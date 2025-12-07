import torch
import torch.nn.functional as F
import argparse
import os
import time
from src.snap import SNAP 
from utils.torch_helpers import all_to_device
from datasets.demo import DemoDatset
import logging
from tqdm import tqdm
from conf_input import semantic_kitti_new, nuscenes_new, pandaset, scannet, s3dis, scanrefer, scannetpp, scannet_block, partnet, kitti360, stpls3d
import numpy as np
import pyvista as pv
import open3d as o3d
from torch.cuda.amp import autocast, GradScaler
import clip
import pdb
import pandas as pd
import hdbscan

# pv.global_theme.allow_empty_mesh = True

def get_args_parser():
    parser = argparse.ArgumentParser()

    # Tensorboard summary name
    parser.add_argument('--enable_amp', action='store_true', default=False)
    # Model
    parser.add_argument('--mask_threshold', default=0.5, type=float)
    parser.add_argument('--resume', default="checkpoints/scannet_kitti_epoch_24.pth", type=str)
    # Output
    parser.add_argument('--save_output', default=False, type=bool)
    # Inference
    # parser.add_argument('--pc_path', default="data_examples/KITTI/000000.bin", type=str)
    parser.add_argument('--pc_path', default="data_examples/PandaSet/30.pkl", type=str)
    # parser.add_argument('--pc_path', default="data_examples/nuScenes/example_3.pcd.bin", type=str)
    # parser.add_argument('--pc_path', default="/home/thor/Projects/Lidar_Segmentation/data_examples/ScanNet/scene0203_02", type=str)
    parser.add_argument('--domain', default='Outdoor', type=str)
    parser.add_argument('--grid_size', default=0.05, type=float)
    parser.add_argument('--seed', default=42, type=int)

    return parser

class PointCloudData:
    def __init__(self):
        self.points = None  # Initialize without points

    def load_point_cloud(self, filename):
        data = {}
        # Load the point cloud from the temporary file
        if filename.lower().endswith('.ply'):
            pcd = o3d.io.read_point_cloud(filename)
        elif filename.lower().endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(filename)
        elif filename.lower().endswith('.pcd.bin'): # Nuscenes format
            pcdata = np.fromfile(filename, dtype=np.float32, count=-1)
            pcdata = pcdata.reshape(-1, 5)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcdata[:, :3])
            data["intensity"] = pcdata[:, 3].reshape(-1, 1)/255.0
        elif filename.lower().endswith('.bin'): # Kitti format
            pcdata = np.fromfile(filename, dtype=np.float32)
            pcdata = pcdata.reshape(-1, 4)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcdata[:, :3])
            data["intensity"] = pcdata[:, 3].reshape(-1, 1)
            print("Intensity data found")
            print("Min and max are: ", data["intensity"].min(), data["intensity"].max())
        elif filename.lower().endswith('.pkl'): # Pandaset format
            pcdata = pd.read_pickle(filename)
            valid_idxs = np.where(pcdata.d==0)[0] # Get the points belonging to the 360 point cloud only
            pcdata_points = np.asarray(pcdata.values[valid_idxs][:, :3])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcdata_points)
            data['intensity'] = np.expand_dims(np.asarray(pcdata.values[valid_idxs][:, 3])/255.0, 1)
            print("Intensity data found")
            print("Min and max are: ", data["intensity"].min(), data["intensity"].max())
        elif filename.lower().endswith('.pth'):
            pcdata = torch.load(filename)
            coord = pcdata[0]
            scale_mat = np.diag([0.1, 0.1, 0.1])
            coord = coord @ scale_mat

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord)
            data['color'] = pcdata[1]

        else:
            # Check if path is a directory
            if os.path.isdir(filename):
                # Check if the following files exist in the directory -> coords.npy, colors.npy, normals.npy
                coord_path = os.path.join(filename, "coord.npy")
                color_path = os.path.join(filename, "color.npy")
                normal_path = os.path.join(filename, "normal.npy")

                if os.path.exists(coord_path):
                    coords = np.load(coord_path)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(coords)
                
                if os.path.exists(color_path):
                    colors = np.load(color_path)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                
                if os.path.exists(normal_path):
                    normals = np.load(normal_path)
                    pcd.normals = o3d.utility.Vector3dVector(normals)
     
            else:
                raise ValueError("Invalid file format. Supported formats are .ply, .pcd, .bin, .pcd.bin")


        data["coord"] = np.asarray(pcd.points)
        if np.asarray(pcd.colors).shape[0] > 0:
            print("Color data found")
            data["color"] = np.asarray(pcd.colors) 
        if np.asarray(pcd.normals).shape[0] > 0:
            data["normal"] = np.asarray(pcd.normals) 

        return data

class SegmentationModel:
    def __init__(self, checkpoint_path=None, domain="Outdoor", grid_size=0.05):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.domain = domain

        # Initialize the datset class
        self.dataset = DemoDatset(domain=self.domain, grid_size=grid_size)

        # Initialize the segmentation model
        self.model = SNAP(num_points=1, num_merge_blocks=1, use_pdnorm=True, return_mid_points=True).to(self.device)

        print(f'Number of trainable params: {sum(p.numel() for p in self.model.parameters())}')

        # Load the pre-trained weights
        if checkpoint_path:
            print(f'Loading checkpoint from {checkpoint_path}')
            self.loadweights(checkpoint_path)
        else:
            print("No checkpoint path provided. Model will be initialized with random weights.")

        # Intialize the clip model
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.point_features = None
        self.mid_points_features = None

    def loadweights(self, path):
        # Load trained weights to the model
        checkpoint = torch.load(path)

        state_dict = {}
        for k, v in checkpoint['model'].items():
            state_dict[k] = v

        self.model.load_state_dict(state_dict)
    
    def intialize_pointcloud(self, point_cloud_data):
        coord = point_cloud_data['coord']
        color = point_cloud_data['color'] if 'color' in point_cloud_data else None
        normal = point_cloud_data['normal'] if 'normal' in point_cloud_data else None
        intensity = point_cloud_data['intensity'] if 'intensity' in point_cloud_data else None

        # Process the point cloud data
        data = self.dataset.process_data(coord, color, normal, intensity)

        return data
    
    def extract_backbone_features(self, data):
        # Put the model in eval mode
        self.model.eval()

        # Put the data on device
        data = all_to_device(data, self.device)

        with torch.no_grad():
            # Run the input encoder
            data = self.model.run_input_encoder(data)

            # Get the point features through model backbone
            self.point_features, self.mid_points_features = self.model.run_backbone(data)
    
    def get_text_label(self, clip_token, pred_token, labels):
        clip_token = F.normalize(clip_token, p=2, dim=1).to(torch.float32)
        pred_token = F.normalize(pred_token, p=2, dim=1)

        logits = pred_token @ clip_token.T
        # pdb.set_trace()
        pred_labels = torch.argmax(logits, dim=1)
        score = logits[0, pred_labels[0]].cpu().numpy()
        return labels[pred_labels], score
    
    def segment_everything(self, data):
        # Cluster the mid points to reduce the number of prompt points -> This will ensure that there are not a lot of points on the same object
        # 1. Use HDBSCAN to cluster the mid points
        if self.domain == 'Outdoor':
            # Make the prompt points using the mid points
            prompt_points = self.point_features['coord'].cpu().numpy()
            print("Mid points shape: ", prompt_points.shape)

            clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=50, cluster_selection_epsilon=0.1)
            cluster_labels = clusterer.fit_predict(prompt_points)
            unique_labels = np.unique(cluster_labels)
            print("Number of clusters: ", len(unique_labels))

            # Get the cluster centers
            cluster_centers = []
            for label in unique_labels:
                cluster_centers.append([np.mean(prompt_points[cluster_labels==label], axis=0)])

            prompt_points = cluster_centers
        
        elif self.domain == 'Indoor':
            prompt_points = self.mid_points_features['coord'].cpu().numpy()
            # Randomly drop 80% of the points
            drop_indices = np.random.choice(len(prompt_points), int(0.8*len(prompt_points)), replace=False)
            prompt_points = np.delete(prompt_points, drop_indices, axis=0)
            print("Mid points shape: ", prompt_points.shape)

            prompt_points_list = []
            for point in prompt_points:
                prompt_points_list.append([point])
            
            prompt_points = prompt_points_list

        masks, text_out_list, iou_out_list = self.segment(data, prompt_points, text_prompt=None)

        return masks, text_out_list, iou_out_list, prompt_points

    def segment(self, data, prompt_points, text_prompt):
        ## Need to create a custom dataloader for the point cloud data with a custom config file
        # 1. This only takes input the pointcloud and gets coord, grid_coord, feat and offset out
        # 2. This sample data needs to be passed to the model to generate the masks in an iterative manner for each prompt
        # 3. If the text prompt is provided, we need to create a uniform grid of points across the pointcloud and pass it to the model
        # 4. If Segment everything is selected, we use the uniform grid of points again.
        # 5. Need to use NMS to remove overlapping masks

        masks = []
        text_out_list = []
        iou_out_list = []
        for idx in range(len(prompt_points)):
            print("Runing for prompt: ", idx)
            # Process the point cloud data
            prompt_points_array = np.expand_dims(np.array(prompt_points[idx]), 0)
            data = self.dataset.process_prompts(data, prompt_points_array, text_prompt)

            # Get text encoding for the text prompt
            if text_prompt:
                print(text_prompt)
                tokenized_text_prompt = clip.tokenize(text_prompt).to(self.device)
                text_features = self.clip_model.encode_text(tokenized_text_prompt)

            text_inputs_vocab = torch.cat([clip.tokenize(f"segment {c}") for c in self.dataset.labels]).to(self.device)
            text_features_vocab = self.clip_model.encode_text(text_inputs_vocab)

            # Extract masks for each prompt
            data = all_to_device(data, self.device)
            data["point_offset"] = [data['point'].shape[0]]
            with torch.no_grad():
                # text_out_list = []
                # Run through the mask decoder to get masks
                for i in range(len(data['point'])):
                    data_dict = data.copy()
                    data_dict["point"] = data['point'][i].unsqueeze(0)

                    mask_logits, text_out, iou_out, _, _, _, _ = self.model.run_mask_decoder(self.point_features, data_dict)
                    masks.append(mask_logits[0].sigmoid().squeeze().cpu().numpy() > 0.5)
                   
                    # Get the text output
                    label, score = self.get_text_label(text_features_vocab, text_out[0], self.dataset.labels)
                    text_out_list.append(label)
                    iou_out_list.append(iou_out[0][0].cpu().numpy())
                    print(f"Label: {label}, IOU: {iou_out[0]} Score: {score}")

        return masks, text_out_list, iou_out_list

class InteractivePlotter:
    """
    Encapsulates all PyVista interactions:
      - Creating the PolyData
      - Enabling point picking
      - Storing user clicks
      - The 'New Object' button logic
    """
    def __init__(self, args, point_cloud_data):
        """
        point_cloud_data: NxM numpy array (M >= 3 for x,y,z; possibly color, etc.)
        """
        # Intialize the model
        self.model = SegmentationModel(args.resume, domain=args.domain, grid_size=args.grid_size)
        # Process input point cloud data
        self.model_data = self.model.intialize_pointcloud(point_cloud_data)
        # Run the model backbone to get the point features
        self.model.extract_backbone_features(self.model_data)

        self.args = args

        self.point_cloud_data = self.model_data.copy()
        self.point_cloud_data['coord'] = self.point_cloud_data['coord'].cpu().numpy()
        if 'color' in point_cloud_data:
            self.point_cloud_data['color'] = self.point_cloud_data['color'].cpu().numpy()
        if 'normal' in point_cloud_data:
            self.point_cloud_data['normal'] = self.point_cloud_data['normal'].cpu().numpy()
        if 'intensity' in point_cloud_data:
            self.point_cloud_data['intensity'] = self.point_cloud_data['strength'].cpu().numpy()
        
        # Convert to a PyVista PolyData (we'll use just the first 3 cols as coordinates).
        coords = self.point_cloud_data['coord']
        self.pc = pv.PolyData(coords)

        if 'color' in self.point_cloud_data:
            if self.args.domain == "Indoor":
                unnormalized_colors = (self.point_cloud_data['color']+1)*127.5
                colors = np.clip(unnormalized_colors, 0, 255).astype(np.uint8)
                self.pc["RGB"] = colors/255.0
                self.vis_colors = colors/255.0
            elif self.args.domain == "Aerial":
                unnormalized_colors = (self.point_cloud_data['color']+1)*127.5 # This will bring the colors between -1 and 1 for stpls3d
                unnormalized_colors = (unnormalized_colors+1) * 127.5
                colors = np.clip(unnormalized_colors, 0, 255).astype(np.uint8)
                self.pc["RGB"] = colors/255.0
                self.vis_colors = colors/255.0
            
        elif 'intensity' in self.point_cloud_data:
            # If only intensity is provided, use it as grayscale
            self.pc['RGB'] = np.tile(self.point_cloud_data['intensity'], (1, 3))
            self.vis_colors = np.tile(self.point_cloud_data['intensity'], (1, 3))

        else:
            print("No color information provided. Using default color gray.")
            # If no color information or intensity is provided, use default color gray
            self.pc['RGB'] = np.tile([0.7, 0.7, 0.7], (len(coords), 1))
            self.vis_colors = np.tile([0.7, 0.7, 0.7], (len(coords), 1))
        
        # Container for user clicks
        self.current_clicks = []
        self.all_click_sets = []
        self.all_prompt_points = []
        self.sphere_list = []
        self.label_list = []
        
        # Initialize the plotter
        self.plotter = pv.Plotter(window_size=(800, 600))

        # Add the point cloud to the scene
        self.plotter.add_mesh(self.pc, scalars='RGB', rgb=True, pickable=True, name="point_cloud", point_size=5, render_points_as_spheres=True)
        
        # Enable point picking (PyVista >= 0.39 requires `use_picker=True`)
        self.plotter.enable_point_picking(
            callback=self.point_picking_callback,
            use_picker=True,
            show_point=True,
            tolerance=0.01,
            color='yellow',    # or 'yellow', or any valid color
            point_size=20,
            render_points_as_spheres=True
        )

        # Add a checkbox widget to mimic a simple push-button
        self.new_object_checkbox = self.plotter.add_checkbox_button_widget(
            callback=self.new_object_button_callback,
            position=(10, 10),
            size=25,
            color_on='white',
            color_off='gray'
        )
        
        self.plotter.add_text(
            "New Object", 
            position=(40, 13),   # shift text to the right of the checkbox
            font_size=12
        )

        # Add a checkbox widget to mimic a simple segment button
        self.segment_checkbox = self.plotter.add_checkbox_button_widget(
            callback=self.segment_button_callback,
            position=(10, 50),
            size=25,
            color_on='white',
            color_off='gray'
        )

        self.plotter.add_text(
            "Segment", 
            position=(40, 53),   # shift text to the right of the checkbox
            font_size=12
        )

        # Add a checkbox widget to for Segment Everything
        self.segment_everything_checkbox = self.plotter.add_checkbox_button_widget(
            callback=self.segment_everything_button_callback,
            position=(10, 90),
            size=25,
            color_on='white',
            color_off='gray'
        )

        self.plotter.add_text(
            "Segment Everything", 
            position=(40, 93),   # shift text to the right of the checkbox
            font_size=12
        )

    def point_picking_callback(self, picked_point, point_index):
        """Triggered on point picks in the PyVista window."""
        if point_index != -1:
            self.current_clicks.append(picked_point)

            # Make a new sphere at the picked point
            if self.args.domain == 'Outdoor':
                sphere = pv.Sphere(radius=0.5, center=picked_point)
            elif self.args.domain == 'Indoor':
                sphere = pv.Sphere(radius=0.1, center=picked_point)
            elif self.args.domain == 'Aerial':
                sphere = pv.Sphere(radius=0.1, center=picked_point)
                
            self.plotter.add_mesh(sphere, color='yellow', name=f"sphere_{len(self.sphere_list)}")
            self.sphere_list.append(f"sphere_{len(self.sphere_list)}")
            self.plotter.update()

            print(f"Picked point #{picked_point}")

    def new_object_button_callback(self, checked):
        """
        Invoked by the "New Object" checkbox. 
        We treat it like a push-button by immediately toggling it off.
        """
        if checked:
            self.new_object()
            # Forcibly revert the checkbox to off
            rep = self.new_object_checkbox.GetRepresentation()
            rep.SetState(0)

    def new_object(self):
        """Finalize the current set of clicks into `all_click_sets`."""
        if self.current_clicks:
            self.all_click_sets.append(list(self.current_clicks))
            self.current_clicks.clear()
        print(f"[New Object] Current sets: {self.all_click_sets}")

    def segment_button_callback(self, checked):
        """
        Invoked by the "Segment" checkbox. 
        We treat it like a push-button by immediately toggling it off.
        """
        if checked:
            self.segment()
            # Forcibly revert the checkbox to off
            rep = self.segment_checkbox.GetRepresentation()
            rep.SetState(0)
        
    def segment(self, everything=False):
        """Segment the point cloud based on the user clicks."""
        # Pass all the user clicks sets to the segmentation model

        if not self.all_click_sets and not everything:
            print("No user-provided click sets available. Please provide prompt points.")
            return
        
        # Clear the previous sphere and label objects
        for sphere in self.sphere_list:
            self.plotter.remove_actor(sphere)

        for label in self.label_list:
            self.plotter.remove_actor(label)

        self.plotter.update()
        self.sphere_list.clear()
        self.label_list.clear()

        if everything:
            masks, text_labels, iou_out, prompt_points_used = self.model.segment_everything(self.model_data)
            self.all_click_sets = prompt_points_used
        else:        
            masks, text_labels, iou_out = self.model.segment(self.model_data, self.all_click_sets, text_prompt=None)

        # Make a new sphere at the picked point
        i=0
        for prompt_points_set in self.all_click_sets:
            for prompt_point in prompt_points_set:
                if self.args.domain == 'Outdoor':
                    sphere = pv.Sphere(radius=0.5, center=prompt_point)
                elif self.args.domain == 'Indoor':
                    sphere = pv.Sphere(radius=0.1, center=prompt_point)
                elif self.args.domain == 'Aerial':
                    sphere = pv.Sphere(radius=0.1, center=prompt_point)
                    
                self.plotter.add_mesh(sphere, color='yellow', name=f"sphere_{i}")
                self.sphere_list.append(f"sphere_{i}")
                self.plotter.update()
                i+=1

        color_array = self.vis_colors.copy()
        for mask in masks:
            color = np.random.rand(3)
            mask_color = color * 0.5 + 0.5
            color_array[mask] = mask_color
        
        self.pc['RGB'] = color_array
        self.pc.active_scalars_name = 'RGB'
        self.pc.set_active_scalars('RGB')
        self.plotter.update()

        # pdb.set_trace()
        # Add the text labels to the scene - only add this when not segmenting everything otherwise it will be too cluttered
        if not everything:
            for idx, (points_set, label, iou) in enumerate(zip(self.all_click_sets, text_labels, iou_out)):
                if len(points_set) == 0:
                    continue
                # Example: Place label at the *first* prompt point
                label_point = points_set[0]
                # If you prefer the center/mean:
                # label_point = np.mean(points_set, axis=0)

                out_str = f"{label} {iou[0]:.2f}"

                # Add the label in 3D
                label_actor = self.plotter.add_point_labels(
                    [label_point],         # a list of 3D coords
                    [out_str],          # the corresponding string label
                    point_size=0,          # no visible marker, just text
                    font_size=14,
                    name=f"label_{idx}",
                    shape=None,            # no bounding shape behind text
                    fill_shape=False,
                    always_visible=True,    # keep it visible through geometry
                    text_color='red',
                )
                self.label_list.append(f"label_{idx}")

            self.plotter.update()

        # Empty the click sets
        self.all_click_sets.clear()
        self.current_clicks.clear()

    def segment_everything_button_callback(self, checked):
        """
        Invoked by the "Segment Everything" checkbox. 
        We treat it like a push-button by immediately toggling it off.
        """
        if checked:
            self.segment(everything=True)
            # Forcibly revert the checkbox to off
            rep = self.segment_everything_checkbox.GetRepresentation()
            rep.SetState(0)
        
    def start_plotting(self):
        """Open the interactive PyVista window."""
        self.plotter.show()

def main(args):
    # Log available GPUs
    logging.info(f'Available GPUs: {torch.cuda.device_count()}')

    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get the point cloud
    point_cloud = PointCloudData()
    point_cloud_data = point_cloud.load_point_cloud(args.pc_path)

    # Create an instance of our interactive plotter class
    interactive_plotter = InteractivePlotter(args, point_cloud_data)
    
    # Launch the interactive session
    interactive_plotter.start_plotting()

    print("User-provided click sets:", interactive_plotter.all_click_sets)
    

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
