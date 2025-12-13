import torch
import torch.nn.functional as F
import argparse
import os
from src.snap import SNAP 
from utils.torch_helpers import all_to_device
from datasets.demo import DemoDatset
import numpy as np
import pyvista as pv
import open3d as o3d
import clip
import pdb
import pandas as pd
import hdbscan

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
    def __init__(self, checkpoint_path=None, domain="Outdoor", grid_size=None):
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

        self.model.load_state_dict(state_dict, strict=False)
    
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

    def visualize_results(self, point_cloud_data, masks, text_labels, iou_scores):
        ## Visualize results using PyVista
        
        # Create a PyVista plotter
        plotter = pv.Plotter()
        # Create a point cloud object
        point_cloud = pv.PolyData(point_cloud_data['coord'].cpu().numpy())
        # Make the point cloud grey in color
        point_cloud['RGB'] = np.tile([0.7, 0.7, 0.7], (len(point_cloud_data['coord']), 1))
        # Add the point cloud to the plotter
        plotter.add_mesh(point_cloud, scalars='RGB', rgb=True, render_points_as_spheres=True, point_size=5, color='white', name='Point Cloud')
        
        # Add the masks to the plotter
        for i, mask in enumerate(masks):
            # Get the points for the mask
            mask_points = point_cloud_data['coord'][mask]
            # Create a PyVista mesh for the mask
            mask_mesh = pv.PolyData(mask_points.cpu().numpy())
            # Make the mask color red
            mask_mesh['RGB'] = np.tile([1, 0, 0], (len(mask_points), 1))
            # Add the mask to the plotter
            plotter.add_mesh(mask_mesh, scalars='RGB', rgb=True, render_points_as_spheres=True, point_size=5, name=f'Mask {i}')
            
            # Add the text label to the mask
            out_str = f"{text_labels[i]}_{iou_scores[i][0]:.2f}"
            label_point = mask_points.mean(axis=0).cpu().numpy()
            plotter.add_point_labels(
                    [label_point],         # a list of 3D coords
                    [out_str],          # the corresponding string label
                    point_size=0,          # no visible marker, just text
                    font_size=14,
                    shape=None,            # no bounding shape behind text
                    fill_shape=False,
                    always_visible=True,    # keep it visible through geometry
                    text_color='red',
                )

        # Show the plotter
        plotter.show()


if __name__ == "__main__":
    # Load the point cloud data
    point_cloud = PointCloudData()
    point_cloud_data = point_cloud.load_point_cloud("data_examples/KITTI/000000.bin")

    # Load the model
    model = SegmentationModel(checkpoint_path="/home/thor/Projects/Lidar_Segmentation/checkpoints/SNAP_aerial_outdoor_indoor_epoch_19.pth", domain="Outdoor", grid_size=0.05)

    # Intialize the model and run the model backbone to extract point features
    point_cloud_data = model.intialize_pointcloud(point_cloud_data)
    model.extract_backbone_features(point_cloud_data)

    # Specify a prompt point
    # prompt_points = [[[x1, y1, z1], [x2, y2, z2], ...]]  # List of prompt pointe, Shape -> M, P, 3 (M = number of objects, P = number of clicks on each object)
    # The prompt points should be in the format [x, y, z] and should be in the same coordinate system as the point cloud data
    prompt_points = [[[ 9.4854517 ,  7.34119511, -0.40044212]]]

    # Run segmentation
    masks, text_labels, iou_scores = model.segment(point_cloud_data, prompt_points, text_prompt=None)

    # Visualize the results
    model.visualize_results(point_cloud_data, masks, text_labels, iou_scores)