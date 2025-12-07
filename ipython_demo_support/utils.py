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
import pandas as pd

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
        elif filename.endswith('.npy'): # DALES format
            pcdata = np.load(filename)
            data["coord"] = pcdata[:, :3]
            return data
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
    

def voxel_downsample(points: np.ndarray, voxel_size: float):
    """
    Downsamples a point cloud by selecting the point closest to the centroid
    of each voxel using a vectorized approach.

    Args:
        points (np.ndarray): The input point cloud as a NumPy array of shape (N, 3).
        voxel_size (float): The side length of the voxel grid cubes. A larger
                            size results in more aggressive downsampling.

    Returns:
        np.ndarray: The downsampled point cloud, where each point is an actual
                    point from the original cloud.
    """
    # 1. Quantize the points to get discrete voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # 2. Get the unique voxel indices and the inverse mapping.
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    # 3. Calculate the centroid for each voxel (as before)
    summed_points = np.zeros((len(unique_voxels), 3))
    np.add.at(summed_points, inverse_indices, points)
    counts = np.bincount(inverse_indices)
    centroids = summed_points / counts[:, np.newaxis]

    # --- 4. Find the point closest to the centroid for each voxel (Vectorized) ---
    # a. Get the corresponding centroid for each original point
    point_centroids = centroids[inverse_indices]
    
    # b. Calculate the squared Euclidean distance for all points from their voxel's centroid
    distances_sq = np.sum((points - point_centroids)**2, axis=1)
    
    # c. Sort points first by their voxel group, then by their distance to the centroid
    # np.lexsort sorts by the last key first, so we pass distance then group index.
    sorted_indices = np.lexsort((distances_sq, inverse_indices))
    
    # d. The first occurrence of each voxel index in the sorted list corresponds
    # to the point with the minimum distance for that voxel. We find these first occurrences.
    _, first_occurrence_indices = np.unique(inverse_indices[sorted_indices], return_index=True)
    
    # e. Get the original indices of these closest points
    min_dist_point_indices = sorted_indices[first_occurrence_indices]
    
    # f. Select the points from the original cloud
    closest_points = points[min_dist_point_indices]

    return closest_points

def non_maximum_suppression_masks(
    masks: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float
):
    """
    Performs Non-Maximum Suppression (NMS) on a set of masks.

    This function is crucial for cleaning up redundant, overlapping predictions
    from an instance segmentation model.

    Args:
        masks (torch.Tensor): A boolean tensor of predicted masks, shape (M, N),
                              where M is the number of predicted instances and N
                              is the number of points.
        scores (torch.Tensor): A tensor of confidence scores for each mask,
                               shape (M,). This is typically the predicted IoU.
        iou_threshold (float): The IoU threshold above which a mask will be
                               suppressed. A common value is 0.7.

    Returns:
        torch.Tensor: A 1D tensor of indices of the masks to keep after NMS.
    """
    # Ensure there are masks to process
    if masks.shape[0] == 0:
        return torch.tensor([], dtype=torch.long)

    # --- Step 1: Sort masks by their confidence scores in descending order ---
    sorted_indices = torch.argsort(scores, descending=True)

    keep_indices = []
    # Use a boolean tensor to keep track of which masks have been suppressed
    suppressed = torch.zeros(masks.shape[0], dtype=torch.bool, device=masks.device)

    # --- Step 2: Iterate through the sorted masks ---
    for i in range(masks.shape[0]):
        current_idx = sorted_indices[i]

        # If the current mask has already been suppressed by a higher-scoring one, skip it
        if suppressed[current_idx]:
            continue

        # Otherwise, keep this mask and use it to suppress others
        keep_indices.append(current_idx)
        current_mask = masks[current_idx]

        # --- Step 3: Compare the current mask with all others ---
        # We only need to compare with masks that appear later in the sorted list
        other_indices = sorted_indices[i + 1:]
        
        # If there are no other masks left to compare, we are done
        if len(other_indices) == 0:
            break

        other_masks = masks[other_indices]

        # --- Step 4: Calculate IoU in a vectorized way for efficiency ---
        # Intersection is the dot product of the boolean masks (converted to float)
        intersection = torch.matmul(current_mask.float().unsqueeze(0), other_masks.float().T)

        # Union = Area(A) + Area(B) - Intersection
        area_current = current_mask.sum()
        areas_other = other_masks.sum(dim=1)
        union = area_current + areas_other - intersection
        
        # Compute IoU, handling the case of zero union
        iou = torch.where(union > 0, intersection / union, torch.tensor(0.0, device=masks.device)).squeeze(0)

        # --- Step 5: Suppress overlapping masks ---
        # Find the indices of masks that have an IoU greater than the threshold
        suppress_mask_indices = other_indices[iou > iou_threshold]
        
        # Mark these masks as suppressed
        suppressed[suppress_mask_indices] = True

    return torch.tensor(keep_indices, dtype=torch.long, device=masks.device)


class SegmentationModel:
    def __init__(self,checkpoint_path=None, domain="Outdoor", grid_size=0.05, num_object_chunk=16, mask_threshold=0.5):
        self.num_object_chunk = num_object_chunk
        self.mask_threshold = mask_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.domain = domain

        # Initialize the datset class
        self.dataset = DemoDatset(domain=self.domain)

        # Initialize the segmentation model
        self.model = SNAP(num_points=1, num_merge_blocks=1, use_pdnorm=True, return_mid_points=True).to(self.device)
        # Put the model in eval mode
        self.model.eval()

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
        score = logits[0, pred_labels[0]].detach().cpu().numpy()
        return labels[pred_labels], score
    
    def find_mask_from_text(self, text_prompt, z_score_threshold=1.25):
        """
        Finds the best matching mask from a cached set based on a text prompt.

        Args:
            cached_tokens (list[torch.Tensor]): A list of cached feature tokens from 'segment_everything'.
            text_prompt (str): The user-provided text query.

        Returns:
            int: The index of the best matching mask, or -1 if no tokens are cached.
        """
        if len(self.cached_mask_features) == 0:
            print("Error: Mask features not cached. Please run `segment_everything` first.")
            return []
        
        with torch.no_grad():
            # Encode the text prompt using CLIP
            txt_in = clip.tokenize(text_prompt).to(self.device)
            prompt_feats = F.normalize(self.clip_model.encode_text(txt_in).to(torch.float32), p=2, dim=-1)
        
        # Calculate similarity between the text feature and all cached mask features
        scores = torch.mm(self.cached_mask_features, prompt_feats.T)

        # mean_score = scores.mean()
        # std_score = scores.std()
        # # Handle case where std_dev is zero to avoid division by zero
        # if std_score == 0:
        #     print("Warning: All scores are identical, cannot compute a meaningful Z-score.")
        #     return [torch.argmax(scores).item()] if scores.numel() > 0 else []

        # z_scores = (scores - mean_score) / std_score
        # print(z_scores, type(z_scores))
        # print(z_score_threshold, type(z_score_threshold))
        # significant_indices = torch.where(z_scores > z_score_threshold)[0]

        significant_indices = torch.where(scores > z_score_threshold)[0]

        if significant_indices.numel() == 0:
            print(f"No statistically significant matches found for '{text_prompt}'. Returning the single best match.")
            return [torch.argmax(scores).item()]
        
        print(f"Found {len(significant_indices)} significant matches for '{text_prompt}'.")
        return significant_indices.tolist()
    
    def find_masks_by_label(self, text_labels, query_label):
        """
        Finds all masks that have been assigned a specific text label.

        This method performs a robust string comparison against the pre-computed
        labels for each mask, avoiding the variability of similarity scores.

        Args:
            text_labels (list[str]): The list of assigned labels for all final masks
                                     (e.g., from the output of `segment_everything`).
            query_label (str): The specific label to search for (e.g., "chair").

        Returns:
            list[int]: A list of indices of the matching masks.
        """
        # Normalize the query for case-insensitive and whitespace-insensitive matching
        normalized_query = query_label.lower().strip()
        
        matching_indices = [
            i for i, label in enumerate(text_labels)
            if label.lower().strip() == normalized_query
        ]

        if matching_indices:
            print(f"Found {len(matching_indices)} mask(s) with the label '{query_label}' at indices: {matching_indices}")
        else:
            print(f"No masks found with the label '{query_label}'.")
            
        return matching_indices
    
    def segment_everything(self, data, max_iters=3, initial_voxel_size=None, iou_score_thresh=0.5, nms_iou_thresh=0.1):
        # --- 1. Initialization ---
        if initial_voxel_size is not None:
            init_voxel_size = initial_voxel_size
        elif self.domain == 'Outdoor':
            init_voxel_size = 10.0 # Start with a coarse grid
        elif self.domain == 'Indoor':
            init_voxel_size = 10.0
        elif self.domain == 'Aerial':
            init_voxel_size = 20.0
        
        max_iterations = max_iters
        min_mask_points = 10 # Ignore tiny, noisy masks

        # Get the original points and keep them unchanged
        original_points = self.point_features['coord'].cpu()
        N = original_points.shape[0]

        # Master mask to track all points that have been covered by a mask
        # Initially, no points are covered.
        covered_mask = torch.zeros(N, dtype=torch.bool, device=original_points.device)
        
        # Lists to accumulate results from all iterations
        all_masks = []
        all_text_outs = []
        all_iou_outs = []
        all_prompt_points = [] # New list to store prompts
        all_feature_tokens = []

        # --- 2. Iterative Masking Loop ---
        for i in range(max_iterations):
            # --- A. Identify unsegmented points ---
            uncovered_indices = torch.where(~covered_mask)[0]
            uncovered_points = original_points[uncovered_indices].numpy()

            print(f"\n--- Iteration {i+1} ---")
            print(f"Points remaining to be segmented: {len(uncovered_points)}")

            if len(uncovered_points) < min_mask_points:
                print("Stopping: Most points have been segmented.")
                break

            # --- B. Generate new prompts from uncovered regions ---
            voxel_size = init_voxel_size / (2**i)
            print(f"Using Voxel Size: {voxel_size:.2f}")
            
            if uncovered_points.shape[0] > 10:
                prompt_points_np = voxel_downsample(uncovered_points, voxel_size)
            else:
                prompt_points_np = np.mean(uncovered_points, axis=0, keepdims=True)
            
            print(f"Generated {prompt_points_np.shape[0]} new prompt points.")
            
            prompt_points_tensor = torch.from_numpy(prompt_points_np).float().to(original_points.device)
            # Assuming model expects prompts as (num_prompts, 1, 3)
            prompt_points_for_model = prompt_points_tensor.unsqueeze(1)
            all_prompt_points.extend(prompt_points_for_model.cpu().numpy().tolist())

            # --- C. Get new masks from the model ---
            masks, text_out_list, iou_out_list, feature_tokens = self.segment(data, prompt_points_for_model)
            
            # --- D. Update the master covered_mask ---
            newly_covered_mask = torch.zeros(N, dtype=torch.bool, device=original_points.device)
            # Iterate with index to keep track of corresponding prompts
            for idx, mask_np in enumerate(masks):
                iou_score = iou_out_list[idx][0] # Assuming score is a single value in a list/tensor

                # Filter by IoU scores
                if iou_score < iou_score_thresh:
                    continue

                # *** FIX: Convert numpy mask to a torch tensor on the correct device ***
                mask = torch.from_numpy(mask_np).to(original_points.device)

                if mask.sum() > min_mask_points:
                    # Accumulate all outputs for accepted masks
                    all_masks.append(mask) # Now appending a tensor
                    all_text_outs.append(text_out_list[idx])
                    all_iou_outs.append(iou_out_list[idx])
                    # all_prompt_points.append(prompt_points_tensor[idx]) # Store the prompt
                    all_feature_tokens.append(feature_tokens[idx])
                    
                    # This operation will now work correctly
                    newly_covered_mask = torch.logical_or(newly_covered_mask, mask)
            
            covered_mask = torch.logical_or(covered_mask, newly_covered_mask)

        # --- 3. Final Non-Maximum Suppression ---
        print("\n--- Post-Processing ---")
        if not all_masks:
            print("No masks were generated.")
            return [], [], [], []

        final_masks_tensor = torch.stack(all_masks)
        # final_scores_tensor = torch.cat([iou.squeeze(1) for iou in all_iou_outs])
        final_tokens_tensor = torch.stack(all_feature_tokens) # Stack feature tensors too
        final_scores_tensor = torch.cat([torch.as_tensor(iou, device=original_points.device).view(-1) for iou in all_iou_outs])

        print(f"Running NMS on {final_masks_tensor.shape[0]} masks...")
        
        keep_indices = non_maximum_suppression_masks(
            final_masks_tensor,
            final_scores_tensor,
            nms_iou_thresh
        )
        
        print(f"Kept {len(keep_indices)} final masks after NMS.")

        # Filter all outputs based on NMS results
        final_masks_nms = [final_masks_tensor[i] for i in keep_indices]
        final_text_out_nms = [all_text_outs[i] for i in keep_indices]
        final_iou_out_nms = [all_iou_outs[i] for i in keep_indices]
        final_prompts_nms = [all_prompt_points[i] for i in keep_indices] # Filter prompts

        # --- CACHING STEP ---
        # Filter the feature tokens with the same indices and store them in the cache
        self.cached_mask_features = final_tokens_tensor[keep_indices]

        return final_masks_nms, final_text_out_nms, final_iou_out_nms, final_prompts_nms

    def segment(self, data, prompt_points):
        """
        Segments the point cloud based on prompt points using batched inference.
        """
        # 1. Prepare prompt points tensor
        if isinstance(prompt_points, list):
            # This is from interactive mode (list of lists of coords)
            # Convert to single-point prompts by taking the centroid of clicks for each object.
            processed_points = [np.mean(points, axis=0) for points in prompt_points if points]
            if not processed_points:
                return [], [], []
            prompt_points_tensor = torch.tensor(processed_points, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            # This is from segment_everything (already a tensor)
            prompt_points_tensor = prompt_points

        # Get text features for vocabulary
        text_inputs_vocab = torch.cat([clip.tokenize(f"segment {c}") for c in self.dataset.labels]).to(self.device)
        text_features_vocab = self.clip_model.encode_text(text_inputs_vocab)

        # 2. Batch inference
        all_mask_logits = []
        all_text_out = []
        all_iou_out = []

        num_prompts = prompt_points_tensor.shape[0]
        chunk_size = self.num_object_chunk

        with torch.no_grad():
            for i in range(0, num_prompts, chunk_size):
                end_idx = min(i + chunk_size, num_prompts)
                prompt_chunk = prompt_points_tensor[i:end_idx]

                data_dict = data.copy()
                data_dict["point"] = prompt_chunk
                data_dict["point_offset"] = [prompt_chunk.shape[0]]
                data_dict = all_to_device(data_dict, self.device)

                mask_logits, text_out, iou_out, _, _, _, _ = self.model.run_mask_decoder(self.point_features, data_dict)
                
                all_mask_logits.append(mask_logits[0])
                all_text_out.append(text_out[0])
                all_iou_out.append(iou_out[0])

        if not all_mask_logits:
            return [], [], []

        # 3. Concatenate and process results
        final_mask_logits = torch.cat(all_mask_logits, dim=0)
        final_text_out = torch.cat(all_text_out, dim=0)
        final_iou_out = torch.cat(all_iou_out, dim=0)

        masks = (final_mask_logits.sigmoid() > self.mask_threshold).cpu().numpy()
        
        # Get text labels
        text_out_list = []
        for i in range(final_text_out.shape[0]):
            label, _ = self.get_text_label(text_features_vocab, final_text_out[i].unsqueeze(0), self.dataset.labels)
            text_out_list.append(label)
            
        iou_out_list = final_iou_out.cpu().numpy()

        return list(masks), text_out_list, list(iou_out_list), final_text_out

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
        
        color_array = np.tile([0.7, 0.7, 0.7], (len(point_cloud_data['coord']), 1))
        for mask in masks:
            color = np.random.rand(3)
            mask_color = color * 0.5 + 0.5
            color_array[mask] = mask_color

        point_cloud['RGB'] = color_array

        # Add the masks to the plotter
        for i, mask in enumerate(masks):
            # # Get the points for the mask
            mask_points = point_cloud_data['coord'][mask]
            
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