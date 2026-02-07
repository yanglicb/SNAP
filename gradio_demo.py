#!/usr/bin/env python3
"""
SNAP Gradio Demo - Interactive 3D Point Cloud Segmentation
Browser-based interface using Gradio (works great in WSL2!)
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import argparse
import logging
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import clip
import open3d as o3d
import trimesh
import gradio as gr
import plotly.graph_objects as go

from src.snap import SNAP
from utils.torch_helpers import all_to_device
from datasets.demo import DemoDatset

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def get_args_parser():
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="SNAP Gradio Interactive Demo")
    parser.add_argument('--mask_threshold', default=0.5, type=float, 
                        help="Threshold for converting sigmoid output to a binary mask.")
    parser.add_argument('--resume', default="checkpoints/scannet_kitti_epoch_24.pth", type=str,
                        help="Path to the pre-trained model checkpoint.")
    parser.add_argument('--pc_path', default="data_examples/PandaSet/30.pkl", type=str,
                        help="Path to the input point cloud file.")
    parser.add_argument('--domain', default='Outdoor', type=str,
                        help="Domain of the point cloud ('Outdoor', 'Indoor', 'Aerial').")
    parser.add_argument('--grid_size', default=0.05, type=float,
                        help="Voxel grid size for initial point cloud processing.")
    parser.add_argument('--seed', default=42, type=int,
                        help="Random seed for reproducibility.")
    parser.add_argument('--num_object_chunk', default=16, type=int,
                        help="Batch size for processing prompt points during inference.")
    parser.add_argument('--port', default=7860, type=int,
                        help="Port for Gradio server.")
    parser.add_argument('--share', action='store_true',
                        help="Create a public link for the demo.")
    return parser


def non_maximum_suppression_masks(masks, scores, iou_threshold):
    """Performs Non-Maximum Suppression (NMS) to remove redundant overlapping masks."""
    if masks.shape[0] == 0:
        return torch.tensor([], dtype=torch.long)
    
    sorted_indices = torch.argsort(scores, descending=True)
    keep_indices = []
    suppressed = torch.zeros(masks.shape[0], dtype=torch.bool, device=masks.device)

    for i in range(masks.shape[0]):
        current_idx = sorted_indices[i]
        if suppressed[current_idx]:
            continue
        
        keep_indices.append(current_idx)
        current_mask = masks[current_idx]
        other_indices = sorted_indices[i + 1:]
        if len(other_indices) == 0:
            break

        other_masks = masks[other_indices]
        intersection = torch.matmul(current_mask.float().unsqueeze(0), other_masks.float().T)
        union = current_mask.sum() + other_masks.sum(dim=1) - intersection
        iou = torch.where(union > 0, intersection / union, 
                         torch.tensor(0.0, device=masks.device)).squeeze(0)
        
        suppress_mask_indices = other_indices[iou > iou_threshold]
        suppressed[suppress_mask_indices] = True
        
    return torch.tensor(keep_indices, dtype=torch.long, device=masks.device)


def voxel_downsample(points, voxel_size):
    """Downsamples a point cloud by taking the point closest to the centroid of each voxel."""
    voxel_indices = np.floor(points / voxel_size).astype(int)
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    
    summed_points = np.zeros((len(unique_voxels), 3))
    np.add.at(summed_points, inverse_indices, points)
    counts = np.bincount(inverse_indices)
    centroids = summed_points / counts[:, np.newaxis]
    
    point_centroids = centroids[inverse_indices]
    distances_sq = np.sum((points - point_centroids)**2, axis=1)
    
    sorted_indices = np.lexsort((distances_sq, inverse_indices))
    _, first_occurrence_indices = np.unique(inverse_indices[sorted_indices], return_index=True)
    
    return points[sorted_indices[first_occurrence_indices]]


class PointCloudData:
    """A class to handle loading point cloud data from various file formats."""
    
    def load_point_cloud(self, filename):
        """Loads a point cloud from a file, supporting multiple formats."""
        data = {}
        if filename.lower().endswith(('.ply', '.pcd')):
            pcd = o3d.io.read_point_cloud(filename)
            # Open3D returns colors in [0, 1] range, convert to [0, 255] for consistency
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                # Check if colors are in [0, 1] range and convert to [0, 255]
                if colors.max() <= 1.0:
                    colors = colors * 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors)
        elif filename.lower().endswith('.pcd.bin'):
            pcdata = np.fromfile(filename, dtype=np.float32).reshape(-1, 5)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcdata[:, :3])
            data["intensity"] = pcdata[:, 3].reshape(-1, 1) / 255.0
        elif filename.lower().endswith('.bin'):
            pcdata = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcdata[:, :3])
            data["intensity"] = pcdata[:, 3].reshape(-1, 1)
        elif filename.lower().endswith('.pkl'):
            pcdata = pd.read_pickle(filename)
            valid_idxs = np.where(pcdata.d == 0)[0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcdata.values[valid_idxs][:, :3]))
            data['intensity'] = np.expand_dims(np.asarray(pcdata.values[valid_idxs][:, 3]) / 255.0, 1)
        elif os.path.isdir(filename):
            coords = np.load(os.path.join(filename, "coord.npy"))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            if os.path.exists(os.path.join(filename, "color.npy")):
                pcd.colors = o3d.utility.Vector3dVector(np.load(os.path.join(filename, "color.npy")))
            if os.path.exists(os.path.join(filename, "normal.npy")):
                pcd.normals = o3d.utility.Vector3dVector(np.load(os.path.join(filename, "normal.npy")))
        else:
            raise ValueError("Unsupported file format.")
            
        data["coord"] = np.asarray(pcd.points)
        if pcd.has_colors():
            data["color"] = np.asarray(pcd.colors)
        if pcd.has_normals():
            data["normal"] = np.asarray(pcd.normals)
        return data


# ==============================================================================
# 3. SEGMENTATION MODEL WRAPPER
# ==============================================================================

class SegmentationModel:
    """A wrapper class for the SNAP segmentation model."""
    
    def __init__(self, args, checkpoint_path=None, domain="Outdoor", grid_size=0.05):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.domain = domain
        self.dataset = DemoDatset(domain=self.domain)
        
        self.model = SNAP(num_points=1, num_merge_blocks=1, use_pdnorm=True, 
                         return_mid_points=True).to(self.device)
        self.model.eval()
        
        if checkpoint_path:
            self.loadweights(checkpoint_path)
            
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.point_features = None

    def loadweights(self, path):
        state_dict = torch.load(path, map_location=self.device)['model']
        self.model.load_state_dict(state_dict, strict=False)

    def intialize_pointcloud(self, d):
        return self.dataset.process_data(d.get('coord'), d.get('color'), 
                                        d.get('normal'), d.get('intensity'))

    def extract_backbone_features(self, data):
        data = all_to_device(data, self.device)
        with torch.no_grad():
            data = self.model.run_input_encoder(data)
            self.point_features, _ = self.model.run_backbone(data)
    
    def get_text_label(self, clip_token, pred_token, labels):
        clip_token = F.normalize(clip_token.to(torch.float32), p=2, dim=1)
        pred_token = F.normalize(pred_token, p=2, dim=1)
        
        logits = pred_token @ clip_token.T
        pred_labels = torch.argmax(logits, dim=1)
        
        return labels[pred_labels], logits[0, pred_labels[0]].detach().cpu().numpy()

    def segment_everything(self, data, return_tokens=False):
        init_voxel_size = {'Outdoor': 10.0, 'Indoor': 2.0, 'Aerial': 20.0}.get(self.domain, 10.0)
        max_iter, min_pts, nms_iou, iou_thresh = 3, 50, 0.02, 0.5

        orig_pts = self.point_features['coord'].cpu()
        N = orig_pts.shape[0]
        covered = torch.zeros(N, dtype=torch.bool, device=orig_pts.device)
        all_masks, all_txt, all_iou, all_prompts = [], [], [], []

        for i in range(max_iter):
            uncovered_idx = torch.where(~covered)[0]
            uncovered_pts = orig_pts[uncovered_idx].numpy()
            if len(uncovered_pts) < min_pts:
                break

            prompts_np = voxel_downsample(uncovered_pts, init_voxel_size / (2**i))
            prompts_tensor = torch.from_numpy(prompts_np).float().to(orig_pts.device).unsqueeze(1)
            
            masks, txt_list, iou_list = self.segment(data, prompts_tensor, None, return_tokens)
            
            newly_covered = torch.zeros(N, dtype=torch.bool, device=orig_pts.device)
            for idx, mask_np in enumerate(masks):
                if iou_list[idx][0] >= iou_thresh:
                    mask = torch.from_numpy(mask_np).to(orig_pts.device)
                    if mask.sum() > min_pts:
                        all_masks.append(mask)
                        all_txt.append(txt_list[idx])
                        all_iou.append(iou_list[idx])
                        all_prompts.append(prompts_tensor[idx].cpu().numpy())
                        newly_covered |= mask
            covered |= newly_covered
            
        if not all_masks:
            return [], [], [], []
        
        final_masks = torch.stack(all_masks)
        final_scores = torch.cat([torch.as_tensor(iou, device=orig_pts.device).view(-1) 
                                 for iou in all_iou])
        keep = non_maximum_suppression_masks(final_masks, final_scores, nms_iou)
        
        return ([final_masks[i] for i in keep], [all_txt[i] for i in keep], 
                [all_iou[i] for i in keep], [all_prompts[i] for i in keep])

    def find_best_mask_from_text(self, cached_tokens, text_prompt):
        if not cached_tokens:
            return -1, 0.0
        
        with torch.no_grad():
            txt_in = clip.tokenize(text_prompt).to(self.device)
            prompt_feats = F.normalize(self.clip_model.encode_text(txt_in).to(torch.float32), 
                                      p=2, dim=-1)
        
        mask_tokens = F.normalize(torch.stack(cached_tokens).to(self.device), p=2, dim=-1)
        scores = torch.mm(mask_tokens, prompt_feats.T)
        
        best_idx = torch.argmax(scores)
        best_score = scores[best_idx].item()
        return best_idx.item(), best_score

    def segment(self, data, prompt_points, text_prompt, return_tokens=False):
        if isinstance(prompt_points, list):
            proc_pts = [np.mean(p, axis=0) for p in prompt_points if p]
            if not proc_pts:
                return [], [], []
            prompt_points_tensor = torch.tensor(proc_pts, dtype=torch.float32, 
                                               device=self.device).unsqueeze(1)
        else:
            prompt_points_tensor = prompt_points

        txt_vocab = torch.cat([clip.tokenize(f"a point cloud of a {c}") 
                               for c in self.dataset.labels]).to(self.device)
        with torch.no_grad():
            txt_feats_vocab = self.clip_model.encode_text(txt_vocab)
            
        all_logits, all_txt, all_iou = [], [], []
        num_prompts, chunk_size = prompt_points_tensor.shape[0], self.args.num_object_chunk
        
        with torch.no_grad():
            for i in range(0, num_prompts, chunk_size):
                chunk = prompt_points_tensor[i:i + chunk_size]
                data_dict = {**data, "point": chunk, "point_offset": [chunk.shape[0]]}
                data_dict = all_to_device(data_dict, self.device)
                
                logits, txt, iou, _, _, _, _ = self.model.run_mask_decoder(
                    self.point_features, data_dict)
                all_logits.append(logits[0])
                all_txt.append(txt[0])
                all_iou.append(iou[0])
        
        if not all_logits:
            return [], [], []
        
        final_logits = torch.cat(all_logits, dim=0)
        final_txt = torch.cat(all_txt, dim=0)
        final_iou = torch.cat(all_iou, dim=0)
        
        masks = (final_logits.sigmoid() > self.args.mask_threshold).cpu().numpy()
        
        if return_tokens:
            txt_list = [final_txt[i] for i in range(final_txt.shape[0])]
        else:
            txt_list = [self.get_text_label(txt_feats_vocab, t.unsqueeze(0), 
                                           self.dataset.labels)[0] for t in final_txt]
            
        return list(masks), txt_list, list(final_iou.cpu().numpy())


# ==============================================================================
# 4. VISUALIZATION FUNCTIONS
# ==============================================================================

def create_plotly_figure(coords, colors, title="Point Cloud", point_size=2, labels=None, masks=None):
    """Creates a Plotly 3D scatter plot for point cloud visualization with optional text labels."""
    # Create point cloud scatter plot
    traces = [go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=colors if colors.shape[1] == 3 else None,
            colorscale='Viridis' if colors.shape[1] != 3 else None,
        ),
        name='Point Cloud',
        showlegend=False
    )]
    
    # Add text labels if provided
    if labels and masks:
        for idx, (label, mask) in enumerate(zip(labels, masks)):
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            masked_coords = coords[mask_np]
            if len(masked_coords) > 0:
                # Calculate centroid of the mask
                centroid = masked_coords.mean(axis=0)
                traces.append(go.Scatter3d(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    z=[centroid[2]],
                    mode='text',
                    text=[label],
                    textposition='top center',
                    textfont=dict(
                        size=14,
                        color='black',
                        family='Arial Black'
                    ),
                    name=f'Label {idx+1}',
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=label
                ))
    
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        height=800,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def get_initial_colors(data, domain):
    """Get initial colors for point cloud visualization."""
    coords_len = len(data['coord'])
    if 'color' in data:
        colors = data['color'].cpu().numpy()
        if domain in ["Indoor", "Aerial"]:
            return np.clip((colors + 1) * 127.5, 0, 255).astype(np.uint8) / 255.0
        return colors
    elif 'strength' in data:
        return np.tile(data['strength'].cpu().numpy(), (1, 3))
    return np.tile([0.7, 0.7, 0.7], (coords_len, 1))


# ==============================================================================
# 5. GRADIO APPLICATION
# ==============================================================================

class SNAPGradioApp:
    """Main application class for SNAP Gradio interface."""
    
    def __init__(self, args, point_cloud_data):
        self.args = args
        self.point_cloud_loader = PointCloudData()
        print("Initializing segmentation model...")
        self.model = SegmentationModel(args, args.resume, domain=args.domain)
        self.model_data = self.model.intialize_pointcloud(point_cloud_data)
        
        print("Extracting point cloud features...")
        self.model.extract_backbone_features(self.model_data)
        
        self.coords = self.model_data['coord'].cpu().numpy()
        self.original_colors = get_initial_colors(self.model_data, args.domain)
        
        # State variables
        self.cached_masks = None
        self.cached_tokens = None
        self.cached_labels = None
        
        print("Model ready!")
    
    def load_new_pointcloud(self, uploaded_file, progress=gr.Progress()):
        """Load a new point cloud from an uploaded file."""
        if uploaded_file is None:
            return None, "⚠️ Please upload a point cloud file."
        
        try:
            progress(0, desc="Loading point cloud...")
            point_cloud_data = self.point_cloud_loader.load_point_cloud(uploaded_file.name)
            
            progress(0.3, desc="Initializing point cloud...")
            self.model_data = self.model.intialize_pointcloud(point_cloud_data)
            
            progress(0.6, desc="Extracting features...")
            self.model.extract_backbone_features(self.model_data)
            
            self.coords = self.model_data['coord'].cpu().numpy()
            self.original_colors = get_initial_colors(self.model_data, self.args.domain)
            
            # Reset cached results
            self.cached_masks = None
            self.cached_tokens = None
            
            progress(1.0, desc="Complete!")
            
            fig = create_plotly_figure(self.coords, self.original_colors, 
                                       "Uploaded Point Cloud")
            return fig, f"✅ Successfully loaded {len(self.coords)} points from uploaded file."
        except Exception as e:
            return None, f"❌ Error loading point cloud: {str(e)}"
    
    def show_original(self):
        """Display the original point cloud."""
        fig = create_plotly_figure(self.coords, self.original_colors, 
                                   "Original Point Cloud")
        return fig, "Showing original point cloud"
    
    def segment_everything(self, progress=gr.Progress()):
        """Segment all objects in the point cloud."""
        progress(0, desc="Running segmentation...")
        
        masks, tokens, iou_out, prompt_points = self.model.segment_everything(
            self.model_data, return_tokens=True)
        
        if not masks:
            return None, "❌ Could not generate any masks."
        
        # Cache masks and tokens for text search
        self.cached_masks = masks
        self.cached_tokens = [t.cpu() for t in tokens]
        
        progress(0.5, desc="Generating labels...")
        
        # Get text labels
        txt_vocab = torch.cat([clip.tokenize(f"a point cloud of a {c}") 
                               for c in self.model.dataset.labels]).to(self.model.device)
        with torch.no_grad():
            txt_feats_vocab = self.model.clip_model.encode_text(txt_vocab)
        text_labels = [self.model.get_text_label(txt_feats_vocab, t.unsqueeze(0), 
                                                 self.model.dataset.labels)[0] 
                      for t in tokens]
        
        # Cache labels for export
        self.cached_labels = text_labels
        
        progress(0.8, desc="Visualizing...")
        
        # Create colored visualization
        color_array = self.original_colors.copy()
        for mask in masks:
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            color_array[mask_np] = np.random.rand(3)
        
        # Create figure with labels
        fig = create_plotly_figure(self.coords, color_array, 
                                   f"Segmentation Result ({len(masks)} objects)",
                                   labels=text_labels, masks=masks)
        
        # Create summary
        summary = f"✅ Segmented {len(masks)} objects:\n\n"
        for i, (label, iou) in enumerate(zip(text_labels, iou_out)):
            summary += f"• Object {i+1}: {label} (IoU: {iou[0]:.3f})\n"
        summary += "\n💾 Click 'Export to GLB' to download the segmented point cloud."
        
        return fig, summary
    
    def segment_from_text(self, text_prompt, progress=gr.Progress()):
        """Find and highlight the best matching object based on text prompt."""
        if not text_prompt:
            return None, "⚠️ Please enter a text prompt."
        
        if self.cached_masks is None:
            return None, "⚠️ Please run 'Segment Everything' first!"
        
        progress(0, desc="Searching for best match...")
        
        best_idx, best_score = self.model.find_best_mask_from_text(
            self.cached_tokens, text_prompt)
        
        if best_idx == -1:
            return None, "❌ No matching object found."
        
        progress(0.5, desc="Visualizing result...")
        
        # Highlight the best matching mask
        color_array = self.original_colors.copy() * 0.3  # Dim background
        best_mask = self.cached_masks[best_idx]
        mask_np = best_mask.cpu().numpy() if isinstance(best_mask, torch.Tensor) else best_mask
        color_array[mask_np] = np.array([1.0, 0.2, 0.2])  # Bright red
        
        # Create figure with label on the best match
        fig = create_plotly_figure(self.coords, color_array, 
                                   f"Text Query Result: '{text_prompt}'",
                                   labels=[text_prompt], masks=[best_mask])
        
        summary = f"✅ Best match: Object #{best_idx + 1}\n"
        summary += f"Similarity score: {best_score:.3f}"
        
        return fig, summary
    
    def export_to_glb(self, progress=gr.Progress()):
        """Export segmented point cloud to GLB format with each segment as a separate object."""
        if self.cached_masks is None or len(self.cached_masks) == 0:
            return None, "⚠️ Please run 'Segment Everything' first!"
        
        progress(0, desc="Preparing export...")
        
        try:
            # Create a scene to hold all segments
            scene = trimesh.Scene()
            
            progress(0.1, desc="Identifying unclassified points...")
            
            # Create a mask for all classified points
            all_classified = np.zeros(len(self.coords), dtype=bool)
            for mask in self.cached_masks:
                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                all_classified |= mask_np
            
            # Get unclassified points
            unclassified_mask = ~all_classified
            num_unclassified = unclassified_mask.sum()
            
            progress(0.2, desc="Exporting segments...")
            
            # Export each segment as a separate object
            for idx, mask in enumerate(self.cached_masks):
                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                segment_coords = self.coords[mask_np]
                
                if len(segment_coords) == 0:
                    continue
                
                # Get colors for this segment (use original colors or generate new ones)
                segment_colors = self.original_colors[mask_np]
                
                # Convert to 0-255 range for trimesh
                if segment_colors.max() <= 1.0:
                    segment_colors = (segment_colors * 255).astype(np.uint8)
                else:
                    segment_colors = segment_colors.astype(np.uint8)
                
                # Create point cloud for this segment
                points = trimesh.PointCloud(
                    vertices=segment_coords,
                    colors=segment_colors
                )
                
                # Add label as metadata
                label = self.cached_labels[idx] if self.cached_labels and idx < len(self.cached_labels) else f"Object_{idx+1}"
                points.metadata['name'] = label
                
                # Add to scene with unique name
                scene.add_geometry(points, node_name=f"{label}_{idx}")
                
                progress((idx + 1) / len(self.cached_masks) * 0.6 + 0.2, 
                        desc=f"Exporting segment {idx+1}/{len(self.cached_masks)}...")
            
            # Add unclassified points if any exist
            if num_unclassified > 0:
                progress(0.85, desc="Exporting unclassified points...")
                
                unclassified_coords = self.coords[unclassified_mask]
                unclassified_colors = self.original_colors[unclassified_mask]
                
                # Convert to 0-255 range for trimesh
                if unclassified_colors.max() <= 1.0:
                    unclassified_colors = (unclassified_colors * 255).astype(np.uint8)
                else:
                    unclassified_colors = unclassified_colors.astype(np.uint8)
                
                # Create point cloud for unclassified points
                unclassified_points = trimesh.PointCloud(
                    vertices=unclassified_coords,
                    colors=unclassified_colors
                )
                unclassified_points.metadata['name'] = "Unclassified"
                
                # Add to scene
                scene.add_geometry(unclassified_points, node_name="Unclassified_Background")
            
            progress(0.9, desc="Writing GLB file...")
            
            # Export to GLB
            output_path = "segmented_pointcloud.glb"
            scene.export(output_path)
            
            progress(1.0, desc="Export complete!")
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            message = f"✅ Exported {len(self.cached_masks)} segments to GLB\n"
            if num_unclassified > 0:
                message += f"📍 Includes {num_unclassified:,} unclassified points\n"
            message += f"📦 File size: {file_size:.2f} MB\n"
            message += f"📄 Each segment is a separate object with label metadata"
            
            return output_path, message
            
        except Exception as e:
            return None, f"❌ Export failed: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="SNAP Interactive Segmentation", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🎯 SNAP: Interactive 3D Point Cloud Segmentation
            
            **Instructions:**
            1. Click "Show Original" to view the point cloud
            2. Click "Segment Everything" to automatically detect all objects
            3. Use "Segment from Text" to search for specific objects by description
            
            *Running on: {}*
            """.format("GPU 🚀" if torch.cuda.is_available() else "CPU 🐢"))
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📂 Upload Point Cloud")
                    file_upload = gr.File(
                        label="Upload Point Cloud",
                        file_types=[".ply", ".pcd", ".bin", ".pkl"],
                        type="filepath"
                    )
                    btn_load = gr.Button("📤 Load Uploaded File", variant="primary", size="lg")
                    
                    gr.Markdown("### 🎮 Controls")
                    
                    btn_show_original = gr.Button("🔍 Show Original", variant="secondary", size="lg")
                    btn_segment_all = gr.Button("🎯 Segment Everything", variant="primary", size="lg")
                    
                    gr.Markdown("### 🔤 Text Search")
                    text_input = gr.Textbox(
                        label="Object Description",
                        placeholder="e.g., 'chair', 'car', 'tree', 'building'...",
                        lines=1
                    )
                    btn_segment_text = gr.Button("🔎 Segment from Text", variant="primary", size="lg")
                    
                    gr.Markdown("### � Export")
                    btn_export = gr.Button("📦 Export to GLB", variant="secondary", size="lg")
                    file_output = gr.File(label="Download Segmented Point Cloud")
                    
                    gr.Markdown("### �📊 Information")
                    status_box = gr.Textbox(
                        label="Status",
                        lines=10,
                        max_lines=15,
                        value="Ready! Click a button to start.",
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    plot_output = gr.Plot(label="3D Visualization")
            
            # Connect buttons to functions
            btn_load.click(
                fn=self.load_new_pointcloud,
                inputs=[file_upload],
                outputs=[plot_output, status_box]
            )
            
            btn_show_original.click(
                fn=self.show_original,
                inputs=[],
                outputs=[plot_output, status_box]
            )
            
            btn_segment_all.click(
                fn=self.segment_everything,
                inputs=[],
                outputs=[plot_output, status_box]
            )
            
            btn_segment_text.click(
                fn=self.segment_from_text,
                inputs=[text_input],
                outputs=[plot_output, status_box]
            )
            
            btn_export.click(
                fn=self.export_to_glb,
                inputs=[],
                outputs=[file_output, status_box]
            )
            
            gr.Markdown("""
            ---
            **Tips:**
            - Rotate: Click and drag
            - Zoom: Scroll wheel or pinch
            - Pan: Right-click and drag
            """)
        
        return demo


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================

def main(args):
    """Main function to set up and run the Gradio application."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load point cloud
    print(f"Loading point cloud from: {args.pc_path}")
    point_cloud_loader = PointCloudData()
    point_cloud_data = point_cloud_loader.load_point_cloud(args.pc_path)
    print(f"Loaded {len(point_cloud_data['coord'])} points")
    
    # Create and launch app
    app = SNAPGradioApp(args, point_cloud_data)
    demo = app.create_interface()
    
    print(f"\n🚀 Launching Gradio interface on port {args.port}...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
