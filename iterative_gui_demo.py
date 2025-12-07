# gui_demo.py (Modified)

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
# --- (Standard Library Imports) ---
import sys
import os
import argparse
import logging

# --- (Scientific Computing and Data Handling Imports) ---
import numpy as np
import open3d as o3d  # For 3D point cloud processing
import pandas as pd
import pyvista as pv   # For 3D visualization and plotting

# --- (Deep Learning Imports) ---
import torch
import torch.nn.functional as F
import clip             # OpenAI's CLIP model for text and image embeddings

# --- (Project-Specific SNAP Model Imports) ---
from src.snap import SNAP
from utils.torch_helpers import all_to_device
from datasets.demo import DemoDatset

# --- (GUI and Plotting Imports) ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLineEdit, QLabel,
                             QGroupBox, QTextEdit)
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
from pyvistaqt import QtInteractor # PyVista widget for embedding in a PyQt application

# ==============================================================================
# 2. HELPER FUNCTIONS AND DATA CLASSES
# ==============================================================================

def get_args_parser():
    """
    Sets up and parses command-line arguments for the application.
    """
    parser = argparse.ArgumentParser(description="SNAP Interactive GUI")
    parser.add_argument('--mask_threshold', default=0.5, type=float, help="Threshold for converting sigmoid output to a binary mask.")
    parser.add_argument('--resume', default="checkpoints/scannet_kitti_epoch_24.pth", type=str, help="Path to the pre-trained model checkpoint.")
    parser.add_argument('--pc_path', default="data_examples/PandaSet/30.pkl", type=str, help="Path to the input point cloud file.")
    parser.add_argument('--domain', default='Outdoor', type=str, help="Domain of the point cloud ('Outdoor', 'Indoor', 'Aerial').")
    parser.add_argument('--grid_size', default=0.05, type=float, help="Voxel grid size for initial point cloud processing.")
    parser.add_argument('--seed', default=42, type=int, help="Random seed for reproducibility.")
    parser.add_argument('--num_object_chunk', default=16, type=int, help="Batch size for processing prompt points during inference.")
    return parser

def non_maximum_suppression_masks(masks, scores, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) on a set of masks to remove redundant, overlapping detections.
    """
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
        iou = torch.where(union > 0, intersection / union, torch.tensor(0.0, device=masks.device)).squeeze(0)
        
        suppress_mask_indices = other_indices[iou > iou_threshold]
        suppressed[suppress_mask_indices] = True
        
    return torch.tensor(keep_indices, dtype=torch.long, device=masks.device)

def voxel_downsample(points, voxel_size):
    """
    Downsamples a point cloud by taking the point closest to the centroid of each voxel.
    """
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
        """
        Loads a point cloud from a file, supporting multiple formats.
        """
        data = {}
        if filename.lower().endswith(('.ply', '.pcd')):
            pcd = o3d.io.read_point_cloud(filename)
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
    """
    A wrapper class for the SNAP segmentation model.
    """
    def __init__(self, args, checkpoint_path=None, domain="Outdoor", grid_size=0.05):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.domain = domain
        self.dataset = DemoDatset(domain=self.domain)
        
        self.model = SNAP(num_points=1, num_merge_blocks=1, use_pdnorm=True, return_mid_points=True).to(self.device)
        self.model.eval()
        
        if checkpoint_path:
            self.loadweights(checkpoint_path)
            
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.point_features = None

    def loadweights(self, path):
        state_dict = torch.load(path, map_location=self.device)['model']
        self.model.load_state_dict(state_dict, strict=False)

    def intialize_pointcloud(self, d):
        return self.dataset.process_data(d.get('coord'), d.get('color'), d.get('normal'), d.get('intensity'))

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
            if len(uncovered_pts) < min_pts: break

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
            
        if not all_masks: return [], [], [], []
        
        final_masks = torch.stack(all_masks)
        final_scores = torch.cat([torch.as_tensor(iou, device=orig_pts.device).view(-1) for iou in all_iou])
        keep = non_maximum_suppression_masks(final_masks, final_scores, nms_iou)
        
        return ([final_masks[i] for i in keep], [all_txt[i] for i in keep], [all_iou[i] for i in keep], [all_prompts[i] for i in keep])

    def find_best_mask_from_text(self, cached_tokens, text_prompt):
        if not cached_tokens: return -1
        
        with torch.no_grad():
            txt_in = clip.tokenize(text_prompt).to(self.device)
            prompt_feats = F.normalize(self.clip_model.encode_text(txt_in).to(torch.float32), p=2, dim=-1)
        
        mask_tokens = F.normalize(torch.stack(cached_tokens).to(self.device), p=2, dim=-1)
        scores = torch.mm(mask_tokens, prompt_feats.T)
        
        best_idx = torch.argmax(scores)
        print(f"Best mask #{best_idx.item()} with score {scores[best_idx].item():.4f}")
        return best_idx

    def segment(self, data, prompt_points, text_prompt, return_tokens=False):
        if isinstance(prompt_points, list):
            proc_pts = [np.mean(p, axis=0) for p in prompt_points if p]
            if not proc_pts:
                return [], [], []
            prompt_points_tensor = torch.tensor(proc_pts, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            prompt_points_tensor = prompt_points

        txt_vocab = torch.cat([clip.tokenize(f"a point cloud of a {c}") for c in self.dataset.labels]).to(self.device)
        with torch.no_grad():
            txt_feats_vocab = self.clip_model.encode_text(txt_vocab)
            
        all_logits, all_txt, all_iou = [], [], []
        num_prompts, chunk_size = prompt_points_tensor.shape[0], self.args.num_object_chunk
        print(prompt_points_tensor)
        with torch.no_grad():
            for i in range(0, num_prompts, chunk_size):
                chunk = prompt_points_tensor[i:i + chunk_size]
                data_dict = {**data, "point": chunk, "point_offset": [chunk.shape[0]]}
                data_dict = all_to_device(data_dict, self.device)
                
                logits, txt, iou, _, _, _, _ = self.model.run_mask_decoder(self.point_features, data_dict)
                all_logits.append(logits[0])
                all_txt.append(txt[0])
                all_iou.append(iou[0])
        
        if not all_logits: return [], [], []
        
        final_logits = torch.cat(all_logits, dim=0)
        final_txt = torch.cat(all_txt, dim=0)
        final_iou = torch.cat(all_iou, dim=0)
        
        masks = (final_logits.sigmoid() > self.args.mask_threshold).cpu().numpy()
        
        if return_tokens:
            txt_list = [final_txt[i] for i in range(final_txt.shape[0])]
        else:
            txt_list = [self.get_text_label(txt_feats_vocab, t.unsqueeze(0), self.dataset.labels)[0] for t in final_txt]
            
        return list(masks), txt_list, list(final_iou.cpu().numpy())

# ==============================================================================
# 4. GUI AND PLOTTER CLASSES
# ==============================================================================

# A modern, clean stylesheet for the PyQt application.
MODERN_STYLESHEET = """
    QMainWindow, QWidget { background-color: #F5F5F5; } 
    QGroupBox { font-size: 14px; font-weight: bold; color: #333;}
    QLabel#title { font-family: Arial, sans-serif; color: #333;}
    QPushButton { 
        background-color: #007AFF; 
        color: white; 
        font-size: 16px; /* Larger font */
        font-weight: bold; 
        border: none; 
        border-radius: 8px; 
        padding: 12px 18px; /* Increased padding */
        margin: 5px 0;
    }
    QPushButton:hover { background-color: #005ECB; } 
    QPushButton:pressed { background-color: #004C9A; }
    QPushButton#clear { background-color: #E74C3C; } 
    QPushButton#clear:hover { background-color: #C0392B; }
    QLineEdit { 
        background-color: white; 
        color: #333; 
        border: 1px solid #E0E0E0; 
        border-radius: 8px; 
        padding: 10px; 
        font-size: 14px;
    }
    QTextEdit { 
        background-color: #FFFFFF; 
        color: #444; 
        border: 1px solid #E0E0E0; 
        border-radius: 8px; 
        font-family: Consolas, Courier New, monospace;
    }
"""

class Stream(QObject):
    """
    A custom stream object that redirects stdout to a PyQt signal.
    """
    new_text = pyqtSignal(str)
    
    def write(self, text):
        self.new_text.emit(str(text))
        
    def flush(self):
        pass

class MainWindow(QMainWindow):
    """The main application window."""
    def __init__(self, args, point_cloud_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SNAP: Interactive 3D Segmentation")
        self.setGeometry(100, 100, 1920, 1080)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_panel.setFixedWidth(350) # Increased width for larger buttons

        title_font = QFont("Arial", 24, QFont.Weight.Bold)
        self.title_label = QLabel("SNAP")
        self.title_label.setObjectName("title")
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_layout.addWidget(self.title_label)
        self.left_layout.addSpacing(15)

        # --- Action Buttons with Shortcuts ---
        self.btn_new_object = QPushButton("New Object (N)")
        self.btn_new_object.setShortcut("N")

        self.btn_segment_points = QPushButton("Finalize & Segment (F)")
        self.btn_segment_points.setShortcut("F")

        self.btn_segment_everything = QPushButton("Segment Everything (E)")
        self.btn_segment_everything.setShortcut("E")

        self.btn_segment_text = QPushButton("Segment from Text (T)")
        self.btn_segment_text.setShortcut("T")

        self.text_prompt_input = QLineEdit()
        self.text_prompt_input.setPlaceholderText("Enter text prompt...")
        self.left_layout.addWidget(self.btn_new_object)
        self.left_layout.addWidget(self.btn_segment_points)
        self.left_layout.addWidget(self.btn_segment_everything)
        self.left_layout.addWidget(self.btn_segment_text)
        self.left_layout.addWidget(self.text_prompt_input)
        self.left_layout.addSpacing(15)
        
        self.btn_clear_clicks = QPushButton("Clear Clicks (C)")
        self.btn_clear_clicks.setObjectName("clear")
        self.btn_clear_clicks.setShortcut("C")

        self.btn_clear_masks = QPushButton("Clear Masks (M)")
        self.btn_clear_masks.setObjectName("clear")
        self.btn_clear_masks.setShortcut("M")

        self.left_layout.addWidget(self.btn_clear_clicks)
        self.left_layout.addWidget(self.btn_clear_masks)
        
        console_group = QGroupBox("Console Output")
        console_layout = QVBoxLayout(console_group)
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        console_layout.addWidget(self.console_output)
        self.left_layout.addWidget(console_group)
        
        self.plotter_widget = QtInteractor(self.central_widget)
        
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.plotter_widget, 1)

        self.plotter_handler = InteractivePlotter(args, point_cloud_data, self.plotter_widget)
        self.plotter_handler.setup_scene()
        
        self.btn_new_object.clicked.connect(self.plotter_handler.new_object)
        self.btn_clear_clicks.clicked.connect(self.plotter_handler.clear_clicks)
        self.btn_clear_masks.clicked.connect(self.plotter_handler.clear_masks)
        self.btn_segment_points.clicked.connect(self.plotter_handler.segment_from_points)
        self.btn_segment_everything.clicked.connect(self.plotter_handler.segment_everything)
        self.btn_segment_text.clicked.connect(self.on_segment_from_text)
        
        self.stream = Stream()
        self.stream.new_text.connect(self.on_new_text)
        sys.stdout = self.stream

    def on_segment_from_text(self):
        prompt = self.text_prompt_input.text()
        if not prompt:
            print("Warning: Text prompt is empty.")
            return
        self.plotter_handler.visualize_best_mask(prompt)

    def on_new_text(self, text):
        self.console_output.moveCursor(QTextCursor.MoveOperation.End)
        self.console_output.insertPlainText(text)


class InteractivePlotter:
    """
    Handles PyVista window, user interactions, and iterative segmentation updates.
    """
    def __init__(self, args, point_cloud_data, plotter):
        self.args, self.plotter = args, plotter
        print("Initializing segmentation model...")
        self.model = SegmentationModel(args, args.resume, domain=args.domain)
        self.model_data = self.model.intialize_pointcloud(point_cloud_data)
        print("Extracting point cloud features...")
        self.model.extract_backbone_features(self.model_data)
        
        coords = self.model_data['coord'].cpu().numpy()
        self.pc = pv.PolyData(coords)
        self.vis_colors = self.get_initial_colors(self.model_data)
        self.pc["RGB"] = self.vis_colors
        
        # --- State Management for Iterative Segmentation ---
        self.current_clicks = []      # Clicks for the current object prompt
        self.all_click_sets = []      # A list of click sets (prompts)
        self.finalized_objects = []   # Stores finalized masks and their colors
        self.actor_names = {"spheres": [], "labels": []}
        self.cached_masks, self.cached_tokens = None, None
        print("Model ready. You can now interact with the scene.")
    
    def get_initial_colors(self, data):
        coords_len = len(data['coord'])
        if 'color' in data:
            colors = data['color'].cpu().numpy()
            if self.args.domain in ["Indoor", "Aerial"]: 
                return np.clip((colors + 1) * 127.5, 0, 255).astype(np.uint8) / 255.0
            return colors
        elif 'strength' in data: 
            return np.tile(data['strength'].cpu().numpy(), (1, 3))
        return np.tile([0.7, 0.7, 0.7], (coords_len, 1))
    
    def setup_scene(self):
        self.plotter.add_mesh(self.pc, scalars='RGB', rgb=True, pickable=True, name="point_cloud", point_size=5, render_points_as_spheres=True)
        self.plotter.enable_point_picking(callback=self.point_picking_callback, use_picker=True, show_point=False, color='yellow', point_size=15)
    
    def point_picking_callback(self, picked_point, point_index):
        """Called on every click, triggers a real-time segmentation update."""
        if point_index != -1:
            self.current_clicks.append(picked_point)
            
            # Visualize the click
            sphere_radius = 0.5 if self.args.domain == 'Outdoor' else 0.1
            actor_name = f"sphere_{len(self.actor_names['spheres'])}"
            self.plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=picked_point), color='yellow', name=actor_name)
            self.actor_names['spheres'].append(actor_name)
            
            # Update segmentation mask in real-time
            self.update_segmentation_preview()

    def new_object(self):
        """Finalizes the current set of clicks into a prompt for a new object."""
        if self.current_clicks:
            self.all_click_sets.append(list(self.current_clicks))
            self.current_clicks.clear()
            print(f"Prompt set {len(self.all_click_sets)} created. Click to start a new one.")
        else:
            print("No new clicks to form an object. Click on the scene first.")

    def update_segmentation_preview(self):
        """Redraws the point cloud with finalized masks and the current in-progress mask."""
        color_array = self.vis_colors.copy()

        # 1. Draw finalized objects with their stored colors
        for obj in self.finalized_objects:
            mask_color = obj['color'] * 0.7 + 0.3
            color_array[obj['mask']] = mask_color

        # 2. If there are active clicks, draw the current object's mask as a preview
        if self.current_clicks:
            # Combine current clicks with already defined sets for a full preview
            preview_sets = self.all_click_sets + [self.current_clicks]
            current_masks, _, _ = self.model.segment(self.model_data, preview_sets, text_prompt=None)
            
            for mask in current_masks:
                # Use a consistent color (e.g., light blue) for the active mask
                active_mask_color = np.array([0.5, 0.7, 1.0])
                color_array[mask] = active_mask_color
        
        # 3. Update the plotter
        self.pc['RGB'] = color_array
        self.plotter.update()

    def _clear_click_actors(self):
        for name in self.actor_names['spheres']: self.plotter.remove_actor(name, render=False)
        self.actor_names['spheres'].clear()
    
    def _clear_mask_actors(self):
        for name in self.actor_names['labels']: self.plotter.remove_actor(name, render=False)
        self.actor_names['labels'].clear()
        self.pc['RGB'] = self.vis_colors
        self.finalized_objects.clear() # Also clear the stored final masks
    
    def clear_clicks(self):
        self._clear_click_actors()
        self.current_clicks.clear()
        self.all_click_sets.clear()
        print("Cleared all click points and prompt sets.")
        self.update_segmentation_preview() # Redraw to remove preview mask
        self.plotter.update()
    
    def clear_masks(self):
        self._clear_mask_actors()
        print("Cleared all segmentation masks and labels.")
        self.plotter.update()
    
    def segment_from_points(self):
        """Finalizes the current clicks and then segments all created prompt sets."""
        self.new_object() # Finalize any clicks that haven't been grouped yet
        if not self.all_click_sets:
            print("No prompt sets created. Please pick points and use 'New Object' first.")
            return
        
        self._clear_mask_actors()
        self._clear_click_actors()
        
        print(f"Segmenting from {len(self.all_click_sets)} prompt set(s)...")
        masks, text_labels, iou_out = self.model.segment(self.model_data, self.all_click_sets, text_prompt=None)
        
        # Store and visualize the final masks
        self.finalized_objects.clear()
        for mask in masks:
            self.finalized_objects.append({'mask': mask, 'color': np.random.rand(3)})
        
        self.update_segmentation_preview()
        self.visualize_labels(text_labels, iou_out, self.all_click_sets)
        
        # Clear the sets for the next operation
        self.all_click_sets.clear()

    def segment_everything(self):
        self._clear_mask_actors()
        self._clear_click_actors()
        print("Running 'Segment Everything'...")
        
        masks, tokens, iou_out, prompt_points = self.model.segment_everything(self.model_data, return_tokens=True)

        if not masks:
            print("Could not generate any masks.")
            return

        self.cached_masks = masks
        self.cached_tokens = [t.cpu() for t in tokens]
        print(f"Cached {len(self.cached_masks)} masks for text search.")

        txt_vocab = torch.cat([clip.tokenize(f"a point cloud of a {c}") for c in self.model.dataset.labels]).to(self.model.device)
        with torch.no_grad():
            txt_feats_vocab = self.model.clip_model.encode_text(txt_vocab)
        text_labels = [self.model.get_text_label(txt_feats_vocab, t.unsqueeze(0), self.model.dataset.labels)[0] for t in tokens]
        
        self.visualize_masks_and_labels(masks, text_labels, iou_out, prompt_points)

    def visualize_masks_and_labels(self, masks, labels, ious, prompt_sets=None):
        """Visualizes masks and labels for segment_everything."""
        color_array = self.vis_colors.copy()
        for mask in masks:
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            color_array[mask_np] = np.random.rand(3) * 0.7 + 0.3
        self.pc['RGB'] = color_array
        self.visualize_labels(labels, ious, prompt_sets)

    def visualize_labels(self, labels, ious, prompt_sets):
        """Adds text labels to the scene."""
        if prompt_sets:
            for idx, points in enumerate(prompt_sets):
                center_point = np.mean(points, axis=0)
                if labels and ious and idx < len(labels):
                    label_name = f"label_{idx}"
                    self.plotter.add_point_labels(
                        center_point, 
                        [f"{labels[idx]} ({ious[idx][0]:.2f})"], 
                        name=label_name, 
                        text_color='black', 
                        font_size=12, 
                        shape=None, 
                        always_visible=True
                    )
                    self.actor_names['labels'].append(label_name)

    def _ensure_cache_is_built(self):
        if self.cached_masks is None:
            print("Cache is empty. Running 'Segment Everything' first to build it.")
            self.segment_everything()
        return self.cached_masks is not None

    def visualize_best_mask(self, text_prompt):
        if not self._ensure_cache_is_built():
            print("Cache building failed. Cannot perform text search.")
            return
        
        self._clear_mask_actors()
        self._clear_click_actors()

        best_mask_idx = self.model.find_best_mask_from_text(self.cached_tokens, text_prompt)
        if best_mask_idx == -1: return

        best_mask = self.cached_masks[best_mask_idx.item()]
        color_array = self.vis_colors.copy()
        color_array[best_mask.cpu().numpy()] = np.array([1.0, 0.2, 0.2]) # Bright red
        self.pc['RGB'] = color_array
        self.plotter.update()

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main(args):
    """The main function to set up and run the application."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    app = QApplication(sys.argv)
    app.setStyleSheet(MODERN_STYLESHEET)
    
    point_cloud_loader = PointCloudData()
    point_cloud_data = point_cloud_loader.load_point_cloud(args.pc_path)
    
    window = MainWindow(args, point_cloud_data)
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    pv.set_plot_theme("document")
    pv.global_theme.background = 'white'
    
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
