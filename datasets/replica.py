import os
import numpy as np
import torch

from .defaults import DefaultDataset_new
import pdb
from glob import glob
import os.path as osp
import plyfile
import pandas as pd
import json


class ReplicaDataset(DefaultDataset_new):
    def __init__(
        self,
        split="train",
        data_root="",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        num_prompt_points=8,
        num_object_points=5,
        agile3d_val=False,
        overfit=False,
        use_centroid=False,
        run_openvocab_eval=False
    ):
        self.split = split
        self.random_points = num_prompt_points
        self.num_object_points = num_object_points
        self.agile3d_val = agile3d_val

        self.data_root = data_root
        self.split = split
        self.overfit = overfit
        self.data_list = self.get_data_list()
        self.use_centroid = use_centroid
        self.run_openvocab_eval=run_openvocab_eval

        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

        # Create learning map for class remapping
        self.class_remap = self.learning_map()

        if self.overfit:
            self.data_list = self.data_list[:10]

    def get_data_list(self):
        if self.split=="val":
            scenes = ["office_0", "office_1", "office_2", "office_3", "office_4", "room_0", "room_1", "room_2"]
            scene_dirs = [osp.join(self.data_root, scene) for scene in scenes]
        elif self.split=="train":
            scenes = ["apartment_0", "apartment_1", "apartment_2", "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
                      "frl_apartment_3", "frl_apartment_4", "frl_apartment_5", "hotel_0"]
            scene_dirs = [osp.join(self.data_root, scene) for scene in scenes]
            

        # For each scene, find the mesh files
        data_files = []
        for scene_dir in scene_dirs:
            # Check for mesh_semantic.ply in habitat folder
            mesh_file = osp.join(scene_dir, 'habitat', 'mesh_semantic.ply')
           
            if osp.exists(mesh_file):
                scene_name = osp.basename(scene_dir)
                data_files.append({
                    'scene_name': osp.basename(scene_dir),
                    'mesh_file': mesh_file,
                    'scene_dir': scene_dir,
                    'gt_file': osp.join(self.data_root, 'ground_truth', scene_name + '.txt') 
                })
        
        assert len(data_files) > 0, f'No valid scenes found in {self.data_root}'

        return data_files
    
    def get_data(self, idx):

        VALID_CLASS_IDS = np.array([
            3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 34, 
            35, 37, 44, 47, 52, 54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 
            79, 80, 82, 83, 87, 88, 91, 92, 95, 97, 98
        ])

        scene_info = self.data_list[idx]
        mesh_file = scene_info['mesh_file']
        scene_dir = scene_info['scene_dir']
        gt_file = scene_info['gt_file']

        # Load semantic information
        json_path = osp.join(scene_dir, 'habitat', 'info_semantic.json')
        with open(json_path, 'r') as f:
            semantic_info = json.load(f)
        
        # Create instance to class mapping
        instance_to_class = {}
        for obj in semantic_info.get('objects', []):
            instance_to_class[obj['id']] = obj['class_id']
        
        # Read mesh file
        plydata = plyfile.PlyData.read(mesh_file)
        vertices = plydata['vertex']
        faces = plydata['face']
        
        # Extract coordinates, colors and normals
        coord = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T.astype(np.float32)
        color = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T.astype(np.float32) / 255.0
        normal = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T.astype(np.float32)
        
        # Extract instance labels if available as numpy array
        instance = pd.read_csv(gt_file, header=None).values.flatten().astype(np.int32)
        
        # Map instance IDs to class IDs
        segment = np.full_like(instance, -1, dtype=np.int32)
        for i in range(len(instance)):
            class_id = instance[i]//1000
            if class_id in VALID_CLASS_IDS:
                if class_id in self.class_remap:
                    segment[i] = self.class_remap[class_id]

        data_dict = dict(coord=coord, color=color, normal=normal, instance=instance, segment=segment)
        return data_dict
    
    def process_data(self, data_dict):
        """
        data_dict contains the following keys after applying the transforms:
        1. coord
        2. grid_coord
        3. colors
        4. labels: instance mask for the processed cloud after transforms
        5. condition: "ScanNet"
        """

        labels = data_dict['instance']
        coord = data_dict['coord']
        grid_coord = data_dict['grid_coord']
        color = data_dict['color']
        normal = data_dict['normal']
        condition = data_dict['condition']
        domain = data_dict['domain']
        segment = data_dict['segment']

        random_indices = []
        masks = []
        prompt_points = []
        mask_labels = []

        ignore_idxs = np.where(segment == -1)[0]
        labels[ignore_idxs] = -1

        unique_labels = np.unique(labels)[1:]
        # unique_labels = np.unique(labels)
        np.random.shuffle(unique_labels)
        if self.split == "train":
            num_obj = min(self.random_points, len(unique_labels))
        else:
            num_obj = len(unique_labels)

        for i in range(num_obj):
            label = unique_labels[i]
            point_idxs = np.where(labels == label)[0]
            if self.split=="val" and len(point_idxs) < 2:
                continue
            seg_label = segment[point_idxs[0]]
            assert seg_label != -1

            obj_coord = coord[point_idxs]

            if self.use_centroid:
                centroid = np.mean(obj_coord, axis=0)
                distances = np.linalg.norm(obj_coord - centroid, axis=1)
                closest_idx = np.argmin(distances)
                closest_point = obj_coord[closest_idx]

                remaining_idxs = np.delete(np.arange(len(obj_coord)), closest_idx)
                sampled_idxs = np.random.choice(remaining_idxs, self.num_object_points - 1, replace=True)
                sampled_points = np.vstack([closest_point, obj_coord[sampled_idxs]])
            
            else:
                sampled_points = obj_coord[np.random.choice(len(obj_coord), self.num_object_points, replace=True)]

            prompt_points.append(sampled_points)

            binary_mask = np.zeros_like(labels)
            binary_mask[point_idxs] = 1
            masks.append(binary_mask)
            mask_labels.append(int(seg_label))

        prompt_points = np.array(prompt_points)
        masks = np.array(masks)
        mask_labels = np.array(mask_labels)
        data_dict = dict(coord=coord, grid_coord=grid_coord, color=color, normal=normal, point=prompt_points, 
                        condition=condition, domain=domain, masks=masks, mask_labels=mask_labels)

        return data_dict

    def learning_map(self):
        CLASS_LABELS = [
            "basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle", 
            "box", "bowl", "camera", "cabinet", "candle", "chair", "clock", "cloth", 
            "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant", 
            "lamp", "monitor", "nightstand", "panel", "picture", "pillar", "pillow", 
            "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa", 
            "stool", "switch", "table", "tablet", "tissue-paper", "tv-screen", 
            "tv-stand", "vase", "vent", "wall-plug", "window", "rug"
        ]
        VALID_CLASS_IDS = np.array([
            3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 34, 
            35, 37, 44, 47, 52, 54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 
            79, 80, 82, 83, 87, 88, 91, 92, 95, 97, 98
        ])

        learning_map = {}
        for i, class_id in enumerate(VALID_CLASS_IDS):
            learning_map[class_id] = i
        return learning_map
    
    def class_labels(self):
        class_mapping = {}

        CLASS_LABELS = [
            "basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle", 
            "box", "bowl", "camera", "cabinet", "candle", "chair", "clock", "cloth", 
            "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant", 
            "lamp", "monitor", "nightstand", "panel", "picture", "pillar", "pillow", 
            "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa", 
            "stool", "switch", "table", "tablet", "tissue-paper", "tv-screen", 
            "tv-stand", "vase", "vent", "wall-plug", "window", "rug"
        ]
        VALID_CLASS_IDS = np.array([
            3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 34, 
            35, 37, 44, 47, 52, 54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 
            79, 80, 82, 83, 87, 88, 91, 92, 95, 97, 98
        ])
        
        class_mapping = {}
        for i, class_name in enumerate(CLASS_LABELS):
            class_mapping[i] = class_name
        return class_mapping