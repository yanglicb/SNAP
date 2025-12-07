import os
import numpy as np
import torch

from .defaults import DefaultDataset_new
import pdb


class Matterport3DDataset(DefaultDataset_new):
    def __init__(
        self,
        split="train",
        data_root="/Data/mnt/scannet",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        num_prompt_points=8,
        num_object_points=5,
        overfit=False,
        use_centroid=False,
    ):
        self.split = split
        self.random_points = num_prompt_points
        self.num_object_points = num_object_points
        self.data_root = data_root
        self.split = split
        
        self.overfit = overfit
        self.data_list = self.get_data_list()
        self.use_centroid = use_centroid

        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

        if self.overfit:
            self.data_list = self.data_list[:10]

    def get_data_list(self):
        # get the path for each point cloud and store them in self.data_list
        split_path = os.path.join(self.data_root, self.split)
        data_list = os.listdir(split_path)
        return data_list
    
    def get_data(self, idx):

        scene_name = self.data_list[idx]

        coord = np.load(os.path.join(self.data_root, self.split, scene_name, "coord.npy"))
        color = np.load(os.path.join(self.data_root, self.split, scene_name, "color.npy"))
        normal = np.load(os.path.join(self.data_root, self.split, scene_name, "normal.npy"))
        instance = np.load(os.path.join(self.data_root, self.split, scene_name, "instance.npy"))
        segment = np.load(os.path.join(self.data_root, self.split, scene_name, "segment.npy"))
        
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
        normal = data_dict["normal"]
        condition = data_dict['condition']
        domain = data_dict['domain']
        segment = data_dict['segment']

        masks = []
        prompt_points = []
        mask_labels = []

        valid_indices = np.where(segment != -1)[0]
        unique_labels = np.unique(labels[valid_indices])

        if unique_labels[0] == -1:
            unique_labels = unique_labels[1:]

        np.random.shuffle(unique_labels)

        if self.split == "train":
            num_obj = min(self.random_points, len(unique_labels))
        else:
            num_obj = len(unique_labels)

        for i in range(num_obj):
            label = unique_labels[i]
            point_idxs = np.where(labels == label)[0]
            if self.split=="val" and len(point_idxs) < 10:
                continue
            seg_label = segment[point_idxs[0]]
            if seg_label == -1:
                continue
            
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
            mask_labels.append(seg_label)

        prompt_points = np.array(prompt_points)
        masks = np.array(masks)
        mask_labels = np.array(mask_labels)
        data_dict = dict(coord=coord, grid_coord=grid_coord, color=color, normal=normal, point=prompt_points, 
                        condition=condition, domain=domain, masks=masks, mask_labels=mask_labels)

        return data_dict

    def class_labels(self):
        CLASS_LABELS = [
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "desk",
            "curtain",
            "refrigerator",
            "shower curtain",
            "toilet",
            "sink",
            "bathtub",
            "other",
            "ceiling"
        ]
                
        class_mapping = {}
        for i in range(len(CLASS_LABELS)):
            class_mapping[i] = CLASS_LABELS[i]

        return class_mapping