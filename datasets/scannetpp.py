import os
import numpy as np
import torch

from .defaults import DefaultDataset_new
import pdb


class ScanNetPPDataset(DefaultDataset_new):
    def __init__(
        self,
        split="train",
        data_root="/Data/mnt/scannet",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        skip=False,
        num_prompt_points=8,
        num_object_points=5,
        agile3d_val=False,
        overfit=False,
        use_random_clicks=True,
        use_centroid=False,
    ):
        self.split = split
        self.random_points = num_prompt_points
        self.num_object_points = num_object_points
        self.agile3d_val = agile3d_val

        self.data_root = data_root
        self.split = split
        self.use_centroid = use_centroid

        if self.split == "train":
            self.real_split = "train_grid1mm_chunk6x6_stride3x3"
        else:
            self.real_split = "val_grid1mm_chunk6x6_stride3x3"
        
        self.overfit = overfit
        self.data_list = self.get_data_list()

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
        split_path = os.path.join(self.data_root, self.real_split)
        data_list = os.listdir(split_path)
        return data_list
    
    def get_data(self, idx):

        scene_name = self.data_list[idx]

        coord = np.load(os.path.join(self.data_root, self.real_split, scene_name, "coord.npy"))
        color = np.load(os.path.join(self.data_root, self.real_split, scene_name, "color.npy"))
        normal = np.load(os.path.join(self.data_root, self.real_split, scene_name, "normal.npy"))
        instance = np.load(os.path.join(self.data_root, self.real_split, scene_name, "instance.npy"))[:, 0]
        segment = np.load(os.path.join(self.data_root, self.real_split, scene_name, "segment.npy"))[:, 0]

        data_dict = dict(coord=coord, color=color, normal=normal, instance=instance, segment=segment)
        return data_dict

    def process_test_data(self, data_dict):
        """
        data_dict contains the following keys after applying the transforms:
        1. coord
        2. colors
        3. labels: instance mask for the processed cloud after transforms
        """
        coord = data_dict["coord"]
        labels = data_dict["instance"]
        color = data_dict["color"]
        normal = data_dict["normal"]
        segment = data_dict["segment"]

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
            # sample P prompt points on the same mask
            # prompt_points.append(coord[np.random.choice(point_idxs, self.num_object_points, replace=True)])

            binary_mask = np.zeros_like(labels)
            binary_mask[point_idxs] = 1
            masks.append(binary_mask)
            mask_labels.append(seg_label)

        prompt_points = np.array(prompt_points)
        masks = np.array(masks)
        mask_labels = np.array(mask_labels)

        data_dict = dict(coord=coord, color=color, normal=normal, point=prompt_points, masks=torch.from_numpy(masks).long(), mask_labels=mask_labels)
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

        random_indices = []
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
            mask_labels.append(seg_label)

        prompt_points = np.array(prompt_points)
        masks = np.array(masks)
        mask_labels = np.array(mask_labels)
        data_dict = dict(coord=coord, grid_coord=grid_coord, color=color, normal=normal, point=prompt_points, 
                        condition=condition, domain=domain, masks=masks, mask_labels=mask_labels)

        return data_dict

    def class_labels(self):
        class_mapping = {}
        class_labels_100 = ["wall", "ceiling", "floor", "table", "door", "ceiling lamp", "cabinet", "blinds", "curtain", "chair", "storage cabinet", "office chair", 
                            "bookshelf", "whiteboard", "window", "box", "window frame", "monitor", "shelf", "doorframe", "pipe", "heater", "kitchen cabinet", "sofa", 
                            "windowsill", "bed", "shower wall", "trash can", "book", "plant", "blanket", "tv", "computer tower", "kitchen counter", "refrigerator", 
                            "jacket", "electrical duct", "sink", "bag", "picture", "pillow", "towel", "suitcase", "backpack", "crate", "keyboard", "rack", "toilet", 
                            "paper", "printer", "poster", "painting", "microwave", "board", "shoes", "socket", "bottle", "bucket", "cushion", "basket", "shoe rack", 
                            "telephone", "file folder", "cloth", "blind rail", "laptop", "plant pot", "exhaust fan", "cup", "coat hanger", "light switch", "speaker", 
                            "table lamp", "air vent", "clothes hanger", "kettle", "smoke detector", "container", "power strip", "slippers", "paper bag", "mouse", 
                            "cutting board", "toilet paper", "paper towel", "pot", "clock", "pan", "tap", "jar", "soap dispenser", "binder", "bowl", "tissue box", 
                            "whiteboard eraser", "toilet brush", "spray bottle", "headphones", "stapler", "marker"]
        for i, label in enumerate(class_labels_100):
            class_mapping[i] = label
        return class_mapping