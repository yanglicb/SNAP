import os
import numpy as np
import glob
from .defaults import DefaultDataset_new

class WaymoDataset(DefaultDataset_new):
    def __init__(
        self,
        split="train",
        data_root="data/waymo",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        num_prompt_points = 32,
        num_object_points = 5,
        overfit = False,
        use_random_clicks=False,
        use_centroid=False,
    ):
        self.ignore_index = ignore_index

        # Variable to define how many random points we are choosing inside the pointcloud
        self.random_points = num_prompt_points
        self.num_object_points = num_object_points
        self.overfit = overfit
        self.use_random_clicks = use_random_clicks
        self.split = split
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
        if self.split=="val":
            processed_data_path = os.path.join(self.data_root, "v2_0_1_processed", "val")
            data_list = glob.glob(os.path.join(processed_data_path, "*/*"))
            
        if self.split=="train":
            processed_data_path = os.path.join(self.data_root, "v2_0_1_processed", "val")
            data_list = glob.glob(os.path.join(processed_data_path, "*/*"))
            
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]

        # Load data
        coord = np.load(os.path.join(data_path, 'coord.npy'))
        strength = np.load(os.path.join(data_path, 'strength.npy')).reshape((-1,1))
        segment = np.load(os.path.join(data_path, 'segment.npy'))
        instance = np.load(os.path.join(data_path, 'instance.npy'))

        data_dict = dict(coord=coord, strength=strength, segment=segment, instance=instance)
        return data_dict

    def process_data(self, data_dict):
        """
        data_dict contains the following keys after applying the transforms:
        1. coord
        2. grid_coord
        3. strength
        4. segment: segmentation masks for the processed cloud after transforms
        5. panoptic: panoptic masks for the processed cloud after transforms
        """

        segment = data_dict['segment']
        labels = data_dict['instance']
        coord = data_dict['coord']
        grid_coord = data_dict['grid_coord']
        strength = data_dict['strength']
        condition = data_dict['condition']
        domain = data_dict['domain']

        masks = []
        mask_labels = []
        prompt_points = []

        unique_labels = np.unique(labels)
        # Remove the -1 labels from instances
        unique_labels = unique_labels[1:]

        if self.split == "train":
            np.random.shuffle(unique_labels)
            num_obj = min(self.random_points, len(unique_labels))
        else:
            num_obj = len(unique_labels)
        
        for i in range(num_obj):
            label = unique_labels[i]
            point_idxs = np.where(labels == label)[0]
            if len(point_idxs) < 10:
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
            binary_mask = np.zeros_like(labels)
            binary_mask[point_idxs] = 1
            masks.append(binary_mask)
            mask_labels.append(seg_label)

        prompt_points = np.array(prompt_points)
        masks = np.array(masks)
        mask_labels = np.array(mask_labels)
        data_dict = dict(coord=coord, grid_coord=grid_coord, strength=strength, point=prompt_points, condition=condition, domain=domain, masks=masks, mask_labels=mask_labels)

        return data_dict

    def class_labels(self):

        class_labels = {
            0: "Car",
            1: "Truck",
            2: "Bus",
            3: "Other Vehicle",
            4: "Motorcyclist",
            5: "Bicyclist",
            6: "Pedestrian",
            7: "Sign",
            8: "Traffic Light",
            9: "Pole",
            10: "Construction Cone",
            11: "Bicycle",
            12: "Motorcycle",
            13: "Building",
            14: "Vegetation",
            15: "Tree Trunk",
            16: "Curb",
            17: "Road",
            18: "Lane Marker",
            19: "Other Ground",
            20: "Walkable",
            21: "Sidewalk"
        }

        return class_labels
