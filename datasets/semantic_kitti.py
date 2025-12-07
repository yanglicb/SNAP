"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

Modifying this to get object-wise segmentation masks.
We have segmentation labels for each point, so here we will choose 64 random points from the dataset, 
for each point, whatever is the class label, we will take all the points belonging to that class label 
from the point cloud and make a segmentation mask for it. 
"""

import os
import numpy as np
# import pickle

from .defaults import DefaultDataset_new

class SemanticKITTIDataset(DefaultDataset_new):
    def __init__(
        self,
        split="train",
        data_root="data/semantic_kitti",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        skip = 1, 
        num_prompt_points = 32,
        num_object_points = 5,
        overfit = False,
        use_random_clicks=False,
        use_centroid=False,
    ):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        self.skip=skip

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

    def get_data_list(self):
        if self.overfit:
            split2seq = dict(
                train=[4],
                val=[4],
                test=[4],
            )
        else:
            split2seq = dict(
                train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
                val=[8],
                test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files_all = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            seq_files = seq_files_all[0::self.skip]
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".txt")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                # segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                label = np.loadtxt(a, dtype=np.int32)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    label & 0xFFFF
                ).astype(np.int32)
                panoptic_label = label
        else:
            print("label file not ffound")
            segment = np.zeros(scan.shape[0]).astype(np.int32)

        data_dict = dict(coord=coord, strength=strength, segment=segment, panoptic=panoptic_label)
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
        panoptic_label = data_dict['panoptic']
        coord = data_dict['coord']
        grid_coord = data_dict['grid_coord']
        strength = data_dict['strength']
        condition = data_dict['condition']
        domain = data_dict['domain']

        # Choose random points for point prompting
        # Find indices of points with non-ignore segmentation labels
        valid_indices = np.where(segment != self.ignore_index)[0]

        # For all unique panoptic labels in the data, get random points on as many labels as possible
        # This is done to ensure that we have a good distribution of points from all the classes
        if self.split == 'train': #or self.split == 'val':
            valid_panoptic_labels = panoptic_label[valid_indices]
            if np.unique(valid_panoptic_labels).shape[0] <= self.random_points:
                # If the number of unique panoptic labels is less than the number of random points choose a point on each label and repeat
                random_indices = []
                for label in np.unique(valid_panoptic_labels):
                    if label==0:
                        continue
                    point_idxs = np.where(panoptic_label == label)[0]
                    random_indices.append(np.random.choice(point_idxs, 1, replace=False))
                    
                random_indices = np.array(random_indices)

            else:
                # If the number of unique panoptic labels is greater than the number of random points, choose a point on each label
                random_indices = []
                i = 0
                unique_valid_panoptic_labels = np.unique(valid_panoptic_labels)
                np.random.shuffle(unique_valid_panoptic_labels)
                for label in unique_valid_panoptic_labels:
                    if label==0:
                        continue
                    point_idxs = np.where(panoptic_label == label)[0]
                    random_indices.append(np.random.choice(point_idxs, 1, replace=False))
                    i+=1
                    if i == self.random_points:
                        break
                random_indices = np.array(random_indices)
        
        elif self.split == 'val':
            valid_panoptic_labels = panoptic_label[valid_indices]
            unique_valid_panoptic_labels = np.unique(valid_panoptic_labels)

            # Exclude label == 0 (assuming 0 is background or invalid label)
            unique_valid_panoptic_labels = unique_valid_panoptic_labels[unique_valid_panoptic_labels != 0]
            random_indices = []
            for label in unique_valid_panoptic_labels:
                point_idxs = np.where(panoptic_label == label)[0]
                random_indices.append(np.random.choice(point_idxs, 1, replace=False))
            random_indices = np.array(random_indices)

            # print("Total objects: ", random_indices.shape[0])
                
        random_indices = random_indices.reshape(len(random_indices), )
        
        # Select the corresponding coordinates and their panoptic labels
        random_coord = coord[random_indices]
        random_instances = panoptic_label[random_indices]
        random_seg_labels = segment[random_indices]

        # For each random point selected, we mark their corresponding instance with its idx
        # In this way we generate a single ground truth mask for the whole scene instead of
        # multiple binary masks
        instance_mask = np.zeros_like(panoptic_label)
        masks = []
        mask_labels = []
        class_weights = [1.0]
        prompt_points = []

        for i in range(len(random_indices)):
            label = random_instances[i] # (instance_id, semantic_label)
            seg_label = random_seg_labels[i]
            point_idxs = np.where(panoptic_label == label)[0]
            if self.split=="val" and len(point_idxs) < 10:
                continue
            instance_mask[point_idxs] = i + 1 # selected instance id in this scene

            obj_coord = coord[point_idxs]

            if self.use_centroid:
                centroid = np.mean(obj_coord, axis=0)
                distances = np.linalg.norm(obj_coord - centroid, axis=1)
                closest_idx = np.argmin(distances)
                closest_point = obj_coord[closest_idx]

                remaining_idxs = np.delete(np.arange(len(obj_coord)), closest_idx)
                try:
                    sampled_idxs = np.random.choice(remaining_idxs, self.num_object_points - 1, replace=True)
                except:
                    print(remaining_idxs)
                    print(self.num_object_points - 1)
                sampled_points = np.vstack([closest_point, obj_coord[sampled_idxs]])
            
            else:
                sampled_points = obj_coord[np.random.choice(len(obj_coord), self.num_object_points, replace=True)]

            prompt_points.append(sampled_points)


            # # Sample P prompt points on the same mask
            # prompt_points.append(coord[np.random.choice(point_idxs, self.num_object_points, replace=True)])

            binary_mask = np.zeros_like(panoptic_label)
            binary_mask[point_idxs] = 1
            masks.append(binary_mask)

            # mask_labels.append(label & 0xFFFF)
            mask_labels.append(seg_label)
            class_weights.append(self.get_class_weights(seg_label))

        prompt_points = np.array(prompt_points)

        # bg_mask = (instance_mask == 0)
        # masks.insert(0, bg_mask)

        masks = np.array(masks)
        mask_labels = np.array(mask_labels)
        class_weights = np.array(class_weights)
        data_dict = dict(coord=coord, grid_coord=grid_coord, strength=strength, segment=segment, point=prompt_points, 
            condition=condition, domain=domain, mask=instance_mask, masks=masks, mask_labels=mask_labels, weights=class_weights)

        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    def get_class_weights(self, idx):
        weights = {
            0: 3.1557,
            1: 8.7029,
            2: 7.8281,
            3: 6.1354,
            4: 6.3161,
            5: 7.9937,
            6: 8.9704,
            7: 10.1922,
            8: 1.6155,
            9: 4.2187,
            10: 1.9385,
            11: 5.5455,
            12: 2.0198,
            13: 2.6261,
            14: 1.3212,
            15: 5.1102,
            16: 2.5492,
            17: 5.8585,
            18: 7.3929,
        }
        return weights[idx]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }
        return learning_map_inv

    def class_labels(self):
        # class_labels = {
        #     0: "ignore_index (unlabeled)",
        #     1: "ignore_index (outlier)", #mapped to "unlabeled" --------------------------mapped
        #     10: "car",
        #     11: "bicycle",
        #     13: "bus", # mapped to "other-vehicle" --------------------------mapped
        #     15: "motorcycle",
        #     16: "on-rails", # mapped to "other-vehicle" ---------------------mapped
        #     18: "truck",
        #     20: "other-vehicle",
        #     30: "person",
        #     31: "bicyclist",
        #     32: "motorcyclist",
        #     40: "road",
        #     44: "parking",
        #     48: "sidewalk",
        #     49: "other-ground",
        #     50: "building",
        #     51: "fence",
        #     52: "ignore_index (other-structure)", # mapped to "unlabeled" ------------------mapped
        #     60: "lane-marking", # to "road" ---------------------------------mapped
        #     70: "vegetation",
        #     71: "trunk",
        #     72: "terrain",
        #     80: "pole",
        #     81: "traffic-sign",
        #     99: "ignore_index (other-object)",  # "other-object" to "unlabeled" ----------------------------mapped
        #     252: "moving-car", # to "car" ------------------------------------mapped
        #     253: "moving-bicyclist", # to "bicyclist" ------------------------mapped
        #     254: "moving-person", # to "person" ------------------------------mapped
        #     255: "moving-motorcyclist", # to "motorcyclist" ------------------mapped
        #     256: "moving-on-rails", # mapped to "other-vehicle" --------------mapped
        #     257: "moving-bus", # mapped to "other-vehicle" -------------------mapped
        #     258: "moving-truck", # to "truck" --------------------------------mapped
        #     259: "moving-other" # vehicle to "other-vehicle" ----------------mapped
        # }

        class_labels = {
            0: "car",           #  10 
            1: "bicycle",       #  11 
            2: "motorcycle",    #  15 
            3: "truck",         #  18 
            4: "other-vehicle", #  20 
            5: "person",        #  30 
            6: "bicyclist",     #  31 
            7: "motorcyclist",  #  32 
            8: "road",          #  40 
            9: "parking",       #  44 
            10: "sidewalk",     #  48 
            11: "other-ground", #  49 
            12: "building",     #  50 
            13: "fence",        #  51 
            14: "vegetation",   #  70 
            15: "trunk",        #  71 
            16: "terrain",      #  72 
            17: "pole",         #  80 
            18: "traffic-sign", #  81 
        }
        return class_labels
