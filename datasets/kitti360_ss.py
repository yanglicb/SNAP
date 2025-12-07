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
import glob
from .defaults import DefaultDataset_new

class KITTI360_SSDataset(DefaultDataset_new):
    def __init__(
        self,
        split="train",
        data_root="data/kitti360_ss",
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
        processed_data_path = os.path.join(self.data_root, "processed")

        if self.split=="val":
            val_processed_data_path = os.path.join(processed_data_path, "val")
            data_list = glob.glob(os.path.join(val_processed_data_path, "*/*/*.bin"))
        if self.split=="train":
            train_processed_data_path = os.path.join(processed_data_path, "val")  # Don't have the train data right now, load this to remove errors. 
            data_list = glob.glob(os.path.join(train_processed_data_path, "*/*/*.bin"))
        
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        
        label_path = data_path
        scan_path = data_path.replace("processed/val", "data_3d_raw").replace("labels", "velodyne_points/data")
        
        # Load the point cloud
        with open(scan_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        # Load the labels
        with open(label_path, "rb") as a:
            label = np.fromfile(a, dtype=np.int16).astype(np.int32)
            label[(label < 0) & (label != -1)] += 65536 
        
        # Get segment and instance ids
        instance_orig = label % 1000
        segment_orig = label // 1000

        segment = []
        instance = []
        for i in range(segment_orig.shape[0]):
            seg = segment_orig[i]
            inst = instance_orig[i]
            seg_new = self.label_conversion(seg)
            segment.append(seg_new)

            if seg_new == -1:
                instance.append(-1)
            else:
                instance.append(inst)

        segment = np.array(segment)
        instance = np.array(instance)

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
        if unique_labels[0] == -1:
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

            # # sample P prompt points on the same mask
            # prompt_points.append(coord[np.random.choice(point_idxs, self.num_object_points, replace=True)])

            binary_mask = np.zeros_like(labels)
            binary_mask[point_idxs] = 1
            masks.append(binary_mask)
            mask_labels.append(seg_label)

        prompt_points = np.array(prompt_points)
        masks = np.array(masks)
        mask_labels = np.array(mask_labels)
        data_dict = dict(coord=coord, grid_coord=grid_coord, strength=strength, point=prompt_points, condition=condition, domain=domain, masks=masks, mask_labels=mask_labels)

        return data_dict

    def label_conversion(self, idx):
        label_conversion_dict = {
            0: -1,
            1: -1,
            2: -1,
            3: -1,
            4: -1,
            5: -1,
            6: -1,
            7: -1, # road
            8: -1,
            9: -1,
            10: -1,
            11: -1,
            12: -1,
            13: -1,
            14: -1,
            15: -1,
            16: -1,
            17: -1,
            18: -1,
            19: -1,
            20: -1,
            21: -1,
            22: -1,
            23: -1,
            24: 0, # person
            25: 1, # rider
            26: 2, # car
            27: 3, # truck
            28: 4, # bus
            29: 5, # caravan
            30: 6, # trailer
            31: 7, # train
            32: 8, # motorcycle
            33: 9, # bicycle
            34: -1,
            35: -1,
            36: -1,
            37: -1,
            38: -1,
            39: -1,
            40: -1, 
            41: -1,
            42: -1, 
            43: -1, 
            44: -1,        
            -1: 10, # license plate
        }

        return label_conversion_dict[idx]

    def class_labels(self):
        # Label = namedtuple("Label", ["name", "id", "kittiId", "trainId", "category", "categoryId", "hasInstances", "ignoreInEval", "ignoreInInst", "color"])
        # labels_info = [
        #     # name. id. kittiId, trainId   category, catId, hasInstances, ignoreInEval, ignoreInInst, color
        #     Label("unlabeled", 0, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
        #     Label("ego vehicle", 1, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
        #     Label("rectification border", 2, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
        #     Label("out of roi", 3, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
        #     Label("static", 4, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
        #     Label("dynamic", 5, -1, 255, "void", 0, False, True, True, (111, 74, 0)),
        #     Label("ground", 6, -1, 255, "void", 0, False, True, True, (81, 0, 81)),
        #     Label("road", 7, 1, 0, "flat", 1, False, False, False, (128, 64, 128)),
        #     Label("sidewalk", 8, 3, 1, "flat", 1, False, False, False, (244, 35, 232)),
        #     Label("parking", 9, 2, 255, "flat", 1, False, True, True, (250, 170, 160)),
        #     Label("rail track", 10, 10, 255, "flat", 1, False, True, True, (230, 150, 140)),
        #     Label("building", 11, 11, 2, "construction", 2, True, False, False, (70, 70, 70)),
        #     Label("wall", 12, 7, 3, "construction", 2, False, False, False, (102, 102, 156)),
        #     Label("fence", 13, 8, 4, "construction", 2, False, False, False, (190, 153, 153)),
        #     Label("guard rail", 14, 30, 255, "construction", 2, False, True, True, (180, 165, 180)),
        #     Label("bridge", 15, 31, 255, "construction", 2, False, True, True, (150, 100, 100)),
        #     Label("tunnel", 16, 32, 255, "construction", 2, False, True, True, (150, 120, 90)),
        #     Label("pole", 17, 21, 5, "object", 3, True, False, True, (153, 153, 153)),
        #     Label("polegroup", 18, -1, 255, "object", 3, False, True, True, (153, 153, 153)),
        #     Label("traffic light", 19, 23, 6, "object", 3, True, False, True, (250, 170, 30)),
        #     Label("traffic sign", 20, 24, 7, "object", 3, True, False, True, (220, 220, 0)),
        #     Label("vegetation", 21, 5, 8, "nature", 4, False, False, False, (107, 142, 35)),
        #     Label("terrain", 22, 4, 9, "nature", 4, False, False, False, (152, 251, 152)),
        #     Label("sky", 23, 9, 10, "sky", 5, False, False, False, (70, 130, 180)),
        #     Label("person", 24, 19, 11, "human", 6, True, False, False, (220, 20, 60)),
        #     Label("rider", 25, 20, 12, "human", 6, True, False, False, (255, 0, 0)),
        #     Label("car", 26, 13, 13, "vehicle", 7, True, False, False, (0, 0, 142)),
        #     Label("truck", 27, 14, 14, "vehicle", 7, True, False, False, (0, 0, 70)),
        #     Label("bus", 28, 34, 15, "vehicle", 7, True, False, False, (0, 60, 100)),
        #     Label("caravan", 29, 16, 255, "vehicle", 7, True, True, True, (0, 0, 90)),
        #     Label("trailer", 30, 15, 255, "vehicle", 7, True, True, True, (0, 0, 110)),
        #     Label("train", 31, 33, 16, "vehicle", 7, True, False, False, (0, 80, 100)),
        #     Label("motorcycle", 32, 17, 17, "vehicle", 7, True, False, False, (0, 0, 230)),
        #     Label("bicycle", 33, 18, 18, "vehicle", 7, True, False, False, (119, 11, 32)),
        #     Label("garage", 34, 12, 2, "construction", 2, True, True, True, (64, 128, 128)),
        #     Label("gate", 35, 6, 4, "construction", 2, False, True, True, (190, 153, 153)),
        #     Label("stop", 36, 29, 255, "construction", 2, True, True, True, (150, 120, 90)),
        #     Label("smallpole", 37, 22, 5, "object", 3, True, True, True, (153, 153, 153)),
        #     Label("lamp", 38, 25, 255, "object", 3, True, True, True, (0, 64, 64)),
        #     Label("trash bin", 39, 26, 255, "object", 3, True, True, True, (0, 128, 192)),
        #     Label("vending machine", 40, 27, 255, "object", 3, True, True, True, (128, 64, 0)),
        #     Label("box", 41, 28, 255, "object", 3, True, True, True, (64, 64, 128)),
        #     Label("unknown construction", 42, 35, 255, "void", 0, False, True, True, (102, 0, 0)),
        #     Label("unknown vehicle", 43, 36, 255, "void", 0, False, True, True, (51, 0, 51)),
        #     Label("unknown object", 44, 37, 255, "void", 0, False, True, True, (32, 32, 32)),
        #     Label("license plate", -1, -1, -1, "vehicle", 7, False, True, True, (0, 0, 142)),
        # ]

        class_labels = {
            0: "person",
            1: "rider",
            2: "car",
            3: "truck",
            4: "bus",
            5: "caravan",
            6: "trailer",
            7: "train",
            8: "motorcycle",
            9: "bicycle",
            10: "license plate"
        }

        return class_labels
