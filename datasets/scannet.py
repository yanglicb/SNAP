import os
import numpy as np
import torch

from .defaults import DefaultDataset_new
import pdb


class ScanNetDataset(DefaultDataset_new):
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
        is_scannet_block=False,
        is_scannet_20=False,
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
        self.is_scannet_block = is_scannet_block
        self.data_list = self.get_data_list()

        self.is_scannet_20 = is_scannet_20
        self.use_centroid = use_centroid
        self.run_openvocab_eval=run_openvocab_eval

        if is_scannet_20:
            self.OPENVOCAB_INVALID_CLASS_IDS = [0, 1, 19] 
        else:
            self.OPENVOCAB_INVALID_CLASS_IDS = [0, 2, ] 
            


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
        if self.agile3d_val and self.split == "val":
            import json
            with open(os.path.join(self.data_root, "val_list.json")) as json_file:
                self.data_samples = json.load(json_file)
                self.scene_to_sample = {s.split('_obj')[0]: s for s in self.data_samples.keys()}

        if self.is_scannet_block:
            data_list = [x for x in data_list if "discard" not in x]
        return data_list
    
    def compute_labels(self, ori_labels, correspondance):
        new_labels = np.zeros_like(ori_labels)
        for new_obj_id, ori_obj_id in correspondance.items():
            new_labels[ori_labels == ori_obj_id - 1] = int(new_obj_id)
        return new_labels

    def get_data(self, idx):

        scene_name = self.data_list[idx]

        coord = np.load(os.path.join(self.data_root, self.split, scene_name, "coord.npy"))
        color = np.load(os.path.join(self.data_root, self.split, scene_name, "color.npy"))
        normal = np.load(os.path.join(self.data_root, self.split, scene_name, "normal.npy"))
        # normal = np.load(os.path.join(self.data_root, self.split, scene_name, "computed_normal.npy"))
        instance = np.load(os.path.join(self.data_root, self.split, scene_name, "instance.npy"))

        if self.is_scannet_20:
            segment = np.load(os.path.join(self.data_root, self.split, scene_name, "segment20.npy"))
        else:
            segment = np.load(os.path.join(self.data_root, self.split, scene_name, "segment200.npy"))


        if self.agile3d_val and self.split == "val":
            unique = np.unique(instance)
            assert unique[1] + len(unique) - 2 == unique[-1]
            instance -= unique[1]
            sample_name = self.scene_to_sample[scene_name]
            data_sample = self.data_samples[sample_name]
            instance = self.compute_labels(instance, data_sample["obj"])
        
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

        unique_labels = np.unique(labels)[1:]
        np.random.shuffle(unique_labels)
        num_obj = min(self.random_points, len(unique_labels))
        for i in range(num_obj):
            label = unique_labels[i]
            point_idxs = np.where(labels == label)[0]

            # sample P prompt points on the same mask
            prompt_points.append(coord[np.random.choice(point_idxs, self.num_object_points, replace=True)])

            binary_mask = np.zeros_like(labels)
            binary_mask[point_idxs] = 1
            masks.append(binary_mask)

        prompt_points = np.array(prompt_points)
        masks = np.array(masks)

        data_dict = dict(coord=coord, color=color, normal=normal, point=prompt_points, masks=torch.from_numpy(masks).long())
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

        # unique_labels = np.unique(labels)[1:]
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

            # Remove the wall and floor classes if running openvocab evaluation
            if self.run_openvocab_eval:
                if seg_label in self.OPENVOCAB_INVALID_CLASS_IDS:
                    continue

                if self.is_scannet_20:
                    seg_label=seg_label-2 # Hack to make openvocab evaluation work properly
                else:
                    if seg_label == 1:
                        seg_label = 0
                    else:
                        seg_label = seg_label - 2
            
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
                        condition=condition, domain=domain, masks=masks, mask_labels=mask_labels, segment=segment)

        return data_dict

    def class_labels(self):

        if self.is_scannet_20:
            VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
            CLASS_LABELS_20 = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
                        'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
                        'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
            
            class_mapping = {}
            class_mapping_ptv3 = {}
            class_mapping_openvocab = {}
            for i in range(len(VALID_CLASS_IDS_20)):
                class_mapping_ptv3[VALID_CLASS_IDS_20[i]] = CLASS_LABELS_20[i]

            for i, key in enumerate(VALID_CLASS_IDS_20):
                class_mapping[i] = class_mapping_ptv3[key]

            # Filter classes for openvocab evaluation
            OV_VALID_CLASS_IDS_20 = [v for v in VALID_CLASS_IDS_20 if v not in [1, 2, 39]]
            for i, key in enumerate(OV_VALID_CLASS_IDS_20):
                if class_mapping_ptv3[key] == "otherfurniture":
                    class_mapping_openvocab[i] = "other"
                else:
                    class_mapping_openvocab[i] = class_mapping_ptv3[key]

            class_mapping = {
                1: "wall",
                2: "floor",
                3: "cabinet",
                4: "bed",
                5: "chair",
                6: "sofa",
                7: "table",
                8: "door",
                9: "window",
                10: "bookshelf",
                11: "picture",
                12: "counter",
                14: "desk",
                16: "curtain",
                24: "refrigerator",
                28: "shower curtain",
                33: "toilet",
                34: "sink",
                36: "bathtub",
                39: "otherfurniture"
            }
            class_mapping = {i: v for i, (k, v) in enumerate(class_mapping.items())}
        
        else:
            class_mapping = {}
            class_mapping_ptv3 = {}
            class_mapping_openvocab = {}

            VALID_CLASS_IDS_200 = (
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
            155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
            488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191)

            CLASS_LABELS_200 = (
            'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
            'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
            'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
            'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
            'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
            'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
            'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
            'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
            'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
            'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
            'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress')

            for i in range(len(VALID_CLASS_IDS_200)):
                class_mapping_ptv3[VALID_CLASS_IDS_200[i]] = CLASS_LABELS_200[i]

            for i, key in enumerate(VALID_CLASS_IDS_200):
                class_mapping[i] = class_mapping_ptv3[key]

            # Filter classes for openvocab evaluation
            OV_VALID_CLASS_IDS_200 = [v for v in VALID_CLASS_IDS_200 if v not in [1, 3]]
            # OV_VALID_CLASS_IDS_200 = [v for v in VALID_CLASS_IDS_200]
            for i, key in enumerate(OV_VALID_CLASS_IDS_200):
                if class_mapping_ptv3[key] == "otherfurniture":
                    class_mapping_openvocab[i] = "other"
                else:
                    class_mapping_openvocab[i] = class_mapping_ptv3[key]
        
        if self.run_openvocab_eval:
            return class_mapping_openvocab
        else:
            return class_mapping