import os
import numpy as np
import torch

from .defaults import DefaultDataset_new
import pdb
from glob import glob
import os.path as osp


class STPLS3DDataset(DefaultDataset_new):
    def __init__(
        self,
        split="train",
        data_root="",
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

        # Merge class ids for the vegetation classes
        self.OPENVOCAB_MERGE_CLASS_IDS = [2,3,4]

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
        if self.split == "val":
            filenames = glob(osp.join(self.data_root, "val_50m", '*.pth'))
        else:
            filenames = glob(osp.join(self.data_root, self.split, '*.pth'))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)

        return filenames
    
    def get_data(self, idx):

        filename = self.data_list[idx]
        pointcloud = torch.load(filename)

        coord = pointcloud[0]
        color = pointcloud[1]
        segment = pointcloud[2]
        instance = pointcloud[3]
        
        data_dict = dict(coord=coord, color=color, instance=instance, segment=segment)
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

            if self.run_openvocab_eval:
                if seg_label in self.OPENVOCAB_MERGE_CLASS_IDS:
                    seg_label = 2

                # Rearranging labels for open-vocabulary evaluation (HACK)
                if seg_label not in [1, 2]:
                    seg_label = seg_label-3
                else:
                    seg_label = seg_label-1

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
            mask_labels.append(int(seg_label))

        prompt_points = np.array(prompt_points)
        masks = np.array(masks)
        mask_labels = np.array(mask_labels)
        data_dict = dict(coord=coord, grid_coord=grid_coord, color=color, point=prompt_points, 
                        condition=condition, domain=domain, masks=masks, mask_labels=mask_labels)

        return data_dict

    def class_labels(self):
        class_mapping = {}
        class_mapping_openvocab = {}

        CLASSES = ('building', 'low vegetation', 'med. vegetation', 'high vegetation', 'vehicle',
               'truck', 'aircraft', 'militaryVehicle', 'bike', 'motorcycle', 'light pole',
               'street sign', 'clutter', 'fence')
        
        OPENVOCAB_CLASSES = ('building', 'vegetation', 'vehicle', 'truck', 'aircraft', 'militaryVehicle',
                             'bike', 'motorcycle', 'light pole', 'street sign', 'clutter', 'fence')
        
        for i, cls in enumerate(OPENVOCAB_CLASSES):
            class_mapping_openvocab[i] = cls

        class_mapping = {
            0: 'ground',
            1: 'building',
            2: 'low vegetation',
            3: 'med. vegetation',
            4: 'high vegetation',
            5: 'vehicle',
            6: 'truck',
            7: 'aircraft',
            8: 'militaryVehicle',
            9: 'bike',
            10: 'motorcycle',
            11: 'light pole',
            12: 'street sign',
            13: 'clutter',
            14: 'fence'
        }

        
        if self.run_openvocab_eval:
            return class_mapping_openvocab
        else:
            return class_mapping