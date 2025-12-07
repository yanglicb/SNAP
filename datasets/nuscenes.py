import os
import numpy as np
from .defaults import DefaultDataset_new
from nuscenes import NuScenes
import pickle

class NuscenesDataset(DefaultDataset_new):
    def __init__(self, 
        split="train",
        data_root="data/nuscenes",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        skip=1, 
        num_prompt_points = 32,
        num_object_points = 5,
        overfit = False,
        use_random_clicks=False,
        use_centroid=False,
    ):
        if overfit:
            pkl_file_path = os.path.join(data_root, f"nuscenes_infos_{split}_mini.pkl")
        else: 
            pkl_file_path = os.path.join(data_root, f"nuscenes_infos_{split}.pkl")

        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
    
        self.nusc_infos = data['infos']
        self.data_path = data_root
        if overfit:
            self.nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=False)
        else:
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=False)
        self.ignore_index = -1
        self.learning_map = self.get_learning_map(self.ignore_index)
        self.random_points = num_prompt_points
        self.num_object_points = num_object_points
        self.skip = skip
        self.overfit = overfit
        self.use_random_clicks = use_random_clicks
        self.use_centroid = use_centroid

        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)
       
    def get_data(self, index):
        info = self.nusc_infos[index]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        lidar_path = info['lidar_path'].split("/")
        lidar_path = os.path.join(lidar_path[-3], lidar_path[-2], lidar_path[-1])

        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        
        # load label
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_sd_token)['filename'])
        panoptic_labels_filename = os.path.join(self.nusc.dataroot, self.nusc.get('panoptic', lidar_sd_token)['filename'])
        
        new_panoptic_labels_filename = panoptic_labels_filename.replace(".npz", ".txt")
        panoptic_label = np.loadtxt(new_panoptic_labels_filename, dtype=np.int32)
        
        segment = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1])
        segment = np.vectorize(self.learning_map.__getitem__)(segment)
        
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]

        data_dict = dict(coord=coord, strength=strength, segment=segment, panoptic=panoptic_label)
        return data_dict

    def process_data(self, data_dict):

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
        
        if self.split == 'train': # or self.split == 'val':
            # For all unique panoptic labels in the data, get random points on as many labels as possible
            # This is done to ensure that we have a good distribution of points from all the classes
            valid_panoptic_labels = panoptic_label[valid_indices]
            
            # Filter the points by the number of points in a given panoptic label as well
            for label in np.unique(valid_panoptic_labels):
                point_idxs = np.where(valid_panoptic_labels == label)[0]
                if point_idxs.shape[0]<10:
                    valid_panoptic_labels[point_idxs] = 0        
            
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
            instance_mask[point_idxs] = i + 1 # selected instance id' in this scene

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

            # # Sample P prompt points on the same mask
            # prompt_points.append(coord[np.random.choice(point_idxs, self.num_object_points, replace=True)])

            binary_mask = np.zeros_like(panoptic_label)
            binary_mask[point_idxs] = 1
            masks.append(binary_mask)
            
            mask_labels.append(seg_label)
            class_weights.append(self.get_class_weights(seg_label))

        prompt_points = np.array(prompt_points)

        masks = np.array(masks)
        mask_labels = np.array(mask_labels)
        class_weights = np.array(class_weights)
        data_dict = dict(coord=coord, grid_coord=grid_coord, strength=strength, segment=segment, point=prompt_points, 
            condition=condition, domain=domain, mask=instance_mask, masks=masks, mask_labels=mask_labels, weights=class_weights)

        return data_dict

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

    def get_learning_map(self, ignore_index):
        learning_map = {
            0: ignore_index,  # noise
            1: ignore_index,  # animal
            2: 7,             # adult
            3: 7,             # child
            4: 7,             # construction worker
            5: ignore_index,  # personal mobility
            6: 7,             # police officer
            7: ignore_index,  # stroller
            8: ignore_index,  # wheelchair
            9: 1,             # barrier
            10: ignore_index, # debris
            11: ignore_index, # pushable pullable
            12: 8,            # traffic cone
            13: ignore_index, # bicycle rack
            14: 2,            # bicycle
            15: 3,            # bendy bus
            16: 3,            # rigid bus
            17: 4,            # car
            18: 5,            # construction vehicle
            19: ignore_index, # ambulance
            20: ignore_index, # police vehicle
            21: 6,            # motorcycle
            22: 9,            # trailer
            23: 10,           # truck
            24: 11,           # road
            25: 12,           # flat surface
            26: 13,           # sidewalk
            27: 14,           # terrain
            28: 15,           # manmade
            29: ignore_index, # other
            30: 16,           # vegetation
            31: ignore_index, # ego vehicle
        }
        return learning_map
    
    def class_labels(self):
        # class_labels = {
        #     0: 'ignore noise',
        #     1: 'barrier',
        #     2: 'bicylce',
        #     3: 'bus',
        #     4: 'car',
        #     5: 'vehicle',
        #     6: 'motorcycle',
        #     7: 'person',
        #     8: 'traficcone',
        #     9: 'vehicle-trailer',
        #     10: 'truck',
        #     11: 'road',
        #     12: 'flat surface',
        #     13: 'sidewalk',
        #     14: 'terrain',
        #     15: 'building',
        #     16: 'vegetation'
        # }

        class_labels = {
            0: 'ignore noise',
            1: 'fence',
            2: 'bicylce',
            3: 'bus',
            4: 'car',
            5: 'construction_vehicle',
            6: 'motorcycle',
            7: 'person',
            8: 'traficcone',
            9: 'trailer',
            10: 'truck',
            11: 'road',
            # 12: 'flat surface',
            12: 'ground',
            13: 'sidewalk',
            14: 'terrain',
            15: 'building',
            16: 'vegetation'
        }
        
        return class_labels