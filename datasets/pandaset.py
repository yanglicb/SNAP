import os
import numpy as np
from .defaults import DefaultDataset_new
import pandas as pd
import pickle


class PandasetDataset(DefaultDataset_new):
    def __init__(self,
        split="train",
        data_root="/home/thor/Datasets/Lidar_Segmentation/pandaset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=0,
        skip = 1, 
        num_prompt_points = 32,
        num_object_points = 5,
        overfit = False,
        use_random_clicks=False,
        cluster=False,
        use_centroid=False,
    ):
        self.ignore_index = ignore_index
        self.skip=skip
        self.split = split
        self.cluster = cluster

        # Variable to define how many random points we are choosing inside the pointcloud
        self.random_points = num_prompt_points
        self.num_object_points = num_object_points
        self.overfit = overfit
        self.use_random_clicks = use_random_clicks

        self.data_root = data_root
        self.use_centroid = use_centroid

        # self.dataset = DataSet(data_root)

        # Get the list of all the sequences in the dataset with semantic segmentation
        self.valid_seqs = ['001', '002', '003', '005', '011', '013', '015', '016', '017', '019', '021', '023', '024', '027', '028', '029', '030', 
                           '032', '033', '034', '035', '037', '038', '039', '040', '041', '042', '043', '044', '046']

        self.train_seqs = ['001', '002', '003', '005', '011', '013', '015', '016', '017', '019', '021', '023', '024', '027', '028', '029', '030',
                           '032', '033', '034', '035', '037', '038', '039', '040']
        
        self.val_seqs = ['041', '042', '043', '044', '046']

        if self.overfit:
            self.train_seqs = ['001']

        # Create the datalist
        self.datalist = self.get_data_list()

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
        return len(self.datalist)

    def get_data_list(self):
        if self.split == "train":
            seqs = self.train_seqs
        elif self.split == "val":
            seqs = self.val_seqs
        elif self.split == "all":
            seqs = self.valid_seqs

        datalist = []

        for seq in seqs:
            lidar_path = os.path.join(self.data_root, seq, 'lidar')
            seg_path = os.path.join(self.data_root, seq, 'annotations/semseg')

            # Get the list of the pickle files in the lidar and segmentation folders
            lidar_files = sorted([file for file in os.listdir(lidar_path) if file.endswith('.pkl')])
            seg_files = sorted([file for file in os.listdir(seg_path) if file.endswith('.pkl')])

            for i in range(len(lidar_files)):
                lidar_file = os.path.join(lidar_path, lidar_files[i])
                seg_file = os.path.join(seg_path, seg_files[i])

                datalist.append((lidar_file, seg_file))
        
        return datalist
   
    def cluster_points(self, coord, segment):
        """
        Args:
            coord: 3D points 
            segment: segmentation class labels 
        Function:
            Take all the points belonging to a particular class and cluster them using HDBSCAN
        Returns:
            panoptic_label: The panoptic label for each point in the pointcloud
        """
        labels_to_skip = [7]

        # Loop over all the classes, run HDBSCAN on it to cluster data, Save these clusters with unique instance ids to the labels
        panoptic_label = np.zeros(segment.shape[0], dtype=np.int32)

        for label in range(1, 43):                 
            # Get all points belonging to this class
            point_idxs = np.where(segment == label)[0]

            if label in labels_to_skip:
                panoptic_label[point_idxs] = label
                continue
            
            if point_idxs.shape[0]>50:
                # Apply HDBSCAN clustering to points of this panoptic class
                min_cluster_size = 50
                min_samples = 50
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
                cluster_labels = clusterer.fit_predict(coord[point_idxs])
                
                instance_ids = cluster_labels + 1  # Start instance IDs from 1

                # Combine instance IDs and class IDs to form the combined labels
                panoptic_label[point_idxs] = label * 1000 + instance_ids
                
                # Now all the noisy points have been marked with the stuff label, so we mark them as 0
                ignore_idxs = np.where(panoptic_label == label * 1000)[0]
                
                panoptic_label[ignore_idxs] = 0
            
            else:
                # If the number of points in this segmentation class is very low, ignore them
                panoptic_label[point_idxs] = 0
                
        return panoptic_label
    
    def get_data(self, idx):
        lidar_file, seg_file = self.datalist[idx]

        # Load the pointcloud and the segmentation labels
        lidar_all = pd.read_pickle(lidar_file)
        valid_idxs = np.where(lidar_all.d == 0)[0] # Get the points belonging to the 360 lidar only. 
        lidar = lidar_all.values[valid_idxs] # Convert to numpy array
        seg_label = pd.read_pickle(seg_file).values # Convert to numpy array
        seg_label = seg_label[valid_idxs] # Get the points belonging to the 360 lidar only.

        assert seg_label.shape[0] == lidar.shape[0], "The number of points in the lidar and segmentation labels do not match"

        # Get the 3D coordinates of the pointcloud
        coord = np.asarray(lidar[:, :3])
        strength = np.expand_dims(np.asarray(lidar[:, 3])/255.0, 1)

        # Get the panoptic labels
        if self.cluster:
            # Cluster the points based on the segmentation labels
            panoptic_label = self.cluster_points(coord, seg_label)
            
            # Save the panoptic labels
            save_path = seg_file.replace('annotations/semseg', 'annotations/panoptic')
            dir_path = os.path.dirname(save_path)
            os.makedirs(dir_path, exist_ok=True)
            pd.DataFrame(panoptic_label).to_pickle(save_path)
        
        else:
            # Load the panoptic labels
            panoptic_label = pd.read_pickle(seg_file.replace('annotations/semseg', 'annotations/panoptic')).values
            # panoptic_label = panoptic_label[valid_idxs]

        data = {'coord': coord, 'strength': strength, 'segment': seg_label, 'panoptic': panoptic_label} 
        return data
    
    def process_data(self, data_dict):
        panoptic_label = data_dict['panoptic'].squeeze()
        coord = np.clip(data_dict['coord'], -1e3, 1e3)
        strength = data_dict['strength']
        segment = data_dict['segment'].squeeze()
        grid_coord = data_dict['grid_coord']
        condition = data_dict['condition']
        domain = data_dict['domain']

        # Choose random points for point prompting
        # Find indices of points with non-ignore segmentation labels
        valid_indices = np.where(panoptic_label != self.ignore_index)[0]
                
        if self.split == 'train':
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

        elif self.split=="val":
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

            # Sample P prompt points on the same mask
            # prompt_points.append(coord[np.random.choice(point_idxs, self.num_object_points, replace=True)])

            binary_mask = np.zeros_like(panoptic_label)
            binary_mask[point_idxs] = 1
            masks.append(binary_mask)
            
            mask_labels.append(seg_label-1)

        prompt_points = np.array(prompt_points)

        masks = np.array(masks)
        mask_labels = np.array(mask_labels)
        data_dict = dict(coord=coord, grid_coord=grid_coord, strength=strength, segment=segment, point=prompt_points, 
            condition=condition, domain=domain, mask=instance_mask, masks=masks, mask_labels=mask_labels)

        return data_dict
    
    def class_labels(self):
        # class_mapping = {
        #     1: 'Smoke', 
        #     2: 'Exhaust', 
        #     3: 'Spray or rain', 
        #     4: 'Reflection', 
        #     5: 'Vegetation', 
        #     6: 'Ground', 
        #     7: 'Road', 
        #     8: 'Lane Line Marking', 
        #     9: 'Stop Line Marking', 
        #     10: 'Other Road Marking', 
        #     11: 'Sidewalk', 
        #     12: 'Driveway', 
        #     13: 'Car', 
        #     14: 'Pickup Truck', 
        #     15: 'Medium-sized Truck', 
        #     16: 'Semi-truck', 
        #     17: 'Towed Object', 
        #     18: 'Motorcycle', 
        #     19: 'Construction Vehicle', 
        #     20: 'Uncommon vecicle', 
        #     21: 'Pedicab', 
        #     22: 'Emergency Vehicle', 
        #     23: 'Bus', 
        #     24: 'Personal Mobility Device', 
        #     25: 'Motorized Scooter', 
        #     26: 'Bicycle', 
        #     27: 'Train', 
        #     28: 'Trolley', 
        #     29: 'Tram / Subway', 
        #     30: 'Pedestrian', 
        #     31: 'Pedestrian with Object', 
        #     32: 'Animals - Bird', 
        #     33: 'Animals - Other', 
        #     34: 'Pylons', 
        #     35: 'Road Barriers', 
        #     36: 'Signs', 
        #     37: 'Cones', 
        #     38: 'Construction Signs', 
        #     39: 'Temporary Construction Barriers', 
        #     40: 'Rolling Containers', 
        #     41: 'Building', 
        #     42: 'Other Static Object',
        #     }

        class_mapping = {
            0: 'Smoke', 
            1: 'Exhaust', 
            2: 'Spray or rain', 
            3: 'Reflection', 
            4: 'Vegetation', 
            5: 'Ground', 
            6: 'Road', 
            7: 'Lane Line Marking', 
            8: 'Stop Line Marking', 
            9: 'Other Road Marking', 
            10: 'Sidewalk', 
            11: 'Driveway', 
            12: 'Car', 
            13: 'Pickup Truck', 
            14: 'Medium-sized Truck', 
            15: 'Semi-truck', 
            16: 'Towed Object', 
            17: 'Motorcycle', 
            18: 'Construction Vehicle', 
            19: 'Uncommon vecicle', 
            20: 'Pedicab', 
            21: 'Emergency Vehicle', 
            22: 'Bus', 
            23: 'Personal Mobility Device', 
            24: 'Motorized Scooter', 
            25: 'Bicycle', 
            26: 'Train', 
            27: 'Trolley', 
            28: 'Tram / Subway', 
            29: 'Pedestrian', 
            30: 'Pedestrian with Object', 
            31: 'Animals - Bird', 
            32: 'Animals - Other', 
            33: 'Pylons', 
            34: 'Road Barriers', 
            35: 'Signs', 
            36: 'Cones', 
            37: 'Construction Signs', 
            38: 'Temporary Construction Barriers', 
            39: 'Rolling Containers', 
            40: 'Building', 
            41: 'Other Static Object',
        }

        return class_mapping
