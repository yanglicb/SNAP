import os
import glob
import numpy as np
import torch
from collections.abc import Sequence

from .defaults import DefaultDataset_new
import pdb
from scipy.spatial.transform import Rotation as R

class S3DISDataset(DefaultDataset_new):
    def __init__(
        self,
        split="train",
        data_root="/Data/mnt/s3dis",
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
        self.overfit = overfit
        self.use_centroid = use_centroid

        self.train_seqs = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
        self.test_seqs = ['Area_5']
        self.val_seqs = ['Area_5']

        self.data_list = self.get_data_list()           

        self.r = R.from_euler("xyz", [-90, 180, 0], degrees=True) 

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

        if self.split == "train":
            seqs = self.train_seqs
        # elif self.split == "test":
        #     seqs = self.test_seqs
        # elif self.split == "val":
        #     seqs = self.val_seqs
        elif self.split == "val":
            self.data_root = "/projects/nufr/aniket/Datasets/S3DIS/single/crops"
            self.data_list = np.load("/projects/nufr/aniket/Datasets/S3DIS/single/object_ids.npy")
            return self.data_list
        
        # get the path for each point cloud and store them in self.data_list
        if isinstance(seqs, str):
            data_list = glob.glob(os.path.join(self.data_root, seqs, "*"))
        elif isinstance(seqs, Sequence):
            data_list = []
            for seq in seqs:
                data_list += glob.glob(os.path.join(self.data_root, seq, "*"))
        
        if self.agile3d_val and self.split == "Area_5":
            import json
            with open(os.path.join(self.data_root, "val_list.json")) as json_file:
                self.data_samples = json.load(json_file)
                self.scene_to_sample = {s.split("_obj")[0]: s for s in self.data_samples.keys()}
                data_list = [x for x in data_list if os.path.basename(x) in self.scene_to_sample]
        return data_list

    def compute_labels(self, ori_labels, correspondance):
        new_labels = np.zeros_like(ori_labels)
        for new_obj_id, ori_obj_id in correspondance.items():
            new_labels[ori_labels == ori_obj_id - 1] = int(new_obj_id)
        return new_labels

    def get_data(self, idx):
        if self.split=="val":
            scene_name = self.data_list[idx,0]
            object_id = self.data_list[idx,1]
            
            point_cloud = read_ply(os.path.join(self.data_root, scene_name, scene_name + '_crop_' + object_id + '.ply'))
            coord = np.column_stack([point_cloud['x'], point_cloud['y'], point_cloud['z']]).astype(np.float64)
            # coord = self.r.apply(coord)
            color = np.column_stack([point_cloud['R'], point_cloud['G'], point_cloud['B']])/255

            # Read normals from the numpy file
            normal = np.load(os.path.join(self.data_root, scene_name, scene_name + '_crop_' + object_id + '_normals.npy'))
            # normal = np.ones_like(coord)
            labels_full = point_cloud['label'].astype(np.int32)

            segment = np.zeros_like(labels_full)

            data_dict = dict(coord=coord, color=color, normal=normal, instance=labels_full, segment=segment)
            return data_dict

        sample_path = self.data_list[idx]
        scene_name = os.path.basename(sample_path)

        coord = np.load(os.path.join(sample_path, "coord.npy"))
        color = np.load(os.path.join(sample_path, "color.npy"))
        normal = np.load(os.path.join(sample_path, "normal.npy"))
        instance = np.load(os.path.join(sample_path, "instance.npy")).reshape(-1).astype(np.int32)
        segment = np.load(os.path.join(sample_path, "segment.npy")).reshape(-1).astype(np.int32)

        if self.agile3d_val and self.split == "Area_5":
            unique = np.unique(instance)
            assert unique[1] + len(unique) - 2 == unique[-1]
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

        if self.split == "val":
            # instance represents the true mask of the object of interest so to sample prompt points, find those points
            object_mask = labels == 1
            obj_coord = coord[object_mask]
            masks = []
            prompt_points = []
            mask_labels = []

            for i in range(self.random_points):
                # prompt_points.append(object_coord[np.random.choice(object_coord.shape[0], self.num_object_points, replace=True)])
                # masks.append(labels)
                # mask_labels.append(0)

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
                masks.append(labels)
                mask_labels.append(0)
                

            prompt_points = np.array(prompt_points)
            masks = np.array(masks)
            mask_labels = np.array(mask_labels)

            data_dict = dict(coord=coord, grid_coord=grid_coord, color=color, normal=normal, point=prompt_points, 
                    condition=condition, domain=domain, masks=masks, mask_labels=mask_labels)

            return data_dict

        random_indices = []
        masks = []
        prompt_points = []
        mask_labels = []

        unique_labels = np.unique(labels)[1:]
        np.random.shuffle(unique_labels)
        if self.split == "train":
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
        data_dict = dict(coord=coord, grid_coord=grid_coord, color=color, normal=normal, point=prompt_points, 
                    condition=condition, domain=domain, masks=masks, mask_labels=mask_labels)

        return data_dict

    def class_labels(self):
        class_mapping = {
            0: "ceiling",
            1: "floor",
            2: "wall",
            3: "beam",
            4: "column",
            5: "window",
            6: "door",
            7: "table",
            8: "chair",
            9: "sofa",
            10: "bookcase",
            11: "board",
            12: "clutter"
        }

        return class_mapping

# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])
    
    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}

    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # Parse header
        num_points, properties = parse_header(plyfile, ext)

        # Get data
        data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data