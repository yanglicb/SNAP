import os
import numpy as np
import torch

from .defaults import DefaultDataset_new
import pdb
from .ply import read_ply


class KITTI360Dataset_full(DefaultDataset_new):
    def __init__(
        self,
        split="train",
        data_root="/work/vig/Datasets/KITTI-360",
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
        self.use_centroid = use_centroid
        
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
        data_path = self.data_root + "/data_3d_semantics/train"
        data_info_path = os.path.join(data_path, f'2013_05_28_drive_{self.split}.txt')
        
        with open(data_info_path, 'r') as f:
            split_list = [line.strip() for line in f.readlines()]

        data_list = []
        for data_file in split_list:
            data_file_path = os.path.join(self.data_root, data_file)
            data_list.append(data_file_path)

        return data_list
    
    def map_segment(self, segment):
        classes = self.class_labels(return_id_mapping=False)
        mapping = {original_id: idx for idx, original_id in enumerate(classes.keys())}
        map_func = np.vectorize(mapping.get)
        return map_func(segment)

    def get_data(self, idx):
        data_path = self.data_list[idx]
        pc = read_ply(data_path)

        coord = np.stack([pc['x'], pc['y'], pc['z']]).T
        color = np.stack([pc['red'], pc['green'], pc['blue']]).T
        segment = self.map_segment(pc['semantic'])
        instance = pc['instance']

        normal = np.ones_like(coord)


        data_dict = dict(coord=coord, color=color, instance=instance, normal=normal, segment=segment)
        return data_dict

    def process_test_data(self, data_dict):
        """
        data_dict contains the following keys after applying the transforms:
        1. coord
        2. colors
        3. normals
        4. labels: instance mask for the processed cloud after transforms
        """
        coord = data_dict["coord"]
        labels = data_dict["instance"]
        color = data_dict["color"]

        masks = []
        prompt_points = []

        unique_labels = np.unique(labels)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
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

        data_dict = dict(coord=coord, color=color, point=prompt_points, masks=torch.from_numpy(masks).long())
        return data_dict
    
    def process_data(self, data_dict):
        """
        data_dict contains the following keys after applying the transforms:
        1. coord
        2. grid_coord
        3. colors
        4. normal
        5. labels: instance mask for the processed cloud after transforms
        6. condition: "KITTI-360"
        """

        labels = data_dict['instance']
        coord = data_dict['coord']
        grid_coord = data_dict['grid_coord']
        color = data_dict['color']
        condition = data_dict['condition']
        domain = data_dict['domain']
        segment = data_dict['segment']
        normal = data_dict["normal"]

        masks = []
        mask_labels = []
        prompt_points = []

        unique_labels = np.unique(labels)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
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
        data_dict = dict(coord=coord, grid_coord=grid_coord, color=color,point=prompt_points, condition=condition, domain=domain, masks=masks, mask_labels=mask_labels)

        return data_dict

    def class_labels(self, return_id_mapping=True):
        class_mapping = {
            6: "ground",
            7: "road",
            8: "sidewalk",
            9: "parking",
            10: "rail track",
            11: "building",
            12: "wall",
            13: "fence",
            14: "guard rail",
            15: "bridge",
            16: "tunnel", 
            17: "pole",
            19: "traffic light",
            20: "traffic sign",
            21: "vegetation",
            22: "terrain",
            24: "person",
            25: "rider",
            26: "car",
            27: "truck",
            28: "bus",
            29: "caravan",
            30: "tariler",
            31: "train",
            32: "motocycle",
            33: "bicycle",
            34: "garage",
            35: "gate",
            36: "stop",
            37: "smallpole",
            38: "lamp",
            39: "trash bin",
            40: "vending machine",
            41: "box",
            42: "unknown construction",
            43: "unknown vehicle",
            44: "unknown object",
        }

        # 16 - stuff and 21 - things

        id_mapping = {i: v for i, (k, v) in enumerate(class_mapping.items())}

        return id_mapping if return_id_mapping else class_mapping


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