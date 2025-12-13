import os
import numpy as np
import torch

import pdb
from .transforms import Compose

class DemoDatset():
    def __init__(self, domain="Outdoor", grid_size=None):
        
        if grid_size is None:
            if domain=="Outdoor":
                grid_size = 0.05
            elif domain=="Indoor":
                grid_size = 0.02
            elif domain=="Aerial":
                grid_size = 0.33

        if domain=="Outdoor":
            transforms_list_1 = [
                dict(
                    type="GridSample",
                    grid_size=grid_size,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "strength"),
                    return_grid_coord=True,
                ),
                dict(type="Add", keys_dict={"condition": "SemanticKITTI"}),
                dict(type="Add", keys_dict={"domain": domain}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "strength", "condition", "domain"),
                )
            ]

            transforms_list_2 = [
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "strength", "condition", "domain", "point", "text"),
                )
            ]

        elif domain=="Indoor":
            transforms_list_1 = [
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=grid_size,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "color", "strength", "normal"),
                    return_grid_coord=True,
                ),
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="Add", keys_dict={"condition": "ScanNet"}),
                dict(type="Add", keys_dict={"domain": domain}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "color", "normal", "strength", "condition", "domain"),
                )
            ]

            transforms_list_2 = [
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "color", "normal", "strength", "condition", "domain", "point", "text"),
                )
            ]

        elif domain=="Aerial":
            print("using the domain Aerial")
            transforms_list_1=[
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=grid_size,
                    hash_type="fnv",
                    mode="train",
                    keys=("coord", "color"),
                    return_grid_coord=True,
                ),
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="Add", keys_dict={"condition": "STPLS3D"}),
                dict(type="Add", keys_dict={"domain": domain}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "color", "condition", "domain"),
                ),
            ]

            transforms_list_2 = [
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "color", "condition", "domain", "point", "text"),
                )
            ]
        

        self.transform1 = Compose(transforms_list_1)
        self.transform2 = Compose(transforms_list_2)

        self.labels_kitti = ["Car", "Bicycle", "Motorcycle", "Truck", "Other_Vehicle", "Person", 
                       "Bicyclist", "Motorcyclist", "Road", "Parking", "Sidewalk", "Other_Ground", 
                       "Building", "Fence", "Vegetation", "Trunk", "Terrain", "Pole", "Traffic_Sign"]
        
        self.labels_scannet = ['wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
            'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag', 'backpack', 'toilet paper',
            'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier',
            'basket', 'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
            'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom stall', 'shower wall',
            'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
            'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser',
            'furniture', 'cart', 'storage container', 'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
            'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider',
            'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity', 'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin', 'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case', 'closet rod',
            'coffee kettle', 'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'storage organizer', 'folded chair', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'luggage', 'mattress']

        self.labels_stpls3d = ['building', 'low vegetation', 'med. vegetation', 'high vegetation', 'vehicle',
               'truck', 'aircraft', 'militaryVehicle', 'bike', 'motorcycle', 'light pole',
               'street sign', 'clutter', 'fence']
        
        self.labels_dales = ['Ground','Vegetation','Cars','Trucks','Power lines','Fences','Poles','Buildings']

        self.labels = self.labels_kitti + self.labels_scannet + self.labels_stpls3d

        self.labels_scannet20 = ("wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture")

        if domain=="Outdoor":
            self.labels = self.labels_kitti
        elif domain=="Indoor":
            self.labels = self.labels_scannet
        elif domain=="Aerial":
            self.labels = self.labels_stpls3d + self.labels_dales
        
    
    
    def process_data(self, pointcloud, color=None, normal=None, intensity=None):
        if pointcloud is None:
            raise ValueError("Pointcloud is not provided")
        
        assert isinstance(pointcloud, np.ndarray)

        coord = pointcloud.astype(np.float32)
        
        if color is not None:
            color = color.astype(np.float32)
        else:
            color = np.ones_like(coord)
        
        if normal is not None:
            normal = normal.astype(np.float32)
        else:
            normal = np.zeros_like(coord)

        if intensity is not None:
            intensity = intensity.astype(np.float32)
        else:
            intensity = np.zeros((coord.shape[0], 1), dtype=np.float32)

        data_dict = dict(coord=coord, color=color, normal=normal, strength=intensity)
        data_dict = self.transform1(data_dict)
        return data_dict
    
    def process_prompts(self, data, point_prompts=None, text_prompts=None):
        if point_prompts is None and text_prompts is None:
            raise ValueError("No prompts provided!")
        
        if point_prompts is not None:
            assert isinstance(point_prompts, np.ndarray)
            point_prompts = point_prompts.astype(np.float32)
        
        if text_prompts is not None:
            assert isinstance(text_prompts, str)

            # Get the uniform grid of points over the entire pointcloud
            point_prompts = self.generate_grid_points(data['coord'])
        else:
            text_prompts = "empty"
        
        data['point'] = point_prompts
        data['text'] = text_prompts
        data = self.transform2(data)
        return data


        

