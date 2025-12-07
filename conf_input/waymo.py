# dataset settings
dataset_type = "KITTI36_Single_Scan"
data_root = "/projects/nufr/aniket/Datasets/Waymo"

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="DropDataKey", key="strength", prob=1.0),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "instance"),
                return_grid_coord=True,
            ),
            # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            # dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            # dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="Add", keys_dict={"condition": "Waymo"}),
            dict(type="Add", keys_dict={"domain": "Outdoor"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "strength", "point", "masks", "mask_labels", "condition", "domain"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="DropDataKey", key="strength", prob=1.0),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "instance"),
                return_grid_coord=True,
            ),
            dict(type="Add", keys_dict={"condition": "Waymo"}),
            dict(type="Add", keys_dict={"domain": "Outdoor"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "strength", "point", "masks", "mask_labels", "condition", "domain"),
            ),
        ],
        test_mode=False,
    )
)