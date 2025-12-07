# dataset settings
dataset_type = "STPLS3DDataset"
data_root = "/projects/nufr/aniket/Datasets/STPLS3D"

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="DropDataKey", key="color", prob=1.0),
            dict(
                type="GridSample",
                grid_size=0.33,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "instance", "segment"),
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="Add", keys_dict={"condition": "STPLS3D"}),
            dict(type="Add", keys_dict={"domain": "Aerial"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "point", "masks", "condition", "domain", "mask_labels"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="DropDataKey", key="color", prob=1.0),
            dict(
                type="GridSample",
                grid_size=0.33,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "instance", "segment"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            # dict(type="NormalizeColor"),
            dict(type="Add", keys_dict={"condition": "STPLS3D"}),
            dict(type="Add", keys_dict={"domain": "Aerial"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "point", "masks", "condition", "domain", "mask_labels"),
            ),
        ],
        test_mode=False,
    )
)