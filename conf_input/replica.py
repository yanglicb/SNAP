# dataset settings
dataset_type = "ReplicaDataset"
# data_root = "/work/vig/Datasets/ptv3_datasets/scannet"
data_root = "/projects/nufr/aniket/Datasets/replica/"

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
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="DropDataKey", key="color", prob=1.0),
            dict(type="DropDataKey", key="normal", prob=1.0),
            # dict(
            #     type="GridSampleProbabilistic",
            #     grid_sizes=[0.01, 0.02, 0.05],
            #     hash_type="fnv",
            #     mode="train",
            #     keys=("coord", "color", "instance", "normal", "segment"),
            #     return_grid_coord=True,
            # ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "instance", "normal", "segment"),
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="Add", keys_dict={"condition": "Replica"}),
            dict(type="Add", keys_dict={"domain": "Indoor"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "normal", "point", "masks", "condition", "domain", "mask_labels"),
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
            dict(type="DropDataKey", key="normal", prob=1.0),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "color", "instance", "normal", "segment"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="Add", keys_dict={"condition": "Replica"}),
            dict(type="Add", keys_dict={"domain": "Indoor"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "normal", "point", "masks", "condition", "domain", "mask_labels"),
            ),
        ],
        test_mode=False,
    )
)