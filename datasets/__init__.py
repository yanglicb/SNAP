from .semantic_kitti import SemanticKITTIDataset
from .nuscenes import NuscenesDataset
from .pandaset import PandasetDataset
from .scannet import ScanNetDataset
from .s3dis import S3DISDataset
from .s3disfull import S3DISDataset_full
from .scannetpp import ScanNetPPDataset
from .kitti360 import KITTI360Dataset
from .kitti360full import KITTI360Dataset_full
from .stpls3d import STPLS3DDataset
from .dales import DalesDataset
from .replica import ReplicaDataset
from .hm3d import HM3DDataset
from .matterport import Matterport3DDataset
from .urbanbis import UrbanBISDataset
from .kitti360_ss import KITTI360_SSDataset
from .waymo import WaymoDataset


def build_dataset_single_mask(args, stage="nuscenes", split="train", skip=1, num_prompt_points=32, num_object_points=5, overfit=False,
    use_random_clicks=False, use_centroid=False, run_openvocab_eval=False):
    if stage=="kitti":
        dataset = SemanticKITTIDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            skip=skip,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_random_clicks=use_random_clicks,
            use_centroid=use_centroid
        )
    elif stage=="nuscenes":
        dataset = NuscenesDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            skip=skip,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_random_clicks=use_random_clicks,
            use_centroid=use_centroid
        )
    elif stage=="kitti360":
        dataset = KITTI360Dataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_centroid=use_centroid
        )
    elif stage=="kitti360full":
        dataset = KITTI360Dataset_full(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_centroid=use_centroid
        )
    elif stage=="pandaset":
        dataset = PandasetDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            skip=skip,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_random_clicks=use_random_clicks,
            use_centroid=use_centroid
        )
    elif stage=="scannet" or stage=="scannet-block" or stage=="scannet20":
        dataset = ScanNetDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            skip=skip,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_random_clicks=use_random_clicks,
            is_scannet_block=(stage == "scannet-block"),
            is_scannet_20=(stage == "scannet20"),
            use_centroid=use_centroid,
            run_openvocab_eval=run_openvocab_eval
        )
    elif stage=="s3dis":
        dataset = S3DISDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            skip=skip,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_random_clicks=use_random_clicks,
            use_centroid=use_centroid
        )
    elif stage=="s3disfull":
        dataset = S3DISDataset_full(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            skip=skip,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_random_clicks=use_random_clicks,
            use_centroid=use_centroid,
            run_openvocab_eval=run_openvocab_eval
        )
    elif stage == "scannetpp":
        dataset = ScanNetPPDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            skip=skip,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_random_clicks=use_random_clicks,
            use_centroid=use_centroid
        )
    elif stage == "stpls3d":
        dataset = STPLS3DDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            skip=skip,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_random_clicks=use_random_clicks,
            use_centroid=use_centroid,
            run_openvocab_eval=run_openvocab_eval
        )
    elif stage == "dales":
        dataset = DalesDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            skip=skip,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_random_clicks=use_random_clicks,
            use_centroid=use_centroid
        )
    elif stage == "urbanbis":
        dataset = UrbanBISDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            skip=skip,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_random_clicks=use_random_clicks,
            use_centroid=use_centroid
        )
    elif stage == "replica":
        dataset = ReplicaDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_centroid=use_centroid,
            run_openvocab_eval=run_openvocab_eval
        )
    elif stage == "hm3d":
        dataset = HM3DDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_centroid=use_centroid
        )
    elif stage == "matterport":
        dataset = Matterport3DDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_centroid=use_centroid
        )
    elif stage=="kitti360_ss":
        dataset = KITTI360_SSDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_centroid=use_centroid
        )
    elif stage == "waymo":
        dataset = WaymoDataset(
            split=split,
            data_root=args.data_root,
            transform=args.data[split]["transform"],
            test_mode=False,
            test_cfg=None,
            num_prompt_points=num_prompt_points,
            num_object_points=num_object_points,
            overfit=overfit,
            use_centroid=use_centroid
        )
    
    return dataset
