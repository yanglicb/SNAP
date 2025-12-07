import torch
import argparse
import os
import time
from datetime import datetime
from datasets import build_dataset_single_mask
from datasets import collate_utils
from dataloaders.multidataset_loader import MultiDataset, RoundRobinCycleDataset
from src.snap import SNAP
from utils.dist_utils import get_dist_info, create_logger, pretty_time_delta
from utils.torch_helpers import all_to_device
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import torch.distributed as dist
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from datetime import timedelta

# Import clip to generate text embeddings for each class in a dataset
import clip
import random

# Import the instance-wise AP calculation
from eval_metrics import *

def get_args_parser():
    parser = argparse.ArgumentParser()

    # Tensorboard summary name
    parser.add_argument('--exp_name', default="test_single_mask", type=str)
    parser.add_argument('--logdir', default="logs/", type=str)
    parser.add_argument('--dev', action='store_true', default=False)
    parser.add_argument('--enable_amp', action='store_true', default=False)

    # Model
    parser.add_argument('--num_prompt_points', default=32, type=int)
    parser.add_argument('--num_object_points', default=10, type=int)
    parser.add_argument('--num_object_chunk', default=32, type=int)
    parser.add_argument('--num_merge', default=1, type=int)
    parser.add_argument('--overfit', action='store_true', default=False)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--use_pdnorm', action='store_true', default=False)
    parser.add_argument('--use_random_clicks', action='store_true', default=False)
    parser.add_argument('--mask_threshold', default=0.5, type=float)
    parser.add_argument('--use_aux_loss', action='store_true', default=False)
    parser.add_argument('--use_localized_loss', action='store_true', default=False)
    parser.add_argument('--iterative', default=False, action='store_true')
    parser.add_argument('--use_var_grid', action='store_true', default=False)
    
    # Evaluation
    parser.add_argument('--val_only', default=False, action='store_true')
    parser.add_argument('--compute_ap', default=False, action='store_true')
    parser.add_argument('--return_class_wise', action='store_true', default=False)
    parser.add_argument('--use_centroid', action='store_true', default=False)
    parser.add_argument('--run_openvocab_eval', action="store_true", default=False)
    parser.add_argument('--compute_PQ', action="store_true", default=False)
    parser.add_argument('--PQ_refer_labels', action="store_true", default=False)
    parser.add_argument('--compute_NOC', action="store_true", default=False)
    parser.add_argument('--exclude_wall_floor', default=False, action='store_true')
    parser.add_argument('--get_confusion', default=False, action='store_true')

    # dataset
    parser.add_argument('--checkpoint_dir', default="checkpoints/", type=str)
    parser.add_argument('--stage', default=['kitti'], type=str, nargs='+')
    parser.add_argument('--val_dataset', default=[''], type=str, nargs='+')

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--weight_decay', default=0.005, type=float)
    parser.add_argument('--val_freq', default=10, type=int)
    parser.add_argument('--epochs', default=500, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str)

    # distributed training
    parser.add_argument('--distributed', action='store_true', default=False)

    # Output
    parser.add_argument('--save_output', default=False, type=bool)

    return parser

def reduce_tensor(tensor):
    # Reduces tensor across all GPUs
    rt = tensor.clone().float()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def reduce_tensor_sum(tensor):
    # Reduces tensor across all GPUs
    rt = tensor.clone().float()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def load_model_weights(model, checkpoint_path):
    # Load trained weights to the model
    checkpoint = torch.load(checkpoint_path)

    state_dict = {}
    for k, v in checkpoint['model'].items():
        state_dict[k] = v

    model.load_state_dict(state_dict, strict=False)
    return model

def random_sample_clicks(preds, masks, coords, offset, threshold):
    batch_offset = [0] + list(offset)
    batch_size = len(masks)
    new_clicks = []
    for i in range(batch_size):
        pred = preds[i].sigmoid() > threshold
        mask = masks[i]
        coords_i = coords[batch_offset[i]:batch_offset[i+1]]
        num_obj = len(mask)
        for idx in range(num_obj):
            pred_idx = pred[idx]
            mask_idx = mask[idx]
            gt_indices = torch.where(mask_idx == 1)[0]
            pred_ = pred_idx[gt_indices]
            error_indices = torch.where(pred_ == 0)[0]
            if len(error_indices) == 0:
                select_idx = np.random.choice(gt_indices.cpu(), 1, replace=False)[0]
            else:
                error_idx = np.random.choice(error_indices.cpu(), 1, replace=False)[0]
                select_idx = gt_indices[error_idx]
            assert mask_idx[select_idx] == 1
            new_clicks.append(coords_i[select_idx].unsqueeze(0))
    new_clicks = torch.stack(new_clicks)
    return new_clicks

def set_seed(seed: int):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.
    """
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch on both CPU and CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set as {seed}")

def compute_iou(pred_mask, gt_mask):
    pred_mask = (pred_mask > 0).float()  # Binarize predicted mask
    gt_mask = (gt_mask > 0).float()  # Binarize ground truth mask

    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection

    if union == 0:
        return 1.0  # If both masks are empty, IoU is 1
    else:
        return intersection / union

def train_epoch(args, model, clip_model, train_loader, optimizer, scheduler, scaler, writer, epoch_num, 
                    device, train_sampler, label_dict):
    ####### Train the model =======================>
    # Manual change random seed for shuffling every epoch
    if args.distributed:
        train_sampler.set_epoch(epoch_num)

    t_epoch_start = time.perf_counter()
    model.train()

    # Vars to store loss and metrics
    total_metrics = {}
    total_count = {}
    loss_avg = {'loss': 0}
    
    # Freeze the clip backbone
    for param in clip_model.parameters():
        param.requires_grad = False

    text_features_dict = {}
    for dataset_name in label_dict:
        classes = label_dict[dataset_name]
        text_inputs = torch.cat([clip.tokenize(f"segment {classes[c]}") for c in classes]).to(device)
        if args.enable_amp:
            text_features_dict[dataset_name] = clip_model.encode_text(text_inputs) # Get text embeddings for each class in the dataset
        else:
            text_features_dict[dataset_name] = clip_model.encode_text(text_inputs).float()

    try:
        # Loop over the dataloader
        with tqdm(total=len(train_loader)) as pbar:
            for batch_idx_data, sample in enumerate(train_loader):
                # Put the data on device
                sample = all_to_device(sample, device)
                # Sanity check for the sample - Check if there are prompt points in the sample, if not, skip the batch
                if sample['point'].shape[0] == 0:
                    logging.info(f"No prompt points in the sample, skipping the batch, condition: {sample['condition']}, num_points: {sample['coord'].shape[0]}")
                    continue

                point_offset = []
                point_num_sum = 0
                for i in range(len(sample["masks"])):
                    point_num_sum += sample["masks"][i].shape[0]
                    point_offset.append(point_num_sum)
                sample["point_offset"] = point_offset

                # If using random clicks, samples random click points from the max available click points
                if args.use_random_clicks:
                    new_click_points = []
                    point_offset_list = [0] + sample["point_offset"]

                    clicks = np.random.randint(1, args.num_object_points+1)
        
                    for i in range(args.batch_size):
                        if args.iterative:
                            new_click_points.append(sample['point'][point_offset_list[i]:point_offset_list[i+1], :1, :])
                        else:
                            new_click_points.append(sample['point'][point_offset_list[i]:point_offset_list[i+1], :clicks, :])

                    sample['point'] = torch.cat(new_click_points, dim=0)


                with autocast(enabled=args.enable_amp):
                    # Forward pass through the model
                    seg_logits, text_out, iou_out, aux_seg_logits_list, loss_weights, confidence_scores_list, confidence_score_gt_list = model(sample, iterative=args.iterative, clicks=clicks)

                    # Compute loss
                    loss, loss_dict = model.module.compute_loss(seg_logits, text_out, iou_out, sample['masks'], 
                                        text_features_dict[sample['condition'][0]], sample['mask_labels'], aux_seg_logits_list, loss_weights,
                                        confidence_scores_list, confidence_score_gt_list, loss_weight_dict, sample['condition'][0])
                
                # Compute Metrics
                with torch.no_grad():
                    batch_metrics, batch_count, _ = model.module.compute_metrics(seg_logits, sample['masks'], sample['mask_labels'], sample['condition'][0],
                                label_dict[sample['condition'][0]], text_out, text_features_dict[sample['condition'][0]],
                                aux_seg_logits_list, return_classwise=False)

                # More efficient zero_grad
                # optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None
                if args.enable_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scale = scaler.get_scale()
                    scaler.update()
                    if scale <= scaler.get_scale():
                        scheduler.step()
                else:
                    # Backward pass
                    loss.backward()
                    optimizer.step() # Optimizer step
                    scheduler.step() # Scheduler step


                ## Log the loss values
                # Aggregate loss values across GPUs
                if args.distributed:
                    loss = reduce_tensor(loss).item()
                if args.local_rank == 0:
                    loss_avg['loss'] += loss

                for key in loss_dict:
                    if args.distributed:
                        loss_data = reduce_tensor(loss_dict[key]).item()
                    else:
                        loss_data = loss_dict[key]
                    
                    if args.local_rank == 0:
                        if key in loss_avg:
                            loss_avg[key] += loss_data
                        else:
                            loss_avg[key] = loss_data 

                # Log the metrics data
                # Aggregate metrics across GPUs
                for key in batch_metrics:
                    if args.distributed:
                        metric_data = reduce_tensor(batch_metrics[key].float()).item()
                    else:
                        metric_data = batch_metrics[key]
                    
                    if args.local_rank == 0:
                        count_key = f"{key}_count"
                        if key in total_metrics:
                            total_metrics[key] += metric_data
                            total_count[key] += batch_count[key]
                        else:
                            total_metrics[key] = metric_data
                            total_count[key] = batch_count[key]

                # Update the tqdm bar
                pbar.set_description(f"Mem: {torch.cuda.max_memory_allocated(device=device)/(1073741824):.2f}")
                pbar.update(1)
    except Exception as e:
        logging.error(f"Error in training epoch: {e}")
        # Print all the info about the sample
        print(f"Num points: {sample['coord'].shape[0]}")
        print(f"Condition: {sample['condition']}")
        print(f"Point: {sample['point']}")
        print(f"Point shape: {sample['point'].shape}")
        print(f"Masks: {sample['masks']}")
        print(f"Mask labels: {sample['mask_labels']}")
        raise e

    # Save data to the log file and tensorboard
    if args.local_rank == 0:
        # Log the learning rate
        writer.add_scalar("Train_epoch/LR", optimizer.param_groups[-1]['lr'], epoch_num)
        logging.info(f"Learning rate after epoch {epoch_num+1}: {optimizer.param_groups[-1]['lr']}")

        mIOU = 0
        categories = 0
        for key in total_metrics:
            avg = total_metrics[key] / total_count[key]

            if "total_IOU" in key:
                condition = key.split("_")[0]
                logging.info(f"After epoch {epoch_num + 1}: For dataset {condition}, IoU = {avg} || Sum: {total_metrics[key]} Count: {total_count[key]}")
                writer.add_scalar(f"Train_epoch/IoU", avg, epoch_num) # tensorboard logs
            elif "total_text_correct" in key:
                condition = key.split("_")[0]
                logging.info(f"After epoch {epoch_num + 1}: For dataset {condition}, Text Accuracy = {avg} || Sum: {total_metrics[key]} Count: {total_count[key]}")
                writer.add_scalar(f"Train_epoch/Text_Accuracy", avg, epoch_num)
            elif "total_acc_25" in key:
                condition = key.split("_")[0]
                logging.info(f"After epoch {epoch_num + 1}: For dataset {condition}, Acc_25 = {avg} || Sum: {total_metrics[key]} Count: {total_count[key]}")
                writer.add_scalar(f"Train_epoch/Acc_25", avg, epoch_num)
            elif "total_acc_50" in key:
                condition = key.split("_")[0]
                logging.info(f"After epoch {epoch_num + 1}: For dataset {condition}, Acc_50 = {avg} || Sum: {total_metrics[key]} Count: {total_count[key]}")
                writer.add_scalar(f"Train_epoch/Acc_50", avg, epoch_num)
            else:
                condition = key.split("_")[0]
                label_idx = int(key.split("_")[3])
                label = label_dict[condition][label_idx]
                logging.info(f"After epoch {epoch_num + 1}: {condition}_{label_idx}_{label} IoU@{args.num_object_points} = {avg} || Sum: {total_metrics[key]} Count: {total_count[key]}")
                writer.add_scalar(f"Train_epoch/{condition}_{label_idx}_{label}_IoU", avg, epoch_num) # tensorboard logs
                if args.exclude_wall_floor and label in ["wall", "floor"]:
                    continue
                    
                mIOU += avg
                categories += 1
        
        # Safe guard when not returning class-wise results
        categories = 1 if categories==0 else categories
        logging.info(f"After epoch {epoch_num + 1}: mIoU@{args.num_object_points} = {mIOU / categories} || Category Count: {categories}{' (excluding wall and floor)'if args.exclude_wall_floor else ''}")
        writer.add_scalar(f"Train_epoch/mIOU", mIOU / categories, epoch_num)
        
        for key in loss_avg:
            logging.info(f"Total {key} after epoch {epoch_num+1}: {loss_avg[key]/len(train_loader)}")
            writer.add_scalar(f"Train_epoch/{str(key)}", loss_avg[key]/len(train_loader), epoch_num) # tensorboard logs

        logging.info(f"Epoch {epoch_num + 1} completed in {pretty_time_delta(time.perf_counter() - t_epoch_start)}\n")

    # Delete the variables to free up memory
    del loss, loss_dict, batch_metrics, batch_count, seg_logits, text_out, iou_out, aux_seg_logits_list

def validation_epoch(args, model, clip_model, val_loader, writer, epoch_num, device, label_dict):
    if args.local_rank == 0:
        logging.info("Starting validation now")

    ####### Perform validation ========================>
    t_epoch_start = time.perf_counter()
    model.eval()

    # Vars to store loss and metrics
    metrics_click = {}
    metrics_click_avg = {}
    metrics_click_pq = {}
    metrics_click_pq_avg = {}
    metrics_click_confusion = {}
    metrics_click_confusion_avg = {}

    count_click = {}
    count_click_avg = {}
    total_ap_info_click = {}
    store_ap_info_click = {}
    scene_data_dict = {}

    # Initialize NoC metrics
    noc_thresholds = [0.5, 0.65, 0.8, 0.85, 0.9]
    noc_metrics = {thresh: [] for thresh in noc_thresholds}

    # Freeze the clip backbone
    for param in clip_model.parameters():
        param.requires_grad = False

    text_features_dict = {}
    for dataset_name in label_dict:
        classes = label_dict[dataset_name]
        text_inputs = torch.cat([clip.tokenize(f"segment {classes[c]}") for c in classes]).to(device)
        text_features_dict[dataset_name] = clip_model.encode_text(text_inputs).float()


    with tqdm(total=len(val_loader)) as pbar: 
        for batch_idx_data, sample in enumerate(val_loader):
            # Put the data on device
            sample = all_to_device(sample, device)
            if sample['point'].shape[0] == 0:
                logging.info(f"No prompt points in the sample, skipping the batch, condition: {sample['condition']}, num_points: {sample['coord'].shape[0]}")
                continue

            scene_id = f"batch_{batch_idx_data}"

            point_offset = []
            point_num_sum = 0
            for i in range(len(sample["masks"])):
                point_num_sum += sample["masks"][i].shape[0]
                point_offset.append(point_num_sum)
            
            sample["point_offset"] = point_offset

            point_offset_list = [0] + sample["point_offset"]
            
            if args.iterative:
                new_click_points = []
                for batch_idx in range(args.batch_size):
                    new_click_points.append(sample['point'][point_offset_list[batch_idx]:point_offset_list[batch_idx+1], :1, :])
                
                sample['point'] = torch.cat(new_click_points, dim=0)
            
            # Run through the model
            with torch.no_grad():
                with autocast(enabled=False):
                    # Run through the backbone first
                    sample = model.module.run_input_encoder(sample)
                    point_features = model.module.run_backbone(sample)

                    # Run through the mask decoder
                    point_offset_list = [0] + sample["point_offset"]
                    offset_list = [0] + sample["offset"].tolist()

                    seg_logits_prompt_list = []
                    text_out_prompt_list = []
                    iou_out_prompt_list = []
                    aux_seg_logits_prompt_list = []
                    loss_weights_prompt_list = []

                    seg_logits_prompt_list_per_click = {}
                    text_out_prompt_list_per_click = {}
                    iou_out_prompt_list_per_click = {}
                    aux_seg_logits_prompt_list_per_click = {}
                    loss_weights_prompt_list_per_click = {}

                    # Add keys to these dictionaries
                    for sample_iter in range(1, args.num_object_points+1):
                        seg_logits_prompt_list_per_click[sample_iter] = []
                        text_out_prompt_list_per_click[sample_iter] = []
                        iou_out_prompt_list_per_click[sample_iter] = []
                        aux_seg_logits_prompt_list_per_click[sample_iter] = []
                        loss_weights_prompt_list_per_click[sample_iter] = []

                    num_objects_chunk = args.num_object_chunk

                    for prompt_idx in range(0, sample['point'].shape[0], num_objects_chunk):
                        end_idx = min(prompt_idx + num_objects_chunk , sample['point'].shape[0])

                        prompt_point_list = []
                        prompt_point_offset_list_new = []
                        masks_new = []
                        text_prompt_list = []
                        for batch_idx in range(args.batch_size):
                            data_dict = sample.copy()
                            
                            # Take the query prompt point and make batch
                            prompt_point_list.append(sample['point'][point_offset_list[batch_idx]:point_offset_list[batch_idx+1], :, :][prompt_idx:end_idx])
                            prompt_point_offset_list_new.append(end_idx - prompt_idx)
                            masks_new.append(sample['masks'][batch_idx][prompt_idx:end_idx])

                        data_dict['point'] = torch.cat(prompt_point_list, dim=0)
                        data_dict['point_offset'] = prompt_point_offset_list_new
                        data_dict['masks'] = masks_new
                        data_dict['text_data'] = text_prompt_list

                        if args.iterative:
                            for sample_iter in range(args.num_object_points):
                                seg_logits_prompt, text_out_prompt, iou_out_prompt, aux_seg_logits_prompt, loss_weights_prompt, confidence_scores_list_prompt, confidence_score_gt_list_prompt = model.module.run_mask_decoder(point_features, data_dict, val=True)

                                seg_logits_prompt_list_per_click[sample_iter+1].append(seg_logits_prompt)
                                text_out_prompt_list_per_click[sample_iter+1].append(text_out_prompt)
                                iou_out_prompt_list_per_click[sample_iter+1].append(iou_out_prompt)

                                if sample_iter < args.num_object_points - 1:
                                    # Based on the ground-truth masks -> Sample new clicks from the error region.
                                    new_clicks = random_sample_clicks(seg_logits_prompt, data_dict['masks'], data_dict['coord'], data_dict['offset'], model.module.mask_threshold)
                                    data_dict['point'] = torch.cat((data_dict['point'], new_clicks), dim=1) # add the new clicks
                        
                        else:
                            seg_logits_prompt, text_out_prompt, iou_out_prompt, aux_seg_logits_prompt, loss_weights_prompt, confidence_scores_list_prompt, confidence_score_gt_list_prompt = model.module(data_dict)
                            seg_logits_prompt_list_per_click[args.num_object_points].append(seg_logits_prompt)
                            text_out_prompt_list_per_click[args.num_object_points].append(text_out_prompt)
                            iou_out_prompt_list_per_click[args.num_object_points].append(iou_out_prompt)

                    if args.compute_NOC:
                        # Initialize per-object IoU tracking
                        per_object_iou_dict = {}  # {(batch_idx, obj_idx): {click: iou}}
                        per_object_noc_dict = {}  # {(batch_idx, obj_idx): {thresh: clicks_needed}}

                    for key in seg_logits_prompt_list_per_click:
                        seg_logits_prompt_list = seg_logits_prompt_list_per_click[key]
                        text_out_prompt_list = text_out_prompt_list_per_click[key]
                        iou_out_prompt_list = iou_out_prompt_list_per_click[key]

                        seg_logits = []
                        text_out = []
                        iou_out = []

                        for batch_idx in range(args.batch_size):
                            seg_logits_list_batch = []
                            text_out_list_batch = []
                            iou_out_list_batch = []

                            for prompt_idx in range(len(seg_logits_prompt_list)):
                                # Get seg_logits_prompt per batch
                                seg_logits_list_batch.append(seg_logits_prompt_list[prompt_idx][batch_idx])
                                text_out_list_batch.append(text_out_prompt_list[prompt_idx][batch_idx])
                                iou_out_list_batch.append(iou_out_prompt_list[prompt_idx][batch_idx])
                        
                            seg_logits_batch = torch.cat(seg_logits_list_batch, dim=0)
                            text_out_batch = torch.cat(text_out_list_batch, dim=0)
                            iou_out_batch = torch.cat(iou_out_list_batch, dim=0)
                            
                            seg_logits.append(seg_logits_batch)
                            text_out.append(text_out_batch)
                            iou_out.append(iou_out_batch)

                        if args.compute_NOC:
                            # Compute per-object IoU
                            num_objects = seg_logits_batch.shape[0]
                            for obj_idx in range(num_objects):
                                pred_mask = seg_logits_batch[obj_idx]  # Predicted mask
                                gt_mask = sample['masks'][batch_idx][obj_idx]  # Ground truth mask

                                # Compute IoU
                                iou = compute_iou(pred_mask, gt_mask)
                                obj_key = (batch_idx, obj_idx)

                                if obj_key not in per_object_iou_dict:
                                    per_object_iou_dict[obj_key] = {}
                                    per_object_noc_dict[obj_key] = {thresh: None for thresh in noc_thresholds}

                                per_object_iou_dict[obj_key][key] = iou

                                # Update NoC metrics
                                for thresh in noc_thresholds:
                                    if per_object_noc_dict[obj_key][thresh] is None and iou >= thresh:
                                        per_object_noc_dict[obj_key][thresh] = key  # Click number

                                    # Also if the object never reaches the threshold, update the NoC metric
                                    if per_object_noc_dict[obj_key][thresh] is None and iou<thresh and key == args.num_object_points:
                                        per_object_noc_dict[obj_key][thresh] = key

                        batch_metrics_temp, batch_count_temp, ap_info = model.module.compute_metrics(seg_logits, sample['masks'], sample['mask_labels'], 
                                            sample['condition'][0], label_dict[sample['condition'][0]], text_out, text_features_dict[sample['condition'][0]],
                                            None, return_classwise=args.return_class_wise, iou_out=iou_out)

                        if args.compute_PQ:
                            # Compute PQ, SQ, RQ metrics
                            num_classes = len(label_dict[sample['condition'][0]])
                            batch_metrics_PQ, batch_confusion_matrix = compute_pq_metrics(num_classes, seg_logits, sample['masks'], sample['mask_labels'], 
                                                                text_features_dict[sample['condition'][0]], text_out, refer_labels=args.PQ_refer_labels, return_confusion=args.get_confusion)

                        if args.run_openvocab_eval:
                            # Save data for open-vocabulary evaluation later                            
                            if key not in scene_data_dict:
                                scene_data_dict[key] = {}

                            scene_data = save_data_openvocab(seg_logits, text_out, text_features_dict[sample['condition'][0]], sample['masks'], 
                                                            sample['mask_labels'], label_dict[sample['condition'][0]], sample['condition'][0], scene_id)
                            
                            for scene_key in scene_data:
                                if scene_key not in scene_data_dict[key]:
                                    scene_data_dict[key][scene_key] = scene_data[scene_key]
                                else:
                                    raise ValueError("Same scene data key found twice...Exiting")

                        
                        del seg_logits
                        del text_out
                        del iou_out
                        del seg_logits_list_batch
                        del text_out_list_batch
                        del iou_out_list_batch
                        # torch.cuda.empty_cache()
                        metrics_click[key] = batch_metrics_temp
                        count_click[key] = batch_count_temp
                        store_ap_info_click[key] = ap_info
                        
                        if args.compute_PQ:
                            metrics_click_pq[key] = batch_metrics_PQ

                        if args.get_confusion:
                            metrics_click_confusion[key] = batch_confusion_matrix


            if args.distributed:
                dist.barrier()

            # Aggregate NoC metrics
            if args.compute_NOC:
                for obj_key in per_object_noc_dict:
                    noc_dict = per_object_noc_dict[obj_key]
                    for thresh in noc_thresholds:
                        clicks_needed = noc_dict[thresh]
                        if clicks_needed is not None:
                            noc_metrics[thresh].append(clicks_needed)

            for click_key in metrics_click:
                metrics = metrics_click[click_key]
                count = count_click[click_key]

                if click_key not in metrics_click_avg:
                    metrics_click_avg[click_key] = {}
                    count_click_avg[click_key] = {}

                for key in metrics:
                    if key in ["TP", "FP", "iou_sum"]:
                        if args.distributed:                      
                            metric_data = reduce_tensor_sum(metrics[key].float()).item()
                        else:
                            metric_data = metrics[key]
                    else:
                        if args.distributed:                      
                            metric_data = reduce_tensor(metrics[key].float()).item()
                        else:
                            metric_data = metrics[key]
                    
                    if args.local_rank == 0:
                        # pdb.set_trace()
                        if key in metrics_click_avg[click_key]:
                            metrics_click_avg[click_key][key] += metric_data
                            count_click_avg[click_key][key] += count[key]
                        else:
                            metrics_click_avg[click_key][key] = metric_data
                            count_click_avg[click_key][key] = count[key]

            if args.compute_PQ:
                for click_key in metrics_click_pq:
                    metrics = metrics_click_pq[click_key]

                    if click_key not in metrics_click_pq_avg:
                        metrics_click_pq_avg[click_key] = {}

                    for class_id in metrics:
                        if args.distributed:
                            temp = {"TP": 0, "FP": 0, "iou_sum": 0}
                            for key in metrics[class_id]: # Iterating over TP, FP, iou_sum keys
                                temp[key] = reduce_tensor_sum(metrics[class_id][key]).item()
                            
                            metric_data = temp
                        else:
                            temp = {"TP": 0, "FP": 0, "iou_sum": 0}
                            for key in metrics[class_id]: # Iterating over TP, FP, iou_sum keys
                                temp[key] = metrics[class_id][key]
                            
                            metric_data = temp

                        if args.local_rank == 0:
                            if class_id in metrics_click_pq_avg[click_key]:
                                for key in metric_data:
                                    metrics_click_pq_avg[click_key][class_id][key] += metric_data[key]
                            else:
                                metrics_click_pq_avg[click_key][class_id] = metric_data

            if args.get_confusion:
                for click_key in metrics_click_confusion:
                    batch_confusion_matrix = metrics_click_confusion[click_key]
                    
                    if click_key not in metrics_click_confusion_avg:
                        metrics_click_confusion_avg[click_key] = batch_confusion_matrix
                    else:
                        metrics_click_confusion_avg[click_key] += batch_confusion_matrix

            if args.compute_ap:
                for click_key in store_ap_info_click:
                    ap_info = store_ap_info_click[click_key]

                    if click_key not in total_ap_info_click:
                        total_ap_info_click[click_key] = {}

                    for cur_ap_info in ap_info:
                        label = cur_ap_info["label"].item()
                        score = cur_ap_info["score"].item()
                        iou = cur_ap_info["iou"].item()

                        if label not in total_ap_info_click[click_key]:
                            total_ap_info_click[click_key][label] = []
                        total_ap_info_click[click_key][label].append((score, iou))
                
            # Update memory usage in the tqdm bar
            pbar.set_description(f"Mem: {torch.cuda.max_memory_allocated(device=device)/(1073741824):.2f}")

            # Update the tqdm bar
            pbar.update(1)

    # Call the barrier to ensure all processes have finished
    if args.distributed:
        dist.barrier()

    # Save data to the log file and tensorboard
    if args.local_rank == 0:

        if args.run_openvocab_eval:
            for click_key in scene_data_dict:                
                ap_results = compute_openvocab_metrics(scene_data_dict[click_key], args.stage[0], label_dict[sample['condition'][0]])
                logging.info(f"Open Vocabulary Evaluation Results after epoch {epoch_num+1} =================> ")
                logging.info(f"At click {click_key}: mAP = {ap_results['all_ap']}")
                logging.info(f"At click {click_key}: AP50 = {ap_results['all_ap_50%']}")
                logging.info(f"At click {click_key}: AP25 = {ap_results['all_ap_25%']}")

                # for class_key in ap_results['classes']:
                #     logging.info(f"At click {click_key}: {class_key} mAP = {ap_results['classes'][class_key]['ap']}")
                #     logging.info(f"At click {click_key}: {class_key} AP50 = {ap_results['classes'][class_key]['ap50%']}")
                #     logging.info(f"At click {click_key}: {class_key} AP25 = {ap_results['classes'][class_key]['ap25%']}")

                if args.stage[0]=="scannet":
                    logging.info(f"At click {click_key}: Head mAP = {ap_results['head_ap']}")
                    logging.info(f"At click {click_key}: Common mAP = {ap_results['common_ap']}")
                    logging.info(f"At click {click_key}: Tail mAP = {ap_results['tail_ap']}")

        if args.get_confusion:
            # Log the confusion matrix
            for click_key in metrics_click_confusion_avg:
                logging.info(f"Confusion Matrix for Click {click_key}")
                confusion_matrix = metrics_click_confusion_avg[click_key]
                logging.info(f"{confusion_matrix}")

                print("="*50)
                for row in confusion_matrix:
                    print('\t'.join(f'{x}' for x in row.tolist()))
                print("="*50)

        if args.compute_PQ:
            # Log the PQ, SQ and RQ metrics
            for click_key in metrics_click_pq_avg:
                logging.info(f"Metrics for Click {click_key}")
                metric_data_per_click = metrics_click_pq_avg[click_key]

                PQ_avg = 0
                SQ_avg = 0
                RQ_avg = 0
                count = 0
                for class_id in metric_data_per_click:
                    TP_sum = metric_data_per_click[class_id]["TP"]
                    FP_sum = metric_data_per_click[class_id]["FP"]
                    iou_sum_all = metric_data_per_click[class_id]["iou_sum"]

                    RQ = TP_sum / (TP_sum + 0.5 * FP_sum) if TP_sum + FP_sum != 0 else 0
                    SQ = iou_sum_all / (TP_sum) if TP_sum != 0 else 0 # Avoid division by zero
                    PQ = RQ * SQ

                    PQ_avg += PQ
                    SQ_avg += SQ
                    RQ_avg += RQ
                    count += 1

                    # logging.info(f"For click {click_key} and class {class_id}: PQ = {PQ} SQ = {SQ} RQ = {RQ} ||| TP: {TP_sum} FP: {FP_sum} iou_sum: {iou_sum_all}")
                    logging.info(f"For class: {class_id}, label: {label_dict[sample['condition'][0]][class_id]} : PQ = {PQ:.4f} SQ = {SQ:.4f} RQ = {RQ:.4f}")
                
                PQ_avg /= count
                SQ_avg /= count
                RQ_avg /= count

                logging.info(f"Metrics for Click {click_key}: PQ = {PQ_avg:.4f} SQ = {SQ_avg:.4f} RQ = {RQ_avg:.4f}")

        if args.compute_NOC:
            # Log NoC metrics
            for thresh in noc_thresholds:
                if len(noc_metrics[thresh]) > 0:
                    avg_noc = sum(noc_metrics[thresh]) / len(noc_metrics[thresh])
                else:
                    avg_noc = float('inf')  # If no samples reached the threshold

                logging.info(f"NoC@{thresh}: {avg_noc} over {len(noc_metrics[thresh])} samples")

        for click_key in metrics_click_avg:
            logging.info(f"Metrics for Click {click_key}")
            mIOU = 0
            categories = 0

            for key in metrics_click_avg[click_key]:
                avg = metrics_click_avg[click_key][key] / count_click_avg[click_key][key]

                if "total_IOU" in key:
                    condition = key.split("_")[0]
                    logging.info(f"After epoch {epoch_num + 1}: For dataset {condition}, IoU = {avg} || Sum: {metrics_click_avg[click_key][key]} Count: {count_click_avg[click_key][key]}")
                    writer.add_scalar(f"Val/IoU_{click_key}", avg, epoch_num) # tensorboard logs
                elif "total_text_correct" in key:
                    condition = key.split("_")[0]
                    logging.info(f"After epoch {epoch_num + 1}: For dataset {condition}, Text Accuracy = {avg} || Sum: {metrics_click_avg[click_key][key]} Count: {count_click_avg[click_key][key]}")
                    writer.add_scalar(f"Val/Text_Accuracy_{click_key}", avg, epoch_num)
                elif "text_correct" in key:
                    condition = key.split("_")[0]
                    print(key)
                    label_idx = int(key.split("_")[3])
                    label = label_dict[condition][label_idx]
                    logging.info(f"After epoch {epoch_num + 1}: {condition}_{label_idx}_{label} Text Accuracy = {avg} || Sum: {metrics_click_avg[click_key][key]} Count: {count_click_avg[click_key][key]}")
                elif "total_acc_25" in key:
                    condition = key.split("_")[0]
                    logging.info(f"After epoch {epoch_num + 1}: For dataset {condition}, Acc_25 = {avg} || Sum: {metrics_click_avg[click_key][key]} Count: {count_click_avg[click_key][key]}")
                    writer.add_scalar(f"Val/Acc_25", avg, epoch_num)
                elif "total_acc_50" in key:
                    condition = key.split("_")[0]
                    logging.info(f"After epoch {epoch_num + 1}: For dataset {condition}, Acc_50 = {avg} || Sum: {metrics_click_avg[click_key][key]} Count: {count_click_avg[click_key][key]}")
                    writer.add_scalar(f"Val/Acc_50", avg, epoch_num)
                else:
                    condition = key.split("_")[0]
                    label_idx = int(key.split("_")[2])
                    label = label_dict[condition][label_idx]
                    logging.info(f"After epoch {epoch_num + 1}: {condition}_{label_idx}_{label} IoU = {avg} || Sum: {metrics_click_avg[click_key][key]} Count: {count_click_avg[click_key][key]}")
                    writer.add_scalar(f"Val/{condition}_{label_idx}_{label}_IoU_{click_key}", avg, epoch_num) # tensorboard logs
                    if args.exclude_wall_floor and label in ["wall", "floor"]:
                        continue
                        
                    mIOU += avg
                    categories += 1

            if args.compute_ap:
                #  Compute the class-wise AP results
                total_ap_info = total_ap_info_click[click_key]
                class_scores, mean_ap, mean_ap_50, mean_ap_25 = compute_ap_classewise(total_ap_info, args.exclude_wall_floor)

                for label in class_scores:
                    label_name = label_dict[condition][label]
                    logging.info(f"After epoch {epoch_num + 1}: {condition}_{label_name} mAP = {class_scores[label]['all_ap']}")
                    logging.info(f"After epoch {epoch_num + 1}: {condition}_{label_name} AP50 = {class_scores[label]['all_ap_50%']}")
                    logging.info(f"After epoch {epoch_num + 1}: {condition}_{label_name} AP25 = {class_scores[label]['all_ap_25%']}")

                logging.info(f"After epoch {epoch_num + 1}: mean mAP = {mean_ap}")
                logging.info(f"After epoch {epoch_num + 1}: mean AP50 = {mean_ap_50}")
                logging.info(f"After epoch {epoch_num + 1}: mean AP25 = {mean_ap_25}")

                # Compute the instance-wise AP results here
                total_ap_info = total_ap_info_click[click_key]
                instance_scores = compute_ap_instancewise(total_ap_info)

                logging.info(f"After epoch {epoch_num + 1}: mean instance mAP = {instance_scores['all_ap']}")
                logging.info(f"After epoch {epoch_num + 1}: mean instance AP50 = {instance_scores['all_ap_50%']}")
                logging.info(f"After epoch {epoch_num + 1}: mean instance AP25 = {instance_scores['all_ap_25%']}")
        
            # Safe guard when not returning class-wise results
            categories = 1 if categories==0 else categories
            logging.info(f"After epoch {epoch_num + 1}: mIoU = {mIOU / categories} || Category Count: {categories}{' (excluding wall and floor)'if args.exclude_wall_floor else ''}")
            writer.add_scalar(f"Val/mIOU_{click_key}", mIOU / categories, epoch_num)
            
        logging.info(f"Validation epoch {epoch_num + 1} completed in {pretty_time_delta(time.perf_counter() - t_epoch_start)}")
        logging.info("Validation ended\n")

def setup_dataloaders(args, config_list):
    assert len(args.stage) == len(config_list)
    
    label_dict = {}
    train_dataset_list = []
    val_dataset_list = []
    for stage, config in zip(args.stage, config_list):
        train_dataset_i = build_dataset_single_mask(config, stage, split="train", num_prompt_points=args.num_prompt_points, 
                            num_object_points=args.num_object_points, overfit=args.overfit, use_random_clicks=args.use_random_clicks, use_centroid=False)
        val_dataset_i = build_dataset_single_mask(config, stage, split="val", num_prompt_points=args.num_prompt_points, 
                            num_object_points=args.num_object_points, overfit=args.overfit, use_random_clicks=args.use_random_clicks, use_centroid=args.use_centroid, run_openvocab_eval=args.run_openvocab_eval)

        train_dataset_list.append(train_dataset_i)
        val_dataset_list.append(val_dataset_i)
        
        if stage == "kitti":
            label_dict["SemanticKITTI"] = train_dataset_i.class_labels()
        elif stage == "nuscenes":
            label_dict["NuScenes"] = train_dataset_i.class_labels()
        elif stage == "pandaset":
            label_dict["Pandaset"] = train_dataset_i.class_labels()
        elif stage == "scannet":
            label_dict["ScanNet"] = val_dataset_i.class_labels()  # Taking from val dataset to run openvocab
        elif stage == "scannet20":
            label_dict["ScanNet"] = val_dataset_i.class_labels() # Taking from val dataset to run openvocab
        elif stage == "s3dis":
            label_dict["S3DIS"] = train_dataset_i.class_labels()
        elif stage == "s3disfull":
            label_dict["S3DIS"] = train_dataset_i.class_labels()
        elif stage == "scannetpp":
            label_dict["ScanNetPP"] = train_dataset_i.class_labels()
        elif stage == "stpls3d":
            label_dict["STPLS3D"] = val_dataset_i.class_labels() # Taking from val dataset to run openvocab
        elif stage == "urbanbis":
            label_dict["UrbanBIS"] = train_dataset_i.class_labels()
        elif stage == "dales":
            label_dict["DALES"] = train_dataset_i.class_labels()
        elif stage == "kitti360":
            label_dict["KITTI-360"] = train_dataset_i.class_labels()
        elif stage == "kitti360full":
            label_dict["KITTI-360"] = train_dataset_i.class_labels()
        elif stage == "replica":
            label_dict["Replica"] = val_dataset_i.class_labels() # Taking from val dataset to run openvocab
        elif stage == "hm3d":
            label_dict["HM3D"] = train_dataset_i.class_labels()
        elif stage == "matterport":
            label_dict["Matterport3D"] = train_dataset_i.class_labels()
        elif stage == "kitti360_ss":
            label_dict["KITTI360SS"] = val_dataset_i.class_labels()
        elif stage == "waymo":
            label_dict["Waymo"] = val_dataset_i.class_labels()
        else:
            raise ValueError(f"Dataset not found: {stage}")

    train_dataset = RoundRobinCycleDataset(train_dataset_list)
    val_dataset = MultiDataset(val_dataset_list)

    if args.local_rank == 0:
        for i, stage in enumerate(args.stage):
            logging.info(f'Number of training samples in {stage}: {len(train_dataset_list[i])}')
            logging.info(f'Number of validation samples in {stage}: {len(val_dataset_list[i])}')

        logging.info(f"Total training samples: {len(train_dataset)}")
        logging.info(f"Total validation samples: {len(val_dataset)}")
    
    # If using distributed, we need to intialize distributed sampler
    if args.distributed:
        _, world_size = get_dist_info()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=args.local_rank)

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=args.local_rank)
    else:
        train_sampler = None
        val_sampler = None

    # Initialize the dataloader
    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler, collate_fn=collate_utils.collate_fn)

    # Load validation dataloader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                            shuffle=False, num_workers=args.num_workers,
                                            pin_memory=True, sampler=val_sampler,
                                            collate_fn=collate_utils.collate_fn)

    return train_loader, val_loader, train_sampler, label_dict

def get_config(args):
    config_list = []

    if args.use_var_grid:
        from conf_input_var import semantic_kitti, nuscenes, pandaset, scannet, stpls3d, dales
    else:
        from conf_input import semantic_kitti, nuscenes, pandaset, scannet, s3dis, scannetpp, kitti360, stpls3d, dales, kitti360full, s3disfull, replica, hm3d, matterport, urbanbis, kitti360_ss, waymo

    for stage in args.stage:
        if stage == "kitti":
            config_list.append(semantic_kitti)
        elif stage == "nuscenes":
            config_list.append(nuscenes)
        elif stage == "pandaset":
            config_list.append(pandaset)
        elif stage == "scannet":
            config_list.append(scannet)
        elif stage == "scannet20":
            config_list.append(scannet)
        elif stage == "s3dis":
            config_list.append(s3dis)
        elif stage == "scannetpp":
            config_list.append(scannetpp)
        elif stage == "stpls3d":
            config_list.append(stpls3d)
        elif stage == "dales":
            config_list.append(dales)
        elif stage == "urbanbis":
            config_list.append(urbanbis)
        elif stage == "kitti360":
            config_list.append(kitti360)
        elif stage == "s3disfull":
            config_list.append(s3disfull)
        elif stage == "kitti360full":
            config_list.append(kitti360full)
        elif stage == "replica":
            config_list.append(replica)
        elif stage == "hm3d":
            config_list.append(hm3d)
        elif stage == "matterport":
            config_list.append(matterport)
        elif stage == "kitti360_ss":
            config_list.append(kitti360_ss)
        elif stage == "waymo":
            config_list.append(waymo)
        else:
            raise ValueError(f"Config file not found for stage: {stage}")

    return config_list

def main(args):
    curr_date_time = datetime.now().strftime('%y%m%d_%H%M%S')
    set_seed(42)
    
    if args.dev:
        args.logdir = "dev_logs/"
        args.checkpoint_dir = "dev_checkpoints/"
    
    print(f"Saving logs in {args.logdir}/{args.exp_name}_{curr_date_time}")
    
    # Modify the current experiment name with current date and time
    args.exp_name = f"{curr_date_time}_{args.exp_name}"

    if args.local_rank == 0:
        # Intialize the tensorboard summary writer
        writer = SummaryWriter(f"{args.logdir}/{args.exp_name}/Tensorboard/")
        # Intialize the logger file
        logger = create_logger(name = f'{args.logdir}/{args.exp_name}/log.txt')
    else:
        writer = None
        logger = None

    if args.local_rank == 0:
        # Log available GPUs
        logging.info(f'Available GPUs: {torch.cuda.device_count()}')

        # Log current settings
        logging.info("The current settings are as follows:")
        for arg in vars(args):
            logging.info(f"{arg}: {getattr(args, arg)}")

    # Params for distributed training
    # torch.backends.cudnn.benchmark = True
    if args.distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=20))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate model
    model = SNAP(num_points=args.num_prompt_points, num_merge_blocks=args.num_merge, use_pdnorm=args.use_pdnorm, 
                    use_aux_loss=args.use_aux_loss, use_localized_loss=args.use_localized_loss).to(device)

    # If resuming from an existing checkpoint
    if args.resume:
        model = load_model_weights(model, args.resume)

    # Generate text embeddings for each class in the dataset
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # If using distributed, create a distributed model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
            )

    # Log the total available training and total parameters
    if args.local_rank == 0:
        logging.info(f'Number of trainable params: {sum(p.numel() for p in model.parameters())}')

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Load config
    config_list = get_config(args)
    
    # Load dataloaders
    train_loader, val_loader, train_sampler, label_dict = setup_dataloaders(args, config_list)

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.0, epochs=args.epochs, 
            steps_per_epoch=len(train_loader), anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0,)

    # Define basic variables to take care of training
    scaler = GradScaler() if args.enable_amp else None
    for epoch_num in range(args.epochs):
        if args.val_only:
            validation_epoch(args, model, clip_model, val_loader, writer, epoch_num, device, label_dict)
            break
        
        # Run one training epoch
        train_epoch(args, model, clip_model, train_loader, optimizer, scheduler, scaler, writer, epoch_num, 
                    device, train_sampler, label_dict)
                
        # Save checkpoint weights
        if args.local_rank == 0 and (epoch_num+1) % args.save_ckpt_freq == 0:
            if not os.path.exists(os.path.join(args.checkpoint_dir, args.exp_name)):
                os.makedirs(os.path.join(args.checkpoint_dir, args.exp_name))
            checkpoint_path = os.path.join(args.checkpoint_dir, args.exp_name, f'epoch_{epoch_num+1}.pth')
            if args.distributed:
                torch.save({'model': model.module.state_dict()}, checkpoint_path)
            else:
                torch.save({'model': model.state_dict()}, checkpoint_path)

        # Run validation epoch
        if (epoch_num+1) % args.val_freq == 0:
            validation_epoch(args, model, clip_model, val_loader, writer, epoch_num, device, label_dict)
    
    # Close the summary writer
    if args.local_rank == 0:
        writer.close()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    main(args)
