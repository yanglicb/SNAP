import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import pdb

def loss_iou(pred_iou, pred_masks, target_masks, threshold):
    batch_size = len(pred_masks)
    loss = 0.0

    for i in range(batch_size):
        pred_iou_i = pred_iou[i]
        pred_mask_i = pred_masks[i]
        target_mask_i = target_masks[i].float()
        
        # Convert to mask by applying threshold
        pred_mask = F.sigmoid(pred_mask_i) > threshold 

        # Calculate IoU for each predicted mask
        intersection = torch.logical_and(pred_mask, target_mask_i).sum(dim=1)
        union = torch.logical_or(pred_mask, target_mask_i).sum(dim=1)

        iou = torch.div(intersection, (union + 1e-10))

        loss_sample = F.mse_loss(pred_iou_i.squeeze(-1), iou)
    
        loss += loss_sample

    loss = loss / batch_size
    return loss


def loss_text(pred_token_list, clip_text_token, target):
    # Compute the cosine distance loss
    batch_size = len(pred_token_list)
    loss = 0.0

    # Normalize the embeddings
    clip_text_token = F.normalize(clip_text_token, p=2, dim=1)

    for i in range(batch_size):
        # Normalize the predicted token
        pred_token_i = F.normalize(pred_token_list[i], p=2, dim=1)
        target_i = target[i]

        # Compute cosine distance
        logits = pred_token_i @ clip_text_token.T

        # Apply binary cross-entropy loss per row
        loss_sample = F.cross_entropy(logits, target_i).mean()
        loss += loss_sample

    loss = loss / batch_size
    return loss        


def loss_focal(pred_masks, target_mask, alpha=1, gamma=2, weights=None):
    batch_size = len(pred_masks)
    loss = 0.0

    for i in range(batch_size):
        # pdb.set_trace()
        pred_mask_i = pred_masks[i]
        target_mask_i = target_mask[i].float()

        if weights is not None:
            weights_i = weights[i]
        else:
            weights_i=None

        # Compute the cross entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(pred_mask_i, target_mask_i, reduction='none', weight=weights_i)
        
        # Get the probability for the true class
        pt = torch.exp(-ce_loss)
        
        # Compute the focal loss
        fl = alpha * (1 - pt) ** gamma * ce_loss
        loss_sample = fl.mean()
        
        loss += loss_sample

    loss = loss / batch_size
    return loss


def loss_ce(pred_masks, target_mask, weights=None):
    batch_size = len(pred_masks)
    loss = 0.0

    for i in range(batch_size):
        # pdb.set_trace()
        pred_mask_i = pred_masks[i]
        target_mask_i = target_mask[i].float()

        if weights is not None:
            weights_i = weights[i]
        else:
            weights_i=None

        loss_sample = F.binary_cross_entropy_with_logits(pred_mask_i, target_mask_i, weight=weights_i)
        loss += loss_sample

    loss = loss / batch_size
    return loss

def loss_dice(pred_masks, target_mask, weights=None):
    batch_size = len(pred_masks)
    loss = 0.0

    for i in range(batch_size):
        pred_mask_i = pred_masks[i]
        target_mask_i = target_mask[i]

        # pred_mask_i = F.softmax(pred_mask_i, dim=0)
        pred_mask_i = F.sigmoid(pred_mask_i)

        loss_sample = dice_loss(pred_mask_i, target_mask_i, eps=1e-6, weights=weights).mean()
        loss += loss_sample

    loss = loss / batch_size
    return loss

def dice_loss(input: Tensor, target: Tensor, eps=1e-6, weights=None):
    input = input.flatten(1)
    target = target.detach().flatten(1)
    weights = weights.flatten(1) if weights is not None else torch.ones_like(input)

    numerator = 2.0 * (input * target * weights).mean(1)
    denominator = (input * weights + target * weights).mean(1)

    soft_iou = (numerator + eps) / (denominator + eps)
    return torch.where(numerator > eps, 1. - soft_iou, soft_iou * 0.)