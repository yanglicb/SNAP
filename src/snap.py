import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .prompt_encoder import PositionEmbeddingCoordsSine
from .merge import Merge
from .losses.arch_sam_loss_weighted import loss_ce, loss_dice, loss_focal, loss_text, loss_iou
from .ptv3_backbone_domain import PointTransformerV3

class InputEncoder(nn.Module):
    def __init__(self, out_channels = 10):
        super().__init__()

        self.xyz_encoder = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU()
        )

        self.rgb_encoder = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU()
        )

        self.normals_encoder = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU()
        )

        self.intensity_encoder = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.ReLU()
        )

    def forward(self, data_dict):
        # Get the features
        xyz = data_dict["coord"] # Shape (N,3)
        
        if "color" in data_dict:
            rgb = data_dict["color"] # Shape (N,3)
        else:
            rgb = torch.zeros_like(xyz)
        
        if "normal" in data_dict: 
            normals = data_dict["normal"] # Shape (N,3)
        else:
            normals = torch.zeros_like(xyz)
        
        if "strength" in data_dict:
            intensity = data_dict["strength"] # Shape (N,1)
        else:
            intensity = torch.zeros(xyz.shape[0], 1).to(xyz.device)

        # Encode the features
        xyz = self.xyz_encoder(xyz)
        rgb = self.rgb_encoder(rgb)
        normals = self.normals_encoder(normals)
        intensity = self.intensity_encoder(intensity)

        # Concatenate the features
        feat = torch.cat([xyz, rgb, normals, intensity], dim=1)

        # Put the output features into the data_dict
        data_dict["feat"] = feat

        return data_dict

class SNAP(nn.Module):
    def __init__(self, num_points=32, num_merge_blocks=1, mask_threshold=0.5, 
                    pretrained_weights_path=None, use_pdnorm=False, use_aux_loss=False, use_localized_loss=False, return_mid_points=False):
        super().__init__()
        
        out_channels = 10
        self.use_aux_loss = use_aux_loss
        self.use_localized_loss = use_localized_loss
        self.w_max = 2
        self.w_min = 1

        # Define the input encoder
        self.input_encoder = InputEncoder(out_channels=out_channels)

        # Define the backbone
        self.backbone = PointTransformerV3(in_channels=out_channels*4, enable_flash=True, 
                                            pdnorm_bn=use_pdnorm, pdnorm_ln=use_pdnorm, return_mid_points=return_mid_points)
        backbone_out_channels = self.backbone.backbone_out_channels

        # Load pretrained weights if provided
        if pretrained_weights_path:
            checkpoint = torch.load(pretrained_weights_path)
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k == "module.seg_head.weight" or k == "module.seg_head.bias":
                    continue
                else:
                    state_dict[k.replace("module.backbone.", "")] = v
            self.backbone.load_state_dict(state_dict)
            print("Weights loaded")

            self.freeze_backbone()

        # Define the IoU token, masks token and the text token
        self.iou_token = nn.Embedding(1, backbone_out_channels)
        self.mask_token = nn.Embedding(1, backbone_out_channels)
        self.text_token = nn.Embedding(1, backbone_out_channels)

        # Prompt Encoder
        self.prompt_encoder = PositionEmbeddingCoordsSine(3, backbone_out_channels)
        self.text_prompt_encoder = nn.Sequential(
            nn.Linear(512, backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, backbone_out_channels)
        )

        # Positional embeddings for the pointcloud
        self.point_pos_embeddings_module = PositionEmbeddingCoordsSine(3, backbone_out_channels)

        # Merge module to add prompt encodings and cloud encodings
        self.merge = Merge(depth=num_merge_blocks, embedding_dim=backbone_out_channels, num_heads=8, mlp_dim=2048)

        # IoU prediction head
        self.iou_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 1)
        )
        self.iou_sigmoid = nn.Sigmoid()

        # Segmentation head
        self.mask_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, backbone_out_channels),
        )

        # Text prediction head
        self.text_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 512)
        )

        # Define the loss function
        self.dice_loss = loss_dice
        self.focal_loss = loss_focal
        self.ce_loss = loss_ce
        self.iou_loss = loss_iou
        self.text_loss = loss_text

        # Define metrics variables
        self.mask_threshold = mask_threshold


    def freeze_backbone(self):
        # Freeze the backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def run_backbone(self, data_dict):
        ## Run the data through the backbone
        point = self.backbone(data_dict)

        return point

    def compute_loss_weights(self, prompt_points, pointcloud):
        """
        prompt_points: Shape -> M, P, 3
        pointcloud: Shape -> N, 3

        return:
        weights: Shape -> M, N, P
        """
        with torch.no_grad():
            M, P, _ = prompt_points.shape
            N, _ = pointcloud.shape
            weights = torch.zeros(M, N, P).to(prompt_points.device)
            # Loop over P
            for i in range(P):
                prompt_point = prompt_points[:, i, :].unsqueeze(1) # Shape: (M, 1, 3)
                # Compute the distance between the prompt point and the pointcloud
                # 1, N, 3 - M, 1, 3
                distance = torch.norm(pointcloud.unsqueeze(0) - prompt_point, dim=2, p=2) # Shape: (M, N)

                # Compute maximum norms for normalization
                prompt_point_norm = torch.norm(prompt_point, dim=2, p=2)
                pointcloud_norm = torch.norm(pointcloud, dim=1, p=2).unsqueeze(0)

                max_norms = torch.max(prompt_point_norm, pointcloud_norm)

                # Compute normalized distance
                normalized_distance = distance / (max_norms + 1e-10)

                # Initialize weights tensor
                weights_i = torch.empty_like(normalized_distance)  # Shape: (M, N)

                # Apply the weight calculation
                weights_i = torch.where(
                    normalized_distance < 0.5,
                    self.w_max - (self.w_max - self.w_min) * normalized_distance,
                    self.w_min
                )  # Shape: (M, N)

                weights[:, :, i] = weights_i

        return weights

    def run_mask_decoder(self, point, data_dict, mid_points=None, val=False):
        # Define the prompt offsets
        batch_size = data_dict['offset'].shape[0]
        offset_list = [0] + list(point.offset)
        point_offset = [0] + data_dict["point_offset"]

        # region = Generate prompt embeddings
        # Loss weights
        loss_weights = [] # Should be a list containing elements of shape (M, N, P)
        ## Run the prompt points through prompt-encoder
        prompt_encoding_list = []
        for i in range(batch_size):
            prompt_points_i = data_dict["point"][point_offset[i]: point_offset[i+1], :, :]

            # Get the positional embeddings for the prompt points
            prompt_encoding_i = self.prompt_encoder(prompt_points_i)

            # Find closest point_cloud embeddings to the prompt points and add them to prompt_encodings
            M, P, _ = prompt_points_i.shape
            prompt_points_i_reshaped = prompt_points_i.view(-1,3)
            distance_matrix = torch.cdist(point.coord[offset_list[i] : offset_list[i+1], :], prompt_points_i_reshaped, p=2)
            value, idx = torch.min(distance_matrix, dim=0)
            prompt_encoding_i = prompt_encoding_i.view(-1,point.feat.size(1))
            prompt_encoding_i = prompt_encoding_i + point.feat[offset_list[i] : offset_list[i+1], :][idx]
            prompt_encoding_i = prompt_encoding_i.view(M, P, -1) #TODO: Double check this

            # Based on the distance matrix, compute the loss weights
            # Weight computation formula is as follows:
            # wi = wmax − (wmax − wmin) · d where d is the normalized distance of points from the click point.
            # wi = wmin otherwise
            if self.use_localized_loss:
                weights = self.compute_loss_weights(prompt_points_i, point.coord[offset_list[i] : offset_list[i+1], :])
                loss_weights.append(weights)

            # Concatenate the output tokens
            output_tokens = torch.cat([self.iou_token.weight, self.mask_token.weight, self.text_token.weight], dim=0)
            output_tokens = output_tokens.unsqueeze(0).expand(M, -1, -1)
            prompt_encoding_i = torch.cat([output_tokens, prompt_encoding_i], dim=1)
            prompt_encoding_list.append(prompt_encoding_i)
        # endregion

        # region = Expand point embeddings and pass through the transformer
        ## Expand per-pointcloud data in the batch direction to be per-mask
        point_embeddings_list = []
        point_pos_embeddings_list = []
        for i in range(batch_size):
            point_embeddings_i = point.feat[offset_list[i] : offset_list[i+1], :]
            # prompt_encoding_i = prompt_encoding[point_offset[i] : point_offset[i+1], :, :]
            prompt_encoding_i = prompt_encoding_list[i]
            
            # Compute positional embeddings for the pointcloud
            point_pos_embeddings_i = self.point_pos_embeddings_module(data_dict['coord'][offset_list[i] : offset_list[i+1], :])

            point_embeddings_i = torch.repeat_interleave(point_embeddings_i.unsqueeze(0), prompt_encoding_i.shape[0], dim=0)
            point_pos_embeddings_i = torch.repeat_interleave(point_pos_embeddings_i.unsqueeze(0), prompt_encoding_i.shape[0], dim=0)

            point_embeddings_list.append(point_embeddings_i)
            point_pos_embeddings_list.append(point_pos_embeddings_i)

        # Add prompt encoding and cloud encodings together
        point_embeddings_list, prompt_encoding_list = self.merge(prompt_encoding_list, point_embeddings_list, point_pos_embeddings_list)
        # endregion

        # region = Pass iou, mask and text encodings through their respective heads and compute masks, iou and text label
        # Get the individual tokens for iou, mask and text for each batch element
        iou_token_out_list = []
        mask_token_out_list = []
        text_token_out_list = []
        aux_mask_token_out_list = []
        for i in range(batch_size):
            iou_token_out_list.append(prompt_encoding_list[i][:, 0, :])
            mask_token_out_list.append(prompt_encoding_list[i][:, 1, :])
            text_token_out_list.append(prompt_encoding_list[i][:, 2, :])
            aux_mask_token_out_list.append(prompt_encoding_list[i][:, 3:, :])

        # Pass the iou_encoding, mask_encoding and text_encoding through the iou head
        iou_out_list = []
        prompt_mask_logits_list = []
        aux_mask_logits_list = []
        text_logits_list = []
        for i in range(batch_size):
            iou_out_list.append(self.iou_sigmoid(self.iou_head(iou_token_out_list[i])))
            prompt_mask_logits_list.append(self.mask_head(mask_token_out_list[i]))
            aux_mask_logits_list.append(self.mask_head(aux_mask_token_out_list[i]))
            text_logits_list.append(self.text_head(text_token_out_list[i]))

        # Take the dot product of prompt_mask_logits and point encodings
        seg_logits_list = []
        aux_seg_logits_list = []
        for i in range(batch_size):
            point_encoding_i = point_embeddings_list[i] # Shape -> M, N, D
            prompt_mask_logits_i = prompt_mask_logits_list[i].unsqueeze(1) # Shape -> M, 1, D
            aux_mask_logits_i = aux_mask_logits_list[i] # Shape -> M, P, D
            # print("aux_mask_logits_i: ", aux_mask_logits_i.shape)

            # seg_logits_list.append((point_encoding_i @ prompt_mask_logits_i.transpose(-1, -2)).squeeze(dim=-1)) # Shape -> M, N
            seg_logits_mat = (point_encoding_i @ prompt_mask_logits_i.transpose(-1, -2)).squeeze(dim=-1) # Shape -> M_t, N
            seg_logits_list.append(seg_logits_mat)
            aux_seg_logits_list.append((point_encoding_i @ aux_mask_logits_i.transpose(-1, -2))) # Shape -> M, N, P
        # endregion

        if self.use_localized_loss:
            return seg_logits_list, text_logits_list, iou_out_list, aux_seg_logits_list, loss_weights, _, _
        else:
            return seg_logits_list, text_logits_list, iou_out_list, aux_seg_logits_list, _, _, _

    def run_input_encoder(self, data_dict):
        # Run the input encoder for prompt points
        data_dict = self.input_encoder(data_dict)
        return data_dict

    def random_sample_clicks(self, preds, masks, coords, offset, threshold):
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

    def forward(self, data_dict, iterative=False, clicks=10, val=False):
        # Run input encoder
        data_dict = self.run_input_encoder(data_dict)

        # Run backbone
        point = self.run_backbone(data_dict)
        
        if iterative:
            # Run mask decoder
            for sample_iter in range(clicks - 1):
                # Run the mask decoder with the first click
                seg_logits_list, text_out_list, iou_out_list, aux_seg_logits_list, loss_weights, confidence_scores_list, confidence_score_gt_list = self.run_mask_decoder(point, data_dict)
                
                # Based on the ground-truth masks -> Sample new clicks from the error region.
                new_clicks = self.random_sample_clicks(seg_logits_list, data_dict['masks'], data_dict['coord'], data_dict['offset'], self.mask_threshold)

                # add the new clicks
                data_dict['point'] = torch.cat((data_dict['point'], new_clicks), dim=1) 
            
            seg_logits_list, text_out_list, iou_out_list, aux_seg_logits_list, loss_weights, confidence_scores_list, confidence_score_gt_list = self.run_mask_decoder(point, data_dict)
        
        else:
            seg_logits_list, text_out_list, iou_out_list, aux_seg_logits_list, loss_weights, confidence_scores_list, confidence_score_gt_list = self.run_mask_decoder(point, data_dict)

        return seg_logits_list, text_out_list, iou_out_list, aux_seg_logits_list, loss_weights, confidence_scores_list, confidence_score_gt_list
        
    def compute_loss(self, pred, text_out_list, iou_out_list, target, clip_text_token, text_target, aux_masks, loss_weights):
        '''
        pred [(M, N, num_mask_tokens)]: predicted mask logits list
        target [(M, N, 1)]: target ground truth masks
        '''

        total_loss = 0
        loss_out_dict = {}
        
        if self.use_aux_loss:
            # Compute Aux losses
            dice_aux = 0
            ce_aux = 0
            focal_aux = 0
            for batch_idx in range(len(aux_masks)):
                for i in range(aux_masks[batch_idx].shape[2]):
                    if self.use_localized_loss and loss_weights is not None:
                        aux_mask_i = aux_masks[batch_idx][:, :, i]
                        weights = loss_weights[batch_idx][:, :, i]
                        dice_aux += self.dice_loss([aux_mask_i], target, weights=weights)
                        ce_aux += self.ce_loss([aux_mask_i], target, weights=weights)
                        focal_aux += self.focal_loss([aux_mask_i], target, weights=weights)
                    else:
                        aux_mask_i = aux_masks[batch_idx][:, :, i]
                        dice_aux += self.dice_loss([aux_mask_i], target)
                        ce_aux += self.ce_loss([aux_mask_i], target)
                        focal_aux += self.focal_loss([aux_mask_i], target)

            total_loss += dice_aux + ce_aux + focal_aux
            loss_out_dict['Dice_aux'] = dice_aux
            loss_out_dict['CE_aux'] = ce_aux
            loss_out_dict['Focal_aux'] = focal_aux

        # Compute IoU Loss
        iou = self.iou_loss(iou_out_list, pred, target, self.mask_threshold)
        total_loss += iou
        loss_out_dict['IoU Loss'] = iou 

        # Compute Text Loss
        text = self.text_loss(text_out_list, clip_text_token, text_target)
        total_loss += text
        loss_out_dict['Text Loss'] = text 
        
        # Compute Dice loss
        dice = self.dice_loss(pred, target)
        total_loss += dice
        loss_out_dict['Dice Loss'] = dice 

        # Compute Cross-entropy loss 
        ce = self.ce_loss(pred, target)
        total_loss += ce
        loss_out_dict['CE Loss'] = ce 
        # Compute the Focal loss
        focal = self.focal_loss(pred, target)
        total_loss += focal
        loss_out_dict['Focal Loss'] = focal 

        loss_out_dict['Total loss'] = total_loss

        return total_loss, loss_out_dict

    def compute_metrics(self, pred, batch, class_labels, condition, label_dict, pred_text_token_list, clip_text_token, aux_masks, return_classwise=False, iou_out=None):
        '''
        pred [(M, N, num_mask_tokens)]: predicted mask logits list
        batch [(M, N, 1)]: target ground truth masks list
        batch binary masks for each instance
        '''
        batch_metrics = {}
        batch_count = {}
        batch_size = len(pred)

        clip_text_token = F.normalize(clip_text_token, p=2, dim=1)
        correct_num = 0
        total_num = 0
        
        ## Calculate text accuracy here
        for i in range(batch_size):
            pred_token_i = F.normalize(pred_text_token_list[i], p=2, dim=1)
            logits = pred_token_i @ clip_text_token.T

            pred_labels = torch.argmax(logits, dim=1)
            correct_num += (pred_labels == class_labels[i]).sum()
            total_num += pred_token_i.shape[0]

        batch_metrics[f'{condition}_total_text_correct'] = correct_num
        batch_count[f'{condition}_total_text_correct'] = total_num

        ## Calculate IOU here
        """
        Calculate IoU for masks of all the prompt points, then take mean thus getting mIoU
        """
        # Get the offsets to get individual point clouds.
        iou = 0
        count = 0

        acc_25 = 0
        acc_50 = 0

        for i in range(batch_size):
            pred_i = pred[i]
            gt_i = batch[i]

            scene_obj_num = gt_i.shape[0]

            # Convert logits to probabilities
            pred_mask = F.sigmoid(pred_i)

            # Convert to mask by applying threshold
            predicted_masks = pred_mask > self.mask_threshold 

            # Calculate IoU for each predicted mask
            intersection = torch.logical_and(predicted_masks, gt_i).sum(dim=1)
            union = torch.logical_or(predicted_masks, gt_i).sum(dim=1)

            iou_now = torch.div(intersection, (union + 1e-10))

            # Also compute acc_25 and acc_50
            acc_25 += (iou_now>=0.25).sum()
            acc_50 += (iou_now>=0.5).sum()

            iou += iou_now.sum()
            count += scene_obj_num

        batch_metrics[f'{condition}_total_IOU'] = iou
        batch_count[f'{condition}_total_IOU'] = count
        batch_metrics[f'{condition}_total_acc_25'] = acc_25
        batch_metrics[f'{condition}_total_acc_50'] = acc_50
        batch_count[f'{condition}_total_acc_25'] = count
        batch_count[f'{condition}_total_acc_50'] = count


        if not return_classwise:
            return batch_metrics, batch_count, None

        ## Calculate class wise text accuracy here
        for i in range(batch_size):
            labels = class_labels[i]
            pred_token_i = F.normalize(pred_text_token_list[i], p=2, dim=1)
            logits = pred_token_i @ clip_text_token.T

            pred_labels = torch.argmax(logits, dim=1)
            for j, label in enumerate(labels):
                key = f'{condition}_text_correct_{label}'
                if key in batch_count:
                    batch_count[key] += 1
                else:
                    batch_count[key] = 1

                if key in batch_metrics:
                    batch_metrics[key] += (pred_labels[j] == label).sum()
                else:
                    batch_metrics[key] = (pred_labels[j] == label).sum()


        ap_info = []
        ## Calculate class-wise mIOU here
        for i in range(batch_size):
            labels = class_labels[i]
            pred_i = pred[i]
            gt_i = batch[i]
            conf_score_i = iou_out[i]

            # Convert logits to probabilities
            pred_mask = F.sigmoid(pred_i)
        
            for j, label in enumerate(labels):
                key = f'{condition}_IOU_{label}'

                if key in batch_count:
                    batch_count[key] += 1
                else:
                    batch_count[key] = 1
                
                pred_class_mask = pred_mask[j, :] > self.mask_threshold 
                gt_class_mask = gt_i[j, :]

                intersection = torch.logical_and(pred_class_mask, gt_class_mask).sum(dim=0)
                union = torch.logical_or(pred_class_mask, gt_class_mask).sum(dim=0)

                cur_ap_info = {
                    "label": label,
                    "score": conf_score_i[j][0],
                    "iou": torch.div(intersection, (union + 1e-10)),
                }

                ap_info.append(cur_ap_info)

                if key in batch_metrics:
                    batch_metrics[key] += torch.div(intersection, (union + 1e-10))
                else:
                    batch_metrics[key] = torch.div(intersection, (union + 1e-10))

        return batch_metrics, batch_count, ap_info


