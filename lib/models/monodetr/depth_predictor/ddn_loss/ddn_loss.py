import torch
import torch.nn as nn
import math

from .balancer import Balancer
from .focalloss import FocalLoss

# based on:
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/ddn_loss.py


class DDNLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 fg_weight=13,
                 bg_weight=1,
                 downsample_factor=1,
                 depth_sort_reverse=False,
                 bg_value=0,
                 shrink_ratio=1,
                 resolution_ratio_zoomup=1,
                 align_by_3d_center=False):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.bg_value = bg_value
        self.shrink_ratio = shrink_ratio
        self.depth_sort_reverse = depth_sort_reverse
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.resolution_ratio_zoomup = resolution_ratio_zoomup
        self.align_by_3d_center = align_by_3d_center

    def build_target_depth_align_2d_center(self, depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img):
        B, _, H, W = depth_logits.shape
        if self.resolution_ratio_zoomup != 1:
            H = H * self.resolution_ratio_zoomup
            W = W * self.resolution_ratio_zoomup
        if self.bg_value == 0:
            depth_maps = torch.zeros((B, H, W), device=depth_logits.device, dtype=depth_logits.dtype)
        else:
            depth_maps = torch.ones((B, H, W), device=depth_logits.device, dtype=depth_logits.dtype) * self.bg_value
        # Set box corners
        gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2]) # x1,y1
        gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:]) # x2,y2
        gt_boxes2d = gt_boxes2d.long()
        if self.shrink_ratio != 1:
            gt_boxes2d_center = (gt_boxes2d[:, :2] + gt_boxes2d[:, 2:]) // 2 
            gt_boxes2d_size = gt_boxes2d[:, 2:] - gt_boxes2d[:, :2]
            gt_boxes2d_size = (gt_boxes2d_size.float() * self.shrink_ratio).long()
            gt_boxes2d = torch.cat([gt_boxes2d_center - gt_boxes2d_size // 2, gt_boxes2d_center + gt_boxes2d_size // 2], dim=1)
        # Set all values within each box to True
        gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
        gt_center_depth = gt_center_depth.split(num_gt_per_img, dim=0)
        B = len(gt_boxes2d)
        for b in range(B):
            center_depth_per_batch = gt_center_depth[b]
            if self.depth_sort_reverse:
                center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=False)
            else:
                center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
            gt_boxes_per_batch = gt_boxes2d[b][sorted_idx]
            for n in range(gt_boxes_per_batch.shape[0]):
                u1, v1, u2, v2 = gt_boxes_per_batch[n]
                depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]

        return depth_maps
    
    def build_target_depth_align_3d_center(self, depth_logits, gt_boxes3d, gt_center_depth, num_gt_per_img):
        B, _, H, W = depth_logits.shape
        if self.bg_value == 0:
            depth_maps = torch.zeros((B, H, W), device=depth_logits.device, dtype=depth_logits.dtype)
        else:
            depth_maps = torch.ones((B, H, W), device=depth_logits.device, dtype=depth_logits.dtype) * self.bg_value
        # Set box corners
        gt_boxes3d[:, :2] = torch.floor(gt_boxes3d[:, :2]) # center_x, center_y
        gt_boxes3d[:, 2:] = torch.ceil(gt_boxes3d[:, 2:]) # rl tb
        gt_boxes3d = gt_boxes3d.long()
  
        gt_boxes3d_center_x = gt_boxes3d[:, :1] 
        gt_boxes3d_center_y = gt_boxes3d[:, 1:2] 
        gt_boxes3d_l = (gt_boxes3d[:, 2:3].float() * self.shrink_ratio).long()
        gt_boxes3d_r = (gt_boxes3d[:, 3:4].float() * self.shrink_ratio).long()
        gt_boxes3d_t = (gt_boxes3d[:, 4:5].float() * self.shrink_ratio).long()
        gt_boxes3d_b = (gt_boxes3d[:, 5:6].float() * self.shrink_ratio).long()
        gt_boxes3d = torch.cat([gt_boxes3d_center_x - gt_boxes3d_l , 
                                gt_boxes3d_center_y - gt_boxes3d_t,
                                gt_boxes3d_center_x + gt_boxes3d_r , 
                                gt_boxes3d_center_y + gt_boxes3d_b], dim=1)
        # make sure gt_boxes3d big than 0
        gt_boxes3d = torch.clamp(gt_boxes3d, min=0)
        # Set all values within each box to True
        gt_boxes3d = gt_boxes3d.split(num_gt_per_img, dim=0)
        gt_center_depth = gt_center_depth.split(num_gt_per_img, dim=0)
        B = len(gt_boxes3d)
        for b in range(B):
            center_depth_per_batch = gt_center_depth[b]
            if self.depth_sort_reverse:
                center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=False)
            else:
                center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
            gt_boxes_per_batch = gt_boxes3d[b][sorted_idx]
            for n in range(gt_boxes_per_batch.shape[0]):
                u1, v1, u2, v2 = gt_boxes_per_batch[n]
                depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]
        return depth_maps
    
    def build_target_depth_map(self, depth_logits, gt_boxes2d, gt_boxes3d, gt_center_depth, num_gt_per_img):
        if self.align_by_3d_center:
            depth_maps = self.build_target_depth_align_3d_center(depth_logits, gt_boxes3d, gt_center_depth, num_gt_per_img)
        else:
            depth_maps = self.build_target_depth_align_2d_center(depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img)
        return depth_maps
    

    def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                      (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)
       
        return indices

    def forward(self, depth_logits, gt_boxes2d, gt_boxes3d, num_gt_per_img, gt_center_depth):
        """
        Gets depth_map loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img:
            gt_center_depth:
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
        """

        # Bin depth map to create target
        depth_maps = self.build_target_depth_map(depth_logits, gt_boxes2d, gt_boxes3d, gt_center_depth, num_gt_per_img)
        #ipdb.set_trace()
        depth_target = self.bin_depths(depth_maps, target=True)
        #ipdb.set_trace()
        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)
        #ipdb.set_trace()
        # Compute foreground/background balancing
        loss = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d, num_gt_per_img=num_gt_per_img)

        return loss
