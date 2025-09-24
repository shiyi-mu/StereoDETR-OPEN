# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from copy import deepcopy
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh, box_cxcylrtb_to_xyxy, box_iou


class StableHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_3dcenter: float = 1, cost_depth: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cec_beta: float = -1, iou_thresh: float = 0.2):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_3dcenter = cost_3dcenter
        self.cost_depth =  cost_depth
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cec_beta = cec_beta
        self.iou_thresh = iou_thresh
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, group_num=11):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # Compute the giou cost betwen boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes_3d"] for v in targets])
         # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()
        bacth_size, class_nums = out_prob.shape
        # 求最后一维度最大值
        out_prob = out_prob.max(-1)[0].reshape(bacth_size, 1)
        # 最后一个维度复制class_nums次
        out_prob = out_prob.expand(-1, class_nums)

        if self.cec_beta > 0:
            giou = (generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcylrtb_to_xyxy(tgt_bbox)) + 1) / 2
            _s = out_prob
            _u = torch.zeros_like(_s)
            _u[:, tgt_ids] = giou
            # scale max(giou) to 1
            _uv = _u.view(bs, num_queries, -1)
            _u_max = _uv.flatten(1, 2).max(-1)[0]
            scalar = (1 / (_u_max + 1e-8))
            scalar = torch.max(scalar, torch.ones_like(scalar))
            _uv = _uv * scalar[:, None, None]
            _u = _uv.view(bs * num_queries, -1)
            # 
            out_prob = (_s * _u.pow(self.cec_beta))

        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        out_3dcenter = outputs["pred_boxes"][:, :, 0: 2].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_3dcenter = torch.cat([v["boxes_3d"][:, 0: 2] for v in targets])
        # Compute the 3dcenter cost between boxes
        cost_3dcenter = torch.cdist(out_3dcenter, tgt_3dcenter, p=1)

        out_depth = outputs['pred_depth'][:, :, 0:1].flatten(0, 1)
        tgt_depth = torch.cat([v['depth'] for v in targets])
        cost_depth = torch.cdist(out_depth, tgt_depth, p=1)

        out_2dbbox = outputs["pred_boxes"][:, :, 2: 6].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_2dbbox = torch.cat([v["boxes_3d"][:, 2: 6] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_2dbbox, tgt_2dbbox, p=1)

        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcylrtb_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_3dcenter * cost_3dcenter \
            + self.cost_class * cost_class \
            + self.cost_giou * cost_giou + self.cost_depth * cost_depth
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        #indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = []
        g_num_queries = num_queries // group_num
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_num):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        indices_match = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        indices_filted = []
       
        for batch_index in range(bs):
            out_bbox_i = outputs["pred_boxes"][batch_index]  # [batch_size * num_queries, 4]
            tgt_bbox_i = targets[batch_index]["boxes_3d"]
            all_iou = box_iou(box_cxcylrtb_to_xyxy(out_bbox_i), box_cxcylrtb_to_xyxy(tgt_bbox_i))[0].view(num_queries, -1)
            indices_i = deepcopy(indices_match[batch_index])
            all_iou_indices = all_iou[indices_i]
            pre_filtered = indices_i[0][all_iou_indices >= self.iou_thresh]
            gt_filtered = indices_i[1][all_iou_indices >= self.iou_thresh]
            indices_filted.append([pre_filtered, gt_filtered])
        return indices_match, indices_filted


def build_fg_Stablematcher(cfg):
    return StableHungarianMatcher(
        cost_class=cfg['set_cost_class'],
        cost_bbox=cfg['set_cost_bbox'],
        cost_3dcenter=cfg['set_cost_3dcenter'],
        cost_depth=cfg['set_cost_depth'],
        cost_giou=cfg['set_cost_giou'],
        cec_beta=cfg['cec_beta'],
        iou_thresh=cfg['iou_thresh'])
        
