import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

def focal_loss(input, target, alpha=0.25, gamma=2.):
    '''
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        alpha: hyper param, default in 0.25
        gamma: hyper param, default in 2.0
    Reference: Focal Loss for Dense Object Detection, ICCV'17
    '''

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    loss = 0

    pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds * alpha
    neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * (1 - alpha)

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss.mean()


def focal_loss_cornernet(input, target, gamma=2.):
    '''
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        gamma: hyper param, default in 2.0
    Reference: Cornernet: Detecting Objects as Paired Keypoints, ECCV'18
    '''

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)

    loss = 0

    pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds
    neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * neg_weights

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss.mean()


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, balance_weights=torch.tensor([1.0], dtype=torch.float)):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.register_buffer("balance_weights", balance_weights)

    def forward(self, classification:torch.Tensor, 
                      targets:torch.Tensor, 
                      gamma:Optional[float]=None, 
                      balance_weights:Optional[torch.Tensor]=None)->torch.Tensor:
        """
            input:
                classification  :[..., num_classes]  linear output
                targets         :[..., num_classes] == -1(ignored), 0, 1
            return:
                cls_loss        :[..., num_classes]  loss with 0 in trimmed or ignored indexes 
        """
        if gamma is None:
            gamma = self.gamma
        if balance_weights is None:
            balance_weights = self.balance_weights

        probs = torch.sigmoid(classification) #[B, N, 1]
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - probs, probs)
        focal_weight = torch.pow(focal_weight, gamma)
        bce = -(targets * nn.functional.logsigmoid(classification)) * balance_weights - ((1-targets) * nn.functional.logsigmoid(-classification)) #[B, N, 1]
        cls_loss = focal_weight * bce

        ## neglect  0.3 < iou < 0.4 anchors
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

        ## clamp over confidence and correct ones to prevent overfitting
        cls_loss = torch.where(torch.lt(cls_loss, 1e-5), torch.zeros(cls_loss.shape).cuda(), cls_loss) #0.02**2 * log(0.98) = 8e-6

        return cls_loss