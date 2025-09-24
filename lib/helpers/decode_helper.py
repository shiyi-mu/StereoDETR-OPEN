import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle
from utils import box_ops


def decode_detections(dets, info, calibs, cls_mean_size, threshold, decoupled):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    results = {}
    results_samples = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        samples = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold:
                continue

            # 2d bboxs decoding
            scale_y = info['img_size_original'][i][1] / (info['upper'][i]+ info['img_size_croped'][i][1])
            scale_w = info['img_size_original'][i][0] / info['img_size_croped'][i][0]
            x = dets[i, j, 2] * info['img_size_original'][i][0]
            y = (dets[i, j, 3] * info['img_size_croped'][i][1] + info['upper'][i])*scale_y
            w = dets[i, j, 4] * info['img_size_original'][i][0] 
            h = dets[i, j, 5] * info['img_size_croped'][i][1]*scale_y
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

            # 3d bboxs decoding
            # depth decoding
            depth = dets[i, j, 6]

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]
            if decoupled:
                # / depth[i] * calib.P2[0, 0] * scale_w
                dimensions[0] = dimensions[0] * depth / calibs[i].P2[0, 0] * scale_y
                dimensions[1:] = dimensions[1:] * depth / calibs[i].P2[0, 0] * scale_w
            # positions decoding
            x3d = dets[i, j, 34] * info['img_size_original'][i][0]
            y3d = (dets[i, j, 35] * info['img_size_croped'][i][1] + info['upper'][i])*scale_y

            sample_x = dets[i, j, 36] * info['img_size_original'][i][0]
            sample_y = (dets[i, j, 37] * info['img_size_croped'][i][1] + info['upper'][i])*scale_y
            samples.append([sample_x, sample_y])
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry(alpha, x)
            score = score * dets[i, j, -1]
            # if score > 0.9:
                # score = score * dets[i, j, -1] * 50
            # else:
            #     score = score * dets[i, j, -1]

            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
        # preds = filter_boxes(preds)
        results[info['img_id'][i]] = preds
        results_samples[info['img_id'][i]] = samples
    return results, results_samples

def mini_nms(preds, threshold=0.1):
    """
    非极大值抑制（NMS）实现
    :param boxes: 二维列表，每个子列表为 [x1, y1, x2, y2, score]
    :param threshold: 交并比（IoU）阈值，用于决定是否抑制
    :return: 经过NMS处理后的边界框列表
    """
    if not preds:
        return []

    # 按照置信度分数降序排序
    preds = sorted(preds, key=lambda x: x[-1], reverse=True)

    keep = []  # 用于保存最终保留的边界框索引

    while preds:
        current_box = preds.pop(0)  # 取出当前置信度最高的边界框
        keep.append(current_box)

        # 计算当前边界框与其他边界框的交并比（IoU）
        preds = [box for box in preds if iou(current_box, box) < threshold]

    return keep

def filter_boxes(boxes, threshold=0.3):
    """
    过滤框，如果有交集，保留深度值较小的框。
    """
    n = len(boxes)
    to_remove = set()  # 用于记录需要移除的框的索引

    # 遍历所有框对
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if iou_of_box1(boxes[i], boxes[j]) > threshold:
                # 如果有交集，保留深度值较小的框
                if boxes[i][-3] > boxes[j][-3]:
                    to_remove.add(i)

    # 保留未被标记移除的框
    # result = [boxes[k] for k in range(n) if k not in to_remove]
    # 衰减 被遮挡框的置信度
    for k in range(n):
        if k in to_remove:
            boxes[k][-1] = boxes[k][-1] * 10
        else:
            boxes[k][-1] = boxes[k][-1] * 50
    return boxes

def mini_nms_depth(preds, threshold=0.8):
    """
    非极大值抑制（NMS）实现
    :param boxes: 二维列表，每个子列表为 [x1, y1, x2, y2, score]
    :param threshold: 交并比（IoU）阈值，用于决定是否抑制
    :return: 经过NMS处理后的边界框列表
    """
    if not preds:
        return []

    # 按照距离排序
    preds = sorted(preds, key=lambda x: x[-3], reverse=True) # 由远及近

    keep = []  # 用于保存最终保留的边界框索引

    while preds:
        current_box = preds.pop(0) 
        keep.append(current_box)
        # 计算当前边界框与其他边界框的交并比（IoU）
        preds_new = []
        for box in preds:
            if iou(current_box, box) > threshold:
                if current_box[-1] < box[-1]:
                    preds_new.append(current_box) #保留更近的框
                else:
                    preds_new.append(box)
        preds = preds_new
    return keep

def iou(pred1, pred2):
    """
    计算两个边界框的交并比（IoU）
    :param box1: [x1, y1, x2, y2, score]
    :param box2: [x1, y1, x2, y2, score]
    :return: 交并比
    """
    # 计算交集部分
    box1 = pred1[2:6]
    box2 = pred2[2:6]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个边界框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = area1 + area2 - intersection_area

    # 计算交并比
    return intersection_area / union_area if union_area != 0 else 0

def iou_of_box1(pred1, pred2):
    """
    计算两个边界框的交并比（IoU）
    :param box1: [x1, y1, x2, y2, score]
    :param box2: [x1, y1, x2, y2, score]
    :return: 交并比
    """
    # 计算交集部分
    box1 = pred1[2:6]
    box2 = pred2[2:6]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个边界框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])

    # 计算交并比
    return intersection_area / area1 if area1 != 0 else 0


def extract_dets_from_outputs(outputs, K=50, topk=50):
    # get src outputs

    # b, q, c
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']
    pred_sample_points = outputs['pred_sample_points']
    prob = out_logits.sigmoid()

    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), topk, dim=1)

    # final scores
    scores = topk_values
    # final indexes
    topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(-1)
    # final labels
    labels = topk_indexes % out_logits.shape[2]
    
    heading = outputs['pred_angle']
    size_3d = outputs['pred_3d_dim']
    depth = outputs['pred_depth'][:, :, 0: 1]
    sigma = outputs['pred_depth'][:, :, 1: 2]
    sigma = torch.exp(-sigma)


    # decode
    boxes = torch.gather(out_bbox, 1, topk_boxes.repeat(1, 1, 6))  # b, q', 4
    pred_sample_points = torch.gather(pred_sample_points, 1, topk_boxes.repeat(1, 1, 2))  # b, q', 2

    xs3d = boxes[:, :, 0: 1] 
    ys3d = boxes[:, :, 1: 2] 

    heading = torch.gather(heading, 1, topk_boxes.repeat(1, 1, 24))
    depth = torch.gather(depth, 1, topk_boxes)
    sigma = torch.gather(sigma, 1, topk_boxes) 
    size_3d = torch.gather(size_3d, 1, topk_boxes.repeat(1, 1, 3))

    corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)

    xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
    size_2d = xywh_2d[:, :, 2: 4]
    
    xs2d = xywh_2d[:, :, 0: 1]
    ys2d = xywh_2d[:, :, 1: 2]
    pred_sample_points_x = pred_sample_points[:, :, 0: 1]
    pred_sample_points_y = pred_sample_points[:, :, 1: 2]

    batch = out_logits.shape[0]
    labels = labels.view(batch, -1, 1)
    scores = scores.view(batch, -1, 1)
    xs2d = xs2d.view(batch, -1, 1)
    ys2d = ys2d.view(batch, -1, 1)
    xs3d = xs3d.view(batch, -1, 1)
    ys3d = ys3d.view(batch, -1, 1)
    sample_x = pred_sample_points_x.view(batch, -1, 1)
    sample_y = pred_sample_points_y.view(batch, -1, 1)

    detections = torch.cat([labels, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sample_x, sample_y, sigma], dim=2)
    samples = torch.cat([sample_x, sample_y], dim=2)
    return detections


############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)
