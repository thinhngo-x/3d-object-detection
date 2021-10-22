from math import sin
import torch
from ...utils.iou3d_nms import iou3d_nms_utils


def nms_single_cls(box_reg: torch.Tensor, box_score: torch.Tensor, cls_code: int,
                   nms_pre_maxsize=4098, nms_post_maxsize=128, nms_thres=0.1,
                   score_thres=0., bbox_size=7):
    """Non-maximum suppressing for a single class.

    Args:
        box_reg (torch.Tensor): of shape (#box, bbox_size)
        box_score (torch.Tensor): of shape (#box,)
        nms_pre_maxsize (int, optional): [description]. Defaults to 4098.
        nms_post_maxsize (int, optional): [description]. Defaults to 128.
        nms_thres (float, optional): [description]. Defaults to 0.1.
        score_thres (float, optional): [description]. Defaults to 0.1.
        bbox_size (int, optional): [description]. Defaults to 7.

    Returns:
        [type]: [description]
    """
    assert box_score.is_cuda
    assert box_reg.is_cuda

    score_mask = box_score > score_thres
    if score_mask.sum() == 0:
        return None
    box_reg_final = box_reg[score_mask]
    box_score_final = box_score[score_mask]
    # print(box_score_final)
    if nms_pre_maxsize < box_score_final.numel():
        _, idx_sorted = torch.topk(box_score_final, k=nms_pre_maxsize)
        box_score_final = box_score_final[idx_sorted]
        box_reg_final = box_reg_final[idx_sorted]
    idx_nms, _ = iou3d_nms_utils.nms_gpu(box_reg_final, box_score_final, thresh=nms_thres)
    box_reg_final = box_reg_final[idx_nms[:nms_post_maxsize]]
    box_score_final = box_score_final[idx_nms[:nms_post_maxsize]][:, None]
    cls_code = cls_code * torch.ones((box_reg_final.shape[0], 1), device=box_reg_final.device)

    return torch.cat((box_reg_final, box_score_final, cls_code), dim=-1)


def batched_nms_multi_cls(batched_box_reg: torch.Tensor, batched_cls_pred: torch.Tensor, batched_obj_pred: torch.Tensor, num_cls,
                          nms_pre_maxsize=4098, nms_post_maxsize=128, nms_thres=0.1,
                          score_thres=0.1, bbox_size=7):
    """Non-maximum suppressing for a single class. Work with batches.

    Args:
        batched_box_reg (List[torch.Tensor]): of shape batch_size * (#voxels, num_anchors_per_vox * bbox_size)
        batched_cls_pred (List[torch.Tensor]): of shape batch_size * (#voxels, num_anchors_per_vox * (num_class+1))
        num_cls ([type]): Number of classes, excluding negative class.
        nms_pre_maxsize (int, optional): [description]. Defaults to 4098.
        nms_post_maxsize (int, optional): [description]. Defaults to 128.
        nms_thres (float, optional): [description]. Defaults to 0.1.
        score_thres (float, optional): [description]. Defaults to 0.1.
        bbox_size (int, optional): [description]. Defaults to 7.

    Returns:
        batched_preds (List[torch.Tensor]): List of predictions corresponding to a single batch, of shape
            (batch_size, #predictions, 9 [*bbox, confidence_score, cls_code])
    """
    assert batched_cls_pred[0].is_cuda
    assert batched_box_reg[0].is_cuda
    # num_cls += 1  # Negative cls
    batch_size = len(batched_cls_pred)
    # torch.save(batched_box_reg, '/home/f90181/boxreg_minke.pt', _use_new_zipfile_serialization=False)
    # batched_box_reg_debug = []

    batched_final_preds = []

    for idx in range(batch_size):
        obj_score = torch.sigmoid(batched_obj_pred[idx].view(-1))
        if num_cls > 1:
            cls_score = torch.nn.functional.softmax(batched_cls_pred[idx].view(-1, num_cls), dim=-1)
        else:
            cls_score = torch.sigmoid(batched_cls_pred[idx].view(-1, num_cls))
        # print(batched_cls_pred[idx].view(-1, num_cls))
        # print((cls_score[:, 0] > 0.5).sum())
        box_reg = batched_box_reg[idx].view(-1, bbox_size).contiguous()
        single_batch_final_preds = []

        for cls in range(num_cls):
            labels = torch.argmax(cls_score[:, :num_cls], dim=-1) == cls
            mask = labels * (cls_score[:, cls] > score_thres)
            mask = mask * (obj_score > 0.5)
            if mask.sum() == 0:
                continue
            box_score_single_cls = cls_score[mask][:, cls].view(-1)
            box_score_single_cls *= obj_score[mask]
            box_reg_single_cls = box_reg[mask]
            single_cls_preds = nms_single_cls(
                box_reg_single_cls,
                box_score_single_cls,
                cls,
                nms_pre_maxsize, nms_post_maxsize,
                nms_thres, score_thres, bbox_size
            )
            # print(box_reg_single_cls.shape)
            if single_cls_preds is not None:
                single_batch_final_preds.append(single_cls_preds)
        if len(single_batch_final_preds) > 0:
            batched_final_preds.append(torch.cat(single_batch_final_preds))
        else:
            batched_final_preds.append(None)
    # torch.save(batched_box_reg_debug, '/home/f90181/boxreg_debug_minke.pt', _use_new_zipfile_serialization=False)
    return batched_final_preds
