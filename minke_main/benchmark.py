import torch
import numpy as np
from minke.utils import boxes_iou3d_gpu
import argparse
from pathlib import Path
import yaml
import os
from minke.datasets import *
from minke.pipelines import *
from functools import partial
CUR_DIR = Path(os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help="Path to config file")
    parser.add_argument('--test_batch_size', help="Batch size of test dataloader", default=16, type=int)
    parser.add_argument('--tag', help="Tag of experiment", default='default', type=str)
    parser.add_argument('--ckpt_dir', help="Directory of checkpoints to evaluate", default=None, type=str)
    parser.add_argument('--ckpt', help="Checkpoint to evaluate", default=None, type=str)
    args = parser.parse_args()
    return args


class MatchedPred:
    __slots__ = ['score', 'true', 'cls_code', 'frame_id']

    def __init__(self, confidence_score: float, true_positive: bool, cls_code: float, frame_id: int) -> None:
        self.score = confidence_score
        self.true = true_positive
        self.cls_code = cls_code.int()
        self.frame_id = frame_id

    def __str__(self) -> str:
        return '({}, {}, {}, {})'.format(self.score, self.true, self.cls_code, self.frame_id)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return self.score < other.score


def match_preds_to_gt(preds, labels, iou_thresh, frame_id=None):
    """Match predictions to ground-truths.

    Args:
        preds (List[torch.Tensor]): List of predictions corresponding to a single batch, of shape
            (batch_size, #predictions, 9 [*bbox, confidence_score, cls_code])
        labels (List[dict]):
            {
                'gt_boxes': torch.Tensor
                'gt_names': numpy.array
                'gt_cls_code': torch.Tensor of shape (#gt)
            }
    """
    matched_preds = []
    for i in range(len(preds)):
        pred = preds[i]
        label = labels[i]
        if pred is None:
            continue
        if label['gt_cls_code'].numel() == 0:
            matched_preds += [MatchedPred(e[-2], False, e[-1]) for e in pred.detach().cpu()]
        else:
            ious = boxes_iou3d_gpu(pred[:, :7].cuda(), label['gt_boxes'].cuda())
            match_matrix = torch.zeros_like(ious).bool()
            for gt_idx, gt_cls_code in enumerate(label['gt_cls_code']):
                overlapping_preds_args = torch.nonzero((ious[:, gt_idx] > iou_thresh), as_tuple=True)[0]
                true_pred_args = overlapping_preds_args[pred[overlapping_preds_args, -1] == gt_cls_code]
                if true_pred_args.numel() == 0:
                    continue
                if true_pred_args.numel() > 1:
                    max_pred_arg = torch.argmax(pred[true_pred_args, -2])
                    true_pred_args = true_pred_args[max_pred_arg]
                match_matrix[true_pred_args.item(), gt_idx] = True
            matched_preds += [MatchedPred(e[-2], match_matrix[pred_idx].any(), e[-1], frame_id[i]) for pred_idx, e in enumerate(pred.detach().cpu())]
            # for pred_idx, e in enumerate(pred):
            #     if not match_matrix[pred_idx].any():
            #         print(frame_id[i])
            #         print(MatchedPred(e[-2], match_matrix[pred_idx].any(), e[-1]))
    return matched_preds


@torch.no_grad()
def eval_single_ckpt(pipeline, dataloader, ckpt_path, iou_thresh, class_names):
    pipeline.load_ckpt(ckpt_path)
    match_preds = []
    num_gt_by_cls = np.zeros(len(class_names))
    for d_batch in dataloader:
        # torch.cuda.empty_cache()
        # print(d_batch['frame_id'])
        inp = pipeline.generate_voxels(d_batch['coordinates'], d_batch['features'])

        # loss, preds = pipeline.get_loss_and_predicts(inp, d_batch['labels'], alpha=.25)
        preds = pipeline.predicts(inp)
        labels = d_batch['labels']
        for label in labels:
            for cls_code in label['gt_cls_code']:
                num_gt_by_cls[cls_code] += 1
        match_preds += match_preds_to_gt(preds, labels, iou_thresh, frame_id=d_batch['frame_id'])
        # print(torch.cuda.memory_allocated('cuda:0'))
    # print(match_preds)
    precisions, recalls = get_precision_recall_curve(match_preds, num_gt_by_cls, class_names)
    ret_str, ret_dict = get_map(precisions, recalls)
    print(ckpt_path)
    print(ret_str)
    return ret_dict


def get_map(precisions, recalls):
    results_dict = {}

    # APs for 11-point interpolation
    recall_points = np.linspace(0., 1., num=11)
    ap_11 = np.zeros(recalls.shape[1])
    if len(precisions) > 0:
        for i in range(recalls.shape[1]):
            single_cls_precisions = precisions[:, i]
            interp_precisions = [
                np.amax(single_cls_precisions[recalls[:, i] >= r]) for r in recall_points
                if np.amax(recalls[:, i]) >= r
            ]
            while len(interp_precisions) < len(recall_points) - 1:
                interp_precisions.append(.0)
            interp_precisions = np.array(interp_precisions)
            ap_11[i] = np.sum((recall_points[1:] - recall_points[:-1]) * interp_precisions)

    # APs for every-point interpolation
    # ap_ept = np.zeros(recalls.shape[1])
    # if len(precisions) > 0:
    #     for i in range(recalls.shape[1]):
    #         idx = np.argsort(recalls[:, i])
    #         single_cls_recalls = recalls[:, i][idx]
    #         single_cls_precisions = precisions[:, i][idx]
    #         interp_precisions = np.array([
    #             np.amax(single_cls_precisions[single_cls_recalls >= r])
    #             for r in single_cls_recalls[:-1]
    #         ])
    #         ap_ept[i] = np.sum((single_cls_recalls[1:] - single_cls_recalls[:-1]) * interp_precisions)

    results_dict['ap_11_point'] = [*ap_11]
    map_11 = ap_11.mean()
    results_dict['map_11_point'] = map_11
    # results_dict['ap_every_point'] = ap_ept
    # map_ept = ap_ept.mean()
    # results_dict['map_every_point'] = map_ept

    results_str = 'APs 11-point interpolation: ' + str([*ap_11]) + '\n'
    results_str += 'mAP 11-point interpolation: %.2f \n' % map_11
    # results_str += 'APs every-point interpolation: ' + np.array_str(ap_ept, precision=2) + '\n'
    # results_str += 'mAP every-point interpolation: %.2f \n' % map_ept

    return results_str, results_dict


def get_precision_recall_curve(match_preds: list, num_gt_by_cls: np.ndarray, class_names: list):
    match_preds.sort(reverse=True)
    # print(match_preds)

    precisions = np.zeros((len(match_preds), len(class_names)))
    recalls = np.zeros((len(match_preds), len(class_names)))
    tp = np.zeros(len(class_names))
    fp = np.zeros(len(class_names))
    for i, pred in enumerate(match_preds):
        is_correct = pred.true
        cls_code = pred.cls_code
        if is_correct:
            tp[cls_code] += 1.0
        else:
            fp[cls_code] += 1.0
        precisions[i, :] = tp / (tp + fp + 1e-5)
        recalls[i, :] = tp / (num_gt_by_cls + 1e-8)
    return precisions, recalls


if __name__ == '__main__':
    args = parse_args()
    with open(Path(args.cfg_file), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    out_dir = CUR_DIR / 'output' / args.tag / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg['Dataset']
    data_cfg['Test_cfg']['root_dir'] = Path(data_cfg['Test_cfg']['root_dir'])
    transforms = DataTransforms(None)
    data = eval(data_cfg['Name'])(**data_cfg['Test_cfg'], transforms=transforms)
    data_cfg['Loader_cfg']['batch_size'] = args.test_batch_size
    data_cfg['Loader_cfg']['collate_fn'] = eval(data_cfg['Loader_cfg']['collate_fn'])
    dataloader = prepare_data(data, False, **data_cfg['Loader_cfg'])

    pipeline_cfg = cfg['Pipeline']
    optim = partial(eval(pipeline_cfg['optimizer']['name']), **pipeline_cfg['optimizer']['optim_cfg'])
    pipeline = eval(pipeline_cfg['Name'])(
        pipeline_cfg['backbone_cfg'],
        pipeline_cfg['head_cfg'],
        pipeline_cfg['voxel_cfg'],
        optim,
        False,
        'cuda'
    )

    if args.ckpt is not None:
        # Evaluate a single checkpoint
        ret_dict = eval_single_ckpt(pipeline, dataloader, Path(args.ckpt), 0.25, data_cfg['Test_cfg']['class_names'])
        print(Path(args.ckpt).name)
        print(ret_dict)
    elif args.ckpt_dir is not None:
        args.ckpt_dir = Path(args.ckpt_dir)
        for ckpt in os.listdir(args.ckpt_dir):
            ret_dict = eval_single_ckpt(pipeline, dataloader, args.ckpt_dir / ckpt, 0.25, data_cfg['Test_cfg']['class_names'])
            print(ckpt)
            print(ret_dict)
