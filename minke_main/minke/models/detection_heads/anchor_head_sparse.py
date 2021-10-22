from typing import Dict

from ...utils import boxes_iou3d_gpu, loss_utils, box_utils
from .funcs import gen_anchor_sizes
import MinkowskiEngine as ME
import torch
from torch import nn
import numpy as np


class AnchorHeadSparse(nn.Module):
    def __init__(self, in_feat, class_names, tensor_stride, voxel_size, anchor_config, device, bbox_size=7):
        """A detection head using 3D anchors and sparse convolutions to predict.

        Args:
            in_feat (int): Nb of input features.
            class_names (List[String]): List of classes that needs to predict.
            tensor_stride (List[int, 3]): The spatial downsample rate of the input, w.r.t x, y, z axis.
            pc_range (List[int, 6]): Range of point cloud [min_x, min_y, min_z, max_x, max_y, max_z]
            voxel_size (List[int, 3]): Size of a voxel of input tensor [x, y, z]
            anchor_config (dict): Configurations of anchor generator.
                {
                    class_name:
                        {
                            'ratio_xy': List[float],
                            'size_xy': List[float],
                            'size_z': List[float],
                            'dir': List[float]
                        }
                }
            device (String or torch.device)
            bbox_size (int): Size of a bounding box prediction. Default set to 7.
        """
        super().__init__()
        tensor_stride = np.array(tensor_stride)
        self.in_feat = in_feat
        self.voxel_size = np.array(voxel_size)
        self.class_names = class_names
        self.num_class = len(self.class_names) + 1  # Neg class
        self.device = device
        self.bbox_size = bbox_size
        assert device != 'cpu'

        self.anchor_sizes = gen_anchor_sizes(anchor_config)
        self.num_anchors_per_vox = len(self.anchor_sizes)

        self.cls = nn.Sequential(
            ME.MinkowskiConvolution(
                in_feat, self.num_anchors_per_vox * self.num_class,
                kernel_size=3, dimension=3
            )
        )

        self.reg_box = nn.Sequential(
            ME.MinkowskiConvolution(
                in_feat, self.num_anchors_per_vox * bbox_size,
                kernel_size=3, dimension=3
            )
        )

    @torch.no_grad()
    def assign_targets(self, inp, labels, bg_thres=0.4, fg_thres=0.6):
        """Assign targets for each anchors.

        Args:
            labels (List[Dict]): list of labels for each batch. Each batch's label includes:
                {
                    'gt_boxes': torch.Tensor of shape (#gt, box_size)
                    'gt_cls_code': torch.Tensor (int) of shape (#gt)
                }
            unmatched_thres (float, optional): [description]. Defaults to 0.45.
            matched_thres (float, optional): [description]. Defaults to 0.6.

        Returns:
            [type]: [description]
        """
        batched_anchors = self.generate_anchors(inp)
        self.anchors = batched_anchors.to(self.device)
        batch_size = len(labels)
        batched_fg_idx = []
        batched_bg_idx = []
        batched_cls_targets = self.anchors.new_zeros((self.anchors.shape[0], self.num_class))
        batched_box_reg_targets = []
        for batch in range(batch_size):
            gt_boxes = labels[batch]['gt_boxes'].to(self.device)
            gt_cls = labels[batch]['gt_cls_code']

            batch_mask = batched_anchors[:, 0] == batch
            batch_idx = torch.arange(batched_anchors.shape[0])[batch_mask]
            anchors = batched_anchors[batch_mask][:, 1:].to(self.device)
            ious = boxes_iou3d_gpu(anchors, gt_boxes.to(self.device))

            fg_idx, fg_gt_idx = torch.nonzero(ious > fg_thres, as_tuple=True)

            if ious.numel() > 0:
                max_iou_anchors_per_gt = torch.argmax(ious, dim=0)
                assert max_iou_anchors_per_gt.shape[0] == gt_boxes.shape[0]
                fg_idx = torch.cat([fg_idx, max_iou_anchors_per_gt])
                fg_gt_idx = torch.cat([fg_gt_idx, torch.arange(gt_boxes.shape[0]).to(fg_gt_idx.device)])
            bg_mask = torch.all(ious < bg_thres, dim=-1)
            if ious.numel() > 0:
                bg_mask[max_iou_anchors_per_gt] = False
            bg_idx = torch.arange(ious.shape[0])[bg_mask].to(fg_idx.device)
            batched_cls_targets[batch_idx[fg_idx], gt_cls[fg_gt_idx]] = 1.
            batched_cls_targets[batch_idx[bg_idx], -1] = 1.

            box_reg_targets = gt_boxes[fg_gt_idx].clone()
            box_reg_targets[:, :3] = (box_reg_targets[:, :3] - anchors[fg_idx, :3]) / anchors[fg_idx, 3:6]
            box_reg_targets[:, 3:6] = torch.log(box_reg_targets[:, 3:6] / anchors[fg_idx, 3:6])

            batched_fg_idx.append(batch_idx[fg_idx])
            batched_bg_idx.append(batch_idx[bg_idx])
            batched_box_reg_targets.append(box_reg_targets)

        batched_bg_idx = torch.cat(batched_bg_idx)
        batched_fg_idx = torch.cat(batched_fg_idx)
        batched_box_reg_targets = torch.cat(batched_box_reg_targets)

        return {
            'fg_idx': batched_fg_idx,  # Indices of positive predictions (List[torch.Tensor])
            'bg_idx': batched_bg_idx,  # Indices of negative predictions (List)
            'cls_targets': batched_cls_targets,  # one-hot targets torch.Tensor((#batch, #anchors, num_class))
            'box_reg_targets': batched_box_reg_targets  # torch.Tensor((#batch * #pos_idx, bbox_size))
        }

    @torch.no_grad()
    def generate_anchors(self, inp: ME.SparseTensor):
        anchor_sizes = self.anchor_sizes
        num_anchors_per_vox = self.num_anchors_per_vox

        stride = self.voxel_size
        coords = inp.C.to(torch.float16).cpu()
        anchors = coords
        anchors[:, 1:] = coords[:, 1:] * torch.Tensor(self.voxel_size)[None, :]  # of shape (#voxels, 4)

        anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32).repeat(
            (*anchors.shape[:-1], 1, 1)
        )  # of shape (#voxels, num_anchors_per_vox, 4)

        anchors = anchors[:, None, :].repeat((1, num_anchors_per_vox, 1))
        # of shape (#voxels, num_anchors_per_vox, 4)

        anchors = torch.cat((anchors, anchor_sizes), dim=-1)
        # of shape (#voxels, num_anchors_per_vox, 8) bbox_size + 1 (batch_coord)
        anchors[..., 1:4] += (stride / 2)[None, :]

        return anchors.view(-1, 8)

    def forward(self, x: torch.Tensor):
        """Forward.

        Args:
            x (torch.Tensor): Output of backbone 3D.

        Returns:
            cls_pred (torch.Tensor): of shape (#voxels, num_anchors_per_vox * num_cls)
            box_reg (torch.Tensor): of shape (#voxels, num_anchors_per_vox * bbox_size)
        """
        cls_pred = self.cls(x)
        box_reg = self.reg_box(x)
        return cls_pred, box_reg

    def get_loss(self, cls_pred_sparse: torch.Tensor, box_reg_sparse: torch.Tensor, targets: Dict, hard_neg_ratio=0., alpha=.25, writer=None):
        """Compute loss between predictions and targets assigned.

        Args:
            cls_pred (torch.Tensor): of shape (#voxels, num_anchors_per_vox * num_cls)
            box_reg (torch.Tensor): of shape (#voxels, num_anchors_per_vox * bbox_size)
            targets (Dict): assigned targets returned by function self.assign_targets()
            hard_neg_ratio (float): Ratio of neg / pos samples for hard negative mining
        """
        # offset = box_reg.F.view(-1, self.bbox_size)
        # offset[:, :3] = offset[:, :3] * self.anchors[:, 4:7] + self.anchors[:, 1:4]
        # offset[:, 3:6] = torch.exp(offset[:, 3:6]) * self.anchors[:, 4:7]
        # cos = torch.sigmoid(offset[:, -1]) * torch.cos(self.anchors[:, -1])
        # offset[..., -1] = torch.atan2(
        #     cos,
        #     torch.sqrt(1 - cos ** 2)
        # )
        # offset[..., -1] = torch.sin(offset[..., -1])
        # box_reg._F = offset.view(-1, self.bbox_size)
        box_reg = box_reg_sparse.F.view(-1, self.bbox_size)
        # assert box_reg.shape[0] == targets['box_reg_targets'].shape[0]
        cls_pred = cls_pred_sparse.F.view(-1, self.num_class)
        assert cls_pred.shape[0] == box_reg.shape[0]
        # print('FG : ', targets['fg_idx'].numel())
        if writer is not None:
            writer.add_scalar('train/fg_targets', targets['fg_idx'].numel())

        # Box regression loss
        RegLoss = loss_utils.WeightedSmoothL1Loss()
        loss_box_reg = RegLoss(
            box_reg[targets['fg_idx'], :6],
            targets['box_reg_targets'].to(box_reg.device)[:, :6]
        ).sum() / max(len(targets['fg_idx']), 1)
        # loss_box_reg = 0

        # Dir regression loss

        DirLoss = loss_utils.WeightedSmoothL1SinLoss()
        loss_dir_reg = DirLoss(
            box_reg[targets['fg_idx'], 6],
            targets['box_reg_targets'].to(box_reg.device)[:, 6]
        ).sum() / max(len(targets['fg_idx']), 1)

        # Corner loss
        decoded_box_reg = box_utils.decode_boxes(
            box_reg[targets['fg_idx'], :].view(-1, self.bbox_size),
            self.anchors[targets['fg_idx'], 1:],
            self.bbox_size
        )

        corners_reg = box_utils.boxes_to_corners_3d(decoded_box_reg)
        with torch.no_grad():
            decoded_box_target = box_utils.decode_boxes(
                targets['box_reg_targets'].to(self.device),
                self.anchors[targets['fg_idx'], 1:],
                self.bbox_size
            )
            corners_target = box_utils.boxes_to_corners_3d(decoded_box_target)
        loss_corners = nn.functional.smooth_l1_loss(
            corners_reg,
            corners_target,
            reduction='sum'
        ) / max(len(targets['fg_idx']), 1)

        loss_reg = [loss_box_reg, loss_dir_reg, loss_corners]
        # Clasification loss

        # anchors_weights = torch.ones(cls_pred.shape[:2], device=cls_pred.device)
        ClsLoss = loss_utils.SoftmaxFocalClassificationLoss(gamma=2., alpha=alpha)
        loss_cls = ClsLoss(
            cls_pred,
            targets['cls_targets'].to(cls_pred.device),
            weights=None
        ).view(-1)
        if writer is not None:
            with torch.no_grad():
                pt = torch.nn.functional.softmax(cls_pred, dim=-1)
                precision = ((pt[..., 0] > 0.5) * targets['cls_targets'][..., 0]).sum() / ((pt[..., 0] > 0.5).sum() + 1e-8)
                recall = ((pt[..., 0] > 0.5) * targets['cls_targets'][..., 0]).sum() / (targets['cls_targets'][..., 0].sum() + 1e-8)
                # pred_by_truth = (pt[..., 0] > 0.5).sum().cpu() / targets['cls_targets'][..., 0].sum()
                writer.add_scalar('train/precision_cls', precision.item())
                writer.add_scalar('train/recall_cls', recall.item())

        # print('Positive cls loss: ', pos_loss)
        # # Hard negative mining

        # num_neg_samples = min(int(len(targets['fg_idx']) * hard_neg_ratio), len(targets['bg_idx']))
        fg_loss = loss_cls[targets['fg_idx']].sum() / max(len(targets['fg_idx']), 1)
        # _, indices = torch.topk(loss_cls[targets['bg_idx']], k=num_neg_samples, sorted=False)
        # bg_loss = loss_cls[targets['bg_idx']][indices].sum() / max(len(targets['fg_idx']), 1)

        bg_loss = loss_cls[targets['bg_idx']].sum() / max(len(targets['bg_idx']), 1)

        loss_cls = fg_loss + bg_loss

        return loss_cls, loss_reg
