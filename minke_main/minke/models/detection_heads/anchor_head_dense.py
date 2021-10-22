from typing import Dict

from numpy.ma import bool_
from torch._C import device
from ...utils import boxes_iou3d_gpu
from ...utils import loss_utils
from .funcs import gen_anchor_sizes
import MinkowskiEngine as ME
import torch
from torch import nn
import numpy as np


class AnchorHeadDense(nn.Module):
    def __init__(self, in_feat, class_names, tensor_stride, pc_range, grid_size, anchor_config, device, bbox_size=7):
        """A detection head using 3D anchors and sparse convolutions to predict.

        Args:
            in_feat (int): Nb of input features.
            class_names (List[String]): List of classes that needs to predict.
            tensor_stride (List[int, 3]): The spatial downsample rate of the input, w.r.t x, y, z axis.
            pc_range (List[int, 6]): Range of point cloud [min_x, min_y, min_z, max_x, max_y, max_z]
            grid_size (Tuple[int, 3]): Spatial shape of input tensor [x, y, z]
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
        pc_range = np.array(pc_range)
        self.in_feat = in_feat
        self.grid_size = np.array(list(grid_size))
        self.class_names = class_names
        self.num_class = len(self.class_names) + 1  # Neg class
        self.device = device
        self.bbox_size = bbox_size
        assert device != 'cpu'

        anchors, num_anchors_per_vox = self.generate_anchors(
            anchor_config, pc_range, self.grid_size
        )
        self.anchors = anchors.to(device)

        self.cls = nn.Sequential(
            nn.Conv3d(
                in_feat, in_feat,
                kernel_size=1
            ),
            nn.BatchNorm3d(in_feat),
            nn.ReLU(),
            nn.Conv3d(
                in_feat, num_anchors_per_vox * self.num_class,
                kernel_size=1
            )
        )

        self.reg_box = nn.Sequential(
            nn.Conv3d(
                in_feat, in_feat,
                kernel_size=1
            ),
            nn.BatchNorm3d(in_feat),
            nn.ReLU(),
            nn.Conv3d(
                in_feat, num_anchors_per_vox * bbox_size,
                kernel_size=1
            )
        )

    @torch.no_grad()
    def encode_label(self, labels):
        """Generate labels for computing loss based on the original labels.

        Args:
            labels (Dict): Dict of labels.
                {
                    'gt_boxes': np.array of gt boxes (#gt, box_size)
                    'gt_names': np.array of gt classes (#gt)
                }

        Returns:
            labels (Dict): Transformed labels.
                {
                    'gt_boxes': torch.Tensor of gt boxes (#gt, box_size)
                    'gt_names': np.array of gt classes (#gt)
                    'gt_cls_code': torch.Tensor of shape (#gt)
                }
        """
        for label in labels:
            gt_names = label['gt_names']
            gt_codes = torch.tensor([self.class_names.index(name) for name in gt_names], dtype=int)
            # gt_onehot = torch.eye(self.num_class)  # num_class = # class_names + 1
            # gt_onehot = gt_onehot[gt_names].detach()
            # label['gt_cls_onehot'] = gt_onehot
            label['gt_cls_code'] = gt_codes
            label['gt_boxes'] = torch.from_numpy(label['gt_boxes'])

        return labels

    @torch.no_grad()
    def assign_targets(self, labels, unmatched_thres=0.01, matched_thres=0.5):
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
        assert self.anchors is not None
        batch_size = len(labels)
        batched_pos_idx = []
        # batched_neg_idx = []
        batched_cls_targets = []
        batched_box_reg_targets = []
        for batch in range(batch_size):
            gt_boxes = labels[batch]['gt_boxes']
            gt_cls = labels[batch]['gt_cls_code']
            anchors = self.anchors.view(-1, self.anchors.shape[-1])
            ious = boxes_iou3d_gpu(anchors.to(self.device), gt_boxes.to(self.device))
            pos_idx, gt_idx = (ious >= matched_thres).nonzero(as_tuple=True)
            neu_idx, neu_gt_idx = ((ious >= unmatched_thres) * (ious < matched_thres)).nonzero(as_tuple=True)
            _, indices = torch.topk(ious[neu_idx, neu_gt_idx], 10)
            neu_idx = neu_idx[indices]
            neu_gt_idx = neu_gt_idx[indices]
            pos_neu_idx = torch.cat([pos_idx, neu_idx])
            # match_inds = (ious >= matched_thres).nonzero(as_tuple=False)
            pos_mask = torch.zeros(anchors.shape[0]).bool()
            pos_mask[pos_neu_idx] = True

            cls_targets = torch.zeros((anchors.shape[0], self.num_class))

            cls_targets[pos_idx, gt_cls[gt_idx]] = 1.0
            cls_targets[neu_idx, gt_cls[neu_gt_idx]] = (ious[neu_idx, neu_gt_idx] / matched_thres).cpu()
            cls_targets[neu_idx, -1] = 1. - cls_targets[neu_idx, gt_cls[neu_gt_idx]]
            cls_targets[~pos_mask, -1] += 1.0  # Neg class

            box_reg_targets = torch.cat([gt_boxes[gt_idx], gt_boxes[neu_gt_idx]])

            batched_pos_idx.append(pos_neu_idx)
            # batched_neg_idx.append(neg_idx)
            batched_cls_targets.append(cls_targets)
            batched_box_reg_targets.append(box_reg_targets)

        batched_box_reg_targets = torch.cat(batched_box_reg_targets)
        batched_cls_targets = torch.stack(batched_cls_targets)
        return {
            'pos_idx': batched_pos_idx,  # Indices of positive predictions (List[torch.Tensor])
            # 'neg_idx': batched_neg_idx,  # Indices of negative predictions (List)
            'cls_targets': batched_cls_targets,  # one-hot targets torch.Tensor((#batch, #anchors, num_class))
            'box_reg_targets': batched_box_reg_targets  # torch.Tensor((#batch * #pos_idx, bbox_size))
        }

    @torch.no_grad()
    def generate_anchors(self, anchor_config: dict, pc_range: np.array, grid_size: np.array):
        anchor_sizes = gen_anchor_sizes(anchor_config)
        num_anchors_per_vox = len(anchor_sizes)
        print(num_anchors_per_vox)

        stride = (pc_range[3:] - pc_range[:3]) / grid_size
        x_coords = torch.arange(
            pc_range[0], pc_range[3], step=stride[0], dtype=torch.float32
        )  # of shape (grid_size[0], )
        y_coords = torch.arange(
            pc_range[1], pc_range[4], step=stride[1], dtype=torch.float32
        )  # of shape (grid_size[1], )
        z_coords = torch.arange(
            pc_range[2], pc_range[5], step=stride[2], dtype=torch.float32
        )  # of shape (grid_size[2], )
        x_coords, y_coords, z_coords = torch.meshgrid([
            x_coords, y_coords, z_coords
        ])  # of shape (grid_size[0], grid_size[1], grid_size[2])
        anchors = torch.stack((x_coords, y_coords, z_coords), dim=-1)  # of shape (*grid_size, 3)

        anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32).repeat(
            (*anchors.shape[:-1], 1, 1)
        )  # of shape (*grid_size, num_anchors_per_vox, 4)

        anchors = anchors[:, :, :, None, :].repeat((1, 1, 1, num_anchors_per_vox, 1))
        # of shape (*grid_size, num_anchors_per_vox, 3)

        anchors = torch.cat((anchors, anchor_sizes), dim=-1)
        # of shape (*grid_size, num_anchors_per_vox, 7)
        anchors[..., :3] += (stride / 2)[None, :]

        return anchors, num_anchors_per_vox

    def forward(self, x: torch.Tensor):
        """Forward.

        Args:
            x (torch.Tensor): Output of backbone 3D.

        Returns:
            cls_pred (torch.Tensor): of shape (batch_size, num_anchors_per_vox * num_cls, *grid_size)
            box_reg (torch.Tensor): of shape (batch_size, num_anchors_per_vox * bbox_size, *grid_size)
        """
        cls_pred = self.cls(x)
        box_reg = self.reg_box(x)
        cls_pred = cls_pred.permute(0, 2, 3, 4, 1).contiguous()
        box_reg = box_reg.permute(0, 2, 3, 4, 1).contiguous()
        for i in range(box_reg.shape[0]):
            offset = box_reg[i].view(*self.grid_size, -1, self.bbox_size)
            offset[..., :3] = offset[..., :3] * self.anchors[..., 3:6] + self.anchors[..., :3]
            offset[..., 3:6] = torch.exp(offset[..., 3:6]) * self.anchors[..., 3:6]
            cos = torch.sigmoid(offset[..., -1]) * torch.cos(self.anchors[..., -1])
            offset[..., -1] = torch.atan2(
                cos,
                torch.sqrt(1 - cos ** 2)
            )
            # offset[..., -1] = torch.sin(offset[..., -1])
            box_reg[i] = offset.view(*self.grid_size, -1)
        return cls_pred, box_reg

    def get_loss(self, cls_pred: torch.Tensor, box_reg: torch.Tensor, targets: Dict, hard_neg_ratio=1.5):
        """Compute loss between predictions and targets assigned.

        Args:
            cls_pred (torch.Tensor): of shape (batch_size, *grid_size, num_anchors_per_vox * num_cls)
            box_reg (torch.Tensor): of shape (batch_size, *grid_size, num_anchors_per_vox * bbox_size)
            targets (Dict): assigned targets returned by function self.assign_targets()
            hard_neg_ratio (float): Ratio of neg / pos samples for hard negative mining
        """
        box_reg = box_reg.view(box_reg.shape[0], -1, self.bbox_size)
        # assert box_reg.shape[0] == targets['box_reg_targets'].shape[0]
        cls_pred = cls_pred.view(cls_pred.shape[0], -1, self.num_class)
        assert cls_pred.shape[0] == targets['cls_targets'].shape[0]

        # Box regression loss - exclude dir
        code_weights = [1.] * (self.bbox_size - 1)
        RegLoss = loss_utils.WeightedSmoothL1Loss(code_weights=code_weights)
        batched_box_reg = []
        for pos_idx, single_box_reg in zip(targets['pos_idx'], box_reg):
            batched_box_reg.append(single_box_reg[pos_idx])
        batched_box_reg = torch.cat(batched_box_reg)
        loss_box_reg = RegLoss(
            batched_box_reg[None, :, :6],
            targets['box_reg_targets'].to(batched_box_reg.device)[None, :, :6]
        ).sum()

        # Dir loss

        loss_dir_reg = batched_box_reg[:, -1] - targets['box_reg_targets'].to(batched_box_reg.device)[:, -1]
        loss_dir_reg = torch.sqrt(2 * (1 - torch.cos(loss_dir_reg))).mean()

        # Clasification loss

        anchors_weights = torch.ones(cls_pred.shape[:2], device=cls_pred.device)
        ClsLoss = loss_utils.SoftmaxFocalClassificationLoss()
        loss_cls = ClsLoss(
            cls_pred,
            targets['cls_targets'].to(cls_pred.device),
            anchors_weights
        ).sum(dim=-1) / self.num_class

        batched_loss_cls = []
        for pos_idx, single_loss_cls in zip(targets['pos_idx'], loss_cls):
            batched_loss_cls.append(single_loss_cls[pos_idx])
        pos_loss_cls = torch.cat(batched_loss_cls).sum()
        # # Hard negative mining

        num_neg_samples = [len(i) * hard_neg_ratio for i in targets['pos_idx']]
        num_neg_samples = int(sum(num_neg_samples))

        _, indices = torch.topk(loss_cls.view(-1), k=num_neg_samples)

        neg_loss_cls = loss_cls.view(-1)[indices].sum()
        loss_cls = pos_loss_cls + neg_loss_cls

        return loss_cls, loss_box_reg, loss_dir_reg
