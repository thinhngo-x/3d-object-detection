from typing import Dict

from ...utils import boxes_iou3d_gpu, loss_utils, box_utils
from .funcs import gen_anchor_sizes
import MinkowskiEngine as ME
import torch
from torch import nn
import numpy as np
from sklearn.metrics import average_precision_score
from ...modules import SparseBasicBlock


class AnchorHeadPrune(nn.Module):
    def __init__(self, in_feat, class_names, voxel_size, anchor_config, device, bbox_size=7, alpha=0.5):
        """A detection head using 3D anchors and sparse convolutions to predict.

        Args:
            in_feat (int): Nb of input features.
            class_names (List[String]): List of classes that needs to predict.
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
        self.in_feat = in_feat
        self.voxel_size = np.array(voxel_size)
        self.class_names = class_names
        self.num_class = len(self.class_names)
        self.device = device
        self.bbox_size = bbox_size
        self.alpha = alpha
        assert device != 'cpu'

        self.anchor_sizes = gen_anchor_sizes(anchor_config)
        self.num_anchors_per_vox = len(self.anchor_sizes)

        # self.conv = nn.Sequential(
        #     SparseBasicBlock(in_feat, in_feat, 3)
        # )

        self.cls = nn.Sequential(
            ME.MinkowskiConvolution(
                in_feat, self.num_anchors_per_vox * self.num_class,
                kernel_size=1, dimension=3, bias=True
            )
        )

        self.objness = nn.Sequential(
            ME.MinkowskiConvolution(
                in_feat, self.num_anchors_per_vox,
                kernel_size=1, dimension=3, bias=True
            )
        )

        self.reg_box = nn.Sequential(
            ME.MinkowskiConvolution(
                in_feat, self.num_anchors_per_vox * bbox_size,
                kernel_size=1, dimension=3, bias=True
            )
        )
        self.weight_initialization()
        self.cls_weight = torch.zeros(self.num_class)
        for m in self.objness.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                nn.init.constant_(m.bias, -4.59)  # According the focal loss

    def weight_initialization(self):
        for m in self.modules():
            print(m)
            if isinstance(m, ME.MinkowskiConvolution) or isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    @torch.no_grad()
    def assign_targets(self, inp, labels, bg_thres=0.2, fg_thres=0.35):
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
        batched_cls_targets = self.anchors.new_zeros((self.anchors.shape[0], 2), dtype=torch.int64)
        batched_box_reg_targets = []
        for batch in range(batch_size):
            gt_boxes = labels[batch]['gt_boxes'].to(self.device)
            gt_cls = labels[batch]['gt_cls_code'].to(self.device)

            batch_mask = batched_anchors[:, 0] == batch
            batch_idx = torch.arange(batched_anchors.shape[0])[batch_mask]
            anchors = batched_anchors[batch_mask][:, 1:].to(self.device)
            ious = boxes_iou3d_gpu(anchors, gt_boxes.to(self.device))
            max_iou_gt_per_anchors = torch.argmax(ious, dim=1)
            fg_mask = ious[torch.arange(ious.shape[0]).to(ious.device), max_iou_gt_per_anchors] > fg_thres
            fg_idx = torch.arange(ious.shape[0]).to(ious.device)[fg_mask]
            fg_gt_idx = max_iou_gt_per_anchors[fg_mask]
            # fg_idx, fg_gt_idx = torch.nonzero(ious > fg_thres, as_tuple=True)

            if ious.numel() > 0:
                max_iou_anchors_per_gt = torch.argmax(ious, dim=0)
                assert max_iou_anchors_per_gt.shape[0] == gt_boxes.shape[0]
                fg_idx = torch.cat([fg_idx, max_iou_anchors_per_gt])
                fg_gt_idx = torch.cat([fg_gt_idx, torch.arange(gt_boxes.shape[0]).to(fg_gt_idx.device)])
            bg_mask = torch.all(ious < bg_thres, dim=-1)
            if ious.numel() > 0:
                bg_mask[max_iou_anchors_per_gt] = False
            bg_idx = torch.arange(ious.shape[0])[bg_mask].to(fg_idx.device)
            batched_cls_targets[batch_idx[fg_idx], 0] = gt_cls[fg_gt_idx]
            # print(torch.unique(gt_cls[fg_gt_idx]))
            batched_cls_targets[batch_idx[bg_idx], 1] = 1

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
        # x = self.conv(x)
        cls_pred = self.cls(x)
        objness_pred = self.objness(x)
        box_reg = self.reg_box(x)
        return cls_pred, box_reg, objness_pred

    def get_loss(self, objness_pred_sparse, cls_pred_sparse, box_reg_sparse,
                 targets: Dict, cur_scale=None, alpha=None, writer=None):
        """Compute loss between predictions and targets assigned.

        Args:
            cls_pred (torch.Tensor): of shape (#voxels, num_anchors_per_vox * num_cls)
            box_reg (torch.Tensor): of shape (#voxels, num_anchors_per_vox * bbox_size)
            targets (Dict): assigned targets returned by function self.assign_targets()
            hard_neg_ratio (float): Ratio of neg / pos samples for hard negative mining
        """
        box_reg = box_reg_sparse.F.view(-1, self.bbox_size)
        cls_pred = cls_pred_sparse.F.view(-1, self.num_class)
        objness_pred = objness_pred_sparse.F.view(-1)
        assert cls_pred.shape[0] == box_reg.shape[0]
        # print('FG : ', targets['fg_idx'].numel())
        if writer is not None:
            writer.add_scalar('train/fg_targets', targets['fg_idx'].numel())

        # Box regression loss
        RegLoss = loss_utils.WeightedSmoothL1Loss()
        loss_box_reg = RegLoss(
            box_reg[targets['fg_idx'], :6],
            targets['box_reg_targets'].to(box_reg.device)[:, :6]
        ).sum() * 0.5 / max(len(targets['fg_idx']), 1)
        # loss_box_reg = 0

        # Dir regression loss

        DirLoss = loss_utils.WeightedSmoothL1SinLoss()
        loss_dir_reg = DirLoss(
            box_reg[targets['fg_idx'], 6],
            targets['box_reg_targets'].to(box_reg.device)[:, 6]
        ).sum() * 0.2 / max(len(targets['fg_idx']), 1)

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
        ) * 0.5 / max(len(targets['fg_idx']), 1)

        loss_reg = [loss_box_reg, loss_dir_reg, loss_corners]

        # Objectness loss

        ObjLoss = loss_utils.SigmoidFocalClassificationLoss(gamma=2., alpha=self.alpha)
        loss_objness_fg = ObjLoss(
            objness_pred[targets['fg_idx']],
            1 - targets['cls_targets'][targets['fg_idx'], 1].to(objness_pred.device),
            weights=None
        ).sum()
        loss_objness_bg = ObjLoss(
            objness_pred[targets['bg_idx']],
            1 - targets['cls_targets'][targets['bg_idx'], 1].to(objness_pred.device),
            weights=None
        ).sum()
        loss_objness = (loss_objness_fg + loss_objness_bg) / (len(targets['fg_idx']) + 1e-8)

        # Clasification loss

        # anchors_weights = torch.ones(cls_pred.shape[:2], device=cls_pred.device)
        if self.num_class > 1:
            ClsLoss = nn.CrossEntropyLoss(
                reduction='mean',
                weight=torch.tensor(
                    [0.0008, 0.6369, 0.0031, 0.0751, 0.0161, 0.1508, 0.0615, 0.0035, 0.0064, 0.0456],
                    device=cls_pred.device
                )
            )
            loss_cls = ClsLoss(
                cls_pred[targets['fg_idx']],
                targets['cls_targets'][targets['fg_idx'], 0].to(cls_pred.device)
            )
            cls = torch.unique(targets['cls_targets'][targets['fg_idx'], 0])
            self.cls_weight[cls] += torch.unique(targets['cls_targets'][targets['fg_idx'], 0], return_counts=True)[1].cpu().detach()
        else:
            ClsLoss = nn.BCEWithLogitsLoss(reduction='mean')
            loss_cls = ClsLoss(
                cls_pred[targets['fg_idx'], 0],
                targets['cls_targets'][targets['fg_idx'], 0].float().to(cls_pred.device) + 1
            )

        # with torch.no_grad():
        #     pt = torch.sigmoid(objness_pred)[torch.cat((targets['fg_idx'], targets['bg_idx']))]
        #     # print(pt.amax())
        #     pos_gt = 1 - targets['cls_targets'][:, -1][torch.cat((targets['fg_idx'], targets['bg_idx']))]
        #     from sklearn.metrics import precision_recall_curve
        #     import matplotlib.pyplot as plt
        #     precision, recall, _ = precision_recall_curve(pos_gt.cpu().numpy(), pt.cpu().numpy())
        #     plt.plot(recall, precision, label=str(cur_scale))
        #     plt.legend()
        #     plt.savefig('pr_curve.png')
        if writer is not None:
            with torch.no_grad():
                pt = torch.sigmoid(objness_pred)[torch.cat((targets['fg_idx'], targets['bg_idx']))]
                # print(pt.amax())
                pos_gt = 1 - targets['cls_targets'][:, -1][torch.cat((targets['fg_idx'], targets['bg_idx']))]
                ap_cls = average_precision_score(pos_gt.cpu().numpy(), pt.cpu().numpy())
                writer.add_scalar('train/ap_cls_' + str(cur_scale), ap_cls)

        return loss_cls, loss_reg, loss_objness
