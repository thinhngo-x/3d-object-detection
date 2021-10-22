from builtins import sum, zip
from ..models import backbones_3d, detection_heads, post_processing
import torch
from torch import nn
import MinkowskiEngine as ME
from ..utils import common_utils, box_utils
import os


class MinkeDetector(nn.Module):

    def __init__(self, backbone_cfg, head_cfg, voxel_cfg, optimizer,
                 training, device, steps_per_epoch=None, epochs=None) -> None:
        super().__init__()
        self.backbone = backbones_3d.MEFPN(**backbone_cfg).to(device)
        self.head = detection_heads.AnchorHeadPrune(**head_cfg).to(device)
        self.voxel_generator = backbones_3d.VoxelGenerator(**voxel_cfg)
        self.training = training
        self.cur_iter = 0
        if training:
            self.params = [*self.backbone.parameters(), *self.head.parameters()]
            self.optimizer = optimizer(self.params)
            self.lr_scheduler = None
            if steps_per_epoch is not None:
                self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=self.optimizer.param_groups[0]["lr"],
                    steps_per_epoch=steps_per_epoch, epochs=epochs,
                    pct_start=0.4, div_factor=5
                )
            self.train()
        else:
            self.eval()

    def generate_voxels(self, coords, feats):
        return self.voxel_generator.voxelize(coords, feats)

    def get_loss(self, voxel_inp, labels, alpha=.2, writer=None, only_seg=False):
        multi_scale_loss = self.backbone(voxel_inp, self.head, labels, writer)
        loss_cls, loss_box_reg, loss_objness = zip(*multi_scale_loss)
        loss_box, loss_dir, loss_corner = zip(*loss_box_reg)
        loss_box_reg = [sum(list(loss_box)), sum(list(loss_dir)), sum(list(loss_corner))]
        return sum(list(loss_cls)), loss_box_reg, sum(list(loss_objness))

    def step(self, grad_norm_clip=5.):
        self.cur_iter += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, grad_norm_clip)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            return grad_norm, self.optimizer.param_groups[0]["lr"]
        return grad_norm, None

    def predicts(self, voxel_inp):
        # assert not self.training
        multi_scale_preds = self.backbone(voxel_inp, self.head)
        cls_pred, box_reg, objness_pred = zip(*multi_scale_preds)
        batched_cls_pred = []
        batched_box_reg = []
        batched_obj_pred = []
        for i in range(len(list(cls_pred))):
            batched_cls_pred.append(self.voxel_generator.devoxelize(cls_pred[i]))
            batched_box_reg.append(self.voxel_generator.devoxelize(box_reg[i]))
            batched_obj_pred.append(self.voxel_generator.devoxelize(objness_pred[i]))
        box_reg = [torch.cat(e, dim=0) for e in zip(*batched_box_reg)]
        cls_pred = [torch.cat(e, dim=0) for e in zip(*batched_cls_pred)]
        obj_pred = [torch.cat(e, dim=0) for e in zip(*batched_obj_pred)]
        batched_preds = post_processing.batched_nms_multi_cls(
            box_reg, cls_pred, obj_pred, self.head.num_class
        )
        return batched_preds

    def save_ckpt(self, out_dir, cur_epoch, interval=5, max_ckpts=10):
        if not out_dir.exists():
            os.mkdir(out_dir)
        if cur_epoch % interval != 0:
            return None
        ckpt_saved = os.listdir(out_dir)
        ckpt_saved = [int(e.split('.')[0].split('_')[1]) for e in ckpt_saved]
        if len(ckpt_saved) >= max_ckpts:
            os.remove(out_dir / ('checkpoint_' + str(min(ckpt_saved)) + '.pth'))
        if self.lr_scheduler is not None:
            lr_scheduler_state = self.lr_scheduler.state_dict()
        else:
            lr_scheduler_state = None
        ckpt_dict = {
            'model': self.state_dict(),
            'optim': self.optimizer.state_dict(),
            'epoch': cur_epoch,
            'iter': self.cur_iter,
            'lr_scheduler': lr_scheduler_state
        }
        torch.save(ckpt_dict, out_dir / ('checkpoint_' + str(cur_epoch) + '.pth'))
        return out_dir / ('checkpoint_' + str(cur_epoch) + '.pth')

    def load_ckpt(self, ckpt_path):
        ckpt_dict = torch.load(ckpt_path)
        self.load_state_dict(ckpt_dict['model'])
        if self.training:
            self.optimizer.load_state_dict(ckpt_dict['optim'])
            self.cur_iter = ckpt_dict['iter']
            # if ckpt_dict['lr_scheduler'] is not None:
            #     self.lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
            return ckpt_dict['epoch']


class MinkeDetector_SparseAnchor(nn.Module):

    def __init__(self, backbone_cfg, head_cfg, voxel_cfg, optimizer,
                 training, device, steps_per_epoch=None, epochs=None) -> None:
        super().__init__()
        self.backbone = backbones_3d.MEUNet(**backbone_cfg).to(device)
        self.head = detection_heads.AnchorHeadSparse(**head_cfg).to(device)
        self.voxel_generator = backbones_3d.VoxelGenerator(**voxel_cfg)
        self.training = training
        self.cur_iter = 0
        if training:
            self.params = [*self.backbone.parameters(), *self.head.parameters()]
            self.optimizer = optimizer(self.params)
            self.lr_scheduler = None
            if steps_per_epoch is not None:
                self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=self.optimizer.param_groups[0]["lr"],
                    steps_per_epoch=steps_per_epoch, epochs=epochs,
                    pct_start=0.4, div_factor=10
                )
            self.train()
        else:
            self.eval()

    def generate_voxels(self, coords, feats):
        return self.voxel_generator.voxelize(coords, feats)

    def get_loss(self, voxel_inp, labels, alpha=.2, writer=None, only_seg=False):
        out = self.backbone(voxel_inp, labels, self.voxel_generator)
        loss_seg = self.backbone.get_seg_loss()
        if only_seg:
            return torch.Tensor([0]), [torch.Tensor([0])] * 3, loss_seg
        cls_pred, box_reg = self.head(out)
        targets = self.head.assign_targets(out, labels)
        loss_cls, loss_box_reg = self.head.get_loss(cls_pred, box_reg, targets, alpha=alpha, writer=writer)
        return loss_cls, loss_box_reg, loss_seg

    def step(self, grad_norm_clip=1.):
        self.cur_iter += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, grad_norm_clip)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            return grad_norm, self.optimizer.param_groups[0]["lr"]
        return grad_norm, None

    def get_loss_and_predicts(self, voxel_inp, labels, alpha=.25, writer=None):
        out = self.backbone(voxel_inp, labels, self.voxel_generator)
        loss_seg = self.backbone.get_seg_loss()
        cls_pred, box_reg = self.head(out)
        targets = self.head.assign_targets(out, labels)
        loss_cls, loss_box_reg = self.head.get_loss(cls_pred, box_reg, targets, alpha=alpha, writer=writer)
        anchors = self.head.anchors.to(voxel_inp.device)
        offset = box_reg.F.view(-1, self.head.bbox_size)
        offset[:, :3] = offset[:, :3] * anchors[:, 4:7] + anchors[:, 1:4]
        offset[:, 3:6] = torch.exp(offset[:, 3:6]) * anchors[:, 4:7]
        box_reg._F = offset.view(box_reg.F.shape)

        cls_pred = self.voxel_generator.devoxelize(cls_pred)
        box_reg = self.voxel_generator.devoxelize(box_reg)
        batched_preds = post_processing.batched_nms_multi_cls(
            box_reg, cls_pred, self.head.num_class - 1
        )
        return (loss_seg + loss_cls + loss_box_reg).detach().item(), batched_preds

    def predicts(self, voxel_inp, labels=None):
        assert not self.training
        if labels is not None:
            out = self.backbone(voxel_inp, voxelization=self.voxel_generator)
        else:
            out = self.backbone(voxel_inp)
        cls_pred, box_reg = self.head(out)
        if labels is not None:
            targets = self.head.assign_targets(out, labels)
            torch.save(targets, '/home/f90181/targets_minke.pt', _use_new_zipfile_serialization=False)
            torch.save(self.head.anchors, '/home/f90181/anchors_minke.pt', _use_new_zipfile_serialization=False)
        anchors = self.head.generate_anchors(out).to(voxel_inp.device)

        box_reg._F = box_utils.decode_boxes(
            box_reg.F, anchors[:, 1:], self.head.bbox_size
        )

        cls_pred = self.voxel_generator.devoxelize(cls_pred)
        box_reg = self.voxel_generator.devoxelize(box_reg)
        batched_preds = post_processing.batched_nms_multi_cls(
            box_reg, cls_pred, self.head.num_class - 1
        )
        return batched_preds

    def save_ckpt(self, out_dir, cur_epoch, interval=5):
        if not out_dir.exists():
            os.mkdir(out_dir)
        if cur_epoch % interval != 0 or cur_epoch == 0:
            return None
        ckpt_saved = os.listdir(out_dir)
        ckpt_saved = [int(e.split('.')[0].split('_')[1]) for e in ckpt_saved]
        if len(ckpt_saved) >= 20:
            os.remove(out_dir / ('checkpoint_' + str(min(ckpt_saved)) + '.pth'))
        if self.lr_scheduler is not None:
            lr_scheduler_state = self.lr_scheduler.state_dict()
        else:
            lr_scheduler_state = None
        ckpt_dict = {
            'model': self.state_dict(),
            'optim': self.optimizer.state_dict(),
            'epoch': cur_epoch,
            'iter': self.cur_iter,
            'lr_scheduler': lr_scheduler_state
        }
        torch.save(ckpt_dict, out_dir / ('checkpoint_' + str(cur_epoch) + '.pth'))
        return out_dir / ('checkpoint_' + str(cur_epoch) + '.pth')

    def load_ckpt(self, ckpt_path):
        ckpt_dict = torch.load(ckpt_path)
        self.load_state_dict(ckpt_dict['model'])
        if self.training:
            self.optimizer.load_state_dict(ckpt_dict['optim'])
            self.cur_iter = ckpt_dict['iter']
            if ckpt_dict['lr_scheduler'] is not None:
                self.lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
            return ckpt_dict['epoch']
