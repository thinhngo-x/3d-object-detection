from typing import Dict, List
import numpy as np
from ..utils import box_utils


class DataTransforms:
    def __init__(self, transforms: Dict) -> None:
        """Aggregate types of transformation.

        Args:
            transforms (Dict): Dict of transformations.
                {
                    'transform_name': transform_kwargs
                }
        """
        self.transforms = transforms

    def center_around_objects(self, pc, label, limit=None):
        ind = np.random.randint(0, len(label['gt_boxes']))
        center_obj = label['gt_boxes'][ind, :3][None, :]
        limit = np.array(limit)
        new_origin = np.random.rand(3) * limit - limit / 2
        center_obj += new_origin[None, :]

        pc[:, :3] -= + center_obj
        label['gt_boxes'][:, :3] -= center_obj

        return pc, label

    def crop_within_limit(self, pc, label, limit=None):
        assert limit is not None
        min_bound = np.array(limit[:3])
        max_bound = np.array(limit[3:])
        coords = pc[:, :3]
        pc = pc[((coords >= min_bound) * (coords <= max_bound)).all(axis=1)]
        gt_boxes = label['gt_boxes']
        if len(gt_boxes) > 0:
            mask_label = box_utils.mask_boxes_outside_range_numpy(gt_boxes, limit, min_num_corners=1)
            label['gt_boxes'] = gt_boxes[mask_label]
            label['gt_names'] = label['gt_names'][mask_label]

        return pc, label

    def translate_to_origin(self, pc, label):
        center = (pc[:, :3].max(axis=0) + pc[:, :3].min(axis=0)) / 2
        pc[:, :3] -= center[None, :]
        label['gt_boxes'][:, :3] -= center[None, :]

        return pc, label

    def translate_to_random_origin(self, pc, label, range_origin=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]):
        range_origin = np.array(range_origin)
        new_origin = np.random.rand(3) * (range_origin[3:] - range_origin[:3]) + range_origin[:3]
        center = (pc[:, :3].max(axis=0) + pc[:, :3].min(axis=0)) / 2
        pc[:, :3] += -center[None, :] + new_origin[None, :]
        label['gt_boxes'][:, :3] += -center[None, :] + new_origin[None, :]

        return pc, label

    def random_rescale(self, pc, label, scale_range=[0.95, 1.05]):
        scale = np.random.rand() * (scale_range[1] - scale_range[0]) + scale_range[0]
        pc[:, :3] *= scale
        label['gt_boxes'][:, 6] *= scale

        return pc, label

    def random_flip(self, pc, label, axis=['x', 'y']):
        for ax in axis:
            p = np.random.rand()
            if p > 0.5:
                continue
            if ax == 'x':
                pc[:, 1] *= -1
                label['gt_boxes'][:, 1] *= -1
                label['gt_boxes'][:, 6] *= -1
            elif ax == 'y':
                pc[:, 0] *= -1
                label['gt_boxes'][:, 0] *= -1
                label['gt_boxes'][:, 6] += np.pi
                label['gt_boxes'][:, 6] *= -1
            else:
                raise NotImplementedError
        return pc, label

    def random_rotate(self, pc, label, angle_range=[0, 1.57]):
        theta = np.random.rand() * (angle_range[1] - angle_range[0])
        rot_mat = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        pc[:, :2] = pc[:, :2] @ rot_mat
        label['gt_boxes'][:, :2] = label['gt_boxes'][:, :2] @ rot_mat
        label['gt_boxes'][:, 6] += theta
        return pc, label

    def transform(self, pc, label):
        """Apply transformations on point clouds and their labels.

        Args:
            pc (np.array): of shape (#points, 3+C)
            label (Dict): {
                'gt_boxes' (np.array): of shape (#objects, 7)
                'gt_names (np.array): of shape (#objects)
            }

        Returns:
            pc (np.array)
            label
        """
        if self.transforms is not None:
            for transformation, kwargs in self.transforms.items():
                if kwargs is None:
                    pc, label = getattr(self, transformation)(pc, label)
                else:
                    pc, label = getattr(self, transformation)(pc, label, **kwargs)
        return pc, label
