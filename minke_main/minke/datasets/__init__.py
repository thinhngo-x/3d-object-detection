import numpy as np
from .sun_rgbd import SunRGBDDataset
from .chaufferie import ChaufferieDataset
import MinkowskiEngine as ME
import torch
from torch.utils.data import DataLoader
from .transforms import DataTransforms
from torch.utils.data import DistributedSampler as _DistributedSampler
from ..utils import common_utils

__all__ = [
    'SunRGBDDataset',
    'ChaufferieDataset',
    'prepare_data',
    'minkowski_collate_fn',
    'custom_collate_fn',
    'DataTransforms',
    'DistributedSampler'
]


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def minkowski_collate_fn(dataset):
    """Collate function for torch.utils.data.DataLoader.

    Args:
        dataset (minke.datasets): Dataset generated, where 1 element contains:
            {
                'coords': Coordinates of a point cloud
                'labels': Labels of a point cloud
                'frame_id': Index of a point cloud
            }
    """
    coords_batch, feats_batch, labels_batch, idx_batch = ME.utils.sparse_collate(
        [d['coords'] for d in dataset],
        [d['features'] for d in dataset],
        [d['labels'] for d in dataset],
        dtype=torch.float32
    )
    return {
        "coordinates": coords_batch,
        "features": feats_batch,
        "labels": labels_batch
    }


def custom_collate_fn(dataset):
    coords, feats, labels, frame_id = (
        [d['coords'] for d in dataset],
        [d['features'] for d in dataset],
        [d['labels'] for d in dataset],
        [d['frame_id'] for d in dataset]
    )
    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords, dtype=torch.float32)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = labels
    frame_id_batch = torch.from_numpy(np.array(frame_id, dtype=int))

    return {
        "coordinates": bcoords,
        "features": feats_batch,
        "labels": labels_batch,
        "frame_id": frame_id_batch
    }


def prepare_data(dataset, dist, **kwargs):
    """[summary]

    Args:
        dataset (minke.datasets): Dataset generated.
    """
    if dist:
        if dataset.training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    kwargs['shuffle'] = (kwargs['shuffle'] and dataset.training) if sampler is None else None
    return DataLoader(dataset, sampler=sampler, **kwargs), sampler
