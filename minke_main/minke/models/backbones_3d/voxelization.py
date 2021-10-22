import torch
import numpy as np
import MinkowskiEngine as ME
from ...utils import box_utils
from MinkowskiEngine.sparse_matrix_functions import MinkowskiSPMMAverageFunction, MinkowskiSPMMFunction


class VoxelGenerator:
    def __init__(self, voxel_size, max_points_per_vox, mode) -> None:
        self.voxel_size = np.array(voxel_size)
        self.max_pts_per_vox = max_points_per_vox
        self.quant_mode = ME.SparseTensorQuantizationMode[mode]

    def voxelize(self, coords: torch.Tensor, feats: torch.Tensor, device='cuda'):
        coords = coords.clone()
        coords[:, 1:] /= torch.Tensor(self.voxel_size)[None, :]
        coords = coords.int()

        # coords, unique_index, inverse_mapping = ME.utils.sparse_quantize(
        #     coordinates=coords,
        #     return_index=True, return_inverse=True
        # )
        # cols = torch.arange(len(feats))
        # size = torch.Size([len(unique_index), len(inverse_mapping)])

        # # UNWEIGHTED_AVERAGE for 2nd feature
        # spmm_avg = MinkowskiSPMMAverageFunction()
        # floor_height = spmm_avg.apply(inverse_mapping, cols, size, feats[:, 1])

        # # UNWEIGHTED_SUM for 1st feature
        # spmm = MinkowskiSPMMFunction()
        # vals = torch.ones(len(feats))
        # intensity = spmm.apply(inverse_mapping, cols, vals, size, feats[:, 0])
        # if self.max_pts_per_vox is not None:
        #     intensity = torch.clamp(intensity, max=self.max_pts_per_vox) / self.max_pts_per_vox

        # feats = torch.cat([intensity[:, None], floor_height[:, None]], dim=1)
        sparse_tensor = ME.SparseTensor(
            coordinates=coords,
            features=feats,
            quantization_mode=self.quant_mode,
            device=device
        )
        if self.max_pts_per_vox is not None:
            sparse_tensor._F = torch.clamp(sparse_tensor._F, max=self.max_pts_per_vox) / self.max_pts_per_vox
        return sparse_tensor

    def devoxelize(self, x: ME.SparseTensor) -> list:
        coords = x.C
        feats = x.F
        batched_feats = []
        for i in torch.unique(coords[:, 0]):
            batch_mask = coords[:, 0] == i
            batched_feats.append(feats[batch_mask])
        return batched_feats

    def take_coords(self, x: ME.SparseTensor):
        coords = x.C.float().cpu()
        coords[:, 1:] *= torch.Tensor(self.voxel_size)[None, :]
        feats = x.F.cpu()
        batched_points = []
        for i in torch.unique(coords[:, 0]):
            batch_mask = coords[:, 0] == i
            batched_points.append(torch.cat([coords[batch_mask, 1:], feats[batch_mask]], dim=-1))
        return batched_points

    @torch.no_grad()
    def get_targets(self, voxels: ME.SparseTensor, labels: dict, tensor_stride: int, ratio_size=0.7) -> torch.Tensor:
        """Get segmentation targets for input voxels.

        Args:
            voxels (ME.SparseTensor): [description]

        Returns:
            torch.Tensor: [description]
        """
        batched_coords = voxels.C
        batched_targets = torch.zeros(batched_coords.shape[0], dtype=int).to(batched_coords.device)
        for i in torch.unique(batched_coords[:, 0]):
            batch_mask = batched_coords[:, 0] == i
            batched_idx = torch.arange(batched_coords.shape[0]).to(batched_coords.device)[batch_mask]
            coords = batched_coords[batch_mask][:, 1:].float()
            label = labels[i]
            coords *= torch.Tensor(self.voxel_size)[None, :].to(coords.device)

            label['gt_boxes'] = label['gt_boxes'].to(coords.device)
            gt_centers = label['gt_boxes'][:, :3]
            limits = torch.amin(label['gt_boxes'][:, 3:6], dim=1).view(1, -1) / 4.

            cdist = torch.cdist(coords, gt_centers, p=1)
            # num_pos = (512 // tensor_stride) // gt_centers.shape[0]
            # if num_pos < coords.shape[0]:
            #     _, pos_indices = torch.topk(cdist, k=num_pos, dim=0, largest=False)
            #     pos_indices = pos_indices.view(-1)
            #     batched_targets[batched_idx[pos_indices]] = 1.
            # else:
            #     batched_targets[batched_idx] = 1.
            # cdist[torch.argmin(cdist, dim=0), torch.arange(gt_centers.shape[0])] = 0.
            pos_mask = torch.any(cdist <= limits * (1 + np.log(tensor_stride)), dim=1)
            batched_targets[batched_idx[pos_mask]] = 1
        return batched_targets
