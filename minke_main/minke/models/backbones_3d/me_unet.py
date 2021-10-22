from torch import nn
import MinkowskiEngine as ME
from ...modules import SparseBasicBlock
import torch
from ...utils.common_utils import me_to_dense


class MEUNet(ME.MinkowskiNetwork):
    def __init__(self, in_feat, mid_feats, out_feat, strides, device='cpu', D=3) -> None:
        """Initialize.

        Args:
            in_feat (int): Nb of input features
            mid_feats (List[int]): Nb of middle features, e.g [16, 32, 64, 128]
            out_feat (int): Nb of output features
            strides (List[int]): List of downsample strides, e.g [1, 2, 2, 2]
            device (String, optional): Device. Defaults to 'cpu'.
            D (int, optional): Nb of spatial dimensions. Defaults to 3.
        """
        super(MEUNet, self).__init__(D)

        assert(len(mid_feats) == len(strides))
        assert(strides[0] == 1)

        self.device = torch.device(device)

        # Downsample: down_conv -> downsample
        self.down_convs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.in_feat = in_feat
        feat = mid_feats[0]
        self.conv_in = SparseBasicBlock(self.in_feat, feat, D)
        self.tensor_strides = []
        tensor_stride = 1
        for stride in strides:
            tensor_stride *= stride
            self.tensor_strides.append(tensor_stride)

        self.in_feat = feat
        for feat, stride in zip(mid_feats, strides):
            conv = nn.Sequential(
                SparseBasicBlock(self.in_feat, self.in_feat, D),
                SparseBasicBlock(self.in_feat, self.in_feat, D),
                SparseBasicBlock(self.in_feat, self.in_feat, D)
            ).to(self.device)
            self.down_convs.append(conv)
            down = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.in_feat, feat, kernel_size=3, stride=stride, dimension=D
                ),
                ME.MinkowskiBatchNorm(feat),
                ME.MinkowskiReLU()
            ).to(self.device)
            self.downsamples.append(down)
            self.in_feat = feat

        # Upsample: up_conv -> upsample
        self.up_convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.pruning_convs = nn.ModuleList()

        for feat, stride in zip(mid_feats[::-1][1:], strides[::-1][:-1]):
            conv = nn.Sequential(
                SparseBasicBlock(self.in_feat, feat, D),
                SparseBasicBlock(feat, feat, D),
                SparseBasicBlock(feat, feat, D)
            ).to(self.device)
            self.up_convs.append(conv)
            up = nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(
                    feat, feat, kernel_size=3, stride=stride, dimension=D
                ),
                ME.MinkowskiBatchNorm(feat),
                ME.MinkowskiReLU()
            ).to(self.device)
            self.upsamples.append(up)
            self.pruning_convs.append(nn.Sequential(
                ME.MinkowskiConvolution(
                    feat, feat, kernel_size=1, dimension=D
                ),
                ME.MinkowskiBatchNorm(feat),
                ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(
                    feat, 1, kernel_size=1, dimension=D
                )
            )
            ).to(self.device)
            self.in_feat = feat
        self.pruning = ME.MinkowskiPruning()
        self.conv_out = nn.Sequential(
            SparseBasicBlock(self.in_feat, out_feat, D),
            SparseBasicBlock(out_feat, out_feat, D),
            SparseBasicBlock(out_feat, out_feat, D)
        ).to(self.device)

    def forward(self, x, labels=None, voxelization=None):
        x = self.conv_in(x)
        x_strided = []
        self.multi_scales_fm_targets = []
        if labels is None and voxelization is not None:
            testing = True
        else:
            testing = False
        if testing:
            hook_dict = {}

        for conv, down in zip(self.down_convs, self.downsamples):
            x_strided.append(x)
            x = down(conv(x))
        for i, (conv, up, pruning_conv) in enumerate(zip(self.up_convs, self.upsamples, self.pruning_convs)):
            # print(x.F.shape)
            # print(x.C.shape)
            x = up(conv(x)) + x_strided[-(i + 1)]
            # if i == len(self.up_convs) - 1:
            #     continue
            y = pruning_conv(x)
            if testing:
                hook_dict[i] = voxelization.take_coords(y)
                print(self.tensor_strides[-(i + 2)])
            if labels is not None:
                seg_targets = voxelization.get_targets(y, labels, self.tensor_strides[-(i + 2)])
                self.multi_scales_fm_targets.append((y.F.view(-1), seg_targets))
            mask = y.F.view(-1) >= 1.
            thres = 1.
            while mask.sum() <= 1:
                thres -= 0.01
                mask = y.F.view(-1) >= thres

            x = self.pruning(x, mask)
        if testing:
            torch.save(hook_dict, '/home/f90181/data_minke.pt', _use_new_zipfile_serialization=False)
        x = self.conv_out(x)
        return x

    def get_seg_loss(self):
        assert self.multi_scales_fm_targets is not None
        loss = 0.
        for i, (fm, targets) in enumerate(self.multi_scales_fm_targets):
            # print(self.tensor_strides[-(i + 2)])
            # print('Recall seg: ', ((fm > 0.) * (targets > 0.)).sum() / (targets > 0.).sum())
            # print('Precision seg: ', ((fm > 0.) * (targets > 0.)).sum() / (fm > 0.).sum())
            loss_all = torch.nn.functional.soft_margin_loss(fm, targets, reduction='none')
            pos_mask = targets > 0
            neg_mask = targets < 0
            # num_pos = pos_mask.sum()
            # neg_num = min(num_pos * 5, neg_mask.sum())
            # if neg_num < neg_mask.sum():
            #     _, neg_indices = torch.topk(loss_all[neg_mask], k=neg_num, sorted=False)
            #     loss += (loss_all[pos_mask].sum() + loss_all[neg_mask][neg_indices].sum()) / num_pos
            # else:
            if pos_mask.sum() == 0:
                print(pos_mask.sum())
            if neg_mask.sum() == 0:
                print(neg_mask.sum())
            loss += (loss_all[pos_mask].mean() + loss_all[neg_mask].mean())
        return loss
