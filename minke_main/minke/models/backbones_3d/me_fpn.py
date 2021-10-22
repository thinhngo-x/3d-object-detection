from torch import nn
import MinkowskiEngine as ME
from ...modules import SparseBasicBlock
import torch
from ...utils.common_utils import me_to_dense
from ...utils import box_utils


class MEFPN(ME.MinkowskiNetwork):
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
        super(MEFPN, self).__init__(D)

        assert(len(mid_feats) == len(strides))
        assert(strides[0] == 1)

        self.device = torch.device(device)

        # Downsample: down_conv -> downsample
        self.down_convs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.in_feat = in_feat
        feat = mid_feats[0]
        self.conv_in = SparseBasicBlock(self.in_feat, feat, D)

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
                    self.in_feat, feat, kernel_size=3, stride=stride, dimension=D, bias=True
                ),
                ME.MinkowskiBatchNorm(feat),
                ME.MinkowskiReLU()
            ).to(self.device)
            self.downsamples.append(down)
            self.in_feat = feat

        # Upsample: up_conv -> upsample
        self.decode_convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for feat, stride in zip(mid_feats[::-1][1:], strides[::-1][:-1]):
            # print(in_feat)
            conv = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.in_feat, self.in_feat, kernel_size=3, dimension=D, bias=True
                ),
                ME.MinkowskiBatchNorm(self.in_feat),
                ME.MinkowskiReLU()
            ).to(self.device)
            self.decode_convs.append(conv)
            conv = nn.Sequential(
                ME.MinkowskiConvolution(
                    feat, self.in_feat, kernel_size=1, dimension=D
                ),
                ME.MinkowskiBatchNorm(self.in_feat)
            ).to(self.device)
            self.skip_convs.append(conv)
            up = nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(
                    self.in_feat, self.in_feat, kernel_size=3, stride=stride, dimension=D
                ),
                ME.MinkowskiBatchNorm(self.in_feat)
            ).to(self.device)
            self.upsamples.append(up)
            # self.in_feat = 64
        self.pruning = ME.MinkowskiPruning()
        self.relu = ME.MinkowskiReLU()
        self.decode_convs.append(nn.Sequential(
            ME.MinkowskiConvolution(
                self.in_feat, self.in_feat, kernel_size=3, dimension=D, bias=True
            ),
            ME.MinkowskiBatchNorm(self.in_feat),
            ME.MinkowskiReLU()
        ).to(self.device)
        )
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution) or isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x, anchor_head, labels=None, writer=None):
        x = self.conv_in(x)
        x_strided = []
        multi_scale_rets = []  # Loss if training and Preds if testing
        debug = True
        if labels is not None:
            debug = False
        if debug:
            hook_dict = {}
        for conv, down in zip(self.down_convs, self.downsamples):
            x_strided.append(x)
            x = down(conv(x))
            # print(x.C)
        x = self.decode_convs[0](x)
        for i, (conv, up, skip) in enumerate(zip(self.decode_convs[1:], self.upsamples, self.skip_convs)):
            # import time
            # s = time.time()
            x = self.relu(up(x) + skip(x_strided[-(i + 1)]))
            y = conv(x)
            # print("Up ", time.time() - s)
            if labels is not None:
                loss, mask = self.get_loss_and_prune(y, labels, anchor_head, cur_scale=i, writer=writer)
                multi_scale_rets.append(loss)
            else:
                preds, mask = self.predict_and_prune(y, anchor_head)
                multi_scale_rets.append(preds)
            assert mask.shape[0] == y.C.shape[0]
            if debug:
                hook_dict[i] = y.C
            x = self.pruning(x, mask)
        if debug:
            torch.save(hook_dict, 'hook_dict.pt', _use_new_zipfile_serialization=False)
        return multi_scale_rets

    @torch.no_grad()
    def predict_and_prune(self, fm, anchor_head):
        cls_pred, box_reg, objness_pred = anchor_head(fm)
        anchors = anchor_head.generate_anchors(fm).to(fm.device)
        box_reg._F = box_utils.decode_boxes(
            box_reg.F, anchors[:, 1:], anchor_head.bbox_size
        )
        pruning_mask = self.get_pruning_mask(objness_pred.F)

        return (cls_pred, box_reg, objness_pred), pruning_mask

    @torch.no_grad()
    def get_pruning_mask(self, pruning_pred, thres=0.2):
        # import time
        # s = time.time()
        pruning_pred = torch.sigmoid(pruning_pred)
        pruning_mask = torch.any(pruning_pred > thres, dim=-1)
        while pruning_mask.sum() >= 10000:
            thres += 0.05
            pruning_mask = torch.any(pruning_pred > thres, dim=-1)

        while pruning_mask.sum() <= 1.:
            rand = torch.randint(0, len(pruning_mask), (2,))
            pruning_mask[rand] = True
        # print("Prune ", time.time() - s)
        # print("Thres ", thres)
        return pruning_mask

    def get_loss_and_prune(self, fm, labels, anchor_head, cur_scale=None, writer=None):
        import time
        cls_pred, box_reg, objness_pred = anchor_head(fm)
        s = time.time()
        targets = anchor_head.assign_targets(fm, labels)
        loss = anchor_head.get_loss(
            objness_pred, cls_pred, box_reg, targets,
            cur_scale=cur_scale, writer=writer
        )
        pruning_mask = self.get_pruning_mask(objness_pred.F)
        # print("Get_loss ", time.time() - s)
        return loss, pruning_mask
