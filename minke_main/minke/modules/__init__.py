import MinkowskiEngine as ME

from torch import nn


class SparseBasicBlock(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D) -> None:
        """Initialize.

        Args:
            in_feat ([int]): Nb of input features
            out_feat ([int]): Nb of output features
            D ([int]): Nb of dimensions (2D, 3D, ...)
        """
        super(SparseBasicBlock, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_feat, out_feat, kernel_size=3, dimension=D
            ),
            ME.MinkowskiBatchNorm(out_feat),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                out_feat, out_feat, kernel_size=3, dimension=D
            ),
            ME.MinkowskiBatchNorm(out_feat)
        )
        if out_feat == in_feat:
            self.skip = None
        else:
            self.skip = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_feat, out_feat, kernel_size=3, dimension=D
                ),
                ME.MinkowskiBatchNorm(out_feat),
            )
        self.out_relu = ME.MinkowskiReLU()

    def forward(self, x):
        out = self.conv1(x)
        out += x if self.skip is None else self.skip(x)
        out = self.out_relu(out)

        return out
