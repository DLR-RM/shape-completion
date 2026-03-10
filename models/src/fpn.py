from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn


def _expand(tensor: Tensor, length: int) -> Tensor:
    return tensor.unsqueeze(1).expand(-1, int(length), *([-1] * (tensor.ndim - 1))).flatten(0, 1)


class FPN(nn.Module):
    """
    Adapted from MaskHeadSmallConv. Excludes bbox_masks. Uses 1x1 conv as final layer.
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim: int, fpn_dims: Sequence[int], context_dim: int, num_groups: int = 8):
        """
        Args:
            dim (int): number of channels of the bottleneck
            fpn_dims (list of ints): number of channels of skip connections
            context_dim (int): context dimensionality
        """
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, dim)
        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, inter_dims[1])
        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = nn.GroupNorm(num_groups, inter_dims[2])
        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = nn.GroupNorm(num_groups, inter_dims[3])
        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(num_groups // 2, inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 1, 1, padding=0)

        self.dim = dim

        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, fpns: Sequence[Tensor]) -> Tensor:
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class SimpleConvDecoder(nn.Module):
    """
    A simple convolutional decoder to predict instance masks from encoder features.

    This decoder takes a feature map from a backbone (like DINOv2) and
    progressively upsamples it to the final output resolution, predicting a
    fixed number of instance masks.
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_masks: int = 100,
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
        num_groups: int = 8,
    ):
        """
        Initializes the SimpleConvDecoder.

        Args:
            input_dim (int): The number of channels in the input feature map
                             from the encoder (e.g., 768 for DINOv2-Base).
            num_masks (int): The fixed number of instance masks to predict.
                             This corresponds to the 'K' value we discussed.
            decoder_channels (List[int]): A list of channel dimensions for
                                          each upsampling stage of the decoder.
        """
        super().__init__()
        self.num_masks = num_masks

        if input_dim != decoder_channels[0]:
            self.proj = nn.Sequential(
                nn.Conv2d(input_dim, decoder_channels[0], kernel_size=1),
                nn.GroupNorm(num_groups, decoder_channels[0]),
                nn.ReLU(),
            )
        else:
            self.proj = nn.Identity()

        self.decoder_layers = nn.ModuleList()
        in_channels = decoder_channels[0]

        for out_channels in decoder_channels[1:]:
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    nn.GroupNorm(num_groups, out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups, out_channels),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels
        self.head = nn.Conv2d(decoder_channels[-1], num_masks, kernel_size=1)

    def forward(self, x: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        x = self.proj(x)
        for layer in self.decoder_layers:
            x = layer(x)
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear")
        return self.head(x)
