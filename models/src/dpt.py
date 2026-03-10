# Adapted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/depth/decode_heads.py
import warnings
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def resize(
    input: Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | tuple[float, float] | None = None,
    mode: Literal["nearest-exact", "bilinear", "bicubic", "area"] = "nearest-exact",
    align_corners: bool | None = None,
    warning: bool = False,
) -> Tensor:
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`",
                        stacklevel=2,
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def _zero_pad2d_arg(padding: int | tuple[int, int]) -> int | tuple[int, int, int, int]:
    if isinstance(padding, tuple):
        # Conv2d uses (h, w); ZeroPad2d uses (left, right, top, bottom).
        return (padding[1], padding[1], padding[0], padding[0])
    return padding


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_layer. Bias will be set as True if `norm_layer` is None, otherwise
            False. Default: "auto".
        conv_layer (nn.Module): Convolution layer. Default: None,
            which means using conv2d.
        norm_layer (nn.Module): Normalization layer. Default: None.
        act_layer (nn.Module): Activation layer. Default: nn.ReLU.
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_: str = "conv_block"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool | str = "auto",
        conv_layer: type[nn.Module] = nn.Conv2d,
        norm_layer: type[nn.Module] | None = None,
        act_layer: type[nn.Module] | None = nn.ReLU,
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
        order: tuple[str, str, str] = ("conv", "norm", "act"),
    ):
        super().__init__()
        self.norm_name: str | None = None
        official_padding_mode = ["zeros", "circular"]
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(["conv", "norm", "act"])

        self.with_norm = norm_layer is not None
        self.with_activation = act_layer is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            if padding_mode == "zeros":
                padding_layer = nn.ZeroPad2d
            else:
                raise AssertionError(f"Unsupported padding mode: {padding_mode}")
            self.pad = padding_layer(_zero_pad2d_arg(padding))

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = self.conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            if norm_layer is None:
                raise ValueError("norm_layer must be provided when with_norm is True")
            self.norm_name = "norm"
            self.add_module(self.norm_name, norm_layer(num_features=norm_channels))
            if self.with_bias:
                if issubclass(norm_layer, (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm)):
                    warnings.warn("Unnecessary conv bias before batch/instance norm", stacklevel=2)

        # build activation layer
        if self.with_activation:
            # nn.Tanh has no 'inplace' argument
            # (nn.Tanh, nn.PReLU, nn.Sigmoid, nn.HSigmoid, nn.Swish, nn.GELU)
            if act_layer is None:
                raise ValueError("act_layer must be provided when with_activation is True")
            if not issubclass(act_layer, (nn.Tanh, nn.PReLU, nn.Sigmoid, nn.GELU)):
                self.activate = act_layer(inplace=inplace)
            else:
                self.activate = act_layer()

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self) -> nn.Module | None:
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self) -> None:
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and isinstance(self.act_layer, nn.LeakyReLU):
                nonlinearity = "leaky_relu"
                a = 0.01  # XXX: default negative_slope
            else:
                nonlinearity = "relu"
                a = 0
            conv_weight = getattr(self.conv, "weight", None)
            if isinstance(conv_weight, Tensor):
                nn.init.kaiming_normal_(conv_weight, a=a, mode="fan_out", nonlinearity=nonlinearity)
            conv_bias = getattr(self.conv, "bias", None)
            if isinstance(conv_bias, Tensor):
                nn.init.constant_(conv_bias, 0)
        if self.with_norm:
            norm_module = self.norm
            if norm_module is not None:
                norm_weight = getattr(norm_module, "weight", None)
                if isinstance(norm_weight, Tensor):
                    nn.init.constant_(norm_weight, 1)
                norm_bias = getattr(norm_module, "bias", None)
                if isinstance(norm_bias, Tensor):
                    nn.init.constant_(norm_bias, 0)

    def forward(self, x: Tensor, activate: bool = True, norm: bool = True) -> Tensor:
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.pad(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                norm_module = self.norm
                if norm_module is None:
                    raise RuntimeError("Expected normalization module when with_norm is True")
                x = norm_module(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class ReassembleBlocks(nn.Module):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.
    Args:
        in_channels (int): ViT feature channels. Default: 768.
        out_channels (List): output channels of each stage.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'project'.
    """

    def __init__(
        self,
        in_channels: int = 768,
        out_channels: tuple[int, ...] = (96, 192, 384, 768),
        readout_type: Literal["add", "project"] | None = "project",
        resize_type: Literal["upsample", "deconv"] = "deconv",
    ):
        super().__init__()

        self.readout_type = readout_type

        self.projects = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    act_layer=None,
                )
                for out_channel in out_channels
            ]
        )

        if resize_type == "upsample":
            self.resize_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
                        ConvModule(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=3, padding=1),
                    ),
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        ConvModule(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=3, padding=1),
                    ),
                    nn.Identity(),
                    nn.Conv2d(
                        in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                    ),
                ]
            )
        elif resize_type == "deconv":
            self.resize_layers = nn.ModuleList(
                [
                    nn.ConvTranspose2d(
                        in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                    ),
                    nn.ConvTranspose2d(
                        in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                    ),
                    nn.Identity(),
                    nn.Conv2d(
                        in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                    ),
                ]
            )
        else:
            raise ValueError(f"Unsupported resize type: {resize_type}")
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))

    def forward(self, inputs: list[tuple[Tensor, Tensor | None]]) -> list[Tensor]:
        assert isinstance(inputs, list)
        out = []
        for i, x in enumerate(inputs):
            assert len(x) == 2
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if cls_token is not None:
                if self.readout_type == "project":
                    x = x.flatten(2).permute((0, 2, 1))
                    readout = cls_token.unsqueeze(1).expand_as(x)
                    x = self.readout_projects[i](torch.cat((x, readout), -1))
                    x = x.permute(0, 2, 1).reshape(feature_shape)
                elif self.readout_type == "add":
                    x = x.flatten(2) + cls_token.unsqueeze(-1)
                    x = x.reshape(feature_shape)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class PreActResidualConvUnit(nn.Module):
    """ResidualConvUnit, pre-activate residual unit.
    Args:
        in_channels (int): number of channels in the input feature map.
        act_layer (nn.Module): activation layer.
        norm_layer (nn.Module): norm layer.
        stride (int): stride of the first block. Default: 1
        dilation (int): dilation rate for convs layers. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        act_layer: type[nn.Module],
        norm_layer: type[nn.Module] | None,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()

        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            act_layer=act_layer,
            bias=False,
            order=("act", "conv", "norm"),
        )

        self.conv2 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            bias=False,
            order=("act", "conv", "norm"),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        inputs_ = inputs.clone()
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs_


class BottleneckResidualConvUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        act_layer: type[nn.Module] = nn.ReLU,
        norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        bottleneck_channels = in_channels // reduction
        self.conv1 = ConvModule(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            order=("act", "conv", "norm"),
        )
        self.conv2 = ConvModule(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            order=("act", "conv", "norm"),
        )
        self.conv3 = ConvModule(
            bottleneck_channels,
            in_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            order=("act", "conv", "norm"),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + inputs


class FeatureFusionBlock(nn.Module):
    """FeatureFusionBlock, merge feature map from different stages.
    Args:
        in_channels (int): Input channels.
        act_layer (nn.Module): activation layer for ResidualConvUnit.
        norm_layer (nn.Module): normalization layer.
        expand (bool): Whether expand the channels in post process block.
            Default: False.
        align_corners (bool): align_corner setting for bilinear upsample.
            Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        act_layer: type[nn.Module],
        norm_layer: type[nn.Module] | None,
        conv_type: Literal["conv", "bottleneck"] = "conv",
        expand: bool = False,
        align_corners: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners

        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2

        self.project = ConvModule(self.in_channels, self.out_channels, kernel_size=1, act_layer=None, bias=True)
        if conv_type == "bottleneck":
            self.res_conv_unit1 = BottleneckResidualConvUnit(
                in_channels=self.in_channels, act_layer=act_layer, norm_layer=norm_layer
            )
            self.res_conv_unit2 = BottleneckResidualConvUnit(
                in_channels=self.out_channels, act_layer=act_layer, norm_layer=norm_layer
            )
        elif conv_type == "conv":
            self.res_conv_unit1 = PreActResidualConvUnit(
                in_channels=self.in_channels, act_layer=act_layer, norm_layer=norm_layer
            )
            self.res_conv_unit2 = PreActResidualConvUnit(
                in_channels=self.out_channels, act_layer=act_layer, norm_layer=norm_layer
            )
        else:
            raise ValueError(f"Unsupported conv type: {conv_type}")

    def forward(self, *inputs: Tensor) -> Tensor:
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = resize(
                    inputs[1], size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
                )  # TODO: why False?
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = resize(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        x = self.project(x)
        return x


class DPTHead(nn.Module):
    """Vision Transformers for Dense Prediction (DPT) Head.

    This module implements the decoder head from the DPT paper, which is designed
    to be attached to a Vision Transformer (ViT) backbone for dense prediction
    tasks like depth estimation or semantic segmentation. It takes multi-scale
    features from the ViT backbone and progressively fuses them to produce a
    dense, high-resolution output.

    The head consists of three main parts:
    1.  Reassemble Blocks: To process and reshape the patch-based features from
        the ViT into standard convolutional feature maps.
    2.  Fusion Blocks: A series of blocks that progressively upsample and merge
        feature maps from different scales, starting from the deepest layer.
    3.  Projection Layer: A final convolutional layer to produce the output map.

    Args:
        channels (int): The number of channels for the intermediate features
            within the fusion blocks. Defaults to 256.
        embed_dims (int): The embedding dimension of the features coming from
            the ViT backbone. Defaults to 768.
        post_process_channels (Tuple[int, ...]): A tuple containing the number
            of output channels for each of the reassemble stages, corresponding
            to the different feature levels from the backbone.
            Defaults to (96, 192, 384, 768).
        readout_type (Optional[Literal["add", "project"]]): The method for
            incorporating the class token information into the patch tokens.
            'project' concatenates the class token with each patch token and
            projects it, while 'add' simply adds it. Defaults to "project".
        expand_channels (bool): If True, the number of channels is
            progressively increased in the post-processing layers.
            Defaults to False.
        **kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        channels: int = 256,
        embed_dims: int = 768,
        post_process_channels: tuple[int, ...] = (96, 192, 384, 768),
        readout_type: Literal["add", "project"] | None = "project",
        resize_type: Literal["upsample", "deconv"] = "deconv",
        conv_type: Literal["conv", "bottleneck"] = "conv",
        expand_channels: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.channels = channels
        self.expand_channels = expand_channels
        self.reassemble_blocks = ReassembleBlocks(
            in_channels=embed_dims,
            out_channels=post_process_channels,
            readout_type=readout_type,
            resize_type=resize_type,
        )

        self.post_process_channels = [
            channel * 2**i if expand_channels else channel for i, channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(ConvModule(channel, channels, kernel_size=3, padding=1, act_layer=None, bias=False))
        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(
                FeatureFusionBlock(
                    in_channels=channels,
                    act_layer=nn.ReLU,
                    norm_layer=None,
                    conv_type=conv_type,
                    expand=expand_channels,
                )
            )
        self.fusion_blocks[0].res_conv_unit1 = nn.Identity()
        self.project = ConvModule(channels, channels, kernel_size=3, padding=1, norm_layer=None)
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels

    def forward(self, inputs: list[tuple[Tensor, Tensor]]) -> Tensor:
        assert len(inputs) == self.num_reassemble_blocks
        x = [inp for inp in inputs]
        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        return self.project(out)
