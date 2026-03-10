from typing import Any, cast

import torch
import torch.nn.functional as F
from pytorch3dunet.unet3d.buildingblocks import SingleConv
from pytorch3dunet.unet3d.model import UNet3D
from torch import Tensor, nn

from utils import DEBUG_LEVEL_1, DEBUG_LEVEL_2, is_distributed, points_to_coordinates

from .mixins import MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model
from .pointnet import GridLocalPoolPointNet
from .resnet import ResNetGridDecoder
from .utils import grid_sample_3d

DEFAULT_KWARGS = {
    "encoder_opt": {
        "hidden_dim": 32,
        "feature_type": ("grid",),
        "grid_resolution": 64,
        "c_dim": 32,
        "norm": None,
        "unet3d": False,
        "downsampler": True,
        "downsampler_kwargs": {"in_channels": 32, "downsample_steps": 2},
    },
    "quantizer_opt": {"vocab_size": 4096, "n_embd": 128},
    "vq_beta": 0.001,
    "decoder_opt": {
        "sample_mode": "bilinear",
        "hidden_dim": 32,
        "c_dim": 32,
        "norm": None,
        "unet3d": True,
        "unet3d_kwargs": {
            "num_levels": 3,
            "f_maps": 128,
            "in_channels": 128,
            "out_channels": 128,
            "is_segmentation": False,
            "layer_order": "crg",
            "mode": "nearest",
        },
        "upsampler": True,
        "upsampler_kwargs": {"in_channels": 128, "upsample_steps": 2},
    },
}


class Downsampler(nn.Module):
    def __init__(self, in_channels, downsample_steps=1):
        super().__init__()
        channels = [in_channels * (2**k) for k in range(0, downsample_steps + 1)]
        blocks: list[nn.Module] = []
        for i in range(downsample_steps):
            in_c, out_c = channels[i], channels[i + 1]
            blocks.append(
                SingleConv(in_channels=in_c, out_channels=out_c, kernel_size=2, order="crg", stride=2, padding=0)
            )
            blocks.append(
                SingleConv(in_channels=out_c, out_channels=out_c, kernel_size=1, order="crg", stride=1, padding=0)
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Upsampler(nn.Module):
    def __init__(self, in_channels, upsample_steps=1, mode="nearest"):
        super().__init__()
        channels = [int(in_channels / (2**k)) for k in range(0, upsample_steps + 1)]
        blocks: list[nn.Module] = []
        for i in range(upsample_steps):
            in_c, out_c = channels[i], channels[i + 1]
            blocks.append(nn.Upsample(scale_factor=2, mode=mode))  # Todo: check if align_corners=True is needed
            blocks.append(
                SingleConv(in_channels=in_c, out_channels=out_c, kernel_size=3, order="crg", stride=1, padding=1)
            )
            blocks.append(
                SingleConv(in_channels=out_c, out_channels=out_c, kernel_size=3, order="crg", stride=1, padding=1)
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Quantizer(nn.Module):
    def __init__(self, vocab_size, n_embd, gamma=0.99):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.embedding.requires_grad_(False)

        self.n_embd = int(n_embd)
        self.vocab_size = int(vocab_size)
        self.gamma = float(gamma)

        # Register buffers
        self.register_buffer("N", torch.zeros(vocab_size))
        self.register_buffer("z_avg", self.embedding.weight.data.clone())

    def codebook_nonzero_count(self):
        count_buffer = cast(Tensor, self.N)
        avg_buffer = cast(Tensor, self.z_avg)
        print(len(count_buffer.nonzero(as_tuple=False)))
        print(len((avg_buffer.abs().sum(dim=-1) > 7.1186e-10).nonzero(as_tuple=False)))
        w = self.embedding.weight
        print(w.shape)
        print(len((w.abs().sum(dim=-1) > 10e-1).nonzero(as_tuple=False)))

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = x.float()
        if x.ndim != 5:
            flat_inputs = x.view(-1, self.n_embd)
        else:
            b, _c, x1, x2, x3 = x.shape
            flat_inputs = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_embd)

        weight = self.embedding.weight
        inputs_l2_norm = flat_inputs.pow(2).sum(dim=1, keepdim=True)
        codebook_l2_norm = weight.pow(2).sum(dim=1, keepdim=True).t()
        distances = inputs_l2_norm - 2 * torch.mm(flat_inputs, weight.t()) + codebook_l2_norm

        encoding_indices = torch.max(-distances, dim=1)[1]
        encode_onehot = F.one_hot(encoding_indices, self.vocab_size).type(flat_inputs.dtype)

        if x.ndim != 5:
            encoding_indices = encoding_indices.view(x.shape[:2])
            quant_feat = self.embedding(encoding_indices)
        else:
            encoding_indices = encoding_indices.view(b, x1, x2, x3)
            quant_feat = self.embedding(encoding_indices).permute(0, 4, 1, 2, 3).contiguous()

        if self.training:
            with torch.no_grad():
                # Synchronize the `encode_onehot` sum across processes
                encode_onehot_sum = encode_onehot.sum(0)
                if is_distributed():
                    torch.distributed.all_reduce(encode_onehot_sum)

                n_buffer = cast(Tensor, self.N)
                n_buffer.mul_(self.gamma).add_(encode_onehot_sum, alpha=1 - self.gamma)

                encode_sum = torch.mm(flat_inputs.t(), encode_onehot)
                if is_distributed():
                    torch.distributed.all_reduce(encode_sum)

                z_avg_buffer = cast(Tensor, self.z_avg)
                z_avg_buffer.mul_(self.gamma).add_(encode_sum.t(), alpha=1 - self.gamma)

                n = n_buffer.sum()
                weights = (n_buffer + 1e-7) / (n + self.vocab_size * 1e-7) * n

                encode_normalized = z_avg_buffer / weights.unsqueeze(1)
                self.embedding.weight.data.copy_(encode_normalized)

        quant_feat_st = (quant_feat - x).detach() + x
        quant_diff = (x - quant_feat.detach()).pow(2).mean()

        return quant_feat_st, encoding_indices, quant_diff

    def sync_buffers(self):
        # Synchronize the buffers manually when necessary
        torch.distributed.broadcast(cast(Tensor, self.N), 0)
        torch.distributed.broadcast(cast(Tensor, self.z_avg), 0)


class VQDIF(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    def __init__(
        self,
        encoder_opt: dict,
        decoder_opt: dict,
        quantizer_opt: dict | None = None,
        vq_beta: float = 0.001,
        cls_num_classes: int | None = None,
        seg_num_classes: int | None = None,
        custom_grid_sampling: bool = False,
    ):
        super().__init__()

        self.cls_head = None
        if cls_num_classes is not None:
            self.cls_head = nn.Linear(decoder_opt["c_dim"], cls_num_classes)

        self.seg_head = None
        if seg_num_classes is not None:
            self.seg_head = nn.Linear(decoder_opt["c_dim"], seg_num_classes + 1)

        self.encoder = GridLocalPoolPointNet(**encoder_opt)
        self.downsampler = None
        if encoder_opt.get("downsampler", False):
            self.downsampler = Downsampler(**encoder_opt["downsampler_kwargs"])
        self.decoder = ResNetGridDecoder(
            grid_sample=grid_sample_3d if custom_grid_sampling else F.grid_sample, **decoder_opt
        )
        self.upsampler = None
        if decoder_opt.get("upsampler", False):
            self.upsampler = Upsampler(**decoder_opt["upsampler_kwargs"])
        self.unet3d = None
        if decoder_opt.get("unet3d", False):
            self.unet3d = UNet3D(**decoder_opt["unet3d_kwargs"])
        self.quantizer = None
        if quantizer_opt is not None:
            self.quantizer = Quantizer(**quantizer_opt)

        self.vq_beta = vq_beta

    def encode(
        self, inputs: Tensor, return_quant: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor, Tensor] | tuple[Tensor, None, None]:
        enc_dict = self.encoder(inputs, return_point_feature=False)
        grid_feat = enc_dict["grid"]

        if self.downsampler is not None:
            # produce (B, k*C, res/k, res/k, res/k), k=2**downsample_steps
            grid_feat = self.downsampler(grid_feat)

        if self.quantizer is not None:
            grid_feat, quant_ind, quant_diff = self.quantizer(grid_feat)
            if return_quant:
                return grid_feat, quant_ind, quant_diff
            return grid_feat

        if return_quant:
            return grid_feat, None, None
        return grid_feat

    @staticmethod
    def get_mask(inputs: Tensor, resolution: int, padding: float) -> Tensor:
        """
        Creates a binary mask tensor based on a set of input points.

        Args:
            inputs (Tensor): A tensor of shape (B, N, 3) containing B batches of N points with 3 coordinates each.
            resolution (int): An integer representing the resolution of the output mask tensor in each dimension.
            padding (float): A float value representing the padding to be applied to the input tensor to normalize the
            coordinates.

        Returns:
            Tensor: A binary mask tensor of shape (B, R, R, R) with indices corresponding to
            the coordinates of the input points set to True.
        """

        # Convert points to normalized coordinates
        coords = cast(Tensor, points_to_coordinates(inputs, max_value=1 + padding))

        # Compute indices of corresponding voxels in the mask tensor
        mask_indices = torch.clamp(coords * resolution, 0, resolution - 1).long()

        # Flatten the indices and repeat batch index for indexing into mask tensor
        flat_indices = mask_indices.view(-1, mask_indices.size(-1))
        binds = torch.repeat_interleave(
            torch.arange(mask_indices.size(0), dtype=mask_indices.dtype), mask_indices.size(1)
        )

        # Create a binary mask tensor and set appropriate indices to True
        mask = torch.zeros(inputs.size(0), resolution, resolution, resolution, dtype=torch.bool)
        mask[binds, flat_indices[:, 2], flat_indices[:, 1], flat_indices[:, 0]] = True  # row-major to column-major?
        return mask

    @torch.no_grad()
    def quantize_cloud(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode input points into a quantized grid feature representation and return quantization indices and a mask.

        Parameters
        ----------
        inputs : Tensor
            The (potentially partial) input pointcloud. Shape (B, N, 3).

        Returns
        -------
        Tuple[Tensor, Tensor, Dict[str, Tensor]]
            A tuple containing:
                - `masked_quant_ind`: (B, R, R, R): A 3D grid where each cell contains the index into the codebook.
                                      Empty input cells are masked with the most common index.
                - `mode` (1,): The most common codebook index.
                - `grid_feat` (B, C, R, R, R): A dictionary containing the grid feature.

        """

        assert not self.training

        # encode input points into a codebook index grid
        quant_ind = self.encode(inputs, return_quant=True)[1]
        if quant_ind is None:
            raise RuntimeError("Quantization indices are unavailable; quantizer is required for `quantize_cloud`.")
        mask = self.get_mask(inputs, resolution=quant_ind.size(-1), padding=self.encoder.padding)

        # compute the mode of the predicted codebook indices, representing empty cells
        mode = quant_ind.view(-1).mode()[0]

        # create a new tensor to represent masked codebook indices, with empty cells set to the mode
        masked_quant_ind = torch.zeros_like(quant_ind) + mode
        masked_quant_ind[mask] = quant_ind[mask]

        return masked_quant_ind, mode

    def decode(self, points: Tensor, feature: Tensor, **kwargs) -> dict[str, Tensor]:
        if self.unet3d is not None:
            feature = self.unet3d(feature)
        if self.upsampler is not None:
            feature = self.upsampler(feature)
        return_point_feature = self.cls_head is not None or self.seg_head is not None
        out = self.decoder(points, {"grid": feature}, return_point_feature=return_point_feature)

        result = dict()
        if isinstance(out, tuple):
            logits, point_feat = out
            if self.cls_head is not None:
                result["cls_logits"] = self.cls_head(point_feat.mean(dim=2))
            if self.seg_head is not None:
                result["seg_logits"] = self.seg_head(point_feat.transpose(1, 2))
            else:
                result["logits"] = logits
        else:
            result["logits"] = out

        return result

    def forward(self, inputs: Tensor, points: Tensor, **kwargs) -> dict[str, Tensor]:
        grid_feat, quant_ind, quant_diff = self.encode(inputs, return_quant=True, **kwargs)
        if quant_ind is None:
            raise RuntimeError("`quant_ind` is None; quantizer is required for `forward`.")
        if quant_diff is None:
            quant_diff = grid_feat.new_zeros(())
        result = dict(grid_feat=grid_feat, quant_ind=quant_ind, quant_diff=quant_diff)
        result.update(self.decode(points, grid_feat))
        return result

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        if not any("logits" in key for key in data.keys()):
            data.update(self(**data, **kwargs))

        reduction = self.reduction if reduction is None else reduction
        data.update(super().loss(data, regression, name, reduction))

        def _get_tensor_or_zero(value: Any) -> Tensor:
            if isinstance(value, Tensor):
                return value
            inputs_value = data.get("inputs")
            device = inputs_value.device if isinstance(inputs_value, Tensor) else None
            return torch.zeros((), device=device)

        def _get_float(value: Any, default: float = 1.0) -> float:
            if isinstance(value, Tensor):
                return float(value.item())
            if isinstance(value, (int, float)):
                return float(value)
            return default

        rec_loss = _get_tensor_or_zero(data.get("occ_loss"))
        vq_loss = _get_tensor_or_zero(data.get("quant_diff"))
        cls_loss = _get_tensor_or_zero(data.get("cls_loss"))
        seg_loss = _get_tensor_or_zero(data.get("seg_loss"))

        if rec_loss.item() > 0:
            self.log("rec_loss", rec_loss.item())
        if vq_loss.item() > 0:
            self.log("vq_loss", vq_loss.item())
        if cls_loss.item() > 0:
            self.log("cls_loss", cls_loss.item())
        if seg_loss.item() > 0:
            self.log("seg_loss", seg_loss.item())

        indices = data.get("quant_ind")
        if indices is not None:
            if not isinstance(indices, Tensor):
                raise TypeError("`quant_ind` must be a tensor.")
            if self.quantizer is None:
                raise RuntimeError("`quantizer` is required when `quant_ind` is present.")
            with torch.no_grad():
                if indices.ndim == 3:
                    b, _n, c = indices.size()
                    offset = torch.arange(c, device=indices.device).view(1, 1, c) * (self.quantizer.vocab_size // c)
                    indices = (indices + offset).view(b, -1)

                indices_count = torch.bincount(indices.view(-1), minlength=self.quantizer.vocab_size)
                if is_distributed():
                    torch.distributed.all_reduce(indices_count)
                avg_probs = indices_count.float() / indices_count.sum()
                active_codes = (indices_count > 0).sum().item()
                perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp().item()

            self.log("codebook_usage", active_codes / self.quantizer.vocab_size * 100)
            self.log("perplexity", perplexity, level=DEBUG_LEVEL_1)
            self.log("active_codes", active_codes, level=DEBUG_LEVEL_2)

        cls_weight = _get_float(data.get("cls_weight", 1.0))
        seg_weight = _get_float(data.get("seg_weight", 1.0))

        loss = rec_loss
        loss += cls_weight * cls_loss
        loss += seg_weight * seg_loss
        loss += self.vq_beta * vq_loss

        return loss
