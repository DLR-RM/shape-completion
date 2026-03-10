from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from utils import setup_logger, unsqueeze_as

from ..mixins import MultiEvalMixin
from .latent import EDMScheduler, edm_sampler
from .model import DiffusionModel
from .unet import UNet

logger = setup_logger(__name__)
try:
    from eval import render_for_fid
except ImportError as e:
    logger.error(f"Could not import from 'eval' submodule: {e}")
    render_for_fid = None


class GridDiffusionModel(MultiEvalMixin, DiffusionModel):
    def __init__(
        self,
        ndim: int = 3,
        channels: int = 1,
        sigma_data: float = 0.5,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        resolution: int = 64,
        **kwargs,
    ):
        super().__init__()
        dim_mults = (
            (1, 2)
            if resolution == 16
            else (1, 2, 4)
            if resolution == 32
            else (1, 2, 4, 8)
            if resolution == 64
            else (1, 2, 2, 2)
        )
        self.scheduler = EDMScheduler(sigma_data=sigma_data)
        self.denoise_fn = UNet(
            dim=resolution, dim_mults=dim_mults, channels=channels, ndim=ndim, resolution=resolution, **kwargs
        )
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.resolution = resolution

    def _get_inputs(self, data: dict[str, Tensor]) -> Tensor:
        if self.denoise_fn.ndim == 2:
            return 2 * data["inputs"] - 1
        elif self.denoise_fn.ndim == 3:
            return 2 * data["points.occ"].float().unsqueeze(1) - 1
        else:
            raise ValueError(f"Unsupported ndim: {self.denoise_fn.ndim}")

    def forward(
        self,
        noisy_samples: Tensor | None = None,
        sigmas: Tensor | None = None,
        cond: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if noisy_samples is None:
            if self.denoise_fn.ndim == 2:
                noisy_samples = torch.zeros(
                    1, self.denoise_fn.channels, self.resolution, self.resolution, dtype=self.dtype, device=self.device
                )
            elif self.denoise_fn.ndim == 3:
                noisy_samples = torch.zeros(1, 1, self.resolution**3, dtype=self.dtype, device=self.device)
        if noisy_samples is None:
            raise ValueError(f"Unsupported ndim: {self.denoise_fn.ndim}")
        if sigmas is None:
            timesteps = unsqueeze_as(
                torch.randn((noisy_samples.size(0),), dtype=noisy_samples.dtype, device=noisy_samples.device),
                noisy_samples,
            )
            sigmas = self.scheduler.precondition_timesteps(timesteps)
        if sigmas is None:
            raise ValueError("sigmas must not be None after scheduler preconditioning")
        if not torch.all(sigmas >= 0):
            raise ValueError(f"sigmas must be non-negative, got {sigmas}")
        cond_samples = self.scheduler.precondition_inputs(noisy_samples, sigmas)
        cond_sigmas = self.scheduler.precondition_noise(sigmas)
        model_output = self.denoise_fn(cond_samples.float(), cond_sigmas.float().flatten(), **kwargs).float()
        return self.scheduler.precondition_outputs(noisy_samples, model_output, sigmas).to(noisy_samples.dtype)

    @torch.no_grad()
    def evaluate(self, data: dict[str, Tensor], prefix: str = "val/", **kwargs) -> dict[str, float]:
        preds = data.get("logits")
        loss = data.get("loss")
        if preds is None or loss is None:
            preds, loss = self.predict(data, return_loss=True, **kwargs)

        if self.denoise_fn.ndim == 2:
            return {f"{prefix}loss": loss.mean().item()}

        data.update({"logits": preds.squeeze(1), "loss": loss})
        return super().evaluate(cast(dict[str, Any], data), prefix=prefix, **kwargs)

    @torch.no_grad()
    def predict(self, data: dict[str, Tensor], return_loss: bool = False, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        inputs = self._get_inputs(data)
        scheduler_call = cast(Any, self.scheduler)
        preds, sigmas = scheduler_call(self.denoise_fn, inputs, **kwargs)
        if return_loss:
            loss = F.mse_loss(preds, inputs, reduction="none")
            loss = self.scheduler.precondition_loss(loss, sigmas)
            if self.denoise_fn.ndim == 2:
                return sigmas, loss
            return preds, loss
        return preds

    @torch.inference_mode()
    def generate(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        threshold: float | None = None,
        points_batch_size: int | None = None,
        show: bool = False,
        render: bool = False,
        num_steps: int = 35,
        progress: bool = True,
        **kwargs,
    ) -> Tensor:
        batch_size = inputs.size(0) if inputs is not None else points.size(0) if points is not None else 1
        if self.denoise_fn.ndim == 2:
            latents = torch.randn(
                batch_size,
                self.denoise_fn.channels,
                self.resolution,
                self.resolution,
                dtype=self.dtype,
                device=self.device,
            )
        elif self.denoise_fn.ndim == 3:
            num_points = points.size(1) if points is not None else self.resolution**3
            latents = torch.randn(batch_size, 1, num_points, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"Unsupported ndim: {self.denoise_fn.ndim}")
        out = cast(Tensor, edm_sampler(self, latents, num_steps=num_steps, progress=progress))
        if show and self.denoise_fn.ndim == 2:
            from matplotlib import pyplot as plt

            b, c, h, w = out.shape
            g = int(b**0.5)
            image = out[: g * g]
            image = (image * 127.5 + 128).clip(0, 255).to(torch.uint8)
            image = image.reshape(g, g, *image.shape[1:]).permute(0, 3, 1, 4, 2)
            image = image.reshape(g * h, g * w, c)
            image = image.cpu().numpy()
            plt.imshow(image)
            plt.axis("off")
            plt.show()
        return out

    def loss(self, data: dict[str, Tensor], **kwargs) -> Tensor:
        inputs = self._get_inputs(data)
        scheduler_call = cast(Any, self.scheduler)
        predictions, sigmas = scheduler_call(self.denoise_fn, inputs, **kwargs)
        loss = F.mse_loss(predictions, inputs, reduction="none")
        loss = self.scheduler.precondition_loss(loss, sigmas)
        return loss.mean()
