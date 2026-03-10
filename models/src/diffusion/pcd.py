from typing import Any, cast

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor
from torch.nn import functional as F

from eval import eval_pointcloud

from .latent import EDMScheduler
from .model import DiffusionModel
from .pvd import PVD


class PVDModel(DiffusionModel):
    def __init__(self):
        super().__init__()
        self.scheduler = DDPMScheduler(clip_sample=False)
        self.denoise_fn = PVD()
        # self.denoise_fn = LatentArrayTransformer(in_channels=3, n_heads=6, depth=8)

    def forward(self, noisy_samples: Tensor | None = None, timesteps: Tensor | None = None, **kwargs) -> Tensor:
        if noisy_samples is None:
            noisy_samples = torch.randn(1, 2048, 3, dtype=self.dtype, device=self.device)
        if timesteps is None:
            scheduler_config = cast(Any, self.scheduler.config)
            timesteps = torch.randint(
                0,
                scheduler_config.num_train_timesteps,
                (noisy_samples.size(0),),
                device=noisy_samples.device,
            )
        return self.denoise_fn(noisy_samples, timesteps, **kwargs)

    def _train_val_step(
        self, inputs: Tensor, **kwargs
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor]:
        timesteps, noise, noisy_inputs = None, None, None
        if isinstance(self.scheduler, EDMScheduler):
            scheduler_call = cast(Any, self.scheduler)
            predictions, sigmas = scheduler_call(self.denoise_fn, inputs, **kwargs)
            loss = F.mse_loss(predictions, inputs, reduction="none")
            loss = self.scheduler.precondition_loss(loss, sigmas).mean()
        else:
            scheduler_config = cast(Any, self.scheduler.config)
            timesteps = torch.randint(
                0,
                scheduler_config.num_train_timesteps,
                (inputs.size(0),),
                device=inputs.device,
            )
            noise = torch.randn_like(inputs)
            noisy_inputs = self.scheduler.add_noise(inputs, noise, cast(torch.IntTensor, timesteps.to(torch.int64)))
            predictions = self(noisy_inputs, timesteps)
            loss = F.mse_loss(predictions, noise)
        return predictions, timesteps, noise, noisy_inputs, loss

    @torch.no_grad()
    def evaluate(self, data: dict[str, Tensor], prefix: str = "val/", **kwargs) -> dict[str, float]:
        inputs = data["inputs"]
        predictions, timesteps, noise, noisy_inputs, loss = self._train_val_step(inputs, **kwargs)
        if not isinstance(self.scheduler, EDMScheduler):
            step = self.scheduler.step
            assert timesteps is not None and noise is not None and noisy_inputs is not None
            predictions = torch.stack(
                [
                    cast(Any, step)(p, int(t.item()), i).prev_sample
                    for p, t, i in zip(predictions, timesteps, noisy_inputs, strict=False)
                ]
            )
            inputs = self.scheduler.add_noise(
                inputs, noise, cast(torch.IntTensor, (timesteps - 1).clamp(min=0).to(torch.int64))
            )
        results = {"loss": loss.item()}
        results.update(eval_pointcloud(predictions, inputs))
        results = {prefix + k: v for k, v in results.items()}
        print("chamfer-l1", results["val/chamfer-l1"])
        return results

    def predict(self, **kwargs) -> Tensor:
        raise NotImplementedError

    def loss(self, data: dict[str, Tensor], **kwargs) -> Tensor:
        return self._train_val_step(data["inputs"], **kwargs)[-1]
