from collections.abc import Callable
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from einops import reduce
from torch import Tensor, nn
from vector_quantize_pytorch import ResidualVQ

from utils import setup_logger, unsqueeze_as

from ..mixins import MultiEvalMixin
from ..transformer import Attention
from ..vae import VAEModel, VQVAEModel
from .edm import EDMScheduler, edm_sampler
from .model import DiffusionModel, Model, bit2int, int2bit
from .transformer import EDMTransformer

logger = setup_logger(__name__)

try:
    from pytorch3d.loss import chamfer_distance
except ImportError:
    logger.warning("The 'PyTorch3D' module is not installed. Chamfer distance loss will not be available.")
    chamfer_distance = None


class LatentDiffusionModel(MultiEvalMixin, DiffusionModel):
    def __init__(
        self,
        vae: VAEModel | VQVAEModel,
        denoise_fn: EDMTransformer,
        conditioner: Model | Callable | None = None,
        scheduler: EDMScheduler | None = None,
        vae_freeze: bool = True,
        conditioner_freeze: bool = True,
        condition_key: str | None = None,
        bit_diffusion: bool = False,
        loss_type: Literal["ce", "bce", "cd", "chamfer", "mse", "l2", "l1", "smooth_l1"] = "mse",
        pre_loss_fn: Callable | Attention | Literal["sigmoid", "tanh", "clamp", "attn"] | None = None,
        use_stats: bool = True,
        requantize: bool = False,
        pos_enc: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(vae, (VAEModel, VQVAEModel)), f"Unsupported VAE: {type(vae)}."
        assert isinstance(denoise_fn, EDMTransformer), f"Unsupported denoiser: {type(denoise_fn)}."
        if not isinstance(vae, VQVAEModel):
            assert not bit_diffusion, "Bit diffusion requires VQ-VAE model."
            assert loss_type not in ["ce", "bce"], "CE loss is only supported for bit diffusion."
        if loss_type in ["ce", "bce"] and not bit_diffusion:
            raise AssertionError("CE loss requires bit diffusion.")

        assert isinstance(pre_loss_fn, (type(None), str, Callable, Attention)), (
            f"Unsupported pre_loss_fn type: {type(pre_loss_fn)}."
        )
        if isinstance(pre_loss_fn, str):
            if pre_loss_fn == "attn":
                pre_loss_fn = Attention()
            elif pre_loss_fn in ["sigmoid", "tanh", "clamp"]:
                pre_loss_fn = getattr(torch, pre_loss_fn)
            else:
                raise ValueError(f"Unsupported pre_loss_fn: {pre_loss_fn}.")

        assert not (loss_type in ["ce", "bce"] and (isinstance(pre_loss_fn, Attention))), (
            "(B)CE loss is not supported with attention."
        )
        assert not (loss_type == "ce" and pre_loss_fn in [F.sigmoid, torch.sigmoid]), (
            "CE loss is not supported with sigmoid activation."
        )

        self.vae_freeze = vae_freeze
        self._vae = vae
        self.requantize = None
        if vae_freeze:
            self._vae.requires_grad_(False)
            self._vae.eval()
            if requantize:
                if bit_diffusion:
                    raise NotImplementedError("Re-quantization is not supported for bit diffusion.")
                if not isinstance(vae, VQVAEModel):
                    raise TypeError("Re-quantization requires VQ-VAE model.")
                vae.vq.train()
                self.requantize = vae.requantize if hasattr(vae, "requantize") else vae.quantize

        self.conditioner_freeze = conditioner_freeze
        self._conditioner = conditioner
        if conditioner_freeze and isinstance(conditioner, nn.Module):
            conditioner.requires_grad_(False)
            conditioner.eval()
        self._denoise_fn = denoise_fn

        self.scheduler = cast(EDMScheduler, scheduler or EDMScheduler(sigma_data=denoise_fn.sigma_data))
        self.sigma_data = float(cast(Any, self.scheduler.config).sigma_data)

        self.bit_diffusion = bit_diffusion
        self.loss_type = loss_type
        self.n_queries = int(getattr(vae, "n_queries", getattr(vae, "n_fps", 1)))
        self.n_latent = int(cast(Any, getattr(denoise_fn, "n_latent", vae.n_latent) or vae.n_latent))
        self.pre_loss_fn = pre_loss_fn
        self.pos_enc = nn.Embedding(self.n_queries, self.n_latent) if pos_enc else None
        self.condition_key = condition_key
        self.use_stats = use_stats

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        vae = cast(VAEModel | VQVAEModel, self._vae)
        vae.setup(*args, **kwargs)
        conditioner = self._conditioner
        if conditioner is not None and hasattr(conditioner, "setup"):
            cast(Any, conditioner).setup(*args, **kwargs)
        if hasattr(self._denoise_fn, "setup"):
            cast(Any, self._denoise_fn).setup(*args, **kwargs)

        has_stats = vae.stats and "mean" in vae.stats and "std" in vae.stats
        if self.use_stats and has_stats and not self.bit_diffusion:
            stats = cast(dict[str, Tensor], vae.stats)
            self.stats = {"mean": stats["mean"], "std": stats["std"]}
            logger.debug_level_2(
                f"Using stats: mean={self.stats['mean'].mean().item():.3f}, std={self.stats['std'].mean().item():.3f}"
            )

    def state_dict(self, *args, **kwargs):
        exclude = ["_vae" if self.vae_freeze else "", "_conditioner" if self.conditioner_freeze else ""]
        return {
            k: v for k, v in super().state_dict(*args, **kwargs).items() if not any(ex in k for ex in exclude if ex)
        }

    def encode(self, inputs: Tensor, **kwargs) -> Tensor:
        vae = cast(VAEModel | VQVAEModel, self._vae)
        with torch.enable_grad() if cast(nn.Module, vae).training else torch.no_grad():
            if isinstance(vae, VAEModel):
                latents = vae.sample_posterior(inputs)
            elif isinstance(vae, VQVAEModel):
                latents, indices, _ = vae.quantize(inputs, **kwargs)
                if self.bit_diffusion:
                    if isinstance(vae.vq, ResidualVQ):
                        """
                        n_quantizer = self._vae.vq.num_quantizers
                        n_code = self._vae.n_code // n_quantizer
                        scale = torch.arange(0, n_quantizer, device=indices.device).view(1, 1, -1) * n_code
                        indices = (indices + scale).view(indices.size(0), -1)
                        """
                        n_quantizers = int(vae.vq.num_quantizers or 1)
                        n_bit = int(self.n_latent) // n_quantizers
                        latents = (
                            2 * torch.cat([int2bit(indices[..., i], n_bit) for i in range(indices.size(-1))], dim=-1)
                            - 1
                        )
                    else:
                        if self.loss_type == "ce":
                            latents = 2 * F.one_hot(indices, int(self.n_latent)).float() - 1
                        else:
                            latents = 2 * int2bit(indices, int(self.n_latent)) - 1  # Analog Bits
            else:
                latents = cast(Tensor, vae.encode(inputs, **kwargs))

            # Standardize to zero mean and sigma_data std
            if self.bit_diffusion:
                latents = cast(Tensor, latents) * self.sigma_data
            elif self.stats is not None:
                stats_mean = cast(Tensor, self.stats["mean"])
                stats_std = cast(Tensor, self.stats["std"])
                latents = (cast(Tensor, latents) - stats_mean.to(cast(Tensor, latents))) * (
                    self.sigma_data / stats_std.to(cast(Tensor, latents))
                )

        return cast(Tensor, latents).float()

    def decode(self, latents: Tensor, **kwargs) -> Tensor:
        vae = cast(VAEModel | VQVAEModel, self._vae)
        with torch.enable_grad() if cast(nn.Module, vae).training else torch.no_grad():
            # Reverse standardization
            if self.bit_diffusion:
                latents = latents / self.sigma_data
            elif self.stats is not None:
                stats_mean = cast(Tensor, self.stats["mean"])
                stats_std = cast(Tensor, self.stats["std"])
                latents = latents / (self.sigma_data / stats_std.to(latents)) + stats_mean.to(latents)

            if self.bit_diffusion and isinstance(vae, VQVAEModel):
                if isinstance(vae.vq, ResidualVQ):
                    n_quantizers = int(vae.vq.num_quantizers or 1)
                    n_bit = int(self.n_latent) // n_quantizers
                    indices = torch.stack(
                        [bit2int(latents[..., i : i + n_bit] > 0) for i in range(0, latents.size(-1), n_bit)], dim=-1
                    )
                    """
                    n_quantizer = self._vae.vq.num_quantizers
                    n_code = self._vae.n_code // n_quantizer
                    indices = indices.view(indices.size(0), indices.size(1) // n_quantizer, n_quantizer)
                    indices = (indices % n_code).long()
                    """
                    latents = cast(Tensor, cast(Any, vae.vq).get_codes_from_indices(indices))
                    latents = reduce(latents, "q ... -> ...", "sum")
                else:
                    if self.loss_type == "ce":
                        indices = torch.argmax(latents, dim=-1)
                    else:
                        indices = bit2int(latents > 0)
                    latents = cast(Tensor, cast(Any, vae.vq).get_codes_from_indices(indices))

            feature = latents
            if hasattr(vae, "latent_to_embd"):
                feature = cast(Any, vae).latent_to_embd(latents)

        return feature

    def get_conditioning(self, conditioning: Tensor, **kwargs) -> Tensor | None:
        if self._conditioner is not None:
            if isinstance(self._conditioner, nn.Module):
                with torch.enable_grad() if self._conditioner.training else torch.no_grad():
                    if isinstance(self._conditioner, Model):
                        return cast(Tensor, cast(Any, self._conditioner).encode(conditioning, **kwargs))
                    return cast(Tensor, self._conditioner(conditioning, **kwargs))
            if callable(self._conditioner):
                return cast(Callable[..., Tensor], self._conditioner)(conditioning, **kwargs)
            vae = cast(VAEModel | VQVAEModel, self._vae)
            if hasattr(vae, "latent_to_feat"):
                with torch.enable_grad() if cast(nn.Module, vae).training else torch.no_grad():
                    return cast(Any, vae).latent_to_feat(self.encode(conditioning, **kwargs))
            else:
                return self.encode(conditioning, **kwargs)
        return conditioning

    def forward(
        self,
        noisy_latents: Tensor | None = None,
        sigmas: Tensor | None = None,
        conditioning: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if noisy_latents is None:
            noisy_latents = self.encode(**kwargs)
        if sigmas is None:
            sigmas = torch.full(
                (noisy_latents.size(0),), self._denoise_fn.sigma_data, dtype=torch.float32, device=noisy_latents.device
            )
            sigmas = unsqueeze_as(sigmas, noisy_latents)
        if conditioning is None and hasattr(self._denoise_fn, "category_emb"):
            conditioning = torch.randint(
                self._denoise_fn.category_emb.num_embeddings, (noisy_latents.size(0),), device=noisy_latents.device
            )

        if not hasattr(self._denoise_fn, "category_emb") and not self._conditioner:
            conditioning = None

        cond_samples = self.scheduler.precondition_inputs(noisy_latents, sigmas)
        cond_sigmas = self.scheduler.precondition_noise(sigmas)
        model_output = self._denoise_fn(cond_samples, cond_sigmas.flatten(), conditioning, **kwargs)
        return self.scheduler.precondition_outputs(noisy_latents, model_output.float(), sigmas)

    def _train_val_step(self, latents: Tensor, conditioning: Tensor | None = None) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(latents, dtype=torch.float32)
        timesteps = unsqueeze_as(torch.randn((latents.size(0),), device=latents.device), latents)
        sigmas = self.scheduler.precondition_timesteps(timesteps)
        noisy_latents = self.scheduler.add_noise(latents.float(), noise, sigmas)

        pos_enc = 0
        if self.pos_enc:
            pos_enc = self.pos_enc(torch.arange(latents.size(1), dtype=torch.long, device=latents.device))

        denoised = self(noisy_latents + pos_enc, sigmas, conditioning) - pos_enc
        return denoised, sigmas

    def _apply_pre_loss_fn(self, latents: Tensor, denoised: Tensor) -> tuple[Tensor, Tensor]:
        pre_loss_fn = self.pre_loss_fn
        if pre_loss_fn:
            if self.bit_diffusion:
                denoised = denoised / self.sigma_data
                latents = latents / self.sigma_data

            if pre_loss_fn in [F.sigmoid, torch.sigmoid]:
                denoised = cast(Callable[[Tensor], Tensor], pre_loss_fn)(denoised)
                if self.bit_diffusion:
                    latents = (latents + 1) / 2
            elif pre_loss_fn in [F.tanh, torch.tanh]:
                denoised = cast(Callable[[Tensor], Tensor], pre_loss_fn)(denoised / 0.1)
            elif pre_loss_fn == torch.clamp:
                denoised = torch.clamp(denoised, -1, 1)
            elif isinstance(pre_loss_fn, Attention):
                denoised = pre_loss_fn(latents, denoised, denoised).float()
                latents = pre_loss_fn(latents).float()
            else:
                denoised = cast(Callable[[Tensor], Tensor], pre_loss_fn)(denoised).float()

            if self.bit_diffusion:
                denoised = denoised * self.sigma_data
                latents = latents * self.sigma_data
        return latents, denoised

    @torch.no_grad()
    def predict(self, data: dict[str, Tensor], return_loss: bool = False, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        latents = self.encode(inputs=data["inputs"], **kwargs)
        conditioning = self.get_conditioning(data[self.condition_key], **kwargs) if self.condition_key else None
        denoised, sigmas = self._train_val_step(latents, conditioning)

        if self.requantize:
            requantize = cast(Callable[[Tensor], tuple[Tensor, Tensor, Tensor]], self.requantize)
            denoised, _indices, _vq_loss = requantize(denoised)

        vae = cast(VAEModel | VQVAEModel, self._vae)
        logits = cast(
            Tensor,
            vae.predict(
                points=data["points"],
                feature=self.decode(denoised, **kwargs),
                points_batch_size=kwargs.get("points_batch_size"),
            ),
        )
        self.log("bce_loss", F.binary_cross_entropy_with_logits(logits, data["points.occ"]).cpu().item())
        if return_loss:
            latents, denoised = self._apply_pre_loss_fn(latents, denoised)
            loss = F.mse_loss(denoised, latents, reduction="none")
            loss = self.scheduler.precondition_loss(loss, sigmas)
            return logits, loss
        return logits

    @torch.no_grad()
    def evaluate(self, data: dict[str, Tensor], **kwargs) -> dict[str, float]:
        logits = data.get("logits")
        loss = data.get("loss")
        if logits is None or loss is None:
            logits, loss = self.predict(data, return_loss=True, **kwargs)
        data.update({"logits": logits, "loss": loss})
        return super().evaluate(cast(Any, data), **kwargs)

    @torch.inference_mode()
    def generate(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        conditioning: Tensor | None = None,
        threshold: float | None = None,
        points_batch_size: int | None = None,
        num_steps: int = 18,
        progress: bool = True,
        return_intermediates: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        batch_size = inputs.size(0) if inputs is not None else points.size(0) if points is not None else 1
        n_queries = getattr(
            self._vae, "n_queries", getattr(self._vae, "n_fps", 1 if inputs is None else inputs.size(0))
        )
        latents = torch.randn(batch_size, n_queries, self.n_latent, device=self.device)

        if hasattr(self._denoise_fn, "category_emb"):
            if conditioning is None:
                conditioning = torch.randint(
                    self._denoise_fn.category_emb.num_embeddings, (batch_size,), device=self.device
                )
        elif self.condition_key:
            if conditioning is not None:
                conditioning = self.get_conditioning(conditioning, **kwargs)

        pos_enc = latents.new_zeros((1, n_queries, self.n_latent))
        if self.pos_enc:
            pos_enc = self.pos_enc(torch.arange(n_queries, dtype=torch.long, device=self.device)).unsqueeze(0)

        sampler_result = edm_sampler(
            self, latents + pos_enc, conditioning,
            num_steps=num_steps, progress=progress, return_intermediates=return_intermediates,
        )

        if return_intermediates:
            latents, raw_intermediates = cast(tuple[Tensor, list[Tensor]], sampler_result)
            latents = latents - pos_enc
            intermediates = [x - pos_enc for x in raw_intermediates]
            if self.requantize:
                requantize = cast(Callable[[Tensor], tuple[Tensor, Tensor, Tensor]], self.requantize)
                latents, _indices, _vq_loss = requantize(latents)

            feature = self.decode(latents, **kwargs)
            vae = cast(VAEModel | VQVAEModel, self._vae)
            logits = cast(Tensor, vae.predict(points=points, feature=feature, points_batch_size=points_batch_size))
            return logits, intermediates

        latents = cast(Tensor, sampler_result) - pos_enc
        if self.requantize:
            requantize = cast(Callable[[Tensor], tuple[Tensor, Tensor, Tensor]], self.requantize)
            latents, _indices, _vq_loss = requantize(latents)

        feature = self.decode(latents, **kwargs)
        vae = cast(VAEModel | VQVAEModel, self._vae)
        logits = cast(Tensor, vae.predict(points=points, feature=feature, points_batch_size=points_batch_size))
        return logits

    def loss(self, data: dict[str, Tensor], **kwargs) -> Tensor:
        latents = self.encode(inputs=data["inputs"], **kwargs)
        conditioning = self.get_conditioning(data[self.condition_key], **kwargs) if self.condition_key else None
        denoised, sigmas = self._train_val_step(latents, conditioning)

        aux_loss = 0
        if self.requantize:
            requantize = cast(Callable[[Tensor], tuple[Tensor, Tensor, Tensor]], self.requantize)
            denoised, _indices, vq_loss = requantize(denoised)
            vae = cast(VAEModel | VQVAEModel, self._vae)
            aux_loss = (vq_loss * cast(float, getattr(vae, "vq_loss_weight", 1.0))).mean()

        latents, denoised = self._apply_pre_loss_fn(latents, denoised)
        if self.bit_diffusion and self.loss_type in ["ce", "bce"]:
            if self.loss_type == "ce":
                loss = F.cross_entropy(
                    denoised.view(-1, denoised.size(-1)), torch.argmax(latents, dim=-1).view(-1), reduction="none"
                ).view(denoised.size(0), -1)
            else:
                if self.pre_loss_fn in [F.sigmoid, torch.sigmoid]:
                    loss = F.binary_cross_entropy(denoised, (latents > 0).float(), reduction="none").mean(-1)
                else:
                    loss = F.binary_cross_entropy_with_logits(denoised, (latents > 0).float(), reduction="none").mean(
                        -1
                    )
        else:
            if self.loss_type in ["cd", "chamfer"]:
                assert chamfer_distance is not None, "Chamfer distance requires pytorch3d."
                loss = chamfer_distance(denoised, latents, batch_reduction=None, point_reduction=None)[0]
                loss = (loss[0] + loss[1]) / (2 * latents.size(-1))
            elif self.loss_type == "l1":
                loss = F.l1_loss(denoised, latents, reduction="none").mean(-1)
            elif "smooth" in self.loss_type:
                loss = F.smooth_l1_loss(denoised, latents, reduction="none").mean(-1)
            else:
                loss = F.mse_loss(denoised, latents, reduction="none").mean(-1)
        loss = self.scheduler.precondition_loss(loss, sigmas.squeeze(-1))
        return loss.mean() + aux_loss
