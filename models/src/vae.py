from abc import abstractmethod
from logging import DEBUG
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor, nn
from vector_quantize_pytorch import FSQ, LFQ, ResidualVQ, VectorQuantize
from vector_quantize_pytorch.vector_quantize_pytorch import gumbel_sample

from utils import DEBUG_LEVEL_1, DEBUG_LEVEL_2, cosine_anneal, is_distributed, setup_logger

from .mixins import MultiEvalMixin, PredictMixin
from .model import Model
from .pointnet import ResNetPointNet
from .resnet import ResNet18
from .utils import check_finite_context, reduce_loss
from .vqdif import Quantizer

logger = setup_logger(__name__)


def _debug_level_2(message: str) -> None:
    debug_level_2 = getattr(cast(Any, logger), "debug_level_2", None)
    if callable(debug_level_2):
        debug_level_2(message)
        return
    logger.debug(message)


def _require_tensor(data: dict[str, list[str] | Tensor], key: str) -> Tensor:
    value = data.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected tensor for `{key}`")
    return value


class IsotropicGaussian(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        # Register mean and logstd as buffers so .to(device) works
        self.register_buffer("mean", torch.zeros(z_dim))
        self.register_buffer("logstd", torch.zeros(z_dim))  # std = exp(logstd) = exp(0) = 1

    @abstractmethod
    def forward(self, **kwargs) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("A derived class should implement this method")

    @property
    def prior(self) -> dist.Normal:
        mean = cast(Tensor, self.mean)
        logstd = cast(Tensor, self.logstd)
        return dist.Normal(mean, logstd.exp())

    def posterior(self, mean: Tensor | None = None, logstd: Tensor | None = None, *args, **kwargs) -> dist.Normal:
        """
        Get the posterior distribution.

        If mean and logstd are provided, uses them directly.
        If either is None, computes them using the forward method.

        Args:
            mean (Optional[Tensor]): Pre-computed mean of the posterior distribution.
            logstd (Optional[Tensor]): Pre-computed log standard deviation of the posterior distribution.
            *args: Additional positional arguments to be passed to the forward method if needed.
            **kwargs: Additional keyword arguments to be passed to the forward method if needed.

        Returns:
            dist.Normal: The posterior distribution (Normal).
        """
        if mean is None or logstd is None:
            mean, logstd = self(*args, **kwargs)
        assert mean is not None and logstd is not None
        return dist.Normal(mean, logstd.exp())

    def sample_prior(self, size: torch.Size, sample: bool = True, use_dist: bool = True) -> Tensor:
        mean = cast(Tensor, self.mean)
        logstd = cast(Tensor, self.logstd)
        if sample:
            if use_dist:
                return self.prior.rsample(size) if self.training else self.prior.sample(size)
            return mean + torch.randn(*size, *mean.size()) * logstd.exp()
        return self.prior.mean.expand(*size, *self.prior.mean.size())

    def sample_posterior(self, mean: Tensor, logstd: Tensor, use_dist: bool = True) -> Tensor:
        if use_dist:
            posterior = dist.Normal(mean, logstd.exp())
            return posterior.rsample() if self.training else posterior.sample()
        return mean + torch.randn_like(mean) * logstd.exp()

    def kl_divergence(
        self,
        mean: Tensor | None = None,
        logstd: Tensor | None = None,
        reduction: str | None = "mean",
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Compute the KL divergence between the posterior and prior distributions.

        If mean and logstd are provided, uses them directly to create the posterior.
        If either is None, computes the posterior using the posterior method.

        Args:
            mean (Optional[Tensor]): Pre-computed mean of the posterior distribution.
            logstd (Optional[Tensor]): Pre-computed log standard deviation of the posterior distribution.
            reduction (str): Reduction method for the KL divergence. Options: None, 'mean', 'sum', 'sum_points'.
            *args: Additional positional arguments to be passed to the posterior method if needed.
            **kwargs: Additional keyword arguments to be passed to the posterior method if needed.

        Returns:
            Tensor: KL divergence between the posterior and prior distributions.

        Raises:
            ValueError: If mean and logstd have incompatible shapes with the prior distribution.
        """
        posterior = self.posterior(mean, logstd, *args, **kwargs)
        kl_div = dist.kl_divergence(posterior, self.prior)  # forward KL, mean-seeking
        return reduce_loss(kl_div.mean(-1), reduction) if reduction else kl_div


class ResNet18VAE(IsotropicGaussian):
    def __init__(self, c_dim: int = 128):
        super().__init__(z_dim=c_dim)
        self.resnet = ResNet18(2 * c_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        mean_logstd = self.resnet(inputs)
        mean, logstd = torch.chunk(mean_logstd, 2, dim=-1)
        return mean, logstd


class ResNetPointNetVAE(IsotropicGaussian):
    def __init__(
        self,
        dim: int = 3,
        c_dim: int = 128,
        hidden_dim: int = 128,
        n_blocks: int = 5,
        norm: str | None = None,
        activation: str = "relu",
        dropout: float = 0,
        reduce: bool = True,
        sum_feature: bool = False,
    ):
        super().__init__(z_dim=c_dim)
        self.pointnet = ResNetPointNet(dim, c_dim, hidden_dim, n_blocks, norm, activation, dropout, reduce, sum_feature)
        self.mean_logstd = nn.Linear(hidden_dim, 2 * c_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        feature = self.pointnet(inputs)
        mean, logstd = torch.chunk(self.mean_logstd(feature), 2, dim=-1)
        return mean, logstd


class UnconditionalVAE(IsotropicGaussian):
    def __init__(self, dim: int = 3, c_dim: int = 128, hidden_dim: int = 128, leaky: bool = False, **kwargs):
        super().__init__(z_dim=c_dim)
        self.hidden_dim = hidden_dim

        self.fc_pos = nn.Linear(dim, hidden_dim)
        self.fc_0 = nn.Linear(1, hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_logstd = nn.Linear(hidden_dim, c_dim)

        if not leaky:
            self.actvn = nn.ReLU(inplace=True)
            self.pool = lambda x: x.max(dim=1, keepdim=True)[0]
        else:
            self.actvn = nn.LeakyReLU(0.2, inplace=True)
            self.pool = torch.mean

    def forward(self, points: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        net = self.fc_0(targets.unsqueeze(-1))  # B x N -> B x N x 1 -> B x N x 128
        net = net + self.fc_pos(points)  # B x N x 128

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))
        net = self.pool(net).squeeze(1)  # B x 128

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd


class VAEModel(IsotropicGaussian, MultiEvalMixin, PredictMixin, Model):
    def __init__(self, ae: Model, n_embd: int, n_latent: int | None = None, kl_loss_weight: float = 1e-2):
        super().__init__(z_dim=n_latent or n_embd)
        self.ae = ae
        self.mean_logstd = nn.Linear(n_embd, 2 * (n_latent or n_embd))
        self.latent_to_embd = nn.Linear(n_latent or n_embd, n_embd) if n_latent != n_embd else nn.Identity()
        self.n_latent = n_latent
        self.kl_loss_weight = kl_loss_weight

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.ae.setup(*args, **kwargs)
        if self.stats:
            _debug_level_2(
                f"{self.name} stats: "
                f"mean={self.stats['mean'].mean().item():.3f}, "
                f"std={self.stats['std'].mean().item():.3f} "
                f"({self.stats['std'].min().item():.3f},{self.stats['std'].max().item():.3f})"
            )

    def teardown(self, *args, **kwargs):
        self.ae.teardown(*args, **kwargs)
        super().teardown(*args, **kwargs)

    def get_mean_logstd(self, inputs: Tensor, clamp: bool = True, **kwargs) -> tuple[Tensor, Tensor]:
        ae_model = cast(Any, self.ae)
        x = cast(Tensor, ae_model.encode(inputs, **kwargs))
        mean_logstd = cast(Tensor, self.mean_logstd(x))
        mean, logstd = torch.chunk(mean_logstd, 2, dim=-1)
        if clamp:
            logstd = logstd.clamp(min=-10, max=10)
        return mean, logstd

    def sample_posterior(
        self, inputs: Tensor, sample: bool = True, return_moments: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        mean, logstd = self.get_mean_logstd(inputs, **kwargs)
        z = super().sample_posterior(mean, logstd) if self.training or sample else mean
        if return_moments:
            return z, mean, logstd
        return z

    def forward(self, inputs: Tensor, points: Tensor | None = None, **kwargs) -> dict[str, Tensor]:
        return self.decode(points, self.encode(inputs, **kwargs), **kwargs)

    def encode(self, inputs: Tensor, sample: bool = False, **kwargs) -> Tensor:
        return self.latent_to_embd(self.sample_posterior(inputs, sample=sample, **kwargs))

    def decode(self, points: Tensor | None, feature: Tensor, **kwargs) -> dict[str, Tensor]:
        if points is None:
            raise ValueError("`points` must be provided for decoding")
        ae_model = cast(Any, self.ae)
        return cast(dict[str, Tensor], ae_model.decode(points, feature, **kwargs))

    @torch.no_grad()
    def predict(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        feature: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        key: str | None = "auto",
        points_batch_size: int | None = None,
        sample: bool = False,
        **kwargs,
    ) -> Tensor:
        if feature is None and sample:
            if inputs is None and points is not None:  # Unconditional
                n = getattr(self, "n_queries", 1)
                if self.stats:
                    prior = super().sample_posterior(
                        self.stats["mean"].expand(points.size(0), n, -1),
                        self.stats["std"].expand(points.size(0), n, -1).log(),
                    )
                else:
                    prior = self.sample_prior(torch.Size((points.size(0), n)))
                feature = self.latent_to_embd(prior.to(self.device))
            elif feature is None:
                if inputs is None:
                    raise ValueError("`inputs` must be provided when `feature` is None")
                feature = self.encode(inputs, sample=sample, **kwargs)
        return cast(Tensor, super().predict(inputs, points, feature, key, points_batch_size, **kwargs))

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        kl_loss_weight: float | None = None,
        global_step: int | None = None,
        total_steps: int | None = None,
        **kwargs,
    ) -> Tensor:
        inputs = _require_tensor(data, "inputs")
        points = _require_tensor(data, "points")

        z, mean, logstd = self.sample_posterior(inputs, return_moments=True, **kwargs)
        if "logits" not in data:
            data.update(self.decode(points, feature=self.latent_to_embd(z), **kwargs))

        nll = self.ae.loss(data, regression, name, reduction="none")
        with check_finite_context(nll, name="nll", do_raise=False, enabled=logger.isEnabledFor(DEBUG_LEVEL_2)) as t:
            if t is None:
                nll = nll.mean()
            else:
                nll[t] = torch.nan
                nll = torch.nanmean(nll)

        kl_div = self.kl_divergence(mean, logstd, reduction=None)
        with check_finite_context(
            kl_div, name="kl_div", do_raise=False, enabled=logger.isEnabledFor(DEBUG_LEVEL_2)
        ) as t:
            if t is None:
                kl_div = kl_div.mean()
            else:
                kl_div[t] = torch.nan
                kl_div = torch.nanmean(kl_div)

        elbo = -nll - kl_div
        if global_step is not None and total_steps and not kl_loss_weight:
            kl_loss_weight = cosine_anneal(self.kl_loss_weight / 100, self.kl_loss_weight, total_steps, global_step)
            self.log("aux_loss_weight", kl_loss_weight, level=DEBUG_LEVEL_2, train_only=True)
        loss = nll + (kl_loss_weight or self.kl_loss_weight) * kl_div

        latent_hist = z.detach().float().cpu().numpy().flatten()
        latent_dim = self.n_latent if self.n_latent is not None else z.size(-1)
        self.log("rec_loss", nll.item())
        self.log("kl_loss", kl_div.item())
        self.log("elbo", elbo.item())
        self.log("mean", latent_hist.mean().item(), level=DEBUG_LEVEL_1)
        self.log("std", latent_hist.std().item(), level=DEBUG_LEVEL_1)
        self.log("latent_hist", latent_hist, level=DEBUG_LEVEL_2)
        self.log("mean", latent_hist.reshape(-1, latent_dim).mean(0), level=DEBUG, train_only=True, ema=True)
        self.log("std", latent_hist.reshape(-1, latent_dim).std(0), level=DEBUG, train_only=True, ema=True)

        return loss


class VQVAEModel(MultiEvalMixin, PredictMixin, Model):
    def __init__(
        self,
        ae: Model,
        n_hidden: int,
        n_code: int,
        n_latent: int | None = None,
        quantizer: Literal["vq", "fsq", "lfq", "legacy", "gumbel"] = "vq",
        learnable_codebook: bool = False,
        vq_loss_weight: float = 1.0,
        decay: float = 0.8,
        kmeans_init: bool = True,
        use_cosine_sim: bool = False,
        threshold_ema_dead_code: int = 0,
        entropy_loss_weight: float | None = None,
        commitment_loss_weight: float | None = None,
        use_cross_entropy: bool = False,
        sample_codes: bool = False,
        n_quantizer: int = 1,
        shared_codebook: bool = False,
        quantize_soft: bool = False,
        **kwargs,
    ):
        """Vector Quantized Variational Autoencoder (VQ-VAE).

        Args:
            ae (Model): Autoencoder model to use as backbone for VQ-VAE.
            n_hidden (int): Number of hidden units in the backbone model.
            n_code (int): Number of codes in the codebook.
            n_latent (Optional[int], optional): Number of latent units in the backbone model. Defaults to None.
            quantizer (Literal["vq", "fsq", "lfq", "legacy"], optional): Quantizer to use. Defaults to "vq".
            learnable_codebook (bool): Whether to use a learnable codebook or not. Defaults to False.
            vq_loss_weight  (float): Weight of the VQ loss. Defaults to 1.
            decay   (float): Decay rate for the exponential moving average of the codebook. Defaults to 0.99.
            kmeans_init   (bool): Whether to use K-Means initialization for the codebook or not. Defaults to False.
            use_cosine_sim (bool): Whether to use cosine similarity for the codebook or not. Defaults to False.
            threshold_ema_dead_code (int): Threshold for the EMA dead code in the codebook. Defaults to 0.
            entropy_loss_weight   (Optional[float], optional): Weight of the entropy loss. Defaults to None.
            commitment_loss_weight    (Optional[float], optional): Weight of the commitment loss. Defaults to None.
            use_cross_entropy (bool): Whether to use cross entropy loss for the commitment loss or not. Defaults to False.
        """
        super().__init__()
        self.ae = ae
        self.vq_loss_weight = vq_loss_weight
        self.n_code = n_code
        self.n_bit = int(np.ceil(np.log2(n_code)))
        self.n_latent = n_latent or n_hidden
        self.sample_codes = sample_codes
        self.quantize_soft = quantize_soft

        latent_dim = self.n_latent
        if quantizer in ["vq", "gumbel"]:
            if n_quantizer > 1:
                assert n_code % n_quantizer == 0, "Codebook size must be divisible by the number of quantizers"
                self.vq = ResidualVQ(
                    dim=n_hidden,
                    num_quantizers=n_quantizer,
                    codebook_dim=latent_dim if latent_dim != n_hidden else None,
                    shared_codebook=shared_codebook,
                    codebook_size=n_code // n_quantizer,
                    stochastic_sample_codes=sample_codes,
                    commitment_weight=commitment_loss_weight or 1,
                    decay=decay,
                    kmeans_init=kmeans_init,
                    use_cosine_sim=use_cosine_sim,
                    threshold_ema_dead_code=threshold_ema_dead_code,
                )
            else:
                self.vq = VectorQuantize(
                    dim=n_hidden,
                    codebook_size=n_code,
                    codebook_dim=latent_dim if latent_dim != n_hidden else None,
                    codebook_diversity_loss_weight=entropy_loss_weight or 0,
                    stochastic_sample_codes=sample_codes,
                    commitment_weight=commitment_loss_weight or 1,
                    commitment_use_cross_entropy_loss=use_cross_entropy,
                    decay=decay,
                    kmeans_init=kmeans_init,
                    use_cosine_sim=use_cosine_sim,
                    threshold_ema_dead_code=threshold_ema_dead_code,
                    ema_update=not (learnable_codebook or quantizer == "gumbel"),
                    learnable_codebook=learnable_codebook or quantizer == "gumbel",
                )
                if quantizer == "gumbel":
                    self.logits = nn.Linear(latent_dim if latent_dim != n_hidden else n_hidden, n_code)
        elif quantizer == "fsq":
            level_map = {
                2**4: [5, 3],
                2**6: [8, 8],
                2**8: [8, 6, 5],
                2**9: [8, 8, 8],
                2**10: [8, 5, 5, 5],
                2**11: [8, 8, 6, 5],
                2**12: [7, 5, 5, 5, 5],
                2**13: [8, 8, 5, 5, 5],
                2**14: [8, 8, 8, 6, 5],
                2**15: [7, 6, 6, 5, 5, 5],
                2**16: [8, 8, 8, 5, 5, 5],
            }
            self.n_code = np.prod(level_map[n_code])
            self.vq = FSQ(levels=level_map[n_code], dim=n_hidden)
        elif quantizer == "lfq":
            self.vq = LFQ(
                dim=n_hidden,
                codebook_size=n_code,
                entropy_loss_weight=entropy_loss_weight or 0.1,
                commitment_loss_weight=commitment_loss_weight or 0.25,
                experimental_softplus_entropy_loss=True,
            )
        elif quantizer == "legacy":
            if latent_dim != n_hidden:
                self.vq = nn.ModuleList(
                    [
                        nn.Linear(n_hidden, latent_dim),
                        Quantizer(n_code, latent_dim, decay),
                        nn.Linear(latent_dim, n_hidden),
                    ]
                )
            else:
                self.vq = Quantizer(n_code, n_hidden, decay)
        else:
            raise ValueError(f"Unknown quantizer: {quantizer}")

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.ae.setup(*args, **kwargs)
        if self.stats:
            _debug_level_2(
                f"{self.name} stats: "
                f"mean={self.stats['mean'].mean().item():.3f}, "
                f"std={self.stats['std'].mean().item():.3f} "
                f"({self.stats['std'].min().item():.3f},{self.stats['std'].max().item():.3f})"
            )

    def teardown(self, *args, **kwargs):
        self.ae.teardown(*args, **kwargs)
        super().teardown(*args, **kwargs)

    @torch.autocast(device_type="cuda", enabled=False)
    def quantize(self, x: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        dtype = x.dtype
        x = x.float()
        vq_loss: Tensor = x.new_zeros(())
        temperature = kwargs.get("temperature", 1.0) if self.training else 0.0
        if isinstance(self.vq, FSQ):
            quantized, indices = self.vq(x)
        elif isinstance(self.vq, nn.ModuleList):
            project_in = cast(nn.Module, self.vq[0])
            quantizer = cast(Any, self.vq[1])
            project_out = cast(nn.Module, self.vq[2])
            quantized, indices, vq_loss = quantizer(cast(Tensor, project_in(x)))
            quantized = cast(Tensor, project_out(quantized))
        elif hasattr(self, "logits"):
            logits_layer = cast(nn.Module, self.logits)
            vq_module = cast(Any, self.vq)
            logits = cast(Tensor, logits_layer(cast(Tensor, vq_module.project_in(x))))
            # indices_one_hot = F.gumbel_softmax(logits, dim=-1, tau=1, hard=True)
            # indices = indices_one_hot.argmax(dim=-1)
            indices, indices_one_hot = gumbel_sample(
                logits,
                temperature=temperature,
                stochastic=self.sample_codes,
                straight_through=True,  # Necessary for gradients
                training=self.training,
            )
            embed = cast(Tensor, vq_module._codebook.embed).squeeze(0)
            quantized = torch.einsum("b...i,ij->b...j", indices_one_hot, embed)
            quantized = cast(Tensor, vq_module.project_out(quantized))

            beta = kwargs.get("beta", vq_module.codebook_diversity_loss_weight)
            if beta:
                log_probs = F.log_softmax(logits, dim=-1)
                uniform = torch.full_like(logits, 1 / self.n_code)
                vq_loss = beta * F.kl_div(log_probs, uniform, reduction="batchmean")
        elif isinstance(self.vq, (VectorQuantize, ResidualVQ)):
            if isinstance(self.vq, VectorQuantize):
                self.vq.codebook_diversity_loss_weight = kwargs.get("beta", self.vq.codebook_diversity_loss_weight)
            quantized, indices, vq_loss = self.vq(x, sample_codebook_temp=temperature)
        else:
            quantized, indices, vq_loss = self.vq(x)

        if not isinstance(vq_loss, Tensor):
            vq_loss = torch.as_tensor(vq_loss, device=x.device, dtype=x.dtype)

        if self.quantize_soft:
            p = kwargs.get("q_prob", 0.5) if self.training else 1.0
            if self.sample_codes:
                # use p% of quantized and (1-p)% of x
                quantized = torch.where(torch.rand_like(x) > p, x, quantized) if p < 1 else quantized
            else:
                # add (1-p)% of x to quantized
                quantized = quantized * p + x * (1 - p) if p < 1 else quantized

        return quantized.type(dtype), indices, vq_loss.mean()

    def forward(self, inputs: Tensor, points: Tensor | None = None, **kwargs) -> dict[str, Tensor]:
        feature = cast(Tensor, self.encode(inputs, **kwargs))
        return self.decode(points, feature, **kwargs)

    def encode(self, inputs: Tensor, return_all: bool = False, **kwargs) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        ae_model = cast(Any, self.ae)
        quantized, indices, vq_loss = self.quantize(cast(Tensor, ae_model.encode(inputs, **kwargs)), **kwargs)

        quant_hist = quantized.detach().float().cpu().numpy().flatten()
        self.log("quantized_hist", quant_hist, level=DEBUG_LEVEL_2)
        self.log("mean", quant_hist.reshape(-1, self.n_latent).mean(0), level=DEBUG, train_only=True, ema=True)
        self.log("std", quant_hist.reshape(-1, self.n_latent).std(0), level=DEBUG, train_only=True, ema=True)

        if return_all:
            return quantized, indices, vq_loss
        return quantized

    def decode(self, points: Tensor | None, feature: Tensor, **kwargs) -> dict[str, Tensor]:
        if points is None:
            raise ValueError("`points` must be provided for decoding")
        ae_model = cast(Any, self.ae)
        return cast(dict[str, Tensor], ae_model.decode(points, feature, **kwargs))

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        vq_loss_weight: float | None = None,
        global_step: int | None = None,
        total_steps: int | None = None,
        **kwargs,
    ) -> Tensor:
        inputs = _require_tensor(data, "inputs")
        points = _require_tensor(data, "points")

        feature, indices, vq_loss = self.encode(
            inputs, return_all=True, global_step=global_step, total_steps=total_steps, **kwargs
        )
        with check_finite_context(
            vq_loss, name="vq_loss", do_raise=False, enabled=logger.isEnabledFor(DEBUG_LEVEL_2)
        ) as t:
            if t is not None:
                vq_loss = vq_loss.new_zeros(())

        if "logits" not in data:
            data.update(self.decode(points, feature, **kwargs))

        rec_loss = self.ae.loss(data, regression, name, reduction="none")
        with check_finite_context(
            rec_loss, name="rec_loss", do_raise=False, enabled=logger.isEnabledFor(DEBUG_LEVEL_2)
        ) as t:
            if t is None:
                rec_loss = rec_loss.mean()
            else:
                rec_loss[t] = torch.nan
                rec_loss = torch.nanmean(rec_loss)

        if global_step and total_steps and not vq_loss_weight:
            vq_loss_weight = cosine_anneal(0.0, 1.0, total_steps, global_step)
            self.log("aux_loss_weight", vq_loss_weight, level=DEBUG_LEVEL_2, train_only=True)
        loss = rec_loss + (vq_loss_weight or self.vq_loss_weight) * vq_loss

        with torch.no_grad():
            if indices.ndim == 3:
                b, _n, c = indices.size()
                offset = torch.arange(c, device=indices.device).view(1, 1, c) * int(self.n_code // c)
                indices = (indices + offset).view(b, -1)

            indices_count = torch.bincount(indices.view(-1), minlength=int(self.n_code))
            if is_distributed():
                torch.distributed.all_reduce(indices_count)
            avg_probs = indices_count.float() / indices_count.sum()
            active_codes = (indices_count > 0).sum().item()
            perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp().item()

        self.log("codebook_usage", active_codes / self.n_code * 100)
        self.log("perplexity", perplexity, level=DEBUG_LEVEL_1)
        self.log("active_codes", active_codes, level=DEBUG_LEVEL_2)
        self.log("index_hist", indices.detach().float().cpu().numpy().flatten(), level=DEBUG_LEVEL_2)
        if vq_loss.item() > 0:
            self.log("rec_loss", rec_loss.item())
            self.log("vq_loss", vq_loss.item())

        return loss
