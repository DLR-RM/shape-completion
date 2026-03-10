import itertools
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from ..mixins import MultiEvalMixin
from ..utils import DropPath
from ..vae import VAEModel
from .model import DiffusionModel
from .utils import unsqueeze_as


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        if context is None:
            context = x

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = AdaLayerNorm(dim)
        self.norm2 = AdaLayerNorm(dim)
        self.norm3 = AdaLayerNorm(dim)
        self.checkpoint = checkpoint

        init_values = 0
        drop_path = 0.0

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, t, context=None):
        x = self.drop_path1(self.ls1(self.attn1(self.norm1(x, t)))) + x
        x = self.drop_path2(self.ls2(self.attn2(self.norm2(x, t), context=context))) + x
        x = self.drop_path3(self.ls3(self.ff(self.norm3(x, t)))) + x
        return x


class LatentArrayTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self, in_channels, t_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None, out_channels=None
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.t_channels = t_channels

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(inner_dim)

        if out_channels is None:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels, bias=False))
        else:
            self.num_cls = out_channels
            self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=False))

        self.context_dim = context_dim

        self.map_noise = PositionalEmbedding(t_channels)

        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim)
        self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim)

        # ###
        # self.pos_emb = nn.Embedding(512, inner_dim)
        # ###

    def forward(self, x, t, cond=None):
        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        x = self.proj_in(x)

        # ###
        # x = x + self.pos_emb.weight[None]
        # ###

        for block in self.transformer_blocks:
            x = block(x, t_emb, context=cond)

        x = self.norm(x)

        x = self.proj_out(x)
        return x


def edm_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    # S_churn=40, S_min=0.05, S_max=50, S_noise=1.003,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(itertools.pairwise(t_steps)):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, inputs, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([inputs.shape[0], 1, 1], device=inputs.device)
        # rnd_normal = torch.randn([1, 1, 1], device=inputs.device).repeat(inputs.shape[0], 1, 1)

        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, _augment_labels = augment_pipe(inputs) if augment_pipe is not None else (inputs, None)

        n = torch.randn_like(y) * sigma

        D_yn = net(y + n, sigma, labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss.mean()


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


class EDMPrecond(MultiEvalMixin, DiffusionModel):
    def __init__(
        self,
        vae: VAEModel,
        n_latent=512,
        channels=32,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
        n_heads=8,
        d_head=64,
        depth=24,
        **kwargs,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.channels = channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self._vae = vae
        self._vae.requires_grad_(False)
        self._vae.eval()
        self.model = LatentArrayTransformer(
            in_channels=channels, t_channels=256, n_heads=n_heads, d_head=d_head, depth=depth
        )
        self.category_emb = nn.Embedding(57, n_heads * d_head)
        self.condition_key = "category.index"

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self._vae.setup(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return {k: v for k, v in super().state_dict(*args, **kwargs).items() if k not in "_vae"}

    def emb_category(self, class_labels):
        return self.category_emb(class_labels).unsqueeze(1)

    def encode(self, inputs: Tensor, **kwargs) -> Tensor:
        with torch.enable_grad() if self._vae.training else torch.no_grad():
            posterior = self._vae.sample_posterior(inputs)
            if isinstance(posterior, tuple):
                posterior = posterior[0]
            return posterior.float()

    def decode(self, latents: Tensor, **kwargs) -> Tensor:
        with torch.enable_grad() if self._vae.training else torch.no_grad():
            return self._vae.latent_to_embd(latents)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **kwargs):
        if x is None:
            x = self.encode(**kwargs)
        if sigma is None:
            sigma = torch.full((x.size(0),), self.sigma_data, dtype=torch.float32, device=x.device)
            sigma = unsqueeze_as(sigma, x)
        if class_labels is None:
            class_labels = torch.randint(self.category_emb.num_embeddings, (x.size(0),), device=x.device)

        if torch.is_floating_point(class_labels):
            cond_emb = class_labels
        else:
            cond_emb = self.category_emb(class_labels).unsqueeze(1)

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), cond=cond_emb, **kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    @torch.no_grad()
    def predict(self, data: dict[str, Tensor], return_loss: bool = False, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        latents = self.encode(inputs=data["inputs"], **kwargs)

        rnd_normal = torch.randn([latents.shape[0], 1, 1], device=latents.device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()
        noise = torch.randn_like(latents) * sigma
        denoised = self(latents + noise, sigma, data[self.condition_key])

        logits = self._vae.predict(
            points=data["points"],
            feature=self.decode(denoised, **kwargs),
            points_batch_size=kwargs.get("points_batch_size"),
        )
        self.log("bce_loss", F.binary_cross_entropy_with_logits(logits, data["points.occ"]).cpu().item())
        if return_loss:
            weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
            loss = weight * ((denoised - latents) ** 2)
            return logits, loss
        return logits

    @torch.no_grad()
    def evaluate(self, data: dict[str, Tensor], **kwargs) -> dict[str, float]:
        logits = data.get("logits")
        loss = data.get("loss")
        if logits is None or loss is None:
            logits, loss = self.predict(data, return_loss=True, **kwargs)
        data.update({"logits": logits, "loss": loss})
        return super().evaluate(cast(dict[str, Any], data), **kwargs)

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
        **kwargs,
    ) -> Tensor:
        latents = self.sample(cond=conditioning).float()
        feature = self.decode(latents, **kwargs)
        logits = self._vae.predict(points=points, feature=feature, points_batch_size=points_batch_size)
        return logits

    def loss(self, data: dict[str, Tensor], **kwargs) -> Tensor:
        latents = self.encode(inputs=data["inputs"], **kwargs)
        return EDMLoss(sigma_data=self.sigma_data)(self, latents, data[self.condition_key])

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    @torch.no_grad()
    def sample(self, cond: Tensor | None, batch_seeds: Tensor | None = None):
        if cond is not None:
            batch_size, device = *cond.shape, cond.device
            if batch_seeds is None:
                batch_seeds = torch.arange(batch_size, device=device)
        else:
            if batch_seeds is None:
                device = torch.device(self.device)
                batch_seeds = torch.arange(1, device=device)
            device = batch_seeds.device
            batch_size = batch_seeds.shape[0]

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, self.n_latent, self.channels], device=device)

        return edm_sampler(self, latents, cond, randn_like=rnd.randn_like)
