import importlib
from collections.abc import Callable
from types import MethodType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_edm_euler import EDMEulerScheduler
from pytorch3d.loss import chamfer_distance
from sklearn.datasets import make_swiss_roll
from torch import Tensor, nn

from utils import Voxelizer

from ..src.diffusion import DiffusersModel, bit2int, int2bit
from ..src.diffusion import diffusers as diffusers_module
from ..src.diffusion import latent as latent_module
from ..src.diffusion.edm import EDMScheduler, edm_sampler
from ..src.diffusion.latent import LatentDiffusionModel
from ..src.diffusion.transformer import EDMTransformer
from ..src.diffusion.unet import UNet
from ..src.model import Model
from ..src.transformer import Attention
from ..src.vae import VQVAEModel

try:
    from diffusers.schedulers.scheduling_vdm import VDMScheduler
except ImportError:
    VDMScheduler = None

try:
    _diffusers_schedulers = importlib.import_module("diffusers.schedulers")
    DiscreteStateScheduler = cast(Any, getattr(_diffusers_schedulers, "DiscreteStateScheduler", None))
except ImportError:
    DiscreteStateScheduler = cast(Any, None)


def _num_train_timesteps(scheduler: Any) -> int:
    config = getattr(scheduler, "config", None)
    if isinstance(config, dict):
        value = config.get("num_train_timesteps")
    else:
        value = getattr(config, "num_train_timesteps", None)
    if value is None:
        raise ValueError("Scheduler does not expose num_train_timesteps")
    return int(value)


class _FakeEDMNet(nn.Module):
    sigma_min = 0.1
    sigma_max = 1.5

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.calls: list[dict[str, Any]] = []

    def forward(self, samples: Tensor, timestep: Tensor, cond: Tensor | None = None) -> Tensor:
        timestep_tensor = torch.as_tensor(timestep, dtype=samples.dtype, device=samples.device)
        view_shape = (1,) + (1,) * (samples.ndim - 1)
        self.calls.append(
            {
                "samples": samples.detach().clone(),
                "timestep": timestep_tensor.detach().clone(),
                "cond": None if cond is None else cond.detach().clone(),
            }
        )
        return samples - timestep_tensor.reshape(view_shape)


class _FakeLatentScheduler:
    def __init__(self, sigma_data: float = 0.5) -> None:
        self.config = SimpleNamespace(sigma_data=sigma_data)
        self.precondition_timesteps_calls: list[Tensor] = []
        self.add_noise_calls: list[dict[str, Tensor]] = []
        self.precondition_inputs_calls: list[dict[str, Tensor]] = []
        self.precondition_loss_calls: list[dict[str, Tensor]] = []

    def precondition_timesteps(self, timesteps: Tensor) -> Tensor:
        self.precondition_timesteps_calls.append(timesteps.detach().clone())
        return timesteps.abs() + 0.5

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor, **kwargs: Any) -> Tensor:
        _ = kwargs
        self.add_noise_calls.append(
            {
                "original_samples": original_samples.detach().clone(),
                "noise": noise.detach().clone(),
                "timesteps": timesteps.detach().clone(),
            }
        )
        return original_samples + noise + timesteps

    def precondition_inputs(self, samples: Tensor, sigmas: Tensor) -> Tensor:
        self.precondition_inputs_calls.append(
            {
                "samples": samples.detach().clone(),
                "sigmas": sigmas.detach().clone(),
            }
        )
        return samples + 1.0

    def precondition_noise(self, sigma: Tensor) -> Tensor:
        return sigma + 2.0

    def precondition_outputs(self, samples: Tensor, model_output: Tensor, sigmas: Tensor) -> Tensor:
        return samples + model_output + sigmas

    def precondition_loss(self, loss: Tensor, sigmas: Tensor) -> Tensor:
        self.precondition_loss_calls.append(
            {
                "loss": loss.detach().clone(),
                "sigmas": sigmas.detach().clone(),
            }
        )
        return loss + sigmas


class _FakeDiffusersDenoiser(nn.Module):
    def __init__(self, *, self_condition: bool = False) -> None:
        super().__init__()
        self.self_condition = self_condition
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        x: tuple[Tensor, Tensor],
        t: Tensor,
        x_self_cond: Tensor | None = None,
    ) -> Tensor:
        inputs, points = x
        del points
        t_tensor = torch.as_tensor(t, dtype=inputs.dtype, device=inputs.device)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.unsqueeze(0)
        t_term = t_tensor.reshape(inputs.size(0), *([1] * (inputs.ndim - 1)))
        output = inputs + 0.1 * t_term
        if x_self_cond is not None:
            output = output + 0.25 * x_self_cond
        self.calls.append(
            {
                "inputs": inputs.detach().clone(),
                "t": t_tensor.detach().clone(),
                "x_self_cond": None if x_self_cond is None else x_self_cond.detach().clone(),
            }
        )
        return output


class _FakeCodebook(nn.Module):
    def __init__(self, n_latent: int) -> None:
        super().__init__()
        self.n_latent = n_latent
        self.num_quantizers = None
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def get_codes_from_indices(self, indices: Tensor) -> Tensor:
        return indices.float().unsqueeze(-1).expand(*indices.shape, self.n_latent)


class _FakeConditionerModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.setup_calls = 0
        self.encode_calls: list[Tensor] = []

    def setup(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self.setup_calls += 1

    def encode(self, conditioning: Tensor, **kwargs: Any) -> Tensor:
        _ = kwargs
        self.encode_calls.append(conditioning.detach().clone())
        return conditioning + 3.0

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        _ = args, kwargs
        return {}

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        _ = args, kwargs
        return {}

    def predict(self, *args: Any, **kwargs: Any) -> Tensor:
        _ = args, kwargs
        return torch.zeros((1, 1))

    def loss(self, *args: Any, **kwargs: Any) -> Tensor:
        _ = args, kwargs
        return torch.tensor(0.0)


class _FakeAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def setup(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs

    def encode(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        _ = kwargs
        return inputs

    def decode(self, points: Tensor, feature: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        _ = kwargs
        return {"logits": feature.sum(dim=-1) + points.sum(dim=-1) * 0}


def _make_vqvae(*, n_queries: int = 3, n_latent: int = 4) -> VQVAEModel:
    vae = cast(Any, VQVAEModel)(
        ae=_FakeAutoencoder(), n_hidden=n_latent, n_code=2**n_latent, n_latent=n_latent, kmeans_init=False
    )
    vae_any = cast(Any, vae)
    vae_any.n_queries = n_queries
    vae_any.setup_calls = 0
    vae_any.predict_calls = []
    vae_any.quantize_calls = []
    vae_any.vq = _FakeCodebook(n_latent)
    vae_any.stats = {
        "mean": torch.full((1, n_queries, n_latent), 2.0),
        "std": torch.full((1, n_queries, n_latent), 4.0),
    }

    def _setup(self: Any, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self.setup_calls += 1

    def _quantize(self: Any, inputs: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor]:
        self.quantize_calls.append({"inputs": inputs.detach().clone(), "kwargs": kwargs})
        batch_size, queries, _ = inputs.shape
        indices = torch.arange(queries, device=inputs.device).view(1, queries).expand(batch_size, -1) % int(self.n_code)
        return inputs + 1.0, indices.long(), inputs.new_tensor(0.25)

    def _predict(
        self: Any,
        points: Tensor | None = None,
        feature: Tensor | None = None,
        points_batch_size: int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        assert points is not None and feature is not None
        self.predict_calls.append(
            {
                "points": points.detach().clone(),
                "feature": feature.detach().clone(),
                "points_batch_size": points_batch_size,
                "kwargs": kwargs,
            }
        )
        return feature.sum(dim=-1)

    def _latent_to_feat(self: Any, latents: Tensor, **kwargs: Any) -> Tensor:
        _ = kwargs
        return latents + 5.0

    def _requantize(self: Any, latents: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor]:
        _ = kwargs
        indices = torch.zeros(latents.shape[:-1], dtype=torch.long, device=latents.device)
        return latents + 2.0, indices, latents.new_tensor(0.75)

    vae_any.setup = MethodType(_setup, vae_any)
    vae_any.quantize = MethodType(_quantize, vae_any)
    vae_any.predict = MethodType(_predict, vae_any)
    vae_any.latent_to_feat = MethodType(_latent_to_feat, vae_any)
    vae_any.requantize = MethodType(_requantize, vae_any)
    vae_any.latent_to_embd = nn.Identity()
    return vae


def _make_denoiser(*, n_latent: int = 4, n_classes: int | None = None) -> EDMTransformer:
    denoise = EDMTransformer(n_latent=n_latent, n_layer=1, n_embd=n_latent, n_head=1, n_classes=n_classes)
    denoise_any = cast(Any, denoise)
    denoise_any.calls = []
    denoise_any.setup_calls = 0

    def _setup(self: Any, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self.setup_calls += 1

    def _forward(self: Any, x: Tensor, sigma: Tensor, cond: Tensor | None = None, **kwargs: Any) -> Tensor:
        sigma_tensor = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        sigma_term = sigma_tensor.view(x.size(0), *([1] * (x.ndim - 1))) if sigma_tensor.ndim == 1 else sigma_tensor
        cond_term = x.new_zeros((x.size(0),) + (1,) * (x.ndim - 1))
        cond_record = None
        if cond is not None:
            cond_record = cond.detach().clone()
            cond_scalar = cond.float().reshape(cond.size(0), -1).mean(dim=1)
            cond_term = cond_scalar.view(x.size(0), *([1] * (x.ndim - 1)))
        self.calls.append(
            {
                "x": x.detach().clone(),
                "sigma": sigma_tensor.detach().clone(),
                "cond": cond_record,
                "kwargs": kwargs,
            }
        )
        return x + sigma_term + cond_term

    denoise_any.setup = MethodType(_setup, denoise_any)
    denoise_any.forward = MethodType(_forward, denoise_any)
    return denoise


def _make_latent_diffusion_model(
    *,
    conditioner: Model | Callable[..., Tensor] | None = None,
    n_classes: int | None = None,
    condition_key: str | None = None,
    bit_diffusion: bool = False,
    loss_type: str = "mse",
    pre_loss_fn: Any = None,
    pos_enc: bool = False,
    requantize: bool = False,
    use_stats: bool = True,
) -> tuple[LatentDiffusionModel, VQVAEModel, EDMTransformer, _FakeLatentScheduler]:
    vae = _make_vqvae()
    denoise = _make_denoiser(n_latent=vae.n_latent, n_classes=n_classes)
    scheduler = _FakeLatentScheduler()
    model = LatentDiffusionModel(
        vae=vae,
        denoise_fn=denoise,
        conditioner=conditioner,
        scheduler=cast(Any, scheduler),
        condition_key=condition_key,
        bit_diffusion=bit_diffusion,
        loss_type=cast(Any, loss_type),
        pre_loss_fn=pre_loss_fn,
        pos_enc=pos_enc,
        requantize=requantize,
        use_stats=use_stats,
    )
    return model, vae, denoise, scheduler


def test_losses():
    b, n, c = 2, 16, 8
    q = torch.randn(b, n, c)
    k = v = torch.randn(b, n, c)
    index = torch.randperm(n)
    attn = Attention()

    x = attn(q)
    y = attn(q, k, v)
    y_perm = attn(q, k[:, index], v[:, index])
    loss = F.mse_loss(x, y)
    loss_perm = F.mse_loss(x, y_perm)

    assert torch.isclose(loss, loss_perm, atol=1e-6)
    assert not torch.isclose(F.mse_loss(q, k), F.mse_loss(q, k[:, index]), atol=1e-6)
    d_chamfer = cast(torch.Tensor, chamfer_distance(q, k)[0])
    d_chamfer_perm = cast(torch.Tensor, chamfer_distance(q, k[:, index])[0])
    assert torch.isclose(d_chamfer, d_chamfer_perm, atol=1e-6)
    assert not torch.isclose(
        F.cross_entropy(q.view(-1, q.size(-1)), k.argmax(-1).view(-1)),
        F.cross_entropy(q.view(-1, q.size(-1)), k[:, index].argmax(-1).view(-1)),
        atol=1e-6,
    )


def test_bit_diffusion():
    n = 14
    x = torch.randint(2**n, (32, 512))
    x_b = int2bit(x.long(), n)
    x_f = 2 * x_b - 1
    x_t = bit2int(x_f.half() > 0)
    assert torch.equal(x, x_t)


def test_noise_schedules_diffusers():
    if DiscreteStateScheduler is None:
        pytest.skip("DiscreteStateScheduler is not available in this diffusers version")

    linear = DDPMScheduler(beta_schedule="linear")
    scaled = DDPMScheduler(beta_schedule="scaled_linear")
    cosine = DDPMScheduler(beta_schedule="squaredcos_cap_v2")
    cosine2 = DiscreteStateScheduler(beta_schedule="squaredcos_cap_v2")
    sigmoid = DDPMScheduler(beta_schedule="sigmoid")

    schedulers = [linear, scaled, cosine, cosine2, sigmoid]
    for scheduler in schedulers:
        alphas_cumprod = np.asarray(scheduler.alphas_cumprod)
        assert alphas_cumprod.shape[0] == _num_train_timesteps(scheduler)
        assert np.all(np.isfinite(alphas_cumprod))
        assert np.all(np.diff(alphas_cumprod) <= 1e-8)
        assert 0 <= alphas_cumprod[-1] <= alphas_cumprod[0] <= 1


def test_edm_scheduler_preconditioning_and_step_outputs():
    scheduler = EDMScheduler(num_steps=4, p_mean=-1.2, p_std=1.2, sigma_data=0.5, sigma_min=0.002, sigma_max=1.0)
    timesteps = torch.tensor([0.0, 1.0], dtype=torch.float32)
    sigmas = scheduler.precondition_timesteps(timesteps)

    expected_sigmas = torch.exp(timesteps * 1.2 - 1.2)
    assert torch.allclose(sigmas, expected_sigmas)
    assert torch.allclose(scheduler.precondition_noise(sigmas), sigmas.log() / 4)

    samples = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    sigma_tensor = sigmas[:1].view(1, 1)
    expected_inputs = samples / torch.sqrt(torch.tensor(0.25, dtype=samples.dtype) + sigma_tensor**2)
    assert torch.allclose(scheduler.precondition_inputs(samples, sigma_tensor), expected_inputs)

    model_output = torch.ones_like(samples)
    c_skip = 0.25 / (sigma_tensor**2 + 0.25)
    c_out = sigma_tensor * 0.5 / torch.sqrt(sigma_tensor**2 + 0.25)
    expected_outputs = c_skip * samples + c_out * model_output
    assert torch.allclose(scheduler.precondition_outputs(samples, model_output, sigma_tensor), expected_outputs)

    loss = torch.ones_like(samples)
    expected_weight = (sigma_tensor**2 + 0.25) / (sigma_tensor * 0.5) ** 2
    assert torch.allclose(scheduler.precondition_loss(loss, sigma_tensor), expected_weight)
    assert torch.allclose(
        scheduler.add_noise(samples, torch.full_like(samples, 2.0), sigma_tensor),
        samples + 2.0 * sigma_tensor,
    )

    step_output = cast(Any, scheduler.step(model_output, sigma_tensor, samples))
    assert torch.equal(step_output.prev_sample, samples)
    assert torch.allclose(step_output.pred_original_sample, expected_outputs)

    prev_sample_only = scheduler.step(model_output, sigma_tensor, samples, return_dict=False)
    assert isinstance(prev_sample_only, tuple)
    assert torch.equal(prev_sample_only[0], samples)


def test_edm_sampler_respects_sigma_bounds_and_can_return_intermediates():
    net = _FakeEDMNet()
    cond = torch.tensor([3])
    randn_like_calls: list[Tensor] = []

    def _zero_noise(samples: Tensor) -> Tensor:
        randn_like_calls.append(samples.detach().clone())
        return torch.zeros_like(samples)

    result, intermediates = cast(
        tuple[Tensor, list[Tensor]],
        edm_sampler(
            net,
            latents=torch.ones((1, 2)),
            cond=cond,
            randn_like=_zero_noise,
            num_steps=3,
            sigma_min=0.01,
            sigma_max=10.0,
            s_churn=0.5,
            progress=False,
            return_intermediates=True,
        ),
    )

    assert torch.isfinite(result).all()
    assert len(intermediates) == 3
    assert len(net.calls) == 5
    assert randn_like_calls and randn_like_calls[0].dtype == torch.float64
    max_gamma = min(0.5 / 3, np.sqrt(2) - 1)
    assert float(net.calls[0]["timestep"]) <= net.sigma_max * (1 + max_gamma)
    assert float(net.calls[0]["timestep"]) >= net.sigma_min
    assert torch.equal(cast(Tensor, net.calls[0]["cond"]), cond)


def test_edm_sampler_returns_tensor_without_intermediates():
    net = _FakeEDMNet()
    result = cast(Tensor, edm_sampler(net, latents=torch.ones((1, 1)), num_steps=2, progress=False))

    assert isinstance(result, Tensor)
    assert result.shape == (1, 1)


def test_latent_diffusion_rejects_invalid_configuration() -> None:
    vae = _make_vqvae()
    denoise = _make_denoiser(n_latent=vae.n_latent)

    with pytest.raises(AssertionError, match="CE loss requires bit diffusion"):
        LatentDiffusionModel(vae=vae, denoise_fn=denoise, loss_type="ce")

    with pytest.raises(ValueError, match="Unsupported pre_loss_fn"):
        LatentDiffusionModel(vae=vae, denoise_fn=denoise, pre_loss_fn=cast(Any, "invalid"))

    with pytest.raises(AssertionError, match="sigmoid"):
        LatentDiffusionModel(vae=vae, denoise_fn=denoise, bit_diffusion=True, loss_type="ce", pre_loss_fn="sigmoid")


def test_latent_diffusion_init_setup_and_state_dict_respect_freeze_and_requantize() -> None:
    conditioner = _FakeConditionerModel()
    model, vae, denoise, _scheduler = _make_latent_diffusion_model(
        conditioner=conditioner,
        condition_key="cond",
        requantize=True,
        pos_enc=True,
    )

    assert model.requantize is not None
    assert model.pos_enc is not None
    assert all(not parameter.requires_grad for parameter in vae.parameters())
    assert all(not parameter.requires_grad for parameter in conditioner.parameters())

    model.setup(stage="fit")

    assert model.stats is not None
    assert torch.equal(cast(Tensor, model.stats["mean"]), cast(Tensor, cast(dict[str, Tensor], vae.stats)["mean"]))
    assert cast(Any, vae).setup_calls >= 1
    assert conditioner.setup_calls >= 1
    assert cast(Any, denoise).setup_calls >= 1

    state = model.state_dict()
    assert not any(key.startswith("_vae.") for key in state)
    assert not any(key.startswith("_conditioner.") for key in state)


def test_latent_diffusion_encode_decode_and_get_conditioning_paths() -> None:
    conditioner = _FakeConditionerModel()
    model, vae, _denoise, _scheduler = _make_latent_diffusion_model(conditioner=conditioner)
    model.setup()
    n_queries = int(cast(Any, vae).n_queries)

    inputs = torch.full((2, n_queries, model.n_latent), 5.0)
    latents = model.encode(inputs)
    mean = cast(Tensor, cast(dict[str, Tensor], vae.stats)["mean"])
    std = cast(Tensor, cast(dict[str, Tensor], vae.stats)["std"])
    expected_quantized = inputs + 1.0
    expected_latents = (expected_quantized - mean) * (model.sigma_data / std)

    assert torch.allclose(latents, expected_latents.float())
    assert torch.allclose(model.decode(latents), expected_quantized)

    conditioning = torch.ones((2, n_queries, model.n_latent))
    assert torch.allclose(cast(Tensor, model.get_conditioning(conditioning)), conditioning + 3.0)

    callable_model, _callable_vae, _callable_denoise, _callable_scheduler = _make_latent_diffusion_model(
        conditioner=cast(Any, lambda x, **kwargs: x + 7.0)
    )
    assert torch.allclose(cast(Tensor, callable_model.get_conditioning(conditioning)), conditioning + 7.0)


def test_latent_diffusion_forward_and_train_val_step_use_category_and_positional_encoding() -> None:
    model, vae, denoise, scheduler = _make_latent_diffusion_model(n_classes=5, pos_enc=True)
    cast(nn.Embedding, model.pos_enc).weight.data.fill_(0.25)
    n_queries = int(cast(Any, vae).n_queries)
    latents = torch.zeros((2, n_queries, model.n_latent))

    output = model.forward(noisy_latents=latents, sigmas=None, conditioning=None)

    assert output.shape == latents.shape
    assert cast(Any, denoise).calls[0]["cond"] is not None
    assert cast(Tensor, cast(Any, denoise).calls[0]["cond"]).shape == (2,)
    assert scheduler.precondition_inputs_calls

    denoised, sigmas = model._train_val_step(latents)
    assert denoised.shape == latents.shape
    assert sigmas.shape == (latents.size(0), 1, 1)
    assert scheduler.precondition_timesteps_calls
    assert scheduler.add_noise_calls


def test_latent_diffusion_predict_evaluate_and_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    conditioner = _FakeConditionerModel()
    model, vae, _denoise, _scheduler = _make_latent_diffusion_model(
        conditioner=conditioner,
        condition_key="cond",
        requantize=True,
        pos_enc=True,
    )
    cast(nn.Embedding, model.pos_enc).weight.data.fill_(0.1)
    n_queries = int(cast(Any, vae).n_queries)
    data = {
        "inputs": torch.ones((2, n_queries, model.n_latent)),
        "points": torch.zeros((2, n_queries, 3)),
        "points.occ": torch.zeros((2, n_queries)),
        "cond": torch.ones((2, n_queries, model.n_latent)),
    }

    logits, loss = cast(tuple[Tensor, Tensor], model.predict(data, return_loss=True, points_batch_size=7))

    assert logits.shape == (2, n_queries)
    assert loss.shape == (2, vae.n_queries, model.n_latent)
    assert cast(Any, vae).predict_calls[-1]["points_batch_size"] == 7
    assert "bce_loss" in cast(dict[str, tuple[Any, int]], model.get_log())

    delegated: list[dict[str, Any]] = []

    def _fake_multi_eval(self: Any, payload: dict[str, Tensor], **kwargs: Any) -> dict[str, float]:
        delegated.append({"payload": dict(payload), "kwargs": kwargs})
        return {"val/delegated": 1.0}

    monkeypatch.setattr(latent_module.MultiEvalMixin, "evaluate", _fake_multi_eval)
    result = model.evaluate({"logits": logits, "loss": loss, "points.occ": data["points.occ"]}, threshold=0.4)

    assert result == {"val/delegated": 1.0}
    assert delegated[0]["kwargs"] == {"threshold": 0.4}

    sampler_calls: list[dict[str, Any]] = []

    def _fake_sampler(
        net: nn.Module,
        latents: Tensor,
        cond: Tensor | None = None,
        num_steps: int = 18,
        progress: bool = True,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        sampler_calls.append(
            {
                "net": net,
                "latents": latents.detach().clone(),
                "cond": None if cond is None else cond.detach().clone(),
                "num_steps": num_steps,
                "progress": progress,
                "return_intermediates": return_intermediates,
            }
        )
        out = latents + 4.0
        if return_intermediates:
            return out, [latents + 1.0, latents + 2.0]
        return out

    monkeypatch.setattr(latent_module, "edm_sampler", _fake_sampler)
    generated_logits, intermediates = cast(
        tuple[Tensor, list[Tensor]],
        model.generate(
            points=cast(Tensor, data["points"]),
            conditioning=cast(Tensor, data["cond"]),
            points_batch_size=5,
            num_steps=6,
            progress=False,
            return_intermediates=True,
        ),
    )

    assert generated_logits.shape == (2, n_queries)
    assert len(intermediates) == 2
    assert sampler_calls[0]["num_steps"] == 6
    assert sampler_calls[0]["progress"] is False
    assert sampler_calls[0]["return_intermediates"] is True
    assert torch.allclose(cast(Tensor, sampler_calls[0]["cond"]), cast(Tensor, data["cond"]) + 3.0)


def test_latent_diffusion_apply_pre_loss_fn_and_loss_cover_bitdiff_and_aux_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bit_model, bit_vae, _bit_denoise, bit_scheduler = _make_latent_diffusion_model(
        bit_diffusion=True,
        loss_type="bce",
        pre_loss_fn="sigmoid",
        use_stats=False,
    )
    bit_queries = int(cast(Any, bit_vae).n_queries)
    bit_latents = torch.tensor([[[-0.5, 0.5, -0.5, 0.5]]], dtype=torch.float32)

    adjusted_latents, adjusted_denoised = bit_model._apply_pre_loss_fn(bit_latents, torch.zeros_like(bit_latents))
    assert torch.allclose(adjusted_latents, torch.tensor([[[0.0, 0.5, 0.0, 0.5]]]))
    assert torch.allclose(adjusted_denoised, torch.full_like(bit_latents, 0.25))

    monkeypatch.setattr(bit_model, "encode", lambda inputs, **kwargs: bit_latents)
    monkeypatch.setattr(
        bit_model,
        "_train_val_step",
        lambda latents, conditioning=None: (torch.zeros_like(latents), torch.ones(latents.shape[:-1])),
    )
    bit_loss = bit_model.loss(
        {
            "inputs": torch.ones((1, bit_queries, bit_model.n_latent)),
            "points": torch.zeros((1, bit_queries, 3)),
            "points.occ": torch.zeros((1, bit_queries)),
        }
    )

    assert torch.isfinite(bit_loss)
    assert bit_scheduler.precondition_loss_calls

    model, vae, _denoise, scheduler = _make_latent_diffusion_model(requantize=True)
    queries = int(cast(Any, vae).n_queries)
    monkeypatch.setattr(model, "encode", lambda inputs, **kwargs: torch.ones((1, queries, model.n_latent)))
    monkeypatch.setattr(
        model,
        "_train_val_step",
        lambda latents, conditioning=None: (torch.zeros_like(latents), torch.ones(latents.shape[:-1])),
    )
    loss = model.loss(
        {
            "inputs": torch.ones((1, queries, model.n_latent)),
            "points": torch.zeros((1, queries, 3)),
            "points.occ": torch.zeros((1, queries)),
        }
    )

    assert torch.isclose(loss, torch.tensor(2.75))
    assert scheduler.precondition_loss_calls


def get_x_0(resolution: int = 64, n_samples: int = 100_000, noise: float = 0.5, padding: float = 0.1):
    data, _color = make_swiss_roll(n_samples=n_samples, noise=noise)

    data -= data.min()
    data /= data.max()
    data -= 0.5
    data[:, 0] += np.abs((data[:, 0].min() + data[:, 0].max()) / 2)
    data[:, 2] += np.abs((data[:, 2].min() + data[:, 2].max()) / 2)

    voxelizer = Voxelizer(resolution, padding)
    grid, _index = voxelizer(data)

    x_0 = torch.from_numpy(grid)[None, None, ...].float()
    return x_0


def test_discrete_state():
    if DiscreteStateScheduler is None:
        pytest.skip("DiscreteStateScheduler is not available in this diffusers version")

    resolution = 64
    x_0 = get_x_0(resolution=resolution, n_samples=8_000)
    scheduler = DiscreteStateScheduler(beta_schedule="squaredcos_cap_v2")
    noise = torch.rand(*x_0.shape, scheduler.config.num_classes, dtype=x_0.dtype, device=x_0.device)
    num_train_timesteps = _num_train_timesteps(scheduler)

    for t in [0, 1, 10, num_train_timesteps - 1]:
        t_tensor = cast(torch.IntTensor, torch.tensor([t], dtype=torch.long))
        x_t = scheduler.add_noise(x_0, noise, t_tensor)
        assert x_t.shape == x_0.shape
        assert torch.isfinite(x_t).all()


def test_log_snr():
    linear = DDPMScheduler(beta_schedule="linear")
    alphas_cumprod = np.asarray(linear.alphas_cumprod, dtype=np.float64)
    eps = 1e-12
    alphas_cumprod = np.clip(alphas_cumprod, eps, 1 - eps)
    log_snr = np.log(alphas_cumprod / (1 - alphas_cumprod))

    assert log_snr.shape[0] == _num_train_timesteps(linear)
    assert np.all(np.isfinite(log_snr))
    assert np.all(np.diff(log_snr) <= 1e-8)


def test_vdm_scheduler():
    if VDMScheduler is None:
        pytest.skip("VDMScheduler is not available in this diffusers version")

    ddpm = DDPMScheduler(beta_schedule="sigmoid")
    vdm = VDMScheduler(beta_schedule="sigmoid", num_train_timesteps=_num_train_timesteps(ddpm))
    timesteps = torch.flip(vdm.timesteps, dims=(0,))

    sigma = vdm.sigmas(timesteps)
    alpha = torch.sqrt(vdm.alphas_cumprod(timesteps))
    assert sigma.shape == alpha.shape == timesteps.shape
    assert torch.isfinite(sigma).all()
    assert torch.isfinite(alpha).all()


def test_noise():
    diffusion = DDPMScheduler(beta_schedule="squaredcos_cap_v2")
    num_train_timesteps = _num_train_timesteps(diffusion)

    for size in [(2, 32), (2, 1, 32), (2, 32, 1), (2, 3, 32), (2, 32, 3)]:
        x = torch.randn(size).clamp(-1, 1)
        noise = torch.randn_like(x)
        t = cast(torch.IntTensor, torch.randint(0, num_train_timesteps, (x.size(0),), dtype=torch.long))
        x_t = diffusion.add_noise(x, noise, t)
        assert x_t.size() == noise.size()
        assert torch.isfinite(x_t).all()

    for size in [(2, 32, 32), (2, 1, 32, 32), (2, 3, 32, 32)]:
        x = torch.randn(size).clamp(-1, 1)
        noise = torch.randn_like(x)
        t = cast(torch.IntTensor, torch.randint(0, num_train_timesteps, (x.size(0),), dtype=torch.long))
        x_t = diffusion.add_noise(x, noise, t)
        assert x_t.size() == noise.size()
        assert torch.isfinite(x_t).all()

    for size in [(2, 32, 32, 32), (2, 1, 32, 32, 32), (2, 3, 32, 32, 32)]:
        x = torch.randn(size).clamp(-1, 1)
        noise = torch.randn_like(x)
        t = cast(torch.IntTensor, torch.randint(0, num_train_timesteps, (x.size(0),), dtype=torch.long))
        x_t = diffusion.add_noise(x, noise, t)
        assert x_t.size() == noise.size()
        assert torch.isfinite(x_t).all()


def test_antithetic_timestep_sampling_ddpm():
    model = DiffusersModel(scheduler="ddpm", num_train_timesteps=32, resolution=16)
    model.antithetic_time_sampling = True

    inputs = torch.zeros((4, 1, model.resolution**3))
    timesteps = model._get_timesteps(inputs)

    assert timesteps.shape == (inputs.size(0),)
    assert timesteps.dtype == torch.long
    assert int(timesteps.min()) >= 0
    assert int(timesteps.max()) < 32


def test_diffusers_model_scheduler_solver_and_helper_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diffusers_module,
        "UNet",
        lambda *args, **kwargs: _FakeDiffusersDenoiser(self_condition=bool(kwargs.get("self_condition", False))),
    )
    ddpm = DiffusersModel(scheduler="ddpm", num_train_timesteps=8, resolution=4)
    ddim = DiffusersModel(scheduler="ddim", num_train_timesteps=8, resolution=4)
    dpmpp = DiffusersModel(scheduler="dpm++", num_train_timesteps=8, resolution=4)
    edm = DiffusersModel(scheduler="edm", num_train_timesteps=8, resolution=4)
    edmpp = DiffusersModel(scheduler="edm++", num_train_timesteps=8, resolution=4)
    karras = DiffusersModel(scheduler="karras", num_train_timesteps=None, resolution=4)

    assert isinstance(ddpm.scheduler, DDPMScheduler)
    assert ddpm.solver is ddpm.scheduler
    assert isinstance(ddim.solver, DDIMScheduler)
    assert isinstance(dpmpp.solver, DPMSolverMultistepScheduler)
    assert isinstance(edm.scheduler, EDMEulerScheduler)
    assert isinstance(edmpp.solver, EDMDPMSolverMultistepScheduler)
    assert isinstance(karras.scheduler, EDMScheduler)

    with pytest.raises(ValueError, match="Invalid scheduler type"):
        len(DiffusersModel(scheduler="bogus", num_train_timesteps=8, resolution=4))

    coerced = DiffusersModel(
        scheduler="ddpm",
        num_train_timesteps=8,
        resolution=4,
        loss="bce",
        prediction_type="epsilon",
    )
    assert coerced.prediction_type == "sample"
    assert coerced.loss_type == "bce"

    with pytest.raises(AssertionError, match="Truncation only supported"):
        DiffusersModel(scheduler="ddpm", num_train_timesteps=8, resolution=4, truncate_sample=True)

    self_cond_model = DiffusersModel(scheduler="ddpm", num_train_timesteps=8, resolution=4, self_condition=True)
    self_cond_model.train()
    monkeypatch.setattr(diffusers_module, "random", lambda: 0.25)
    assert self_cond_model.self_condition is False
    assert self_cond_model.self_condition is True
    self_cond_model.eval()
    assert self_cond_model.self_condition is True

    sample = torch.zeros((1, 3))
    mask_points = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]])
    points = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    masked_sample, mask, augmented_points = ddpm._mask_sample(sample.clone(), mask_points, points.clone())
    assert masked_sample.shape == (1, 5)
    assert mask.shape == (1, 5)
    assert augmented_points.shape == (1, 5, 3)

    extracted_inputs, extracted_points, extracted_mask = ddpm._extract_data(
        {
            "inputs": mask_points,
            "points": points,
            "points.occ": torch.tensor([[0.0, 1.0, 0.0]]),
        }
    )
    assert extracted_inputs.shape == (1, 1, 5)
    assert extracted_points.shape == (1, 5, 3)
    assert extracted_mask is not None and extracted_mask.shape == (1, 5)

    ddpm.antithetic_time_sampling = True
    antithetic = ddpm._get_timesteps(torch.zeros((4, 1, ddpm.resolution**3)))
    assert antithetic.shape == (4,)
    assert antithetic.dtype == torch.long

    karras.antithetic_time_sampling = True
    with pytest.raises(ValueError, match="Antithetic time sampling not supported for EDM"):
        karras._get_timesteps(torch.zeros((4, 1, karras.resolution**3)))

    edm_continuous = DiffusersModel(scheduler="karras", num_train_timesteps=None, resolution=4)
    edm_times = edm_continuous._get_timesteps(torch.zeros((3, 1, edm_continuous.resolution**3)))
    assert edm_times.shape == (3,)
    assert edm_times.dtype == torch.float32

    assert ddpm._check_inputs(points=torch.zeros((2, 7, 3))) == (2, 7)
    assert ddpm._check_inputs(inputs=torch.zeros((1, 4, 4, 4)), points=torch.zeros((1, 64, 3))) == (1, 64)
    with pytest.raises(ValueError, match="Invalid resolution"):
        ddpm._check_inputs(inputs=torch.zeros((1, 3, 3, 3)))
    with pytest.raises(ValueError, match="Points shape"):
        ddpm._check_inputs(points=torch.zeros((1, 7, 2)))

    stopped = DiffusersModel(scheduler="ddpm", num_train_timesteps=8, num_inference_steps=5, stop_steps=2, resolution=4)
    timesteps = stopped._get_gen_timesteps()
    assert len(timesteps) == 2
    assert stopped._get_gen_noise(2, 7).shape == (2, 1, 7)


def test_diffusers_model_predict_generate_evaluate_and_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diffusers_module,
        "UNet",
        lambda *args, **kwargs: _FakeDiffusersDenoiser(self_condition=bool(kwargs.get("self_condition", False))),
    )
    model = DiffusersModel(
        scheduler="ddpm",
        num_train_timesteps=8,
        num_inference_steps=3,
        num_eval_steps=1,
        resolution=4,
        min_snr_gamma=0.5,
    )
    fake_denoiser = cast(_FakeDiffusersDenoiser, model.denoise_fn)

    data = {
        "inputs": torch.tensor([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]], dtype=torch.float32),
        "points": torch.tensor(
            [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]], dtype=torch.float32
        ),
        "points.occ": torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
    }

    logits, loss = cast(tuple[Tensor, Tensor], model.predict(data, return_loss=True))
    assert logits.shape == (1, 1, 4)
    assert loss.shape == (1, 4)
    assert fake_denoiser.calls

    delegated: list[dict[str, Any]] = []

    def _fake_multi_eval(self: Any, payload: dict[str, Tensor], **kwargs: Any) -> dict[str, float]:
        delegated.append({"payload": dict(payload), "kwargs": kwargs})
        return {"eval/delegated": 1.0}

    monkeypatch.setattr(diffusers_module.MultiEvalMixin, "evaluate", _fake_multi_eval)
    result = model.evaluate(dict(data), threshold=0.4)
    assert result == {"eval/delegated": 1.0}
    assert delegated[0]["kwargs"] == {"threshold": 0.4}

    generated = model.generate(points=data["points"], points_batch_size=2)
    assert generated.shape == (1, data["points"].size(1))

    training_loss = model.loss(data)
    assert torch.isfinite(training_loss)


def test_diffusers_model_predict_self_condition_and_sample_prediction_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diffusers_module,
        "UNet",
        lambda *args, **kwargs: _FakeDiffusersDenoiser(self_condition=bool(kwargs.get("self_condition", False))),
    )
    model = DiffusersModel(
        scheduler="ddpm",
        num_train_timesteps=8,
        num_inference_steps=3,
        resolution=4,
        prediction_type="sample",
        loss="bce",
        self_condition=True,
        self_cond_on_prev_step=True,
    )
    model.train()

    monkeypatch_target = cast(Any, model)
    predict_step_calls: list[dict[str, Tensor | None]] = []

    def _fake_predict_step(
        self: Any,
        x: tuple[Tensor, Tensor],
        timesteps: Tensor,
        noise: Tensor,
        x_cond_mask: Tensor | None = None,
        x_self_cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del noise, x_cond_mask, x_self_cond
        predict_step_calls.append({"timesteps": timesteps.detach().clone()})
        predictions = torch.ones_like(x[0], dtype=torch.float32)
        noisy = torch.full_like(predictions, 2.0)
        return predictions, noisy

    monkeypatch_target._predict_step = MethodType(_fake_predict_step, monkeypatch_target)
    prev_predictions = model._predict_self_cond(
        (torch.zeros((2, 1, 4)), torch.zeros((2, 4, 3))),
        torch.tensor([0, 3]),
        torch.zeros((2, 1, 4)),
    )

    assert torch.equal(cast(Tensor, predict_step_calls[0]["timesteps"]), torch.tensor([0, 2]))
    assert torch.allclose(prev_predictions[0], torch.zeros_like(prev_predictions[0]))
    assert torch.allclose(prev_predictions[1], torch.ones_like(prev_predictions[1]))

    sample_data = {
        "inputs": torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
        "points": torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]], dtype=torch.float32),
        "points.occ": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    }
    logits, loss = cast(tuple[Tensor, Tensor], model.predict(sample_data, return_loss=True))
    assert logits.shape == (1, 1, 2)
    assert loss.shape == (1, 2)

    generated = model.generate(points=sample_data["points"])
    assert generated.shape == (1, sample_data["points"].size(1))


def test_unet_cpu_self_condition_path():
    model = UNet(dim=16, dim_mults=(1, 2), channels=1, resolution=8, self_condition=True)

    x = torch.zeros((1, 1, model.resolution**3))
    p = torch.randn((1, model.resolution**3, 3))
    t = torch.randint(0, 1000, (1,))

    with torch.no_grad():
        self_cond = model((x, p), t)
        output = model((x, p), t, self_cond)

    assert self_cond.shape == output.shape == x.shape
    assert torch.isfinite(self_cond).all()
    assert torch.isfinite(output).all()


def test_unet_rejects_self_condition_when_disabled():
    model = UNet(dim=16, dim_mults=(1, 2), channels=1, resolution=8, self_condition=False)

    x = torch.zeros((1, 1, model.resolution**3))
    p = torch.randn((1, model.resolution**3, 3))
    t = torch.randint(0, 1000, (1,))

    with pytest.raises(ValueError, match="self conditioning"):
        model((x, p), t, x_self_cond=torch.zeros_like(x))


class TestDiffusionModel:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_unet(self):
        model = UNet(dim=16, dim_mults=(1, 2), channels=1, resolution=16, self_condition=True).cuda()

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        o = torch.zeros((1, 1, model.resolution**3), device=device, dtype=dtype)
        p = torch.randn((1, model.resolution**3, 3), device=device, dtype=dtype)
        t = torch.randint(0, 1000, (1,), device=device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            self_cond = model((o, p), t)
            model((o, p), t, self_cond)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_diffusion_model(self):
        model = DiffusersModel(resolution=16).cuda()

        o = torch.zeros((1, 1, model.resolution**3), device=model.device, dtype=model.dtype)
        p = torch.randn((1, model.resolution**3, 3), device=model.device, dtype=model.dtype)
        t = torch.randint(0, len(model), (1,), device=model.device)

        with torch.autocast(device_type=model.device.type, dtype=torch.float16):
            model((o, p), t)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss(self):
        model = DiffusersModel(scheduler="vdm", num_train_timesteps=0, prediction_type="sample", resolution=16).cuda()

        i = torch.randint(0, 1, (1, model.resolution**3), device=model.device).bool()
        o = torch.randint(0, 1, (1, model.resolution**3), device=model.device).bool()
        p = torch.randn((1, model.resolution**3, 3), device=model.device)

        with torch.autocast(device_type=model.device.type, dtype=torch.float16):
            model.loss({"inputs": i, "points": p, "points.occ": o})
