from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import pytest
import torch
from torch import Tensor, nn

from ..src.diffusion import grid as grid_module
from ..src.diffusion.grid import GridDiffusionModel


class _FakeDenoiseFn(nn.Module):
    def __init__(self, *, ndim: int, channels: int) -> None:
        super().__init__()
        self.ndim = ndim
        self.channels = channels
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.calls: list[dict[str, Any]] = []

    def forward(self, samples: Tensor, sigmas: Tensor, **kwargs: Any) -> Tensor:
        self.calls.append(
            {
                "samples": samples.detach().clone(),
                "sigmas": sigmas.detach().clone(),
                "kwargs": kwargs,
            }
        )
        return torch.full_like(samples, 2.0)


class _FakeScheduler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.precondition_timesteps_calls: list[Tensor] = []

    def precondition_timesteps(self, timesteps: Tensor) -> Tensor:
        self.precondition_timesteps_calls.append(timesteps.detach().clone())
        return timesteps.abs() + 0.25

    def precondition_inputs(self, noisy_samples: Tensor, sigmas: Tensor) -> Tensor:
        _ = sigmas
        return noisy_samples + 1.0

    def precondition_noise(self, sigmas: Tensor) -> Tensor:
        return sigmas + 2.0

    def precondition_outputs(self, noisy_samples: Tensor, model_output: Tensor, sigmas: Tensor) -> Tensor:
        return noisy_samples + model_output + sigmas

    def precondition_loss(self, loss: Tensor, sigmas: Tensor) -> Tensor:
        return loss + sigmas

    def __call__(self, denoise_fn: nn.Module, inputs: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        self.calls.append(
            {
                "denoise_fn": denoise_fn,
                "inputs": inputs.detach().clone(),
                "kwargs": kwargs,
            }
        )
        preds = inputs + 1.0
        sigmas = torch.full_like(inputs, 0.5)
        return preds, sigmas


def _make_model(*, ndim: int, channels: int = 1, resolution: int = 16) -> tuple[GridDiffusionModel, _FakeScheduler]:
    model = GridDiffusionModel(ndim=ndim, channels=channels, resolution=resolution)
    scheduler = _FakeScheduler()
    model_any = cast(Any, model)
    model_any.scheduler = scheduler
    model_any.denoise_fn = _FakeDenoiseFn(ndim=ndim, channels=channels)
    return model, scheduler


def test_get_inputs_handles_2d_and_3d_and_rejects_unknown_ndim() -> None:
    model_2d, _ = _make_model(ndim=2, channels=2)
    inputs = torch.tensor([[[[0.0, 1.0], [0.5, 0.25]], [[1.0, 0.0], [0.25, 0.5]]]])

    assert torch.equal(model_2d._get_inputs({"inputs": inputs}), 2 * inputs - 1)

    model_3d, _ = _make_model(ndim=3)
    occupancy = torch.tensor([[0, 1, 1, 0]], dtype=torch.int64)

    assert torch.equal(
        model_3d._get_inputs({"points.occ": occupancy}),
        torch.tensor([[[-1.0, 1.0, 1.0, -1.0]]]),
    )

    cast(_FakeDenoiseFn, model_3d.denoise_fn).ndim = 4
    with pytest.raises(ValueError, match="Unsupported ndim: 4"):
        model_3d._get_inputs({"points.occ": occupancy})


@pytest.mark.parametrize(
    ("ndim", "channels", "expected_shape"),
    [
        (2, 2, (1, 2, 16, 16)),
        (3, 1, (1, 1, 4096)),
    ],
)
def test_forward_initializes_samples_when_missing(
    ndim: int, channels: int, expected_shape: tuple[int, ...]
) -> None:
    model, scheduler = _make_model(ndim=ndim, channels=channels)
    denoise_fn = cast(_FakeDenoiseFn, model.denoise_fn)

    output = model.forward()

    assert output.shape == expected_shape
    assert scheduler.precondition_timesteps_calls
    assert denoise_fn.calls[0]["samples"].shape == expected_shape
    assert denoise_fn.calls[0]["sigmas"].shape == (expected_shape[0],)
    assert torch.isfinite(output).all()


def test_forward_uses_scheduler_preconditioning_and_rejects_negative_sigmas() -> None:
    model, _ = _make_model(ndim=2)
    denoise_fn = cast(_FakeDenoiseFn, model.denoise_fn)
    noisy_samples = torch.zeros((1, 1, 2, 2))
    sigmas = torch.full_like(noisy_samples, 0.5)

    output = model.forward(noisy_samples=noisy_samples, sigmas=sigmas, example=True)

    assert torch.equal(denoise_fn.calls[0]["samples"], torch.ones_like(noisy_samples))
    assert torch.allclose(denoise_fn.calls[0]["sigmas"], torch.full((4,), 2.5))
    assert denoise_fn.calls[0]["kwargs"] == {"example": True}
    assert torch.allclose(output, torch.full_like(noisy_samples, 2.5))

    with pytest.raises(ValueError, match="sigmas must be non-negative"):
        model.forward(noisy_samples=noisy_samples, sigmas=-torch.ones_like(noisy_samples))


def test_predict_returns_sigmas_for_2d_and_predictions_for_3d() -> None:
    model_2d, scheduler_2d = _make_model(ndim=2)
    data_2d = {"inputs": torch.tensor([[[[0.0, 1.0], [0.25, 0.75]]]])}
    expected_2d_inputs = 2 * data_2d["inputs"] - 1

    sigmas_2d, loss_2d = cast(tuple[Tensor, Tensor], model_2d.predict(data_2d, return_loss=True, stage="predict"))

    assert torch.equal(scheduler_2d.calls[0]["inputs"], expected_2d_inputs)
    assert scheduler_2d.calls[0]["kwargs"] == {"stage": "predict"}
    assert torch.allclose(sigmas_2d, torch.full_like(expected_2d_inputs, 0.5))
    assert torch.allclose(loss_2d, torch.full_like(expected_2d_inputs, 1.5))
    assert torch.allclose(cast(Tensor, model_2d.predict(data_2d)), expected_2d_inputs + 1.0)

    model_3d, _ = _make_model(ndim=3)
    data_3d = {"points.occ": torch.tensor([[0, 1, 1, 0]], dtype=torch.int64)}
    expected_3d_inputs = torch.tensor([[[-1.0, 1.0, 1.0, -1.0]]])

    preds_3d, loss_3d = cast(tuple[Tensor, Tensor], model_3d.predict(data_3d, return_loss=True))

    assert torch.equal(preds_3d, expected_3d_inputs + 1.0)
    assert torch.allclose(loss_3d, torch.full_like(expected_3d_inputs, 1.5))
    assert torch.isclose(model_3d.loss(data_3d), torch.tensor(1.5))


def test_evaluate_returns_mean_loss_for_2d(monkeypatch: pytest.MonkeyPatch) -> None:
    model, _ = _make_model(ndim=2)
    losses = torch.tensor([[[[1.0, 3.0]]]])

    def _fake_predict(data: dict[str, Tensor], return_loss: bool = False, **kwargs: Any) -> tuple[Tensor, Tensor]:
        _ = data, kwargs
        assert return_loss is True
        return torch.zeros_like(losses), losses

    monkeypatch.setattr(model, "predict", _fake_predict)

    result = model.evaluate({"inputs": torch.zeros((1, 1, 1, 2))}, prefix="metric/")

    assert result == {"metric/loss": 2.0}


def test_evaluate_delegates_to_multi_eval_for_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    model, _ = _make_model(ndim=3)
    super_calls: list[dict[str, Any]] = []

    def _fake_super_evaluate(self: Any, data: dict[str, Tensor], prefix: str = "val/", **kwargs: Any) -> dict[str, float]:
        super_calls.append({"data": dict(data), "prefix": prefix, "kwargs": kwargs})
        return {f"{prefix}delegated": 1.0}

    def _fail_predict(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("predict should not be called when logits and loss are already present")

    monkeypatch.setattr(grid_module.MultiEvalMixin, "evaluate", _fake_super_evaluate)
    monkeypatch.setattr(model, "predict", _fail_predict)

    data = {
        "logits": torch.ones((1, 1, 4)),
        "loss": torch.tensor([1.0]),
        "points.occ": torch.ones((1, 4), dtype=torch.int64),
    }
    result = model.evaluate(data, prefix="test/", threshold=0.4)

    assert result == {"test/delegated": 1.0}
    assert super_calls[0]["prefix"] == "test/"
    assert super_calls[0]["kwargs"] == {"threshold": 0.4}
    assert torch.equal(super_calls[0]["data"]["logits"], torch.ones((1, 4)))
    assert torch.equal(super_calls[0]["data"]["loss"], torch.tensor([1.0]))


def test_generate_uses_latent_shapes_for_2d_and_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler_calls: list[dict[str, Any]] = []
    images: list[Any] = []
    axes: list[str] = []
    shown: list[bool] = []

    def _fake_sampler(model: GridDiffusionModel, latents: Tensor, num_steps: int, progress: bool) -> Tensor:
        sampler_calls.append(
            {
                "model": model,
                "latents": latents.detach().clone(),
                "num_steps": num_steps,
                "progress": progress,
            }
        )
        return torch.full_like(latents, 0.5)

    monkeypatch.setattr(grid_module, "edm_sampler", _fake_sampler)
    monkeypatch.setattr(plt, "imshow", lambda image: images.append(image))
    monkeypatch.setattr(plt, "axis", lambda value: axes.append(str(value)))
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    model_2d, _ = _make_model(ndim=2, channels=3)
    out_2d = model_2d.generate(inputs=torch.zeros((4, 3, 2, 2)), show=True, num_steps=7, progress=False)

    assert sampler_calls[0]["model"] is model_2d
    assert sampler_calls[0]["latents"].shape == (4, 3, 16, 16)
    assert sampler_calls[0]["num_steps"] == 7
    assert sampler_calls[0]["progress"] is False
    assert out_2d.shape == (4, 3, 16, 16)
    assert images[0].shape == (32, 32, 3)
    assert axes == ["off"]
    assert shown == [True]

    model_3d, _ = _make_model(ndim=3)
    out_3d = model_3d.generate(points=torch.zeros((2, 5, 3)), num_steps=3, progress=False)

    assert sampler_calls[1]["model"] is model_3d
    assert sampler_calls[1]["latents"].shape == (2, 1, 5)
    assert out_3d.shape == (2, 1, 5)

    cast(_FakeDenoiseFn, model_3d.denoise_fn).ndim = 4
    with pytest.raises(ValueError, match="Unsupported ndim: 4"):
        model_3d.generate()
