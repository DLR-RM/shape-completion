from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from train.src import model as script


class _FrozenSubmodule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.eval()


class _FakeOrigModule(nn.Module):
    def __init__(self, include_frozen: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.load_calls: list[dict[str, Any]] = []
        if include_frozen:
            self._vae = _FrozenSubmodule()

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        _ = args, kwargs
        return {"weight": self.weight.detach().clone()}

    def load_state_dict(self, state_dict: dict[str, Any], *args: Any, strict: bool = True, **kwargs: Any) -> Any:
        self.load_calls.append({"state_dict": state_dict, "strict": strict})
        return SimpleNamespace()


class _FakeWrappedModel(nn.Module):
    def __init__(self, include_frozen: bool = False) -> None:
        super().__init__()
        self.orig_mod = _FakeOrigModule(include_frozen=include_frozen)
        self.extra = nn.Parameter(torch.tensor([2.0]))
        self.name = "fake_model"
        self.some_attr = "delegated"
        self.forward_result = torch.tensor([3.0])
        self.evaluate_result: dict[str, Any] = {"loss": torch.tensor(1.25), "precision": 0.5}
        self.log_values: dict[str, tuple[Any, int]] = {}
        self.loss_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.evaluate_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.clear_log_calls = 0
        self.begin_calls = 0

    def forward(self, **batch: Any) -> torch.Tensor:
        _ = batch
        return self.forward_result

    def loss(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        self.loss_calls.append((batch, kwargs))
        return torch.tensor(2.5)

    def evaluate(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.evaluate_calls.append((batch, kwargs))
        return self.evaluate_result

    def get_log(self) -> dict[str, tuple[Any, int]]:
        return dict(self.log_values)

    def clear_log(self) -> None:
        self.clear_log_calls += 1
        self.log_values = {}

    def on_validation_epoch_end(self) -> dict[str, Any]:
        return {"loss": torch.tensor(0.25), "f1": 0.75}

    def begin(self) -> None:
        self.begin_calls += 1


class _FakeEMA:
    def __init__(
        self,
        orig_mod: _FakeOrigModule,
        result: torch.Tensor | None = None,
        evaluate_result: dict[str, Any] | None = None,
    ) -> None:
        self.result = result if result is not None else torch.tensor([9.0])
        self.evaluate_result = evaluate_result if evaluate_result is not None else {"loss": torch.tensor(0.75)}
        self.calls: list[dict[str, Any]] = []
        self.update_calls: list[nn.Module] = []
        self.module = SimpleNamespace(orig_mod=orig_mod, evaluate=self.evaluate)

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        self.calls.append(kwargs)
        return self.result

    def update_parameters(self, model: nn.Module) -> None:
        self.update_calls.append(model)

    def evaluate(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        _ = batch, kwargs
        return self.evaluate_result


class _FakeExperiment:
    def __init__(self) -> None:
        self.hist_calls: list[tuple[str, np.ndarray, int]] = []

    def add_histogram(self, name: str, values: np.ndarray, step: int) -> None:
        self.hist_calls.append((name, np.asarray(values), step))


def _make_lit_model(wrapped: _FakeWrappedModel, **kwargs: Any) -> script.LitModel:
    return script.LitModel("fake", ".", cast(Any, wrapped), **kwargs)


def _attach_trainer(
    lit: script.LitModel,
    *,
    global_step: int = 0,
    log_every_n_steps: int = 1,
    estimated_stepping_batches: int = 8,
    experiment: _FakeExperiment | None = None,
) -> _FakeExperiment:
    exp = experiment if experiment is not None else _FakeExperiment()
    object.__setattr__(
        lit,
        "_trainer",
        SimpleNamespace(
        global_step=global_step,
        log_every_n_steps=log_every_n_steps,
        estimated_stepping_batches=estimated_stepping_batches,
        logger=SimpleNamespace(experiment=exp),
        ),
    )
    return exp


def test_state_forward_and_checkpoint_hooks_delegate_to_wrapped_model() -> None:
    wrapped = _FakeWrappedModel(include_frozen=True)
    ema_orig = _FakeOrigModule(include_frozen=True)
    ema = _FakeEMA(ema_orig, result=torch.tensor([9.0]))
    lit = _make_lit_model(wrapped)
    cast(Any, lit).ema_model = ema

    assert lit.some_attr == "delegated"
    assert torch.equal(lit.state_dict()["model.weight"], torch.tensor([1.0]))

    lit.load_state_dict({"weight": torch.tensor([2.0])})
    assert wrapped.orig_mod.load_calls[-1]["strict"] is False

    checkpoint: dict[str, Any] = {}
    lit.on_save_checkpoint(checkpoint)
    assert torch.equal(checkpoint["ema_state_dict"]["weight"], torch.tensor([1.0]))

    lit.on_load_checkpoint({"ema_state_dict": {"weight": torch.tensor([5.0])}})
    assert ema_orig.load_calls[-1]["strict"] is False

    wrapped.orig_mod.eval()
    output = lit.forward({"inputs": torch.ones((1, 1))})
    assert torch.equal(output, torch.tensor([9.0]))
    assert torch.equal(ema.calls[0]["inputs"], torch.ones((1, 1)))

    lit.on_before_zero_grad(cast(Any, None))
    assert ema.update_calls == [wrapped]


def test_training_and_validation_generic_paths_log_and_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped = _FakeWrappedModel()
    lit = _make_lit_model(
        wrapped,
        threshold=0.25,
        reduction="sum",
        points_batch_size=16,
        monitor="val/loss",
    )
    experiment = _attach_trainer(lit, estimated_stepping_batches=12)
    logged: list[tuple[str, Any]] = []
    warnings: list[str] = []

    monkeypatch.setattr(
        pl.LightningModule,
        "log",
        lambda self, name, value, *args, **kwargs: logged.append((name, value)),
    )
    monkeypatch.setattr(script.logger, "isEnabledFor", lambda level: True)
    monkeypatch.setattr(script.logger, "warning", lambda msg: warnings.append(str(msg)))

    batch = {"inputs": torch.zeros((2, 3))}
    wrapped.log_values = {
        "train/scalar": (np.array(1.5, dtype=np.float32), script.DEBUG_LEVEL_1),
        "train/hist": (torch.tensor([1.0, 2.0]), script.DEBUG_LEVEL_1),
        "train/bad": ({}, script.DEBUG_LEVEL_1),
    }
    loss = lit.training_step(batch, 0)

    assert torch.isclose(loss, torch.tensor(2.5))
    assert wrapped.loss_calls[0][1]["log_freq"] == 1
    assert wrapped.loss_calls[0][1]["total_steps"] == 12
    assert any(name == "train/loss" for name, _ in logged)
    assert any(name == "train/scalar" for name, _ in logged)
    assert experiment.hist_calls[0][0] == "train/hist"
    assert warnings == ["Cannot log train/bad of type <class 'dict'>"]
    assert wrapped.clear_log_calls == 1

    wrapped.orig_mod.eval()
    wrapped.log_values = {"val/aux": (torch.tensor(0.3), script.DEBUG_LEVEL_1)}
    result = lit.validation_step(batch, 0)

    assert float(result["val/loss"]) == pytest.approx(1.25)
    assert float(result["val/precision"]) == pytest.approx(0.5)
    assert float(result["val/aux"]) == pytest.approx(0.3)
    assert any(name == "val/loss" for name, _ in logged)
    assert any(name == "val/precision" for name, _ in logged)
    assert any(name == "val/aux" for name, _ in logged)
    assert wrapped.clear_log_calls == 2

    lit.on_validation_epoch_end()
    assert any(name == "val/f1" for name, _ in logged)
    lit.on_validation_end()


def test_backward_clipping_and_validation_end_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped = _FakeWrappedModel()
    lit = _make_lit_model(wrapped, hypergradients=True, monitor="val/f1")
    _attach_trainer(lit)
    backward_calls: list[dict[str, Any]] = []
    clip_calls: list[tuple[Any, Any, Any]] = []

    monkeypatch.setattr(
        pl.LightningModule,
        "backward",
        lambda self, loss, *args, **kwargs: backward_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        lit,
        "clip_gradients",
        lambda optimizer, gradient_clip_val, gradient_clip_algorithm: clip_calls.append(
            (optimizer, gradient_clip_val, gradient_clip_algorithm)
        ),
    )

    with pytest.raises(ValueError, match="Loss is not finite"):
        lit.on_before_backward(torch.tensor(float("nan")))

    lit.on_train_batch_start({}, 0)
    assert wrapped.begin_calls == 1

    lit.backward(torch.tensor(1.0, requires_grad=True))
    assert backward_calls[-1]["create_graph"] is True

    optimizer = SGD([wrapped.extra], lr=0.1)
    lit.configure_gradient_clipping(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
    assert clip_calls == [(optimizer, 0.5, "norm")]

    with pytest.raises(ValueError, match="Metric val/f1"):
        lit.on_validation_end()


def test_on_before_optimizer_step_logs_norms_and_histograms(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped = _FakeWrappedModel()
    lit = _make_lit_model(wrapped)
    experiment = _attach_trainer(lit)
    logged: list[str] = []

    monkeypatch.setattr(
        pl.LightningModule,
        "log",
        lambda self, name, value, *args, **kwargs: logged.append(name),
    )
    monkeypatch.setattr(
        script,
        "grad_norm",
        lambda model, norm_type=2: {
            "grad_2.0_norm_total": torch.tensor(3.0),
            "grad_2.0_norm/weight": torch.tensor(1.0),
        },
    )
    monkeypatch.setattr(
        script,
        "weight_norm",
        lambda model, norm_type=2: {
            "weight_2.0_norm_total": 5.0,
            "weight_2.0_norm/weight": 2.0,
        },
    )
    monkeypatch.setattr(script.logger, "isEnabledFor", lambda level: True)

    lit.on_before_optimizer_step(cast(Any, None))

    assert "train/grad_norm" in logged
    assert "train/weight_norm" in logged
    assert experiment.hist_calls[0][0] == "gradients/grad_hist"
    assert experiment.hist_calls[1][0] == "gradients/weight_hist"


def test_configure_optimizers_handles_scheduler_and_learning_rate_groups() -> None:
    wrapped = _FakeWrappedModel()
    optimizer = SGD(
        [{"params": [wrapped.orig_mod.weight], "lr": 0.02}, {"params": [wrapped.extra], "lr": 0.002}],
        lr=0.02,
    )
    lit = _make_lit_model(wrapped, optimizer=optimizer)

    configured = lit.configure_optimizers(lr=0.1)
    assert configured is optimizer
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.1, 0.01])

    wrapped_scheduler = _FakeWrappedModel()
    optimizer_scheduler = SGD(wrapped_scheduler.parameters(), lr=0.03)
    scheduler = StepLR(optimizer_scheduler, step_size=1)
    lit_scheduler = _make_lit_model(
        wrapped_scheduler,
        optimizer=optimizer_scheduler,
        scheduler=scheduler,
        interval="step",
        frequency=2,
        monitor="val/loss",
    )

    scheduler_config = cast(dict[str, Any], lit_scheduler.configure_optimizers())
    assert scheduler_config["optimizer"] is optimizer_scheduler
    assert scheduler_config["lr_scheduler"]["scheduler"] is scheduler
    assert scheduler_config["lr_scheduler"]["interval"] == "step"
    assert scheduler_config["lr_scheduler"]["frequency"] == 2
    assert scheduler_config["lr_scheduler"]["monitor"] == "val/loss"

    wrapped_auto = _FakeWrappedModel()
    lit_auto = _make_lit_model(wrapped_auto, learning_rate=0.05)
    auto_optimizer = cast(torch.optim.AdamW, lit_auto.configure_optimizers())
    assert auto_optimizer.param_groups[0]["lr"] == pytest.approx(0.05)

    wrapped_error = _FakeWrappedModel()
    optimizer_error = SGD(wrapped_error.parameters(), lr=0.03)
    scheduler_error = StepLR(optimizer_error, step_size=1)
    lit_error = _make_lit_model(
        wrapped_error,
        optimizer=optimizer_error,
        scheduler=scheduler_error,
        learning_rate=0.2,
    )

    with pytest.raises(NotImplementedError, match="Setting learning rate with scheduler"):
        lit_error.configure_optimizers()
