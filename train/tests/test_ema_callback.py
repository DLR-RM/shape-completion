from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from lightning.fabric.utilities.exceptions import MisconfigurationException

from ..src.callbacks.ema import EMACallback, EMAOptimizer


def _build_model_and_optimizer() -> tuple[torch.nn.Linear, EMAOptimizer]:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float32))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    ema_optimizer = EMAOptimizer(
        optimizer=optimizer,
        device=torch.device("cpu"),
        decay=0.5,
        every_n_steps=1,
        current_step=0,
    )
    return model, ema_optimizer


def test_ema_callback_rejects_invalid_decay() -> None:
    with pytest.raises(MisconfigurationException):
        EMACallback(decay=1.1)


def test_ema_optimizer_step_updates_ema_params() -> None:
    model, optimizer = _build_model_and_optimizer()
    for parameter in model.parameters():
        parameter.grad = parameter.detach().clone()

    optimizer.step()
    optimizer.join()

    expected_weights = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    expected_ema = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
    assert torch.allclose(model.weight.detach(), expected_weights)
    assert torch.allclose(optimizer.ema_params[0], expected_ema)
    assert optimizer.current_step == 1


def test_swap_ema_weights_context_swaps_and_restores() -> None:
    model, optimizer = _build_model_and_optimizer()
    for parameter in model.parameters():
        parameter.grad = parameter.detach().clone()

    optimizer.step()
    optimizer.join()
    expected_model_weights = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    expected_ema_weights = torch.tensor([[0.5, -0.5]], dtype=torch.float32)

    with optimizer.swap_ema_weights():
        assert torch.allclose(model.weight.detach(), expected_ema_weights)
    assert torch.allclose(model.weight.detach(), expected_model_weights)


def test_on_fit_start_wraps_optimizers() -> None:
    model, _ = _build_model_and_optimizer()
    trainer = cast(Any, SimpleNamespace(optimizers=[torch.optim.SGD(model.parameters(), lr=1.0)], global_step=0))
    callback = EMACallback(decay=0.9)

    callback.on_fit_start(trainer, cast(Any, SimpleNamespace(device=torch.device("cpu"))))

    assert len(trainer.optimizers) == 1
    assert isinstance(trainer.optimizers[0], EMAOptimizer)


def test_save_original_optimizer_state_context_toggles_flag() -> None:
    _, ema_optimizer = _build_model_and_optimizer()
    trainer = cast(Any, SimpleNamespace(optimizers=[ema_optimizer]))
    callback = EMACallback(decay=0.9)

    assert ema_optimizer.save_original_optimizer_state is False
    with callback.save_original_optimizer_state(trainer):
        assert ema_optimizer.save_original_optimizer_state is True
    assert ema_optimizer.save_original_optimizer_state is False
