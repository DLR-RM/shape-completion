from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from ..src import model as model_module
from ..src.model import Model


class ToyModel(Model):
    def __init__(self):
        super().__init__(reduction="sum")
        self.encoder = nn.Linear(3, 4)
        self.mask_head = nn.Linear(4, 2)
        self.frozen = nn.Parameter(torch.ones(5), requires_grad=False)
        self.setup_calls = 0
        self.teardown_calls = 0

    def setup(self, *args, **kwargs):
        self.setup_calls += 1
        super().setup(*args, **kwargs)

    def teardown(self, *args, **kwargs):
        self.teardown_calls += 1
        super().teardown(*args, **kwargs)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        hidden = self.encoder(x)
        return {"pred": self.mask_head(hidden)}

    def evaluate(self, x: Tensor) -> dict[str, float]:
        return {"mean": float(self.predict(x).mean().item())}

    def predict(self, x: Tensor) -> Tensor:
        return self.forward(x)["pred"]

    def loss(self, x: Tensor) -> Tensor:
        return self.predict(x).sum()


class SuperCallingModel(Model):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, *args, **kwargs) -> dict[str, Tensor]:
        return cast(Any, super()).forward(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> dict[str, float]:
        return cast(Any, super()).evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs) -> Tensor:
        return cast(Any, super()).predict(*args, **kwargs)

    def loss(self, *args, **kwargs) -> Tensor:
        return cast(Any, super()).loss(*args, **kwargs)


def test_model_logging_state_dict_and_properties(monkeypatch: pytest.MonkeyPatch):
    debug_messages: list[str] = []
    pretty_prints: list[object] = []

    monkeypatch.setattr(model_module.logger, "debug_level_2", debug_messages.append, raising=False)
    monkeypatch.setattr(model_module.logger, "isEnabledFor", lambda _level: True)
    monkeypatch.setattr(model_module, "pprint", pretty_prints.append)

    model = ToyModel()
    model.stats = {"loss": torch.tensor(1.0)}
    model.setup()
    model.teardown()
    assert debug_messages[:2] == ["ToyModel.setup()", "ToyModel stats: dict_keys(['loss'])"]
    assert pretty_prints == [model.stats]

    assert model.reduction == "sum"
    assert model.device.type == "cpu"
    assert model.dtype == torch.float32
    assert model.num_params == 31
    assert model.num_trainable_params == 26
    assert model.every_n_steps(12, 3) is True
    assert model.every_n_steps(13, 3) is False
    model.on_validation_epoch_end()

    assert model.orig_mod is model
    wrapper = ToyModel()
    model._orig_mod = wrapper
    assert model.orig_mod is wrapper

    model.train()
    model.log("loss", 1.0, ema=True)
    model.log("loss", 3.0, ema=0.5)
    model.log_dict({"iou": np.array([0.25, 0.75])}, ema=False)
    model.eval()
    model.log("train_only", 7.0, train_only=True)
    log_dict = model.get_log()
    assert isinstance(log_dict, dict)
    assert "train_only" not in log_dict
    assert model.get_log("loss") == 3.0
    assert model._ema_dict["loss"] == pytest.approx(2.0)

    state = model.state_dict()
    assert model.teardown_calls >= 2
    assert model.setup_calls >= 2
    assert "stats" in state
    assert torch.equal(state["stats"]["loss"], torch.tensor(2.0))

    model.clear_log("loss", ema=True)
    log_dict = model.get_log()
    assert isinstance(log_dict, dict)
    assert "loss" not in log_dict
    assert "loss" not in model._ema_dict
    model.clear_log(ema=True)
    assert model.get_log() == {}
    assert model._ema_dict == {}

    sample = torch.ones((2, 3), dtype=torch.float32)
    assert model.forward(sample)["pred"].shape == (2, 2)
    assert isinstance(model.evaluate(sample)["mean"], float)
    assert model.predict(sample).shape == (2, 2)
    assert torch.is_tensor(model.loss(sample))

    with pytest.raises(RuntimeError, match="only available for CUDA devices"):
        _ = model.nvml_handle


def test_model_state_dict_remapping_and_load(monkeypatch: pytest.MonkeyPatch):
    info_messages: list[str] = []
    monkeypatch.setattr(model_module.logger, "info", info_messages.append)

    model = ToyModel()
    fuzzy_input = {
        "encodr.weight": model.encoder.weight.detach().clone(),
        "encoder.bias": model.encoder.bias.detach().clone(),
        "mask_head.weight": torch.ones((3, 3), dtype=model.encoder.weight.dtype),
        "unused.weight": torch.ones((1, 1), dtype=model.encoder.weight.dtype),
    }
    remapped = model._fuzzy_remap_state_dict_keys(fuzzy_input, cutoff=0.8, drop_unmatched=False)
    assert "encoder.weight" in remapped
    assert "encoder.bias" in remapped
    assert "mask_head.weight" in remapped
    assert "unused.weight" in remapped
    assert any("Fuzzy-mapped" in msg for msg in info_messages)

    dropped = model._fuzzy_remap_state_dict_keys({"unused.weight": torch.ones((1, 1))}, drop_unmatched=True)
    assert dropped == {}

    suffix_input = {
        "encoder.weight": model.encoder.weight.detach().clone(),
        "encoder.bias": model.encoder.bias.detach().clone(),
        "mash_head.weight": model.mask_head.weight.detach().clone(),
        "mash_head.bias": model.mask_head.bias.detach().clone(),
    }
    resolved = model._resolve_missing_by_shape_suffix(suffix_input.copy(), min_ratio=0.5)
    assert "mask_head.weight" in resolved
    assert "mask_head.bias" in resolved
    assert any("Shape-suffix mapped" in msg for msg in info_messages)

    source = ToyModel()
    source._ema_dict["ema_loss"] = 1.5
    checkpoint = {
        "state_dict": {
            "model._orig_mod.encoder.weight": source.encoder.weight.detach().clone(),
            "model._orig_mod.encoder.bias": source.encoder.bias.detach().clone(),
            "model.mash_head.weight": source.mask_head.weight.detach().clone(),
            "model.mash_head.bias": source.mask_head.bias.detach().clone(),
            "model._orig_mod.frozen": source.frozen.detach().clone(),
            "stats": {"ema_loss": torch.tensor(1.5)},
        }
    }

    target = ToyModel()
    result = target.load_state_dict(checkpoint, strict=False, fuzzy_match=False, shape_suffix_match=True)
    assert isinstance(result, torch.nn.modules.module._IncompatibleKeys)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert torch.allclose(target.encoder.weight, source.encoder.weight)
    assert torch.allclose(target.mask_head.bias, source.mask_head.bias)
    assert target.stats is not None
    assert torch.equal(target.stats["ema_loss"], torch.tensor(1.5))
    assert target.setup_calls == 1
    assert target.teardown_calls == 1


def test_model_super_methods_raise_not_implemented():
    model = SuperCallingModel()

    with pytest.raises(NotImplementedError, match="forward method is not implemented"):
        model.forward(torch.ones(1))
    with pytest.raises(NotImplementedError, match="evaluate method is not implemented"):
        model.evaluate(torch.ones(1))
    with pytest.raises(NotImplementedError, match="predict method is not implemented"):
        model.predict(torch.ones(1))
    with pytest.raises(NotImplementedError, match="loss method is not implemented"):
        model.loss(torch.ones(1))
