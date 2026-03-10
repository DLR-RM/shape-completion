import difflib
from abc import ABC, abstractmethod
from functools import cached_property
from logging import DEBUG, INFO
from pprint import pprint
from typing import Any

import numpy as np
import pynvml
import torch
from torch import Tensor, nn

from utils import DEBUG_LEVEL_2, setup_logger, to_tensor

logger = setup_logger(__name__)
LogDictValueTypes = int | float | np.ndarray
LogDictTypes = tuple[LogDictValueTypes, int | None]


def _log_debug_level_2(msg: str) -> None:
    debug_level_2 = getattr(logger, "debug_level_2", logger.debug)
    debug_level_2(msg)


class Model(nn.Module, ABC):
    def __init__(self, reduction: str | None = "mean"):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.type = self.__class__  # pyright: ignore[reportAttributeAccessIssue]
        self.reduction = reduction
        self.stats: dict[str, Tensor] | None = None
        self._log_dict: dict[str, LogDictTypes] = dict()
        self._ema_dict: dict[str, LogDictValueTypes] = dict()

    def setup(self, *args, **kwargs):
        _log_debug_level_2(f"{self.name}.setup()")
        if logger.isEnabledFor(DEBUG_LEVEL_2) and self.stats:
            _log_debug_level_2(f"{self.name} stats: {self.stats.keys()}")
            if logger.isEnabledFor(DEBUG):
                pprint(self.stats)

    def teardown(self, *args, **kwargs):
        _log_debug_level_2(f"{self.name}.teardown()")

    @property
    def orig_mod(self) -> "Model":
        return getattr(self, "_orig_mod", self)  # torch.compile fix

    def state_dict(self, *args, **kwargs):
        self.teardown(*args, **kwargs)

        _log_debug_level_2(f"{self.name}.state_dict()")
        state_dict = super().state_dict(*args, **kwargs)
        if self._ema_dict:
            state_dict["stats"] = {
                k: to_tensor(v, unsqueeze=False, device=self.device) for k, v in self._ema_dict.items()
            }

        self.setup(*args, **kwargs)
        return state_dict

    def _fuzzy_remap_state_dict_keys(
        self, incoming: dict[str, Any], cutoff: float = 0.84, drop_unmatched: bool = False
    ) -> dict[str, Any]:
        """
        Try to remap checkpoint keys to the closest model keys based on string similarity.
        Only remap when tensor shapes match and similarity >= cutoff.
        If drop_unmatched is False, unmatched keys are left as-is (PyTorch may flag them as unexpected
        when strict=True). If True, they are removed.
        """

        # Get current model's expected keys and their tensor shapes, without triggering our override.
        model_state = nn.Module.state_dict(self)  # base class state_dict
        expected_keys = set(model_state.keys())

        # We'll track which target keys are already taken to avoid many-to-one collisions.
        used_targets = set()

        remapped: dict[str, Any] = {}
        for src_key, value in incoming.items():
            # If exact match exists, keep it.
            if src_key in expected_keys and src_key not in used_targets:
                remapped[src_key] = value
                used_targets.add(src_key)
                continue

            # Search for the closest match among remaining candidates.
            candidates = [k for k in expected_keys if k not in used_targets]
            if not candidates:
                if not drop_unmatched:
                    remapped[src_key] = value
                continue

            match = difflib.get_close_matches(src_key, candidates, n=1, cutoff=cutoff)
            if match:
                tgt_key = match[0]
                # Shape check (buffers may not be tensors; be permissive there)
                try:
                    shapes_match = (
                        hasattr(value, "shape")
                        and hasattr(model_state[tgt_key], "shape")
                        and tuple(value.shape) == tuple(model_state[tgt_key].shape)
                    )
                except Exception:
                    shapes_match = True  # in doubt (e.g. non-tensor buffers), allow

                if shapes_match:
                    if src_key != tgt_key:
                        logger.info(f"[{self.name}] Fuzzy-mapped '{src_key}' -> '{tgt_key}'")
                    remapped[tgt_key] = value
                    used_targets.add(tgt_key)
                else:
                    # Shape mismatch: keep original (or drop)
                    if not drop_unmatched:
                        remapped[src_key] = value
            else:
                # No close match found
                if not drop_unmatched:
                    remapped[src_key] = value

        return remapped

    def _resolve_missing_by_shape_suffix(self, incoming: dict[str, Any], min_ratio: float = 0.5) -> dict[str, Any]:
        """
        For keys still missing after fuzzy remap, try to map unexpected keys to missing keys
        by:
            - identical parameter suffix (e.g., 'weight', 'bias')
            - exact tensor shape match
            - highest name similarity on the path without the suffix
        """

        model_state = nn.Module.state_dict(self)  # base class state_dict (no overrides)
        expected_keys = set(model_state.keys())
        incoming_keys = set(incoming.keys())

        # Only consider tensors
        def is_tensor_shape(x):
            return hasattr(x, "shape")

        missing = [k for k in expected_keys if k not in incoming_keys and is_tensor_shape(model_state[k])]
        extras = [k for k in incoming_keys if k not in expected_keys and is_tensor_shape(incoming[k])]

        if not missing or not extras:
            return incoming

        used_extra = set()

        for m in missing:
            m_suffix = m.split(".")[-1]  # e.g., weight/bias
            m_shape = tuple(model_state[m].shape)
            m_base = m.rsplit(".", 1)[0]  # path without suffix

            best_key = None
            best_ratio = -1.0

            for e in extras:
                if e in used_extra:
                    continue
                if e.split(".")[-1] != m_suffix:
                    continue
                v = incoming[e]
                if tuple(v.shape) != m_shape:
                    continue

                e_base = e.rsplit(".", 1)[0]
                ratio = difflib.SequenceMatcher(a=m_base, b=e_base).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key = e

            if best_key is not None and best_ratio >= min_ratio:
                incoming[m] = incoming.pop(best_key)
                used_extra.add(best_key)
                logger.info(f"[{self.name}] Shape-suffix mapped '{best_key}' -> '{m}' (sim={best_ratio:.2f})")

        return incoming

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        replace_keys: bool = True,
        fuzzy_match: bool = True,
        fuzzy_cutoff: float = 0.84,
        fuzzy_drop_unmatched: bool = False,
        shape_suffix_match: bool = True,
        shape_suffix_min_ratio: float = 0.5,
        *args,
        **kwargs,
    ):
        self.teardown(*args, **kwargs)

        if replace_keys:
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}  # Lightning fix
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}  # PyTorch compile fix

        # Pull out any stored stats before fuzzy mapping.
        self.stats = state_dict.pop("stats", None)

        # 1) Fuzzy remap for near-identical names
        if fuzzy_match:
            strict = kwargs.get("strict", True)
            state_dict = self._fuzzy_remap_state_dict_keys(
                state_dict, cutoff=fuzzy_cutoff, drop_unmatched=fuzzy_drop_unmatched and strict
            )

        # 2) Resolve remaining Missing/Unexpected by shape + suffix (e.g., points_head <-> mask_head)
        if shape_suffix_match:
            state_dict = self._resolve_missing_by_shape_suffix(state_dict, min_ratio=shape_suffix_min_ratio)

        _log_debug_level_2(f"{self.name}.load_state_dict()")
        result = super().load_state_dict(state_dict, *args, **kwargs)

        self.setup(*args, **kwargs)
        return result

    @staticmethod
    def every_n_steps(global_step: int, n_steps: int) -> bool:
        return global_step % n_steps == 0

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @cached_property
    def nvml_handle(self):
        if self.device.type != "cuda":
            raise RuntimeError("nvml_handle is only available for CUDA devices")
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in super().parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def forward(self, *args, **kwargs) -> dict[str, Tensor]:
        raise NotImplementedError(f"{self.name}.forward method is not implemented")

    @torch.no_grad()
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict[str, float]:
        raise NotImplementedError(f"{self.name}.evaluate method is not implemented")

    @torch.inference_mode()
    @abstractmethod
    def predict(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(f"{self.name}.predict method is not implemented")

    @abstractmethod
    def loss(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(f"{self.name}.loss method is not implemented")

    def on_validation_epoch_end(self, *args, **kwargs):
        pass

    def log(
        self,
        key: str,
        value: LogDictValueTypes,
        level: int = INFO,
        train_only: bool = False,
        ema: bool | float = False,
    ):
        if not train_only or self.training:
            self._log_dict[key] = (value, level)
            if ema:
                if key not in self._ema_dict:
                    self._ema_dict[key] = value
                else:
                    if not isinstance(ema, float):
                        ema = 0.99
                    self._ema_dict[key] = ema * value + (1 - ema) * self._ema_dict[key]

    def log_dict(
        self,
        log_dict: dict[str, LogDictValueTypes],
        level: int = INFO,
        train_only: bool = False,
        ema: bool | float = False,
    ):
        for key, value in log_dict.items():
            self.log(key, value, level=level, train_only=train_only, ema=ema)

    def get_log(self, key: str | None = None) -> dict[str, LogDictTypes] | LogDictValueTypes:
        if key is None:
            return self._log_dict
        return self._log_dict[key][0]

    def clear_log(self, key: str | None = None, ema: bool = False):
        if key is None:
            self._log_dict.clear()
            if ema:
                self._ema_dict.clear()
        else:
            self._log_dict.pop(key, None)
            if ema:
                self._ema_dict.pop(key, None)
