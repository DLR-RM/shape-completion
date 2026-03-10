from __future__ import annotations

import importlib
import logging as py_logging
import sys
import warnings
from types import SimpleNamespace
from typing import Any, cast

import pytest

from utils.src import runtime as runtime_module

logging_module = importlib.import_module("utils.src.logging")


def test_suppress_known_optional_dependency_warnings_updates_filters_and_cflags(monkeypatch: pytest.MonkeyPatch) -> None:
    filter_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(warnings, "filterwarnings", lambda action, message: filter_calls.append((action, message)))
    monkeypatch.setenv("CFLAGS", "-Wall -Wno-cpp")

    runtime_module.suppress_known_optional_dependency_warnings()
    runtime_module.suppress_known_optional_dependency_warnings()

    assert filter_calls == [
        ("ignore", r"xFormers is not available \(SwiGLU\)"),
        ("ignore", r"xFormers is not available \(Attention\)"),
        ("ignore", r"xFormers is not available \(Block\)"),
        ("ignore", r"xFormers is not available \(SwiGLU\)"),
        ("ignore", r"xFormers is not available \(Attention\)"),
        ("ignore", r"xFormers is not available \(Block\)"),
    ]
    assert logging_module is not None  # keep import used for pyright
    assert runtime_module.os.environ["CFLAGS"].count("-Wno-cpp") == 1
    assert runtime_module.os.environ["CFLAGS"].count("-Wno-macro-redefined") == 1


def test_runtime_dependency_helpers_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    spec_map = {
        "available.mod": object(),
        "broken.mod": object(),
        "xformers": object(),
        "bpy": object(),
        "blenderproc": None,
        "transforms3d": None,
        "torch_cluster": None,
    }

    monkeypatch.setattr(runtime_module.importlib.util, "find_spec", lambda name: spec_map.get(name))

    def _fake_import_module(name: str) -> object:
        if name == "broken.mod":
            raise RuntimeError("broken import")
        return object()

    monkeypatch.setattr(runtime_module.importlib, "import_module", _fake_import_module)

    assert runtime_module._module_spec_exists("available.mod") is True
    assert runtime_module._module_spec_exists("missing.mod") is False
    assert runtime_module._module_import_status("missing.mod") == (False, "missing")
    assert runtime_module._module_import_status("broken.mod") == (False, "import failed (RuntimeError: broken import)")
    assert runtime_module._module_import_status("available.mod") == (True, "available")

    cfg = {
        "model": {"arch": "shape3d2vecset", "attn_backend": "xformers"},
        "data": {"train_ds": "Completion3D", "val_ds": ["ShapeNet"], "test_ds": None},
        "vis": {"method": "blender"},
    }
    obj = SimpleNamespace(model=SimpleNamespace(arch="other"))

    assert runtime_module._nested_get(cfg, "model.arch") == "shape3d2vecset"
    assert runtime_module._nested_get(obj, "model.arch") == "other"
    assert runtime_module._nested_get(obj, "model.missing") is None
    assert runtime_module._nested_get(None, "model.arch") is None
    assert runtime_module._as_str_list(None) == []
    assert runtime_module._as_str_list("chair") == ["chair"]
    assert runtime_module._as_str_list(("chair", 3)) == ["chair", "3"]
    assert runtime_module._as_str_list(5) == ["5"]
    assert runtime_module._extract_dataset_names(cfg) == ["completion3d", "shapenet"]

    def _fake_module_import_status(module_name: str) -> tuple[bool, str]:
        status = {
            "transforms3d": (False, "missing"),
            "xformers": (False, "import failed (ImportError: kernels missing)"),
            "torch_cluster": (True, "available"),
            "bpy": (True, "available"),
            "blenderproc": (False, "missing"),
        }
        return status[module_name]

    monkeypatch.setattr(runtime_module, "_module_import_status", _fake_module_import_status)
    monkeypatch.setattr(runtime_module, "_module_spec_exists", lambda name: name in {"xformers", "bpy"})

    lines_required = runtime_module.optional_dependency_summary(cfg)
    assert lines_required[0] == "MISSING (required):"
    assert any("transforms3d" in line and "Completion3D-specific" in line for line in lines_required)
    assert any("xformers" in line and "import failed" in line for line in lines_required)
    assert any("OK:" == line for line in lines_required)
    assert any("bpy" in line for line in lines_required)

    lines_optional = runtime_module.optional_dependency_summary(
        model_arch="onet",
        attn_backend="torch",
        dataset_names=["shapenet"],
        needs_blender=False,
    )
    assert lines_optional[0] == "OK:"
    assert any("xformers" in line for line in lines_optional)
    assert any("Skipped (not needed):" == line for line in lines_optional)
    assert any("transforms3d" in line for line in lines_optional)

    info_calls: list[str] = []
    runtime_module.log_optional_dependency_summary(SimpleNamespace(info=info_calls.append), cfg)
    assert info_calls == ["Optional dependency summary:\n" + "\n".join(lines_required)]


def test_setup_logger_debug_helpers_and_set_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeLogger:
        def __init__(self, enabled: bool):
            self.enabled = enabled
            self.calls: list[tuple[int, str, tuple[Any, ...], dict[str, Any]]] = []

        def isEnabledFor(self, level: int) -> bool:
            return self.enabled

        def _log(self, level: int, message: str, args: tuple[Any, ...], **kwargs: Any) -> None:
            self.calls.append((level, message, args, kwargs))

    direct_logger = _FakeLogger(enabled=True)
    wrapped_debug = cast(Any, logging_module.debug_with_level).__wrapped__
    wrapped_debug(direct_logger, "hello", py_logging.INFO, 1, extra={"k": "v"})
    assert direct_logger.calls == [(py_logging.INFO, "hello", (1,), {"extra": {"k": "v"}})]

    silent_logger = _FakeLogger(enabled=False)
    wrapped_debug(silent_logger, "ignored", py_logging.INFO)
    assert silent_logger.calls == []

    monkeypatch.setattr(logging_module, "_created_loggers", set())
    monkeypatch.setattr(logging_module.os, "getcwd", lambda: "/tmp/project")
    monkeypatch.setattr(sys, "argv", ["/tmp/project/scripts/run.py"])

    debug_calls: list[tuple[str, str, int, tuple[Any, ...], dict[str, Any]]] = []

    def _fake_debug_with_level(
        logger: Any, message: str, level: int = py_logging.DEBUG, *args: Any, **kwargs: Any
    ) -> None:
        debug_calls.append((logger.name, message, level, args, kwargs))

    monkeypatch.setattr(logging_module, "debug_with_level", _fake_debug_with_level)

    logger = logging_module.setup_logger("__main__")
    assert logger.name == "scripts.run"
    assert logger.level == py_logging.INFO

    logger.debug("debug")
    logger.info("info")
    logger.warning("warn")
    logger.error("error")
    logger.fatal("fatal")
    logger.critical("critical")
    cast(Any, logger).debug_level_1("d1")
    cast(Any, logger).debug_level_2("d2")
    logger.exception("boom")

    expected_levels = [
        py_logging.DEBUG,
        py_logging.INFO,
        py_logging.WARNING,
        py_logging.ERROR,
        py_logging.FATAL,
        py_logging.CRITICAL,
        logging_module.DEBUG_LEVEL_1,
        logging_module.DEBUG_LEVEL_2,
        py_logging.ERROR,
    ]
    assert [call[2] for call in debug_calls] == expected_levels
    assert debug_calls[-1][4]["exc_info"] is True

    same_logger = logging_module.setup_logger("scripts.run")
    assert same_logger is logger

    logging_module.set_log_level("DEBUG")
    assert logger.level == py_logging.DEBUG
    assert any("Using log level DEBUG" in call[1] for call in debug_calls)
