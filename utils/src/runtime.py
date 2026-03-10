from __future__ import annotations

import importlib
import importlib.util
import os
import warnings
from collections.abc import Iterable, Mapping
from typing import Any


def suppress_known_optional_dependency_warnings() -> None:
    warnings.filterwarnings("ignore", message=r"xFormers is not available \(SwiGLU\)")
    warnings.filterwarnings("ignore", message=r"xFormers is not available \(Attention\)")
    warnings.filterwarnings("ignore", message=r"xFormers is not available \(Block\)")

    # Suppress _POSIX_C_SOURCE redefinition warning from Triton JIT compilation
    # (Python 3.11 pyconfig.h vs glibc features.h conflict)
    cflags = os.environ.get("CFLAGS", "")
    for flag in ("-Wno-cpp", "-Wno-macro-redefined"):
        if flag not in cflags:
            cflags = f"{cflags} {flag}".strip()
    os.environ["CFLAGS"] = cflags


def _module_spec_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _module_import_status(module_name: str) -> tuple[bool, str]:
    if not _module_spec_exists(module_name):
        return False, "missing"

    try:
        importlib.import_module(module_name)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}".splitlines()[0]
        return False, f"import failed ({reason})"
    return True, "available"


def _nested_get(source: Any, path: str) -> Any:
    value = source
    for key in path.split("."):
        if value is None:
            return None
        if isinstance(value, Mapping):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
    return value


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _extract_dataset_names(cfg: Any) -> list[str]:
    names: list[str] = []
    for split in ("train", "val", "test"):
        key = f"data.{split}_ds"
        names.extend(_as_str_list(_nested_get(cfg, key)))
    return [name.lower() for name in names]


def optional_dependency_summary(
    cfg: Any | None = None,
    *,
    model_arch: str | None = None,
    attn_backend: str | None = None,
    dataset_names: Iterable[str] | None = None,
    needs_blender: bool | None = None,
) -> list[str]:
    if model_arch is None and cfg is not None:
        model_arch = _nested_get(cfg, "model.arch")
    if attn_backend is None and cfg is not None:
        attn_backend = _nested_get(cfg, "model.attn_backend")
    if dataset_names is None and cfg is not None:
        dataset_names = _extract_dataset_names(cfg)

    model_arch_str = str(model_arch or "").lower()
    attn_backend_str = str(attn_backend or "torch").lower()
    datasets = [str(name).lower() for name in (dataset_names or [])]

    if needs_blender is None:
        vis_method = str(_nested_get(cfg, "vis.method") or "").lower() if cfg is not None else ""
        needs_blender = vis_method in {"blender", "cycles", "eevee"}

    checks: list[tuple[str, str, bool]] = [
        ("transforms3d", "Completion3D-specific augmentation helpers", any("completion3d" in n for n in datasets)),
        ("xformers", "memory-efficient attention backends (and DINO xformers kernels)", attn_backend_str == "xformers"),
        ("torch_cluster", "original Shape3D2VecSet FPS path", "shape3d2vecset" in model_arch_str),
        ("bpy", "Blender Python module for rendering", bool(needs_blender)),
        ("blenderproc", "BlenderProc rendering pipeline", bool(needs_blender)),
    ]

    available: list[str] = []
    not_needed: list[str] = []
    problems: list[str] = []

    for module_name, purpose, required in checks:
        if required:
            ok, detail = _module_import_status(module_name)
            if ok:
                available.append(f"  \u2713 {module_name:<16} {purpose}")
            else:
                problems.append(f"  \u2717 {module_name:<16} {detail} — {purpose}")
        else:
            if _module_spec_exists(module_name):
                available.append(f"  \u2713 {module_name:<16} {purpose}")
            else:
                not_needed.append(f"  \u2013 {module_name:<16} {purpose}")

    lines: list[str] = []
    if problems:
        lines.append("MISSING (required):")
        lines.extend(problems)
    if available:
        lines.append("OK:")
        lines.extend(available)
    if not_needed:
        lines.append("Skipped (not needed):")
        lines.extend(not_needed)
    return lines


def log_optional_dependency_summary(
    logger: Any,
    cfg: Any | None = None,
    *,
    model_arch: str | None = None,
    attn_backend: str | None = None,
    dataset_names: Iterable[str] | None = None,
    needs_blender: bool | None = None,
) -> None:
    lines = optional_dependency_summary(
        cfg, model_arch=model_arch, attn_backend=attn_backend, dataset_names=dataset_names, needs_blender=needs_blender
    )
    logger.info("Optional dependency summary:\n" + "\n".join(lines))
