"""Verify constructed model settings against an immutable runtime contract."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

MODEL_CONTRACT_SCHEMA = "dino_instseg3d_effective_config_v1"


class ModelContractError(ValueError):
    """Raised when a constructed model does not match its arm contract."""


MODEL_CONTRACT_BASE_FIELDS = (
    "model_class",
    "branch_quality_heads",
    "branch_quality_loss_balance",
    "inputs_stop_gradient",
    "multitask",
    "pred_cls",
    "points_weight",
    "inputs_weight",
    "matcher_points_weight",
    "matcher_inputs_weight",
    "depth_descriptor",
    "salient_anchor_fraction",
    "salient_candidate_multiplier",
    "salient_memory_attention",
    "voxel_anchor_resolutions",
    "voxel_neighborhood_resolutions",
    "voxel_bias_resolution",
    "spatial_reference_weight",
    "spatial_feedback",
    "allow_branch_warmstart",
)
MODEL_CONTRACT_RGBD_FIELDS = (
    "rgb_fusion",
    "fusion_mode",
    "train_mode",
    "rgb_repo_or_dir",
    "rgb_backbone",
    "freeze_rgb_encoder",
    "rgb_image_normalization",
    "fusion_gate_init_bias",
    "featup_repo",
    "featup_checkpoint",
    "rgb_metric_checkpoint",
    "fusion_max_ratio",
    "point_fusion_max_ratio",
    "query_fusion_max_ratio",
    "query_rgb_tokens",
    "rgb_delivery",
    "query_fusion_stage",
    "rgb_residual_dropout",
    "depth_fusion_dropout",
    "rgb_token_dropout",
)
CANONICAL_MODEL_CLASSES = ("DinoInstSeg3D", "DinoInstSegRGBD3D")


def canonical_effective_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project a runtime effective config onto the immutable contract fields."""
    fields: list[str] = list(MODEL_CONTRACT_BASE_FIELDS)
    model_class = config.get("model_class")
    if model_class not in CANONICAL_MODEL_CLASSES:
        raise ModelContractError(f"unsupported canonical model class: {model_class!r}")
    if model_class == "DinoInstSegRGBD3D":
        fields.extend(MODEL_CONTRACT_RGBD_FIELDS)
    missing = [field for field in fields if field not in config]
    if missing:
        raise ModelContractError(f"effective model config is missing canonical fields: {missing}")
    return {field: config[field] for field in fields}


def compare_canonical_effective_config(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a complete canonical contract and compare its normalized fields."""
    canonical = canonical_effective_config(actual)
    missing = sorted(set(canonical) - set(expected))
    extra = sorted(set(expected) - set(canonical))
    if missing or extra:
        raise ModelContractError(
            f"canonical model contract fields mismatch: missing_expected={missing}, unexpected_expected={extra}"
        )
    return compare_effective_config(canonical, expected)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return _jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"model contract contains non-serializable value: {type(value).__name__}")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MODEL_CONTRACT_SCHEMA:
        raise ModelContractError(f"unsupported model contract schema: {payload.get('schema')!r}")
    expected = payload.get("expected")
    if not isinstance(expected, Mapping) or not expected:
        raise ModelContractError("model contract must contain a non-empty expected mapping")
    return {"schema": payload["schema"], "arm": payload.get("arm"), "expected": dict(expected)}


def effective_model_config(model: Any) -> dict[str, Any]:
    config = getattr(model, "effective_model_config", None)
    if callable(config):
        config = config()
    if not isinstance(config, Mapping):
        raise ModelContractError("constructed model exposes no effective_model_config mapping")
    return _jsonable(dict(config))


def compare_effective_config(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    differences = {
        key: {"actual": actual.get(key), "expected": value}
        for key, value in expected.items()
        if key not in actual or actual.get(key) != value
    }
    if differences:
        raise ModelContractError(f"effective model contract mismatch: {differences}")
    return {str(key): actual[key] for key in expected}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_and_write_model_contract(
    model: Any,
    *,
    contract_path: Path,
    artifact_path: Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    actual = effective_model_config(model)
    artifact: dict[str, Any] = {
        "schema": MODEL_CONTRACT_SCHEMA,
        "status": "rejected",
        "contract_path": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "arm": contract.get("arm"),
        "expected": contract["expected"],
        "actual": actual,
    }
    try:
        if expected_sha256 is not None and artifact["contract_sha256"] != expected_sha256:
            raise ModelContractError(
                f"model contract file hash mismatch: actual={artifact['contract_sha256']} expected={expected_sha256}"
            )
        if actual.get("model_class") in CANONICAL_MODEL_CLASSES:
            artifact["matched"] = compare_canonical_effective_config(actual, contract["expected"])
        else:
            artifact["matched"] = compare_effective_config(actual, contract["expected"])
    except ModelContractError as error:
        artifact["error"] = str(error)
        _write_json_atomic(artifact_path, artifact)
        raise
    artifact["status"] = "verified"
    _write_json_atomic(artifact_path, artifact)
    return artifact
