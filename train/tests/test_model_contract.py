from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest
from omegaconf import OmegaConf

from ..scripts.resolve_model_contract import effective_from_config, effective_from_kwargs
from ..src.model_contract import (
    MODEL_CONTRACT_BASE_FIELDS,
    MODEL_CONTRACT_SCHEMA,
    ModelContractError,
    compare_canonical_effective_config,
    compare_effective_config,
    verify_and_write_model_contract,
)


class Model:
    effective_model_config: ClassVar[dict[str, object]] = {
        "branch_quality_heads": True,
        "branch_quality_loss_balance": "task_weights",
        "inputs_stop_gradient": False,
    }


def write_contract(path: Path, expected: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"schema": MODEL_CONTRACT_SCHEMA, "arm": "c1", "expected": expected}),
        encoding="utf-8",
    )


def test_matching_contract_writes_verified_artifact(tmp_path: Path) -> None:
    contract = tmp_path / "c1.json"
    artifact = tmp_path / "run" / "effective_model_config.json"
    write_contract(
        contract,
        {
            "branch_quality_heads": True,
            "branch_quality_loss_balance": "task_weights",
            "inputs_stop_gradient": False,
        },
    )

    result = verify_and_write_model_contract(Model(), contract_path=contract, artifact_path=artifact)

    assert result["status"] == "verified"
    assert json.loads(artifact.read_text(encoding="utf-8"))["status"] == "verified"


def test_mismatch_writes_rejected_artifact_before_raising(tmp_path: Path) -> None:
    contract = tmp_path / "c1.json"
    artifact = tmp_path / "run" / "effective_model_config.json"
    write_contract(contract, {"branch_quality_loss_balance": "equal"})

    with pytest.raises(ModelContractError, match="effective model contract mismatch"):
        verify_and_write_model_contract(Model(), contract_path=contract, artifact_path=artifact)

    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["status"] == "rejected"
    assert saved["actual"]["branch_quality_loss_balance"] == "task_weights"


def test_compare_reports_only_expected_fields() -> None:
    assert compare_effective_config(
        {"branch_quality_heads": True, "extra": "ignored"},
        {"branch_quality_heads": True},
    ) == {"branch_quality_heads": True}


def test_missing_key_does_not_match_expected_none() -> None:
    with pytest.raises(ModelContractError, match="effective model contract mismatch"):
        compare_effective_config({}, {"optional_setting": None})


def test_canonical_contract_rejects_partial_expected_fields(tmp_path: Path) -> None:
    actual: dict[str, object] = {field: None for field in MODEL_CONTRACT_BASE_FIELDS}
    actual["model_class"] = "DinoInstSeg3D"
    contract = tmp_path / "c1.json"
    artifact = tmp_path / "run" / "effective_model_config.json"
    write_contract(contract, {"model_class": "DinoInstSeg3D"})

    class CanonicalModel:
        effective_model_config = actual

    with pytest.raises(ModelContractError, match="canonical model contract fields mismatch"):
        verify_and_write_model_contract(CanonicalModel(), contract_path=contract, artifact_path=artifact)

    with pytest.raises(ModelContractError, match="canonical model contract fields mismatch"):
        compare_canonical_effective_config(actual, {"model_class": "DinoInstSeg3D"})


def test_resolver_uses_constructor_default_when_balance_override_is_absent() -> None:
    from models import resolve_dino_instseg3d_kwargs

    cfg = OmegaConf.create(
        {
            "inputs": {"dim": 3},
            "model": {"bias": True, "dropout": 0.0},
            "train": {"loss": None},
            "multitask": "head",
            "mlp_heads": True,
            "inputs_weight": 0.2,
            "branch_quality_heads": True,
        }
    )

    effective = effective_from_kwargs(resolve_dino_instseg3d_kwargs(cfg))

    assert effective["branch_quality_loss_balance"] == "equal"
    with pytest.raises(ModelContractError, match="effective model contract mismatch"):
        compare_effective_config(
            effective,
            {"branch_quality_loss_balance": "task_weights"},
        )


def test_resolver_includes_rgbd_wrapper_contract_fields() -> None:
    config = OmegaConf.create(
        {
            "model": {
                "arch": "dino_inst_rgbd3d",
                "bias": True,
                "dropout": 0.0,
            },
            "inputs": {"dim": 3},
            "train": {"loss": None},
            "rgb_fusion": "raw_dual",
            "fusion_mode": "gate",
            "fusion_max_ratio": 0.1,
            "point_fusion_max_ratio": 0.2,
            "query_fusion_max_ratio": 0.3,
            "query_fusion_stage": "early",
        }
    )

    effective = effective_from_config(config)

    assert effective["model_class"] == "DinoInstSegRGBD3D"
    assert effective["rgb_fusion"] == "raw_dual"
    assert effective["train_mode"] == "finetune"
    assert effective["rgb_repo_or_dir"] == "facebookresearch/dinov2"
    assert effective["rgb_backbone"] == "dinov2_vits14"
    assert effective["freeze_rgb_encoder"] is True
    assert effective["rgb_image_normalization"] == "auto"
    assert effective["fusion_gate_init_bias"] == -10.0
    assert effective["rgb_residual_dropout"] == 0.0
    assert effective["depth_fusion_dropout"] == 0.0
    assert effective["rgb_token_dropout"] == 0.0
    assert effective["point_fusion_max_ratio"] == 0.2
    assert effective["query_fusion_max_ratio"] == 0.3
    assert effective["query_fusion_stage"] == "early"
    assert compare_effective_config(
        effective,
        {
            "model_class": "DinoInstSegRGBD3D",
            "rgb_fusion": "raw_dual",
            "point_fusion_max_ratio": 0.2,
            "query_fusion_max_ratio": 0.3,
            "query_fusion_stage": "early",
        },
    ) == {
        "model_class": "DinoInstSegRGBD3D",
        "rgb_fusion": "raw_dual",
        "point_fusion_max_ratio": 0.2,
        "query_fusion_max_ratio": 0.3,
        "query_fusion_stage": "early",
    }
