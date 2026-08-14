#!/usr/bin/env python3
"""Resolve a Hydra job and compare its model-construction settings without building a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from models import resolve_dino_instseg3d_kwargs, resolve_dino_instseg_rgbd3d_kwargs
from train.src.model_contract import (
    MODEL_CONTRACT_SCHEMA,
    compare_canonical_effective_config,
    load_contract,
    sha256,
)


def jsonable(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    return parser.parse_args()


def effective_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return jsonable(
        {
            "model_class": "DinoInstSeg3D",
            "branch_quality_heads": kwargs["branch_quality_heads"],
            "branch_quality_loss_balance": kwargs["branch_quality_loss_balance"],
            "inputs_stop_gradient": kwargs["inputs_stop_gradient"],
            "multitask": kwargs["multitask"],
            "pred_cls": "" if kwargs["pred_cls"] is None else kwargs["pred_cls"],
            "points_weight": kwargs["points_weight"],
            "inputs_weight": kwargs["inputs_weight"],
            "matcher_points_weight": kwargs["matcher_points_weight"],
            "matcher_inputs_weight": kwargs["matcher_inputs_weight"],
            "depth_descriptor": kwargs["depth_descriptor"],
            "salient_anchor_fraction": kwargs["salient_anchor_fraction"],
            "salient_candidate_multiplier": kwargs["salient_candidate_multiplier"],
            "salient_memory_attention": kwargs["salient_memory_attention"],
            "voxel_anchor_resolutions": kwargs["voxel_anchor_resolutions"] or (),
            "voxel_neighborhood_resolutions": kwargs["voxel_neighborhood_resolutions"] or (),
            "voxel_bias_resolution": kwargs["voxel_bias_resolution"],
            "spatial_reference_weight": kwargs["spatial_reference_weight"],
            "spatial_feedback": kwargs["spatial_feedback"],
            "allow_branch_warmstart": kwargs["allow_branch_warmstart"],
        }
    )


def effective_from_config(config: DictConfig) -> dict[str, Any]:
    effective = effective_from_kwargs(resolve_dino_instseg3d_kwargs(config))
    if "rgbd3d" not in str(config.model.arch):
        return effective

    rgb_kwargs = resolve_dino_instseg_rgbd3d_kwargs(config)
    effective.update(
        {
            "model_class": "DinoInstSegRGBD3D",
            "rgb_fusion": rgb_kwargs["rgb_fusion"],
            "fusion_mode": rgb_kwargs["fusion_mode"],
            "train_mode": rgb_kwargs["train_mode"],
            "rgb_repo_or_dir": rgb_kwargs["repo_or_dir"],
            "rgb_backbone": rgb_kwargs["backbone"],
            "freeze_rgb_encoder": rgb_kwargs["freeze_rgb_encoder"],
            "rgb_image_normalization": rgb_kwargs["rgb_image_normalization"],
            "fusion_gate_init_bias": rgb_kwargs["fusion_gate_init_bias"],
            "featup_repo": rgb_kwargs["featup_repo"],
            "featup_checkpoint": rgb_kwargs["featup_checkpoint"],
            "rgb_metric_checkpoint": rgb_kwargs["rgb_metric_checkpoint"],
            "fusion_max_ratio": rgb_kwargs["fusion_max_ratio"],
            "point_fusion_max_ratio": rgb_kwargs["point_fusion_max_ratio"],
            "query_fusion_max_ratio": rgb_kwargs["query_fusion_max_ratio"],
            "query_rgb_tokens": rgb_kwargs["query_rgb_tokens"],
            "rgb_delivery": rgb_kwargs["rgb_delivery"],
            "query_fusion_stage": rgb_kwargs["query_fusion_stage"],
            "rgb_residual_dropout": rgb_kwargs["rgb_residual_dropout"],
            "depth_fusion_dropout": rgb_kwargs["depth_fusion_dropout"],
            "rgb_token_dropout": rgb_kwargs["rgb_token_dropout"],
        }
    )
    return effective


def resolve(config: DictConfig, contract_path: Path, expected_sha256: str | None) -> dict[str, Any]:
    contract = load_contract(contract_path)
    contract_sha256 = sha256(contract_path)
    if expected_sha256 is not None and contract_sha256 != expected_sha256:
        raise ValueError(f"model contract file hash mismatch: actual={contract_sha256} expected={expected_sha256}")
    kwargs = resolve_dino_instseg3d_kwargs(config)
    effective = effective_from_config(config)
    matched = compare_canonical_effective_config(effective, contract["expected"])
    return {
        "schema": MODEL_CONTRACT_SCHEMA,
        "status": "verified",
        "arm": contract.get("arm"),
        "contract_sha256": contract_sha256,
        "effective": effective,
        "matched": matched,
        "constructor_kwargs": jsonable(kwargs),
    }


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if not isinstance(config, DictConfig):
        raise TypeError(f"expected a mapping config, got {type(config).__name__}")
    print(json.dumps(resolve(config, args.contract, args.expected_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
