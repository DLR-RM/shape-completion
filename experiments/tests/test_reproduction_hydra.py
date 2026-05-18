from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from experiments.reproduction import RECIPES, Recipe, get_recipe

hydra = pytest.importorskip("hydra")

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = REPO_ROOT / "conf"


def _compose_recipe(key: str, **values: str) -> Any:
    defaults = {
        "weights": "runs/model_best.pt",
        "vae_weights": "runs/vae.pt",
        "occ_weights": "runs/occ.pt",
        "inst_weights": "runs/inst.pt",
        "bop_split": "bopscene_hb_val_primesense",
    }
    defaults.update(values)
    args = get_recipe(key).render_args(defaults)
    assert args[0] == "./scripts/run.sh"
    assert "-cn" in args
    config_index = args.index("-cn") + 1
    config_name = args[config_index]
    overrides = [arg for arg in args[config_index + 1 :] if "=" in arg]

    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return hydra.compose(config_name=config_name, overrides=[*overrides, "dirs=default", "load.hdf5=False"])


RUNNER_RECIPES = tuple(recipe for recipe in RECIPES if recipe.use_runner)


@pytest.mark.parametrize("recipe", RUNNER_RECIPES, ids=lambda recipe: recipe.key)
def test_all_runner_recipes_compose_with_public_configs(recipe: Recipe) -> None:
    cfg = _compose_recipe(recipe.key)

    assert cfg is not None


@pytest.mark.parametrize(
    ("key", "expected_arch"),
    [
        ("iros2023:mugs-main-train", "conv_onet_grid"),
        ("humanoids2023:vqdif-train", "vqdif"),
        ("3dv2026:ldm-uncond-train", "ldm"),
        ("3dv2026:ar-uncond-train", "larm"),
        ("inst3d2026:joint-train", "dino_inst_mask"),
        ("inst3d2026:pipeline-pile-eval", "inst_pipe"),
    ],
)
def test_representative_recipes_compose_with_public_configs(key: str, expected_arch: str) -> None:
    cfg = _compose_recipe(key)

    assert cfg.model.arch == expected_arch


def test_inst3d_bop_recipe_composes_with_camera_frame_overrides() -> None:
    cfg = _compose_recipe("inst3d2026:bop-joint-eval")

    assert cfg.data.frame == "cam"
    assert cfg.aug.rotate is False
    assert cfg.points.subsample is False
    assert cfg.apply_filter is False
    assert cfg.align_to_gt is True
    assert cfg.single_view is False
    assert cfg.refine_pose is False
