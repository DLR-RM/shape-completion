from __future__ import annotations

from pathlib import Path

import pytest

from experiments.reproduction import RECIPES, Recipe, get_recipe, iter_recipes, main, parse_values, repo_root

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VALUES = {
    "weights": "runs/model_best.pt",
    "vae_weights": "runs/vae.pt",
    "mesh": "mesh.obj",
    "occ_weights": "runs/occ.pt",
    "inst_weights": "runs/inst.pt",
    "bop_split": "bopscene_hb_val_primesense",
    "mesh_root": "data/meshes",
    "watertight_root": "data/watertight",
    "samples_root": "data/samples",
    "depth_root": "data/depth",
    "poses_root": "data/poses",
    "hdf5_root": "data/hdf5",
    "packed_root": "data/packed",
    "shapenet_root": "data/shapenet/ShapeNetCore.v1",
    "n_jobs": "1",
    "num_scenes": "1",
    "num_views": "1",
    "shard": "0",
    "tabletop_dataset": "tabletop.v3",
}


def _tokens(key: str, **values: str) -> list[str]:
    defaults = dict(SAMPLE_VALUES)
    defaults.update(values)
    return get_recipe(key).render_args(defaults)


def test_public_suites_have_train_and_runnable_non_train_recipes() -> None:
    for suite in ("iros2023", "humanoids2023", "3dv2026", "inst3d2026"):
        stages = {recipe.stage for recipe in iter_recipes(suite=suite)}
        assert "train" in stages
        assert stages & {"eval", "preprocess"}


def test_recipe_names_are_unique_within_suite() -> None:
    keys = [recipe.key for recipe in RECIPES]
    assert len(keys) == len(set(keys))


def test_execute_uses_repository_root_as_working_directory() -> None:
    assert repo_root() == REPO_ROOT
    assert (repo_root() / "scripts" / "run.sh").is_file()


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda recipe: recipe.key)
def test_all_recipes_render_with_sample_values(recipe: Recipe) -> None:
    args = recipe.render_args(SAMPLE_VALUES)

    assert args
    assert all("{" not in arg and "}" not in arg for arg in args)


def test_missing_required_values_fail_before_rendering() -> None:
    recipe = get_recipe("3dv2026:ar-uncond-eval")

    with pytest.raises(ValueError, match="requires: vae_weights, weights"):
        recipe.render_args({})


def test_parse_values_requires_key_value_pairs() -> None:
    assert parse_values(["weights=foo", "bop_split=bopscene_hb_val_primesense"]) == {
        "weights": "foo",
        "bop_split": "bopscene_hb_val_primesense",
    }

    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_values(["weights"])


def test_ar_recipe_contains_required_vqvae_overrides() -> None:
    args = _tokens("3dv2026:ar-uncond-train")

    assert "model.arch=larm" in args
    assert "model.compile=False" in args
    assert "++vae_arch=3dshape2vecset_vqvae" in args
    assert "+vae_weights=runs/vae.pt" in args


def test_3dv_eval_recipes_preserve_runtime_normalization_drift_fix() -> None:
    args = _tokens("3dv2026:ldm-uncond-eval")

    assert "norm.center=True" in args
    assert "norm.scale=True" in args
    assert "train.batch_size=64" in args
    assert "train.epochs=2000" in args
    assert "train.lr=3.90625e-07" in args


def test_inst3d_bop_recipe_uses_camera_frame_and_filter_overrides() -> None:
    args = _tokens("inst3d2026:bop-pipeline-eval")

    assert "data.frame=cam" in args
    assert "aug.rotate=False" in args
    assert "inputs.crop=511" in args
    assert "points.subsample=False" in args
    assert "+apply_filter=False" in args
    assert "+align_to_gt=True" in args
    assert "+single_view=False" in args
    assert "+refine_pose=False" in args


def test_inst3d_pile_recipe_uses_kinect_sim_and_no_aug_eval_settings() -> None:
    args = _tokens("inst3d2026:joint-pile-ft")

    assert "inputs.type=kinect_sim" in args
    assert "data.train_ds=[tabletop_pile]" in args
    assert "test.no_aug=True" in args


def test_datagen_recipes_cover_all_public_suites() -> None:
    recipes_by_suite = {
        suite: {recipe.name for recipe in iter_recipes(suite=suite, stage="preprocess")}
        for suite in ("iros2023", "humanoids2023", "3dv2026", "inst3d2026")
    }

    assert {"watertight-meshes", "sample-points", "uncertain-labels"} <= recipes_by_suite["iros2023"]
    assert {"watertight-meshes", "render-kinect", "render-kinect-parallel", "stable-poses"} <= recipes_by_suite[
        "humanoids2023"
    ]
    assert {"watertight-meshes", "sample-points", "render-kinect-parallel", "pack-hdf5"} <= recipes_by_suite["3dv2026"]
    assert {"render-tabletop-packed", "render-tabletop-pile", "add-kinect-sim", "pack-hdf5"} <= recipes_by_suite[
        "inst3d2026"
    ]


def test_cli_lists_and_renders_recipes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list", "--suite", "iros2023"]) == 0
    listed = capsys.readouterr().out
    assert "iros2023:mugs-main-train" in listed

    assert main(["--recipe", "iros2023:mugs-main-train"]) == 0
    rendered = capsys.readouterr().out
    assert "./scripts/run.sh train -cn mugs_paper" in rendered


def test_public_reproduction_docs_do_not_expose_local_infrastructure() -> None:
    public_text = "\n".join(
        [
            (REPO_ROOT / "README.md").read_text(),
            (REPO_ROOT / "docs" / "reproduction.md").read_text(),
        ]
    )

    forbidden_tokens = [
        "rmc-" + "gpu19",
        "build_" + "cuda.sh",
        "dirs=" + "dlr",
        "/vol" + "ume/",
        "/run/" + "media/",
        "shape-completion-" + "ci" + "ssy",
        "osl" + "155",
    ]
    leaked = [token for token in forbidden_tokens if token in public_text]
    assert leaked == []
