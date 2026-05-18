from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from experiments.reproduction import get_recipe
from experiments.tests.test_reproduction import SAMPLE_VALUES

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "key",
    [
        "iros2023:watertight-meshes",
        "iros2023:sample-points",
        "iros2023:uncertain-labels",
        "humanoids2023:render-kinect",
        "humanoids2023:render-kinect-parallel",
        "humanoids2023:stable-poses",
        "3dv2026:pack-hdf5",
        "inst3d2026:add-kinect-sim",
    ],
)
def test_reproduction_preprocess_entrypoints_show_help(key: str) -> None:
    args = get_recipe(key).render_args(SAMPLE_VALUES)
    if args[:2] != ["python3", "-m"]:
        pytest.skip(f"{key} is not a Python module entry point")

    command = [sys.executable, *args[1:], "--help"]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=45)

    assert result.returncode == 0, result.stderr


def test_reproduction_shell_wrappers_parse() -> None:
    result = subprocess.run(
        [
            "bash",
            "-n",
            "scripts/render_tabletop.sh",
            "scripts/train_eval.sh",
            "scripts/train_eval_select.sh",
            "scripts/test/test_eval.sh",
            "scripts/test/test_eval_gen.sh",
            "scripts/test/test_eval_img.sh",
            "scripts/test/test_eval_mesh.sh",
            "scripts/test/test_eval_pcd.sh",
            "scripts/test/test_eval_pcd_select.sh",
            "scripts/test/test_eval_pcd_uncond.sh",
            "scripts/test/test_eval_uncond.sh",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
