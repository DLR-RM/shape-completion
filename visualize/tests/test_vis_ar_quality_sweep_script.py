from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import trimesh

from visualize.scripts import vis_ar_quality_sweep as script


def _write_mesh(path: Path, x_offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(
        vertices=[
            [0.0 + x_offset, 0.0, 0.0],
            [1.0 + x_offset, 0.0, 0.0],
            [0.0 + x_offset, 1.0, 0.0],
        ],
        faces=[[0, 1, 2]],
        process=False,
    )
    mesh.export(path)


def test_choose_best_supports_both_criteria() -> None:
    low_cd = script.CandidateMetrics(
        seed=0,
        run_name="run0",
        object_key="03001627/object",
        mesh_path="pred0.ply",
        gt_path="gt.ply",
        cd_l1=0.1,
        cd_l2=0.2,
        fscore_01=0.4,
        precision_01=0.5,
        recall_01=0.6,
        component_count=1,
        largest_component_ratio=1.0,
    )
    high_fscore = script.CandidateMetrics(
        seed=1,
        run_name="run1",
        object_key="03001627/object",
        mesh_path="pred1.ply",
        gt_path="gt.ply",
        cd_l1=0.3,
        cd_l2=0.4,
        fscore_01=0.9,
        precision_01=0.7,
        recall_01=0.8,
        component_count=1,
        largest_component_ratio=1.0,
    )

    assert script.choose_best([high_fscore, low_cd], "cd_l1") is low_cd
    assert script.choose_best([low_cd, high_fscore], "fscore_01") is high_fscore

    with pytest.raises(ValueError, match="Unknown criterion"):
        script.choose_best([low_cd, high_fscore], "invalid")


def test_run_generation_for_seed_builds_command_and_tails_stderr(monkeypatch: Any, tmp_path: Path) -> None:
    stderr = "\n".join(f"line {idx}" for idx in range(30))

    def fake_run(cmd: list[str], cwd: Path, capture_output: bool, text: bool, check: bool) -> Any:
        assert cwd == tmp_path
        assert capture_output is True
        assert text is True
        assert check is False
        return SimpleNamespace(returncode=7, stderr=stderr)

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    cmd, return_code, stderr_tail = script.run_generation_for_seed(
        python_executable=".venv/bin/python",
        repo_root=tmp_path,
        config_name="demo_cfg",
        base_log_name="demo",
        seed=3,
        objects=[19, 41],
        overrides=["model.arch=larm"],
    )

    assert cmd == [
        ".venv/bin/python",
        "visualize/scripts/vis_generation_process.py",
        "-cn",
        "demo_cfg",
        "log.name=demo_s3",
        "misc.seed=3",
        "+vis.objects=[19,41]",
        "+vis.ar_steps=[512]",
        "model.arch=larm",
    ]
    assert return_code == 7
    assert stderr_tail == "\n".join(f"line {idx}" for idx in range(10, 30))


def test_main_smoke_writes_report_and_summary(monkeypatch: Any, tmp_path: Path) -> None:
    generation_root = tmp_path / "generation"
    output_dir = tmp_path / "audit"
    object_key = "03001627/common_obj"

    for run_name, x_offset in (("demo_s0", 0.3), ("demo_s1", 0.0)):
        object_dir = generation_root / run_name / "generation_process" / "03001627" / "common_obj"
        _write_mesh(object_dir / "ar" / "token_512.ply", x_offset=x_offset)
        _write_mesh(object_dir / "gt.ply")

    only_seed0_dir = generation_root / "demo_s0" / "generation_process" / "03001627" / "only_seed0"
    _write_mesh(only_seed0_dir / "ar" / "token_512.ply", x_offset=0.1)
    _write_mesh(only_seed0_dir / "gt.ply")

    monkeypatch.setattr(
        script.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(
            generation_root=generation_root,
            base_log_name="demo",
            seeds="0,1",
            objects="41,19",
            config_name="cvpr_2025",
            python=".venv/bin/python",
            run_generate=False,
            override=[],
            token_name="token_512.ply",
            points=32,
            criterion="cd_l1",
            output_dir=output_dir,
        ),
    )

    script.main()

    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    summary = (output_dir / "summary.csv").read_text(encoding="utf-8")

    assert report["base_log_name"] == "demo"
    assert report["objects_requested"] == [19, 41]
    assert report["common_object_keys"] == [object_key]
    assert report["missing_objects_by_run"]["demo_s0"] == []
    assert report["missing_objects_by_run"]["demo_s1"] == ["03001627/only_seed0"]
    assert report["selection"][object_key]["best_seed"] == 1
    assert report["diagnostics"]["object_identity_consistent_across_seeds"] is False
    assert "common_obj,1,cd_l1" in summary
