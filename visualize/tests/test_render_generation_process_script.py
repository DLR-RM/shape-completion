from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from visualize.scripts import render_generation_process as script


def _save_image(path: Path, color: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (8, 8), color).save(path)


def test_load_rgba_and_compose_strip() -> None:
    img_a = Image.new("RGB", (4, 6), (255, 0, 0))
    img_b = Image.new("RGBA", (4, 6), (0, 255, 0, 255))

    strip = script.compose_strip([img_a, img_b], labels=["A", "B"], padding=2, label_height=10)

    assert strip.mode == "RGBA"
    assert strip.size == (10, 16)


def test_label_suffix_and_timeline_helpers(tmp_path: Path) -> None:
    render_dir = tmp_path / "frames"
    _save_image(render_dir / "step_02_normals.png", (1, 2, 3, 255))
    _save_image(render_dir / "step_00_normals.png", (1, 2, 3, 255))
    _save_image(render_dir / "token_003.png", (1, 2, 3, 255))
    _save_image(render_dir / "token_001.png", (1, 2, 3, 255))

    assert script.label_from_stem("step_07") == "t=07"
    assert script.label_from_stem("token_128") == "n=128"
    assert script.extract_suffix("step_00_normals") == "normals"
    assert script.extract_suffix("input_depth") == "depth"
    assert script.extract_suffix("gt") == ""
    assert [path.name for path in script.collect_timeline(render_dir, "normals")] == [
        "step_00_normals.png",
        "step_02_normals.png",
    ]
    assert [path.name for path in script.collect_timeline(render_dir, "")] == ["token_001.png", "token_003.png"]


def test_save_mp4_handles_missing_and_failed_ffmpeg(monkeypatch: Any, tmp_path: Path) -> None:
    frames = [Image.new("RGB", (4, 4), (255, 255, 255))]
    output_path = tmp_path / "movie.mp4"

    monkeypatch.setattr(script.shutil, "which", lambda name: None)
    assert script.save_mp4(frames, output_path, fps=8.0) is False

    monkeypatch.setattr(script.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(script.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="boom"))
    assert script.save_mp4(frames, output_path, fps=8.0) is False


def test_main_smoke_creates_strips_animations_and_comparison(monkeypatch: Any, tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "out"
    obj_dir = input_dir / "chairs" / "shape_001"

    for method in ("diffusion", "ar"):
        render_dir = obj_dir / method
        _save_image(render_dir / "input.png", (255, 0, 0, 255))
        _save_image(render_dir / "step_00.png", (0, 255, 0, 255))
        _save_image(render_dir / "step_01.png", (0, 0, 255, 255))
        _save_image(render_dir / "gt.png", (255, 255, 0, 255))
        _save_image(render_dir / "input_normals.png", (255, 0, 255, 255))
        _save_image(render_dir / "step_00_normals.png", (0, 255, 255, 255))
        _save_image(render_dir / "gt_normals.png", (128, 128, 128, 255))

    args = Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        method="both",
        labels=True,
        unconditional=False,
        gif_duration_ms=120,
        gif_loop=0,
        gif_bg_color=[255, 255, 255],
        animation_format="both",
        mp4_fps=None,
        show=False,
    )
    saved_mp4: list[tuple[Path, float, int]] = []

    monkeypatch.setattr(script.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(
        script,
        "save_mp4",
        lambda frames, path, fps: saved_mp4.append((Path(path), fps, len(frames))) or Path(path).write_text("mp4") or True,
    )

    script.main()

    obj_out = output_dir / "chairs" / "shape_001"
    assert (obj_out / "diffusion_strip.png").is_file()
    assert (obj_out / "diffusion_strip_normals.png").is_file()
    assert (obj_out / "ar_strip.png").is_file()
    assert (obj_out / "comparison.png").is_file()
    assert (obj_out / "comparison_normals.png").is_file()
    assert (obj_out / "diffusion_process.gif").is_file()
    assert (obj_out / "ar_process.gif").is_file()
    assert (obj_out / "diffusion_process_normals.mp4").is_file()
    assert (obj_out / "ar_process.mp4").is_file()
    assert saved_mp4
    assert saved_mp4[0][1] == 1000.0 / 120.0
