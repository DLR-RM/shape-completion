from argparse import Namespace
from pathlib import Path

import pytest

from process.scripts import render_kinect_parallel as rkp


def test_run_single_shard_calls_render_and_clears_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    in_path = tmp_path / "mesh.obj"
    in_path.write_text("dummy")
    out_root = tmp_path / "out"
    out_root.mkdir()

    calls: list[tuple[Path, Path]] = []

    def fake_resolve_out_dir(_in_path: Path, _in_dir: Path, _out_dir: Path, shard):
        assert shard is None
        target = out_root / "single"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def fake_render(mesh_path: Path, out_dir: Path, _args) -> None:
        calls.append((mesh_path, out_dir))
        (out_dir / "lock").write_text("locked")

    monkeypatch.setattr(rkp, "resolve_out_dir", fake_resolve_out_dir)
    monkeypatch.setattr(rkp, "render", fake_render)

    args = Namespace(n_shards=1, in_dir=tmp_path, out_dir=out_root, remove=False)
    rkp.run(in_path, args)

    assert calls == [(in_path, out_root / "single")]
    assert not (out_root / "single" / "lock").exists()


def test_run_multiple_shards_remove_on_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    in_path = tmp_path / "mesh.obj"
    in_path.write_text("dummy")
    out_root = tmp_path / "out"
    out_root.mkdir()

    calls: list[int] = []
    removed: list[Path] = []

    def fake_resolve_out_dir(_in_path: Path, _in_dir: Path, _out_dir: Path, shard):
        target = out_root / f"shard_{shard}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "lock").write_text("locked")
        return target

    def fake_render(_mesh_path: Path, out_dir: Path, _args) -> None:
        calls.append(int(out_dir.name.split("_")[-1]))
        if out_dir.name.endswith("_1"):
            raise RuntimeError("render failed")

    def fake_rmtree(path: Path, ignore_errors: bool = False) -> None:
        removed.append(path)

    monkeypatch.setattr(rkp, "resolve_out_dir", fake_resolve_out_dir)
    monkeypatch.setattr(rkp, "render", fake_render)
    monkeypatch.setattr(rkp.shutil, "rmtree", fake_rmtree)

    args = Namespace(n_shards=3, in_dir=tmp_path, out_dir=out_root, remove=True)
    rkp.run(in_path, args)

    assert calls == [0, 1]
    assert removed == [out_root / "shard_1"]
    assert not (out_root / "shard_1" / "lock").exists()


def test_run_uses_input_parent_when_out_dir_not_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    in_path = tmp_path / "mesh.obj"
    in_path.write_text("dummy")
    (tmp_path / "lock").write_text("locked")

    calls: list[Path] = []

    def fail_resolve_out_dir(*_args, **_kwargs):
        raise AssertionError("resolve_out_dir should not be called when args.out_dir is None")

    def fake_render(_mesh_path: Path, out_dir: Path, _args) -> None:
        calls.append(out_dir)

    monkeypatch.setattr(rkp, "resolve_out_dir", fail_resolve_out_dir)
    monkeypatch.setattr(rkp, "render", fake_render)

    args = Namespace(n_shards=1, in_dir=tmp_path, out_dir=None, remove=False)
    rkp.run(in_path, args)

    assert calls == [tmp_path]
    assert not (tmp_path / "lock").exists()
