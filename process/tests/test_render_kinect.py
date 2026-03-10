import sys
from argparse import Namespace
from pathlib import Path

from process.scripts import render_kinect as rk


def test_run_uses_input_parent_when_out_dir_not_set(monkeypatch, tmp_path: Path) -> None:
    in_file = tmp_path / "a" / "b" / "mesh.obj"
    in_file.parent.mkdir(parents=True)
    in_file.write_text("dummy")

    calls: list[Path] = []

    def fake_render(in_path: Path, out_dir: Path, _args: Namespace) -> None:
        assert in_path == in_file
        calls.append(out_dir)
        (out_dir / "lock").write_text("locked")

    monkeypatch.setattr(rk, "render", fake_render)

    args = Namespace(in_file=in_file, out_dir=None, remove=False)
    rk.run(args)

    assert calls == [in_file.parent]
    assert not (in_file.parent / "lock").exists()


def test_run_uses_configured_out_dir_layout(monkeypatch, tmp_path: Path) -> None:
    in_file = tmp_path / "dataset" / "object_01" / "mesh.obj"
    in_file.parent.mkdir(parents=True)
    in_file.write_text("dummy")
    out_root = tmp_path / "renders"
    out_root.mkdir()

    out_dirs: list[Path] = []

    def fake_render(_in_path: Path, out_dir: Path, _args: Namespace) -> None:
        out_dirs.append(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "lock").write_text("locked")

    monkeypatch.setattr(rk, "render", fake_render)

    args = Namespace(in_file=in_file, out_dir=out_root, remove=False)
    rk.run(args)

    expected = out_root / "dataset" / "object_01"
    assert out_dirs == [expected]
    assert not (expected / "lock").exists()


def test_main_writes_command_file_to_cwd_when_out_dir_missing(monkeypatch, tmp_path: Path) -> None:
    in_file = tmp_path / "mesh.obj"
    in_file.write_text("dummy")

    captured: dict[str, Path | bool] = {}

    def fake_save_command(path: Path, _args: Namespace) -> None:
        captured["command_path"] = path

    def fake_run(_args: Namespace) -> None:
        captured["run_called"] = True

    monkeypatch.setattr(rk, "save_command_and_args_to_file", fake_save_command)
    monkeypatch.setattr(rk, "run", fake_run)
    monkeypatch.setattr(rk, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(rk, "log_optional_dependency_summary", lambda _logger: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["render_kinect.py", str(in_file)])

    rk.main()

    assert captured["command_path"] == tmp_path / "command.txt"
    assert captured["run_called"] is True
