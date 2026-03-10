from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dataset.scripts import process_dataset as process_dataset_script


@contextmanager
def _null_context(*_args: Any, **_kwargs: Any):
    yield


class _ImmediateParallel:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __call__(self, tasks: Any) -> list[Any]:
        return [task() for task in tasks]


def _delayed(func: Any) -> Any:
    def _wrap(*args: Any, **kwargs: Any) -> Any:
        return lambda: func(*args, **kwargs)

    return _wrap


def test_create_splits_writes_expected_lists(tmp_path: Path) -> None:
    in_dir = tmp_path / "shapenet"
    class_dir = in_dir / "001"
    class_dir.mkdir(parents=True)
    for name in ["obj_a", "obj_b", "obj_c", "obj_d", "obj_e"]:
        (class_dir / name).mkdir()

    args = SimpleNamespace(in_dir=in_dir, out_dir=None, val_size=0.2, test_size=0.2, verbose=False)

    process_dataset_script.create_splits(args)

    assert (class_dir / "train.lst").read_text() == "obj_a\nobj_b\nobj_c"
    assert (class_dir / "val.lst").read_text() == "obj_d"
    assert (class_dir / "test.lst").read_text() == "obj_e"
    assert (class_dir / "all.lst").read_text() == "obj_a\nobj_b\nobj_c\nobj_d\nobj_e"
    assert (class_dir / "train_val.lst").read_text() == "obj_a\nobj_b\nobj_c\nobj_d"
    assert (class_dir / "train_test.lst").read_text() == "obj_a\nobj_b\nobj_c\nobj_e"


def test_merge_splits_writes_combined_object_lists(tmp_path: Path) -> None:
    in_dir = tmp_path / "src"
    class_a = in_dir / "001"
    class_b = in_dir / "002_v2"
    class_a.mkdir(parents=True)
    class_b.mkdir(parents=True)

    (class_a / "train.lst").write_text("mesh_a\n")
    (class_a / "val.lst").write_text("mesh_b\n")
    (class_a / "test.lst").write_text("mesh_c\n")

    (class_b / "train.lst").write_text("mesh_d\n")
    (class_b / "val.lst").write_text("")
    (class_b / "test.lst").write_text("mesh_e\n")

    out_dir = tmp_path / "merged"
    args = SimpleNamespace(in_dir=in_dir, out_dir=out_dir)

    process_dataset_script.merge_splits(args)

    assert (out_dir / "train_objs.txt").read_text() == (
        f"{class_a}/mesh_a/model.obj\n"
        f"{class_b}/mesh_d/models/model_normalized.obj\n"
    )
    assert (out_dir / "val_objs.txt").read_text() == f"{class_a}/mesh_b/model.obj\n"
    assert (out_dir / "test_objs.txt").read_text() == (
        f"{class_a}/mesh_c/model.obj\n"
        f"{class_b}/mesh_e/models/model_normalized.obj\n"
    )


def test_binary_hdf5_round_trip_preserves_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    obj_path = dataset_root / "001" / "obj"
    (obj_path / "depth").mkdir(parents=True)
    (obj_path / "normal").mkdir()
    (obj_path / "samples").mkdir()

    depth_bytes = b"depth-bytes"
    normal_bytes = b"normal-bytes"
    sample_bytes = b"sample-bytes"
    model_bytes = b"OFF\n0 0 0\n"
    params_bytes = b"PK\x03\x04"

    (obj_path / "depth" / "000.bin").write_bytes(depth_bytes)
    (obj_path / "normal" / "000.bin").write_bytes(normal_bytes)
    (obj_path / "samples" / "000.bin").write_bytes(sample_bytes)
    (obj_path / "model.off").write_bytes(model_bytes)
    (obj_path / "parameters.npz").write_bytes(params_bytes)

    packed_root = tmp_path / "packed"
    process_dataset_script.save_binary_hdf5(obj_path, out_dir=packed_root)

    hdf5_path = packed_root / "001" / "obj.hdf5"
    assert hdf5_path.is_file()

    restored_root = tmp_path / "restored"
    process_dataset_script.load_binary_hdf5(hdf5_path, out_dir=restored_root)

    restored_obj = restored_root / "001" / "obj"
    assert (restored_obj / "depth" / "000.bin").read_bytes() == depth_bytes
    assert (restored_obj / "normal" / "000.bin").read_bytes() == normal_bytes
    assert (restored_obj / "samples" / "000.bin").read_bytes() == sample_bytes
    assert (restored_obj / "model.off").read_bytes() == model_bytes
    assert (restored_obj / "parameters.npz").read_bytes() == params_bytes


def test_main_dispatches_split(monkeypatch: Any, tmp_path: Path) -> None:
    in_dir = tmp_path / "shapenet"
    args = SimpleNamespace(
        in_dir=in_dir,
        task="split",
        out_dir=None,
        val_size=0.1,
        test_size=0.2,
        n_jobs=1,
        verbose=False,
    )
    called: list[Any] = []

    monkeypatch.setattr(process_dataset_script.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(process_dataset_script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(process_dataset_script, "log_optional_dependency_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(process_dataset_script, "create_splits", lambda parsed_args: called.append(parsed_args))

    process_dataset_script.main()

    assert called == [args]


def test_main_pack_copies_metadata_and_runs_workers(monkeypatch: Any, tmp_path: Path) -> None:
    in_dir = tmp_path / "packed_src"
    class_dir = in_dir / "001"
    obj_dir = class_dir / "obj_a"
    obj_dir.mkdir(parents=True)
    (obj_dir / "model.off").write_text("off")
    (class_dir / "train.lst").write_text("obj_a\n")
    (in_dir / "taxonomy.json").write_text("{}")

    out_dir = tmp_path / "packed_out"
    args = SimpleNamespace(
        in_dir=in_dir,
        task="pack",
        out_dir=out_dir,
        val_size=0.1,
        test_size=0.2,
        n_jobs=1,
        verbose=False,
    )
    calls: list[tuple[Path, Path | None]] = []

    monkeypatch.setattr(process_dataset_script.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(process_dataset_script, "suppress_known_optional_dependency_warnings", lambda: None)
    monkeypatch.setattr(process_dataset_script, "log_optional_dependency_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(process_dataset_script, "tqdm_joblib", _null_context)
    monkeypatch.setattr(process_dataset_script, "Parallel", _ImmediateParallel)
    monkeypatch.setattr(process_dataset_script, "delayed", _delayed)
    monkeypatch.setattr(
        process_dataset_script,
        "save_binary_hdf5",
        lambda obj_path, out_dir=None: calls.append((Path(obj_path), Path(out_dir) if out_dir is not None else None)),
    )

    process_dataset_script.main()

    assert calls == [(obj_dir, out_dir)]
    assert (out_dir / "taxonomy.json").read_text() == "{}"
    assert (out_dir / "001" / "train.lst").read_text() == "obj_a\n"
