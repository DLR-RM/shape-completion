from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from ..src import tabletop as tabletop_module
from ..src.tabletop import TableTop

pytestmark = pytest.mark.integration


def test_tabletop_dataset_loads():
    data_dir = os.environ.get("TABLETOP_ROOT")
    if not data_dir:
        pytest.skip("Set TABLETOP_ROOT to run TableTop integration tests.")

    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        pytest.skip(f"TableTop dataset directory not found: {data_dir}")

    data_dir_3d_env = os.environ.get("SHAPENET_V1_FUSED_SIMPLE")
    data_dir_3d = Path(data_dir_3d_env).expanduser().resolve() if data_dir_3d_env else None
    if data_dir_3d is not None and not data_dir_3d.is_dir():
        data_dir_3d = None

    tabletop_cls = cast(Any, TableTop)
    ds = tabletop_cls(
        data_dir=data_dir,
        data_dir_3d=data_dir_3d,
        split="train",
    )

    assert len(ds) > 0
    item = ds[0]
    assert "inputs" in item


def test_tabletop_init_accepts_str_coco_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "tabletop"
    split_dir = data_dir / "train"
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "0.png").touch()
    (split_dir / "0.hdf5").touch()

    class FakeCocoInstanceSegmentation:
        def __init__(
            self,
            data_dir: Path,
            split: str = "train",
            train_dir: str = "",
            val_dir: str = "",
            test_dir: str = "",
            **_: Any,
        ):
            if split == "train":
                root = data_dir / train_dir
            elif split == "val":
                root = data_dir / val_dir
            else:
                root = data_dir / test_dir
            self.root = str(root)

        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(tabletop_module, "CocoInstanceSegmentation", FakeCocoInstanceSegmentation)

    tabletop_cls = cast(Any, TableTop)
    ds = tabletop_cls(data_dir=data_dir, split="train")

    assert len(ds.coco_ds) == 1
    assert isinstance(ds.coco_ds[0].root, str)
    assert len(ds) == 1


def test_tabletop_loads_stereo_depth_channel(tmp_path: Path):
    expected = np.asarray([[0.5, 0.0], [0.7, 0.8]], dtype=np.float32)
    metadata = {
        "intrinsic": np.eye(3).tolist(),
        "extrinsic": np.eye(4).tolist(),
    }
    with h5py.File(tmp_path / "0.hdf5", "w") as output:
        output.create_dataset("data", data=json.dumps(metadata).encode("utf-8"))
        output.create_dataset("stereo_depth", data=expected)

    dataset = cast(Any, TableTop).__new__(TableTop)
    dataset.load_depth = "stereo_depth"
    dataset.load_normals = False

    actual = dataset._load_data(tmp_path, 0)

    np.testing.assert_array_equal(actual["depth"], expected)
