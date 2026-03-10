import numpy as np
import pytest
from trimesh import Trimesh

from process.src import fill as fill_mod


def test_voxelize_returns_bool_grid() -> None:
    mesh = Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
        validate=False,
    )

    voxel = fill_mod.voxelize(mesh, resolution=8)

    assert voxel.shape == (8, 8, 8)
    assert voxel.dtype == np.bool_
    assert voxel.any()


def test_kaolin_pipeline_raises_when_kaolin_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fill_mod, "kaolin", None)
    mesh = {
        "vertices": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "faces": np.array([[0, 1, 2]], dtype=np.int64),
    }

    with pytest.raises(ImportError, match="kaolin"):
        fill_mod.kaolin_pipeline(mesh, resolution=16)
