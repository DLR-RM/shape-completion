from pathlib import Path
from typing import Any, cast

import numpy as np
import pymeshlab
import pytest
from trimesh import Trimesh

from process.scripts import make_watertight as mw


def test_load_falls_back_when_primary_loader_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    calls: list[str] = []

    def fake_load_mesh(_path: Path, load_with: str | None = None):
        calls.append(load_with if load_with is not None else "default")
        if load_with == "pymeshlab":
            raise ValueError("primary loader failed")
        return vertices.copy(), faces.copy()

    monkeypatch.setattr(mw, "load_mesh", fake_load_mesh)

    out = mw.load(Path("mesh.obj"), loader="pymeshlab", return_type="dict")
    assert isinstance(out, dict)
    np.testing.assert_array_equal(out["vertices"], vertices)
    np.testing.assert_array_equal(out["faces"], faces)
    assert calls == ["pymeshlab", "default"]


def test_load_unknown_return_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown return type"):
        mw.load(Path("mesh.obj"), return_type="bad")


def test_normalize_trimesh_returns_translation_and_scale() -> None:
    mesh = Trimesh(
        vertices=np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
        validate=False,
    )

    out_mesh, translation, scale = mw.normalize(mesh, padding=0.1)

    assert isinstance(out_mesh, Trimesh)
    assert translation.shape == (3,)
    assert isinstance(scale, float)


def test_normalize_dict_uses_referenced_vertices() -> None:
    mesh = {
        "vertices": np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [100.0, 100.0, 100.0],  # unreferenced outlier; should not affect bounds
            ],
            dtype=np.float32,
        ),
        "faces": np.array([[0, 1, 2]], dtype=np.int64),
    }

    out_mesh, translation, scale = mw.normalize(mesh, padding=0.1)

    assert isinstance(out_mesh, dict)
    assert np.allclose(translation, np.array([-1.0, -1.0, 0.0], dtype=np.float32))
    assert np.isclose(scale, 1 / 2.4)
    transformed = out_mesh["vertices"][:3]
    assert transformed.max() <= 0.5 + 1e-6
    assert transformed.min() >= -0.5 - 1e-6


def test_process_preserves_input_type(monkeypatch: pytest.MonkeyPatch) -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    scripts = [Path("dummy.mlx")]

    def fake_apply_meshlab_filters(v: np.ndarray, f: np.ndarray, _scripts):
        return v.copy(), f.copy()

    monkeypatch.setattr(mw, "apply_meshlab_filters", fake_apply_meshlab_filters)

    mesh_dict = {"vertices": vertices.copy(), "faces": faces.copy()}
    out_dict = mw.process(mesh_dict, scripts)
    assert isinstance(out_dict, dict)
    np.testing.assert_array_equal(out_dict["vertices"], vertices)
    np.testing.assert_array_equal(out_dict["faces"], faces)

    mesh_tri = Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False, validate=False)
    out_tri = mw.process(mesh_tri, scripts)
    assert isinstance(out_tri, Trimesh)


def test_save_invalid_precision_raises(tmp_path: Path) -> None:
    mesh = Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
        validate=False,
    )
    with pytest.raises(ValueError, match="Invalid precision"):
        mw.save(mesh, tmp_path / "out.obj", precision=8)


def test_save_pymeshlab_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_save_mesh(
        path: Path, _vertices: np.ndarray, _faces: np.ndarray, save_with: str, digits: int | None = None
    ):
        calls.append(save_with)

    monkeypatch.setattr(mw, "save_mesh", fake_save_mesh)

    pymeshlab_mod = cast(Any, pymeshlab)
    meshset = pymeshlab_mod.MeshSet()
    meshset.add_mesh(
        pymeshlab_mod.Mesh(
            vertex_matrix=np.zeros((3, 3), dtype=np.float32), face_matrix=np.array([[0, 1, 2]], dtype=np.int32)
        )
    )
    mw.save(meshset, tmp_path / "out.obj", precision=32)
    assert calls == ["pymeshlab"]
