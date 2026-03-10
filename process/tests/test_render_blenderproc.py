"""Tests for render_blenderproc.py helper functions and add_kinect_sim.py.

Note: Tests that require blenderproc imports must be run via `blenderproc run`.
This file stubs BlenderProc modules for pure/helper function coverage.
Kinect simulation integration tests are in libs/tests/test_libkinect.py.
"""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import trimesh


@pytest.fixture()
def render_blenderproc_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import render_blenderproc with lightweight stubs for BlenderProc/bpy modules."""

    @contextmanager
    def _stdout_redirected(*_: Any, **__: Any):
        yield None

    for module_name in [name for name in list(sys.modules) if name == "blenderproc" or name.startswith("blenderproc.")]:
        sys.modules.pop(module_name, None)

    blenderproc = types.ModuleType("blenderproc")
    blenderproc.__path__ = []  # type: ignore[attr-defined]
    blenderproc_python = types.ModuleType("blenderproc.python")
    blenderproc_python.__path__ = []  # type: ignore[attr-defined]
    blenderproc_python_types = types.ModuleType("blenderproc.python.types")
    blenderproc_python_types.__path__ = []  # type: ignore[attr-defined]
    blenderproc_python_utility = types.ModuleType("blenderproc.python.utility")
    blenderproc_python_utility.__path__ = []  # type: ignore[attr-defined]

    blenderproc_any = cast(Any, blenderproc)
    blenderproc_any.sampler = types.SimpleNamespace(
        sphere=lambda *args, **kwargs: np.zeros(3),
        part_sphere=lambda *args, **kwargs: np.zeros(3),
        shell=lambda *args, **kwargs: np.zeros(3),
    )
    blenderproc_any.camera = types.SimpleNamespace(get_camera_pose=lambda frame: np.eye(4))
    blenderproc_any.types = types.SimpleNamespace(
        MeshObject=object,
        Material=object,
        Light=object,
    )
    blenderproc_any.python = blenderproc_python
    cast(Any, blenderproc_python).types = blenderproc_python_types
    cast(Any, blenderproc_python).utility = blenderproc_python_utility
    cast(Any, blenderproc_python_types).MeshObjectUtility = None
    cast(Any, blenderproc_python_utility).LabelIdMapping = None
    cast(Any, blenderproc_python_utility).Utility = None

    bpy = types.ModuleType("bpy")
    cast(Any, bpy).app = types.SimpleNamespace(version=(4, 0, 0))
    cast(Any, bpy).context = types.SimpleNamespace(
        scene=types.SimpleNamespace(
            objects=[],
            world=types.SimpleNamespace(node_tree=types.SimpleNamespace(nodes=[])),
            render=types.SimpleNamespace(),
            eevee=types.SimpleNamespace(),
            view_layers={"ViewLayer": types.SimpleNamespace()},
            frame_end=1,
        )
    )
    cast(Any, bpy).data = types.SimpleNamespace(images=types.SimpleNamespace(load=lambda *args, **kwargs: None))

    class _BpyTypes(types.SimpleNamespace):
        def __getattr__(self, _: str) -> type[object]:
            return object

    cast(Any, bpy).types = _BpyTypes(NodeSocket=object, Object=object)

    mesh_util = types.ModuleType("blenderproc.python.types.MeshObjectUtility")
    cast(Any, mesh_util).__package__ = "blenderproc.python.types"

    class _MeshObject:
        pass

    cast(Any, mesh_util).MeshObject = _MeshObject

    label_mapping = types.ModuleType("blenderproc.python.utility.LabelIdMapping")
    cast(Any, label_mapping).__package__ = "blenderproc.python.utility"

    class _LabelIdMapping:
        @staticmethod
        def from_dict(data: dict[str, int]) -> dict[str, int]:
            return data

    cast(Any, label_mapping).LabelIdMapping = _LabelIdMapping

    utility = types.ModuleType("blenderproc.python.utility.Utility")
    cast(Any, utility).__package__ = "blenderproc.python.utility"

    class _Utility:
        @staticmethod
        def get_the_one_node_with_type(*_: Any, **__: Any) -> Any:
            return types.SimpleNamespace(
                image=None,
                inputs={"Strength": types.SimpleNamespace(default_value=1.0)},
            )

    cast(Any, utility).Utility = _Utility
    cast(Any, utility).stdout_redirected = _stdout_redirected

    monkeypatch.setitem(sys.modules, "blenderproc", blenderproc)
    monkeypatch.setitem(sys.modules, "blenderproc.python", blenderproc_python)
    monkeypatch.setitem(sys.modules, "blenderproc.python.types", blenderproc_python_types)
    monkeypatch.setitem(sys.modules, "blenderproc.python.utility", blenderproc_python_utility)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "blenderproc.python.types.MeshObjectUtility", mesh_util)
    monkeypatch.setitem(sys.modules, "blenderproc.python.utility.LabelIdMapping", label_mapping)
    monkeypatch.setitem(sys.modules, "blenderproc.python.utility.Utility", utility)

    module_name = "process.scripts.render_blenderproc"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class _FakeMeshObject:
    def __init__(
        self,
        vertices: list[list[float]],
        faces: list[list[int]],
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        local2world: np.ndarray | None = None,
        name: str = "obj",
    ):
        self._mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self._scale = np.array(scale, dtype=np.float32)
        self._local2world = np.eye(4, dtype=np.float32) if local2world is None else local2world.astype(np.float32)
        self._name = name

    def mesh_as_trimesh(self) -> trimesh.Trimesh:
        return self._mesh.copy()

    def get_scale(self) -> np.ndarray:
        return self._scale

    def get_local2world_mat(self) -> np.ndarray:
        return self._local2world

    def get_name(self) -> str:
        return self._name


class _PoseObject:
    def __init__(self):
        self.location: np.ndarray | None = None
        self.rotation: np.ndarray | None = None

    def set_location(self, location: Any) -> None:
        self.location = np.asarray(location, dtype=np.float32)

    def set_rotation_euler(self, rotation: Any) -> None:
        self.rotation = np.asarray(rotation, dtype=np.float32)


class TestRenderBlenderprocHelpers:
    def test_convert_coordinates_identity(self, render_blenderproc_module: Any):
        points = np.random.rand(10, 3).astype(np.float32)

        result = render_blenderproc_module.convert_coordinates(points, "opengl", "opengl")
        np.testing.assert_array_equal(result, points)

        result = render_blenderproc_module.convert_coordinates(points, "opencv", "opencv")
        np.testing.assert_array_equal(result, points)

    def test_convert_coordinates_opengl_opencv(self, render_blenderproc_module: Any):
        points = np.array([[1, 2, 3]], dtype=np.float32)
        result = render_blenderproc_module.convert_coordinates(points, "opengl", "opencv")
        expected = np.array([[1, -2, -3]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_inv_trafo(self, render_blenderproc_module: Any):
        rotation = np.eye(3)
        rotation[:2, :2] = [[0, -1], [1, 0]]
        translation = np.array([1, 2, 3], dtype=np.float32)

        trafo = np.eye(4, dtype=np.float32)
        trafo[:3, :3] = rotation
        trafo[:3, 3] = translation

        inv = render_blenderproc_module.inv_trafo(trafo)
        np.testing.assert_array_almost_equal(trafo @ inv, np.eye(4), decimal=6)

    def test_convert_extrinsic(self, render_blenderproc_module: Any):
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, 3] = [0.1, -0.2, 0.3]

        converted = render_blenderproc_module.convert_extrinsic(extrinsic, "opengl", "opencv")
        restored = render_blenderproc_module.convert_extrinsic(converted, "opencv", "opengl")

        np.testing.assert_allclose(restored, extrinsic, atol=1e-6)

    def test_sample_truncnorm_fixed_scale(self, render_blenderproc_module: Any):
        sampled = render_blenderproc_module.sample_truncnorm((0.3, 0.3), size=5)
        np.testing.assert_allclose(sampled, np.full((5,), 0.3, dtype=np.float32))

    def test_sample_from_file_weighted(self, tmp_path: Path, render_blenderproc_module: Any):
        file_path = tmp_path / "objects.txt"
        file_path.write_text(
            "\n".join(
                [
                    "/tmp/root/class_a/obj1/model.obj",
                    "/tmp/root/class_a/obj2/model.obj",
                    "/tmp/root/class_b/obj3/model.obj",
                ]
            ),
            encoding="utf-8",
        )

        sampled = render_blenderproc_module.sample_from_file_weighted(file_path=file_path, n_samples=4)

        assert len(sampled) == 4
        assert all(isinstance(path, Path) for path in sampled)

    def test_camera_defaults(self, render_blenderproc_module: Any):
        camera = render_blenderproc_module.Camera(width=640, height=480, rotation=(0.0, 90.0, 180.0))

        assert camera.fx == 640
        assert camera.fy == 640
        assert camera.cx == 320
        assert camera.cy == 240
        np.testing.assert_allclose(
            camera.intrinsics,
            np.array([[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]]),
        )
        np.testing.assert_allclose(camera.rotation, np.deg2rad([0.0, 90.0, 180.0]))

    def test_camera_from_npz(self, tmp_path: Path, render_blenderproc_module: Any):
        intr = np.array([[500.0, 0.0, 256.0], [0.0, 501.0, 255.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        extr = np.eye(4, dtype=np.float32)
        extr[:3, 3] = [1.0, 2.0, 3.0]
        file_path = tmp_path / "camera.npz"
        np.savez(file_path, intrinsics=intr, extrinsics=extr)

        camera = render_blenderproc_module.Camera(file=file_path)

        np.testing.assert_array_equal(camera.intrinsics, intr)
        np.testing.assert_array_equal(camera.position, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(camera.rotation, np.eye(3, dtype=np.float32))

    def test_config_post_init_normalizes_engine_and_scale(self, tmp_path: Path, render_blenderproc_module: Any):
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            engine="eevee",
            max_samples="auto",
            scale=None,
        )

        assert cfg.engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}
        assert cfg.max_samples == 32
        assert callable(cfg.scale)
        np.testing.assert_allclose(np.asarray(cfg.scale(), dtype=np.float32), np.ones(3, dtype=np.float32))

    def test_config_scale_range_sampler(self, tmp_path: Path, render_blenderproc_module: Any):
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            scale=(0.2, 0.4),
            distort=0.1,
        )

        assert callable(cfg.scale)
        sampled = np.asarray(cfg.scale(), dtype=np.float32)
        assert sampled.shape == (3,)
        assert np.all(sampled > 0.0)

    def test_random_pose_fn_accepts_explicit_position_rotation(self, render_blenderproc_module: Any):
        obj = _PoseObject()

        render_blenderproc_module.random_pose_fn(
            obj=obj,
            position=(1.0, 2.0, 3.0),
            rotation=(10.0, 20.0, 30.0),
        )

        assert obj.location is not None
        assert obj.rotation is not None
        np.testing.assert_allclose(obj.location, np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(obj.rotation, np.deg2rad([10.0, 20.0, 30.0]), atol=1e-6)

    def test_volume_pose_fn_accepts_explicit_rotation(self, render_blenderproc_module: Any):
        obj = _PoseObject()

        render_blenderproc_module.volume_pose_fn(
            obj=obj,
            bounds_xy=(0.0, 0.0),
            bounds_z=(0.0, 0.0),
            rotation=(0.0, 90.0, 180.0),
            upright=False,
        )

        assert obj.location is not None
        assert obj.rotation is not None
        np.testing.assert_allclose(obj.location, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(obj.rotation, np.deg2rad([0.0, 90.0, 180.0]), atol=1e-6)

    def test_merge_scene_meshes(self, render_blenderproc_module: Any):
        obj1 = _FakeMeshObject(
            vertices=[[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            faces=[[0, 1, 2]],
            name="obj1",
        )
        obj2 = _FakeMeshObject(
            vertices=[[2, 0, 1], [3, 0, 1], [2, 1, 1]],
            faces=[[0, 1, 2]],
            name="obj2",
        )
        extrinsic = np.eye(4, dtype=np.float32)

        verts, faces = render_blenderproc_module.merge_scene_meshes(
            [obj1, obj2], surface=None, camera_extrinsic=extrinsic
        )

        assert verts.shape == (6, 3)
        assert faces.shape == (2, 3)
        np.testing.assert_array_equal(faces[0], [0, 1, 2])
        np.testing.assert_array_equal(faces[1], [3, 4, 5])

    def test_merge_scene_meshes_with_surface(self, render_blenderproc_module: Any):
        obj = _FakeMeshObject(
            vertices=[[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            faces=[[0, 1, 2]],
            name="obj",
        )
        surface = _FakeMeshObject(
            vertices=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
            faces=[[0, 1, 2], [0, 2, 3]],
            name="surface",
        )
        extrinsic = np.eye(4, dtype=np.float32)

        verts, faces = render_blenderproc_module.merge_scene_meshes([obj], surface=surface, camera_extrinsic=extrinsic)

        assert verts.shape == (7, 3)
        assert faces.shape == (3, 3)
        np.testing.assert_array_equal(faces[1], [3, 4, 5])
        np.testing.assert_array_equal(faces[2], [3, 5, 6])


class TestAddKinectSimScript:
    """Tests for the standalone add_kinect_sim.py script."""

    def test_create_plane_mesh(self):
        """Test that create_plane_mesh produces correct geometry."""
        from process.scripts.add_kinect_sim import create_plane_mesh

        plane = create_plane_mesh(size=5.0)

        assert len(plane.vertices) == 4
        assert len(plane.faces) == 2

        bounds = plane.bounds
        np.testing.assert_array_almost_equal(bounds[0], [-2.5, -2.5, 0])
        np.testing.assert_array_almost_equal(bounds[1], [2.5, 2.5, 0])

    def test_create_plane_mesh_custom_size(self):
        """Test plane mesh with custom size."""
        from process.scripts.add_kinect_sim import create_plane_mesh

        plane = create_plane_mesh(size=2.0)
        bounds = plane.bounds
        np.testing.assert_array_almost_equal(bounds[0], [-1.0, -1.0, 0])
        np.testing.assert_array_almost_equal(bounds[1], [1.0, 1.0, 0])

    def test_merge_meshes_to_camera(self):
        """Test merge_meshes_to_camera function."""
        from process.scripts.add_kinect_sim import merge_meshes_to_camera

        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])

        pose = np.eye(4)
        pose[:3, 3] = [5, 0, 0]

        scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        extrinsic = np.eye(4)

        verts, faces = merge_meshes_to_camera([mesh], [pose], [scale], extrinsic)

        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)

        expected = np.array([[5, 0, 0], [6, 0, 0], [5, 1, 0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(verts, expected)

    def test_merge_meshes_with_scale(self):
        """Test that scale is applied correctly."""
        from process.scripts.add_kinect_sim import merge_meshes_to_camera

        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])

        pose = np.eye(4)
        scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        extrinsic = np.eye(4)

        verts, _ = merge_meshes_to_camera([mesh], [pose], [scale], extrinsic)

        expected = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(verts, expected)

    def test_merge_meshes_with_surface(self):
        """Test that surface mesh is included."""
        from process.scripts.add_kinect_sim import merge_meshes_to_camera

        obj_mesh = trimesh.Trimesh(vertices=[[0, 0, 1], [1, 0, 1], [0, 1, 1]], faces=[[0, 1, 2]])
        surface = trimesh.Trimesh(
            vertices=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
            faces=[[0, 1, 2], [0, 2, 3]],
        )

        pose = np.eye(4)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        extrinsic = np.eye(4)

        verts, faces = merge_meshes_to_camera([obj_mesh], [pose], [scale], extrinsic, surface_mesh=surface)

        assert verts.shape == (7, 3)
        assert faces.shape == (3, 3)
        np.testing.assert_array_equal(faces[1], [3, 4, 5])
        np.testing.assert_array_equal(faces[2], [3, 5, 6])
