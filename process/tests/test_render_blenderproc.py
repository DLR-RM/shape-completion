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
        view_layer=types.SimpleNamespace(update=lambda: None),
        scene=types.SimpleNamespace(
            objects=[],
            world=types.SimpleNamespace(node_tree=types.SimpleNamespace(nodes=[])),
            render=types.SimpleNamespace(),
            eevee=types.SimpleNamespace(),
            view_layers={"ViewLayer": types.SimpleNamespace()},
            frame_end=1,
        ),
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
        hidden: bool = False,
    ):
        self._mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self._scale = np.array(scale, dtype=np.float32)
        self._local2world = np.eye(4, dtype=np.float32) if local2world is None else local2world.astype(np.float32)
        self._name = name
        self.blender_obj = types.SimpleNamespace(hide_render=hidden)

    def mesh_as_trimesh(self) -> trimesh.Trimesh:
        return self._mesh.copy()

    def get_scale(self) -> np.ndarray:
        return self._scale

    def get_local2world_mat(self) -> np.ndarray:
        return self._local2world

    def get_name(self) -> str:
        return self._name


class _PoseObject:
    def __init__(self, source_asset_path: str | None = None):
        self.location: np.ndarray | None = None
        self.rotation: np.ndarray | None = None
        self.custom_properties: dict[str, Any] = {}
        if source_asset_path is not None:
            self.custom_properties["source_asset_path"] = source_asset_path

    def has_cp(self, key: str) -> bool:
        return key in self.custom_properties

    def get_cp(self, key: str) -> Any:
        return self.custom_properties[key]

    def set_location(self, location: Any) -> None:
        self.location = np.asarray(location, dtype=np.float32)

    def set_rotation_euler(self, rotation: Any) -> None:
        self.rotation = np.asarray(rotation, dtype=np.float32)


class TestRenderBlenderprocHelpers:
    def test_surface_size_must_be_positive(self, tmp_path: Path, render_blenderproc_module: Any):
        with pytest.raises(ValueError, match="surface_size must be positive"):
            render_blenderproc_module.Config(
                object_path=tmp_path / "dummy.obj",
                output_dir=tmp_path / "out",
                surface_size=0.0,
            )
        with pytest.raises(ValueError, match="surface_thickness must be positive"):
            render_blenderproc_module.Config(
                object_path=tmp_path / "dummy.obj",
                output_dir=tmp_path / "out",
                surface="table",
                surface_thickness=0.0,
            )

        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
        )
        assert cfg.surface_size == 5.0

    def test_right_stereo_camera_pose_uses_camera_local_x(self, render_blenderproc_module: Any):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 0] = [0.0, 1.0, 0.0]

        right = render_blenderproc_module.right_stereo_camera_pose(pose, 0.05)

        np.testing.assert_allclose(right[:3, 3], [0.0, 0.05, 0.0])
        np.testing.assert_allclose(right[:3, :3], pose[:3, :3])

    def test_stereo_depth_config_validation(self, tmp_path: Path, render_blenderproc_module: Any):
        base = {"object_path": tmp_path / "dummy.obj", "output_dir": tmp_path / "out"}
        cfg = render_blenderproc_module.Config(**base, stereo_depth=True)
        assert cfg.stereo_baseline == 0.05
        assert cfg.stereo_num_disparities == 128

        with pytest.raises(ValueError, match="positive and divisible by 16"):
            render_blenderproc_module.Config(**base, stereo_depth=True, stereo_num_disparities=100)

    def test_kinect_darkness_threshold_validation(self, tmp_path: Path, render_blenderproc_module: Any) -> None:
        base = {"object_path": tmp_path / "dummy.obj", "output_dir": tmp_path / "out"}
        cfg = render_blenderproc_module.Config(**base, kinect=True)
        assert cfg.kinect_darkness_threshold == 15

        with pytest.raises(ValueError, match="between 0 and 255"):
            render_blenderproc_module.Config(
                **base,
                kinect=True,
                kinect_darkness_threshold=256,
            )

    def test_extract_stereo_depth_keeps_canonical_frames(
        self,
        tmp_path: Path,
        render_blenderproc_module: Any,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_sgm(pairs: list[np.ndarray], **kwargs: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
            captured["pairs"] = pairs
            captured["kwargs"] = kwargs
            depths = [
                np.asarray([[0.5, 2.0], [np.inf, -1.0]], dtype=np.float32),
                np.full((2, 2), 0.8, dtype=np.float32),
            ]
            return depths, [np.zeros((2, 2), dtype=np.float32) for _ in depths]

        render_blenderproc_module.bproc.postprocessing = types.SimpleNamespace(stereo_global_matching=fake_sgm)
        data = {
            "colors": [np.full((2, 2, 4), frame, dtype=np.uint8) for frame in range(4)],
            "depth": [np.full((2, 2), frame, dtype=np.float32) for frame in range(4)],
        }
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            stereo_depth=True,
        )

        render_blenderproc_module.extract_stereo_depth(data, 2, cfg)

        assert len(data["colors"]) == len(data["depth"]) == len(data["stereo_depth"]) == 2
        assert captured["pairs"][0].shape == (2, 2, 2, 3)
        assert captured["kwargs"]["disparity_filter"] is False
        assert captured["kwargs"]["depth_completion"] is False
        assert render_blenderproc_module.bpy.context.scene.frame_end == 2
        np.testing.assert_array_equal(data["stereo_depth"][0], [[0.5, 0.0], [0.0, 0.0]])

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

        assert len(sampled) == 3
        assert all(isinstance(path, Path) for path in sampled)
        assert len(set(sampled)) == len(sampled)

    def test_loader_category_name_from_path(self, render_blenderproc_module: Any):
        glb_path = Path("/tmp/render_assets/02876657/abcdef/model.glb")
        obj_path = Path("/tmp/render_assets/02876657/abcdef/meshes/model.obj")

        assert render_blenderproc_module.loader_category_name_from_path(glb_path) == ("02876657", "abcdef")
        assert render_blenderproc_module.loader_category_name_from_path(obj_path) == ("02876657", "abcdef")

    def test_node_output_by_name_falls_back_to_iteration(self, render_blenderproc_module: Any):
        class _Outputs(list):
            def __getitem__(self, key: Any) -> Any:
                if isinstance(key, str):
                    raise KeyError(key)
                return super().__getitem__(key)

        expected = types.SimpleNamespace(name="IndexOB", identifier="IndexOB")
        node = types.SimpleNamespace(outputs=_Outputs([types.SimpleNamespace(name="Image"), expected]))

        assert render_blenderproc_module._node_output_by_name(node, "IndexOB") is expected

    def test_enabled_node_output_by_name_rejects_disabled_socket(self, render_blenderproc_module: Any):
        node = types.SimpleNamespace(
            outputs=[types.SimpleNamespace(name="IndexOB", identifier="IndexOB", enabled=False)]
        )

        with pytest.raises(KeyError):
            render_blenderproc_module._enabled_node_output_by_name(node, "IndexOB")

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

    def test_config_preserves_zero_objects_for_empty_scene(self, tmp_path: Path, render_blenderproc_module: Any):
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "objects.txt",
            output_dir=tmp_path / "out",
            num_objects=0,
        )

        assert cfg.num_objects == 0

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

    def test_config_accepts_real_depth_randomization_controls(
        self,
        tmp_path: Path,
        render_blenderproc_module: Any,
    ) -> None:
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            lights=(1, 4),
            light_type="AREA",
            light_radius=(0.6, 1.2),
            light_elevation=(35.0, 85.0),
            light_energy=(250.0, 600.0),
            light_size=(0.25, 0.75),
            light_color_mode="neutral",
            surface_material_profile="tabletop",
            world_background_strength=0.05,
            cc_material_assets=("Wood", "Plastic"),
            scene_metadata={"randomization": {"profile": "real_depth_v1"}},
        )

        assert cfg.light_type == "AREA"
        assert cfg.light_elevation == (35.0, 85.0)
        assert cfg.surface_material_profile == "tabletop"
        assert cfg.scene_metadata == {"randomization": {"profile": "real_depth_v1"}}

    def test_config_validates_upright_path_rotation_overrides(
        self,
        tmp_path: Path,
        render_blenderproc_module: Any,
    ) -> None:
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "objects.txt",
            output_dir=tmp_path / "out",
            upright_x_rotation_path_overrides={"omniobject3d": 90.0},
        )
        assert cfg.upright_x_rotation_path_overrides == {"omniobject3d": 90.0}

        with pytest.raises(ValueError, match="non-empty path tokens"):
            render_blenderproc_module.Config(
                object_path=tmp_path / "objects.txt",
                output_dir=tmp_path / "out",
                upright_x_rotation_path_overrides={"": 90.0},
            )

    def test_config_accepts_optional_local_lights_with_hdri(
        self,
        tmp_path: Path,
        render_blenderproc_module: Any,
    ) -> None:
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            lights=(0, 3),
            hdri_path=tmp_path / "haven",
            hdri_assets=("art_studio",),
        )

        assert cfg.lights == (0, 3)
        assert cfg.hdri_assets == ("art_studio",)

    def test_sample_hdri_path_respects_allowlist(
        self,
        tmp_path: Path,
        render_blenderproc_module: Any,
    ) -> None:
        hdri = tmp_path / "haven" / "hdris" / "art_studio" / "art_studio_2k.hdr"
        hdri.parent.mkdir(parents=True)
        hdri.touch()
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            hdri_path=tmp_path / "haven",
            hdri_assets=("art_studio",),
        )

        assert render_blenderproc_module.sample_hdri_path(cfg) == str(hdri)

    def test_tabletop_surface_material_stays_within_bounded_ranges(
        self,
        render_blenderproc_module: Any,
    ) -> None:
        values: dict[str, Any] = {}

        class _Material:
            def set_principled_shader_value(self, key: str, value: Any) -> None:
                values[key] = value

        render_blenderproc_module.randomize_tabletop_surface(_Material())

        assert len(values["Base Color"]) == 4
        assert 0.2 <= values["Specular IOR Level"] <= 0.5
        assert 0.35 <= values["Roughness"] <= 0.85
        assert values["Metallic"] == 0.0

    def test_ensure_render_material_only_fills_missing_slot(
        self,
        render_blenderproc_module: Any,
    ) -> None:
        values: dict[str, Any] = {}

        class _Material:
            def set_principled_shader_value(self, key: str, value: Any) -> None:
                values[key] = value

        class _Object:
            def __init__(self, has_materials: bool) -> None:
                self._has_materials = has_materials
                self.created_names: list[str] = []
                self.custom_properties: dict[str, Any] = {}

            def has_materials(self) -> bool:
                return self._has_materials

            def get_name(self) -> str:
                return "asset"

            def new_material(self, name: str) -> _Material:
                self.created_names.append(name)
                return _Material()

            def set_cp(self, key: str, value: Any) -> None:
                self.custom_properties[key] = value

        textured = _Object(has_materials=True)
        render_blenderproc_module.ensure_render_material(textured)
        assert textured.created_names == []

        materialless = _Object(has_materials=False)
        render_blenderproc_module.ensure_render_material(materialless)
        assert materialless.created_names == ["asset default"]
        assert values == {
            "Base Color": [0.35, 0.38, 0.40, 1.0],
            "Specular IOR Level": 0.3,
            "Roughness": 0.6,
            "Metallic": 0.0,
        }
        assert materialless.custom_properties == {"materialless_fallback_profile": "fixed"}

    def test_industrial_material_fallback_is_bounded_and_deterministic(
        self,
        render_blenderproc_module: Any,
    ) -> None:
        class _Material:
            def __init__(self) -> None:
                self.values: dict[str, Any] = {}

            def set_principled_shader_value(self, key: str, value: Any) -> None:
                self.values[key] = value

        class _Object:
            def __init__(self) -> None:
                self.material = _Material()
                self.custom_properties: dict[str, Any] = {}

            def has_materials(self) -> bool:
                return False

            def get_name(self) -> str:
                return "abc asset"

            def new_material(self, _name: str) -> _Material:
                return self.material

            def set_cp(self, key: str, value: Any) -> None:
                self.custom_properties[key] = value

        render_blenderproc_module.random.seed(17)
        render_blenderproc_module.np.random.seed(17)
        first = _Object()
        render_blenderproc_module.ensure_render_material(first, profile="industrial")

        render_blenderproc_module.random.seed(17)
        render_blenderproc_module.np.random.seed(17)
        second = _Object()
        render_blenderproc_module.ensure_render_material(second, profile="industrial")

        assert first.material.values == second.material.values
        values = first.material.values
        assert len(values["Base Color"]) == 4
        assert 0.2 <= values["Specular IOR Level"] <= 0.5
        assert 0.25 <= values["Roughness"] <= 0.85
        assert values["Metallic"] in {0.0, 1.0}
        assert first.custom_properties == {"materialless_fallback_profile": "industrial"}

    def test_materialless_profile_resolution_is_source_specific(
        self,
        render_blenderproc_module: Any,
    ) -> None:
        overrides = {"/abc/": "industrial"}

        assert (
            render_blenderproc_module.materialless_object_profile_for_path(
                "/datasets/abc/object/model.obj",
                overrides,
            )
            == "industrial"
        )
        assert (
            render_blenderproc_module.materialless_object_profile_for_path(
                "/datasets/omniobject3d/object/model.glb",
                overrides,
            )
            == "fixed"
        )

    def test_packed_physics_preserves_explicit_settings(self, tmp_path: Path, render_blenderproc_module: Any):
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            scene="packed",
            physics=True,
            containment_walls=True,
            collision_shape="MESH",
            collision_margin=0.002,
            rigidbody_friction=0.8,
            linear_damping=0.3,
            angular_damping=0.4,
            solidify=0.0,
            physics_min_simulation_time=1.0,
            physics_max_simulation_time=6.0,
            physics_check_interval=0.25,
            physics_substeps_per_frame=12,
            physics_solver_iters=14,
        )

        assert cfg.physics
        assert cfg.containment_walls
        assert cfg.collision_shape == "MESH"
        assert cfg.solidify == 0.0
        assert cfg.physics_max_simulation_time == 6.0
        assert not cfg.correct_floor_penetration

    def test_pile_defaults_use_fast_sequential_hulls(self, tmp_path: Path, render_blenderproc_module: Any):
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            scene="pile",
            physics=True,
            scale=(0.08, 0.22),
        )

        assert cfg.placement == "sequential"
        assert cfg.collision_shape == "CONVEX_HULL"
        assert cfg.solidify == 0.0
        assert cfg.containment_walls
        assert not cfg.upright
        assert cfg.pile_batch_size == 2
        assert cfg.pile_drop_margin == (0.01, 0.03)
        assert cfg.pile_settle_time == 0.5
        assert cfg.pile_wall_margin == 0.01
        assert cfg.pile_center_radius == 0.0
        assert cfg.rigidbody_friction == 0.8

    def test_pile_preserves_explicit_wall_override(self, tmp_path: Path, render_blenderproc_module: Any):
        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            scene="pile",
            physics=True,
            containment_walls=False,
        )

        assert not cfg.containment_walls

    def test_pile_drop_pose_fits_rotated_bounds_inside_region(
        self,
        tmp_path: Path,
        render_blenderproc_module: Any,
    ) -> None:
        class _DropObject:
            def __init__(self) -> None:
                self.location = np.zeros(3)
                self.rotation = np.zeros(3)
                self.custom_properties = {
                    "placement_bounds_min": [-0.05, -0.04, -0.03],
                    "placement_bounds_max": [0.05, 0.04, 0.03],
                }

            def has_cp(self, key: str) -> bool:
                return key in self.custom_properties

            def get_cp(self, key: str) -> Any:
                return self.custom_properties[key]

            def get_scale(self) -> np.ndarray:
                return np.ones(3)

            def set_rotation_euler(self, value: Any) -> None:
                self.rotation = np.asarray(value, dtype=np.float64)

            def set_location(self, value: Any) -> None:
                self.location = np.asarray(value, dtype=np.float64)

        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            scene="pile",
            physics=True,
            rotation=False,
            pile_center_radius=0.0,
        )
        obj = _DropObject()

        bounds = render_blenderproc_module.sample_pile_drop_pose(
            obj,
            cfg,
            bounds_xy=(-0.12, 0.12),
            target_bottom=0.2,
        )

        assert bounds is not None
        assert bounds[:, 0].min() >= -0.12
        assert bounds[:, 0].max() <= 0.12
        assert bounds[:, 1].min() >= -0.12
        assert bounds[:, 1].max() <= 0.12
        assert bounds[:, 2].min() == pytest.approx(0.2)
        np.testing.assert_allclose(bounds[:, :2].mean(axis=0), np.zeros(2), atol=1e-7)

    def test_rigidbody_uses_configured_contact_parameters(self, tmp_path: Path, render_blenderproc_module: Any):
        calls: list[dict[str, Any]] = []

        class _RigidBodyObject:
            def enable_rigidbody(self, **kwargs: Any) -> None:
                calls.append(kwargs)

        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            collision_shape="BOX",
            collision_margin=0.002,
            rigidbody_friction=0.7,
            linear_damping=0.25,
            angular_damping=0.35,
        )

        render_blenderproc_module.enable_rigidbody_with_decomposition(_RigidBodyObject(), cfg)

        assert calls == [
            {
                "active": True,
                "collision_shape": "BOX",
                "collision_margin": 0.002,
                "friction": 0.7,
                "linear_damping": 0.25,
                "angular_damping": 0.35,
            }
        ]

    def test_decomposition_cache_key_is_stable_and_parameter_bound(
        self,
        render_blenderproc_module: Any,
    ) -> None:
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])

        first = render_blenderproc_module.decomposition_cache_key(
            vertices,
            faces,
            method="coacd",
            parameters={"format": 2, "threshold": 0.05},
        )
        repeated = render_blenderproc_module.decomposition_cache_key(
            vertices.copy(),
            faces.copy(),
            method="coacd",
            parameters={"threshold": 0.05, "format": 2},
        )
        different_threshold = render_blenderproc_module.decomposition_cache_key(
            vertices,
            faces,
            method="coacd",
            parameters={"format": 2, "threshold": 0.02},
        )

        assert first == repeated
        assert len(first) == 64
        assert first != different_threshold

    def test_aabb_surface_placement_separates_collision_proxies(
        self,
        tmp_path: Path,
        render_blenderproc_module: Any,
    ) -> None:
        class _PlacementObject:
            def __init__(self, extents: tuple[float, float, float]) -> None:
                self.blender_obj = types.SimpleNamespace(hide_render=False, hide_viewport=False)
                self.location = np.zeros(3)
                self.rotation = np.zeros(3)
                self.corners = trimesh.creation.box(extents=extents).bounding_box.vertices
                self.custom_properties: dict[str, Any] = {}

            def has_cp(self, key: str) -> bool:
                return key in self.custom_properties

            def get_cp(self, key: str) -> Any:
                return self.custom_properties[key]

            def set_location(self, value: Any) -> None:
                self.location = np.asarray(value, dtype=np.float64)

            def set_rotation_euler(self, value: Any) -> None:
                self.rotation = np.asarray(value, dtype=np.float64)

            def get_scale(self) -> np.ndarray:
                return np.ones(3)

            def get_bound_box(self, local_coords: bool = False) -> np.ndarray:
                if local_coords:
                    return self.corners.copy()
                transform = trimesh.transformations.euler_matrix(
                    float(self.rotation[0]),
                    float(self.rotation[1]),
                    float(self.rotation[2]),
                )
                points = trimesh.transform_points(self.corners, transform)
                return points + self.location

        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            scene="packed",
            placement="surface_aabb",
            physics=True,
            solidify=0.0,
            upright_x_rotation=0.0,
            placement_min_distance=0.005,
        )
        assert cfg.rigidbody_friction == 0.5
        objects = [
            _PlacementObject((0.06, 0.05, 0.08)),
            _PlacementObject((0.12, 0.08, 0.10)),
            _PlacementObject((0.08, 0.07, 0.06)),
            _PlacementObject((0.10, 0.07, 0.09)),
        ]
        for obj in objects:
            obj.custom_properties["placement_bounds_min"] = obj.corners.min(axis=0).tolist()
            obj.custom_properties["placement_bounds_max"] = obj.corners.max(axis=0).tolist()

        placed = render_blenderproc_module.place_objects_surface_aabb(
            objects,
            cfg,
            bounds_xy=(-0.25, 0.25),
            spawn_height=(0.01, 0.05),
        )

        assert len(placed) == 4
        footprint_areas = [float(np.ptp(obj.corners, axis=0)[:2].prod()) for obj in placed]
        assert footprint_areas == sorted(footprint_areas, reverse=True)
        bounds = [np.asarray(obj.get_bound_box()) for obj in placed]
        for index, first in enumerate(bounds):
            assert float(first[:, 2].min()) >= 0.01 - 1e-7
            assert float(first[:, 2].min()) <= 0.05 + 1e-7
            for second in bounds[index + 1 :]:
                first_min, first_max = first[:, :2].min(axis=0), first[:, :2].max(axis=0)
                second_min, second_max = second[:, :2].min(axis=0), second[:, :2].max(axis=0)
                assert (
                    first_max[0] + cfg.placement_min_distance <= second_min[0]
                    or second_max[0] + cfg.placement_min_distance <= first_min[0]
                    or first_max[1] + cfg.placement_min_distance <= second_min[1]
                    or second_max[1] + cfg.placement_min_distance <= first_min[1]
                )

    def test_render_hidden_tracks_placement_rejection(self, render_blenderproc_module: Any) -> None:
        visible = types.SimpleNamespace(blender_obj=types.SimpleNamespace(hide_render=False))
        hidden = types.SimpleNamespace(blender_obj=types.SimpleNamespace(hide_render=True))

        assert not render_blenderproc_module.is_render_hidden(visible)
        assert render_blenderproc_module.is_render_hidden(hidden)

    def test_hdri_rotation_randomizes_scene_azimuth(
        self,
        monkeypatch: pytest.MonkeyPatch,
        render_blenderproc_module: Any,
    ) -> None:
        monkeypatch.setattr(render_blenderproc_module.np.random, "uniform", lambda *_: 1.25)

        rotation = render_blenderproc_module.sample_hdri_rotation_euler(Path("scene.txt"), True)

        np.testing.assert_allclose(rotation, [0.0, 0.0, 1.25])

    def test_hdri_rotation_preserves_obj_axis_conversion(self, render_blenderproc_module: Any) -> None:
        rotation = render_blenderproc_module.sample_hdri_rotation_euler(Path("asset.obj"), False)

        np.testing.assert_allclose(rotation, [np.pi / 2, 0.0, 0.0])

    def test_floor_correction_lifts_only_penetrating_visible_objects(self, render_blenderproc_module: Any) -> None:
        class _FloorObject:
            def __init__(
                self,
                name: str,
                min_z: float,
                hidden: bool = False,
                bound_box_min_z: float | None = None,
            ) -> None:
                self.name = name
                self.location = np.zeros(3)
                self.base_min_z = min_z
                self.bound_box_min_z = min_z if bound_box_min_z is None else bound_box_min_z
                self.blender_obj = types.SimpleNamespace(hide_render=hidden)
                self.custom_properties: dict[str, float] = {}

            def get_bound_box(self) -> np.ndarray:
                low = np.array([-0.1, -0.1, self.bound_box_min_z]) + self.location
                high = np.array([0.1, 0.1, 0.1]) + self.location
                return np.array(
                    [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
                )

            def mesh_as_trimesh(self) -> trimesh.Trimesh:
                vertices = np.array(
                    [
                        [-0.1, -0.1, self.base_min_z],
                        [0.1, -0.1, self.base_min_z],
                        [0.0, 0.1, 0.1],
                    ]
                )
                return trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)

            def get_scale(self) -> np.ndarray:
                return np.ones(3)

            def get_local2world_mat(self) -> np.ndarray:
                transform = np.eye(4)
                transform[:3, 3] = self.location
                return transform

            def get_location(self) -> np.ndarray:
                return self.location.copy()

            def set_location(self, value: Any) -> None:
                self.location = np.asarray(value, dtype=np.float64)

            def set_cp(self, key: str, value: float) -> None:
                self.custom_properties[key] = value

            def get_name(self) -> str:
                return self.name

        penetrating = _FloorObject("penetrating", -0.025)
        touching = _FloorObject("touching", 0.0)
        hidden = _FloorObject("hidden", -1.0, hidden=True)
        false_corner_penetration = _FloorObject("false_corner", 0.01, bound_box_min_z=-0.05)

        corrections = render_blenderproc_module.correct_objects_below_floor(
            [penetrating, touching, hidden, false_corner_penetration]
        )

        assert corrections == {"penetrating": pytest.approx(0.025)}
        assert render_blenderproc_module.mesh_object_world_bounds(penetrating)[0][2] == pytest.approx(0.0)
        assert touching.location[2] == 0.0
        assert hidden.location[2] == 0.0
        assert false_corner_penetration.location[2] == 0.0

    def test_post_physics_outlier_is_hidden_before_floor_correction(
        self,
        tmp_path: Path,
        render_blenderproc_module: Any,
    ) -> None:
        class _LaunchedObject:
            def __init__(self) -> None:
                self.blender_obj = types.SimpleNamespace(hide_render=False, hide_viewport=False)
                self.disabled = False
                self.custom_properties: dict[str, str] = {}

            def get_bound_box(self) -> np.ndarray:
                return np.array([[0.8, 0.0, -10.0], [0.9, 0.1, -9.9]])

            def has_rigidbody_enabled(self) -> bool:
                return True

            def disable_rigidbody(self) -> None:
                self.disabled = True

            def set_cp(self, key: str, value: str) -> None:
                self.custom_properties[key] = value

            def get_name(self) -> str:
                return "launched"

        cfg = render_blenderproc_module.Config(
            object_path=tmp_path / "dummy.obj",
            output_dir=tmp_path / "out",
            scene="pile",
            physics=True,
        )
        obj = _LaunchedObject()

        rejected = render_blenderproc_module.reject_unstable_physics_objects([obj], cfg)

        assert rejected == {"launched": "xy_escape"}
        assert obj.disabled
        assert obj.blender_obj.hide_render
        assert obj.custom_properties["physics_rejection_reason"] == "xy_escape"

    def test_normalization_parameters_center_off_origin_mesh(self, render_blenderproc_module: Any):
        bounds = np.array([[2.0, -4.0, 8.0], [6.0, 2.0, 10.0]], dtype=np.float64)

        translation, scale = render_blenderproc_module.normalization_parameters(bounds)
        normalized = bounds * scale + translation

        np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(3), atol=1e-7)
        assert np.max(normalized[1] - normalized[0]) == pytest.approx(1.0)

    def test_bake_world_transform_copies_shared_mesh_data(
        self,
        render_blenderproc_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _MeshData:
            def __init__(self) -> None:
                self.transform_value: np.ndarray | None = None

            def copy(self) -> _MeshData:
                return _MeshData()

            def transform(self, value: np.ndarray) -> None:
                self.transform_value = value

        mathutils = types.ModuleType("mathutils")
        cast(Any, mathutils).Matrix = types.SimpleNamespace(Identity=lambda size: np.eye(size))
        monkeypatch.setitem(sys.modules, "mathutils", mathutils)
        original_data = _MeshData()
        world = np.arange(16, dtype=np.float64).reshape(4, 4)
        obj = types.SimpleNamespace(data=original_data, matrix_world=world, parent=object())

        render_blenderproc_module.bake_blender_object_world_transform(obj)

        assert obj.data is not original_data
        np.testing.assert_array_equal(obj.data.transform_value, world)
        assert obj.parent is None
        np.testing.assert_array_equal(obj.matrix_world, np.eye(4))

    def test_upright_pose_supports_z_up_assets(self, render_blenderproc_module: Any):
        obj = _PoseObject()

        render_blenderproc_module.upper_region_pose_fn(
            obj=obj,
            surface=None,
            rotation=True,
            upright=True,
            upright_x_rotation=0.0,
        )

        assert obj.rotation is not None
        assert obj.rotation[0] == pytest.approx(0.0)

    def test_upright_pose_applies_source_path_rotation_override(
        self,
        render_blenderproc_module: Any,
    ) -> None:
        obj = _PoseObject("/datasets/omniobject3d/bottle/model.glb")

        render_blenderproc_module.upper_region_pose_fn(
            obj=obj,
            surface=None,
            rotation=True,
            upright=True,
            upright_x_rotation=0.0,
            upright_x_rotation_path_overrides={"omniobject3d": 90.0},
        )

        assert obj.rotation is not None
        assert obj.rotation[0] == pytest.approx(np.pi / 2)

    def test_upright_path_rotation_rejects_conflicting_matches(
        self,
        render_blenderproc_module: Any,
    ) -> None:
        with pytest.raises(ValueError, match="Conflicting upright rotation"):
            render_blenderproc_module.upright_x_rotation_for_path(
                "/datasets/omniobject3d/bottle/model.glb",
                0.0,
                {"omniobject3d": 90.0, "bottle": -90.0},
            )

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

    def test_merge_scene_meshes_excludes_render_hidden_objects(self, render_blenderproc_module: Any):
        visible = _FakeMeshObject(
            vertices=[[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            faces=[[0, 1, 2]],
            name="visible",
        )
        hidden = _FakeMeshObject(
            vertices=[[2, 0, 1], [3, 0, 1], [2, 1, 1]],
            faces=[[0, 1, 2]],
            name="hidden",
            hidden=True,
        )

        verts, faces = render_blenderproc_module.merge_scene_meshes(
            [visible, hidden],
            surface=None,
            camera_extrinsic=np.eye(4, dtype=np.float32),
        )

        assert verts.shape == (3, 3)
        assert faces.shape == (1, 3)
        np.testing.assert_array_equal(faces[0], [0, 1, 2])

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
