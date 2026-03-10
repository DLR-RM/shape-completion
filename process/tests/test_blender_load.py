"""Tests for blender_load.py — pure functions and geometry math.

The module imports bpy/bmesh at the top level, so we stub them before import.
Standalone ``mathutils`` (PyPI) is used directly for geometry tests.
"""

from __future__ import annotations

import importlib
import math
import struct
import sys
import types
from pathlib import Path
from typing import Any, cast

import mathutils
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture: import blender_load with lightweight bpy/bmesh stubs
# ---------------------------------------------------------------------------


def _make_bpy_stub() -> types.ModuleType:
    """Minimal bpy stub sufficient for module-level imports in blender_load."""
    bpy = types.ModuleType("bpy")

    class _Types(types.SimpleNamespace):
        def __getattr__(self, _: str) -> type[object]:
            return object

    cast(Any, bpy).types = _Types(Object=object, Collection=object, GeometryNodeTree=object)
    cast(Any, bpy).data = types.SimpleNamespace(
        collections=types.SimpleNamespace(get=lambda *a, **kw: None),
        objects=types.SimpleNamespace(get=lambda *a, **kw: None),
    )
    cast(Any, bpy).context = types.SimpleNamespace(
        scene=types.SimpleNamespace(
            collection=types.SimpleNamespace(children=types.SimpleNamespace(link=lambda *a: None)),
            render=types.SimpleNamespace(resolution_x=1024, resolution_y=1024, resolution_percentage=100),
            frame_current=1,
            frame_start=1,
            frame_end=120,
        ),
    )
    cast(Any, bpy).ops = types.SimpleNamespace(
        render=types.SimpleNamespace(render=lambda **kw: None),
        import_scene=types.SimpleNamespace(),
    )
    cast(Any, bpy).app = types.SimpleNamespace(version=(4, 4, 0))
    return bpy


@pytest.fixture()
def bl(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import blender_load with bpy/bmesh stubbed out."""
    bpy_stub = _make_bpy_stub()
    bmesh_stub = types.ModuleType("bmesh")
    cast(Any, bmesh_stub).new = lambda: None

    # kdtree is part of mathutils but needs to be importable as 'mathutils.kdtree'
    # — the standalone mathutils package should already have it.

    # The standalone mathutils PyPI package lacks kdtree; stub it.
    kdtree_stub = types.ModuleType("mathutils.kdtree")
    cast(Any, kdtree_stub).KDTree = type("KDTree", (), {"__init__": lambda *a, **kw: None})

    monkeypatch.setitem(sys.modules, "bpy", bpy_stub)
    monkeypatch.setitem(sys.modules, "bmesh", bmesh_stub)
    monkeypatch.setitem(sys.modules, "mathutils.kdtree", kdtree_stub)

    module_name = "process.scripts.blender_load"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestNormalizeKeepRatio:
    def test_none_returns_none(self, bl: Any):
        assert bl._normalize_keep_ratio(None) is None

    def test_zero_returns_none(self, bl: Any):
        assert bl._normalize_keep_ratio(0.0) is None

    def test_one_returns_none(self, bl: Any):
        assert bl._normalize_keep_ratio(1.0) is None

    def test_near_one_returns_none(self, bl: Any):
        assert bl._normalize_keep_ratio(0.9995) is None

    def test_negative_returns_none(self, bl: Any):
        assert bl._normalize_keep_ratio(-0.5) is None

    def test_valid_ratio(self, bl: Any):
        assert bl._normalize_keep_ratio(0.5) == pytest.approx(0.5)

    def test_small_valid(self, bl: Any):
        assert bl._normalize_keep_ratio(0.01) == pytest.approx(0.01)


class TestNormalizeColor:
    def test_none(self, bl: Any) -> None:
        assert bl._normalize_color(None) is None

    def test_tuple_three_values(self, bl: Any) -> None:
        assert bl._normalize_color((1, 0.5, 0)) == (1.0, 0.5, 0.0)

    def test_sequence_three_values(self, bl: Any) -> None:
        assert bl._normalize_color([0.2, 0.4, 0.6]) == (0.2, 0.4, 0.6)

    def test_tuple_wrong_length_raises(self, bl: Any) -> None:
        with pytest.raises(ValueError):
            bl._normalize_color((1.0, 0.0))

    def test_sequence_wrong_length_raises(self, bl: Any) -> None:
        with pytest.raises(ValueError):
            bl._normalize_color([1.0, 0.0, 0.0, 1.0])

    def test_non_sequence_raises(self, bl: Any) -> None:
        with pytest.raises(TypeError):
            bl._normalize_color(1.0)


class TestNormalizeVec:
    def test_normalize_vec3_none(self, bl: Any) -> None:
        assert bl._normalize_vec3(None, "camera_pos") is None

    def test_normalize_vec3_valid(self, bl: Any) -> None:
        assert bl._normalize_vec3([1, 2, 3], "camera_pos") == (1.0, 2.0, 3.0)

    def test_normalize_vec3_invalid_len(self, bl: Any) -> None:
        with pytest.raises(ValueError):
            bl._normalize_vec3([1, 2], "camera_pos")

    def test_normalize_vec4_none(self, bl: Any) -> None:
        assert bl._normalize_vec4(None, "camera_rot_quat") is None

    def test_normalize_vec4_valid(self, bl: Any) -> None:
        assert bl._normalize_vec4([1, 0, 0, 0], "camera_rot_quat") == (1.0, 0.0, 0.0, 0.0)

    def test_normalize_vec4_invalid_len(self, bl: Any) -> None:
        with pytest.raises(ValueError):
            bl._normalize_vec4([1, 0, 0], "camera_rot_quat")


class TestNormalizeCompositorPreset:
    def test_default(self, bl: Any) -> None:
        assert bl._normalize_compositor_preset(None) == "paper_clean"

    def test_aliases(self, bl: Any) -> None:
        assert bl._normalize_compositor_preset("paper") == "paper_clean"
        assert bl._normalize_compositor_preset("none") == "off"

    def test_valid_values(self, bl: Any) -> None:
        assert bl._normalize_compositor_preset("off") == "off"
        assert bl._normalize_compositor_preset("paper_clean") == "paper_clean"
        assert bl._normalize_compositor_preset("beauty") == "beauty"

    def test_invalid_raises(self, bl: Any) -> None:
        with pytest.raises(ValueError):
            bl._normalize_compositor_preset("cinematic")


class TestClampFloat:
    def test_within_range(self, bl: Any):
        assert bl._clamp_float(0.5, 0.0, 1.0) == 0.5

    def test_below(self, bl: Any):
        assert bl._clamp_float(-1.0, 0.0, 1.0) == 0.0

    def test_above(self, bl: Any):
        assert bl._clamp_float(2.0, 0.0, 1.0) == 1.0

    def test_swapped_bounds(self, bl: Any):
        assert bl._clamp_float(0.5, 1.0, 0.0) == 0.5


class TestSplitSubsampleRatio:
    def test_none(self, bl: Any):
        assert bl._split_subsample_ratio(None) == (None, None)

    def test_one(self, bl: Any):
        assert bl._split_subsample_ratio(1.0) == (None, None)

    def test_valid_split(self, bl: Any):
        a, b = bl._split_subsample_ratio(0.25)
        assert a is not None and b is not None
        assert a * b == pytest.approx(0.25)
        assert a == pytest.approx(0.5)


class TestNpMat44:
    def test_4x4_passthrough(self, bl: Any):
        m = np.eye(4, dtype=np.float64)
        result = bl.np_mat44(m)
        np.testing.assert_array_equal(result, m)

    def test_3x3_to_4x4(self, bl: Any):
        m = np.eye(3, dtype=np.float64) * 2
        result = bl.np_mat44(m)
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result[:3, :3], m)
        assert result[3, 3] == 1.0


class TestComputePlaneParams:
    def test_centered_principal_point(self, bl: Any):
        intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        w, h, ox, oy = bl._compute_plane_params_from_intrinsics(intrinsic, 640, 480, depth_m=1.0)
        assert w == pytest.approx(640 / 500)
        assert h == pytest.approx(480 / 500)
        assert ox == pytest.approx(0.0)
        assert oy == pytest.approx(0.0)


class TestFindStem:
    def test_explicit_stem(self, bl: Any, tmp_path: Path):
        assert bl.find_stem(tmp_path, "foo") == "foo"

    def test_auto_detect(self, bl: Any, tmp_path: Path):
        (tmp_path / "scene01_camera.json").touch()
        assert bl.find_stem(tmp_path, None) == "scene01"

    def test_no_match_raises(self, bl: Any, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            bl.find_stem(tmp_path, None)


class TestListVersionedFiles:
    def test_single_file(self, bl: Any, tmp_path: Path):
        p = tmp_path / "cloud.ply"
        p.touch()
        result = bl.list_versioned_files(p)
        assert result == [p]

    def test_series(self, bl: Any, tmp_path: Path):
        for name in ["cloud.ply", "cloud_1.ply", "cloud_2.ply"]:
            (tmp_path / name).touch()
        result = bl.list_versioned_files(tmp_path / "cloud.ply")
        assert len(result) == 3
        assert result[0].name == "cloud.ply"
        assert result[1].name == "cloud_1.ply"

    def test_missing(self, bl: Any, tmp_path: Path):
        result = bl.list_versioned_files(tmp_path / "nope.ply")
        assert result == []


# ---------------------------------------------------------------------------
# PLY header parsing tests
# ---------------------------------------------------------------------------


def _write_ascii_ply(path: Path, n_vertices: int) -> None:
    header = f"ply\nformat ascii 1.0\nelement vertex {n_vertices}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    path.write_text(header + "0 0 0\n" * min(n_vertices, 3))


def _write_binary_ply(path: Path, n_vertices: int) -> None:
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n_vertices}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        for _ in range(n_vertices):
            f.write(struct.pack("<fff", 0.0, 0.0, 0.0))


class TestReadPlyVertexCount:
    def test_ascii_ply(self, bl: Any, tmp_path: Path):
        p = tmp_path / "test.ply"
        _write_ascii_ply(p, 42)
        assert bl._read_ply_vertex_count(p) == 42

    def test_binary_ply(self, bl: Any, tmp_path: Path):
        p = tmp_path / "test.ply"
        _write_binary_ply(p, 1000)
        assert bl._read_ply_vertex_count(p) == 1000

    def test_missing_file(self, bl: Any, tmp_path: Path):
        assert bl._read_ply_vertex_count(tmp_path / "nope.ply") is None


class TestFindDensestPath:
    def test_picks_densest(self, bl: Any, tmp_path: Path):
        for name, n in [("a.ply", 100), ("b.ply", 500), ("c.ply", 200)]:
            _write_ascii_ply(tmp_path / name, n)
        paths = [tmp_path / "a.ply", tmp_path / "b.ply", tmp_path / "c.ply"]
        densest, count = bl._find_densest_path(paths)
        assert densest.name == "b.ply"
        assert count == 500


class TestComputeGroupKeepRatio:
    def test_no_auto(self, bl: Any, tmp_path: Path):
        _write_ascii_ply(tmp_path / "a.ply", 1000)
        ratio, _path, count = bl._compute_group_keep_ratio(
            [tmp_path / "a.ply"],
            explicit_ratio=0.5,
            auto_tune_points=False,
            auto_target_points=None,
        )
        assert ratio == pytest.approx(0.5)
        assert count == 1000

    def test_auto_downsample(self, bl: Any, tmp_path: Path):
        _write_ascii_ply(tmp_path / "a.ply", 10000)
        ratio, _, _ = bl._compute_group_keep_ratio(
            [tmp_path / "a.ply"],
            explicit_ratio=None,
            auto_tune_points=True,
            auto_target_points=5000,
        )
        assert ratio == pytest.approx(0.5)

    def test_auto_zero_target(self, bl: Any, tmp_path: Path):
        _write_ascii_ply(tmp_path / "a.ply", 1000)
        ratio, _, _ = bl._compute_group_keep_ratio(
            [tmp_path / "a.ply"],
            explicit_ratio=None,
            auto_tune_points=True,
            auto_target_points=0,
        )
        assert ratio is None


# ---------------------------------------------------------------------------
# Geometry / reframe camera tests (standalone mathutils)
# ---------------------------------------------------------------------------


class TestReframeCameraGeometry:
    """Test the AABB projection math inside _reframe_camera.

    We can't easily call _reframe_camera directly (it modifies bpy objects),
    so we replicate and test the core projection logic.
    """

    @staticmethod
    def _compute_min_distance(
        cam_location: mathutils.Vector,
        center: mathutils.Vector,
        bounds_min: mathutils.Vector,
        bounds_max: mathutils.Vector,
        half_fov: float,
        fill_factor: float = 0.95,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        aspect: float = 1.0,
    ) -> float:
        """Pure-math replica of _reframe_camera's projection logic.

        Supports asymmetric frustum via shift_x/shift_y (in sensor-width
        fractions) and non-square renders via aspect (res_y / res_x).
        """
        tan_hfov = math.tan(half_fov)

        # Per-edge fill tangents (from optical axis to fill boundary).
        fill_right = max(fill_factor * (0.5 - shift_x) * 2.0 * tan_hfov, 1e-6)
        fill_left = max(fill_factor * (0.5 + shift_x) * 2.0 * tan_hfov, 1e-6)
        fill_up = max(fill_factor * (0.5 - shift_y) * 2.0 * tan_hfov * aspect, 1e-6)
        fill_down = max(fill_factor * (0.5 + shift_y) * 2.0 * tan_hfov * aspect, 1e-6)

        direction = cam_location - center
        if direction.length < 1e-8:
            direction = mathutils.Vector((0.0, -1.0, 0.5))
        view_dir = direction.normalized()

        world_up = mathutils.Vector((0.0, 1.0, 0.0))
        if abs(view_dir.dot(world_up)) > 0.999:
            world_up = mathutils.Vector((0.0, 0.0, 1.0))
        right = cast(mathutils.Vector, view_dir.cross(world_up)).normalized()
        up = cast(mathutils.Vector, right.cross(view_dir)).normalized()

        corners = [
            mathutils.Vector((x, y, z))
            for x in (bounds_min.x, bounds_max.x)
            for y in (bounds_min.y, bounds_max.y)
            for z in (bounds_min.z, bounds_max.z)
        ]

        min_distance = 0.0
        for corner in corners:
            offset = corner - center
            d_along = offset.dot(view_dir)
            lat_x = offset.dot(right)
            lat_y = offset.dot(up)
            req_x = lat_x / fill_right if lat_x >= 0.0 else -lat_x / fill_left
            req_y = lat_y / fill_up if lat_y >= 0.0 else -lat_y / fill_down
            required = d_along + max(req_x, req_y)
            min_distance = max(min_distance, required)

        max_d_along = max((c - center).dot(view_dir) for c in corners)
        min_distance = max(min_distance, max_d_along + 0.01)
        return min_distance

    def test_cube_front_view(self):
        """Camera looking along -Y at a unit cube centered at origin."""
        center = mathutils.Vector((0, 0, 0))
        bmin = mathutils.Vector((-0.5, -0.5, -0.5))
        bmax = mathutils.Vector((0.5, 0.5, 0.5))
        cam_loc = mathutils.Vector((0, 5, 0))
        half_fov = math.atan(1.0)  # 45 deg -> tan = 1.0

        dist = self._compute_min_distance(cam_loc, center, bmin, bmax, half_fov, fill_factor=1.0)

        # From the front, the cube projects as 1x1 in XZ.
        # view_dir = (0,1,0). d_along = corner.y.
        # Worst corner (0.5, 0.5, 0.5): d_along=0.5, lat_x=0.5, lat_y=0.5 -> 0.5 + 0.5 = 1.0
        assert dist == pytest.approx(1.0, abs=0.02)

    def test_flat_scene_top_view(self):
        """Camera directly above a flat 2x2 slab at z=0. Should give tight framing."""
        center = mathutils.Vector((0, 0, 0))
        bmin = mathutils.Vector((-1, -1, -0.1))
        bmax = mathutils.Vector((1, 1, 0.1))
        cam_loc = mathutils.Vector((0, 0, 10))  # directly above
        half_fov = math.radians(25)

        dist = self._compute_min_distance(cam_loc, center, bmin, bmax, half_fov, fill_factor=0.95)

        # view_dir = (0,0,1), right = (-1,0,0), up = (0,-1,0).
        # Worst corners: lat_x=1, lat_y=1, d_along=0.1.
        # fill_tan = 0.95 * tan(25) = 0.4428
        # required = 0.1 + 1.0 / 0.4428 = 2.359
        expected = 0.1 + 1.0 / (0.95 * math.tan(math.radians(25)))
        assert dist == pytest.approx(expected, rel=0.01)

    def test_flat_scene_sphere_vs_aabb(self):
        """AABB framing should be much tighter than sphere for an oblique view of a flat scene."""
        center = mathutils.Vector((0, 0, 0))
        bmin = mathutils.Vector((-1, -1, -0.05))
        bmax = mathutils.Vector((1, 1, 0.05))

        sphere_radius = (bmax - center).length
        half_fov = math.radians(25)
        fill = 0.95

        sphere_dist = sphere_radius / (fill * math.tan(half_fov))

        cam_loc = mathutils.Vector((0, -3, 1.73))  # roughly 30 deg elevation
        aabb_dist = self._compute_min_distance(cam_loc, center, bmin, bmax, half_fov, fill)

        assert aabb_dist < sphere_dist, f"AABB dist {aabb_dist:.3f} should be less than sphere dist {sphere_dist:.3f}"

    def test_elongated_scene(self):
        """A long thin object viewed from the side should frame tightly."""
        center = mathutils.Vector((0, 0, 0))
        bmin = mathutils.Vector((-5, -0.1, -0.1))
        bmax = mathutils.Vector((5, 0.1, 0.1))
        cam_loc = mathutils.Vector((0, 10, 0))
        half_fov = math.radians(30)
        fill = 0.95

        sphere_radius = (bmax - center).length
        sphere_dist = sphere_radius / (fill * math.tan(half_fov))

        aabb_dist = self._compute_min_distance(cam_loc, center, bmin, bmax, half_fov, fill)
        assert aabb_dist < sphere_dist * 1.05

    def test_shift_x_asymmetric(self):
        """Positive shift_x extends left edge, so a left-heavy scene should frame tighter."""
        center = mathutils.Vector((0, 0, 0))
        # Content extends further to the left than right.
        bmin = mathutils.Vector((-2, -0.5, -0.5))
        bmax = mathutils.Vector((0.5, 0.5, 0.5))
        cam_loc = mathutils.Vector((0, 10, 0))
        half_fov = math.radians(30)
        fill = 0.95

        # shift_x > 0: principal point right, frustum LEFT edge extends more.
        # The left-heavy content fits better in the wider left half.
        dist_shifted = self._compute_min_distance(cam_loc, center, bmin, bmax, half_fov, fill, shift_x=0.15)
        dist_centered = self._compute_min_distance(cam_loc, center, bmin, bmax, half_fov, fill, shift_x=0.0)
        assert dist_shifted < dist_centered, (
            f"Shifted dist {dist_shifted:.3f} should be less than centered {dist_centered:.3f}"
        )

    def test_zero_shift_matches_symmetric(self):
        """With zero shifts, asymmetric logic should match the old symmetric result."""
        center = mathutils.Vector((0, 0, 0))
        bmin = mathutils.Vector((-1, -1, -1))
        bmax = mathutils.Vector((1, 1, 1))
        cam_loc = mathutils.Vector((0, 5, 2))
        half_fov = math.radians(30)
        fill = 0.95

        dist_zero = self._compute_min_distance(cam_loc, center, bmin, bmax, half_fov, fill, shift_x=0.0, shift_y=0.0)

        # Old symmetric formula: fill_tan = fill * tan(half_fov), max(|lat_x|, |lat_y|) / fill_tan
        fill_tan = fill * math.tan(half_fov)
        direction = cam_loc - center
        view_dir = direction.normalized()
        world_up = mathutils.Vector((0.0, 1.0, 0.0))
        right = cast(mathutils.Vector, view_dir.cross(world_up)).normalized()
        up = cast(mathutils.Vector, right.cross(view_dir)).normalized()
        corners = [
            mathutils.Vector((x, y, z)) for x in (bmin.x, bmax.x) for y in (bmin.y, bmax.y) for z in (bmin.z, bmax.z)
        ]
        old_dist = 0.0
        for c in corners:
            o = c - center
            d = o.dot(view_dir)
            lx = abs(o.dot(right))
            ly = abs(o.dot(up))
            old_dist = max(old_dist, d + max(lx, ly) / fill_tan)
        max_d = max((c - center).dot(view_dir) for c in corners)
        old_dist = max(old_dist, max_d + 0.01)

        assert dist_zero == pytest.approx(old_dist, rel=1e-6)


class TestObjectsAabb:
    """Test _objects_aabb with real mathutils but stubbed bpy objects."""

    @staticmethod
    def _make_fake_obj(
        corners: list[tuple[float, float, float]] | list[tuple[int, int, int]],
        name: str = "fake",
    ) -> Any:
        """Create a minimal object with bound_box and matrix_world."""
        obj = types.SimpleNamespace()
        obj.bound_box = corners
        obj.matrix_world = mathutils.Matrix.Identity(4)
        obj.name = name
        obj.hide_render = False
        obj.data = True  # non-None -> _is_placeholder returns False
        obj.type = "MESH"
        obj._fake_props: dict = {}  # type: ignore[annotation-unchecked]

        def get(key: str, default: Any = None) -> Any:
            return obj._fake_props.get(key, default)

        obj.get = get
        return obj

    def test_single_object(self, bl: Any):
        corners = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
        obj = self._make_fake_obj(corners)
        center, bmin, bmax = bl._objects_aabb([obj])
        assert bmin.x == pytest.approx(0.0)
        assert bmax.x == pytest.approx(1.0)
        assert center.x == pytest.approx(0.5)

    def test_empty(self, bl: Any):
        center, _bmin, _bmax = bl._objects_aabb([])
        assert center.length == pytest.approx(0.0)

    def test_no_outlier_filtering(self, bl: Any):
        """_objects_aabb should NOT filter — it trusts the caller's curated set."""
        unit = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        objs = [self._make_fake_obj(unit, name=f"obj_{i}") for i in range(5)]
        outlier_corners = [
            (100, 0, 0),
            (101, 0, 0),
            (101, 1, 0),
            (100, 1, 0),
            (100, 0, 1),
            (101, 0, 1),
            (101, 1, 1),
            (100, 1, 1),
        ]
        objs.append(self._make_fake_obj(outlier_corners, name="distant"))

        _center, _bmin, bmax = bl._objects_aabb(objs)
        # AABB must include the distant object — no filtering applied.
        assert bmax.x == pytest.approx(101.0)


class TestFilterOutlierBounds:
    """Test _filter_outlier_bounds (used by _robust_bounds_from_stats)."""

    @staticmethod
    def _make_bounds(cx: float, cy: float, cz: float, r: float) -> Any:
        """Create an ObjectBounds-like namedtuple."""
        from collections import namedtuple

        OB = namedtuple("OB", ["obj", "min", "max", "center", "radius"])
        center = mathutils.Vector((cx, cy, cz))
        bmin = mathutils.Vector((cx - r, cy - r, cz - r))
        bmax = mathutils.Vector((cx + r, cy + r, cz + r))
        return OB(obj=None, min=bmin, max=bmax, center=center, radius=r)

    def test_outlier_dropped(self, bl: Any):
        """A distant outlier should be filtered when enough inliers exist."""
        stats = [self._make_bounds(float(i), 0.0, 0.0, 0.5) for i in range(5)]
        stats.append(self._make_bounds(100.0, 0.0, 0.0, 0.5))  # outlier
        filtered = bl._filter_outlier_bounds(stats)
        centers_x = [b.center.x for b in filtered]
        assert 100.0 not in centers_x, "Outlier should have been removed"
        assert len(filtered) == 5

    def test_small_set_unfiltered(self, bl: Any):
        """Sets with fewer than 4 items should be returned unchanged."""
        stats = [self._make_bounds(0, 0, 0, 1), self._make_bounds(100, 0, 0, 1)]
        filtered = bl._filter_outlier_bounds(stats)
        assert len(filtered) == 2


class TestLightingSubjectBounds:
    @staticmethod
    def _make_fake_obj(
        corners: list[tuple[float, float, float]] | list[tuple[int, int, int]],
        name: str = "fake",
    ) -> Any:
        obj = types.SimpleNamespace()
        obj.bound_box = corners
        obj.matrix_world = mathutils.Matrix.Identity(4)
        obj.name = name
        obj.hide_render = False
        obj.data = True  # non-None -> _is_placeholder returns False
        obj.type = "MESH"
        obj._fake_props = cast(dict[str, Any], {})

        def get(key: str, default: Any = None) -> Any:
            return obj._fake_props.get(key, default)

        obj.get = get
        return obj

    def test_prefers_mesh_and_pred_over_inputs(self, bl: Any):
        mesh_left = self._make_fake_obj(
            [
                (-2.0, -1.0, -1.0),
                (-1.0, -1.0, -1.0),
                (-2.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-2.0, -1.0, 1.0),
                (-1.0, -1.0, 1.0),
                (-2.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
            ],
            name="mesh_left",
        )
        pred_right = self._make_fake_obj(
            [
                (1.0, -1.0, -1.0),
                (2.0, -1.0, -1.0),
                (1.0, 1.0, -1.0),
                (2.0, 1.0, -1.0),
                (1.0, -1.0, 1.0),
                (2.0, -1.0, 1.0),
                (1.0, 1.0, 1.0),
                (2.0, 1.0, 1.0),
            ],
            name="pred_right",
        )
        inputs_far = self._make_fake_obj(
            [
                (100.0, -1.0, -1.0),
                (101.0, -1.0, -1.0),
                (100.0, 1.0, -1.0),
                (101.0, 1.0, -1.0),
                (100.0, -1.0, 1.0),
                (101.0, -1.0, 1.0),
                (100.0, 1.0, 1.0),
                (101.0, 1.0, 1.0),
            ],
            name="inputs_far",
        )

        subject, center, radius, min_z = bl._lighting_subject_bounds(
            {
                "mesh": [mesh_left],
                "pred": [pred_right],
                "inputs": [inputs_far],
            }
        )

        assert subject in (mesh_left, pred_right)
        assert center is not None and center.x == pytest.approx(0.0)
        assert center is not None and center.y == pytest.approx(0.0)
        assert center is not None and center.z == pytest.approx(0.0)
        assert radius is not None and radius == pytest.approx(math.sqrt(6.0))
        assert min_z == pytest.approx(-1.0)

    def test_falls_back_to_inputs_when_mesh_pred_missing(self, bl: Any):
        inputs = self._make_fake_obj(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ],
            name="inputs",
        )

        subject, center, radius, min_z = bl._lighting_subject_bounds({"inputs": [inputs]})

        assert subject is inputs
        assert center is not None and center.x == pytest.approx(0.5)
        assert center is not None and center.y == pytest.approx(0.5)
        assert center is not None and center.z == pytest.approx(0.5)
        assert radius is not None and radius == pytest.approx(math.sqrt(3.0) * 0.5)
        assert min_z == pytest.approx(0.0)

    def test_choose_subject_handles_multi_object_groups(self, bl: Any):
        mesh_a = self._make_fake_obj(
            [
                (-1.0, -1.0, -1.0),
                (-0.5, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-0.5, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (-0.5, -1.0, 1.0),
                (-1.0, 1.0, 1.0),
                (-0.5, 1.0, 1.0),
            ],
            name="mesh_a",
        )
        mesh_b = self._make_fake_obj(
            [
                (0.5, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (0.5, 1.0, -1.0),
                (1.0, 1.0, -1.0),
                (0.5, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (0.5, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ],
            name="mesh_b",
        )

        subject = bl.choose_subject({"mesh": [mesh_a, mesh_b]})
        assert subject in (mesh_a, mesh_b)


class TestPointRadiusFloor:
    def test_probe_points_include_center_and_bbox_corners(self, bl: Any, monkeypatch: pytest.MonkeyPatch):
        obj = types.SimpleNamespace(
            bound_box=[
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (-1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0),
            ],
            matrix_world=mathutils.Matrix.Identity(4),
        )
        center = mathutils.Vector((0.25, -0.5, 0.75))
        monkeypatch.setattr(bl, "_object_bounds_world", lambda _obj: (center, 1.0))

        probes = bl._point_radius_probe_points_world(obj)
        probe_tuples = {tuple(round(float(v), 6) for v in p) for p in probes}

        assert tuple(round(float(v), 6) for v in center) in probe_tuples
        assert (-1.0, -1.0, -1.0) in probe_tuples
        assert (1.0, 1.0, 1.0) in probe_tuples
        assert len(probes) >= 9

    def test_camera_floor_uses_max_over_probe_points(self, bl: Any, monkeypatch: pytest.MonkeyPatch):
        obj = types.SimpleNamespace(name="inputs_cloud")
        scene = bl.bpy.context.scene
        cam_obj = types.SimpleNamespace()
        set_calls: list[float] = []

        monkeypatch.setattr(bl, "_iter_point_objects", lambda _groups, _labels=None: [obj])
        monkeypatch.setattr(bl, "_base_point_radius", lambda _obj: 0.05)
        monkeypatch.setattr(
            bl,
            "_point_radius_probe_points_world",
            lambda _obj: [mathutils.Vector((0.5, 0.0, 0.0)), mathutils.Vector((2.0, 0.0, 0.0))],
        )
        monkeypatch.setattr(
            bl,
            "_min_world_radius_for_pixels",
            lambda _cam, point, min_px, scene: float(point.x) * 0.1,
        )
        monkeypatch.setattr(bl, "_set_point_radius", lambda _obj, r: set_calls.append(float(r)) or True)

        bl._apply_point_radius_floor_for_camera({"inputs": [obj]}, cam_obj, min_px=1.0, labels=("inputs",))

        assert set_calls
        assert set_calls[-1] == pytest.approx(0.2)
        assert scene is not None

    def test_compensated_keep_ratio_scales_with_radius_squared(self, bl: Any):
        keep = bl._compensated_keep_ratio_from_radius(base_keep_ratio=1.0, base_radius=0.01, active_radius=0.02)
        assert keep == pytest.approx(0.25)

        keep_no_increase = bl._compensated_keep_ratio_from_radius(
            base_keep_ratio=0.6, base_radius=0.01, active_radius=0.005
        )
        assert keep_no_increase == pytest.approx(0.6)

    def test_camera_floor_applies_density_compensation(self, bl: Any, monkeypatch: pytest.MonkeyPatch):
        obj = types.SimpleNamespace(name="inputs_cloud")
        cam_obj = types.SimpleNamespace()
        set_radius_calls: list[float] = []
        set_keep_calls: list[float] = []

        monkeypatch.setattr(bl, "_iter_point_objects", lambda _groups, _labels=None: [obj])
        monkeypatch.setattr(bl, "_base_point_radius", lambda _obj: 0.1)
        monkeypatch.setattr(bl, "_required_point_radius_for_camera", lambda _obj, _cam, min_px, scene: 0.2)
        monkeypatch.setattr(bl, "_base_point_keep_ratio", lambda _obj: 1.0)
        monkeypatch.setattr(bl, "_set_point_radius", lambda _obj, r: set_radius_calls.append(float(r)) or True)
        monkeypatch.setattr(bl, "_set_point_keep_ratio", lambda _obj, k: set_keep_calls.append(float(k)) or True)

        bl._apply_point_radius_floor_for_camera({"inputs": [obj]}, cam_obj, min_px=1.0, labels=("inputs",))

        assert set_radius_calls and set_radius_calls[-1] == pytest.approx(0.2)
        assert set_keep_calls and set_keep_calls[-1] == pytest.approx(0.25)


class TestRemoveExistingLightingRig:
    def test_removes_named_rig_artifacts_and_keeps_other_objects(self, bl: Any, monkeypatch: pytest.MonkeyPatch):
        class _ObjectStore(list):
            def remove(self, item: Any, do_unlink: bool = False) -> None:
                _ = do_unlink
                super().remove(item)

        class _DataStore(list):
            def remove(self, item: Any) -> None:
                super().remove(item)

        def _obj(name: str) -> Any:
            return types.SimpleNamespace(name=name)

        def _block(name: str, users: int) -> Any:
            return types.SimpleNamespace(name=name, users=users)

        object_store = _ObjectStore(
            [
                _obj("Key_Light"),
                _obj("Key_Light.001"),
                _obj("ShadowCatcher"),
                _obj("BackgroundMesh"),
            ]
        )
        light_store = _DataStore(
            [
                _block("Key_Light", 0),
                _block("Other_Light", 0),
            ]
        )
        mesh_store = _DataStore(
            [
                _block("ShadowCatcher_mesh", 0),
                _block("OtherMesh", 0),
            ]
        )
        material_store = _DataStore(
            [
                _block("ShadowCatcher_Mat", 0),
                _block("OtherMat", 0),
            ]
        )

        monkeypatch.setattr(
            bl.bpy,
            "data",
            types.SimpleNamespace(
                objects=object_store,
                lights=light_store,
                meshes=mesh_store,
                materials=material_store,
            ),
        )

        bl._remove_existing_lighting_rig()

        assert [o.name for o in object_store] == ["BackgroundMesh"]
        assert [light.name for light in light_store] == ["Other_Light"]
        assert [m.name for m in mesh_store] == ["OtherMesh"]
        assert [m.name for m in material_store] == ["OtherMat"]


class TestCreateObjectCamerasForTargets:
    class _FakeObj:
        def __init__(self, name: str):
            self.name = name
            self.data = True
            self.type = "MESH"
            self._props: dict[str, Any] = {}

        def __setitem__(self, key: str, value: Any) -> None:
            self._props[key] = value

        def get(self, key: str, default: Any = None) -> Any:
            return self._props.get(key, default)

        def keys(self) -> list[str]:
            return list(self._props.keys())

    def test_gt_anchor_shared_camera_and_orphan_pred_gets_individual(self, bl: Any, monkeypatch: pytest.MonkeyPatch):
        """Shared GT/pred cameras anchored on GT; orphan pred gets individual camera."""
        default_cam = object()
        gt0 = self._FakeObj("00010850_mesh_0")
        pred0 = self._FakeObj("00010850_pred_mesh_0")
        pred1 = self._FakeObj("00010850_pred_mesh_1")  # orphan (no gt_1)

        created_for: list[str] = []
        fill_factors: list[float] = []

        def _fake_create_object_camera(obj: Any, _reference: Any, fill_factor: float = 0.95) -> Any:
            created_for.append(obj.name)
            fill_factors.append(float(fill_factor))
            return types.SimpleNamespace(name="tmp_cam")

        monkeypatch.setattr(bl, "ensure_collection", lambda _name: object())
        monkeypatch.setattr(bl, "_move_object_to_collection", lambda *_a, **_k: None)
        monkeypatch.setattr(bl, "create_object_camera", _fake_create_object_camera)

        bl.create_object_cameras_for_targets(
            "00010850",
            default_cam,
            {"mesh": [gt0], "pred": [pred0, pred1]},
        )

        # GT camera created first, then individual camera for orphan pred.
        assert created_for == ["00010850_mesh_0", "00010850_pred_mesh_1"]
        assert fill_factors == [pytest.approx(0.85), pytest.approx(0.85)]

        # GT and paired pred must share the same GT-derived camera.
        assert gt0.get("_solo_camera") == "Camera_pair_0"
        assert pred0.get("_solo_camera") == "Camera_pair_0"
        assert gt0.get("_solo_light_anchor") == "00010850_mesh_0"
        assert pred0.get("_solo_light_anchor") == "00010850_mesh_0"

        # Orphan pred gets its own individual camera.
        assert pred1.get("_solo_camera") == "Camera_00010850_pred_mesh_1"
        assert pred1.get("_solo_light_anchor") == "00010850_pred_mesh_1"

    def test_series_index_ignores_blender_numeric_suffix(self, bl: Any, monkeypatch: pytest.MonkeyPatch):
        """Names like *_mesh_2.001 should still pair by index 2, not collapse to 0."""
        default_cam = object()
        gt2 = self._FakeObj("00001067_mesh_2.001")
        gt10 = self._FakeObj("00001067_mesh_10.001")
        pred2 = self._FakeObj("00001067_pred_mesh_2.001")
        pred10 = self._FakeObj("00001067_pred_mesh_10.001")

        created_for: list[str] = []

        def _fake_create_object_camera(obj: Any, _reference: Any, fill_factor: float = 0.95) -> Any:
            created_for.append(obj.name)
            return types.SimpleNamespace(name="tmp_cam")

        monkeypatch.setattr(bl, "ensure_collection", lambda _name: object())
        monkeypatch.setattr(bl, "_move_object_to_collection", lambda *_a, **_k: None)
        monkeypatch.setattr(bl, "create_object_camera", _fake_create_object_camera)

        bl.create_object_cameras_for_targets(
            "00001067",
            default_cam,
            {"mesh": [gt2, gt10], "pred": [pred2, pred10]},
        )

        assert created_for == ["00001067_mesh_2.001", "00001067_mesh_10.001"]
        assert gt2.get("_solo_camera") == "Camera_pair_2"
        assert pred2.get("_solo_camera") == "Camera_pair_2"
        assert gt10.get("_solo_camera") == "Camera_pair_10"
        assert pred10.get("_solo_camera") == "Camera_pair_10"

    def test_spatial_fallback_reassigns_pred_when_index_alignment_is_bad(
        self, bl: Any, monkeypatch: pytest.MonkeyPatch
    ):
        """If index pairing is spatially inconsistent, preds should match nearest GT camera."""
        default_cam = object()
        gt0 = self._FakeObj("00001067_mesh_0")
        gt1 = self._FakeObj("00001067_mesh_1")
        pred0 = self._FakeObj("00001067_pred_mesh_0")
        pred1 = self._FakeObj("00001067_pred_mesh_1")

        def _fake_create_object_camera(obj: Any, _reference: Any, fill_factor: float = 0.95) -> Any:
            return types.SimpleNamespace(name="tmp_cam")

        centers = {
            "00001067_mesh_0": mathutils.Vector((0.0, 0.0, 0.0)),
            "00001067_mesh_1": mathutils.Vector((10.0, 0.0, 0.0)),
            # Intentionally swapped: pred_0 lies near gt_1, pred_1 lies near gt_0.
            "00001067_pred_mesh_0": mathutils.Vector((10.1, 0.0, 0.0)),
            "00001067_pred_mesh_1": mathutils.Vector((0.1, 0.0, 0.0)),
        }

        def _fake_bounds(obj: Any) -> tuple[mathutils.Vector, float]:
            return centers[obj.name], 0.2

        monkeypatch.setattr(bl, "ensure_collection", lambda _name: object())
        monkeypatch.setattr(bl, "_move_object_to_collection", lambda *_a, **_k: None)
        monkeypatch.setattr(bl, "create_object_camera", _fake_create_object_camera)
        monkeypatch.setattr(bl, "_object_bounds_world", _fake_bounds)

        bl.create_object_cameras_for_targets(
            "00001067",
            default_cam,
            {"mesh": [gt0, gt1], "pred": [pred0, pred1]},
        )

        # Cameras stay GT-indexed.
        assert gt0.get("_solo_camera") == "Camera_pair_0"
        assert gt1.get("_solo_camera") == "Camera_pair_1"

        # Preds must be remapped spatially (not by their own index).
        assert pred0.get("_solo_camera") == "Camera_pair_1"
        assert pred1.get("_solo_camera") == "Camera_pair_0"


class TestCreateObjectCamera:
    def test_uses_frame_points_for_tightening(self, bl: Any, monkeypatch: pytest.MonkeyPatch):
        """create_object_camera should pass sampled frame points into reframing."""
        obj = types.SimpleNamespace(name="mesh")
        ref_cam = types.SimpleNamespace(
            data=types.SimpleNamespace(
                type="PERSP",
                lens=50.0,
                sensor_fit="HORIZONTAL",
                sensor_width=36.0,
                sensor_height=36.0,
                clip_start=0.01,
                clip_end=100.0,
            ),
            matrix_world=mathutils.Matrix.Identity(4),
        )
        cam_obj = types.SimpleNamespace(
            name="Camera_mesh",
            data=types.SimpleNamespace(),
            location=mathutils.Vector((0.0, 0.0, 0.0)),
            rotation_euler=mathutils.Vector((0.0, 0.0, 0.0)),
        )

        frame_points = [mathutils.Vector((1.0, 0.0, 0.0)), mathutils.Vector((-1.0, 0.0, 0.0))]
        calls: list[tuple[mathutils.Vector, list[mathutils.Vector], float]] = []

        monkeypatch.setattr(bl, "new_camera", lambda name="Camera": cam_obj)
        monkeypatch.setattr(bl, "_object_bounds_world", lambda _obj: (mathutils.Vector((0.0, 0.0, 0.0)), 1.0))
        monkeypatch.setattr(bl, "_object_frame_points_world", lambda _obj, max_mesh_points=4000: frame_points)

        def _fake_reframe(
            _cam_obj: Any, center: mathutils.Vector, corners: list[mathutils.Vector], fill_factor: float = 0.95
        ) -> None:
            calls.append((center, corners, fill_factor))

        monkeypatch.setattr(bl, "_reframe_camera_to_corners", _fake_reframe)

        bl.create_object_camera(obj, ref_cam, fill_factor=0.97)

        assert len(calls) == 1
        center, corners, ff = calls[0]
        assert center == mathutils.Vector((0.0, 0.0, 0.0))
        assert corners is frame_points
        assert ff == pytest.approx(0.97)


class TestObjectFramePoints:
    def test_mesh_vertices_prefer_geometry_over_bbox_corners(self, bl: Any):
        verts = [
            types.SimpleNamespace(co=mathutils.Vector((-1.0, -0.5, -0.25))),
            types.SimpleNamespace(co=mathutils.Vector((1.0, 0.5, 0.25))),
            types.SimpleNamespace(co=mathutils.Vector((0.0, 0.1, 0.2))),
        ]
        obj = types.SimpleNamespace(
            type="MESH",
            data=types.SimpleNamespace(vertices=verts),
            matrix_world=mathutils.Matrix.Identity(4),
            bound_box=[
                (-100.0, -100.0, -100.0),
                (-100.0, -100.0, 100.0),
                (-100.0, 100.0, -100.0),
                (-100.0, 100.0, 100.0),
                (100.0, -100.0, -100.0),
                (100.0, -100.0, 100.0),
                (100.0, 100.0, -100.0),
                (100.0, 100.0, 100.0),
            ],
        )

        points = bl._object_frame_points_world(obj, max_mesh_points=100)

        assert points
        assert max(abs(p.x) for p in points) <= 1.0
        assert max(abs(p.y) for p in points) <= 0.5
        assert max(abs(p.z) for p in points) <= 0.25

    def test_non_mesh_falls_back_to_bbox_corners(self, bl: Any):
        corners = [
            (-2.0, -3.0, -4.0),
            (-2.0, -3.0, 4.0),
            (-2.0, 3.0, -4.0),
            (-2.0, 3.0, 4.0),
            (2.0, -3.0, -4.0),
            (2.0, -3.0, 4.0),
            (2.0, 3.0, -4.0),
            (2.0, 3.0, 4.0),
        ]
        obj = types.SimpleNamespace(
            type="EMPTY",
            data=None,
            matrix_world=mathutils.Matrix.Identity(4),
            bound_box=corners,
        )

        points = bl._object_frame_points_world(obj, max_mesh_points=100)

        assert len(points) == 8
        got = {(round(p.x, 5), round(p.y, 5), round(p.z, 5)) for p in points}
        assert got == set(corners)


class TestReframeCameraToCorners:
    def test_does_not_mutate_input_center(self, bl: Any):
        cam = types.SimpleNamespace(
            data=types.SimpleNamespace(
                sensor_width=36.0,
                lens=50.0,
                shift_x=0.0,
                shift_y=0.0,
                clip_start=0.01,
                clip_end=10.0,
            ),
            location=mathutils.Vector((0.0, -5.0, 1.0)),
            rotation_euler=mathutils.Vector((0.0, 0.0, 0.0)),
        )
        center = mathutils.Vector((1.0, 2.0, 3.0))
        center_before = center.copy()
        corners = [
            mathutils.Vector((0.0, 1.5, 2.5)),
            mathutils.Vector((2.0, 2.5, 3.5)),
            mathutils.Vector((0.5, 1.8, 3.2)),
        ]

        bl._reframe_camera_to_corners(cam, center, corners, fill_factor=0.99)

        assert center == center_before

    def test_reduces_clip_start_when_tight(self, bl: Any):
        cam = types.SimpleNamespace(
            data=types.SimpleNamespace(
                sensor_width=36.0,
                lens=50.0,
                shift_x=0.0,
                shift_y=0.0,
                clip_start=1.0,
                clip_end=10.0,
            ),
            location=mathutils.Vector((0.0, -3.0, 0.0)),
            rotation_euler=mathutils.Vector((0.0, 0.0, 0.0)),
        )
        center = mathutils.Vector((0.0, 0.0, 0.0))
        corners = [
            mathutils.Vector((-0.6, -0.4, -0.3)),
            mathutils.Vector((0.6, 0.4, 0.3)),
            mathutils.Vector((0.2, -0.5, 0.1)),
        ]

        bl._reframe_camera_to_corners(cam, center, corners, fill_factor=0.99)

        assert float(cam.data.clip_start) < 1.0

    def test_does_not_increase_clip_start(self, bl: Any):
        cam = types.SimpleNamespace(
            data=types.SimpleNamespace(
                sensor_width=36.0,
                lens=50.0,
                shift_x=0.0,
                shift_y=0.0,
                clip_start=1e-7,
                clip_end=10.0,
            ),
            location=mathutils.Vector((0.0, -3.0, 0.0)),
            rotation_euler=mathutils.Vector((0.0, 0.0, 0.0)),
        )
        center = mathutils.Vector((0.0, 0.0, 0.0))
        corners = [
            mathutils.Vector((-0.6, -0.4, -0.3)),
            mathutils.Vector((0.6, 0.4, 0.3)),
            mathutils.Vector((0.2, -0.5, 0.1)),
        ]

        bl._reframe_camera_to_corners(cam, center, corners, fill_factor=0.99)

        assert float(cam.data.clip_start) <= 1e-7


class TestSolveProjectedMidShift:
    def test_zero_for_symmetric_case(self, bl: Any):
        delta = bl._solve_projected_mid_shift([-1.0, 1.0], [2.0, 2.0])
        assert delta == pytest.approx(0.0, abs=1e-9)

    def test_nonzero_for_depth_asymmetry(self, bl: Any):
        # lat = [-1, 1], depth = [1, 2] has projected center at -0.25 when delta=0.
        # Expected centering shift is -1/3:
        #   (-1 - d)/1 and (1 - d)/2 -> symmetric when d = -1/3.
        delta = bl._solve_projected_mid_shift([-1.0, 1.0], [1.0, 2.0])
        assert delta == pytest.approx(-1.0 / 3.0, rel=1e-3, abs=1e-4)
