from types import SimpleNamespace

import numpy as np
import pytest

from process.src import fuse as fuse_mod


def test_get_views_returns_expected_rotation_for_forward_view() -> None:
    points = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    views = fuse_mod.get_views(points)

    assert len(views) == 1
    np.testing.assert_allclose(views[0], np.eye(3), atol=1e-7)


def test_fuse_transposes_tsdf_output(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, np.ndarray] = {}

    class FakeViews:
        def __init__(self, depthmaps, ks, rs, ts):
            captured["depthmaps"] = np.array(depthmaps)
            captured["ks"] = np.array(ks)
            captured["rs"] = np.array(rs)
            captured["ts"] = np.array(ts)

    def fake_tsdf_fusion(_views, depth, height, width, vx_size, truncation, unknown_is_free):
        assert depth == height == width == 8
        assert np.isclose(vx_size, 1 / 8)
        assert np.isclose(truncation, 10 / 8)
        assert unknown_is_free is False
        return (np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),)

    monkeypatch.setattr(fuse_mod, "PyViews", FakeViews)
    monkeypatch.setattr(fuse_mod, "tsdf_fusion", fake_tsdf_fusion)

    depthmaps = [np.ones((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32)]
    rotations = [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)]
    out = fuse_mod.fuse(depthmaps, rotations, resolution=8, fx=10.0, fy=10.0, cx=2.0, cy=2.0)

    assert out.shape == (4, 3, 2)
    np.testing.assert_array_equal(out, np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 1, 0))
    assert captured["depthmaps"].shape == (2, 4, 4)
    assert captured["ks"].shape == (2, 3, 3)
    assert captured["rs"].shape == (2, 3, 3)
    assert captured["ts"].shape == (2, 3)


def test_render_handles_tuple_render_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRenderer:
        def __init__(self, width: int, height: int):
            self.width = width
            self.height = height
            self.deleted = False

        def render(self, _scene, flags=None):
            assert flags == 1
            depth = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
            color = np.zeros((2, 2, 3), dtype=np.uint8)
            return color, depth

        def delete(self):
            self.deleted = True

    class FakeIntrinsicsCamera:
        def __init__(self, *args):
            self.args = args

    class FakePrimitive:
        def __init__(self, positions, indices):
            self.positions = positions
            self.indices = indices

    class FakeMesh:
        def __init__(self, primitives):
            self.primitives = primitives

    class FakeScene:
        def __init__(self):
            self.nodes = []

        def add(self, node):
            self.nodes.append(node)

    fake_pyrender = SimpleNamespace(
        OffscreenRenderer=FakeRenderer,
        IntrinsicsCamera=FakeIntrinsicsCamera,
        Primitive=FakePrimitive,
        Mesh=FakeMesh,
        Scene=FakeScene,
        RenderFlags=SimpleNamespace(DEPTH_ONLY=1),
    )
    monkeypatch.setattr(fuse_mod, "pyrender", fake_pyrender)

    mesh = {
        "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        "faces": np.array([[0, 1, 2]], dtype=np.int32),
    }
    rotations = [np.eye(3, dtype=np.float32)]

    depthmaps = fuse_mod.render(
        mesh=mesh,
        rotations=rotations,
        resolution=2,
        width=2,
        height=2,
        fx=1.0,
        fy=1.0,
        cx=1.0,
        cy=1.0,
        znear=0.1,
        zfar=10.0,
        offset=1.5,
        erode=False,
        flip_faces=False,
        show=False,
    )

    assert len(depthmaps) == 1
    expected = np.array([[9.25, 0.25], [1.25, 9.25]], dtype=np.float32)
    np.testing.assert_allclose(depthmaps[0], expected)
