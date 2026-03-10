import itertools
import os
from typing import Any, cast

import pytest

if os.environ.get("SC_RUN_RENDERER_TESTS", "").lower() not in {"1", "true", "yes"}:
    pytest.skip(
        "Renderer tests are opt-in. Set SC_RUN_RENDERER_TESTS=1 to run this module.",
        allow_module_level=True,
    )

o3d = pytest.importorskip("open3d")
pyrender = pytest.importorskip("pyrender")
torch = pytest.importorskip("torch")

renderer_module = pytest.importorskip("visualize.src.renderer")
PACKAGES = renderer_module.PACKAGES
Renderer = renderer_module.Renderer

if not torch.cuda.is_available():
    pytest.skip("Renderer tests require CUDA.", allow_module_level=True)

HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

pytestmark = [pytest.mark.renderer, pytest.mark.gpu]


def _methods_to_test() -> list[str]:
    methods = [method for method in ["blender", "open3d", "pyrender", "pytorch3d"] if method in PACKAGES]
    methods.append("auto")
    return methods


def _offscreen_values(method: str) -> list[bool]:
    if HAS_DISPLAY:
        return [False, True]
    if method in {"open3d", "pyrender", "auto"}:
        return [True]
    return [False, True]


class TestRenderer:
    def test_init(self):
        differentiable_values = [False, True] if "pytorch3d" in PACKAGES else [False]
        raytracing_values = [False, True] if "blender" in PACKAGES else [False]
        for method in _methods_to_test():
            for offscreen, differentiable, raytracing, color, depth, normal in itertools.product(
                _offscreen_values(method),
                differentiable_values,
                raytracing_values,
                [False, True],
                [False, True],
                [False, True],
            ):
                if differentiable and method not in {"pytorch3d", "auto"}:
                    continue
                if raytracing and method not in {"blender", "auto"}:
                    continue
                if offscreen and normal and method == "open3d":
                    continue
                if normal and method == "pytorch3d":
                    continue

                renderer = Renderer(
                    method=method,
                    offscreen=offscreen,
                    render_color=color,
                    render_depth=depth,
                    render_normal=normal,
                    differentiable=differentiable,
                    raytracing=raytracing,
                )
                if method == "blender" or raytracing:
                    assert renderer.renderer is None
                elif method == "pyrender":
                    if offscreen:
                        assert isinstance(renderer.renderer, pyrender.OffscreenRenderer)
                        assert os.environ["PYOPENGL_PLATFORM"] == "egl"
                    else:
                        assert isinstance(renderer.renderer, pyrender.Renderer)
                        assert os.environ.get("PYOPENGL_PLATFORM") in [renderer.pyopengl_platform, None]
                elif method == "open3d":
                    visualization = getattr(o3d, "visualization", None)
                    if visualization is None:
                        pytest.skip("open3d visualization backend unavailable")
                    visualization = cast(Any, visualization)
                    if offscreen:
                        assert isinstance(renderer.renderer, visualization.rendering.OffscreenRenderer)
                        assert not normal
                    else:
                        assert isinstance(renderer.renderer, visualization.Visualizer)
                elif method == "pytorch3d" or differentiable:
                    assert renderer.renderer is None
                else:
                    assert renderer.renderer is not None

    def test_method(self):
        renderer = Renderer()
        methods = ["auto"]
        methods.extend([method for method in ["blender", "pyrender", "open3d", "pytorch3d"] if method in PACKAGES])
        for method in methods:
            renderer.method = method

    def test_offscreen(self):
        if "open3d" in PACKAGES:
            renderer = Renderer(method="open3d", offscreen=True)
            visualization = getattr(o3d, "visualization", None)
            if visualization is None:
                pytest.skip("open3d visualization backend unavailable")
            visualization = cast(Any, visualization)
            assert isinstance(renderer.renderer, visualization.rendering.OffscreenRenderer)
            if HAS_DISPLAY:
                renderer.offscreen = False
                assert isinstance(renderer.renderer, visualization.Visualizer)

        if "pyrender" in PACKAGES:
            renderer = Renderer(method="pyrender", offscreen=True)
            assert isinstance(renderer.renderer, pyrender.OffscreenRenderer)
            assert os.environ["PYOPENGL_PLATFORM"] == "egl"
            if HAS_DISPLAY:
                renderer.offscreen = False
                assert isinstance(renderer.renderer, pyrender.Renderer)
                assert os.environ.get("PYOPENGL_PLATFORM") in [renderer.pyopengl_platform, None]
            renderer.offscreen = True
            assert os.environ["PYOPENGL_PLATFORM"] == "egl"
            assert isinstance(renderer.renderer, pyrender.OffscreenRenderer)

    def test_differentiable(self):
        if "pytorch3d" not in PACKAGES:
            pytest.skip("PyTorch3D backend unavailable")

        renderer = Renderer(method="pytorch3d", differentiable=False)
        assert renderer.renderer is None

        renderer = Renderer(differentiable=True)
        assert renderer.method == "pytorch3d"
        assert renderer.renderer is None

        renderer = Renderer()
        renderer.differentiable = True
        assert renderer.method == "pytorch3d"
        assert renderer.renderer is None

    def test_raytracing(self):
        if "blender" not in PACKAGES:
            pytest.skip("Blender backend unavailable")

        renderer = Renderer(method="blender", raytracing=False)
        assert renderer.renderer is None

        renderer = Renderer(raytracing=True)
        assert renderer.renderer is None

        renderer = Renderer()
        renderer.raytracing = True
        assert renderer.renderer is None

        renderer = Renderer(method="pyrender")
        assert isinstance(renderer.renderer, pyrender.OffscreenRenderer)
        renderer.raytracing = True
        assert renderer.method == "blender"
        assert renderer.renderer is None
