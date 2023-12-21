import itertools
import os

import open3d as o3d
import pyrender
from pytorch3d.renderer import MeshRasterizer

from ..src.renderer import Renderer


class TestRenderer:
    def test_init(self):
        for method in ["blender", "open3d", "pyrender", "pytorch3d", "auto", "foo"]:
            for offscreen, differentiable, raytracing, color, depth, normal in itertools.product([False, True],
                                                                                                 repeat=6):
                try:
                    renderer = Renderer(method=method,
                                        offscreen=offscreen,
                                        render_color=color,
                                        render_depth=depth,
                                        render_normal=normal,
                                        differentiable=differentiable,
                                        raytracing=raytracing)
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
                        if offscreen:
                            assert isinstance(renderer.renderer, o3d.visualization.rendering.OffscreenRenderer)
                            assert not normal
                        else:
                            assert isinstance(renderer.renderer, o3d.visualization.Visualizer)
                    elif method == "pytorch3d" or differentiable:
                        assert isinstance(renderer.renderer, MeshRasterizer)
                    else:
                        renderer.renderer  # Always initialize the renderer
                except AssertionError as e:
                    if method == "foo":
                        pass
                    elif differentiable and method != "pytorch3d":
                        pass
                    elif raytracing and method != "blender":
                        pass
                    elif offscreen and normal and method == "open3d":
                        pass
                    elif normal and method == "pytorch3d":
                        pass
                    else:
                        raise e

    def test_method(self):
        renderer = Renderer()
        for method in ["auto", "blender", "pyrender", "open3d", "pytorch3d"]:
            renderer.method = method

    def test_offscreen(self):
        renderer = Renderer(method="open3d", offscreen=True)
        assert isinstance(renderer.renderer, o3d.visualization.rendering.OffscreenRenderer)
        renderer.offscreen = False
        assert isinstance(renderer.renderer, o3d.visualization.Visualizer)
        # Fixme: Why doesn't this work?
        # renderer.offscreen = True
        # assert isinstance(renderer.renderer, o3d.visualization.rendering.OffscreenRenderer)

        renderer = Renderer(method="pyrender", offscreen=True)
        assert isinstance(renderer.renderer, pyrender.OffscreenRenderer)
        assert os.environ["PYOPENGL_PLATFORM"] == "egl"
        renderer.offscreen = False
        assert isinstance(renderer.renderer, pyrender.Renderer)
        assert os.environ.get("PYOPENGL_PLATFORM") in [renderer.pyopengl_platform, None]
        renderer.offscreen = True
        assert os.environ["PYOPENGL_PLATFORM"] == "egl"
        assert isinstance(renderer.renderer, pyrender.OffscreenRenderer)

    def test_differentiable(self):
        renderer = Renderer(method="pytorch3d", differentiable=False)
        assert isinstance(renderer.renderer, MeshRasterizer)

        renderer = Renderer(differentiable=True)
        assert isinstance(renderer.renderer, MeshRasterizer)

        renderer = Renderer()
        renderer.differentiable = True
        assert isinstance(renderer.renderer, MeshRasterizer)

    def test_raytracing(self):
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
