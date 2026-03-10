import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def o3d_viz_enabled() -> bool:
    return os.environ.get("SHOW_O3D_TEST_VIZ") == "1"


@pytest.fixture(scope="session")
def shapenet_data_root() -> Path:
    root = os.environ.get("SHAPENET_ROOT")
    if root:
        return Path(root)

    candidates = [
        Path("/data/shapenet"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


@pytest.fixture(autouse=True)
def _disable_open3d_windows(monkeypatch: pytest.MonkeyPatch, o3d_viz_enabled: bool) -> None:
    if o3d_viz_enabled:
        return

    try:
        import open3d as o3d
    except ImportError:
        return

    visualization = getattr(o3d, "visualization", None)
    if visualization is None:
        return

    def _no_window(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(visualization, "draw_geometries", _no_window, raising=False)

    visualizer_cls = getattr(visualization, "Visualizer", None)
    if visualizer_cls is not None and hasattr(visualizer_cls, "create_window"):
        original_create_window: Callable[..., Any] = visualizer_cls.create_window

        def _create_window_no_visible(self: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs["visible"] = False
            return original_create_window(self, *args, **kwargs)

        monkeypatch.setattr(visualizer_cls, "create_window", _create_window_no_visible, raising=False)
