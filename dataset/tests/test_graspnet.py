from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..src import graspnet as graspnet_module
from ..src.graspnet import GraspNetEval


def test_mesh_decimation_failure_disables_once(monkeypatch, caplog):
    dataset = GraspNetEval.__new__(GraspNetEval)
    dataset.mesh_simplify_fraction = 0.1
    dataset._mesh_decimation_enabled = True
    dataset._mesh_decimation_disabled_reason = None
    dataset._mesh_cache = {}
    dataset._points_cache = {}

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    monkeypatch.setattr(graspnet_module, "load_mesh", lambda _path: (vertices, faces))

    call_counter = {"count": 0}

    def _raise_missing_dependency(self, _target_fraction):
        call_counter["count"] += 1
        raise ModuleNotFoundError("No module named 'fast_simplification'")

    monkeypatch.setattr(graspnet_module.Trimesh, "simplify_quadric_decimation", _raise_missing_dependency)

    with caplog.at_level(logging.WARNING):
        dataset._load_mesh_cached(Path("obj_001.ply"))
        dataset._load_mesh_cached(Path("obj_002.ply"))

    warning_lines = [
        record.message for record in caplog.records if "Mesh decimation disabled after first failure" in record.message
    ]
    assert len(warning_lines) == 1
    assert call_counter["count"] == 1
    assert dataset._mesh_decimation_enabled is False
    assert dataset._mesh_decimation_disabled_reason is not None
