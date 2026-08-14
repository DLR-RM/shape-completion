import numpy as np
import pytest
import trimesh

from .. import check_mesh_contains

if check_mesh_contains is None:
    pytest.skip("libintersect extension is not installed", allow_module_level=True)


def test_check_mesh_contains():
    points = np.random.rand(100, 3)
    mesh = trimesh.primitives.creation.uv_sphere(radius=1.0)
    check_mesh_contains(mesh, points)
