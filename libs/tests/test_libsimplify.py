import pytest
import trimesh

from .. import simplify_mesh

if simplify_mesh is None:
    pytest.skip("libsimplify extension is not installed", allow_module_level=True)


def test_simplify_mesh():
    mesh = trimesh.primitives.creation.icosphere()
    assert len(simplify_mesh(mesh, target_faces=100).faces) == 100
