import trimesh

from .. import simplify_mesh


def test_simplify_mesh():
    mesh = trimesh.primitives.creation.icosphere()
    assert len(simplify_mesh(mesh, target_faces=100).faces) == 100
