from typing import Optional
import warnings

from trimesh import Trimesh

try:
    from simplify_mesh import mesh_simplify
except ImportError:
    warnings.warn('The `simplify` library is not installed.'
                  'Please install it using `python libs/libmanager.py install simplify`')
    raise


def simplify_mesh(mesh: Trimesh,
                  target_faces: Optional[int] = None,
                  target_percent: Optional[float] = None,
                  aggressiveness: float = 7.0) -> Trimesh:
    """
    Simplify a mesh using the mesh_simplify function from libsimplify.
    Args:
        mesh: The mesh to simplify.
        target_faces: The target number of faces.
        target_percent: The target percentage of faces.
        aggressiveness: The aggressiveness of the simplification.

    Returns:
        The simplified mesh.
    """
    assert target_faces is not None or target_percent is not None, "Either target_faces or target_percent must be set."
    if target_percent is not None:
        target_faces = int(target_percent * len(mesh.faces))
    elif target_percent is not None and target_faces is not None:
        target_faces = min(target_faces, int(target_percent * len(mesh.faces)))
    vertices, faces = mesh_simplify(mesh.vertices, mesh.faces, target_faces, aggressiveness)
    return Trimesh(vertices, faces, process=False)
