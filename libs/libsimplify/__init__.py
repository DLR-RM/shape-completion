import warnings

from trimesh import Trimesh

try:
    from simplify_ext import mesh_simplify
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        "The `simplify` library is not installed. Please install it using `python libs/libmanager.py install simplify`",
        stacklevel=2,
    )
    raise


def simplify_mesh(
    mesh: Trimesh, target_faces: int | None = None, target_percent: int | None = None, aggressiveness: float = 7.0
) -> Trimesh:
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
    if target_faces is not None and target_percent is not None:
        num_faces = min(target_faces, int(target_percent / 100 * len(mesh.faces)))
    elif target_faces is not None:
        num_faces = target_faces
    elif target_percent is not None:
        num_faces = int(target_percent / 100 * len(mesh.faces))
    else:
        raise ValueError("Either target_faces or target_percent must be set.")

    vertices, faces = mesh_simplify(mesh.vertices, mesh.faces, num_faces, aggressiveness)
    return Trimesh(vertices, faces, process=False)
