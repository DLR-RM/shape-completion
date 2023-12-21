from typing import Union, Type, Dict, Tuple, List

import mcubes
import numpy as np
from trimesh import Trimesh

from utils import setup_logger

logger = setup_logger(__name__)


def resolve_dtype(precision: int,
                  integer: bool = False,
                  unsigned: bool = False) -> Type[np.dtype]:
    if precision == 8:
        return np.uint8 if unsigned else np.int8
    elif precision == 16:
        return np.float16 if not integer else np.uint16 if unsigned else np.int16
    elif precision == 32:
        return np.float32 if not integer else np.uint32 if unsigned else np.int32
    elif precision == 64:
        return np.float64 if not integer else np.uint64 if unsigned else np.int64
    else:
        raise Exception(f"Invalid precision: {precision}.")


def normalize_mesh(mesh: Trimesh,
                   center: bool = True,
                   scale: bool = True) -> Trimesh:
    if center:
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
    if scale:
        mesh.apply_scale(1 / mesh.extents.max())
    return mesh


def look_at(eye: np.ndarray,
            target: np.ndarray,
            up: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    return np.array([
        [x_axis[0], y_axis[0], z_axis[0], eye[0]],
        [x_axis[1], y_axis[1], z_axis[1], eye[1]],
        [x_axis[2], y_axis[2], z_axis[2], eye[2]],
        [0, 0, 0, 1]
    ])


def part_sphere(center: Union[np.ndarray, list],
                radius: float, mode: str,
                dist_above_center: float = 0.0,
                part_sphere_dir_vector: Union[np.ndarray, list] = None) -> np.ndarray:
    """ Samples a point from the surface or from the interior of solid sphere which is split in two parts.

    https://math.stackexchange.com/a/87238
    https://math.stackexchange.com/a/1585996

    Example 1: Sample a point from the surface of the sphere that is split by a plane with displacement of 0.5
    above center and a normal of [1, 0, 0].

    .. code-block:: python

        PartSphere.sample(
            center=[0, 0, 0],
            part_sphere_vector=[1, 0, 0],
            mode="SURFACE",
            distance_above_center=0.5
        )

    :param center: Location of the center of the sphere.
    :param radius: The radius of the sphere.
    :param mode: Mode of sampling. Determines the geometrical structure used for sampling. Available: SURFACE (sampling
                 from the 2-sphere), INTERIOR (sampling from the 3-ball).
    :param dist_above_center: The distance above the center, which should be used. Default: 0.0 (half of the sphere).
    :param part_sphere_dir_vector: The direction in which the sphere should be split, the end point of the vector, will be in the middle of
                                   the sphere pointing towards the middle of the resulting surface. Default: [0, 0, 1].
    :return: A random point lying inside or on the surface of a solid sphere.
    """
    if part_sphere_dir_vector is None:
        part_sphere_dir_vector = np.array([0, 0, 1], np.float32)
    else:
        part_sphere_dir_vector = np.array(part_sphere_dir_vector).astype(np.float32)
    part_sphere_dir_vector /= np.linalg.norm(part_sphere_dir_vector)

    if dist_above_center >= radius:
        raise Exception("The dist_above_center value is bigger or as big as the radius!")

    while True:
        location = sphere(center, radius, mode)
        # project the location onto the part_sphere_dir_vector and get the length
        loc_in_sphere = location - np.array(center)
        length = loc_in_sphere.dot(part_sphere_dir_vector)
        if length > dist_above_center:
            return location


def sphere(center: Union[np.ndarray, list],
           radius: float,
           mode: str) -> np.ndarray:
    """ Samples a point from the surface or from the interior of solid sphere.

    https://math.stackexchange.com/a/87238
    https://math.stackexchange.com/a/1585996

    Example 1: Sample a point from the surface of the solid sphere of a defined radius and center location.

    .. code-block:: python

        Sphere.sample(
            center=Vector([0, 0, 0]),
            radius=2,
            mode="SURFACE"
        )

    :param center: Location of the center of the sphere.
    :param radius: The radius of the sphere.
    :param mode: Mode of sampling. Determines the geometrical structure used for sampling. Available: SURFACE (sampling
                 from the 2-sphere), INTERIOR (sampling from the 3-ball).
    """
    center = np.array(center)

    # Sample
    direction = np.random.normal(loc=0.0, scale=1.0, size=3)

    if np.count_nonzero(direction) == 0:  # Check no division by zero
        direction[0] = 1e-5

    # For normalization
    norm = np.sqrt(direction.dot(direction))

    # If sampling from the surface set magnitude to radius of the sphere
    if mode == "SURFACE":
        magnitude = radius
    # If sampling from the interior set it to scaled radius
    elif mode == "INTERIOR":
        magnitude = radius * np.cbrt(np.random.uniform())
    else:
        raise Exception("Unknown sampling mode: " + mode)

    # Normalize
    sampled_point = list(map(lambda x: magnitude * x / norm, direction))

    # Add center
    location = np.array(sampled_point) + center

    return location


def extract(grid: np.ndarray,
            level: float,
            resolution: int,
            pad: bool = True,
            return_type: str = "dict") -> Union[Trimesh, Dict[str, np.ndarray]]:
    if pad:
        grid = np.pad(grid, 1, "constant", constant_values=-1e6)
    vertices, triangles = mcubes.marching_cubes(grid, level)
    if pad:
        vertices -= 1
    vertices /= (resolution - 1)
    vertices -= 0.5

    if return_type == "trimesh":
        return Trimesh(vertices=vertices,
                       faces=triangles,
                       process=False,
                       validate=False)
    elif return_type == "dict":
        return {"vertices": vertices, "faces": triangles}
    else:
        raise ValueError(f"Unknown return type '{return_type}'.")


def get_vertices_and_faces(mesh: Union[Trimesh, Dict[str, np.ndarray], List[np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, Trimesh):
        vertices, faces = mesh.vertices, mesh.faces
    elif isinstance(mesh, dict):
        vertices, faces = mesh["vertices"], mesh["faces"]
    elif isinstance(mesh, (list, tuple)):
        assert len(mesh) == 2, f"Expected mesh to be a list or vertices and faces."
        vertices, faces = mesh
    else:
        raise ValueError(f"Unknown mesh type '{type(mesh)}'.")
    return vertices.astype(np.float32), faces.astype(np.int64)
