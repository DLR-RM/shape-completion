import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import mcubes
import numpy as np
from trimesh import Trimesh

from utils import setup_logger

logger = setup_logger(__name__)


def normalize_pointcloud(
    pointcloud: np.ndarray, center: bool = True, scale: bool = True, cube_or_sphere: str = "cube"
) -> np.ndarray:
    if center:
        if cube_or_sphere in ["cube", "sphere"]:
            pointcloud -= (pointcloud.max(axis=0) + pointcloud.min(axis=0)) / 2  # Center to midpoint
        # elif cube_or_sphere == "sphere":
        #     pointcloud -= pointcloud.mean(axis=0)  # Center to centroid
        else:
            raise ValueError(f"Normalization shape '{cube_or_sphere}' not supported.")
    if scale:
        if cube_or_sphere == "cube":
            pointcloud /= ((pointcloud.max(axis=0) - pointcloud.min(axis=0)).max()) / 2  # Scale to [-1, 1]
        elif cube_or_sphere == "sphere":
            pointcloud /= np.linalg.norm(pointcloud, axis=1).max()  # Scale to unit sphere
        else:
            raise ValueError(f"Normalization shape '{cube_or_sphere}' not supported.")
    return pointcloud


def part_sphere(
    center: np.ndarray | Sequence[float],
    radius: float,
    mode: str,
    dist_above_center: float = 0.0,
    part_sphere_dir_vector: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Samples a point from the surface or from the interior of solid sphere which is split in two parts.

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


def sphere(center: np.ndarray | Sequence[float], radius: float, mode: str) -> np.ndarray:
    """Samples a point from the surface or from the interior of solid sphere.

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


def extract(
    grid: np.ndarray, level: float, resolution: int, pad: bool = True, return_type: str = "dict"
) -> Trimesh | dict[str, np.ndarray]:
    if pad:
        grid = np.pad(grid, 1, "constant", constant_values=-1e6)
    vertices, triangles = mcubes.marching_cubes(grid, level)
    if pad:
        vertices -= 1
    vertices /= resolution - 1
    vertices -= 0.5

    if return_type == "trimesh":
        return Trimesh(vertices=vertices, faces=triangles, process=False, validate=False)
    elif return_type == "dict":
        return {"vertices": vertices, "faces": triangles}
    else:
        raise ValueError(f"Unknown return type '{return_type}'.")


def get_vertices_and_faces(
    mesh: Trimesh | dict[str, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, Trimesh):
        vertices, faces = mesh.vertices, mesh.faces
    elif isinstance(mesh, dict):
        vertices, faces = mesh["vertices"], mesh["faces"]
    elif isinstance(mesh, (list, tuple)):
        assert len(mesh) == 2, "Expected mesh to be a list or vertices and faces."
        vertices, faces = mesh
    else:
        raise ValueError(f"Unknown mesh type '{type(mesh)}'.")
    return vertices.astype(np.float32), faces.astype(np.int64)


def load_scripts(
    script_dir: Path, num_vertices: int | None = None, min_vertices: int = 20_000, max_vertices: int = 200_000
) -> list[Path]:
    """Load and modify MeshLab scripts based on the number of vertices.

    This function searches for MeshLab script files (*.mlx) in the specified directory
    and returns a list of script file paths. If a 'simplify.mlx' script is found and
    the `num_vertices` parameter is provided, the function modifies the simplification
    script to adjust the percentage reduction based on the number of vertices.

    Args:
        script_dir (Path): The directory path where the MeshLab script files are located.
        num_vertices (int | None): The number of vertices in the mesh. If provided,
            the simplification script will be modified based on this value.
            Default is None.
        min_vertices (int): The minimum number of vertices for determining the percentage
            reduction in the simplification script. Default is 20000.
        max_vertices (int): The maximum number of vertices for determining the percentage
            reduction in the simplification script. Default is 200000.

    Returns:
        list[Path]: A list of file paths to the MeshLab script files, including the
            modified simplification script if applicable.

    Raises:
        AssertionError: If the 'simplify.mlx' script does not contain the expected content
            or if the modification of the script fails.

    Notes:
        - The function assumes that the 'simplify.mlx' script contains the line
          'Simplification: Quadric Edge Collapse Decimation'.
        - The percentage reduction is calculated based on the `num_vertices` value and
          the range defined by `min_vertices` and `max_vertices`.
        - If `num_vertices` is less than `min_vertices`, the percentage reduction is set
          to 0.5 (50%).
        - If `num_vertices` is greater than `max_vertices`, the percentage reduction is
          set to 0.1 (10%).
        - If `num_vertices` is between `min_vertices` and `max_vertices`, the percentage
          reduction is interpolated linearly between 0.5 and 0.1 based on the number of
          vertices.
        - The modified simplification script is saved to a temporary file, and the file
          path is included in the returned list of script paths.
    """
    scripts = sorted(script_dir.glob("*.mlx"))
    logger.debug(f"Found {len(scripts)} scripts in {script_dir}.")
    simplify = any(script.name == "simplify.mlx" for script in scripts)
    if simplify and num_vertices is not None:
        index = next(i for i, s in enumerate(scripts) if s.name == "simplify.mlx")
        script_path = scripts[index]
        scripts[index] = modify_simplify(script_path, num_vertices, min_vertices, max_vertices)
    return scripts


def modify_simplify(
    script_path: Path, num_vertices: int, max_vertices: int = 20_000, min_vertices: int = 200_000
) -> Path:
    assert script_path.name == "simplify.mlx", f"Expected script to be 'simplify.mlx', but got '{script_path.name}'."

    if num_vertices < min_vertices:
        percentage = 0.5
    elif num_vertices > max_vertices:
        percentage = 0.1
    else:
        percentage = round(0.5 + (num_vertices - min_vertices) / (max_vertices - min_vertices) * (0.1 - 0.5), 2)
    logger.debug(f"\tload_scripts: Simplifying mesh by {100 * (1 - percentage):.0f}%.")

    script = script_path.read_text()
    assert "Simplification: Quadric Edge Collapse Decimation" in script
    script = script.replace(
        '"Percentage reduction (0..1)" value="0.05"', f'"Percentage reduction (0..1)" value="{percentage}"'
    )
    assert f'"Percentage reduction (0..1)" value="{percentage}"' in script

    script_path = Path(tempfile.mkstemp(suffix=".mlx")[1])
    script_path.write_text(script)
    logger.debug(f"\tload_scripts: Saved modified simplification script to {script_path}.")
    return script_path


def apply_meshlab_filters(
    vertices: np.ndarray, faces: np.ndarray, script_paths: list[Path]
) -> tuple[np.ndarray, np.ndarray]:
    import pymeshlab

    pymeshlab_mod = cast(Any, pymeshlab)
    ms = pymeshlab_mod.MeshSet()
    pymesh = pymeshlab_mod.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms.add_mesh(pymesh)
    for script_path in script_paths:
        logger.debug(f"\tprocess: Applying script {script_path.name}.")
        ms.load_filter_script(str(script_path))
        ms.apply_filter_script()
    return ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()
