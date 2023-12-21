import math
from time import sleep, time
from typing import Union, Dict, List, Any

import numpy as np
import pyrender
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation
from trimesh import Trimesh

from utils import setup_logger
from libs import PyViews, tsdf_fusion
from .utils import extract

logger = setup_logger(__name__)


def render(mesh: Union[Trimesh, Dict[str, np.ndarray]],
           rotations: List[np.ndarray],
           resolution: int,
           width: int,
           height: int,
           fx: float,
           fy: float,
           cx: float,
           cy: float,
           znear: float,
           zfar: float,
           offset: float = 1.5,
           erode: bool = True,
           flip_faces: bool = False,
           show: bool = False) -> List[np.ndarray]:
    renderer = pyrender.OffscreenRenderer(width, height)
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear, zfar)
    rot_x_180 = Rotation.from_euler('x', 180, degrees=True).as_matrix()

    depthmaps = list()
    for R in rotations:
        R_pyrender = rot_x_180 @ R

        if isinstance(mesh, Trimesh):
            trafo = np.eye(4)
            trafo[:3, :3] = R_pyrender
            trafo[:3, 3] = np.array([0, 0, -1])

            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(trafo)

            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_copy)

            if flip_faces:
                mesh_copy.invert()
                pyrender_mesh = [pyrender_mesh, pyrender.Mesh.from_trimesh(mesh_copy)]
        elif isinstance(mesh, dict):
            vertices = mesh["vertices"].copy()
            vertices = vertices @ R_pyrender.T
            vertices[:, 2] -= 1
            faces = mesh["faces"].copy()

            primitives = [pyrender.Primitive(positions=vertices, indices=faces)]

            if flip_faces:
                primitives.append(pyrender.Primitive(positions=vertices, indices=np.flip(faces, axis=1)))

            pyrender_mesh = pyrender.Mesh(primitives=primitives)
        else:
            raise ValueError(f"Unknown mesh type '{type(mesh)}'.")

        scene = pyrender.Scene()
        scene.add(pyrender_mesh)
        scene.add(camera)

        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

        if show:
            Image.fromarray(((depth / depth.max()) * 256).astype(np.uint8)).convert('L').show()
            sleep(1)

        depth[depth == 0] = zfar

        # Optionally thicken object by offsetting and eroding the depth maps.
        depth -= offset * (1 / resolution)
        if erode:
            depth = ndimage.grey_erosion(depth, size=(3, 3))
        depthmaps.append(depth)

    renderer.delete()
    return depthmaps


def fuse(depthmaps: List[np.ndarray],
         rotations: List[np.ndarray],
         resolution: int,
         fx: float,
         fy: float,
         cx: float,
         cy: float) -> np.ndarray:
    Ks = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0, 0, 1]]).reshape((1, 3, 3))

    Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)
    Rs = np.array(rotations).astype(np.float32)
    Ts = np.array([np.array([0, 0, 1]) for _ in range(len(Rs))]).astype(np.float32)
    depthmaps = np.array(depthmaps).astype(np.float32)
    voxel_size = 1 / resolution

    views = PyViews(depthmaps, Ks, Rs, Ts)
    tsdf = tsdf_fusion(views,
                       depth=resolution,
                       height=resolution,
                       width=resolution,
                       vx_size=voxel_size,
                       truncation=10 * voxel_size,
                       unknown_is_free=False)[0].transpose((2, 1, 0))
    return tsdf


def get_points(n_views: int = 100) -> np.ndarray:
    """See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere."""
    rnd = 1.
    points = []
    offset = 2. / n_views
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(n_views):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % n_views) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)


def get_views(points: np.ndarray) -> List[np.ndarray]:
    """Generate a set of views to generate depth maps from."""
    Rs = []
    for i in range(points.shape[0]):
        # https://np.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
        longitude = - math.atan2(points[i, 0], points[i, 1])
        latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

        R_x = np.array(
            [[1, 0, 0], [0, math.cos(latitude), -math.sin(latitude)], [0, math.sin(latitude), math.cos(latitude)]])
        R_y = np.array(
            [[math.cos(longitude), 0, math.sin(longitude)], [0, 1, 0], [-math.sin(longitude), 0, math.cos(longitude)]])

        R = R_y.dot(R_x)
        Rs.append(R)

    return Rs


def pyfusion_pipeline(mesh: Union[Trimesh, Dict[str, np.ndarray]],
                      args: Any) -> Union[Trimesh, Dict[str, np.ndarray]]:
    start = time()
    rotations = get_views(get_points(args.n_views))
    logger.debug(f"Generated rotations in {time() - start:.3f}s.")

    restart = time()
    depthmaps = render(mesh,
                       rotations,
                       args.resolution,
                       args.width,
                       args.height,
                       args.fx,
                       args.fy,
                       args.cx,
                       args.cy,
                       args.znear,
                       args.zfar,
                       args.depth_offset,
                       not args.no_erosion,
                       args.flip_faces,
                       show=False)
    logger.debug(f"Rendered depth maps in {time() - restart:.3f}s.")

    restart = time()
    tsdf = fuse(depthmaps,
                rotations,
                args.resolution,
                args.fx,
                args.fy,
                args.cx,
                args.cy)
    logger.debug(f"Fused depth maps in {time() - restart:.3f}s.")

    restart = time()
    mesh = extract(grid=-tsdf,
                   level=0,
                   resolution=args.resolution,
                   return_type="trimesh" if args.use_trimesh else "dict")
    logger.debug(f"Extracted mesh in {time() - restart:.3f}s.")
    return mesh
