import os
import copy
import time
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
from typing import Union, Optional, List
import subprocess

from trimesh import Trimesh
from joblib import Parallel, delayed, cpu_count
import numpy as np
import open3d as o3d
import pybullet
import pybullet_data
from pybullet_utils import bullet_client
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation as R

from utils import (setup_logger, set_log_level, tqdm_joblib, stdout_redirected, eval_input, load_mesh, save_mesh,
                   save_command_and_args_to_file)
from libs import simplify_mesh

logger = setup_logger(__name__)


def build_vhacd() -> Path:
    import git
    import libs
    vhacd_path = Path(libs.__file__).parent
    if not (vhacd_path / "v-hacd").exists():
        vhacd_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Downloading v-hacd library into {vhacd_path}")
        git.Git(vhacd_path).clone("https://github.com/kmammou/v-hacd.git")
    binary_path = vhacd_path / "v-hacd" / "app" / "build" / "TestVHACD"
    if not binary_path.exists():
        logger.debug("Building v-hacd")
        os.system(f"cd {str(vhacd_path)}/v-hacd/app; cmake -S . -B build -DCMAKE_BUILD_TYPE=Release; cmake --build build")
    assert binary_path.exists(), f"Could not build v-hacd. Please check {vhacd_path / 'v-hacd' / 'app' / 'build'}"
    return binary_path


def run_vhacd(path_to_mesh: Path,
              resolution: int = int(1e6),
              max_num_vertices_per_ch: int = 64,
              depth: int = 20,
              verbose: bool = False) -> Path:
    vhacd_binary = build_vhacd()
    if vhacd_binary.exists():
        logger.debug(f"Found V-HACD binary at {vhacd_binary}")

    out_dir = path_to_mesh.parent
    cmd_line = f'{vhacd_binary} {path_to_mesh} -r {resolution} -v {max_num_vertices_per_ch} -d {depth}'
    logger.debug(f"Running V-HACD with command line: {cmd_line}")
    with stdout_redirected(enabled=not verbose):
        subprocess.run(cmd_line, bufsize=-1, close_fds=True, shell=True, cwd=out_dir)

    for file in ["decomp.stl", "decomp.mtl"]:
        (out_dir / file).unlink(missing_ok=True)

    out_path = out_dir / f"vhacd.obj"
    (out_dir / "decomp.obj").rename(out_path)
    assert out_path.exists(), f"V-HACD failed to produce output file {out_path}"

    return out_path


def center_mesh(mesh: Trimesh) -> Trimesh:
    offset = -mesh.extents.mean(axis=0)
    return mesh.apply_translation(offset)


def angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    cosine_theta = np.dot(u, v)
    cosine_theta = np.clip(cosine_theta, -1.0, 1.0)
    theta = np.arccos(cosine_theta)
    return theta


def duplicate(r1: np.ndarray,
              r2: np.ndarray,
              up_axis: int = 2,
              tolerance: float = 1.0) -> bool:
    return np.rad2deg(angle_between_vectors(r1[:3, up_axis], r2[:3, up_axis])) < tolerance


def filter_matrices(rotation_matrices: List[np.ndarray], up_axis: int = 2) -> List[np.ndarray]:
    lst = rotation_matrices
    unique_matrices = [x for i, x in enumerate(lst) if not any(duplicate(x, y, up_axis) for y in lst[:i])]
    logger.debug(f"Found {len(unique_matrices)} unique poses.")
    return unique_matrices


def simplify_types(value: str) -> Union[bool, int, float]:
    if value == "True":
        return True
    elif value == "False":
        return False
    try:
        int_val = int(value)
        if 1 <= int_val <= int(1e5):
            return int_val
        else:
            raise ArgumentTypeError("Integer value for --simplify must be in [1, 10000]")
    except ValueError:
        try:
            float_val = float(value)
            if 0 < float_val < 1:
                return float_val
            else:
                raise ArgumentTypeError("Float value for --simplify must be in (0, 1)")
        except ValueError:
            raise ArgumentTypeError("Invalid value provided for --simplify. Must be one of [True, False, int, float]")


def show_types(value: str) -> Union[bool, str]:
    if value == "True":
        return True
    elif value == "False":
        return False
    elif value == "poses":
        return value
    else:
        raise ArgumentTypeError(f"Invalid value '{value}' provided for --show. Must be one of [True, False, poses].")


def simulate(file: Path, args: Namespace, p: Optional[bullet_client.BulletClient] = None):
    start = time.time()
    logger.debug(f"Processing {file}")

    out_dir = file.parent if args.out_dir is None else args.out_dir.expanduser().resolve()
    out_path = out_dir / "poses.npy"
    if out_path.exists() and not args.overwrite and not args.show:
        logger.debug(f"File {out_path} already processed. Skipping.")
        return

    if p is None:
        with stdout_redirected():
            p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

    restart = time.time()
    obj_path = file
    obj_mesh = Trimesh(*load_mesh(obj_path))
    if obj_path.suffix != ".obj":
        obj_path = obj_path.parent / "physics.obj"
        save_mesh(obj_path, obj_mesh.vertices, obj_mesh.faces)
    logger.debug(f"Loading mesh took {time.time() - restart:.2f}s.")

    if args.simplify:
        restart = time.time()
        num_faces = len(obj_mesh.triangles)
        if type(args.simplify) == float:
            target_num_faces = int(args.simplify * num_faces)
        elif type(args.simplify) == int:
            target_num_faces = args.simplify
        else:
            target_num_faces = int(1e5)
        if num_faces > target_num_faces:
            obj_path = file.parent / "simplified.obj"
            if not obj_path.exists():
                logger.debug(f"Simplifying mesh to {target_num_faces} faces.")
                obj_mesh = simplify_mesh(obj_mesh, target_num_faces)
                logger.debug(f"Writing simplified mesh to {obj_path}.")
                save_mesh(obj_path, obj_mesh.vertices, obj_mesh.faces)
            else:
                logger.debug(f"Found simplified mesh at {obj_path}.")
                obj_mesh = Trimesh(*load_mesh(obj_path))
        else:
            logger.debug(f"Mesh has less than {target_num_faces} faces. Skipping simplification.")
        logger.debug(f"Simplifying mesh took {time.time() - restart:.2f}s.")

    if args.solidify:
        restart = time.time()
        try:
            import bpy
        except ImportError:
            logger.error("Could not import the Blender Python API (bpy). Use `pip install bpy` to use this feature.")
            raise

        obj_path = file.parent / "solidified.obj"
        if not obj_path.exists():
            logger.debug(f"Solidifying mesh.")
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)

            bpy_mesh = bpy.data.meshes.new("Mesh")
            bpy_mesh.from_pydata(obj_mesh.vertices.tolist(),
                                 [],
                                 obj_mesh.triangles.tolist())
            bpy_mesh.update()
            bpy_mesh.validate()

            bpy_obj = bpy.data.objects.new("Object", bpy_mesh)
            bpy.context.collection.objects.link(bpy_obj)

            bpy.data.objects["Object"].select_set(True)
            bpy.context.view_layer.objects.active = bpy.data.objects["Object"]

            solidify = bpy.data.objects["Object"].modifiers.new(name="Solidify", type='SOLIDIFY')
            solidify.thickness = 0.001
            bpy.ops.object.modifier_apply(modifier=solidify.name)

            triangulate = bpy.data.objects["Object"].modifiers.new(name="Triangulate", type='TRIANGULATE')
            bpy.ops.object.modifier_apply(modifier=triangulate.name)

            bpy_mesh = bpy.data.objects["Object"].data
            vertices = np.array([v.co for v in bpy_mesh.vertices])
            faces = np.array([p.vertices[:] for p in bpy_mesh.polygons])

            obj_mesh = Trimesh(vertices, faces)
            save_mesh(obj_path, obj_mesh.vertices, obj_mesh.faces)
        else:
            logger.debug(f"Found solidified mesh at {obj_path}.")
            obj_mesh = Trimesh(*load_mesh(obj_path))
        logger.debug(f"Solidifying mesh took {time.time() - restart:.2f}s.")

    if args.center:
        restart = time.time()
        obj_mesh = center_mesh(obj_mesh)
        obj_path = file.parent / "centered.obj"
        save_mesh(obj_path, obj_mesh.vertices, obj_mesh.faces)
        logger.debug(f"Centering mesh took {time.time() - restart:.2f}s.")

    if args.vhacd:
        restart = time.time()
        vhacd_path = obj_path.parent / "vhacd.obj"
        log_path = obj_path.parent / "vhacd_log.txt"
        if not vhacd_path.exists():
            logger.debug(f"Creating V-HACD mesh at {vhacd_path}.")
            try:
                run_vhacd(obj_path, verbose=args.verbose)
            except Exception as e:
                logger.exception(e)
                with stdout_redirected(enabled=not args.verbose):
                    p.vhacd(str(obj_path),
                            str(vhacd_path),
                            str(log_path),
                            resolution=int(1e6))
            log_path.unlink(missing_ok=True)
        else:
            logger.debug(f"Found V-HACD mesh at {vhacd_path}.")
        logger.debug(f"V-HACD took {time.time() - restart:.2f}s.")

    restart = time.time()
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane100.urdf",
               useMaximalCoordinates=True,
               useFixedBase=True)

    # Load object into pybullet
    collisionShapeId = p.createCollisionShape(shapeType=pybullet.GEOM_MESH,
                                              fileName=str(vhacd_path) if args.vhacd else str(obj_path),
                                              collisionFramePosition=[0, 0, 0],
                                              meshScale=1)
    visualShapeId = p.createVisualShape(shapeType=pybullet.GEOM_MESH,
                                        fileName=str(obj_path),
                                        rgbaColor=[1, 1, 1, 1],
                                        specularColor=[0.4, 0.4, 0],
                                        visualFramePosition=[0, 0, 0],
                                        meshScale=1)

    if obj_mesh.is_watertight:
        volume = obj_mesh.volume
        mass = 1000 * volume
        center = obj_mesh.center_mass
        logger.debug(f"Mesh is watertight. Volume: {volume:.5f} m^3. Mass: {mass:.4f} kg. Center: {center}.")
    else:
        area = obj_mesh.area
        mass = area
        center = obj_mesh.centroid
        logger.debug(f"Mesh is NOT watertight. Area/Mass: {area:.4f} m^2/kg. Center: {center}.")

    object_id = p.createMultiBody(baseMass=mass,
                                  baseInertialFramePosition=center,
                                  baseVisualShapeIndex=visualShapeId,
                                  baseCollisionShapeIndex=collisionShapeId,
                                  basePosition=[0, 0, 0],
                                  baseOrientation=[0, 0, 0, 1])

    def get_pose(rot: R) -> np.ndarray:
        trafo = np.eye(4)
        trafo[:3, :3] = rot.as_matrix()
        mesh = obj_mesh.copy()
        mesh = mesh.apply_translation(-center)
        mesh = mesh.apply_transform(trafo)
        trafo[2, 3] = -mesh.bounds[0, 2] + 0.003
        pose = trafo[:3, 3].T.tolist() + rot.as_quat().tolist()

        p.resetBasePositionAndOrientation(bodyUniqueId=object_id, posObj=pose[:3], ornObj=pose[3:])
        p.setGravity(0, 0, args.gravity)
        p.changeDynamics(bodyUniqueId=object_id,
                         linkIndex=-1,
                         lateralFriction=0.3,
                         spinningFriction=0.0005,
                         rollingFriction=0.0002,
                         restitution=0.1)

        for _ in range(10000):
            p.stepSimulation()
            if args.debug:
                time.sleep(1. / 240.)

        t, quat = p.getBasePositionAndOrientation(bodyUniqueId=object_id)
        trafo = np.eye(4)
        trafo[:3, :3] = np.array(p.getMatrixFromQuaternion(quat)).reshape((3, 3))
        trafo[:3, 3] = np.zeros(3)
        return trafo

    poses = list()
    if args.rotate == "principal":
        for axis in ["x", "y"]:
            for angle in [0, np.pi / 2, -np.pi / 2, np.pi] if axis == "x" else [np.pi / 2, -np.pi / 2]:
                poses.append(get_pose(R.from_euler(axis, angle)))
    elif args.rotate == "random":
        for _ in trange(args.num_poses, desc="Pose", disable=not args.verbose):
            poses.append(get_pose(R.random()))

    poses = filter_matrices(poses, args.up_axis)
    if not out_path.exists() or args.overwrite:
        np.save(out_path, poses)

    if not args.keep_intermediate:
        for obj in ["physics", "simplified", "solidified", "centered", "vhacd"]:
            (file.parent / f"{obj}.obj").unlink(missing_ok=True)

    logger.debug(f"PyBullet simulation took {time.time() - restart:.2f}s.")
    logger.debug(f"Total time: {time.time() - start:.2f}s.")

    if args.show == "poses":
        obj_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(obj_mesh.vertices),
                                             o3d.utility.Vector3iVector(obj_mesh.faces))
        obj_mesh.compute_vertex_normals()
        rot_x_90 = R.from_euler("x", -np.pi / 2).as_matrix()
        for pose in poses:
            pose = rot_x_90 @ pose[:3, :3]
            o3d.visualization.draw_geometries([copy.deepcopy(obj_mesh).rotate(pose, center=[0, 0, 0]),
                                               o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)])


def run(file: Path, args: Namespace, p: Optional[bullet_client.BulletClient] = None):
    start = time.time()
    try:
        simulate(file, args, p)
    except Exception as e:
        logger.exception(e)
        logger.error(f"Failed to process {file}.")
    logger.debug(f"Runtime: {time.time() - start:.2f}s.\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("in_dir_or_path", type=Path, help="Path to input data.")
    parser.add_argument("--out_dir", type=Path, help="Path to output directory.")
    parser.add_argument("--in_format", type=str, default=".obj", help="Input file format.")
    parser.add_argument("--recursion_depth", type=int, help="Depth of recursive glob pattern matching.")
    parser.add_argument('--simplify', nargs='?', const=True, default=False, type=simplify_types,
                        help='Simplify meshes before processing. If integer, it is used as the target '
                             'number of faces. If float, it is used as the target ratio of faces.')
    parser.add_argument("--solidify", action="store_true", help="Solidify meshes before processing.")
    parser.add_argument("--vhacd", action="store_true", help="Use V-HACD to create collision mesh.")
    parser.add_argument("--keep_intermediate", action="store_true", help="Keep intermediate files.")
    parser.add_argument("--center", action="store_true", help="Center meshes before processing.")
    parser.add_argument("--rotate", type=str, default="principal", choices=["principal", "random"],
                        help="Rotate object around principal axes or randomly.")
    parser.add_argument("--num_poses", type=int, default=100, help="Number of poses to generate.")
    parser.add_argument("--up_axis", type=int, default=2, choices=[0, 1, 2], help="Up axis of inputs.")
    parser.add_argument("--show", nargs='?', const=True, default=False, type=show_types,
                        help="Visualize physics simulation (and optionally the identified poses).")
    parser.add_argument("--sort", action="store_true", help="Sort files before processing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--n_jobs", type=int, default=cpu_count(), help="Number of parallel jobs.")
    parser.add_argument("--gravity", type=float, default=-9.81, help="Gravity in z-direction.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    save_command_and_args_to_file(args.out_dir if args.out_dir else Path.cwd() / "command.txt", args)

    if args.verbose:
        set_log_level(logging.DEBUG)

    p = None
    if args.show:
        with stdout_redirected():
            p = bullet_client.BulletClient(connection_mode=pybullet.GUI)

    files = eval_input(args.in_dir_or_path, args.in_format, args.recursion_depth, args.sort)
    with tqdm_joblib(tqdm(desc="Simulating", total=len(files), disable=args.verbose)):
        Parallel(n_jobs=1 if args.show or args.verbose else min(args.n_jobs, len(files)),
                 verbose=args.verbose)(delayed(run)(file, args, p) for file in files)


if __name__ == '__main__':
    main()
