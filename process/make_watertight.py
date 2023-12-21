import gc
import os
import tempfile

from typing import Any, List, Tuple, Optional, Union, Dict
from pathlib import Path
from argparse import ArgumentParser
from joblib import Parallel, delayed, cpu_count
from time import time
import logging
import tracemalloc
import linecache
import shutil

import pymeshlab
from tqdm import tqdm
import trimesh
from trimesh import Trimesh
import numpy as np

from .src.utils import get_vertices_and_faces
from utils import (setup_logger, set_log_level, load_mesh, save_mesh, tqdm_joblib, eval_input, disable_multithreading,
                   resolve_out_dir, save_command_and_args_to_file)

logger = setup_logger(__name__)

MODES = ["fuse", "carve", "fill"]
try:
    from .src.fuse import pyfusion_pipeline
except ImportError:
    logger.warning("Could not import PyFusion, 'fuse' mode disabled.")
    MODES.remove("fuse")
try:
    from .src.carve import voxel_carving_pipeline
except ImportError:
    logger.warning("Could not import Open3D, 'carve' mode disabled.")
    MODES.remove("carve")
try:
    from .src.fill import kaolin_pipeline
except ImportError:
    logger.warning("Could not import PyTorch and/or NVIDIA Kaolin, 'fill' mode disabled.")
    MODES.remove("fill")
assert len(MODES) > 0, "No modes available, exiting."
logger.debug(f"Enabled modes: {MODES}")


def load(in_path: Path,
         loader: str = "pymeshlab",
         return_type: str = "dict") -> Union[Trimesh, Dict[str, np.ndarray]]:
    try:
        vertices, faces = load_mesh(in_path, load_with=loader)
    except (ValueError, pymeshlab.PyMeshLabException):
        logger.warning(f"Could not load mesh {in_path} with {loader}. Trying alternatives.")
        vertices, faces = load_mesh(in_path)

    if return_type == "dict":
        return {"vertices": vertices,
                "faces": faces}
    elif return_type == "trimesh":
        return Trimesh(vertices=vertices,
                       faces=faces,
                       process=False,
                       validate=False)
    else:
        raise ValueError(f"Unknown return type '{return_type}'.")


def normalize(mesh: Union[Trimesh, Dict[str, np.ndarray]],
              translation: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
              scale: Optional[Union[float, Tuple[float, float, float], np.ndarray]] = None,
              padding: float = 0.1) -> Tuple[Union[Trimesh, Dict[str, np.ndarray]], np.ndarray, float]:
    if isinstance(mesh, Trimesh):
        if translation is None:
            translation = -mesh.bounds.mean(axis=0)
        if scale is None:
            max_extents = mesh.extents.max()
            scale = 1 / (max_extents + 2 * padding * max_extents)
            # scale = (1 - padding) / max_extents

        mesh.apply_translation(translation)
        mesh.apply_scale(scale)
    elif isinstance(mesh, dict):
        if translation is None or scale is None:
            vertices = mesh["vertices"]
            faces = mesh["faces"]
            referenced = np.zeros(len(vertices), dtype=bool)
            referenced[faces] = True
            in_mesh = vertices[referenced]
            bounds = np.array([in_mesh.min(axis=0), in_mesh.max(axis=0)])
            if translation is None:
                translation = -bounds.mean(axis=0)
            if scale is None:
                extents = bounds.ptp(axis=0)
                max_extents = extents.max()
                scale = 1 / (max_extents + 2 * padding * max_extents)
                # scale = (1 - padding) / max_extents

        mesh["vertices"] += translation
        mesh["vertices"] *= scale
    else:
        raise ValueError(f"Unknown mesh type '{type(mesh)}'.")

    return mesh, translation, scale


def load_scripts(script_dir: Path,
                 num_vertices: Optional[int] = None,
                 min_vertices: int = 20000,
                 max_vertices: int = 200000) -> List[Path]:
    scripts = sorted(script_dir.glob("*.mlx"))
    logger.debug(f"Found {len(scripts)} scripts in {script_dir}.")
    simplify = any(script.name == 'simplify.mlx' for script in scripts)
    if simplify and num_vertices is not None:
        if num_vertices < min_vertices:
            percentage = 0.5
        elif num_vertices > max_vertices:
            percentage = 0.1
        else:
            percentage = round(0.5 + (num_vertices - min_vertices) / (max_vertices - min_vertices) * (0.1 - 0.5), 2)
        logger.debug(f"\tload_scripts: Simplifying mesh by {100 * (1 - percentage):.0f}%.")

        index = next(i for i, s in enumerate(scripts) if s.name == 'simplify.mlx')
        with open(scripts[index], 'r') as f:
            script = f.read()
            assert 'Simplification: Quadric Edge Collapse Decimation' in script
            script = script.replace('"Percentage reduction (0..1)" value="0.05"',
                                    f'"Percentage reduction (0..1)" value="{percentage}"')
            assert f'"Percentage reduction (0..1)" value="{percentage}"' in script

            scripts[index] = Path(tempfile.mkstemp(suffix=".mlx")[1])
            scripts[index].write_text(script)
            logger.debug(f"\tload_scripts: Saved modified simplification script to {scripts[index]}.")
    return scripts


def process(mesh: Union[Trimesh, Dict[str, np.ndarray]],
            script_paths: List[Path]) -> Union[Trimesh, Dict[str, np.ndarray]]:
    vertices, faces = get_vertices_and_faces(mesh)

    ms = pymeshlab.MeshSet()
    pymesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms.add_mesh(pymesh)

    for script_path in script_paths:
        logger.debug(f"\tprocess: Applying script {script_path.name}.")
        ms.load_filter_script(str(script_path))
        ms.apply_filter_script()

    if isinstance(mesh, Trimesh):
        return Trimesh(vertices=ms.current_mesh().vertex_matrix(),
                       faces=ms.current_mesh().face_matrix(),
                       process=False,
                       validate=False)
    elif isinstance(mesh, dict):
        return {"vertices": ms.current_mesh().vertex_matrix(),
                "faces": ms.current_mesh().face_matrix()}


def save(mesh: Union[Trimesh, pymeshlab.MeshSet, pymeshlab.Mesh, Dict[str, np.ndarray]],
         path: Path,
         precision: int = 32):
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = get_vertices_and_faces(mesh)
    if isinstance(mesh, Trimesh):
        if precision == 16:
            precision = np.float16
        elif precision == 32:
            precision = np.float32
        elif precision == 64:
            precision = np.float64
        else:
            raise ValueError(f"Invalid precision: {precision}.")
        precision = np.finfo(precision).precision
        save_mesh(path, vertices, faces, save_with='trimesh', digits=precision)
    elif isinstance(mesh, (pymeshlab.MeshSet, pymeshlab.Mesh, dict)):
        save_mesh(path, vertices, faces, save_with='pymeshlab')


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def trace_run(in_path: Path, args: Any):
    snapshot1 = tracemalloc.take_snapshot()

    run(in_path, args)

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("[ Top 10 differences ]")
    for stat in top_stats[:10]:
        print(stat)

    display_top(snapshot2)


def check(out_path: Path, args: Any):
    try:
        mesh = load(out_path,
                    loader="trimesh" if args.use_trimesh else "pymeshlab",
                    return_type="trimesh" if args.use_trimesh else "dict")
        vertices, faces = get_vertices_and_faces(mesh)
        assert len(vertices) > 0 and len(faces) > 0, f"Mesh {out_path} is empty."
        if args.check_watertight:
            mesh = Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
            assert mesh.is_watertight, f"Mesh {out_path} is not watertight."
    except Exception as e:
        logger.exception(e)
        if args.fix:
            try:
                os.remove(out_path)
            except Exception as e:
                logger.exception(e)
                pass


def run(in_path: Path, args: Any):
    start = time()
    logger.debug(f"Processing file {in_path}:")

    in_dir = args.in_dir_or_path.parent if args.in_dir_or_path.is_file() else args.in_dir_or_path
    out_dir = resolve_out_dir(in_path, in_dir, args.out_dir) if args.out_dir else in_path.parent
    out_path = out_dir / (in_path.stem + args.out_format)
    if args.check:
        check(out_path, args)
        if not args.fix:
            return

    if out_path.exists() and not args.overwrite:
        logger.debug(f"File {out_path} already exists. Skipping.")
        return

    try:
        restart = time()
        mesh = load(in_path,
                    loader="trimesh" if args.use_trimesh else "pymeshlab",
                    return_type="trimesh" if args.use_trimesh else "dict")
        vertices = get_vertices_and_faces(mesh)[0]
        logger.debug(f"Loaded mesh ({len(vertices)} vertices) in {time() - restart:.2f}s.")

        translation = np.zeros(3)
        scale = 1.0
        if not args.no_normalization:
            restart = time()
            mesh, translation, scale = normalize(mesh, padding=args.padding)
            logger.debug(f"Normalized mesh in {time() - restart:.2f}s.")

        if args.mode == "fill":
            restart = time()
            mesh = kaolin_pipeline(mesh,
                                   resolution=args.resolution,
                                   save_voxel_path=out_path.parent / "voxel.npz",
                                   try_cpu=args.try_cpu)
            logger.debug(f"Ran Kaolin pipeline in {time() - restart:.2f}s.")
        elif args.mode == "fuse":
            restart = time()
            mesh = pyfusion_pipeline(mesh, args)
            logger.debug(f"Ran PyFusion pipeline in {time() - restart:.2f}s.")
        elif args.mode == "carve":
            restart = time()
            mesh = voxel_carving_pipeline(mesh)
            logger.debug(f"Ran voxel carving pipeline in {time() - restart:.2f}s.")

        vertices, faces = get_vertices_and_faces(mesh)
        if len(vertices) == 0 or len(faces) == 0:
            logger.warning(f"Extracted mesh is empty. Skipping.")
            return
        logger.debug(f"Extracted mesh ({len(vertices)} vertices, {len(faces)} faces) in {time() - restart:.2f}s.")

        if args.script_dir:
            restart = time()

            script_dir = args.script_dir
            if not script_dir.is_dir():
                script_dir = Path(__file__).parent / script_dir
            assert script_dir.is_dir(), f"Script dir {script_dir} is not a directory."
            scripts = load_scripts(script_dir, num_vertices=len(vertices))
            mesh = process(mesh, scripts)

            vertices, faces = get_vertices_and_faces(mesh)
            if len(vertices) == 0 or len(faces) == 0:
                logger.warning(f"Filtered mesh is empty. Skipping.")
                return
            logger.debug(f"Filtered mesh ({len(vertices)} vertices, {len(faces)} faces) in {time() - restart:.2f}s.")

        if not args.no_normalization:
            restart = time()
            mesh, _, _ = normalize(mesh, translation=-translation * 1 / scale, scale=1 / scale)
            logger.debug(f"Normalized mesh in {time() - restart:.2f}s.")

        if args.check_watertight:
            restart = time()
            vertices, faces = get_vertices_and_faces(mesh)
            _mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if not _mesh.is_watertight:
                logger.warning(f"Mesh {in_path} is not watertight. Skipping.")
                return
            logger.debug(f"Checked watertightness in {time() - restart:.2f}s.")

        if args.show:
            try:
                import open3d as o3d
                vertices, faces = get_vertices_and_faces(mesh)
                _mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                                  o3d.utility.Vector3iVector(faces))
                _mesh.compute_vertex_normals()
                o3d.visualization.draw_geometries([_mesh])
            except ImportError:
                logger.warning("Could not import Open3D, skipping visualization.")

        restart = time()
        save(mesh, out_path, args.precision)
        logger.debug(f"Saved mesh in {time() - restart:.2f}s.")

        del mesh
        gc.collect()
    except Exception as e:
        logger.exception(e)
        if args.remove:
            logger.warning(f"Exception occurred. Removing {out_dir}.")
            shutil.rmtree(out_dir, ignore_errors=True)
    logger.debug(f"Runtime: {time() - start:.2f}s.\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("in_dir_or_path", type=Path, help="Path to input data.")
    parser.add_argument("--out_dir", type=Path, help="Path to output directory.")
    parser.add_argument("--in_format", type=str, default=".obj", help="Input file format.")
    parser.add_argument("--out_format", type=str, default=".off", choices=[".obj", ".off", ".ply", ".stl"],
                        help="Output file format.")
    parser.add_argument("--recursion_depth", type=int, help="Depth of recursive glob pattern matching.")
    parser.add_argument("--script_dir", type=Path, default="assets/meshlab_filter_scripts",
                        help="Name of or (absolute or relative) path to directory containing MeshLab scripts.")
    parser.add_argument("--width", type=int, default=640, help="Width of the depth map.")
    parser.add_argument("--height", type=int, default=640, help="Height of the depth map.")
    parser.add_argument("--fx", type=float, default=640, help="Focal length in x.")
    parser.add_argument("--fy", type=float, default=640, help="Focal length in y.")
    parser.add_argument("--cx", type=float, default=320, help="Principal point in x.")
    parser.add_argument("--cy", type=float, default=320, help="Principal point in y.")
    parser.add_argument("--znear", type=float, default=0.25, help="Near clipping plane.")
    parser.add_argument("--zfar", type=float, default=1.75, help="Far clipping plane.")
    parser.add_argument("--padding", type=float, default=0.1, help="Relative padding applied on each side.")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of the TSDF fusion voxel grid.")
    parser.add_argument("--depth_offset", type=float, default=1.5,
                        help="Thicken object through offsetting of rendered depth maps.")
    parser.add_argument("--no_erosion", action="store_true",
                        help="Do not erode rendered depth maps to thicken thin structures.")
    parser.add_argument("--no_normalization", action="store_true", help="Do not normalize the mesh.")
    parser.add_argument("--n_jobs", type=int, default=cpu_count(), help="Number of parallel jobs.")
    parser.add_argument("--n_views", type=int, default=100, help="Number of views to render.")
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32, 64], help="Data precision.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--flip_faces", action="store_true", help="Flip faces (i.e. invert normals) of the mesh.")
    parser.add_argument("--use_trimesh", action="store_true", help="Use trimesh for loading and saving meshes.")
    parser.add_argument("--mode", type=str, default="fuse", choices=["fuse", "carve", "fill", "script"],
                        help="Apply TSDF fusion, voxel carving or Kaolin hole filling to the meshes."
                             "Use 'script' to only apply MeshLab filter scripts from the provided 'script_dir'.")
    parser.add_argument("--try_cpu", action="store_true", help="Fallback to CPU if GPU fails.")
    parser.add_argument("--sort", action="store_true", help="Sort files before processing.")
    parser.add_argument("--check", action="store_true", help="Check results.")
    parser.add_argument("--check_watertight", action="store_true", help="Verify that generated mesh is watertight.")
    parser.add_argument("--fix", action="store_true", help="Fix results that failed check.")
    parser.add_argument('--remove', action='store_true', help='Removes results that failed check.')
    parser.add_argument('--show', action='store_true', help='Visualize renders.')
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--debug", action="store_true", help="Enable debugging.")
    args = parser.parse_args()

    assert args.mode in MODES, f"Mode '{args.mode}' not in available modes: {MODES}."

    save_command_and_args_to_file(args.out_dir if args.out_dir else Path.cwd() / "command.txt", args)

    if args.use_trimesh:
        logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Disable multithreading if multiprocessing is used.
    if args.n_jobs > 1:
        disable_multithreading()

    # Check that fix or remove are only set when check is set.
    if args.fix or args.remove:
        assert args.check, "Fix or remove can only be set when check is set."

    # Check that fix and remove are not set at the same time.
    assert not (args.fix and args.remove), "Fix and remove cannot be set at the same time."

    if args.verbose:
        set_log_level(logging.DEBUG)

    if args.debug:
        tracemalloc.start()

    # Set EGL for offscreen rendering if not visualizing.
    if not args.show:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

    files = eval_input(args.in_dir_or_path, args.in_format, args.recursion_depth, args.sort)
    start = time()
    desc = "Checking" if args.check else "Processing"
    if args.fix:
        desc += " & Fixing"
    if args.remove:
        desc += " & Removing"
    logger.debug(desc)
    with tqdm_joblib(tqdm(desc=desc, total=len(files), disable=args.verbose)):
        Parallel(n_jobs=1 if args.verbose else min(args.n_jobs, len(files)),
                 verbose=args.verbose)(delayed(run)(file, args) for file in files)
    logger.debug(f"Total runtime: {time() - start:.2f}s.")


if __name__ == "__main__":
    main()
