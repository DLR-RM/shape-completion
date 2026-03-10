import gc
import linecache
import logging
import os
import shutil
import time
import tracemalloc
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm
from trimesh import Trimesh

from utils import (
    disable_multithreading,
    eval_input,
    load_mesh,
    log_optional_dependency_summary,
    resolve_out_dir,
    save_command_and_args_to_file,
    save_mesh,
    set_log_level,
    setup_logger,
    suppress_known_optional_dependency_warnings,
    tqdm_joblib,
)

from ..src.utils import apply_meshlab_filters, get_vertices_and_faces, load_scripts

logger = setup_logger(__name__)

MODES = ["fuse", "carve", "fill"]
try:
    from process.src.fuse import pyfusion_pipeline
except ImportError as e:
    logger.warning(f"Could not import PyFusion, 'fuse' mode disabled: {e}")
    MODES.remove("fuse")
try:
    from process.src.carve import voxel_carving_pipeline
except ImportError as e:
    logger.warning(f"Could not import Open3D, 'carve' mode disabled: {e}")
    MODES.remove("carve")
try:
    from process.src.fill import kaolin_pipeline
except ImportError as e:
    logger.warning(f"Could not import PyTorch and/or NVIDIA Kaolin, 'fill' mode disabled: {e}.")
    MODES.remove("fill")
assert len(MODES) > 0, "No modes available, exiting."
logger.debug(f"Enabled modes: {MODES}")


def load(in_path: Path, loader: str = "pymeshlab", return_type: str = "dict") -> Trimesh | dict[str, np.ndarray]:
    try:
        vertices, faces = load_mesh(in_path, load_with=loader)
    except Exception as exc:
        if not (isinstance(exc, ValueError) or exc.__class__.__name__ == "PyMeshLabException"):
            raise
        logger.warning(f"Could not load mesh {in_path} with {loader}. Trying alternatives.")
        vertices, faces = load_mesh(in_path)

    if faces is None:
        raise ValueError(f"Could not load faces from {in_path}.")

    if return_type == "dict":
        return {"vertices": vertices, "faces": faces}
    elif return_type == "trimesh":
        return Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    else:
        raise ValueError(f"Unknown return type '{return_type}'.")


def _normalize_scale(scale: float | tuple[float, float, float] | np.ndarray) -> float | np.ndarray:
    if isinstance(scale, tuple):
        return np.asarray(scale, dtype=np.float64)
    if isinstance(scale, np.ndarray):
        return np.asarray(scale, dtype=np.float64)
    return float(scale)


def normalize(
    mesh: Trimesh | dict[str, np.ndarray],
    translation: tuple[float, float, float] | np.ndarray | None = None,
    scale: float | tuple[float, float, float] | np.ndarray | None = None,
    padding: float = 0.1,
) -> tuple[Trimesh | dict[str, np.ndarray], np.ndarray, float | np.ndarray]:
    translation_arr: np.ndarray
    scale_value: float | np.ndarray

    if isinstance(mesh, Trimesh):
        if translation is None:
            translation_arr = -mesh.bounds.mean(axis=0)
        else:
            translation_arr = np.asarray(translation, dtype=np.float64)

        if scale is None:
            max_extents = mesh.extents.max()
            scale_value = float(1 / (max_extents + 2 * padding * max_extents))
        else:
            scale_value = _normalize_scale(scale)

        mesh.apply_translation(translation_arr)
        mesh.apply_scale(scale_value)
    elif isinstance(mesh, dict):
        if translation is None or scale is None:
            vertices = mesh["vertices"]
            faces = mesh["faces"]
            referenced = np.zeros(len(vertices), dtype=bool)
            referenced[faces] = True
            in_mesh = vertices[referenced]
            bounds = np.array([in_mesh.min(axis=0), in_mesh.max(axis=0)])
            if translation is None:
                translation_arr = -bounds.mean(axis=0)
            else:
                translation_arr = np.asarray(translation, dtype=np.float64)
            if scale is None:
                extents = np.ptp(bounds, axis=0)
                max_extents = extents.max()
                scale_value = float(1 / (max_extents + 2 * padding * max_extents))
            else:
                scale_value = _normalize_scale(scale)
        else:
            translation_arr = np.asarray(translation, dtype=np.float64)
            scale_value = _normalize_scale(cast(float | tuple[float, float, float] | np.ndarray, scale))

        mesh["vertices"] += translation_arr
        mesh["vertices"] *= scale_value
    else:
        raise ValueError(f"Unknown mesh type '{type(mesh)}'.")

    return mesh, translation_arr, scale_value


def process(mesh: Trimesh | dict[str, np.ndarray], script_paths: list[Path]) -> Trimesh | dict[str, np.ndarray]:
    vertices, faces = get_vertices_and_faces(mesh)
    vertices, faces = apply_meshlab_filters(vertices, faces, script_paths)

    if isinstance(mesh, Trimesh):
        return Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    elif isinstance(mesh, dict):
        return {"vertices": vertices, "faces": faces}
    raise ValueError(f"Unknown mesh type '{type(mesh)}'.")


def _get_vertices_and_faces_for_save(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, (Trimesh, dict, list, tuple)):
        return get_vertices_and_faces(mesh)

    mesh_obj = mesh
    if hasattr(mesh_obj, "current_mesh"):
        mesh_obj = mesh_obj.current_mesh()

    if hasattr(mesh_obj, "vertex_matrix") and hasattr(mesh_obj, "face_matrix"):
        vertices = np.asarray(mesh_obj.vertex_matrix())
        faces = np.asarray(mesh_obj.face_matrix())
        return vertices.astype(np.float32), faces.astype(np.int64)

    raise ValueError(f"Unknown mesh type '{type(mesh)}'.")


def save(mesh: Trimesh | dict[str, np.ndarray] | Any, path: Path, precision: int = 32):
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces = _get_vertices_and_faces_for_save(mesh)
    if isinstance(mesh, Trimesh):
        if precision == 16:
            digits = 3
        elif precision == 32:
            digits = 6
        elif precision == 64:
            digits = 15
        else:
            raise ValueError(f"Invalid precision: {precision}.")
        save_mesh(path, vertices, faces, save_with="trimesh", digits=digits)
    else:
        save_mesh(path, vertices, faces, save_with="pymeshlab")


def display_top(snapshot, key_type="lineno", limit=10):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print(f"Top {limit} lines")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(f"#{index}: {frame.filename}:{frame.lineno}: {stat.size / 1024:.1f} KiB")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"    {line}")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print(f"{len(other)} other: {size / 1024:.1f} KiB")
    total = sum(stat.size for stat in top_stats)
    print(f"Total allocated size: {total / 1024:.1f} KiB")


def trace_run(in_path: Path, args: Any):
    snapshot1 = tracemalloc.take_snapshot()

    run(in_path, args)

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, "lineno")

    print("[ Top 10 differences ]")
    for stat in top_stats[:10]:
        print(stat)

    display_top(snapshot2)


def check(out_path: Path, args: Any):
    try:
        mesh = load(
            out_path,
            loader="trimesh" if args.use_trimesh else "pymeshlab",
            return_type="trimesh" if args.use_trimesh else "dict",
        )
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
    start = time.perf_counter()
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
        restart = time.perf_counter()
        mesh = load(
            in_path,
            loader="trimesh" if args.use_trimesh else "pymeshlab",
            return_type="trimesh" if args.use_trimesh else "dict",
        )
        vertices = get_vertices_and_faces(mesh)[0]
        logger.debug(f"Loaded mesh ({len(vertices)} vertices) in {time.perf_counter() - restart:.2f}s.")

        translation = np.zeros(3)
        scale = 1.0
        if not args.no_normalization:
            restart = time.perf_counter()
            mesh, translation, scale = normalize(mesh, padding=args.padding)
            logger.debug(f"Normalized mesh in {time.perf_counter() - restart:.2f}s.")

        if args.mode == "fill":
            restart = time.perf_counter()
            mesh = kaolin_pipeline(
                mesh, resolution=args.resolution, save_voxel_path=out_path.parent / "voxel.npz", try_cpu=args.try_cpu
            )
            logger.debug(f"Ran Kaolin pipeline in {time.perf_counter() - restart:.2f}s.")
        elif args.mode == "fuse":
            restart = time.perf_counter()
            mesh = pyfusion_pipeline(mesh, args)
            logger.debug(f"Ran PyFusion pipeline in {time.perf_counter() - restart:.2f}s.")
        elif args.mode == "carve":
            restart = time.perf_counter()
            mesh = voxel_carving_pipeline(mesh)
            logger.debug(f"Ran voxel carving pipeline in {time.perf_counter() - restart:.2f}s.")

        vertices, faces = get_vertices_and_faces(mesh)
        if len(vertices) == 0 or len(faces) == 0:
            logger.warning("Extracted mesh is empty. Skipping.")
            return
        logger.debug(
            f"Extracted mesh ({len(vertices)} vertices, {len(faces)} faces) in {time.perf_counter() - restart:.2f}s."
        )

        if args.script_dir:
            restart = time.perf_counter()

            script_dir = args.script_dir
            if not script_dir.is_dir():
                script_dir = Path(__file__).parent.parent / script_dir
            assert script_dir.is_dir(), f"Script dir {script_dir} is not a directory."
            scripts = load_scripts(script_dir, num_vertices=len(vertices))
            mesh = process(mesh, scripts)

            vertices, faces = get_vertices_and_faces(mesh)
            if len(vertices) == 0 or len(faces) == 0:
                logger.warning("Filtered mesh is empty. Skipping.")
                return
            logger.debug(
                f"Filtered mesh ({len(vertices)} vertices, {len(faces)} faces) in {time.perf_counter() - restart:.2f}s."
            )

        if not args.no_normalization:
            restart = time.perf_counter()
            mesh, _, _ = normalize(mesh, translation=-translation * 1 / scale, scale=1 / scale)
            logger.debug(f"Normalized mesh in {time.perf_counter() - restart:.2f}s.")

        if args.check_watertight:
            restart = time.perf_counter()
            vertices, faces = get_vertices_and_faces(mesh)
            _mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if not _mesh.is_watertight:
                logger.warning(f"Mesh {in_path} is not watertight. Skipping.")
                return
            logger.debug(f"Checked watertightness in {time.perf_counter() - restart:.2f}s.")

        if args.show:
            try:
                import open3d as o3d

                vertices, faces = get_vertices_and_faces(mesh)
                _mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces)
                )
                _mesh.compute_vertex_normals()
                o3d_visualization = cast(Any, o3d).visualization
                o3d_visualization.draw_geometries([_mesh])
            except ImportError:
                logger.warning("Could not import Open3D, skipping visualization.")

        restart = time.perf_counter()
        save(mesh, out_path, args.precision)
        logger.debug(f"Saved mesh in {time.perf_counter() - restart:.2f}s.")

        del mesh
        gc.collect()
    except Exception as e:
        logger.exception(e)
        if args.remove:
            logger.warning(f"Exception occurred. Removing {out_dir}.")
            shutil.rmtree(out_dir, ignore_errors=True)
    logger.debug(f"Runtime: {time.perf_counter() - start:.2f}s.\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("in_dir_or_path", type=Path, help="Path to input data.")
    parser.add_argument("--out_dir", type=Path, help="Path to output directory.")
    parser.add_argument("--in_format", type=str, default=".obj", help="Input file format.")
    parser.add_argument(
        "--out_format", type=str, default=".off", choices=[".obj", ".off", ".ply", ".stl"], help="Output file format."
    )
    parser.add_argument("--recursion_depth", type=int, help="Depth of recursive glob pattern matching.")
    parser.add_argument(
        "--script_dir",
        type=Path,
        default="assets/meshlab_filter_scripts",
        help="Name of or (absolute or relative) path to directory containing MeshLab scripts.",
    )
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
    parser.add_argument(
        "--depth_offset", type=float, default=1.5, help="Thicken object through offsetting of rendered depth maps."
    )
    parser.add_argument(
        "--no_erosion", action="store_true", help="Do not erode rendered depth maps to thicken thin structures."
    )
    parser.add_argument("--no_normalization", action="store_true", help="Do not normalize the mesh.")
    parser.add_argument("--n_jobs", type=int, default=cpu_count(), help="Number of parallel jobs.")
    parser.add_argument("--n_views", type=int, default=100, help="Number of views to render.")
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32, 64], help="Data precision.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--flip_faces", action="store_true", help="Flip faces (i.e. invert normals) of the mesh.")
    parser.add_argument("--use_trimesh", action="store_true", help="Use trimesh for loading and saving meshes.")
    parser.add_argument(
        "--mode",
        type=str,
        default="fuse",
        choices=["fuse", "carve", "fill", "script"],
        help="Apply TSDF fusion, voxel carving or Kaolin hole filling to the meshes."
        "Use 'script' to only apply MeshLab filter scripts from the provided 'script_dir'.",
    )
    parser.add_argument("--try_cpu", action="store_true", help="Fallback to CPU if GPU fails.")
    parser.add_argument("--sort", action="store_true", help="Sort files before processing.")
    parser.add_argument("--check", action="store_true", help="Check results.")
    parser.add_argument("--check_watertight", action="store_true", help="Verify that generated mesh is watertight.")
    parser.add_argument("--fix", action="store_true", help="Fix results that failed check.")
    parser.add_argument("--remove", action="store_true", help="Removes results that failed check.")
    parser.add_argument("--show", action="store_true", help="Visualize renders.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--debug", action="store_true", help="Enable debugging.")
    args = parser.parse_args()

    suppress_known_optional_dependency_warnings()
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
    log_optional_dependency_summary(logger)

    if args.debug:
        tracemalloc.start()

    # Set EGL for offscreen rendering if not visualizing.
    if not args.show:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    files = eval_input(args.in_dir_or_path, args.in_format, args.recursion_depth, args.sort)
    start = time.perf_counter()
    desc = "Checking" if args.check else "Processing"
    if args.fix:
        desc += " & Fixing"
    if args.remove:
        desc += " & Removing"
    logger.debug(desc)
    with tqdm_joblib(tqdm(desc=desc, total=len(files), disable=args.verbose)):
        Parallel(n_jobs=1 if args.verbose else min(args.n_jobs, len(files)), verbose=args.verbose)(
            delayed(run)(file, args) for file in files
        )
    logger.debug(f"Total runtime: {time.perf_counter() - start:.2f}s.")


if __name__ == "__main__":
    main()
