import logging
import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import time
from typing import Any, Tuple

import numpy as np
import open3d as o3d
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from trimesh import Trimesh

from dataset import Visualize
from libs import check_mesh_contains
from utils import (tqdm_joblib, generate_random_basis, setup_logger, set_log_level, load_mesh, save_mesh, eval_input,
                   disable_multithreading, resolve_out_dir, save_command_and_args_to_file)
from .src.utils import resolve_dtype, normalize_mesh

logger = setup_logger(__name__)


def normalize(in_path: Path, out_dir: Path, args: Any):
    start = time()
    out_format = in_path.suffix
    if args.out_format is not None:
        out_format = args.out_format
    out_path = (out_dir / in_path.name).with_suffix(out_format)
    if out_path.exists() and not args.overwrite:
        logger.debug(f"File {out_path} already processed. Skipping.")
        return
    restart = time()
    mesh = Trimesh(*load_mesh(in_path,
                              force='mesh',
                              process=args.process,
                              validate=args.process,
                              enable_post_processing=args.process))
    logger.debug(f"Loading mesh took {time() - restart:.2f}s.")
    restart = time()
    mesh = normalize_mesh(mesh, args.center, args.scale)
    logger.debug(f"Normalizing mesh took {time() - restart:.2f}s.")
    restart = time()
    save_mesh(out_path, mesh.vertices, mesh.faces)
    logger.debug(f"Saving mesh took {time() - restart:.2f}s.")
    logger.debug(f"Total time: {time() - start:.2f}s.")


def sample_pointcloud(mesh: Trimesh, args: Any) -> Tuple[np.ndarray, np.ndarray]:
    points, face_idx = mesh.sample(args.num_points, return_index=True)
    normals = mesh.face_normals[face_idx]

    dtype = resolve_dtype(args.precision)
    points = points.astype(dtype)
    normals = normals.astype(dtype)

    return points, normals


def sample_points(mesh: Trimesh, args: Any) -> Tuple[np.ndarray, np.ndarray]:
    boxsize = 1 + args.padding
    points_uniform = np.random.rand(args.num_points, 3)
    points_uniform = boxsize * (points_uniform - 0.5)

    samples = list()
    sigmas = [0.001, 0.0015, 0.0025, 0.01, 0.015, 0.025, 0.05, 0.1, 0.15, 0.25]
    num_points = args.num_points // len(sigmas)
    for sigma in sigmas:
        noise = sigma * np.random.standard_normal((num_points, 3))
        samples.append(mesh.sample(num_points) + noise)
    points_surface = np.concatenate(samples, axis=0)

    points_sphere_1 = generate_random_basis(args.num_points, radius=1, seed=None)
    points_sphere_2 = generate_random_basis(args.num_points, radius=2, seed=None)
    points_sphere_5 = generate_random_basis(args.num_points, radius=5, seed=None)
    points_sphere_10 = generate_random_basis(args.num_points, radius=10, seed=None)
    points_sphere_20 = generate_random_basis(args.num_points, radius=20, seed=None)
    points_sphere_50 = generate_random_basis(args.num_points, radius=50, seed=None)

    points = np.concatenate([points_uniform,
                             points_surface,
                             points_sphere_1,
                             points_sphere_2,
                             points_sphere_5,
                             points_sphere_10,
                             points_sphere_20,
                             points_sphere_50], axis=0)

    occupancy = check_mesh_contains(mesh, points)

    dtype = resolve_dtype(args.precision)
    points = points.astype(dtype)

    return points, occupancy


def sample(in_path: Path, out_dir: Path, args: Any):
    out_dir /= "samples"
    pointcloud_path = out_dir / "surface.npz"
    uniform_points_path = out_dir / "uniform_random.npz"
    surface_points_path = out_dir / "surface_random.npz"

    if args.check:
        try:
            for path in [pointcloud_path, uniform_points_path, surface_points_path]:
                data = np.load(str(path))
                for key in data.keys():
                    value = data[key]
            for index in range(6):
                points_path = out_dir / f"uniform_sphere_{index}.npz"
                data = np.load(str(points_path))
                for key in data.keys():
                    value = data[key]
        except Exception as e:
            logger.error(f"Error: Files for {in_path} are corrupted.")
            logger.exception(e)
            if args.fix:
                try:
                    os.remove(pointcloud_path)
                    os.remove(uniform_points_path)
                    os.remove(surface_points_path)
                    shutil.rmtree(out_dir, ignore_errors=True)
                except Exception as e:
                    logger.exception(e)
                    pass

    if pointcloud_path.exists() and uniform_points_path.exists() and surface_points_path.exists() and not args.overwrite:
        logger.debug(f"Files for {in_path} already processed. Skipping.")
        return

    restart = time()
    mesh = Trimesh(*load_mesh(in_path,
                              force='mesh',
                              process=args.process,
                              validate=args.process,
                              enable_post_processing=args.process))
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        logger.error(f"Error: Mesh {in_path} is empty.")
        return
    logger.debug(f"Loaded mesh in {time() - restart:.4f}s.")

    if not mesh.is_watertight:
        if args.process:
            restart = time()
            mesh = o3d.io.read_triangle_mesh(str(in_path), enable_post_processing=True)
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.remove_non_manifold_edges()
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()
            mesh = mesh.select_by_index(np.arange(len(np.asarray(mesh.vertices))), cleanup=True)
            mesh = Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
            logger.debug(f"Processed mesh in {time() - restart:.4f}s.")
        if not mesh.is_watertight:
            if args.accept_non_watertight:
                logger.warning(f"Warning: Mesh {in_path} is not watertight.")
            else:
                logger.error(f"Error: Mesh {in_path} is not watertight.")
                return

    out_dir.mkdir(parents=True, exist_ok=True)

    restart = time()
    mesh = normalize_mesh(mesh)
    logger.debug(f"Normalized mesh in {time() - restart:.4f}s.")

    pcd, normals = None, None
    if not pointcloud_path.exists() or args.overwrite:
        restart = time()
        pcd, normals = sample_pointcloud(mesh, args)
        np.savez_compressed(pointcloud_path, points=pcd, normals=normals)
        logger.debug(f"Sampled pointcloud in {time() - restart:.4f}s.")

    points, occupancy = None, None
    if not uniform_points_path.exists() or not surface_points_path.exists() or args.overwrite:
        restart = time()
        points, occupancy = sample_points(mesh, args)
        for index, step in enumerate(range(0, len(points), args.num_points)):
            if index == 0:
                points_path = out_dir / "uniform_random.npz"
            elif index == 1:
                points_path = out_dir / "surface_random.npz"
            else:
                points_path = out_dir / f"uniform_sphere_{index - 2}.npz"
            np.savez_compressed(points_path,
                                points=points[step:step + args.num_points],
                                occupancy=np.packbits(occupancy[step:step + args.num_points]))
        logger.debug(f"Sampled points in {time() - restart:.4f}s.")

    if args.show and not any([pcd is None, normals is None, points is None, occupancy is None]):
        vis = Visualize(show_inputs=False,
                        show_mesh=True,
                        show_pointcloud=True)
        vis({"obj_name": in_path.stem,
             "inputs.path": str(in_path),
             "mesh": mesh,
             "pointcloud": pcd,
             "normals": normals,
             "points": points,
             "points.occ": occupancy})


def run(in_path: Path, args: Namespace):
    start = time()
    logger.debug(f"Processing file {in_path}.")
    in_dir = args.in_dir_or_path.parent if args.in_dir_or_path.is_file() else args.in_dir_or_path
    out_dir = resolve_out_dir(in_path, in_dir, args.out_dir) if args.out_dir else in_path.parent
    logger.debug(f"Output directory is {out_dir}.")
    try:
        if args.task == "sample":
            sample(in_path, out_dir, args)
        elif args.task == "normalize":
            normalize(in_path, out_dir, args)
    except Exception as e:
        logger.exception(e)
        if args.remove:
            logger.warning(f"Exception occurred. Removing {out_dir}.")
            shutil.rmtree(out_dir, ignore_errors=True)
    logger.debug(f"Runtime: {time() - start:.2f}s.\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("in_dir_or_path", type=Path, help="Path to input data.")
    parser.add_argument("task", type=str, choices=["sample", "normalize"], help="Task to perform.")
    parser.add_argument("--out_dir", type=Path, help="Path to output directory.")
    parser.add_argument("--in_format", type=str, default=".off",
                        help="Input file format.")
    parser.add_argument("--out_format", type=str, choices=[".off", ".obj", ".ply"],
                        help="Output file format.")
    parser.add_argument("--recursion_depth", type=int, help="Depth of recursive glob pattern matching.")
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32, 64], help="Data precision.")
    parser.add_argument("--padding", type=float, default=0.1, help="Padding applied during sampling.")
    parser.add_argument("--num_points", type=int, default=int(1e5), help="Number of points to sample.")
    parser.add_argument("--center", action="store_true", help="Center mesh(es).")
    parser.add_argument("--scale", action="store_true", help="Scale mesh(es).")
    parser.add_argument("--process", action="store_true", help="Process meshes before sampling/rendering.")
    parser.add_argument("--accept_non_watertight", action="store_true", help="Accept non-watertight meshes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--check", action="store_true", help="Check results.")
    parser.add_argument("--fix", action="store_true", help="Fix results that failed check.")
    parser.add_argument("--remove", action="store_true", help="Removes results that failed check.")
    parser.add_argument("--sort", action="store_true", help="Sort files before processing.")
    parser.add_argument("--n_jobs", type=int, default=cpu_count(), help="Number of parallel jobs.")
    parser.add_argument("--show", action="store_true", help="Visualize renders.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    save_command_and_args_to_file(args.out_dir if args.out_dir else Path.cwd() / "command.txt", args)

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
