import argparse
import copy
import logging
import os
import shutil
import time
from glob import glob
from joblib import cpu_count
from pathlib import Path
from typing import Any, List, Tuple, Union

import mcubes
import numpy as np
import open3d as o3d
import trimesh
from joblib import Parallel, delayed
from matplotlib.cm import get_cmap
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R
from skimage.measure import marching_cubes
from tqdm import tqdm

from utils import (generate_random_basis, resolve_path, tqdm_joblib, setup_logger, load_mesh,
                   save_command_and_args_to_file)
from libs import compute_distance_field, compute_marching_cubes


rng = np.random.default_rng()
logger = setup_logger(__name__)


def normalize_mesh(input_path: str, output_path: str, args: Any) -> Tuple[float, float]:
    path = input_path
    """
    if args.shapenet and input_path.endswith(".obj"):
        path = input_path.replace(".obj", ".off")
        command = f"meshlabserver -i {input_path} -o {path}"
        subprocess.run(command.split(' '), stdout=None if args.verbose else subprocess.DEVNULL)
        # os.system(command + " &> /dev/null")
    # mesh = trimesh.load(path, process=False)
    #
    # total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    # scale = total_size / (1 - args.padding)
    # centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=not args.dont_fix_mesh)
    """
    vertices, faces = load_mesh(path, load_with="pymeshlab")
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))

    if "ShapeNet.v1" in input_path and "03797390" in input_path:
        ids = ["3c0467f96e26b8c6a93445a1757adf6",
               "1a1c0a8d4bad82169f0594e65f756cf5",
               "1f035aa5fc6da0983ecac81e09b15ea9",
               "68f4428c0b38ae0e2469963e6d044dfe",
               "f1866a48c2fc17f85b2ecd212557fda0"]
        if any(_id in input_path for _id in ids):
            mesh.rotate(R.from_euler('x', -180, degrees=True).as_matrix(), center=(0, 0, 0))

    if not args.dont_fix_mesh:
        indices, length, area = mesh.cluster_connected_triangles()
        mesh.remove_triangles_by_index(np.argwhere(indices != np.argmax(length)))
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh = mesh.select_by_index(np.arange(len(np.asarray(mesh.vertices))), cleanup=True)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    loc = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    scale = (mesh.get_max_bound() - mesh.get_min_bound()).max() / (1 - args.padding)

    np.savez(output_path.replace(output_path.split('/')[-1], "transform"), loc=loc, scale=scale)

    # mesh.apply_translation(-centers)
    # mesh.apply_scale(1 / scale)

    mesh.translate(-loc, relative=True)
    mesh.scale(1 / scale, center=(0, 0, 0))
    # mesh.scale(2.8, center=(0, 0, 0))
    args.e /= (mesh.get_max_bound() - mesh.get_min_bound()).max()

    # pymesh.save_mesh_raw(output_path, mesh.vertices, mesh.faces)
    # mesh.export(output_path, file_type="obj")
    # o3d.io.write_triangle_mesh(output_path,
    #                            mesh,
    #                            write_vertex_colors=False,
    #                            write_triangle_uvs=False)
    trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                    faces=np.asarray(mesh.triangles),
                    vertex_normals=np.asarray(mesh.vertex_normals),
                    face_normals=np.asarray(mesh.triangle_normals)).export(output_path)
    # if not input_path.endswith(".off"):
    #     os.remove(path)
    return loc, scale


def mesh_from_sdf(sdf_path: str,
                  args: Any = None,
                  **kwargs: Any) -> bool:
    output_path = kwargs.get('o')
    if kwargs.get("method") == "vega":
        return compute_marching_cubes(sdf_path, output_path, **kwargs)
    elif kwargs.get("method") == "skimage":
        sdf_dict = load_sdf(sdf_path, args.res)
        volume = sdf_dict["values"].copy().transpose((2, 1, 0))

        level = kwargs.get('i')
        level = 0.0 if level is None else level
        # box_size = (1 - args.padding) * args.e
        box_size = 1.1
        voxel_size = box_size / (np.array(volume.shape) - 1)

        vertices, faces, normals, _ = marching_cubes(volume=volume,
                                                     level=level,
                                                     spacing=voxel_size,
                                                     step_size=1,
                                                     allow_degenerate=False,
                                                     method="lewiner")  # lorensen or lewiner

        mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals if kwargs.get("normals") else None)
        offsets = np.repeat(0.5 * box_size, 3)
        mesh.apply_translation(-offsets)
        mesh.export(output_path)

        # mesh.apply_scale(kwargs.get("scale"))
        # mesh.apply_translation(kwargs.get("loc"))
        # mesh.export(output_path.split('.')[0] + "_orig.ply")

        return mesh.is_watertight
    elif kwargs.get("method") == "mcubes":
        sdf_dict = load_sdf(sdf_path, args.res)
        volume = sdf_dict["values"].copy().transpose((2, 1, 0))

        level = kwargs.get('i')
        level = 0.0 if level is None else level
        box_size = 1.1

        volume_padded = np.pad(volume, 1, "constant", constant_values=1e6)
        vertices, triangles = mcubes.marching_cubes(-volume_padded, level)
        vertices -= 1

        vertices /= (np.array(volume.shape) - 1)
        vertices = box_size * (vertices - 0.5)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(output_path)

        return mesh.is_watertight
    else:
        raise ValueError


def uniform_grid_sampling(grid: np.ndarray,
                          bounds: Union[np.ndarray, List, Tuple],
                          num_points: int,
                          mask: Union[np.ndarray, List, Tuple] = None) -> np.ndarray:
    assert len(grid.shape) == 3, f"{grid.shape}"
    assert len(bounds) == 6, f"{bounds}"

    res_x, res_y, res_z = grid.shape
    x = np.linspace(bounds[0], bounds[3], num=res_x)
    y = np.linspace(bounds[1], bounds[4], num=res_y)
    z = np.linspace(bounds[2], bounds[5], num=res_z)

    if mask is None:
        choice = rng.integers(grid.size, size=num_points)
        x_ind = choice % res_x
        y_ind = (choice // res_y) % res_y
        z_ind = choice // res_z ** 2
        x_vals = x[x_ind]
        y_vals = y[y_ind]
        z_vals = z[z_ind]
        vals = grid.flatten()[choice]
    else:
        choice = rng.choice(np.argwhere(mask), size=num_points)
        x_vals = x[choice[:, 2]]
        y_vals = y[choice[:, 1]]
        z_vals = z[choice[:, 0]]
        vals = grid[choice[:, 0], choice[:, 1], choice[:, 2]]
    return np.vstack((x_vals, y_vals, z_vals, vals)).T


def uniform_random_sampling(grid: np.ndarray,
                            bounds: Union[np.ndarray, List, Tuple],
                            num_points: int = 0,
                            points: np.ndarray = None,
                            ignore_bounds: bool = False) -> np.ndarray:
    assert len(grid.shape) == 3
    assert len(bounds) == 6
    assert num_points > 0 or points is not None

    res_x, res_y, res_z = grid.shape
    x = np.linspace(bounds[0], bounds[3], num=res_x)
    y = np.linspace(bounds[1], bounds[4], num=res_y)
    z = np.linspace(bounds[2], bounds[5], num=res_z)

    interpolator = RegularGridInterpolator(points=(z, y, x),
                                           values=grid,
                                           bounds_error=not ignore_bounds,
                                           fill_value=grid.max() if ignore_bounds else np.nan)

    if points is None:
        x = (bounds[3] - bounds[0]) * rng.random(num_points) + bounds[0]
        y = (bounds[4] - bounds[1]) * rng.random(num_points) + bounds[1]
        z = (bounds[5] - bounds[2]) * rng.random(num_points) + bounds[2]
    else:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        if not ignore_bounds:
            x = np.clip(x, bounds[0], bounds[3])
            y = np.clip(y, bounds[1], bounds[4])
            z = np.clip(z, bounds[2], bounds[5])
    points = np.vstack((z, y, x)).T
    return np.vstack((points[:, 2], points[:, 1], points[:, 0], interpolator(points))).T


def sample(sdf_path: str, mesh_path: str, args: Any) -> None:
    sdf_dict = load_sdf(sdf_path, args.res)
    sdf_values = sdf_dict["values"]
    sdf_bounds = sdf_dict["bounds"]

    mesh = trimesh.load(mesh_path, process=False)
    if args.vis:
        mesh.show()
    assert mesh.is_watertight, f"Mesh {mesh_path} is not watertight. Skipping."

    if args.voxel:
        voxel_path = os.path.join('/'.join(sdf_path.split('/')[:-1]), "model.binvox")
        if args.overwrite or not os.path.isfile(voxel_path):
            binvox_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "src", "lib", "binvox")
            voxel = trimesh.exchange.binvox.voxelize_mesh(mesh,
                                                          dimension=32,
                                                          remove_internal=False,
                                                          center=True,
                                                          binvox_path=binvox_path)

            binvox = trimesh.exchange.binvox.export_binvox(voxel)  # Writes in 'xzy' format by default

            with open(voxel_path, "wb") as f:
                f.write(binvox)

    sample_path = get_sample_path(sdf_path, args)
    replace = sample_path.split('/')[-1]

    all_samples = list()
    sample_names = list()

    # 1. Uniform samples from voxel grid
    name = "uniform_grid.npy"
    uniform_grid_samples = None
    if name in args.samples and (args.overwrite or not os.path.isfile(sample_path.replace(replace, name))):
        uniform_grid_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points)
        sample_names.append(name)
        all_samples.append(uniform_grid_samples)

    # 2. Equal (inside/outside) samples from voxel grid
    name = "equal_grid.npy"
    inside_samples = None
    outside_samples = None
    if name in args.samples and (args.overwrite or not os.path.isfile(sample_path.replace(replace, name))):
        inside_mask = sdf_values <= 0
        outside_mask = sdf_values > 0
        inside_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points // 2, inside_mask)
        outside_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points // 2, outside_mask)
        equal_grid_samples = np.concatenate((inside_samples, outside_samples))
        sample_names.append(name)
        all_samples.append(equal_grid_samples)

    # 3. Surface/uniform samples from voxel grid
    name = "surface_grid.npy"
    if name in args.samples and (args.overwrite or not os.path.isfile(sample_path.replace(replace, name))):
        voxel_size = (1 - args.padding) * args.e / args.res
        surface_mask = (sdf_values <= voxel_size) & (sdf_values >= -voxel_size)
        surface_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points // 2, surface_mask)
        if uniform_grid_samples is None:
            uniform_grid_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points)
        uniform_samples = uniform_grid_samples[:args.num_points // 2]
        surface_grid_samples = np.concatenate((surface_samples, uniform_samples))
        sample_names.append(name)
        all_samples.append(surface_grid_samples)

    # 4. Uniform random samples in volume
    name = "uniform_random.npy"
    if name in args.samples:
        for n in range(args.num_uniform_random):
            if n > 0:
                name = f"uniform_random_{n - 1}.npy"
            if args.overwrite or not os.path.isfile(sample_path.replace(replace, name)):
                uniform_random_samples = uniform_random_sampling(sdf_values, sdf_bounds, args.num_points)
                sample_names.append(name)
                all_samples.append(uniform_random_samples)

    # 5. Equal (inside/outside) random samples in volume
    name = "equal_random.npy"
    if name in args.samples and (args.overwrite or not os.path.isfile(sample_path.replace(replace, name))):
        if inside_samples is None:
            inside_mask = sdf_values <= 0
            inside_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points // 2, inside_mask)
        inside_points = inside_samples[:, :3] + args.noise * rng.standard_normal((args.num_points // 2, 3))
        if outside_samples is None:
            outside_mask = sdf_values > 0
            outside_samples = uniform_grid_sampling(sdf_values, sdf_bounds, args.num_points // 2, outside_mask)
        outside_points = outside_samples[:, :3] + args.noise * rng.standard_normal((args.num_points // 2, 3))
        inside_samples = uniform_random_sampling(sdf_values, sdf_bounds, points=inside_points)
        outside_samples = uniform_random_sampling(sdf_values, sdf_bounds, points=outside_points)
        equal_random_samples = np.concatenate((inside_samples, outside_samples))
        sample_names.append(name)
        all_samples.append(equal_random_samples)

    # 6. Surface/uniform random samples in volume
    name = "surface_random.npy"
    if name in args.samples and (args.overwrite or not os.path.isfile(sample_path.replace(replace, name))):
        samples = list()
        sigmas = [0.001, 0.0015, 0.0025, 0.01, 0.015, 0.025, 0.05, 0.1, 0.15, 0.25]
        for sigma in sigmas:
            num_points = args.num_points // len(sigmas)
            noise = sigma * rng.standard_normal((num_points, 3))
            noisy_points = mesh.sample(num_points) + noise
            samples.append(uniform_random_sampling(sdf_values, sdf_bounds, points=noisy_points))
        surface_random_samples = np.concatenate(samples)
        sample_names.append(name)
        all_samples.append(surface_random_samples)

    # 7. IF-Net samples
    name = "if_net.npy"
    if name in args.samples and (args.overwrite or not os.path.isfile(sample_path.replace(replace, name))):
        samples = list()
        sigmas = [0.01, 0.015, 0.1, 0.15]
        for sigma in sigmas:
            num_points = args.num_points // len(sigmas)
            noise = sigma * rng.standard_normal((num_points, 3))
            noisy_points = mesh.sample(num_points) + noise
            samples.append(uniform_random_sampling(sdf_values, sdf_bounds, points=noisy_points))
        if_net_samples = np.concatenate(samples)
        sample_names.append(name)
        all_samples.append(if_net_samples)

    # 8. DeepSDF samples
    name = "deepsdf.npy"
    if name in args.samples and (args.overwrite or not os.path.isfile(sample_path.replace(replace, name))):
        samples = list()
        sigmas = [0.0025, 0.00025]
        for sigma in sigmas:
            num_points = int(args.num_points * 47 / 50) // len(sigmas)
            noise = sigma * rng.standard_normal((num_points, 3))
            noisy_points = mesh.sample(num_points) + noise
            samples.append(uniform_random_sampling(sdf_values, sdf_bounds, points=noisy_points))
        deepsdf_samples = np.concatenate(samples)
        deepsdf_samples = np.concatenate([deepsdf_samples,
                                          uniform_random_sampling(sdf_values,
                                                                  sdf_bounds,
                                                                  num_points=args.num_points - len(deepsdf_samples))])
        sample_names.append(name)
        all_samples.append(deepsdf_samples)

    # 9. Uniform random samples in sphere
    name = "uniform_sphere.npy"
    if name in args.samples:
        for radius in args.uniform_sphere_radii:
            name = f"uniform_sphere_{radius}.npy"
            if args.overwrite or not os.path.isfile(sample_path.replace(replace, name)):
                sphere_points = generate_random_basis(n_points=args.num_points, radius=radius, seed=None)
                uniform_sphere_samples = uniform_random_sampling(sdf_values, sdf_bounds, points=sphere_points,
                                                                 ignore_bounds=True)
                sample_names.append(name)
                all_samples.append(uniform_sphere_samples)

    # 10. Surface points and normals
    no_surface_file = not os.path.isfile(sample_path.replace(replace, "surface.npy"))
    no_normals_file = not os.path.isfile(sample_path.replace(replace, "normals.npy"))
    if args.overwrite or no_surface_file or no_normals_file:
        surface_points, index = mesh.sample(args.num_points, return_index=True)
        surface_normals = mesh.face_normals[index]

        if args.precision == 16:
            surface_points = surface_points.astype(np.float16)
            surface_normals = surface_normals.astype(np.float16)
        elif args.precision == 32:
            surface_points = surface_points.astype(np.float32)
            surface_normals = surface_normals.astype(np.float32)
        np.save(sample_path.replace(replace, "surface.npy"), surface_points)
        np.save(sample_path.replace(replace, "normals.npy"), surface_normals)
        if args.vis:
            trimesh.PointCloud(surface_points[:int(1e5)]).show()

    for samples, name in zip(all_samples, sample_names):
        if name in ["surface_random.npy", "equal_random.npy"]:
            assert len(samples) == args.num_points
        if args.precision == 16:
            samples = samples.astype(np.float16)
        elif args.precision == 32:
            samples = samples.astype(np.float32)
        np.save(sample_path.replace(replace, name), samples)

        if args.vis:
            points = samples[:, :3]
            sdfs = samples[:, 3]

            reds = get_cmap("Reds")
            blues = get_cmap("Blues").reversed()
            inside = sdfs[sdfs <= 0]
            outside = sdfs[sdfs > 0]
            inside_norm = (inside - inside.min()) / (inside.max() - inside.min())
            outside_norm = (outside - outside.min()) / (outside.max() - outside.min())
            inside = [reds(i) for i in inside_norm]
            outside = [blues(o) for o in outside_norm]

            colors = np.array([(0.0, 0.0, 0.0, 1.0) for _ in sdfs])
            colors[sdfs <= 0] = inside
            colors[sdfs > 0] = outside

            trimesh.PointCloud(points, colors).show()


def load_sdf(sdf_path: str, resolution: int = 256):
    intsize = 4
    floatsize = 8
    sdf = {
        "bounds": [],
        "values": []
    }
    with open(sdf_path, "rb") as f:
        try:
            bytes = f.read()
            ress = np.frombuffer(bytes[:intsize * 3], dtype=np.int32)
            if -1 * ress[0] != resolution or ress[1] != resolution or ress[2] != resolution:
                raise Exception(sdf_path, "res not consistent with ", str(resolution))
            positions = np.frombuffer(bytes[intsize * 3:intsize * 3 + floatsize * 6], dtype=np.float64)
            # bottom left corner, x,y,z and top right corner, x, y, z
            sdf["bounds"] = [positions[0], positions[1], positions[2], positions[3], positions[4], positions[5]]
            sdf["bounds"] = np.float32(sdf["bounds"])
            sdf["values"] = np.frombuffer(bytes[intsize * 3 + floatsize * 6:], dtype=np.float32)
            sdf["values"] = np.reshape(sdf["values"], (resolution + 1, resolution + 1, resolution + 1))
        finally:
            f.close()
    return sdf


def get_mesh_path(mesh_path: str, args: Any) -> str:
    filename = mesh_path.split('/')[-1].split('.')[0]
    if args.shapenet:
        synthset = mesh_path.split('/')[-3] if args.version == 1 else mesh_path.split('/')[-4]
        uid = mesh_path.split('/')[-2] if args.version == 1 else mesh_path.split('/')[-3]
        dir_path = os.path.join(args.o, synthset, uid)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(args.o, synthset, uid, f"{uid}.obj")
    return os.path.join(args.o, "mesh", filename + ".obj")


def get_sdf_path(mesh_path: str, args: Any) -> str:
    if not args.shapenet:
        return mesh_path.replace("/mesh/", "/sdf/").replace(".obj", ".dist")
    return mesh_path.replace(".obj", ".dist")


def get_sample_path(sdf_path, args):
    if not args.shapenet:
        return sdf_path.replace("/sdf/", "/samples/").replace(".dist", ".npz")
        # filename = sdf_path_list[-1].split('.')[0]
        # sdf_path_list.insert(-1, filename)
        # sdf_path = '/'.join(sdf_path_list)
        # os.makedirs('/'.join(sdf_path_list[:-1]), exist_ok=True)
    return sdf_path.replace(".dist", ".npz")


def run(mesh: str, args: Any) -> None:
    logger.setLevel(logging.WARNING)
    if args.verbose:
        logger.setLevel(logging.INFO)
    start_run = time.time()

    if not args.o:
        args.o = os.path.dirname(mesh)
    elif not args.shapenet:
        args.o = os.path.join(args.o, os.path.basename(mesh).split('.')[0])

    os.makedirs(args.o, exist_ok=True)
    if not args.shapenet:
        os.makedirs(os.path.join(args.o, "mesh"), exist_ok=True)
        os.makedirs(os.path.join(args.o, "sdf"), exist_ok=True)
        os.makedirs(os.path.join(args.o, "samples"), exist_ok=True)

    mesh_path = get_mesh_path(mesh, args)
    sdf_path = get_sdf_path(mesh_path, args)

    sdf_exists = os.path.isfile(sdf_path)
    mesh_exists = os.path.isfile(mesh_path.replace(".obj", ".ply"))
    sample_dir = '/'.join(get_sample_path(sdf_path, args).split('/')[:-1])
    sample_files = glob(sample_dir + "/*.npy")
    expected_num_samples = args.num_uniform_random + len(args.uniform_sphere_radii) + len(
        list(set(args.samples) & {"uniform_sphere", "uniform_random"})) + 2
    samples_exist = len(sample_files) == expected_num_samples
    if sdf_exists and mesh_exists and samples_exist and not args.overwrite:
        logger.info(f"Everything done. Skipping mesh {mesh}")
        return

    if not mesh_exists or args.overwrite:
        try:
            loc, scale = normalize_mesh(mesh, mesh_path, args)
        except Exception as e:
            logger.exception(e)
            return

    if not sdf_exists or args.overwrite:
        try:
            logger.info(f"Saving SDF to: {sdf_path}")
            kwargs = {'n': args.n,
                      's': False if args.u else True,
                      'o': os.path.relpath(sdf_path),
                      'm': args.m,
                      'b': args.b,
                      'c': args.c,
                      'e': args.e,
                      'd': args.d,
                      't': args.t,
                      'w': args.w,
                      'W': args.W,
                      'g': args.g,
                      'G': args.G,
                      'r': args.r,
                      'i': args.i,
                      'v': args.v,
                      'p': args.p,
                      "verbose": args.verbose}

            start = time.time()
            compute_distance_field(mesh_path, resolution=args.res, **kwargs)
            logger.info(f"SDF time: {time.time() - start}")
        except Exception as e:
            logger.exception(e)
            return
    else:
        logger.info(f"SDF computation for {mesh} done. Skipping.")

    mesh_path = mesh_path.replace(".obj", ".ply")
    if not mesh_exists or args.overwrite:
        try:
            logger.info(f"Saving mesh from marching cubes to: {mesh_path}")
            kwargs = {'o': mesh_path,
                      'i': 0.0,
                      'n': args.n,
                      "method": args.method,
                      "normals": False,
                      "loc": loc,
                      "scale": scale,
                      "verbose": args.verbose}

            start = time.time()
            if not mesh_from_sdf(sdf_path, args=args, **kwargs):
                kwargs["method"] = "vega"
                mesh_from_sdf(sdf_path, args=args, **kwargs)
            logger.info(f"Marching cubes time: {time.time() - start}")
        except Exception as e:
            logger.exception(e)
            return
    else:
        logger.info(f"Marching cubes for {mesh} done. Skipping.")

    if not samples_exist or args.overwrite:
        try:
            start = time.time()
            sample(sdf_path, mesh_path, args)
            logger.info(f"Sample time: {time.time() - start}")
        except Exception as e:
            logger.exception(e)
            return
    else:
        logger.info(f"Sampling for mesh {mesh} done. Skipping.")

    if not args.sdf:
        shutil.rmtree(Path(sdf_path).parent)
    if not args.mesh:
        shutil.rmtree(Path(mesh_path).parent)

    # Remove unnecessary files
    dir_name = Path(mesh_path).parent
    test = os.listdir(dir_name)
    keep = ["npy", "npz", "obj", "off", "ply", "binvox", "dist"]
    for item in test:
        if not item.split('.')[-1] in keep:
            path = os.path.join(dir_name, item)
            logger.info(f"Removing {path}")
            os.remove(path)

    logger.info(f"Runtime: {time.time() - start_run}")


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Computes SDFs from meshes.")
    parser.add_argument("meshes", nargs='+', type=str, help="List of meshes or glob pattern.")
    parser.add_argument("res", type=int, default=256, help="Voxel grid resolution.")
    parser.add_argument('-n', action="store_true", help="Compute only narrow band distance field.")
    parser.add_argument('-u', action="store_true", help="Compute unsigned distance field.")
    parser.add_argument('-o', "--out_dir", type=str, help="Output path.")
    parser.add_argument('-m', type=int, default=2, help="Signed field computation mode.")
    parser.add_argument('-b', help="Specify scene bounding box.")
    parser.add_argument('-c', action="store_true", help="Force bounding box into a cube.")
    parser.add_argument('-e', type=float, default=1.1, help="Expansion ratio for box.")
    parser.add_argument('-d', type=int, help="Max octree depth to use.")
    parser.add_argument('-t', type=int, help="Max num triangles per octree cell.")
    parser.add_argument('-w', type=float, help="The band width for narrow band distance field.")
    parser.add_argument('-W', type=int, help="Band width represented as #grid sizes.")
    parser.add_argument('-g', type=float, help="Sigma value.")
    parser.add_argument('-G', type=int, default=1, help="Sigma value represented as #grid sizes.")
    parser.add_argument('-r', action="store_true", help="Do not subtract sigma when creating signed field.")
    parser.add_argument('-i', type=str, default=None, help="Precomputed unsigned field for creating signed field.")
    parser.add_argument('-v', action="store_true", help="Also compute voronoi diagram.")
    parser.add_argument('-p', action="store_true", help="also compute closest points.")
    parser.add_argument("--num_points", type=int, default=int(1e5),
                        help="Number of points to sample from the SDF grid.")
    parser.add_argument("--num_uniform_random", type=int, default=1, help="Number of uniform random samples to take.")
    parser.add_argument("--uniform_sphere_radii", nargs='*', type=float, default=[1, 2, 5, 10],
                        help="Radii to use for uniform sphere sampling.")
    parser.add_argument("--samples", nargs='*', type=str, default=["uniform_random.npy", "uniform_sphere.npy"],
                        help="List of samples to take.")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise variance added to surface samples.")
    parser.add_argument("--padding", type=float, default=0.0, help="Padding applied when normalizing mesh.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--sdf", action="store_true", help="Store the computed SDF.")
    parser.add_argument("--mesh", action="store_true", help="Compute and store object mesh created from SDF.")
    parser.add_argument("--voxel", action="store_true", help="Compute and store voxelized mesh.")
    parser.add_argument("--shapenet", action="store_true", help="Assumes ShapeNet file structure for in- and output.")
    parser.add_argument("--version", type=int, default=1, help="ShapeNet version.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output during execution.")
    parser.add_argument("--visualize", action="store_true", help="Visualize SDF samples.")
    parser.add_argument("--method", type=str, default="skimage",
                        help="Marching cubes method (one of 'skimage' or 'vega').")
    parser.add_argument("--parallel", type=int, default=-1, help="Number of parallel processes to spawn.")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32, 64], help="Floating point precision.")
    parser.add_argument("--dont_fix_mesh", action="store_true", help="Keep mesh as is.")
    args = parser.parse_args()

    save_command_and_args_to_file(args.out_dir if args.out_dir else Path.cwd() / "command.txt", args)

    meshes = args.meshes
    if len(meshes) == 1:
        meshes = glob(resolve_path(meshes[0]))
    else:
        meshes = [resolve_path(mesh) for mesh in meshes]
    if args.verbose:
        print("Path(s) to mesh(es):", meshes)
    print("Processing", len(meshes), "mesh(es) using",
          cpu_count() if args.parallel == -1 else min(args.parallel, cpu_count()), "processes.")

    if 1 < args.parallel < len(meshes):
        with tqdm_joblib(tqdm(desc="SDF", total=len(meshes), disable=args.verbose)):
            with Parallel(n_jobs=min(args.parallel, cpu_count())) as parallel:
                parallel(delayed(run)(mesh, copy.deepcopy(args)) for mesh in meshes)
    else:
        for mesh in tqdm(meshes, disable=args.verbose):
            run(mesh, copy.deepcopy(args))

    print("Time taken:", time.time() - start)


if __name__ == "__main__":
    main()
