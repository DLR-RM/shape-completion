import copy
import logging
import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import open3d as o3d
import pyrender
from PIL import Image
from easy_o3d.utils import convert_depth_image_to_point_cloud, convert_rgbd_image_to_point_cloud
from pyrender.shader_program import ShaderProgramCache
from scipy.spatial.transform import Rotation as R
from trimesh import Trimesh

from libs.libkinect import KinectSim
from utils import inv_trafo, setup_logger, set_log_level, load_mesh, save_command_and_args_to_file
from .src.utils import resolve_dtype, normalize_mesh, look_at, part_sphere

logger = setup_logger(__name__)


def render(in_path: Path, out_dir: Path, args: Any):
    params_path = out_dir / 'parameters.npz'
    lock_path = out_dir / 'lock'
    kinect_path = out_dir / 'kinect'
    kinect_path.mkdir(parents=True, exist_ok=True)
    kinect_files = list(kinect_path.glob(f'*{args.out_format_depth}'))

    depth_path = out_dir / 'depth'
    normal_path = out_dir / 'normal'
    depth_path.mkdir(parents=True, exist_ok=True)
    normal_path.mkdir(parents=True, exist_ok=True)
    depth_files = list(depth_path.glob(f'*{args.out_format_depth}'))
    normal_files = list(normal_path.glob(f'*{args.out_format_normal}'))

    if args.check:
        try:
            data = np.load(str(params_path))
            for key in data:
                value = data[key]
                if key in ['max_depths', 'kinect_max_depths']:
                    for max_depth in value:
                        assert max_depth > 0, 'Error: Max depth is invalid.'
            for depth_file in depth_files:
                depth = np.asarray(Image.open(str(depth_file)))
                assert depth.any(), 'Error: Depth image is empty.'
            for normal_file in normal_files:
                normal = np.asarray(Image.open(str(normal_file)))
                assert normal.any(), 'Error: Normal image is empty.'
            for kinect_file in kinect_files:
                kinect_depth_map = np.asarray(Image.open(str(kinect_file)))
                assert kinect_depth_map.any(), 'Error: Kinect image is empty.'
        except Exception as e:
            logger.error(f'Files for {in_path} are corrupted.')
            logger.exception(e)
            if args.fix:
                logger.warning(f'Removing corrupted files for {in_path}.')
                params_path.unlink(missing_ok=True)
                lock_path.unlink(missing_ok=True)
                shutil.rmtree(kinect_path, ignore_errors=True)
                kinect_path.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(depth_path, ignore_errors=True)
                shutil.rmtree(normal_path, ignore_errors=True)
                depth_path.mkdir(parents=True, exist_ok=True)
                normal_path.mkdir(parents=True, exist_ok=True)
            elif args.remove:
                logger.warning(f'Check failed. Removing {out_dir}.')
                shutil.rmtree(out_dir, ignore_errors=True)
                return
            else:
                return

    lock = lock_path.exists()
    all_files = all(len(files) >= args.n_views for files in [depth_files, normal_files, kinect_files])

    if not all_files and not lock:
        open(out_dir / 'lock', 'w').close()

    if (all_files or lock) and not args.overwrite:
        logger.debug(f'File {in_path} already {"being " if lock else ""}processed. Skipping.')
        return

    depth_dtype = resolve_dtype(args.depth_precision, integer=True, unsigned=True)

    uint8_max = 2 ** 8 - 1
    uint16_max = 2 ** 16 - 1
    max_value = uint8_max if depth_dtype == np.uint8 else uint16_max

    intrinsic = np.array([[args.fx, 0, args.cx],
                          [0, args.fy, args.cy],
                          [0, 0, 1]])

    restart = time()
    mesh = Trimesh(*load_mesh(in_path,
                              force='mesh',
                              process=args.process,
                              validate=args.process,
                              enable_post_processing=args.process),
                   process=False,
                   validate=False)
    vertices, faces = mesh.vertices, mesh.faces
    if len(vertices) == 0 or len(faces) == 0:
        logger.error(f'Error: Mesh {in_path} is empty.')
        return
    logger.debug(f'Loaded mesh ({len(vertices)} vertices, {len(faces)} faces) in {time() - restart:.4f}s.')

    restart = time()
    scale_y = mesh.extents.max()
    scale = np.array([scale_y, scale_y, scale_y])
    logger.debug(f'Object scale is {scale_y}.')
    mesh = normalize_mesh(mesh)
    logger.debug(f'Normalized mesh in {time() - restart:.4f}s.')

    restart = time()
    scene = pyrender.Scene()
    camera = pyrender.IntrinsicsCamera(args.fx, args.fy, args.cx, args.cy, args.znear, args.zfar)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    logger.debug(f'Prepared scene in {time() - restart:.4f}s.')

    restart = time()
    renderer = pyrender.OffscreenRenderer(args.width, args.height)
    shader_dir = Path(__file__).parent.parent / 'utils' / 'assets' / 'shaders'
    renderer._renderer._program_cache = ShaderProgramCache(shader_dir=shader_dir)
    logger.debug(f'Prepared renderer in {time() - restart:.4f}s.')

    restart = time()
    kinect_sim = KinectSim()
    logger.debug(f'Prepared Kinect simulator in {time() - restart:.4f}s.')

    max_depths = list()
    kinect_max_depths = list()
    extrinsics = list()
    scales = list()
    rotations = list()

    for index in range(args.n_views):
        logger.debug(f'Rendering view {index + 1}/{args.n_views}.')

        tmp_mesh = mesh.copy()
        if args.scale_object:
            scale_xz = np.clip(np.abs(np.random.normal(0.15, 0.06)), 0.05, 0.5)
            scale = np.array([scale_xz, scale_xz, scale_xz])
            if args.distort_object:
                scale_y_offset = np.clip(np.random.normal(0, 0.1), -0.2, 0.2)
                scale_y = scale_xz * (1 + scale_y_offset)
                scale = np.array([scale_xz, scale_y, scale_xz])
            logger.debug(f'Sampled scale {scale}.')
        tmp_mesh.vertices *= scale

        trafo = np.eye(4)
        if args.rotate_object or args.axis or args.angle:
            poses_path = in_path.parent / 'poses.npy'
            if poses_path.exists():
                logger.debug(f'Loading poses from {poses_path}.')
                poses = np.load(str(poses_path))
                choice = np.random.randint(len(poses)) if np.random.random() < 0.5 else 0
                pose = poses[choice]
                trafo[:3, :3] = pose[:3, :3]

            axis = args.axis if args.axis else np.random.choice(['', 'x', 'z', 'xz'], p=[0.5, 0.2, 0.2, 0.1])
            if axis:
                angle = args.angle if args.angle else np.random.choice([0, 90, 180], size=len(axis))
                logger.debug(f'Rotating object around {axis} axis by {angle} degrees.')
                rot = R.from_euler(axis, angle, degrees=True).as_matrix()
                trafo[:3, :3] = rot @ trafo[:3, :3]

            tmp_mesh.apply_transform(trafo)

        mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(tmp_mesh, smooth=False))
        scene.add_node(mesh_node)

        inplane_rot = R.from_euler('z',
                                   args.inplane_rotation,
                                   degrees=True).as_matrix() if args.inplane_rotation else None

        min_pixels = 100
        inv_extrinsic = np.eye(4)
        depth_map = np.zeros((args.height, args.width), dtype=np.float32)
        normal_image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        kinect_depth_map = np.zeros((args.height, args.width), dtype=np.float32)
        for _ in range(100):
            radius = np.clip(np.abs(np.random.normal(0.65, 0.1)), 0.5, 1) + scale_y
            logger.debug(f'Sampled radius {radius:.4f}.')
            offset_x = np.clip(np.random.normal(0, 0.2), -0.2, 0.2)
            offset_y = 0
            offset_z = 0
            offset = np.array([offset_x, offset_y, offset_z])
            logger.debug(f'Sampled offset {offset}.')
            eye = part_sphere(center=np.zeros(3),
                              radius=radius,
                              mode='SURFACE',
                              part_sphere_dir_vector=[0, 1, 0])
            logger.debug(f'Sampled eye {eye}.')
            inv_extrinsic = look_at(eye=eye, target=np.array(offset))

            if inplane_rot is not None:
                inv_extrinsic[:3, :3] = inv_extrinsic[:3, :3] @ inplane_rot

            restart = time()
            scene.set_pose(camera_node, inv_extrinsic)
            normal_image, depth_map = renderer.render(scene, flags=pyrender.RenderFlags.OFFSCREEN)
            logger.debug(f'Rendered depth and normal map in {time() - restart:.4f}s.')

            restart = time()
            extrinsic = inv_trafo(inv_extrinsic)
            extrinsic[1, :] *= -1
            extrinsic[2, :] *= -1
            tmp_mesh.apply_transform(extrinsic)
            vertices, faces = tmp_mesh.vertices, tmp_mesh.faces
            kinect_depth_map = kinect_sim.simulate(vertices,
                                                   faces,
                                                   args.width,
                                                   args.height,
                                                   args.fx,
                                                   args.fy,
                                                   args.cx,
                                                   args.cy,
                                                   args.verbose)
            tmp_mesh.apply_transform(inv_trafo(extrinsic))
            logger.debug(f'Simulated Kinect depth in {time() - restart:.4f}s.')

            # Checks if the depth map contains enough pixels and if the minimum depth is greater than znear.
            if (kinect_depth_map > 0).sum() > min_pixels and kinect_depth_map[kinect_depth_map > 0].min() > args.znear:
                break
            else:
                # If not, decrease the number of pixels and scale the mesh.
                min_pixels -= 1
                scale *= 1.05
                tmp_mesh.vertices *= 1.05
                scene.remove_node(mesh_node)
                mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(tmp_mesh, smooth=False))
                scene.add_node(mesh_node)

                if args.show and args.verbose:
                    Image.fromarray((depth_map / depth_map.max() * 255).astype(np.uint8), mode='L').show()
                    Image.fromarray((kinect_depth_map / kinect_depth_map.max() * 255).astype(np.uint8), mode='L').show()
                    Image.fromarray(normal_image).show()

        if (kinect_depth_map > 0).sum() <= min_pixels or kinect_depth_map[kinect_depth_map > 0].min() <= args.znear:
            logger.warning(f'Could not find a valid view for mesh {in_path}. Removing {out_dir}.')
            shutil.rmtree(out_dir, ignore_errors=True)
            return

        scene.remove_node(mesh_node)
        scales.append(scale)
        rotations.append(trafo[:3, :3])

        restart = time()
        depth_map[(depth_map > args.zfar) | (depth_map < args.znear)] = 0
        depth_max = depth_map.max()
        depth_scale = max_value / depth_max
        depth_map *= depth_scale
        depth_map[depth_map > max_value] = max_value
        depth_map = np.round(depth_map).astype(depth_dtype)
        max_depths.append(depth_max)

        normal_map = normal_image.copy()
        normal_map[depth_map == 0] = np.zeros(3, dtype=np.uint8)
        logger.debug(f'Processed depth and normal map in {time() - restart:.4f}s.')

        restart = time()
        kinect_depth_map[(kinect_depth_map > args.zfar) | (kinect_depth_map < args.znear)] = 0
        kinect_depth_max = kinect_depth_map.max()
        kinect_depth_scale = max_value / kinect_depth_max
        kinect_depth_map *= kinect_depth_scale
        kinect_depth_map[kinect_depth_map > max_value] = max_value
        kinect_depth_map = np.round(kinect_depth_map).astype(depth_dtype)
        kinect_max_depths.append(kinect_depth_max)
        logger.debug(f'Processed Kinect depth map in {time() - restart:.4f}s.')

        restart = time()
        depth_path = out_dir / 'depth' / f'{index:05d}.png'
        normal_path = out_dir / 'normal' / f'{index:05d}.jpg'
        Image.fromarray(depth_map, mode='L' if depth_dtype == np.uint8 else 'I;16').save(depth_path)
        Image.fromarray(normal_map).save(normal_path, quality=args.normal_quality)
        logger.debug(f'Saved depth and normal map in {time() - restart:.4f}s.')

        restart = time()
        kinect_depth_path = out_dir / 'kinect' / f'{index:05d}.png'
        Image.fromarray(kinect_depth_map, mode='L' if depth_dtype == np.uint8 else 'I;16').save(kinect_depth_path)
        logger.debug(f'Saved Kinect depth map in {time() - restart:.4f}s.')

        extrinsic = inv_trafo(inv_extrinsic)
        extrinsic[1, :] *= -1
        extrinsic[2, :] *= -1
        extrinsics.append(extrinsic)

        if args.show:
            Image.fromarray((depth_map / depth_map.max() * 255).astype(np.uint8), mode='L').show()
            Image.fromarray((kinect_depth_map / kinect_depth_map.max() * 255).astype(np.uint8), mode='L').show()
            Image.fromarray(normal_image).show()

            restart = time()
            ds = 1 / depth_max if args.depth_precision == 8 else depth_scale
            pcd = convert_rgbd_image_to_point_cloud([str(normal_path), str(depth_path)],
                                                    intrinsic,
                                                    extrinsic,
                                                    depth_scale=ds,
                                                    depth_trunc=args.zfar,
                                                    convert_rgb_to_intensity=False)
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.colors) * 2 - 1)
            logger.debug(f'Converted depth and normal map to point cloud in {time() - restart:.4f}s.')

            restart = time()
            kinect_ds = 1 / kinect_depth_max if args.depth_precision == 8 else kinect_depth_scale
            kinect_pcd = convert_depth_image_to_point_cloud(str(kinect_depth_path),
                                                            intrinsic,
                                                            extrinsic,
                                                            depth_scale=kinect_ds,
                                                            depth_trunc=args.zfar,
                                                            convert_rgb_to_intensity=False)
            logger.debug(f'Converted Kinect depth map to point cloud in {time() - restart:.4f}s.')

            if args.verbose:
                o3d.visualization.draw_geometries([copy.deepcopy(kinect_pcd).paint_uniform_color([1, 0, 0]), pcd])
            o3d.visualization.draw_geometries([kinect_pcd,
                                               o3d.geometry.TriangleMesh(
                                                   vertices=o3d.utility.Vector3dVector(tmp_mesh.vertices),
                                                   triangles=o3d.utility.Vector3iVector(
                                                       tmp_mesh.faces)).compute_vertex_normals(),
                                               o3d.geometry.TriangleMesh().create_coordinate_frame(size=scale_y / 2),
                                               # o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1).transform(
                                               #     inv_extrinsic),
                                               o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1).transform(
                                                   inv_trafo(extrinsic))])

    data = {'scales': np.asarray(scales, dtype=np.float32),
            'max_depths': np.asarray(max_depths, dtype=np.float32),
            'kinect_max_depths': np.asarray(kinect_max_depths, dtype=np.float32),
            'intrinsic': intrinsic.astype(np.float32),
            'extrinsics': np.asarray(extrinsics, dtype=np.float32)}
    # Test if any rotation in rotations is not the identity
    rotations = np.asarray(rotations, dtype=np.float32)
    if not np.allclose(rotations, np.eye(3)):
        data['rotations'] = rotations
    np.savez_compressed(params_path, **data)

    renderer.delete()
    del kinect_sim
    lock_path.unlink(missing_ok=True)


def run(args: Namespace):
    start = time()
    in_path = args.in_file.expanduser().resolve()
    logger.debug(f'Processing file {in_path}.')
    out_dir = in_path.parent
    if args.out_dir:
        out_dir = args.out_dir.expanduser().resolve() / in_path.parent.parent.name / in_path.parent.name
    logger.debug(f'Output directory is {out_dir}.')
    try:
        render(in_path, out_dir, args)
    except Exception as e:
        logger.exception(e)
        if args.remove:
            logger.warning(f'Exception occurred. Removing {out_dir}.')
            shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / 'lock').unlink(missing_ok=True)
    logger.debug(f'Runtime: {time() - start:.2f}s.\n')


def get_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=Path, help='Path to output directory.')
    parser.add_argument('--out_format_depth', type=str, default='.png', help='Output file format for depth maps.')
    parser.add_argument('--out_format_normal', type=str, default='.jpg', help='Output file format for normal maps.')
    parser.add_argument('--depth_precision', type=int, default=16, choices=[8, 16], help='Precision of depth maps.')
    parser.add_argument('--normal_quality', type=str, default='web_low',
                        choices=['web_low', 'web_medium', 'web_high', 'web_very_high', 'web_maximum',
                                 'low', 'medium', 'high', 'maximum'],
                        help='JPEG quality of normal maps.')
    parser.add_argument('--n_views', type=int, default=100, help='Number of depth and normal views to render.')
    parser.add_argument('--width', type=int, default=640, help='Width of the depth map.')
    parser.add_argument('--height', type=int, default=480, help='Height of the depth map.')
    parser.add_argument('--fx', type=float, default=582.6989, help='Focal length in x.')
    parser.add_argument('--fy', type=float, default=582.6989, help='Focal length in y.')
    parser.add_argument('--cx', type=float, default=320.7906, help='Principal point in x.')
    parser.add_argument('--cy', type=float, default=245.2647, help='Principal point in y.')
    parser.add_argument('--znear', type=float, default=0.5, help='Near clipping plane.')
    parser.add_argument('--zfar', type=float, default=6.0, help='Far clipping plane.')
    parser.add_argument('--process', action='store_true', help='Process meshes before sampling/rendering.')
    parser.add_argument('--inplane_rotation', type=float, default=0.0, help='In-plane rotation of the camera.')
    parser.add_argument('--scale_object', action='store_true', help='Apply random scale to the object.')
    parser.add_argument('--distort_object', action='store_true', help='Apply random distortion to the object.')
    parser.add_argument('--rotate_object', action='store_true', help='Apply random rotation to the object.')
    parser.add_argument('--axis', type=str, choices=['x', 'y', 'z'], help='Rotation axis.')
    parser.add_argument('--angle', type=float, help='Rotation angle in degrees.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    parser.add_argument('--check', action='store_true', help='Check results.')
    parser.add_argument('--fix', action='store_true', help='Fix results that failed check.')
    parser.add_argument('--remove', action='store_true', help='Removes results that failed check.')
    parser.add_argument('--show', action='store_true', help='Visualize renders.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    return parser


def main():
    parser = get_argument_parser()
    parser.add_argument('in_file', type=Path, help='Path to input file.')
    args = parser.parse_args()

    save_command_and_args_to_file(args.out_dir / "command.txt", args)

    # Check that fix or remove are only set when check is set.
    if args.fix or args.remove:
        assert args.check, 'Fix or remove can only be set when check is set.'

    # Check that fix and remove are not set at the same time.
    assert not (args.fix and args.remove), 'Fix and remove cannot be set at the same time.'

    if args.verbose:
        set_log_level(logging.DEBUG)

    # Set EGL for offscreen rendering if not visualizing.
    if not args.show:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

    run(args)


if __name__ == '__main__':
    main()
