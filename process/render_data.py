import argparse
import os
import shutil
from glob import glob
from joblib import cpu_count
from typing import Any, List
from pathlib import Path

import blenderproc as bp
import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import save_command_and_args_to_file


def get_output_path(path: str, args):
    if "ShapeNetCore.v1" in path:
        output_path = os.path.join(args.out_dir, path.split('/')[-3], path.split('/')[-2])
    elif "ShapeNetCore.v2" in path:
        output_path = os.path.join(args.out_dir, path.split('/')[-4], path.split('/')[-3])
    elif "ShapeNet.build" in path:
        output_path = os.path.join(args.out_dir, path.split('/')[-3], path.split('/')[-1].split('.')[0])
    else:
        output_path = os.path.join(args.out_dir, path.split('/')[-3], "blenderproc", args.split)
    return output_path


def cleanup(path, args):
    output_path = get_output_path(path, args)
    for s in range(args.num_scales):
        done = False
        shard_path = os.path.join(output_path, "train_pbr", str(s).zfill(6))
        if os.path.isdir(shard_path):
            depth_dir = os.path.isdir(os.path.join(shard_path, "depth"))
            scale_file = os.path.isfile(os.path.join(shard_path, "scale.npy"))
            cam_file = os.path.isfile(os.path.join(shard_path, "scene_camera.json"))
            gt_file = os.path.isfile(os.path.join(shard_path, "scene_gt.json"))
            if depth_dir and scale_file and cam_file and gt_file:
                if all(os.path.isfile(os.path.join(shard_path,
                                                   "depth",
                                                   str(f).zfill(6) + ".png")) for f in range(args.num_cams)):
                    done = True
        if not done:
            print(f"Shard {shard_path} not complete. Deleting.")
            # shutil.rmtree(shard_path)


def main() -> List[str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_objects", nargs='+', type=str, help="Glob pattern to directory containing objects.")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--split", type=str.lower, default="", help="Output to train/val/test split directory.")
    parser.add_argument("--distance", type=float, nargs=2, default=[0.5, 1],
                        help="Min/max distance from camera/light to object center.")
    parser.add_argument("--elevation", type=int, nargs=2, default=[10, 80], help="Min/max camera elevation.")
    parser.add_argument("--azimuth", type=int, nargs=2, default=[-180, 180], help="Min/max azimuth angle.")
    parser.add_argument("--scale", type=float, nargs=2, default=[0.05, 0.4], help="Scale range to sample from.")
    parser.add_argument("--distort", type=float, default=0, help="Min/max amount of distortion to apply to the mesh")
    parser.add_argument("--symmetric_distortion", action="store_true", help="Apply distortion symmetrically.")
    parser.add_argument("--num_scales", type=int, default=10, help="Number of scales to sample.")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize object size to unit length prior to scaling.")
    parser.add_argument("--center", action="store_true", help="Center object at origin.")
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480], help="Width and height of render.")
    parser.add_argument("--camera_intrinsics", nargs=4, type=float, default=[640, 640, 320, 240],
                        help="Camera intrinsic parameters")
    parser.add_argument("--inplane_rot", type=int, default=90, help="Camera roll in degrees.")
    parser.add_argument("--num_cams", type=int, default=100, help="Number of camera positions to sample.")
    parser.add_argument("--use_cycles", action="store_true",
                        help="Use the Cycles raytracer instead of Eevee for rendering.")
    parser.add_argument("--remove_rgb", action="store_true", help="Remove color image after rendering.")
    parser.add_argument("--remove_normals", action="store_true", help="Remove normal map after rendering.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data.")
    parser.add_argument("--background_color", type=float, nargs=3, default=[0.05, 0.05, 0.05],
                        help="RGB background values (0-1).")
    parser.add_argument("--transparent_background", action="store_true", help="Make background transparent.")
    parser.add_argument("--random_background_color", action="store_true", help="Random background color.")
    parser.add_argument("--random_backlight", action="store_true", help="Random background light strength.")
    parser.add_argument("--random_appearance", action="store_true", help="Random specularity and roughness.")
    parser.add_argument("--random_metallic", action="store_true", help="Random 'metallicness'.")
    parser.add_argument("--random_color", action="store_true", help="Random object color.")
    parser.add_argument("--random_transparency", action="store_true", help="Random object transparency.")
    parser.add_argument("--random_light_color", action="store_true", help="Random light color.")
    parser.add_argument("--random_light_strength", action="store_true", help="Random light strength.")
    parser.add_argument("--randomize_everything", action="store_true", help="Randomize everything.")
    parser.add_argument("--parallel", action="store_true", help="Use parallel execution.")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup generated data.")
    args = parser.parse_args()

    save_command_and_args_to_file(args.out_dir if args.out_dir else Path.cwd() / "command.txt", args)

    paths = args.path_to_objects
    if len(paths) == 1:
        paths = glob(os.path.realpath(os.path.expanduser(paths[0])))
        np.random.shuffle(paths)
    else:
        paths = [os.path.realpath(os.path.expanduser(path)) for path in paths]
    print("Path(s) to data:", paths)
    print(f"Found {len(paths)} objects in the provided directory.")

    error_list = list()
    if len(paths) > cpu_count() and args.parallel:
        with Parallel(n_jobs=min(len(paths), cpu_count())) as parallel:
            if args.cleanup:
                parallel(delayed(cleanup)(path, args) for path in tqdm(paths))
            else:
                error_list.extend(parallel(delayed(setup_and_run)(p, path, args) for p, path in enumerate(paths)))
    else:
        setup(args)
        for p, path in enumerate(paths):
            try:
                error_list.extend(run(p, path, args))
                bp.object.delete_multiple(bp.object.get_all_mesh_objects())
            except FileNotFoundError as e:
                print(e)
    return error_list


def setup_and_run(p: int, path: str, args: Any) -> List[str]:
    setup(args)
    try:
        return run(p, path, args)
    except FileNotFoundError as e:
        print(e)


def setup(args):
    resolution = args.resolution
    camera_intrinsics = args.camera_intrinsics
    cam_intrinsics = np.eye(3)
    cam_intrinsics[0, 0] = camera_intrinsics[0]
    cam_intrinsics[1, 1] = camera_intrinsics[1]
    cam_intrinsics[0, 2] = camera_intrinsics[2]
    cam_intrinsics[1, 2] = camera_intrinsics[3]

    if args.use_cycles:
        bp.init(horizon_color=args.background_color,
                compute_device="GPU",
                compute_device_type="OPTIX")
        # bp.renderer.set_denoiser("OPTIX")
        bp.renderer.set_max_amount_of_samples(0 if args.remove_rgb else 50)
        bp.renderer.set_light_bounces(diffuse_bounces=0,
                                      glossy_bounces=0,
                                      ao_bounces_render=0,
                                      max_bounces=0,
                                      transmission_bounces=0,
                                      transparent_max_bounces=0,
                                      volume_bounces=0)
        if args.transparent_background:
            bp.renderer.set_output_format(enable_transparency=True)

    bp.camera.set_resolution(*resolution)
    bp.camera.set_intrinsics_from_K_matrix(cam_intrinsics, image_width=resolution[0], image_height=resolution[1])
    bp.renderer.enable_depth_output(activate_antialiasing=False)
    if not args.remove_normals:
        bp.renderer.enable_normals_output()


def run(p: int, path: str, args: Any) -> List[str]:
    output_path = get_output_path(path, args)
    bp.object.delete_multiple(bp.object.get_all_mesh_objects())
    print(f"Loading object {p + 1} from {path}:")
    obj = bp.loader.load_obj(path)
    assert len(obj) == 1, print(obj)
    os.makedirs(output_path, exist_ok=True)
    obj = obj[0]
    obj.set_shading_mode("auto")
    obj.set_cp("category_id", p)
    if args.center:
        location = (obj.get_bound_box().max(axis=0) + obj.get_bound_box().min(axis=0)) / 2
        obj.set_location(obj.get_location() - location)
        new_location = (obj.get_bound_box().max(axis=0) + obj.get_bound_box().min(axis=0)) / 2
        assert np.allclose(new_location, np.zeros(3), atol=1e-7), new_location
    if args.normalize:
        max_extend = (obj.get_bound_box().max(axis=0) - obj.get_bound_box().min(axis=0)).max()
        obj.set_scale(obj.get_scale() * np.reciprocal(max_extend))
        new_max_extend = (obj.get_bound_box().max(axis=0) - obj.get_bound_box().min(axis=0)).max()
        assert np.isclose(new_max_extend, 1), new_max_extend
    if "shapenet" in path.lower():
        obj.set_rotation_euler([np.deg2rad(90), 0, 0])
    orig_scale = obj.get_scale()

    error_list = list()
    for s, scale in enumerate(np.random.uniform(*args.scale, size=args.num_scales)):
        done = False
        shard_path = os.path.join(output_path, "train_pbr", str(s).zfill(6))
        if not args.overwrite and os.path.isdir(shard_path):
            depth_dir = os.path.isdir(os.path.join(shard_path, "depth"))
            scale_file = os.path.isfile(os.path.join(shard_path, "scale.npy"))
            cam_file = os.path.isfile(os.path.join(shard_path, "scene_camera.json"))
            gt_file = os.path.isfile(os.path.join(shard_path, "scene_gt.json"))
            if depth_dir and scale_file and cam_file and gt_file:
                if all(os.path.isfile(os.path.join(shard_path,
                                                   "depth",
                                                   str(f).zfill(6) + ".png")) for f in range(args.num_cams)):
                    done = True
        if not done and os.path.exists(shard_path):
            print(f"{output_path} partially finished. Reporting.")
            error_list.append(shard_path)
            continue
        elif done:
            print(f"Rendering for path {output_path} (shard: {s + 1}/{args.num_scales}) finished. Skipping.")
            continue
        print(f"Using scale {scale} ({s + 1}/{args.num_scales}):")
        bp.utility.reset_keyframes()

        if args.distort:
            if args.symmetric_distortion:
                scale_x = scale_z = np.random.uniform(1 - args.distort, 1 + args.distort)
            else:
                scale_x, scale_z = np.random.uniform(1 - args.distort, 1 + args.distort, size=2)
            scale = np.array([scale * scale_x, scale, scale * scale_z])
        obj.set_scale(orig_scale * scale)
        if args.normalize:
            max_extend = (obj.get_bound_box().max(axis=0) - obj.get_bound_box().min(axis=0)).max()
            assert np.isclose(max_extend, scale), max_extend

        if not args.remove_rgb:
            num_lights = np.random.randint(1, 4)
            for _ in range(num_lights):
                light = bp.types.Light()
                light.set_type("POINT")
                light.set_location(bp.sampler.shell(center=(0, 0, 0) if args.center else obj.get_location(),
                                                    radius_min=args.distance[0],
                                                    radius_max=1.5,
                                                    elevation_min=1,
                                                    elevation_max=89))
                if args.random_light_color or args.randomize_everything:
                    light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
                if args.random_light_strength or args.randomize_everything:
                    light.set_energy(np.random.uniform(50 // num_lights))
                else:
                    light.set_energy(50)

            if args.random_appearance or args.randomize_everything:
                for mat in obj.get_materials():
                    mat.set_principled_shader_value("Specular", np.random.uniform())
                    mat.set_principled_shader_value("Roughness", np.random.uniform())
                    if args.random_metallic or args.randomize_everything:
                        mat.set_principled_shader_value("Metallic", np.random.uniform())
                    if args.random_color or args.randomize_everything:
                        color = list(np.random.uniform(size=3))
                        if args.random_transparency or args.randomize_everything:
                            color += [np.random.uniform(0.5, 1)]
                        else:
                            color += [1]
                        mat.set_principled_shader_value("Base Color", color)

        for c in range(args.num_cams):
            print(f"Sampling camera position {c + 1}/{args.num_cams}:")
            az_min = args.azimuth[0]
            az_max = args.azimuth[1]
            if args.split == "test":
                # Generates views +/- 10 degrees from the negative X-axis
                az_min = 170 if np.random.random() >= 0.5 else -180
                az_max = 180 if az_min == 170 else -170
            location = bp.sampler.shell(center=(0, 0, 0) if args.center else obj.get_location(),
                                        radius_min=args.distance[0],
                                        radius_max=args.distance[1],
                                        elevation_min=args.elevation[0],
                                        elevation_max=args.elevation[1],
                                        azimuth_min=az_min,
                                        azimuth_max=az_max)
            forward_vec = (obj.get_location() - location)
            rotation_matrix = bp.camera.rotation_from_forward_vec(forward_vec,
                                                                  inplane_rot=np.deg2rad(args.inplane_rot))
            cam2world_matrix = bp.math.build_transformation_mat(location, rotation_matrix)
            bp.camera.add_camera_pose(cam2world_matrix)

        data = bp.renderer.get_image()

        bp.writer.write_bop(output_path,
                            target_objects=[obj],
                            depths=data["depth"],
                            colors=data["colors"],
                            color_file_format="PNG" if args.transparent_background else "JPEG",
                            frames_per_chunk=args.num_cams)

        chunks_dir = os.path.join(output_path, 'train_pbr')
        chunk_dirs = sorted(glob(os.path.join(chunks_dir, '*')))
        chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d)]

        if len(chunk_dirs):
            last_chunk_dir = sorted(chunk_dirs)[-1]
            if last_chunk_dir != shard_path:
                os.makedirs(shard_path, exist_ok=True)
                os.rename(last_chunk_dir, shard_path)

        if not args.remove_normals:
            normal_path = os.path.join(shard_path, "normal")
            os.makedirs(normal_path, exist_ok=True)
            bp.writer.write_hdf5(normal_path, dict(normals=data["normals"]))
            for index, normal in enumerate(data["normals"]):
                print(normal.min(), normal.max())
                normal[np.all(normal == (0.5, 0.5, 0.5), axis=-1)] = np.ones(3)
                cv2.imwrite(os.path.join(normal_path, f"{str(index).zfill(6)}.png"),
                            np.clip(normal[..., ::-1] * 255, 0, 255))

        rgb_path = os.path.join(shard_path, "rgb")
        if args.remove_rgb and os.path.exists(rgb_path):
            shutil.rmtree(rgb_path, ignore_errors=True)
        np.save(os.path.join(shard_path, "scale"), scale)
    return error_list


if __name__ == "__main__":
    errors = main()
    if errors:
        print("Errors occurred:")
        for error in errors:
            print(error)
