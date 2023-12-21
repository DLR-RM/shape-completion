import glob
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from typing import List, Any

import h5py
import numpy as np
from PIL import Image
from rembg import remove
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

camera_models = {0: "SIMPLE_PINHOLE",
                 1: "PINHOLE",
                 2: "SIMPLE_RADIAL",
                 3: "RADIAL",
                 4: "OPENCV",
                 5: "OPENCV_FISHEYE",
                 6: "FULL_OPENCV",
                 7: "FOV",
                 8: "SIMPLE_RADIAL_FISHEYE",
                 9: "RADIAL_FISHEYE",
                 10: "THIN_PRISM_FISHEYE"}


def inv_trafo(trafo: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = trafo[:3, :3].T
    inverse[:3, 3] = -trafo[:3, :3].T @ trafo[:3, 3]
    return inverse


def get_camera(cam_id: int,
               camera_model: str,
               width: int,
               height: int,
               calibration: h5py.File) -> List:
    K = calibration[f"N{cam_id}_rgb_K"][:]
    d = calibration[f"N{cam_id}_rgb_d"][:]

    fx, fy, cx, cy, s = K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]
    k1, k2, p1, p2, k3, k4, k5, k6 = d[0], d[1], d[2], d[3], d[4], 0, 0, 0

    camera = [cam_id, camera_model, width, height, fx]
    if camera_model == camera_models[0]:
        camera.extend([cx, cy])
    elif camera_model == camera_models[1]:
        camera.extend([fy, cx, cy])
    elif camera_model in [camera_models[2], camera_models[8]]:
        camera.extend([cx, cy, k1])
    elif camera_model in [camera_models[3], camera_models[9]]:
        camera.extend([cx, cy, k1, k2])
    elif camera_model == camera_models[4]:
        camera.extend([fy, cx, cy, k1, k2, p1, p2])
    elif camera_model == camera_models[5]:
        camera.extend([fy, cx, cy, k1, k2, k3, 0])
    elif camera_model in [camera_models[6], camera_models[10]]:
        camera.extend([fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6])
    elif camera_model == camera_models[7]:
        camera.extend([fy, cx, cy, 0])
    else:
        raise ValueError
    return camera


def get_image(cam_id: int,
              angle: int,
              image_counter: int,
              data_dir: str,
              calibration: h5py.File) -> List:
    image_id = angle // 3 + 1
    image_name = "image" + str(image_id).zfill(3) + ".jpg"

    pose_data = h5py.File(os.path.join(data_dir, "poses", f"NP5_{angle}_pose.h5"))
    f_table_ref = pose_data["H_table_from_reference_camera"][:]
    pose_data.close()

    f_rgb_ref = calibration[f"H_N{cam_id}_from_NP5"][:]
    f_ref_rgb = inv_trafo(f_rgb_ref)
    f_table_rgb = f_table_ref @ f_ref_rgb
    f_rgb_table = inv_trafo(f_table_rgb)
    QX, QY, QZ, QW = R.from_matrix(f_rgb_table[:3, :3]).as_quat()
    TX, TY, TZ = f_rgb_table[:3, 3]

    image = [image_counter, QW, QX, QY, QZ, TX, TY, TZ, cam_id, f"cam{cam_id}/{image_name}"]
    return image


def get_mask(data_dir: str,
             image_path: str,
             pad: List[int],
             image: np.ndarray) -> np.ndarray:
    image_id = int(image_path.split('/')[-1].split('_')[-1].split('.')[0]) // 3 + 1
    init_mask_path = os.path.join(data_dir, "masks",
                                  image_path.split('/')[-1].replace(".jpg", "_mask.pbm"))
    mask = np.asarray(Image.open(init_mask_path, formats=["PPM"]).convert('L'), dtype=bool)
    extend = np.argwhere(~mask)
    _min, _max = extend.min(axis=0), extend.max(axis=0)
    mask = np.ones_like(mask, dtype=bool)
    mask[_min[0] - pad[0]:_max[0] + pad[1], _min[1] - pad[2]:_max[1] + pad[3]] = False

    rembg_input = image[_min[0] - pad[0]:_max[0] + pad[1], _min[1] - pad[2]:_max[1] + pad[3]]
    if args.show_bbox and image_id == 1:
        from cv2 import imshow, waitKey, destroyAllWindows, resize
        imshow("Bounding Box", resize(rembg_input, (rembg_input.shape[1] // 2, rembg_input.shape[0] // 2)))
        waitKey(0)
        destroyAllWindows()

    rembg_output = remove(rembg_input, alpha_matting=True, alpha_matting_erode_size=15, only_mask=True)
    test = Image.composite(Image.fromarray(rembg_input),
                           Image.fromarray(np.ones_like(rembg_input) * 255),
                           Image.fromarray(rembg_output))
    test.thumbnail((224, 224))
    new_im = Image.new("RGB", size=(224, 224), color="white")
    new_im.paste(test, (224 - test.size[0] - (224 - test.size[0]) // 2,
                        224 - test.size[1] - (224 - test.size[1]) // 2))
    new_im.save(path, quality=100, subsampling=0)

    mask = np.zeros_like(mask, dtype=rembg_output.dtype)
    mask[_min[0] - pad[0]:_max[0] + pad[1], _min[1] - pad[2]:_max[1] + pad[3]] = rembg_output
    return mask


def main(args: Any):
    all_images_dir = os.path.join(args.data_dir, "sfm", "images")
    all_masks_dir = all_images_dir.replace("images", "masks")
    colmap_dir = os.path.join(args.data_dir, "sfm", "colmap")
    database_path = os.path.join(colmap_dir, "database.db")
    camera_model_id = dict((v, k) for k, v in camera_models.items())[args.camera_model]
    camera_model = camera_models[camera_model_id]
    cameras = list()
    images = list()

    if args.data or (args.sparse and not os.path.isfile(database_path)):
        data_dir = args.data_dir
        pad = args.padding

        calibration = h5py.File(os.path.join(data_dir, "calibration.h5"))
        image_counter = 1

        for cam_id in args.camera_ids:
            cameras.append(get_camera(cam_id, camera_model, args.width, args.height, calibration))

            image_dir = os.path.join(all_images_dir, f"cam{cam_id}")
            mask_dir = image_dir.replace("images", "masks")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            for angle in tqdm(np.arange(0, 360, 3), desc=f"Processing images [camera {cam_id}/{len(args.camera_ids)}]"):
                image_id = angle // 3 + 1
                image_name = "image" + str(image_id).zfill(3) + ".jpg"

                image_path = os.path.join(data_dir, f"N{cam_id}_{angle}.jpg")
                copy_image_path = os.path.join(image_dir, image_name)
                if not os.path.isfile(copy_image_path):
                    shutil.copy(image_path, copy_image_path)

                images.append(get_image(cam_id, angle, image_counter, data_dir, calibration))
                image_counter += 1

                depth_mask_path = os.path.join(mask_dir, image_name)
                mask_depth = args.mask_depth and not os.path.isfile(depth_mask_path)

                masked_dir = image_dir.replace("images", "masked")
                masked_path = os.path.join(masked_dir, image_name)
                masked = args.masked and not os.path.isfile(masked_path)

                refined_mask_path = os.path.join(mask_dir, image_name + ".png")
                if not os.path.isfile(refined_mask_path) or masked:
                    pil_image = Image.open(image_path, formats=["JPEG"])
                    image = np.asarray(pil_image)
                if not os.path.isfile(refined_mask_path) or mask_depth or masked:
                    try:
                        mask = np.asarray(Image.open(refined_mask_path, formats=["PNG"]))
                    except FileNotFoundError:
                        mask = get_mask(data_dir, image_path, pad, image)
                        Image.fromarray(mask).save(refined_mask_path)

                if mask_depth:
                    Image.fromarray(mask).save(depth_mask_path,
                                               quality=100,
                                               subsampling=0)

                if masked:
                    os.makedirs(masked_dir, exist_ok=True)
                    masked_image = Image.composite(pil_image,
                                                   Image.fromarray(np.zeros_like(image)),
                                                   Image.fromarray(mask))
                    masked_image.save(masked_path,
                                      quality=100,
                                      subsampling=0,
                                      exif=pil_image.info["exif"])

        calibration.close()

    sparse_dir = os.path.join(colmap_dir, "sparse")
    dense_dir = os.path.join(colmap_dir, "dense")
    pipeline = list()

    if args.sparse or args.dense:
        os.makedirs(colmap_dir, exist_ok=True)
        if args.sparse or args.mask_depth:
            sys.path.insert(0, os.path.join(args.colmap_github_dir, "scripts", "python"))
        if args.sparse:
            os.makedirs(sparse_dir, exist_ok=True)
        if args.dense:
            os.makedirs(dense_dir, exist_ok=True)

    if args.sparse and not os.path.isfile(os.path.join(sparse_dir, "database.db")):
        extract_features = f"colmap feature_extractor" \
                           f" --database_path {database_path}" \
                           f" --image_path {all_images_dir}" \
                           f" --ImageReader.mask_path {all_masks_dir}" \
                           f" --ImageReader.camera_model {camera_model}" \
                           f" --ImageReader.single_camera_per_folder 1" \
                           f" --SiftExtraction.max_image_size {args.max_image_size}" \
                           f" --SiftExtraction.estimate_affine_shape {args.estimate_affine_shape}" \
                           f" --SiftExtraction.domain_size_pooling {args.domain_size_pooling}"
        subprocess.run(extract_features.split(' '))

        from database import COLMAPDatabase, blob_to_array
        db = COLMAPDatabase.connect(database_path)
        rows = db.execute("SELECT * FROM images")
        db_images = list(rows)

        new_images = [list()] * len(images)
        for db_image in db_images:
            for image in images:
                if db_image[1] == image[-1]:
                    image[0] = db_image[0]
                    new_images[db_image[0] - 1] = image

        assert len(db_images) == len(new_images)
        for db_image, image in zip(db_images, new_images):
            assert db_image[0] == image[0]
            assert db_image[1] == image[-1]
        db.close()

        if args.update_cameras:
            db = COLMAPDatabase.connect(database_path)
            for camera in cameras:
                command = f"UPDATE cameras SET params=(?) WHERE camera_id={camera[0]}"
                db.execute(command, (np.array(camera[4:]).tobytes(),))
            db.commit()
            db.close()

        db = COLMAPDatabase.connect(database_path)
        rows = db.execute("SELECT * FROM cameras")
        db_cams = list(rows)

        assert len(db_cams) == len(cameras)
        for db_cam, cam in zip(db_cams, cameras):
            camera_id, model, width, height, params, prior = db_cam
            if args.update_cameras:
                assert camera_id == cam[0]
                assert model == camera_model_id and width == cam[2] and height == cam[3]
            else:
                cameras[camera_id - 1] = [camera_id,
                                          camera_models[model],
                                          width,
                                          height,
                                          *blob_to_array(params, np.float64)]
        db.close()

        with open(os.path.join(colmap_dir, "cameras.txt"), 'w') as outfile:
            outfile.write("\n".join(' '.join(str(item) for item in camera) for camera in cameras))
        with open(os.path.join(colmap_dir, "images.txt"), 'w') as outfile:
            outfile.write("\n".join(' '.join(str(item) for item in image) + "\n" for image in new_images) + "\n")
        open(os.path.join(colmap_dir, "points3D.txt"), 'a').close()

        pipeline.append(f"colmap exhaustive_matcher"
                        f" --database_path {database_path}"
                        f" --SiftMatching.cross_check 1"
                        f" --SiftMatching.multiple_models {args.multiple_models}"
                        f" --SiftMatching.guided_matching {args.guided_matching}")
        if not os.path.isfile(os.path.join(sparse_dir, "points3D.bin")):
            pipeline.append(f"colmap point_triangulator"
                            f" --database_path {database_path}"
                            f" --image_path {all_images_dir}"
                            f" --input_path {colmap_dir}"
                            f" --output_path {sparse_dir}"
                            f" --Mapper.multiple_models {args.multiple_models}"
                            f" --Mapper.max_reg_trials 5"
                            f" --Mapper.tri_ignore_two_view_tracks {args.ignore_two_view_tracks}")
    sparse_exists = os.path.isfile(os.path.join(sparse_dir, "points3D.bin"))
    sparse_will_exist = pipeline and "point_triangulator" in pipeline[-1]
    if args.show_sparse and (sparse_exists or sparse_will_exist):
        pipeline.append(f"python {args.colmap_github_dir}/scripts/python/visualize_model.py"
                        f" --input_model {os.path.join(sparse_dir)}"
                        f" --input_format .bin")

    if args.dense:
        if args.mask_depth and not os.path.isdir(os.path.join(dense_dir, "masks")):
            pipeline.append(f"colmap image_undistorter"
                            f" --image_path {all_masks_dir}"
                            f" --input_path {sparse_dir} "
                            f" --output_path {dense_dir}"
                            f" --max_image_size {args.max_depth_size}")
        if not os.path.isdir(os.path.join(dense_dir, "images")):
            pipeline.append(f"colmap image_undistorter"
                            f" --image_path {all_images_dir}"
                            f" --input_path {sparse_dir}"
                            f" --output_path {dense_dir}"
                            f" --max_image_size {args.max_depth_size}")
        pipeline.append(f"colmap patch_match_stereo"
                        f" --workspace_path {dense_dir}"
                        f" --PatchMatchStereo.max_image_size {args.max_depth_size}"
                        f" --PatchMatchStereo.window_radius {args.window_radius}"
                        f" --PatchMatchStereo.geom_consistency {0 if args.fast_depth else 1}"
                        f" --PatchMatchStereo.window_step {2 if args.fast_depth else 1}"
                        f" --PatchMatchStereo.num_samples {10 if args.fast_depth else 15}"
                        f" --PatchMatchStereo.num_iterations {3 if args.fast_depth else 5}"
                        f" --PatchMatchStereo.cache_size 32")
        if not os.path.isfile(os.path.join(dense_dir, "fused.ply")):
            pipeline.append(f"colmap stereo_fusion"
                            f" --workspace_path {dense_dir}"
                            f" --input_type {'photometric' if args.fast_depth else 'geometric'}"
                            f" --output_path {dense_dir}/fused.ply"
                            f" --StereoFusion.min_num_pixels {args.min_num_pixels}"
                            f" --StereoFusion.max_image_size {args.max_depth_size}"
                            f" --StereoFusion.check_num_images {120 * len(args.camera_ids)}"
                            f" --StereoFusion.cache_size 32")

    if args.mesh:
        pipeline.append(f"colmap poisson_mesher"
                        f" --input_path {dense_dir}/fused.ply"
                        f" --output_path {dense_dir}/meshed-poisson.ply"
                        f" --PoissonMeshing.trim 7")
        pipeline.append(f"colmap delaunay_mesher"
                        f" --input_path {dense_dir}"
                        f" --output_path {dense_dir}/meshed-delaunay.ply"
                        f" --DelaunayMeshing.quality_regularization 5"
                        f" --DelaunayMeshing.max_proj_dist 5")

    for step in pipeline:
        if args.mask_depth and "stereo_fusion" in step:
            from read_write_dense import read_array, write_array
            for cam_id in args.camera_ids:
                os.makedirs(os.path.join(dense_dir, "masks", f"cam{cam_id}"), exist_ok=True)
                os.makedirs(os.path.join(dense_dir, "stereo", "depth_maps_copy", f"cam{cam_id}"), exist_ok=True)
                os.makedirs(os.path.join(dense_dir, "stereo", "normal_maps_copy", f"cam{cam_id}"), exist_ok=True)
                path = os.path.join(dense_dir, "stereo", "depth_maps", f"cam{cam_id}")
                for depth_path in tqdm(
                        glob.glob(os.path.join(path, f"*{'photometric' if args.fast_depth else 'geometric'}.bin")),
                        desc=f"Masking depth [camera {cam_id}/{len(args.camera_ids)}]"):
                    normals_path = depth_path.replace("depth_maps", "normal_maps")
                    depth_copy_path = depth_path.replace("depth_maps", "depth_maps_copy")
                    normals_copy_path = normals_path.replace("normal_maps", "normal_maps_copy")

                    if not os.path.isfile(depth_copy_path) or not os.path.isfile(normals_copy_path):
                        if not os.path.isfile(depth_copy_path):
                            shutil.copy(depth_path, depth_copy_path)
                        if not os.path.isfile(normals_copy_path):
                            shutil.copy(normals_path, normals_copy_path)

                        depth = read_array(depth_copy_path)
                        normals = read_array(normals_copy_path)

                        image_name = depth_path.split('/')[-1].split('.')[0] + ".jpg"
                        image_path = os.path.join(dense_dir, "images", f"cam{cam_id}", image_name)
                        mask_path = image_path.replace("images", "masks")
                        if os.path.isfile(mask_path):
                            mask = np.asarray(Image.open(mask_path, formats=["JPEG"]).convert('L'))
                        elif "PINHOLE" in args.camera_model and args.max_image_size == args.max_depth_size:
                            mask_path = os.path.join(all_masks_dir, f"cam{cam_id}", image_name + ".png")
                            mask = np.asarray(Image.open(mask_path, formats=["PNG"]).convert('L'))
                        else:
                            image = Image.open(image_path, formats=["JPEG"])
                            if image.size[:2] != depth.shape:
                                image.thumbnail(size=(depth.shape[1], depth.shape[0]))
                            mask = remove(np.asarray(image),
                                          alpha_matting=True,
                                          alpha_matting_erode_size=15,
                                          only_mask=True)
                            Image.fromarray(mask).save(mask_path,
                                                       quality=100,
                                                       subsampling=0)
                        if mask.shape != depth.shape:
                            mask = Image.fromarray(mask)
                            mask.thumbnail(size=(depth.shape[1], depth.shape[0]))
                            mask = np.asarray(mask)

                        if depth.shape == normals.shape[:2] == mask.shape:
                            depth[mask == 0] = 0
                            normals[mask == 0] = 0
                            write_array(depth, depth_path)
                            write_array(normals, normals_path)
                        else:
                            print("Warning: Mask shape != depth shape", mask.shape, depth.shape)
        subprocess.run(step.split(' '))
        if args.mask_depth and "image_undistorter" in step and all_masks_dir in step:
            shutil.move(os.path.join(dense_dir, "images"), os.path.join(dense_dir, "masks"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Reconstruct YCB objects from images using photogrammetry.")
    parser.add_argument("data_dir", type=str, help="Path to YCB object data directory.")
    parser.add_argument("-gh", "--colmap_github_dir", required=True, type=str, help="Path to colmap GitHub directory.")
    parser.add_argument("--padding", nargs=4, type=int, default=[150, 150, 150, 150],
                        help="Padding around existing default object masks.")
    parser.add_argument("--max_image_size", type=int, default=3200, help="Maximum image size. Decrease if OOM occurs.")
    parser.add_argument("--max_depth_size", type=int, default=1600,
                        help="Maximum estimated depth map size."
                             "Increase to obtain denser output for weakly textured surfaces.")
    parser.add_argument("--window_radius", type=int, default=10,
                        help="Stereo reconstruction window size. Increase for weakly textured surfaces.")
    parser.add_argument("--min_num_pixels", type=int, default=10,
                        help="Minimum allowed number of pixels in stereo fusion. Increase to reduce outliers.")
    parser.add_argument("--camera_model", type=str, default="OPENCV", choices=camera_models.values(),
                        help="Camera model defining the distortion parameters.")
    parser.add_argument("--guided_matching", type=int, default=True, help="Use guided SIFT feature matching.")
    parser.add_argument("--ignore_two_view_tracks", action="store_true", help="Enable if sparse model is too noisy.")
    parser.add_argument("--estimate_affine_shape", type=int, default=True, help="Estimate affine SIFT features.")
    parser.add_argument("--domain_size_pooling", type=int, default=True, help="Use DSP SIFT features.")
    parser.add_argument("--data", action="store_true", help="Copy and rename images. Extract masks.")
    parser.add_argument("--sparse", action="store_true", help="Perform sparse reconstruction.")
    parser.add_argument("--dense", action="store_true", help="Perform dense reconstruction.")
    parser.add_argument("--mesh", action="store_true",
                        help="Run Poisson surface reconstruction on the dense MVS output.")
    parser.add_argument("--fast_depth", type=int, default=True,
                        help="Perform fast but potentially less accurate depth estimation.")
    parser.add_argument("--mask_depth", type=int, default=True, help="Mask estimated depth maps prior to fusion.")
    parser.add_argument("--masked", action="store_true", help="Produce masked images.")
    parser.add_argument("--width", type=int, default=4272, help="Image width.")
    parser.add_argument("--height", type=int, default=2848, help="Image height.")
    parser.add_argument("--update_cameras", type=int, default=True,
                        help="Updates camera parameters from calibration data file.")
    parser.add_argument("--show_bbox", action="store_true",
                        help="Show image cut by bounding box defined by the initial mask and the padding.")
    parser.add_argument("--multiple_models", action="store_true",
                        help="Allow reconstruction of multiple partial models.")
    parser.add_argument("--camera_ids", nargs='*', type=int, default=[1, 2, 3],
                        help="Cameras to include in reconstruction.")
    parser.add_argument("--show_sparse", action="store_true", help="Visualize sparse model.")
    args = parser.parse_args()

    main(args)
