import copy
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from easy_o3d.utils import draw_geometries, eval_data, eval_transformation_data
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from trimesh import Trimesh

from utils import setup_logger, subsample_indices
from utils.src.voxelizer import Voxelizer

logger = setup_logger(__name__)

try:
    from libs import check_mesh_contains
except ImportError:
    logger.warning("Could not import 'check_mesh_contains' from the 'libs' submodule")
    check_mesh_contains = None


def unproject_kinect_depth(
    depth_raw: np.ndarray,
    depth_scale: float = 1000.0,
    depth_trunc: float | None = None,
    invalid_depth_value: int = 2047,
    fx: float = 582.6989,
    fy: float = 582.6989,
    cx: float = 320.7906,
    cy: float = 245.2647,
    baseline: float = 74.3428,
    ir_depth_offset: float = -4.5,
    disparity_offset: float = 1081.5151,
    disparity_precision: float = 1 / 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject raw Kinect depth data into 3D point coordinates along with the camera intrinsic matrix.

    This function converts a raw depth image from a Kinect sensor into a 3D point cloud by applying
    a series of calibration transforms. It accounts for sensor-specific corrections such as disparity accuracy,
    baseline separation, and IR image center offsets.

    Parameters:
        depth_raw (np.ndarray): The raw depth image captured by the Kinect sensor.
        depth_scale (float, optional): Scaling factor to convert depth values from sensor units
            to real-world units (typically millimeters to meters). Default is 1000.0.
        depth_trunc (float, optional): Maximum allowable depth value; any depth exceeding this value
            will be truncated. If None, no truncation is applied.
        invalid_depth_value (int, optional): Marker value in the raw depth image that indicates an invalid depth.
            Depth entries equal to this value are ignored. Default is 2047.
        fx (float, optional): Focal length along the x-axis in pixels for the depth camera. Default is 582.6989.
        fy (float, optional): Focal length along the y-axis in pixels for the depth camera. Default is 582.6989.
        cx (float, optional): x-coordinate of the principal point (optical center) of the sensor. Default is 320.7906.
        cy (float, optional): y-coordinate of the principal point (optical center) of the sensor. Default is 245.2647.
        baseline (float, optional): The baseline, representing the physical distance between the infrared (IR) sensor
            and the projector (or between two IR sensors). Used in the triangulation computation. Default is 74.3428.
        ir_depth_offset (float, optional): A correction factor to account for misalignment between the IR sensor and
            the depth image. It adjusts the pixel coordinate when mapping to 3D world coordinates. Default is -4.5.
        disparity_offset (float, optional): Disparity offset used in Kinect sensor calibration. It is subtracted
            from the raw depth value before computing disparity. Default is 1081.5151.
        disparity_precision (float, optional): The factor representing the Kinect's disparity precision (typically 1/8 pixel).
            It scales the difference between the calibrated offset and the raw depth value to obtain the disparity value.
            Default is 1/8.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points: A (N, 3) array of unprojected 3D points in the camera coordinate system.
            - intrinsic: The 3x3 camera intrinsic matrix.
    """

    d = disparity_precision * (disparity_offset - depth_raw)
    depth = baseline * fx / d

    depth[(depth < 0) | (depth_raw == invalid_depth_value)] = 0
    z = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)
    v, u = np.nonzero(z)
    z = z[v, u] / depth_scale
    if depth_trunc is not None:
        mask = z <= depth_trunc
        u = u[mask]
        v = v[mask]
        z = z[mask]

    x = (u - ir_depth_offset + 0.5 - cx) * z / fx
    y = (v - ir_depth_offset + 0.5 - cy) * z / fy
    points = np.column_stack([x, y, z])

    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return points, intrinsic


def remove_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.006,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    outlier_neighbors: int = 50,
    outlier_radius: float = 0.1,
    outlier_std: float = 10,
    cluster: bool = False,
    cluster_eps: float = 0.075,
    cluster_min_points: int = 1000,
    num_cluster: int = 1,
    crop: bool = True,
    crop_scale: float = 1.0,
    crop_up_axis: int = 1,
    show: bool = False,
) -> tuple[list[o3d.geometry.PointCloud], np.ndarray]:
    start = time.perf_counter()
    in_pcd = copy.deepcopy(pcd)
    logger.debug(f"Attempting plane removal on {in_pcd}")
    restart = time.perf_counter()
    plane_model, plane_indices = in_pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations
    )
    plane_pcd = in_pcd.select_by_index(plane_indices)
    not_plane_pcd = in_pcd.select_by_index(plane_indices, invert=True)
    logger.debug(f"Plane segmentation took {time.perf_counter() - restart:.2f}s")
    logger.debug(f"Plane: {plane_pcd}")
    logger.debug(f"Not plane: {not_plane_pcd}")

    points = np.asarray(not_plane_pcd.points)
    above_plane_mask = points @ np.array(plane_model[:3]) + plane_model[3] > 2 * distance_threshold
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[above_plane_mask]))
    above_plane_pcd = copy.deepcopy(pcd)
    logger.debug(f"Above plane: {above_plane_pcd}")

    pcds = [pcd]
    if cluster:
        restart = time.perf_counter()
        labels = np.asarray(pcd.cluster_dbscan(eps=cluster_eps, min_points=cluster_min_points))
        classes = np.unique(labels)
        num_classes = (classes >= 0).sum()
        logger.debug(f"Clustered {num_classes} classes: {classes}")
        noise_indices = np.argwhere(labels == -1)
        logger.debug(f"Noise points: {len(noise_indices)}")

        pcds = []
        cluster_pcds = []
        if num_cluster is None and num_classes:
            for c in range(num_classes):
                ci = np.argwhere(labels == c)
                logger.debug(f"Class {c} points: {len(ci)}")
                p = copy.deepcopy(pcd).select_by_index(ci)
                pcds.append(p)
                cluster_pcds.append(copy.deepcopy(p))
        else:
            pcds = [copy.deepcopy(pcd).select_by_index(noise_indices, invert=num_classes > 0)]
            cluster_pcds = [copy.deepcopy(pcds[0])]
        logger.debug(f"Clustering took {time.perf_counter() - restart:.2f}s")
    if outlier_neighbors and (outlier_std or outlier_radius):
        restart = time.perf_counter()
        outlier_indices = list()
        for i, p in enumerate(pcds):
            ois = list()
            if outlier_std:
                p, oi = p.remove_statistical_outlier(nb_neighbors=outlier_neighbors, std_ratio=outlier_std)
                ois.extend(oi)
            if outlier_radius:
                p, oi = p.remove_radius_outlier(nb_points=outlier_neighbors, radius=outlier_radius)
                ois.extend(oi)
            outlier_indices.append(ois)
            pcds[i] = p
        logger.debug(f"Outlier removal took {time.perf_counter() - restart:.2f}s")
    if crop:
        restart = time.perf_counter()
        for i, p in enumerate(pcds):
            points = np.asarray(p.points)
            hull_points = points.copy()
            if crop_scale != 1:
                hull_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hull_points))
                hull_points = np.asarray(hull_pcd.scale(crop_scale, hull_pcd.get_center()).points)
            hull_points[:, crop_up_axis] -= 2 * distance_threshold

            convex_hull_start = time.perf_counter()
            logger.debug(f"Convex hull took {time.perf_counter() - convex_hull_start:.3f}s")

            convex_hull_start = time.perf_counter()
            in_pcd_points = np.asarray(in_pcd.points)
            if check_mesh_contains is None:
                in_hull_mask = Delaunay(hull_points).find_simplex(in_pcd_points) >= 0
            else:
                hull_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hull_points))
                hull_mesh, _ = hull_pcd.compute_convex_hull()
                hull_mesh = Trimesh(vertices=np.asarray(hull_mesh.vertices), faces=np.asarray(hull_mesh.triangles))
                in_hull_mask = check_mesh_contains(hull_mesh, in_pcd_points)
            in_hull_points = in_pcd_points[in_hull_mask]
            logger.debug(f"Inside hull check took {time.perf_counter() - convex_hull_start:.2f}s")

            logger.debug(f"In hull: {len(in_hull_points)}")
            logger.debug(f"Before crop: {len(points)}")
            points = np.unique(np.concatenate([in_hull_points, points], axis=0), axis=0)
            logger.debug(f"After crop: {len(points)}")
            pcds[i] = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        logger.debug(f"Crop took {time.perf_counter() - restart:.2f}s")
    if show:
        plane_pcd.paint_uniform_color([0.8, 0.8, 0.8])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries = [plane_pcd, frame]

        if cluster:
            noise_pcd = above_plane_pcd.select_by_index(noise_indices)
            noise_pcd.paint_uniform_color([0, 0, 1])
            geometries.append(noise_pcd)

        for index, p in enumerate(pcds):
            center = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            center.paint_uniform_color([1, 0, 0])
            center.translate((p.get_max_bound() + p.get_min_bound()) / 2, relative=False)
            geometries.extend([p.paint_uniform_color(np.random.uniform(size=3)), p.get_oriented_bounding_box(), center])
            if outlier_neighbors and (outlier_std or outlier_radius):
                if cluster:
                    outlier_pcd = cluster_pcds[index].select_by_index(outlier_indices[index], invert=True)
                else:
                    outlier_pcd = above_plane_pcd.select_by_index(outlier_indices[index], invert=True)
                if outlier_pcd.has_points():
                    outlier_pcd.paint_uniform_color([0, 1, 0])
                    geometries.append(outlier_pcd)
            if crop:
                hull_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hull_points))
                hull_mesh, _ = hull_pcd.compute_convex_hull()
                geometries.append(o3d.geometry.LineSet.create_from_triangle_mesh(hull_mesh))
        draw_geometries(geometries, window_name="Plane Removal")
    logger.debug(f"Plane removal took {time.perf_counter() - start:.2f}s")
    return pcds, plane_model


def get_point_cloud(
    in_path: Path,
    depth_scale: float = 1000.0,
    depth_trunc: float = 1.1,
    intrinsic: str | np.ndarray | None = None,
    extrinsic: str | np.ndarray | None = None,
    pcd_crop: list[float] | None = None,
    show: bool = False,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    intrinsic_arr: np.ndarray
    if intrinsic is None:
        f_x = f_y = 582.6989
        c_x, c_y = 320.7906, 245.2647
        intrinsic_arr = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    elif isinstance(intrinsic, str):
        intrinsic_arr = np.asarray(np.load(intrinsic))
    else:
        intrinsic_arr = intrinsic

    logger.debug(f"Attempting to load point cloud from {in_path}")
    if in_path.suffix in [".ply", ".obj", ".off"]:
        pcd = eval_data(str(in_path))
    else:
        if in_path.suffix in [".npy", ".npz"]:
            logger.debug("Assuming Microsoft Kinect camara intrinsic parameters.")
            depth = np.load(str(in_path))
            points, intrinsic_arr = unproject_kinect_depth(depth, depth_scale, depth_trunc)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        elif in_path.suffix in [".png", ".jpg", ".exr"]:
            if in_path.suffix == ".exr":
                try:
                    import cv2

                    depth = cv2.imread(str(in_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                except ImportError as e:
                    logger.error("OpenCV is not installed. Please install it to read EXR files.")
                    raise e
            else:
                depth = o3d.io.read_image(str(in_path)).numpy()
            pcd = eval_data(data=depth, camera_intrinsic=intrinsic_arr, depth_scale=depth_scale, depth_trunc=depth_trunc)
        else:
            raise ValueError(f"Unsupported file format: {in_path.suffix}")

    extrinsic_arr = np.eye(4)
    if extrinsic is not None:
        extrinsic_data: str | np.ndarray = extrinsic
        if isinstance(extrinsic_data, str) and (extrinsic_data.endswith(".npy") or extrinsic_data.endswith(".npz")):
            extrinsic_data = np.load(extrinsic_data)
        extrinsic_mat = np.asarray(eval_transformation_data(extrinsic_data))
        """
        z, y, x = R.from_matrix(extrinsic[:3, :3].T).as_euler("zyx", degrees=True)
        logger.debug(f"Camera extrinsic rotation: z={z}, y={y}, x={x}")
        rot_z = R.from_euler("z", z, degrees=True).as_matrix().T
        rot_y = R.from_euler("y", y, degrees=True).as_matrix()
        rot_x = R.from_euler("x", x + 90, degrees=True).as_matrix().T
        pcd.rotate(rot_x @ rot_y @ rot_z, center=np.zeros(3))
        """

        rot_x, rot_y, rot_z = get_rot_from_extrinsic(extrinsic_mat)
        pcd.rotate(rot_x @ rot_y @ rot_z, center=np.zeros(3))
        extrinsic_arr[:3, :3] = (rot_x @ rot_y).T
    if pcd_crop is not None:
        loc = (pcd.get_min_bound() + pcd.get_max_bound()) / 2
        pcd.translate(-loc)
        points = np.asarray(pcd.points)
        points = points[(points[:, 0] > pcd_crop[0]) & (points[:, 0] < pcd_crop[3])]
        points = points[(points[:, 1] > pcd_crop[1]) & (points[:, 1] < pcd_crop[4])]
        points = points[(points[:, 2] > pcd_crop[2]) & (points[:, 2] < pcd_crop[5])]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.translate(loc)
    logger.debug(f"Loaded {pcd}")
    if show:
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
        draw_geometries([pcd, frame], window_name="Input Point Cloud")
    return pcd, intrinsic_arr, extrinsic_arr


def get_rot_from_extrinsic(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = R.from_matrix(extrinsic[:3, :3]).as_euler("xyz", degrees=True)
    logger.debug(f"Camera extrinsic rotation: z={z}, y={y}, x={x}")
    rot_z = R.from_euler("z", z, degrees=True).as_matrix()
    rot_y = R.from_euler("y", y, degrees=True).as_matrix().T
    rot_x = R.from_euler("x", x - 90, degrees=True).as_matrix()
    return rot_x, rot_y, rot_z


def get_input_data_from_point_cloud(
    pcd: o3d.geometry.PointCloud,
    transform: np.ndarray | None = None,
    num_input_points: int | None = None,
    noise_std: float | None = None,
    rotate_z: float | None = None,
    offset_y: float = 0,
    center: bool = False,
    scale: bool | float | None = None,
    crop: float | None = 0.55,
    voxelize: int | None = None,
    padding: float = 0.1,
    show: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    if rotate_z is not None:
        pcd.rotate(R.from_euler("z", rotate_z, degrees=False).as_matrix(), center=(0, 0, 0))

    loc = np.zeros(3)
    if center:
        loc = pcd.get_min_bound() + pcd.get_max_bound()
    loc[1] -= offset_y
    loc /= 2
    if center:
        pcd.translate(-loc)

    scale_value = 1.0
    if isinstance(scale, float):
        scale_value = scale
        pcd.scale(scale_value, center=(0, 0, 0))
    elif scale:
        scale_extent = pcd.get_max_bound() - pcd.get_min_bound()
        scale_extent[1] += offset_y
        scale_value = float(np.max(scale_extent))
        pcd.scale(1 / scale_value, center=(0, 0, 0))

    if transform is not None:
        trafo = eval_transformation_data(transform)
        pcd.transform(trafo)

    if crop is not None:
        uncropped_pcd = copy.deepcopy(pcd)
        crop_box = o3d.geometry.AxisAlignedBoundingBox((-crop,) * 3, (crop,) * 3)
        pcd = pcd.crop(crop_box)

    points = np.asarray(pcd.points)

    if num_input_points:
        indices = subsample_indices(points, num_input_points)
        points = points[indices]

    if noise_std is not None and noise_std > 0:
        noise = noise_std * np.random.randn(*points.shape)
        points = points + noise

    if voxelize is not None:
        voxelizer = Voxelizer(resolution=voxelize, padding=padding, method="kdtree")
        points, indices = voxelizer(points)

    if show:
        inputs = points
        if voxelize is not None:
            inputs = voxelizer.grid_points[indices]
        geometries = [
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs)),
            o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5),
        ]
        if crop:
            geometries.extend([crop_box, uncropped_pcd.paint_uniform_color([0, 0, 1])])
        draw_geometries(geometries, window_name="Input Data")

    return points.astype(np.float32), loc.astype(np.float32), float(scale_value)
