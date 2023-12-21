import copy
import gc
import json
import os
from pathlib import Path
from typing import Union, Tuple, List, Dict, Any, Optional

import binvox
import h5py
import numpy as np
import open3d as o3d
import pyrender
import trimesh
from trimesh import Trimesh
from PIL import Image
from easy_o3d.registration import IterativeClosestPoint
from easy_o3d.utils import (convert_rgbd_image_to_point_cloud,
                            convert_depth_image_to_point_cloud,
                            get_camera_parameters_from_blenderproc_bopwriter)
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from skimage.util.shape import view_as_windows

from libs import check_mesh_contains
from utils import inv_trafo, setup_logger, load_mesh, generate_random_basis, load_from_binary_hdf5
from .transforms import apply_transform

logger = setup_logger(__name__)


def _load_mesh(file_path: Union[str, Path],
               normalize: bool = False,
               return_values: bool = False,
               pose_file: str = "pose.npy",
               transform: np.ndarray = None) -> Union[Trimesh, Tuple[Trimesh, float, float]]:
    mesh = Trimesh(*load_mesh(file_path), process=False, validate=False)
    pose_path = os.path.join(os.path.dirname(file_path), pose_file)
    if os.path.isfile(pose_path):
        mesh = mesh.apply_transform(np.load(pose_path))
    if transform is not None:
        mesh = mesh.apply_transform(transform)
    return _normalize_mesh(mesh, return_values) if normalize else mesh


def _normalize_mesh(mesh: Trimesh,
                    return_values: bool = False) -> Union[Trimesh, Tuple[Trimesh, float, float]]:
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

    mesh.apply_translation(-centers)
    mesh.apply_scale(1 / total_size)

    return (mesh, centers, total_size) if return_values else mesh


def _load_pointcloud(obj_dir: Union[str, Path],
                     file: str,
                     normals_file: Optional[str] = None,
                     load_normals: bool = False,
                     normalize_normals: bool = False,
                     pose_file: str = "pose.npy",
                     from_hdf5: bool = False,
                     transform: np.ndarray = None) -> Union[Tuple[np.ndarray, None], Tuple[np.ndarray, np.ndarray]]:
    normals = None
    file_path = Path(obj_dir) / file
    if from_hdf5:
        pointcloud_dict = load_from_binary_hdf5(file_path.parent.parent,
                                                [file_path.name],
                                                [file_path.parent.name])[0]
        points = pointcloud_dict["points"]
        if load_normals:
            normals = pointcloud_dict["normals"]
    else:
        if file.lower().endswith(".npz"):
            pointcloud_dict = np.load(file_path)
            points = pointcloud_dict["points"]
            if load_normals:
                normals = pointcloud_dict["normals"]
        elif file.lower().endswith(".npy"):
            points = np.load(file_path)
            if load_normals and normals_file:
                normals_path = os.path.join(obj_dir, normals_file)
                normals = np.load(normals_path)
        elif file.lower().endswith(".obj") or file.lower().endswith(".off") or file.lower().endswith(".ply"):
            mesh = _load_mesh(file_path, pose_file=pose_file)
            points, indices = mesh.sample(int(1e5), return_index=True)
            if load_normals:
                normals = mesh.face_normals[indices]
        else:
            raise ValueError(f"Unknown file type: {file_path}")
    if transform is not None:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if load_normals:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd = pcd.transform(transform)
        points = np.asarray(pcd.points)
        if load_normals:
            normals = np.asarray(pcd.normals)
    if normalize_normals and normals is not None:
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def _filter_discontinuities(depth: np.ndarray,
                            filter_size: int = 7,
                            threshold: int = 1000) -> np.ndarray:
    assert filter_size % 2 == 1

    offset = (filter_size - 1) // 2
    patches = view_as_windows(depth, (filter_size, filter_size))
    mids = patches[:, :, offset, offset]
    mins = np.min(patches, axis=(2, 3))
    maxes = np.max(patches, axis=(2, 3))

    discont = np.maximum(np.abs(mins - mids), np.abs(maxes - mids))
    mark = discont > threshold

    final_mark = np.zeros_like(depth)
    final_mark[offset:offset + mark.shape[0], offset:offset + mark.shape[1]] = mark

    return depth * (1 - final_mark)


def _load_cam(index: int, cam_index: Union[Tuple[int, int], int], num_cams: int, path: str):
    if isinstance(cam_index, (tuple, list)):
        image_index = index + cam_index[0] - index // cam_index[1] * cam_index[1]
    elif isinstance(cam_index, int):
        if cam_index == -1:
            image_index = np.random.randint(num_cams)
        else:
            image_index = cam_index
    else:
        raise TypeError

    camera_dict = np.load(path)
    trafo = camera_dict[f"world_mat_{image_index}"]
    rot = trafo[:3, :3]
    x_angle = -np.arctan2(rot.T[2, 1], rot.T[2, 2])
    x_angle = np.rad2deg(x_angle - np.pi if x_angle > 0 else x_angle)  # Todo: why?
    K = camera_dict[f"camera_mat_{image_index}"]

    return {"rot": rot,
            "x_angle": x_angle,
            "cam": trafo[:3, 3],
            "K": K,
            "index": image_index}


class Field:
    def __init__(self):
        self.name = self.__class__.__name__

    def load(self, obj_dir: str, index: int, category: int) -> Dict[Any, np.ndarray]:
        raise NotImplementedError

    def clear_cache(self):
        if hasattr(self, "cache"):
            if isinstance(self.cache, dict):
                self.cache.clear()
            del self.cache
            gc.collect()


class EmptyField(Field):
    def load(self, obj_dir: str, index: int, category: int) -> Dict[Any, np.ndarray]:
        return {}


class ImageField(Field):
    def __init__(self,
                 data_dir: str = "img_choy2016",
                 extension: str = "jpg",
                 index: Union[Tuple[int, int], int] = -1,
                 num_images: int = 24,
                 transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.extension = extension
        self.transform = transform
        self.index = index
        self.num_images = num_images
        if isinstance(index, int):
            assert index <= self.num_images

    def load(self, obj_dir, index, category):
        object_dir = os.path.join(obj_dir, self.data_dir)
        data = _load_cam(index, self.index, self.num_images, os.path.join(object_dir, "cameras.npz"))

        filename = os.path.join(object_dir, str(data["index"]).zfill(3) + '.' + self.extension)
        image = Image.open(filename).convert("RGB")

        image = apply_transform({"image": image}, self.transform)

        data[None] = image
        return data


class BOPField(Field):
    def __init__(self,
                 file_ids: List[int],
                 camera: Optional[str] = None,
                 max_outlier_std: float = 0,
                 max_correspondence_distance: float = 0,
                 transform=None):
        super().__init__()

        self.camera_file = "camera.json" if camera is None else f"camera_{camera}.json"
        self.file_ids = file_ids
        self.transform = transform
        self.max_outlier_std = max_outlier_std
        if max_correspondence_distance:
            self.icp = IterativeClosestPoint(max_correspondence_distance=max_correspondence_distance,
                                             max_iteration=30,
                                             relative_fitness=1e-6,
                                             relative_rmse=1e-6)
        else:
            self.icp = None

    def load(self, obj_dir, index, category):
        skip = False

        obj_id = category
        obj_name = str(category).zfill(6)
        file_id = self.file_ids[index]

        data_dir = '/'.join(obj_dir.split('/')[:-2])
        path_to_scene_camera_json = os.path.join(obj_dir, "scene_camera.json")
        path_to_scene_gt_json = os.path.join(obj_dir, "scene_gt.json")
        path_to_scene_gt_info_json = os.path.join(obj_dir, "scene_gt_info.json")

        with open(path_to_scene_camera_json) as f:
            camera_data = json.load(f)
        intrinsic = np.asarray(camera_data[str(file_id)]["cam_K"]).reshape(3, 3)
        depth_scale = camera_data[str(file_id)]["depth_scale"]

        with open(path_to_scene_gt_json) as f:
            gt_poses = json.load(f)
        for mask_id, pose_data in enumerate(gt_poses[str(file_id)]):
            pose = np.eye(4)
            pose[:3, :3] = np.asarray(pose_data["cam_R_m2c"]).reshape(3, 3)
            pose[:3, 3] = np.asarray(pose_data["cam_t_m2c"])
            if pose_data["obj_id"] == obj_id:
                break
        up = pose[:3, 2][1] < 0 and pose[:3, 2][2] < 0
        if not up:
            skip = True

        with open(path_to_scene_gt_info_json) as f:
            info = json.load(f)
        info = info[str(file_id)][mask_id]
        visib_fract = info["visib_fract"]

        if not visib_fract > 0.85:
            skip = True

        depth_path = os.path.join(obj_dir, "depth", f"{file_id:06d}.png")
        mask_path = os.path.join(obj_dir, "mask", f"{file_id:06d}_{mask_id:06d}.png")

        depth = np.asarray(o3d.io.read_image(depth_path))
        mask = np.asarray(o3d.io.read_image(mask_path)).astype(bool)
        masked_depth = depth * mask
        height, width = depth.shape

        # plt.imshow(masked_depth, cmap="gray")
        # plt.show()

        pcd = convert_depth_image_to_point_cloud(masked_depth, intrinsic, pose, depth_scale=1 / depth_scale)
        pcd.scale(0.001, center=(0, 0, 0))

        if not len(np.asarray(pcd.points)):
            skip = True

        inv_pose = inv_trafo(pose)
        pose[:3, 3] /= 1000
        inv_pose[:3, 3] /= 1000

        z_proj = (pose[:3, :3].T @ np.array([0, 0, 1]).T).T
        z_proj[2] = 0
        z_proj = z_proj / np.linalg.norm(z_proj)
        z_angle = np.arctan2(np.linalg.det([z_proj[:2], np.array([0, 1])]), np.dot(z_proj[:2], np.array([0, 1])))
        rot_to_cam = R.from_euler('z', z_angle).as_matrix()

        mesh = o3d.io.read_triangle_mesh(os.path.join(data_dir, "models_eval", f"obj_{obj_name}.ply"))
        mesh.scale(0.001, center=(0, 0, 0))

        render_intrinsic = intrinsic.copy()
        render_intrinsic[0, 2] = width / 2
        render_intrinsic[1, 2] = height / 2

        if self.icp is not None and not skip:
            pyrender_mesh = Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)

            pyrender_extrinsic = np.eye(4)
            rot_x_180 = R.from_euler('x', 180, degrees=True).as_matrix()
            pyrender_extrinsic[:3, :3] = rot_x_180 @ pose[:3, :3]
            pyrender_extrinsic[:3, 3] = pose[:3, 3] @ rot_x_180
            pyrender_extrinsic = inv_trafo(pyrender_extrinsic)

            camera = pyrender.IntrinsicsCamera(render_intrinsic[0, 0],
                                               render_intrinsic[1, 1],
                                               render_intrinsic[0, 2],
                                               render_intrinsic[1, 2],
                                               znear=0.01,
                                               zfar=10)

            scene = pyrender.Scene()
            scene.add(pyrender.Mesh.from_trimesh(pyrender_mesh, smooth=False))
            scene.add(camera, pose=pyrender_extrinsic)

            renderer = pyrender.OffscreenRenderer(width, height)
            mesh_depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            mesh_depth[mesh_depth >= 10] = 0

            renderer.delete()

            """
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=width, height=height, visible=False)

            vis.add_geometry(mesh)

            ctr = vis.get_view_control()
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
            param.intrinsic.intrinsic_matrix = render_intrinsic
            param.extrinsic = pose
            ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

            mesh_depth = vis.capture_depth_float_buffer(do_render=True)
            vis.destroy_window()
            vis.close()
            
            plt.imshow(np.asarray(mesh_depth), cmap="gray")
            plt.show()
            """

            mesh_pcd = convert_depth_image_to_point_cloud(mesh_depth,
                                                          render_intrinsic,
                                                          pose,
                                                          depth_scale=1,
                                                          depth_trunc=10)

            fisher1 = o3d.pipelines.registration.get_information_matrix_from_point_clouds(mesh_pcd,
                                                                                          pcd,
                                                                                          self.icp.max_correspondence_distance,
                                                                                          np.eye(4))

            result = self.icp.run(mesh_pcd, pcd, draw=False)

            fisher2 = o3d.pipelines.registration.get_information_matrix_from_point_clouds(mesh_pcd,
                                                                                          pcd,
                                                                                          self.icp.max_correspondence_distance,
                                                                                          result.transformation)
            trafo = result.transformation.copy() if fisher2.mean() > fisher1.mean() else np.eye(4)
            """
            z_new = trafo[:3, 2]
            yz = np.array([z_new[1], z_new[2]])
            yz /= np.linalg.norm(yz)
            x_angle = np.rad2deg(np.math.atan2(np.linalg.det([yz, np.array([0, 1])]), np.dot(yz, np.array([0, 1]))))
            rot_x = R.from_euler('x', x_angle, degrees=True).as_matrix()

            xz = np.array([z_new[0], z_new[2]])
            xz /= np.linalg.norm(xz)
            y_angle = np.rad2deg(np.math.atan2(np.linalg.det([xz, np.array([0, 1])]), np.dot(xz, np.array([0, 1]))))
            rot_y = R.from_euler('y', y_angle, degrees=True).as_matrix()

            pcd.rotate(rot_x.T, center=(0, 0, 0))
            pcd.rotate(rot_y.T, center=(0, 0, 0))
            trafo[:3, :3] = rot_y.T @ rot_x.T @ trafo[:3, :3]

            mesh.compute_vertex_normals()
            pcd.paint_uniform_color((1, 0, 0))
            mesh_pcd.paint_uniform_color(np.zeros(3))
            o3d.visualization.draw_geometries([pcd,
                                               mesh_pcd.transform(trafo),
                                               mesh,
                                               o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1),
                                               # o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1).transform(inv_pose)
                                               ])
            """
        else:
            trafo = np.eye(4)

        if self.max_outlier_std and not skip:
            mesh.transform(trafo)
            mesh.scale(1.1, center=(0, 0, 0))
            hull = Delaunay(np.asarray(mesh.vertices))
            mask = hull.find_simplex(np.asarray(pcd.points)) >= 0
            pcd = pcd.select_by_index(np.argwhere(mask))

            nb_neighbors = max(len(np.asarray(pcd.points)) // 100, 10)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=self.max_outlier_std)

        if not skip:
            rot = np.eye(3)
            rot_z = np.eye(3)
            if data_dir.split('/')[-1] == "hb" and obj_id == 6:
                rot_x = R.from_euler('x', -1.44, degrees=True).as_matrix()
                rot_y = R.from_euler('y', -1.99, degrees=True).as_matrix()
                rot_z = R.from_euler('z', 187.14, degrees=True).as_matrix()
                rot = rot_z @ rot_y @ rot_x
            elif data_dir.split('/')[-1] == "tyol":
                if obj_id == 3:
                    rot_x = R.from_euler('x', -0.89, degrees=True).as_matrix()
                    rot_y = R.from_euler('y', 2.11, degrees=True).as_matrix()
                    rot_z = R.from_euler('z', 180, degrees=True).as_matrix()
                elif obj_id == 4:
                    rot_x = R.from_euler('x', 0.54, degrees=True).as_matrix()
                    rot_y = R.from_euler('y', 3, degrees=True).as_matrix()
                    rot_z = R.from_euler('z', 189, degrees=True).as_matrix()
                elif obj_id == 5:
                    rot_x = R.from_euler('x', -2.52, degrees=True).as_matrix()
                    rot_y = R.from_euler('y', 0.351, degrees=True).as_matrix()
                    rot_z = R.from_euler('z', 182, degrees=True).as_matrix()
                elif obj_id == 6:
                    rot_x = R.from_euler('x', 3.43, degrees=True).as_matrix()
                    rot_y = R.from_euler('y', 1.2, degrees=True).as_matrix()
                    rot_z = R.from_euler('z', 182, degrees=True).as_matrix()
                elif obj_id == 20:
                    rot_x = R.from_euler('x', 8.27, degrees=True).as_matrix()
                    rot_y = R.from_euler('y', 4.47, degrees=True).as_matrix()
                    rot_z = R.from_euler('z', 179, degrees=True).as_matrix()
                elif obj_id == 21:
                    rot_x = R.from_euler('x', -5.1, degrees=True).as_matrix()
                    rot_y = R.from_euler('y', 0.15, degrees=True).as_matrix()
                    rot_z = R.from_euler('z', 184, degrees=True).as_matrix()
                else:
                    raise ValueError(f"Object id {obj_id} not supported.")
                rot = rot_z @ rot_y @ rot_x
            elif data_dir.split('/')[-1] == "ycbv" and obj_id == 14:
                rot_z = R.from_euler('z', 3.12, degrees=True).as_matrix()
                rot = rot_z

            iv_trafo = inv_trafo(trafo)
            inv_pose[:3, :3] = inv_pose[:3, :3] @ iv_trafo[:3, :3].T
            inv_pose[:3, 3] += iv_trafo[:3, 3]

            # mesh.scale(1 / 1.1, center=(0, 0, 0))
            # mesh.transform(iv_trafo)

            pcd.transform(iv_trafo)
            trafo = np.eye(4)
            trafo[:3, :3] = rot
            pcd.rotate(rot, center=(0, 0, 0))
            # mesh.rotate(rot, center=(0, 0, 0))

            pose = inv_trafo(inv_pose)
            pose[:3, :3] = pose[:3, :3] @ rot.T
            inv_pose = inv_trafo(pose)

            rot_to_cam = rot_to_cam @ rot_z.T

            """
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh,
                                               pcd,
                                               o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1),
                                               o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1).transform(inv_pose)
                                               ])
            """

        points = np.asarray(pcd.points)
        data = {None: points,
                "intrinsic": render_intrinsic,
                "extrinsic": pose,
                "width": width,
                "height": height,
                "cam": inv_pose[:3, 3],
                "rot": rot_to_cam,
                "x_angle": 0,
                "visib_fract": visib_fract,
                "up": up,
                "skip": skip or len(points) == 0,
                "pose": trafo,
                "name": obj_name,
                "path": depth_path}
        if pcd.has_normals():
            data["normals"] = np.asarray(pcd.normals)

        data = apply_transform(data, self.transform)

        return data


class DepthField(Field):
    def __init__(self,
                 data_dir: Optional[str] = None,
                 unscale: bool = True,
                 unrotate: bool = True,
                 load_normals: bool = True,
                 kinect: bool = False,
                 path_suffix: Optional[str] = None,
                 num_objects: int = 1,
                 num_files: int = 100,
                 file_offset: int = 0,
                 random_file: bool = False,
                 file_weights: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                 precision: int = 16,
                 simulate_depth_noise: bool = False,
                 cache: bool = False,
                 from_hdf5: bool = False,
                 transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.unscale = unscale
        self.unrotate = unrotate
        self.load_normals = load_normals
        self.kinect = kinect
        self.path_suffix = "" if path_suffix is None else path_suffix
        self.num_objects = num_objects
        self.num_files = num_files
        self.file_offset = file_offset
        self.random_file = random_file
        self.file_weights = file_weights
        self.precision = precision
        self.cache = dict() if cache else None
        self.from_hdf5 = from_hdf5
        self.transform = transform

        self.simulator = None
        if simulate_depth_noise:
            self.simulator = o3d.t.io.DepthNoiseSimulator(o3d.data.RedwoodIndoorLivingRoom1().noise_model_path)

    def get_file(self, index: int) -> int:
        epoch = index // self.num_objects
        return epoch - epoch // self.num_files + self.file_offset

    def get_depth_path(self, index: int, obj_dir: Path, file: Optional[int] = None) -> Path:
        if file is None:
            file = self.get_file(index)
        return obj_dir / Path('kinect' if self.kinect else 'depth') / f"{str(file).zfill(5)}.png"

    def get_normal_path(self, index: int, obj_dir: Path, file: Optional[int] = None) -> Path:
        if file is None:
            file = self.get_file(index)
        return obj_dir / "normal" / f"{str(file).zfill(5)}.jpg"

    @staticmethod
    def get_opencv_inv_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
        opengl_extrinsic = extrinsic.copy()
        opengl_extrinsic[1, :] *= -1
        opengl_extrinsic[2, :] *= -1
        return inv_trafo(opengl_extrinsic)

    @staticmethod
    def get_x_angle_from_rot(rot: np.ndarray) -> float:
        rot = rot.T
        # Calculate the angle of rotation around the x-axis
        x_angle = np.degrees(np.arctan2(-rot[2, 1], rot[2, 2]))

        # Adjusting angle to be in the range [0, 180] union [-180, 0]
        if x_angle < -90:
            x_angle += 180
        elif x_angle > 90:
            x_angle -= 180

        # x = np.arctan2(rot[2, 1], rot[2, 2])
        # x = -np.rad2deg(x - np.pi if x > 0 else x)  # resolve ambiguity

        return x_angle

    def load(self, obj_dir: str, index: int, category: int) -> Dict[Any, np.ndarray]:
        obj_dir = Path(obj_dir)
        obj_name = obj_dir.name
        obj_category = obj_dir.parent.name
        obj_dir = obj_dir / self.path_suffix

        if self.data_dir:
            obj_dir = Path(self.data_dir) / obj_category / obj_name / self.path_suffix

        if self.random_file and self.num_files > 1:
            if self.file_weights and obj_category in self.file_weights and obj_name in self.file_weights[obj_category]:
                weights = self.file_weights[obj_category][obj_name][:self.num_files]
                file = np.random.choice(len(weights), p=weights)
                if self.cache:
                    """
                    easy = (Path("out/automatica_2023/easy") / obj_category / obj_name).with_suffix(".png").expanduser().resolve()
                    hard = (Path("out/automatica_2023/hard") / obj_category / obj_name).with_suffix(".png").expanduser().resolve()
                    easy.parent.mkdir(parents=True, exist_ok=True)
                    hard.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.get_depth_path(index, obj_dir, np.argmin(weights)), easy)
                    shutil.copyfile(self.get_depth_path(index, obj_dir, np.argmax(weights)), hard)
                    """
                    logger.debug(f"Chose file {file} for {obj_category}/{obj_name} (p: {weights[file]:.3f})")
                    # print((100 * weights).min(), np.quantile(100 * weights, [0.1, 0.5, 0.9]), (100 * weights).max())
            else:
                file = np.random.randint(self.num_files)
        else:
            file = self.get_file(index)

        depth_path = self.get_depth_path(index, obj_dir, file)
        normal_path = self.get_normal_path(index, obj_dir, file)

        params_path = obj_dir / "parameters.npz"
        if self.from_hdf5:
            parameters = load_from_binary_hdf5(obj_dir, ["parameters.npz"])[0]
        else:
            parameters = np.load(params_path)
        logger.debug(f"Found keys {list(parameters.keys())} in {params_path}")
        max_depths_key = "kinect_max_depths" if self.kinect else "max_depths"
        depth_max = parameters[max_depths_key][file]
        scale = parameters["scales"][file]
        rotation = parameters["rotations"][file] if "rotations" in parameters else None
        intrinsic = parameters["intrinsic"]
        extrinsic = parameters["extrinsics"][file]

        data = self.cache.get(depth_path) if isinstance(self.cache, dict) else None
        if data is None:
            depth_scale = 1 / depth_max if self.precision == 8 else (2 ** 16 - 1) / depth_max
            if self.from_hdf5:
                depth, normal = load_from_binary_hdf5(depth_path.parent.parent,
                                                      [depth_path.name, normal_path.name],
                                                      [depth_path.parent.name, normal_path.parent.name])
                depth = o3d.geometry.Image(depth)
                normal = o3d.geometry.Image(normal)
            else:
                depth = o3d.io.read_image(str(depth_path))
                normal = o3d.io.read_image(str(normal_path))

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(normal,
                                                                            depth,
                                                                            depth_scale=depth_scale,
                                                                            depth_trunc=6.0,
                                                                            convert_rgb_to_intensity=False)

            depth = rgbd_image.depth
            if self.simulator is not None:
                depth = o3d.t.geometry.Image.from_legacy(depth).cuda()
                depth = self.simulator.simulate(depth, depth_scale=1.0).cpu().to_legacy()

            pcd = convert_rgbd_image_to_point_cloud([rgbd_image.color, depth],
                                                    intrinsic,
                                                    extrinsic,
                                                    depth_scale=1.0,
                                                    depth_trunc=6.0,
                                                    convert_rgb_to_intensity=False)
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.colors) * 2 - 1)

            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)

            inv_extrinsic_openvc = self.get_opencv_inv_extrinsic(extrinsic)
            rot = inv_extrinsic_openvc[:3, :3].T
            cam = inv_extrinsic_openvc[:3, 3]
            x_angle = self.get_x_angle_from_rot(rot)

            if isinstance(self.cache, dict):
                logger.debug(f"{self.name}: Caching depth data.")
                self.cache[depth_path] = {None: points.astype(np.float32),
                                          "normals": normals.astype(np.float32),
                                          "rot": rot.astype(np.float32),
                                          "cam": cam.astype(np.float32),
                                          "x_angle": x_angle}
        else:
            logger.debug(f"{self.name}: Retrieving depth data from cache.")
            points = data[None]
            normals = data["normals"]
            rot = data["rot"]
            cam = data["cam"]
            x_angle = data["x_angle"]

        data = {None: points,
                "normals": normals,
                "rot": rot,
                "cam": cam,
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
                "x_angle": x_angle,
                "name": obj_name,
                "path": depth_path,
                "file": file}
        data = copy.deepcopy(data)

        if self.unrotate and rotation is not None:
            data[None] = data[None] @ rotation
            data["normals"] = data["normals"] @ rotation
            data["rotation"] = rotation
        if self.unscale:
            data[None] /= scale
            if not isinstance(scale, np.ndarray):
                data["scale"] = np.asarray([scale, scale, scale])
            else:
                data["scale"] = scale

        data = apply_transform(data, self.transform)

        return data


class BlenderProcRGBDField(Field):
    def __init__(self,
                 data_dir: Optional[str] = None,
                 unscale: bool = True,
                 undistort: bool = True,
                 path_prefix: str = "",
                 num_objects: int = 1,
                 num_shards: int = 1,
                 files_per_shard: int = 1,
                 random_file: bool = False,
                 random_shard: bool = False,
                 depth_trunc: int = 10,
                 fuse: Union[int, Tuple[int, int]] = 0,
                 fuse_thresholds: Tuple[float] = (0.3, 1.0, 0.1),
                 input_type: Optional[str] = None,
                 swap_xy: bool = True,
                 load_fixed: bool = False,
                 cache: bool = False,
                 transform=None):
        super().__init__()

        self.data_dir = "" if data_dir is None else data_dir
        self.unscale = unscale
        self.undistort = undistort
        self.path_prefix = path_prefix
        self.num_objects = num_objects
        self.num_shards = num_shards
        self.files_per_shard = files_per_shard
        self.random_file = random_file
        self.random_shard = random_shard
        self.depth_trunc = depth_trunc
        self.fuse = fuse
        self.fuse_thresholds = fuse_thresholds
        self.input_type = input_type
        self.load_fixed = load_fixed
        self.cache = dict() if cache else None
        self.transform = transform

        if swap_xy:
            self.swap_xy = np.array([[0, 1, 0],
                                     [1, 0, 0],
                                     [0, 0, 1]])
        else:
            self.swap_xy = np.eye(3)
        self.swap_yz = np.array([[1, 0, 0],
                                 [0, 0, 1],
                                 [0, -1, 0]])

    def _get_points_from_depth(self, depth, camera_intrinsic, camera_extrinsic):
        pcd = convert_depth_image_to_point_cloud(depth,
                                                 camera_intrinsic,
                                                 camera_extrinsic,
                                                 depth_trunc=self.depth_trunc)
        pcd.rotate(self.swap_yz, center=(0, 0, 0))
        return np.asarray(pcd.points)

    @staticmethod
    def get_x_angle_from_rot(rot: np.ndarray) -> float:
        x_angle = np.arctan2(rot.T[2, 1], rot.T[2, 2])
        return -np.rad2deg(x_angle - np.pi if x_angle > 0 else x_angle)  # Todo: why?

    def get_shard_path(self, index: int, obj_dir: str, shard: int = None) -> str:
        if shard is None:
            shard = self.get_shard(index)
        return os.path.join(obj_dir, self.data_dir, "train_pbr", str(shard).zfill(6))

    def get_shard(self, index: int) -> int:
        epoch = index // self.num_objects
        return epoch // self.files_per_shard

    def get_file(self, index: int) -> int:
        epoch = index // self.num_objects
        return epoch - epoch // self.files_per_shard * self.files_per_shard

    def get_depth_path(self, index: int, obj_dir: str, shard: int = None, file: int = None) -> str:
        if shard is None:
            shard = self.get_shard(index)
        if file is None:
            file = self.get_file(index)
        shard_path = self.get_shard_path(index, obj_dir, shard)
        return os.path.join(shard_path, "depth", str(file).zfill(6) + ".png")

    def load(self, obj_dir, index, category):
        if self.path_prefix:
            synthset = obj_dir.split('/')[-2]
            model = obj_dir.split('/')[-1]
            obj_dir = os.path.join(self.path_prefix, synthset, model)
        obj_name = obj_dir.split('/')[-1]
        path_to_camera_json = os.path.join(obj_dir, self.data_dir, "camera.json")

        shard = self.get_shard(index)
        file = self.get_file(index)

        if self.random_shard and self.num_shards > 1:
            shard = np.random.randint(self.num_shards)
        if self.random_file and self.files_per_shard > 1:
            file = np.random.randint(self.files_per_shard)

        shard_path = self.get_shard_path(index, obj_dir, shard)
        path_to_scene_camera_json = os.path.join(shard_path, "scene_camera.json")

        camera_parameters = get_camera_parameters_from_blenderproc_bopwriter(path_to_scene_camera_json,
                                                                             path_to_camera_json,
                                                                             scene_id=-1 if self.fuse else file)
        if self.fuse:
            chosen_cam_params = camera_parameters[file]
            chosen_location = -chosen_cam_params.extrinsic[:3, :3].T @ chosen_cam_params.extrinsic[:3, 3]
            other_locations = np.vstack([-c.extrinsic[:3, :3].T @ c.extrinsic[:3, 3] for c in camera_parameters])
            distances = np.abs(other_locations - chosen_location)
            indices = np.arange(len(distances))
            within_x = distances[:, 0] <= self.fuse_thresholds[0]
            within_y = distances[:, 1] <= self.fuse_thresholds[1]
            within_z = distances[:, 2] <= self.fuse_thresholds[2]
            same_sign_y = np.sign(other_locations[:, 1]) == np.sign(chosen_location[1])
            same_sign_z = np.sign(other_locations[:, 2]) == np.sign(chosen_location[2])
            n_closest = indices[within_x & within_y & within_z & same_sign_y & same_sign_z]
            n_closest = n_closest[n_closest != file]

            if isinstance(self.fuse, (tuple, list)):
                fuse = np.random.randint(*self.fuse)
            else:
                fuse = self.fuse
            if fuse == 0:
                n_closest = []
            elif len(n_closest) > fuse:
                n_closest = np.random.choice(n_closest, size=fuse, replace=False)
        else:
            chosen_cam_params = camera_parameters[0]
            n_closest = []

        intrinsic = chosen_cam_params.intrinsic
        extrinsic = chosen_cam_params.extrinsic

        depth_path = self.get_depth_path(index, obj_dir, shard, file)
        if self.load_fixed:
            fixed_depth_path = depth_path.replace(".png", "_fixed.png")
            if os.path.exists(fixed_depth_path):
                depth_path = fixed_depth_path

        data = self.cache.get(depth_path) if isinstance(self.cache, dict) else None
        if data is None:
            if self.input_type in ["image", "rgbd"]:
                image = Image.open(depth_path.replace("depth", "rgb").replace(".png", ".jpg")).convert("RGB")
                if self.input_type == "image":
                    inputs = image

            if self.input_type != "image":
                inputs = self._get_points_from_depth(depth_path, intrinsic, extrinsic)

                opengl_extrinsic = extrinsic.copy()
                # opengl_extrinsic[1, :] *= -1
                opengl_extrinsic[2, :] *= -1
                inv_extrinsic = inv_trafo(opengl_extrinsic)

                rot = self.swap_xy @ (self.swap_yz @ inv_extrinsic[:3, :3]).T
                cam = self.swap_yz @ inv_extrinsic[:3, 3]
                x_angle = self.get_x_angle_from_rot(rot)

                for n in n_closest:
                    depth_path = os.path.join(shard_path, "depth", str(n).zfill(6) + ".png")
                    intrinsic = camera_parameters[n].intrinsic
                    extrinsic = camera_parameters[n].extrinsic
                    inputs = np.concatenate([inputs, self._get_points_from_depth(depth_path, intrinsic, extrinsic)])

            if isinstance(self.cache, dict):
                logger.debug(f"{self.name}: Caching {self.input_type} data.")
                self.cache[depth_path] = {None: inputs.astype(np.float32)}
                if self.input_type != "image":
                    self.cache[depth_path].update({"rot": rot.astype(np.float32),
                                                   "cam": cam.astype(np.float32),
                                                   "x_angle": x_angle})
                    if self.input_type == "rgbd":
                        self.cache[depth_path]["image"] = image
        else:
            logger.debug(f"{self.name}: Retrieving {self.input_type} data from cache.")
            inputs = data[None]
            if self.input_type != "image":
                rot = data["rot"]
                cam = data["cam"]
                x_angle = data["x_angle"]
                if self.input_type == "rgbd":
                    image = data["image"]

        data = {None: inputs,
                "rot": rot,
                "cam": cam,
                "intrinsic": intrinsic.intrinsic_matrix,
                "extrinsic": extrinsic,
                "x_angle": x_angle,
                "name": obj_name,
                "path": depth_path}
        if self.input_type == "rgbd":
            data["image"] = image
        data = copy.deepcopy(data)

        if self.unscale or self.undistort:
            scale_path = os.path.join(shard_path, "scale.npy")
            scale = np.load(scale_path)
            data["scale"] = scale
            if isinstance(data[None], np.ndarray):
                if self.unscale and self.undistort:
                    data[None] /= scale
                elif self.unscale:
                    if isinstance(scale, float):
                        data[None] /= scale
                    else:
                        data[None] /= scale[1]
                elif self.undistort and not isinstance(scale, float):
                    data[None] /= scale
                    data[None] *= scale[1]

        data = apply_transform(data, self.transform)

        return data


class RGBDField(Field):
    def __init__(self,
                 rgb: bool = False,
                 high_res_rgb: bool = False,
                 mask: bool = False,
                 crop: float = 0.2,
                 cam_id_range: Tuple[int, int] = (1, 5),
                 merge_cam_id_range: Tuple[int, int] = None,
                 merge_angles: int = 0,
                 stride: int = 1,
                 filter_discontinuities: int = None,
                 transform=None):
        super().__init__()

        if cam_id_range:
            assert max(cam_id_range) <= 5 and min(cam_id_range) >= 1
        if merge_cam_id_range:
            assert merge_cam_id_range == cam_id_range
        if merge_angles:
            assert merge_angles % 2 == 1
        if high_res_rgb:
            logger.debug("Warning: High-res RGB is not properly calibrated.")
        assert stride > 0

        self.rgb = rgb
        self.high_res_rgb = high_res_rgb
        self.mask = mask
        self.crop = crop
        self.cam_id_range = cam_id_range if cam_id_range else (1, 5)
        self.merge_cam_id_range = merge_cam_id_range
        self.merge_angles = merge_angles if merge_angles else 0
        self.stride = stride
        self.filter_discontinuities = filter_discontinuities

        self.transform = transform

        self.depth_shape = 480, 640
        self.rgb_shape = 1024, 1280
        self.high_res_rgb_shape = 2848, 4272

        self.rotation_angle = 3
        self.num_rotations = 360 // self.rotation_angle
        self.max_index = self.num_rotations * max(self.cam_id_range)
        self.min_index = self.num_rotations * (min(self.cam_id_range) - 1)
        self.num_images = self.max_index - self.min_index

        self.rot_x_180 = R.from_euler('x', np.pi).as_matrix()

    @staticmethod
    def undistort_points(points: np.ndarray, d: np.ndarray):
        _points = copy.deepcopy(points)
        k1 = d[0]
        k2 = d[1]
        k3 = d[4]
        p1 = d[2]
        p2 = d[3]
        z = _points[:, 2]
        x_p = _points[:, 0] / z
        y_p = _points[:, 1] / z
        r_2 = x_p ** 2 + y_p ** 2
        r_4 = x_p ** 4 + y_p ** 4
        r_6 = x_p ** 6 + y_p ** 6

        # Fix radial distortion
        radial = (1 + k1 * r_2 + k2 * r_4 + k3 * r_6)
        x_r = x_p * radial
        y_r = y_p * radial

        # Fix tangential distortion
        x_t = x_r + (2 * p1 * x_r * y_r + p2 * (r_2 + 2 * x_r ** 2))
        y_t = y_r + (p1 * (r_2 + 2 * y_r ** 2) + 2 * p2 * x_r * y_r)
        _points[:, 0] = z * x_t
        _points[:, 1] = z * y_t
        return _points

    def get_indices(self, index: int) -> Tuple[int, int]:
        """Computes camera index and turntable index from dataset instance index.

        Args:
            index: Dataset instance index (0...#instance).

        Returns: Camera ID and turntable angle.
        """
        image_index = index + self.min_index - index // self.num_images * self.num_images
        cam_index = image_index // self.num_rotations
        angle = self.rotation_angle * image_index - cam_index * 360
        return cam_index + 1, angle

    def load_data(self, obj_dir: str, cam_index: int, angle: int):
        depth_path = os.path.join(obj_dir, f"NP{cam_index}_{angle}.h5")
        calibration_path = os.path.join(obj_dir, "calibration.h5")
        pose_path = os.path.join(obj_dir, "poses", f"NP5_{angle}_pose.h5")

        calibration = h5py.File(calibration_path)
        if self.high_res_rgb:
            rgb_intrinsics = calibration[f"N{cam_index}_rgb_K"][:]
        else:
            rgb_intrinsics = calibration[f"NP{cam_index}_rgb_K"][:]
        depth_intrinsics = calibration[f"NP{cam_index}_depth_K"][:]
        depth_scale = calibration[f"NP{cam_index}_ir_depth_scale"][0] * 1e-4  # 100um to meters
        if self.high_res_rgb:
            f_rgb_ref = calibration[f"H_N{cam_index}_from_NP5"][:]  # N[1...5] RGB cam pose in NP5 RGB cam coords
        else:
            f_rgb_ref = calibration[f"H_NP{cam_index}_from_NP5"][:]  # NP[1-5] RGB cam pose in NP5 RGB cam coords
        f_ir_ref = calibration[f"H_NP{cam_index}_ir_from_NP5"][:]  # NP[1-5] depth cam poise in NP5 RGB cam coords
        calibration.close()

        f_ref_rgb = inv_trafo(f_rgb_ref)  # NP5 RGB cam pose in N/NP[1-5] RGB cam coords
        f_ref_ir = inv_trafo(f_ir_ref)  # NP5 RGB cam pose in NP[1-5] depth cam coords
        f_rgb_ir = f_rgb_ref @ f_ref_ir  # N/NP[1-5] RGB cam pose in NP[1-5] depth cam coords

        pose_data = h5py.File(pose_path)
        f_table_ref = pose_data["H_table_from_reference_camera"][:]  # Table pose in NP5 coords
        f_ir_table = inv_trafo(f_table_ref @ f_ref_ir)  # NP[1-5] depth cam pose in table coords
        f_rgb_table = inv_trafo(f_table_ref @ f_ref_rgb)  # N/NP[1-5] RGB cam pose in table coords
        pose_data.close()

        # table_data = h5py.File(os.path.join(obj_dir, "poses", "turntable.h5"))
        # print([v[:] for v in table_data.values()])
        # table_data.close()

        depth_data = h5py.File(depth_path)
        depth = depth_data["depth"][:]
        depth_data.close()

        if self.filter_discontinuities:
            depth = _filter_discontinuities(depth, filter_size=self.filter_discontinuities)
        depth *= depth_scale

        return depth, depth_intrinsics, rgb_intrinsics, f_rgb_ir, f_ir_table, f_rgb_table

    def load_single(self, obj_dir, index):
        cam_index, angle = self.get_indices(index)
        rgb_shape = self.high_res_rgb_shape if self.high_res_rgb else self.rgb_shape
        depth, depth_intrinsics, rgb_intrinsics, f_rgb_ir, f_ir_table, f_rgb_table = self.load_data(obj_dir,
                                                                                                    cam_index,
                                                                                                    angle)

        # Project depth_image to point cloud
        depth_extrinsics = np.eye(4) if self.rgb or self.mask else f_ir_table
        pcd = convert_depth_image_to_point_cloud(depth,
                                                 camera_intrinsic=depth_intrinsics,
                                                 camera_extrinsic=depth_extrinsics,
                                                 depth_scale=1,
                                                 depth_trunc=3.5,
                                                 stride=1)

        if self.rgb or self.mask:
            base_path = f"N{cam_index}_{angle}" if self.high_res_rgb else f"NP{cam_index}_{angle}"
            points = np.asarray(pcd.points)

            # Transform depth image point cloud to RGB camera coordinates
            points = (f_rgb_ir[:3, :3] @ points.T).T + f_rgb_ir[:3, 3]

            # Reproject transformed depth image to image plane
            z = points[:, 2]
            inverse_z = np.reciprocal(z).repeat(3).reshape(-1, 3)
            u, v, _ = (inverse_z * (rgb_intrinsics @ points.T).T).T

            # Fixme: Why? Like np.ceil before typecasting to int?
            u = (u + 0.5).astype(int)
            v = (v + 0.5).astype(int)

            # Handle image boundaries
            height, width = rgb_shape
            mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            z = z[mask]
            u = u[mask]
            v = v[mask]

            # Handle occlusions
            d = depth.ravel()
            d = d[d > 0]
            d = d[mask]
            occ_mask = (z == 0) | (z > d)
            z[occ_mask] = d[occ_mask]

            depth = np.zeros(rgb_shape, dtype=np.float32)
            depth[v, u] = z

            # Cut object from background using segmentation mask
            if self.mask:
                depth_mask_path = os.path.join(obj_dir, "masks", base_path + "_mask.pbm")
                depth_mask = np.asarray(Image.open(depth_mask_path).convert('L'), dtype=bool)
                depth[depth_mask] = 0

            if self.rgb:
                rgb_path = os.path.join(obj_dir, base_path + ".jpg")
                rgb = np.asarray(Image.open(rgb_path))
                pcd = convert_rgbd_image_to_point_cloud([rgb, depth],
                                                        camera_intrinsic=rgb_intrinsics,
                                                        camera_extrinsic=f_rgb_table,
                                                        depth_scale=1,
                                                        depth_trunc=3.5,
                                                        convert_rgb_to_intensity=False)
            else:
                pcd = convert_depth_image_to_point_cloud(depth,
                                                         camera_intrinsic=rgb_intrinsics,
                                                         camera_extrinsic=f_rgb_table,
                                                         depth_scale=1,
                                                         depth_trunc=3.5,
                                                         stride=1)

        if self.crop:
            pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=(-self.crop, -self.crop, -0.01),
                                                               max_bound=(self.crop, self.crop, 0.75)))

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        trafo = f_rgb_table if self.rgb or self.mask else f_ir_table
        rot = self.rot_x_180 @ trafo[:3, :3]
        x_angle = -np.arctan2(rot.T[2, 1], rot.T[2, 2])
        x_angle += np.pi / 2  # Switch z-axis up to y-axis up
        x_angle = np.rad2deg(x_angle)

        data = {None: points,
                "rot": rot,
                "x_angle": x_angle}
        if self.rgb:
            data["colors"] = colors
        return data

    def get_merged_cams(self, obj_dir, index):
        points = list()
        colors = list()
        rot = np.eye(3)
        x_angle = 0
        start, end = self.merge_cam_id_range
        for cam_index in range(start - 1, end):
            data = self.load_single(obj_dir, index + cam_index * self.num_rotations)
            points.append(data[None])
            if "colors" in data:
                colors.append(data["colors"])
            if cam_index == (start - 1) + (end - start + 1) // 2:
                rot = data["rot"]
                x_angle = data["x_angle"]
        data = {None: np.concatenate(points),
                "rot": rot,
                "x_angle": x_angle}
        if colors:
            data["colors"] = np.concatenate(colors)
        return data

    def get_merged_angles(self, obj_dir, index):
        points = list()
        colors = list()
        rot = np.eye(3)
        x_angle = 0
        cam_index, angle = self.get_indices(index)
        offset = self.merge_angles // 2
        for o in range(-offset * self.stride, offset * self.stride + 1, self.stride):
            i = o
            if angle // self.rotation_angle + o < 0:
                i += self.num_rotations
            elif angle // self.rotation_angle + o >= self.num_rotations:
                i -= self.num_rotations
            if self.merge_cam_id_range:
                data = self.get_merged_cams(obj_dir, index + i)
            else:
                data = self.load_single(obj_dir, index + i)
            points.append(data[None])
            if "colors" in data:
                colors.append(data["colors"])
            if o == 0:
                rot = data["rot"]
                x_angle = data["x_angle"]
        data = {None: np.concatenate(points),
                "rot": rot,
                "x_angle": x_angle}
        if colors:
            data["colors"] = np.concatenate(colors)
        return data

    def load(self, obj_dir, index, category):
        if self.merge_cam_id_range or self.merge_angles > 1:
            if self.merge_angles:
                data = self.get_merged_angles(obj_dir, index)
            else:
                data = self.get_merged_cams(obj_dir, index)
        else:
            data = self.load_single(obj_dir, index)

        data["name"] = obj_dir.split('/')[-1]
        data["path"] = obj_dir

        data = apply_transform(data, self.transform)

        return data


class PointCloudField(Field):
    def __init__(self,
                 file: str = "pointcloud.npz",
                 data_dir: Optional[str] = None,
                 cam_dir: Optional[str] = None,
                 index: Union[Tuple[int, int], int] = -1,
                 num_cams: int = 24,
                 normals_file: Optional[str] = None,
                 load_normals: bool = False,
                 pose_file: Optional[str] = None,
                 min_max_only: bool = False,
                 cache: bool = False,
                 from_hdf5: bool = False,
                 transform=None):
        super().__init__()

        assert not (load_normals and min_max_only), "Normals and min_max_only not supported yet"

        self.file = file
        self.data_dir = "" if data_dir is None else data_dir
        self.cam_dir = cam_dir
        self.index = index
        self.num_cams = num_cams
        self.normals_file = normals_file
        self.load_normals = load_normals
        self.pose_file = pose_file
        self.min_max_only = min_max_only
        self.cache = dict() if cache else None
        self.from_hdf5 = from_hdf5
        self.transform = transform

    def load(self, obj_dir, index, category):
        obj_dir = Path(obj_dir)
        obj_name = obj_dir.name
        obj_dir = obj_dir / self.data_dir
        file_path = obj_dir / self.file
        data = self.cache.get(obj_name) if isinstance(self.cache, dict) else None

        if data is None:
            points, normals = _load_pointcloud(obj_dir,
                                               self.file,
                                               self.normals_file,
                                               self.load_normals,
                                               from_hdf5=self.from_hdf5)

            if self.pose_file and isinstance(self.pose_file, str):
                pose_path = os.path.join(obj_dir, self.pose_file)
                if os.path.isfile(pose_path):
                    if self.pose_file.endswith(".npy"):
                        pose = np.load(pose_path)
                        points = points @ pose[:3, :3].T + pose[:3, 3]
                        if normals is not None:
                            normals = normals @ pose[:3, :3].T
                    elif self.pose_file.endswith(".npz"):
                        pose = np.load(pose_path)
                        offset = pose["loc"]
                        scale = pose["scale"]
                        points *= scale
                        points += offset
                    else:
                        raise TypeError(f"Pose file {self.pose_file} is not supported.")

            if self.min_max_only:
                points = np.array([points.min(0), points.max(0)])

            if isinstance(self.cache, dict):
                logger.debug(f"{self.name}: Caching pointcloud.")
                self.cache[obj_name] = {None: points.astype(np.float32)}
                if normals is not None:
                    self.cache[obj_name]["normals"] = normals.astype(np.float32)
        else:
            logger.debug(f"{self.name}: Retrieving pointcloud from cache.")
            points = data[None]
            normals = data.get("normals")

        data = {None: points,
                "name": obj_name,
                "path": file_path}
        if normals is not None:
            data["normals"] = normals

        if self.cam_dir:
            cam_path = os.path.join(obj_dir, self.cam_dir, "cameras.npz")
            assert os.path.isfile(cam_path), f"Camera file {cam_path} does not exist."
            data.update(_load_cam(index, self.index, self.num_cams, cam_path))

        data = copy.deepcopy(data)

        data = apply_transform(data, self.transform)

        return data


class PointsField(Field):
    def __init__(self,
                 file: Union[str, Path, List[str], List[Path]] = "points.npz",
                 data_dir: Optional[str] = None,
                 pose_file: Optional[str] = None,
                 params_dir: Optional[str] = None,
                 path_suffix: Optional[str] = None,
                 padding: float = 0.1,
                 num_points: int = int(1e5),
                 occ_from_sdf: bool = True,
                 tsdf: float = 0,
                 sigmas: Optional[List[float]] = None,
                 cache: bool = False,
                 crop: bool = False,
                 normalize: bool = True,
                 load_all_files: bool = False,
                 load_random_file: bool = False,
                 load_surface_file: Optional[Union[bool, float]] = True,
                 from_hdf5: bool = False,
                 transform=None):
        super().__init__()

        assert not (load_all_files and load_random_file), \
            "Cannot load all files and a random file at the same time."
        assert padding >= 0
        if sigmas is not None:
            assert all([sigma > 0 for sigma in sigmas])
        assert not (occ_from_sdf and tsdf)
        self.file = [file] if isinstance(file, (str, Path)) else file
        self.data_dir = data_dir
        self.pose_file = pose_file
        self.params_dir = params_dir
        self.path_suffix = "" if path_suffix is None else path_suffix
        self.transform = transform
        self.occ_from_sdf = occ_from_sdf
        self.tsdf = tsdf
        self.sigmas = sigmas
        self.padding = padding
        self.num_points = num_points
        self.cache = dict() if cache else None
        self.crop = crop
        self.normalize = normalize
        self.load_all_files = load_all_files
        self.load_random_file = load_random_file
        self.load_surface_file = False if load_surface_file is None else load_surface_file
        self.from_hdf5 = from_hdf5

    def load_file(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if self.from_hdf5:
            points_data = load_from_binary_hdf5(file_path.parent.parent,
                                                [file_path.name],
                                                [file_path.parent.name])[0]
            points = points_data["points"]
            occupancy = points_data["occupancies"]
            occupancy = np.unpackbits(occupancy)
            return points, occupancy

        if file_path.suffix == ".npz":
            points_data = np.load(file_path)
            points = points_data["points"]
            occupancy = points_data["occupancies"]
            occupancy = np.unpackbits(occupancy)
        elif file_path.suffix == ".npy":
            points_data = np.load(file_path)
            points = points_data[:, :3]
            occupancy = points_data[:, 3]
            if (self.occ_from_sdf or self.tsdf) and occupancy.min() < 0:
                if self.occ_from_sdf:
                    occupancy = occupancy <= 0
                elif self.tsdf:
                    occupancy = np.clip(occupancy, -self.tsdf, self.tsdf)
        elif file_path.suffix in [".obj", ".off", ".ply"]:
            boxsize = 1 + self.padding
            points = np.random.rand(self.num_points, 3)
            points = boxsize * (points - 0.5)

            if self.normalize:
                mesh, loc, scale = _load_mesh(file_path, normalize=True, return_values=True)
            else:
                mesh = _load_mesh(file_path)
                loc = 0
                scale = 1

            if self.sigmas:
                samples = list()
                num_points = self.num_points // len(self.sigmas)
                for sigma in self.sigmas:
                    noise = sigma * np.random.standard_normal((num_points, 3))
                    samples.append(mesh.sample(num_points) + noise)
                points_surface = np.concatenate(samples, axis=0)

                points_sphere_1 = generate_random_basis(self.num_points, radius=1, seed=None)
                points_sphere_2 = generate_random_basis(self.num_points, radius=2, seed=None)
                points_sphere_5 = generate_random_basis(self.num_points, radius=5, seed=None)
                points_sphere_10 = generate_random_basis(self.num_points, radius=10, seed=None)
                points_sphere_20 = generate_random_basis(self.num_points, radius=20, seed=None)
                points_sphere_50 = generate_random_basis(self.num_points, radius=50, seed=None)

                points = np.concatenate([points,
                                         points_surface,
                                         points_sphere_1,
                                         points_sphere_2,
                                         points_sphere_5,
                                         points_sphere_10,
                                         points_sphere_20,
                                         points_sphere_50], axis=0)
            occupancy = check_mesh_contains(mesh, points)
            points *= scale
            points += loc
        else:
            raise TypeError
        return points, occupancy

    def load(self, obj_dir, index, category):
        obj_dir = Path(obj_dir)
        obj_name = obj_dir.name
        obj_category = obj_dir.parent.name
        obj_dir = obj_dir / self.path_suffix

        if self.data_dir:
            obj_dir = Path(self.data_dir) / obj_category / obj_name / self.path_suffix

        file = self.file if self.file else [obj_name + ".ply"]
        file_path = [obj_dir / f for f in file]
        data = self.cache.get(obj_name) if isinstance(self.cache, dict) else None

        if data is None:
            points_list = list()
            occupancy_list = list()

            if not self.load_all_files:
                if self.load_random_file:
                    assert len(file_path) > 2, "Must have more than 2 query points files to load a random file"
                    assert "uniform_random" in str(
                        file_path[0]), "First query points file in list must be 'uniform_random'"
                    file_path = [file_path[0], np.random.choice(file_path[1:])]
                elif len(file_path) == 1:
                    pass
                else:
                    assert len(file_path) in [7, 8], "7 or 8 query points files required when not loading a random file"
                    assert "uniform_random" in str(file_path[0]), \
                        "First query points file in list must be 'uniform_random'"
                    if len(file_path) == 8:
                        assert "surface_random" in str(file_path[1]), \
                            "Second query points file in list must be 'surface_random'"

                    params_dir = obj_dir if self.params_dir is None else Path(self.params_dir) / obj_category / obj_name
                    logger.debug(f"{self.name}: Loading parameters from {params_dir / 'parameters.npz'}.")
                    if self.from_hdf5:
                        parameters = load_from_binary_hdf5(params_dir, ["parameters.npz"])[0]
                    else:
                        parameters = np.load(obj_dir / "parameters.npz")

                    scale = parameters["scales"][category]
                    if isinstance(scale, np.ndarray):
                        scale = max(scale)
                    files = [file_path[0]]
                    if "surface" in str(file_path[1]) and np.random.random() < self.load_surface_file:
                        files.append(file_path[1])
                    if 1 / scale <= 1:
                        files.append(file_path[2] if len(file_path) == 8 else file_path[1])
                    elif 1 / scale <= 2:
                        files.append(file_path[3] if len(file_path) == 8 else file_path[2])
                    elif 1 / scale <= 5:
                        files.append(file_path[4] if len(file_path) == 8 else file_path[3])
                    elif 1 / scale <= 10:
                        files.append(file_path[5] if len(file_path) == 8 else file_path[4])
                    elif 1 / scale <= 20:
                        files.append(file_path[6] if len(file_path) == 8 else file_path[5])
                    else:
                        files.append(file_path[7] if len(file_path) == 8 else file_path[6])
                    file_path = files

            logger.debug(f"{self.name}: Loading points from {[f.name for f in file_path]}.")
            for path in file_path:
                points, occupancy = self.load_file(path)
                points_list.append(points)
                occupancy_list.append(occupancy)
            points = np.concatenate(points_list)
            occupancy = np.concatenate(occupancy_list)

            if self.pose_file is not None:
                pose_path = obj_dir / self.pose_file
                if pose_path.is_file():
                    if self.pose_file.endswith(".npy"):
                        pose = np.load(pose_path)
                        points = points @ pose[:3, :3].T + pose[:3, 3]
                    elif self.pose_file.endswith(".npz"):
                        pose = np.load(pose_path)
                        offset = pose["loc"]
                        scale = pose["scale"]
                        points *= scale
                        points += offset
                    else:
                        raise TypeError(f"Pose file {self.pose_file} is not supported.")

            if isinstance(self.cache, dict):
                logger.debug(f"{self.name}: Caching points.")
                self.cache[obj_name] = {None: points.astype(np.float32),
                                        "occ": occupancy.astype(bool),
                                        "path": file_path}
        else:
            logger.debug(f"{self.name}: Retrieving points from cache.")
            points = data[None]
            occupancy = data["occ"]
            file_path = data["path"]

        if self.crop:
            mask = np.all(np.abs(points) <= 0.5 + self.padding / 2, axis=1)
            if mask.sum() > 0:
                points = points[mask]
                occupancy = occupancy[mask]

        data = {None: points,
                "occ": occupancy,
                "name": obj_name,
                "path": file_path}
        data = copy.deepcopy(data)

        data = apply_transform(data, self.transform)

        return data


class VoxelsField(Field):
    def __init__(self,
                 file: str = "model.binvox",
                 use_trimesh: bool = True,
                 transform=None):
        super().__init__()

        self.file = file
        self.use_trimesh = use_trimesh
        self.transform = transform

    def load(self, obj_dir, index, category):
        file_path = os.path.join(obj_dir, self.file)

        with open(file_path, "rb") as f:
            if self.use_trimesh:
                voxels = np.asarray(trimesh.exchange.binvox.load_binvox(f).matrix)
            else:
                voxels = binvox.Binvox.read(f, mode="dense").data

        data = {None: voxels,
                "name": obj_dir.split('/')[-1],
                "path": file_path}

        data = apply_transform(data, self.transform)

        return data


class MeshField(Field):
    def __init__(self,
                 file: Optional[str] = None,
                 file_prefix: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 pose_file: Optional[str] = None,
                 cache: bool = False,
                 process: bool = False,
                 from_hdf5: bool = False,
                 transform=None):
        super().__init__()

        self.file = file
        self.file_prefix = file_prefix if file_prefix else ""
        self.data_dir = "" if data_dir is None else data_dir
        self.pose_file = pose_file
        self.cache = dict() if cache else None
        self.process = process
        self.from_hdf5 = from_hdf5
        self.transform = transform

    def load(self, obj_dir, index, category):
        obj_name = obj_dir.split('/')[-1]
        obj_dir = os.path.join(obj_dir, self.data_dir)
        file_path = os.path.join(obj_dir, self.file) if self.file else os.path.join(obj_dir,
                                                                                    self.file_prefix + obj_name + ".ply")
        data = self.cache.get(obj_name) if isinstance(self.cache, dict) else None

        if data is None:
            if self.from_hdf5:
                file_path = Path(file_path)
                mesh = load_from_binary_hdf5(file_path.parent, [file_path.name])[0].as_open3d
            else:
                mesh = o3d.io.read_triangle_mesh(file_path, enable_post_processing=self.process)
            if self.process and self.cache is not None and not mesh.is_watertight():
                # indices, length, area = mesh.cluster_connected_triangles()
                # mesh.remove_triangles_by_index(np.argwhere(indices != np.argmax(length)))
                mesh.remove_duplicated_triangles()
                mesh.remove_degenerate_triangles()
                mesh.remove_non_manifold_edges()
                mesh.remove_duplicated_vertices()
                mesh.remove_unreferenced_vertices()
                mesh = mesh.select_by_index(np.arange(len(np.asarray(mesh.vertices))), cleanup=True)
            if self.pose_file and isinstance(self.pose_file, str):
                pose_path = os.path.join(obj_dir, self.pose_file)
                if os.path.isfile(pose_path):
                    logger.debug(f"{self.name}: Applying pose {self.pose_file}.")
                    if self.pose_file.endswith(".npy"):
                        mesh = mesh.transform(np.load(pose_path))
                    elif self.pose_file.endswith(".npz"):
                        pose = np.load(pose_path)
                        offset = pose["loc"]
                        scale = pose["scale"]
                        mesh.scale(scale, center=(0, 0, 0))
                        mesh.translate(offset)
                    else:
                        raise TypeError(f"Pose file {self.pose_file} is not supported.")
                logger.debug(f"{self.name}: Pose file {self.pose_file} does not exist.")

            data = {"vertices": np.asarray(mesh.vertices, dtype=np.float32),
                    "triangles": np.asarray(mesh.triangles)}
            if mesh.has_triangle_uvs():
                data["uvs"] = np.asarray(mesh.triangle_uvs)
            if mesh.has_triangle_material_ids():
                data["ids"] = np.asarray(mesh.triangle_material_ids)
            if mesh.has_vertex_colors():
                data["colors"] = np.asarray(mesh.vertex_colors)
            if mesh.has_vertex_normals():
                data["normals"] = np.asarray(mesh.vertex_normals, dtype=np.float32)
            if mesh.has_textures():
                data["textures"] = [np.asarray(texture) for texture in mesh.textures]

            if isinstance(self.cache, dict):
                logger.debug(f"{self.name}: Caching mesh.")
                self.cache[obj_name] = data
        else:
            logger.debug(f"{self.name}: Retrieving mesh from cache.")

        data.update({"name": obj_name,
                     "path": file_path})
        data = copy.deepcopy(data)

        data = apply_transform(data, self.transform)

        return data


class PartNetField(Field):
    def __init__(self,
                 data_dir: Union[str, Path],
                 transform=None):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.transform = transform

    def load(self, obj_dir, index, category):
        obj_dir = Path(obj_dir)
        obj_name = obj_dir.name
        obj_category = obj_dir.parent.name
        category_path = self.data_dir / obj_category
        points_path = (category_path / "points" / obj_name).with_suffix(".pts")
        label_path = (category_path / "points_label" / obj_name).with_suffix(".seg")

        points = np.loadtxt(points_path)
        labels = np.loadtxt(label_path, dtype=np.int64)

        bounds = np.array([points.min(axis=0), points.max(axis=0)])
        points -= bounds.mean(axis=0)
        points /= bounds.ptp(axis=0).max()

        data = {"points": points,
                "labels": labels,
                "name": obj_name,
                "path": points_path}
        data = copy.deepcopy(data)

        data = apply_transform(data, self.transform)

        return data
