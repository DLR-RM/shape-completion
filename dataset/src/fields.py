import copy
import functools
import json
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import binvox
import h5py
import numpy as np
import open3d as o3d
import pyrender
import torch.cuda
from easy_o3d.registration import IterativeClosestPoint
from easy_o3d.utils import (
    convert_depth_image_to_point_cloud,
    convert_rgbd_image_to_point_cloud,
    get_camera_parameters_from_blenderproc_bopwriter,
)
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from skimage.util.shape import view_as_windows
from trimesh import Trimesh
from trimesh.exchange import binvox as trimesh_binvox

from utils import (
    adjust_intrinsic,
    apply_trafo,
    bbox_from_mask,
    convert_extrinsic,
    crop_and_resize_image,
    inv_trafo,
    load_from_binary_hdf5,
    load_mesh,
    pitch_from_trafo,
    points_to_depth,
    setup_logger,
)

from .utils import calibration_from_blender

logger = setup_logger(__name__)


def _load_pointcloud(
    obj_dir: str | Path,
    file: str,
    normals_file: str | None = None,
    load_normals: bool = False,
    normalize_normals: bool = False,
    from_hdf5: bool = False,
    transform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    normals: np.ndarray | None = None
    file_path = Path(obj_dir) / file
    if from_hdf5:
        raw_data = load_from_binary_hdf5(file_path.parent.parent, [file_path.name], [file_path.parent.name])[0]
        if not isinstance(raw_data, Mapping):
            raise TypeError(f"Expected mapping pointcloud payload, got {type(raw_data)!r}")
        pointcloud_dict = cast(Mapping[str, Any], raw_data)
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


def _load_points(
    file_path: Path, occ_from_sdf: bool = False, tsdf: float = 0.0, from_hdf5: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if from_hdf5:
        raw_data = load_from_binary_hdf5(file_path.parent.parent, [file_path.name], [file_path.parent.name])[0]
        if not isinstance(raw_data, Mapping):
            raise TypeError(f"Expected mapping points payload, got {type(raw_data)!r}")
        points_data = cast(Mapping[str, Any], raw_data)
        points = points_data["points"]
        occupancy = points_data["occupancies"]
        occupancy = np.unpackbits(occupancy).astype(bool)
        return points, occupancy

    if file_path.suffix == ".npz":
        points_data = np.load(file_path)
        points = points_data["points"]
        occupancy = points_data["occupancies"]
        occupancy = np.unpackbits(occupancy).astype(bool)
    elif file_path.suffix == ".npy":
        points_data = np.load(file_path)
        points = points_data[:, :3]
        occupancy = points_data[:, 3]
        if (occ_from_sdf or tsdf) and occupancy.min() < 0:
            if occ_from_sdf:
                occupancy = occupancy <= 0
            elif tsdf:
                occupancy = np.clip(occupancy, -tsdf, tsdf)
    else:
        raise TypeError(f"File type {file_path.suffix} is not supported.")
    return points, occupancy


def _filter_discontinuities(depth: np.ndarray, filter_size: int = 7, threshold: int = 1000) -> np.ndarray:
    assert filter_size % 2 == 1

    offset = (filter_size - 1) // 2
    patches = view_as_windows(depth, (filter_size, filter_size))
    mids = patches[:, :, offset, offset]
    mins = np.min(patches, axis=(2, 3))
    maxes = np.max(patches, axis=(2, 3))

    discont = np.maximum(np.abs(mins - mids), np.abs(maxes - mids))
    mark = discont > threshold

    final_mark = np.zeros_like(depth)
    final_mark[offset : offset + mark.shape[0], offset : offset + mark.shape[1]] = mark

    return depth * (1 - final_mark)


def _load_cam(index: int, cam_index: tuple[int, int] | int, num_cams: int, path: Path) -> dict[str, Any]:
    if isinstance(cam_index, (tuple, list)):
        image_index = index + cam_index[0] - index // cam_index[1] * cam_index[1]
    elif isinstance(cam_index, int):
        image_index = np.random.randint(num_cams) if cam_index == -1 else cam_index
    else:
        raise TypeError(f"Unknown type for cam_index: {type(cam_index)}")

    if path.suffix in [".npz", ".npy"]:
        camera_dict = np.load(path)
        extrinsic = camera_dict[f"world_mat_{image_index}"]
        intrinsic = camera_dict[f"camera_mat_{image_index}"]
    elif path.suffix == ".txt":
        with open(path) as f:
            lines = f.readlines()
        params = eval(lines[image_index])[0]
        az, el, distance_ratio = params[0], params[1], params[3]
        intrinsic, extrinsic = calibration_from_blender(az, el, distance_ratio, width=224, height=224, padding=0)
    else:
        raise ValueError(f"Unknown file type: {path.suffix}")

    extrinsic_opencv = np.eye(4)
    extrinsic_opencv[:3, :] = extrinsic
    pitch = pitch_from_trafo(extrinsic_opencv)

    return {"pitch": pitch, "intrinsic": intrinsic, "extrinsic": extrinsic_opencv, "index": image_index}


class Field(ABC):
    def __init__(self, cachable: bool = True, cache: bool = False):
        self.name = self.__class__.__name__
        self.cachable = cachable

        if cachable and cache:
            orig_load = self.load
            cached = functools.cache(orig_load)

            def _load_with_copy(obj_dir, index, category=None):
                return copy.deepcopy(cached(obj_dir, index, category))

            _load_with_copy.__name__ = orig_load.__name__
            _load_with_copy.__doc__ = orig_load.__doc__

            self.load = _load_with_copy

    @abstractmethod
    def load(self, obj_dir: str | Path, index: int, category: int | None) -> dict[Any, int | np.ndarray]:
        raise NotImplementedError("A derived class must implement this method.")

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class RandomField(Field):
    def __init__(self, fields: list[Field], p: list[float] | None = None):
        super().__init__(cachable=False)
        if p is not None:
            if len(p) != len(fields):
                raise ValueError(f"Length of p ({len(p)}) must match the number of fields ({len(fields)}).")
            if not sum(p) == 1:
                raise ValueError(f"sum(p)={sum(p)}!=1")

        self.fields = fields
        self.p = p

    def load(self, obj_dir, index, category=None):
        field = random.choices(self.fields, weights=self.p, k=1)[0]
        logger.debug(f"Selected field: {field.name}")
        return field.load(obj_dir, index, category)


class MixedField(Field):
    def __init__(self, fields: list[Field], merge_keys: list[str] | None = None, p: list[float] | None = None):
        super().__init__(cachable=p is None and all(field.cachable for field in fields))
        if merge_keys is not None and len(merge_keys) != len(fields):
            raise ValueError(f"#merge_keys ({len(merge_keys)}) must match the #fields ({len(fields)}).")
        if p is not None:
            if len(p) != len(fields):
                raise ValueError(f"#p ({len(p)}) must match the #fields ({len(fields)}).")
            if not sum(p) == 1:
                raise ValueError(f"sum(p)={sum(p)}!=1")

        self.fields = fields
        self.merge_map = (
            {field.name: key for key, field in zip(merge_keys, fields, strict=False)} if merge_keys else None
        )
        self.p = p

    def load(self, obj_dir, index, category=None):
        data = dict()

        sorted_fields = self.fields
        if self.p is not None:
            idx = np.random.choice(len(self.fields), p=self.p)
            sorted_fields = [field for i, field in enumerate(self.fields) if i != idx] + [self.fields[idx]]

        if self.merge_map is None:
            for field in sorted_fields:
                data.update(field.load(obj_dir, index, category))
        else:
            for field in sorted_fields:
                field_data = field.load(obj_dir, index, category)
                if field.name in self.merge_map:
                    points = field_data.pop(None)
                    normals = field_data.pop("normals", None)
                    data.update(field_data)
                    merge_key = self.merge_map[field.name]
                    data[merge_key] = points
                    if normals is not None:
                        merge_key = merge_key + ".normals" if merge_key else "normals"
                        data[merge_key] = normals
                else:
                    data.update(field_data)

        return data


class EmptyField(Field):
    def load(self, obj_dir, index, category=None) -> dict:
        return {}


class IndexField(Field):
    def load(self, obj_dir, index, category=None) -> dict[None, int | str | Path]:
        obj_dir = Path(obj_dir)
        obj_name = obj_dir.name
        data = {None: index, "name": obj_name, "path": obj_dir}
        return data


class ImageField(Field):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        path_suffix: str | None = "img_choy2016",
        cam_filename: str = "cameras.npz",
        extension: str = ".jpg",
        index: tuple[int, int] | int = -1,
        num_images: int = 24,
        convention: str = "opencv",
    ):
        super().__init__()

        self.data_dir = data_dir
        self.extension = extension
        self.index = index
        self.num_images = num_images
        self.convention = convention
        self.path_suffix = "" if path_suffix is None else path_suffix
        self.cam_filename = cam_filename
        if isinstance(index, int):
            assert index <= self.num_images

    def load(self, obj_dir, index, category=None):
        obj_dir = Path(obj_dir)
        obj_name = obj_dir.name
        obj_category = obj_dir.parent.name
        obj_dir = obj_dir / self.path_suffix

        if self.data_dir:
            obj_dir = Path(self.data_dir) / obj_category / obj_name / self.path_suffix

        data: dict[Any, Any] = _load_cam(index, self.index, self.num_images, obj_dir / self.cam_filename)
        data["extrinsic"] = convert_extrinsic(data["extrinsic"], "opencv", self.convention)

        filename = (obj_dir / str(data["index"]).zfill(2 if self.data_dir else 3)).with_suffix(self.extension)
        image = Image.open(filename).convert("RGB")
        width, height = image.size
        data[None] = image
        data["width"] = width
        data["height"] = height
        data["name"] = obj_name
        data["path"] = filename

        return copy.deepcopy(data)


class BOPField(Field):
    def __init__(
        self,
        file_ids: list[int],
        camera: str | None = None,
        max_outlier_std: float = 0,
        max_correspondence_distance: float = 0,
    ):
        super().__init__()

        self.camera_file = "camera.json" if camera is None else f"camera_{camera}.json"
        self.file_ids = file_ids
        self.max_outlier_std = max_outlier_std
        if max_correspondence_distance:
            self.icp = IterativeClosestPoint(
                max_correspondence_distance=max_correspondence_distance,
                max_iteration=30,
                relative_fitness=1e-6,
                relative_rmse=1e-6,
            )
        else:
            self.icp = None

    def load(self, obj_dir, index, category):
        obj_dir = str(obj_dir)
        skip = False

        obj_id = category
        obj_name = str(category).zfill(6)
        file_id = self.file_ids[index]

        data_dir = "/".join(obj_dir.split("/")[:-2])
        path_to_scene_camera_json = os.path.join(obj_dir, "scene_camera.json")
        path_to_scene_gt_json = os.path.join(obj_dir, "scene_gt.json")
        path_to_scene_gt_info_json = os.path.join(obj_dir, "scene_gt_info.json")

        with open(path_to_scene_camera_json) as f:
            camera_data = json.load(f)
        intrinsic = np.asarray(camera_data[str(file_id)]["cam_K"]).reshape(3, 3)
        depth_scale = camera_data[str(file_id)]["depth_scale"]

        with open(path_to_scene_gt_json) as f:
            gt_poses = json.load(f)
        selected_mask_id: int | None = None
        for mask_id, pose_data in enumerate(gt_poses[str(file_id)]):
            pose = np.eye(4)
            pose[:3, :3] = np.asarray(pose_data["cam_R_m2c"]).reshape(3, 3)
            pose[:3, 3] = np.asarray(pose_data["cam_t_m2c"])
            if pose_data["obj_id"] == obj_id:
                selected_mask_id = mask_id
                break
        if selected_mask_id is None:
            selected_mask_id = 0
            skip = True
        up = pose[:3, 2][1] < 0 and pose[:3, 2][2] < 0
        if not up:
            skip = True

        with open(path_to_scene_gt_info_json) as f:
            info = json.load(f)
        info = info[str(file_id)][selected_mask_id]
        visib_fract = info["visib_fract"]

        if not visib_fract > 0.85:
            skip = True

        depth_path = os.path.join(obj_dir, "depth", f"{file_id:06d}.png")
        mask_path = os.path.join(obj_dir, "mask", f"{file_id:06d}_{selected_mask_id:06d}.png")

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

        inv_pose = np.asarray(inv_trafo(pose))
        pose[:3, 3] /= 1000
        inv_pose[:3, 3] /= 1000

        z_proj = (pose[:3, :3].T @ np.array([0, 0, 1]).T).T
        z_proj[2] = 0
        z_proj = z_proj / np.linalg.norm(z_proj)
        z_angle = np.arctan2(np.linalg.det([z_proj[:2], np.array([0, 1])]), np.dot(z_proj[:2], np.array([0, 1])))
        rot_to_cam = R.from_euler("z", z_angle).as_matrix()

        mesh = o3d.io.read_triangle_mesh(os.path.join(data_dir, "models_eval", f"obj_{obj_name}.ply"))
        mesh.scale(0.001, center=(0, 0, 0))

        render_intrinsic = intrinsic.copy()
        render_intrinsic[0, 2] = width / 2
        render_intrinsic[1, 2] = height / 2

        if self.icp is not None and not skip:
            pyrender_mesh = Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)

            pyrender_extrinsic = np.eye(4)
            rot_x_180 = R.from_euler("x", 180, degrees=True).as_matrix()
            pyrender_extrinsic[:3, :3] = rot_x_180 @ pose[:3, :3]
            pyrender_extrinsic[:3, 3] = pose[:3, 3] @ rot_x_180
            pyrender_extrinsic = inv_trafo(pyrender_extrinsic)

            camera = pyrender.IntrinsicsCamera(
                render_intrinsic[0, 0],
                render_intrinsic[1, 1],
                render_intrinsic[0, 2],
                render_intrinsic[1, 2],
                znear=0.01,
                zfar=10,
            )

            scene = pyrender.Scene()
            scene.add(pyrender.Mesh.from_trimesh(pyrender_mesh, smooth=False))
            scene.add(camera, pose=pyrender_extrinsic)

            renderer = pyrender.OffscreenRenderer(width, height)
            rendered_depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            mesh_depth = rendered_depth[1] if isinstance(rendered_depth, tuple) else rendered_depth
            mesh_depth = np.asarray(mesh_depth)
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

            mesh_pcd = convert_depth_image_to_point_cloud(
                mesh_depth, render_intrinsic, pose, depth_scale=1, depth_trunc=10
            )

            fisher1 = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                mesh_pcd, pcd, self.icp.max_correspondence_distance, np.eye(4)
            )

            result = self.icp.run(mesh_pcd, pcd, draw=False)

            fisher2 = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                mesh_pcd, pcd, self.icp.max_correspondence_distance, result.transformation
            )
            trafo = result.transformation.copy() if fisher2.mean() > fisher1.mean() else np.eye(4)
            """
            z_new = trafo[:3, 2]
            yz = np.array([z_new[1], z_new[2]])
            yz /= np.linalg.norm(yz)
            pitch = np.rad2deg(np.math.atan2(np.linalg.det([yz, np.array([0, 1])]), np.dot(yz, np.array([0, 1]))))
            rot_x = R.from_euler('x', pitch, degrees=True).as_matrix()

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
            if data_dir.split("/")[-1] == "hb" and obj_id == 6:
                rot_x = R.from_euler("x", -1.44, degrees=True).as_matrix()
                rot_y = R.from_euler("y", -1.99, degrees=True).as_matrix()
                rot_z = R.from_euler("z", 187.14, degrees=True).as_matrix()
                rot = rot_z @ rot_y @ rot_x
            elif data_dir.split("/")[-1] == "tyol":
                if obj_id == 3:
                    rot_x = R.from_euler("x", -0.89, degrees=True).as_matrix()
                    rot_y = R.from_euler("y", 2.11, degrees=True).as_matrix()
                    rot_z = R.from_euler("z", 180, degrees=True).as_matrix()
                elif obj_id == 4:
                    rot_x = R.from_euler("x", 0.54, degrees=True).as_matrix()
                    rot_y = R.from_euler("y", 3, degrees=True).as_matrix()
                    rot_z = R.from_euler("z", 189, degrees=True).as_matrix()
                elif obj_id == 5:
                    rot_x = R.from_euler("x", -2.52, degrees=True).as_matrix()
                    rot_y = R.from_euler("y", 0.351, degrees=True).as_matrix()
                    rot_z = R.from_euler("z", 182, degrees=True).as_matrix()
                elif obj_id == 6:
                    rot_x = R.from_euler("x", 3.43, degrees=True).as_matrix()
                    rot_y = R.from_euler("y", 1.2, degrees=True).as_matrix()
                    rot_z = R.from_euler("z", 182, degrees=True).as_matrix()
                elif obj_id == 20:
                    rot_x = R.from_euler("x", 8.27, degrees=True).as_matrix()
                    rot_y = R.from_euler("y", 4.47, degrees=True).as_matrix()
                    rot_z = R.from_euler("z", 179, degrees=True).as_matrix()
                elif obj_id == 21:
                    rot_x = R.from_euler("x", -5.1, degrees=True).as_matrix()
                    rot_y = R.from_euler("y", 0.15, degrees=True).as_matrix()
                    rot_z = R.from_euler("z", 184, degrees=True).as_matrix()
                else:
                    raise ValueError(f"Object id {obj_id} not supported.")
                rot = rot_z @ rot_y @ rot_x
            elif data_dir.split("/")[-1] == "ycbv" and obj_id == 14:
                rot_z = R.from_euler("z", 3.12, degrees=True).as_matrix()
                rot = rot_z

            iv_trafo = np.asarray(inv_trafo(trafo))
            inv_pose[:3, :3] = inv_pose[:3, :3] @ iv_trafo[:3, :3].T
            inv_pose[:3, 3] += iv_trafo[:3, 3]

            # mesh.scale(1 / 1.1, center=(0, 0, 0))
            # mesh.transform(iv_trafo)

            pcd.transform(iv_trafo)
            trafo = np.eye(4)
            trafo[:3, :3] = rot
            pcd.rotate(rot, center=(0, 0, 0))
            # mesh.rotate(rot, center=(0, 0, 0))

            pose = np.asarray(inv_trafo(inv_pose))
            pose[:3, :3] = pose[:3, :3] @ rot.T
            inv_pose = np.asarray(inv_trafo(pose))

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
        data = {
            None: points,
            "intrinsic": render_intrinsic,
            "extrinsic": pose,
            "width": width,
            "height": height,
            "cam": inv_pose[:3, 3],
            "pitch": 0,
            "visib_fract": visib_fract,
            "up": up,
            "skip": skip or len(points) == 0,
            "pose": trafo,
            "name": obj_name,
            "path": depth_path,
        }
        if pcd.has_normals():
            data["normals"] = np.asarray(pcd.normals)

        return copy.deepcopy(data)


class DepthField(Field):
    def __init__(
        self,
        data_dir: str | None = None,
        unscale: bool = True,
        unrotate: bool = True,
        load_depth: bool = True,
        load_normals: bool = True,
        load_colors: bool = False,
        kinect: bool = False,
        path_suffix: str | None = None,
        num_objects: int = 1,
        num_files: int = 100,
        file_offset: int = 0,
        random_file: bool = False,
        file_weights: dict[str, dict[str, np.ndarray]] | None = None,
        precision: int = 16,
        simulate_depth_noise: bool = False,
        from_hdf5: bool = False,
        project: bool = True,
        crop: bool = False,
        resize: int | None = None,
        convention: str = "opencv",
    ):
        super().__init__(cachable=not random_file)

        if project:
            if not load_depth:
                raise ValueError("Projecting requires loading depth.")
            if load_normals and load_colors:
                raise ValueError("Projecting both normals and colors is not supported.")

        self.data_dir = data_dir
        self.unscale = unscale
        self.unrotate = unrotate
        self.load_depth = load_depth
        self.load_normals = load_normals
        self.load_colors = load_colors
        self.kinect = kinect
        self.path_suffix = "" if path_suffix is None else path_suffix
        self.num_objects = num_objects
        self.num_files = num_files
        self.file_offset = file_offset
        self.random_file = random_file
        self.file_weights = file_weights
        self.precision = precision
        self.from_hdf5 = from_hdf5
        self.project = project
        self.crop = crop
        self.resize = resize
        self.convention = convention

        self.simulator = None
        if simulate_depth_noise:
            self.simulator = o3d.t.io.DepthNoiseSimulator(o3d.data.RedwoodIndoorLivingRoom1().noise_model_path)

    def get_file(self, index: int) -> int:
        epoch = index // self.num_objects
        return epoch - epoch // self.num_files + self.file_offset

    def get_depth_path(self, index: int, obj_dir: Path, file: int | None = None) -> Path:
        if file is None:
            file = self.get_file(index)
        return obj_dir / Path("kinect" if self.kinect else "depth") / f"{str(file).zfill(5)}.png"

    def get_normal_path(self, index: int, obj_dir: Path, file: int | None = None) -> Path:
        if file is None:
            file = self.get_file(index)
        return obj_dir / "normal" / f"{str(file).zfill(5)}.jpg"

    def get_color_path(self, index: int, obj_dir: Path, file: int | None = None) -> Path:
        if file is None:
            file = self.get_file(index)
        return Path(str(obj_dir).replace("fused.simple.kinect", "bproc")) / "colors" / f"{str(file).zfill(5)}.jpg"

    def load(self, obj_dir, index, category=None):
        obj_dir = Path(obj_dir)
        obj_name = obj_dir.name
        obj_category = obj_dir.parent.name
        obj_dir = obj_dir / self.path_suffix

        if self.data_dir:
            obj_dir = Path(self.data_dir) / obj_category / obj_name / self.path_suffix

        if self.random_file and self.num_files > 1:
            if self.file_weights and obj_category in self.file_weights and obj_name in self.file_weights[obj_category]:
                weights = self.file_weights[obj_category][obj_name][: self.num_files]
                file = np.random.choice(len(weights), p=weights)
            else:
                file = np.random.randint(self.num_files)
        else:
            file = self.get_file(index)

        depth_path = self.get_depth_path(index, obj_dir, file)
        normal_path = self.get_normal_path(index, obj_dir, file)
        color_path = self.get_color_path(index, obj_dir, file)

        params_path = obj_dir / "parameters.npz"
        if self.from_hdf5:
            raw_parameters = load_from_binary_hdf5(obj_dir, ["parameters.npz"])[0]
            if not isinstance(raw_parameters, Mapping):
                raise TypeError(f"Expected mapping parameters payload, got {type(raw_parameters)!r}")
            parameters = cast(Mapping[str, Any], raw_parameters)
        else:
            parameters = cast(Mapping[str, Any], np.load(params_path))
        logger.debug(f"Found keys {list(parameters.keys())} in {params_path}")
        max_depths_key = "kinect_max_depths" if self.kinect else "max_depths"
        depth_max = float(np.asarray(parameters[max_depths_key])[file])
        scale = parameters["scales"][file] if "scales" in parameters else None
        rotation = parameters["rotations"][file] if "rotations" in parameters else None
        intrinsic = np.asarray(parameters["intrinsic"])
        extrinsic = np.asarray(parameters["extrinsics"])[file]
        extrinsic = convert_extrinsic(extrinsic, "opencv", self.convention)
        pitch = pitch_from_trafo(extrinsic)

        depth_scale = 1 / depth_max if self.precision == 8 else (2**16 - 1) / depth_max
        depth = None
        normals = None
        colors = None
        if self.from_hdf5:
            if self.load_colors:
                raise NotImplementedError("Loading colors from HDF5 is not supported yet.")
            if self.load_depth and self.load_normals:
                raw_depth, raw_normals = load_from_binary_hdf5(
                    depth_path.parent.parent,
                    [depth_path.name, normal_path.name],
                    [depth_path.parent.name, normal_path.parent.name],
                )
                depth = np.asarray(raw_depth)
                normals = np.asarray(raw_normals)
            elif self.load_depth:
                depth = np.asarray(
                    load_from_binary_hdf5(depth_path.parent.parent, [depth_path.name], [depth_path.parent.name])[0]
                )
            elif self.load_normals:
                normals = np.asarray(
                    load_from_binary_hdf5(normal_path.parent.parent, [normal_path.name], [normal_path.parent.name])[0]
                )
        else:
            if self.load_depth:
                depth = np.asarray(o3d.io.read_image(str(depth_path)))
            if self.load_normals:
                normals = np.asarray(o3d.io.read_image(str(normal_path)))
            if self.load_colors:
                colors = np.asarray(o3d.io.read_image(str(color_path)))

        # FIXME: Hack to fix bad colors/normals
        if self.load_colors or self.load_normals:
            if depth is None:
                if self.from_hdf5:
                    depth = np.asarray(
                        load_from_binary_hdf5(depth_path.parent.parent, [depth_path.name], [depth_path.parent.name])[0]
                    )
                else:
                    depth = np.asarray(o3d.io.read_image(str(depth_path)))
            if self.load_colors and colors is not None:
                colors[depth == 0] = 255
            elif self.load_normals and normals is not None:
                normals[depth == 0] = 0

        if self.load_depth and depth is not None:
            shape = depth.shape
        elif self.load_colors and colors is not None:
            shape = colors.shape
        elif normals is not None:
            shape = normals.shape
        else:
            raise RuntimeError("No image modality loaded.")
        height, width = shape[:2]
        if self.crop:
            if depth is not None:
                bbox_input: np.ndarray = depth
            else:
                raise RuntimeError("Depth is required for crop bbox fallback.")
            if self.load_colors and colors is not None:
                bbox_input = np.any(colors < 230, axis=2)
            elif self.load_normals and depth is None and normals is not None:
                bbox_input = normals.sum(axis=2) > 0
            bbox = bbox_from_mask(bbox_input, padding=0.1)

            if self.load_depth and depth is not None:
                _depth = np.asarray(crop_and_resize_image(depth, box=bbox))
                if np.all(_depth == 0):
                    logger.warning(f"Depth {depth_path} is empty after cropping.")
                    bbox = bbox_from_mask(depth, padding=0.1)
                    depth = np.asarray(crop_and_resize_image(depth, box=bbox))
                else:
                    depth = _depth
            if normals is not None:
                normals = np.asarray(crop_and_resize_image(normals, box=bbox))
            if colors is not None:
                colors = np.asarray(crop_and_resize_image(colors, box=bbox, color="white"))
            intrinsic = adjust_intrinsic(intrinsic, width, height, box=bbox)
            if self.load_depth and depth is not None:
                shape = depth.shape
            elif self.load_colors and colors is not None:
                shape = colors.shape
            elif normals is not None:
                shape = normals.shape
            else:
                raise RuntimeError("No image modality loaded after crop.")
            height, width = shape[:2]
        if self.resize:
            if depth is not None:
                depth = np.asarray(crop_and_resize_image(depth, size=self.resize, interpolation="nearest"))
            if normals is not None:
                normals = np.asarray(crop_and_resize_image(normals, size=self.resize))
            if colors is not None:
                colors = np.asarray(crop_and_resize_image(colors, size=self.resize))
            intrinsic = adjust_intrinsic(intrinsic, width, height, size=self.resize)
            if self.load_depth and depth is not None:
                shape = depth.shape
            elif self.load_colors and colors is not None:
                shape = colors.shape
            elif normals is not None:
                shape = normals.shape
            else:
                raise RuntimeError("No image modality loaded after resize.")
            height, width = shape[:2]

        data: dict[str | None, Any] = {
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "pitch": pitch,
            "width": width,
            "height": height,
            "name": obj_name,
            "path": (color_path if normals is None else normal_path) if depth is None else depth_path,
            "file": file,
        }
        if self.load_colors and colors is not None:
            data["image"] = colors
            data["mask"] = colors.mean(axis=2) != 255
        elif self.load_normals and normals is not None:
            data["image"] = normals
        if self.load_normals and normals is not None:
            data["normals"] = normals / 255
            data["mask"] = normals.sum(axis=2) > 0
        if self.load_depth and depth is not None:
            data["depth"] = depth / depth_scale
            data["mask"] = depth > 0

        inputs: np.ndarray
        if self.project:
            if self.load_normals or self.load_colors:
                if depth is None:
                    raise RuntimeError("Depth is required for RGBD projection.")
                if self.load_normals:
                    if normals is None:
                        raise RuntimeError("Normals are required for normal projection.")
                    rgb = normals.copy()
                else:
                    if colors is None:
                        raise RuntimeError("Colors are required for RGB projection.")
                    rgb = colors.copy()
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(rgb),
                    o3d.geometry.Image(depth),
                    depth_scale=depth_scale,
                    depth_trunc=float("inf"),
                    convert_rgb_to_intensity=False,
                )

                depth_image = rgbd_image.depth
                if self.simulator is not None and torch.cuda.is_available():
                    depth_cuda = o3d.t.geometry.Image.from_legacy(depth_image).cuda()
                    depth_image = self.simulator.simulate(depth_cuda, depth_scale=1.0).cpu().to_legacy()

                pcd = convert_rgbd_image_to_point_cloud(
                    [rgbd_image.color, depth_image],
                    intrinsic,
                    extrinsic,
                    depth_scale=1.0,
                    depth_trunc=float("inf"),
                    convert_rgb_to_intensity=False,
                )
                if self.load_normals:
                    pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.colors) * 2 - 1)
                    normals = np.asarray(pcd.normals)
                elif self.load_colors:
                    colors = np.asarray(pcd.colors)

                inputs = np.asarray(pcd.points)
            else:
                if depth is None:
                    raise RuntimeError("Depth is required for depth projection.")
                pcd = convert_depth_image_to_point_cloud(
                    o3d.geometry.Image(depth), intrinsic, extrinsic, depth_scale=depth_scale, depth_trunc=float("inf")
                )
                inputs = np.asarray(pcd.points)
        elif self.load_depth:
            if depth is None:
                raise RuntimeError("Depth is required when load_depth is enabled.")
            inputs = np.asarray(depth / depth_scale)
        else:
            if self.load_colors and colors is not None:
                inputs = np.asarray(colors)
            elif normals is not None:
                inputs = np.asarray(normals)
            else:
                raise RuntimeError("No inputs available for non-projected mode.")

        data[None] = inputs
        if self.project:
            if len(inputs) == 0:
                logger.warning(f"Depth {depth_path} is empty after projection.")
                data[None] = np.zeros((1, 3))
            if self.load_normals:
                data["normals"] = normals
            elif self.load_colors:
                data["colors"] = colors

        # Undo object rotation applied during rendering
        if self.unrotate and rotation is not None:
            data["rotation"] = rotation
            data["extrinsic"][:3, :3] = extrinsic[:3, :3] @ rotation.T
            if self.project:
                data[None] = data[None] @ rotation
                if "normals" in data:
                    normals = data["normals"] @ rotation
                    data["normals"] = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # Undo object scaling applied during rendering
        if self.unscale and scale is not None:
            scale = scale if isinstance(scale, np.ndarray) else np.array((scale,) * 3)
            data["scale"] = scale
            data["extrinsic"] = extrinsic @ np.diag([*scale, 1])
            if self.project:
                data[None] /= scale

        return copy.deepcopy(data)


class BlenderProcRGBDField(Field):
    def __init__(
        self,
        data_dir: str | None = None,
        unscale: bool = True,
        undistort: bool = True,
        path_prefix: str = "",
        num_objects: int = 1,
        num_shards: int = 1,
        files_per_shard: int = 1,
        random_file: bool = False,
        random_shard: bool = False,
        depth_trunc: int = 10,
        fuse: int | tuple[int, int] = 0,
        fuse_thresholds: tuple[float, float, float] = (0.3, 1.0, 0.1),
        input_type: str | None = None,
        swap_xy: bool = True,
        load_fixed: bool = False,
    ):
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

        if swap_xy:
            self.swap_xy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        else:
            self.swap_xy = np.eye(3)
        self.swap_yz = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    def _get_points_from_depth(self, depth, camera_intrinsic, camera_extrinsic):
        pcd = convert_depth_image_to_point_cloud(
            depth, camera_intrinsic, camera_extrinsic, depth_trunc=self.depth_trunc
        )
        pcd.rotate(self.swap_yz, center=(0, 0, 0))
        return np.asarray(pcd.points)

    @staticmethod
    def get_pitch_from_rot(rot: np.ndarray) -> float:
        pitch = np.arctan2(rot.T[2, 1], rot.T[2, 2])
        return -np.rad2deg(pitch - np.pi if pitch > 0 else pitch)  # Todo: why?

    def get_shard_path(self, index: int, obj_dir: str, shard: int | None = None) -> str:
        if shard is None:
            shard = self.get_shard(index)
        return os.path.join(obj_dir, self.data_dir, "train_pbr", str(shard).zfill(6))

    def get_shard(self, index: int) -> int:
        epoch = index // self.num_objects
        return epoch // self.files_per_shard

    def get_file(self, index: int) -> int:
        epoch = index // self.num_objects
        return epoch - epoch // self.files_per_shard * self.files_per_shard

    def get_depth_path(self, index: int, obj_dir: str, shard: int | None = None, file: int | None = None) -> str:
        if shard is None:
            shard = self.get_shard(index)
        if file is None:
            file = self.get_file(index)
        shard_path = self.get_shard_path(index, obj_dir, shard)
        return os.path.join(shard_path, "depth", str(file).zfill(6) + ".png")

    def load(self, obj_dir, index, category=None):
        obj_dir = str(obj_dir)
        if self.path_prefix:
            synthset = obj_dir.split("/")[-2]
            model = obj_dir.split("/")[-1]
            obj_dir = os.path.join(self.path_prefix, synthset, model)
        obj_name = obj_dir.split("/")[-1]
        path_to_camera_json = os.path.join(obj_dir, self.data_dir, "camera.json")

        shard = self.get_shard(index)
        file = self.get_file(index)

        if self.random_shard and self.num_shards > 1:
            shard = np.random.randint(self.num_shards)
        if self.random_file and self.files_per_shard > 1:
            file = np.random.randint(self.files_per_shard)

        shard_path = self.get_shard_path(index, obj_dir, shard)
        path_to_scene_camera_json = os.path.join(shard_path, "scene_camera.json")

        camera_parameters = get_camera_parameters_from_blenderproc_bopwriter(
            path_to_scene_camera_json, path_to_camera_json, scene_id=-1 if self.fuse else file
        )
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

        if self.input_type in ["image", "rgbd"]:
            image = Image.open(depth_path.replace("depth", "rgb").replace(".png", ".jpg")).convert("RGB")
            if self.input_type == "image":
                inputs = image

        if self.input_type != "image":
            inputs = self._get_points_from_depth(depth_path, intrinsic, extrinsic)
            pitch = pitch_from_trafo(extrinsic)

            for n in n_closest:
                depth_path = os.path.join(shard_path, "depth", str(n).zfill(6) + ".png")
                intrinsic = camera_parameters[n].intrinsic
                extrinsic = camera_parameters[n].extrinsic
                inputs = np.concatenate([inputs, self._get_points_from_depth(depth_path, intrinsic, extrinsic)])

        data = {
            None: inputs,
            "intrinsic": intrinsic.intrinsic_matrix,
            "extrinsic": extrinsic,
            "pitch": pitch,
            "name": obj_name,
            "path": depth_path,
        }
        if self.input_type == "rgbd":
            data["image"] = image

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

        return copy.deepcopy(data)


class RGBDField(Field):
    def __init__(
        self,
        rgb: bool = False,
        high_res_rgb: bool = False,
        mask: bool = False,
        crop: float = 0.2,
        cam_id_range: tuple[int, int] = (1, 5),
        merge_cam_id_range: tuple[int, int] | None = None,
        merge_angles: int = 0,
        stride: int = 1,
        filter_discontinuities: int | None = None,
    ):
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

        self.depth_shape = 480, 640
        self.rgb_shape = 1024, 1280
        self.high_res_rgb_shape = 2848, 4272

        self.rotation_angle = 3
        self.num_rotations = 360 // self.rotation_angle
        self.max_index = self.num_rotations * max(self.cam_id_range)
        self.min_index = self.num_rotations * (min(self.cam_id_range) - 1)
        self.num_images = self.max_index - self.min_index

        self.rot_x_180 = R.from_euler("x", np.pi).as_matrix()

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
        r_2 = x_p**2 + y_p**2
        r_4 = x_p**4 + y_p**4
        r_6 = x_p**6 + y_p**6

        # Fix radial distortion
        radial = 1 + k1 * r_2 + k2 * r_4 + k3 * r_6
        x_r = x_p * radial
        y_r = y_p * radial

        # Fix tangential distortion
        x_t = x_r + (2 * p1 * x_r * y_r + p2 * (r_2 + 2 * x_r**2))
        y_t = y_r + (p1 * (r_2 + 2 * y_r**2) + 2 * p2 * x_r * y_r)
        _points[:, 0] = z * x_t
        _points[:, 1] = z * y_t
        return _points

    def get_indices(self, index: int) -> tuple[int, int]:
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

        def _h5_array(handle: Any, key: str) -> np.ndarray:
            return np.asarray(handle[key][:])

        calibration = h5py.File(calibration_path)
        if self.high_res_rgb:
            rgb_intrinsics = _h5_array(calibration, f"N{cam_index}_rgb_K")
        else:
            rgb_intrinsics = _h5_array(calibration, f"NP{cam_index}_rgb_K")
        depth_intrinsics = _h5_array(calibration, f"NP{cam_index}_depth_K")
        depth_scale = float(_h5_array(calibration, f"NP{cam_index}_ir_depth_scale")[0]) * 1e-4  # 100um to meters
        if self.high_res_rgb:
            f_rgb_ref = _h5_array(
                calibration, f"H_N{cam_index}_from_NP5"
            )  # N[1...5] RGB cam pose in NP5 RGB cam coords
        else:
            f_rgb_ref = _h5_array(
                calibration, f"H_NP{cam_index}_from_NP5"
            )  # NP[1-5] RGB cam pose in NP5 RGB cam coords
        f_ir_ref = _h5_array(
            calibration, f"H_NP{cam_index}_ir_from_NP5"
        )  # NP[1-5] depth cam poise in NP5 RGB cam coords
        calibration.close()

        f_ref_rgb = np.asarray(inv_trafo(f_rgb_ref))  # NP5 RGB cam pose in N/NP[1-5] RGB cam coords
        f_ref_ir = np.asarray(inv_trafo(f_ir_ref))  # NP5 RGB cam pose in NP[1-5] depth cam coords
        f_rgb_ir = f_rgb_ref @ f_ref_ir  # N/NP[1-5] RGB cam pose in NP[1-5] depth cam coords

        pose_data = h5py.File(pose_path)
        f_table_ref = _h5_array(pose_data, "H_table_from_reference_camera")  # Table pose in NP5 coords
        f_ir_table = np.asarray(inv_trafo(f_table_ref @ f_ref_ir))  # NP[1-5] depth cam pose in table coords
        f_rgb_table = np.asarray(inv_trafo(f_table_ref @ f_ref_rgb))  # N/NP[1-5] RGB cam pose in table coords
        pose_data.close()

        # table_data = h5py.File(os.path.join(obj_dir, "poses", "turntable.h5"))
        # print([v[:] for v in table_data.values()])
        # table_data.close()

        depth_data = h5py.File(depth_path)
        depth = _h5_array(depth_data, "depth")
        depth_data.close()

        if self.filter_discontinuities:
            depth = np.asarray(_filter_discontinuities(depth, filter_size=self.filter_discontinuities))
        depth *= depth_scale

        return depth, depth_intrinsics, rgb_intrinsics, f_rgb_ir, f_ir_table, f_rgb_table

    def load_single(self, obj_dir, index):
        cam_index, angle = self.get_indices(index)
        rgb_shape = self.high_res_rgb_shape if self.high_res_rgb else self.rgb_shape
        depth, depth_intrinsics, rgb_intrinsics, f_rgb_ir, f_ir_table, f_rgb_table = self.load_data(
            obj_dir, cam_index, angle
        )

        # Project depth_image to point cloud
        depth_extrinsics = np.eye(4) if self.rgb or self.mask else np.asarray(f_ir_table)
        pcd = convert_depth_image_to_point_cloud(
            depth,
            camera_intrinsic=depth_intrinsics,
            camera_extrinsic=depth_extrinsics,
            depth_scale=1,
            depth_trunc=3.5,
            stride=1,
        )

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
                depth_mask = np.asarray(Image.open(depth_mask_path).convert("L"), dtype=bool)
                depth[depth_mask] = 0

            if self.rgb:
                rgb_path = os.path.join(obj_dir, base_path + ".jpg")
                rgb = np.asarray(Image.open(rgb_path))
                pcd = convert_rgbd_image_to_point_cloud(
                    [rgb, depth],
                    camera_intrinsic=rgb_intrinsics,
                    camera_extrinsic=f_rgb_table,
                    depth_scale=1,
                    depth_trunc=3.5,
                    convert_rgb_to_intensity=False,
                )
            else:
                pcd = convert_depth_image_to_point_cloud(
                    depth,
                    camera_intrinsic=rgb_intrinsics,
                    camera_extrinsic=f_rgb_table,
                    depth_scale=1,
                    depth_trunc=3.5,
                    stride=1,
                )

        if self.crop:
            pcd = pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-self.crop, -self.crop, -0.01), max_bound=(self.crop, self.crop, 0.75)
                )
            )

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        trafo = f_rgb_table if self.rgb or self.mask else f_ir_table
        rot = self.rot_x_180 @ trafo[:3, :3]
        pitch = -np.arctan2(rot.T[2, 1], rot.T[2, 2])
        pitch += np.pi / 2  # Switch z-axis up to y-axis up
        pitch = np.rad2deg(pitch)

        data = {None: points, "pitch": pitch}
        if self.rgb:
            data["colors"] = colors
        return data

    def get_merged_cams(self, obj_dir, index):
        if self.merge_cam_id_range is None:
            raise RuntimeError("merge_cam_id_range must be set when merging cameras.")
        points = list()
        colors = list()
        pitch = 0
        start, end = self.merge_cam_id_range
        for cam_index in range(start - 1, end):
            data = self.load_single(obj_dir, index + cam_index * self.num_rotations)
            points.append(data[None])
            if "colors" in data:
                colors.append(data["colors"])
            if cam_index == (start - 1) + (end - start + 1) // 2:
                pitch = data["pitch"]
        data = {None: np.concatenate(points), "pitch": pitch}
        if colors:
            data["colors"] = np.concatenate(colors)
        return data

    def get_merged_angles(self, obj_dir, index):
        points = list()
        colors = list()
        pitch = 0
        _cam_index, angle = self.get_indices(index)
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
                pitch = data["pitch"]
        data = {None: np.concatenate(points), "pitch": pitch}
        if colors:
            data["colors"] = np.concatenate(colors)
        return data

    def load(self, obj_dir, index, category=None):
        obj_dir = str(obj_dir)
        if self.merge_cam_id_range or self.merge_angles > 1:
            if self.merge_angles:
                data = self.get_merged_angles(obj_dir, index)
            else:
                data = self.get_merged_cams(obj_dir, index)
        else:
            data = self.load_single(obj_dir, index)

        data["name"] = obj_dir.split("/")[-1]
        data["path"] = obj_dir

        return copy.deepcopy(data)


class PointCloudField(Field):
    def __init__(
        self,
        file: str = "pointcloud.npz",
        data_dir: str | None = None,
        cam_dir: str | None = None,
        index: tuple[int, int] | int = -1,
        num_cams: int = 24,
        normals_file: str | None = None,
        load_normals: bool = False,
        pose_file: str | None = None,
        min_max_only: bool = False,
        from_hdf5: bool = False,
    ):
        super().__init__()

        if load_normals and min_max_only:
            logger.warning("Normals and min_max_only not supported yet")
            load_normals = False

        self.file = file
        self.data_dir = data_dir
        self.cam_dir = cam_dir
        self.index = index
        self.num_cams = num_cams
        self.normals_file = normals_file
        self.load_normals = load_normals
        self.pose_file = pose_file
        self.min_max_only = min_max_only
        self.from_hdf5 = from_hdf5
        self.load_file = _load_pointcloud

    def load(self, obj_dir, index, category=None):
        obj_dir = Path(obj_dir)
        obj_name = obj_dir.name
        obj_category = obj_dir.parent.name
        if self.data_dir:
            obj_dir = Path(self.data_dir) / obj_category / obj_name
        file_path = obj_dir / self.file

        points, normals = self.load_file(
            obj_dir, self.file, self.normals_file, self.load_normals, from_hdf5=self.from_hdf5
        )

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
            points = np.array([points.min(axis=0), points.max(axis=0)])

        data = {None: points, "name": obj_name, "path": file_path}
        if normals is not None:
            data["normals"] = normals

        if self.cam_dir:
            cam_path = obj_dir / self.cam_dir / "cameras.npz"
            assert cam_path.is_file(), f"Camera file {cam_path} if not a file or does not exist."
            data.update(_load_cam(index, self.index, self.num_cams, cam_path))

        return copy.deepcopy(data)


class PointsField(Field):
    def __init__(
        self,
        file: str | Path | list[str] | list[Path] = "points.npz",
        data_dir: str | None = None,
        pose_file: str | None = None,
        params_dir: str | None = None,
        path_suffix: str | None = None,
        padding: float = 0.1,
        num_points: int = int(1e5),
        occ_from_sdf: bool = True,
        tsdf: float = 0,
        sigmas: list[float] | tuple[float, ...] | None = None,
        spheres: list[float] | tuple[float, ...] | None = None,
        crop: bool = False,
        normalize: bool = True,
        load_all_files: bool = False,
        load_random_file: bool = False,
        load_surface_file: bool | float | None = True,
        from_hdf5: bool = False,
    ):
        super().__init__(cachable=load_all_files)

        if sigmas is not None:
            assert all([sigma > 0 for sigma in sigmas]), "Sigmas must be positive"
        if spheres is not None:
            assert all([sphere > 0 for sphere in spheres]), "Spheres must be positive"
        assert not (occ_from_sdf and tsdf), "Cannot have occupancy and TSDF at the same time"

        self.file = [file] if isinstance(file, (str, Path)) else file
        self.data_dir = data_dir
        self.pose_file = pose_file
        self.params_dir = params_dir
        self.path_suffix = "" if path_suffix is None else path_suffix
        self.occ_from_sdf = occ_from_sdf
        self.tsdf = tsdf
        self.sigmas = sigmas
        self.spheres = spheres
        self.padding = padding
        self.num_points = num_points
        self.crop = crop
        self.normalize = normalize
        self.load_all_files = load_all_files
        self.load_random_file = load_random_file
        self.load_surface_file = False if load_surface_file is None else load_surface_file
        self.from_hdf5 = from_hdf5

    def load(self, obj_dir, index, category=None):
        obj_dir = Path(obj_dir)
        obj_name = obj_dir.name
        obj_category = obj_dir.parent.name
        obj_dir = obj_dir / self.path_suffix

        if self.data_dir:
            obj_dir = Path(self.data_dir) / obj_category / obj_name / self.path_suffix

        file = self.file if self.file else [obj_name + ".ply"]
        file_path = [obj_dir / f for f in file]

        if any(fp.suffix not in [".npy", ".npz"] for fp in file_path):
            raise TypeError(f"File type {file_path[0].suffix} is not supported yet.")

        else:
            points_list = list()
            occupancy_list = list()
            if not self.load_all_files:
                files = [file_path[0]]
                if self.load_surface_file:
                    assert len(file_path) > 1 and "surface" in str(file_path[1]), "Second file must be 'surface'"
                    if np.random.random() < self.load_surface_file:  # Always True if load_surface_file is 1 (i.e. True)
                        files.append(file_path[1])
                if self.load_random_file:
                    assert len(file_path) > 2, "Need at lest 2 files to load a random file"
                    assert "uniform_random" in str(file_path[0]), "First file must be 'uniform_random'"
                    files.append(random.choice(file_path[1:]))
                elif len(file_path) in [7, 8]:
                    assert "uniform_random" in str(file_path[0]), "First file must be 'uniform_random'"
                    if len(file_path) == 8:
                        assert "surface_random" in str(file_path[1]), "Second file must be 'surface_random'"

                    scale = 1
                    if isinstance(category, (int, np.integer)) and category >= 0:
                        params_dir = obj_dir
                        if self.params_dir is not None:
                            params_dir = Path(self.params_dir) / obj_category / obj_name
                        logger.debug(f"{self.name}: Loading parameters from {params_dir / 'parameters.npz'}.")
                        if self.from_hdf5:
                            raw_parameters = load_from_binary_hdf5(params_dir, ["parameters.npz"])[0]
                            if not isinstance(raw_parameters, Mapping):
                                raise TypeError(f"Expected mapping parameters payload, got {type(raw_parameters)!r}")
                            parameters = cast(Mapping[str, Any], raw_parameters)
                        else:
                            parameters = cast(Mapping[str, Any], np.load(params_dir / "parameters.npz"))
                        scale = np.asarray(parameters["scales"])[category]

                    if isinstance(scale, np.ndarray):
                        scale = max(scale)
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
                points, occupancy = _load_points(
                    file_path=path, occ_from_sdf=self.occ_from_sdf, tsdf=self.tsdf, from_hdf5=self.from_hdf5
                )
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

            if self.crop:
                mask = np.all(np.abs(points) <= 0.5 + self.padding / 2, axis=1)
                if mask.sum() > 0:
                    points = points[mask]
                    occupancy = occupancy[mask]

        data = {None: points, "occ": occupancy.astype(bool), "name": obj_name, "path": file_path}

        return copy.deepcopy(data)


class VoxelsField(Field):
    def __init__(self, file: str = "model.binvox", use_trimesh: bool = True):
        super().__init__()

        self.file = file
        self.use_trimesh = use_trimesh

    def load(self, obj_dir, index, category=None):
        obj_dir = str(obj_dir)
        file_path = os.path.join(obj_dir, self.file)

        with open(file_path, "rb") as f:
            if self.use_trimesh:
                voxels = np.asarray(trimesh_binvox.load_binvox(f).matrix)
            else:
                voxels = binvox.Binvox.read(f, mode="dense").data

        data = {None: voxels, "name": obj_dir.split("/")[-1], "path": file_path}

        return copy.deepcopy(data)


class MeshField(Field):
    def __init__(
        self,
        file: str | None = None,
        file_prefix: str | None = None,
        data_dir: str | None = None,
        pose_file: str | None = None,
        process: bool = False,
        from_hdf5: bool = False,
        path_only: bool = False,
        geometry_only: bool = True,
    ):
        super().__init__()

        self.file = file
        self.file_prefix = file_prefix if file_prefix else ""
        self.data_dir = data_dir
        self.pose_file = pose_file
        self.process = process
        self.from_hdf5 = from_hdf5
        self.path_only = path_only
        self.geometry_only = geometry_only

    def load(self, obj_dir, index, category=None):
        obj_dir = Path(obj_dir)
        obj_name = obj_dir.name
        obj_category = obj_dir.parent.name
        if self.data_dir:
            obj_dir = Path(self.data_dir) / obj_category / obj_name
        file_path = obj_dir / self.file if self.file else obj_dir / (self.file_prefix + obj_name + ".ply")

        if self.path_only:
            return {"name": obj_name, "path": file_path}

        if self.from_hdf5:
            mesh = cast(Any, load_from_binary_hdf5(file_path.parent, [file_path.name])[0]).as_open3d
        else:
            if self.geometry_only:
                mesh = Trimesh(*load_mesh(file_path), process=False, validate=False)
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
                )
            else:
                mesh = o3d.io.read_triangle_mesh(str(file_path))

        if self.process and not mesh.is_watertight():
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

        data = {"vertices": np.asarray(mesh.vertices, dtype=np.float32), "triangles": np.asarray(mesh.triangles)}
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

        data.update({"name": obj_name, "path": file_path})

        return copy.deepcopy(data)


class PartNetField(Field):
    def __init__(self, data_dir: str | Path):
        super().__init__()

        self.data_dir = Path(data_dir)

    def load(self, obj_dir, index, category=None):
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
        points /= np.ptp(bounds, axis=0).max()

        data = {"points": points, "labels": labels, "name": obj_name, "path": points_path}

        return copy.deepcopy(data)


class DTUField(Field):
    """Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): image folder name
        mask_folder_name (str): mask folder name
        depth_folder_name (str): depth folder name
        visual_hull_depth_folder (str): visual hull depth folder name
        transform (transform): transformations applied to images
        extension (str): image extension
        mask_extension (str): mask extension
        depth_extension (str): depth extension
        with_camera (bool): whether camera data should be provided
        with_mask (bool): whether object masks should be provided
        with_depth (bool): whether depth maps should be provided
        random_view (bool): whether a random view should be used
        all_images (bool): whether all images should be returned (instead of
            one); only used for rendering
        n_views (int): number of views that should be used; if < 1, all views
            in the folder are used
        depth_from_visual_hull (bool): whether the visual hull depth map
            should be provided
        ignore_image_idx (list): list of IDs which should be ignored (only
            used for the multi-view reconstruction experiments)
    """

    def __init__(
        self,
        image_folder_name: str = "image",
        mask_folder_name: str = "mask",
        depth_folder_name: str = "depth",
        image_extension: str = "png",
        mask_extension: str = "png",
        depth_extension: str = "exr",
        load_depth: bool = True,
        random_view: bool = False,
        n_views: int = 0,
        ignore_image_idx: tuple[int, ...] | None = None,
    ):
        super().__init__(cachable=not random_view)
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.depth_folder_name = depth_folder_name
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        self.depth_extension = depth_extension
        self.random_view = random_view
        self.n_views = n_views
        self.load_depth_enabled = load_depth
        self.ignore_image_idx = ignore_image_idx

    def get_number_files(self, image_path: Path, ignore_filtering: bool = False):
        """Returns how many views are present for the model.

        Args:
            model_path (str): path to model
            ignore_filtering (bool): whether the image filtering should be
                ignored
        """
        files = image_path.glob(f"*.{self.image_extension}")
        files = sorted(files)

        if not ignore_filtering and self.ignore_image_idx:
            files = [files[i] for i in range(len(files)) if i not in self.ignore_image_idx]

        if not ignore_filtering and self.n_views > 0:
            files = files[: self.n_views]
        return len(files)

    def return_idx_filename(self, obj_dir: Path, folder_name: str, extension: str, index: int) -> Path:
        """Loads the "index" filename from the folder.

        Args:
            model_path (str): path to model
            folder_name (str): name of the folder
            extension (str): string of the extension
            index (int): ID of data point
        """
        files = (obj_dir / folder_name).glob(f"*.{extension}")
        files = sorted(files)

        if self.ignore_image_idx:
            files = [files[i] for i in range(len(files)) if i not in self.ignore_image_idx]

        if self.n_views > 0:
            files = files[: self.n_views]
        return files[index]

    def load_image(self, obj_dir: Path, index: int) -> np.ndarray:
        """Loads an image.

        Args:
            model_path (str): path to model
            index (int): ID of data point
            data (dict): data dictionary
        """
        filename = self.return_idx_filename(obj_dir, self.image_folder_name, self.image_extension, index)
        return np.asarray(Image.open(filename).convert("RGB")).astype(np.uint8)

    def load_camera(self, obj_dir: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Loads an image.

        Args:
            model_path (str): path to model
            index (int): ID of data point
            data (dict): data dictionary
        """
        import cv2

        if self.ignore_image_idx:
            n_files = self.get_number_files(obj_dir / self.image_folder_name, ignore_filtering=True)
            idx_list = [i for i in range(n_files) if i not in self.ignore_image_idx]
            index = idx_list[index]

        camera_file = obj_dir / "cameras.npz"
        camera_dict = np.load(camera_file)
        Rt = camera_dict[f"world_mat_{index}"]
        S = camera_dict[f"scale_mat_{index}"]
        P = Rt @ S

        out = cv2.decomposeProjectionMatrix(P[:3, :])
        K = out[0]
        R = out[1]
        t = out[2]

        K /= K[2, 2]

        pose = np.eye(4)
        pose[:3, :3] = R.T
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        return K.astype(np.float32), np.linalg.inv(pose).astype(np.float32)

    def load_mask(self, obj_dir: Path, idx: int) -> np.ndarray:
        """Loads an object mask.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        """
        filename = self.return_idx_filename(obj_dir, self.mask_folder_name, self.mask_extension, idx)
        return np.asarray(Image.open(filename).convert("L")).astype(np.bool)

    def load_depth(self, obj_dir: Path, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Loads a depth map.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        """
        import cv2

        filename = self.return_idx_filename(obj_dir, self.depth_folder_name, self.depth_extension, idx)
        depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth image: {filename}")
        return depth.astype(np.float32), np.isfinite(depth)

    def _get_original_index(self, index: int) -> int:
        """Maps filtered index back to original index when ignore_image_idx is set.

        Args:
            index (int): filtered index

        Returns:
            int: original index
        """
        if self.ignore_image_idx:
            # Get total number of files without filtering
            all_files = list(
                range(self.get_number_files(Path("dummy") / self.image_folder_name, ignore_filtering=True))
            )
            # Remove ignored indices
            valid_indices = [i for i in all_files if i not in self.ignore_image_idx]
            # Apply n_views filtering if needed
            if self.n_views > 0:
                valid_indices = valid_indices[: self.n_views]
            return valid_indices[index]
        return index

    def load(self, obj_dir, index, category):
        """Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
            input_idx_img (int): image id which should be used (this
                overwrites any other id). This is used when the fields are
                cached.
        """

        model_path = Path(obj_dir)
        filename = self.return_idx_filename(model_path, self.image_folder_name, self.image_extension, index)
        image = self.load_image(model_path, index)
        mask = self.load_mask(model_path, index)
        intrinsic, extrinsic = self.load_camera(model_path, index)
        image[~mask] = 0

        height, width = image.shape[:2]
        data = {
            None: image,
            "mask": mask,
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "width": width,
            "height": height,
            "name": filename.name,
            "path": filename,
        }

        if self.load_depth_enabled:
            data["image"] = image
            depth_raw, depth_mask = self.load_depth(model_path, index)
            depth = depth_raw.copy()
            depth[~mask | ~depth_mask] = 0

            if self.ignore_image_idx:
                n_files = self.get_number_files(obj_dir / self.image_folder_name, ignore_filtering=True)
                idx_list = [i for i in range(n_files) if i not in self.ignore_image_idx]
                index = idx_list[index]
            cam = np.load(model_path / "cameras.npz")
            cm = cam[f"camera_mat_inv_{index}"]
            wm = cam[f"world_mat_inv_{index}"]
            sm = cam[f"scale_mat_inv_{index}"]

            u, v = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
            u, v = u.flatten().astype(np.float32), v.flatten().astype(np.float32)

            u_norm = (u / (width - 1)) * 2 - 1
            v_norm = (v / (height - 1)) * 2 - 1
            uv_coords = np.stack((u_norm, v_norm), axis=1)
            xyz = np.concatenate((uv_coords, np.ones((width * height, 1))), axis=1)

            p_depth = xyz * depth.reshape(-1, 1)
            P = sm @ wm @ cm
            p_depth_h = np.concatenate((p_depth, np.ones((width * height, 1))), axis=1)
            p_world = P @ p_depth_h.transpose()
            points = p_world.transpose()[:, :3]

            points = apply_trafo(points[depth.flatten() > 0], extrinsic)
            depth = np.asarray(points_to_depth(points, intrinsic, width, height), dtype=np.float32)
            depth[mask & ~depth_mask] = depth_raw[mask & ~depth_mask]  # Fill in missing depth values
            data[None] = depth

        return copy.deepcopy(data)
