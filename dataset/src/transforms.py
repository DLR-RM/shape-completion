import os
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Tuple, Union, Dict, List, Optional, Any

import numpy as np
import open3d as o3d
import pyrender
import torch
from PIL import Image
from easy_o3d.utils import (process_point_cloud, DownsampleTypes, OutlierTypes, SearchParamTypes,
                            convert_depth_image_to_point_cloud, convert_rgbd_image_to_point_cloud)
from einops import rearrange
from joblib import Parallel, delayed
from mathutils import Vector, Euler
from matplotlib.cm import get_cmap
from pykdtree.kdtree import KDTree
from pyrender.shader_program import ShaderProgramCache
from pytorch3d.renderer import (RasterizationSettings, MeshRasterizer, PerspectiveCameras)
from pytorch3d.structures import Meshes
from scipy.spatial import Delaunay, KDTree as SKDTree
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import Compose, Resize, Normalize as NormalizeImage, ToTensor, ToPILImage
from trimesh import Trimesh

from libs import check_mesh_contains
from utils import (subsample, generate_random_basis, rot_from_euler, make_3d_grid, inv_trafo, to_tensor, normalize,
                   setup_logger, points_to_coordinates, get_partnet_colors)

logger = setup_logger(__name__)


class Transform(ABC):
    @abstractmethod
    def __call__(self, data: Dict[Union[str, None], Any]) -> Dict[Union[str, None], Any]:
        raise NotImplementedError("Transform is an abstract class and cannot be called directly.")

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def message(self):
        return f"Applying {self.name} transform"


def apply_transform(data: Dict[Any, np.ndarray],
                    transform: Optional[Union[List[Transform], Transform, Compose]] = None,
                    cache: bool = False):
    if transform is not None:
        if isinstance(transform, list):
            for t in transform:
                trafo_timer = time()
                if not cache or (cache and hasattr(t, "cache") and t.cache is not None):
                    data = t(data)
                    logger.debug(f"Transform {t.name} takes {time() - trafo_timer:.4f}s.")
        elif isinstance(transform, (Transform, Compose)) and not cache:
            trafo_timer = time()
            data = transform(data)
            if isinstance(transform, Transform):
                logger.debug(f"Transform {transform.name} takes {time() - trafo_timer:.4f}s.")
            else:
                logger.debug(f"Compose transform takes {time() - trafo_timer:.4f}s.")
        else:
            raise TypeError(f"Transform must be a list of transforms, a single transform or a Compose object.")
    return data


class ReturnTransform(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        return data


class RandomChoice(Transform):
    def __init__(self, transforms: List[Transform]):
        super().__init__()
        print(self.message, f"(transforms: {[t.name for t in transforms]})")
        self.transforms = transforms

    def __call__(self, data):
        transform = np.random.choice(self.transforms)
        return transform(data)


class RandomApply(Transform):
    def __init__(self, transform: Transform, p: float = 0.5):
        super().__init__()
        print(self.message, f"(transform: {transform.name}, p: {p})")
        self.transform = transform
        self.p = p

    def __call__(self, data):
        if np.random.rand() < self.p:
            return self.transform(data)
        return data


class NeRFEncoding(Transform):
    def __init__(self,
                 dim: int = 3,
                 num_freqs: int = 10,
                 min_freq_exp: int = 0,
                 max_freq_exp: int = 8,
                 include_inputs: bool = True,
                 normalize_inputs: bool = True,
                 scale_inputs: bool = True,
                 apply_to: Union[List[str], Tuple[str, ...]] = ("inputs",),
                 replace_key: bool = True,
                 padding: float = 0.1):
        super().__init__()
        print(self.message, f"(in_dim: {dim}, num_frequencies: {num_freqs}, "
                            f"min_freq_exp: {min_freq_exp}, max_freq_exp: {max_freq_exp}, "
                            f"include_inputs: {include_inputs}, apply_to: {apply_to})")

        assert dim >= 1, "Input dimension must be at least 1"
        assert all([key in ["inputs", "points", "pointcloud"] for key in apply_to]), \
            "Invalid key in 'apply_to' argument"

        self.dim = dim
        self.num_freq = num_freqs
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_inputs = include_inputs
        self.normalize_inputs = normalize_inputs
        self.scale_inputs = scale_inputs
        self.apply_to = apply_to
        self.replace_key = replace_key
        self.padding = padding
        self.out_dim = self.get_out_dim()

    def get_out_dim(self):
        out_dim = self.dim * self.num_freq * 2
        if self.include_inputs:
            out_dim += self.dim
        return out_dim

    def apply(self, points: np.ndarray, covs: Optional[np.ndarray] = None):
        coords = points_to_coordinates(points, self.padding) if self.normalize_inputs else points
        scaled_in_tensor = 2 * np.pi * coords
        freqs = 2 ** np.linspace(self.min_freq, self.max_freq, self.num_freq)
        scaled_inputs = scaled_in_tensor[..., None] * freqs
        scaled_inputs = scaled_inputs.reshape(*scaled_inputs.shape[:-2], -1)

        if covs is None:
            encoded_inputs = np.sin(np.concatenate([scaled_inputs, scaled_inputs + np.pi / 2.0], axis=-1))
        else:
            input_var = np.diagonal(covs, axis1=-2, axis2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = self.expected_sin(np.concatenate([scaled_inputs, scaled_inputs + np.pi / 2.0], axis=-1),
                                               np.concatenate(2 * [input_var], axis=-1))

        if self.include_inputs:
            inputs = 2 * (coords - 0.5) if self.scale_inputs else points
            encoded_inputs = np.concatenate([inputs, encoded_inputs], axis=-1)
        return encoded_inputs

    def __call__(self, data):
        for key in self.apply_to:
            data[key if self.replace_key else f"{key}.nerf"] = self.apply(data[key])
        return data

    @staticmethod
    def expected_sin(x_means, x_vars):
        return np.exp(-0.5 * x_vars) * np.sin(x_means)


class MinMaxNumPoints(Transform):
    def __init__(self,
                 min_num_points: Dict[str, int],
                 max_num_points: Dict[str, int]):
        super().__init__()
        print(self.message,
              f"(min: {[f'{k}: {v}' for k, v in min_num_points.items()]}, "
              f"max: {[f'{k}: {v}' for k, v in max_num_points.items()]})")

        self.min_num_points = min_num_points
        self.max_num_points = max_num_points

    def __call__(self, data):
        if "inputs" in data:
            inputs = data["inputs"]
            normals = data.get("inputs.normals")
            colors = data.get("inputs.colors")
            labels = data.get("inputs.labels")
            if "inputs" in self.min_num_points:
                min_num_points = self.min_num_points["inputs"]
                if len(inputs) == 0:
                    data["inputs"] = np.zeros((min_num_points, 3))
                    if normals is not None:
                        data["inputs.normals"] = np.zeros((min_num_points, 3))
                    if colors is not None:
                        data["inputs.colors"] = np.zeros((min_num_points, 3))
                    if labels is not None:
                        data["inputs.labels"] = np.zeros(min_num_points, dtype=labels.dtype)
                elif len(inputs) < min_num_points:
                    indices = subsample(inputs, min_num_points)
                    data["inputs"] = inputs[indices]
                    if normals is not None:
                        data["inputs.normals"] = normals[indices]
                    if colors is not None:
                        data["inputs.colors"] = colors[indices]
                    if labels is not None:
                        data["inputs.labels"] = labels[indices]
            if "inputs" in self.max_num_points:
                max_num_points = self.max_num_points["inputs"]
                if len(inputs) > max_num_points:
                    indices = subsample(inputs, max_num_points)
                    data["inputs"] = inputs[indices]
                    if normals is not None:
                        data["inputs.normals"] = normals[indices]
                    if colors is not None:
                        data["inputs.colors"] = colors[indices]
                    if labels is not None:
                        data["inputs.labels"] = labels[indices]
        if "points" in data:
            points = data["points"]
            occ = data["points.occ"]
            labels = data.get("points.labels")
            if "points" in self.min_num_points:
                min_num_points = self.min_num_points["points"]
                if len(points) == 0:
                    data["points"] = np.zeros((min_num_points, 3))
                    data["points.occ"] = np.ones(min_num_points, dtype=bool)
                    if labels is not None:
                        data["points.labels"] = np.zeros(min_num_points, dtype=labels.dtype)
                elif len(points) < min_num_points:
                    indices = subsample(points, min_num_points)
                    data["points"] = points[indices]
                    data["points.occ"] = occ[indices]
                    data["points.indices"] = indices
                    if labels is not None:
                        data["points.labels"] = labels[indices]
            if "points" in self.max_num_points:
                max_num_points = self.max_num_points["points"]
                if len(points) > max_num_points:
                    indices = subsample(points, max_num_points)
                    data["points"] = points[indices]
                    data["points.occ"] = occ[indices]
                    data["points.indices"] = indices
                    if labels is not None:
                        data["points.labels"] = labels[indices]
        if "pointcloud" in data:
            pointcloud = data["pointcloud"]
            normals = data.get("pointcloud.normals")
            labels = data.get("pointcloud.labels")
            if "pointcloud" in self.min_num_points:
                min_num_points = self.min_num_points["pointcloud"]
                if len(pointcloud) == 0:
                    data["pointcloud"] = np.zeros((min_num_points, 3))
                    if normals is not None:
                        data["pointcloud.normals"] = np.zeros((min_num_points, 3))
                    if labels is not None:
                        data["pointcloud.labels"] = np.zeros(min_num_points, dtype=labels.dtype)
                elif len(pointcloud) < min_num_points:
                    indices = subsample(pointcloud, min_num_points)
                    data["pointcloud"] = pointcloud[indices]
                    if normals is not None:
                        data["pointcloud.normals"] = normals[indices]
                    if labels is not None:
                        data["pointcloud.labels"] = labels[indices]
            if "pointcloud" in self.max_num_points:
                max_num_points = self.max_num_points["pointcloud"]
                if len(pointcloud) > max_num_points:
                    indices = subsample(pointcloud, max_num_points)
                    data["pointcloud"] = pointcloud[indices]
                    if normals is not None:
                        data["pointcloud.normals"] = normals[indices]
                    if labels is not None:
                        data["pointcloud.labels"] = labels[indices]
        return data


class BoundingBox(Transform):
    def __init__(self,
                 reference: str = "pointcloud",
                 remove_reference: bool = False,
                 cache: bool = False,
                 verbose: bool = False):
        super().__init__()
        print(self.message, f"(reference: {reference}, remove_reference: {remove_reference})")

        self.reference = reference
        self.remove_reference = remove_reference
        self.cache = dict() if cache else None
        self.verbose = verbose

    @staticmethod
    def get_aabb_from_pts(points: np.ndarray) -> np.ndarray:
        ub, lb = np.max(points, axis=0), np.min(points, axis=0)
        borders = [[lb[0], lb[1], lb[2]],
                   [lb[0], lb[1], ub[2]],
                   [lb[0], ub[1], lb[2]],
                   [lb[0], ub[1], ub[2]],
                   [ub[0], lb[1], lb[2]],
                   [ub[0], lb[1], ub[2]],
                   [ub[0], ub[1], lb[2]],
                   [ub[0], ub[1], ub[2]]]
        return np.asarray(borders)

    def __call__(self, data):
        obj_name = data.get("obj_name")
        bbox = None
        if obj_name is not None:
            bbox = self.cache.get(obj_name) if isinstance(self.cache, dict) else None
        if bbox is None:
            ref = data[self.reference]
            bbox = self.get_aabb_from_pts(ref)
            if isinstance(self.cache, dict) and obj_name is not None:
                if self.verbose:
                    print(f"{self.__class__.__name__}: Caching bbox.")
                self.cache[obj_name] = bbox
        else:
            if self.verbose:
                print(f"{self.__class__.__name__}: Retrieving bbox from cache.")

        data["bbox"] = bbox.copy()
        if self.remove_reference:
            data.pop(self.reference)
        return data


class BoundingBoxJitter(Transform):
    def __init__(self, max_jitter: List[float]):
        super().__init__()
        print(self.message)

        self.max_jitter = max_jitter

    def __call__(self, data):
        dx = np.random.uniform(-self.max_jitter[0], self.max_jitter[0])
        dy = np.random.uniform(-self.max_jitter[1], self.max_jitter[1])
        dz = np.random.uniform(-self.max_jitter[2], self.max_jitter[2])
        data["bbox"] += np.asarray([dx, dy, dz])
        return data


class SegmentationFromPartNet(Transform):
    def __init__(self,
                 apply_to: Union[List[str], Tuple[str, ...]] = ("inputs",),
                 num_classes: int = 50):
        super().__init__()
        print(self.message, f"(apply to: {apply_to}, num classes: {num_classes})")

        self.apply_to = apply_to
        self.num_classes = num_classes

        self.cat_part_map = {"03642806_2": 29, "03642806_1": 28, "04379243_1": 47, "03636649_4": 27, "02773838_2": 5,
                             "02773838_1": 4, "03797390_1": 36, "03636649_1": 24, "03636649_2": 25, "03636649_3": 26,
                             "02691156_4": 3, "02691156_1": 0, "04379243_2": 48, "02691156_3": 2, "02691156_2": 1,
                             "02954340_1": 6, "02954340_2": 7, "04099429_2": 42, "04099429_3": 43, "04099429_1": 41,
                             "03261776_3": 18, "03261776_2": 17, "03261776_1": 16, "02958343_3": 10, "02958343_2": 9,
                             "02958343_1": 8, "03467517_2": 20, "03467517_3": 21, "04379243_3": 49, "02958343_4": 11,
                             "04225987_2": 45, "04225987_3": 46, "04225987_1": 44, "03790512_3": 32, "03790512_1": 30,
                             "03790512_2": 31, "03467517_1": 19, "03790512_4": 33, "03790512_5": 34, "03790512_6": 35,
                             "03797390_2": 37, "03001627_4": 15, "03001627_1": 12, "03001627_2": 13, "03001627_3": 14,
                             "03948459_1": 38, "03948459_3": 40, "03948459_2": 39, "03624134_1": 22, "03624134_2": 23}

    def __call__(self, data):
        partnet_points = data.pop("partnet.points")
        partnet_labels = data.pop("partnet.labels")
        category = data["category.id"]
        for index, part_label in enumerate(np.unique(partnet_labels)):
            partnet_labels[partnet_labels == part_label] = self.cat_part_map[f"{category}_{index + 1}"]

        if not self.apply_to:
            idx = subsample(np.random.permutation(len(partnet_points)), len(data["inputs"]))
            data["inputs"] = partnet_points[idx]
            data["inputs.labels"] = partnet_labels[idx]
            return data

        kdtree = KDTree(partnet_points)
        for key in self.apply_to:
            value = data.get(key)
            if value is not None:
                value = value.astype(partnet_points.dtype)
                if key == "points":
                    occ = data["points.occ"]
                    if occ.dtype in [np.float16, np.float32, np.float64]:
                        occ = occ <= 0
                    else:
                        occ = occ == 1
                    data["points.labels"] = self.num_classes * np.ones_like(occ, dtype=partnet_labels.dtype)
                    data["points.labels"][occ] = partnet_labels[kdtree.query(value[occ])[1]]
                else:
                    data[f"{key}.labels"] = partnet_labels[kdtree.query(value)[1]]

        return data


class NormalsCameraCosineSimilarity(Transform):
    def __init__(self, remove_normals: bool = False):
        super().__init__()
        print(self.message)

        self.remove_normals = remove_normals

    def __call__(self, data):
        normals = data["inputs.normals"]
        rot = data["inputs.rot"]
        cam = rot.T[:, 2]
        cos_sim = np.dot(normals, cam) / (np.linalg.norm(normals, axis=1) * np.linalg.norm(cam))
        data["inputs.cos_sim"] = cos_sim
        if self.remove_normals:
            data.pop("inputs.normals")
        return data


class AngleOfIncidenceRemoval(Transform):
    def __init__(self,
                 random: bool = True,
                 cos_sim_threshold: Optional[float] = None,
                 remove_cos_sim: bool = False):
        super().__init__()
        if cos_sim_threshold is None:
            print(self.message)
        else:
            print(self.message, f"(angle threshold: {cos_sim_threshold})")

        self.random = random
        self.cos_sim_threshold = cos_sim_threshold
        self.remove_cos_sim = remove_cos_sim

    def __call__(self, data):
        points = data["inputs"]
        normals = data.get("inputs.normals")
        cos_sim = data["inputs.cos_sim"]

        indices = np.ones(len(points), dtype=bool)
        if self.random:
            indices = np.abs(cos_sim) >= (0.8 - 0.2) * np.random.random_sample(len(indices)) + 0.2
        if self.cos_sim_threshold is not None:
            cos_sim_threshold = np.random.uniform(0, self.cos_sim_threshold) if self.random else self.cos_sim_threshold
            indices = indices & (np.abs(cos_sim) >= cos_sim_threshold)

        """
        if data.get("pointcloud") is not None:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["pointcloud"]))
            pcd.normals = o3d.utility.Vector3dVector(data["pointcloud.normals"])
            o3d.visualization.draw_geometries([pcd], point_show_normal=True)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([pcd.select_by_index(np.argwhere(indices)).paint_uniform_color((0, 0, 1)),
                                           pcd.select_by_index(np.argwhere(~indices)).paint_uniform_color((1, 0, 0))],
                                          point_show_normal=True)
        """

        if indices.sum() == 0:
            return data

        data["inputs"] = points[indices]
        if normals is not None:
            data["inputs.normals"] = normals[indices]
        if self.remove_cos_sim:
            data.pop("inputs.cos_sim")
        else:
            data["inputs.cos_sim"] = cos_sim[indices]
        colors = data.get("inputs.colors")
        if colors is not None:
            data["inputs.colors"] = colors[indices]

        return data


class EdgeNoise(Transform):
    def __init__(self,
                 stddev: float = 0.005,
                 angle_threshold: float = None,
                 remove_cos_sim: bool = False):
        super().__init__()
        print(self.message, f"(STD={stddev})")

        self.stddev = stddev
        self.angle_threshold = angle_threshold
        self.remove_cos_sim = remove_cos_sim

    def __call__(self, data):
        points = data["inputs"]
        cos_sim = data["inputs.cos_sim"]

        if self.angle_threshold is None:
            indices = np.abs(cos_sim) >= (0.8 - 0.2) * np.random.random_sample(len(points)) + 0.2
        else:
            indices = np.abs(cos_sim) >= self.angle_threshold

        noise = self.stddev * np.random.randn(*points[~indices].shape)
        points[~indices] += noise

        """
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        o3d.visualization.draw_geometries([pcd.select_by_index(np.argwhere(indices)).paint_uniform_color((0, 0, 1)),
                                           pcd.select_by_index(np.argwhere(~indices)).paint_uniform_color((1, 0, 0))])
        """

        data["inputs"] = points
        if self.remove_cos_sim:
            data.pop("inputs.cos_sim")
        return data


class ImageToTensor(Transform):
    def __init__(self,
                 resize: int = 224,
                 normalize: bool = True,
                 format: str = "torchvision"):
        super().__init__()
        print(self.message, f"(Resize: {resize if resize else False}, Normalize: {normalize})")

        self.transform = list()
        if resize:
            self.transform.append(Resize(resize))
        self.transform.append(ToTensor())
        if normalize:
            if format == "detectron2":
                mean = np.array([103.530, 116.280, 123.675]) / 255.
                std = np.array([57.375, 57.120, 58.395]) / 255.
                std = np.ones(3)
            elif format == "torchvision":
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            else:
                raise ValueError
            self.transform.append(NormalizeImage(mean, std))
        self.transform = Compose(self.transform)
        self.format = format

    def __call__(self, data):
        image_key = "inputs.image"
        image = data.get(image_key)
        if image is None:
            image_key = "inputs"
            image = data[image_key]
        if isinstance(image, Image.Image):

            if self.format == "detectron2":
                image = np.asarray(image)
                image = image[:, :, ::-1]
                image = Image.fromarray(image)
            data[image_key] = self.transform(image)
        else:
            raise TypeError
        return data


class Rotate(Transform):
    def __init__(self,
                 to_cam_frame: bool = False,
                 to_world_frame: bool = False,
                 axes: str = "",
                 angles: Tuple[float] = None,
                 matrix: np.ndarray = None,
                 upper_hemisphere: bool = True,
                 reverse: bool = False,
                 angle_from_index: bool = False,
                 choose_random: bool = False,
                 from_inputs: bool = False,
                 print_message: bool = True):
        assert not (to_cam_frame and to_world_frame)
        assert not ((to_cam_frame or to_world_frame) and from_inputs)

        super().__init__()
        if print_message:
            print(self.message, f"(to world: {to_world_frame}, to cam: {to_cam_frame}, from inputs: {from_inputs})")

        self.to_cam_frame = to_cam_frame
        self.to_world_frame = to_world_frame
        self.axes = axes
        self.angles = angles
        self.matrix = matrix
        self.upper_hemisphere = upper_hemisphere
        self.reverse = reverse
        self.angle_from_index = angle_from_index
        self.choose_random = choose_random
        self.from_inputs = from_inputs

    def __call__(self, data):
        points = data.get("points")

        inputs = data.get("inputs")
        normals = data.get("inputs.normals")

        pcd = data.get("pointcloud")
        pcd_normals = data.get("pointcloud.normals")

        pcd100k = data.get("pcd100k")
        pcd100k_normals = data.get("pcd100k.normals")

        mesh_vertices = data.get("mesh.vertices")
        mesh_normals = data.get("mesh.normals")

        bbox = data.get("bbox")

        partnet = data.get("partnet.points")

        if self.reverse:
            rot = data.get("inputs.frame")
            rot = np.eye(3) if rot is None else rot.T.copy()
        elif self.from_inputs:
            rot = data.get("inputs.rotation", np.eye(3))
        else:
            if self.to_cam_frame:
                rot = data["inputs.rot"].copy()
            elif self.to_world_frame:
                rot = data["inputs.rot"].copy()
                x_angle = data["inputs.x_angle"]
                if x_angle != 0:
                    rot_x = R.from_euler('x', -x_angle, degrees=True).as_matrix()
                    rot = rot_x @ rot
            elif self.axes:
                if self.angle_from_index:
                    if len(self.axes) > 1:
                        raise NotImplementedError("Angle from index only works for single axis rotation.")
                    self.angles = [(5 * data["index"]) % 360]
                if self.angles:
                    if self.choose_random:
                        axis = np.random.choice(list(self.axes))
                        angle = np.random.choice(self.angles)
                        rot = R.from_euler(axis, angle, degrees=True).as_matrix()
                    else:
                        rot = R.from_euler(self.axes,
                                           self.angles[0] if len(self.angles) == 1 else self.angles,
                                           degrees=True).as_matrix()
                else:
                    rot, x_angle = rot_from_euler(self.axes, self.upper_hemisphere)
                    data["inputs.x_angle"] = x_angle
                # data["inputs.rot"] = rot
            elif isinstance(self.matrix, np.ndarray):
                rot = self.matrix
            else:
                rot = R.random().as_matrix()
                data["inputs.rot"] = rot
            data["inputs.frame"] = rot

        for k, v in zip(["points",
                         "pointcloud",
                         "pointcloud.normals",
                         "pcd100k",
                         "pcd100k.normals",
                         "inputs",
                         "inputs.normals",
                         "mesh.vertices",
                         "mesh.normals",
                         "bbox",
                         "partnet.points"],
                        [points,
                         pcd,
                         pcd_normals,
                         pcd100k,
                         pcd100k_normals,
                         inputs,
                         normals,
                         mesh_vertices,
                         mesh_normals,
                         bbox,
                         partnet]):
            if isinstance(v, np.ndarray):
                data[k] = v @ rot.T
        return data


class PointcloudNoise(Transform):
    def __init__(self,
                 stddev: float = 0.005,
                 clip: Optional[float] = None,
                 return_noise: bool = False):
        super().__init__()
        print(self.message, f"(STD={stddev}, clip={clip}, return_noise={return_noise})")

        self.stddev = stddev
        self.clip = clip
        self.return_noise = return_noise

    def __call__(self, data):
        if "inputs" in data:
            points = data["inputs"]
        else:
            points = data[None]

        noise = self.stddev * np.random.randn(*points.shape)
        if self.clip is not None:
            noise = np.clip(noise, -self.clip, self.clip)
        scale = (points.max(axis=0) - points.min(axis=0)).max()
        points += noise * scale

        if "inputs" in data:
            data["inputs"] = points
            if self.return_noise:
                data["inputs.noise"] = noise
        else:
            data[None] = points
            if self.return_noise:
                data["noise"] = noise

        return data


class SubsamplePointcloud(Transform):
    def __init__(self, num_samples: int = 3000):
        super().__init__()
        print(self.message, f"(N={num_samples})")

        self.num_samples = num_samples

    def __call__(self, data):
        if not self.num_samples:
            return data

        if "inputs" in data:
            points = data["inputs"]
            normals = data.get("inputs.normals")
            colors = data.get("inputs.colors")
        else:
            points = data[None]
            normals = data.get("normals")
            colors = data.get("colors")

        if self.num_samples == len(points):
            return data

        indices = subsample(points, self.num_samples)

        if "inputs" in data:
            data["inputs"] = points[indices]
            if normals is not None:
                data["inputs.normals"] = normals[indices]
            if colors is not None:
                data["inputs.colors"] = colors[indices]
        else:
            data[None] = points[indices]
            if normals is not None:
                data["normals"] = normals[indices]
            if colors is not None:
                data["colors"] = colors[indices]

        return data


class ProcessPointcloud(Transform):
    def __init__(self,
                 downsample: DownsampleTypes = None,
                 downsample_factor: Union[float, int] = 1,
                 remove_outlier: OutlierTypes = None,
                 outlier_std_ratio: float = 1.0,
                 estimate_normals: bool = False,
                 search_param: SearchParamTypes = SearchParamTypes.HYBRID,
                 search_param_knn: int = 30,
                 search_param_radius: float = 0.02):
        super().__init__()
        print(self.message)

        self.downsample = downsample
        self.downsample_factor = downsample_factor
        self.remove_outlier = remove_outlier
        self.outlier_std_ratio = outlier_std_ratio
        self.estimate_normals = estimate_normals
        self.search_param = search_param
        self.search_param_knn = search_param_knn
        self.search_param_radius = search_param_radius

    def __call__(self, data):
        points = data[None]
        normals = data.get("normals")
        colors = data.get("colors")

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = process_point_cloud(pcd,
                                  downsample=self.downsample,
                                  downsample_factor=self.downsample_factor,
                                  remove_outlier=self.remove_outlier,
                                  outlier_std_ratio=self.outlier_std_ratio,
                                  estimate_normals=self.estimate_normals,
                                  search_param=self.search_param,
                                  search_param_knn=self.search_param_knn,
                                  search_param_radius=self.search_param_radius)

        data[None] = np.asarray(pcd.points)
        if pcd.has_normals():
            data["normals"] = np.asarray(pcd.normals)
        if pcd.has_colors():
            data["colors"] = np.asarray(pcd.colors)
        return data


class RotatePointcloud(Transform):
    def __init__(self,
                 axes: Optional[str] = None,
                 angles: Tuple[float] = None):
        super().__init__()
        print(self.message, f"(axes: {axes}, angles: {angles})")

        self.axes = axes
        self.angles = angles

    def __call__(self, data):
        points = data[None]
        normals = data.get("normals")

        rot = R.from_euler(self.axes,
                           self.angles[0] if len(self.angles) == 1 else self.angles,
                           degrees=True).as_matrix()

        data[None] = (rot @ points.T).T
        if normals is not None:
            data["normals"] = (rot @ normals.T).T
        return data


class RotateMesh(Transform):
    def __init__(self,
                 axes: Optional[str] = None,
                 angles: Tuple[float] = None):
        super().__init__()
        print(self.message, f"(axes: {axes}, angles: {angles})")

        self.axes = axes
        self.angles = angles

    def __call__(self, data):
        vertices = data["vertices"]
        normals = data.get("normals")

        rot = R.from_euler(self.axes,
                           self.angles[0] if len(self.angles) == 1 else self.angles,
                           degrees=True).as_matrix()

        data["vertices"] = (rot @ vertices.T).T
        if normals is not None:
            data["normals"] = (rot @ normals.T).T
        return data


class ScaleMesh(Transform):
    def __init__(self, scale: float):
        super().__init__()
        print(self.message, f"(scale: {scale})")

        self.scale = scale

    def __call__(self, data):
        vertices = data["vertices"]

        data["vertices"] = self.scale * vertices
        data["scale"] = self.scale
        return data


class ScalePoints(Transform):
    def __init__(self, scale: float):
        super().__init__()
        print(self.message, f"(scale: {scale})")

        self.scale = scale

    def __call__(self, data):
        points = data[None]

        data[None] = self.scale * points
        return data


class TransformMesh(Transform):
    def __init__(self):
        super().__init__()
        print(self.message)

    def __call__(self, data):
        trafo = data.get("inputs.trafo")
        if trafo is not None:
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data["mesh.vertices"]),
                                             o3d.utility.Vector3iVector(data["mesh.triangles"]))
            mesh.transform(trafo)

            data["mesh.vertices"] = np.asarray(mesh.vertices)
            return data
        else:
            return data


class NormalizeMesh(Transform):
    def __init__(self):
        super().__init__()
        print(self.message)

    def __call__(self, data):
        vertices = data["vertices"]
        triangles = data["triangles"]

        referenced = np.zeros(len(vertices), dtype=bool)
        referenced[triangles] = True
        in_mesh = vertices[referenced]
        bounds = np.array([in_mesh.min(axis=0), in_mesh.max(axis=0)])

        translation = -bounds.mean(axis=0)

        extents = bounds.ptp(axis=0)
        max_extents = extents.max()
        scale = 1 / max_extents

        vertices += translation
        vertices *= scale

        data["vertices"] = vertices
        return data


class ApplyPose(Transform):
    def __init__(self,
                 pose: Optional[np.ndarray] = None,
                 keys: Tuple[str, ...] = ("points",
                                          "pointcloud",
                                          "pointcloud.normals",
                                          "pcd100k",
                                          "pcd100k.normals",
                                          "inputs",
                                          "inputs.normals",
                                          "mesh.vertices",
                                          "mesh.normals",
                                          "bbox"),
                 exclude: Optional[Tuple[str, ...]] = None):
        super().__init__()
        print(self.message, f"to {list(k for k in keys if exclude and k not in exclude)}")

        self.pose = pose
        self.keys = keys
        self.exclude = list() if exclude is None else exclude

    def __call__(self, data):
        pose = self.pose
        if pose is None:
            pose = data.get("inputs.pose")
        if pose is not None:
            apply_list = list()
            for key in self.keys:
                if key not in self.exclude:
                    apply_list.append((key, data.get(key)))

            rot = pose[:3, :3]
            trans = pose[:3, 3]

            data["pose"] = pose
            for k, v in apply_list:
                if isinstance(v, np.ndarray):
                    data[k] = v @ rot.T
                    if "normals" not in k:
                        data[k] += trans
            return data
        else:
            return data


class AddRandomRotation(Transform):
    def __init__(self,
                 axes: Optional[str] = None,
                 upper_hemisphere: bool = False):
        super().__init__()
        print(self.message, f"(axes: {axes}, upper_hemisphere: {upper_hemisphere})")

        self.axes = axes
        self.upper_hemisphere = upper_hemisphere

    def __call__(self, data):
        rot, x_angle = rot_from_euler(self.axes, self.upper_hemisphere)
        data["rot"] = rot
        data["x_angle"] = x_angle
        return data


class ProcessInputs(ProcessPointcloud):
    def __init__(self,
                 downsample: DownsampleTypes = None,
                 downsample_factor: Union[float, int] = 1,
                 remove_outlier: OutlierTypes = None,
                 outlier_std_ratio: float = 1.0,
                 estimate_normals: bool = False,
                 search_param: SearchParamTypes = SearchParamTypes.HYBRID,
                 search_param_knn: int = 30,
                 search_param_radius: float = 0.02):
        super().__init__(downsample,
                         downsample_factor,
                         remove_outlier,
                         outlier_std_ratio,
                         estimate_normals,
                         search_param,
                         search_param_knn,
                         search_param_radius)

    def __call__(self, data):
        super_data = super().__call__({None: data["inputs"],
                                       "normals": data.get("inputs.normals"),
                                       "colors": data.get("inputs.colors")})

        data["inputs"] = super_data[None]
        if super_data.get("normals") is not None:
            data["inputs.normals"] = super_data["normals"]
        if super_data.get("colors") is not None:
            data["inputs.colors"] = super_data["colors"]
        return data


class InputNormalsFromPointcloud(Transform):
    def __init__(self,
                 cache_kdtree: bool = False,
                 cache_normals: bool = False,
                 remove_pointcloud: bool = False,
                 verbose: bool = False):
        super().__init__()
        print(self.message, f"(cache: {cache_kdtree}, {cache_normals})")

        self.cache = dict() if cache_kdtree or cache_normals else None
        self.remove_pointcloud = remove_pointcloud
        self.verbose = verbose

    def __call__(self, data):
        if data.get("inputs.normals") is not None:
            if self.verbose:
                print(f"{self.__class__.__name__}: Input normals already exist. Returning data.")
            return data

        pointcloud = data["pointcloud"]
        normals = data["pointcloud.normals"]
        inputs = data["inputs"]
        obj_name = data.get("inputs.name")
        input_path = data.get("inputs.path")

        kdtree = None
        if obj_name is not None:
            kdtree = self.cache.get(obj_name) if isinstance(self.cache, dict) else None
        if kdtree is None:
            kdtree = KDTree(pointcloud, leafsize=100)
            if isinstance(self.cache, dict) and obj_name is not None:
                if self.verbose:
                    print(f"{self.__class__.__name__}: Caching KDTree.")
                self.cache[obj_name] = kdtree
        else:
            if self.verbose:
                print(f"{self.__class__.__name__}: Retrieving KDTree from cache.")

        input_normals = None
        if input_path is not None:
            input_normals = self.cache.get(input_path) if isinstance(self.cache, dict) else None
        if input_normals is None:
            _, index = kdtree.query(inputs.astype(pointcloud.dtype))
            input_normals = normals[index]
            if isinstance(self.cache, dict) and input_path is not None:
                if self.verbose:
                    print(f"{self.__class__.__name__}: Caching normals.")
                self.cache[input_path] = input_normals.astype(np.float32)
        else:
            if self.verbose:
                print(f"{self.__class__.__name__}: Retrieving normals from cache.")

        data["inputs.normals"] = input_normals.copy()
        if self.remove_pointcloud:
            data.pop("pointcloud")
            data.pop("pointcloud.normals")
        return data


class CropPointcloud(Transform):
    def __init__(self,
                 crop_type: str = "box",
                 scale: float = 1.2,
                 mesh_dirname: str = "google_16k",
                 mesh_filename: str = "nontextured.ply",
                 pose_filename: str = "pose.npy"):
        assert crop_type.lower() in ["box", "hull"]

        super().__init__()
        print(self.message, f"with {crop_type}")

        self.crop_type = crop_type.lower()
        self.scale = scale
        self.crop_shapes = dict()
        self.mesh_dirname = mesh_dirname
        self.mesh_filename = mesh_filename
        self.pose_filename = pose_filename

    def get_mesh(self, mesh_dir: str) -> o3d.geometry.TriangleMesh:
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, self.mesh_filename))
        if self.pose_filename and isinstance(self.pose_filename, str):
            pose_path = os.path.join(mesh_dir, self.pose_filename)
            if os.path.isfile(pose_path):
                mesh = mesh.transform(np.load(pose_path))
        center = mesh.get_center()
        center[2] = 0
        mesh = mesh.scale(self.scale, center=center)
        return mesh

    def __call__(self, data):
        obj_name = data["name"]
        points = data[None]
        normals = data.get("normals")
        colors = data.get("colors")

        if obj_name not in self.crop_shapes:
            mesh_dir = os.path.join(data["path"], self.mesh_dirname)
            mesh = self.get_mesh(mesh_dir)
            if self.crop_type == "box":
                self.crop_shapes[obj_name] = mesh.get_axis_aligned_bounding_box()
            elif self.crop_type == "hull":
                self.crop_shapes[obj_name] = Delaunay(np.asarray(mesh.vertices))
            else:
                raise ValueError

        if self.crop_type == "box":
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            if normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(normals)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)

            pcd = pcd.crop(self.crop_shapes[obj_name])

            points = np.asarray(pcd.points)
            if pcd.has_normals():
                normals = np.asarray(pcd.normals)
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
        elif self.crop_type == "hull":
            indices = self.crop_shapes[obj_name].find_simplex(points) >= 0
            points = points[indices]
            if normals is not None:
                normals = normals[indices]
            if colors is not None:
                colors = colors[indices]

        data[None] = points
        if normals is not None:
            data["normals"] = normals
        if colors is not None:
            data["colors"] = colors
        return data


class AxesCutPointcloud(Transform):
    def __init__(self,
                 axes: str = "xyz",
                 cut_ratio: Union[Tuple[float, float], float] = 0.5,
                 rotate_object: str = "",
                 upper_hemisphere: bool = False):
        super().__init__()
        print(self.message, f"(axes={axes}, ratio={cut_ratio}, rotation={rotate_object})")

        self.axes = axes
        self.cut_ratio = cut_ratio
        self.rotate_object = rotate_object
        self.upper_hemisphere = upper_hemisphere

    def __call__(self, data):
        points = data[None]
        normals = data.get("normals")

        x_angle = 0
        rot = np.eye(3)
        if self.rotate_object:
            rot, x_angle = rot_from_euler(self.rotate_object, self.upper_hemisphere)
            points = (rot @ points.T).T

        if all(axis in self.axes for axis in "xyz"):
            side = np.random.randint(3)
        else:
            side = list()
            if 'x' in self.axes:
                side.append(0)
            if 'y' in self.axes:
                side.append(1)
            if 'z' in self.axes:
                side.append(2)
            side = np.random.choice(side)

        size = points[:, side].max() - points[:, side].min()

        if isinstance(self.cut_ratio, float):
            length = np.random.uniform(0, self.cut_ratio) * size
            indices = (points[:, side] - points[:, side].min()) > length
        elif isinstance(self.cut_ratio, (tuple, list)):
            length = np.random.uniform(self.cut_ratio[0], self.cut_ratio[1]) * size
            indices = (points[:, side] - points[:, side].min()) > length
        else:
            raise TypeError(f"cut_ratio must be float or tuple, not {type(self.cut_ratio)}")

        if self.rotate_object:
            data["rot"] = rot
            data["x_angle"] = x_angle
        if indices.sum() == 0:
            return data

        data[None] = data[None][indices]
        if normals is not None:
            data["normals"] = data["normals"][indices]

        return data


class SphereCutPointcloud(Transform):
    def __init__(self,
                 radius: float = 0.01,
                 num_spheres: int = 100,
                 random: bool = True,
                 max_percent: float = 0.5):
        super().__init__()
        print(self.message, f"(radius={radius}, num_spheres={num_spheres})")

        self.radius = radius
        self.num_spheres = num_spheres
        self.random = random
        self.max_percent = max_percent

    def __call__(self, data):
        points = data[None]
        normals = data.get("normals")

        if len(points) < self.num_spheres:
            return data

        num_spheres = self.num_spheres
        if self.random:
            num_spheres = np.random.randint(self.num_spheres)

        indices = np.zeros(len(points), dtype=bool)
        center_indices = np.random.randint(0, len(points), num_spheres)
        centers = points[center_indices]
        radii = np.random.uniform(0, self.radius, size=num_spheres) if self.random else self.radius

        distances = np.linalg.norm(points - centers[:, np.newaxis], axis=2)
        new_indices = np.any(distances < radii[:, np.newaxis], axis=0)
        indices[new_indices] = True

        if (indices.sum() / len(points)) > self.max_percent:
            indices = np.zeros(len(points), dtype=bool)

        data[None] = points[~indices]
        if normals is not None:
            data["normals"] = normals[~indices]

        return data


class RenderPointcloud(Transform):
    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 verbose: bool = False):
        super().__init__()
        print(self.message, f"(width={width}, height={height})")

        self.width = width
        self.height = height

        self.default_intrinsic = np.eye(3)
        self.default_intrinsic[0, 0] = self.width
        self.default_intrinsic[1, 1] = self.width
        self.default_intrinsic[0, 2] = self.width / 2
        self.default_intrinsic[1, 2] = self.height / 2

        self.default_extrinsic = np.eye(4)
        self.default_extrinsic[:3, 3] = np.array([0, 0, 2])

        self.verbose = verbose

    def __call__(self, data):

        vertices = data["mesh.vertices"]
        triangles = data["mesh.triangles"]
        intrinsic = data.get("inputs.intrinsic")
        extrinsic = data.get("inputs.extrinsic")

        if intrinsic is None:
            intrinsic = self.default_intrinsic
            data["inputs.intrinsic"] = intrinsic
        if extrinsic is None:
            extrinsic = self.default_extrinsic
            data["inputs.extrinsic"] = extrinsic

        vis_timer = time()
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                         o3d.utility.Vector3iVector(triangles))
        if self.verbose:
            print(f"{self.__class__.__name__}: Loading mesh takes {time() - vis_timer:.3f} seconds")

        vis_timer = time()
        open3d_extrinsic = np.eye(4)
        rot_yz_180 = R.from_euler("yz", (180, 180), degrees=True).as_matrix()
        open3d_extrinsic[:3, :3] = rot_yz_180 @ extrinsic[:3, :3]
        open3d_extrinsic[:3, 3] = extrinsic[:3, 3]
        if self.verbose:
            print(f"{self.__class__.__name__}: Setting camera extrinsics takes {time() - vis_timer:.3f} seconds")

        vis_timer = time()
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.width, height=self.height, visible=False)

        vis.add_geometry(mesh)

        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = intrinsic
        param.extrinsic = open3d_extrinsic
        vis.get_view_control().convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        if self.verbose:
            print(f"{self.__class__.__name__}: Visualizer setup takes {time() - vis_timer:.3f}s")

        vis_timer = time()
        vis.capture_depth_point_cloud("test.ply", do_render=True, convert_to_world_coordinate=True)
        pcd = o3d.io.read_point_cloud("test.ply")
        if self.verbose:
            print(f"{self.__class__.__name__}: Rendering pointcloud takes {time() - vis_timer:.3f}s")

        # vis.clear_geometries()
        # vis.remove_geometry(mesh)
        vis.destroy_window()
        vis.close()

        data["inputs"] = np.asarray(pcd.points)
        data["inputs.rot"] = data.get("inputs.rot", extrinsic[:3, :3].T)  # TODO: Is this correct?
        data["inputs.path"] = data["mesh.path"]

        return data


class RenderDepthMap(Transform):
    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 offscreen: bool = True,
                 method: str = "pyrender",
                 device: str = "cuda",
                 remove_mesh: bool = True,
                 render_normals: bool = False,
                 print_message: bool = True,
                 verbose: bool = False):
        if render_normals and (method == "pytorch3d" or (method == "open3d" and offscreen)):
            raise ValueError("Normals can only be rendered with pyrender or open3d with onscreen rendering")

        super().__init__()
        if print_message:
            print(self.message, f"{'(offscreen)' if offscreen else ''} (width={width}, height={height})")

        self.width = width
        self.height = height
        self.offscreen = offscreen
        self.method = method
        self.device = device
        self.remove_mesh = remove_mesh
        self.render_normals = render_normals
        self.verbose = verbose

        self.default_intrinsic = np.eye(3)
        self.default_intrinsic[0, 0] = self.width
        self.default_intrinsic[1, 1] = self.width
        self.default_intrinsic[0, 2] = self.width / 2
        self.default_intrinsic[1, 2] = self.height / 2

        self.default_extrinsic = np.eye(4)
        self.default_extrinsic[:3, 3] = np.array([0, 0, 1.5])

        if method == "pyrender" and offscreen:
            os.environ["PYOPENGL_PLATFORM"] = "egl"

    def render_open3d(self,
                      vertices: np.ndarray,
                      triangles: np.ndarray,
                      intrinsic: np.ndarray,
                      extrinsic: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        vis_timer = time()
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                         o3d.utility.Vector3iVector(triangles))
        if self.verbose:
            print(f"{self.__class__.__name__}: Loading mesh takes {time() - vis_timer:.3f} seconds")

        if self.render_normals:
            vis_timer = time()
            mesh.compute_vertex_normals()
            if self.verbose:
                print(f"{self.__class__.__name__}: Computing normals takes {time() - vis_timer:.3f} seconds")
        normal_map = None

        vis_timer = time()
        if self.verbose:
            print(f"{self.__class__.__name__}: Setting camera extrinsics takes {time() - vis_timer:.3f} seconds")

        if self.offscreen:
            render_timer = time()
            renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
            material = o3d.visualization.rendering.MaterialRecord()
            renderer.scene.add_geometry("mesh", mesh, material)
            renderer.setup_camera(intrinsic, extrinsic, self.width, self.height)
            if self.verbose:
                print(f"{self.__class__.__name__}: Renderer setup takes {time() - render_timer:.3f}s")

            render_timer = time()
            depth_map = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
            depth_map[depth_map == np.inf] = 0
            if self.verbose:
                print(f"{self.__class__.__name__}: Rendering depth map takes {time() - render_timer:.3f}s")

            # renderer.scene.clear_geometry()
        else:
            vis_timer = time()
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=self.width, height=self.height, visible=False)

            vis.add_geometry(mesh)

            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width,
                                                                self.height,
                                                                intrinsic[0, 0],
                                                                intrinsic[1, 1],
                                                                intrinsic[0, 2] - 0.5,
                                                                intrinsic[1, 2] - 0.5)
            # param.intrinsic.intrinsic_matrix = intrinsic
            param.extrinsic = extrinsic
            vis.get_view_control().convert_from_pinhole_camera_parameters(param, allow_arbitrary=False)
            if self.verbose:
                print(f"{self.__class__.__name__}: Visualizer setup takes {time() - vis_timer:.3f}s")

            if self.render_normals:
                vis_timer = time()
                vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
                normal_map = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                if self.verbose:
                    print(f"{self.__class__.__name__}: Rendering normals takes {time() - vis_timer:.3f}s")

            vis_timer = time()
            depth_map = np.asarray(vis.capture_depth_float_buffer(do_render=True))
            if self.verbose:
                print(f"{self.__class__.__name__}: Rendering depth map takes {time() - vis_timer:.3f}s")

            # vis.clear_geometries()
            # vis.remove_geometry(mesh)
            vis.destroy_window()
            vis.close()
        return depth_map, normal_map

    def render_pytorch3d(self,
                         vertices: np.ndarray,
                         triangles: np.ndarray,
                         intrinsic: np.ndarray,
                         extrinsic: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        intrinsic = intrinsic.astype(np.float32)

        meshes = Meshes(verts=to_tensor(vertices, device=self.device),
                        faces=to_tensor(triangles, device=self.device))

        cameras = PerspectiveCameras(focal_length=((intrinsic[0, 0], intrinsic[1, 1]),),
                                     principal_point=((intrinsic[0, 2], intrinsic[1, 2]),),
                                     R=to_tensor(extrinsic[:3, :3], device=self.device),
                                     T=to_tensor(extrinsic[:3, 3], device=self.device),
                                     device=self.device,
                                     in_ndc=False,
                                     image_size=((self.height, self.width),))
        raster_settings = RasterizationSettings(image_size=(self.height, self.width),
                                                blur_radius=0.0,
                                                faces_per_pixel=1,
                                                bin_size=0)

        rasterizer = MeshRasterizer(cameras=cameras,
                                    raster_settings=raster_settings)
        fragments = rasterizer(meshes)
        depth_map = fragments.zbuf.squeeze().cpu().numpy()
        depth_map[depth_map == -1] = 0

        return depth_map, None

    def render_pyrender(self,
                        vertices: np.ndarray,
                        triangles: np.ndarray,
                        intrinsic: np.ndarray,
                        extrinsic: np.ndarray,
                        z_near: float = 0.01,
                        z_far: float = 10.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        normal_map = None

        renderer_timer = time()
        mesh = Trimesh(vertices=vertices, faces=triangles, process=False)
        if self.verbose:
            print(f"{self.__class__.__name__}: Loading mesh takes {time() - renderer_timer:.3f} seconds")

        renderer_timer = time()
        renderer = pyrender.OffscreenRenderer(self.width, self.height)
        if self.render_normals:
            flags = pyrender.renderer.RenderFlags.NONE
            shader_dir = Path(__file__).parent.parent.parent / "utils" / "assets" / "shaders"
            renderer._renderer._program_cache = ShaderProgramCache(shader_dir=shader_dir)
        else:
            flags = pyrender.RenderFlags.DEPTH_ONLY
        if self.offscreen:
            flags |= pyrender.RenderFlags.OFFSCREEN

        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        camera = pyrender.IntrinsicsCamera(intrinsic[0, 0],
                                           intrinsic[1, 1],
                                           intrinsic[0, 2],
                                           intrinsic[1, 2],
                                           znear=z_near,
                                           zfar=z_far)
        scene.add(camera, pose=inv_trafo(extrinsic))
        if self.verbose:
            print(f"{self.__class__.__name__}: Renderer setup takes {time() - renderer_timer:.2f} seconds")

        renderer_timer = time()
        if self.render_normals:
            normal_image, depth_map = renderer.render(scene, flags=flags)
            normal_map = normal_image / 255
        else:
            depth_map = renderer.render(scene, flags=flags)
        depth_map[depth_map >= z_far] = 0
        if self.verbose:
            if self.render_normals:
                print(f"{self.__class__.__name__}: Rendering takes {time() - renderer_timer:.2f} seconds")
            else:
                print(f"{self.__class__.__name__}: Rendering depth map takes {time() - renderer_timer:.3f}s")

        renderer.delete()

        return depth_map, normal_map

    def __call__(self, data):

        vertices = data["mesh.vertices"]
        triangles = data["mesh.triangles"]
        intrinsic = data.get("inputs.intrinsic")
        extrinsic = data.get("inputs.extrinsic")

        if intrinsic is None:
            intrinsic = self.default_intrinsic
            data["inputs.intrinsic"] = intrinsic
        if extrinsic is None:
            extrinsic = self.default_extrinsic
            data["inputs.extrinsic"] = extrinsic

        if self.method == "open3d":
            depth, normals = self.render_open3d(vertices, triangles, intrinsic, extrinsic)
        elif self.method == "pytorch3d":
            depth, normals = self.render_pytorch3d(vertices, triangles, intrinsic, extrinsic)
        elif self.method == "pyrender":
            depth, normals = self.render_pyrender(vertices, triangles, intrinsic, extrinsic)
        else:
            raise ValueError(f"Unknown rendering method {self.method}.")

        depth[depth < 0] = 0
        data["inputs.depth"] = depth
        data["inputs.rot"] = data.get("inputs.rot", extrinsic[:3, :3].T)  # TODO: Is this correct?
        data["inputs.path"] = data["mesh.path"]

        if normals is not None:
            data["inputs.normals"] = normals

        if self.remove_mesh:
            data.pop("mesh.vertices")
            data.pop("mesh.triangles")
        return data


class DepthToPointcloud(Transform):
    def __init__(self):
        super().__init__()
        print(self.message)

        self.rot_x_180 = R.from_euler('x', 180, degrees=True).as_matrix()

    def __call__(self, data):
        depth_map = data["inputs.depth_map"]
        normal_map = data.get("inputs.normal_map")
        intrinsic = data["inputs.intrinsic"]
        extrinsic = data["inputs.extrinsic"]

        if normal_map is None:
            pcd = convert_depth_image_to_point_cloud(depth_map, intrinsic, extrinsic)
        else:
            normal_image = (normal_map * 255).astype(np.uint8)
            pcd = convert_rgbd_image_to_point_cloud([normal_image, depth_map],
                                                    intrinsic,
                                                    extrinsic,
                                                    depth_scale=1.0,
                                                    depth_trunc=10.0,
                                                    convert_rgb_to_intensity=False)
        pcd.rotate(self.rot_x_180, center=(0, 0, 0))
        points = np.asarray(pcd.points)

        data["inputs"] = points
        if normal_map is not None:
            data["inputs.normals"] = np.asarray(pcd.colors) * 2 - 1
        return data


class RenderDepthMaps(Transform):
    def __init__(self,
                 step: int = 5,
                 min_angle: int = 0,
                 max_angle: int = 360,
                 width: int = 640,
                 height: int = 480,
                 offscreen: bool = True,
                 inplane_rot: Optional[float] = None,
                 print_message: bool = True):
        super().__init__()
        if print_message:
            print(self.message)

        self.step = step
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.width = width
        self.height = height
        self.offscreen = offscreen
        self.inplane_rot = np.deg2rad(inplane_rot) if inplane_rot is not None else None

        self.intrinsic = None
        self.extrinsic = None
        self.mesh = None
        self.camera_position = None

        self.depth_list = list()
        self.extrinsic_list = list()
        self.angle_list = list()

        self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        self.flags = pyrender.RenderFlags.DEPTH_ONLY
        if self.offscreen:
            self.flags |= pyrender.RenderFlags.OFFSCREEN

    @staticmethod
    def draw_camera(K, R, t, width, height, scale=1, color=None):
        if color is None:
            color = [0.8, 0.2, 0.8]

        # camera model scale
        s = 1 / scale

        # intrinsics
        Ks = np.array([[K[0, 0] * s, 0, K[0, 2]],
                       [0, K[1, 1] * s, K[1, 2]],
                       [0, 0, K[2, 2]]])
        Kinv = np.linalg.inv(Ks)

        # 4x4 transformation
        T = np.column_stack((R, t))
        T = np.vstack((T, (0, 0, 0, 1)))

        # axis
        axis = o3d.geometry.TriangleMesh().create_coordinate_frame(size=scale * 0.5).transform(T)

        # points in pixel
        points_pixel = [
            [0, 0, 0],
            [0, 0, 1],
            [width, 0, 1],
            [0, height, 1],
            [width, height, 1],
        ]

        # pixel to camera coordinate system
        points = [scale * Kinv @ p for p in points_pixel]

        # image plane
        width = abs(points[1][0]) + abs(points[3][0])
        height = abs(points[1][1]) + abs(points[3][1])
        plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
        plane.paint_uniform_color(color)
        plane.transform(T)
        plane.translate(R @ [points[1][0], points[1][1], scale])

        # pyramid
        points_in_world = [(R @ p + t) for p in points]
        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
        ]
        colors = [color for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_in_world),
            lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return [axis, plane, line_set]

    @staticmethod
    def compute_look_at_matrix(from_vec: np.ndarray,
                               to_vec: np.ndarray) -> np.ndarray:
        forward_vec = to_vec - from_vec
        forward_vec /= np.linalg.norm(forward_vec)

        right_vec = np.cross(np.array([0, 1, 0]), forward_vec)
        up_vec = np.cross(forward_vec, right_vec)
        return np.array([right_vec, up_vec, forward_vec]).T

    @staticmethod
    def rotation_from_forward_vec(forward_vec: Union[np.ndarray, Vector],
                                  up_axis: str = 'Y',
                                  inplane_rot: Optional[float] = None) -> np.ndarray:
        rotation_matrix = Vector(forward_vec).to_track_quat('-Z', up_axis).to_matrix()
        if inplane_rot is not None:
            rotation_matrix = rotation_matrix @ Euler((0.0, 0.0, inplane_rot)).to_matrix()
        return np.array(rotation_matrix)

    @staticmethod
    def get_position(x: float, y: float, angle: float) -> Tuple[float, float]:
        angle = np.deg2rad(angle)
        x_new = y * np.sin(angle) + x * np.cos(angle)
        y_new = y * np.cos(angle) - x * np.sin(angle)
        return x_new, y_new

    def get_params(self, extrinsic):
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height,
                                                             self.intrinsic[0, 0], self.intrinsic[1, 1],
                                                             self.intrinsic[0, 2] - 0.5, self.intrinsic[1, 2] - 0.5)
        # params.intrinsic.intrinsic_matrix = self.intrinsic
        params.extrinsic = extrinsic
        return params

    def update_camera(self, angle: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = self.get_position(self.camera_position[0], self.camera_position[1], angle)
        new_camera_position = np.array([x, y, self.camera_position[2]])

        x, y = self.get_position(self.camera_look_at[0], self.camera_look_at[1], angle)
        new_camera_look_at = np.array([x, y, self.camera_look_at[2]])

        inv_extrinsic = np.eye(4)
        inv_extrinsic[:3, 3] = new_camera_position

        inv_extrinsic[:3, :3] = self.rotation_from_forward_vec(forward_vec=new_camera_position - new_camera_look_at,
                                                               up_axis='Y',
                                                               inplane_rot=self.inplane_rot)

        extrinsic = inv_trafo(inv_extrinsic)
        return extrinsic, new_camera_position, new_camera_look_at

    def show(self, box: bool = False):
        self.mesh.compute_vertex_normals()
        geometries = [self.mesh, o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)]

        if box:
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.55,) * 3,
                                                      max_bound=(0.55,) * 3)
            box.color = np.zeros(3)
            geometries.append(box)

        for depth, extrinsic in zip(self.depth_list, self.extrinsic_list):
            pcd = convert_depth_image_to_point_cloud(depth,
                                                     self.intrinsic,
                                                     extrinsic,
                                                     depth_scale=1,
                                                     depth_trunc=10)
            geometries.append(pcd)

            cam = self.draw_camera(self.intrinsic,
                                   extrinsic[:3, :3].T,
                                   -extrinsic[:3, :3].T @ extrinsic[:3, 3],
                                   width=640,
                                   height=480,
                                   scale=0.25)
            geometries.extend(cam)
        o3d.visualization.draw_geometries(geometries)

    def __call__(self, data: Dict) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        self.depth_list.clear()
        self.extrinsic_list.clear()
        self.angle_list.clear()

        self.intrinsic = data["inputs.intrinsic"]
        self.extrinsic = data["inputs.extrinsic"]

        inv_extrinsic = inv_trafo(self.extrinsic)
        self.camera_position = inv_extrinsic[:3, 3]
        self.camera_look_at = data.get("inputs.look_at")
        if self.camera_look_at is None:
            self.camera_look_at = np.zeros(3)

        self.mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data["mesh.vertices"]),
                                              o3d.utility.Vector3iVector(data["mesh.triangles"]))

        scene = pyrender.Scene()
        mesh = Trimesh(vertices=data["mesh.vertices"], faces=data["mesh.triangles"], process=False)
        mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False))
        scene.add_node(mesh_node)
        camera = pyrender.IntrinsicsCamera(self.intrinsic[0, 0],
                                           self.intrinsic[1, 1],
                                           self.intrinsic[0, 2],
                                           self.intrinsic[1, 2],
                                           znear=0.1,
                                           zfar=10.0)
        camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
        scene.add_node(camera_node)

        for angle in range(0, 360, self.step):
            extrinsic, _, _ = self.update_camera(-angle)
            inv_extrinsic = inv_trafo(extrinsic)
            rot_z = 180 + np.rad2deg(np.arctan2(inv_extrinsic[1, 0], inv_extrinsic[0, 0]))
            if self.min_angle <= rot_z < self.max_angle and inv_extrinsic[0, 3] < 0:
                extrinsic_opengl = extrinsic.copy()
                extrinsic_opengl[1, :] *= -1
                extrinsic_opengl[2, :] *= -1
                scene.set_pose(camera_node, pose=inv_trafo(extrinsic_opengl))
                depth_map = self.renderer.render(scene, flags=self.flags)

                self.depth_list.append(depth_map)
                self.extrinsic_list.append(extrinsic)
                self.angle_list.append(angle)

        return self.depth_list, self.extrinsic_list, self.angle_list

    def __del__(self):
        self.renderer.delete()


class FindUncertainPoints(Transform):
    def __init__(self,
                 depth_list: List[np.ndarray],
                 angle_list: List[float],
                 max_chamfer_dist: float = 0.01,
                 parallel: bool = False,
                 debug: bool = False,
                 show: bool = False,
                 print_message: bool = True):
        super().__init__()
        if print_message:
            print(self.message)

        self.depth_list = depth_list
        self.angle_list = angle_list
        self.max_chamfer_dist = max_chamfer_dist
        self.parallel = parallel
        self.debug = debug
        self.show = show

        self.init_depth = self.depth_list.pop(0)
        self.init_angle = self.angle_list.pop(0)
        self.init_inputs = None

    @staticmethod
    def get_init_inputs(depth: np.ndarray, intrinsic: np.ndarray, scale: float) -> np.ndarray:
        pcd = convert_depth_image_to_point_cloud(depth,
                                                 intrinsic,
                                                 depth_scale=1,
                                                 depth_trunc=10)
        inputs = np.asarray(pcd.points).copy()
        inputs /= scale
        return inputs

    def eval_single(self,
                    data: Dict,
                    scale: float,
                    depth: np.ndarray,
                    angle: float) -> List[np.ndarray]:

        pcd = convert_depth_image_to_point_cloud(depth,
                                                 data["inputs.intrinsic"],
                                                 depth_scale=1,
                                                 depth_trunc=10)
        inputs = np.asarray(pcd.points).copy()
        inputs /= scale

        offset = (self.init_inputs.max(axis=0) + self.init_inputs.min(axis=0)) / 2
        init_inputs = self.init_inputs - offset
        inputs -= offset

        too_large1 = inputs.min(axis=0)[1] / init_inputs.min(axis=0)[1]
        too_large2 = inputs.max(axis=0)[1] / init_inputs.max(axis=0)[1]

        if self.debug:
            print("Size:", too_large1, too_large2)
        if too_large1 > 1.06 or too_large2 > 1.06:
            if self.debug:
                print("Size skip")
            return []

        kdtree = SKDTree(inputs)
        dist1, index = kdtree.query(init_inputs, workers=1 if self.parallel else -1)

        kdtree = SKDTree(init_inputs)
        dist2, index = kdtree.query(inputs, workers=1 if self.parallel else -1)

        dist1_mean = dist1.mean()
        dist2_mean = dist2.mean()
        chamfer_l1 = 0.5 * (dist1_mean + dist2_mean)

        if self.debug:
            print("Chamfer L1:", chamfer_l1)
        if chamfer_l1 > self.max_chamfer_dist:
            if self.debug:
                print("Chamfer skip")
            return []

        if self.debug and self.show:
            o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs)).paint_uniform_color((1, 0, 0)),
                                               o3d.geometry.PointCloud(o3d.utility.Vector3dVector(init_inputs)).paint_uniform_color((0, 0, 1)),
                                               o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)])

        rot = R.from_euler('y', -angle, degrees=True).as_matrix()
        vertices = data["mesh.vertices"].copy()
        vertices = (rot @ vertices.T).T
        points = data["points"]
        mesh = Trimesh(vertices, data["mesh.triangles"], process=False)

        return check_mesh_contains(mesh, points)

    @staticmethod
    def check_occupancy(occupancy: List[np.ndarray]) -> List[np.ndarray]:
        occupancy = [occ for occ in occupancy if len(occ)]
        if len(occupancy) <= 3:
            return []
        return occupancy

    def get_uncertain(self, data: Dict, occupancy: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        occupancy = np.vstack(occupancy)
        always = np.all(occupancy, axis=0)
        sometimes = np.any(occupancy, axis=0)
        uncertain = sometimes & ~always

        # Remove uncertain points in observed free space
        item = Rotate(to_world_frame=True, print_message=False)(data)
        visib_left = item["points"][:, 0] < item["inputs"][:, 0].min()
        visib_rigth = item["points"][:, 0] > item["inputs"][:, 0].max()
        uncertain[visib_left | visib_rigth] = False

        # Remove uncertain points that are too close to occupied points
        points = data["points"]
        kdtree = SKDTree(points[always])
        always_points_dist, _ = kdtree.query(points[always], k=2, workers=-1 if self.parallel else 1)
        always_points_dist = always_points_dist[always_points_dist > 0]
        uncertain_to_always_points_dist, _ = kdtree.query(points[uncertain], workers=-1 if self.parallel else 1)
        uncertain[uncertain] = uncertain_to_always_points_dist > np.quantile(always_points_dist, 0.99)

        # Remove statistical outlier
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[uncertain]))
        _, indices = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1)

        uncertain_indices = np.argwhere(uncertain)
        uncertain_indices = uncertain_indices[indices]
        uncertain = np.zeros_like(uncertain, dtype=bool)
        uncertain[uncertain_indices] = True

        return uncertain, always, sometimes

    def __call__(self, data):
        intrinsic = data["inputs.intrinsic"]
        scale = (data["pointcloud"].max(axis=0) - data["pointcloud"].min(axis=0)).max()
        self.init_inputs = self.get_init_inputs(self.init_depth, intrinsic, scale)

        depth_scale = self.init_inputs.max(axis=0) - self.init_inputs.min(axis=0)
        if self.debug:
            print("Depth scale:", depth_scale)
        if depth_scale[0] > 0.95:
            if self.debug:
                print("Scale skip")
            return data

        occupancy = [data["points.occ"] > 0]
        if self.parallel:
            with Parallel(n_jobs=-1) as p:
                occupancy.extend(p(delayed(self.eval_single)(data, scale, depth, angle) for depth, angle in
                                   zip(self.depth_list, self.angle_list)))
        else:
            for depth, angle in zip(self.depth_list, self.angle_list):
                occupancy.append(self.eval_single(data, scale, depth, angle))

        occupancy = self.check_occupancy(occupancy)
        if occupancy:
            uncertain, always, sometimes = self.get_uncertain(data, occupancy)

            data["points.uncertain"] = uncertain
            data["points.occ"] = always

            return data
        return data


class LoadUncertain(Transform):
    def __init__(self):
        super().__init__()
        print(self.message)

    def __call__(self, data):
        inputs_path = Path(data["inputs.path"])
        uncertain = list()
        for path in data["points.path"]:
            uncertain_path = inputs_path.parent.joinpath('_'.join([inputs_path.stem, Path(path).stem, "uncertain.npy"]))

            if os.path.isfile(uncertain_path):
                uncertain.append(np.unpackbits(np.load(str(uncertain_path))).astype(bool))
        if uncertain:
            uncertain = np.concatenate(uncertain)

            indices = data.get("points.indices")
            if indices is not None:
                uncertain = uncertain[indices]

            occ = data["points.occ"].astype(int)
            occ[uncertain] = 2
            data["points.occ"] = occ
            return data
        return data


class DepthLikePointcloud(Transform):
    def __init__(self,
                 rotate_object: str = "",
                 upper_hemisphere: bool = False,
                 rot_from_inputs: bool = False,
                 cam_from_inputs: bool = False,
                 distance_multiplier: int = 100):
        assert not (rotate_object and rot_from_inputs)

        super().__init__()
        if rotate_object:
            print(self.message, f"(rotation={rotate_object}, upper hemisphere={upper_hemisphere})")
        elif rot_from_inputs:
            print(self.message, f"(rotation from input=True, cam from input={cam_from_inputs})")
        else:
            print(self.message)

        self.rotate_object = rotate_object
        self.upper_hemisphere = upper_hemisphere
        self.cam_from_inputs = cam_from_inputs
        self.rot_from_inputs = rot_from_inputs
        self.distance_multiplier = distance_multiplier

    def __call__(self, data):
        points = data[None]
        normals = data.get("normals")

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        x_angle = 0
        rot = np.eye(3)
        if self.rotate_object:
            rot, x_angle = rot_from_euler(self.rotate_object, self.upper_hemisphere)
            trafo = np.eye(4)
            trafo[:3, :3] = rot
            pcd.transform(trafo)
        elif self.rot_from_inputs:
            rot = data["rot"]
            pcd.rotate(rot, center=(0, 0, 0))

        if self.cam_from_inputs:
            camera = data["cam"]
            diameter = np.linalg.norm(camera)
        else:
            diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
            camera = np.array([0, 0, diameter])
        _, indices = pcd.hidden_point_removal(camera, self.distance_multiplier * diameter)

        data[None] = points[indices]
        data["rot"] = rot
        # data["x_angle"] = 180 + x_angle if x_angle is not None else None
        data["x_angle"] = x_angle
        if normals is not None:
            data["normals"] = normals[indices]

        return data


class RemoveHiddenPointsFromInputs(Transform):
    def __init__(self, cam_from_inputs: bool = False):
        super().__init__()
        print(self.message, f"(cam from input={cam_from_inputs})")

        self.cam_from_inputs = cam_from_inputs

    def __call__(self, data):
        inputs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["inputs"]))
        if data.get("inputs.normals") is not None:
            inputs.normals = o3d.utility.Vector3dVector(data["inputs.normals"])
        if data.get("inputs.colors") is not None:
            inputs.colors = o3d.utility.Vector3dVector(data["inputs.colors"])

        if self.cam_from_inputs:
            cam = data["inputs.cam"]
            camera = -cam
            diameter = cam[2]
        else:
            diameter = np.linalg.norm(np.asarray(inputs.get_max_bound()) - np.asarray(inputs.get_min_bound()))
            camera = np.array([0, 0, diameter])
        _, indices = inputs.hidden_point_removal(camera, 1000 * diameter)

        inputs = inputs.select_by_index(indices)

        data["inputs"] = np.asarray(inputs.points)
        if inputs.has_normals():
            data["inputs.normals"] = np.asarray(inputs.normals)
        if inputs.has_colors():
            data["inputs.colors"] = np.asarray(inputs.colors)
        return data


class SubsamplePoints(Transform):
    def __init__(self, num_samples: int = 2048, in_out_ratio: Optional[float] = None):
        super().__init__()
        print(self.message, f"(N={num_samples}, ratio={in_out_ratio})")

        self.num_samples = num_samples
        self.in_out_ratio = in_out_ratio

    def __call__(self, data):
        if not self.num_samples:
            return data

        if "points" in data:
            points = data["points"]
            occ = data["points.occ"]
        else:
            points = data[None]
            occ = data["occ"]

        if self.in_out_ratio is not None:
            try:
                n_in = int(self.num_samples * self.in_out_ratio)
                n_out = self.num_samples - n_in
                indices_in = np.argwhere(occ == 1).squeeze()
                indices_out = np.argwhere(occ == 0).squeeze()
                if len(indices_in) > n_in:
                    indices_in = np.random.choice(indices_in, n_in, replace=False)
                else:
                    n_out = self.num_samples - len(indices_in)
                indices_out = np.random.choice(indices_out, n_out, replace=False if n_out < len(indices_out) else True)
                indices = np.concatenate([indices_in, indices_out])
            except (ValueError, TypeError):
                indices = subsample(points, self.num_samples)
        else:
            indices = subsample(points, self.num_samples)

        if "points" in data:
            data["points"] = points[indices]
            data["points.occ"] = occ[indices]
            data["points.indices"] = indices
        else:
            data[None] = points[indices]
            data["occ"] = occ[indices]
            data["indices"] = indices

        return data


class SdfFromOcc(Transform):
    def __init__(self,
                 tsdf: Optional[float] = None,
                 remove_pointcloud: bool = True):
        super().__init__()
        if tsdf is not None:
            print(self.message, f"(TSDF={tsdf})")
        else:
            print(self.message)

        self.tsdf = tsdf
        self.remove_pointcloud = remove_pointcloud

    @staticmethod
    def occ_to_sdf(points: np.ndarray,
                   occ: np.ndarray,
                   pcd: np.ndarray,
                   distance_upper_bound: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        kdtree = KDTree(pcd, leafsize=100)
        dist, idx = kdtree.query(points,
                                 eps=1e-4,
                                 # distance_upper_bound=distance_upper_boun)
                                 )
        if distance_upper_bound is not None:
            # dist[dist == np.inf] = distance_upper_bound
            dist[dist > distance_upper_bound] = distance_upper_bound
        dist[occ == 1] *= -1
        return dist, idx

    def __call__(self, data):
        points = data["points"]
        occ = data["points.occ"]
        pcd = data["pointcloud"]

        sdf, idx = self.occ_to_sdf(points, occ, pcd, distance_upper_bound=self.tsdf)

        data["points.occ"] = sdf
        if self.remove_pointcloud:
            data.pop("pointcloud")
            if "pointcloud.normals" in data:
                data.pop("pointcloud.normals")
        else:
            data["pointcloud.index"] = idx
        return data


class Voxelizer:
    def __init__(self,
                 voxel_size: float = 1.0,
                 resolution: int = 128,
                 padding: float = 0.1,
                 method: str = "simple"):
        assert voxel_size is None or voxel_size == 1 or resolution is None
        assert method.lower() in ["simple", "kdtree", "open3d"]

        self.voxel_size = voxel_size
        self.res = resolution
        self.method = method.lower()
        self.padding = padding

        self.min = -0.5 - padding / 2
        self.max = 0.5 + padding / 2
        self.grid_points = make_3d_grid((self.min,) * 3, (self.max,) * 3, (self.res,) * 3).numpy()

        self.kdtree = None
        if self.method == "kdtree":
            self.kdtree = KDTree(self.grid_points, leafsize=100)

    def __call__(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.method == "simple":
            points /= (1 + self.padding)
            points += 0.5
            if points.max() >= 1:
                points[points >= 1] = 1 - 1e-6
            if points.min() < 0:
                points[points < 0] = 0
            voxel_indices = (points * self.res).astype(np.int64)
            voxel_indices = tuple(voxel_indices.T)
            occupancy = np.zeros((self.res,) * 3, dtype=bool)
            occupancy[voxel_indices] = True
        elif self.method == "kdtree":
            occupancy = np.zeros(len(self.grid_points), dtype=bool)
            _, voxel_indices = self.kdtree.query(points.astype(self.grid_points.dtype))
            occupancy[voxel_indices] = True
            occupancy = rearrange(occupancy, "(h w d) -> h w d", h=self.res, w=self.res, d=self.res)
        elif self.method == "open3d":
            if (self.voxel_size is None or self.voxel_size == 1) and self.res is not None:
                voxel_size = (self.max - self.min) / self.res
            else:
                voxel_size = self.voxel_size

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            voxel_grid = o3d.geometry.VoxelGrid().create_from_point_cloud_within_bounds(input=pcd,
                                                                                        voxel_size=voxel_size,
                                                                                        min_bound=self.min * np.ones(3),
                                                                                        max_bound=self.max * np.ones(3))
            voxel_indices = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])
            voxel_indices = tuple(voxel_indices.T)
            occupancy = np.zeros((self.res,) * 3, dtype=bool)
            occupancy[voxel_indices] = True
        else:
            raise ValueError(f"Unknown voxelization method: {self.method}")

        return occupancy, voxel_indices


class VoxelizeInputs(Transform):
    def __init__(self,
                 voxel_size: float = 1.0,
                 resolution: int = 128,
                 padding: float = 0.1,
                 method: str = "simple"):
        super().__init__()
        print(self.message, f"(voxel_size={voxel_size}, resolution={resolution}, method={method})")

        self.voxelizer = Voxelizer(voxel_size, resolution, padding, method)

    def __call__(self, data):
        points = data["inputs"]
        normals = data.get("inputs.normals")

        voxelized_inputs, voxel_indices = self.voxelizer(points)

        data["inputs"] = voxelized_inputs
        if normals is not None:
            voxelized_normals = np.zeros((len(self.voxelizer.grid_points), 3), dtype=normals.dtype)
            voxelized_normals[voxel_indices] = normals
            data["inputs.normals"] = rearrange(voxelized_normals, "(h w d) c -> h w d c",
                                               h=self.voxelizer.res,
                                               w=self.voxelizer.res,
                                               d=self.voxelizer.res)

        return data


class VoxelizePoints(Transform):
    def __init__(self,
                 voxel_size: float = 1.0,
                 resolution: int = 128,
                 padding: float = 0.1,
                 method: str = "simple"):
        super().__init__()
        print(self.message, f"(voxel_size={voxel_size}, resolution={resolution}, method={method})")

        self.voxelizer = Voxelizer(voxel_size, resolution, padding, method)

    def __call__(self, data):
        points_key = "points" if "points" in data else None
        occ_key = "points.occ" if "points.occ" in data else "occ"

        points = data[points_key]
        occ = data[occ_key]

        voxel_indices = self.voxelizer(points[occ.astype(bool)])[1]
        voxel_occ = np.zeros(len(self.voxelizer.grid_points), dtype=occ.dtype)
        voxel_occ[voxel_indices] = occ[occ.astype(bool)]

        data[points_key] = self.voxelizer.grid_points
        data[occ_key] = voxel_occ
        return data


class Translate(Transform):
    def __init__(self,
                 offset: Union[float, Tuple[float, float, float]],
                 reverse: bool = False):
        super().__init__()
        print(self.message)

        self.offset = offset
        self.reverse = reverse

    def __call__(self, data):
        points = data.get("points")
        pointcloud = data.get("pointcloud")
        pcd100k = data.get("pcd100k")
        inputs = data.get("inputs")
        mesh_vertices = data.get("mesh.vertices")

        if self.reverse:
            offset = data.get("offset")
            offset = 0 if offset is None else -offset
        else:
            offset = self.offset

        for k, v in zip(["points", "pointcloud", "pcd100k", "inputs", "mesh.vertices"],
                        [points, pointcloud, pcd100k, inputs, mesh_vertices]):
            if isinstance(v, np.ndarray):
                v = v.copy()
                v += offset
                data[k] = v
        data["offset"] = offset
        return data


class Normalize(Transform):
    def __init__(self,
                 center: Optional[Union[bool, str]] = None,
                 to_min_val: Optional[str] = None,
                 to_max_val: Optional[str] = None,
                 to_front: bool = False,
                 scale: bool = True,
                 offset: Optional[Union[float, Tuple[float, float, float], List[float]]] = None,
                 true_height: bool = False,
                 reference: str = "inputs",
                 method: str = "cube",
                 reverse: bool = False,
                 padding: float = 0.1):
        assert reference.lower() in ["pointcloud", "inputs", "mesh", "points"], f"Unknown reference: {reference}"
        assert method.lower() in ["cube", "sphere"], f"Unknown method: {method}"

        super().__init__()
        print(self.message, f"(centering: {center}, scaling: {scale}, reference: {reference})")

        self.center = center.lower() if isinstance(center, str) else "xyz" if center else ""
        self.to_min_val = to_min_val.lower() if to_min_val is not None else ""
        self.to_max_val = to_max_val.lower() if to_max_val is not None else ""
        self.to_front = to_front
        self.scale = scale
        self.offset = offset
        self.true_height = true_height
        self.reference = reference.lower()
        self.method = method.lower()
        self.reverse = reverse
        self.padding = padding

    def __call__(self, data):

        points = data.get("points")
        occ = data.get("points.occ")
        pointcloud = data.get("pointcloud")
        pcd100k = data.get("pcd100k")
        inputs = data.get("inputs")
        mesh_vertices = data.get("mesh.vertices")
        bbox = data.get("bbox")
        partnet = data.get("partnet.points")

        if self.reverse:
            offset = data.get("inputs.norm_offset")
            scale = data.get("inputs.norm_scale")
            offset = -offset if offset is not None else 0
            scale = 1 / scale if scale else 1
        else:
            if self.reference == "inputs":
                ref = inputs
            elif self.reference == "pointcloud":
                ref = pointcloud
            elif self.reference == "mesh":
                ref = mesh_vertices
            elif self.reference == "points":
                ref = points[occ > 0]
            else:
                raise ValueError(f"Reference cannot be {self.reference}.")

            min_vals = ref.min(axis=0)
            max_vals = ref.max(axis=0)

            if self.true_height:
                ref = np.array([np.array([min_vals[0], np.min(pointcloud[:, 1]), min_vals[2]]), max_vals])

            scale = 1
            if self.scale:
                if self.method == "cube":
                    scale = (ref.max(axis=0) - ref.min(axis=0)).max()
                elif self.method == "sphere":
                    scale = 2 * np.max(np.sqrt(np.sum(np.square(ref), axis=1)))
                else:
                    raise ValueError(f"Method cannot be {self.method}.")

            offset = (ref.max(axis=0) + ref.min(axis=0)) / 2
            offset_x = offset_y = offset_z = 0

            if self.center:
                offset_x = offset[0] if "x" in self.center else offset_x
                offset_y = offset[1] if "y" in self.center else offset_y
                offset_z = offset[2] if "z" in self.center else offset_z

            if self.to_min_val:
                offset_x = min_vals[0] if "x" in self.to_min_val else offset_x
                offset_y = min_vals[1] if "y" in self.to_min_val else offset_y
                offset_z = min_vals[2] if "z" in self.to_min_val else offset_z

            if self.to_max_val:
                offset_x = max_vals[0] if "x" in self.to_max_val else offset_x
                offset_y = max_vals[1] if "y" in self.to_max_val else offset_y
                offset_z = max_vals[2] if "z" in self.to_max_val else offset_z

            if self.to_front:
                offset_z = scale * (-0.5 + max_vals[2]) + (max_vals[2] - scale * max_vals[2])

            offset = np.array([offset_x, offset_y, offset_z])
            if self.offset is not None:
                if isinstance(self.offset, float):
                    offset -= self.offset
                else:
                    offset[0] -= self.offset[0]
                    offset[1] -= self.offset[1]
                    offset[2] -= self.offset[2]

            if self.center:
                data["inputs.norm_offset"] = offset
            if self.scale:
                data["inputs.norm_scale"] = scale

        if self.scale and scale == 0:
            scale = 1

        for k, v in zip(["points", "pointcloud", "pcd100k", "inputs", "mesh.vertices", "bbox", "partnet.points"],
                        [points, pointcloud, pcd100k, inputs, mesh_vertices, bbox, partnet]):
            if isinstance(v, np.ndarray):
                v_out = v - offset
                if self.scale:
                    v_out = v_out / scale
                data[k] = v_out

        return data


class BPS(Transform):
    def __init__(self,
                 num_points: int = 1024,
                 resolution: int = 32,
                 padding: float = 0.1,
                 bounds: Tuple[float, float] = (-0.5, 0.5),
                 method: str = "kdtree",
                 feature: Union[str, List[str]] = "delta",
                 basis: str = "sphere",
                 squeeze: bool = False,
                 seed: int = 0):
        super().__init__()
        if basis == "sphere":
            text = f"({num_points} point sphere)"
        elif basis == "cube":
            text = f"({num_points} point cube)"
        else:
            text = f"({resolution}^3 resolution grid)"
        print(self.message, f"with {feature} encoding", text)

        if isinstance(feature, str):
            feature = [feature]
        self.method = method
        self.feature = feature
        self.basis_type = basis
        self.num_points = num_points
        self.res = resolution
        self.squeeze = squeeze

        box_size = (bounds[1] - bounds[0]) + padding
        if self.basis_type == "sphere":
            self.basis = generate_random_basis(self.num_points,
                                               radius=box_size / 2,
                                               seed=seed)
        elif self.basis_type == "cube":
            self.basis = np.random.default_rng(seed).random((self.num_points, 3))
            self.basis = (box_size * (self.basis - 0.5))
        elif self.basis_type == "grid":
            # self.basis = create_grid_points_from_bounds(-0.55, 0.55, res=self.res)
            self.basis = box_size * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (self.res,) * 3).numpy()
        else:
            raise ValueError

    def __call__(self, data):
        inputs = data.get("inputs")
        normals = data.get("inputs.normals")

        if self.method == "kdtree":
            # kdtree = KDTree(inputs, leafsize=100)
            # kdtree = SLKDTree(inputs, leaf_size=100)
            kdtree = KDTree(inputs, leafsize=100)
            input_dist, input_index = kdtree.query(self.basis)
        else:
            nn = NearestNeighbors(n_neighbors=1, leaf_size=100, algorithm="ball_tree").fit(inputs)
            input_dist, input_index = nn.kneighbors(self.basis)

        input_feature = list()
        for f in self.feature:
            if f == "closest":
                input_feature.append(inputs[input_index])
            elif f == "basis":
                input_feature.append(self.basis)
            elif f == "delta":
                input_feature.append(inputs[input_index] - self.basis)
            elif f == "distance":
                input_feature.append(np.expand_dims(input_dist, axis=1))
        input_feature = np.concatenate(input_feature, axis=1)

        if self.basis_type == "grid" and input_feature is not None:
            input_feature = input_feature.reshape((self.res, self.res, self.res, -1))

        data["bps.inputs"] = inputs
        data["bps.basis"] = self.basis
        data["inputs"] = input_feature.squeeze() if self.squeeze else input_feature
        if normals is not None:
            data["inputs.normals"] = normals[input_index]
        data["inputs.bps_index"] = input_index
        data["inputs.bps_dist"] = input_dist
        return data


class Scale(Transform):
    def __init__(self,
                 axes: str = "xyz",
                 amount: Union[float, Tuple[float, float], List[float]] = None,
                 random: bool = False,
                 from_inputs: bool = False,
                 multiplier: float = None):
        super().__init__()
        if from_inputs:
            print(self.message, "(from inputs)")
        else:
            if isinstance(amount, float):
                print(self.message, f"(scaling {axes} by min/max (+-){amount})")
            else:
                print(self.message, f"(scaling {axes} by {1 - amount[0]} to {1 + amount[1]})")

        self.axes = axes
        self.amount = amount
        self.random = random
        self.from_inputs = from_inputs
        self.multiplier = multiplier

    def __call__(self, data):
        if not self.axes:
            print("Warning: Didn't provide any axes. Returning.")
            return data

        points = data.get("points")

        inputs = data.get("inputs")
        normals = data.get("inputs.normals")

        pcd = data.get("pointcloud")
        pcd_normals = data.get("pointcloud.normals")
        pcd100k = data.get("pcd100k")
        pcd100k_normals = data.get("pcd100k.normals")

        mesh_vertices = data.get("mesh.vertices")
        mesh_normals = data.get("mesh.normals")

        bbox = data.get("bbox")

        partnet = data.get("partnet.points")

        if self.from_inputs:
            scale = data.get("scale")
            if scale is None:
                scale = data.get("inputs.scale")
            if scale is None:
                if self.multiplier is None:
                    scale_x = scale_y = scale_z = 1
                else:
                    scale_x = scale_y = scale_z = self.multiplier
            else:
                if isinstance(scale, float):
                    scale_x = scale if "x" in self.axes else 1
                    scale_y = scale if "y" in self.axes else 1
                    scale_z = scale if "z" in self.axes else 1
                elif isinstance(scale, (tuple, list, np.ndarray)) and len(scale) == 3 and len(self.axes) == 3:
                    if self.multiplier is None:
                        scale_x = scale[0] / scale[1]
                        scale_y = 1
                        scale_z = scale[2] / scale[1]
                    else:
                        scale_x, scale_y, scale_z = self.multiplier * scale
                else:
                    raise ValueError("Scale must be a float or a list of 3 floats.")
        elif self.amount is not None:
            if isinstance(self.amount, (float, int)):
                if self.random:
                    scale_x = np.random.uniform(1 - self.amount, 1 + self.amount) if "x" in self.axes else 1
                    scale_y = np.random.uniform(1 - self.amount, 1 + self.amount) if "y" in self.axes else 1
                    scale_z = np.random.uniform(1 - self.amount, 1 + self.amount) if "z" in self.axes else 1
                else:
                    scale_x = self.amount if "x" in self.axes else 1
                    scale_y = self.amount if "y" in self.axes else 1
                    scale_z = self.amount if "z" in self.axes else 1
            elif isinstance(self.amount, (tuple, list)):
                if len(self.amount) == 2:
                    if self.random:
                        scale = np.random.uniform(1 - self.amount[0], 1 + self.amount[1])
                        scale_x = scale if "x" in self.axes else 1
                        scale_y = scale if "y" in self.axes else 1
                        scale_z = scale if "z" in self.axes else 1
                    else:
                        print(f"Warning: Scaling {self.axes} with {self.amount} is ambiguous. Returning.")
                        return data
                elif len(self.amount) == 3 and len(self.axes) == 3:
                    if self.random:
                        scale_x = np.random.uniform(1 - self.amount[0], 1 + self.amount[0])
                        scale_y = np.random.uniform(1 - self.amount[1], 1 + self.amount[1])
                        scale_z = np.random.uniform(1 - self.amount[2], 1 + self.amount[2])
                    else:
                        scale_x, scale_y, scale_z = self.amount
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            print("Warning: No scaling amount provided and not taken from inputs. Returning.")
            return data

        scale = np.array([scale_x, scale_y, scale_z])
        if "inputs.scale" in data:
            data["inputs.scale"] = scale
        else:
            data["scale"] = scale

        for k, v in zip(["points",
                         "pointcloud",
                         "pointcloud.normals",
                         "pcd100k",
                         "pcd100k.normals",
                         "inputs",
                         "inputs.normals",
                         "mesh.vertices",
                         "mesh.normals",
                         "bbox",
                         "partnet.points"],
                        [points,
                         pcd,
                         pcd_normals,
                         pcd100k,
                         pcd100k_normals,
                         inputs,
                         normals,
                         mesh_vertices,
                         mesh_normals,
                         bbox,
                         partnet]):
            if isinstance(v, np.ndarray):
                v_scaled = v * scale
                if "normals" in k:
                    v_scaled /= np.linalg.norm(v_scaled, axis=1, keepdims=True)
                data[k] = v_scaled
        return data


class PointcloudFromMesh(Transform):
    def __init__(self,
                 num_points: int = int(1e5),
                 cache: bool = False):
        super().__init__()
        print(self.message, f"(sampling {num_points} points)")

        self.num_points = num_points
        self.cache = dict() if cache else None

    def __call__(self, data):
        mesh_name = data["mesh.name"]
        points = self.cache.get(mesh_name) if isinstance(self.cache, dict) else None
        if points is None:
            vertices = data["mesh.vertices"]
            triangles = data["mesh.triangles"]
            mesh = Trimesh(vertices, triangles)
            points = mesh.sample(self.num_points)

            if isinstance(self.cache, dict):
                self.cache[mesh_name] = points.astype(np.float32)

        data["pointcloud"] = points.copy()
        return data


class PointsFromMesh(Transform):
    def __init__(self,
                 padding: float = 0.1,
                 sigmas: List[float] = None,
                 num_samples: int = int(1e5),
                 cache: bool = False):
        super().__init__()
        print(self.message, f"(subsampling to {num_samples} points)")

        self.padding = padding
        self.sigmas = sigmas
        self.num_samples = num_samples
        self.cache = dict() if cache else None

    def __call__(self, data):
        mesh_name = data["mesh.name"]
        mesh_data = self.cache.get(mesh_name) if isinstance(self.cache, dict) else None
        if mesh_data is None:
            vertices = data["mesh.vertices"]
            triangles = data["mesh.triangles"]
            mesh = Trimesh(vertices, triangles)

            points = np.random.rand(max(int(1e5), self.num_samples), 3)
            box_size = 1 + self.padding
            points = box_size * (points - 0.5)

            if self.sigmas:
                points_list = list()
                occupancy_list = list()

                num_points = len(points) // (len(self.sigmas) + 1)
                for sigma in self.sigmas:
                    points = mesh.sample(num_points)
                    boundary_points = points + sigma * np.random.randn(*points.shape)
                    occupancy = check_mesh_contains(mesh, boundary_points)

                    points_list.append(boundary_points)
                    occupancy_list.append(occupancy)

                indices = np.random.randint(len(points), size=num_points)
                points_list.append(points[indices])
                occupancy_list.append(check_mesh_contains(mesh, points[indices]))

                points = np.concatenate(points_list)
                occupancy = np.concatenate(occupancy_list)
            else:
                occupancy = check_mesh_contains(mesh, points)

            mesh_data = {"points": points.astype(np.float32),
                         "occupancy": occupancy.astype(bool)}

            if isinstance(self.cache, dict):
                self.cache[mesh_name] = mesh_data

        indices = subsample(mesh_data["points"], self.num_samples)

        data["points"] = mesh_data["points"][indices].copy()
        data["points.occ"] = mesh_data["occupancy"][indices].copy()
        return data


class CropPoints(Transform):
    def __init__(self,
                 padding: float = 0.1,
                 cache: bool = False,
                 verbose: bool = False):
        super().__init__()
        print(self.message, f"(cache: {cache})")

        self.padding = padding
        # self.box = create_box((1 + self.padding / 2,) * 3)
        self.cache = dict() if cache else None
        self.verbose = verbose

    def __call__(self, data):
        points_key = "points" if "points" in data else None
        occ_key = "points.occ" if "points.occ" in data else "occ"
        inputs_path = data.get("inputs.path")

        points_data = None
        if inputs_path is not None:
            points_data = self.cache.get(inputs_path) if isinstance(self.cache, dict) else None

        if points_data is None:
            points = data[points_key]
            occ = data[occ_key]

            mask = np.all(np.abs(points) <= 0.5 + self.padding / 2, axis=1)
            if mask.sum() > 0:
                points = points[mask]
                occ = occ[mask]

            if inputs_path is not None and isinstance(self.cache, dict):
                if self.verbose:
                    print(f"{self.__class__.__name__}: Caching cropped points.")
                self.cache[inputs_path] = {points_key: points.astype(np.float32),
                                           occ_key: occ}
        else:
            if self.verbose:
                print(f"{self.__class__.__name__}: Retrieving cropped points from cache.")
            points = points_data[points_key]
            occ = points_data[occ_key]

        """
        box_occ = check_mesh_contains(self.box, points)
        points = points[box_occ]
        occ = occ[box_occ]
        """

        data[points_key] = points.copy()
        data[occ_key] = occ.copy()

        return data


class Visualize(Transform):
    def __init__(self,
                 show_inputs: bool = True,
                 show_occupancy: bool = True,
                 show_points: bool = False,
                 show_frame: bool = True,
                 show_box: bool = True,
                 show_bbox: bool = False,
                 show_cam: bool = False,
                 show_mesh: bool = False,
                 show_pointcloud: bool = False,
                 threshold: float = 0.5,
                 sdf: bool = False,
                 padding: float = 0.1,
                 n_calls: int = None,
                 cam_forward: Tuple[int, int, int] = (0, 0, 1),
                 cam_up: Tuple[int, int, int] = (0, 1, 0),
                 frame_size: float = 0.5):
        super().__init__()
        print(self.message)

        self.show_inputs = show_inputs
        self.show_occupancy = show_occupancy
        self.show_points = show_points
        self.show_frame = show_frame
        self.show_box = show_box
        self.show_bbox = show_bbox
        self.show_cam = show_cam
        self.show_mesh = show_mesh
        self.show_pointcloud = show_pointcloud
        self.threshold = threshold
        self.sdf = sdf
        self.padding = padding
        self.n_calls = n_calls
        self.call_counter = 0
        self.cam_forward = cam_forward
        self.cam_up = cam_up
        self.frame_size = frame_size
        self.unnormalize = Compose([NormalizeImage(mean=np.zeros(3), std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                    NormalizeImage(mean=[-0.485, -0.456, -0.406], std=np.ones(3))])
        self.to_pil_image = ToPILImage()

    def __call__(self, data):
        if self.n_calls:
            self.call_counter += 1
            if self.call_counter > self.n_calls:
                return data
        geometries = list()

        """
        if "partnet.points" in data:
            points = data["partnet.points"]
            labels = data["partnet.labels"]
            colors = get_partnet_colors()[labels]
            points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            points.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(points)
        """

        if self.show_inputs:
            if "bps.basis" in data:
                basis = data["bps.basis"]
                inputs = data["bps.inputs"]
                index = data["inputs.bps_index"]
                dist = data["inputs.bps_dist"]

                norm_dist = (dist - dist.min()) / (dist.max() - dist.min())
                plasma = get_cmap("plasma")
                greys = get_cmap("Greys")
                # viridis = get_cmap("viridis")
                #
                point_colors = np.array([plasma(d) for d in norm_dist])[:, :3]
                line_colors = np.array([greys(1 - d) for d in norm_dist])[:, :3]
                # bar_colors = [viridis(d) for d in norm_dist]
                #
                # plt.bar(np.arange(len(dist)), dist, color=bar_colors, width=1)
                # plt.show()

                basis_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(basis))
                basis_points.colors = o3d.utility.Vector3dVector(point_colors)
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(np.concatenate([inputs[index], basis], axis=0))
                lines = list(zip(np.arange(len(index)).astype(int),
                                 len(index) + np.arange(len(basis)).astype(int)))
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
                selection = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs[np.unique(index)]))
                selection.paint_uniform_color((0, 0, 0))
                inputs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs))
                geometries.extend([basis_points, inputs, selection, line_set])
            else:
                inputs = data["inputs"]
                if isinstance(inputs, np.ndarray):
                    normals = data.get("inputs.normals")
                    if len(inputs.shape) == 3:
                        if normals is not None and len(normals.shape) == 4:
                            normals = normals.reshape(-1, 3)
                            normals = normals[inputs.ravel() == 1]

                        min_bound = -0.5 - self.padding / 2
                        max_bound = 0.5 + self.padding / 2
                        grid = make_3d_grid((min_bound,) * 3, (max_bound,) * 3, (inputs.shape[0],) * 3).numpy()
                        inputs = grid[inputs.ravel() == 1]

                    inputs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs))
                    if normals is not None:
                        inputs.normals = o3d.utility.Vector3dVector(normals)

                    colors = data.get("inputs.colors")
                    labels = data.get("inputs.labels")
                    if colors is not None:
                        inputs.colors = o3d.utility.Vector3dVector(colors)
                    elif labels is not None:
                        inputs.colors = o3d.utility.Vector3dVector(get_partnet_colors()[labels])
                    elif inputs.normals:
                        colors = np.asarray(inputs.normals) / 2 + 0.5
                        inputs.colors = o3d.utility.Vector3dVector(colors)

                    geometries.append(inputs)
                elif isinstance(inputs, Image.Image):
                    inputs.show()
                elif isinstance(inputs, torch.Tensor):
                    self.to_pil_image(self.unnormalize(inputs)).show()
                else:
                    raise TypeError(type(inputs))

            image = data.get("inputs.image")
            if isinstance(image, Image.Image):
                image.show()
            elif isinstance(image, (torch.Tensor, np.ndarray)):
                self.to_pil_image(self.unnormalize(image)).show()

            depth_map = data.get("inputs.depth_map")
            if isinstance(depth_map, Image.Image):
                depth_map.show()
            elif isinstance(depth_map, (torch.Tensor, np.ndarray)):
                depth_map[depth_map == 0] = 1.02 * depth_map.max()
                depth_map = normalize(depth_map, depth_map.min(), depth_map.max())
                cmap = get_cmap("Greys")
                depth_image = cmap(depth_map)
                Image.fromarray((depth_image[..., :3] * 255).astype(np.uint8)).show()

            normal_map = data.get("inputs.normal_map")
            if isinstance(normal_map, Image.Image):
                normal_map.show()
            elif isinstance(normal_map, (torch.Tensor, np.ndarray)):
                Image.fromarray((normal_map * 255).astype(np.uint8)).show()

        if self.show_occupancy or self.show_points:
            points_occ = data["points.occ"]
            indices = points_occ <= self.threshold if self.sdf else ((points_occ >= self.threshold) & (points_occ <= 1))
            occ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["points"][indices]))
            labels = data.get("points.labels")
            if labels is None:
                occ = occ.paint_uniform_color((0.7, 0.7, 0.7))
            else:
                occ.colors = o3d.utility.Vector3dVector(get_partnet_colors()[labels[indices]])
            if points_occ.max() == 2:
                indices = points_occ == 2
                uncertain = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["points"][indices]))
                uncertain.paint_uniform_color((171 / 255, 99 / 255, 250 / 255))
                geometries.append(uncertain)
            if self.show_points:
                outside = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["points"][~indices]))
                outside.paint_uniform_color((0.9, 0.9, 0.9))
                geometries.append(outside)
            geometries.append(occ)

        if self.show_frame:
            frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=self.frame_size)
            geometries.append(frame)

        if self.show_box:
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.5 - self.padding / 2,) * 3,
                                                      max_bound=(0.5 + self.padding / 2,) * 3)
            box.color = np.zeros(3)
            geometries.append(box)

        if self.show_bbox:
            bbox = data.get("bbox")
            if bbox is not None:
                bbox_points = o3d.utility.Vector3dVector(bbox)
                bbox_pcd = o3d.geometry.PointCloud(bbox_points)
                bbox_pcd.paint_uniform_color((1, 0, 0))
                geometries.append(bbox_pcd)

                lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]]
                colors = [[1, 0, 0] for _ in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = bbox_points
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                geometries.append(line_set)

        if self.show_cam:
            cam = data.get("inputs.cam")
            rot = data.get("inputs.rot")
            inv_extrinsic = np.eye(4)
            inv_extrinsic[:3, :3] = np.eye(3) if rot is None else rot.T
            inv_extrinsic[:3, 3] = np.zeros(3) if cam is None else cam
            cam = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1).transform(inv_extrinsic)
            geometries.append(cam)

        if self.show_mesh:
            if "mesh" in data:
                mesh = data["mesh"]
                if isinstance(mesh, Trimesh):
                    data["mesh.vertices"] = mesh.vertices
                    data["mesh.triangles"] = mesh.faces
            if "mesh.vertices" in data and "mesh.triangles" in data:
                if data["mesh.triangles"] is None:
                    mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["mesh.vertices"]))
                    mesh = mesh.paint_uniform_color((102 / 255, 102 / 255, 102 / 255))
                else:
                    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data["mesh.vertices"]),
                                                     o3d.utility.Vector3iVector(data["mesh.triangles"]))
                    if "mesh.normals" in data:
                        mesh.vertex_normals = o3d.utility.Vector3dVector(data["mesh.normals"])
                    if "mesh.colors" in data:
                        mesh.vertex_colors = o3d.utility.Vector3dVector(data["mesh.colors"])
                    if "mesh.textures" in data and "mesh.uvs" in data and "mesh.ids" in data:
                        mesh.textures = [o3d.geometry.Image(texture) for texture in data["mesh.textures"]]
                        mesh.triangle_uvs = o3d.utility.Vector2dVector(data["mesh.uvs"])
                        mesh.triangle_material_ids = o3d.utility.IntVector(data["mesh.ids"])

                    if not mesh.has_vertex_colors() and not mesh.has_textures():
                        mesh = mesh.paint_uniform_color((102 / 255, 102 / 255, 102 / 255))
                    if not mesh.has_vertex_normals():
                        mesh.compute_vertex_normals()
                    if not mesh.has_triangle_normals():
                        mesh.compute_triangle_normals()
                geometries.append(mesh)
            else:
                print("Warning: No mesh data available for object", data["obj_name"])

        if self.show_pointcloud:
            if "pointcloud" in data:
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["pointcloud"]))
                if "pointcloud.normals" in data:
                    pcd.normals = o3d.utility.Vector3dVector(data["pointcloud.normals"])
                if "pointcloud.labels" in data:
                    pcd.colors = o3d.utility.Vector3dVector(get_partnet_colors()[data["pointcloud.labels"]])
                else:
                    pcd = pcd.paint_uniform_color([0, 0, 0])
                geometries.append(pcd)
            else:
                print("Warning: No pointcloud data available for object", data["obj_name"])

        if self.show_pointcloud and self.show_points and "pointcloud.index" in data:
            points = data["points"]
            dist = data["points.occ"]
            pcd = data["pointcloud"]
            index = data["pointcloud.index"]

            if index.max() == len(pcd):
                index = index[index < len(pcd)]
                points = points[dist < dist.max()]
                dist = dist[dist < dist.max()]

            cmap = get_cmap("bwr")
            inside = dist < 0
            outside = dist >= 0
            norm_dist = np.zeros(len(dist))
            norm_dist[inside] = normalize(-dist[inside], 0, 0.5)
            norm_dist[outside] = normalize(-dist[outside], 0.5, 1)
            line_colors = np.array([cmap(d) for d in norm_dist])[:, :3]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.concatenate([pcd[index], points], axis=0))
            lines = list(zip(np.arange(len(index)).astype(int),
                             len(index) + np.arange(len(points)).astype(int)))
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            geometries.append(line_set)

        if not geometries:
            print("Warning: No data to show")

        o3d.visualization.draw_geometries(geometries,
                                          window_name=data["inputs.path"],
                                          mesh_show_back_face=True,
                                          mesh_show_wireframe=False,
                                          zoom=1 if self.show_box else 0.75,
                                          lookat=(0, 0, 0),
                                          front=self.cam_forward,
                                          up=self.cam_up)
        return data


class KeysToKeep(Transform):
    def __init__(self,
                 preset: str = "train",
                 keys: Optional[List[str]] = None):
        super().__init__()
        print(self.message)

        if keys:
            self.keys = keys
        else:
            if preset == "train":
                self.keys = ["index",
                             "category.index",
                             "inputs",
                             "inputs.cos_sim",
                             "inputs.skip",
                             "inputs.path",
                             "inputs.name",
                             "inputs.labels",
                             "inputs.nerf"
                             "points",
                             "points.occ",
                             "points.labels",
                             "points.nerf",
                             "pointcloud",
                             "pointcloud.normals",
                             "pointcloud.index",
                             "pointcloud.labels",
                             "pointcloud.nerf",
                             "voxels",
                             "mesh.vertices",
                             "mesh.triangles",
                             "mesh.labels",
                             "bbox"]
            elif preset == "visualize":
                self.keys = ["index",
                             "category.index",
                             "inputs",
                             "inputs.cos_sim",
                             "inputs.rot",
                             "inputs.normals",
                             "inputs.colors"
                             "inputs.image",
                             "inputs.depth",
                             "inputs.up",
                             "inputs.visib_fract",
                             "inputs.path",
                             "inputs.skip",
                             "inputs.name",
                             "inputs.cam",
                             "inputs.index",
                             "inputs.intrinsic",
                             "inputs.extrinsic",
                             "points",
                             "points.occ",
                             "pointcloud",
                             "pointcloud.normals",
                             "pointcloud.index",
                             "pointcloud.labels",
                             "voxels",
                             "mesh.vertices",
                             "mesh.triangles",
                             "mesh.labels",
                             "bbox",
                             "partnet.points",
                             "partnet.labels"]

    def add_key(self, key: str):
        self.keys.append(key)

    def add_keys(self, keys: List[str]):
        self.keys.extend(keys)

    def remove_key(self, key: str):
        self.keys.remove(key)

    def __call__(self, data):
        return {k: data[k] for k in data.keys() & self.keys}


class CheckDtype(Transform):
    def __init__(self, float16_to_float32: bool = True):
        super().__init__()
        print(self.message)

        self.float16_to_float32 = float16_to_float32

    def cast(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            if data.dtype in [int, np.uint32, np.int64]:
                return data
            if self.float16_to_float32 and data.dtype == np.float16:
                # float16 only represents 3.31 decimal digits
                data = data.astype(np.float32) + 1e-4 * np.random.randn(*data.shape)
            return data.astype(np.float32)
        elif isinstance(data, Path):
            return str(data)
        return data

    def __call__(self, data):
        return {k: self.cast(v) for k, v in data.items()}


class Compress(Transform):
    def __init__(self,
                 dtype: np.dtype = np.float32,
                 packbits: bool = True):
        assert dtype in [np.float16, np.float32], f"Cannot compress to {dtype}."

        super().__init__()
        print(self.message)

        self.dtype = dtype
        self.packbits = packbits

    def compress(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            if data.dtype == np.float64:
                return data.astype(self.dtype)
            if self.packbits and data.dtype == bool:
                return np.packbits(data)
        return data

    def __call__(self, data):
        return {k: self.compress(v) for k, v in data.items()}


class Unpack(Transform):
    def __init__(self):
        super().__init__()
        print(self.message)

    @staticmethod
    def unpack(data: Any) -> Any:
        if isinstance(data, np.ndarray) and data.dtype == np.uint8:
            return np.unpackbits(data)
        return data

    def __call__(self, data):
        return {k: self.unpack(v) for k, v in data.items()}


__all__ = ['Transform',
           'apply_transform',
           'ReturnTransform',
           'RandomChoice',
           'RandomApply',
           'MinMaxNumPoints',
           'BoundingBox',
           'BoundingBoxJitter',
           'NormalsCameraCosineSimilarity',
           'AngleOfIncidenceRemoval',
           'EdgeNoise',
           'ImageToTensor',
           'Rotate',
           'PointcloudNoise',
           'SubsamplePointcloud',
           'ProcessPointcloud',
           'RotatePointcloud',
           'RotateMesh',
           'ScaleMesh',
           'ScalePoints',
           'TransformMesh',
           'NormalizeMesh',
           'ApplyPose',
           'AddRandomRotation',
           'ProcessInputs',
           'InputNormalsFromPointcloud',
           'CropPointcloud',
           'AxesCutPointcloud',
           'SphereCutPointcloud',
           'RenderPointcloud',
           'RenderDepthMaps',
           'DepthToPointcloud',
           'RenderDepthMap',
           'FindUncertainPoints',
           'LoadUncertain',
           'DepthLikePointcloud',
           'RemoveHiddenPointsFromInputs',
           'SubsamplePoints',
           'SdfFromOcc',
           'Voxelizer',
           'VoxelizeInputs',
           'VoxelizePoints',
           'Translate',
           'Normalize',
           'BPS',
           'Scale',
           'PointcloudFromMesh',
           'PointsFromMesh',
           'CropPoints',
           'Visualize',
           'KeysToKeep',
           'CheckDtype',
           'Compress',
           'Unpack',
           'SegmentationFromPartNet',
           'NeRFEncoding']
