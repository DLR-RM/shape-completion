import json
import os
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import open3d as o3d
import pyrender
import torch
import torchvision.transforms.v2 as tv_transforms
from easy_o3d.interfaces import MyRegistrationResult
from easy_o3d.registration import ICPTypes, IterativeClosestPoint, KernelTypes
from easy_o3d.utils import (
    DownsampleTypes,
    OrientationTypes,
    OutlierTypes,
    SearchParamTypes,
    convert_depth_image_to_point_cloud,
    convert_rgbd_image_to_point_cloud,
    get_point_cloud_from_points,
    process_point_cloud,
)
from einops import rearrange
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from PIL import Image, ImageShow
from pykdtree.kdtree import KDTree
from pyrender.shader_program import ShaderProgramCache
from scipy.spatial import Delaunay
from scipy.spatial import KDTree as SKDTree
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torchvision import models
from torchvision.transforms import CenterCrop, Compose, Resize, ToPILImage, ToTensor
from torchvision.transforms import Normalize as NormalizeImage
from trimesh import PointCloud, Trimesh

from libs import check_mesh_contains
from utils import (
    PLOTLY_COLORS,
    Voxelizer,
    adjust_intrinsic,
    apply_trafo,
    convert_extrinsic,
    default_on_exception,
    depth_to_image,
    depth_to_points,
    draw_camera,
    filter_dict,
    generate_random_basis,
    get_args,
    get_rays,
    inv_trafo,
    is_in_frustum,
    look_at,
    make_3d_grid,
    normalize,
    points_to_depth,
    points_to_uv,
    resolve_dtype,
    rot_from_euler,
    sample_distances,
    setup_logger,
    subsample_indices,
    to_tensor,
)

logger = setup_logger(__name__)

DataDict = dict[Any, Any]


def _as_point_cloud(result: Any) -> o3d.geometry.PointCloud:
    if isinstance(result, tuple):
        return cast(o3d.geometry.PointCloud, result[0])
    return cast(o3d.geometry.PointCloud, result)


def _log_debug_level_1(message: str) -> None:
    debug_level_1 = getattr(logger, "debug_level_1", None)
    if callable(debug_level_1):
        debug_level_1(message)
    else:
        logger.debug(message)


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


try:
    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes
except ImportError:
    logger.warning("The 'PyTorch3D' module is not installed. Some transformations will not be available.")


class Transform(ABC):
    @get_args()
    def __init__(
        self,
        apply_to: str | list[str] | tuple[str, ...] | None = None,
        allowed: str | list[str] | tuple[str, ...] | None = None,
        cachable: bool = False,
    ):
        if isinstance(apply_to, str):
            self.apply_to: set[str] | str | None = {apply_to}
        elif apply_to is None:
            self.apply_to = None
        else:
            self.apply_to = set(apply_to)
        self.cachable = cachable

        if allowed is None:
            allowed_set = {
                "inputs",
                "inputs.depth",
                "inputs.normals",
                "inputs.image",
                "points",
                "pointcloud",
                "pointcloud.normals",
                "mesh.vertices",
                "mesh.normals",
                "voxels",
                "bbox",
                "partnet.points",
            }
        elif isinstance(allowed, str):
            allowed_set = {allowed}
        else:
            allowed_set = set(allowed)

        if self.apply_to is not None:
            if isinstance(self.apply_to, str):
                apply_set = {cast(str, self.apply_to)}
            else:
                apply_set = cast(set[str], self.apply_to)
            assert all(key in allowed_set for key in apply_set), f"Invalid key in apply_to: {self.apply_to}"

    def before_apply(self, data: DataDict) -> DataDict:
        return data

    @abstractmethod
    def apply(self, data: DataDict, key: str | None) -> DataDict:
        raise NotImplementedError("Method must be implemented in derived classes.")

    def after_apply(self, data: DataDict) -> DataDict:
        return data

    def __call__(self, data: DataDict) -> DataDict:
        try:
            data = self.before_apply(data)
            if self.apply_to is None:
                data = self.apply(data, key=None)
                return self.after_apply(data)
            if isinstance(self.apply_to, str):
                apply_set = {cast(str, self.apply_to)}
            else:
                apply_set = cast(set[str], self.apply_to)
            for key in list(apply_set & set(data.keys())):
                data = self.apply(data, key)
            return self.after_apply(data)
        except Exception as e:
            logger.exception(f"Transformation {self.name} failed with error: {e}")
            print(data)
            raise e

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self) -> dict[str, Any]:
        return cast(dict[str, Any], getattr(self, "_args", {}))

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


def apply_transforms(data: DataDict, transforms: list[Transform] | Transform | Compose | None = None) -> DataDict:
    if transforms is None:
        return data

    if isinstance(transforms, list):
        for t in transforms:
            trafo_timer = time.perf_counter()
            data = t(data)
            logger.debug(f"Transform {t.name} takes {time.perf_counter() - trafo_timer:.4f}s.")
    elif isinstance(transforms, (Transform, Compose)):
        trafo_timer = time.perf_counter()
        data = transforms(data)
        if isinstance(transforms, Transform):
            logger.debug(f"Transform {transforms.name} takes {time.perf_counter() - trafo_timer:.4f}s.")
        else:
            logger.debug(f"Compose transform takes {time.perf_counter() - trafo_timer:.4f}s.")
    else:
        raise TypeError("'transforms' must be a single or a list of transforms or a `Compose` object.")
    return data


class Return(Transform):
    def __init__(self):
        super().__init__()

    def apply(self, data, key):
        return data


class RandomChoice(Transform):
    @get_args()
    def __init__(self, transforms: list[Transform], p: list[float] | None = None):
        super().__init__()
        self.transformations = transforms
        self.p = p

    def apply(self, data, key):
        transformation_idx = int(np.random.choice(len(self.transformations), p=self.p))
        transformation = self.transformations[transformation_idx]
        return transformation(data)


class RandomApply(Transform):
    @get_args()
    def __init__(self, transform: Transform, p: float = 0.5):
        super().__init__()
        self.transform = transform
        self.p = p

    def apply(self, data, key):
        if np.random.rand() < self.p:
            return self.transform(data)
        return data


class Torchvision(Transform):
    @get_args()
    def __init__(self, transform: tv_transforms.Transform, apply_to=None, **kwargs):
        super().__init__(apply_to=apply_to)
        self.transform = transform(**kwargs)

    def apply(self, data, key):
        image = data[key]
        transforms = self.transform
        if not isinstance(image, Tensor):
            transforms = Compose([ToTensor(), self.transform])
        data[key] = transforms(image)
        return data


class Permute(Transform):
    @get_args()
    def __init__(self, apply_to=("inputs", "points", "pointcloud")):
        super().__init__(apply_to=apply_to, allowed=("inputs", "points", "pointcloud"))

    def apply(self, data, key):
        value = data[key]
        if isinstance(value, np.ndarray):
            idx = np.random.permutation(len(value))
            data[key] = value[idx]
            if key == "points":
                data["points.occ"] = data["points.occ"][idx]
            elif f"{key}.normals" in data:
                data[f"{key}.normals"] = data[f"{key}.normals"][idx]
        return data


class MinMaxNumPoints(Transform):
    @get_args()
    def __init__(
        self,
        min_num_points: dict[str, int] | None = None,
        max_num_points: dict[str, int] | None = None,
        apply_to=("inputs", "points", "pointcloud"),
    ):
        super().__init__(apply_to=apply_to)

        self.min_num_points = min_num_points or dict()
        self.max_num_points = max_num_points or dict()

    def apply(self, data, key):
        if key == "inputs" and key in data:
            inputs = data["inputs"]
            normals = data.get("inputs.normals")
            depth = data.get("inputs.depth")
            colors = data.get("inputs.colors")
            labels = data.get("inputs.labels")
            if isinstance(labels, np.ndarray) and len(labels) != len(inputs):
                labels = None

            if "inputs" in self.min_num_points:
                min_num_points = self.min_num_points["inputs"] or 0
                if depth is not None:
                    if len(depth) == 0:
                        data["inputs.depth"] = np.zeros((min_num_points, 3))
                    elif len(depth) < min_num_points:
                        data["inputs.depth"] = depth[subsample_indices(depth, min_num_points)]
                if inputs is None or len(inputs) == 0:
                    logger.warning(f"{self.name}: No input points for {data['inputs.path']}")
                    data["inputs.skip"] = True

                    data["inputs"] = np.zeros((min_num_points, 3))
                    if normals is not None:
                        data["inputs.normals"] = np.zeros((min_num_points, 3))
                    if colors is not None:
                        data["inputs.colors"] = np.zeros((min_num_points, 3))
                    if isinstance(labels, np.ndarray):
                        data["inputs.labels"] = np.zeros(min_num_points, dtype=np.int64)
                elif len(inputs) < min_num_points:
                    indices = subsample_indices(inputs, min_num_points)
                    data["inputs"] = inputs[indices]
                    if normals is not None:
                        data["inputs.normals"] = normals[indices]
                    if colors is not None:
                        data["inputs.colors"] = colors[indices]
                    if isinstance(labels, np.ndarray):
                        data["inputs.labels"] = labels[indices]
            if "inputs" in self.max_num_points:
                max_num_points = self.max_num_points["inputs"]
                if (
                    max_num_points is not None
                    and max_num_points > 0
                    and depth is not None
                    and len(depth) > max_num_points
                ):
                    data["inputs.depth"] = depth[subsample_indices(depth, max_num_points)]
                if max_num_points is not None and max_num_points > 0 and len(inputs) > max_num_points:
                    indices = subsample_indices(inputs, max_num_points)
                    data["inputs"] = inputs[indices]
                    if normals is not None:
                        data["inputs.normals"] = normals[indices]
                    if colors is not None:
                        data["inputs.colors"] = colors[indices]
                    if isinstance(labels, np.ndarray):
                        data["inputs.labels"] = labels[indices]
        elif key == "points" and key in data:
            points = data["points"]
            occ = data["points.occ"]
            labels = data.get("points.labels")
            if "points" in self.min_num_points:
                min_num_points = self.min_num_points["points"] or 0
                if len(points) == 0:
                    logger.warning(f"{self.name}: No query points for {data['points.path']}.")
                    data["inputs.skip"] = True

                    data["points"] = np.zeros((min_num_points, 3))
                    data["points.occ"] = np.ones(min_num_points, dtype=bool)
                    if isinstance(labels, np.ndarray):
                        data["points.labels"] = np.zeros(min_num_points, dtype=np.int64)
                elif len(points) < min_num_points:
                    indices = subsample_indices(points, min_num_points)
                    data["points"] = points[indices]
                    data["points.occ"] = occ[indices]
                    if isinstance(labels, np.ndarray):
                        data["points.labels"] = labels[indices]
            if "points" in self.max_num_points:
                max_num_points = self.max_num_points["points"]
                if max_num_points is not None and max_num_points > 0 and len(points) > max_num_points:
                    indices = subsample_indices(points, max_num_points)
                    data["points"] = points[indices]
                    data["points.occ"] = occ[indices]
                    if isinstance(labels, np.ndarray):
                        data["points.labels"] = labels[indices]
        elif key == "pointcloud" and key in data:
            pointcloud = data["pointcloud"]
            normals = data.get("pointcloud.normals")
            labels = data.get("pointcloud.labels")
            if "pointcloud" in self.min_num_points:
                min_num_points = self.min_num_points["pointcloud"] or 0
                if len(pointcloud) == 0:
                    logger.warning(f"{self.name}: Point cloud {data['pointcloud.path']} has no points.")
                    data["inputs.skip"] = True

                    data["pointcloud"] = np.zeros((min_num_points, 3))
                    if normals is not None:
                        data["pointcloud.normals"] = np.zeros((min_num_points, 3))
                    if isinstance(labels, np.ndarray):
                        data["pointcloud.labels"] = np.zeros(min_num_points, dtype=np.int64)
                elif len(pointcloud) < min_num_points:
                    indices = subsample_indices(pointcloud, min_num_points)
                    data["pointcloud"] = pointcloud[indices]
                    if normals is not None:
                        data["pointcloud.normals"] = normals[indices]
                    if isinstance(labels, np.ndarray):
                        data["pointcloud.labels"] = labels[indices]
            if "pointcloud" in self.max_num_points:
                max_num_points = self.max_num_points["pointcloud"]
                if max_num_points is not None and max_num_points > 0 and len(pointcloud) > max_num_points:
                    indices = subsample_indices(pointcloud, max_num_points)
                    data["pointcloud"] = pointcloud[indices]
                    if normals is not None:
                        data["pointcloud.normals"] = normals[indices]
                    if isinstance(labels, np.ndarray):
                        data["pointcloud.labels"] = labels[indices]
        return data


class BoundingBox(Transform):
    @get_args()
    def __init__(self, reference: str = "pointcloud", remove_reference: bool = False):
        super().__init__(apply_to=reference)

        self.reference = reference
        self.remove_reference = remove_reference

    @staticmethod
    def get_aabb_from_pts(points: np.ndarray) -> np.ndarray:
        ub, lb = np.max(points, axis=0), np.min(points, axis=0)
        borders = [
            [lb[0], lb[1], lb[2]],
            [lb[0], lb[1], ub[2]],
            [lb[0], ub[1], lb[2]],
            [lb[0], ub[1], ub[2]],
            [ub[0], lb[1], lb[2]],
            [ub[0], lb[1], ub[2]],
            [ub[0], ub[1], lb[2]],
            [ub[0], ub[1], ub[2]],
        ]
        return np.asarray(borders)

    def apply(self, data, key):
        ref = data[key]
        bbox = self.get_aabb_from_pts(ref)

        data["bbox"] = bbox.copy()
        if self.remove_reference:
            data.pop(key)
        return data


class BoundingBoxJitter(Transform):
    @get_args()
    def __init__(self, max_jitter: list[float]):
        super().__init__(apply_to="bbox")

        self.max_jitter = max_jitter

    def apply(self, data, key):
        dx = np.random.uniform(-self.max_jitter[0], self.max_jitter[0])
        dy = np.random.uniform(-self.max_jitter[1], self.max_jitter[1])
        dz = np.random.uniform(-self.max_jitter[2], self.max_jitter[2])
        data[key] += np.asarray([dx, dy, dz])
        return data


class SegmentationFromPartNet(Transform):
    @get_args()
    def __init__(self, apply_to=("inputs", "points", "pointcloud", "mesh"), num_classes: int = 50):
        super().__init__(apply_to=apply_to)
        self.num_classes = num_classes
        self.cat_part_map = {
            "03642806_2": 29,
            "03642806_1": 28,
            "04379243_1": 47,
            "03636649_4": 27,
            "02773838_2": 5,
            "02773838_1": 4,
            "03797390_1": 36,
            "03636649_1": 24,
            "03636649_2": 25,
            "03636649_3": 26,
            "02691156_4": 3,
            "02691156_1": 0,
            "04379243_2": 48,
            "02691156_3": 2,
            "02691156_2": 1,
            "02954340_1": 6,
            "02954340_2": 7,
            "04099429_2": 42,
            "04099429_3": 43,
            "04099429_1": 41,
            "03261776_3": 18,
            "03261776_2": 17,
            "03261776_1": 16,
            "02958343_3": 10,
            "02958343_2": 9,
            "02958343_1": 8,
            "03467517_2": 20,
            "03467517_3": 21,
            "04379243_3": 49,
            "02958343_4": 11,
            "04225987_2": 45,
            "04225987_3": 46,
            "04225987_1": 44,
            "03790512_3": 32,
            "03790512_1": 30,
            "03790512_2": 31,
            "03467517_1": 19,
            "03790512_4": 33,
            "03790512_5": 34,
            "03790512_6": 35,
            "03797390_2": 37,
            "03001627_4": 15,
            "03001627_1": 12,
            "03001627_2": 13,
            "03001627_3": 14,
            "03948459_1": 38,
            "03948459_3": 40,
            "03948459_2": 39,
            "03624134_1": 22,
            "03624134_2": 23,
        }

    def apply(self, data, key):
        partnet_points = data["partnet.points"]
        partnet_labels = data["partnet.labels"]
        category = data["category.id"]
        for index, part_label in enumerate(np.unique(partnet_labels)):
            partnet_labels[partnet_labels == part_label] = self.cat_part_map[f"{category}_{index + 1}"]

        if key is None:
            idx = subsample_indices(np.random.permutation(len(partnet_points)), len(data["inputs"]))
            data["inputs"] = partnet_points[idx]
            data["inputs.labels"] = partnet_labels[idx]
            return data

        kdtree = KDTree(partnet_points)
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
                data["points.labels"][occ] = partnet_labels[kdtree.query(value[occ], k=1, eps=0)[1]]
            else:
                data[f"{key}.labels"] = partnet_labels[kdtree.query(value, k=1, eps=0)[1]]

        return data


class NormalsCameraCosineSimilarity(Transform):
    @get_args()
    def __init__(self, apply_to="inputs", remove_normals: bool = False):
        super().__init__(apply_to=apply_to)

        self.remove_normals = remove_normals

    def apply(self, data, key):
        normals = data[f"{key}.normals"]
        rot = data[f"{key}.rot"]
        cam = rot.T[:, 2]
        cos_sim = np.dot(normals, cam) / (np.linalg.norm(normals, axis=1) * np.linalg.norm(cam))
        data[f"{key}.cos_sim"] = cos_sim
        if self.remove_normals:
            data.pop(f"{key}.normals")
        return data


class AngleOfIncidenceRemoval(Transform):
    @get_args()
    def __init__(
        self,
        apply_to="inputs",
        random: bool = True,
        cos_sim_threshold: float | None = None,
        remove_cos_sim: bool = False,
    ):
        super().__init__(apply_to=apply_to)

        self.apply_to = apply_to
        self.random = random
        self.cos_sim_threshold = cos_sim_threshold
        self.remove_cos_sim = remove_cos_sim

    def apply(self, data, key):
        points = data[key]
        normals = data.get(f"{key}.normals")
        cos_sim = data[f"{key}.cos_sim"]

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

        if indices.any():
            return data

        data[key] = points[indices]
        if normals is not None:
            data[f"{key}.normals"] = normals[indices]
        if self.remove_cos_sim:
            data.pop(f"{key}.cos_sim")
        else:
            data[f"{key}.cos_sim"] = cos_sim[indices]
        colors = data.get(f"{key}.colors")
        if colors is not None:
            data[f"{key}.colors"] = colors[indices]

        return data


class EdgeNoise(Transform):
    @get_args()
    def __init__(
        self,
        apply_to="inputs",
        stddev: float = 0.005,
        angle_threshold: float | None = None,
        remove_cos_sim: bool = False,
    ):
        super().__init__(apply_to=apply_to)

        self.stddev = stddev
        self.angle_threshold = angle_threshold
        self.remove_cos_sim = remove_cos_sim

    def apply(self, data, key):
        points = data[key]
        cos_sim = data[f"{key}.cos_sim"]

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

        data[key] = points
        if self.remove_cos_sim:
            data.pop(f"{key}.cos_sim")
        return data


class ImageBorderNoise(Transform):
    @get_args()
    def __init__(self, apply_to, range: tuple[int, int] = (230, 255)):
        super().__init__(apply_to=apply_to)
        self.range = range

    def apply(self, data, key):
        image = data[key]
        white_mask = np.all(image > self.range[0], axis=2) & np.all(image < self.range[1], axis=2)
        noise = 255 * np.random.randn(*image.shape)
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        image[white_mask] = noise[white_mask]
        data[key] = image
        return data


class ImageToTensor(Transform):
    @get_args()
    def __init__(
        self,
        apply_to="inputs",
        resize: int | None = 256,
        crop: int | None = 224,
        normalize: bool = True,
        format: str = "ResNet18_Weights",
    ):
        super().__init__(apply_to=apply_to)

        self.transformation: Any = list()
        if format and hasattr(models, format):
            weights = getattr(models, format)
            self.transformation = weights.DEFAULT.transforms()
            print(self.transformation)
        else:
            transformations: list[Any] = []
            if resize is not None:
                image_module = cast(Any, Image)
                bicubic = image_module.Resampling.BICUBIC if hasattr(Image, "Resampling") else image_module.BICUBIC
                transformations.append(Resize(resize, interpolation=bicubic))
            if crop is not None:
                transformations.append(CenterCrop(crop))
            transformations.append(ToTensor())
            if normalize:
                if format == "detectron2":
                    mean = np.array([103.530, 116.280, 123.675]) / 255.0
                    std = np.array([57.375, 57.120, 58.395]) / 255.0
                    std = np.ones(3)  # TODO: Check if this is correct
                elif format == "torchvision":
                    # https://pytorch.org/vision/0.8/models.html
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]
                else:
                    raise ValueError(f"Unknown format {format}")
                transformations.append(NormalizeImage(mean, std))
            self.transformation = Compose(transformations)

        self.resize = resize
        self.crop = crop
        self.normalize = normalize
        self.format = format

    def apply(self, data, key):
        image = data[key]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if self.format == "detectron2":
            image = np.asarray(image)
            image = image[:, :, ::-1]  # RGB to BGR
            image = Image.fromarray(image)
        data[key] = self.transformation(image)

        if "inputs.intrinsic" in data:
            if self.resize:
                data["inputs.intrinsic"] = adjust_intrinsic(data["inputs.intrinsic"], *image.size, size=self.resize)
                data["inputs.height"], data["inputs.width"] = data[key].shape[-2:]
            if self.crop:
                data["inputs.intrinsic"] = adjust_intrinsic(data["inputs.intrinsic"], *image.size, size=self.crop)
                data["inputs.height"], data["inputs.width"] = data[key].shape[-2:]

        return data


class ShadingImageFromNormals(Transform):
    @get_args()
    def __init__(
        self,
        light_dir: tuple[float, float, float] | Literal["random"] = (0.5, 0.5, 1.0),
        ambient: float | Literal["random"] = 0.2,
        diffuse: float | Literal["random"] = 0.7,
        specular: float | Literal["random"] = 0.1,
        shininess: int | Literal["random"] = 32,
        view_dir: tuple[float, float, float] | Literal["random"] = (0.0, 0.0, 1.0),
        multi_light: bool = False,
        use_hdri: bool = False,
        hdri_path: Path | None = None,
        remove_normals: bool = False,
        replace: bool = False,
        cache: bool = False,
    ):
        super().__init__(apply_to="inputs.normals", cachable=cache)

        self.light_dir = light_dir
        self.view_dir = view_dir

        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess

        self.multi_light = multi_light
        self.use_hdri = use_hdri
        self.hdri_path = hdri_path

        self.remove_normals = remove_normals
        self.replace = replace

        # Default light sources for multi-light setup
        self.light_sources = [
            {
                "direction": [0.5, 0.5, 1.0],
                "color": [1.0, 0.9, 0.8],  # Warm light
                "intensity": 0.7,
                "diffuse": 0.7,
                "specular": 0.3,
                "shininess": 32,
            },
            {
                "direction": [-0.5, -0.3, 0.5],
                "color": [0.8, 0.9, 1.0],  # Cool light (fill)
                "intensity": 0.4,
                "diffuse": 0.5,
                "specular": 0.1,
                "shininess": 16,
            },
            {
                "direction": [0.0, -1.0, 0.1],
                "color": [1.0, 1.0, 1.0],  # Rim light
                "intensity": 0.3,
                "diffuse": 0.6,
                "specular": 0.4,
                "shininess": 64,
            },
        ]

        # Load HDRI if specified
        self.hdri = None
        if self.use_hdri and self.hdri_path:
            try:
                from PIL import Image

                self.hdri = np.array(Image.open(self.hdri_path))
            except Exception as e:
                logger.warning(f"Failed to load HDRI from {self.hdri_path}: {e}")
                self.use_hdri = False

    def _normalize_vector(self, v):
        """Normalize vectors to unit length."""
        norm = np.linalg.norm(v, axis=2, keepdims=True)
        # Avoid division by zero
        norm[norm == 0] = 1
        return v / norm

    def _normal_map_to_rgb(self, normal_map):
        normals = normal_map
        if normal_map.dtype == np.uint8 or normal_map.max() > 1:
            normals = normal_map.astype(np.float32) / 127.5 - 1
        elif normal_map.min() >= 0:
            normals = normal_map * 2 - 1
        return self._normalize_vector(normals)

    def _resolved_single_light_params(self) -> tuple[np.ndarray, np.ndarray, float, float, float, int]:
        light_dir = self.light_dir
        if light_dir == "random":
            light_dir = np.random.uniform(-1, 1, 3)
            light_dir[2] = abs(light_dir[2])
        light_dir = np.array(light_dir) / np.linalg.norm(light_dir)

        view_dir = self.view_dir
        if view_dir == "random":
            view_dir = np.random.uniform(-1, 1, 3)
            view_dir[2] = abs(view_dir[2])
        view_dir = np.array(view_dir) / np.linalg.norm(view_dir)

        ambient = np.random.uniform(0.1, 0.3) if self.ambient == "random" else float(self.ambient)
        diffuse = np.random.uniform(0.5, 0.8) if self.diffuse == "random" else float(self.diffuse)
        specular = np.random.uniform(0.1, 0.5) if self.specular == "random" else float(self.specular)
        shininess = np.random.randint(8, 64) if self.shininess == "random" else int(self.shininess)
        return light_dir, view_dir, ambient, diffuse, specular, shininess

    def _generate_single_light_shading(self, normal_map, light_dir, ambient, diffuse, specular, shininess, view_dir):
        """Generate shading from a normal map with a single light source."""
        # Convert normal map to normalized RGB values
        normals = self._normal_map_to_rgb(normal_map)

        # Calculate diffuse component (Lambert's law)
        # dot product between normal and light direction
        diffuse_factor = np.maximum(0, np.sum(normals * light_dir, axis=2))

        # Calculate half vector between light and view
        half_vector = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)

        # Calculate specular component
        specular_factor = np.power(np.maximum(0, np.sum(normals * half_vector, axis=2)), shininess)

        # Combine components
        shading = ambient + diffuse * diffuse_factor + specular * specular_factor

        # Clip to 0-1 range
        shading = np.clip(shading, 0, 1)

        return shading

    def _generate_multi_light_shading(self, normal_map, view_dir):
        """Generate shading from multiple light sources."""
        # Convert normal map to normalized RGB values
        normals = self._normal_map_to_rgb(normal_map)

        # Start with ambient lighting
        h, w = normal_map.shape[:2]
        shading = np.ones((h, w, 3)) * 0.2

        # Add contribution from each light source
        for light in self.light_sources:
            # Extract light properties
            direction = np.array(light["direction"])
            direction = direction / np.linalg.norm(direction)

            intensity = light.get("intensity", 1.0)
            color = np.array(light.get("color", [1.0, 1.0, 1.0]))
            diffuse = light.get("diffuse", 0.7)
            specular = light.get("specular", 0.3)
            shininess = light.get("shininess", 32)

            # Calculate diffuse component
            diffuse_factor = np.maximum(0, np.sum(normals * direction, axis=2))
            diffuse_contribution = diffuse * diffuse_factor[:, :, np.newaxis] * color * intensity

            # Calculate half vector between light and view
            half_vector = (direction + view_dir) / np.linalg.norm(direction + view_dir)

            # Calculate specular component
            specular_factor = np.power(np.maximum(0, np.sum(normals * half_vector, axis=2)), shininess)
            specular_contribution = specular * specular_factor[:, :, np.newaxis] * color * intensity

            # Add this light's contribution
            shading += diffuse_contribution + specular_contribution

        # Tone mapping to keep values in 0-1 range
        shading = np.clip(shading, 0, 1)

        return shading

    def _generate_hdri_shading(self, normal_map):
        """Generate shading using an HDRI environment map."""
        if self.hdri is None:
            logger.warning("HDRI is None, falling back to single light shading")
            light_dir, view_dir, ambient, diffuse, specular, shininess = self._resolved_single_light_params()
            return self._generate_single_light_shading(
                normal_map,
                light_dir=light_dir,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                shininess=shininess,
                view_dir=view_dir,
            )

        # Convert normal map to normalized RGB values
        normals = self._normal_map_to_rgb(normal_map)

        # Get dimensions
        h_hdri, w_hdri = self.hdri.shape[:2]
        h_norm, w_norm = normals.shape[:2]

        # Convert normals to spherical coordinates
        phi = np.arctan2(normals[:, :, 1], normals[:, :, 0])
        theta = np.arccos(np.clip(normals[:, :, 2], -1, 1))

        # Map spherical coordinates to HDRI coordinates
        u = ((phi / (2 * np.pi)) + 0.5) % 1.0  # Map [-pi, pi] to [0, 1]
        v = theta / np.pi  # Map [0, pi] to [0, 1]

        # Convert to pixel coordinates in the HDRI
        hdri_x = (u * w_hdri).astype(int) % w_hdri
        hdri_y = (v * h_hdri).astype(int) % h_hdri

        # Sample the HDRI - basic point sampling
        hdri_values = self.hdri[hdri_y, hdri_x] / 255.0

        # Start with ambient lighting
        ambient = np.random.uniform(0.1, 0.3) if self.ambient == "random" else float(self.ambient)
        diffuse = np.random.uniform(0.5, 0.8) if self.diffuse == "random" else float(self.diffuse)
        shading = np.ones((h_norm, w_norm, 3)) * ambient

        # Add diffuse contribution from HDRI
        shading += hdri_values * diffuse

        # Simple tone mapping
        shading = np.clip(shading, 0, 1)

        return shading

    def apply(self, data, key):
        normal_map = data[key]

        if normal_map.ndim == 2 and normal_map.shape[1] == 3:
            points = data["inputs"]
            intrinsic = data["inputs.intrinsic"]
            extrinsic = data.get("inputs.extrinsic", np.eye(4))
            width = data.get("inputs.width", int(intrinsic[0, 2] * 2))
            height = data.get("inputs.height", int(intrinsic[1, 2] * 2))

            points_cam = apply_trafo(points, extrinsic)
            u, v, mask = points_to_uv(points_cam, intrinsic, width=width, height=height)

            normal_image = np.zeros((height, width, 3), dtype=np.float32)
            normal_image[v, u] = normal_map[mask] / 2 + 0.5
            normal_map = normal_image

        view_dir = self.view_dir
        if view_dir == "random":
            view_dir = np.random.uniform(-1, 1, 3)
            view_dir[2] = abs(view_dir[2])
        view_dir = np.array(view_dir) / np.linalg.norm(view_dir)

        if self.use_hdri:
            shaded = self._generate_hdri_shading(normal_map)
        elif self.multi_light:
            shaded = self._generate_multi_light_shading(normal_map, view_dir)
        else:
            light_dir = self.light_dir
            if light_dir == "random":
                light_dir = np.random.uniform(-1, 1, 3)
                light_dir[2] = abs(light_dir[2])
            light_dir = np.array(light_dir) / np.linalg.norm(light_dir)

            ambient = np.random.uniform(0.1, 0.3) if self.ambient == "random" else self.ambient
            diffuse = np.random.uniform(0.5, 0.8) if self.diffuse == "random" else self.diffuse
            specular = np.random.uniform(0.1, 0.5) if self.specular == "random" else self.specular
            shininess = np.random.randint(8, 64) if self.shininess == "random" else self.shininess
            logger.debug(f"Ambient: {ambient}, Diffuse: {diffuse}, Specular: {specular}, Shininess: {shininess}")

            shaded = self._generate_single_light_shading(
                normal_map, light_dir, ambient, diffuse, specular, shininess, view_dir
            )

        shaded[normal_map.sum(2) == 0] = 0
        image = Image.fromarray((shaded * 255).astype(np.uint8))
        if self.remove_normals:
            data.pop(key)
        if self.replace:
            data["inputs"] = image
        else:
            data["inputs.image"] = image

        return data


class Rotate(Transform):
    @get_args()
    def __init__(
        self,
        axes: str | None = None,
        angles: tuple[float, ...] | None = None,
        matrix: np.ndarray | None = None,
        upper_hemisphere: bool = True,
        reverse: bool = False,
        angle_from_index: bool = False,
        choose_random: bool = False,
        from_inputs: bool = False,
    ):
        super().__init__()

        self.axes = axes
        self.angles = angles
        self.matrix = matrix
        self.upper_hemisphere = upper_hemisphere
        self.reverse = reverse
        self.angle_from_index = angle_from_index
        self.choose_random = choose_random
        self.from_inputs = from_inputs

    def apply(self, data, key):
        points = data.get("points")

        inputs = data.get("inputs")
        normals = data.get("inputs.normals")
        depth = data.get("inputs.depth")

        pcd = data.get("pointcloud")
        pcd_normals = data.get("pointcloud.normals")

        mesh_vertices = data.get("mesh.vertices")
        mesh_normals = data.get("mesh.normals")

        bbox = data.get("bbox")

        partnet = data.get("partnet.points")

        if self.reverse:
            rot = data.get("inputs.frame")
            rot = np.eye(3) if rot is None else rot.T
        elif self.from_inputs:
            rot = data.get("inputs.rotation", np.eye(3))
            if self.axes == "x":
                extrinsic = data.get("inputs.extrinsic")
                if extrinsic is not None:
                    x = data.get("inputs.pitch")
                    if x is None:
                        # x, y, z = R.from_matrix(extrinsic[:3, :3]).as_euler("XYZ", degrees=True)
                        _z, _y, x = R.from_matrix(extrinsic[:3, :3]).as_euler("zyx", degrees=True)
                        # x, y, z = R.from_matrix(extrinsic[:3, :3].T).as_euler("xyz", degrees=True)
                        # z, y, x = R.from_matrix(extrinsic[:3, :3].T).as_euler("ZYX", degrees=True)
                        x = x - 180 if x > 90 else x + 180 if x < -90 else x
                        # z = z if abs(z) < 90 else 180 - abs(z)
                        # rot_z = R.from_euler("z", z, degrees=True).as_matrix().T
                        # rot_y = R.from_euler("y", y, degrees=True).as_matrix().T
                    rot_x = R.from_euler("x", x + 180, degrees=True).as_matrix().T
                    rot = rot_x @ rot
                    data["inputs.extrinsic"] = np.eye(4)
        else:
            if self.axes is not None:
                if self.angle_from_index:
                    if len(self.axes) > 1:
                        raise NotImplementedError("Angle from index not implemented for multi-axis rotation.")
                    self.angles = [(5 * data["index"]) % 360]
                if self.angles:
                    if self.choose_random:
                        axis = np.random.choice(list(self.axes))
                        angle = np.random.choice(self.angles)
                        rot = R.from_euler(axis, angle, degrees=True).as_matrix()
                    else:
                        rot = R.from_euler(
                            self.axes, self.angles[0] if len(self.angles) == 1 else self.angles, degrees=True
                        ).as_matrix()
                else:
                    rot, _pitch = rot_from_euler(self.axes, self.upper_hemisphere)
            elif isinstance(self.matrix, np.ndarray):
                rot = self.matrix
            else:
                rot = R.random().as_matrix()
            data["inputs.frame"] = rot

        for key, value in zip(
            [
                "points",
                "pointcloud",
                "pointcloud.normals",
                "inputs",
                "inputs.normals",
                "inputs.depth",
                "mesh.vertices",
                "mesh.normals",
                "bbox",
                "partnet.points",
            ],
            [
                points,
                pcd,
                pcd_normals,
                inputs,
                normals,
                depth,
                mesh_vertices,
                mesh_normals,
                bbox,
                partnet,
            ],
            strict=False,
        ):
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 3:
                value_rot = value @ rot.T
                if "normals" in key:
                    norm = np.linalg.norm(value_rot, axis=1, keepdims=True)
                    norm[norm == 0] = 1
                    value_rot = value_rot / norm
                data[key] = value_rot.astype(value.dtype)
            elif isinstance(value, np.ndarray) and "normals" in key:
                normals = value
                if value.min() > 1:
                    v, u = np.nonzero(value.sum(axis=2))
                    normals = (value[v, u] / 255) * 2 - 1
                normals = normals @ rot.T
                norm = np.linalg.norm(normals, axis=1, keepdims=True)
                norm[norm == 0] = 1
                normals = normals / norm
                if value.min() > 1:
                    value[v, u] = ((normals / 2 + 0.5) * 255).astype(np.uint8)
                else:
                    value = normals
                data[key] = value

        extrinsic = data.get("inputs.extrinsic")
        if extrinsic is not None:
            # When from_inputs with no axes, we're reversing DepthField's unrotate
            # which already applied @ rot.T to the extrinsic — so apply @ rot to cancel.
            # For fresh rotations (augmentation etc.), @ rot.T compensates for points @ rot.T.
            rot_ext = rot if (self.from_inputs and self.axes is None) else rot.T
            data["inputs.extrinsic"][:3, :3] = extrinsic[:3, :3] @ rot_ext

        if self.from_inputs:
            data.pop("inputs.rotation", None)
            if self.axes == "x":
                data.pop("inputs.pitch", None)

        return data


class Affine(Transform):
    @get_args()
    def __init__(
        self,
        apply_to=(
            "inputs",
            "inputs.depth",
            "inputs.normals",
            "points",
            "pointcloud",
            "pointcloud.normals",
            "mesh.vertices",
            "mesh.normals",
            "bbox",
            "partnet.points",
        ),
        trafo_key: str = "inputs.extrinsic",
        trafo: np.ndarray | None = None,
        remove_roll: bool = False,
        replace: bool = False,
    ):
        super().__init__(apply_to=apply_to)
        self.trafo = trafo
        self.trafo_key = trafo_key
        self.replace = replace
        self.remove_roll = remove_roll
        self._roll = None

    def before_apply(self, data):
        trafo = data.get(self.trafo_key, self.trafo)
        if trafo is None:
            return data

        if self.remove_roll:
            angles = R.from_matrix(trafo[:3, :3]).as_euler("zyx")
            new_trafo = trafo.copy()
            roll = R.from_euler("z", angles[1]).as_matrix()  # TODO: OpenGL camera convetion w/ Y up?
            trafo_z = np.eye(4)
            trafo_z[:3, :3] = roll.T
            new_trafo = trafo_z @ new_trafo
            data[self.trafo_key] = new_trafo
            self._roll = roll

        return data

    def apply(self, data, key):
        trafo = data.get(self.trafo_key, self.trafo)
        if trafo is None:
            return data

        value = data.get(key)

        if isinstance(value, np.ndarray) and (
            (value.ndim == 2 and value.shape[1] == 3) or (value.ndim == 3 and value.shape[2] == 3)
        ):
            if key is not None and "normals" in key:
                value_t = value @ trafo[:3, :3].T
                norm = np.linalg.norm(value_t, axis=1, keepdims=True)
                norm[norm == 0] = 1
                value_t = value_t / norm
            else:
                value_t = apply_trafo(value, trafo)
            if torch.is_tensor(value_t):
                value_t = value_t.detach().cpu().numpy()
            data[key] = value_t.astype(value.dtype)
        elif isinstance(value, np.ndarray) and key is not None and "normals" in key:
            normals = value
            if value.min() > 1:
                v, u = np.nonzero(value.sum(axis=2))
                normals = (value[v, u] / 255) * 2 - 1
            normals = normals @ trafo[:3, :3].T
            norm = np.linalg.norm(normals, axis=1, keepdims=True)
            norm[norm == 0] = 1
            normals = normals / norm
            if value.min() > 1:
                value[v, u] = ((normals / 2 + 0.5) * 255).astype(np.uint8)
            else:
                value = normals
            data[key] = value

        return data

    def after_apply(self, data):
        trafo = data.get(self.trafo_key, self.trafo)
        data[self.trafo_key.split(".")[0] + ".inv_extrinsic"] = inv_trafo(trafo)
        if self.replace:
            data[self.trafo_key] = np.eye(4)
            if self._roll is not None:
                data[self.trafo_key][:3, :3] = self._roll
                self._roll = None
            data[self.trafo_key.split(".")[0] + ".frame"] = trafo[:3, :3]
        return data


class AddGaussianNoise(Transform):
    @get_args()
    def __init__(
        self,
        apply_to="inputs",
        stddev: float | tuple[float, float] = 0.005,
        clip: float | None = None,
        return_noise: bool = False,
    ):
        super().__init__(apply_to=apply_to)

        self.stddev = stddev
        self.clip = clip
        self.return_noise = return_noise

    def apply(self, data, key):
        value = data[key]

        noise = np.random.randn(*value.shape)
        if isinstance(self.stddev, tuple):
            stddev_low, stddev_high = self.stddev
            stddev = np.random.uniform(stddev_low, stddev_high)
        else:
            stddev = float(self.stddev)
        noise *= stddev

        if self.clip is not None:
            noise = np.clip(noise, -self.clip, self.clip)
        scale = (value.max(axis=0) - value.min(axis=0)).max()
        value += noise * scale

        data[key] = value
        if self.return_noise:
            data["noise" if key is None else f"{key}.noise"] = noise

        return data


class SubsamplePointcloud(Transform):
    @get_args()
    def __init__(
        self, apply_to, num_samples: int | tuple[float, float] = 3000, fps: bool = False, cachable: bool = False
    ):
        super().__init__(apply_to=apply_to, cachable=cachable)

        self.num_samples = num_samples
        self.fps = fps
        self.warn_num_samples = 0

        if fps:
            logger.warning("CPU FPS is slow. Consider using GPU FPS from libs.")

    def apply(self, data, key):
        num_samples = self.num_samples

        if not num_samples:
            return data

        value = data[key]
        if key is None:
            normals = data.get("normals")
            colors = data.get("colors")
            labels = data.get("labels")
        else:
            normals = data.get(f"{key}.normals")
            colors = data.get(f"{key}.colors")
            labels = data.get(f"{key}.labels")

        if isinstance(labels, np.ndarray) and len(labels) != len(value):
            labels = None

        if not isinstance(num_samples, (int, float)):
            num_samples = int(np.random.uniform(*num_samples) * len(value))

        if num_samples == len(value) or value.ndim != 2 or value.shape[1] != 3:
            return data

        if len(value) < num_samples:
            self.warn_num_samples += 1
            if self.warn_num_samples == 10:
                logger.warning(f"{self.name}: Not enough points to sample from: {len(value)} < {num_samples}")

        if self.fps:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(value))
            if normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(normals)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            elif isinstance(labels, np.ndarray):
                colors = o3d.utility.Vector3dVector(labels / 255)
            pcd = pcd.farthest_point_down_sample(num_samples)

            value = np.asarray(pcd.points)
            if normals is not None:
                normals = np.asarray(pcd.normals)
            if colors is not None:
                colors = np.asarray(pcd.colors)
            if isinstance(labels, np.ndarray):
                colors = np.round(255 * np.asarray(pcd.colors)).astype(np.int64)
        else:
            indices = subsample_indices(value, num_samples)
            value = value[indices]
            if normals is not None:
                normals = normals[indices]
            if colors is not None:
                colors = colors[indices]
            if isinstance(labels, np.ndarray):
                labels = labels[indices]

        data[key] = value
        if key is None:
            if normals is not None:
                data["normals"] = normals
            if colors is not None:
                data["colors"] = colors
            if isinstance(labels, np.ndarray):
                data["labels"] = labels
        else:
            if normals is not None:
                data[f"{key}.normals"] = normals
            if colors is not None:
                data[f"{key}.colors"] = colors
            if isinstance(labels, np.ndarray):
                data[f"{key}.labels"] = labels

        return data


class ProcessPointcloud(Transform):
    @get_args()
    def __init__(
        self,
        apply_to,
        downsample: DownsampleTypes | None = None,
        downsample_factor: float | int = 1,
        remove_outlier: OutlierTypes | None = None,
        outlier_std_ratio: float = 1.0,
        estimate_normals: bool = False,
        search_param: SearchParamTypes = SearchParamTypes.HYBRID,
        search_param_knn: int = 30,
        search_param_radius: float = 0.02,
    ):
        super().__init__(apply_to=apply_to, cachable=True)

        self.downsample = downsample
        self.downsample_factor = downsample_factor
        self.remove_outlier = remove_outlier
        self.outlier_std_ratio = outlier_std_ratio
        self.estimate_normals = estimate_normals
        self.search_param = search_param
        self.search_param_knn = search_param_knn
        self.search_param_radius = search_param_radius

    def apply(self, data, key):
        points = data[key]
        if key is None:
            normals = data.get("normals")
            colors = data.get("colors")
        else:
            normals = data.get(f"{key}.normals")
            colors = data.get(f"{key}.colors")

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_processed = process_point_cloud(
            pcd,
            downsample=self.downsample,
            downsample_factor=self.downsample_factor,
            remove_outlier=self.remove_outlier,
            outlier_std_ratio=self.outlier_std_ratio,
            estimate_normals=self.estimate_normals,
            search_param=self.search_param,
            search_param_knn=self.search_param_knn,
            search_param_radius=self.search_param_radius,
        )
        pcd = pcd_processed[0] if isinstance(pcd_processed, tuple) else pcd_processed
        pcd = cast(Any, pcd)

        data[key] = np.asarray(pcd.points)
        if key is None:
            if pcd.has_normals():
                data["normals"] = np.asarray(pcd.normals)
            if pcd.has_colors():
                data["colors"] = np.asarray(pcd.colors)
        else:
            if pcd.has_normals():
                data[f"{key}.normals"] = np.asarray(pcd.normals)
            if pcd.has_colors():
                data[f"{key}.colors"] = np.asarray(pcd.colors)

        return data


class RotatePointcloud(Transform):
    @get_args()
    def __init__(self, axes: str | None = None, angles: tuple[float, ...] | None = None):
        super().__init__(apply_to="pointcloud")

        self.axes = axes
        self.angles = angles

    def apply(self, data, key):
        points = data[key]
        normals = data.get(f"{key}.normals")

        axes = cast(str, self.axes)
        angles = cast(tuple[float, ...], self.angles)
        rot = R.from_euler(axes, angles[0] if len(angles) == 1 else angles, degrees=True).as_matrix()

        data[key] = (rot @ points.T).T
        if normals is not None:
            data[f"{key}.normals"] = (rot @ normals.T).T
        return data


class RotateMesh(Transform):
    @get_args()
    def __init__(self, axes: str | None = None, angles: tuple[float, ...] | None = None):
        super().__init__(apply_to="mesh.vertices", cachable=True)

        self.axes = axes
        self.angles = angles

    def apply(self, data, key):
        vertices = data[key]
        normals = data.get("mesh.normals")

        axes = cast(str, self.axes)
        angles = cast(tuple[float, ...], self.angles)
        rot = R.from_euler(axes, angles[0] if len(angles) == 1 else angles, degrees=True).as_matrix()

        data[key] = (rot @ vertices.T).T
        if normals is not None:
            data["mesh.normals"] = (rot @ normals.T).T
        return data


class NormalizeMesh(Transform):
    def __init__(self):
        super().__init__(apply_to="mesh.vertices", cachable=True)

    def apply(self, data, key):
        vertices = data[key]
        triangles = data["mesh.triangles"]

        referenced = np.zeros(len(vertices), dtype=bool)
        referenced[triangles] = True
        in_mesh = vertices[referenced]
        bounds = np.array([in_mesh.min(axis=0), in_mesh.max(axis=0)])

        translation = -bounds.mean(axis=0)

        extents = np.ptp(bounds, axis=0)
        max_extents = extents.max()
        scale = 1 / max_extents

        vertices += translation
        vertices *= scale

        data[key] = vertices
        return data


class ApplyPose(Transform):
    @get_args()
    def __init__(
        self,
        apply_to=("inputs", "points", "pointcloud", "mesh.vertices", "bbox", "partnet.points"),
        pose: np.ndarray | None = None,
    ):
        super().__init__(apply_to=apply_to)

        self.pose = pose

    def apply(self, data, key):
        value = data.get(key)
        if isinstance(value, np.ndarray):
            pose = data.get("inputs.pose", self.pose)

            rot = pose[:3, :3]
            trans = pose[:3, 3]

            data["pose"] = pose
            data[key] = value @ rot.T
            if key is None or "normals" not in key:
                data[key] += trans

        return data


class InputsNormalsFromPointcloud(Transform):
    @get_args()
    def __init__(self, remove_pointcloud: bool = False, cachable: bool = False):
        super().__init__(cachable=cachable)

        self.remove_pointcloud = remove_pointcloud

    def apply(self, data, key):
        pointcloud = data["pointcloud"]
        normals = data["pointcloud.normals"]
        inputs = data["inputs"]

        kdtree = KDTree(pointcloud)
        _, index = kdtree.query(inputs.astype(pointcloud.dtype), k=1, eps=0)
        inputs_normals = normals[index]

        data["inputs.normals"] = inputs_normals
        if self.remove_pointcloud:
            data.pop("pointcloud")
            data.pop("pointcloud.normals")
        return data


class CropPointcloud(Transform):
    @get_args()
    def __init__(
        self,
        apply_to,
        mode: Literal["cube", "sphere", "frustum"] = "cube",
        padding: float = 0.1,
        scale_factor: float = 1.0,
        cache: bool = False,
    ):
        super().__init__(apply_to=apply_to, cachable=cache)

        self.mode = mode
        self.padding = padding
        self.scale_factor = scale_factor

    def apply(self, data, key):
        points = data[key]
        colors = data.get(f"{key}.colors")
        normals = data.get(f"{key}.normals")
        labels = data.get(f"{key}.labels")

        bound = (0.5 + self.padding / 2.0) * self.scale_factor

        if self.mode == "cube":
            mask = np.all(np.abs(points) <= bound, axis=1)
        elif self.mode == "sphere":
            mask = np.einsum("ij,ij->i", points, points) <= (bound * bound)
        elif self.mode == "frustum":
            is_in_frustum_fn = cast(Any, is_in_frustum)
            mask = is_in_frustum_fn(
                points=points,
                intrinsic=data["inputs.intrinsic"],
                extrinsic=data["inputs.extrinsic"],
                width=data["inputs.width"],
                height=data["inputs.height"],
                near=data.get("inputs.near", 0.2),
                far=data.get("inputs.far", 2.4),
            )
        else:
            raise ValueError(f"Unknown CropPoints mode: {self.mode}")

        if mask.all():
            return data

        data[key] = points[mask]
        if colors is not None:
            data[f"{key}.colors"] = colors[mask]
        if normals is not None:
            data[f"{key}.normals"] = normals[mask]
        if isinstance(labels, np.ndarray) and len(labels) == len(points):
            data[f"{key}.labels"] = labels[mask]
        return data


class CropPointcloudWithMesh(Transform):
    @get_args()
    def __init__(
        self,
        apply_to,
        crop_type: str = "box",
        scale: float = 1.2,
        mesh_dirname: str = "google_16k",
        mesh_filename: str = "nontextured.ply",
        pose_filename: str = "pose.npy",
    ):
        super().__init__(apply_to=apply_to)

        assert crop_type.lower() in ["box", "hull"]

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

    def apply(self, data, key):
        obj_name = data[f"{key}.name"]
        points = data[key]
        normals = data.get(f"{key}.normals")
        colors = data.get(f"{key}.colors")

        if obj_name not in self.crop_shapes:
            mesh_dir = os.path.join(data[f"{key}.path"], self.mesh_dirname)
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

        data[key] = points
        if normals is not None:
            data[f"{key}.normals"] = normals
        if colors is not None:
            data[f"{key}.colors"] = colors
        return data


class RefinePose(Transform):
    @get_args()
    def __init__(
        self,
        pose_key: str = "inputs.extrinsic",
        src_key: str = "mesh.vertices",
        tgt_key: str = "inputs",
        max_correspondence_distance: float = 0.0125,
        max_iterations: int = 100,
        voxel_size: float | None = 0.005,
        point_to_plane: bool = False,
        projective_icp: bool = True,
        remove_outlier: bool = False,
        z_near: float = 0.01,
        z_far: float = 10.0,
        crop_target_around_source: bool = True,
        crop_scale: float = 1.4,
        robust_kernel: KernelTypes | None = KernelTypes.TUKEY,
        kernel_noise_std: float | None = 0.01,
        max_rot_deg: float = 30.0,
        max_trans: float = 0.1,
        show: bool = False,
    ):
        super().__init__()
        self.pose_key = pose_key
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.max_corr = float(max_correspondence_distance)
        self.max_iter = int(max_iterations)
        self.voxel = voxel_size
        self.ptp = not point_to_plane
        self.projective_icp = projective_icp
        self.remove_outlier = remove_outlier
        self.z_near = float(z_near)
        self.z_far = float(z_far)
        self.crop_target_around_source = crop_target_around_source
        self.crop_scale = crop_scale
        self.robust_kernel = robust_kernel
        self.kernel_noise_std = kernel_noise_std
        self.max_rot_deg = max_rot_deg
        self.max_trans = max_trans
        self.show = show

        if projective_icp and not show:
            os.environ["PYOPENGL_PLATFORM"] = "egl"

    def apply(self, data: dict[str | None, Any], key: str | None):
        src = data[self.src_key]
        tgt = data[self.tgt_key]
        intrinsic = data["inputs.intrinsic"]

        _MIN_ICP_POINTS = 10

        @default_on_exception(default=(np.eye(4), None))
        def _run_icp(src_pts: np.ndarray, tgt_pts: np.ndarray) -> tuple[np.ndarray, MyRegistrationResult | None]:
            if len(src_pts) < _MIN_ICP_POINTS or len(tgt_pts) < _MIN_ICP_POINTS:
                _log_debug_level_1(f"_run_icp: skipping, too few input points (src={len(src_pts)}, tgt={len(tgt_pts)})")
                return np.eye(4), None

            downsample = DownsampleTypes.VOXEL if self.voxel else None
            search_radius = (self.voxel or 0.005) * 3.0
            kernel_noise_std = (self.voxel or 0.005) * 2.0 if self.kernel_noise_std is None else self.kernel_noise_std
            want_normals = not self.ptp
            orient = OrientationTypes.TANGENT_PLANE if want_normals else None
            src_pcd = _as_point_cloud(
                process_point_cloud(
                    get_point_cloud_from_points(src_pts),
                    downsample=downsample,
                    downsample_factor=self.voxel or 1.0,
                    estimate_normals=want_normals,
                    orient_normals=orient,
                    search_param=SearchParamTypes.HYBRID,
                    search_param_knn=30,
                    search_param_radius=search_radius,
                )
            )
            tgt_pcd = _as_point_cloud(
                process_point_cloud(
                    get_point_cloud_from_points(tgt_pts),
                    downsample=downsample,
                    downsample_factor=self.voxel or 1.0,
                    remove_outlier=OutlierTypes.STATISTICAL if self.remove_outlier else None,
                    outlier_std_ratio=1.0,
                    estimate_normals=want_normals,
                    orient_normals=orient,
                    search_param=SearchParamTypes.HYBRID,
                    search_param_knn=30,
                    search_param_radius=search_radius,
                )
            )

            n_src = len(np.asarray(src_pcd.points))
            n_tgt = len(np.asarray(tgt_pcd.points))
            if n_src < _MIN_ICP_POINTS or n_tgt < _MIN_ICP_POINTS:
                _log_debug_level_1(f"_run_icp: skipping, too few points after processing (src={n_src}, tgt={n_tgt})")
                return np.eye(4), None

            has_normals = src_pcd.has_normals() and tgt_pcd.has_normals()
            use_plane = want_normals and has_normals
            icp = IterativeClosestPoint(
                max_correspondence_distance=self.max_corr,
                max_iteration=self.max_iter,
                estimation_method=ICPTypes.PLANE if use_plane else ICPTypes.POINT,
                kernel=self.robust_kernel if use_plane else None,
                kernel_noise_std=kernel_noise_std,
            )
            result = icp.run(
                src_pcd,
                tgt_pcd,
                crop_target_around_source=self.crop_target_around_source,
                crop_scale=self.crop_scale,
                draw=self.show,
            )
            trafo = np.asarray(result.transformation, dtype=np.float32)
            return trafo, result

        if self.projective_icp:
            V = data["mesh.vertices"]
            F = data["mesh.triangles"]
            pose = data[self.pose_key]
            H = data["inputs.height"]
            W = data["inputs.width"]
            scene = pyrender.Scene()
            mesh = pyrender.Mesh.from_trimesh(Trimesh(V, F, process=False), smooth=False)
            scene.add(mesh)
            cam = pyrender.IntrinsicsCamera(
                float(intrinsic[0, 0]),
                float(intrinsic[1, 1]),
                float(intrinsic[0, 2]),
                float(intrinsic[1, 2]),
                znear=self.z_near,
                zfar=self.z_far,
            )
            scene.add(cam, pose=inv_trafo(convert_extrinsic(pose, "opencv", "opengl")))
            renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
            flags = pyrender.RenderFlags.DEPTH_ONLY | pyrender.RenderFlags.OFFSCREEN
            depth_syn = renderer.render(scene, flags=flags)
            renderer.delete()
            depth_syn = np.asarray(depth_syn)
            depth_syn[(depth_syn >= self.z_far) | (depth_syn <= self.z_near)] = 0
            pcd_syn = convert_depth_image_to_point_cloud(depth_syn.astype(np.float32), intrinsic, pose, depth_scale=1.0)
            syn_pts = np.asarray(pcd_syn.points)
            trafo, reg = _run_icp(syn_pts, tgt)
        else:
            trafo, reg = _run_icp(src, tgt)

        if self.show and reg is not None:
            logger.info(f"{self.name}: fitness={reg.fitness:.4f}, inlier_rmse={reg.inlier_rmse:.4f}")

        rot = trafo[:3, :3]
        rvec = R.from_matrix(rot).as_rotvec()
        angle = float(np.linalg.norm(rvec)) if rvec is not None else 0.0
        if angle > np.deg2rad(self.max_rot_deg):
            trafo = np.eye(4)

        tnorm = float(np.linalg.norm(trafo[:3, 3]))
        if tnorm > self.max_trans:
            trafo = np.eye(4)

        if "mesh.vertices" in data and isinstance(data["mesh.vertices"], np.ndarray):
            data["mesh.vertices"] = apply_trafo(data["mesh.vertices"], trafo)
            if "mesh.normals" in data and isinstance(data["mesh.normals"], np.ndarray):
                data["mesh.normals"] = (rot @ data["mesh.normals"].T).T
        for k in ("mesh.vertices_world", "mesh.vertices_cam", "mesh.vertices_table"):
            if k in data and isinstance(data[k], np.ndarray):
                data[k] = apply_trafo(data[k], trafo)
        if (
            "points" in data
            and isinstance(data["points"], np.ndarray)
            and data["points"].ndim == 2
            and data["points"].shape[1] == 3
        ):
            data["points"] = apply_trafo(data["points"], trafo)
        if (
            "pointcloud" in data
            and isinstance(data["pointcloud"], np.ndarray)
            and data["pointcloud"].ndim == 2
            and data["pointcloud"].shape[1] == 3
        ):
            data["pointcloud"] = apply_trafo(data["pointcloud"], trafo)
            if "pointcloud.normals" in data and isinstance(data["pointcloud.normals"], np.ndarray):
                data["pointcloud.normals"] = (rot @ data["pointcloud.normals"].T).T

        return data


class RefinePosePerInstance(Transform):
    """
    - Kinect-like: voxel_size=0.005, max_corr≈0.0125, kernel_noise_std≈0.01
    - D435: voxel_size=0.003, max_corr≈0.0075, kernel_noise_std≈0.006
    """

    @get_args()
    def __init__(
        self,
        tgt_key: str = "inputs",
        max_correspondence_distance: float = 0.0125,
        max_iterations: int = 25,
        voxel_size: float | None = 0.005,
        point_to_plane: bool = False,
        remove_outlier: bool = False,
        z_near: float = 0.01,
        z_far: float = 10.0,
        crop_target_around_source: bool = True,
        crop_scale: float = 1.4,
        robust_kernel: KernelTypes | None = KernelTypes.TUKEY,
        kernel_noise_std: float | None = 0.01,
        max_rot_deg: float = 30.0,
        max_trans: float = 0.1,
        show: bool = False,
    ):
        super().__init__()
        self.tgt_key = tgt_key
        self.max_corr = float(max_correspondence_distance)
        self.max_iter = int(max_iterations)
        self.voxel = voxel_size
        self.ptp = not point_to_plane
        self.remove_outlier = remove_outlier
        self.z_near = float(z_near)
        self.z_far = float(z_far)
        self.crop_target_around_source = crop_target_around_source
        self.crop_scale = crop_scale
        self.robust_kernel = robust_kernel
        self.kernel_noise_std = kernel_noise_std
        self.max_rot_deg = max_rot_deg
        self.max_trans = max_trans
        self.show = show

        if not show:
            os.environ["PYOPENGL_PLATFORM"] = "egl"

    def apply(self, data: dict[str | None, Any], key: str | None):
        V_all = data["mesh.vertices"]
        F_all = data["mesh.triangles"]
        v_lens: list[int] = data["mesh.num_vertices"]
        f_lens: list[int] = data["mesh.num_triangles"]
        intrinsic = data["inputs.intrinsic"]
        extrinsic = data["inputs.extrinsic"]
        H = int(data["inputs.height"])
        W = int(data["inputs.width"])

        tgt = data[self.tgt_key]
        want_normals = not self.ptp
        tgt_pcd = _as_point_cloud(
            process_point_cloud(
                get_point_cloud_from_points(tgt),
                downsample=DownsampleTypes.VOXEL if self.voxel else None,
                downsample_factor=self.voxel or 1.0,
                remove_outlier=OutlierTypes.STATISTICAL if self.remove_outlier else None,
                outlier_std_ratio=1.0,
                estimate_normals=want_normals,
                orient_normals=OrientationTypes.TANGENT_PLANE if want_normals else None,
                search_param=SearchParamTypes.HYBRID,
                search_param_knn=30,
                search_param_radius=(self.voxel or 0.005) * 3.0,
            )
        )

        v_offsets = np.cumsum([0, *v_lens[:-1]])
        f_offsets = np.cumsum([0, *f_lens[:-1]])

        _MIN_ICP_POINTS = 10

        @default_on_exception(default=(np.eye(4), None))
        def _run_icp(
            src_pts: np.ndarray, tgt_pcd: o3d.geometry.PointCloud
        ) -> tuple[np.ndarray, MyRegistrationResult | None]:
            if len(src_pts) < _MIN_ICP_POINTS:
                _log_debug_level_1(f"_run_icp: skipping, too few source points ({len(src_pts)})")
                return np.eye(4), None

            downsample = DownsampleTypes.VOXEL if self.voxel else None
            search_radius = (self.voxel or 0.005) * 3.0
            kernel_noise_std = (self.voxel or 0.005) * 2.0 if self.kernel_noise_std is None else self.kernel_noise_std
            want_normals = not self.ptp
            src_pcd = _as_point_cloud(
                process_point_cloud(
                    get_point_cloud_from_points(src_pts),
                    downsample=downsample,
                    downsample_factor=self.voxel or 1.0,
                    estimate_normals=want_normals,
                    orient_normals=OrientationTypes.TANGENT_PLANE if want_normals else None,
                    search_param=SearchParamTypes.HYBRID,
                    search_param_knn=30,
                    search_param_radius=search_radius,
                )
            )

            n_src = len(np.asarray(src_pcd.points))
            if n_src < _MIN_ICP_POINTS:
                _log_debug_level_1(f"_run_icp: skipping, too few source points after processing ({n_src})")
                return np.eye(4), None

            has_normals = src_pcd.has_normals() and tgt_pcd.has_normals()
            use_plane = want_normals and has_normals
            icp = IterativeClosestPoint(
                max_correspondence_distance=self.max_corr,
                max_iteration=self.max_iter,
                estimation_method=ICPTypes.PLANE if use_plane else ICPTypes.POINT,
                kernel=self.robust_kernel if use_plane else None,
                kernel_noise_std=kernel_noise_std,
            )
            result = icp.run(
                src_pcd,
                tgt_pcd,
                crop_target_around_source=self.crop_target_around_source,
                crop_scale=self.crop_scale,
                draw=self.show,
            )
            trafo = np.asarray(result.transformation, dtype=np.float32)
            return trafo, result

        p_lengths = data.get("points.lengths")
        pc_lengths = data.get("pointcloud.lengths")
        p_offsets = np.cumsum([0] + (p_lengths[:-1] if p_lengths else [])).tolist() if p_lengths else None
        pc_offsets = np.cumsum([0] + (pc_lengths[:-1] if pc_lengths else [])).tolist() if pc_lengths else None

        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        cam = pyrender.IntrinsicsCamera(
            float(intrinsic[0, 0]),
            float(intrinsic[1, 1]),
            float(intrinsic[0, 2]),
            float(intrinsic[1, 2]),
            znear=self.z_near,
            zfar=self.z_far,
        )

        for i, (vl, fl, vo, fo) in enumerate(zip(v_lens, f_lens, v_offsets, f_offsets, strict=False)):
            V = V_all[vo : vo + vl].copy()
            F = F_all[fo : fo + fl].copy()
            F_local = F - vo

            scene = pyrender.Scene()
            mesh = pyrender.Mesh.from_trimesh(Trimesh(V, F_local, process=False), smooth=False)
            scene.add(mesh)
            scene.add(cam, pose=inv_trafo(convert_extrinsic(extrinsic, "opencv", "opengl")))
            flags = pyrender.RenderFlags.DEPTH_ONLY | pyrender.RenderFlags.OFFSCREEN
            depth_syn = renderer.render(scene, flags=flags)
            depth_syn = np.asarray(depth_syn)
            depth_syn[(depth_syn >= self.z_far) | (depth_syn <= self.z_near)] = 0
            syn_pcd = convert_depth_image_to_point_cloud(
                depth_syn.astype(np.float32), intrinsic, extrinsic, depth_scale=1.0
            )
            syn_pts = np.asarray(syn_pcd.points)
            trafo, reg = _run_icp(syn_pts, tgt_pcd)

            if self.show and reg is not None:
                logger.info(f"{self.name} (instance {i}): fitness={reg.fitness:.4f}, inlier_rmse={reg.inlier_rmse:.4f}")

            rot = trafo[:3, :3]
            rvec = R.from_matrix(rot).as_rotvec()
            angle = float(np.linalg.norm(rvec)) if rvec is not None else 0.0
            if angle > np.deg2rad(self.max_rot_deg):
                continue

            tnorm = float(np.linalg.norm(trafo[:3, 3]))
            if tnorm > self.max_trans:
                continue

            V_all[vo : vo + vl] = apply_trafo(V_all[vo : vo + vl], trafo)
            if "mesh.normals" in data and isinstance(data["mesh.normals"], np.ndarray):
                data["mesh.normals"][vo : vo + vl] = (rot @ data["mesh.normals"][vo : vo + vl].T).T
            if p_lengths and p_offsets and "points" in data and isinstance(data["points"], np.ndarray):
                po = p_offsets[i]
                pl = p_lengths[i]
                data["points"][po : po + pl] = apply_trafo(data["points"][po : po + pl], trafo)
            if pc_lengths and pc_offsets and "pointcloud" in data and isinstance(data["pointcloud"], np.ndarray):
                pco = pc_offsets[i]
                pcl = pc_lengths[i]
                data["pointcloud"][pco : pco + pcl] = apply_trafo(data["pointcloud"][pco : pco + pcl], trafo)
                if "pointcloud.normals" in data and isinstance(data["pointcloud.normals"], np.ndarray):
                    data["pointcloud.normals"][pco : pco + pcl] = (
                        rot @ data["pointcloud.normals"][pco : pco + pcl].T
                    ).T
        renderer.delete()

        data["mesh.vertices"] = V_all
        return data


class AxesCutPointcloud(Transform):
    @get_args()
    def __init__(
        self,
        apply_to="inputs",
        axes: str = "xyz",
        cut_ratio: tuple[float, float] | float = 0.5,
        rotate_object: str = "",
        upper_hemisphere: bool = False,
    ):
        super().__init__(apply_to=apply_to)

        self.axes = axes
        self.cut_ratio = cut_ratio
        self.rotate_object = rotate_object
        self.upper_hemisphere = upper_hemisphere

    def apply(self, data, key):
        points = data[key]
        normals = data.get(f"{key}.normals")

        pitch = 0
        if self.rotate_object:
            rot, pitch = rot_from_euler(self.rotate_object, self.upper_hemisphere)
            points = (rot @ points.T).T

        if all(axis in self.axes for axis in "xyz"):
            side = np.random.randint(3)
        else:
            side = list()
            if "x" in self.axes:
                side.append(0)
            if "y" in self.axes:
                side.append(1)
            if "z" in self.axes:
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
            data["pitch"] = pitch
        if indices.any():
            return data

        data[key] = data[key][indices]
        if normals is not None:
            data[f"{key}.normals"] = data[f"{key}.normals"][indices]

        return data


class SphereCutPointcloud(Transform):
    @get_args()
    def __init__(
        self, apply_to, radius: float = 0.01, num_spheres: int = 100, random: bool = True, max_percent: float = 0.5
    ):
        super().__init__(apply_to=apply_to)

        self.radius = radius
        self.num_spheres = num_spheres
        self.random = random
        self.max_percent = max_percent

    def apply(self, data, key):
        points = data[key]
        normals = data.get(f"{key}.normals")

        if len(points) < self.num_spheres:
            return data

        num_spheres = self.num_spheres
        if self.random:
            num_spheres = np.random.randint(self.num_spheres)

        indices = np.zeros(len(points), dtype=bool)
        center_indices = np.random.randint(0, len(points), num_spheres)
        centers = points[center_indices]
        radii = self.radius * np.random.rand(num_spheres) if self.random else np.full(num_spheres, self.radius)

        distances = np.linalg.norm(points - centers[:, np.newaxis], axis=2)
        new_indices = np.any(distances < radii[:, np.newaxis], axis=0)
        indices[new_indices] = True

        if (indices.sum() / len(points)) > self.max_percent:
            indices = np.zeros(len(points), dtype=bool)

        data[key] = points[~indices]
        if normals is not None:
            data[f"{key}.normals"] = normals[~indices]

        return data


class SphereMovePointcloud(Transform):
    @get_args()
    def __init__(
        self,
        apply_to,
        radius: float = 0.01,
        num_spheres: int = 100,
        random: bool = True,
        offset_amount: float = 0.01,
        inward_probability: float = 0.5,
        max_percent: float = 0.5,
    ):
        super().__init__(apply_to=apply_to)

        self.radius = radius
        self.num_spheres = num_spheres
        self.random = random
        self.offset_amount = offset_amount
        self.inward_probability = inward_probability
        self.max_percent = max_percent

    def apply(self, data, key):
        points = data[key]
        normals = data[f"{key}.normals"]

        if len(points) < self.num_spheres:
            return data

        # Determine how many spheres to create
        num_spheres = self.num_spheres
        if self.random:
            num_spheres = np.random.randint(self.num_spheres)

        # Generate random sphere centers and radii
        center_indices = np.random.randint(0, len(points), num_spheres)
        centers = points[center_indices]
        radii = self.radius * np.random.randn(num_spheres) if self.random else np.full(num_spheres, self.radius)

        # Calculate distances from each point to each sphere center
        distances = np.linalg.norm(points - centers[:, np.newaxis], axis=2)

        # Find points that are inside at least one sphere
        indices = np.any(distances < radii[:, np.newaxis], axis=0)

        # Check if we're modifying too many points
        if (indices.sum() / len(points)) > self.max_percent:
            return data

        # Create a copy of the points
        new_points = points.copy()

        # For each affected point, determine direction (inward/outward)
        directions = np.random.random(indices.sum()) < self.inward_probability

        # Calculate offsets (positive for outward, negative for inward)
        offset_multipliers = np.ones(indices.sum())
        offset_multipliers[directions] = -1

        # Calculate random offset amounts
        if self.random:
            offset_amounts = self.offset_amount * np.random.randn(indices.sum())
        else:
            offset_amounts = np.full(indices.sum(), self.offset_amount)

        # Move points along their normals
        new_points[indices] += normals[indices] * offset_multipliers[:, np.newaxis] * offset_amounts[:, np.newaxis]

        # Update data with modified points
        data[key] = new_points

        return data


class RenderPointcloud(Transform):
    @get_args()
    def __init__(self, width: int = 640, height: int = 480, verbose: bool = False):
        super().__init__()

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

    def apply(self, data, key):
        vertices = data["mesh.vertices"]
        triangles = data["mesh.triangles"]
        intrinsic = data.get("inputs.intrinsic")
        extrinsic = data.get("inputs.extrinsic")

        if intrinsic is None:
            intrinsic = self.default_intrinsic
            data["inputs.intrinsic"] = intrinsic
        else:
            intrinsic = _to_numpy_array(intrinsic)
            data["inputs.intrinsic"] = intrinsic
        if extrinsic is None:
            extrinsic = self.default_extrinsic
            data["inputs.extrinsic"] = extrinsic
        else:
            extrinsic = _to_numpy_array(extrinsic)
            data["inputs.extrinsic"] = extrinsic

        vis_timer = time.perf_counter()
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
        if self.verbose:
            print(f"{self.__class__.__name__}: Loading mesh takes {time.perf_counter() - vis_timer:.3f} seconds")

        vis_timer = time.perf_counter()
        open3d_extrinsic = np.eye(4)
        rot_yz_180 = R.from_euler("yz", (180, 180), degrees=True).as_matrix()
        open3d_extrinsic[:3, :3] = rot_yz_180 @ extrinsic[:3, :3]
        open3d_extrinsic[:3, 3] = extrinsic[:3, 3]
        if self.verbose:
            print(
                f"{self.__class__.__name__}: Setting camera extrinsics takes {time.perf_counter() - vis_timer:.3f} seconds"
            )

        vis_timer = time.perf_counter()
        vis = cast(Any, o3d).visualization.Visualizer()
        vis.create_window(width=self.width, height=self.height, visible=False)

        vis.add_geometry(mesh)

        param = o3d.camera.PinholeCameraParameters()
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        param.intrinsic.intrinsic_matrix = intrinsic
        param.extrinsic = open3d_extrinsic
        vis.get_view_control().convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        if self.verbose:
            print(f"{self.__class__.__name__}: Visualizer setup takes {time.perf_counter() - vis_timer:.3f}s")

        vis_timer = time.perf_counter()
        vis.capture_depth_point_cloud("test.ply", do_render=True, convert_to_world_coordinate=True)
        pcd = o3d.io.read_point_cloud("test.ply")
        if self.verbose:
            print(f"{self.__class__.__name__}: Rendering pointcloud takes {time.perf_counter() - vis_timer:.3f}s")

        # vis.clear_geometries()
        # vis.remove_geometry(mesh)
        vis.destroy_window()
        vis.close()

        data["inputs"] = np.asarray(pcd.points)
        data["inputs.rot"] = data.get("inputs.rot", extrinsic[:3, :3].T)  # TODO: Is this correct?
        data["inputs.path"] = data["mesh.path"]

        return data


class Render(Transform):
    @get_args()
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        offscreen: bool = True,
        method: str = "pyrender",
        device: str = "cuda",
        remove_mesh: bool = True,
        render_color: bool = True,
        render_depth: bool = False,
        render_normals: bool = False,
        sample_cam: bool = True,
        cache: bool = False,
    ):
        super().__init__(cachable=not sample_cam or cache)
        if not (render_color or render_depth or render_normals):
            raise ValueError("Must render at least one of color, depth, or normals")
        if render_normals and (method == "pytorch3d" or (method == "open3d" and offscreen)):
            raise ValueError("Normals can only be rendered with pyrender or open3d with onscreen rendering")
        if render_color and render_normals and method == "pyrender":
            raise ValueError("Color and normals cannot be rendered simultaneously with pyrender")

        self.width = width
        self.height = height
        self.offscreen = offscreen
        self.method = method
        self.device = device
        self.remove_mesh = remove_mesh
        self.render_color = render_color
        self.render_depth = render_depth
        self.render_normals = render_normals
        self.sample_cam = sample_cam

        self.default_intrinsic = np.eye(3)
        self.default_intrinsic[0, 0] = max(width, height)
        self.default_intrinsic[1, 1] = max(width, height)
        self.default_intrinsic[0, 2] = (width - 1) / 2
        self.default_intrinsic[1, 2] = (height - 1) / 2

        self.default_extrinsic_opengl = _to_numpy_array(inv_trafo(look_at(np.array([1, 0.5, 1]), np.zeros(3))))
        self.default_extrinsic_opencv = _to_numpy_array(
            convert_extrinsic(self.default_extrinsic_opengl, "opengl", "opencv")
        )

        if method == "pyrender" and offscreen:
            os.environ["PYOPENGL_PLATFORM"] = "egl"

    def render_open3d(
        self, vertices: np.ndarray, triangles: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        vis_timer = time.perf_counter()
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
        logger.debug(f"{self.name}: Loading mesh takes {time.perf_counter() - vis_timer:.3f} seconds")

        if self.render_normals:
            vis_timer = time.perf_counter()
            mesh.compute_vertex_normals()
            logger.debug(f"{self.name}: Computing normals takes {time.perf_counter() - vis_timer:.3f} seconds")
        normal_map = None

        vis_timer = time.perf_counter()
        logger.debug(f"{self.name}: Setting camera extrinsics takes {time.perf_counter() - vis_timer:.3f} seconds")

        if self.offscreen:
            render_timer = time.perf_counter()
            renderer = cast(Any, o3d).visualization.rendering.OffscreenRenderer(self.width, self.height)
            material = cast(Any, o3d).visualization.rendering.MaterialRecord()
            renderer.scene.add_geometry("mesh", mesh, material)
            renderer.setup_camera(intrinsic, extrinsic, self.width, self.height)
            logger.debug(f"{self.name}: Renderer setup takes {time.perf_counter() - render_timer:.3f}s")

            render_timer = time.perf_counter()
            depth_map = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
            depth_map[depth_map == np.inf] = 0
            logger.debug(f"{self.name}: Rendering depth map takes {time.perf_counter() - render_timer:.3f}s")

            # renderer.scene.clear_geometry()
        else:
            vis_timer = time.perf_counter()
            vis = cast(Any, o3d).visualization.Visualizer()
            vis.create_window(width=self.width, height=self.height, visible=False)

            vis.add_geometry(mesh)

            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.width, self.height, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2] - 0.5, intrinsic[1, 2] - 0.5
            )
            # param.intrinsic.intrinsic_matrix = intrinsic
            param.extrinsic = extrinsic
            vis.get_view_control().convert_from_pinhole_camera_parameters(param, allow_arbitrary=False)
            logger.debug(f"{self.name}: Visualizer setup takes {time.perf_counter() - vis_timer:.3f}s")

            if self.render_normals:
                vis_timer = time.perf_counter()
                vis.get_render_option().mesh_color_option = cast(Any, o3d).visualization.MeshColorOption.Normal
                normal_map = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                logger.debug(f"{self.name}: Rendering normals takes {time.perf_counter() - vis_timer:.3f}s")

            vis_timer = time.perf_counter()
            depth_map = np.asarray(vis.capture_depth_float_buffer(do_render=True))
            logger.debug(f"{self.name}: Rendering depth map takes {time.perf_counter() - vis_timer:.3f}s")

            # vis.clear_geometries()
            # vis.remove_geometry(mesh)
            vis.destroy_window()
            vis.close()
        return depth_map, normal_map

    def render_pytorch3d(
        self, vertices: np.ndarray, triangles: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        intrinsic = intrinsic.astype(np.float32)

        meshes = Meshes(verts=to_tensor(vertices, device=self.device), faces=to_tensor(triangles, device=self.device))

        cameras = PerspectiveCameras(
            focal_length=((intrinsic[0, 0], intrinsic[1, 1]),),
            principal_point=((intrinsic[0, 2], intrinsic[1, 2]),),
            R=to_tensor(extrinsic[:3, :3], device=self.device),
            T=to_tensor(extrinsic[:3, 3], device=self.device),
            device=self.device,
            in_ndc=False,
            image_size=((self.height, self.width),),
        )
        raster_settings = RasterizationSettings(
            image_size=(self.height, self.width), blur_radius=0.0, faces_per_pixel=1, bin_size=0
        )

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(meshes)
        depth_map = fragments.zbuf.squeeze().cpu().numpy()
        depth_map[depth_map == -1] = 0

        return depth_map, None

    def render_pyrender(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        z_near: float = 0.01,
        z_far: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        renderer_timer = time.perf_counter()
        mesh = Trimesh(vertices=vertices, faces=triangles, process=False, validate=False)
        logger.debug(f"{self.name}: Loading mesh takes {time.perf_counter() - renderer_timer:.3f} seconds")

        renderer_timer = time.perf_counter()
        flags = pyrender.renderer.RenderFlags.NONE
        if not self.render_color and not self.render_normals:
            flags = pyrender.RenderFlags.DEPTH_ONLY
        if self.offscreen:
            flags |= pyrender.RenderFlags.OFFSCREEN

        scene = pyrender.Scene(ambient_light=0.2 * np.ones(3) if self.render_color else None)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        camera = pyrender.IntrinsicsCamera(
            intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2], znear=z_near, zfar=z_far
        )
        scene.add(camera, pose=np.linalg.inv(extrinsic))
        if self.render_color:
            position = np.random.uniform(-2, 2, 3)
            while np.max(np.abs(position)) <= 1:
                position = np.random.uniform(-5, 5, 3)
            pose = np.eye(4)
            pose[:3, 3] = position
            point_intensity = np.linalg.norm(position) * np.clip(np.random.normal(loc=15, scale=5), 10, 20)
            light = pyrender.PointLight(intensity=point_intensity)
            scene.add(light, pose=pose)

        logger.debug(f"{self.name}: Renderer setup takes {time.perf_counter() - renderer_timer:.2f} seconds")

        image = None
        if self.render_color or self.render_normals:
            image, depth_map = cast(tuple[np.ndarray, np.ndarray], self.renderer.render(scene, flags=flags))
            if self.render_normals:
                image = image / 255
        else:
            depth_map = cast(np.ndarray, self.renderer.render(scene, flags=flags))
        depth_map[depth_map >= z_far] = 0
        depth_map[depth_map < z_near] = 0

        return depth_map, image.copy() if image is not None else None

    @cached_property
    def renderer(self):
        renderer = pyrender.OffscreenRenderer(self.width, self.height)
        if self.render_normals:
            shader_dir = Path(__file__).parent.parent.parent / "utils" / "assets" / "shaders"
            cast(Any, renderer)._renderer._program_cache = ShaderProgramCache(shader_dir=shader_dir)
        return renderer

    def __del__(self):
        renderer = self.__dict__.get("renderer")
        if renderer is not None:
            cast(Any, renderer).delete()
        vis = self.__dict__.get("vis")
        if vis is not None:
            cast(Any, vis).destroy_window()
            cast(Any, vis).close()

    def apply(self, data, key):
        vertices = _to_numpy_array(data["mesh.vertices"])
        triangles = _to_numpy_array(data["mesh.triangles"])
        intrinsic = data.get("inputs.intrinsic")
        extrinsic = data.get("inputs.extrinsic")

        if intrinsic is None:
            intrinsic = self.default_intrinsic
            data["inputs.intrinsic"] = intrinsic
        else:
            intrinsic = _to_numpy_array(intrinsic)
            data["inputs.intrinsic"] = intrinsic
        if extrinsic is None:
            if self.sample_cam:
                loc = np.random.uniform(-1, 1, 3)
                loc[1] = np.abs(loc[1])
                loc /= np.random.uniform(0.2, 0.8) * np.linalg.norm(loc)
                extrinsic = _to_numpy_array(inv_trafo(look_at(loc, np.zeros(3))))
                data["inputs.extrinsic"] = _to_numpy_array(convert_extrinsic(extrinsic, "opengl", "opencv"))
                if self.method != "pyrender":
                    extrinsic = _to_numpy_array(data["inputs.extrinsic"])
            else:
                extrinsic = (
                    self.default_extrinsic_opengl if self.method == "pyrender" else self.default_extrinsic_opencv
                )
                data["inputs.extrinsic"] = self.default_extrinsic_opencv
        else:
            extrinsic = _to_numpy_array(extrinsic)
            data["inputs.extrinsic"] = extrinsic

        intrinsic_np = _to_numpy_array(intrinsic)
        extrinsic_np = _to_numpy_array(extrinsic)

        if self.method == "open3d":
            depth, image = self.render_open3d(vertices, triangles, intrinsic_np, extrinsic_np)
        elif self.method == "pytorch3d":
            depth, image = self.render_pytorch3d(vertices, triangles, intrinsic_np, extrinsic_np)
        elif self.method == "pyrender":
            depth, image = self.render_pyrender(vertices, triangles, intrinsic_np, extrinsic_np)
        else:
            raise ValueError(f"Unknown rendering method {self.method}.")

        data["inputs"] = depth
        data["inputs.mask"] = depth > 0
        data["inputs.rot"] = data.get("inputs.rot", extrinsic_np[:3, :3].T)  # TODO: Is this correct?
        data["inputs.path"] = data["mesh.path"]
        data["inputs.name"] = data["mesh.name"]
        data["inputs.width"] = self.width
        data["inputs.height"] = self.height

        if image is not None:
            if self.render_color:
                image[depth == 0] = 255
                data["inputs.image"] = image.astype(np.uint8)
            if self.render_normals:
                image[depth == 0] = 0
                data["inputs.image"] = (255 * image).astype(np.uint8)  # FIXME: For DVR testing
                data["inputs.normals"] = image

        if not self.render_depth:
            data["inputs"] = data.pop("inputs.image")

        if self.remove_mesh:
            for key in list(data.keys()):
                if key.startswith("mesh."):
                    data.pop(key)
        return data


class DepthToPointcloud(Transform):
    def __init__(self, apply_extrinsic: bool = True, cache: bool = False):
        super().__init__(cachable=cache)
        self.apply_extrinsic = apply_extrinsic

    def apply(self, data, key):
        depth = data.get("inputs.depth", data["inputs"])
        normals = data.get("inputs.normals")
        colors = data.get("inputs.image")
        intrinsic = data["inputs.intrinsic"]
        extrinsic = data.get("inputs.extrinsic") if self.apply_extrinsic else None

        if normals is not None:
            pcd = convert_rgbd_image_to_point_cloud(
                [normals, depth],
                intrinsic,
                extrinsic,
                depth_scale=1.0,
                depth_trunc=float("inf"),
                convert_rgb_to_intensity=False,
            )
        elif colors is not None:
            pcd = convert_rgbd_image_to_point_cloud(
                [colors, depth],
                intrinsic,
                extrinsic,
                depth_scale=1.0,
                depth_trunc=float("inf"),
                convert_rgb_to_intensity=False,
            )
        else:
            pcd = convert_depth_image_to_point_cloud(
                depth, intrinsic, extrinsic, depth_scale=1.0, depth_trunc=float("inf")
            )

        data["inputs"] = np.asarray(pcd.points)
        data["inputs.depth"] = depth
        if normals is not None:
            data["inputs.normals"] = np.asarray(pcd.colors) * 2 - 1
        elif colors is not None:
            data["inputs.colors"] = np.asarray(pcd.colors)
        return data


class RenderDepthMaps(Transform):
    @get_args()
    def __init__(
        self,
        step: int = 5,
        min_angle: int = 0,
        mapitch: int = 360,
        width: int = 640,
        height: int = 480,
        offscreen: bool = True,
        inplane_rot: float | None = None,
    ):
        super().__init__()

        self.step = step
        self.min_angle = min_angle
        self.mapitch = mapitch
        self.width = width
        self.height = height
        self.offscreen = offscreen
        self.inplane_rot = np.deg2rad(inplane_rot) if inplane_rot is not None else None

        self.intrinsic: np.ndarray | None = None
        self.extrinsic: np.ndarray | None = None
        self.mesh: o3d.geometry.TriangleMesh | None = None
        self.camera_position: np.ndarray | None = None
        self.camera_look_at: np.ndarray | None = None

        self.depth_list: list[np.ndarray] = []
        self.extrinsic_list: list[np.ndarray] = []
        self.angle_list: list[float] = []

        self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        self.flags = pyrender.RenderFlags.DEPTH_ONLY
        if self.offscreen:
            self.flags |= pyrender.RenderFlags.OFFSCREEN

    @staticmethod
    def compute_look_at_matrix(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
        forward_vec = to_vec - from_vec
        forward_vec /= np.linalg.norm(forward_vec)

        right_vec = np.cross(np.array([0, 1, 0]), forward_vec)
        up_vec = np.cross(forward_vec, right_vec)
        return np.array([right_vec, up_vec, forward_vec]).T

    @staticmethod
    def rotation_from_forward_vec(
        forward_vec: np.ndarray | Any, up_axis: str = "Y", inplane_rot: float | None = None
    ) -> np.ndarray:
        from mathutils import Euler, Vector

        forward = np.asarray(forward_vec, dtype=np.float64).reshape(-1)
        rotation_matrix = Vector(forward.tolist()).to_track_quat("-Z", up_axis).to_matrix()
        if inplane_rot is not None:
            rotation_matrix = rotation_matrix @ Euler((0.0, 0.0, inplane_rot)).to_matrix()
        return np.array(rotation_matrix)

    @staticmethod
    def get_position(x: float, y: float, angle: float) -> tuple[float, float]:
        angle = np.deg2rad(angle)
        x_new = y * np.sin(angle) + x * np.cos(angle)
        y_new = y * np.cos(angle) - x * np.sin(angle)
        return x_new, y_new

    def get_params(self, extrinsic: np.ndarray):
        if self.intrinsic is None:
            raise ValueError("RenderDepthMaps requires intrinsic before camera params.")
        intrinsic = self.intrinsic
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2] - 0.5, intrinsic[1, 2] - 0.5
        )
        # params.intrinsic.intrinsic_matrix = self.intrinsic
        params.extrinsic = extrinsic
        return params

    def update_camera(self, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.camera_position is None or self.camera_look_at is None:
            raise ValueError("RenderDepthMaps camera state is not initialized.")
        camera_position = np.asarray(self.camera_position, dtype=np.float32)
        camera_look_at = np.asarray(self.camera_look_at, dtype=np.float32)
        x, y = self.get_position(float(camera_position[0]), float(camera_position[1]), angle)
        new_camera_position = np.array([x, y, float(camera_position[2])], dtype=np.float32)

        x, y = self.get_position(float(camera_look_at[0]), float(camera_look_at[1]), angle)
        new_camera_look_at = np.array([x, y, float(camera_look_at[2])], dtype=np.float32)

        inv_extrinsic = np.eye(4)
        inv_extrinsic[:3, 3] = new_camera_position

        inv_extrinsic[:3, :3] = self.rotation_from_forward_vec(
            forward_vec=new_camera_position - new_camera_look_at, up_axis="Y", inplane_rot=self.inplane_rot
        )

        extrinsic = _to_numpy_array(inv_trafo(inv_extrinsic))
        return extrinsic, new_camera_position, new_camera_look_at

    def show(self, box: bool = False):
        if self.mesh is None or self.intrinsic is None:
            raise ValueError("RenderDepthMaps requires mesh and intrinsic to visualize.")
        mesh = self.mesh
        intrinsic = self.intrinsic
        mesh.compute_vertex_normals()
        geometries = [mesh, o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)]

        if box:
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.55,) * 3, max_bound=(0.55,) * 3)
            bbox.color = np.zeros(3)
            geometries.append(bbox)

        for depth, extrinsic in zip(self.depth_list, self.extrinsic_list, strict=False):
            pcd = convert_depth_image_to_point_cloud(depth, intrinsic, extrinsic, depth_scale=1, depth_trunc=10)
            geometries.append(pcd)

            cam = draw_camera(intrinsic, extrinsic, width=640, height=480, scale=0.25)
            geometries.extend(cam)
        cast(Any, o3d).visualization.draw_geometries(geometries)

    def apply(self, data, key):
        self.depth_list.clear()
        self.extrinsic_list.clear()
        self.angle_list.clear()

        self.intrinsic = _to_numpy_array(data["inputs.intrinsic"])
        self.extrinsic = _to_numpy_array(data["inputs.extrinsic"])

        inv_extrinsic = _to_numpy_array(inv_trafo(self.extrinsic))
        self.camera_position = inv_extrinsic[:3, 3]
        camera_look_at = data.get("inputs.look_at")
        self.camera_look_at = (
            np.zeros(3, dtype=np.float32) if camera_look_at is None else _to_numpy_array(camera_look_at)
        )

        vertices = _to_numpy_array(data["mesh.vertices"])
        triangles = _to_numpy_array(data["mesh.triangles"])
        self.mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
        )

        scene = pyrender.Scene()
        mesh = Trimesh(vertices=vertices, faces=triangles, process=False)
        mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False))
        scene.add_node(mesh_node)
        camera = pyrender.IntrinsicsCamera(
            float(self.intrinsic[0, 0]),
            float(self.intrinsic[1, 1]),
            float(self.intrinsic[0, 2]),
            float(self.intrinsic[1, 2]),
            znear=0.1,
            zfar=10.0,
        )
        camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
        scene.add_node(camera_node)

        for angle in range(0, 360, self.step):
            extrinsic, _, _ = self.update_camera(-angle)
            inv_extrinsic = inv_trafo(extrinsic)
            rot_z = 180 + np.rad2deg(np.arctan2(inv_extrinsic[1, 0], inv_extrinsic[0, 0]))
            if self.min_angle <= rot_z < self.mapitch and inv_extrinsic[0, 3] < 0:
                extrinsic_opengl = extrinsic.copy()
                extrinsic_opengl[1, :] *= -1
                extrinsic_opengl[2, :] *= -1
                scene.set_pose(camera_node, pose=inv_trafo(extrinsic_opengl))
                depth_map = cast(np.ndarray, self.renderer.render(scene, flags=self.flags))

                self.depth_list.append(depth_map)
                self.extrinsic_list.append(extrinsic)
                self.angle_list.append(angle)

        return self.depth_list, self.extrinsic_list, self.angle_list

    def __del__(self):
        self.renderer.delete()


class FindUncertainPoints(Transform):
    @get_args()
    def __init__(
        self,
        depth_list: list[np.ndarray],
        angle_list: list[float],
        max_chamfer_dist: float = 0.01,
        parallel: bool = False,
        debug: bool = False,
        show: bool = False,
    ):
        super().__init__()

        self.depth_list = depth_list
        self.angle_list = angle_list
        self.max_chamfer_dist = max_chamfer_dist
        self.parallel = parallel
        self.debug = debug
        self.show = show

        self.init_depth = self.depth_list.pop(0)
        self.init_angle = self.angle_list.pop(0)
        self.init_inputs: np.ndarray | None = None

    @staticmethod
    def get_init_inputs(depth: np.ndarray, intrinsic: np.ndarray, scale: float) -> np.ndarray:
        pcd = convert_depth_image_to_point_cloud(depth, intrinsic, depth_scale=1, depth_trunc=10)
        inputs = np.asarray(pcd.points).copy()
        inputs /= scale
        return inputs

    def eval_single(self, data: dict, scale: float, depth: np.ndarray, angle: float) -> np.ndarray:
        if self.init_inputs is None:
            raise ValueError("FindUncertainPoints.init_inputs is not initialized.")

        pcd = convert_depth_image_to_point_cloud(depth, data["inputs.intrinsic"], depth_scale=1, depth_trunc=10)
        inputs = np.asarray(pcd.points).copy()
        inputs /= scale

        init_inputs_raw = self.init_inputs
        offset = (init_inputs_raw.max(axis=0) + init_inputs_raw.min(axis=0)) / 2
        init_inputs = init_inputs_raw - offset
        inputs -= offset

        too_large1 = inputs.min(axis=0)[1] / init_inputs.min(axis=0)[1]
        too_large2 = inputs.max(axis=0)[1] / init_inputs.max(axis=0)[1]

        if self.debug:
            print("Size:", too_large1, too_large2)
        if too_large1 > 1.06 or too_large2 > 1.06:
            if self.debug:
                print("Size skip")
            return np.zeros((0,), dtype=bool)

        kdtree = SKDTree(inputs)
        dist1, _index = kdtree.query(init_inputs, workers=1 if self.parallel else -1)

        kdtree = SKDTree(init_inputs)
        dist2, _index = kdtree.query(inputs, workers=1 if self.parallel else -1)

        dist1_mean = dist1.mean()
        dist2_mean = dist2.mean()
        chamfer_l1 = 0.5 * (dist1_mean + dist2_mean)

        if self.debug:
            print("Chamfer L1:", chamfer_l1)
        if chamfer_l1 > self.max_chamfer_dist:
            if self.debug:
                print("Chamfer skip")
            return np.zeros((0,), dtype=bool)

        if self.debug and self.show:
            cast(Any, o3d).visualization.draw_geometries(
                [
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs)).paint_uniform_color((1, 0, 0)),
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(init_inputs)).paint_uniform_color((0, 0, 1)),
                    o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1),
                ]
            )

        rot = R.from_euler("y", -angle, degrees=True).as_matrix()
        vertices = _to_numpy_array(data["mesh.vertices"]).copy()
        vertices = (rot @ vertices.T).T
        points = _to_numpy_array(data["points"])
        mesh = Trimesh(vertices, _to_numpy_array(data["mesh.triangles"]), process=False)

        return np.asarray(check_mesh_contains(mesh, points), dtype=bool)

    @staticmethod
    def check_occupancy(occupancy: list[np.ndarray]) -> list[np.ndarray]:
        occupancy = [occ for occ in occupancy if len(occ)]
        if len(occupancy) <= 3:
            return []
        return occupancy

    def get_uncertain(self, data: dict, occupancy: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        occupancy_arr = np.vstack(occupancy)
        always = np.all(occupancy_arr, axis=0)
        sometimes = np.any(occupancy_arr, axis=0)
        uncertain = sometimes & ~always

        # Remove uncertain points in observed free space
        item = Rotate(axes="x", from_inputs=True)(data)
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

    def apply(self, data, key):
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
                occupancy.extend(
                    p(
                        delayed(self.eval_single)(data, scale, depth, angle)
                        for depth, angle in zip(self.depth_list, self.angle_list, strict=False)
                    )
                )
        else:
            for depth, angle in zip(self.depth_list, self.angle_list, strict=False):
                occupancy.append(self.eval_single(data, scale, depth, angle))

        occupancy = self.check_occupancy(occupancy)
        if occupancy:
            uncertain, always, _sometimes = self.get_uncertain(data, occupancy)

            data["points.uncertain"] = uncertain
            data["points.occ"] = always

            return data
        return data


class LoadUncertain(Transform):
    def __init__(self):
        super().__init__()

    def apply(self, data, key):
        inputs_path = Path(data["inputs.path"])
        uncertain = list()
        for path in data["points.path"]:
            uncertain_path = inputs_path.parent.joinpath("_".join([inputs_path.stem, Path(path).stem, "uncertain.npy"]))

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
    @get_args()
    def __init__(
        self,
        apply_to="inputs",
        rotate_object: str = "",
        upper_hemisphere: bool = False,
        rot_from_inputs: bool = False,
        cam_from_inputs: bool = False,
        distance_multiplier: int = 100,
    ):
        super().__init__(apply_to=apply_to)

        self.rotate_object = rotate_object
        self.upper_hemisphere = upper_hemisphere
        self.cam_from_inputs = cam_from_inputs
        self.rot_from_inputs = rot_from_inputs
        self.distance_multiplier = distance_multiplier

    def apply(self, data, key):
        points = data[key]
        normals = data.get(f"{key}.normals")

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        pitch = 0
        if self.rotate_object:
            rot, pitch = rot_from_euler(self.rotate_object, self.upper_hemisphere)
            trafo = np.eye(4)
            trafo[:3, :3] = rot
            pcd.transform(trafo)
        elif self.rot_from_inputs:
            rot = data["extrinsic"][:3, :3]
            pcd.rotate(rot, center=(0, 0, 0))

        if self.cam_from_inputs:
            camera = inv_trafo(data["extrinsic"])[:3, 3]
            diameter = np.linalg.norm(camera)
        else:
            diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
            camera = np.array([0, 0, diameter])
        _, indices = pcd.hidden_point_removal(camera, self.distance_multiplier * diameter)

        data[key] = points[indices]
        data["pitch"] = pitch
        if normals is not None:
            data[f"{key}.normals"] = normals[indices]

        return data


class RemoveHiddenPointsFromInputs(Transform):
    @get_args()
    def __init__(self, cam_from_inputs: bool = False):
        super().__init__(apply_to="inputs")

        self.cam_from_inputs = cam_from_inputs

    def apply(self, data, key):
        inputs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data[key]))
        if f"{key}.normals" in data:
            inputs.normals = o3d.utility.Vector3dVector(data[f"{key}.normals"])
        if f"{key}.colors" in data:
            inputs.colors = o3d.utility.Vector3dVector(data[f"{key}.colors"])

        if self.cam_from_inputs:
            cam = data[f"{key}.cam"]
            camera = -cam
            diameter = cam[2]
        else:
            diameter = np.linalg.norm(np.asarray(inputs.get_max_bound()) - np.asarray(inputs.get_min_bound()))
            camera = np.array([0, 0, diameter])
        _, indices = inputs.hidden_point_removal(camera, 1000 * diameter)

        inputs = inputs.select_by_index(indices)

        data[key] = np.asarray(inputs.points)
        if inputs.has_normals():
            data[f"{key}.normals"] = np.asarray(inputs.normals)
        if inputs.has_colors():
            data[f"{key}.colors"] = np.asarray(inputs.colors)

        return data


class SubsamplePoints(Transform):
    @get_args()
    def __init__(
        self,
        num_samples: int | tuple[float, float] = 2048,
        in_out_ratio: float | None = None,
        in_volume: Literal["cube", "frustum"] | None = None,
        padding: float = 0.1,
        scale_factor: float = 1.0,
        near: float = 0.2,
        far: float = 2.4,
        voxel_res: int | None = None,
        per_voxel_cap: int = 1,
        inverse_density: bool = False,
        cachable: bool = False,
    ):
        super().__init__(apply_to="points", cachable=cachable)

        self.num_samples = num_samples
        self.in_out_ratio = in_out_ratio
        self.in_volume = in_volume
        self.padding = padding
        self.scale_factor = scale_factor
        self.near = near
        self.far = far
        self.voxel_res = voxel_res
        self.per_voxel_cap = per_voxel_cap
        self.inverse_density = inverse_density
        self.warn_num_samples = 0

    @staticmethod
    def _sample_ratio(num_samples: int, points: np.ndarray, occ: np.ndarray, ratio: float) -> np.ndarray:
        try:
            indices_in_all = np.where(occ == 1)[0]
            indices_out_all = np.where(occ == 0)[0]

            n_in_target = int(num_samples * ratio)
            n_out_target = num_samples - n_in_target

            if not len(indices_in_all):
                n_in_target = 0
                n_out_target = num_samples
            elif not len(indices_out_all):
                n_in_target = num_samples
                n_out_target = 0

            replace_in = n_in_target > len(indices_in_all)
            indices_in = np.random.choice(indices_in_all, n_in_target, replace=replace_in)

            replace_out = n_out_target > len(indices_out_all)
            indices_out = np.random.choice(indices_out_all, n_out_target, replace=replace_out)

            return np.concatenate([indices_in, indices_out])
        except (ValueError, TypeError):
            return subsample_indices(points, num_samples)

    @staticmethod
    def _voxel_keys(points: np.ndarray, res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Compute voxel indices in [0, res-1]^3 for the AABB of points
        eps = 1e-12
        mn = points.min(axis=0)
        mx = points.max(axis=0)
        span = np.maximum(mx - mn, eps)
        ijk = np.floor((points - mn) / span * res).astype(np.int64)
        ijk = np.clip(ijk, 0, res - 1)
        # Flatten key for fast grouping
        key = (ijk[:, 0] * res + ijk[:, 1]) * res + ijk[:, 2]
        return key, mn, mx

    @staticmethod
    def _sample_voxelgrid(points: np.ndarray, num_samples: int, res: int, per_voxel_cap: int = 1) -> np.ndarray:
        # One random representative per voxel (very fast); optional multi-cap per voxel
        N = len(points)
        if N == 0:
            return np.zeros((0,), dtype=np.int64)
        key, _, _ = SubsamplePoints._voxel_keys(points, res)

        order = np.random.permutation(N)
        key_shuf = key[order]

        # First take 1 per voxel
        _, first_idx = np.unique(key_shuf, return_index=True)
        sel = order[first_idx]

        if per_voxel_cap > 1:
            # Additional passes to allow more per voxel if needed (cheap loop for small caps)
            rem_mask = np.ones(N, dtype=bool)
            rem_mask[sel] = False
            remaining = np.where(rem_mask)[0]
            if len(remaining) > 0:
                key_rem = key[remaining]
                counts = np.bincount(key[sel], minlength=key.max() + 1)  # faster than zeros + assign

                # iterate once; promote some remaining where cap not hit
                rem_order_idx = np.random.permutation(len(remaining))
                rem_order = remaining[rem_order_idx]
                k_rem_order = key_rem[rem_order_idx]

                picked_extra = []
                for idx, k in zip(rem_order, k_rem_order, strict=False):
                    if counts[k] < per_voxel_cap:
                        picked_extra.append(idx)
                        counts[k] += 1
                        if (len(sel) + len(picked_extra)) >= num_samples:
                            break
                if picked_extra:
                    sel = np.concatenate([sel, np.array(picked_extra, dtype=np.int64)], axis=0)

        # Trim or pad
        if len(sel) >= num_samples:
            return sel[:num_samples]
        else:
            # Fill remainder with random picks (keeps it fast)
            need = num_samples - len(sel)

            # Prefer a quick boolean mask over setdiff1d
            rem_mask = np.ones(N, dtype=bool)
            rem_mask[sel] = False
            candidates = np.where(rem_mask)[0]

            if candidates.size == 0:
                # No remaining unique candidates; sample with replacement from all points
                fill = np.random.choice(np.arange(N, dtype=np.int64), size=need, replace=True)
            else:
                replace = need > candidates.size
                fill = np.random.choice(candidates, size=need, replace=replace)

            return np.concatenate([sel, fill], axis=0)

    @staticmethod
    def _sample_volume(
        num_samples: int,
        points: np.ndarray,
        volume: Literal["cube", "frustum"],
        padding: float = 0.1,
        intrinsic: np.ndarray | None = None,
        extrinsic: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
        near: float = 0.2,
        far: float = 2.4,
        scale_factor: float = 1.0,
        inverse_density: bool = False,
    ) -> np.ndarray:  # default False for speed
        if volume == "cube":
            bound = 0.5 * scale_factor + padding / 2
            p_unif = np.random.uniform(-bound, bound, size=(num_samples, 3))
        elif volume == "frustum":
            if intrinsic is None or extrinsic is None or width is None or height is None:
                raise ValueError("Frustum sampling requires intrinsic, extrinsic, width, and height.")
            # Note: this path is slower (rays + KDTree). Prefer voxel_res for speed.
            ray0, ray_dirs, _u, _v = get_rays(
                np.ones((int(height), int(width))), intrinsic, extrinsic, num_samples=num_samples
            )
            d = sample_distances(n_points=len(ray0), near=near, far=far)
            p_unif = ray0 + d * ray_dirs
        else:
            raise ValueError(f"Unknown volume type: {volume}")

        # Single nearest-neighbor map into existing set
        kdtree = KDTree(points)
        indices = kdtree.query(p_unif, k=1)[1]

        if inverse_density:
            # Optional blue-noise-ish thinning; costs an extra query
            candidate_indices = np.unique(indices)
            candidate_points = points[candidate_indices]
            dist_to_neighbor = kdtree.query(candidate_points, k=2)[0][:, 1]
            probabilities = dist_to_neighbor + 1e-8
            probabilities /= probabilities.sum()
            return np.random.choice(
                candidate_indices, size=num_samples, replace=num_samples > len(candidate_indices), p=probabilities
            )
        return indices

    def _sample(
        self,
        num_samples: int,
        points: np.ndarray,
        occ: np.ndarray,
        labels: np.ndarray | None = None,
        data: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        def _to_optional_int(value: Any) -> int | None:
            if value is None:
                return None
            if isinstance(value, Tensor):
                return int(value.item())
            if isinstance(value, np.ndarray):
                return int(np.asarray(value).reshape(-1)[0])
            return int(value)

        def _to_float(value: Any, default: float) -> float:
            if value is None:
                return default
            if isinstance(value, Tensor):
                return float(value.item())
            if isinstance(value, np.ndarray):
                return float(np.asarray(value).reshape(-1)[0])
            return float(value)

        # Fast path: voxel-stratified sampling over existing points
        if self.voxel_res:
            if self.in_out_ratio is not None:
                # Keep class balance with separate voxel passes
                n_in = int(num_samples * self.in_out_ratio)
                n_out = num_samples - n_in
                idx_in_all = np.where(occ == 1)[0]
                idx_out_all = np.where(occ == 0)[0]

                sel_in = (
                    self._sample_voxelgrid(points[idx_in_all], n_in, self.voxel_res, self.per_voxel_cap)
                    if len(idx_in_all)
                    else np.zeros((0,), dtype=np.int64)
                )
                sel_out = (
                    self._sample_voxelgrid(points[idx_out_all], n_out, self.voxel_res, self.per_voxel_cap)
                    if len(idx_out_all)
                    else np.zeros((0,), dtype=np.int64)
                )

                indices = np.concatenate([idx_in_all[sel_in], idx_out_all[sel_out]], axis=0)
            else:
                indices = self._sample_voxelgrid(points, num_samples, self.voxel_res, self.per_voxel_cap)

        # Optional: volume-based (slower). Use only when needed.
        elif self.in_volume:
            if data is None:
                raise ValueError("Volume sampling requires camera metadata in data.")
            volume = cast(Literal["cube", "frustum"], self.in_volume)
            near = _to_float(data.get("inputs.near"), self.near)
            far = _to_float(data.get("inputs.far"), self.far)
            indices = self._sample_volume(
                num_samples,
                points,
                volume=volume,
                padding=self.padding,
                intrinsic=cast(np.ndarray | None, data.get("inputs.intrinsic")),
                extrinsic=cast(np.ndarray | None, data.get("inputs.extrinsic")),
                width=_to_optional_int(data.get("inputs.width")),
                height=_to_optional_int(data.get("inputs.height")),
                near=near,
                far=far,
                scale_factor=self.scale_factor,
                inverse_density=self.inverse_density,
            )
        elif self.in_out_ratio:
            indices = self._sample_ratio(num_samples, points, occ, self.in_out_ratio)
        else:
            # Fast random subset from existing distribution
            indices = subsample_indices(points, num_samples)

        if labels is None:
            return points[indices], occ[indices], labels, indices
        return points[indices], occ[indices], labels[indices], indices

    def apply(self, data, key):
        points = data[key]
        occ = data[f"{key}.occ"]
        labels = data.get(f"{key}.labels")

        num_samples = self.num_samples
        if isinstance(num_samples, tuple):
            num_samples = int(np.random.uniform(*num_samples) * len(points))

        if len(points) < num_samples:
            self.warn_num_samples += 1
            if self.warn_num_samples == 10:
                logger.warning(f"{self.name}: Not enough points to sample from: {len(points)} < {num_samples}")

        if points.ndim == 2:
            points, occ, labels, indices = self._sample(num_samples, points, occ, labels, data)
        elif points.ndim == 3:
            points_list = list()
            occ_list = list()
            indices_list = list()
            for p, o in zip(points, occ, strict=False):
                p, o, _, idx = self._sample(num_samples, p, o, data=data)
                points_list.append(p)
                occ_list.append(o)
                indices_list.append(idx)
            points = np.stack(points_list, axis=0)
            occ = np.stack(occ_list, axis=0)
            indices = np.stack(indices_list, axis=0)
        else:
            raise ValueError(f"Unsupported points dimension: {points.ndim}")

        data[key] = points
        data[f"{key}.occ"] = occ
        data[f"{key}.indices"] = indices
        if isinstance(labels, np.ndarray):
            data[f"{key}.labels"] = labels
        return data


class SdfFromOcc(Transform):
    @get_args()
    def __init__(self, tsdf: float | None = None, remove_pointcloud: bool = True):
        super().__init__()

        self.tsdf = tsdf
        self.remove_pointcloud = remove_pointcloud

    @staticmethod
    def occ_to_sdf(
        points: np.ndarray, occ: np.ndarray, pcd: np.ndarray, distance_upper_bound: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        kdtree = KDTree(pcd)
        dist, idx = kdtree.query(
            points,
            eps=1e-4,
            # distance_upper_bound=distance_upper_bound)
        )
        if distance_upper_bound is not None:
            # dist[dist == np.inf] = distance_upper_bound
            dist[dist > distance_upper_bound] = distance_upper_bound
        dist[occ == 1] *= -1
        return dist, idx

    def apply(self, data, key):
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


class VoxelizePointcloud(Transform):
    @get_args()
    def __init__(
        self,
        apply_to,
        resolution: int = 128,
        padding: float = 0.1,
        method: Literal["simple", "kdtree", "open3d"] = "simple",
        scale_factor: float = 1.0,
        bounds: tuple[float, float] | tuple[tuple[float, float, float], tuple[float, float, float]] = (-0.5, 0.5),
        cachable: bool = True,
    ):
        super().__init__(apply_to=apply_to, cachable=cachable)

        self.voxelizer = Voxelizer(
            resolution=resolution, padding=padding, method=method, scale_factor=scale_factor, bounds=bounds
        )

    def apply(self, data, key):
        points = data[key]
        normals = data.get(f"{key}.normals")

        voxelized_inputs, voxel_indices = self.voxelizer(points)

        data[key] = voxelized_inputs

        if normals is not None:
            nx, ny, nz = self.voxelizer.grid_shape
            voxelized_normals = np.zeros((len(self.voxelizer.grid_points), 3), dtype=normals.dtype)
            voxelized_normals[voxel_indices] = normals
            data[f"{key}.normals"] = rearrange(
                voxelized_normals,
                pattern="(x y z) c -> x y z c",
                x=nx,
                y=ny,
                z=nz,
            )
            del normals

        return data


class VoxelizePoints(Transform):
    @get_args()
    def __init__(
        self,
        resolution: int = 128,
        padding: float = 0.1,
        scale_factor: float = 1.0,
        bounds: tuple[float, float] | tuple[tuple[float, float, float], tuple[float, float, float]] = (-0.5, 0.5),
        cachable: bool = True,
    ):
        super().__init__(apply_to="points", cachable=cachable)

        self.voxelizer = Voxelizer(resolution=resolution, padding=padding, scale_factor=scale_factor, bounds=bounds)

    def apply(self, data, key):
        points = data[key]
        occ = data[f"{key}.occ"]
        labels = data.get(f"{key}.labels")

        kdtree = KDTree(points)
        idx = kdtree.query(self.voxelizer.grid_points, k=1)[1]

        data[key] = self.voxelizer.grid_points
        data[f"{key}.occ"] = occ[idx]
        if isinstance(labels, np.ndarray):
            data[f"{key}.labels"] = labels[idx]

        return data


class Translate(Transform):
    @get_args()
    def __init__(
        self,
        axes: str = "xyz",
        amount: float | int | tuple[float, float] | tuple[float, float, float] | list[float] = 0.0,
        random: bool = False,
        reverse: bool = False,
    ):
        super().__init__()

        self.axes = axes
        self.amount = amount
        self.random = random
        self.reverse = reverse

    def apply(self, data, key):
        def _offset_from_value(value: Any) -> np.ndarray:
            if isinstance(value, (int, float)):
                scalar = float(value)
                if self.random:
                    return np.asarray(np.random.uniform(-scalar, scalar, size=3), dtype=np.float32)
                return np.full(3, scalar, dtype=np.float32)

            if isinstance(value, (tuple, list, np.ndarray)):
                values = np.asarray(value, dtype=np.float32).reshape(-1)
                if values.size == 2:
                    if self.random:
                        return np.asarray(
                            np.random.uniform(-float(values[0]), float(values[1]), size=3), dtype=np.float32
                        )
                    print(f"Warning: Translating {self.axes} with {self.amount} is ambiguous. Returning.")
                    return np.asarray([], dtype=np.float32)
                if values.size == 3:
                    if self.random:
                        return np.asarray([np.random.uniform(-float(v), float(v)) for v in values], dtype=np.float32)
                    return values.astype(np.float32, copy=False)

            raise ValueError("Translation amount must be scalar, 2-value range, or 3-value vector.")

        if self.reverse:
            previous_offset = data.get("offset")
            if previous_offset is None:
                offset = np.zeros(3, dtype=np.float32)
            else:
                previous = np.asarray(previous_offset, dtype=np.float32).reshape(-1)
                if previous.size == 1:
                    previous = np.full(3, float(previous[0]), dtype=np.float32)
                elif previous.size != 3:
                    raise ValueError("Offset must be scalar or a 3-value vector.")
                offset = -previous
        else:
            offset = _offset_from_value(self.amount)
            if offset.size == 0:
                return data

        if "x" not in self.axes:
            offset[0] = 0
        if "y" not in self.axes:
            offset[1] = 0
        if "z" not in self.axes:
            offset[2] = 0

        keys = [
            "inputs",
            "inputs.depth",
            "points",
            "pointcloud",
            "mesh.vertices",
            "bbox",
            "partnet.points",
        ]
        for key in keys:
            value = data.get(key)
            if isinstance(value, np.ndarray) and (
                (value.ndim == 2 and value.shape[1] == 3) or (value.ndim == 3 and value.shape[2] == 3)
            ):
                data[key] = value + offset

        extrinsic = data.get("inputs.extrinsic")
        if isinstance(extrinsic, np.ndarray):
            extrinsic = extrinsic.copy()
            extrinsic[:3, 3] = extrinsic[:3, 3] - extrinsic[:3, :3] @ offset
            data["inputs.extrinsic"] = extrinsic
            if "inputs.inv_extrinsic" in data:
                data["inputs.inv_extrinsic"] = inv_trafo(extrinsic)

        data["offset"] = offset

        return data


class Normalize(Transform):
    @get_args()
    def __init__(
        self,
        center: bool | str | None = None,
        to_min_val: str | None = None,
        to_max_val: str | None = None,
        to_front: bool = False,
        scale: bool = True,
        offset: float | tuple[float, float, float] | list[float] | None = None,
        true_height: bool = False,
        reference: str = "inputs",
        scale_method: str = "cube",
        scale_factor: float = 1.0,
        scale_quantiles: tuple[float, float] | None = None,
        center_method: Literal["bbox", "mean", "median", "percentile"] = "bbox",
        center_quantiles: tuple[float, float] = (0.02, 0.98),
        reverse: bool = False,
    ):
        super().__init__()
        assert reference.lower() in ["inputs", "depth", "inputs.depth", "points", "pointcloud", "mesh"], (
            f"Unknown reference: {reference}"
        )
        assert scale_method.lower() in ["cube", "sphere"], f"Unknown method: {scale_method}"
        assert center_method.lower() in ["bbox", "mean", "median"], f"Unknown center_method: {center_method}"

        self.center = center.lower() if isinstance(center, str) else "xyz" if center else ""
        self.to_min_val = to_min_val.lower() if to_min_val is not None else ""
        self.to_max_val = to_max_val.lower() if to_max_val is not None else ""
        self.to_front = to_front
        self.scale = scale
        self.offset = offset
        self.true_height = true_height
        self.reference = reference.lower()
        self.scale_method = scale_method.lower()
        self.scale_factor = scale_factor
        self.scale_quantiles = scale_quantiles
        self.center_method = center_method.lower()
        self.center_quantiles = center_quantiles
        self.reverse = reverse

    def apply(self, data, key):
        points = data.get("points")
        occ = data.get("points.occ")
        pointcloud = data.get("pointcloud")
        inputs = data.get("inputs")
        depth = data.get("inputs.depth")
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
            elif self.reference in ["depth", "inputs.depth"]:
                ref = depth
            elif self.reference == "pointcloud":
                ref = pointcloud
            elif self.reference == "mesh":
                ref = mesh_vertices
            elif self.reference == "points":
                if not isinstance(points, np.ndarray) or not isinstance(occ, np.ndarray):
                    raise ValueError("Reference 'points' requires 'points' and 'points.occ' arrays.")
                ref = points[occ > 0]
            else:
                raise ValueError(f"Reference cannot be {self.reference}.")

            assert isinstance(ref, np.ndarray) and ref.shape[1] == 3, "Reference must be a pointcloud."
            if len(ref) == 0:
                logger.warning(f"Reference {self.reference} for {data.get('inputs.path', data['index'])} is empty.")
                min_vals = -0.5 * np.ones(3) * self.scale_factor
                max_vals = 0.5 * np.ones(3) * self.scale_factor
            else:
                min_vals = ref.min(axis=0)
                max_vals = ref.max(axis=0)

            if self.true_height:
                min_height = float(min_vals[1])
                if isinstance(pointcloud, np.ndarray) and len(pointcloud):
                    min_height = float(np.min(pointcloud[:, 1]))
                ref = np.array([np.array([min_vals[0], min_height, min_vals[2]]), max_vals])

            base_center = (max_vals + min_vals) / 2.0
            scale = 1
            if self.scale:
                if self.scale_method == "cube":
                    if self.scale_quantiles and len(ref):
                        ql, qh = self.scale_quantiles
                        qmin = np.quantile(ref, ql, axis=0)
                        qmax = np.quantile(ref, qh, axis=0)
                        scale = (qmax - qmin).max()
                    else:
                        scale = (max_vals - min_vals).max()
                elif self.scale_method == "sphere":
                    if len(ref):
                        r = np.linalg.norm(ref - base_center, axis=1)
                        if self.scale_quantiles is not None:
                            scale = 2 * np.quantile(r, self.scale_quantiles[1])
                        else:
                            scale = 2 * r.max()
                else:
                    raise ValueError(f"Method cannot be {self.scale_method}.")
                scale /= self.scale_factor

            if self.center_method == "bbox":
                base_center = (max_vals + min_vals) / 2.0
            elif self.center_method == "mean":
                base_center = ref.mean(axis=0) if len(ref) else np.zeros(3)
            elif self.center_method == "median":
                base_center = np.median(ref, axis=0) if len(ref) else np.zeros(3)
            elif self.center_method == "percentile":
                ql, qh = self.center_quantiles
                q_min = np.quantile(ref, ql, axis=0) if len(ref) else -0.5 * np.ones(3)
                q_max = np.quantile(ref, qh, axis=0) if len(ref) else 0.5 * np.ones(3)
                base_center = (q_max + q_min) / 2.0
            else:
                raise ValueError(f"center_method cannot be {self.center_method}.")

            offset_x = offset_y = offset_z = 0
            if self.center:
                offset_x = base_center[0] if "x" in self.center else offset_x
                offset_y = base_center[1] if "y" in self.center else offset_y
                offset_z = base_center[2] if "z" in self.center else offset_z

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
                if isinstance(self.offset, (int, float)):
                    offset -= float(self.offset)
                else:
                    offset_shift = np.asarray(self.offset, dtype=np.float32).reshape(-1)
                    if offset_shift.size != 3:
                        raise ValueError("Offset must be a scalar or 3-value vector.")
                    offset -= offset_shift

            if self.center:
                data["inputs.norm_offset"] = offset
            if self.scale:
                data["inputs.norm_scale"] = scale

        if self.scale and scale == 0:
            scale = 1

        keys = [
            "inputs",
            "inputs.depth",
            "points",
            "pointcloud",
            "mesh.vertices",
            "bbox",
            "partnet.points",
        ]
        values = [
            inputs,
            depth,
            points,
            pointcloud,
            mesh_vertices,
            bbox,
            partnet,
        ]

        for key, value in zip(keys, values, strict=False):
            if isinstance(value, np.ndarray) and (
                (value.ndim == 2 and value.shape[1] == 3) or (value.ndim == 3 and value.shape[2] == 3)
            ):
                v_scale = value - offset
                if self.scale:
                    v_scale = v_scale / scale
                data[key] = v_scale.astype(value.dtype)

        extrinsic = data.get("inputs.extrinsic")
        if extrinsic is not None:
            norm_mat = np.eye(4, dtype=extrinsic.dtype)
            norm_mat[:3, :3] = np.diag((scale,) * 3)
            norm_mat[:3, 3] = offset
            data["inputs.extrinsic"] = extrinsic @ norm_mat

        return data


class BPS(Transform):
    @get_args()
    def __init__(
        self,
        num_points: int = 1024,
        resolution: int = 32,
        padding: float = 0.1,
        bounds: tuple[float, float] = (-0.5, 0.5),
        method: str = "kdtree",
        feature: str | list[str] = "delta",
        basis: str = "sphere",
        squeeze: bool = False,
        seed: int = 0,
        cachable: bool = True,
    ):
        super().__init__(cachable=cachable)
        if basis not in ["sphere", "cube", "grid"]:
            raise ValueError(f"Unknown BPS basis type: {basis}. Choose from 'sphere', 'cube', 'grid'.")

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
            self.basis = generate_random_basis(self.num_points, radius=box_size, seed=seed)
        elif self.basis_type == "cube":
            self.basis = np.random.default_rng(seed).random((self.num_points, 3))
            self.basis = box_size * (self.basis - 0.5)
        elif self.basis_type == "grid":
            # self.basis = create_grid_points_from_bounds(-0.55, 0.55, res=self.res)
            grid_min: tuple[float, float, float] = (-0.5, -0.5, -0.5)
            grid_max: tuple[float, float, float] = (0.5, 0.5, 0.5)
            grid_shape: tuple[int, int, int] = (self.res, self.res, self.res)
            self.basis = box_size * make_3d_grid(grid_min, grid_max, grid_shape).numpy()

    def apply(self, data, key):
        inputs = data.get("inputs")
        normals = data.get("inputs.normals")
        if not isinstance(inputs, np.ndarray):
            raise ValueError("BPS requires 'inputs' to be a numpy array.")
        basis = self.basis.astype(inputs.dtype, copy=False)

        if self.method == "kdtree":
            kdtree = KDTree(inputs)
            input_dist, input_index = kdtree.query(basis, k=1, eps=0)
        elif self.method == "skdtree":
            kdtree = SKDTree(inputs)
            input_dist, input_index = kdtree.query(basis, workers=1)
        elif self.method == "balltree":
            nn = NearestNeighbors(n_neighbors=1, leaf_size=100, algorithm="ball_tree").fit(inputs)
            input_dist, input_index = nn.kneighbors(basis)
            input_dist = input_dist.squeeze()
            input_index = input_index.squeeze()
        else:
            raise ValueError(f"Unknown BPS method: {self.method}. Choose from 'kdtree', 'skdtree', 'balltree'.")

        input_feature = list()
        for f in self.feature:
            if f == "closest":
                input_feature.append(inputs[input_index])
            elif f == "basis":
                input_feature.append(basis)
            elif f == "delta":
                input_feature.append(inputs[input_index] - basis)
            elif f == "distance":
                input_feature.append(np.expand_dims(input_dist, axis=1))
            else:
                raise ValueError(f"Unknown BPS feature type: {f}. Choose from 'closest', 'basis', 'delta', 'distance'.")
        input_feature = np.concatenate(input_feature, axis=1)

        if self.basis_type == "grid" and input_feature is not None:
            input_feature = input_feature.reshape((self.res, self.res, self.res, -1))

        data["bps.inputs"] = inputs
        data["bps.basis"] = basis
        data["inputs"] = input_feature.squeeze() if self.squeeze else input_feature
        if isinstance(normals, np.ndarray):
            data["inputs.normals"] = normals[input_index]
        data["inputs.bps_index"] = input_index
        data["inputs.bps_dist"] = input_dist
        return data


class Scale(Transform):
    @get_args()
    def __init__(
        self,
        axes: str = "xyz",
        amount: float | tuple[float, float] | list[float] | None = None,
        random: bool = False,
        from_inputs: bool = False,
        multiplier: float | None = None,
    ):
        super().__init__()

        self.axes = axes
        self.amount = amount
        self.random = random
        self.from_inputs = from_inputs
        self.multiplier = multiplier

    def apply(self, data, key):
        points = data.get("points")
        inputs = data.get("inputs")
        depth = data.get("inputs.depth")
        pcd = data.get("pointcloud")
        mesh_vertices = data.get("mesh.vertices")
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
                    scale_array = np.asarray(scale, dtype=np.float32).reshape(3)
                    if self.multiplier is None:  # Undistort
                        scale_x = scale_array[0] / scale_array[1]
                        scale_y = 1
                        scale_z = scale_array[2] / scale_array[1]
                    else:
                        scaled = float(self.multiplier) * scale_array
                        scale_x = scaled[0]
                        scale_y = scaled[1]
                        scale_z = scaled[2]
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
        data["scale" if "scale" in data else "inputs.scale"] = scale

        for k, v in zip(
            ["inputs", "inputs.depth", "points", "pointcloud", "mesh.vertices", "bbox", "partnet.points"],
            [inputs, depth, points, pcd, mesh_vertices, bbox, partnet],
            strict=False,
        ):
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3:
                data[k] = (v * scale).astype(v.dtype)

        extrinsic = data.get("inputs.extrinsic")
        if extrinsic is not None:
            data["inputs.extrinsic"] = extrinsic @ np.diag([*(1 / scale), 1])

        if self.from_inputs:
            data.pop("scale" if "scale" in data else "inputs.scale", None)

        return data


class PointcloudFromMesh(Transform):
    @get_args()
    def __init__(self, num_points: int = int(1e5)):
        super().__init__(apply_to="mesh.vertices", cachable=True)

        self.num_points = num_points

    def apply(self, data, key):
        vertices = _to_numpy_array(data["mesh.vertices"])
        triangles = _to_numpy_array(data["mesh.triangles"])

        mesh = Trimesh(vertices, triangles, process=False, validate=False)
        points = np.asarray(mesh.sample(self.num_points))

        data["pointcloud"] = points.copy()
        return data


class PointsFromMesh(Transform):
    @get_args()
    def __init__(
        self,
        padding: float = 0.1,
        sigmas: list[float] | tuple[float, ...] | None = None,
        spheres: list[float] | tuple[float, ...] | None = None,
        points: np.ndarray | None = None,
        num_points: int = int(1e5),
        precision: int = 16,
        cache: bool = True,
    ):
        super().__init__(apply_to="mesh.vertices", cachable=cache)

        self.padding = padding
        self.sigmas = sigmas
        self.spheres = spheres
        self.points = points
        self.num_points = num_points
        self.precision = precision

    def apply(self, data, key):
        vertices = data[key]
        triangles = data["mesh.triangles"]

        mesh = Trimesh(vertices, triangles, process=False, validate=False)
        points = self.points

        if points is None:
            boxsize = 1 + self.padding
            points_uniform = np.random.rand(self.num_points, 3)
            points = [boxsize * (points_uniform - 0.5)]

            if self.sigmas:
                num_points = self.num_points // len(self.sigmas)
                for sigma in self.sigmas:
                    noise = sigma * np.random.standard_normal((num_points, 3))
                    points.append(mesh.sample(num_points) + noise)

            if self.spheres:
                for sphere in self.spheres:
                    points.append(generate_random_basis(self.num_points, radius=sphere, seed=None))

            points = np.concatenate(points, axis=0)

        occupancy = check_mesh_contains(mesh, points).astype(bool)
        points = points.astype(resolve_dtype(self.precision))

        data["points"] = points.copy()
        data["points.occ"] = occupancy.copy()
        return data


class PointsFromPointcloud(Transform):
    def __init__(self, append: bool = False):
        super().__init__(apply_to="pointcloud", cachable=True)
        self.append = append

    def apply(self, data, key):
        pointcloud = data[key]
        if "points" in data and "points.occ" in data and self.append:
            data["points"] = np.concatenate([data["points"], pointcloud], axis=0)
            data["points.occ"] = np.concatenate([data["points.occ"], np.ones(len(pointcloud), dtype=bool)], axis=0)
            return data

        data["points"] = pointcloud
        data["points.occ"] = np.ones(len(pointcloud), dtype=bool)
        return data


class CropPoints(Transform):
    @get_args()
    def __init__(
        self,
        mode: Literal["cube", "sphere", "frustum"] = "cube",
        padding: float = 0.1,
        scale_factor: float = 1.0,
        cache: bool = False,
    ):
        super().__init__(apply_to="points", cachable=cache)
        self.padding = padding
        self.scale_factor = scale_factor
        self.mode = mode

    def apply(self, data, key):
        points = data[key]
        occ = data[f"{key}.occ"]
        labels = data.get(f"{key}.labels")

        bound = (0.5 + self.padding / 2.0) * self.scale_factor

        if self.mode == "cube":
            mask = np.all(np.abs(points) <= bound, axis=1)
        elif self.mode == "sphere":
            mask = np.einsum("ij,ij->i", points, points) <= (bound * bound)
        elif self.mode == "frustum":
            in_frustum = cast(Any, is_in_frustum)
            mask = in_frustum(
                points=points,
                intrinsic=data["inputs.intrinsic"],
                extrinsic=data["inputs.extrinsic"],
                width=data["inputs.width"],
                height=data["inputs.height"],
                near=data.get("inputs.near", 0.2),
                far=data.get("inputs.far", 2.4),
            )
        else:
            raise ValueError(f"Unknown CropPoints mode: {self.mode}")

        if mask.all():
            return data

        points = points[mask]
        occ = occ[mask]
        if isinstance(labels, np.ndarray):
            labels = labels[mask]

        data[key] = points
        data[f"{key}.occ"] = occ
        if isinstance(labels, np.ndarray):
            data[f"{key}.labels"] = labels
        return data


class Visualize(Transform):
    def __init__(
        self,
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
        scale_factor: float = 1.0,
        n_calls: int | None = None,
        frame: str = "world",
        convention: str = "opencv",
        frame_size: float = 0.5,
    ):
        super().__init__()

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
        self.scale_factor = scale_factor
        self.n_calls = n_calls
        self.call_counter = 0
        self.frame = frame
        self.convention = convention
        self.frame_size = frame_size
        self.unnormalize = Compose(
            [
                NormalizeImage(mean=np.zeros(3), std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                NormalizeImage(mean=[-0.485, -0.456, -0.406], std=np.ones(3)),
            ]
        )

    def apply(self, data, key):
        if self.n_calls:
            self.call_counter += 1
            if self.call_counter > self.n_calls:
                return data
        geometries = list()
        image = None
        depth_map = None
        normal_map = None
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
                lines = list(
                    zip(np.arange(len(index)).astype(int), len(index) + np.arange(len(basis)).astype(int), strict=False)
                )
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
                selection = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs[np.unique(index)]))
                selection.paint_uniform_color((0, 0, 0))
                inputs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs))
                geometries.extend([basis_points, inputs, selection, line_set])
            else:
                inputs = data["inputs"]
                if inputs.ndim == 3 and inputs.shape[0] == 3:
                    image = ToPILImage()(self.unnormalize(torch.from_numpy(inputs)))
                elif inputs.ndim == 3 or (inputs.ndim == 2 and inputs.shape[1] == 3):
                    normals = data.get("inputs.normals")
                    if inputs.ndim == 3:
                        if normals is not None and len(normals.shape) == 4:
                            normals = normals.reshape(-1, 3)
                            normals = normals[inputs.ravel() == 1]

                        min_bound = -0.5 - self.padding / 2
                        max_bound = 0.5 + self.padding / 2
                        res = inputs.shape[-1]
                        voxel_size = (max_bound - min_bound) / res
                        grid = make_3d_grid(min_bound + voxel_size / 2, max_bound - voxel_size / 2, res).numpy()
                        inputs = grid[inputs.ravel() == 1]

                    inputs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs))
                    if normals is not None:
                        inputs.normals = o3d.utility.Vector3dVector(normals)

                    colors = data.get("inputs.colors")
                    labels = data.get("inputs.labels")
                    if colors is not None:
                        inputs.colors = o3d.utility.Vector3dVector(colors)
                    elif isinstance(labels, np.ndarray) and len(labels) == len(inputs.points):
                        colors = np.array([[0.9, 0.9, 0.9], *list(PLOTLY_COLORS)])[labels]
                        inputs.colors = o3d.utility.Vector3dVector(colors)
                    elif inputs.normals:
                        colors = np.asarray(inputs.normals) / 2 + 0.5
                        inputs.colors = o3d.utility.Vector3dVector(colors)
                    geometries.append(inputs)

                    if self.show_cam:
                        if "inputs.intrinsic" in data:
                            intrinsic = data["inputs.intrinsic"]
                            extrinsic = data.get("inputs.extrinsic", np.eye(4))
                            width = data.get("inputs.width", int(intrinsic[0, 2] * 2))
                            height = data.get("inputs.height", int(intrinsic[1, 2] * 2))

                            points = np.asarray(inputs.points)
                            points = apply_trafo(points, extrinsic)
                            depth = points_to_depth(points, intrinsic, width, height)
                            image = depth_to_image(_to_numpy_array(depth))
                            if colors is not None:
                                u, v, mask = points_to_uv(points, intrinsic, width=width, height=height)
                                image = np.asarray(image).copy()
                                image[v, u] = colors[mask] * 255
                                image = Image.fromarray(image.astype(np.uint8))
                        else:
                            logger.warning("Cannot project inputs w/o intrinsic.")
                elif inputs.ndim == 2:
                    image = depth_to_image(inputs)

                    if "inputs.intrinsic" in data:
                        intrinsic = data.get("inputs.intrinsic")
                        extrinsic = data.get("inputs.extrinsic", np.eye(4))
                        colors = data.get("inputs.image")
                        normals = data.get("inputs.normals")
                        if colors is not None:
                            if colors.shape[0] == 3:
                                colors = colors.transpose(1, 2, 0)
                            colors = normalize(colors)
                            colors = colors[inputs > 0]
                        elif normals is not None and normals.shape[:2] == inputs.shape:
                            if normals.min() < 0:
                                normals = normals / 2 + 0.5
                            if normals.max() <= 1:
                                normals = normals * 255
                            image = Image.fromarray(normals.astype(np.uint8))
                            normals = normals[inputs > 0]

                        if intrinsic is not None:
                            inputs_points = depth_to_points(inputs, intrinsic)
                            inputs_points = apply_trafo(inputs_points, inv_trafo(extrinsic))
                            inputs_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs_points))
                            if normals is not None:
                                inputs_pcd.normals = o3d.utility.Vector3dVector((normals / 255) * 2 - 1)
                                inputs_pcd.colors = o3d.utility.Vector3dVector(normals / 255)
                            if colors is not None:
                                inputs_pcd.colors = o3d.utility.Vector3dVector(colors)
                            geometries.append(inputs_pcd)
                else:
                    raise ValueError(f"Unknown inputs shape: {inputs.shape}")

            _image = data.get("inputs.image", image)
            if isinstance(_image, np.ndarray) and _image.shape[0] == 3:
                image = ToPILImage()(self.unnormalize(torch.from_numpy(_image)))

        if self.show_occupancy or self.show_points:
            if "points" in data and "points.occ" in data:
                points_occ = data["points.occ"]
                indices = (
                    points_occ <= self.threshold if self.sdf else ((points_occ >= self.threshold) & (points_occ <= 1))
                )
                occ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["points"][indices]))
                labels = data.get("points.labels")
                if labels is None:
                    occ = occ.paint_uniform_color((0.7, 0.7, 0.7))
                    if points_occ.max() == 2:
                        indices = points_occ == 2
                        uncertain = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["points"][indices]))
                        uncertain.paint_uniform_color((171 / 255, 99 / 255, 250 / 255))
                        geometries.append(uncertain)
                else:
                    colors = np.array([[0.8, 0.8, 0.8], *list(np.clip(PLOTLY_COLORS - 0.1, 0, 1))])
                    occ.colors = o3d.utility.Vector3dVector(colors[labels[indices]])
                if self.show_points:
                    outside = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["points"][~indices]))
                    outside.paint_uniform_color((0.9, 0.9, 0.9))
                    geometries.append(outside)
                if self.show_occupancy:
                    geometries.append(occ)
            else:
                logger.warning("No points or occupancy data available for visualization.")

        if self.show_frame:
            frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=self.frame_size)
            geometries.append(frame)

        if self.show_box:
            bound = (0.5 + self.padding / 2) * self.scale_factor
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-bound,) * 3, max_bound=(bound,) * 3)
            box.color = np.zeros(3)
            geometries.append(box)

        if self.show_bbox:
            bbox = data.get("bbox")
            if bbox is not None:
                bbox_points = o3d.utility.Vector3dVector(bbox)
                bbox_pcd = o3d.geometry.PointCloud(bbox_points)
                bbox_pcd.paint_uniform_color((1, 0, 0))
                geometries.append(bbox_pcd)

                lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
                colors = [[1, 0, 0] for _ in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = bbox_points
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                geometries.append(line_set)

        if self.show_cam:
            intrinsic = data.get("inputs.intrinsic", np.eye(3))
            width = data.get("inputs.width")
            if width is None:
                if image is not None:
                    width = image.size[0]
                elif intrinsic[0, 2]:
                    width = 2 * intrinsic[0, 2]
                else:
                    width = 640
            height = data.get("inputs.height")
            if height is None:
                if image is not None:
                    height = image.size[1]
                elif intrinsic[1, 2]:
                    height = 2 * intrinsic[1, 2]
                else:
                    height = 480

            if np.all(intrinsic == np.eye(3)):
                intrinsic[0, 2] = width / 2
                intrinsic[1, 2] = height / 2
                intrinsic[0, 0] = intrinsic[1, 1] = intrinsic[0, 2] * 2

            extrinsic = data.get("inputs.extrinsic", np.eye(4))

            geometries.extend(
                draw_camera(
                    intrinsic,
                    convert_extrinsic(extrinsic, self.convention, "opencv"),
                    width,
                    height,
                    scale=0.5 if image is None else 1,
                    color=None if image is None else np.asarray(image),
                )
            )

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
                    mesh = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(data["mesh.vertices"]),
                        o3d.utility.Vector3iVector(data["mesh.triangles"]),
                    )
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
                logger.warning("Warning: No mesh data available for object", data["mesh.name"])

        if self.show_pointcloud:
            if "pointcloud" in data:
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data["pointcloud"]))
                if "pointcloud.normals" in data:
                    pcd.normals = o3d.utility.Vector3dVector(data["pointcloud.normals"])
                if "pointcloud.labels" in data:
                    pcd.colors = o3d.utility.Vector3dVector(PLOTLY_COLORS[data["pointcloud.labels"]])
                else:
                    pcd = pcd.paint_uniform_color([0, 0, 0])
                geometries.append(pcd)
            else:
                logger.warning("Warning: No pointcloud data available for object", data["pointcloud.name"])

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
            lines = list(
                zip(np.arange(len(index)).astype(int), len(index) + np.arange(len(points)).astype(int), strict=False)
            )
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            geometries.append(line_set)

        if not geometries:
            logger.warning("Warning: No data to show")

        if self.show_inputs and not self.show_cam:

            def show_image(np_or_pil_image: Image.Image | np.ndarray):
                tmp_path = tempfile.mkstemp(suffix=".PNG")[1]
                pil_image = (
                    np_or_pil_image if isinstance(np_or_pil_image, Image.Image) else ToPILImage()(np_or_pil_image)
                )
                pil_image.save(tmp_path)
                if not ImageShow.DisplayViewer().show_file(tmp_path):
                    pil_image.show()

            if image is not None:
                show_image(image)
            if depth_map is not None:
                show_image(depth_map)
            if normal_map is not None:
                show_image(normal_map)

        front = [0, -1, 0] if self.convention == "blender" else [0, 0, 1]
        up = [0, 0, 1] if self.convention == "blender" else [0, 1, 0]
        if self.convention == "opencv" and self.frame == "cam":
            front = -np.array(front)
            up = -np.array(up)
        cast(Any, o3d).visualization.draw_geometries(
            geometries,
            window_name=f"{data['index']}: {data['inputs.path']}",
            mesh_show_back_face=True,
            mesh_show_wireframe=False,
            zoom=1 if self.show_box else 0.75,
            lookat=(0, 0, 0),
            front=front,
            up=up,
        )

        return data


class KeysToKeep(Transform):
    @get_args()
    def __init__(self, keys: tuple[str, ...] | list[str] | None = None):
        super().__init__()
        self.keys: list[str] | None = None if keys is None else list(keys)

    def add_key(self, key: str):
        if self.keys is None:
            self.keys = []
        self.keys.append(key)

    def add_keys(self, keys: list[str]):
        if self.keys is None:
            self.keys = []
        self.keys.extend(keys)

    def remove_key(self, key: str):
        if self.keys is None:
            return
        self.keys.remove(key)

    def apply(self, data, key):
        keep = set(self.keys) if self.keys else None
        return filter_dict(data, keep=keep)  # intersection of keys


class CheckDtype(Transform):
    @get_args()
    def __init__(self, exclude: tuple[str, ...] | list[str] | None = None, dither: bool = False):
        super().__init__()

        self.exclude = set() if exclude is None else set(exclude)
        self.dither = dither

    def cast(self, data: Any) -> Any:
        if isinstance(data, (tuple, list)):
            return [self.cast(d) for d in data]
        if isinstance(data, np.ndarray):
            if np.issubdtype(data.dtype, np.integer):
                return data
            data32 = data.astype(np.float32)
            if data.dtype == np.float16 and self.dither:
                next_up = np.nextafter(data, np.float16(np.inf)).astype(np.float32)
                ulp = (next_up - data.astype(np.float32)).astype(np.float32)
                noise = (np.random.random(size=data.shape).astype(np.float32) - 0.5) * ulp
                return data32 + noise
            return data32
        elif isinstance(data, Path):
            return str(data)
        return data

    def apply(self, data, key):
        return {k: v if k in self.exclude else self.cast(v) for k, v in data.items()}


class Compress(Transform):
    @get_args()
    def __init__(self, dtype: Any = np.float16, packbits: bool = True):
        super().__init__()

        assert dtype in [np.float16, np.float32], f"Cannot compress to {dtype}."

        self.dtype = dtype
        self.packbits = packbits

    def compress(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            if data.dtype == np.float64:
                return data.astype(self.dtype)
            if self.packbits and data.dtype == bool:
                return np.packbits(data)
        return data

    def apply(self, data, key):
        return {k: self.compress(v) for k, v in data.items()}


class Unpack(Transform):
    def __init__(self):
        super().__init__()

    @staticmethod
    def unpack(data: Any) -> Any:
        if isinstance(data, np.ndarray) and data.dtype == np.uint8:
            return np.unpackbits(data)
        return data

    def apply(self, data, key):
        return {k: self.unpack(v) for k, v in data.items()}


class SaveData(Transform):
    @get_args()
    def __init__(
        self,
        output_dir: str | Path,
        save_inputs: bool = True,
        save_image: bool = True,
        save_depth: bool = True,
        save_normals: bool = True,
        save_colors: bool = True,
        save_pointcloud: bool = True,
        save_mesh: bool = True,
        save_points: bool = True,
        save_logits: bool = True,
        save_instseg: bool = True,
        save_cam: bool = True,
        threshold: float = 0.5,
        sdf: bool = False,
    ):
        super().__init__()

        self.output_dir = Path(output_dir)
        self.save_inputs = save_inputs
        self.save_image = save_image
        self.save_depth = save_depth
        self.save_normals = save_normals
        self.save_colors = save_colors
        self.save_pointcloud = save_pointcloud
        self.save_mesh = save_mesh
        self.save_points = save_points
        self.save_logits = save_logits
        self.save_instseg = save_instseg
        self.save_cam = save_cam
        self.threshold = threshold
        self.sdf = sdf

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.unnormalize = Compose(
            [
                NormalizeImage(mean=np.zeros(3), std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                NormalizeImage(mean=[-0.485, -0.456, -0.406], std=np.ones(3)),
            ]
        )

    @staticmethod
    def _unwrap_batch(value: Any) -> Any:
        if torch.is_tensor(value):
            tensor = value
            while tensor.ndim > 0 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            return tensor
        if isinstance(value, np.ndarray):
            array = value
            while array.ndim > 0 and array.shape[0] == 1:
                array = np.squeeze(array, axis=0)
            return array
        return value

    @staticmethod
    def _as_numpy(array: Any) -> np.ndarray | None:
        if array is None:
            return None
        array = SaveData._unwrap_batch(array)
        if torch.is_tensor(array):
            return array.detach().cpu().numpy()
        if isinstance(array, np.ndarray):
            return array
        try:
            return np.asarray(array)
        except Exception:
            return None

    @staticmethod
    def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> Tensor | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = SaveData._unwrap_batch(value)
        else:
            array = SaveData._as_numpy(value)
            if array is None:
                return None
            tensor = torch.from_numpy(array)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    @staticmethod
    def _as_scalar(value: Any) -> int | float | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            tensor = SaveData._unwrap_batch(value.detach().cpu())
            flat = tensor.reshape(-1)
            if flat.numel() == 0:
                return None
            return flat[0].item()
        if isinstance(value, np.ndarray):
            array = SaveData._unwrap_batch(value)
            flat = array.reshape(-1)
            if flat.size == 0:
                return None
            return flat[0].item()
        if isinstance(value, (int, float)):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _split_by_lengths(array: Any | None, lengths: Sequence[int] | None) -> list[Any] | None:
        if array is None or lengths is None:
            return None
        array = SaveData._unwrap_batch(array)
        source = cast(Any, array)
        lengths_np = SaveData._as_numpy(lengths)
        if lengths_np is not None:
            lengths_list = lengths_np.astype(int).tolist()
        else:
            lengths_list = list(map(int, lengths))
        offsets = np.cumsum([0, *lengths_list])
        slices: list[Any] = []
        for start, end in pairwise(offsets):
            slices.append(source[start:end])
        return slices

    @staticmethod
    def _align_chunks(value: Any, lengths: Sequence[int] | None, count: int) -> list[np.ndarray | None]:
        if value is None:
            return [None] * count
        if isinstance(value, list):
            assert len(value) == count, "Length mismatch between points and auxiliary data."
            return [SaveData._as_numpy(v) for v in value]
        value = SaveData._unwrap_batch(value)
        if lengths is not None:
            split = SaveData._split_by_lengths(value, lengths)
            assert split is not None and len(split) == count
            return [SaveData._as_numpy(v) for v in split]
        if count == 1:
            return [SaveData._as_numpy(value)]
        raise ValueError("Cannot align auxiliary data without lengths when multiple chunks exist.")

    def apply(self, data, key):
        index_value = SaveData._as_scalar(data.get("index"))
        if index_value is None:
            raise ValueError("SaveData: 'index' must be convertible to a scalar")
        index = int(index_value)
        filename = str(index).zfill(8)
        if self.save_cam:
            self._save_camera_params(data, filename)
        if self.save_inputs and "inputs" in data:
            self._save_inputs(data, filename)
        if self.save_pointcloud and "pointcloud" in data:
            self._save_pointcloud(data, filename)
        if self.save_points and "points" in data:
            self._save_points(data, filename)
        if self.save_mesh and ("mesh" in data or "mesh.vertices" in data):
            self._save_mesh(data, filename)
        if self.save_logits and "logits" in data and "points" in data:
            self._save_logits(data, filename)
        if self.save_instseg and ("inputs.logits" in data or "inputs.labels" in data):
            self._save_instseg(data, filename)
        return data

    def _save_camera_params(self, data: dict, filename: str):
        camera_data = {}

        intrinsic = self._as_numpy(data.get("inputs.intrinsic"))
        if intrinsic is not None:
            camera_data["intrinsic"] = intrinsic.tolist()

        extrinsic = self._as_numpy(data.get("inputs.extrinsic"))
        if extrinsic is not None:
            camera_data["extrinsic"] = extrinsic.tolist()

        width = self._as_scalar(data.get("inputs.width"))
        if width is not None:
            camera_data["width"] = int(width)

        height = self._as_scalar(data.get("inputs.height"))
        if height is not None:
            camera_data["height"] = int(height)

        if camera_data:
            with open(self.output_dir / f"{filename}_camera.json", "w") as f:
                json.dump(camera_data, f, indent=2)

    def _save_logits(self, data: dict, filename: str, cmap: str | None = "plasma"):
        points = self._as_numpy(data.get("points"))
        logits = self._as_tensor(data.get("logits"), dtype=torch.float32)
        if points is None or logits is None:
            logger.warning("SaveData: Missing points or logits, skipping logits export.")
            return
        occ = np.atleast_2d(logits.sigmoid().cpu().numpy())
        for i, o in enumerate(occ):
            idx = o <= self.threshold if self.sdf else ((o >= self.threshold) & (o <= 1))
            pts = points[idx]
            if len(pts) == 0:
                continue
            pcd = PointCloud(pts)

            if self.save_colors:
                o_norm = normalize(o[idx])
                if cmap is None:
                    color = ((o_norm[:, None] + 0.5) * PLOTLY_COLORS[i]).clip(0, 1)
                else:
                    colormap = plt.get_cmap(cmap)
                    color = np.asarray(colormap(o_norm))[:, :3]
                pcd.colors = (255 * np.c_[color, o_norm]).astype(np.uint8)

            filename_i = f"{filename}_logits.ply"
            if i > 0:
                filename_i = f"{filename}_logits_{i}.ply"
            try:
                pcd.export(str(self.output_dir / filename_i))
            except ValueError:
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
                pcd.colors = o3d.utility.Vector3dVector(color)
                o3d.io.write_point_cloud(str(self.output_dir / filename_i), pcd)

    def _labels_to_map(self, data: dict) -> np.ndarray:
        gt = self._as_numpy(data.get("inputs.labels"))
        if gt is None:
            raise ValueError("SaveData: Missing inputs.labels for instance segmentation export")
        gt = gt.astype(np.uint16)
        if gt.ndim == 2:
            return gt

        width = self._as_scalar(data.get("inputs.width"))
        height = self._as_scalar(data.get("inputs.height"))
        if width is None or height is None:
            raise ValueError("SaveData: inputs.width/height required for label export")
        width = int(width)
        height = int(height)
        if gt.ndim == 1 and gt.size == width * height:
            return gt.reshape(height, width)

        inputs = self._as_tensor(data.get("inputs"), dtype=torch.float32)
        intrinsic = self._as_tensor(data.get("inputs.intrinsic"), dtype=torch.float32)
        extrinsic = self._as_tensor(data.get("inputs.extrinsic"), dtype=torch.float32)
        if inputs is None or intrinsic is None:
            raise ValueError("SaveData: inputs/intrinsic required for label projection")
        if extrinsic is None:
            extrinsic = torch.eye(4)
        u, v, mask = points_to_uv(inputs, intrinsic, extrinsic, width, height)
        out = np.zeros((height, width), dtype=np.uint16)
        u_idx = np.asarray(SaveData._as_numpy(u), dtype=np.int64).reshape(-1)
        v_idx = np.asarray(SaveData._as_numpy(v), dtype=np.int64).reshape(-1)
        mask_idx = np.asarray(SaveData._as_numpy(mask), dtype=bool).reshape(-1)
        out[v_idx, u_idx] = gt[mask_idx]
        return out

    def _preds_to_map(self, data: dict) -> np.ndarray:
        in_logits = self._as_tensor(data.get("inputs.logits"), dtype=torch.float32)
        inputs = self._as_tensor(data.get("inputs"), dtype=torch.float32)
        intrinsic = self._as_tensor(data.get("inputs.intrinsic"), dtype=torch.float32)
        extrinsic = self._as_tensor(data.get("inputs.extrinsic"), dtype=torch.float32)
        width = self._as_scalar(data.get("inputs.width"))
        height = self._as_scalar(data.get("inputs.height"))
        if in_logits is None or inputs is None or intrinsic is None:
            raise ValueError("SaveData: Missing inputs/logits/intrinsics for prediction export")
        if extrinsic is None:
            extrinsic = torch.eye(4)
        if width is None or height is None:
            raise ValueError("SaveData: inputs.width/height required for prediction export")
        width = int(width)
        height = int(height)

        if in_logits.ndim < 2 or in_logits.shape[0] == 0:
            return np.zeros((height, width), dtype=np.uint16)
        probs = in_logits.sigmoid()
        max_prob, inst = probs.max(dim=0)
        labels_pred = torch.where(max_prob > self.threshold, inst + 1, torch.zeros_like(inst))

        u, v, mask = points_to_uv(inputs, intrinsic, extrinsic, width, height)
        label_map = np.zeros((height, width), dtype=np.uint16)
        u_idx = np.asarray(SaveData._as_numpy(u), dtype=np.int64).reshape(-1)
        v_idx = np.asarray(SaveData._as_numpy(v), dtype=np.int64).reshape(-1)
        mask_idx = np.asarray(SaveData._as_numpy(mask), dtype=bool).reshape(-1)
        labels_np = labels_pred.detach().cpu().numpy()
        label_map[v_idx, u_idx] = labels_np[mask_idx]
        return label_map

    def _save_seg_color(self, seg: np.ndarray, filename: str):
        palette = np.vstack((np.zeros((1, 3)), np.asarray(PLOTLY_COLORS)))
        color_img = (255.0 * palette[seg]).astype(np.uint8)
        Image.fromarray(color_img).save(self.output_dir / filename)

    def _save_instseg(self, data: dict, filename: str):
        if "inputs.labels" in data:
            self._save_seg_color(self._labels_to_map(data), f"{filename}_instseg.png")
        if "inputs.logits" in data:
            self._save_seg_color(self._preds_to_map(data), f"{filename}_pred_instseg.png")

    def _save_inputs(self, data: dict, filename: str):
        inputs = self._unwrap_batch(data["inputs"])
        inputs_np = self._as_numpy(inputs)
        if isinstance(inputs, (torch.Tensor, np.ndarray)) and inputs.ndim == 3 and inputs.shape[0] == 3:
            self._save_image(inputs, filename + "_rgb.png")
            if self.save_normals and "inputs.normals" in data:
                normals = self._as_numpy(data.get("inputs.normals"))
                if normals is not None:
                    normals = normals.copy()
                    if normals.min() < 0:
                        normals = normals / 2 + 0.5
                    if normals.max() <= 1:
                        normals = normals * 255
                    self._save_image(normals, filename + "_normals.png")
        elif "inputs.image" in data:
            image = self._unwrap_batch(data["inputs.image"])
            self._save_image(image, filename + "_rgb.png")
        if self.save_depth and "inputs.depth" in data:
            depth = self._unwrap_batch(data["inputs.depth"])
            self._save_depth(depth, filename + "_depth.png")

        if isinstance(inputs_np, np.ndarray) and inputs_np.ndim == 2 and inputs_np.shape[1] == 3:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs_np))
            if self.save_normals and "inputs.normals" in data:
                normals = self._as_numpy(data.get("inputs.normals"))
                if normals is not None and len(normals) == len(inputs_np):
                    pcd.normals = o3d.utility.Vector3dVector(normals)
            if self.save_colors:
                if "inputs.labels" in data:
                    labels = self._as_numpy(data.get("inputs.labels"))
                    if labels is not None and len(labels) == len(inputs_np):
                        colors = np.array([[0.9, 0.9, 0.9], *list(PLOTLY_COLORS)])[labels.astype(int)]
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                elif "inputs.colors" in data:
                    colors = self._as_numpy(data.get("inputs.colors"))
                    if colors is not None and len(colors) == len(inputs_np):
                        pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(self.output_dir / f"{filename}_inputs.ply"), pcd)

    def _save_image(self, image: np.ndarray | Tensor | Image.Image, filename: str):
        if isinstance(image, Image.Image):
            pil_image = image
        elif torch.is_tensor(image) or (isinstance(image, np.ndarray) and image.ndim > 0 and image.shape[0] == 3):
            image_tensor = torch.from_numpy(image) if isinstance(image, np.ndarray) else image
            pil_image = ToPILImage()(self.unnormalize(image_tensor))
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            raise TypeError(f"Cannot save image of type {type(image)}.")
        pil_image.save(self.output_dir / filename)

    def _save_depth(self, depth: np.ndarray | Tensor, filename: str, cmap: str = "Greys"):
        if torch.is_tensor(depth):
            depth = depth.cpu().numpy()
        depth_to_image(depth, cmap).save(self.output_dir / filename)

    @staticmethod
    def _slice_pointcloud_attr(value: Any, index: int) -> Any:
        """Return the per-pointcloud entry when auxiliary data is batched."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value[index] if len(value) > index else value
        if torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] > index:
                return value[index]
            return value
        if isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] > index:
                return value[index]
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            try:
                return value[index]
            except (IndexError, TypeError):
                return value
        return value

    def _save_pointcloud(self, data: dict, filename: str, index: int | None = None):
        pcds = data["pointcloud"]
        if isinstance(pcds, list):
            for i in range(len(pcds)):
                d = {k: self._slice_pointcloud_attr(v, i) for k, v in data.items() if k.startswith("pointcloud")}
                self._save_pointcloud(d, filename, index=i)
            return

        pcd = data["pointcloud"]
        if not isinstance(pcd, o3d.geometry.PointCloud):
            pts = self._as_numpy(pcd)
            if pts is None:
                logger.warning("SaveData: pointcloud is None, skipping save.")
                return
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

        if self.save_colors:
            if "pointcloud.labels" in data:
                labels = self._as_numpy(data.get("pointcloud.labels"))
                if labels is not None:
                    pcd.colors = o3d.utility.Vector3dVector(PLOTLY_COLORS[labels.astype(int)])
            elif "pointcloud.colors" in data:
                colors = self._as_numpy(data.get("pointcloud.colors"))
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors)

        if self.save_normals and "pointcloud.normals" in data:
            normals = self._as_numpy(data.get("pointcloud.normals"))
            if normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(normals)

        filepath = self.output_dir / f"{filename}_pointcloud.ply"
        if index is not None:
            filepath = self.output_dir / f"{filename}_pointcloud_{index}.ply"
            if self.save_colors and not len(pcd.colors):
                pcd.paint_uniform_color(PLOTLY_COLORS[index])
        o3d.io.write_point_cloud(str(filepath), pcd)

    def _save_points(self, data: dict, filename: str):
        def _write_single(
            points_arr: np.ndarray | None,
            occ_arr: np.ndarray | None,
            labels_arr: np.ndarray | None,
            suffix: str | None = None,
        ) -> None:
            if points_arr is None or occ_arr is None:
                logger.warning("SaveData: Missing points or occupancy for suffix '%s'", suffix)
                return

            indices = occ_arr <= self.threshold if self.sdf else ((occ_arr >= self.threshold) & (occ_arr <= 1))
            pts_occ = points_arr[indices]
            pts_free = points_arr[~indices]

            occ_name = f"{filename}_occ{suffix}.ply" if suffix else f"{filename}_occ.ply"
            free_name = f"{filename}_free{suffix}.ply" if suffix else f"{filename}_free.ply"

            if len(pts_occ) > 0:
                pcd_occ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_occ))
                if self.save_colors and labels_arr is not None:
                    colors = np.array([[0.8, 0.8, 0.8], *list(np.clip(PLOTLY_COLORS - 0.1, 0, 1))])
                    if len(labels_arr) == len(points_arr):
                        pcd_occ.colors = o3d.utility.Vector3dVector(colors[labels_arr[indices]])
                o3d.io.write_point_cloud(str(self.output_dir / occ_name), pcd_occ)

            if len(pts_free) > 0:
                pcd_free = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_free))
                o3d.io.write_point_cloud(str(self.output_dir / free_name), pcd_free)

        points = data["points"]
        if not isinstance(points, list):
            points = self._unwrap_batch(points)
        lengths = None if isinstance(points, list) else data.get("points.lengths")
        point_chunks: list[np.ndarray | None]
        if isinstance(points, list):
            point_chunks = [self._as_numpy(p) for p in points]
        elif lengths is not None:
            split = self._split_by_lengths(points, lengths)
            point_chunks = [self._as_numpy(p) for p in split] if split is not None else [self._as_numpy(points)]
        else:
            point_chunks = [self._as_numpy(points)]

        occ_chunks = self._align_chunks(data["points.occ"], lengths, len(point_chunks))
        label_chunks = self._align_chunks(data.get("points.labels"), lengths, len(point_chunks))

        for idx, (pts, occ_part, lbl_part) in enumerate(zip(point_chunks, occ_chunks, label_chunks, strict=False)):
            suffix = f"_{idx}" if len(point_chunks) > 1 else None
            _write_single(pts, occ_part, lbl_part, suffix=suffix)

    def _save_mesh(self, data: dict, filename: str, index: int | None = None, suffix: Literal[".obj", ".glb"] = ".obj"):
        meshes = data.get("mesh")
        if isinstance(meshes, list):
            for i, mesh in enumerate(meshes):
                self._save_mesh({"mesh": mesh}, filename, index=i, suffix=suffix)
            return

        if "mesh" in data:
            mesh = data["mesh"]
            if isinstance(mesh, Trimesh):
                data["mesh.vertices"] = mesh.vertices
                data["mesh.triangles"] = mesh.faces
                # if mesh.vertex_normals is not None:
                #    data["mesh.normals"] = mesh.vertex_normals
                # if mesh.visual.vertex_colors is not None:
                #    data["mesh.colors"] = mesh.visual.vertex_colors

        vertices = self._as_numpy(data.get("mesh.vertices"))
        triangles = self._as_numpy(data.get("mesh.triangles"))
        if vertices is None:
            logger.warning("SaveData: mesh vertices missing, skipping mesh export")
            return

        if triangles is None:
            mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
        else:
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
            )
            if self.save_normals and "mesh.normals" in data:
                normals = self._as_numpy(data.get("mesh.normals"))
                if normals is not None:
                    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
            if self.save_colors and "mesh.colors" in data:
                colors = self._as_numpy(data.get("mesh.colors"))
                if colors is not None:
                    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            if self.save_colors and "mesh.textures" in data and "mesh.uvs" in data and "mesh.ids" in data:
                textures = list()
                for texture in data["mesh.textures"]:
                    if isinstance(texture, o3d.geometry.Image):
                        textures.append(texture)
                        continue
                    tex_np = self._as_numpy(texture)
                    if tex_np is not None:
                        textures.append(o3d.geometry.Image(tex_np))
                if textures:
                    mesh.textures = textures
                uvs = self._as_numpy(data["mesh.uvs"])
                if uvs is not None:
                    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
                ids = self._as_numpy(data["mesh.ids"])
                if ids is not None:
                    mesh.triangle_material_ids = o3d.utility.IntVector(ids.astype(int))
            if self.save_normals:
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()
                if not mesh.has_triangle_normals():
                    mesh.compute_triangle_normals()

        filepath = self.output_dir / f"{filename}_mesh{suffix}"
        if index is not None:
            filepath = self.output_dir / f"{filename}_mesh_{index}{suffix}"
        if index is not None and self.save_colors and not len(mesh.vertex_colors):
            mesh.paint_uniform_color(PLOTLY_COLORS[index])
        o3d.io.write_triangle_mesh(str(filepath), mesh)


class SplitData(Transform):
    @get_args()
    def __init__(
        self,
        split_text: bool = True,
        split_mesh: bool = True,
        split_pointcloud: bool = True,
        split_points: bool = True,
        apply_to: str | list[str] | tuple[str, ...] | None = None,
    ):
        super().__init__(apply_to=apply_to)
        self.split_text = split_text
        self.split_mesh = split_mesh
        self.split_pointcloud = split_pointcloud
        self.split_points = split_points

    def apply(self, data: dict[str | None, Any], key: str | None):
        if self.split_text:
            data["category.name"] = data["category.name"].split("_")
            # data["category.index"] = sum([CATEGORIES_MAP[id] for id in item["category.id"]])
            data["category.id"] = data["category.id"].split("_")
            data["inputs.name"] = data["inputs.name"].split("_")

        if self.split_mesh:
            data["mesh"] = list()
            v = data["mesh.vertices"]
            f = data["mesh.triangles"]
            if "mesh.lengths" in data:
                mesh_lengths = data["mesh.lengths"]
                v_lens = mesh_lengths["vertices"]
                f_lens = mesh_lengths["triangles"]
            elif "mesh.num_vertices" in data and "mesh.num_triangles" in data:
                v_lens = data["mesh.num_vertices"]
                f_lens = data["mesh.num_triangles"]
            else:
                raise ValueError("Cannot split mesh without lengths or num_vertices/num_faces.")
            v_offsets = np.cumsum([0, *v_lens[:-1]]).tolist()
            f_offsets = np.cumsum([0, *f_lens[:-1]]).tolist()
            for vi, vf, vo, fo in zip(v_lens, f_lens, v_offsets, f_offsets, strict=False):
                v_slice = v[vo : vo + vi].copy()
                f_slice = f[fo : fo + vf].copy() - vo
                data["mesh"].append(Trimesh(vertices=v_slice, faces=f_slice))

        if self.split_pointcloud:
            pcd = data["pointcloud"]
            normals = data.get("pointcloud.normals")
            pcd_lengths = data["pointcloud.lengths"]
            offsets = np.cumsum([0, *list(map(int, pcd_lengths))[:-1]]).tolist()
            pcs = [pcd[offset : offset + length] for offset, length in zip(offsets, pcd_lengths, strict=False)]
            data["pointcloud"] = pcs
            if isinstance(normals, np.ndarray) and len(normals) == len(pcd):
                data["pointcloud.normals"] = [
                    normals[offset : offset + length] for offset, length in zip(offsets, pcd_lengths, strict=False)
                ]

        if self.split_points and "points" in data and not isinstance(data["points"], list):
            points = data["points"]
            lengths = data.get("points.lengths")
            if lengths is None:
                raise ValueError("Cannot split points without points.lengths.")

            def _as_list(seq):
                if isinstance(seq, np.ndarray):
                    return seq.tolist()
                return list(seq)

            lengths = [int(length) for length in _as_list(lengths)]
            total = int(np.sum(lengths))
            if hasattr(points, "shape") and points.shape[0] != total:
                raise ValueError(f"Sum of points.lengths ({total}) != len(points) ({len(points)})")

            def _slice(arr, start, end):
                if arr is None:
                    return None
                if torch.is_tensor(arr):
                    return arr[start:end].clone()
                if isinstance(arr, np.ndarray):
                    return arr[start:end].copy()
                return arr[start:end]

            offsets = np.cumsum([0, *lengths])
            data["points"] = [_slice(points, start, end) for start, end in pairwise(offsets)]

            occ = data.get("points.occ")
            if occ is not None and not isinstance(occ, list) and len(occ) == total:
                data["points.occ"] = [_slice(occ, start, end) for start, end in pairwise(offsets)]

            labels = data.get("points.labels")
            if labels is not None and not isinstance(labels, list) and len(labels) == total:
                data["points.labels"] = [_slice(labels, start, end) for start, end in pairwise(offsets)]
            points_chunks = cast(list[Any], data["points"])
            data["points.lengths"] = [len(chunk) for chunk in points_chunks]
        elif self.split_points and "points" in data and isinstance(data["points"], list):
            points_chunks = cast(list[Any], data["points"])
            data["points.lengths"] = [len(chunk) for chunk in points_chunks]

        return data


__all__ = [
    "BPS",
    "AddGaussianNoise",
    "Affine",
    "AngleOfIncidenceRemoval",
    "ApplyPose",
    "AxesCutPointcloud",
    "BoundingBox",
    "BoundingBoxJitter",
    "CheckDtype",
    "Compress",
    "CropPointcloud",
    "CropPointcloudWithMesh",
    "CropPoints",
    "DepthLikePointcloud",
    "DepthToPointcloud",
    "EdgeNoise",
    "FindUncertainPoints",
    "ImageBorderNoise",
    "ImageToTensor",
    "InputsNormalsFromPointcloud",
    "KeysToKeep",
    "LoadUncertain",
    "MinMaxNumPoints",
    "Normalize",
    "NormalizeMesh",
    "NormalsCameraCosineSimilarity",
    "Permute",
    "PointcloudFromMesh",
    "PointsFromMesh",
    "PointsFromPointcloud",
    "ProcessPointcloud",
    "RandomApply",
    "RandomChoice",
    "RefinePose",
    "RefinePosePerInstance",
    "RemoveHiddenPointsFromInputs",
    "Render",
    "RenderDepthMaps",
    "RenderPointcloud",
    "Return",
    "Rotate",
    "RotateMesh",
    "RotatePointcloud",
    "SaveData",
    "Scale",
    "Scale",
    "SdfFromOcc",
    "SegmentationFromPartNet",
    "ShadingImageFromNormals",
    "SphereCutPointcloud",
    "SphereMovePointcloud",
    "SplitData",
    "SubsamplePointcloud",
    "SubsamplePoints",
    "Transform",
    "Translate",
    "Unpack",
    "Visualize",
    "VoxelizePointcloud",
    "VoxelizePoints",
    "apply_transforms",
]
