import copy
import json
import random
import time
from collections.abc import Callable, Sequence
from functools import cached_property, lru_cache, partial
from itertools import pairwise
from logging import DEBUG
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import torch
from easy_o3d.utils import convert_depth_image_to_point_cloud
from scipy.ndimage import gaussian_filter
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.transforms.v2 import functional as F
from tqdm import tqdm
from trimesh import Trimesh

from utils import (
    DEBUG_LEVEL_2,
    apply_trafo,
    depth_to_image,
    depth_to_points,
    get_rays,
    inv_trafo,
    is_in_frustum,
    load_mesh,
    measure_runtime,
    normalize,
    normalize_mesh,
    points_to_uv,
    sample_distances,
    setup_logger,
    stdout_redirected,
    subsample_indices,
)

from .coco import CocoInstanceSegmentation
from .fields import _load_pointcloud, _load_points
from .shapenet import CATEGORIES_MAP
from .transforms import Transform, apply_transforms
from .tv_transforms import CameraIntrinsic

logger = setup_logger(__name__)

# Pyright helper aliases for dynamically-typed utility decorators/functions.
_measure_runtime = cast(Any, measure_runtime)
_get_rays = cast(Any, get_rays)
_sample_distances = cast(Any, sample_distances)
_is_in_frustum = cast(Any, is_in_frustum)
_apply_transforms_3d = cast(Any, apply_transforms)
_convert_depth_image_to_point_cloud = cast(Any, convert_depth_image_to_point_cloud)
_points_to_uv = cast(Any, points_to_uv)


@lru_cache(maxsize=2048)
def _cached_load_points(path_str: str, from_hdf5: bool = False) -> tuple[np.ndarray, np.ndarray]:
    points, occ = _load_points(Path(path_str), from_hdf5=from_hdf5)
    points = np.asarray(points, dtype=np.float16)
    occ = np.asarray(occ, dtype=bool)
    points.setflags(write=False)
    occ.setflags(write=False)
    return points, occ


class TableTop(VisionDataset):
    @_measure_runtime
    def __init__(
        self,
        data_dir: Path,
        split: Literal["train", "val", "test"] = "train",
        train_ann_file: str = "coco_annotations.json",
        val_ann_file: str = "coco_annotations.json",
        test_ann_file: str = "coco_annotations.json",
        load_color: bool = True,
        load_depth: bool | Literal["kinect", "kinect_sim"] = True,
        load_normals: bool = True,
        apply_filter: Literal["edge"] | None = None,
        project: bool = True,
        data_dir_3d: Path | None = None,
        mesh_file: str | None = None,
        pcd_file: str | None = None,
        points_file: str | list[str] | None = None,
        crop_points: Literal["sphere", "cube", "frustum"] | None = None,
        sample_free_points: Literal["sphere", "cube", "frustum"] = "cube",
        points_fraction: float | Literal["auto"] | None = "auto",
        append_pcd_to_points: bool = False,
        points_grid_resolution: int = 128,
        near: float | None = None,
        far: float | None = None,
        padding: float = 0.1,
        scale_factor: float = 1.0,
        collate_3d: Literal["cat", "rand", "size", "list", "stack", "seq"] | None = None,
        stack_2d: bool = False,
        apply_pose: bool = True,
        from_hdf5: bool = False,
        cache_points: bool = False,
        transforms: Callable[..., Any] | None = None,
        transforms_3d: Transform | list[Transform] | None = None,
    ):
        super().__init__(root=data_dir, transforms=transforms)

        get_coco_ds = partial(
            CocoInstanceSegmentation,
            split=split,
            train_dir="",
            val_dir="",
            test_dir="",
            ann_dir="",
            train_ann_file=train_ann_file,
            val_ann_file=val_ann_file,
            test_ann_file=test_ann_file,
        )

        def validate_coco_ds(ds: CocoInstanceSegmentation) -> CocoInstanceSegmentation:
            ds_root = Path(ds.root)
            assert len(list((ds_root / "images").glob("*.png"))) == len(ds), f"len(images) != len(dataset): {ds_root}"
            assert len(list(ds_root.glob("*.hdf5"))) == len(ds), f"len(hdf5) != len(dataset): {ds_root}"
            return ds

        self.coco_ds: list[CocoInstanceSegmentation] = list()
        if (data_dir / split / "images").exists():
            self.coco_ds.append(
                validate_coco_ds(
                    get_coco_ds(
                        data_dir,
                        train_dir=split,
                        val_dir=split,
                        test_dir=split,
                        ann_dir=split,
                    )
                )
            )
        else:
            shards = sorted(
                shard for shard in (data_dir / split).iterdir() if shard.is_dir() and (shard / "images").exists()
            )
            for shard in tqdm(shards, desc=f"Loading {split} shards", disable=logger.isEnabledFor(DEBUG_LEVEL_2)):
                with stdout_redirected(enabled=not logger.isEnabledFor(DEBUG_LEVEL_2)):
                    self.coco_ds.append(validate_coco_ds(get_coco_ds(shard)))
        self._cumsize = np.cumsum([len(ds) for ds in self.coco_ds])

        self.name = self.__class__.__name__
        self.split = split
        self.load_color = load_color
        self.load_depth = load_depth
        self.load_normals = load_normals
        self.apply_filter = apply_filter
        self.project = project
        self.data_dir_3d = data_dir_3d
        self.mesh_file = mesh_file
        self.pcd_file = pcd_file
        self.points_file = points_file
        self.crop_points = crop_points
        self.sample_free_points = sample_free_points
        self.points_fraction = points_fraction
        self.append_pcd_to_points = append_pcd_to_points
        self.points_grid_resolution = points_grid_resolution
        self.near = near
        self.far = far
        self.padding = padding
        self.scale_factor = scale_factor
        self.collate_3d = collate_3d
        self.stack_2d = stack_2d
        self.from_hdf5 = from_hdf5
        self.cache_points = cache_points
        self.apply_pose = apply_pose
        self.transforms_3d = transforms_3d

        if not isinstance(points_file, str) and isinstance(points_file, Sequence):
            self.points_file = list(points_file)

        self.categories = list(CATEGORIES_MAP.keys())

        # Precompute sequential instance index if requested
        self._seq_indices: list[tuple[int, int, int]] | None = None
        if collate_3d == "seq":
            self._seq_indices = self._build_seq_index()

        logger.info(f"Found {len(self.coco_ds)} '{split}' shards (len={len(self)}).")

    @cached_property
    def volume(self) -> float:
        if self.sample_free_points == "cube":
            bound = (0.5 + self.padding / 2.0) * self.scale_factor
            return (2.0 * bound) ** 3
        if self.sample_free_points == "sphere":
            bound = (0.5 + self.padding / 2.0) * self.scale_factor
            return (4.0 / 3.0) * np.pi * (bound**3)
        raise ValueError("Frustum volume is not fixed.")

    @staticmethod
    def volume_frustum(intrinsic: np.ndarray, width: int, height: int, near: float, far: float) -> float:
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        k = (width * height) / (fx * fy)
        return k * (far**3 - near**3) / 3.0

    @_measure_runtime
    def _index_to_coco_ds(self, index: int) -> tuple[CocoInstanceSegmentation, int]:
        """
        Convert a global index to the corresponding COCO dataset instance using cached sizes.
        """
        ds_idx = np.searchsorted(self._cumsize, index, side="right")
        if ds_idx >= len(self.coco_ds):
            raise IndexError(f"Index {index} out of bounds: size {len(self)} with {len(self.coco_ds)} shards.")

        local_index = index - (self._cumsize[ds_idx - 1] if ds_idx > 0 else 0)
        return self.coco_ds[ds_idx], local_index

    @_measure_runtime
    def _build_seq_index(self) -> list[tuple[int, int, int]]:
        """
        Build a flat list mapping each visible instance (area > 0) to its (shard, local image, instance) triple.
        Returns:
            List of (ds_idx, local_idx, inst_idx).
        """
        seq: list[tuple[int, int, int]] = []
        total_imgs = sum(len(ds) for ds in self.coco_ds)
        total_insts = 0
        with tqdm(
            total=total_imgs, desc=f"Indexing {self.split} seq instances", disable=logger.isEnabledFor(DEBUG)
        ) as pbar:
            for ds_idx, ds in enumerate(self.coco_ds):
                n_ds = len(ds)
                for local_idx in range(n_ds):
                    item = ds[local_idx]
                    info = item.get("inputs.info", [])
                    for inst_idx, item_info in enumerate(info):
                        if item_info.get("area", 0) > 0:
                            seq.append((ds_idx, local_idx, inst_idx))
                            total_insts += 1
                    pbar.update(1)
        logger.info(f"Built seq index over {total_imgs} images: {total_insts} visible instances.")
        return seq

    def _load_data(self, data_dir: Path, index: int) -> dict[str, Any]:
        hdf5_path = data_dir / f"{index}.hdf5"
        with h5py.File(hdf5_path, "r") as hdf5_file:
            raw_json = cast(bytes, cast(Any, hdf5_file["data"])[()])
            data = json.loads(raw_json.decode("utf-8"))
            if self.load_depth:
                if self.load_depth == "kinect":
                    data["depth"] = np.asarray(hdf5_file["kinect"], dtype=np.float32)
                elif self.load_depth == "kinect_sim":
                    data["depth"] = np.asarray(hdf5_file["kinect_sim"], dtype=np.float32)
                else:
                    data["depth"] = np.asarray(hdf5_file["depth"], dtype=np.float32)
            if self.load_normals:
                data["normals"] = np.asarray(hdf5_file["normals"], dtype=np.float32)
        data["intrinsic"] = np.asarray(data["intrinsic"], dtype=np.float32)
        data["extrinsic"] = np.asarray(data["extrinsic"], dtype=np.float32)
        return data

    @_measure_runtime
    def _apply_transforms(self, item: dict[str, Any]) -> dict[str, Any]:
        target = {
            "boxes": item["inputs.boxes"],
            "masks": item["inputs.masks"],
            "labels": item["inputs.labels"],
            "intrinsic": item["inputs.intrinsic"],
        }

        image = item["inputs"]
        if self.load_depth and (self.load_color or self.project):
            image = item["inputs.image"]
            target["depth"] = tv_tensors.Mask(item["inputs"])
        if self.load_normals:
            target["normals"] = tv_tensors.Mask(torch.from_numpy(item["inputs.normals"]).permute(2, 0, 1))

        if self.transforms is None:
            return item
        image, target = self.transforms(image, target)
        height, width = image.shape[-2:]

        info = item["inputs.info"]
        for i, elem in enumerate(info):
            if "area" in elem:
                elem["area"] = target["masks"][i].sum().item()

        item["inputs"] = image.contiguous()
        if self.load_depth:
            item.pop("inputs.image", None)
            if self.load_color or self.project:
                item["inputs"] = target["depth"].contiguous().numpy()
                if self.load_color:
                    item["inputs.image"] = image.contiguous()
        if self.load_normals:
            item["inputs.normals"] = target["normals"].permute(1, 2, 0).contiguous().numpy()

        item["inputs.boxes"] = target["boxes"]
        item["inputs.masks"] = target["masks"]
        item["inputs.labels"] = target["labels"]
        item["inputs.width"] = width
        item["inputs.height"] = height
        item["inputs.intrinsic"] = target["intrinsic"]

        return item

    @staticmethod
    @_measure_runtime
    def _get_crop_mask_from_boxes(
        points: np.ndarray,
        boxes: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        valid_boxes = [box for box in boxes if box is not None]
        if not valid_boxes:
            return np.ones((len(points),), dtype=bool)

        # Ensure float32 once
        points = points.astype(np.float32, copy=False)

        # Stack box mins/maxs
        box_mins = np.stack([box[0] for box in valid_boxes], axis=0).astype(np.float32, copy=False)  # (N,3)
        box_maxs = np.stack([box[1] for box in valid_boxes], axis=0).astype(np.float32, copy=False)  # (N,3)

        # Coarse union prefilter: points outside the union cannot be inside any boxox
        union_min = box_mins.min(axis=0)
        union_max = box_maxs.max(axis=0)
        in_union = (points >= union_min).all(axis=1) & (points <= union_max).all(axis=1)

        # Start with keep-all; refine only for potentially-in-union points
        keep = np.ones((len(points),), dtype=bool)
        if not in_union.any():
            return keep

        idx = np.where(in_union)[0]
        pf = points[idx][:, None, :]  # (K,1,3)
        in_box = (pf >= box_mins[None, :, :]) & (pf <= box_maxs[None, :, :])  # (K,N,3)
        in_any = np.any(np.all(cast(Any, in_box), axis=2), axis=1)  # (K,)

        # Exclude those that are inside any box
        keep[idx] = ~in_any
        return keep

    @_measure_runtime
    def _filter_free_points_against_other_boxes(
        self,
        points_list: list[tuple[np.ndarray, np.ndarray]],
        boxes: Sequence[tuple[np.ndarray, np.ndarray] | None],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        out: list[tuple[np.ndarray, np.ndarray]] = []
        valid_idx = [k for k, box in enumerate(boxes) if box is not None]
        if not valid_idx:
            return points_list

        for i, (p, occ) in enumerate(points_list):
            if len(p) == 0 or not np.any(~occ):
                out.append((p, occ))
                continue

            other_boxs = [boxes[k] for k in valid_idx if k != i]
            if not other_boxs:
                out.append((p, occ))
                continue

            keep = np.ones((len(p),), dtype=bool)
            free_idx = np.where(~occ)[0]
            if len(free_idx) > 0:
                free_mask = self._get_crop_mask_from_boxes(p[free_idx], other_boxs)
                keep[free_idx] = free_mask

            out.append((p[keep], occ[keep]))
        return out

    @_measure_runtime
    def _sample_free_points(
        self,
        mask: np.ndarray | None = None,
        intrinsic: np.ndarray | None = None,
        extrinsic: np.ndarray | None = None,
        near: float = 0.2,
        far: float = 2.4,
        num_samples: int | None = None,
        volume: Literal["sphere", "cube", "frustum"] = "frustum",
    ) -> np.ndarray:
        n_samples = num_samples if num_samples is not None else 100_000
        if volume != "frustum" or mask is None or intrinsic is None or extrinsic is None:
            if volume == "sphere":
                bound = (0.5 + self.padding / 2.0) * self.scale_factor
                dirs = np.random.normal(size=(n_samples, 3)).astype(np.float32)
                norms = np.linalg.norm(dirs, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0  # avoid divide-by-zero
                dirs /= norms
                radii = (np.random.rand(n_samples, 1).astype(np.float32)) ** (1.0 / 3.0) * bound
                p_free = dirs * radii
            else:
                bound = (0.5 + self.padding / 2) * self.scale_factor
                p_free = np.random.uniform(-bound, bound, size=(n_samples, 3))
        else:
            ray0, ray_dirs, _u, _v = _get_rays(mask, intrinsic, extrinsic, num_samples=n_samples)
            d = _sample_distances(n_points=len(ray0), near=near, far=far, uniform_in_volume=True)
            p_free = ray0 + d * ray_dirs

        return p_free.astype(np.float32, copy=False)

    @_measure_runtime
    def _estimate_num_samples(
        self,
        num_points: int,
        boxes: Sequence[tuple[np.ndarray, np.ndarray] | None],
        intrinsic: np.ndarray | None = None,
        width: int | None = None,
        height: int | None = None,
        near: float | None = None,
        far: float | None = None,
        min_samples: int = 10_000,
        max_samples: int = 250_000,
    ) -> int:
        valid_boxes = [box for box in boxes if box is not None]
        vol_boxes = sum(np.prod(box[1] - box[0]) for box in valid_boxes)
        if self.sample_free_points == "frustum":
            if intrinsic is None or width is None or height is None or near is None or far is None:
                raise ValueError("intrinsic/width/height/near/far are required for frustum volume estimation")
            vol_domain = self.volume_frustum(intrinsic, width, height, near, far)
        else:
            vol_domain = self.volume
        vol_frac = np.clip(1.0 - vol_boxes / vol_domain, 0.0, 1.0)
        num_samples = num_points * vol_frac
        num_samples = np.clip(num_samples, min_samples, max_samples)
        return int(num_samples)

    @staticmethod
    def _edge_map(depth: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        """
        Compute a Sobel-based edge map on depth (and optionally image).
        Returns a float array in [0, 1] aligned to depth shape.
        """
        # Local imports to avoid adding top-level deps if not used
        from skimage.color import rgb2gray
        from skimage.feature import canny
        from skimage.filters import scharr

        # Depth edges (ignore strict zeros; match old behavior)
        depth_f = depth.astype(np.float32, copy=False)
        depth_edges = scharr(depth_f)
        depth_edges[depth == 0.0] = 0.0

        if image is None:
            m = float(depth_edges.max())
            return depth_edges / m if m > 0 else depth_edges

        # Image edge proxy: Sobel on grayscale
        if image.ndim == 3:
            img_f = image.astype(np.float32, copy=False)
            # Bring to [0,1] if needed (rgb2gray expects float in [0,1])
            if img_f.max() > 1.0:
                img_f = img_f / 255.0
            gray = rgb2gray(img_f)
        else:
            gray = image.astype(np.float32, copy=False)
            if gray.max() > 1.0:
                gray = gray / 255.0

        image_edges = canny(gray)

        # Normalize and combine (weighted OR)
        dmax = float(depth_edges.max())
        imax = float(image_edges.max())
        depth_norm = depth_edges / dmax if dmax > 0 else depth_edges
        image_norm = image_edges / imax if imax > 0 else image_edges

        combined = 0.7 * depth_norm + 0.3 * image_norm
        cmax = float(combined.max())
        return combined / cmax if cmax > 0 else combined

    @_measure_runtime
    def _make_query_grid(
        self,
        mins: np.ndarray | None = None,
        maxs: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Create a uniform 3D grid of voxel centers covering the AABB [mins, maxs],
        expanded by self.padding and scaled by self.scale_factor, using the same
        lattice construction as visualize/src/generator.py.
        """
        resolution = self.points_grid_resolution
        if mins is None:
            mins = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        if maxs is None:
            maxs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        mins = np.asarray(mins, dtype=np.float32)
        maxs = np.asarray(maxs, dtype=np.float32)

        # Compute center and axis sizes with scale + padding
        center = ((mins + maxs) / 2.0).astype(np.float32)  # (3,)
        axis_extent = (maxs - mins).astype(np.float32) * float(self.scale_factor)
        axis_box_size = axis_extent + float(self.padding)  # (3,)

        # Uniform voxel size taken from the largest axis; grid is rectangular with uniform spacing
        box_size = float(np.max(axis_box_size))
        voxel_size = box_size / float(resolution)

        # Per-axis integer voxel counts (>=2)
        shape = np.maximum(2, np.round(axis_box_size / voxel_size).astype(int))
        nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])

        vs = float(voxel_size)
        total = np.array([nx, ny, nz], dtype=np.float32) * vs
        offset = (total - vs) / 2.0
        x0, y0, z0 = (center - offset).tolist()

        xs = np.linspace(x0, x0 + vs * (nx - 1), nx, dtype=np.float32)
        ys = np.linspace(y0, y0 + vs * (ny - 1), ny, dtype=np.float32)
        zs = np.linspace(z0, z0 + vs * (nz - 1), nz, dtype=np.float32)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32, copy=False)
        return points

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Map global index either by image (default) or by instance (seq)
        global_index = index
        seq_inst_idx: int | None = None
        if self.collate_3d == "seq" and self._seq_indices is not None:
            ds_idx, local_idx, seq_inst_idx = self._seq_indices[index]
            coco_ds = self.coco_ds[ds_idx]
            index = local_idx
        else:
            coco_ds, index = cast(tuple[CocoInstanceSegmentation, int], self._index_to_coco_ds(index))

        item = coco_ds[index]
        item["index"] = global_index  # coco_ds stores shard-local index; restore global
        image = item["inputs"]
        width = item["inputs.width"]
        height = item["inputs.height"]
        info = item["inputs.info"]

        data = self._load_data(Path(coco_ds.root), index)
        canvas_size = (int(image.shape[-2]), int(image.shape[-1]))
        intrinsic = CameraIntrinsic._wrap(torch.as_tensor(data["intrinsic"]), canvas_size=canvas_size)
        extrinsic = cast(np.ndarray, data["extrinsic"])

        if self.load_depth:
            item["inputs"] = data["depth"]
            if self.load_color or self.project:
                item["inputs.image"] = image
            else:
                item["inputs"] = tv_tensors.Image(depth_to_image(data["depth"], a=0.0))
        if self.load_normals:
            item["inputs.normals"] = data["normals"]
        item["inputs.intrinsic"] = intrinsic
        item["inputs.extrinsic"] = extrinsic

        if self.transforms is not None:
            item = self._apply_transforms(item)
            intrinsic = item["inputs.intrinsic"]
            width = item["inputs.width"]
            height = item["inputs.height"]

        masks = item["inputs.masks"]
        if self.stack_2d:
            n = len(masks)
            if n > 0:
                indices = torch.arange(1, n + 1).view(n, 1, 1)
                masks = torch.max(masks.long() * indices, dim=0).values
                item["inputs.labels"] = indices.view(n)
            else:
                masks = torch.zeros((height, width), dtype=torch.long)
                item["inputs.labels"] = torch.empty((0,), dtype=torch.long)
            if self.data_dir_3d is None:
                item["category.id"] = [str(id).zfill(8) for id in item["category.id"]]
                item["category.name"] = "_".join(item["category.name"])
                item["category.index"] = sum([CATEGORIES_MAP[id] for id in item["category.id"]])
                item["category.id"] = "_".join(item["category.id"])
                if not (self.load_depth and self.load_color):
                    item["inputs.masks"] = masks
                item.pop("inputs.boxes")

        near = self.near or 0.2
        far = self.far or 2.4
        if self.load_depth:
            depth = item["inputs"]
            if depth.any():
                if self.near is None:
                    near = max(0, depth[depth > 0].min() - self.padding)
                if self.far is None:
                    far = depth.max() + self.padding
            item["inputs.near"] = near
            item["inputs.far"] = far

            if self.project and self.collate_3d not in ["rand", "size", "seq"]:
                start = time.perf_counter()
                start_project = time.perf_counter()
                if self.near:
                    depth[depth < near] = 0.0
                if self.far:
                    depth[depth > far] = 0.0
                pcd = _convert_depth_image_to_point_cloud(depth, intrinsic[0].numpy(), extrinsic, depth_scale=1.0)
                projected_points = np.asarray(pcd.points, dtype=np.float32)
                logger.debug(f"depth to world points: {time.perf_counter() - start_project:.4f}s")
                item["inputs.depth"] = depth
                item["inputs"] = projected_points
                if self.load_color or self.load_normals or self.stack_2d:
                    u, v, _ = _points_to_uv(projected_points, intrinsic[0].numpy(), extrinsic)
                    u = np.clip(np.asarray(u).astype(np.int64, copy=False), 0, width - 1)
                    v = np.clip(np.asarray(v).astype(np.int64, copy=False), 0, height - 1)
                    if self.load_color:
                        colors = item["inputs.image"].permute(1, 2, 0).numpy()
                        item["inputs.colors"] = colors[v, u]
                        assert len(item["inputs.colors"]) == len(item["inputs"])
                    if self.load_normals:
                        normals = 2 * item["inputs.normals"] - 1
                        item["inputs.normals"] = normals[v, u]
                        assert len(item["inputs.normals"]) == len(item["inputs"])
                    if self.stack_2d:
                        instance_mask_2d = masks.numpy()
                        item["inputs.labels"] = instance_mask_2d[v, u].astype(np.int64, copy=False)
                        assert len(item["inputs.labels"]) == len(item["inputs"])
                if self.apply_filter == "edge":
                    item["points"] = projected_points
                    if self.stack_2d:
                        item["points.occ"] = item["inputs.labels"] > 0
                        item["points.labels"] = item["inputs.labels"]

                    start_edge = time.perf_counter()
                    start_filter = time.perf_counter()
                    edge_map = gaussian_filter(
                        self._edge_map(depth, cast(np.ndarray | None, normalize(colors) if self.load_color else None)),
                        sigma=1.0,
                    )
                    logger.debug(f"edge filter: {time.perf_counter() - start_filter:.4f}s")
                    d_flat = depth.ravel()
                    valid_depth = d_flat > 0
                    assert valid_depth.sum() == len(projected_points)

                    N = len(projected_points)
                    k_edge = N // 2
                    k_orig = N - k_edge

                    start_sample = time.perf_counter()
                    w = edge_map.ravel()[valid_depth].astype(np.float64, copy=False)
                    w_sum = w.sum()
                    if w_sum > 0:
                        p = w / w_sum
                        orig_idx = np.random.randint(0, N, size=k_orig)
                        counts = np.random.multinomial(k_edge, p)
                        edge_idx = np.repeat(np.arange(N), counts)
                        idx = np.concatenate([orig_idx, edge_idx], axis=0)
                    else:
                        idx = np.random.randint(0, N, size=N)
                    logger.debug(f"edge sampling: {time.perf_counter() - start_sample:.4f}s")

                    start_index = time.perf_counter()
                    item["inputs"] = projected_points[idx]
                    if self.load_color:
                        c = item["inputs.colors"]
                        item["inputs.colors"] = c[idx]
                    if self.load_normals:
                        nrm = item["inputs.normals"]
                        item["inputs.normals"] = nrm[idx]
                    if self.stack_2d:
                        lbl = item["inputs.labels"]
                        item["inputs.labels"] = lbl[idx]
                    logger.debug(f"edge indexing: {time.perf_counter() - start_index:.4f}s")
                    logger.debug(f"edge runtime: {time.perf_counter() - start_edge:.4f}s")
                logger.debug(f"project depth: {time.perf_counter() - start:.4f}s")

        # TODO: Extract method
        if self.data_dir_3d is not None:
            load_start = time.perf_counter()
            num_objects = len(info)
            num_non_zero_area = sum(i["area"] > 0 for i in info if "area" in i)
            meshes = [Trimesh()] * num_objects
            pcd_default: tuple[np.ndarray, np.ndarray | None] = (
                (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32))
                if self.load_normals
                else (np.zeros((1, 3), dtype=np.float32), None)
            )
            pcds: list[tuple[np.ndarray, np.ndarray | None]] = [pcd_default for _ in range(num_objects)]
            point: tuple[np.ndarray, np.ndarray] = (np.zeros((1, 3), dtype=np.float32), np.zeros((1,), dtype=bool))
            points: list[tuple[np.ndarray, np.ndarray]] = [point for _ in range(num_objects)]
            names: list[str] = ["" for _ in range(num_objects)]
            boxes: list[tuple[np.ndarray, np.ndarray] | None] = [None for _ in range(num_objects)]
            points_file = (
                self.points_file
                if isinstance(self.points_file, str)
                else (self.points_file[0] if isinstance(self.points_file, list) and len(self.points_file) > 0 else "")
            )

            if any("name" in n for n in info):
                for name, pose in zip(data["names"], data["poses"], strict=False):
                    start = time.perf_counter()
                    found = next(((i, n) for i, n in enumerate(info) if n["name"] == name), None)
                    if found is None:
                        continue

                    idx, match = found
                    category, name, _ = str(name).split("_")
                    assert category == str(item["category.id"][idx]).zfill(8)
                    names[idx] = str(name)
                    obj_path = self.data_dir_3d / category / name

                    if match["area"] == 0:
                        start = time.perf_counter()
                        mask = item["inputs.masks"][idx].bool().numpy()
                        assert mask.sum() == 0, f"Mask for {name} is not empty but has zero area."
                        if self.points_file and self.collate_3d:
                            if self.collate_3d in ["rand", "size", "seq"] or not self.apply_pose:
                                p_free = self._sample_free_points(num_samples=100_000)
                            else:
                                p_free = self._sample_free_points(
                                    mask=~mask,
                                    intrinsic=intrinsic[0].numpy(),
                                    extrinsic=extrinsic,
                                    near=near,
                                    far=far,
                                    num_samples=100_000,
                                    volume=self.sample_free_points,
                                )
                            free = np.zeros((len(p_free),), dtype=bool)
                            points[idx] = (p_free, free)
                        logger.debug(f"skip object {name}: {time.perf_counter() - start:.4f}s")
                        continue

                    pose = np.array(pose)
                    logger.debug(f"find object {name}: {time.perf_counter() - start:.4f}s")

                    if self.mesh_file is not None:
                        start = time.perf_counter()
                        if self.from_hdf5:
                            raise NotImplementedError("Loading mesh from HDF5 is not implemented yet.")
                        mesh = normalize_mesh(Trimesh(*load_mesh(obj_path / self.mesh_file)))
                        if self.apply_pose:
                            mesh.apply_transform(pose)
                        meshes[idx] = mesh
                        logger.debug(f"load mesh: {time.perf_counter() - start:.4f}s")
                    if self.pcd_file is not None:
                        start = time.perf_counter()
                        loaded_pcd, normals = _load_pointcloud(
                            obj_path, self.pcd_file, load_normals=self.load_normals, from_hdf5=self.from_hdf5
                        )
                        if self.apply_pose:
                            loaded_pcd = np.asarray(apply_trafo(loaded_pcd, pose), dtype=np.float32)
                            if normals is not None:
                                normals = normals @ pose[:3, :3].T
                                norm = np.linalg.norm(normals, axis=1, keepdims=True)
                                norm[norm == 0.0] = 1.0
                                normals = normals / norm

                        pcds[idx] = (loaded_pcd, normals)
                        logger.debug(f"load pointcloud: {time.perf_counter() - start:.4f}s")
                    if self.points_file is not None:
                        start = time.perf_counter()
                        if self.points_file == "grid":
                            p = self._make_query_grid()
                            occ = np.zeros((len(p),), dtype=bool)
                        else:
                            if isinstance(self.points_file, list):
                                points_file = random.choice(self.points_file)
                            if self.cache_points:
                                p, occ = _cached_load_points(str(obj_path / points_file), self.from_hdf5)
                            else:
                                p, occ = _load_points(obj_path / points_file, from_hdf5=self.from_hdf5)
                            p = np.asarray(p, dtype=np.float32)

                        if self.crop_points in ["cube", "sphere"]:
                            p_occ = p[occ]
                            if self.crop_points == "sphere":
                                radius = np.linalg.norm(p_occ, axis=1).max()
                                crop_mask = np.linalg.norm(p, axis=1) <= radius
                            elif self.crop_points == "cube":
                                box_min = p_occ.min(axis=0)
                                box_max = p_occ.max(axis=0)
                                crop_mask = np.all(p >= box_min, axis=1) & np.all(p <= box_max, axis=1)
                            p = p[crop_mask]
                            occ = occ[crop_mask]

                        if self.apply_pose:
                            p = np.asarray(apply_trafo(p, pose), dtype=np.float32)

                        if occ.any():
                            p_occ = np.asarray(p[occ], dtype=np.float32)
                            boxes[idx] = (p_occ.min(axis=0), p_occ.max(axis=0))

                        if self.points_fraction:
                            frac = self.points_fraction
                            if frac == "auto":
                                frac = max(1 / num_non_zero_area, 0.1)
                            p_idx = np.random.choice(len(p), size=int(frac * len(p)), replace=False)
                            p = p[p_idx]
                            occ = occ[p_idx]

                        points[idx] = (p, occ)

                        logger.debug(f"load points: {time.perf_counter() - start:.4f}s")
            logger.debug(f"load 3D data: {time.perf_counter() - load_start:.4f}s")

            # TODO: Extract method
            collate_start = time.perf_counter()
            if self.collate_3d in ["cat", "stack"]:
                # FIXME: Is there a better way?
                if self.collate_3d == "cat":
                    if item["inputs.skip"]:
                        item["category.id"] = "00000000"
                        item["category.name"] = "unknown"
                        item["category.index"] = -1
                        item["inputs.name"] = "empty"
                        item["inputs.labels"] = torch.zeros((0,), dtype=torch.int64)
                    else:
                        item["category.id"] = str(item["category.id"][0]).zfill(8)
                        item["category.name"] = item["category.name"][0]
                        item["category.index"] = CATEGORIES_MAP[item["category.id"]]
                        item["inputs.name"] = names[0]
                        item["inputs.boxes"] = item["inputs.boxes"][0]
                        item["inputs.labels"] = item["inputs.labels"][0]
                        item["inputs.masks"] = torch.any(item["inputs.masks"], dim=0)
                else:
                    item["category.id"] = [str(id).zfill(8) for id in item["category.id"]]
                    item["category.name"] = "_".join(item["category.name"])
                    item["category.index"] = sum([CATEGORIES_MAP[id] for id in item["category.id"]])
                    item["category.id"] = "_".join(item["category.id"])
                    item["inputs.name"] = "_".join(names)

                if self.mesh_file is not None:
                    # if self.collate_3d == "cat":
                    vertex_offset = 0
                    new_faces = []
                    mesh_vertex_lengths = []
                    mesh_triangle_lengths = []
                    for mesh in meshes:
                        mesh_vertex_lengths.append(len(mesh.vertices))
                        mesh_triangle_lengths.append(len(mesh.faces))
                        new_faces.append(mesh.faces + vertex_offset)
                        vertex_offset += len(mesh.vertices)

                    item["mesh.vertices"] = np.concatenate([mesh.vertices for mesh in meshes])
                    item["mesh.triangles"] = np.concatenate(new_faces)
                    # Store exact lengths for deterministic splitting later
                    item["mesh.lengths"] = {"vertices": mesh_vertex_lengths, "triangles": mesh_triangle_lengths}
                    # else:
                    #     item["mesh"] = meshes
                    item["mesh.path"] = str(self.data_dir_3d / self.mesh_file)
                    item["mesh.name"] = self.mesh_file.split(".")[0]
                if self.pcd_file is not None:
                    # if self.collate_3d == "cat":
                    pcd_cat = np.concatenate([p[0] for p in pcds])
                    # Subsample and keep per-instance counts after subsampling for exact split later
                    pcd_idx = subsample_indices(pcd_cat, 100_000)
                    pcd_idx = np.sort(pcd_idx)
                    # Build instance id map for the concatenated array
                    inst_lengths = [len(p[0]) for p in pcds]
                    if sum(inst_lengths) == len(pcd_cat):
                        inst_ids = np.repeat(np.arange(len(inst_lengths)), inst_lengths)
                        ids_after = inst_ids[pcd_idx]
                        # bincount to lengths after subsample
                        pcd_lengths = np.bincount(ids_after, minlength=len(inst_lengths)).tolist()
                    else:
                        # Fallback: unknown; keep zeros of correct length
                        pcd_lengths = [0 for _ in pcds]
                    item["pointcloud"] = pcd_cat[pcd_idx]
                    if pcds[0][1] is not None:
                        normals_list = [n for _, n in pcds if n is not None]
                        if len(normals_list) == len(pcds):
                            item["pointcloud.normals"] = np.concatenate(normals_list)[pcd_idx]
                    item["pointcloud.lengths"] = pcd_lengths
                    # else:
                    #     item["pointcloud"] = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[0])) for p in pcds]
                    #     if pcds[0][1] is not None:
                    #         for pcd in pcds:
                    #            pcd.normals = o3d.utility.Vector3dVector(pcd[1])
                    item["pointcloud.path"] = str(self.data_dir_3d / self.pcd_file)
                    item["pointcloud.name"] = self.pcd_file.split(".")[0]
                if self.points_file is not None:
                    if self.append_pcd_to_points and self.pcd_file is not None:
                        for i, ((p, o), (pcd_obj, nrm)) in enumerate(zip(points, pcds, strict=False)):
                            if pcd_obj is None or len(pcd_obj) <= 1 or nrm is None:
                                continue

                            # Subsample surface anchors
                            k = min(len(pcd_obj), len(p) // 4)  # keep growth bounded
                            if k == 0 or len(p) <= 2 * k:
                                continue
                            idx_pcd = subsample_indices(pcd_obj, k)
                            anchors = pcd_obj[idx_pcd]
                            normals = cast(np.ndarray, nrm)[idx_pcd]

                            # Small bidirectional offsets along normals -> clean labels
                            eps = 0.001 * np.random.uniform(0.5, 1.5, size=(k, 1))
                            p_pos = anchors - eps * normals  # inside
                            p_neg = anchors + eps * normals  # outside

                            o_pos = np.ones((k,), dtype=bool)
                            o_neg = np.zeros((k,), dtype=bool)

                            # Keep a balanced slice of original points to preserve global ratio
                            keep = subsample_indices(p, len(p) - 2 * k)
                            p_new = np.concatenate([p[keep], p_pos, p_neg], axis=0)
                            o_new = np.concatenate([o[keep], o_pos, o_neg], axis=0)
                            points[i] = (p_new, o_new)

                    points = self._filter_free_points_against_other_boxes(points, boxes)

                    item["points"] = np.concatenate([p[0] for p in points])
                    item["points.occ"] = np.concatenate([p[1] for p in points])
                    item["points.path"] = str(self.data_dir_3d / points_file)
                    item["points.name"] = points_file.split(".")[0]
                    item["points.lengths"] = [len(p[0]) for p in points]

                    if self.collate_3d == "stack":
                        start = time.perf_counter()
                        labels_list = list()
                        for i, (_p, occ) in enumerate(points):
                            instance_id = i + 1
                            labels = np.zeros_like(occ, dtype=np.int64)
                            labels[occ] = instance_id
                            labels_list.append(labels)
                        item["points.labels"] = np.concatenate(labels_list)
                        logger.debug(f"stack points labels: {time.perf_counter() - start:.4f}s")

                    if self.crop_points == "frustum":
                        p, occ = item["points"], item["points.occ"]
                        mask = _is_in_frustum(
                            points=p,
                            intrinsic=intrinsic[0].numpy(),
                            extrinsic=extrinsic,
                            width=width,
                            height=height,
                            near=near,
                            far=far,
                        )
                        item["points"] = p[mask]
                        item["points.occ"] = occ[mask]
                        if "points.labels" in item:
                            item["points.labels"] = item["points.labels"][mask]
                        if "points.lengths" in item:
                            lengths = item["points.lengths"]
                            offsets = np.cumsum([0, *lengths])
                            updated = []
                            for start_idx, end_idx in pairwise(offsets):
                                updated.append(int(mask[start_idx:end_idx].sum()))
                            item["points.lengths"] = updated

                    if self.sample_free_points:
                        p, occ = item["points"], item["points.occ"]
                        num_samples = self._estimate_num_samples(
                            num_points=occ.sum(),
                            boxes=boxes,
                            intrinsic=intrinsic[0].numpy(),
                            width=width,
                            height=height,
                            near=near,
                            far=far,
                        )
                        p_free = self._sample_free_points(
                            mask=np.ones((height, width), dtype=bool),
                            intrinsic=intrinsic[0].numpy(),
                            extrinsic=extrinsic,
                            near=near,
                            far=far,
                            num_samples=num_samples,
                            volume=self.sample_free_points,
                        )

                        p_free = p_free[self._get_crop_mask_from_boxes(p_free, boxes)]
                        free = np.zeros((len(p_free),), dtype=bool)

                        item["points"] = np.concatenate([p, p_free])
                        item["points.occ"] = np.concatenate([occ, free])
                        if "points.labels" in item:
                            labels_free = np.zeros_like(free, dtype=np.int64)
                            item["points.labels"] = np.concatenate([item["points.labels"], labels_free])
                        if "points.lengths" in item:
                            item["points.lengths"].append(len(p_free))
            elif self.collate_3d in ["rand", "size", "seq"]:
                start = time.perf_counter()
                idx = None
                valid_indices = np.where([item_info.get("area", 0) > 0 for item_info in info])[0]
                if len(valid_indices) > 0:
                    if self.collate_3d == "seq":
                        if seq_inst_idx is not None and int(seq_inst_idx) in valid_indices:
                            idx = int(seq_inst_idx)
                    else:
                        probs = None
                        if self.collate_3d == "size":
                            areas = np.array([info[i].get("area", 0) for i in valid_indices])
                            probs = areas / areas.sum() if areas.sum() > 0 else None
                        idx = np.random.choice(valid_indices, p=probs)
                logger.debug(
                    f"select {'seq' if self.collate_3d == 'seq' else 'random'} index: {time.perf_counter() - start:.4f}s"
                )

                _masks = item.pop("inputs.masks")
                _boxes = item.pop("inputs.boxes")
                _labels = item.pop("inputs.labels")
                if idx is None:
                    item["inputs.skip"] = True
                    if item["category.id"]:
                        item["category.id"] = str(item["category.id"][0]).zfill(8)
                        item["category.name"] = item["category.name"][0]
                        item["category.index"] = CATEGORIES_MAP[item["category.id"]]
                    else:
                        item["category.id"] = "00000000"
                        item["category.name"] = "unknown"
                        item["category.index"] = -1
                    if self.load_depth and self.project:
                        item["inputs"] = None
                    if self.points_file is not None:
                        p_free = self._sample_free_points(num_samples=100_000)
                        item["points"] = p_free
                        item["points.occ"] = np.zeros((len(p_free),), dtype=bool)
                        item["points.path"] = str(self.data_dir_3d / points_file)
                        item["points.name"] = points_file
                else:
                    if self.mesh_file is not None:
                        mesh = meshes[idx]
                        item["mesh.vertices"] = mesh.vertices
                        item["mesh.triangles"] = mesh.faces
                        item["mesh.path"] = str(self.data_dir_3d / self.mesh_file)
                        item["mesh.name"] = self.mesh_file
                    if self.pcd_file is not None:
                        pcd = pcds[idx]
                        item["pointcloud"] = pcd[0]
                        if pcd[1] is not None:
                            item["pointcloud.normals"] = pcd[1]
                        item["pointcloud.path"] = str(self.data_dir_3d / self.pcd_file)
                        item["pointcloud.name"] = self.pcd_file
                    if self.points_file is not None:
                        p, occ = points[idx]
                        item["points"] = p
                        item["points.occ"] = occ
                        item["points.path"] = str(self.data_dir_3d / points_file)
                        item["points.name"] = points_file

                        if self.sample_free_points:
                            num_samples = self._estimate_num_samples(
                                num_points=occ.sum(),
                                boxes=[boxes[idx]],
                                intrinsic=intrinsic[0].numpy(),
                                width=width,
                                height=height,
                                near=near,
                                far=far,
                            )
                            p_free = self._sample_free_points(
                                mask=np.ones((height, width), dtype=bool),
                                intrinsic=intrinsic[0].numpy(),
                                extrinsic=extrinsic,
                                near=near,
                                far=far,
                                num_samples=num_samples,
                                volume=self.sample_free_points,
                            )
                            p_free = p_free[self._get_crop_mask_from_boxes(p_free, [boxes[idx]])]
                            free = np.zeros((len(p_free),), dtype=bool)
                            item["points"] = np.concatenate([p, p_free])
                            item["points.occ"] = np.concatenate([occ, free])

                    item["inputs.name"] = names[idx]
                    item["category.id"] = str(item["category.id"][idx]).zfill(8)
                    item["category.name"] = item["category.name"][idx]
                    item["category.index"] = CATEGORIES_MAP[item["category.id"]]

                    start = time.perf_counter()
                    # TODO: Make this a torchvision.transforms.v2.Transform
                    mask = _masks[idx]
                    box = _boxes[idx]
                    width, height = mask.size()  # original image size (W, H) for reference

                    x, y, w, h = box.numpy()

                    # Make a slightly larger square crop around the object
                    side = max(w, h)
                    side = int(np.round(side * 1.1))
                    cx = x + w // 2
                    cy = y + h // 2
                    x_new = int(cx - side // 2)
                    y_new = int(cy - side // 2)
                    box = (y_new, x_new, side, side)  # (top, left, height, width) for F.crop

                    # Crop intrinsics to match the new canvas
                    intrinsic = F.crop(intrinsic, *box)

                    # Record the actual cropped canvas size
                    item["inputs.width"] = int(side)
                    item["inputs.height"] = int(side)

                    # Always compute a cropped instance mask for consistent zeroing
                    _mask_cropped = F.crop(mask.unsqueeze(0), *box).squeeze(0)

                    if self.load_color:
                        image = item["inputs"] if not self.load_depth else item["inputs.image"]
                        norm_zero = -torch.tensor([0.485 / 0.229, 0.456 / 0.224, 0.406 / 0.225])
                        image = F.crop(image, *box)
                        image[:, _mask_cropped == 0] = norm_zero.unsqueeze(1)

                    if self.load_depth:
                        # Crop depth then zero non-object pixels using the cropped mask
                        depth = item["inputs"]
                        depth = F.crop(torch.from_numpy(depth).unsqueeze(0), *box).squeeze(0).numpy()
                        depth[_mask_cropped.numpy() == 0] = 0.0

                        if self.project:
                            depth = apply_trafo(depth_to_points(depth, intrinsic[0].numpy()), inv_trafo(extrinsic))
                        item["inputs"] = depth
                        if self.load_color:
                            item["inputs.image"] = image
                    elif self.load_color:
                        item["inputs"] = image
                    logger.debug(f"crop and resize: {time.perf_counter() - start:.4f}s")
            elif self.collate_3d == "list":
                if self.mesh_file is not None:
                    item["mesh.vertices"] = [mesh.vertices for mesh in meshes]
                    item["mesh.triangles"] = [mesh.faces for mesh in meshes]
                    item["mesh.path"] = [str(self.data_dir_3d / self.mesh_file) for _ in meshes]
                    item["mesh.name"] = self.mesh_file
                if self.pcd_file is not None:
                    item["pointcloud"] = [pcd[0] for pcd in pcds]
                    if pcds[0][1] is not None:
                        item["pointcloud.normals"] = [pcd[1] for pcd in pcds]
                    item["pointcloud.path"] = str(self.data_dir_3d / self.pcd_file)
                    item["pointcloud.name"] = self.pcd_file
                if self.points_file is not None:
                    start = time.perf_counter()

                    p_occ_list: list[tuple[np.ndarray, np.ndarray]] = []
                    p_free_list: list[np.ndarray] = []

                    for i, (p, occ) in enumerate(points):
                        # Optional per-instance frustum crop
                        if self.crop_points == "frustum":
                            mask = _is_in_frustum(
                                points=p,
                                intrinsic=intrinsic[0].numpy(),
                                extrinsic=extrinsic,
                                width=width,
                                height=height,
                                near=near,
                                far=far,
                            )
                            p = p[mask]
                            occ = occ[mask]

                        # Per-instance free sampling (rand/size-like)
                        if self.sample_free_points:
                            num_samples = self._estimate_num_samples(
                                num_points=int(occ.sum()),
                                boxes=[boxes[i]],
                                intrinsic=intrinsic[0].numpy(),
                                width=width,
                                height=height,
                                near=near,
                                far=far,
                            )
                            p_free = self._sample_free_points(
                                mask=np.ones((height, width), dtype=bool),
                                intrinsic=intrinsic[0].numpy(),
                                extrinsic=extrinsic,
                                near=near,
                                far=far,
                                num_samples=num_samples,
                                volume=self.sample_free_points,
                            )
                            # Exclude free points that land inside this instance's occupied box
                            if boxes[i] is not None:
                                p_free = p_free[self._get_crop_mask_from_boxes(p_free, [boxes[i]])]

                            _free = np.zeros((len(p_free),), dtype=bool)
                            p = np.concatenate([p, p_free], axis=0)
                            occ = np.concatenate([occ, _free], axis=0)
                            p_free_list.append(p_free)
                        else:
                            # Keep indices aligned if no free points are sampled
                            p_free_list.append(np.empty((0, 3), dtype=np.float32))

                        p_occ_list.append((p, occ))

                    logger.debug(f"list-mode per-instance points: {time.perf_counter() - start:.4f}s")

                    # Pad to common length (with free points) and stack
                    if self.sample_free_points and len(p_occ_list) > 0:
                        start = time.perf_counter()
                        p_max = max(len(p) for p, _ in p_occ_list)
                        for i, ((p, occ), p_free) in enumerate(zip(p_occ_list, p_free_list, strict=False)):
                            n_pad = p_max - len(p)
                            if n_pad > 0:
                                if len(p_free) > 0:
                                    idx_pad = np.random.choice(len(p_free), size=n_pad, replace=n_pad > len(p_free))
                                    p_pad = p_free[idx_pad]
                                else:
                                    # Fallback: duplicate from existing p if p_free is empty
                                    idx_pad = np.random.choice(len(p), size=n_pad, replace=len(p) < n_pad)
                                    p_pad = p[idx_pad]
                                occ_pad = np.zeros((n_pad,), dtype=bool)
                                p = np.concatenate([p, p_pad], axis=0)
                                occ = np.concatenate([occ, occ_pad], axis=0)
                                p_occ_list[i] = (p, occ)
                        logger.debug(f"pad points: {time.perf_counter() - start:.4f}s")

                    # Stack across instances
                    item["points"] = np.stack([p for p, _ in p_occ_list], axis=0)
                    item["points.occ"] = np.stack([occ for _, occ in p_occ_list], axis=0)
                    item["points.path"] = str(self.data_dir_3d / points_file)
                    item["points.name"] = points_file
            else:
                if self.mesh_file is not None:
                    item["mesh.vertices"] = [mesh.vertices for mesh in meshes]
                    item["mesh.triangles"] = [mesh.faces for mesh in meshes]
                    item["mesh.path"] = [str(self.data_dir_3d / self.mesh_file) for _ in meshes]
                    item["mesh.name"] = self.mesh_file
                if self.pcd_file is not None:
                    item["pointcloud"] = [pcd[0] for pcd in pcds]
                    if pcds[0][1] is not None:
                        item["pointcloud.normals"] = [pcd[1] for pcd in pcds]
                    item["pointcloud.path"] = str(self.data_dir_3d / self.pcd_file)
                    item["pointcloud.name"] = self.pcd_file
                if self.points_file is not None:
                    for i, (p, occ) in enumerate(points):
                        if len(p) == 1:
                            p = self._sample_free_points(num_samples=100_000)
                            occ = np.zeros((len(p),), dtype=bool)
                            points[i] = (p, occ)

                    item["points"] = np.stack([p[0] for p in points], axis=0)
                    item["points.occ"] = np.stack([p[1] for p in points], axis=0)
                    item["points.path"] = str(self.data_dir_3d / points_file)
                    item["points.name"] = points_file
            logger.debug(f"collate 3D data: {time.perf_counter() - collate_start:.4f}s")

        item["inputs.intrinsic"] = intrinsic[0].numpy()
        item = copy.deepcopy(item)
        if self.transforms_3d is not None:
            start = time.perf_counter()
            item = _apply_transforms_3d(item, self.transforms_3d)
            logger.debug(f"3D transforms: {time.perf_counter() - start:.4f}s")

        return item

    def __len__(self) -> int:
        if self.collate_3d == "seq" and self._seq_indices is not None:
            return len(self._seq_indices)
        return sum(len(ds) for ds in self.coco_ds)

    def __repr__(self):
        repr_str = super().__repr__()
        if self.transforms_3d:
            trafo_str = "\n3D Transformations:\n"
            transforms_3d: list[Transform] = (
                self.transforms_3d if isinstance(self.transforms_3d, list) else [self.transforms_3d]
            )
            for trafo in transforms_3d:
                if trafo.args:
                    trafo_str += f"  {trafo.name}:\n"
                    for k, v in trafo.args.items():
                        if isinstance(v, np.ndarray):
                            with np.printoptions(precision=3, suppress=True):
                                if v.ndim == 1:
                                    trafo_str += f"    {k}: {v}\n"
                                else:
                                    trafo_str += f"    {k}: array({v.shape})\n"
                        elif isinstance(v, Sequence):
                            if not isinstance(v, str) and len(v) > 6:
                                list_str = f"[{v[0]}, {v[1]}, {v[2]}, ..., {v[-3]}, {v[-2]}, {v[-1]}]"
                                trafo_str += f"    {k}: {list_str} (len={len(v)})\n"
                            else:
                                trafo_str += f"    {k}: {v}\n"
                        else:
                            trafo_str += f"    {k}: {v}\n"
                else:
                    trafo_str += f"  {trafo.name}\n"
            return repr_str + trafo_str
        return repr_str
