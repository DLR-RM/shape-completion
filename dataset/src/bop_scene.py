from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from easy_o3d.utils import convert_depth_image_to_point_cloud
from PIL import Image
from trimesh import Trimesh

from utils import load_mesh as _load_mesh
from utils import setup_logger

from .graspnet import HAS_LIBMESH, GraspNetEval
from .transforms import Transform, apply_transforms

logger = setup_logger(__name__)


@dataclass(frozen=True)
class _BOPSample:
    scene_dir: Path
    ann_id: str

    def __post_init__(self) -> None:
        if len(self.ann_id) != 6 or not self.ann_id.isdigit():
            raise ValueError(f"BOP ann_id must be a 6-digit string, got {self.ann_id!r}")


def _fit_plane_ransac(
    points: np.ndarray,
    threshold: float,
    *,
    num_iterations: int,
    max_fit_points: int,
    min_inliers: int,
    seed: int = 0,
) -> np.ndarray | None:
    if len(points) < max(3, min_inliers):
        return None

    fit_points = points
    if len(fit_points) > max_fit_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(fit_points), size=max_fit_points, replace=False)
        fit_points = fit_points[indices]

    rng = np.random.default_rng(seed)
    best_plane: np.ndarray | None = None
    best_count = -1
    best_error = np.inf
    for _ in range(num_iterations):
        sample = fit_points[rng.choice(len(fit_points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal = normal / norm
        d = -float(normal @ sample[0])
        distances = np.abs(fit_points @ normal + d)
        inliers = distances <= threshold
        count = int(inliers.sum())
        if count < min_inliers:
            continue
        error = float(distances[inliers].mean())
        if count > best_count or (count == best_count and error < best_error):
            best_plane = np.asarray([normal[0], normal[1], normal[2], d], dtype=np.float32)
            best_count = count
            best_error = error

    if best_plane is None:
        return None

    distances = np.abs(fit_points @ best_plane[:3] + best_plane[3])
    inlier_points = fit_points[distances <= threshold]
    if len(inlier_points) < min_inliers:
        return best_plane

    center = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return best_plane
    normal = normal / norm
    d = -float(normal @ center)
    return np.asarray([normal[0], normal[1], normal[2], d], dtype=np.float32)


def _plane_distances(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    return np.abs(points @ plane[:3] + plane[3]) / max(float(np.linalg.norm(plane[:3])), 1e-8)


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {path}: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to read JSON file {path}: {e}") from e


class BOPSceneEval(GraspNetEval):
    """Scene-level BOP evaluation dataset with the same item contract as GraspNetEval.

    Unlike the legacy :class:`BOP` dataset, this loader keeps the full depth frame,
    projects all visible instance masks to 3D labels, and loads all frame-level CAD
    meshes for instance-completion evaluation.

    BOP ``models_eval`` meshes are stored in millimeters, while this loader's
    item contract uses meters. Keep the default ``mesh_scale=0.001`` for normal
    BOP meshes; set it to ``1.0`` only for pre-scaled mesh directories.
    """

    def __init__(
        self,
        root: Path | str,
        name: str,
        split: str = "test",
        mesh_dir: Path | str | None = None,
        scene_ids: Iterable[int] | None = None,
        project: bool = False,
        load_mesh: bool = True,
        crop_to_mesh: bool = False,
        crop_padding: float = 0.1,
        mesh_simplify_fraction: float | None = None,
        mesh_scale: float = 0.001,
        load_pcd: bool = False,
        pcd_num_points: int = 100_000,
        generate_points: bool = False,
        points_sampling: Literal["uniform", "surface", "both"] = "uniform",
        num_points: int = 100_000,
        points_padding: float = 0.1,
        load_points: bool = False,
        collate_points: Literal["stack"] | None = None,
        sample_scene_points: bool = False,
        scene_points_volume: Literal["cube", "sphere"] = "cube",
        num_scene_points: int = 100_000,
        padding: float = 0.1,
        scale_factor: float = 1.0,
        load_label: bool = False,
        stack_2d: bool = False,
        depth_dir: str = "depth",
        mask_dir: str = "mask_visib",
        filter_background: float | None = None,
        background_plane_threshold: float | None = None,
        background_plane_iterations: int = 256,
        background_plane_max_fit_points: int = 5000,
        background_plane_min_inliers: int = 100,
        background_plane_show: bool = False,
        statistical_outlier_removal: bool = False,
        statistical_outlier_neighbors: int = 50,
        statistical_outlier_std_ratio: float = 2.5,
        statistical_outlier_min_points: int | None = None,
        statistical_outlier_show: bool = False,
        dbscan_filter: bool = False,
        dbscan_eps: float = 0.05,
        dbscan_min_points: int = 10,
        dbscan_keep: Literal["non_noise", "largest"] = "non_noise",
        dbscan_include_background: bool = False,
        dbscan_show: bool = False,
        one_view_per_scene: bool = False,
        target_filename: str | None = "test_targets_bop19.json",
        transforms: Callable | list[Transform] | None = None,
    ) -> None:
        self.name = name
        self.root = Path(root)
        self.dataset_root = self.root / name
        self.split = split
        self.load_mesh = load_mesh
        self.mesh_dir = Path(mesh_dir) if mesh_dir is not None else self.dataset_root / "models_eval"
        self.scene_ids = set(scene_ids) if scene_ids is not None else None
        self.project = project
        self.crop_to_mesh = crop_to_mesh
        self.crop_padding = crop_padding
        self.mesh_simplify_fraction = mesh_simplify_fraction
        self.mesh_scale = mesh_scale
        self.load_pcd = load_pcd
        self.pcd_num_points = pcd_num_points
        self.generate_points = generate_points
        self.points_sampling = points_sampling
        self.num_points = num_points
        self.points_padding = points_padding
        self.load_points = load_points
        self.collate_points = collate_points
        self.sample_scene_points = sample_scene_points
        self.scene_points_volume = scene_points_volume
        self.num_scene_points = num_scene_points
        self.padding = padding
        self.scale_factor = scale_factor
        self.load_label = load_label
        self.stack_2d = stack_2d
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.filter_background = filter_background
        self.background_plane_threshold = background_plane_threshold
        self.background_plane_iterations = background_plane_iterations
        self.background_plane_max_fit_points = background_plane_max_fit_points
        self.background_plane_min_inliers = background_plane_min_inliers
        self.background_plane_show = background_plane_show
        self.statistical_outlier_removal = statistical_outlier_removal
        self.statistical_outlier_neighbors = statistical_outlier_neighbors
        self.statistical_outlier_std_ratio = statistical_outlier_std_ratio
        self.statistical_outlier_min_points = statistical_outlier_min_points
        self.statistical_outlier_show = statistical_outlier_show
        self.dbscan_filter = dbscan_filter
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points
        self.dbscan_keep = dbscan_keep
        self.dbscan_include_background = dbscan_include_background
        self.dbscan_show = dbscan_show
        self.one_view_per_scene = one_view_per_scene
        self.target_filename = target_filename
        self.transforms = transforms
        self._mesh_decimation_enabled = self.mesh_simplify_fraction is not None
        self._mesh_decimation_disabled_reason: str | None = None

        if self.load_mesh and not self.mesh_dir.exists():
            raise FileNotFoundError(f"Could not find BOP models directory at {self.mesh_dir}.")
        if self.crop_to_mesh and not (self.project and self.load_mesh):
            raise ValueError("crop_to_mesh requires both project=True and load_mesh=True")
        if self.mesh_simplify_fraction is not None and not 0.0 < self.mesh_simplify_fraction <= 1.0:
            raise ValueError(f"mesh_simplify_fraction must be in (0.0, 1.0], got {self.mesh_simplify_fraction}")
        if self.mesh_scale <= 0.0:
            raise ValueError(f"mesh_scale must be positive, got {self.mesh_scale}")
        if self.background_plane_threshold is not None and self.background_plane_threshold <= 0.0:
            raise ValueError(f"background_plane_threshold must be positive, got {self.background_plane_threshold}")
        if self.background_plane_iterations < 1:
            raise ValueError(f"background_plane_iterations must be positive, got {self.background_plane_iterations}")
        if self.background_plane_max_fit_points < 3:
            raise ValueError(
                f"background_plane_max_fit_points must be at least 3, got {self.background_plane_max_fit_points}"
            )
        if self.background_plane_min_inliers < 1:
            raise ValueError(f"background_plane_min_inliers must be positive, got {self.background_plane_min_inliers}")
        if self.filter_background is not None and self.background_plane_threshold is not None:
            raise ValueError("Use either filter_background or background_plane_threshold, not both.")
        if self.statistical_outlier_neighbors < 1:
            raise ValueError(f"statistical_outlier_neighbors must be positive, got {self.statistical_outlier_neighbors}")
        if self.statistical_outlier_std_ratio <= 0.0:
            raise ValueError(f"statistical_outlier_std_ratio must be positive, got {self.statistical_outlier_std_ratio}")
        if self.statistical_outlier_min_points is not None and self.statistical_outlier_min_points < 1:
            raise ValueError(f"statistical_outlier_min_points must be positive, got {self.statistical_outlier_min_points}")
        if self.dbscan_eps <= 0.0:
            raise ValueError(f"dbscan_eps must be positive, got {self.dbscan_eps}")
        if self.dbscan_min_points < 1:
            raise ValueError(f"dbscan_min_points must be positive, got {self.dbscan_min_points}")
        if self.dbscan_keep not in {"non_noise", "largest"}:
            raise ValueError(f"dbscan_keep must be 'non_noise' or 'largest', got {self.dbscan_keep}")
        if self.mesh_simplify_fraction is not None and importlib.util.find_spec("fast_simplification") is None:
            self._mesh_decimation_enabled = False
            self._mesh_decimation_disabled_reason = "missing optional dependency 'fast_simplification'"
            logger.warning(
                "Mesh decimation disabled because 'fast_simplification' is missing. "
                "Install package 'fast-simplification' to enable mesh simplification."
            )
        if self.load_pcd and not self.load_mesh:
            raise ValueError("load_pcd=True requires load_mesh=True because pointclouds are sampled from meshes.")
        if self.load_points and not self.load_mesh:
            raise ValueError("load_points=True requires load_mesh=True because points are generated from meshes.")
        if self.generate_points and not HAS_LIBMESH:
            raise RuntimeError("generate_points=True requires libmesh. Please compile libs/libmesh.")
        if self.generate_points and not self.load_mesh:
            raise ValueError("generate_points=True requires load_mesh=True")
        if self.sample_scene_points and not HAS_LIBMESH:
            raise RuntimeError("sample_scene_points=True requires libmesh. Please compile libs/libmesh.")
        if self.sample_scene_points and not self.load_mesh:
            raise ValueError("sample_scene_points=True requires load_mesh=True")
        if self.stack_2d and not self.load_label:
            raise ValueError("stack_2d=True requires load_label=True")

        self.scene_dirs = self._find_scene_dirs()
        self.categories = [scene_dir.name for scene_dir in self.scene_dirs]
        self.samples = self._enumerate_samples()

        self._camera_cache: dict[Path, dict[str, Any]] = {}
        self._gt_cache: dict[Path, dict[str, Any]] = {}
        self._gt_info_cache: dict[Path, dict[str, Any]] = {}
        self._mesh_cache: dict[tuple[Path, float | None, float], Trimesh] = {}
        self._points_cache: dict[tuple[Path, str], tuple[np.ndarray, np.ndarray]] = {}

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        depth, meta = self._load_depth_and_meta(sample)
        frame_id = int(sample.ann_id)
        depth_path = self._depth_path(sample)

        item: dict[str, Any] = {
            "index": index,
            "category.name": f"{self.name}/{sample.scene_dir.name}",
            "category.id": f"{self.name}/{sample.scene_dir.name}",
            "scene.id": int(sample.scene_dir.name),
            "frame.id": frame_id,
            "inputs": depth,
            "inputs.name": depth_path.name,
            "inputs.path": depth_path,
            "inputs.intrinsic": np.asarray(meta["intrinsic_matrix"], dtype=np.float32),
            "inputs.height": depth.shape[0],
            "inputs.width": depth.shape[1],
        }

        if self.load_label:
            self._populate_masks(item, sample)

        self._populate_camera_transforms(item, sample.scene_dir, frame_id)

        if self.project:
            pcd = convert_depth_image_to_point_cloud(
                depth, item["inputs.intrinsic"], item["inputs.extrinsic"], depth_scale=1.0
            )
            item["inputs.depth"] = depth
            item["inputs"] = np.asarray(pcd.points)
            if self.load_label and "label" in item:
                v_coords, u_coords = np.nonzero(depth > 0)
                item["inputs.pixel_coords"] = np.column_stack([v_coords, u_coords]).astype(np.int32)
                instance_map = item["label"]
                item["inputs.labels"] = instance_map[v_coords, u_coords].astype(np.int64)

        if self.load_mesh:
            self._populate_meshes(item, meta)

        if self.project and self.load_mesh and self.crop_to_mesh:
            self._crop_points_to_mesh(item)
        if self.project:
            self._filter_background_points(item)
            self._remove_statistical_outlier_points(item)
            self._filter_dbscan_components(item)

        self._align_2d_3d_instances(item)

        item = copy.deepcopy(item)
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                item = apply_transforms(item, self.transforms)
            else:
                item = self.transforms(item)
        return item

    def _find_scene_dirs(self) -> list[Path]:
        split_dir = self.dataset_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Could not locate BOP split directory {split_dir}.")
        scene_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and (p / self.depth_dir).exists())
        if self.scene_ids is not None:
            scene_names = {f"{scene_id:06d}" for scene_id in self.scene_ids}
            scene_dirs = [p for p in scene_dirs if p.name in scene_names]
        if not scene_dirs:
            raise FileNotFoundError(f"No scene directories found in {split_dir}.")
        return scene_dirs

    def _enumerate_samples(self) -> list[_BOPSample]:
        samples: list[_BOPSample] = []
        target_ann_ids = self._load_target_ann_ids()
        for scene_dir in self.scene_dirs:
            ann_ids = sorted(p.stem for p in (scene_dir / self.depth_dir).glob("*.png"))
            if target_ann_ids is not None:
                ann_ids = [ann_id for ann_id in ann_ids if ann_id in target_ann_ids.get(int(scene_dir.name), set())]
            if self.one_view_per_scene and ann_ids:
                samples.append(_BOPSample(scene_dir=scene_dir, ann_id=ann_ids[len(ann_ids) // 2]))
            else:
                samples.extend(_BOPSample(scene_dir=scene_dir, ann_id=ann_id) for ann_id in ann_ids)
        if not samples:
            raise FileNotFoundError(f"No depth frames found for BOP split '{self.name}/{self.split}'.")
        return samples

    def _load_target_ann_ids(self) -> dict[int, set[str]] | None:
        # BOP test_targets_bop19.json is a test-split protocol file. Do not
        # apply it to val/train splits; HB, for example, ships test targets
        # next to val_primesense, but they do not define the val protocol.
        if self.target_filename is None or not self.split.startswith("test"):
            return None
        target_path = self.dataset_root / self.target_filename
        if not target_path.exists():
            return None

        targets: dict[int, set[str]] = {}
        for row in _load_json(target_path):
            scene_id = int(row["scene_id"])
            if self.scene_ids is not None and scene_id not in self.scene_ids:
                continue
            targets.setdefault(scene_id, set()).add(f"{int(row['im_id']):06d}")
        return targets

    def _depth_path(self, sample: _BOPSample) -> Path:
        return sample.scene_dir / self.depth_dir / f"{sample.ann_id}.png"

    def _load_depth_and_meta(self, sample: _BOPSample) -> tuple[np.ndarray, dict[str, Any]]:
        frame_key = str(int(sample.ann_id))
        camera_data = self._load_scene_camera(sample.scene_dir)[frame_key]
        gt_data = self._load_scene_gt(sample.scene_dir)[frame_key]
        gt_info = self._load_scene_gt_info(sample.scene_dir).get(frame_key, [])

        depth_raw = np.asarray(Image.open(self._depth_path(sample)), dtype=np.float32)
        depth_scale = float(camera_data.get("depth_scale", 1.0))
        # BOP depth images are raw integer depth values; depth_scale converts
        # them to millimeters, and the dataset contract uses meters.
        depth = depth_raw * depth_scale / 1000.0
        meta = {
            "intrinsic_matrix": np.asarray(camera_data["cam_K"], dtype=np.float32).reshape(3, 3),
            "gt": gt_data,
            "gt_info": gt_info,
        }
        return depth, meta

    def _load_scene_camera(self, scene_dir: Path) -> dict[str, Any]:
        if scene_dir not in self._camera_cache:
            self._camera_cache[scene_dir] = _load_json(scene_dir / "scene_camera.json")
        return self._camera_cache[scene_dir]

    def _load_scene_gt(self, scene_dir: Path) -> dict[str, Any]:
        if scene_dir not in self._gt_cache:
            self._gt_cache[scene_dir] = _load_json(scene_dir / "scene_gt.json")
        return self._gt_cache[scene_dir]

    def _load_scene_gt_info(self, scene_dir: Path) -> dict[str, Any]:
        if scene_dir not in self._gt_info_cache:
            path = scene_dir / "scene_gt_info.json"
            if path.exists():
                self._gt_info_cache[scene_dir] = _load_json(path)
            else:
                logger.debug(f"Optional BOP scene_gt_info.json is missing: {path}")
                self._gt_info_cache[scene_dir] = {}
        return self._gt_info_cache[scene_dir]

    def _populate_masks(self, item: dict[str, Any], sample: _BOPSample) -> None:
        gt = self._load_scene_gt(sample.scene_dir)[str(int(sample.ann_id))]
        gt_info = self._load_scene_gt_info(sample.scene_dir).get(str(int(sample.ann_id)), [])
        depth_shape = (int(item["inputs.height"]), int(item["inputs.width"]))
        instance_map = np.zeros(depth_shape, dtype=np.int64)
        masks: list[np.ndarray] = []
        boxes: list[list[int]] = []
        labels: list[int] = []
        info: list[dict[str, Any]] = []
        obj_ids: list[int] = []
        gt_indices: list[int] = []
        missing_mask_count = 0
        empty_mask_count = 0

        for gt_index, pose_data in enumerate(gt):
            mask_path = sample.scene_dir / self.mask_dir / f"{int(sample.ann_id):06d}_{gt_index:06d}.png"
            if not mask_path.exists():
                missing_mask_count += 1
                continue
            mask = np.asarray(Image.open(mask_path)).astype(bool)
            if not mask.any():
                empty_mask_count += 1
                continue

            seq_id = len(labels) + 1
            rows, cols = np.where(mask)
            x_min, x_max = int(cols.min()), int(cols.max())
            y_min, y_max = int(rows.min()), int(rows.max())
            masks.append(mask)
            boxes.append([x_min, y_min, x_max - x_min + 1, y_max - y_min + 1])
            labels.append(seq_id)
            obj_id = int(pose_data["obj_id"])
            obj_ids.append(obj_id)
            gt_indices.append(gt_index)
            instance_map[mask] = seq_id
            visib_fract = gt_info[gt_index].get("visib_fract") if gt_index < len(gt_info) else None
            info.append(
                {
                    "area": int(mask.sum()),
                    "instance_id": seq_id,
                    "label": seq_id,
                    "obj_id": obj_id,
                    "gt_index": gt_index,
                    "visib_fract": visib_fract,
                }
            )

        item["label"] = instance_map
        item["label.name"] = f"{sample.ann_id}.png"
        item["label.path"] = sample.scene_dir / self.mask_dir
        item["inputs.obj_id_order_2d"] = np.asarray(obj_ids, dtype=np.int32)
        item["inputs.gt_indices_2d"] = np.asarray(gt_indices, dtype=np.int64)
        item["scene.object_id_list"] = np.asarray(obj_ids, dtype=np.int32)
        item["inputs.missing_mask_count"] = missing_mask_count
        item["inputs.empty_mask_count"] = empty_mask_count
        if missing_mask_count:
            logger.warning(
                f"Missing {missing_mask_count} BOP visible mask(s) for "
                f"{self.name} frame {sample.ann_id} in {sample.scene_dir}."
            )

        if masks:
            item["inputs.masks"] = instance_map if self.stack_2d else np.stack(masks, axis=0).astype(np.uint8)
            item["inputs.boxes"] = np.asarray(boxes, dtype=np.float32)
            item["inputs.labels"] = np.asarray(labels, dtype=np.int64)
            item["inputs.info"] = info
            item["inputs.instance_ids_2d"] = np.asarray(labels, dtype=np.int32)
            item["inputs.instance_id_to_seq"] = {label: label for label in labels}
        else:
            item["inputs.masks"] = instance_map if self.stack_2d else np.zeros((0, *depth_shape), dtype=np.uint8)
            item["inputs.boxes"] = np.zeros((0, 4), dtype=np.float32)
            item["inputs.labels"] = np.zeros((0,), dtype=np.int64)
            item["inputs.info"] = []
            item["inputs.instance_ids_2d"] = np.zeros((0,), dtype=np.int32)

    def _populate_camera_transforms(self, item: dict[str, Any], scene_dir: Path, frame_id: int) -> None:
        # BOP does not provide scene-level camera poses. Keep all geometry in the
        # OpenCV camera frame while preserving the GraspNet-style key contract.
        eye = np.eye(4, dtype=np.float32)
        item["inputs.cam_to_world"] = eye
        item["inputs.extrinsic_world"] = eye
        item["inputs.extrinsic"] = eye

    def _filter_background_points(self, item: dict[str, Any]) -> None:
        if (
            (self.filter_background is None and self.background_plane_threshold is None)
            or "inputs.labels" not in item
            or not isinstance(item["inputs"], np.ndarray)
            or item["inputs"].ndim != 2
            or item["inputs"].shape[1] < 3
            or len(item["inputs.labels"]) != len(item["inputs"])
        ):
            return
        pts = item["inputs"]
        lbl = item["inputs.labels"]
        if self.background_plane_threshold is not None:
            background = pts[lbl == 0]
            item["inputs.background_plane_status"] = "not_enough_background"
            item["inputs.background_plane_num_background"] = len(background)
            item["inputs.background_plane_num_background_kept"] = 0
            item["inputs.background_plane_inlier_ratio"] = 0.0
            item["inputs.background_plane_mean_distance"] = np.nan
            if len(background) < max(3, self.background_plane_min_inliers):
                return
            plane = _fit_plane_ransac(
                background,
                float(self.background_plane_threshold),
                num_iterations=self.background_plane_iterations,
                max_fit_points=self.background_plane_max_fit_points,
                min_inliers=self.background_plane_min_inliers,
            )
            if plane is None:
                item["inputs.background_plane_status"] = "failed"
                return
            item["inputs.background_plane"] = plane
            distances = _plane_distances(pts, plane)
            keep_mask = (lbl != 0) | (distances <= float(self.background_plane_threshold))
            background_distances = distances[lbl == 0]
            background_keep = background_distances <= float(self.background_plane_threshold)
            item["inputs.background_plane_status"] = "ok"
            item["inputs.background_plane_num_background_kept"] = int(background_keep.sum())
            item["inputs.background_plane_inlier_ratio"] = float(background_keep.mean()) if len(background_keep) else 0.0
            item["inputs.background_plane_mean_distance"] = (
                float(background_distances[background_keep].mean()) if np.any(background_keep) else np.nan
            )
            if self.background_plane_show:
                self._show_background_plane_filter(pts[:, :3], lbl, keep_mask)
            self._apply_input_point_keep_mask(item, keep_mask)
            return

        z_thresh = float(self.filter_background)
        keep_mask = ~((lbl == 0) & ((pts[:, 2] > z_thresh) | (pts[:, 2] < -z_thresh)))
        self._apply_input_point_keep_mask(item, keep_mask)

    def _apply_input_point_keep_mask(self, item: dict[str, Any], keep_mask: np.ndarray) -> None:
        old_labels = np.asarray(item["inputs.labels"]) if "inputs.labels" in item else None
        item["inputs"] = item["inputs"][keep_mask]
        if old_labels is not None and len(old_labels) == len(keep_mask):
            item["inputs.labels"] = old_labels[keep_mask]
        if "inputs.pixel_coords" in item and len(item["inputs.pixel_coords"]) == len(keep_mask):
            self._sync_2d_labels_from_point_filter(item, old_labels, keep_mask)
            item["inputs.pixel_coords"] = item["inputs.pixel_coords"][keep_mask]

    def _sync_2d_labels_from_point_filter(
        self,
        item: dict[str, Any],
        old_labels: np.ndarray | None,
        keep_mask: np.ndarray,
    ) -> None:
        pixel_coords = np.asarray(item["inputs.pixel_coords"])
        removed = ~keep_mask
        if old_labels is None or not np.any(removed):
            return
        removed_pixels = pixel_coords[removed]
        removed_labels = old_labels[removed]

        if "label" in item and isinstance(item["label"], np.ndarray):
            label_map = item["label"].copy()
            label_map[removed_pixels[:, 0], removed_pixels[:, 1]] = 0
            item["label"] = label_map
        if "inputs.masks" not in item or not isinstance(item["inputs.masks"], np.ndarray):
            return
        masks = item["inputs.masks"].copy()
        if self.stack_2d and masks.ndim == 2:
            masks[removed_pixels[:, 0], removed_pixels[:, 1]] = 0
        elif masks.ndim == 3:
            for label in np.unique(removed_labels):
                if label <= 0:
                    continue
                label_pixels = removed_pixels[removed_labels == label]
                mask_index = int(label) - 1
                if 0 <= mask_index < len(masks):
                    masks[mask_index, label_pixels[:, 0], label_pixels[:, 1]] = 0
        item["inputs.masks"] = masks

    def _show_background_plane_filter(self, points: np.ndarray, labels: np.ndarray, keep_mask: np.ndarray) -> None:
        import open3d as o3d

        colors = np.full((len(points), 3), 0.55, dtype=np.float64)
        background = labels == 0
        colors[background & keep_mask] = np.asarray([0.1, 0.7, 0.2], dtype=np.float64)
        colors[background & ~keep_mask] = np.asarray([0.9, 0.1, 0.1], dtype=np.float64)
        colors[~background] = np.asarray([0.1, 0.3, 0.9], dtype=np.float64)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, frame], window_name="BOP background plane filter")

    def _remove_statistical_outlier_points(self, item: dict[str, Any]) -> None:
        if (
            not self.statistical_outlier_removal
            or "inputs.labels" not in item
            or not isinstance(item["inputs"], np.ndarray)
            or item["inputs"].ndim != 2
            or item["inputs"].shape[1] < 3
            or len(item["inputs.labels"]) != len(item["inputs"])
        ):
            return

        pts = item["inputs"]
        labels = np.asarray(item["inputs.labels"])
        keep_mask = np.ones(len(pts), dtype=bool)
        min_points = self.statistical_outlier_min_points or self.statistical_outlier_neighbors + 1
        min_points = max(min_points, self.statistical_outlier_neighbors + 1)

        for label in np.unique(labels):
            label_indices = np.flatnonzero(labels == label)
            if len(label_indices) < min_points:
                continue
            inlier_indices = self._statistical_inlier_indices(
                pts[label_indices, :3],
                nb_neighbors=self.statistical_outlier_neighbors,
                std_ratio=self.statistical_outlier_std_ratio,
            )
            if len(inlier_indices) == 0:
                continue
            label_keep = np.zeros(len(label_indices), dtype=bool)
            label_keep[inlier_indices] = True
            keep_mask[label_indices[~label_keep]] = False

        removed = ~keep_mask
        item["inputs.outlier_removal_num_removed"] = int(removed.sum())
        item["inputs.outlier_removal_num_removed_background"] = int(np.count_nonzero(removed & (labels == 0)))
        item["inputs.outlier_removal_num_removed_objects"] = int(np.count_nonzero(removed & (labels != 0)))
        if self.statistical_outlier_show:
            self._show_statistical_outlier_filter(pts[:, :3], labels, keep_mask)
        if not np.any(removed):
            return

        self._apply_input_point_keep_mask(item, keep_mask)

    def _show_statistical_outlier_filter(self, points: np.ndarray, labels: np.ndarray, keep_mask: np.ndarray) -> None:
        import open3d as o3d

        colors = np.full((len(points), 3), 0.55, dtype=np.float64)
        background = labels == 0
        colors[background & keep_mask] = np.asarray([0.1, 0.7, 0.2], dtype=np.float64)
        colors[background & ~keep_mask] = np.asarray([0.9, 0.1, 0.1], dtype=np.float64)
        colors[~background & keep_mask] = np.asarray([0.1, 0.3, 0.9], dtype=np.float64)
        colors[~background & ~keep_mask] = np.asarray([1.0, 0.45, 0.0], dtype=np.float64)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, frame], window_name="BOP statistical outlier filter")

    @staticmethod
    def _statistical_inlier_indices(points: np.ndarray, *, nb_neighbors: int, std_ratio: float) -> np.ndarray:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        _, indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return np.asarray(indices, dtype=np.int64)

    def _filter_dbscan_components(self, item: dict[str, Any]) -> None:
        if (
            not self.dbscan_filter
            or "inputs.labels" not in item
            or not isinstance(item["inputs"], np.ndarray)
            or item["inputs"].ndim != 2
            or item["inputs"].shape[1] < 3
            or len(item["inputs.labels"]) != len(item["inputs"])
        ):
            return

        pts = item["inputs"]
        labels = np.asarray(item["inputs.labels"])
        keep_mask = np.ones(len(pts), dtype=bool)
        clustered_labels = 0
        debug_cluster_ids = np.full(len(pts), -2, dtype=np.int64) if self.dbscan_show else None

        for label in np.unique(labels):
            if label == 0 and not self.dbscan_include_background:
                continue
            label_indices = np.flatnonzero(labels == label)
            if len(label_indices) < self.dbscan_min_points:
                continue
            cluster_ids = self._dbscan_cluster_ids(
                pts[label_indices, :3],
                eps=self.dbscan_eps,
                min_points=self.dbscan_min_points,
            )
            if debug_cluster_ids is not None:
                debug_cluster_ids[label_indices] = cluster_ids
            valid_clusters = cluster_ids[cluster_ids >= 0]
            if len(valid_clusters) == 0:
                continue
            if self.dbscan_keep == "largest":
                cluster_values, cluster_counts = np.unique(valid_clusters, return_counts=True)
                largest_cluster = int(cluster_values[int(np.argmax(cluster_counts))])
                label_keep = cluster_ids == largest_cluster
            else:
                label_keep = cluster_ids >= 0
            keep_mask[label_indices[~label_keep]] = False
            clustered_labels += 1

        removed = ~keep_mask
        item["inputs.dbscan_num_removed"] = int(removed.sum())
        item["inputs.dbscan_num_removed_background"] = int(np.count_nonzero(removed & (labels == 0)))
        item["inputs.dbscan_num_removed_objects"] = int(np.count_nonzero(removed & (labels != 0)))
        item["inputs.dbscan_num_clustered_labels"] = clustered_labels
        if self.dbscan_show:
            self._show_dbscan_filter(pts[:, :3], labels, keep_mask, debug_cluster_ids)
        if not np.any(removed):
            return

        self._apply_input_point_keep_mask(item, keep_mask)

    @staticmethod
    def _dbscan_cluster_ids(points: np.ndarray, *, eps: float, min_points: int) -> np.ndarray:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        return np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False), dtype=np.int64)

    def _show_dbscan_filter(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        keep_mask: np.ndarray,
        cluster_ids: np.ndarray | None = None,
    ) -> None:
        import open3d as o3d

        colors = np.full((len(points), 3), 0.55, dtype=np.float64)
        background = labels == 0
        colors[background & ~keep_mask] = np.asarray([0.9, 0.1, 0.1], dtype=np.float64)
        colors[~background & ~keep_mask] = np.asarray([1.0, 0.45, 0.0], dtype=np.float64)
        if cluster_ids is None:
            colors[background & keep_mask] = np.asarray([0.1, 0.7, 0.2], dtype=np.float64)
            colors[~background & keep_mask] = np.asarray([0.1, 0.3, 0.9], dtype=np.float64)
        else:
            palette = np.asarray(
                [
                    [0.1, 0.3, 0.9],
                    [0.1, 0.7, 0.2],
                    [0.8, 0.2, 0.8],
                    [0.0, 0.7, 0.8],
                    [0.9, 0.8, 0.1],
                    [0.6, 0.4, 0.1],
                    [0.5, 0.2, 0.9],
                    [0.2, 0.6, 0.5],
                ],
                dtype=np.float64,
            )
            clustered = cluster_ids >= 0
            colors[keep_mask & clustered] = palette[cluster_ids[keep_mask & clustered] % len(palette)]
            colors[keep_mask & ~clustered & background] = np.asarray([0.1, 0.7, 0.2], dtype=np.float64)
            colors[keep_mask & ~clustered & ~background] = np.asarray([0.1, 0.3, 0.9], dtype=np.float64)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, frame], window_name="BOP DBSCAN component filter")

    def _populate_meshes(self, item: dict[str, Any], meta: dict[str, Any]) -> None:
        gt_indices = item.get("inputs.gt_indices_2d")
        if gt_indices is None:
            gt_indices = np.arange(len(meta["gt"]), dtype=np.int64)
        gt_indices_arr = np.asarray(gt_indices, dtype=np.int64)
        gt_selected = [meta["gt"][int(i)] for i in gt_indices_arr.tolist()]
        if not gt_selected:
            return

        obj_ids_arr = np.asarray([int(pose_data["obj_id"]) for pose_data in gt_selected], dtype=np.int32)
        obj_poses = self._bop_poses_to_matrices(gt_selected)
        mesh_paths = self._get_mesh_paths(obj_ids_arr.tolist())

        item.setdefault("scene.object_id_list", obj_ids_arr.copy())
        item["mesh.obj_poses_cam"] = obj_poses.copy()
        item["mesh.obj_id_order_3d"] = obj_ids_arr.copy()
        item["mesh.gt_indices_3d"] = gt_indices_arr.copy()

        vertices_cam, vertices_world, vertices_table, faces_list = self._transform_meshes(
            mesh_paths, obj_poses, item.get("inputs.cam_to_world"), item.get("inputs.cam_to_table")
        )
        if not vertices_cam:
            return

        item["mesh.vertices_cam"] = np.concatenate(vertices_cam, axis=0)
        item["mesh.triangles"] = np.concatenate(faces_list, axis=0)
        if vertices_world:
            item["mesh.vertices_world"] = np.concatenate(vertices_world, axis=0)
        if vertices_table:
            item["mesh.vertices_table"] = np.concatenate(vertices_table, axis=0)
        item["mesh.vertices"] = (
            item["mesh.vertices_table"]
            if vertices_table
            else item["mesh.vertices_world"]
            if vertices_world
            else item["mesh.vertices_cam"]
        )
        item["mesh.num_vertices"] = [v.shape[0] for v in vertices_cam]
        item["mesh.num_triangles"] = [f.shape[0] for f in faces_list]
        item["mesh.name"] = mesh_paths[0].stem
        item["mesh.path"] = mesh_paths[0]

        if self.load_pcd:
            self._populate_pcd(item, obj_ids_arr, mesh_paths, obj_poses, item.get("inputs.cam_to_world"), None)
        if self.load_points:
            self._populate_points(item, obj_ids_arr, mesh_paths, obj_poses, item.get("inputs.cam_to_world"), None)

    def _align_2d_3d_instances(self, item: dict[str, Any]) -> None:
        gt2d = item.get("inputs.gt_indices_2d")
        gt3d = item.get("mesh.gt_indices_3d")
        if gt2d is None or gt3d is None:
            super()._align_2d_3d_instances(item)
            return

        gt2d_arr = np.asarray(gt2d, dtype=np.int64)
        gt3d_arr = np.asarray(gt3d, dtype=np.int64)
        common = np.intersect1d(gt2d_arr, gt3d_arr, assume_unique=False)
        only_2d = np.setdiff1d(gt2d_arr, gt3d_arr, assume_unique=False)
        only_3d = np.setdiff1d(gt3d_arr, gt2d_arr, assume_unique=False)
        if np.array_equal(gt2d_arr, gt3d_arr):
            status = "perfect"
        elif len(common) > 0:
            status = "partial"
        else:
            status = "disjoint"

        ids2d = np.asarray(item.get("inputs.obj_id_order_2d", []), dtype=np.int32)
        ids3d = np.asarray(item.get("mesh.obj_id_order_3d", []), dtype=np.int32)
        item["alignment.status"] = status
        item["alignment.only_2d_gt_indices"] = only_2d
        item["alignment.only_3d_gt_indices"] = only_3d
        item["alignment.gt_index_order_2d"] = gt2d_arr.copy()
        item["alignment.gt_index_order_3d"] = gt3d_arr.copy()
        item["alignment.obj_id_order_2d"] = ids2d.copy()
        item["alignment.obj_id_order_3d"] = ids3d.copy()
        item["alignment.gt_index_to_2d_index"] = {int(gt_index): idx for idx, gt_index in enumerate(gt2d_arr)}
        item["alignment.gt_index_to_3d_index"] = {int(gt_index): idx for idx, gt_index in enumerate(gt3d_arr)}
        item["alignment.obj_id_to_2d_indices"] = self._indices_by_id(ids2d)
        item["alignment.obj_id_to_3d_indices"] = self._indices_by_id(ids3d)
        item["alignment.obj_id_to_2d_index"] = self._first_index_by_id(ids2d)
        item["alignment.obj_id_to_3d_index"] = self._first_index_by_id(ids3d)
        item["alignment.only_2d_ids"] = self._obj_ids_for_gt_indices(gt2d_arr, ids2d, only_2d)
        item["alignment.only_3d_ids"] = self._obj_ids_for_gt_indices(gt3d_arr, ids3d, only_3d)
        item["alignment.seq_to_obj_id"] = [int(obj_id) for obj_id in ids2d]

    @staticmethod
    def _indices_by_id(ids: np.ndarray) -> dict[int, list[int]]:
        indices_by_id: dict[int, list[int]] = {}
        for idx, obj_id in enumerate(ids):
            indices_by_id.setdefault(int(obj_id), []).append(idx)
        return indices_by_id

    @staticmethod
    def _first_index_by_id(ids: np.ndarray) -> dict[int, int]:
        return {obj_id: indices[0] for obj_id, indices in BOPSceneEval._indices_by_id(ids).items()}

    @staticmethod
    def _obj_ids_for_gt_indices(gt_indices: np.ndarray, obj_ids: np.ndarray, selected_gt_indices: np.ndarray) -> np.ndarray:
        if len(selected_gt_indices) == 0 or len(gt_indices) != len(obj_ids):
            return np.zeros((0,), dtype=np.int32)
        mask = np.isin(gt_indices, selected_gt_indices)
        return obj_ids[mask].astype(np.int32, copy=False)

    @staticmethod
    def _bop_poses_to_matrices(gt_selected: list[dict[str, Any]]) -> np.ndarray:
        poses = []
        for pose_data in gt_selected:
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = np.asarray(pose_data["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
            pose[:3, 3] = np.asarray(pose_data["cam_t_m2c"], dtype=np.float32) / 1000.0
            poses.append(pose)
        return np.stack(poses, axis=0)

    def _resolve_mesh_path(self, obj_id: int) -> Path:
        candidates = [
            self.mesh_dir / f"obj_{obj_id:06d}.ply",
            self.mesh_dir / f"obj_{obj_id:03d}.ply",
            self.mesh_dir / f"{obj_id:03d}" / "nontextured.ply",
            self.mesh_dir / f"{obj_id:03d}" / "textured.obj",
        ]
        for path in candidates:
            if path.exists():
                return path
        searched = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(f"Mesh for BOP obj_id={obj_id} not found. Searched:\n{searched}")

    def _load_mesh_cached(self, path: Path) -> Trimesh:
        effective_fraction = self.mesh_simplify_fraction if self._mesh_decimation_enabled else None
        cache_key = (path, effective_fraction, self.mesh_scale)
        if cache_key not in self._mesh_cache:
            vertices, faces = _load_mesh(path)
            if faces is None:
                raise ValueError(f"Loaded mesh {path} has no faces.")
            vertices = np.asarray(vertices, dtype=np.float32) * self.mesh_scale
            mesh = Trimesh(vertices, faces, process=False)
            original_face_count = len(faces)
            if effective_fraction is not None:
                target_face_count = int(original_face_count * effective_fraction)
                if target_face_count < original_face_count:
                    try:
                        mesh = mesh.simplify_quadric_decimation(1 - effective_fraction)
                    except (ImportError, ModuleNotFoundError) as e:
                        original_cache_key = (path, None, self.mesh_scale)
                        self._mesh_cache[original_cache_key] = mesh
                        logger.warning(
                            f"Mesh decimation unavailable for {path} ({type(e).__name__}: {e}); "
                            "using original mesh for this load."
                        )
                        return mesh
            self._mesh_cache[cache_key] = mesh
        return self._mesh_cache[cache_key]


__all__ = ["BOPSceneEval"]
