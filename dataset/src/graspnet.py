import copy
import importlib.util
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from easy_o3d.utils import convert_depth_image_to_point_cloud
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from trimesh import Trimesh

from utils import apply_trafo, inv_trafo, load_mesh, normalize_mesh, setup_logger

from .transforms import Transform, apply_transforms

try:
    from libs import check_mesh_contains

    HAS_LIBMESH = True
except ImportError:
    HAS_LIBMESH = False

logger = setup_logger(__name__)


def _log_debug_level_1(message: str) -> None:
    getattr(logger, "debug_level_1", logger.debug)(message)


def _log_debug_level_2(message: str) -> None:
    getattr(logger, "debug_level_2", logger.debug)(message)


def _find_scene_dirs(root: Path) -> list[Path]:
    scenes_root = root / "scenes"
    if not scenes_root.exists():
        raise FileNotFoundError(f"Could not locate GraspNet scenes under {scenes_root}.")
    scene_dirs = sorted(p for p in scenes_root.glob("scene_*") if p.is_dir())
    if not scene_dirs:
        raise FileNotFoundError(f"No scene directories found in {scenes_root}.")
    return scene_dirs


def _camera_dir(scene_dir: Path, camera: str, suffix: str) -> Path:
    preferred = scene_dir / camera / suffix
    if preferred.exists():
        return preferred
    legacy = scene_dir / f"{camera}_{suffix}"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Expected '{camera}/{suffix}' (or '{camera}_{suffix}') inside {scene_dir}.")


def _load_depth(depth_path: Path, depth_scale: float) -> np.ndarray:
    depth = np.array(Image.open(depth_path), dtype=np.float32)
    # Depth is stored as uint16 in millimetres; convert to float depth (metres by default).
    depth /= depth_scale
    return depth


@dataclass(frozen=True)
class _Sample:
    scene_dir: Path
    ann_id: str


class GraspNetEval(Dataset):
    """
    Minimal GraspNet dataset wrapper that loads per-frame depth maps and object meshes.

    This implementation avoids any heavy preprocessing and only fetches the pieces needed for
    evaluation-time integration. It operates directly on the original GraspNet directory layout:

        root/
          scenes/
            scene_0000/
              kinect/
                depth/0000.png
                meta/0000.mat
          models/
            000/nontextured.ply

    Parameters
    ----------
    root: str | Path
        Root directory of the extracted GraspNet dataset.
    split: {"train", "val", "test", "test_seen", "test_similar", "test_novel", "all"}
        Dataset split to iterate over. Splits follow the official GraspNet scene ID ranges.
    camera: {"kinect", "realsense"}
        Which camera stream to read (affects directory names for depth/meta files).
    depth_scale: float
        Scale factor applied to convert the stored depth values (default assumes millimetres → metres).
    load_mesh: bool
        If True, loads mesh paths and poses for all instances in the frame.
    mesh_dir: Optional[Path]
        Optional override for the models directory. Defaults to `<root>/models`.
    project: bool
        If True, converts depth image to 3D point cloud.
    crop_to_mesh: bool
        If True, crops the projected point cloud to the mesh bounding box (with padding).
        Only applies when both project=True and load_mesh=True.
    crop_padding: float
        Padding (in meters) to add around the mesh bounding box when cropping.
    mesh_simplify_fraction: Optional[float]
        If specified, decimates loaded meshes to this fraction of original faces (0.0-1.0).
        For example, 0.1 keeps 10% of faces. Decimated meshes are cached.
        Recommended value: 0.05-0.2 for fast loading while maintaining shape.
    load_pcd: bool
        If True, loads surface point clouds for each mesh.
    pcd_num_points: int
        Number of points to sample for surface point cloud (default: 2048).
    generate_points: bool
        If True, generates occupancy points for each mesh and caches them.
    points_sampling: Literal["uniform", "surface", "both"]
        Sampling strategy for occupancy points:
        - "uniform": Only uniform random samples in padded bounding box
        - "surface": Only near-surface samples with multiple noise levels
        - "both": Both uniform and surface samples (default)
    num_points: int
        Number of points to sample for occupancy testing (default: 100000).
    points_padding: float
        Padding factor for point sampling volume (default: 0.1).
    load_points: bool
        If True, loads generated occupancy points for meshes (cached in memory).
    collate_points: Optional[Literal["stack"]]
        How to collate points across multiple objects. "stack" creates instance labels.
    sample_scene_points: bool
        If True, samples scene-level points in a volume around all meshes.
    scene_points_volume: Literal["cube", "sphere"]
        Volume shape for scene-level point sampling (default: "cube").
    num_scene_points: int
        Number of scene-level points to sample (default: 100000).
    padding: float
        Padding for scene-level point sampling (default: 0.1).
    scale_factor: float
        Scale factor for scene-level point sampling volume (default: 1.0).
    load_label: bool
        If True, loads 2D segmentation masks/labels.
    stack_2d: bool
        If True, treats labels as instance segmentation and extracts instance IDs.
    one_view_per_scene: bool
        If True, randomly selects only one view (frame) per scene instead of all views.
        Useful to avoid overfitting when multiple views of the same scene are available.
    transforms: Optional[Callable]
        Optional data transformations to apply.
    filter_background: Optional[float]
        If set, removes projected depth points whose label==0 (background) and whose Z coordinate
        exceeds this threshold. Z is interpreted in the coordinate frame of `inputs` after projection
        (table frame if available, else world, else camera). Units are meters.
    """

    def __init__(
        self,
        root: Path | str,
        split: str = "test_novel",
        camera: Literal["kinect", "realsense"] = "kinect",
        depth_scale: float = 1000.0,
        load_mesh: bool = True,
        mesh_dir: Path | str | None = None,
        scene_ids: Iterable[int] | None = None,
        project: bool = False,
        crop_to_mesh: bool = False,
        crop_padding: float = 0.1,
        mesh_simplify_fraction: float | None = 0.1,
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
        one_view_per_scene: bool = False,
        filter_background: float | None = None,
        transforms: Callable | list[Transform] | None = None,
    ) -> None:
        self.name = self.__class__.__name__
        self.root = Path(root)
        self.split = split
        self.camera = camera
        self.depth_scale = depth_scale
        self.load_mesh = load_mesh
        self.mesh_dir = Path(mesh_dir) if mesh_dir is not None else self.root / "models"
        self.scene_ids = set(scene_ids) if scene_ids is not None else None
        self.project = project
        self.crop_to_mesh = crop_to_mesh
        self.crop_padding = crop_padding
        self.mesh_simplify_fraction = mesh_simplify_fraction
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
        self.one_view_per_scene = one_view_per_scene
        self.filter_background = filter_background
        self.transforms = transforms
        self._mesh_decimation_enabled = self.mesh_simplify_fraction is not None
        self._mesh_decimation_disabled_reason: str | None = None

        if self.load_mesh and not self.mesh_dir.exists():
            raise FileNotFoundError(f"Could not find GraspNet models directory at {self.mesh_dir}.")

        if self.crop_to_mesh and not (self.project and self.load_mesh):
            raise ValueError("crop_to_mesh requires both project=True and load_mesh=True")

        if self.mesh_simplify_fraction is not None:
            if not 0.0 < self.mesh_simplify_fraction <= 1.0:
                raise ValueError(f"mesh_simplify_fraction must be in (0.0, 1.0], got {self.mesh_simplify_fraction}")
            if importlib.util.find_spec("fast_simplification") is None:
                self._mesh_decimation_enabled = False
                self._mesh_decimation_disabled_reason = "missing optional dependency 'fast_simplification'"
                logger.warning(
                    "Mesh decimation disabled because 'fast_simplification' is missing. "
                    "Install package 'fast-simplification' to enable mesh simplification."
                )

        if self.generate_points and not HAS_LIBMESH:
            raise RuntimeError("generate_points=True requires libmesh. Please compile libs/libmesh.")

        if self.generate_points and not self.load_mesh:
            raise ValueError("generate_points=True requires load_mesh=True")

        # Ensure derived artifacts are only produced when meshes are loaded
        if self.load_pcd and not self.load_mesh:
            raise ValueError("load_pcd=True requires load_mesh=True because pointclouds are sampled from meshes.")
        if self.load_points and not self.load_mesh:
            raise ValueError("load_points=True requires load_mesh=True because points are generated from meshes.")

        if self.sample_scene_points and not HAS_LIBMESH:
            raise RuntimeError("sample_scene_points=True requires libmesh. Please compile libs/libmesh.")

        if self.sample_scene_points and not self.load_mesh:
            raise ValueError("sample_scene_points=True requires load_mesh=True")

        if self.stack_2d and not self.load_label:
            raise ValueError("stack_2d=True requires load_label=True")

        self.scene_dirs = _find_scene_dirs(self.root)
        self.categories = [scene_dir.name for scene_dir in self.scene_dirs]
        self.samples = self._enumerate_samples(split=split)

        self._meta_cache: dict[tuple[Path, str], dict[str, Any]] = {}
        self._intrinsic_cache: dict[Path, np.ndarray] = {}
        self._pose_cache: dict[Path, np.ndarray | None] = {}
        self._table_from_world: dict[Path, np.ndarray | None] = {}
        self._scene_obj_id_list: dict[Path, np.ndarray] = {}
        self._mesh_cache: dict[tuple[Path, float | None], Trimesh] = {}
        self._points_cache: dict[tuple[Path, str], tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        depth, meta = self._load_depth_and_meta(sample)
        frame_id = int(sample.ann_id)
        depth_path = self._depth_path(sample)

        item: dict[str, Any] = {
            "index": index,
            "category.name": sample.scene_dir.name,
            "category.id": int(sample.ann_id),
            "inputs": depth,
            "inputs.name": depth_path.name,
            "inputs.path": depth_path,
        }

        # Load intrinsic matrix
        intrinsic = meta.get("intrinsic_matrix")
        if intrinsic is None:
            intrinsic = self._load_intrinsic(sample.scene_dir)
        item["inputs.intrinsic"] = np.asarray(intrinsic, dtype=np.float32)

        # Always load scene object id list from file for correct 2D↔3D mapping
        try:
            item["scene.object_id_list"] = self._read_object_id_list(sample.scene_dir)
        except FileNotFoundError as e:
            logger.warning(str(e))

        # Store image dimensions
        item["inputs.height"] = depth.shape[0]
        item["inputs.width"] = depth.shape[1]

        # Load and populate masks/labels if requested
        if self.load_label:
            self._populate_masks(item, sample)

        # Load and populate camera transforms
        self._populate_camera_transforms(item, sample.scene_dir, frame_id)

        if self.project:
            pcd = convert_depth_image_to_point_cloud(
                depth, item["inputs.intrinsic"], item["inputs.extrinsic"], depth_scale=1.0
            )
            item["inputs.depth"] = depth
            item["inputs"] = np.asarray(pcd.points)

            # Project labels to 3D points if stack_2d is enabled
            if self.stack_2d and "inputs.masks" in item:
                v_coords, u_coords = np.nonzero(depth > 0)
                instance_map = item["inputs.masks"]
                point_labels = instance_map[v_coords, u_coords].astype(np.int64)
                item["inputs.labels"] = point_labels

            # Remove background (label==0) points whose Z exceeds threshold
            if (
                self.filter_background is not None
                and "inputs.labels" in item
                and isinstance(item["inputs"], np.ndarray)
                and item["inputs"].ndim == 2
                and item["inputs"].shape[1] >= 3
                and len(item["inputs.labels"]) == len(item["inputs"])
            ):
                pts = item["inputs"]
                lbl = item["inputs.labels"]
                z_thresh = float(self.filter_background)
                keep_mask = ~((lbl == 0) & ((pts[:, 2] > z_thresh) | (pts[:, 2] < -z_thresh)))
                item["inputs"] = pts[keep_mask]
                item["inputs.labels"] = lbl[keep_mask]

        # Load and transform meshes if requested
        if self.load_mesh:
            self._populate_meshes(item, meta)

        # Crop point cloud to mesh bounding box if requested
        if self.project and self.load_mesh and self.crop_to_mesh:
            self._crop_points_to_mesh(item)

        # Build alignment and reorder 3D per-object data if perfect match
        self._align_2d_3d_instances(item)

        item = copy.deepcopy(item)
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                item = apply_transforms(item, self.transforms)
            else:
                item = self.transforms(item)

        return item

    def _read_object_id_list(self, scene_dir: Path) -> np.ndarray:
        """Read and cache scene-level model IDs from object_id_list.txt.

        The file lists model IDs (0..87) for objects present in the scene.
        """
        if scene_dir in self._scene_obj_id_list:
            return self._scene_obj_id_list[scene_dir]

        obj_list_path = scene_dir / "object_id_list.txt"
        if not obj_list_path.exists():
            raise FileNotFoundError(f"Missing object_id_list.txt at {obj_list_path}")
        with obj_list_path.open("r") as f:
            ids = [int(line.strip()) for line in f if line.strip() != ""]
        arr = np.asarray(ids, dtype=np.int32)
        self._scene_obj_id_list[scene_dir] = arr
        return arr

    def _compute_obj_permutation(self, item: dict[str, Any], obj_ids: Sequence[int]) -> list[int] | None:
        """Compute permutation of per-object data to match 2D instance order.

        - If `inputs.obj_id_order_2d` is present, we build a permutation that
          places objects in that order first, followed by any remaining
          objects in their original order.
        - Returns None if no 2D order is available.
        """
        ids2d = item.get("inputs.obj_id_order_2d")
        if ids2d is None:
            return None
        ids2d_arr = np.asarray(ids2d, dtype=np.int32)
        obj_ids_arr = np.asarray(list(obj_ids), dtype=np.int32)
        idx_map = {int(oid): i for i, oid in enumerate(obj_ids_arr.tolist())}
        used: set[int] = set()
        perm: list[int] = []
        for oid in ids2d_arr.tolist():
            j = idx_map.get(int(oid))
            if j is not None and j not in used:
                perm.append(j)
                used.add(j)
        for j in range(len(obj_ids_arr)):
            if j not in used:
                perm.append(j)
        return perm

    def _align_2d_3d_instances(self, item: dict[str, Any]) -> None:
        """Align 2D instance ordering with 3D per-object data when possible.

        Adds mapping fields irrespective of alignment quality and, when a perfect
        ID match exists, reorders stacked occupancy point segments so that
        instance index i matches between 2D masks and 3D points.

        Fields added:
        - alignment.status: "perfect" | "partial" | "disjoint" | "none"
        - alignment.only_2d_ids / alignment.only_3d_ids: np.ndarray of IDs
        - alignment.obj_id_to_2d_index / alignment.obj_id_to_3d_index: dict
        - alignment.seq_to_obj_id: list mapping sequential 1..N to original obj id when 2D is present
        - alignment.obj_id_order_2d / alignment.obj_id_order_3d: arrays (copies for convenience)
        Adds mapping fields irrespective of alignment quality and, when IDs are
        available, reorders stacked occupancy point segments so that instance
        index i matches between 2D masks and 3D points using the canonical
        GraspNet mapping: seg==k ↔ object_id_list[k-1]. No geometric matching
        or heuristic offsets are used.
          points, occ, labels per object.
        Reordering logic (only when collate_points == "stack"):
        - Uses boundaries in item["points.lengths"] to slice per-object segments.
        - Applies a permutation derived from IDs so 3D order matches 2D order.
                - Rebuilds concatenated arrays and regenerates labels so occupied points
                    get their 2D-aligned sequential instance index.
        """

        # Fetch 2D/3D object-id orders; derive 2D from instance ids if needed
        ids2d = item.get("inputs.obj_id_order_2d")
        ids3d = item.get("mesh.obj_id_order_3d")
        if (
            ids2d is None
            and item.get("inputs.instance_ids_2d") is not None
            and item.get("scene.object_id_list") is not None
        ):
            inst_ids = np.asarray(item["inputs.instance_ids_2d"], dtype=np.int32)
            obj_list = np.asarray(item["scene.object_id_list"], dtype=np.int32)
            valid = inst_ids[(inst_ids > 0) & (inst_ids <= len(obj_list))]
            ids2d = obj_list[valid - 1]
            item["inputs.obj_id_order_2d"] = ids2d.copy()

        if ids2d is None or ids3d is None:
            item["alignment.status"] = "none"
            return

        ids2d = np.asarray(ids2d, dtype=np.int32)
        ids3d = np.asarray(ids3d, dtype=np.int32)
        # Drop placeholders (e.g., -1) from 2D mapping if any
        if ids2d.ndim == 1:
            ids2d = ids2d[ids2d >= 0]

        # First pass intersection / differences
        common = np.intersect1d(ids2d, ids3d, assume_unique=True)
        only_2d = np.setdiff1d(ids2d, ids3d, assume_unique=True)
        only_3d = np.setdiff1d(ids3d, ids2d, assume_unique=True)

        correction_applied = False
        offset_used: int | None = None

        if len(common) == len(ids2d) == len(ids3d) and len(only_2d) == 0 and len(only_3d) == 0:
            status = "perfect"
        elif len(common) > 0:
            status = "partial"
        else:
            status = "disjoint"

        item["alignment.status"] = status
        item["alignment.only_2d_ids"] = only_2d
        item["alignment.only_3d_ids"] = only_3d
        item["alignment.obj_id_order_2d"] = ids2d.copy()
        item["alignment.obj_id_order_3d"] = ids3d.copy()
        item["alignment.auto_offset_correction"] = correction_applied
        if correction_applied:
            item["alignment.offset_used"] = offset_used

        obj_id_to_2d_index = {int(obj_id): idx for idx, obj_id in enumerate(ids2d)}
        obj_id_to_3d_index = {int(obj_id): idx for idx, obj_id in enumerate(ids3d)}
        item["alignment.obj_id_to_2d_index"] = obj_id_to_2d_index
        item["alignment.obj_id_to_3d_index"] = obj_id_to_3d_index
        item["alignment.seq_to_obj_id"] = [int(obj_id) for obj_id in ids2d]

        # Log non-perfect alignment (typically from occluded/out-of-frame objects)
        if status == "partial":
            _log_debug_level_1(
                f"GraspNetEval: partial 2D/3D alignment (only_2d={only_2d.tolist()}, only_3d={only_3d.tolist()})."
            )
        elif status == "disjoint" and len(ids2d) and len(ids3d):
            logger.warning(f"GraspNetEval: disjoint 2D/3D object ID sets (2D={ids2d.tolist()}, 3D={ids3d.tolist()}).")

        # If already perfectly aligned in the same sequence, just record metadata and stop early
        if status == "perfect" and np.array_equal(ids2d, ids3d):
            item["alignment.only_2d_ids"] = only_2d
            item["alignment.only_3d_ids"] = only_3d
            item["alignment.obj_id_order_2d"] = ids2d.copy()
            item["alignment.obj_id_order_3d"] = ids3d.copy()
            item["alignment.auto_offset_correction"] = False
            item["alignment.obj_id_to_2d_index"] = {int(obj_id): idx for idx, obj_id in enumerate(ids2d)}
            item["alignment.obj_id_to_3d_index"] = {int(obj_id): idx for idx, obj_id in enumerate(ids3d)}
            item["alignment.seq_to_obj_id"] = [int(obj_id) for obj_id in ids2d]
            item["alignment.method"] = "ids"
            return

        # Decide whether to reorder stacked points:
        # - If perfect ID match: reorder by IDs
        # - Else, if counts match and we have masks+poses+intrinsics: try centroid-based matching
        # Disable centroid-based geometric matching; keep alignment purely ID-driven
        try_centroid = False

        if not ((status == "perfect" and self.collate_points == "stack") or try_centroid):
            return

        # Need points data with lengths and occupancy
        if "points" not in item or "points.lengths" not in item or "points.occ" not in item:
            return

        lengths = item["points.lengths"]
        pts_cat = item["points"]
        occ_cat = item["points.occ"]
        labels_cat = item.get("points.labels")  # may be None

        if not isinstance(lengths, (list, tuple)):
            return
        num_obj_segments = len(ids3d)
        if len(lengths) == num_obj_segments + 1:
            pass
        elif len(lengths) != num_obj_segments:
            # Unexpected layout; skip reordering
            return

        # Slice concatenated arrays into per-object segments (current 3D order)
        segments_pts: list[np.ndarray] = []
        segments_occ: list[np.ndarray] = []
        segments_labels: list[np.ndarray] = []
        start = 0
        for _i_seg, L in enumerate(lengths):
            end = start + L
            segments_pts.append(pts_cat[start:end])
            segments_occ.append(occ_cat[start:end])
            if labels_cat is not None and len(labels_cat) == len(pts_cat):
                segments_labels.append(labels_cat[start:end])
            start = end

        # Simplified mode: no late reordering here; upstream populate methods already emit 2D-ordered data.
        return

    def _split_scene_ids(self, split: str) -> set | None:
        """Get the set of scene IDs for a given split."""
        if self.scene_ids is not None:
            return set(self.scene_ids)

        split_map = {
            "train": set(range(0, 100)),
            "val": set(range(100, 130)),
            "test": set(range(100, 190)),
            "test_seen": set(range(100, 130)),
            "test_similar": set(range(130, 160)),
            "test_novel": set(range(160, 190)),
            "all": None,
        }
        if split in split_map:
            return split_map[split]
        logger.warning("Unknown split '%s'; defaulting to all scenes.", split)
        return None

    def _enumerate_samples(self, split: str) -> list[_Sample]:
        """Enumerate all depth frame samples for a given split."""
        target_ids = self._split_scene_ids(split)
        samples: list[_Sample] = []

        for scene_dir in self.scene_dirs:
            scene_id = self._scene_id(scene_dir)
            if scene_id is None or (target_ids is not None and scene_id not in target_ids):
                continue

            depth_dir = _camera_dir(scene_dir, self.camera, "depth")
            ann_ids = sorted(p.stem for p in depth_dir.glob("*.png"))

            if self.one_view_per_scene and len(ann_ids) > 0:
                # Randomly select one view per scene
                selected_ann_id = np.random.choice(ann_ids)
                samples.append(_Sample(scene_dir=scene_dir, ann_id=selected_ann_id))
            else:
                # Use all views
                samples.extend(_Sample(scene_dir=scene_dir, ann_id=ann_id) for ann_id in ann_ids)

        if not samples:
            raise FileNotFoundError(f"No depth frames found for split='{split}' and camera='{self.camera}'.")
        return samples

    @staticmethod
    def _scene_id(scene_dir: Path) -> int | None:
        """Extract numeric scene ID from scene directory name."""
        try:
            return int(scene_dir.name.split("_")[-1])
        except ValueError:
            logger.warning("Could not parse scene id from %s.", scene_dir)
            return None

    def _depth_path(self, sample: _Sample) -> Path:
        """Get the path to the depth image for a sample."""
        return _camera_dir(sample.scene_dir, self.camera, "depth") / f"{sample.ann_id}.png"

    def _meta_path(self, sample: _Sample) -> Path:
        """Get the path to the metadata file for a sample."""
        return _camera_dir(sample.scene_dir, self.camera, "meta") / f"{sample.ann_id}.mat"

    def _label_path(self, sample: _Sample) -> Path:
        """Get the path to the label/mask file for a sample."""
        return _camera_dir(sample.scene_dir, self.camera, "label") / f"{sample.ann_id}.png"

    def _load_depth_and_meta(self, sample: _Sample) -> tuple[np.ndarray, dict[str, Any]]:
        """Load both depth image and metadata for a sample."""
        meta = self._load_meta(sample)
        depth_scale = float(meta.get("factor_depth", self.depth_scale))
        depth_path = self._depth_path(sample)
        depth = _load_depth(depth_path, depth_scale)
        return depth, meta

    def _load_meta(self, sample: _Sample) -> dict[str, Any]:
        """Load and cache metadata for a sample."""
        cache_key = (sample.scene_dir, sample.ann_id)
        if cache_key in self._meta_cache:
            return self._meta_cache[cache_key]

        meta_path = self._meta_path(sample)
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta file {meta_path}.")
        meta = loadmat(meta_path, squeeze_me=True)
        self._meta_cache[cache_key] = meta
        return meta

    def _load_intrinsic(self, scene_dir: Path) -> np.ndarray:
        """Load and cache camera intrinsic matrix for a scene."""
        if scene_dir in self._intrinsic_cache:
            return self._intrinsic_cache[scene_dir]

        camk_path = scene_dir / self.camera / "camK.npy"
        if not camk_path.exists():
            logger.warning("Intrinsic matrix not found at %s; using identity.", camk_path)
            intrinsic = np.eye(3, dtype=np.float32)
        else:
            intrinsic = np.load(camk_path).astype(np.float32)
        self._intrinsic_cache[scene_dir] = intrinsic
        return intrinsic

    def _load_label(self, sample: _Sample) -> np.ndarray | None:
        """
        Load segmentation label/mask for a sample.

        GraspNet labels are instance segmentation maps where each pixel value
        corresponds to an object ID (0 = background, 1+ = object instances).
        """
        label_path = self._label_path(sample)
        if not label_path.exists():
            logger.warning("Label file not found at %s.", label_path)
            return None
        # Load as integer array (uint8 or uint16)
        label = np.array(Image.open(label_path))
        return label

    def _load_camera_pose(self, scene_dir: Path, frame_id: int) -> np.ndarray | None:
        """Load camera-to-world pose for a specific frame."""
        if scene_dir not in self._pose_cache:
            pose_path = scene_dir / self.camera / "camera_poses.npy"
            if not pose_path.exists():
                logger.warning("Camera poses not found at %s.", pose_path)
                self._pose_cache[scene_dir] = None
                self._table_from_world[scene_dir] = None
            else:
                poses = np.load(pose_path).astype(np.float32)
                self._pose_cache[scene_dir] = poses
                # Load alignment matrix (cam0_wrt_table) if available
                table_path = scene_dir / self.camera / "cam0_wrt_table.npy"
                if table_path.exists():
                    cam0_wrt_table = np.load(table_path).astype(np.float32)
                    self._table_from_world[scene_dir] = cam0_wrt_table
                else:
                    logger.warning("cam0_wrt_table.npy not found at %s; no table alignment available.", table_path)
                    self._table_from_world[scene_dir] = None

        poses = self._pose_cache[scene_dir]
        if poses is None:
            return None
        if frame_id >= len(poses):
            logger.warning("Frame %d exceeds pose array length %d for scene %s.", frame_id, len(poses), scene_dir)
            return None
        return poses[frame_id]

    def _crop_points_to_mesh(self, item: dict[str, Any]) -> None:
        """
        Crop the projected point cloud to the mesh bounding box with padding.

        This removes background points and focuses on the objects of interest.
        Also crops corresponding labels if they exist.
        """
        if "inputs" not in item or "mesh.vertices" not in item:
            return

        points = item["inputs"]
        mesh_vertices = item["mesh.vertices"]

        # Compute mesh bounding box with padding
        bbox_min = mesh_vertices.min(axis=0) - self.crop_padding
        bbox_max = mesh_vertices.max(axis=0) + self.crop_padding

        # Filter points within bounding box
        mask = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
        item["inputs"] = points[mask]

        # Also crop labels if they exist (from 2D label projection)
        if "inputs.labels" in item and isinstance(item["inputs.labels"], np.ndarray):
            if len(item["inputs.labels"]) == len(points):
                item["inputs.labels"] = item["inputs.labels"][mask]

    def _populate_masks(self, item: dict[str, Any], sample: _Sample) -> None:
        """
        Load and populate instance masks in COCO-like format.

        Converts GraspNet instance segmentation maps to COCO-style annotations:
        - inputs.masks: (N, H, W) binary masks or (H, W) stacked map if stack_2d=True
        - inputs.boxes: (N, 4) bounding boxes in (x, y, w, h) format
        - inputs.labels: (N,) sequential instance indices (1, 2, 3, ...) remapped from original IDs
        - inputs.info: List of dicts with metadata (includes both original instance_id and remapped label)

        Note: GraspNet label values can be non-sequential (e.g., 5, 17, 75).
        We remap them to sequential indices (1, 2, 3, ...) for compatibility with visualization.
        Original IDs are preserved in inputs.info[i]["instance_id"].
        """
        label = self._load_label(sample)
        if label is None:
            return

        # Store original label map
        item["label"] = label
        item["label.name"] = self._label_path(sample).name
        item["label.path"] = self._label_path(sample)

        # Get image dimensions
        height, width = label.shape

        # Convert to COCO-like format: extract individual masks and boxes
        instance_ids = np.unique(label)
        instance_ids = instance_ids[instance_ids > 0]  # Exclude background

        # Keep raw 2D instance indices (1..N) for mapping & debugging
        if instance_ids.size > 0:
            item["inputs.instance_ids_2d"] = instance_ids.copy()

        if len(instance_ids) > 0:
            # Create mapping from original IDs to sequential indices (1, 2, 3, ...)
            id_to_index = {orig_id: idx + 1 for idx, orig_id in enumerate(instance_ids)}

            # Extract binary masks for each instance (N, H, W)
            masks = []
            boxes = []
            labels = []
            info = []

            for instance_id in instance_ids:
                # Binary mask for this instance
                mask = label == instance_id
                masks.append(mask)

                # Compute bounding box (x, y, w, h) format like COCO
                rows, cols = np.where(mask)
                if len(rows) > 0:
                    y_min, y_max = rows.min(), rows.max()
                    x_min, x_max = cols.min(), cols.max()
                    w = x_max - x_min + 1
                    h = y_max - y_min + 1
                    boxes.append([x_min, y_min, w, h])
                    area = mask.sum()
                else:
                    boxes.append([0, 0, 0, 0])
                    area = 0

                # Use sequential index instead of original ID
                sequential_id = id_to_index[instance_id]
                labels.append(sequential_id)
                info.append({"area": int(area), "instance_id": int(instance_id), "label": int(sequential_id)})

            # Store in COCO-like format
            item["inputs.masks"] = np.stack(masks, axis=0).astype(np.uint8)  # (N, H, W)
            item["inputs.boxes"] = np.array(boxes, dtype=np.float32)  # (N, 4)
            item["inputs.labels"] = np.array(labels, dtype=np.int64)  # (N,) - sequential 1, 2, 3, ...
            item["inputs.info"] = info  # List of dicts with metadata

            # In GraspNet masks, pixel value v corresponds to model id v-1 (0..87), 0 is background
            item["inputs.obj_id_order_2d"] = np.array(instance_ids, dtype=np.int32) - 1
            item["inputs.instance_id_to_seq"] = id_to_index

            # If stack_2d, convert masks back to single instance map with remapped IDs
            if self.stack_2d:
                # Remap label values to sequential indices
                remapped_label = np.zeros_like(label, dtype=np.int64)
                for orig_id, seq_id in id_to_index.items():
                    remapped_label[label == orig_id] = seq_id
                item["inputs.masks"] = remapped_label
        else:
            # No instances found
            item["inputs.masks"] = np.zeros((0, height, width), dtype=np.uint8)
            item["inputs.boxes"] = np.zeros((0, 4), dtype=np.float32)
            item["inputs.labels"] = np.zeros((0,), dtype=np.int64)
            item["inputs.info"] = []

    def _populate_camera_transforms(self, item: dict[str, Any], scene_dir: Path, frame_id: int) -> None:
        """
        Populate camera transformation matrices in the item dictionary.

        GraspNet stores camera-to-world transforms, but standard CV uses world-to-camera extrinsics.
        This method stores both conventions:
        - cam_to_world/cam_to_table: for transforming mesh vertices
        - extrinsic_world/extrinsic_table: inverted, for depth projection
        """
        cam_to_world = self._load_camera_pose(scene_dir, frame_id)
        if cam_to_world is None:
            return

        cam0_wrt_table = self._table_from_world.get(scene_dir)

        # Store camera-to-world transform
        item["inputs.cam_to_world"] = cam_to_world
        item["inputs.extrinsic_world"] = inv_trafo(cam_to_world)

        # Apply table alignment if available (following GraspNet API)
        if cam0_wrt_table is not None:
            cam_to_table = cam0_wrt_table @ cam_to_world
            item["inputs.cam_to_table"] = cam_to_table
            item["inputs.cam0_wrt_table"] = cam0_wrt_table
            item["inputs.extrinsic_table"] = inv_trafo(cam_to_table)
            item["inputs.extrinsic"] = item["inputs.extrinsic_table"]
        else:
            item["inputs.extrinsic"] = item["inputs.extrinsic_world"]

    def _populate_meshes(self, item: dict[str, Any], meta: dict[str, Any]) -> None:
        """Load and transform object meshes in multiple coordinate frames."""
        # Prefer scene-level object list for stable model IDs
        obj_ids_raw = item.get("scene.object_id_list")
        if obj_ids_raw is None or len(obj_ids_raw) == 0:
            obj_ids_raw = self._extract_obj_ids(meta)
        obj_ids_arr = np.asarray(obj_ids_raw, dtype=np.int32)
        if len(obj_ids_arr) == 0:
            return

        # Keep canonical scene object list for potential diagnostics or mapping
        item["scene.object_id_list"] = obj_ids_arr.copy()

        # If 2D order exists, reorder per-object arrays before any concatenation
        obj_ids = obj_ids_arr.tolist()
        perm = self._compute_obj_permutation(item, obj_ids)
        if perm is not None and len(perm) == len(obj_ids_arr):
            obj_ids_arr = obj_ids_arr[perm]
            obj_ids = obj_ids_arr.tolist()

        obj_poses = self._extract_obj_poses(meta, len(obj_ids))
        if perm is not None and obj_poses.shape[0] == len(perm):
            obj_poses = obj_poses[perm]

        # Store object poses in camera frame for potential 2D-3D centroid matching
        item["mesh.obj_poses_cam"] = obj_poses.copy()
        mesh_paths = self._get_mesh_paths(obj_ids)

        # Store 3D object ID order for later 2D-3D alignment
        item["mesh.obj_id_order_3d"] = obj_ids_arr.copy()

        cam_to_world = item.get("inputs.cam_to_world")
        cam_to_table = item.get("inputs.cam_to_table")

        # Transform meshes to different coordinate frames
        vertices_cam, vertices_world, vertices_table, faces_list = self._transform_meshes(
            mesh_paths, obj_poses, cam_to_world, cam_to_table
        )

        if not vertices_cam:
            return

        # Store mesh data
        v_counts = [v.shape[0] for v in vertices_cam]
        f_counts = [f.shape[0] for f in faces_list]
        item["mesh.vertices_cam"] = np.concatenate(vertices_cam, axis=0)
        item["mesh.triangles"] = np.concatenate(faces_list, axis=0)

        if vertices_world:
            item["mesh.vertices_world"] = np.concatenate(vertices_world, axis=0)
        if vertices_table:
            item["mesh.vertices_table"] = np.concatenate(vertices_table, axis=0)

        # Set default vertices (prefer table > world > cam)
        if vertices_table:
            item["mesh.vertices"] = item["mesh.vertices_table"]
        elif vertices_world:
            item["mesh.vertices"] = item["mesh.vertices_world"]
        else:
            item["mesh.vertices"] = item["mesh.vertices_cam"]

        # Per-object lengths for downstream per-instance processing
        item["mesh.num_vertices"] = v_counts
        item["mesh.num_triangles"] = f_counts

        item["mesh.name"] = mesh_paths[0].stem
        item["mesh.path"] = mesh_paths[0]

        # Load and transform surface point clouds if requested
        if self.load_pcd:
            self._populate_pcd(item, obj_ids, mesh_paths, obj_poses, cam_to_world, cam_to_table)

        # Load and transform occupancy points if requested
        if self.load_points:
            self._populate_points(item, obj_ids, mesh_paths, obj_poses, cam_to_world, cam_to_table)

    def _populate_pcd(
        self,
        item: dict[str, Any],
        obj_ids: np.ndarray,
        mesh_paths: list[Path],
        obj_poses: np.ndarray,
        cam_to_world: np.ndarray | None,
        cam_to_table: np.ndarray | None,
    ) -> None:
        """Load and transform surface point clouds for meshes."""
        pcd_list: list[np.ndarray] = []
        pcd_labels: list[np.ndarray] = []

        for _obj_id, mesh_path, obj_pose in zip(obj_ids, mesh_paths, obj_poses, strict=False):
            # Always derive surface pointclouds from meshes (no external files)
            mesh = self._load_mesh_cached(mesh_path)
            pcd = np.asarray(mesh.sample(self.pcd_num_points), dtype=np.float32)

            # Transform point cloud from normalized object space to scene space
            # Apply object pose to get points in camera frame
            pcd_cam = np.asarray(apply_trafo(pcd, obj_pose), dtype=np.float32)

            # Optionally transform to table frame (preferred)
            if cam_to_table is not None:
                pcd_table = np.asarray(apply_trafo(pcd_cam, cam_to_table), dtype=np.float32)
                pcd_list.append(pcd_table)
            elif cam_to_world is not None:
                pcd_world = np.asarray(apply_trafo(pcd_cam, cam_to_world), dtype=np.float32)
                pcd_list.append(pcd_world)
            else:
                pcd_list.append(pcd_cam)

            # Per-object labels (sequential 1..N)
            pcd_labels.append(np.full((len(pcd_list[-1]),), len(pcd_list), dtype=np.int64))

        if not pcd_list:
            return

        # Concatenate all point clouds and record lengths + labels
        item["pointcloud"] = np.concatenate(pcd_list, axis=0)
        item["pointcloud.lengths"] = [len(p) for p in pcd_list]
        item["pointcloud.labels"] = np.concatenate(pcd_labels, axis=0) if pcd_labels else np.zeros((0,), np.int64)
        item["pointcloud.name"] = "sampled"
        item["pointcloud.path"] = str(self.mesh_dir)

    def _sample_scene_points(self, mesh_vertices: np.ndarray) -> np.ndarray:
        """
        Sample points in a volume around all meshes in the scene.

        Similar to tabletop.py, samples points uniformly in a cube or sphere.
        Uses the same formula: bound = (0.5 + padding/2) * scale_factor

        Parameters
        ----------
        mesh_vertices : np.ndarray
            All mesh vertices in the scene (concatenated), shape (N, 3)

        Returns
        -------
        scene_points : np.ndarray
            Sampled points in scene coordinates, shape (num_scene_points, 3)
        """
        # Compute scene center from all mesh vertices
        bbox_min = mesh_vertices.min(axis=0)
        bbox_max = mesh_vertices.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2

        # Use same formula as tabletop.py
        bound = (0.5 + self.padding / 2.0) * self.scale_factor

        if self.scene_points_volume == "cube":
            # Sample uniformly in a cube
            scene_points = np.random.uniform(-bound, bound, size=(self.num_scene_points, 3))
            scene_points += bbox_center
        elif self.scene_points_volume == "sphere":
            # Sample uniformly in a sphere
            dirs = np.random.normal(size=(self.num_scene_points, 3)).astype(np.float32)
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0  # avoid divide-by-zero
            dirs /= norms
            radii = (np.random.rand(self.num_scene_points, 1).astype(np.float32)) ** (1.0 / 3.0) * bound
            scene_points = (dirs * radii) + bbox_center
        else:
            raise ValueError(f"Unknown scene_points_volume: {self.scene_points_volume}")

        return scene_points.astype(np.float32)

    def _populate_points(
        self,
        item: dict[str, Any],
        obj_ids: np.ndarray,
        mesh_paths: list[Path],
        obj_poses: np.ndarray,
        cam_to_world: np.ndarray | None,
        cam_to_table: np.ndarray | None,
    ) -> None:
        """Load and transform occupancy points for meshes."""
        points_list: list[np.ndarray] = []
        occ_list: list[np.ndarray] = []

        for obj_id, mesh_path, obj_pose in zip(obj_ids, mesh_paths, obj_poses, strict=False):
            # Load or generate points
            points, occupancy = self._get_or_generate_points(mesh_path, int(obj_id))

            if len(points) == 0:
                # No points available
                continue

            # Transform points from normalized object space to scene space
            # Apply object pose to get points in camera frame
            points_cam = np.asarray(apply_trafo(points, obj_pose), dtype=np.float32)

            # Optionally transform to table frame (preferred)
            if cam_to_table is not None:
                points_table = np.asarray(apply_trafo(points_cam, cam_to_table), dtype=np.float32)
                points_list.append(points_table)
            elif cam_to_world is not None:
                points_world = np.asarray(apply_trafo(points_cam, cam_to_world), dtype=np.float32)
                points_list.append(points_world)
            else:
                points_list.append(points_cam)

            occ_list.append(occupancy)

        if not points_list:
            # No per-object points, but we might still have scene points
            if not self.sample_scene_points:
                return
            all_points = []
            all_occ = []
            all_labels = []
        else:
            # Collate per-object points
            if self.collate_points == "stack":
                # Create instance labels like tabletop.py. We align instance IDs
                # to the order of 3D objects here; a later alignment step may
                # reorder 3D objects to match 2D instances when available.
                all_points = []
                all_occ = []
                all_labels = []

                for instance_id, (pts, occ) in enumerate(zip(points_list, occ_list, strict=False), start=1):
                    labels = np.zeros_like(occ, dtype=np.int64)
                    labels[occ] = instance_id  # Occupied points get instance ID
                    all_points.append(pts)
                    all_occ.append(occ)
                    all_labels.append(labels)
            else:
                # Simple concatenation
                all_points = [np.concatenate(points_list, axis=0)]
                all_occ = [np.concatenate(occ_list, axis=0)]
                all_labels = []

        # Sample scene-level points if requested
        if self.sample_scene_points:
            # Get mesh vertices for scene bounding box (use preferred frame)
            if "mesh.vertices" in item and len(item["mesh.vertices"]) > 0:
                mesh_vertices = item["mesh.vertices"]

                # Sample points in scene volume
                scene_points = self._sample_scene_points(mesh_vertices)

                # Check occupancy against all meshes
                # Need to check against individual meshes to assign instance labels
                scene_occ = np.zeros(len(scene_points), dtype=bool)
                scene_labels = np.zeros(len(scene_points), dtype=np.int64)

                if self.collate_points == "stack":
                    # Check each mesh and assign instance labels
                    for instance_id, (mesh_path, obj_pose) in enumerate(
                        zip(mesh_paths, obj_poses, strict=False), start=1
                    ):
                        # Load mesh and transform to scene coordinates
                        mesh = self._load_mesh_cached(mesh_path).copy()

                        # Transform mesh to scene frame (same as mesh vertices)
                        if cam_to_table is not None:
                            mesh.apply_transform(cam_to_table @ obj_pose)
                        elif cam_to_world is not None:
                            mesh.apply_transform(cam_to_world @ obj_pose)
                        else:
                            mesh.apply_transform(obj_pose)

                        # Check which scene points are inside this mesh
                        mesh_occ = check_mesh_contains(mesh, scene_points)

                        # Update labels for points inside this mesh
                        scene_labels[mesh_occ] = instance_id
                        scene_occ |= mesh_occ
                else:
                    # Just check occupancy, no instance labels
                    for mesh_path, obj_pose in zip(mesh_paths, obj_poses, strict=False):
                        mesh = self._load_mesh_cached(mesh_path).copy()

                        if cam_to_table is not None:
                            mesh.apply_transform(cam_to_table @ obj_pose)
                        elif cam_to_world is not None:
                            mesh.apply_transform(cam_to_world @ obj_pose)
                        else:
                            mesh.apply_transform(obj_pose)

                        scene_occ |= check_mesh_contains(mesh, scene_points)

                # Add scene points to the lists
                all_points.append(scene_points)
                all_occ.append(scene_occ)
                if self.collate_points == "stack":
                    all_labels.append(scene_labels)

        # Concatenate all points and record lengths
        if all_points:
            item["points"] = np.concatenate(all_points, axis=0)
            item["points.occ"] = np.concatenate(all_occ, axis=0)
            item["points.lengths"] = [len(p) for p in all_points]
            if self.collate_points == "stack" and all_labels:
                item["points.labels"] = np.concatenate(all_labels, axis=0)

            item["points.name"] = "generated"
            item["points.path"] = "memory"  # Points are cached in memory, not on disk

    def _transform_meshes(
        self,
        mesh_paths: list[Path],
        obj_poses: np.ndarray,
        cam_to_world: np.ndarray | None,
        cam_to_table: np.ndarray | None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Transform meshes into camera, world, and table coordinate frames."""
        vertices_cam: list[np.ndarray] = []
        vertices_world: list[np.ndarray] = []
        vertices_table: list[np.ndarray] = []
        faces_list: list[np.ndarray] = []
        vertex_offset = 0

        for path, obj_pose in zip(mesh_paths, obj_poses, strict=False):
            base_mesh = self._load_mesh_cached(path)

            # Camera frame: apply object pose
            mesh_cam = base_mesh.copy()
            mesh_cam.apply_transform(obj_pose)
            vertices_cam.append(mesh_cam.vertices.copy())

            # World frame: compose cam_to_world @ obj_pose (following GraspNet API)
            if cam_to_world is not None:
                mesh_world = base_mesh.copy()
                mesh_world.apply_transform(cam_to_world @ obj_pose)
                vertices_world.append(mesh_world.vertices.copy())

            # Table frame: compose cam_to_table @ obj_pose
            if cam_to_table is not None:
                mesh_table = base_mesh.copy()
                mesh_table.apply_transform(cam_to_table @ obj_pose)
                vertices_table.append(mesh_table.vertices.copy())

            faces_list.append(base_mesh.faces + vertex_offset)
            vertex_offset += len(base_mesh.vertices)

        return vertices_cam, vertices_world, vertices_table, faces_list

    @staticmethod
    def _extract_obj_ids(meta: dict[str, Any]) -> np.ndarray:
        """Extract object IDs from metadata, handling different key names."""
        for key in ("obj_ids", "object_ids", "objectid_list", "cls_indexes"):
            if key in meta:
                ids = np.atleast_1d(meta[key]).astype(np.int32)
                # cls_indexes are 1-indexed, convert to 0-indexed
                if key == "cls_indexes":
                    ids = ids - 1
                return ids
        logger.warning("No object IDs found in meta; returning empty array.")
        return np.zeros((0,), dtype=np.int32)

    @staticmethod
    def _extract_obj_poses(meta: dict[str, Any], num_objects: int) -> np.ndarray:
        """
        Extract and normalize object poses from metadata.

        GraspNet stores poses in various formats; this method normalizes them to (N, 4, 4).
        """
        # Try to find poses in metadata
        poses = None
        for key in ("poses", "object_poses", "pose"):
            if key in meta:
                poses = np.asarray(meta[key], dtype=np.float32)
                break

        if poses is None:
            return np.eye(4, dtype=np.float32)[None, ...].repeat(num_objects, axis=0)

        # Handle single pose: (4, 4) -> (1, 4, 4)
        if poses.ndim == 2:
            poses = poses[None, ...]

        # Handle transposed formats: (3, 4, N) or (4, 4, N) -> (N, 3, 4) or (N, 4, 4)
        if poses.ndim == 3 and poses.shape[2] == num_objects:
            poses = np.transpose(poses, (2, 0, 1))

        if poses.ndim != 3:
            logger.warning("Unexpected pose array shape %s; returning identity.", poses.shape)
            return np.eye(4, dtype=np.float32)[None, ...].repeat(num_objects, axis=0)

        # Convert (N, 3, 4) to (N, 4, 4) by adding homogeneous row
        if poses.shape[1:] == (3, 4):
            bottom = np.zeros((poses.shape[0], 1, 4), dtype=np.float32)
            bottom[:, 0, 3] = 1.0
            poses = np.concatenate([poses, bottom], axis=1)

        if poses.shape[0] != num_objects:
            logger.warning("Mismatch between number of poses (%d) and objects (%d).", poses.shape[0], num_objects)

        return poses

    def _get_mesh_paths(self, obj_ids: Sequence[int]) -> list[Path]:
        """Get mesh file paths for a list of object IDs."""
        return [self._resolve_mesh_path(int(obj_id)) for obj_id in obj_ids]

    def _load_mesh_cached(self, path: Path) -> Trimesh:
        """
        Load and cache a mesh from disk, optionally decimating it.

        Meshes are cached with their decimation fraction to avoid reprocessing.
        """
        effective_fraction = self.mesh_simplify_fraction if self._mesh_decimation_enabled else None
        cache_key = (path, effective_fraction)

        if cache_key not in self._mesh_cache:
            vertices, faces = load_mesh(path)
            if faces is None:
                raise ValueError(f"Loaded mesh {path} has no faces.")
            mesh = Trimesh(vertices, faces)
            original_face_count = len(faces)
            _log_debug_level_2(f"Loaded mesh {path.name} with {original_face_count} faces.")

            # Decimate mesh if requested
            if effective_fraction is not None:
                target_face_count = int(original_face_count * effective_fraction)
                # Only decimate if there's significant reduction
                if target_face_count < original_face_count:
                    try:
                        mesh = mesh.simplify_quadric_decimation(1 - effective_fraction)
                        _log_debug_level_2(
                            f"Decimated mesh {path.name} from {original_face_count} to {len(mesh.faces)} faces ({100.0 * len(mesh.faces) / original_face_count:.1f}%)"
                        )
                    except Exception as e:
                        self._mesh_decimation_enabled = False
                        self._mesh_decimation_disabled_reason = f"{type(e).__name__}: {e}"
                        logger.warning(
                            "Mesh decimation disabled after first failure "
                            f"({self._mesh_decimation_disabled_reason}). "
                            "Using original meshes for remaining objects."
                        )

            self._mesh_cache[cache_key] = mesh

        return self._mesh_cache[cache_key]

    def _generate_points_for_mesh(self, mesh: Trimesh) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate occupancy points for a mesh.

        Samples points based on points_sampling strategy:
        - "uniform": Uniform random in padded bounding box
        - "surface": Near-surface with multiple noise levels
        - "both": Both uniform and surface samples

        Note: The mesh is normalized to a unit cube before sampling to ensure
        the sampling bounds match the mesh scale. Points are then transformed
        back to the original mesh space before being returned.
        """
        # Store original mesh bounds for inverse transformation
        original_center = mesh.bounds.mean(axis=0)
        original_scale = mesh.extents.max()

        # Normalize mesh to unit cube (critical for correct occupancy sampling)
        mesh_normalized = normalize_mesh(mesh.copy(), center=True, scale=True)

        points_list = []

        if self.points_sampling in ["uniform", "both"]:
            # Uniform random samples in padded box
            boxsize = 1 + self.points_padding
            points_uniform = np.random.rand(self.num_points, 3).astype(np.float32)
            points_uniform = boxsize * (points_uniform - 0.5)
            points_list.append(points_uniform)

        if self.points_sampling in ["surface", "both"]:
            # Surface samples with varying noise levels
            samples = []
            sigmas = [0.001, 0.0015, 0.0025, 0.01, 0.015, 0.025, 0.05, 0.1, 0.15, 0.25]
            num_points = self.num_points // len(sigmas)
            for sigma in sigmas:
                noise = sigma * np.random.standard_normal((num_points, 3)).astype(np.float32)
                samples.append(np.asarray(mesh_normalized.sample(num_points), dtype=np.float32) + noise)
            points_surface = np.concatenate(samples, axis=0)
            points_list.append(points_surface)

        points = np.concatenate(points_list, axis=0).astype(np.float32)

        # Check occupancy against normalized mesh
        occupancy = check_mesh_contains(mesh_normalized, points)

        # Transform points back to original mesh space
        # Inverse of normalization: scale up, then translate back
        points = points * original_scale + original_center

        return points, occupancy

    def _get_or_generate_points(self, mesh_path: Path, obj_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get cached points or generate them if needed.

        Points are cached in memory similar to meshes, avoiding disk I/O.
        """
        # Include sampling strategy in cache key to avoid conflicts
        cache_key = (mesh_path, self.points_sampling)

        # Check memory cache
        if cache_key in self._points_cache:
            return self._points_cache[cache_key]

        # Generate points if requested
        if not self.generate_points:
            # Return empty if not generating
            empty_result = (np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool))
            self._points_cache[cache_key] = empty_result
            return empty_result

        # Generate and cache points
        mesh = self._load_mesh_cached(mesh_path)
        points, occupancy = self._generate_points_for_mesh(mesh)

        self._points_cache[cache_key] = (points, occupancy)
        _log_debug_level_2(f"Generated and cached points for obj {obj_id} (occ={occupancy.sum()})")

        return points, occupancy

    def _resolve_mesh_path(self, obj_id: int) -> Path:
        """Resolve the file path for a mesh given an object ID."""
        obj_str = f"{obj_id:03d}"
        candidates = [
            self.mesh_dir / obj_str / "nontextured.ply",
            self.mesh_dir / obj_str / "textured.obj",
            self.mesh_dir / f"obj_{obj_id:06d}.ply",
            self.mesh_dir / f"obj_{obj_id:03d}.ply",
        ]
        for path in candidates:
            if path.exists():
                return path
        logger.warning("Mesh for obj_id=%s not found in %s.", obj_id, self.mesh_dir)
        return candidates[0]

    def __repr__(self):
        repr_str = super().__repr__()
        config_str = "\nConfiguration:\n"
        config_str += f"  split: {self.split}\n"
        config_str += f"  camera: {self.camera}\n"
        config_str += f"  project: {self.project}\n"
        config_str += f"  load_mesh: {self.load_mesh}\n"
        if self.mesh_simplify_fraction is not None:
            if self._mesh_decimation_enabled:
                config_str += f"  mesh_simplify_fraction: {self.mesh_simplify_fraction:.2%}\n"
            else:
                reason = self._mesh_decimation_disabled_reason or "disabled"
                config_str += f"  mesh_simplify_fraction: disabled ({reason})\n"
        if self.crop_to_mesh:
            config_str += f"  crop_to_mesh: True (padding={self.crop_padding}m)\n"
        if self.load_pcd:
            config_str += f"  load_pcd: True (sampling {self.pcd_num_points} points)\n"
        if self.load_points:
            config_str += "  load_points: True"
            if self.generate_points:
                config_str += f" (generating {self.points_sampling}, num_points={self.num_points})"
            if self.collate_points:
                config_str += f" (collate={self.collate_points})"
            config_str += "\n"
        if self.sample_scene_points:
            config_str += f"  sample_scene_points: True ({self.scene_points_volume}, num={self.num_scene_points}, padding={self.padding}, scale={self.scale_factor})\n"
        if self.load_label:
            config_str += "  load_label: True"
            if self.stack_2d:
                config_str += " (stack_2d=True)"
            config_str += "\n"
        if self.one_view_per_scene:
            config_str += "  one_view_per_scene: True\n"
        if self.filter_background is not None:
            config_str += f"  remove_background_z_above: {self.filter_background}\n"
        repr_str += config_str

        if isinstance(self.transforms, list) and self.transforms:
            trafo_str = "\n3D Transformations:\n"
            for trafo in self.transforms:
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
        if self.transforms is not None:
            return repr_str + "\n3D Transformations:\n  callable\n"
        return repr_str


__all__ = ["GraspNetEval"]
