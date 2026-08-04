import blenderproc as bproc  # noqa: I001  # pyright: ignore[reportMissingImports]

import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast
from collections.abc import Callable

import bpy
import cv2
import numpy as np
import trimesh
import tyro
from blenderproc.python.types.MeshObjectUtility import MeshObject  # pyright: ignore[reportMissingImports]
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping  # pyright: ignore[reportMissingImports]
from blenderproc.python.utility.Utility import Utility, stdout_redirected  # pyright: ignore[reportMissingImports]
from loguru import logger
from PIL import Image
from scipy.stats import truncnorm
from tqdm import trange


def inv_trafo(trafo: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a 4x4 homogeneous transformation matrix (or a batch of them).

    Args:
        trafo: A (4, 4) or (B, 4, 4) transformation matrix (PyTorch Tensor or NumPy ndarray).
               Assumes the bottom row is [0, 0, 0, 1].

    Returns:
        The inverse transformation matrix (or batch of matrices) with the same type as input.
    """
    if trafo.shape[-2:] != (4, 4):
        raise ValueError(f"Input matrices must be of shape (..., 4, 4), got {trafo.shape}")

    rot = trafo[..., :3, :3]
    trans = trafo[..., :3, 3]
    rot_t = rot.swapaxes(-1, -2)

    identity = np.eye(3, dtype=rot.dtype)
    if rot.ndim == 3:
        identity = np.broadcast_to(identity, rot.shape)
    if not np.allclose(rot @ rot_t, identity, atol=1e-6):
        return np.linalg.inv(trafo)

    if trafo.ndim == 3:
        inverse = np.broadcast_to(np.eye(4), trafo.shape)
        inverse[..., :3, :3] = rot_t
        inverse[..., :3, 3] = (-rot_t @ trans[..., None]).squeeze(-1)
    elif trafo.ndim == 2:
        inverse = np.eye(4)
        inverse[:3, :3] = rot_t
        inverse[:3, 3] = -rot_t @ trans
    else:
        raise ValueError(f"Input must have 2 or 3 dimensions, got {trafo.ndim}")

    return inverse


def convert_coordinates(points: np.ndarray, input_format: str, output_format: str) -> np.ndarray:
    if input_format.lower() == output_format.lower():
        return points

    # Transformation matrices from OpenGL to other formats
    transforms = {
        "opengl": np.eye(3),
        "opencv": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # Invert y and z axes
        "blender": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),  # Swap y and z axes, invert y axis
    }

    # Validate formats
    assert input_format.lower() in transforms, f"Unknown input format: {input_format}"
    assert output_format.lower() in transforms, f"Unknown output format: {output_format}"

    # Validate shape
    assert points.shape[-1] == 3, "Point coordinates must be 3D."
    assert points.ndim in [2, 3], "Point coordinates (N, 3) or (B, N, 3)."

    # Select transformation matrices
    in_trafo = transforms[input_format.lower()]
    out_trafo = transforms[output_format.lower()]

    # Convert from input format to OpenGL
    points_opengl = points @ in_trafo.T

    # Convert from OpenGL to output format
    return points_opengl @ out_trafo


def convert_extrinsic(extrinsic: np.ndarray, input_format: str, output_format: str) -> np.ndarray:
    inv_extrinsic = inv_trafo(extrinsic)
    inv_extrinsic[:3, :3] = convert_coordinates(inv_extrinsic[:3, :3], input_format, output_format)
    return inv_trafo(inv_extrinsic)


def sample_truncnorm(scale: tuple[float, float], size: int = 3) -> np.ndarray:
    low, high = float(scale[0]), float(scale[1])
    if low > high:
        low, high = high, low

    if np.isclose(low, high):
        return np.full((size,), low, dtype=np.float32)

    mean = 0.5 * (low + high)
    std = 0.5 * (high - low)
    a, b = (low - mean) / std, (high - mean) / std
    value = float(truncnorm.rvs(a, b, loc=mean, scale=std))
    return np.full((size,), value, dtype=np.float32)


ScaleSpec = float | tuple[float, float] | tuple[float, float, float] | None
ScaleSampler = Callable[[], np.ndarray | list[float] | tuple[float, float, float]]


@dataclass
class Camera:
    """Configure camera for depth image processing."""

    fx: float | None = None
    """Focal length in the x-direction"""
    fy: float | None = None
    """Focal length in the y-direction"""
    cx: float | None = None
    """Principal point in the x-direction"""
    cy: float | None = None
    """Principal point in the y-direction"""
    width: int | None = 512
    """Width of the image."""
    height: int | None = 512
    """Height of the image."""
    position: tuple[float, float, float] | np.ndarray | None = (7.35889, -6.92579, 4.95831)
    """Camera position in meters."""
    rotation: tuple[float, float, float] | np.ndarray | None = (63.5593, 0.0, 46.6919)
    """Camera rotation as XYZ Euler angles in degrees."""
    inplane_rotation: float | None = None
    """In-plane rotation in degrees."""
    near: float | None = None
    """Near clipping plane distance."""
    far: float | None = None
    """Far clipping plane distance."""
    intrinsics: np.ndarray | None = None
    """Camera intrinsics matrix."""
    extrinsics: int | np.ndarray | None = None
    """Camera extrinsics matrices or number of random poses to sample."""
    convention: Literal["opencv", "opengl", "blender"] = "opencv"
    """Coordinate system convention for the camera."""
    file: Path | None = None
    """File path to load camera parameters from."""
    sampler: Literal["sphere", "part_sphere", "shell"] | None = None
    """Sampler to use for generating camera poses."""
    jitter: tuple[float, float] | None = None
    """Min/max jitter to be applied to the camera position when sampling."""

    def __post_init__(self):
        if self.file and self.file.is_file():
            self.from_file(self.file)
        elif self.sampler:
            self.sampler = getattr(bproc.sampler, self.sampler)
            if self.extrinsics is None:
                self.extrinsics = 1

        width = int(self.width) if self.width is not None else 512
        height = int(self.height) if self.height is not None else 512
        self.width = width
        self.height = height

        if self.fx is None:
            self.fx = float(width)
        if self.fy is None:
            self.fy = float(width)
        if self.cx is None:
            self.cx = width / 2
        if self.cy is None:
            self.cy = height / 2
        if self.intrinsics is None:
            self.intrinsics = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        if self.rotation is not None and not isinstance(self.rotation, np.ndarray):
            self.rotation = np.deg2rad(np.asarray(self.rotation, dtype=np.float32))

    def from_file(self, file_path: Path):
        if file_path.suffix in [".npy", ".npz"]:
            self.data = np.load(file_path)
            for key, value in self.data.items():
                if "intr" in key:
                    self.fx = value[0, 0]
                    self.fy = value[1, 1]
                    self.cx = value[0, 2]
                    self.cy = value[1, 2]
                    self.intrinsics = value
                if "extr" in key:
                    if value.ndim == 2:
                        self.position = value[:3, 3]
                        self.rotation = value[:3, :3]
                    else:
                        self.position = None
                        self.rotation = None
                        self.extrinsics = value
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


@dataclass
class Config:
    """Configuration for BlenderProc rendering."""

    object_path: Path | Literal["primitives"]
    """Path to the object file."""
    output_dir: Path
    """Output directory for the rendered images."""
    metadata_path: Path | None = None
    """Path to a metadata file."""
    normalize: bool = False
    """Normalize the object to fit into the unit cube."""
    shading: str = "auto"
    """Shading mode."""
    backface_culling: bool = False
    """Enable backface culling."""
    flip_normals: bool = False
    """Flip normals."""
    fix_normals: bool = False
    """Fix normals of the object."""
    clear_normals: bool = False
    """Clear custom split normals of the object."""
    cleanup: bool = False
    """Cleanup the object by removing unused vertices and faces."""
    validate: bool = False
    """Validate meshes after loading."""
    scale: ScaleSpec | ScaleSampler = None
    """Scale the object."""
    distort: float | None = None
    """Distort the object through non-uniform scaling."""
    position: tuple[float, float] | tuple[float, float, float] | None = None
    """Set min/max or xyz object position."""
    rotation: bool | tuple[float, float, float] | None = None
    """XYZ-Euler angles of the object in degrees or True for random rotation."""
    upright: bool = True
    """Constrain rotation to Z-axis only (objects stay upright). Preserves coordinate convention."""
    upright_x_rotation: float = 90.0
    """X-axis rotation in degrees applied before random yaw for upright placement."""
    upright_x_rotation_path_overrides: dict[str, float] = field(default_factory=dict)
    """Path-token overrides for source-specific upright X rotations in mixed asset lists."""
    spawn_height: tuple[float, float] | None = None
    """Min/max height for spawning objects above surface (overrides default 1-4m)."""
    spawn_bounds: tuple[float, float] | None = None
    """XY spawn bounds: fraction of surface for 'surface' placement (e.g., 0.4-0.6), absolute meters for 'volume' (e.g., -0.3 to 0.3)."""
    containment_walls: bool | None = None
    """Add temporary containment walls during physics simulation."""
    placement: Literal["surface", "surface_aabb", "volume", "sequential", "tower"] | None = None
    """Placement strategy: packed surface, collision-proxy-safe surface, volume, sequential drop, or tower."""
    placement_min_distance: float = 0.005
    """Minimum XY separation between world AABBs for surface_aabb placement."""
    placement_bounds_ratio_limit: float = 4.0
    """Reject objects whose evaluated and normalized bounds disagree by more than this factor."""
    pile_batch_size: int = 2
    """Number of vertically separated objects released per pile-settling step."""
    pile_drop_margin: tuple[float, float] = (0.01, 0.03)
    """Minimum and maximum clearance above the current pile for each release."""
    pile_settle_time: float = 0.5
    """Minimum simulated settling time after each pile batch."""
    pile_wall_margin: float = 0.01
    """Horizontal margin between the pile spawn region and temporary walls."""
    pile_center_radius: float | None = 0.0
    """Optional maximum absolute XY center offset for pile releases."""
    collision_shape: Literal["CONVEX_HULL", "MESH", "BOX", "COMPOUND"] = "CONVEX_HULL"
    """Collision shape for physics: CONVEX_HULL (fast, poor for concave), MESH (accurate, slow), BOX (fastest), COMPOUND (VHACD decomposition)."""
    decomposition: Literal["vhacd", "coacd"] = "vhacd"
    """Convex decomposition method for COMPOUND collision shape."""
    collision_margin: float = 0.001
    """Rigid-body collision margin in meters."""
    rigidbody_friction: float | None = None
    """Object and floor friction coefficient."""
    linear_damping: float = 0.1
    """Rigid-body linear damping."""
    angular_damping: float = 0.15
    """Rigid-body angular damping."""
    coacd_threshold: float = 0.05
    """CoACD concavity threshold (lower = more parts, higher quality)."""
    coacd_path: Path = Path(__file__).resolve().parent.parent.parent / "libs" / "coacd"
    """Path to CoACD directory (for cache storage)."""
    vhacd_path: Path = Path(__file__).resolve().parent.parent.parent / "libs"  # Contains v-hacd/app/TestVHACD
    """Path to VHACD directory (auto-downloaded if not present)."""
    decomposition_cache_dir: Path | None = None
    """Optional shared root for stable per-mesh collision decomposition caches."""
    scene: Literal["packed", "pile"] | None = None
    """Preset scene configuration (sets spawn_height, spawn_bounds, containment_walls, upright, placement)."""
    solidify: float | None = None
    """Thickness of solidify modifier added to the object."""
    add_uv: bool = False
    """Add UV mapping to the object."""
    hdri_path: Path | Literal["haven"] | None = None
    """Load random Haven HDRI image from this path."""
    hdri_strength: float | Literal["random"] = 1.0
    """Emission strength of the HDRI image."""
    hdri_assets: tuple[str, ...] | None = None
    """Optional Haven HDRI asset allowlist."""
    randomize_hdri: bool = False
    """Randomize the HDRI image for each frame."""
    randomize_hdri_rotation: bool = False
    """Randomize the scene-level HDRI azimuth."""
    lights: int | tuple[int, int] | None = None
    """Fixed count or (inclusive min, exclusive max) random light count."""
    light_type: Literal["POINT", "AREA"] = "POINT"
    """Type of direct light used for random scene illumination."""
    light_radius: tuple[float, float] = (5.0, 10.0)
    """Min/max radius in meters for sampled lights."""
    light_elevation: tuple[float, float] = (1.0, 89.0)
    """Min/max elevation in degrees for sampled lights."""
    light_energy: tuple[float, float] = (0.0, 250.0)
    """Min/max total direct-light energy, divided across sampled lights."""
    light_size: tuple[float, float] = (0.25, 0.25)
    """Min/max area-light size or point-light shadow radius in meters."""
    light_color_mode: Literal["rgb", "neutral"] = "rgb"
    """Sample legacy independent RGB colors or correlated warm-to-cool neutral light."""
    randomize_lights: bool = False
    """Randomize lights for each frame."""
    materials: bool = False
    """Set random material properties."""
    randomize_materials: bool = False
    """Randomize material properties for each frame."""
    colors: bool | float | Literal["auto"] = False
    """Set random colors for the objects."""
    randomize_colors: bool = False
    """Randomize colors for each frame."""
    displacement: bool | float = False
    """Set random displacement for the materials."""
    replace: bool | float = False
    """Set random texture replacement for the materials."""
    cc_material_path: Path | None = None
    """Path to the CC materials directory."""
    cc_material_assets: tuple[str, ...] | None = None
    """Optional CC material name prefixes used to restrict support texture replacement."""
    surface_material_profile: Literal["tabletop"] | None = None
    """Bounded procedural material profile for the support surface."""
    materialless_object_path_profiles: dict[str, Literal["fixed", "industrial"]] = field(default_factory=dict)
    """Source path tokens mapped to fallback profiles for material-less objects."""
    world_background_strength: float = 0.1
    """World background emission strength when no HDRI is active."""
    engine: str = "cycles"
    """The Blender render eninge to use."""
    max_samples: int | Literal["auto"] = "auto"
    """Maximum number of samples rendering."""
    noise_threshold: float = 0.1
    """Noise threshold for Cycles rendering."""
    denoiser: str = "optix"
    """Denoiser for Cycles rendering."""
    camera: Camera = field(default_factory=Camera)
    """Camera intrinsic and extrinsic parematers."""
    primitive_type: Literal["sphere", "cube", "cone", "cylinder", "torus", "monkey", "random"] = "random"
    """Primitive to render if `object_path` is set to "primitives"."""
    num_objects: int | tuple[int, int] | None = None
    """Number of objects to render."""
    physics: bool = False
    """Enable physics simulation for the objects."""
    physics_min_simulation_time: float = 2.0
    """Minimum simulated settling time in seconds."""
    physics_max_simulation_time: float = 15.0
    """Maximum simulated settling time in seconds."""
    physics_check_interval: float = 0.5
    """Interval between settled-object checks in simulated seconds."""
    physics_substeps_per_frame: int = 25
    """Bullet substeps per frame."""
    physics_solver_iters: int = 20
    """Bullet solver iterations per step."""
    correct_floor_penetration: bool = False
    """Lift penetrating objects to the surface after packed-scene settling."""
    physics_reject_max_abs_xy: float | None = None
    """Hide post-physics objects whose AABB center exceeds this XY magnitude."""
    physics_reject_min_z: float | None = None
    """Hide post-physics objects whose rendered bounds fall below this Z value."""
    physics_reject_max_z: float | None = None
    """Hide post-physics objects whose rendered bounds exceed this Z value."""
    surface: Literal["plane", "table"] | None = None
    """Surface to place the objects on."""
    surface_size: float = 5.0
    """Side length in meters for a generated plane surface."""
    surface_thickness: float = 0.04
    """Thickness in meters for a generated table slab."""
    normals: bool = False
    """Enable normal output."""
    depth: bool = False
    """Enable depth output."""
    kinect: bool = False
    """Enable Kinect Azure noise on depth output."""
    kinect_darkness_threshold: int = 15
    """Grayscale threshold below which Kinect Azure depth is invalidated."""
    kinect_sim: bool = False
    """Enable structured light Kinect v1 depth simulation (libkinect)."""
    stereo_depth: bool = False
    """Enable BlenderProc semi-global matching depth from a rendered right-camera view."""
    stereo_baseline: float = 0.05
    """Horizontal stereo baseline in meters."""
    stereo_depth_max: float = 2.0
    """Maximum valid stereo depth in meters."""
    stereo_window_size: int = 7
    """Semi-global matching window size; must be odd."""
    stereo_num_disparities: int = 128
    """Semi-global matching disparity search width; must be divisible by 16."""
    diffuse: bool = False
    """Enable diffuse color output."""
    segmentation: bool = False
    """Enable segmentation output."""
    mask: bool = True
    """Zero-mask empty regions in outputs (reduces file size)."""
    background: Literal["white", "black", "transparent", "random"] | tuple[float, float, float] | None = None
    """Background color."""
    jpg_quality: int = 95
    """JPEG quality."""
    view_transform: str | None = None
    """Color curve applied to the rendered image."""
    writer: Literal["bop", "coco", "hdf5", "usd"] | None = None
    """Output writer type."""
    overwrite: bool = False
    """Overwrite existing files."""
    progress: bool = True
    """Show progress bar."""
    seed: int | None = None
    """Set random seed for reproducibility."""
    verbose: bool = False
    """Enable verbose logging."""
    quiet: bool = False
    """Disable logging to stdout."""
    scene_metadata: dict[str, Any] | None = None
    """Serializable scene-generation settings persisted into each HDF5 frame record."""

    def __post_init__(self):
        logger.remove()
        if self.quiet:
            logger.add(sys.stderr, level="ERROR")
        else:
            level = "DEBUG" if self.verbose else "INFO"
            logger.add(sys.stderr, level=level)

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            logger.debug(f"Set random seed to {self.seed}")

        if "eevee" in self.engine.lower():
            self.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version >= (2, 93) else "BLENDER_EEVEE"
            if self.max_samples == "auto":
                self.max_samples = 32
        elif "cycles" in self.engine.lower():
            self.engine = "CYCLES"
            if self.max_samples == "auto":
                self.max_samples = 128
        self.denoiser = self.denoiser.upper()
        self.shading = self.shading.upper()

        is_primitives = str(self.object_path) == "primitives"
        is_txt_list = isinstance(self.object_path, Path) and self.object_path.suffix == ".txt"
        if is_primitives or is_txt_list:
            if self.num_objects is None:
                self.num_objects = 1
            if isinstance(self.num_objects, (tuple, list)):
                low, high = self.num_objects
                mean = (low + high) / 2
                std = (high - low) / 4
                a, b = (low - mean) / std, (high - mean) / std
                sampled = int(truncnorm.rvs(a, b, loc=mean, scale=std))
                sampled = max(low, min(high, sampled))
                self.num_objects = sampled
                logger.info(f"Sampled number of objects: {self.num_objects}")

        self.metadata: LabelIdMapping | None = None
        if self.metadata_path:
            with open(self.metadata_path, encoding="utf-8") as f:
                taxonomy_data = json.load(f)

            label_to_id: dict[str, int] = {}
            seen_labels: set[str] = set()
            seen_ids: set[int] = set()
            for item in taxonomy_data:
                primary_label: str = item["name"].split(",")[0]
                id_value: int = int(item["synsetId"])

                if primary_label not in seen_labels and id_value not in seen_ids:
                    label_to_id[primary_label] = id_value
                    seen_labels.add(primary_label)
                    seen_ids.add(id_value)
            self.metadata = LabelIdMapping.from_dict(label_to_id)

        # Save original scale range for spawn bounds calculation
        scale_spec = self.scale
        if isinstance(scale_spec, (tuple, list)) and len(scale_spec) == 2:
            self._scale_range = (float(scale_spec[0]), float(scale_spec[1]))
        else:
            self._scale_range = None

        if self.scale is None:
            self.scale = lambda: [1.0] * 3
        elif isinstance(self.scale, (int, float)):
            scale_scalar = float(self.scale)
            self.scale = lambda: [scale_scalar] * 3
        elif isinstance(self.scale, (tuple, list)):
            if len(self.scale) == 2:
                scale_range = (float(self.scale[0]), float(self.scale[1]))

                def sample_distortion(distort: float) -> np.ndarray:
                    return np.random.uniform(1.0 - distort, 1.0 + distort, size=3)

                self.scale = lambda: sample_truncnorm(scale_range)
                if self.distort:
                    distort = float(self.distort)
                    self.scale = lambda: sample_truncnorm(scale_range) * sample_distortion(distort)
            elif len(self.scale) == 3:
                scale_xyz = tuple(float(value) for value in self.scale)
                self.scale = lambda: np.array(scale_xyz, dtype=np.float32)

        if self.kinect:
            self.depth = True
            if not 0 <= self.kinect_darkness_threshold <= 255:
                raise ValueError("kinect_darkness_threshold must be between 0 and 255")

        if self.kinect_sim:
            self.depth = True

        if self.stereo_depth:
            if self.stereo_baseline <= 0 or self.stereo_depth_max <= 0:
                raise ValueError("stereo baseline and maximum depth must be positive")
            if self.stereo_window_size <= 0 or self.stereo_window_size % 2 == 0:
                raise ValueError("stereo_window_size must be a positive odd integer")
            if self.stereo_num_disparities <= 0 or self.stereo_num_disparities % 16:
                raise ValueError("stereo_num_disparities must be positive and divisible by 16")
            if self.randomize_hdri or self.randomize_lights or self.randomize_materials or self.randomize_colors:
                raise ValueError("stereo_depth requires scene-level, not per-frame, appearance randomization")

        if self.segmentation and not self.writer:
            self.writer = "bop"

        if self.replace:
            if not self.cc_material_path:
                raise ValueError("CC material path must be set when using texture replacement.")

        if isinstance(self.lights, int) and self.lights <= 0:
            raise ValueError("lights must be positive")
        if isinstance(self.lights, tuple) and (
            len(self.lights) != 2 or self.lights[0] < 0 or self.lights[0] >= self.lights[1]
        ):
            raise ValueError("lights must be a non-negative (inclusive min, exclusive max) pair")
        for name, value, minimum in (
            ("light_radius", self.light_radius, 0.0),
            ("light_energy", self.light_energy, 0.0),
            ("light_size", self.light_size, 0.0),
        ):
            if len(value) != 2 or value[0] < minimum or value[0] > value[1]:
                raise ValueError(f"{name} must be a non-negative (min, max) pair")
        if self.light_size[0] <= 0:
            raise ValueError("light_size must be positive")
        if len(self.light_elevation) != 2 or not 0 <= self.light_elevation[0] <= self.light_elevation[1] <= 90:
            raise ValueError("light_elevation must lie within [0, 90] degrees")
        if self.world_background_strength < 0:
            raise ValueError("world_background_strength must be non-negative")
        if self.surface_size <= 0:
            raise ValueError("surface_size must be positive")
        if self.surface_thickness <= 0:
            raise ValueError("surface_thickness must be positive")
        for token, rotation in self.upright_x_rotation_path_overrides.items():
            if not token.strip() or not np.isfinite(float(rotation)):
                raise ValueError(
                    "upright_x_rotation_path_overrides requires non-empty path tokens and finite rotations"
                )
        for token, profile in self.materialless_object_path_profiles.items():
            if not token.strip() or profile not in {"fixed", "industrial"}:
                raise ValueError(
                    "materialless_object_path_profiles requires non-empty path tokens and fixed or industrial profiles"
                )

        if self.pile_batch_size < 1:
            raise ValueError("pile_batch_size must be positive")
        if self.pile_settle_time <= 0:
            raise ValueError("pile_settle_time must be positive")
        if self.pile_wall_margin < 0:
            raise ValueError("pile_wall_margin must be non-negative")
        if self.pile_center_radius is not None and self.pile_center_radius < 0:
            raise ValueError("pile_center_radius must be non-negative")
        if (
            len(self.pile_drop_margin) != 2
            or self.pile_drop_margin[0] < 0
            or self.pile_drop_margin[0] > self.pile_drop_margin[1]
        ):
            raise ValueError("pile_drop_margin must be a non-negative (min, max) pair")

        # Apply scene presets (VGN-style configurations)
        # Compute average object size for spawn bounds scaling
        avg_scale = np.mean(self._scale_range) if self._scale_range else 0.25
        max_scale = self._scale_range[1] if self._scale_range else 0.25
        num_objs = self.num_objects if isinstance(self.num_objects, int) else 5

        if self.scene == "packed":
            # Packed: objects placed upright on surface, then settle with physics
            # VGN packed: XY 0.08-0.22 in 0.3m workspace for ~5 objects
            if self.spawn_height is None:
                self.spawn_height = (0.01, 0.05)  # Just above surface
            if self.spawn_bounds is None:
                # Scale spawn area based on object count and size
                # VGN ratio: ~0.07m half-width for 5 objects at ~0.1m scale
                spawn_half = max(0.15, np.sqrt(num_objs) * avg_scale * 0.7)
                plane_half = self.surface_size / 2.0
                self.spawn_bounds = (
                    0.5 - spawn_half / plane_half,
                    0.5 + spawn_half / plane_half,
                )  # Relative to plane
            # upright=True is already default; placement defaults to surface at end of __post_init__
            if self.containment_walls is None:
                self.containment_walls = False
            if not self.physics:
                self.physics = True  # ShapeNet objects need settling
                logger.warning("Enabling physics for packed scene (required for settling)")
            logger.info(
                f"Scene preset 'packed': {num_objs} objects, avg_scale={avg_scale:.2f}, bounds={self.spawn_bounds}"
            )
        elif self.scene == "pile":
            # Pile: objects dropped with random orientation, physics settling
            # VGN pile: middle third of 0.3m workspace for ~5 objects
            if self.spawn_height is None:
                # Drop height: must clear largest object + margin; not too high to avoid bounce
                # VGN uses 0.2m for small objects; we scale with max object size
                self.spawn_height = (max_scale + 0.1, max_scale + 0.3)
            if self.spawn_bounds is None:
                # Scale spawn area: tighter packing for more stacking/occlusion
                # Formula: sqrt(num_objects) * avg_scale * packing_factor
                spawn_half = max(0.1, np.sqrt(num_objs) * avg_scale * 0.35)
                self.spawn_bounds = (-spawn_half, spawn_half)  # Absolute bounds
            self.upright = False  # Full SO3 rotation
            if self.placement is None:
                self.placement = "sequential"
            if self.containment_walls is None:
                self.containment_walls = True
            if self.solidify is None:
                self.solidify = 0.0
            if self.physics_reject_max_abs_xy is None:
                self.physics_reject_max_abs_xy = 0.5
            if self.physics_reject_min_z is None:
                self.physics_reject_min_z = -0.1
            if self.physics_reject_max_z is None:
                self.physics_reject_max_z = 2.0
            if not self.physics:
                self.physics = True  # Pile requires physics
                logger.warning("Enabling physics for pile scene (required for settling)")
            logger.info(
                f"Scene preset 'pile': {num_objs} objects, avg_scale={avg_scale:.2f}, collision={self.collision_shape}, bounds={self.spawn_bounds}"
            )

        if self.containment_walls is None:
            self.containment_walls = False

        if self.rigidbody_friction is None:
            self.rigidbody_friction = 0.8 if self.scene == "pile" else 0.5

        # Preserve the historical generic behavior outside the pile preset.
        if self.physics and self.solidify is None:
            self.solidify = 0.0025

        # Apply default placement if not set by user or preset
        if self.placement is None:
            self.placement = "surface"


def random_pose_fn(
    obj: bproc.types.MeshObject,
    position: tuple[float, float] | tuple[float, float, float] = (-1, 1),
    rotation: bool | tuple[float, float, float] | None = False,
):
    if len(position) == 2:
        loc = np.random.uniform(position[0], position[1], size=3)
    else:
        loc = np.asarray(position, dtype=np.float32)
    obj.set_location(loc)
    if isinstance(rotation, tuple):
        obj.set_rotation_euler(np.deg2rad(np.asarray(rotation, dtype=np.float32)))
    elif rotation:
        obj.set_rotation_euler(bproc.sampler.uniformSO3())


def upright_x_rotation_for_path(
    path: str,
    default: float,
    overrides: dict[str, float] | None,
) -> float:
    """Resolve one unambiguous source-specific upright rotation from an asset path."""
    matches = {float(rotation) for token, rotation in (overrides or {}).items() if token.lower() in path.lower()}
    if len(matches) > 1:
        raise ValueError(f"Conflicting upright rotation overrides match asset path {path!r}")
    return matches.pop() if matches else float(default)


def upright_x_rotation_for_object(
    obj: bproc.types.MeshObject,
    default: float,
    overrides: dict[str, float] | None,
) -> float:
    path = str(obj.get_cp("source_asset_path")) if obj.has_cp("source_asset_path") else ""
    return upright_x_rotation_for_path(path, default, overrides)


def upper_region_pose_fn(
    obj: bproc.types.MeshObject,
    surface: bproc.types.MeshObject | None = None,
    rotation: bool | tuple[float, float, float] | None = False,
    upright: bool = False,
    upright_x_rotation: float = 90.0,
    upright_x_rotation_path_overrides: dict[str, float] | None = None,
    spawn_height: tuple[float, float] = (1.0, 4.0),
    spawn_bounds: tuple[float, float] = (0.4, 0.6),
):
    if surface is None:
        loc = np.random.uniform([-1, -1, 0.0], [1, 1, 1])
    else:
        loc = bproc.sampler.upper_region(
            objects_to_sample_on=surface,
            min_height=spawn_height[0],
            max_height=spawn_height[1],
            face_sample_range=list(spawn_bounds),
        )
    obj.set_location(loc)
    if isinstance(rotation, tuple):
        obj.set_rotation_euler(np.deg2rad(np.asarray(rotation, dtype=np.float32)))
    elif rotation:
        if upright:
            x_rotation = upright_x_rotation_for_object(obj, upright_x_rotation, upright_x_rotation_path_overrides)
            obj.set_rotation_euler([np.deg2rad(x_rotation), 0, np.random.uniform(0, np.pi * 2)])
        else:
            # Full SO3 rotation
            obj.set_rotation_euler(bproc.sampler.uniformSO3())


def volume_pose_fn(
    obj: bproc.types.MeshObject,
    bounds_xy: tuple[float, float] = (-0.5, 0.5),
    bounds_z: tuple[float, float] = (0.1, 0.6),
    rotation: bool | tuple[float, float, float] | None = True,
    upright: bool = False,
    upright_x_rotation: float = 90.0,
    upright_x_rotation_path_overrides: dict[str, float] | None = None,
):
    """Pose function for pile-style volume placement (no surface constraint).

    Places objects randomly within XY bounds (default: unit cube -1 to 1).
    """
    loc = np.array(
        [
            np.random.uniform(bounds_xy[0], bounds_xy[1]),
            np.random.uniform(bounds_xy[0], bounds_xy[1]),
            np.random.uniform(bounds_z[0], bounds_z[1]),
        ]
    )
    obj.set_location(loc)
    if isinstance(rotation, tuple):
        obj.set_rotation_euler(np.deg2rad(np.asarray(rotation, dtype=np.float32)))
    elif rotation:
        if upright:
            x_rotation = upright_x_rotation_for_object(obj, upright_x_rotation, upright_x_rotation_path_overrides)
            obj.set_rotation_euler([np.deg2rad(x_rotation), 0, np.random.uniform(0, np.pi * 2)])
        else:
            # Full SO3 rotation
            obj.set_rotation_euler(bproc.sampler.uniformSO3())


def place_objects_tower(
    objs: list[bproc.types.MeshObject],
    bounds_xy: tuple[float, float] = (-0.5, 0.5),
    jitter_xy: float = 0.02,
    base_z: float = 0.05,
    gap: float = 0.02,
    rotation: bool | tuple[float, float, float] | None = True,
    upright: bool = False,
    upright_x_rotation: float = 90.0,
    upright_x_rotation_path_overrides: dict[str, float] | None = None,
) -> None:
    """Place objects in a vertical tower at a random XY position.

    Objects are stacked vertically with small random XY jitter per object.
    Physics then collapses the tower.

    Args:
        objs: Objects to stack
        bounds_xy: (min, max) range for base XY position of the tower
        jitter_xy: Small random offset per object around the base XY
        base_z: Starting Z height for bottom object
        gap: Vertical gap between objects
        rotation: Apply random rotation
        upright: If True, only randomize yaw (keep objects upright)
    """
    if not objs:
        return

    # Sample base XY for the tower (can be anywhere in bounds)
    base_xy = np.array(
        [
            np.random.uniform(bounds_xy[0], bounds_xy[1]),
            np.random.uniform(bounds_xy[0], bounds_xy[1]),
        ]
    )

    current_z = base_z

    for obj in objs:
        # Apply rotation first (affects bounding box)
        if isinstance(rotation, tuple):
            obj.set_rotation_euler(np.deg2rad(np.asarray(rotation, dtype=np.float32)))
        elif rotation:
            if upright:
                x_rotation = upright_x_rotation_for_object(obj, upright_x_rotation, upright_x_rotation_path_overrides)
                obj.set_rotation_euler([np.deg2rad(x_rotation), 0, np.random.uniform(0, np.pi * 2)])
            else:
                obj.set_rotation_euler(bproc.sampler.uniformSO3())

        # Get bounding box after rotation
        bb = obj.get_bound_box()
        min_z = bb[:, 2].min()
        max_z = bb[:, 2].max()
        height = max_z - min_z

        # Position so bottom of bbox is at current_z
        # Object origin may not be at center, so compute offset
        current_loc = obj.get_location()
        z_offset = current_loc[2] - min_z  # Distance from origin to bottom

        # Add small random jitter per object
        xy = base_xy + np.random.uniform(-jitter_xy, jitter_xy, size=2)

        obj.set_location([xy[0], xy[1], current_z + z_offset])

        # Move up for next object
        current_z += height + gap


def create_containment_walls(
    size: float = 2.5,
    height: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
) -> list[bproc.types.MeshObject]:
    """Create temporary containment walls for pile physics simulation."""
    walls = []
    half = size / 2
    wall_thickness = 0.02
    cx, cy = center

    # Four walls around the perimeter, offset by center
    wall_specs = [
        ([cx + half, cy, height / 2], [wall_thickness, size, height]),  # +X wall
        ([cx - half, cy, height / 2], [wall_thickness, size, height]),  # -X wall
        ([cx, cy + half, height / 2], [size, wall_thickness, height]),  # +Y wall
        ([cx, cy - half, height / 2], [size, wall_thickness, height]),  # -Y wall
    ]

    for i, (loc, scale) in enumerate(wall_specs):
        wall = bproc.object.create_primitive("CUBE", location=loc, scale=[s / 2 for s in scale])
        wall.set_name(f"ContainmentWall_{i}")
        wall.enable_rigidbody(
            active=False,
            collision_shape="BOX",
            collision_margin=0.001,
            friction=0.3,  # Lower than floor (0.5) so objects slide down walls naturally
            linear_damping=0.1,
            angular_damping=0.15,
        )
        walls.append(wall)

    logger.debug(f"Created {len(walls)} containment walls (size={size}, height={height})")
    return walls


def enable_rigidbody_with_decomposition(obj: bproc.types.MeshObject, cfg: "Config") -> None:
    """Enable rigidbody on object with proper convex decomposition if configured.

    Handles COMPOUND collision shapes with either CoACD or V-HACD decomposition,
    as well as simple shapes (CONVEX_HULL, MESH, BOX).

    Args:
        obj: BlenderProc MeshObject to enable physics on.
        cfg: Config with collision_shape, decomposition method, and paths.
    """
    if cfg.collision_shape == "COMPOUND":
        if cfg.decomposition == "coacd":
            # CoACD decomposition (SIGGRAPH 2022)
            # Must enable parent rigidbody FIRST with COMPOUND shape
            obj.enable_rigidbody(
                active=True,
                collision_shape="COMPOUND",
                collision_margin=cfg.collision_margin,
                friction=cfg.rigidbody_friction,
                linear_damping=cfg.linear_damping,
                angular_damping=cfg.angular_damping,
            )
            # Then decompose and add children with CONVEX_HULL
            cache_dir = (
                cfg.decomposition_cache_dir / "coacd"
                if cfg.decomposition_cache_dir is not None
                else cfg.coacd_path / ".cache"
            )
            parts = coacd_decomposition(obj, cfg.coacd_threshold, cache_dir)
            for part in parts:
                part_obj = MeshObject(part)
                part_obj.set_parent(obj)
                part_obj.enable_rigidbody(True, "CONVEX_HULL")
                part_obj.hide()
        else:
            # V-HACD fallback
            obj.enable_rigidbody(
                active=True,
                collision_shape=cfg.collision_shape,
                collision_margin=cfg.collision_margin,
                friction=cfg.rigidbody_friction,
                linear_damping=cfg.linear_damping,
                angular_damping=cfg.angular_damping,
            )
            cache_dir = (
                cfg.decomposition_cache_dir / "vhacd"
                if cfg.decomposition_cache_dir is not None
                else cfg.vhacd_path / "v-hacd" / ".cache"
            )
            obj.build_convex_decomposition_collision_shape(
                str(cfg.vhacd_path),
                cache_dir=str(cache_dir),
            )
    else:
        obj.enable_rigidbody(
            active=True,
            collision_shape=cfg.collision_shape,
            collision_margin=cfg.collision_margin,
            friction=cfg.rigidbody_friction,
            linear_damping=cfg.linear_damping,
            angular_damping=cfg.angular_damping,
        )


def is_render_hidden(obj: bproc.types.MeshObject) -> bool:
    """Return whether placement rejected an object from rendering."""
    return bool(getattr(obj.blender_obj, "hide_render", False))


def mesh_object_world_vertices(obj: bproc.types.MeshObject) -> np.ndarray:
    """Return tight world-space vertices for a BlenderProc mesh object."""
    mesh = obj.mesh_as_trimesh()
    scale = np.asarray(obj.get_scale(), dtype=np.float64)
    local2world = np.asarray(obj.get_local2world_mat(), dtype=np.float64)
    # mesh_as_trimesh() pre-multiplies vertices by scale. Undo that before
    # applying the complete object transform, matching merge_scene_meshes().
    vertices_local = np.asarray(mesh.vertices, dtype=np.float64) / np.where(np.abs(scale) > 1e-8, scale, 1.0)
    return vertices_local @ local2world[:3, :3].T + local2world[:3, 3]


def mesh_object_world_bounds(obj: bproc.types.MeshObject) -> tuple[np.ndarray, np.ndarray]:
    """Return tight world-space bounds from rendered mesh vertices."""
    vertices_world = mesh_object_world_vertices(obj)
    if not len(vertices_world) or not np.isfinite(vertices_world).all():
        raise ValueError(f"Invalid rendered mesh vertices for {obj.get_name()}")
    return vertices_world.min(axis=0), vertices_world.max(axis=0)


def rotated_bounds_at_origin(obj: bproc.types.MeshObject, rotation: np.ndarray) -> np.ndarray:
    """Compute scaled, rotated bounds from the normalization-time geometry bounds."""
    if obj.has_cp("placement_bounds_min") and obj.has_cp("placement_bounds_max"):
        bounds_min = np.asarray(obj.get_cp("placement_bounds_min"), dtype=np.float64)
        bounds_max = np.asarray(obj.get_cp("placement_bounds_max"), dtype=np.float64)
        corners = np.array(
            [
                [x, y, z]
                for x in (bounds_min[0], bounds_max[0])
                for y in (bounds_min[1], bounds_max[1])
                for z in (bounds_min[2], bounds_max[2])
            ]
        )
        corners *= np.asarray(obj.get_scale(), dtype=np.float64)
        transform = trimesh.transformations.euler_matrix(*rotation)
        return trimesh.transform_points(corners, transform)

    # Fallback for objects created outside this renderer's load/normalize path.
    obj.set_rotation_euler(rotation)
    obj.set_location([0.0, 0.0, 0.0])
    bpy.context.view_layer.update()
    return np.asarray(obj.get_bound_box(), dtype=np.float64)


def placement_bounds_compatible(
    obj: bproc.types.MeshObject,
    cfg: "Config",
    probe_rotation: np.ndarray,
) -> bool:
    """Reject assets whose evaluated bounds do not match normalized placement bounds."""
    if not (obj.has_cp("placement_bounds_min") and obj.has_cp("placement_bounds_max")):
        return True

    expected_probe = rotated_bounds_at_origin(obj, probe_rotation)
    obj.set_rotation_euler(probe_rotation)
    obj.set_location([0.0, 0.0, 0.0])
    bpy.context.view_layer.update()
    evaluated_probe = np.asarray(obj.get_bound_box(), dtype=np.float64)
    expected_extent = float(np.ptp(expected_probe, axis=0).max())
    evaluated_extent = float(np.ptp(evaluated_probe, axis=0).max())
    bounds_ratio = evaluated_extent / expected_extent if expected_extent > 0 else float("inf")
    ratio_limit = cfg.placement_bounds_ratio_limit
    compatible = (
        np.isfinite(expected_probe).all()
        and np.isfinite(evaluated_probe).all()
        and 1.0 / ratio_limit <= bounds_ratio <= ratio_limit
    )
    if not compatible:
        obj.blender_obj.hide_render = True
        obj.blender_obj.hide_viewport = True
        logger.warning(
            f"Placement rejected transform-incompatible {obj.get_name()}: "
            f"evaluated_extent={evaluated_extent:.6g}, expected_extent={expected_extent:.6g}, "
            f"ratio={bounds_ratio:.6g}"
        )
    return compatible


def sample_pile_drop_pose(
    obj: bproc.types.MeshObject,
    cfg: "Config",
    bounds_xy: tuple[float, float],
    target_bottom: float,
) -> np.ndarray | None:
    """Sample a rotated pile-drop pose whose world AABB starts inside the spawn region."""
    if cfg.upright:
        x_rotation = upright_x_rotation_for_object(
            obj,
            cfg.upright_x_rotation,
            cfg.upright_x_rotation_path_overrides,
        )
        rotation = np.array([np.deg2rad(x_rotation), 0.0, np.random.uniform(0.0, 2.0 * np.pi)])
    elif cfg.rotation:
        rotation = np.asarray(bproc.sampler.uniformSO3(), dtype=np.float64)
    else:
        rotation = np.zeros(3)
    obj.set_rotation_euler(rotation)
    bounds = rotated_bounds_at_origin(obj, rotation)
    if not np.isfinite(bounds).all():
        return None
    bounds_min = bounds.min(axis=0)
    bounds_max = bounds.max(axis=0)
    x_range = (bounds_xy[0] - bounds_min[0], bounds_xy[1] - bounds_max[0])
    y_range = (bounds_xy[0] - bounds_min[1], bounds_xy[1] - bounds_max[1])
    if cfg.pile_center_radius is not None:
        radius = cfg.pile_center_radius
        x_range = (max(x_range[0], -radius), min(x_range[1], radius))
        y_range = (max(y_range[0], -radius), min(y_range[1], radius))
    if x_range[0] > x_range[1] or y_range[0] > y_range[1]:
        return None
    translation = np.array(
        [
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            target_bottom - bounds_min[2],
        ]
    )
    obj.set_location(translation)
    return bounds + translation


def place_objects_surface_aabb(
    objs: list[bproc.types.MeshObject],
    cfg: "Config",
    bounds_xy: tuple[float, float],
    spawn_height: tuple[float, float],
    max_tries: int = 100,
) -> list[bproc.types.MeshObject]:
    """Place objects upright with disjoint world AABBs before rigid-body setup."""
    placed: list[bproc.types.MeshObject] = []
    occupied_xy: list[tuple[np.ndarray, np.ndarray]] = []

    def probe_rotation(obj: bproc.types.MeshObject) -> np.ndarray:
        x_rotation = upright_x_rotation_for_object(
            obj,
            cfg.upright_x_rotation,
            cfg.upright_x_rotation_path_overrides,
        )
        return np.array([np.deg2rad(x_rotation), 0.0, 0.0])

    def footprint_area(obj: bproc.types.MeshObject) -> float:
        bounds = rotated_bounds_at_origin(obj, probe_rotation(obj))
        extent = np.ptp(bounds, axis=0)
        return float(extent[0] * extent[1]) if np.isfinite(extent).all() else float("inf")

    for obj in sorted(objs, key=footprint_area, reverse=True):
        obj.blender_obj.hide_render = False
        obj.blender_obj.hide_viewport = False
        object_probe_rotation = probe_rotation(obj)
        if not placement_bounds_compatible(obj, cfg, object_probe_rotation):
            continue
        success = False
        attempted_centers: list[np.ndarray] = []
        last_bounds = np.empty((0, 3))
        for _ in range(max_tries):
            if cfg.upright:
                x_rotation = upright_x_rotation_for_object(
                    obj,
                    cfg.upright_x_rotation,
                    cfg.upright_x_rotation_path_overrides,
                )
                rotation = np.array([np.deg2rad(x_rotation), 0.0, np.random.uniform(0.0, 2.0 * np.pi)])
            elif cfg.rotation:
                rotation = bproc.sampler.uniformSO3()
            else:
                rotation = np.zeros(3)
            obj.set_rotation_euler(rotation)

            bounds = rotated_bounds_at_origin(obj, rotation)
            last_bounds = bounds
            bounds_min = bounds.min(axis=0)
            bounds_max = bounds.max(axis=0)
            x_range = (bounds_xy[0] - bounds_min[0], bounds_xy[1] - bounds_max[0])
            y_range = (bounds_xy[0] - bounds_min[1], bounds_xy[1] - bounds_max[1])
            if x_range[0] > x_range[1] or y_range[0] > y_range[1]:
                continue

            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            target_bottom = np.random.uniform(*spawn_height)
            translation = np.array([x, y, target_bottom - bounds_min[2]])
            obj.set_location(translation)
            candidate = bounds + translation
            candidate_min = candidate.min(axis=0)
            candidate_max = candidate.max(axis=0)
            attempted_centers.append(0.5 * (candidate_min + candidate_max))
            last_bounds = candidate
            separated = all(
                candidate_max[0] + cfg.placement_min_distance <= other_min[0]
                or other_max[0] + cfg.placement_min_distance <= candidate_min[0]
                or candidate_max[1] + cfg.placement_min_distance <= other_min[1]
                or other_max[1] + cfg.placement_min_distance <= candidate_min[1]
                for other_min, other_max in occupied_xy
            )
            if separated:
                occupied_xy.append((candidate_min[:2], candidate_max[:2]))
                placed.append(obj)
                success = True
                break

        if not success:
            obj.blender_obj.hide_render = True
            obj.blender_obj.hide_viewport = True
            centers = np.asarray(attempted_centers)
            center_range = (
                (centers.min(axis=0).tolist(), centers.max(axis=0).tolist()) if centers.size else (None, None)
            )
            logger.warning(
                f"AABB placement rejected {obj.get_name()}: "
                f"extent={np.ptp(last_bounds, axis=0).tolist()}, "
                f"scale={np.asarray(obj.get_scale()).tolist()}, "
                f"sampled_center_range={center_range}, "
                f"occupied={[(low.tolist(), high.tolist()) for low, high in occupied_xy]}"
            )

    return placed


def sequential_drop_objects(
    objs: list[bproc.types.MeshObject],
    cfg: "Config",
    plane: bproc.types.MeshObject,
    walls: list[bproc.types.MeshObject],
    bounds_xy: tuple[float, float],
    drop_margin: tuple[float, float] = (0.02, 0.1),
    batch_size: int = 1,
    settle_time: float = 1.0,
) -> list[bproc.types.MeshObject]:
    """Drop objects sequentially with physics settling between each (VGN-style pile generation).

    Args:
        objs: Objects to drop
        cfg: Config with physics/collision settings
        plane: Ground plane (must have rigidbody enabled)
        walls: Containment walls (must have rigidbody enabled)
        bounds_xy: (min, max) XY spawn bounds (objects placed randomly within)
        drop_margin: (min, max) height above current pile to spawn new objects
        batch_size: Number of objects to drop per physics step (1 = true sequential)
        settle_time: Seconds to simulate per batch
    """
    del plane, walls
    dropped_objs: list[bproc.types.MeshObject] = []
    pile_height = 0.0  # Track current pile height

    # Process in batches
    for i in range(0, len(objs), batch_size):
        batch = objs[i : i + batch_size]
        placed_batch: list[bproc.types.MeshObject] = []
        spawn_top = pile_height

        # Objects in one batch are vertically separated. This permits overlapping
        # XY drop paths without starting Bullet from intersecting collision hulls.
        for obj in batch:
            obj.blender_obj.hide_render = False
            obj.blender_obj.hide_viewport = False
            if not placement_bounds_compatible(obj, cfg, np.zeros(3)):
                continue
            target_bottom = spawn_top + np.random.uniform(drop_margin[0], drop_margin[1])
            world_bounds = sample_pile_drop_pose(obj, cfg, bounds_xy, target_bottom)
            if world_bounds is None:
                obj.blender_obj.hide_render = True
                obj.blender_obj.hide_viewport = True
                logger.warning(f"Pile placement rejected oversized or non-finite {obj.get_name()}")
                continue
            spawn_top = float(world_bounds[:, 2].max()) + cfg.placement_min_distance
            obj.set_cp("pile_drop_batch", i // batch_size)

            # Enable rigidbody with proper decomposition (same logic as volume/surface)
            enable_rigidbody_with_decomposition(obj, cfg)
            placed_batch.append(obj)

        if not placed_batch:
            continue

        # Run physics for this batch
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=settle_time,
            max_simulation_time=max(
                settle_time + cfg.physics_check_interval,
                min(cfg.physics_max_simulation_time, settle_time * 3.0),
            ),
            check_object_interval=cfg.physics_check_interval,
            substeps_per_frame=cfg.physics_substeps_per_frame,
            solver_iters=cfg.physics_solver_iters,
        )

        # Track dropped objects and update pile height
        dropped_objs.extend(placed_batch)
        pile_height = max(obj.get_bound_box()[:, 2].max() for obj in dropped_objs)

        logger.debug(
            f"Dropped batch {i // batch_size + 1}/{(len(objs) + batch_size - 1) // batch_size}, pile_height={pile_height:.3f}"
        )

    return dropped_objs


def correct_objects_below_floor(
    objs: list[bproc.types.MeshObject],
    floor_z: float = 0.0,
    tolerance: float = 1e-4,
) -> dict[str, float]:
    """Lift visible objects whose rendered geometry penetrates the floor."""
    corrections: dict[str, float] = {}
    for obj in objs:
        if is_render_hidden(obj):
            continue
        bpy.context.view_layer.update()
        bounds_min, _ = mesh_object_world_bounds(obj)
        min_z = float(bounds_min[2])
        correction = floor_z - min_z
        if correction <= tolerance:
            continue
        location = np.asarray(obj.get_location(), dtype=np.float64)
        location[2] += correction
        obj.set_location(location)
        obj.set_cp("physics_floor_correction", correction)
        corrections[obj.get_name()] = correction
    if corrections:
        bpy.context.view_layer.update()
        logger.warning(
            f"Lifted {len(corrections)} penetrating render objects; max_correction={max(corrections.values()):.6f}m"
        )
    return corrections


def reject_unstable_physics_objects(objs: list[bproc.types.MeshObject], cfg: "Config") -> dict[str, str]:
    """Hide objects whose post-physics render bounds indicate a simulation launch."""
    rejected: dict[str, str] = {}
    for obj in objs:
        if is_render_hidden(obj):
            continue
        bpy.context.view_layer.update()
        bounds = np.asarray(obj.get_bound_box(), dtype=np.float64)
        if not np.isfinite(bounds).all():
            reason = "non_finite_bounds"
        else:
            bounds_min = bounds.min(axis=0)
            bounds_max = bounds.max(axis=0)
            center = 0.5 * (bounds_min + bounds_max)
            reason = ""
            if cfg.physics_reject_max_abs_xy is not None and np.max(np.abs(center[:2])) > cfg.physics_reject_max_abs_xy:
                reason = "xy_escape"
            elif cfg.physics_reject_min_z is not None and bounds_min[2] < cfg.physics_reject_min_z:
                reason = "below_floor_escape"
            elif cfg.physics_reject_max_z is not None and bounds_max[2] > cfg.physics_reject_max_z:
                reason = "height_escape"
        if not reason:
            continue
        if obj.has_rigidbody_enabled():
            obj.disable_rigidbody()
        obj.blender_obj.hide_render = True
        obj.blender_obj.hide_viewport = True
        obj.set_cp("physics_rejection_reason", reason)
        rejected[obj.get_name()] = reason
    if rejected:
        logger.warning(f"Rejected {len(rejected)} post-physics outliers: {rejected}")
    return rejected


def sample_hdri_rotation_euler(
    object_path: Path | str | None,
    randomize_azimuth: bool,
) -> np.ndarray:
    """Sample one HDRI rotation while preserving the legacy OBJ axis conversion."""
    rotation = np.zeros(3, dtype=np.float64)
    if object_path is not None and Path(object_path).suffix == ".obj":
        rotation[0] = np.pi / 2
    if randomize_azimuth:
        rotation[2] = np.random.uniform(0.0, 2.0 * np.pi)
    return rotation


def sample_hdri_path(cfg: Config) -> str:
    """Sample a Haven HDRI, optionally restricted to an explicit asset allowlist."""
    if cfg.hdri_path is None:
        raise ValueError("hdri_path must be configured before sampling an HDRI")
    if not cfg.hdri_assets:
        return bproc.loader.get_random_world_background_hdr_img_path_from_haven(cfg.hdri_path)

    root = Path(cfg.hdri_path)
    candidates: list[Path] = []
    for asset in cfg.hdri_assets:
        asset_dir = root / "hdris" / asset
        if not asset_dir.is_dir():
            asset_dir = root / asset
        candidates.extend(sorted(asset_dir.glob("*.hdr")))
        candidates.extend(sorted(asset_dir.glob("*.exr")))
    if not candidates:
        raise FileNotFoundError(f"No allowlisted HDRIs found below {root}")
    return str(random.choice(candidates))


def set_random_hdri(cfg: Config):
    world = cast(Any, bpy.context.scene).world
    nodes = world.node_tree.nodes
    texture_node = Utility.get_the_one_node_with_type(nodes, "TexEnvironment")
    hdri_path = sample_hdri_path(cfg)
    logger.debug(f"Setting HDRI image to {hdri_path}")
    texture_node.image = bpy.data.images.load(hdri_path, check_existing=True)

    if cfg.hdri_strength == "random":
        background_node = Utility.get_the_one_node_with_type(nodes, "Background")
        background_node.inputs["Strength"].default_value = np.random.uniform(0.5, 1.5)


def set_random_light(cfg: Config):
    scene_objects = cast(list[Any], cast(Any, bpy.context.scene).objects)
    bproc.object.delete_multiple([bproc.types.Light(obj) for obj in scene_objects if obj.type == "Light"])
    if cfg.lights is None:
        return
    if isinstance(cfg.lights, int):
        num_lights = cfg.lights
    else:
        num_lights = int(np.random.randint(cfg.lights[0], cfg.lights[1]))
    total_energy = float(np.random.uniform(*cfg.light_energy))
    for i in range(num_lights):
        light = bproc.types.Light(light_type=cfg.light_type, name=f"Light {i}")
        location = bproc.sampler.shell(
            center=(0, 0, 0),
            radius_min=cfg.light_radius[0],
            radius_max=cfg.light_radius[1],
            elevation_min=cfg.light_elevation[0],
            elevation_max=cfg.light_elevation[1],
        )
        light.set_location(location)
        if cfg.light_type == "AREA":
            light.set_rotation_mat(bproc.camera.rotation_from_forward_vec(-np.asarray(location)))
            light.blender_obj.data.shape = "DISK"
            light.blender_obj.data.size = float(np.random.uniform(*cfg.light_size))
        else:
            light.set_radius(float(np.random.uniform(*cfg.light_size)))
        if cfg.light_color_mode == "neutral":
            blend = float(np.random.uniform())
            warm = np.array([1.0, 0.88, 0.75])
            cool = np.array([0.82, 0.90, 1.0])
            color = (1.0 - blend) * warm + blend * cool
        else:
            color = np.random.uniform([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        light.set_color(color)
        light.set_energy(total_energy / num_lights)
        logger.debug(
            f"Created light {i} at location {light.get_location()} with {light.get_color()} and strength {light.get_energy()}"
        )


def randomize_tabletop_surface(material: bproc.types.Material) -> None:
    """Apply bounded support-surface appearance variation without changing geometry."""
    colors = np.array(
        [
            [0.16, 0.18, 0.20, 1.0],
            [0.35, 0.38, 0.40, 1.0],
            [0.62, 0.61, 0.57, 1.0],
            [0.72, 0.66, 0.55, 1.0],
            [0.78, 0.79, 0.76, 1.0],
        ]
    )
    material.set_principled_shader_value("Base Color", random.choice(colors).tolist())
    material.set_principled_shader_value("Specular IOR Level", float(np.random.uniform(0.2, 0.5)))
    material.set_principled_shader_value("Roughness", float(np.random.uniform(0.35, 0.85)))
    material.set_principled_shader_value("Metallic", 0.0)


def materialless_object_profile_for_path(
    path: str,
    overrides: dict[str, Literal["fixed", "industrial"]] | None,
) -> Literal["fixed", "industrial"]:
    """Resolve one source-specific material fallback from an asset path."""
    matches = {profile for token, profile in (overrides or {}).items() if token.lower() in path.lower()}
    if len(matches) > 1:
        raise ValueError(f"Conflicting material-less profiles match asset path {path!r}")
    return matches.pop() if matches else "fixed"


def ensure_render_material(
    obj: bproc.types.MeshObject,
    profile: Literal["fixed", "industrial"] = "fixed",
) -> None:
    """Give material-less imports a visible slot that domain randomization can modify."""
    if obj.has_materials():
        return
    material = obj.new_material(f"{obj.get_name()} default")
    if profile == "industrial":
        base_color, metallic = random.choice(
            (
                ([0.05, 0.055, 0.06, 1.0], 0.0),
                ([0.24, 0.27, 0.30, 1.0], 0.0),
                ([0.72, 0.73, 0.70, 1.0], 0.0),
                ([0.82, 0.80, 0.72, 1.0], 0.0),
                ([0.48, 0.055, 0.035, 1.0], 0.0),
                ([0.035, 0.14, 0.42, 1.0], 0.0),
                ([0.045, 0.25, 0.10, 1.0], 0.0),
                ([0.65, 0.42, 0.025, 1.0], 0.0),
                ([0.50, 0.53, 0.57, 1.0], 1.0),
            )
        )
        material.set_principled_shader_value("Base Color", base_color)
        material.set_principled_shader_value("Specular IOR Level", float(np.random.uniform(0.2, 0.5)))
        material.set_principled_shader_value("Roughness", float(np.random.uniform(0.25, 0.85)))
        material.set_principled_shader_value("Metallic", metallic)
    else:
        material.set_principled_shader_value("Base Color", [0.35, 0.38, 0.40, 1.0])
        material.set_principled_shader_value("Specular IOR Level", 0.3)
        material.set_principled_shader_value("Roughness", 0.6)
        material.set_principled_shader_value("Metallic", 0.0)
    obj.set_cp("materialless_fallback_profile", profile)


def replace_materials(
    objs: list[bproc.types.MeshObject],
    materials: list[bproc.types.Material],
    p: float = 0.5,
):
    for obj in objs:
        if obj.has_uv_mapping():
            for i, material in enumerate(obj.get_materials()):
                if np.random.uniform() < p:
                    new_material = random.choice(materials)
                    obj.set_material(i, new_material)
                    logger.debug(f"Replaced material {material.get_name()} with {new_material.get_name()}")


def randomize_materials(
    objs: list[bproc.types.MeshObject],
    specular: float = 0.5,
    roughness: float = 0.5,
    metallic: float = 0.5,
    color: float | Literal["auto"] = 0.5,
    displacement: float = 0.5,
):
    for obj in objs:
        if color == "auto":  # Randomize color if no valid uv mapping exists
            color = 0.7
            if obj.has_uv_mapping():
                color = 0.1

        for material in obj.get_materials():
            if material is not None and material.get_principled_shader_value("Alpha") == 1:
                log_str = f"Properties for material {material.get_name()}: "
                if np.random.uniform() < specular:
                    material.set_principled_shader_value("Specular IOR Level", np.random.uniform())
                    log_str += f"specular={material.get_principled_shader_value('Specular IOR Level')} "
                if np.random.uniform() < roughness:
                    material.set_principled_shader_value("Roughness", np.random.uniform())
                    log_str += f"roughness={material.get_principled_shader_value('Roughness')} "
                if np.random.uniform() < metallic:
                    material.set_principled_shader_value("Metallic", np.random.uniform())
                    log_str += f"metallic={material.get_principled_shader_value('Metallic')} "

                if np.random.uniform() < color:
                    c = np.random.uniform(size=3)
                    log_str += f"color={c}"
                    material.set_principled_shader_value("Base Color", [*list(c), 1])

                if np.random.uniform() < displacement:
                    if isinstance(
                        material.get_principled_shader_value("Base Color"),
                        bpy.types.NodeSocket,
                    ):
                        log_str += "displacement=True "
                        material.set_displacement_from_principled_shader_value(
                            "Base Color", np.random.uniform(0.001, 0.15)
                        )
                logger.debug(log_str)


def sample_from_file_weighted(file_path: Path, n_samples: int, alpha: float = 0.2) -> list[Path]:
    """
    Loads object paths from a file and samples N objects
    using softened inverse frequency weighting.

    Args:
        file_path: The pathlib.Path object for the text file.
        n_samples: The number of objects to sample.
        alpha: Controls strength of inverse frequency weighting (0=uniform, 1=full inverse).

    Returns:
        A list containing N sampled pathlib.Path objects.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found at: {file_path}")

    with file_path.open("r") as f:
        obj_paths = [Path(line.strip()) for line in f if line.strip()]

    if not obj_paths or n_samples <= 0:
        return []

    try:
        classes = [p.parts[-3] for p in obj_paths]
    except IndexError as error:
        raise ValueError("Path format is not as expected for class ID extraction.") from error

    class_counts = Counter(classes)
    sample_weights = np.asarray([(1 / class_counts[c]) ** alpha for c in classes], dtype=np.float64)

    if n_samples >= len(obj_paths):
        return random.sample(obj_paths, k=len(obj_paths))

    sample_weights = sample_weights / sample_weights.sum()
    indices = np.random.choice(len(obj_paths), size=n_samples, replace=False, p=sample_weights)
    return [obj_paths[int(index)] for index in indices]


def merge_scene_meshes(
    objs: list[MeshObject],
    surface: MeshObject | None,
    camera_extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge all scene objects into single mesh in camera coordinates.

    Args:
        objs: List of MeshObjects in the scene.
        surface: Optional surface MeshObject (plane or table).
        camera_extrinsic: World-to-camera transform (4x4) in OpenCV convention.

    Returns:
        vertices: (N, 3) float32 in camera coords (Z forward positive).
        faces: (M, 3) int32 vertex indices.
    """
    all_verts = []
    all_faces = []
    vert_offset = 0

    w2c = camera_extrinsic

    for obj in objs:
        if is_render_hidden(obj):
            logger.debug(f"Skipping render-hidden object in merged sensor mesh: {obj.get_name()}")
            continue
        mesh = obj.mesh_as_trimesh()
        scale = np.array(obj.get_scale())
        local2world = obj.get_local2world_mat()

        # mesh_as_trimesh() pre-multiplies vertices by scale (BlenderProc quirk)
        # Undo that so we can apply the full local2world matrix
        verts_local = mesh.vertices / np.where(scale > 1e-8, scale, 1.0)
        verts_world = verts_local @ local2world[:3, :3].T + local2world[:3, 3]
        verts_cam = verts_world @ w2c[:3, :3].T + w2c[:3, 3]

        logger.debug(
            f"  {obj.get_name()}: {len(mesh.vertices)} verts, "
            f"world_bounds={verts_world.min(axis=0)}..{verts_world.max(axis=0)}"
        )

        all_verts.append(verts_cam)
        all_faces.append(mesh.faces + vert_offset)
        vert_offset += len(verts_cam)

    if surface is not None:
        mesh = surface.mesh_as_trimesh()
        scale = np.array(surface.get_scale())
        local2world = surface.get_local2world_mat()

        verts_local = mesh.vertices / np.where(scale > 1e-8, scale, 1.0)
        verts_world = verts_local @ local2world[:3, :3].T + local2world[:3, 3]
        verts_cam = verts_world @ w2c[:3, :3].T + w2c[:3, 3]
        all_verts.append(verts_cam)
        all_faces.append(mesh.faces + vert_offset)

    return (
        np.vstack(all_verts).astype(np.float32),
        np.vstack(all_faces).astype(np.int32),
    )


def bake_blender_object_world_transform(blender_obj: Any) -> None:
    from mathutils import Matrix

    world_transform = blender_obj.matrix_world.copy()
    if getattr(blender_obj, "animation_data", None) is not None:
        blender_obj.animation_data_clear()
    for constraint in list(getattr(blender_obj, "constraints", [])):
        blender_obj.constraints.remove(constraint)
    blender_obj.data = blender_obj.data.copy()
    blender_obj.data.transform(world_transform)
    blender_obj.parent = None
    blender_obj.matrix_world = Matrix.Identity(4)


def import_gltf_as_mesh_object(path: str) -> MeshObject:
    before = set(cast(list[Any], cast(Any, bpy.context.scene).objects))
    bpy.ops.import_scene.gltf(filepath=path)
    imported = [obj for obj in cast(list[Any], cast(Any, bpy.context.scene).objects) if obj not in before]
    mesh_objs = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh objects imported from {path}")
    for mesh_obj in mesh_objs:
        bake_blender_object_world_transform(mesh_obj)
    if len(mesh_objs) == 1:
        return MeshObject(mesh_objs[0])

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    active = mesh_objs[0]
    for obj in mesh_objs:
        obj.select_set(True)
    cast(Any, bpy.context.view_layer).objects.active = active
    with bpy.context.temp_override(
        active_object=active,
        selected_objects=mesh_objs,
        selected_editable_objects=mesh_objs,
    ):
        bpy.ops.object.join()
    return MeshObject(cast(Any, bpy.context.view_layer).objects.active)


def normalization_parameters(bounds: np.ndarray) -> tuple[np.ndarray, float]:
    bounds = np.asarray(bounds, dtype=np.float64)
    extent = float(np.max(bounds[1] - bounds[0]))
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not np.isfinite(extent) or extent <= 0:
        raise ValueError(f"Invalid mesh bounds for normalization: {bounds}")
    scale = 1.0 / extent
    translation = -bounds.mean(axis=0) * scale
    return translation, scale


def loader_category_name_from_path(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    for index in range(len(parts) - 2, -1, -1):
        if parts[index].isdigit() and len(parts[index]) == 8 and index + 1 < len(parts):
            return parts[index], parts[index + 1]
    return None


def _node_output_by_name(node: Any, name: str) -> Any:
    try:
        return node.outputs[name]
    except (KeyError, TypeError):
        for output in node.outputs:
            if output.name == name or getattr(output, "identifier", None) == name:
                return output
        raise


def _enabled_node_output_by_name(node: Any, name: str) -> Any:
    output = _node_output_by_name(node, name)
    if not getattr(output, "enabled", True):
        raise KeyError(name)
    return output


def enable_segmentation_output_compat(
    map_by: str | list[str] = "category_id",
    default_values: dict[str, Any] | None = None,
    pass_alpha_threshold: float = 0.05,
    output_dir: str | None = None,
    file_prefix: str = "segmap_",
    output_key: str = "segmap",
) -> None:
    from blenderproc.python.utility.Utility import Utility

    for index, mesh_obj in enumerate(bpy.context.scene.objects):
        if getattr(mesh_obj, "type", None) == "MESH":
            mesh_obj.pass_index = index + 1

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
    tree = bpy.context.scene.node_tree
    render_layer_node = Utility.get_the_one_node_with_type(tree.nodes, "CompositorNodeRLayers")
    if render_layer_node is None:
        raise RuntimeError("Render Layers compositor node missing")
    try:
        object_index_output = _enabled_node_output_by_name(render_layer_node, "IndexOB")
    except KeyError:
        render_layer_node = tree.nodes.new("CompositorNodeRLayers")
        object_index_output = _enabled_node_output_by_name(render_layer_node, "IndexOB")

    if output_dir is None:
        output_dir = Utility.get_temporary_directory()
    output_node = tree.nodes.new("CompositorNodeOutputFile")
    output_node.base_path = output_dir
    output_node.format.file_format = "OPEN_EXR"
    output_node.file_slots.values()[0].path = file_prefix
    Utility.add_output_entry(
        {
            "key": output_key,
            "path": os.path.join(output_dir, file_prefix) + "%04d" + ".exr",
            "version": "3.0.0",
            "trim_redundant_channels": True,
            "is_semantic_segmentation": True,
            "semantic_segmentation_mapping": map_by,
            "semantic_segmentation_default_values": default_values,
        }
    )

    combine_color = tree.nodes.new("CompositorNodeCombineColor")
    combine_color.mode = "HSV"
    tree.links.new(object_index_output, combine_color.inputs[2])
    tree.links.new(combine_color.outputs["Image"], output_node.inputs["Image"])
    bpy.context.scene.view_layers["ViewLayer"].pass_alpha_threshold = pass_alpha_threshold


def simulate_kinect_depth(
    objs: list[MeshObject],
    surface: MeshObject | None,
    cfg: "Config",
    num_frames: int,
) -> list[np.ndarray]:
    """Run libkinect structured light depth simulation on scene meshes.

    Simulates Kinect v1-style depth sensing using ray casting and stereo matching.
    Imports libkinect directly to bypass libs/__init__.py (which imports torch,
    unavailable in Blender's bundled Python).

    Args:
        objs: Scene objects to include in simulation.
        surface: Optional ground plane/table surface.
        cfg: Render config with camera parameters.
        num_frames: Number of frames to simulate.

    Returns:
        List of depth images as (H, W) float32 arrays in meters.
    """
    import sys

    repo_root = Path(__file__).resolve().parent.parent.parent
    libs_dir = repo_root / "libs"
    if str(libs_dir) not in sys.path:
        sys.path.insert(0, str(libs_dir))

    import libkinect  # pyright: ignore[reportMissingImports]

    sim = libkinect.KinectSimCython()
    results = []

    logger.info(f"Running libkinect structured light simulation on {num_frames} frame(s)...")
    total_start = time.perf_counter()

    for frame in range(num_frames):
        frame_start = time.perf_counter()

        extrinsic = convert_extrinsic(
            inv_trafo(bproc.camera.get_camera_pose(frame)),
            "opengl",
            "opencv",
        )

        merge_start = time.perf_counter()
        vertices, faces = merge_scene_meshes(objs, surface, extrinsic)
        merge_time = time.perf_counter() - merge_start

        sim_start = time.perf_counter()
        depth_sim = sim.simulate(
            vertices,
            faces,
            width=cfg.camera.width,
            height=cfg.camera.height,
            fx=cfg.camera.fx,
            fy=cfg.camera.fy,
            cx=cfg.camera.cx,
            cy=cfg.camera.cy,
            noise=libkinect.NoiseType.PERLIN,
        )
        sim_time = time.perf_counter() - sim_start
        results.append(depth_sim)

        frame_time = time.perf_counter() - frame_start
        logger.debug(
            f"  Frame {frame + 1}/{num_frames}: "
            f"{len(vertices):,} verts, {len(faces):,} faces | "
            f"merge={merge_time:.3f}s, sim={sim_time:.3f}s, total={frame_time:.3f}s"
        )

    total_time = time.perf_counter() - total_start
    avg_time = total_time / num_frames if num_frames > 0 else 0
    logger.info(f"libkinect simulation complete: {total_time:.2f}s total, {avg_time:.2f}s/frame avg")

    return results


def right_stereo_camera_pose(cam2world: np.ndarray, baseline: float) -> np.ndarray:
    """Translate a camera along its local +X axis without changing orientation."""
    pose = np.asarray(cam2world, dtype=np.float64).copy()
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("cam2world must be a finite 4x4 matrix")
    if baseline <= 0:
        raise ValueError("stereo baseline must be positive")
    pose[:3, 3] += baseline * pose[:3, 0]
    return pose


def append_right_stereo_camera_poses(baseline: float) -> int:
    """Append one rectified right-camera pose for each canonical camera pose."""
    primary_frame_count = int(cast(Any, bpy.context.scene).frame_end)
    if primary_frame_count <= 0:
        raise RuntimeError("stereo depth requested without canonical camera poses")
    primary_poses = [np.asarray(bproc.camera.get_camera_pose(frame)).copy() for frame in range(primary_frame_count)]
    for frame, pose in enumerate(primary_poses, start=primary_frame_count):
        bproc.camera.add_camera_pose(right_stereo_camera_pose(pose, baseline), frame=frame)
    bproc.camera.set_stereo_parameters(
        interocular_distance=baseline,
        convergence_mode="PARALLEL",
        convergence_distance=0.00001,
    )
    return primary_frame_count


def extract_stereo_depth(
    data: dict[str, Any],
    primary_frame_count: int,
    cfg: "Config",
) -> None:
    """Compute SGM depth and remove right-camera outputs from the canonical render data."""
    colors = data.get("colors")
    expected_frames = 2 * primary_frame_count
    if not isinstance(colors, (list, np.ndarray)) or len(colors) != expected_frames:
        raise RuntimeError(
            f"stereo RGB output has {len(colors) if colors is not None else 0} frames, expected {expected_frames}"
        )
    stereo_pairs = [
        np.stack((np.asarray(colors[frame])[..., :3], np.asarray(colors[primary_frame_count + frame])[..., :3]))
        for frame in range(primary_frame_count)
    ]
    stereo_depths, _disparities = bproc.postprocessing.stereo_global_matching(
        stereo_pairs,
        depth_max=cfg.stereo_depth_max,
        window_size=cfg.stereo_window_size,
        num_disparities=cfg.stereo_num_disparities,
        disparity_filter=False,
        depth_completion=False,
    )
    for depth in stereo_depths:
        invalid = ~np.isfinite(depth) | (depth <= 0) | (depth >= cfg.stereo_depth_max)
        depth[invalid] = 0.0

    for key, value in list(data.items()):
        if isinstance(value, (list, np.ndarray)) and len(value) == expected_frames:
            data[key] = value[:primary_frame_count]
    data["stereo_depth"] = stereo_depths
    cast(Any, bpy.context.scene).frame_end = primary_frame_count


def coacd_decomposition(
    obj: MeshObject,
    threshold: float = 0.05,
    cache_dir: Path | None = None,
    apply_modifiers: bool = True,
) -> list[bpy.types.Object]:
    """Decompose mesh into convex parts using CoACD (SIGGRAPH 2022).

    Args:
        obj: BlenderProc MeshObject to decompose.
        threshold: Concavity threshold (lower = more parts, higher quality).
        cache_dir: Directory to cache decomposition results.
        apply_modifiers: Whether to apply modifiers before decomposition.

    Returns:
        List of Blender objects representing convex hull parts.
    """
    import coacd

    # Get evaluated mesh (with modifiers applied)
    if apply_modifiers:
        mesh = obj.blender_obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.copy()
    else:
        mesh = obj.blender_obj.data.copy()

    # Triangulate mesh (CoACD requires triangles)
    mesh.calc_loop_triangles()

    # Convert to numpy arrays
    verts = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
    faces = np.array(
        [[mesh.loops[li].vertex_index for li in tri.loops] for tri in mesh.loop_triangles],
        dtype=np.int32,
    )

    mesh_hash = decomposition_cache_key(
        verts,
        faces,
        method="coacd",
        parameters={"threshold": threshold, "format": 2},
    )
    cache_file = cache_dir / f"{mesh_hash}.obj" if cache_dir else None
    decomposition_started = time.perf_counter()

    # Check cache or run CoACD
    if cache_file and cache_file.exists():
        obj.set_cp("collision_decomposition_cache_hit", True)
        logger.info(f"Loading cached CoACD for '{obj.get_name()}' from {cache_file}")
        # Deselect all before import to isolate newly imported objects
        bpy.ops.object.select_all(action="DESELECT")
        existing_objs = set(cast(list[Any], cast(Any, bpy.context.scene).objects))
        bpy.ops.wm.obj_import(filepath=str(cache_file))
        # Get only the newly imported objects
        hulls = [o for o in cast(list[Any], cast(Any, bpy.context.scene).objects) if o not in existing_objs]
        logger.info(f"Loaded {len(hulls)} cached hulls for '{obj.get_name()}'")
    else:
        obj.set_cp("collision_decomposition_cache_hit", False)
        logger.info(
            f"Running CoACD on '{obj.get_name()}': {len(verts)} verts, {len(faces)} tris, threshold={threshold}"
        )
        coacd_mesh = coacd.Mesh(verts, faces)
        parts = coacd.run_coacd(coacd_mesh, threshold=threshold)
        logger.info(f"CoACD produced {len(parts)} convex parts for '{obj.get_name()}'")

        hulls = []
        for i, (hull_verts, hull_faces) in enumerate(parts):
            hull_mesh = bpy.data.meshes.new(f"{obj.get_name()}_hull_{i}")
            hull_mesh.from_pydata(hull_verts.tolist(), [], hull_faces.tolist())
            hull_mesh.update()
            hull_obj = bpy.data.objects.new(hull_mesh.name, hull_mesh)
            cast(Any, bpy.context.collection).objects.link(hull_obj)
            hulls.append(hull_obj)

        # Cache the result
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            bpy.ops.object.select_all(action="DESELECT")
            for h in hulls:
                h.select_set(True)
            temporary_cache_file = cache_file.with_name(f".{cache_file.name}.{os.getpid()}.tmp.obj")
            bpy.ops.wm.obj_export(
                filepath=str(temporary_cache_file),
                export_selected_objects=True,
            )
            temporary_cache_file.replace(cache_file)
            logger.info(f"Cached {len(hulls)} hulls for '{obj.get_name()}' to {cache_file}")

    obj.set_cp("collision_decomposition_cache_key", mesh_hash)
    obj.set_cp("collision_decomposition_part_count", len(hulls))
    obj.set_cp("collision_decomposition_seconds", time.perf_counter() - decomposition_started)
    bpy.data.meshes.remove(mesh)
    return hulls


def decomposition_cache_key(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    method: str,
    parameters: dict[str, Any],
) -> str:
    """Build a stable cache key from local mesh geometry and decomposition settings."""
    vertices_array = np.ascontiguousarray(vertices, dtype="<f8")
    faces_array = np.ascontiguousarray(faces, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(method.encode("utf-8"))
    digest.update(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.asarray(vertices_array.shape, dtype="<i8").tobytes())
    digest.update(vertices_array.tobytes())
    digest.update(np.asarray(faces_array.shape, dtype="<i8").tobytes())
    digest.update(faces_array.tobytes())
    return digest.hexdigest()


def run(cfg: Config):
    output_dir = cfg.output_dir
    object_path = cfg.object_path if isinstance(cfg.object_path, Path) else None
    if object_path is not None and "shapenet" in str(object_path).lower():
        obj_id = object_path.parent.name
        obj_category = object_path.parent.parent.name
        if cfg.camera.file and cfg.camera.file.is_dir():
            camera_file = cfg.camera.file / obj_category / obj_id / "parameters.npz"
            logger.debug(f"Set camera parameters file to: {camera_file}")
            cfg.camera.from_file(camera_file)
        if output_dir and not (obj_category == output_dir.parent.parent and obj_id == output_dir.parent):
            output_dir = output_dir / obj_category / obj_id
            logger.debug(f"Set output path to: {output_dir}")

    if output_dir.exists() and not cfg.overwrite:
        logger.info(f"Output directory {output_dir} already exists. Use --overwrite to overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    with stdout_redirected(enabled=cfg.quiet):
        bproc.init()
        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

    if "cycles" in cfg.engine.lower():
        bproc.renderer.set_max_amount_of_samples(cfg.max_samples)
        bproc.renderer.set_noise_threshold(cfg.noise_threshold)
        bproc.renderer.set_denoiser(cfg.denoiser)
    elif "eevee" in cfg.engine.lower():
        scene = cast(Any, bpy.context.scene)
        view_layer = cast(Any, bpy.context.view_layer)
        scene.render.engine = cfg.engine
        scene.eevee.use_raytracing = True
        scene.eevee.taa_render_samples = cfg.max_samples

        scene.render.compositor_device = "GPU"
        scene.render.use_compositing = False
        view_layer.use_pass_diffuse_color = False
        view_layer.use_pass_normal = False

    bproc.renderer.set_output_format(
        file_format="JPEG" if cfg.background is None else "PNG",
        enable_transparency=cfg.background is not None,
        jpg_quality=cfg.jpg_quality,
        view_transform=cfg.view_transform,
    )

    if object_path is None and str(cfg.object_path) == "primitives":
        objs = list()
        if not isinstance(cfg.num_objects, int):
            raise TypeError("Config.num_objects must be an integer for primitive generation")
        for i in range(cfg.num_objects):
            primitive_type = cfg.primitive_type
            if primitive_type == "random":
                types = ["sphere", "cube", "cone", "cylinder", "torus", "monkey"]
                primitive_type = random.choice(types)
            if primitive_type == "torus":
                bpy.ops.mesh.primitive_torus_add()
                obj = MeshObject(bpy.context.object)
            else:
                obj = bproc.object.create_primitive(primitive_type.upper())
            obj.set_cp("category_id", types.index(primitive_type) + 1)
            obj.set_name(f"{primitive_type.capitalize()} {i + 1}")
            obj.set_shading_mode(cfg.shading)
            if primitive_type in ["cube", "cylinder", "cone"] and random.random() < 0.5:
                obj.add_modifier("BEVEL")
                obj.set_name(f"{primitive_type.capitalize()} Bevel {i + 1}")
            objs.append(obj)
    elif object_path is not None and object_path.suffix == ".off":
        mesh = cast(trimesh.Trimesh, trimesh.load(object_path, force="mesh", process=False, validate=cfg.validate))
        obj = bproc.object.create_with_empty_mesh(object_name="off_object")
        obj.get_mesh().from_pydata(mesh.vertices, [], mesh.faces)
        if cfg.validate:
            obj.get_mesh().validate()
        ensure_render_material(obj)
        obj.set_shading_mode(cfg.shading)
        objs = [obj]
    elif object_path is not None and object_path.suffix in [".obj", ".glb", ".gltf"]:
        obj_path_str = str(object_path)
        is_shapenet = "shapenet" in obj_path_str.lower()
        if is_shapenet:
            if "v2" in obj_path_str.lower():
                obj_id = object_path.parent.parent.name
                obj_category = object_path.parent.parent.parent.name
            else:
                obj_id = object_path.parent.name
                obj_category = object_path.parent.parent.name

        with stdout_redirected(enabled=cfg.quiet):
            if object_path.suffix in [".glb", ".gltf"]:
                obj = import_gltf_as_mesh_object(obj_path_str)
                if is_shapenet:
                    obj.set_rotation_euler([-np.pi / 2, 0, 0])
            else:
                obj = bproc.loader.load_obj(
                    obj_path_str,
                    use_split_objects=False,
                    validate_meshes=cfg.validate,
                    forward_axis="Y",
                    up_axis="Z",
                )[0]
                if is_shapenet:
                    obj.add_modifier("EDGE_SPLIT")

        ensure_render_material(
            obj,
            materialless_object_profile_for_path(obj_path_str, cfg.materialless_object_path_profiles),
        )
        obj.set_cp("source_asset_path", obj_path_str)
        obj.set_shading_mode(cfg.shading)
        if is_shapenet:
            obj.set_name(f"{obj_category}_{obj_id}")
            obj.set_cp("category_id", int(obj_category))
            bproc.python.loader.ShapeNetLoader._ShapeNetLoader.correct_materials(obj)
            obj.persist_transformation_into_mesh()
            obj.set_rotation_euler([np.pi / 2, 0, 0])
            if obj_category == "02958343":
                obj.get_mesh().flip_normals()
        else:
            loader_identity = loader_category_name_from_path(object_path)
            if loader_identity is None:
                obj.set_name(object_path.parent.name)
                obj.set_cp("category_id", 1)
            else:
                obj_category, obj_id = loader_identity
                obj.set_name(f"{obj_category}_{obj_id}_1")
                obj.set_cp("category_id", int(obj_category))
        objs = [obj]
    elif object_path is not None and object_path.suffix == ".txt":
        if not isinstance(cfg.num_objects, int):
            raise TypeError("Config.num_objects must be an integer when sampling objects from a file")
        obj_paths = sample_from_file_weighted(object_path, cfg.num_objects)
        objs = list()
        for i, obj_path in enumerate(obj_paths):
            if obj_path.with_suffix(".glb").exists():
                obj_path = obj_path.with_suffix(".glb")

            obj_path_str = str(obj_path)
            is_shapenet = "shapenet" in obj_path_str.lower()
            if is_shapenet:
                if "v2" in obj_path_str.lower():
                    obj_id = obj_path.parent.parent.name
                    obj_category = obj_path.parent.parent.parent.name
                else:
                    obj_id = obj_path.parent.name
                    obj_category = obj_path.parent.parent.name

            with stdout_redirected(enabled=cfg.quiet):
                if obj_path.suffix in [".glb", ".gltf"]:
                    obj = import_gltf_as_mesh_object(obj_path_str)
                    if is_shapenet:
                        obj.set_rotation_euler([-np.pi / 2, 0, 0])
                else:
                    obj = bproc.loader.load_obj(
                        str(obj_path),
                        use_split_objects=False,
                        validate_meshes=cfg.validate,
                        forward_axis="Y",
                        up_axis="Z",
                    )[0]
                    if is_shapenet:
                        obj.add_modifier("EDGE_SPLIT")

            ensure_render_material(
                obj,
                materialless_object_profile_for_path(obj_path_str, cfg.materialless_object_path_profiles),
            )
            obj.set_cp("source_asset_path", obj_path_str)
            obj.set_shading_mode(cfg.shading)
            if is_shapenet:
                obj.set_name(f"{obj_category}_{obj_id}_{i + 1}")
                obj.set_cp("category_id", int(obj_category))
                bproc.python.loader.ShapeNetLoader._ShapeNetLoader.correct_materials(obj)
                obj.persist_transformation_into_mesh()
                obj.set_rotation_euler([np.pi / 2, 0, 0])
                if obj_category == "02958343":
                    obj.get_mesh().flip_normals()
            else:
                loader_identity = loader_category_name_from_path(obj_path)
                if loader_identity is None:
                    obj.set_name(f"{obj_path.stem.capitalize()} {i + 1}")
                    obj.set_cp("category_id", i + 1)
                else:
                    obj_category, obj_id = loader_identity
                    obj.set_name(f"{obj_category}_{obj_id}_{i + 1}")
                    obj.set_cp("category_id", int(obj_category))
            objs.append(obj)
    else:
        suffix = object_path.suffix if object_path is not None else str(cfg.object_path)
        raise ValueError(f"Unsupported object file format: {suffix}.")

    scales = list()
    for i, obj in enumerate(objs):
        if cfg.fix_normals or cfg.cleanup:
            obj.edit_mode()
            bpy.ops.mesh.select_all(action="SELECT")
            if cfg.cleanup:
                bpy.ops.mesh.delete_loose()
                bpy.ops.mesh.dissolve_degenerate()
                bpy.ops.mesh.remove_doubles()
            if cfg.fix_normals:
                bpy.ops.mesh.normals_make_consistent(inside=False)
            obj.object_mode()
        if cfg.flip_normals:
            obj.get_mesh().flip_normals()
        if cfg.clear_normals:
            cast(Any, bpy.context.view_layer).objects.active = obj.blender_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

        mesh_bounds = np.asarray(obj.mesh_as_trimesh().bounds, dtype=np.float64)
        placement_bounds = mesh_bounds
        if cfg.normalize:
            translation, scale = normalization_parameters(mesh_bounds)
            obj.set_location(translation)
            obj.set_scale([scale] * 3)
            obj.persist_transformation_into_mesh(rotation=False)
            placement_bounds = mesh_bounds * scale + translation

        if cfg.add_uv:
            obj.add_uv_mapping(projection="smart")

        if not callable(cfg.scale):
            raise TypeError("Config.scale must be callable after initialization")
        obj.set_scale(cfg.scale())
        obj.set_cp("placement_bounds_min", placement_bounds[0].tolist())
        obj.set_cp("placement_bounds_max", placement_bounds[1].tolist())
        scales.append(obj.get_scale())

        if not any(obj.get_materials()):
            obj.new_material(f"Material {i + 1}")

        if cfg.backface_culling:
            for material in obj.get_materials():
                if material is not None:
                    engine = "BLENDER_EEVEE_NEXT" if bpy.app.version >= (2, 93) else "BLENDER_EEVEE"
                    cast(Any, bpy.context.scene).render.engine = engine
                    material.blender_obj.use_backface_culling = True
                    cast(Any, bpy.context.scene).render.engine = cfg.engine
                    if cfg.engine == "CYCLES":
                        # From https://github.com/DLR-RM/BlenderProc/issues/634
                        principled_bsdf_node = material.get_the_one_node_with_type("BsdfPrincipled")
                        material_output_node = material.get_the_one_node_with_type("OutputMaterial")
                        mix_shader_node = material.new_node("ShaderNodeMixShader")
                        geometry_node = material.new_node("ShaderNodeNewGeometry")
                        transparent_bsdf_node = material.new_node("ShaderNodeBsdfTransparent")

                        material.links.new(
                            geometry_node.outputs["Backfacing"],
                            mix_shader_node.inputs[0],
                        )
                        material.links.new(
                            transparent_bsdf_node.outputs["BSDF"],
                            mix_shader_node.inputs[2],
                        )
                        material.insert_node_instead_existing_link(
                            principled_bsdf_node.outputs["BSDF"],
                            mix_shader_node.inputs[1],
                            mix_shader_node.outputs["Shader"],
                            material_output_node.inputs["Surface"],
                        )
        if cfg.solidify:
            obj.add_modifier("SOLIDIFY", thickness=cfg.solidify)
        # NOTE: Rigidbody and VHACD are set up AFTER pose sampling (see below)
        # to avoid hull children interfering with collision checks

    if cfg.cc_material_path:
        cc_materials = bproc.loader.load_ccmaterials(
            cfg.cc_material_path,
            used_assets=cfg.cc_material_assets,
            preload=True,
        )

    with stdout_redirected(enabled=cfg.quiet):
        if cfg.surface in {"plane", "table"}:
            if cfg.surface == "plane":
                plane = bproc.object.create_primitive("PLANE", size=cfg.surface_size, location=(0, 0, 0))
            else:
                plane = bproc.object.create_primitive(
                    "CUBE",
                    location=(0, 0, -cfg.surface_thickness / 2.0),
                    scale=(cfg.surface_size / 2.0, cfg.surface_size / 2.0, cfg.surface_thickness / 2.0),
                )
            plane.new_material("Plane Material")
            if cfg.cc_material_path:
                replace_materials([plane], cc_materials, p=1.0)
            elif cfg.surface_material_profile == "tabletop":
                randomize_tabletop_surface(plane.get_materials()[0])
            else:
                randomize_materials(
                    [plane],
                    specular=0.5,
                    roughness=0.5,
                    metallic=0.1,
                    color=cfg.colors,
                    displacement=cfg.displacement,
                )

            # Configure spawn parameters
            spawn_height = cfg.spawn_height or (0.02, 1.0)
            # Default XY bounds: (-0.5, 0.5); configurable via spawn_bounds
            if cfg.spawn_bounds is None:
                bounds_xy = (-0.5, 0.5)
            elif cfg.spawn_bounds[0] < 0:
                # Absolute bounds (e.g., -0.3, 0.3)
                bounds_xy = cfg.spawn_bounds
            else:
                # Relative bounds (0-1) - convert to absolute based on plane size
                plane_half = cfg.surface_size / 2.0
                bounds_xy = (
                    -plane_half * cfg.spawn_bounds[1],
                    plane_half * cfg.spawn_bounds[1],
                )

            # Enable plane rigidbody once before any physics (shared by all placement modes)
            if cfg.physics:
                plane.enable_rigidbody(
                    active=False,
                    collision_shape="BOX",
                    collision_margin=cfg.collision_margin,
                    friction=cfg.rigidbody_friction,
                    linear_damping=cfg.linear_damping,
                    angular_damping=cfg.angular_damping,
                )

            if cfg.placement == "sequential":
                # Sequential dropping (VGN-style): drop objects one at a time with physics settling
                # Drop height is adaptive (just above current pile)
                logger.info(f"Sequential placement: XY={bounds_xy}")

                containment_walls = []
                if cfg.containment_walls:
                    spawn_half = max(abs(bounds_xy[0]), abs(bounds_xy[1]))
                    wall_size = 2.0 * (spawn_half + cfg.pile_wall_margin)
                    containment_walls = create_containment_walls(
                        size=wall_size,
                        height=1.5,
                        center=(0.0, 0.0),
                    )

                # Drop objects sequentially (adaptive height above pile)
                dropped_objects = sequential_drop_objects(
                    objs=objs,
                    cfg=cfg,
                    plane=plane,
                    walls=containment_walls,
                    bounds_xy=bounds_xy,
                    drop_margin=cfg.pile_drop_margin,
                    batch_size=cfg.pile_batch_size,
                    settle_time=cfg.pile_settle_time,
                )

                # Final settling pass - let whole pile relax (terminates early if already at rest)
                if dropped_objects:
                    logger.debug("Running final settling pass for sequential pile")
                    bproc.object.simulate_physics_and_fix_final_poses(
                        min_simulation_time=cfg.physics_min_simulation_time,
                        max_simulation_time=cfg.physics_max_simulation_time,
                        check_object_interval=cfg.physics_check_interval,
                        substeps_per_frame=cfg.physics_substeps_per_frame,
                        solver_iters=cfg.physics_solver_iters,
                    )
                else:
                    logger.warning("Skipping final pile settling because placement rejected every object")

                if containment_walls:
                    bproc.object.delete_multiple(containment_walls)
                    logger.debug("Removed containment walls after sequential pile settling")

                # Skip later physics block (already handled)
                containment_walls = []

            elif cfg.placement == "volume":
                # Volume placement (pile-style): sample in 3D volume, drop all at once
                logger.info(f"Volume placement: XY={bounds_xy}, Z={spawn_height}")
                p_fn = partial(
                    volume_pose_fn,
                    bounds_xy=bounds_xy,
                    bounds_z=spawn_height,
                    rotation=cfg.rotation,
                    upright=cfg.upright,
                    upright_x_rotation=cfg.upright_x_rotation,
                    upright_x_rotation_path_overrides=cfg.upright_x_rotation_path_overrides,
                )
                results = bproc.object.sample_poses(objs, p_fn, max_tries=100)
                # Log placement failures
                failed = [obj.get_name() for obj, (_, success) in results.items() if not success]
                if failed:
                    logger.warning(f"{len(failed)} objects failed initial placement: {failed[:3]}...")

            elif cfg.placement == "tower":
                # Tower placement: vertical column at random XY, collapses with physics
                logger.info(f"Tower placement: XY={bounds_xy}")
                place_objects_tower(
                    objs=objs,
                    bounds_xy=bounds_xy,
                    jitter_xy=0.02,  # Small jitter for natural instability
                    base_z=0.02,  # Above ground plane (margin for bbox precision)
                    gap=0.02,
                    rotation=cfg.rotation,
                    upright=cfg.upright,
                    upright_x_rotation=cfg.upright_x_rotation,
                    upright_x_rotation_path_overrides=cfg.upright_x_rotation_path_overrides,
                )

            elif cfg.placement == "surface_aabb":
                logger.info(f"AABB-safe surface placement: XY={bounds_xy}")
                placed = place_objects_surface_aabb(
                    objs=objs,
                    cfg=cfg,
                    bounds_xy=bounds_xy,
                    spawn_height=spawn_height,
                )
                failed_count = len(objs) - len(placed)
                if failed_count > 0:
                    logger.warning(f"{failed_count}/{len(objs)} objects failed AABB-safe placement (hidden)")

            else:
                # Surface placement (packed-style): sample above surface with distance constraints
                # Convert bounds_xy to face_sample_range (0-1 fraction of plane)
                plane_size = cfg.surface_size
                face_sample_range = (
                    (bounds_xy[0] + plane_size / 2) / plane_size,
                    (bounds_xy[1] + plane_size / 2) / plane_size,
                )
                logger.info(f"Surface placement: XY={bounds_xy} (face_range={face_sample_range})")
                p_fn = partial(
                    upper_region_pose_fn,
                    surface=plane,
                    rotation=cfg.rotation,
                    upright=cfg.upright,
                    upright_x_rotation=cfg.upright_x_rotation,
                    upright_x_rotation_path_overrides=cfg.upright_x_rotation_path_overrides,
                    spawn_height=spawn_height,
                    spawn_bounds=face_sample_range,
                )
                placed = bproc.object.sample_poses_on_surface(
                    objs,
                    plane,
                    p_fn,
                    max_tries=100,
                    min_distance=0.01,
                    max_distance=0.3,
                )
                # Log placement results
                failed_count = len(objs) - len(placed)
                if failed_count > 0:
                    logger.warning(f"{failed_count}/{len(objs)} objects failed surface placement (hidden)")

        elif cfg.position or cfg.rotation:
            position = cfg.position if isinstance(cfg.position, tuple) else (-1.0, 1.0)
            p_fn = partial(random_pose_fn, position=position, rotation=cfg.rotation)
            bproc.object.sample_poses(objs, p_fn, max_tries=100)

        # Create containment walls for pile-style physics (skip for sequential - handled above)
        containment_walls = []
        if cfg.containment_walls and cfg.physics and cfg.placement != "sequential":
            wall_size = abs(bounds_xy[1]) * 2 + 0.5  # bounds + margin
            containment_walls = create_containment_walls(size=wall_size, height=1.5)

        # Run physics simulation (skip for sequential - handled above)
        if cfg.physics and cfg.placement != "sequential":
            # Enable rigidbody and convex decomposition AFTER pose sampling (BlenderProc recommended order)
            # This prevents hull children from interfering with sample_poses collision checks
            physics_objs = [obj for obj in objs if not is_render_hidden(obj)]
            for obj in physics_objs:
                enable_rigidbody_with_decomposition(obj, cfg)

            # Simulation parameters (Blender defaults: substeps=10, solver_iters=10)
            # Higher values for stability with convex decomposition and stacking
            if physics_objs:
                bproc.object.simulate_physics_and_fix_final_poses(
                    min_simulation_time=cfg.physics_min_simulation_time,
                    max_simulation_time=cfg.physics_max_simulation_time,
                    check_object_interval=cfg.physics_check_interval,
                    substeps_per_frame=cfg.physics_substeps_per_frame,
                    solver_iters=cfg.physics_solver_iters,
                    verbose=cfg.verbose,
                )
            else:
                logger.warning("Skipping physics because placement rejected every object")

        # Remove containment walls after physics simulation
        if containment_walls:
            bproc.object.delete_multiple(containment_walls)
            logger.debug("Removed containment walls after physics simulation")

        if cfg.physics and any(
            value is not None
            for value in (cfg.physics_reject_max_abs_xy, cfg.physics_reject_min_z, cfg.physics_reject_max_z)
        ):
            reject_unstable_physics_objects(objs, cfg)

        if cfg.physics and cfg.correct_floor_penetration and cfg.surface in {"plane", "table"}:
            correct_objects_below_floor(objs)

    bproc.renderer.set_world_background(
        color=np.ones(4),
        strength=0 if cfg.hdri_path else cfg.world_background_strength,
    )
    bproc.camera.set_resolution(cfg.camera.width, cfg.camera.height)
    bproc.camera.set_intrinsics_from_K_matrix(
        K=cfg.camera.intrinsics,
        image_width=cfg.camera.width,
        image_height=cfg.camera.height,
        clip_start=cfg.camera.near,
        clip_end=cfg.camera.far,
    )

    if cfg.hdri_path:
        hdri_path = sample_hdri_path(cfg)
        hdri_rotation = sample_hdri_rotation_euler(cfg.object_path, cfg.randomize_hdri_rotation)
        bproc.world.set_world_background_hdr_img(
            hdri_path,
            strength=np.random.uniform(0.5, 1.5) if cfg.hdri_strength == "random" else cfg.hdri_strength,
            rotation_euler=hdri_rotation,
        )
        if cfg.scene_metadata is not None:
            randomization_metadata = cfg.scene_metadata.setdefault("randomization", {})
            randomization_metadata["selected_hdri"] = Path(hdri_path).stem
            randomization_metadata["hdri_rotation_euler_rad"] = hdri_rotation.tolist()
    if cfg.lights or cfg.randomize_lights:
        set_random_light(cfg)
    if cfg.replace:
        replace_materials(objs, cc_materials, p=cfg.replace)
    if cfg.cc_material_path:
        bproc.loader.load_ccmaterials(
            cfg.cc_material_path,
            used_assets=cfg.cc_material_assets,
            fill_used_empty_materials=True,
        )
    if cfg.materials or cfg.randomize_materials or cfg.colors or cfg.randomize_colors:
        if cfg.materials or cfg.randomize_materials:
            randomize_materials(
                objs,
                specular=0.5,
                roughness=0.5,
                metallic=0.1,
                color=cfg.colors or cfg.randomize_colors,
                displacement=cfg.displacement,
            )
        else:
            randomize_materials(objs, specular=0, roughness=0, metallic=0, color=True)

    primary_frame_count = 0
    if cfg.camera.extrinsics is not None:
        if isinstance(cfg.camera.extrinsics, (np.ndarray, list)):
            for frame, pose in enumerate(cfg.camera.extrinsics):
                pose = convert_extrinsic(pose, cfg.camera.convention, "opencv")
                bproc.camera.add_camera_pose(inv_trafo(pose), frame=frame)

                if cfg.camera.data is not None and "scales" in cfg.camera.data:
                    scale = cfg.camera.data["scales"][frame]
                    for s, obj in zip(scales, objs, strict=False):
                        obj.set_scale(s * scale, frame=frame)
        else:
            bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)
            frame = int(cast(Any, bpy.context.scene).frame_end)
            while frame < cfg.camera.extrinsics:
                poi_objs = np.random.choice(objs, size=np.random.randint(1, len(objs))) if len(objs) > 1 else objs
                poi = bproc.object.compute_poi(poi_objs)
                if cfg.camera.jitter:
                    poi += np.random.uniform(*cfg.camera.jitter, size=3)
                logger.debug(f"Point of interest: {poi}")

                radius = sample_truncnorm(scale=(0.2, 2.0), size=1)
                logger.debug(f"Sampling camera location with radius: {radius}")
                if not callable(cfg.camera.sampler):
                    raise ValueError("Camera sampler must be callable when sampling random camera poses")
                location = cfg.camera.sampler(
                    center=poi,
                    radius_min=radius,
                    radius_max=radius,
                    elevation_min=5,
                    elevation_max=89,
                )
                logger.debug(f"Camera location: {location}")

                inplane_rot = None
                if cfg.camera.inplane_rotation:
                    ip_rot = np.deg2rad(cfg.camera.inplane_rotation)
                    inplane_rot = np.random.uniform(-ip_rot, ip_rot)
                rotation_matrix = bproc.camera.rotation_from_forward_vec(
                    forward_vec=poi - location, inplane_rot=inplane_rot
                )
                cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
                if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.25}, bvh_tree):
                    frame = bproc.camera.add_camera_pose(cam2world_matrix) + 1

        primary_frame_count = int(cast(Any, bpy.context.scene).frame_end)
        if cfg.stereo_depth:
            primary_frame_count = append_right_stereo_camera_poses(cfg.stereo_baseline)

        if cfg.normals:
            bproc.renderer.enable_normals_output()
        if cfg.depth:
            bproc.renderer.enable_depth_output(activate_antialiasing=False)
        if cfg.diffuse:
            bproc.renderer.enable_diffuse_color_output()
        if cfg.segmentation:
            enable_segmentation_output_compat(
                map_by=["category_id", "instance", "name"],
                default_values={"category_id": 0},
            )

        if cfg.randomize_hdri or cfg.randomize_lights or cfg.randomize_materials or cfg.randomize_colors:
            if isinstance(cfg.camera.extrinsics, np.ndarray):
                n_frames = len(cfg.camera.extrinsics)
            elif isinstance(cfg.camera.extrinsics, int):
                n_frames = int(cfg.camera.extrinsics)
            else:
                n_frames = 0
            for frame in trange(
                n_frames,
                disable=not cfg.progress or cfg.quiet or cfg.verbose,
            ):
                with stdout_redirected(enabled=not cfg.verbose):
                    bproc.utility.set_keyframe_render_interval(frame, frame + 1)
                    if frame == 0:
                        data = bproc.renderer.render(verbose=cfg.verbose)
                    else:
                        if cfg.randomize_hdri:
                            set_random_hdri(cfg)
                        if cfg.randomize_lights:
                            set_random_light(cfg)
                        if cfg.randomize_materials or cfg.randomize_colors:
                            if cfg.randomize_materials:
                                randomize_materials(
                                    objs,
                                    specular=0.5,
                                    roughness=0.5,
                                    metallic=0.1,
                                    color=cfg.randomize_colors,
                                )
                            else:
                                randomize_materials(
                                    objs,
                                    specular=0,
                                    roughness=0,
                                    metallic=0,
                                    color=True,
                                )

                        for key, value in bproc.renderer.render(verbose=cfg.verbose).items():
                            data[key].extend(value)
        else:
            with stdout_redirected(enabled=not cfg.progress or cfg.quiet):
                data = bproc.renderer.render(verbose=cfg.verbose)
    else:
        pose = bproc.math.build_transformation_mat(cfg.camera.position, cfg.camera.rotation)
        bproc.camera.add_camera_pose(pose)
        primary_frame_count = int(cast(Any, bpy.context.scene).frame_end)
        if cfg.stereo_depth:
            primary_frame_count = append_right_stereo_camera_poses(cfg.stereo_baseline)
        if cfg.normals:
            bproc.renderer.enable_normals_output()
        if cfg.depth:
            bproc.renderer.enable_depth_output(activate_antialiasing=False)
        if cfg.diffuse:
            bproc.renderer.enable_diffuse_color_output()
        if cfg.segmentation:
            enable_segmentation_output_compat(
                map_by=["category_id", "instance", "name"],
                default_values={"category_id": 0},
            )
        with stdout_redirected(enabled=not cfg.progress or cfg.quiet):
            data = bproc.renderer.render(verbose=cfg.verbose)

    if cfg.stereo_depth:
        extract_stereo_depth(data, primary_frame_count, cfg)

    if cfg.mask:
        masks = list()
        if cfg.depth:
            for depth in data["depth"]:
                mask = depth == depth.max()
                masks.append(mask)
                depth[mask] = 0
        if cfg.normals:
            for i, normals in enumerate(data["normals"]):
                mask = normals == (0.5, 0.5, 0.5)
                if masks:
                    mask = masks[i]
                normals[mask] = 0
        if cfg.diffuse:
            for i, diffuse in enumerate(data["diffuse"]):
                mask = diffuse.sum(axis=-1) <= 3
                if masks:
                    mask = masks[i]
                diffuse[mask] = 0
        if cfg.kinect:
            logger.debug("Adding Kinect Azure noise to depth images.")
            data["kinect"] = bproc.postprocessing.add_kinect_azure_noise(
                data["depth"],
                data["colors"],
                missing_depth_darkness_thres=cfg.kinect_darkness_threshold,
            )
            for depth, kinect in zip(data["depth"], data["kinect"], strict=False):
                kinect[depth == 0] = 0

        if cfg.kinect_sim:
            surface_obj = plane if cfg.surface in {"plane", "table"} else None
            data["kinect_sim"] = simulate_kinect_depth(objs, surface_obj, cfg, len(data["depth"]))

    if cfg.writer:
        if cfg.overwrite:
            shutil.rmtree(cfg.output_dir, ignore_errors=True)

        if cfg.writer == "bop":
            bproc.writer.write_bop(
                output_dir=cfg.output_dir,
                target_objects=objs,
                depths=data.get("depth"),
                colors=data.get("colors"),
                append_to_existing_output=not cfg.overwrite,
                depth_scale=1000.0,
            )
            data.pop("colors", None)
            data.pop("depth", None)
        elif cfg.writer == "coco":
            bproc.writer.write_coco_annotations(
                output_dir=cfg.output_dir,
                instance_segmaps=data.get("instance_segmaps"),
                instance_attribute_maps=data.get("instance_attribute_maps"),
                colors=data.get("colors"),
                append_to_existing_output=not cfg.overwrite,
                label_mapping=cfg.metadata,
            )
            data.pop("colors", None)
            data.pop("instance_segmaps", None)
            data.pop("category_id_segmaps", None)
            data.pop("instance_attribute_maps", None)
        if cfg.writer == "usd" or str(cfg.object_path) == "primitives":
            usd_export_settings = {
                "selected_objects_only": False,
                "visible_objects_only": True,
                "export_animation": False,
                "export_materials": cfg.writer == "usd",
                "export_uvmaps": cfg.writer == "usd",
                "export_normals": cfg.writer == "usd",
                "use_instancing": True,
                "evaluation_mode": "RENDER",
                "relative_paths": True,
            }
            bpy.ops.wm.usd_export(filepath=str(cfg.output_dir / "scene.usd"), **usd_export_settings)
            shutil.rmtree(cfg.output_dir / "textures", ignore_errors=True)

        frame_data = list()
        for frame in range(primary_frame_count):
            frame_data.append(
                {
                    "names": [obj.get_name() for obj in objs],
                    "categories": [obj.get_cp("category_id") for obj in objs],
                    "scales": [obj.get_scale() for obj in objs],
                    "poses": [obj.get_local2world_mat() for obj in objs],
                    "intrinsic": bproc.camera.get_intrinsics_as_K_matrix(),
                    "extrinsic": convert_extrinsic(
                        inv_trafo(bproc.camera.get_camera_pose(frame)),
                        "opengl",
                        "opencv",
                    ),
                    "surface": cfg.surface,
                    "scene_metadata": cfg.scene_metadata,
                }
            )
        data["data"] = frame_data
        bproc.writer.write_hdf5(
            output_dir_path=cfg.output_dir,
            output_data_dict=data,
            append_to_existing_output=not cfg.overwrite,
        )

        return

    masks = data.get("depth")
    for obj_id, image_data in data.items():
        (output_dir / obj_id).mkdir(exist_ok=True)

        for frame, values in enumerate(image_data):
            output_file = output_dir / obj_id / f"{frame:05d}"

            mask = None
            if masks is not None:
                mask = masks[frame] == masks[frame].max()

            if obj_id == "normals":
                if mask is None:
                    mask = np.all(values == (0.5, 0.5, 0.5), axis=-1)
                values[mask] = 0
                cv2.imwrite(
                    str(output_file.with_suffix(".exr")),
                    cv2.cvtColor(values, cv2.COLOR_RGB2BGR),
                )
            elif obj_id == "depth":
                if mask is None:
                    mask = values == values.max()
                values[mask] = 0
                cv2.imwrite(str(output_file.with_suffix(".exr")), values)
            elif obj_id == "diffuse":
                if mask is None:
                    mask = values.sum(axis=-1) <= 3
                values[mask] = 0
                Image.fromarray(values).save(output_file.with_suffix(".png"))
            elif obj_id == "colors":
                if cfg.background:
                    values[values[..., 3] == 0] = 0
                image = Image.fromarray(values)
                if cfg.background == "transparent":
                    image.save(output_file.with_suffix(".png"))
                elif cfg.background:
                    background = Image.new(
                        mode="RGBA",
                        size=image.size,
                        color=tuple(np.random.randint(255, size=3)) if cfg.background == "random" else cfg.background,
                    )
                    image = Image.alpha_composite(background, image)
                    image.convert("RGB").save(output_file.with_suffix(".jpg"), quality=cfg.jpg_quality)
                else:
                    image.save(output_file.with_suffix(".jpg"), quality=cfg.jpg_quality)
            # elif name in ["instance_segmaps", "category_id_segmaps"]:
            #     Image.fromarray(data.astype(np.uint16)).save(output_file.with_suffix(".png"))

    logger.info(f"Images saved to {output_dir}")


def main(cfg: Config):
    start = time.perf_counter()
    run(cfg)
    logger.info(f"Runtime: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main(tyro.cli(Config))
