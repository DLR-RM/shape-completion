import blenderproc as bproc  # noqa: I001  # pyright: ignore[reportMissingImports]

import json
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
    spawn_height: tuple[float, float] | None = None
    """Min/max height for spawning objects above surface (overrides default 1-4m)."""
    spawn_bounds: tuple[float, float] | None = None
    """XY spawn bounds: fraction of surface for 'surface' placement (e.g., 0.4-0.6), absolute meters for 'volume' (e.g., -0.3 to 0.3)."""
    containment_walls: bool = False
    """Add temporary containment walls during physics simulation."""
    placement: Literal["surface", "volume", "sequential", "tower"] | None = None
    """Placement strategy: 'surface' (packed), 'volume' (simultaneous drop), 'sequential' (VGN-style pile), 'tower' (vertical column, collapses). Default: surface."""
    collision_shape: Literal["CONVEX_HULL", "MESH", "BOX", "COMPOUND"] = "CONVEX_HULL"
    """Collision shape for physics: CONVEX_HULL (fast, poor for concave), MESH (accurate, slow), BOX (fastest), COMPOUND (VHACD decomposition)."""
    decomposition: Literal["vhacd", "coacd"] = "vhacd"
    """Convex decomposition method for COMPOUND collision shape: CoACD (SIGGRAPH 2022, default) or V-HACD (fallback)."""
    coacd_threshold: float = 0.05
    """CoACD concavity threshold (lower = more parts, higher quality)."""
    coacd_path: Path = Path(__file__).resolve().parent.parent.parent / "libs" / "coacd"
    """Path to CoACD directory (for cache storage)."""
    vhacd_path: Path = Path(__file__).resolve().parent.parent.parent / "libs"  # Contains v-hacd/app/TestVHACD
    """Path to VHACD directory (auto-downloaded if not present)."""
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
    randomize_hdri: bool = False
    """Randomize the HDRI image for each frame."""
    lights: int | tuple[int, int] | None = None
    """Add (random) number of lights to add to the scene."""
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
    surface: Literal["plane", "table"] | None = None
    """Surface to place the objects on."""
    normals: bool = False
    """Enable normal output."""
    depth: bool = False
    """Enable depth output."""
    kinect: bool = False
    """Enable Kinect Azure noise on depth output."""
    kinect_sim: bool = False
    """Enable structured light Kinect v1 depth simulation (libkinect)."""
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

    def __post_init__(self):
        logger.remove()
        if self.quiet:
            logger.add(sys.stderr, level="ERROR")
        else:
            level = "DEBUG" if self.verbose else "INFO"
            logger.add(sys.stderr, level=level)

        if self.seed:
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
            self.num_objects = self.num_objects or 1
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

        if self.kinect_sim:
            self.depth = True

        if self.segmentation and not self.writer:
            self.writer = "bop"

        if self.replace:
            if not self.cc_material_path:
                raise ValueError("CC material path must be set when using texture replacement.")

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
                self.spawn_bounds = (
                    0.5 - spawn_half / 2.5,
                    0.5 + spawn_half / 2.5,
                )  # Relative to plane
            # upright=True is already default; placement defaults to surface at end of __post_init__
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
                self.placement = "volume"  # Use sample_poses, not sample_poses_on_surface
            self.containment_walls = True  # Prevent objects flying off
            if self.collision_shape == "CONVEX_HULL":
                self.collision_shape = "COMPOUND"  # Use VHACD for accurate concave object physics
            if not self.physics:
                self.physics = True  # Pile requires physics
                logger.warning("Enabling physics for pile scene (required for settling)")
            logger.info(
                f"Scene preset 'pile': {num_objs} objects, avg_scale={avg_scale:.2f}, collision={self.collision_shape}, bounds={self.spawn_bounds}"
            )

        # Auto-enable solidify for physics (thin-shell meshes need thickness for stability)
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


def upper_region_pose_fn(
    obj: bproc.types.MeshObject,
    surface: bproc.types.MeshObject | None = None,
    rotation: bool | tuple[float, float, float] | None = False,
    upright: bool = False,
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
            # Upright with random yaw (preserves π/2 X rotation for coordinate convention)
            obj.set_rotation_euler([np.pi / 2, 0, np.random.uniform(0, np.pi * 2)])
        else:
            # Full SO3 rotation
            obj.set_rotation_euler(bproc.sampler.uniformSO3())


def volume_pose_fn(
    obj: bproc.types.MeshObject,
    bounds_xy: tuple[float, float] = (-0.5, 0.5),
    bounds_z: tuple[float, float] = (0.1, 0.6),
    rotation: bool | tuple[float, float, float] | None = True,
    upright: bool = False,
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
            # Upright with random yaw (preserves π/2 X rotation for coordinate convention)
            obj.set_rotation_euler([np.pi / 2, 0, np.random.uniform(0, np.pi * 2)])
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
                obj.set_rotation_euler([np.pi / 2, 0, np.random.uniform(0, np.pi * 2)])
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
                collision_margin=0.0005,  # Smaller for COMPOUND to avoid gaps between parts
                friction=0.5,
                linear_damping=0.1,
                angular_damping=0.15,
            )
            # Then decompose and add children with CONVEX_HULL
            cache_dir = cfg.coacd_path / ".cache"
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
                collision_margin=0.0005,  # Smaller for COMPOUND to avoid gaps between parts
                friction=0.5,
                linear_damping=0.1,
                angular_damping=0.15,
            )
            obj.build_convex_decomposition_collision_shape(
                str(cfg.vhacd_path),
                cache_dir=str(cfg.vhacd_path / "v-hacd" / ".cache"),
            )
    else:
        obj.enable_rigidbody(
            active=True,
            collision_shape=cfg.collision_shape,
            collision_margin=0.001,  # BlenderProc default; Blender's 0.04 causes gaps
            friction=0.5,
            linear_damping=0.1,
            angular_damping=0.15,
        )


def sequential_drop_objects(
    objs: list[bproc.types.MeshObject],
    cfg: "Config",
    plane: bproc.types.MeshObject,
    walls: list[bproc.types.MeshObject],
    bounds_xy: tuple[float, float],
    drop_margin: tuple[float, float] = (0.02, 0.1),
    batch_size: int = 1,
    settle_time: float = 1.0,
) -> None:
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
    dropped_objs: list[bproc.types.MeshObject] = []
    pile_height = 0.0  # Track current pile height

    # Process in batches
    for i in range(0, len(objs), batch_size):
        batch = objs[i : i + batch_size]

        # Spawn batch at random positions above current pile
        for obj in batch:
            xy = np.random.uniform(bounds_xy[0], bounds_xy[1], 2)
            # Spawn just above current pile height
            z = pile_height + np.random.uniform(drop_margin[0], drop_margin[1])
            obj.set_location([xy[0], xy[1], z])

            # Random rotation
            if cfg.upright:
                # Only rotate around Z axis
                angle = np.random.uniform(0, 2 * np.pi)
                obj.set_rotation_euler([0, 0, angle])
            else:
                # Full SO3 rotation
                obj.set_rotation_euler(bproc.sampler.uniformSO3())

            # Enable rigidbody with proper decomposition (same logic as volume/surface)
            enable_rigidbody_with_decomposition(obj, cfg)

        # Run physics for this batch
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=settle_time,
            max_simulation_time=settle_time + 1.0,
            check_object_interval=0.25,
            substeps_per_frame=25,
            solver_iters=20,
        )

        # Track dropped objects and update pile height
        dropped_objs.extend(batch)
        pile_height = max(obj.get_bound_box()[:, 2].max() for obj in dropped_objs)

        logger.debug(
            f"Dropped batch {i // batch_size + 1}/{(len(objs) + batch_size - 1) // batch_size}, pile_height={pile_height:.3f}"
        )


def set_random_hdri(cfg: Config):
    world = cast(Any, bpy.context.scene).world
    nodes = world.node_tree.nodes
    texture_node = Utility.get_the_one_node_with_type(nodes, "TexEnvironment")
    hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(cfg.hdri_path)
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
    for i in range(num_lights):
        light = bproc.types.Light(name=f"Light {i}")
        location = bproc.sampler.shell(
            center=(0, 0, 0),
            radius_min=5,
            radius_max=10,
            elevation_min=1,
            elevation_max=89,
        )
        light.set_location(location)
        light.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
        light.set_energy(np.random.uniform(250 // num_lights))
        logger.debug(
            f"Created light {i} at location {light.get_location()} with {light.get_color()} and strength {light.get_energy()}"
        )


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
    sample_weights = [(1 / class_counts[c]) ** alpha for c in classes]

    return random.choices(population=obj_paths, weights=sample_weights, k=n_samples)


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

    # Compute hash for caching
    mesh_hash = abs(hash(verts.tobytes() + faces.tobytes()))
    cache_file = cache_dir / f"{mesh_hash}.obj" if cache_dir else None

    # Check cache or run CoACD
    if cache_file and cache_file.exists():
        logger.info(f"Loading cached CoACD for '{obj.get_name()}' from {cache_file}")
        # Deselect all before import to isolate newly imported objects
        bpy.ops.object.select_all(action="DESELECT")
        existing_objs = set(cast(list[Any], cast(Any, bpy.context.scene).objects))
        bpy.ops.wm.obj_import(filepath=str(cache_file))
        # Get only the newly imported objects
        hulls = [o for o in cast(list[Any], cast(Any, bpy.context.scene).objects) if o not in existing_objs]
        logger.info(f"Loaded {len(hulls)} cached hulls for '{obj.get_name()}'")
    else:
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
            bpy.ops.wm.obj_export(
                filepath=str(cache_file),
                export_selected_objects=True,
            )
            logger.info(f"Cached {len(hulls)} hulls for '{obj.get_name()}' to {cache_file}")

    bpy.data.meshes.remove(mesh)
    return hulls


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
        if cfg.seed:
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
                bpy.ops.import_scene.gltf(filepath=obj_path_str)
                bpy.ops.object.join()
                obj = MeshObject(bpy.context.object)
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

        obj.set_shading_mode(cfg.shading)
        if is_shapenet:
            obj.set_name(f"{obj_category}_{obj_id}")
            obj.set_cp("category_id", int(obj_category))
            bproc.python.loader.ShapeNetLoader._ShapeNetLoader.correct_materials(obj)
            obj.persist_transformation_into_mesh()
            obj.set_rotation_euler([np.pi / 2, 0, 0])
            if obj_category == "02958343":
                obj.get_mesh().flip_normals()
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
                    bpy.ops.import_scene.gltf(filepath=obj_path_str)
                    bpy.ops.object.join()
                    obj = MeshObject(bpy.context.object)
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
                obj.set_name(f"{obj_path.stem.capitalize()} {i + 1}")
                obj.set_cp("category_id", {i + 1})
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

        if cfg.normalize:
            mesh = obj.mesh_as_trimesh()
            offset = -mesh.bounds.mean(axis=0)
            scale = 1 / mesh.extents.max()
            obj.set_location(offset)
            obj.set_scale([scale] * 3)
            obj.persist_transformation_into_mesh(rotation=False)

        if cfg.add_uv:
            obj.add_uv_mapping(projection="smart")

        if not callable(cfg.scale):
            raise TypeError("Config.scale must be callable after initialization")
        obj.set_scale(cfg.scale())
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
        cc_materials = bproc.loader.load_ccmaterials(cfg.cc_material_path, preload=True)

    with stdout_redirected(enabled=cfg.quiet):
        if cfg.surface == "plane":
            plane = bproc.object.create_primitive("PLANE", size=5, location=(0, 0, 0))
            plane.new_material("Plane Material")
            if cfg.cc_material_path:
                replace_materials([plane], cc_materials, p=1.0)
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
                plane_half = 2.5  # Half the plane size (plane is size=5)
                bounds_xy = (
                    -plane_half * cfg.spawn_bounds[1],
                    plane_half * cfg.spawn_bounds[1],
                )

            # Enable plane rigidbody once before any physics (shared by all placement modes)
            if cfg.physics:
                plane.enable_rigidbody(
                    active=False,
                    collision_shape="BOX",
                    collision_margin=0.001,
                    friction=0.5,
                    linear_damping=0.1,
                    angular_damping=0.15,
                )

            if cfg.placement == "sequential":
                # Sequential dropping (VGN-style): drop objects one at a time with physics settling
                # Drop height is adaptive (just above current pile)
                logger.info(f"Sequential placement: XY={bounds_xy}")

                # Create containment walls covering the spawn area
                wall_size = abs(bounds_xy[1]) * 2 + 0.5  # bounds + margin
                containment_walls = create_containment_walls(size=wall_size, height=1.5, center=(0.0, 0.0))

                # Drop objects sequentially (adaptive height above pile)
                sequential_drop_objects(
                    objs=objs,
                    cfg=cfg,
                    plane=plane,
                    walls=containment_walls,
                    bounds_xy=bounds_xy,
                    drop_margin=(0.02, 0.1),  # 2-10cm above current pile
                    batch_size=1,  # True sequential; increase for speed
                    settle_time=3.0,
                )

                # Remove walls after all objects dropped
                bproc.object.delete_multiple(containment_walls)
                logger.debug("Removed containment walls after sequential dropping")

                # Final settling pass - let whole pile relax (terminates early if already at rest)
                logger.debug("Running final settling pass for sequential pile")
                bproc.object.simulate_physics_and_fix_final_poses(
                    min_simulation_time=0.5,  # Short min; early exit if settled
                    max_simulation_time=5.0,  # Allow time for complex piles to fully settle
                    check_object_interval=0.25,
                    substeps_per_frame=25,
                    solver_iters=20,
                )

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
                )

            else:
                # Surface placement (packed-style): sample above surface with distance constraints
                # Convert bounds_xy to face_sample_range (0-1 fraction of plane)
                plane_size = 5.0
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
            for obj in objs:
                enable_rigidbody_with_decomposition(obj, cfg)

            # Simulation parameters (Blender defaults: substeps=10, solver_iters=10)
            # Higher values for stability with convex decomposition and stacking
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=2,
                max_simulation_time=15,  # Allow time for complex pile settling
                check_object_interval=0.5,  # Check settling 2x more frequently than default
                substeps_per_frame=25,  # 2.5x default; stability for multi-contact scenarios
                solver_iters=20,  # 2x default; better constraint solving for stacking
                verbose=cfg.verbose,
            )

        # Remove containment walls after physics simulation
        if containment_walls:
            bproc.object.delete_multiple(containment_walls)
            logger.debug("Removed containment walls after physics simulation")

    bproc.renderer.set_world_background(color=np.ones(4), strength=0 if cfg.hdri_path else 0.1)
    bproc.camera.set_resolution(cfg.camera.width, cfg.camera.height)
    bproc.camera.set_intrinsics_from_K_matrix(
        K=cfg.camera.intrinsics,
        image_width=cfg.camera.width,
        image_height=cfg.camera.height,
        clip_start=cfg.camera.near,
        clip_end=cfg.camera.far,
    )

    if cfg.hdri_path:
        hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(cfg.hdri_path)
        bproc.world.set_world_background_hdr_img(
            hdri_path,
            strength=np.random.uniform(0.5, 1.5) if cfg.hdri_strength == "random" else cfg.hdri_strength,
            rotation_euler=[np.pi / 2, 0, 0] if object_path is not None and object_path.suffix == ".obj" else None,
        )
    if cfg.lights or cfg.randomize_lights:
        set_random_light(cfg)
    if cfg.replace:
        replace_materials(objs, cc_materials, p=cfg.replace)
    if cfg.cc_material_path:
        bproc.loader.load_ccmaterials(cfg.cc_material_path, fill_used_empty_materials=True)
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

        if cfg.normals:
            bproc.renderer.enable_normals_output()
        if cfg.depth:
            bproc.renderer.enable_depth_output(activate_antialiasing=False)
        if cfg.diffuse:
            bproc.renderer.enable_diffuse_color_output()
        if cfg.segmentation:
            bproc.renderer.enable_segmentation_output(
                map_by=["category_id", "instance", "name"],
                default_values={"category_id": 0},
            )

        if cfg.randomize_hdri or cfg.randomize_lights or cfg.randomize_materials:
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
        if cfg.normals:
            bproc.renderer.enable_normals_output()
        if cfg.depth:
            bproc.renderer.enable_depth_output(activate_antialiasing=False)
        if cfg.diffuse:
            bproc.renderer.enable_diffuse_color_output()
        if cfg.segmentation:
            bproc.renderer.enable_segmentation_output(
                map_by=["category_id", "instance", "name"],
                default_values={"category_id": 0},
            )
        with stdout_redirected(enabled=not cfg.progress or cfg.quiet):
            data = bproc.renderer.render(verbose=cfg.verbose)

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
            data["kinect"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"])
            for depth, kinect in zip(data["depth"], data["kinect"], strict=False):
                kinect[depth == 0] = 0

        if cfg.kinect_sim:
            surface_obj = plane if cfg.surface == "plane" else None
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
        for frame in range(int(cast(Any, bpy.context.scene).frame_end)):
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
