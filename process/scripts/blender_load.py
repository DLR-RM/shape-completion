# Usage (zsh):
#   blender --python scripts/blender_import_sc_data.py -- \
#       --dir /path/to/exported/output --stem <filename_stem>
#
# If --stem is omitted, the script will pick the first "*_camera.json" it finds.
"""Load shape-completion exports into Blender 4/5 scenes."""

import argparse
import ctypes.util
import json
import math
import os
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Literal, NamedTuple, cast

import bmesh
import bpy
import mathutils
import numpy as np
from loguru import logger
from mathutils import kdtree  # pyright: ignore[reportMissingModuleSource]


def _configure_logger(level: str = os.getenv("BLENDER_LOAD_LOG_LEVEL", "INFO")) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper())


_configure_logger()


def _require_scene() -> Any:
    scene = bpy.context.scene
    if scene is None:
        raise RuntimeError("No active Blender scene available.")
    return cast(Any, scene)


def _context_any() -> Any:
    return cast(Any, bpy.context)


def _normalize_color(value: Sequence[float] | tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError("Color tuples must contain exactly three components")
        return float(value[0]), float(value[1]), float(value[2])
    if isinstance(value, Sequence):
        seq = list(value)
        if len(seq) != 3:
            raise ValueError("Color sequences must contain exactly three components")
        return float(seq[0]), float(seq[1]), float(seq[2])
    raise TypeError("Color values must be a sequence of three numeric components or None")


def _normalize_vec3(value: Sequence[float] | None, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    seq = list(value)
    if len(seq) != 3:
        raise ValueError(f"{name} must contain exactly three components")
    return float(seq[0]), float(seq[1]), float(seq[2])


def _normalize_vec4(value: Sequence[float] | None, name: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    seq = list(value)
    if len(seq) != 4:
        raise ValueError(f"{name} must contain exactly four components")
    return float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3])


def _normalize_compositor_preset(value: str | None) -> Literal["off", "paper_clean", "beauty"]:
    preset = str(value or "paper_clean").strip().lower()
    aliases = {
        "none": "off",
        "false": "off",
        "0": "off",
        "paper": "paper_clean",
        "clean": "paper_clean",
    }
    preset = aliases.get(preset, preset)
    if preset == "off":
        return "off"
    if preset == "paper_clean":
        return "paper_clean"
    if preset == "beauty":
        return "beauty"
    raise ValueError(f"Invalid compositor preset '{value}'. Expected one of: ['beauty', 'off', 'paper_clean']")


def _get_node_tree(owner: object):
    """Return owner's node tree, enabling nodes only as a legacy fallback."""
    tree = getattr(owner, "node_tree", None)
    if tree is not None:
        return tree
    # Scene.use_nodes is deprecated in Blender 4.x and can warn on access.
    # For scenes we skip the legacy enablement path; AO has a material fallback.
    owner_type = type(owner).__name__
    if owner_type == "Scene":
        return None

    has_use_nodes = False
    try:
        bl_rna = getattr(owner, "bl_rna", None)
        props = getattr(bl_rna, "properties", None)
        if props is not None:
            _ = props["use_nodes"]
            has_use_nodes = True
    except Exception:
        has_use_nodes = False

    if has_use_nodes:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                cast(Any, owner).use_nodes = True
        except Exception:
            pass
    return getattr(owner, "node_tree", None)


def _set_node_input_default(node: object, names: Sequence[str], value: float) -> bool:
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        return False
    for name in names:
        try:
            socket = inputs[name]
        except Exception:
            continue
        try:
            socket.default_value = value
            return True
        except Exception:
            continue
    return False


def _apply_principled_product_look(
    principled_node: object,
    *,
    roughness: float,
    specular_level: float,
    coat_weight: float,
    coat_roughness: float,
    alpha: float = 1.0,
) -> None:
    """Set a neutral product-style Principled profile (Blender-version safe)."""
    roughness_v = _clamp_float(float(roughness), 0.0, 1.0)
    specular_v = _clamp_float(float(specular_level), 0.0, 1.0)
    coat_v = _clamp_float(float(coat_weight), 0.0, 1.0)
    coat_roughness_v = _clamp_float(float(coat_roughness), 0.0, 1.0)
    alpha_v = _clamp_float(float(alpha), 0.0, 1.0)
    _set_node_input_default(principled_node, ("Roughness",), roughness_v)
    _set_node_input_default(principled_node, ("Metallic",), 0.0)
    _set_node_input_default(principled_node, ("Specular IOR Level", "Specular"), specular_v)
    _set_node_input_default(principled_node, ("Coat Weight", "Coat"), coat_v)
    _set_node_input_default(principled_node, ("Coat Roughness",), coat_roughness_v)
    _set_node_input_default(principled_node, ("Alpha",), alpha_v)


@dataclass(frozen=True)
class LoaderConfig:
    directory: Path
    stem: str | None
    hdri: Path | None
    as_planes: bool = True
    plane_depth: float = 1.0
    add_camera_body: bool = True
    enable_lighting: bool = True
    lighting_samples: int = 384
    lighting_scale: float = 0.5
    inputs_subsample: float = 1.0
    render_width: int = 1024
    render_height: int = 768
    hdri_strength: float = 0.06
    create_object_cameras: bool = True
    shadow_catcher: bool = True
    lighting_hdr_strength: float = 0.06
    lighting_world_ambient_scale: float = 1.0
    lighting_distance_scale: float = 1.7
    lighting_fill_distance_ratio: float = 1.28
    lighting_rim_distance_ratio: float = 1.18
    lighting_key_energy_base: float = 100.0
    lighting_fill_ratio: float = 0.25
    lighting_rim_ratio: float = 0.68
    lighting_kicker_ratio: float = 0.03
    lighting_key_strength: float = 1.0
    lighting_fill_strength: float = 1.0
    lighting_rim_strength: float = 1.0
    lighting_kicker_strength: float = 1.0
    lighting_scene_scale: float = 0.9
    lighting_scene_fill_ratio_scale: float = 0.75
    lighting_scene_ambient_scale: float = 1.0
    lighting_solo_scale: float = 0.9
    lighting_solo_fill_ratio_scale: float = 1.6
    view_exposure: float = 0.0
    ao_strength: float = 0.3
    ao_distance: float = 0.2
    compositor_preset: Literal["off", "paper_clean", "beauty"] = "paper_clean"
    inputs_point_radius: float | None = None
    inputs_color: tuple[float, float, float] | None = None
    pcd_point_radius: float | None = None
    pcd_color: tuple[float, float, float] | None = None
    occ_point_radius: float | None = None
    occ_color: tuple[float, float, float] | None = None
    free_point_radius: float | None = None
    free_color: tuple[float, float, float] | None = (0.8, 0.8, 0.8)
    logits_point_radius: float | None = None
    logits_color: tuple[float, float, float] | None = None
    point_auto_enabled: bool = True
    point_auto_target: int | None = 100_000
    point_auto_radius_scale: float = 0.2
    point_auto_radius_min: float = 0.0012
    point_auto_radius_max: float = 0.012
    point_min_px: float = 1.0
    point_auto_use_nn: bool = True
    point_auto_nn_sample: int = 5000
    point_auto_nn_subsample_if_small: bool = True
    cycles_samples: int = 384
    cycles_denoise: bool = True
    cycles_preview_factor: float = 0.25
    cycles_adaptive_threshold: float = 0.03
    cycles_light_threshold: float = 0.01
    cycles_clamp_direct: float = 0.0
    cycles_clamp_indirect: float = 2.5
    cycles_blur_glossy: float = 0.0
    cycles_max_bounces: int = 4
    cycles_diffuse_bounces: int = 2
    cycles_glossy_bounces: int = 2
    cycles_transmission_bounces: int = 1
    cycles_transparent_max_bounces: int = 4
    cycles_caustics_reflective: bool = False
    cycles_caustics_refractive: bool = False
    group_visibility_overrides: dict[str, Literal["viewport", "render", "both"] | None] | None = None
    # Default camera overrides from CLI
    camera_location: tuple[float, float, float] | None = None
    camera_rotation_euler_deg: tuple[float, float, float] | None = None
    camera_rotation_mode: str = "XYZ"
    camera_rotation_quat: tuple[float, float, float, float] | None = None
    camera_look_at: tuple[float, float, float] | None = None
    # Canonical viewpoints
    canonical_viewpoints: bool = True
    canonical_elevation_deg: float = 30.0
    canonical_distance_scale: float = 2.5
    # Orbit camera
    orbit_camera: bool = True
    orbit_frames: int = 120
    orbit_elevation_deg: float = 25.0
    orbit_distance_scale: float = 2.5
    # Batch rendering
    render: bool = False
    render_output: Path | None = None
    render_orbit: bool = False
    render_composite_resolution: int = 1024
    render_solo_resolution: int = 512


def list_versioned_files(base: Path) -> list[Path]:
    """Return [base, base_1, ...] for existing numbered files sorted by suffix."""
    parent = base.parent
    stem = base.stem
    suffix = base.suffix

    series: list[Path] = []
    if base.exists():
        series.append(base)

    pat = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
    extras = [p for p in parent.glob(f"{stem}_*{suffix}") if pat.match(p.name)]

    def _suffix_index(path: Path) -> int:
        match = pat.match(path.name)
        if match is None:
            return -1
        return int(match.group(1))

    extras.sort(key=_suffix_index)
    series.extend(extras)
    return series


def import_series(base: Path, importer: Callable[..., bpy.types.Object | None], /, **kwargs) -> list[bpy.types.Object]:
    objs: list[bpy.types.Object] = []
    for path in list_versioned_files(base):
        obj = importer(path, **kwargs)
        if obj is not None:
            objs.append(obj)
    return objs


def _read_ply_vertex_count(path: Path) -> int | None:
    """Read vertex count from a PLY header without loading full geometry."""
    try:
        with path.open("rb") as handle:
            for raw in handle:
                line = raw.decode("ascii", errors="ignore").strip()
                if not line:
                    continue
                lower = line.lower()
                if lower.startswith("element vertex "):
                    parts = lower.split()
                    if len(parts) >= 3:
                        return int(parts[2])
                if lower == "end_header":
                    break
    except Exception:
        return None
    return None


def _find_densest_path(paths: Sequence[Path]) -> tuple[Path, int | None]:
    """Return the densest PLY path (by header vertex count) and its count."""
    densest = paths[0]
    max_points: int | None = None
    for path in paths:
        count = _read_ply_vertex_count(path)
        if count is None:
            continue
        if max_points is None or count > max_points:
            max_points = count
            densest = path
    return densest, max_points


def _normalize_keep_ratio(ratio: float | None) -> float | None:
    if ratio is None:
        return None
    if not (0.0 < ratio < 1.0):
        return None
    # Treat near-1.0 as no-op to avoid pointless subsampling overhead.
    if ratio >= 0.999:
        return None
    return float(ratio)


def _compute_group_keep_ratio(
    paths: Sequence[Path],
    explicit_ratio: float | None,
    auto_tune_points: bool,
    auto_target_points: int | None,
) -> tuple[float | None, Path, int | None]:
    """Choose one keep ratio per source group from the densest cloud.

    Returns (ratio, densest_path, densest_count) so callers can reuse the
    densest-path lookup without a second scan.
    """
    densest_path, max_points = _find_densest_path(paths)
    ratio = _normalize_keep_ratio(explicit_ratio)

    if auto_tune_points and auto_target_points is not None and auto_target_points > 0 and paths:
        if max_points is not None and max_points > 0:
            target_ratio = min(1.0, float(auto_target_points) / float(max_points))
            target_ratio = _normalize_keep_ratio(target_ratio)
            if target_ratio is not None:
                ratio = target_ratio if ratio is None else min(ratio, target_ratio)

    return ratio, densest_path, max_points


def _object_prop_float(obj: bpy.types.Object, key: str) -> float | None:
    try:
        value = obj.get(key)
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _object_prop_str(obj: bpy.types.Object, key: str) -> str | None:
    try:
        value = obj.get(key)
        if value is None:
            return None
        return str(value)
    except Exception:
        return None


def import_pcd_series_shared(base: Path, **kwargs) -> list[bpy.types.Object]:
    """Import a PCD series with shared radius anchored to the densest cloud."""
    paths = list_versioned_files(base)
    if not paths:
        return []

    explicit_ratio = kwargs.get("subsample_ratio")
    auto_tune_points = bool(kwargs.get("auto_tune_points", False))
    auto_target_points = kwargs.get("auto_target_points")
    user_radius = kwargs.get("point_radius")

    anchor_ratio, densest_path, densest_count = _compute_group_keep_ratio(
        paths,
        explicit_ratio=explicit_ratio,
        auto_tune_points=auto_tune_points,
        auto_target_points=auto_target_points,
    )

    imported_by_path: dict[Path, bpy.types.Object] = {}

    # First import the densest cloud to anchor group-wide auto heuristics.
    anchor_kwargs = dict(kwargs)
    anchor_kwargs["subsample_ratio"] = anchor_ratio
    anchor_obj = import_pcd(densest_path, **anchor_kwargs)
    if anchor_obj is not None:
        imported_by_path[densest_path] = anchor_obj

    shared_radius = user_radius
    if shared_radius is None and anchor_obj is not None:
        anchor_radius = _object_prop_float(anchor_obj, "_pcd_effective_radius")
        if anchor_radius is not None:
            shared_radius = anchor_radius

    anchor_keep = _object_prop_float(anchor_obj, "_pcd_effective_keep_ratio") if anchor_obj is not None else None

    logger.debug(
        f"PCD group {base.stem}: densest={densest_path.name} n={densest_count if densest_count is not None else 'n/a'} "
        f"anchor_keep={anchor_keep if anchor_keep is not None else 1.0:.3f} "
        f"shared_radius={shared_radius if shared_radius is not None else 'auto'}"
    )

    # Import remaining clouds with anchored radius. Keep-ratio stays per-cloud.
    fixed_kwargs = dict(kwargs)
    fixed_kwargs["point_radius"] = shared_radius
    for path in paths:
        if path == densest_path:
            continue
        obj = import_pcd(path, **fixed_kwargs)
        if obj is not None:
            imported_by_path[path] = obj

    return [imported_by_path[path] for path in paths if path in imported_by_path]


def ensure_collection(name: str) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        scene = _require_scene()
        scene.collection.children.link(coll)
    return coll


def _move_object_to_collection(obj: bpy.types.Object, coll: bpy.types.Collection) -> None:
    if obj is None or coll is None:
        return
    if obj.name not in coll.objects:
        coll.objects.link(obj)
    for other in list(obj.users_collection):
        if other != coll:
            try:
                other.objects.unlink(obj)
            except Exception:
                pass


def group_objects_into_collection(objs: list[bpy.types.Object], coll_name: str) -> bpy.types.Collection | None:
    # Always group into a dedicated collection so visibility can be managed per-group
    coll = ensure_collection(coll_name)
    for obj in objs:
        _move_object_to_collection(obj, coll)
    return coll


def _set_collection_visibility(
    coll: bpy.types.Collection, visible: Literal["viewport", "render", "both"] | None
) -> None:
    """Set collection visibility consistently for viewport and render.

    - "viewport": visible in viewport only (hidden in renders)
    - "render": hidden in viewport, visible in renders
    - "both": visible in both
    - None: hidden in both

    Viewport visibility is stored on LayerCollection per View Layer, so we
    synchronize all active view layers. Render visibility uses collection.hide_render.
    """
    if coll is None:
        return

    if visible == "viewport":
        show_viewport, show_render = True, False
    elif visible == "render":
        show_viewport, show_render = False, True
    elif visible == "both":
        show_viewport, show_render = True, True
    elif visible is None:
        show_viewport, show_render = False, False
    else:
        raise ValueError(f"Unknown visibility mode: {visible}")

    # Render visibility
    if hasattr(coll, "hide_render"):
        try:
            coll.hide_render = not show_render
        except Exception:
            pass

    # View layer layer-collection visibility controls actual viewport drawing
    def _sync_layer_coll_visibility(layer_coll, target_coll, hide_flag: bool) -> bool:
        if getattr(layer_coll, "collection", None) is target_coll:
            if hasattr(layer_coll, "hide_viewport"):
                try:
                    layer_coll.hide_viewport = hide_flag
                except Exception:
                    pass
            return True
        for child in getattr(layer_coll, "children", []) or []:
            if _sync_layer_coll_visibility(child, target_coll, hide_flag):
                return True
        return False

    try:
        scene = _require_scene()
        for view_layer in getattr(scene, "view_layers", []) or []:
            _sync_layer_coll_visibility(view_layer.layer_collection, coll, not show_viewport)
    except Exception:
        pass


def _set_visibility(obj: bpy.types.Object, visible: Literal["viewport", "render", "both"] | None) -> None:
    if visible == "viewport":
        show_viewport, show_render = True, False
    elif visible == "render":
        show_viewport, show_render = False, True
    elif visible == "both":
        show_viewport, show_render = True, True
    elif visible is None:
        show_viewport, show_render = False, False
    else:
        raise ValueError(f"Unknown visibility mode: {visible}")

    obj.hide_set(not show_viewport)
    obj.hide_render = not show_render


def _create_import_placeholder(base_name: str, *, suffix: str, message: str | None = None) -> bpy.types.Object:
    name = f"{base_name}_{suffix}" if suffix else base_name
    placeholder = bpy.data.objects.new(name, None)
    try:
        placeholder.empty_display_type = "CUBE"
        placeholder.empty_display_size = 0.25
    except Exception:
        pass
    scene = _require_scene()
    scene.collection.objects.link(placeholder)
    if message is not None:
        try:
            placeholder["import_status"] = message
        except Exception:
            pass
    return placeholder


def _is_placeholder(obj: bpy.types.Object | None) -> bool:
    if obj is None or obj.data is not None:
        return False
    if getattr(obj, "type", None) != "EMPTY":
        return False
    try:
        return "import_status" in obj.keys()
    except Exception:
        return False


def np_mat44(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape == (3, 3):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = arr
        return mat
    assert arr.shape == (4, 4), f"Expected 4x4 or 3x3, got {arr.shape}"
    return arr


def set_scene_resolution(width: int, height: int):
    scene = _require_scene()
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0


def modify_default_scene_objects(remove: set | None = None):
    if remove is None:
        remove = {"Cube", "Light"}
    cam = bpy.data.objects.get("Camera")
    if cam is not None:
        cam.location = (2.0, -2.0, 1.8)
    for name in remove:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        # Make sure we're not in edit mode for that object
        try:
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
        # Unlink from all collections then remove
        for coll in list(obj.users_collection):
            try:
                coll.objects.unlink(obj)
            except Exception:
                pass
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass


def new_camera(name: str = "Camera") -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    scene = _require_scene()
    scene.collection.objects.link(cam_obj)
    return cam_obj


def set_camera_from_opencv_params(
    cam_obj: bpy.types.Object,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    width: int,
    height: int,
    sensor_width_mm: float = 36.0,
):
    """
    intrinsic: 3x3 (fx, 0, cx; 0, fy, cy; 0, 0, 1)
    extrinsic_w2c: 4x4 world->camera (OpenCV)
    width, height: image size in pixels
    """

    cam = cast(Any, cam_obj.data)
    cam.type = "PERSP"
    cam.lens_unit = "MILLIMETERS"
    cam.sensor_fit = "HORIZONTAL"  # we will match fx exactly; fy via sensor_height

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    # Compute Blender lens (in mm) from fx, and sensor_height to satisfy fy.
    # fx = f_mm * width_px / sensor_width_mm  => f_mm = fx * sensor_width_mm / width_px
    f_mm = fx * sensor_width_mm / float(width)
    sensor_height_mm = (f_mm * float(height)) / fy

    cam.lens = f_mm
    cam.sensor_width = sensor_width_mm
    cam.sensor_height = sensor_height_mm

    # Principal point -> shifts (relative to sensor size). Note the sign on shift_y.
    cam.shift_x = (cx - (width / 2.0)) / float(width)
    cam.shift_y = -(cy - (height / 2.0)) / float(height)

    # Pose: Convert OpenCV extrinsic (world->camera) to Blender camera matrix_world.
    # OpenCV camera coords: +X right, +Y down, +Z forward
    # Blender camera coords: +X right, +Y up, -Z forward
    # So X_cv = R_cv_bcam * X_bl_cam, with R_cv_bcam = diag([1, -1, -1]).
    R_cv_bcam = np.diag([1.0, -1.0, -1.0, 1.0])

    # Camera-to-world in OpenCV
    c2w_cv = np.linalg.inv(extrinsic_w2c)

    # Blender camera matrix_world
    M_world = c2w_cv @ R_cv_bcam
    cam_obj.matrix_world = mathutils.Matrix(M_world.tolist())


def import_mesh(mesh_path: Path, color: Sequence[float] | np.ndarray | None = None) -> bpy.types.Object | None:
    """
    Import a mesh file (.glb/.gltf or .obj) and return the primary imported object.
    Applies a Principled BSDF material (with optional base color) to all imported meshes.

    :param mesh_path: Path to the mesh file (.glb/.gltf or .obj).
    :param color: Optional RGB color [r,g,b] in 0..1 for the material's Base Color.
                  If None, the material will read from a mesh attribute named "Color".
    :return: The first imported mesh object.
    """
    # Deselect to reliably capture what this import selects
    try:
        bpy.ops.object.select_all(action="DESELECT")
    except Exception:
        pass

    ext = mesh_path.suffix.lower()

    # Import by format with version-safe fallbacks
    try:
        if ext in {".glb", ".gltf"}:
            bpy.ops.import_scene.gltf(filepath=str(mesh_path))
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=str(mesh_path))
        else:
            raise ValueError(f"Unsupported mesh format: {mesh_path.suffix}")
    except Exception as e:
        logger.error(f"Error importing mesh '{mesh_path}': {e}")
        return _create_import_placeholder(mesh_path.stem, suffix="import_error", message=str(e))

    # Collect imported mesh objects (there can be multiple)
    imported = list(bpy.context.selected_objects)
    mesh_objs = [o for o in imported if isinstance(getattr(o, "data", None), bpy.types.Mesh)]
    if not mesh_objs:
        logger.error(f"No mesh data found in '{mesh_path}'")
        return _create_import_placeholder(mesh_path.stem, suffix="no_mesh", message="no_mesh_objects_found")

    # Name the primary object after the file stem; keep others as-is
    primary = mesh_objs[0]
    primary.name = mesh_path.stem

    # Fix coordinate system: OBJ importer rotates meshes 90° around X.
    # GLTF already handles Y-up → Z-up per spec, so skip it there.
    if ext == ".obj":
        for obj in mesh_objs:
            obj.rotation_euler[0] -= math.radians(90)

    # Create a shared material
    mat = bpy.data.materials.new(name=f"{primary.name}_mat")
    mat_tree = _get_node_tree(mat)
    if mat_tree is None:
        logger.warning(f"Node tree unavailable for material '{mat.name}'; using default material setup")
        for obj in mesh_objs:
            try:
                cast(Any, obj.data).materials.append(mat)
            except Exception:
                pass
        return primary
    mat_nodes = mat_tree.nodes
    mat_links = mat_tree.links
    principled_bsdf_node = next(n for n in mat_nodes if "BsdfPrincipled" in n.bl_idname)
    _apply_principled_product_look(
        principled_bsdf_node,
        roughness=0.62,
        specular_level=0.32,
        coat_weight=0.06,
        coat_roughness=0.35,
    )

    if color is not None:
        # Explicit color override
        principled_bsdf_node.inputs["Base Color"].default_value = [*color, 1]
    else:
        # Use attribute-driven color, mirroring import_pcd's behavior but with "Color"
        # Determine a suitable attribute name with small fallbacks for robustness
        attr_name = "Color"
        try:
            data = cast(Any, primary.data)
            names = []
            if hasattr(data, "color_attributes"):
                names = [a.name for a in data.color_attributes]
            elif hasattr(data, "vertex_colors"):
                names = [a.name for a in data.vertex_colors]
            # Prefer "Color", else typical alternatives
            if "Color" in names:
                attr_name = "Color"
            elif "Col" in names:
                attr_name = "Col"
            elif "COLOR_0" in names:
                attr_name = "COLOR_0"
        except Exception:
            pass

        attribute_node = mat_nodes.new(type="ShaderNodeAttribute")
        attribute_node.attribute_name = attr_name  # "Color" preferred
        # Connect attribute color → Principled Base Color
        if "Color" in attribute_node.outputs and "Base Color" in principled_bsdf_node.inputs:
            mat_links.new(attribute_node.outputs["Color"], principled_bsdf_node.inputs["Base Color"])

    # Assign material to all imported mesh objects
    for obj in mesh_objs:
        try:
            cast(Any, obj.data).materials.append(mat)
        except Exception:
            pass

    return primary


def add_geometry_nodes(obj: bpy.types.Object) -> bpy.types.GeometryNodeTree:
    with bpy.context.temp_override(object=obj):
        bpy.ops.node.new_geometry_nodes_modifier()
    modifier = obj.modifiers[-1]
    return cast(Any, modifier).node_group


def import_pcd(
    obj_path: Path,
    point_radius: float | None,
    color: Sequence[float] | np.ndarray | None = None,
    roughness: float = 0.5,
    specular_level: float = 0.18,
    alpha: float = 1.0,
    subsample_ratio: float | None = None,
    auto_tune_points: bool = False,
    auto_target_points: int | None = None,
    auto_radius_scale: float = 0.15,
    auto_radius_min: float = 5e-4,
    auto_radius_max: float = 0.02,
    auto_use_nn: bool = True,
    auto_nn_sample_size: int = 5000,
    auto_nn_subsample_if_small: bool = True,
) -> bpy.types.Object:
    """
    Import a point cloud from a PLY file and render as spheres using Geometry Nodes.

    :param obj_path: Path to the PLY file
    :param point_radius: Radius of each rendered point sphere. When None and auto tuning is enabled,
                        the radius is derived from bounding size and density.
    :param color: Optional RGB color override. If None, uses 'Col' attribute from PLY
    :param roughness: Material roughness value
    :param specular_level: Principled specular level (or Specular IOR Level in Blender 4.x)
    :param alpha: Material alpha/transparency value
    :param subsample_ratio: Optional fraction (0-1) of points to randomly keep (e.g., 0.5 = 50%)
    :param auto_tune_points: Enable automatic point-budget + radius heuristics when True
    :param auto_target_points: Target upper bound for visible points per cloud when auto tuning
    :param auto_radius_scale: Scale factor applied to the estimated spacing for automatic radius
    :param auto_radius_min: Minimum radius allowed for automatic sizing
    :param auto_radius_max: Maximum radius allowed for automatic sizing
    :return: The imported point cloud object
    """
    # Load point cloud data
    try:
        bpy.ops.object.select_all(action="DESELECT")
    except Exception:
        pass
    try:
        bpy.ops.wm.ply_import(filepath=str(obj_path))
    except Exception as e:
        logger.error(f"Error importing point cloud '{obj_path}': {e}")
        return _create_import_placeholder(obj_path.stem, suffix="import_error", message=str(e))

    imported = list(bpy.context.selected_objects)
    if not imported:
        logger.error(f"No objects selected after importing '{obj_path}'")
        return _create_import_placeholder(obj_path.stem, suffix="no_selection", message="no_objects_selected")

    obj = imported[0]
    if not isinstance(getattr(obj, "data", None), bpy.types.Mesh):
        logger.error(f"Imported object for '{obj_path}' is not a mesh")
        return _create_import_placeholder(obj_path.stem, suffix="not_mesh", message="imported_object_not_mesh")
    obj.name = obj_path.stem

    mesh = cast(Any, obj.data)
    initial_points = len(mesh.vertices)

    final_ratio = None
    had_cap = False
    if subsample_ratio is not None and 0 < subsample_ratio < 1.0:
        final_ratio = subsample_ratio

    if auto_tune_points and auto_target_points and auto_target_points > 0 and initial_points > 0:
        target_ratio = min(1.0, float(auto_target_points) / float(initial_points))
        if target_ratio < 0.999:
            if final_ratio is None:
                final_ratio = target_ratio
                had_cap = True
            else:
                had_cap = had_cap or (target_ratio < final_ratio)
                final_ratio = min(final_ratio, target_ratio)

    destructive_ratio, runtime_ratio = _split_subsample_ratio(final_ratio)

    if destructive_ratio is not None:
        rng = Random(42)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        verts_to_remove = [v for v in bm.verts if rng.random() > destructive_ratio]
        if verts_to_remove:
            bmesh.ops.delete(bm, geom=verts_to_remove, context="VERTS")
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    active_points = len(mesh.vertices)
    if final_ratio is not None and initial_points > 0:
        actual_ratio = active_points / float(initial_points)
        logger.debug(
            f"Subsample {obj.name}: {initial_points:.0f} -> {active_points:.0f} pts "
            f"(target {final_ratio:.3f}, actual {actual_ratio:.3f})"
        )

    effective_radius = point_radius
    auto_radius_enabled = auto_tune_points and point_radius is None
    # For logging
    avg_nn_dbg: float | None = None
    raw_r_dbg: float | None = None
    nn_keep_dbg: float | None = None
    nn_floor_dbg: float | None = None
    sources: list[str] = []

    if auto_radius_enabled:
        if auto_use_nn:
            eff_r, suggested_keep, avg_nn_dbg, raw_r_dbg = _estimate_auto_point_radius_nn(
                obj,
                scale=auto_radius_scale,
                radius_min=auto_radius_min,
                radius_max=auto_radius_max,
                sample_size=auto_nn_sample_size,
            )
            effective_radius = eff_r
            if auto_nn_subsample_if_small and suggested_keep is not None:
                runtime_ratio = suggested_keep if runtime_ratio is None else min(runtime_ratio, suggested_keep)
                nn_keep_dbg = suggested_keep
                sources.append("nn")
            logger.debug(f"Auto NN radius {obj.name}: {float(effective_radius):.6f}")

            # Density-aware floor: if an auto point cap was applied but the cloud is sparse,
            # do not reduce below the density-implied keep. This avoids over-thinning.
            if had_cap and avg_nn_dbg is not None and auto_radius_scale > 0:
                target_spacing = float(auto_radius_min) / float(max(auto_radius_scale, 1e-12))
                density_floor = _clamp_float((avg_nn_dbg / max(target_spacing, 1e-12)) ** 3, 0.0, 1.0)
                if runtime_ratio is not None and density_floor > runtime_ratio:
                    runtime_ratio = density_floor
                    nn_floor_dbg = density_floor
                    sources.append("nn_floor")
        else:
            effective_radius = _estimate_auto_point_radius(
                obj,
                point_count=max(active_points, 1),
                scale=auto_radius_scale,
                radius_min=auto_radius_min,
                radius_max=auto_radius_max,
            )
            logger.debug(f"Auto radius {obj.name}: {float(effective_radius):.6f}")

    if effective_radius is None:
        effective_radius = 0.003

    # Setup Geometry Nodes to render points as spheres
    node_group = add_geometry_nodes(obj)
    links = node_group.links
    nodes = node_group.nodes

    # Get input/output nodes
    group_input_node = next(n for n in nodes if "NodeGroupInput" in n.bl_idname)
    group_output_node = next(n for n in nodes if "NodeGroupOutput" in n.bl_idname)

    # Create mesh-to-points node
    mesh_to_points_node = cast(Any, nodes.new(type="GeometryNodeMeshToPoints"))
    mesh_to_points_node.inputs["Radius"].default_value = float(effective_radius)
    upstream = group_input_node.outputs["Geometry"]

    # Random subsampling via geometry nodes fields (runtime_ratio = keep probability)
    if runtime_ratio is not None and runtime_ratio < 1.0:
        delete_node = cast(Any, nodes.new(type="GeometryNodeDeleteGeometry"))
        delete_node.domain = "POINT"
        rand_node = cast(Any, nodes.new(type="FunctionNodeRandomValue"))
        rand_node.data_type = "FLOAT"
        rand_node.inputs["Min"].default_value = 0.0
        rand_node.inputs["Max"].default_value = 1.0
        compare_node = cast(Any, nodes.new(type="FunctionNodeCompare"))
        compare_node.data_type = "FLOAT"
        # Delete when random > keep_ratio (i.e., delete probability = 1 - keep)
        compare_node.operation = "GREATER_THAN"
        compare_node.label = "PCD_KEEP_COMPARE"
        compare_node.name = "PCD_KEEP_COMPARE"
        compare_node.inputs[1].default_value = runtime_ratio  # B input is keep_ratio

        links.new(group_input_node.outputs["Geometry"], delete_node.inputs["Geometry"])
        links.new(rand_node.outputs["Value"], compare_node.inputs[0])
        links.new(compare_node.outputs["Result"], delete_node.inputs["Selection"])
        upstream = delete_node.outputs["Geometry"]

    links.new(upstream, mesh_to_points_node.inputs["Mesh"])

    # Material setup
    mat = bpy.data.materials.new(name=f"{obj.name}_mat")
    mat_tree = _get_node_tree(mat)
    if mat_tree is None:
        raise RuntimeError(f"Node tree unavailable for point-cloud material '{mat.name}'")
    mat_nodes = mat_tree.nodes
    mat_links = mat_tree.links

    principled_bsdf_node = next(n for n in mat_nodes if "BsdfPrincipled" in n.bl_idname)
    pcd_roughness = max(float(roughness), 0.72)
    _apply_principled_product_look(
        principled_bsdf_node,
        roughness=pcd_roughness,
        specular_level=float(specular_level),
        coat_weight=0.0,
        coat_roughness=0.25,
        alpha=float(alpha),
    )
    if color is None:
        attribute_node = mat_nodes.new(type="ShaderNodeAttribute")
        attribute_node.attribute_name = "Col"
        mat_links.new(attribute_node.outputs["Color"], principled_bsdf_node.inputs["Base Color"])
    else:
        principled_bsdf_node.inputs["Base Color"].default_value = [*color, 1]

    set_material_node = cast(Any, nodes.new(type="GeometryNodeSetMaterial"))
    set_material_node.inputs["Material"].default_value = mat

    cast(Any, obj.data).materials.append(mat)

    # Final wiring: Points → Set Material → Output
    links.new(mesh_to_points_node.outputs["Points"], set_material_node.inputs["Geometry"])
    links.new(set_material_node.outputs["Geometry"], group_output_node.inputs["Geometry"])

    # Compose per-cloud debug log
    try:
        if subsample_ratio is not None and 0.0 < subsample_ratio < 1.0:
            sources.append("explicit")
        if (
            auto_tune_points
            and auto_target_points
            and auto_target_points > 0
            and initial_points > 0
            and (float(auto_target_points) / float(initial_points)) < 0.999
        ):
            if "cap" not in sources:
                sources.append("cap")
        destr_str = f"{destructive_ratio:.3f}" if destructive_ratio is not None else "1.000"
        run_str = f"{runtime_ratio:.3f}" if runtime_ratio is not None else "1.000"
        nn_part = (
            f" | NN avg={avg_nn_dbg:.6f} raw={raw_r_dbg:.6f}"
            if (avg_nn_dbg is not None and raw_r_dbg is not None)
            else ""
        )
        nn_keep_part = f" nn_keep={nn_keep_dbg:.3f}" if nn_keep_dbg is not None else ""
        nn_floor_part = f" nn_floor={nn_floor_dbg:.3f}" if nn_floor_dbg is not None else ""
        src_part = f" [{','.join(sources)}]" if sources else ""
        logger.debug(
            f"PCD {obj.name}: pts {initial_points}->{len(cast(Any, obj.data).vertices)} | "
            f"r={float(effective_radius):.6f} | keep destr={destr_str} runtime={run_str}"
            f"{nn_part}{nn_keep_part}{nn_floor_part}{src_part}"
        )
    except Exception:
        pass

    # Persist effective settings for group-level anchoring across series imports.
    try:
        destr_keep = destructive_ratio if destructive_ratio is not None else 1.0
        runtime_keep = runtime_ratio if runtime_ratio is not None else 1.0
        effective_keep = _clamp_float(float(destr_keep) * float(runtime_keep), 0.0, 1.0)
        obj["_pcd_effective_radius"] = float(effective_radius)
        obj["_pcd_runtime_keep_ratio"] = float(runtime_keep)
        obj["_pcd_effective_keep_ratio"] = float(effective_keep)
    except Exception:
        pass

    return obj


def _vertex_positions_world(obj: bpy.types.Object) -> list[mathutils.Vector]:
    try:
        mw = obj.matrix_world
        verts = getattr(obj.data, "vertices", [])
        return [mw @ v.co for v in verts]
    except Exception:
        return []


def _estimate_auto_point_radius_nn(
    obj: bpy.types.Object, scale: float, radius_min: float, radius_max: float, sample_size: int = 5000
) -> tuple[float, float | None, float | None, float | None]:
    """
    Estimate point radius from the average nearest-neighbor distance measured on a sample
    of points (in world space). Returns (effective_radius, suggested_keep_ratio, avg_nn, raw_radius)
    where suggested_keep_ratio is an optional additional subsampling ratio to increase spacing
    when the raw estimate falls below radius_min. avg_nn and raw_radius may be None on failure.
    """
    pts = _vertex_positions_world(obj)
    n = len(pts)
    if n <= 1:
        return _clamp_float(0.003, radius_min, radius_max), None, None, None

    # Deterministic stride sampling to avoid heavy memory/use of random state
    if n > sample_size > 0:
        step = max(n // sample_size, 1)
        pts_s = pts[0:n:step][:sample_size]
    else:
        pts_s = pts

    m = len(pts_s)
    if m <= 1:
        return _clamp_float(0.003, radius_min, radius_max), None, None, None

    try:
        tree = kdtree.KDTree(m)
        for i, p in enumerate(pts_s):
            tree.insert((float(p.x), float(p.y), float(p.z)), i)
        tree.balance()

        dists = []
        for p in pts_s:
            hits = tree.find_n((float(p.x), float(p.y), float(p.z)), 2)
            if not hits or len(hits) < 2:
                continue
            # hits: list of (co, index, dist); second entry is nearest neighbor (first is itself)
            dists.append(hits[1][2])

        if not dists:
            return _clamp_float(0.003, radius_min, radius_max), None, None, None

        avg_nn = float(sum(dists) / len(dists))
    except Exception:
        # Fallback: approximate via bounds if KDTree fails
        eff = _estimate_auto_point_radius(
            obj,
            point_count=max(len(getattr(obj.data, "vertices", [])), 1),
            scale=scale,
            radius_min=radius_min,
            radius_max=radius_max,
        )
        return eff, None, None, None

    raw_radius = avg_nn * float(scale)
    eff_radius = _clamp_float(raw_radius, float(radius_min), float(radius_max))

    # If radius is clamped to min due to dense sampling, suggest an additional keep ratio
    keep_ratio: float | None = None
    if raw_radius < float(radius_min):
        target_spacing = float(radius_min) / max(float(scale), 1e-12)
        # spacing ∝ n^{-1/d} ⇒ keep ≈ (d_current / d_target)^d
        factor = (avg_nn / max(target_spacing, 1e-12)) ** 3
        keep_ratio = float(_clamp_float(factor, 0.0, 1.0))
        if keep_ratio >= 0.999:
            keep_ratio = None

    return eff_radius, keep_ratio, avg_nn, raw_radius


def set_camera_background_images(
    cam_obj: bpy.types.Object,
    rgb_path: Path | None = None,
    normals_path: Path | None = None,
    depth_path: Path | None = None,
):
    cam = cast(Any, cam_obj.data)
    cam.show_background_images = True

    def add_bg(img_path: Path | None, alpha: float, display_depth: str):
        if not img_path or not img_path.is_file():
            return
        img = bpy.data.images.load(str(img_path), check_existing=True)
        bg = cam.background_images.new()
        bg.image = img
        bg.alpha = alpha
        bg.display_depth = display_depth  # 'FRONT' or 'BACK'
        bg.frame_method = "FIT"

    # RGB front, others behind (semi-transparent)
    add_bg(rgb_path, alpha=1.0, display_depth="FRONT")
    if (rgb_path and rgb_path.is_file()) or (depth_path and depth_path.is_file()):
        add_bg(normals_path, alpha=0.5, display_depth="BACK")
    else:
        add_bg(normals_path, alpha=1.0, display_depth="FRONT")
    if (rgb_path and rgb_path.is_file()) or (normals_path and normals_path.is_file()):
        add_bg(depth_path, alpha=0.5, display_depth="BACK")
    else:
        logger.debug("Showing depth image in front layer")
        add_bg(depth_path, alpha=1.0, display_depth="FRONT")


def _compute_plane_params_from_intrinsics(intrinsic: np.ndarray, width_px: int, height_px: int, depth_m: float):
    """
    Returns (plane_width_m, plane_height_m, offset_x_m, offset_y_m)
    so that the plane placed at camera local (x=offx, y=offy, z=-depth_m),
    with its surface on the local XY plane, exactly spans the camera view.
    """
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    plane_width = depth_m * (width_px / fx)
    plane_height = depth_m * (height_px / fy)
    # Principal point offset: note Y sign (OpenCV down -> Blender up)
    off_x = depth_m * ((cx - (width_px / 2.0)) / fx)
    off_y = -depth_m * ((cy - (height_px / 2.0)) / fy)
    return plane_width, plane_height, off_x, off_y


def _make_plane_object(name: str, width_m: float, height_m: float) -> bpy.types.Object:
    """
    Create a mesh plane of exact size (width_m x height_m) centered at origin,
    oriented on the local XY plane (normal +Z), with basic UVs.
    Uses data API (no operators) so it works in background mode.
    """
    mesh = bpy.data.meshes.new(name + "_mesh")
    hw, hh = width_m / 2.0, height_m / 2.0
    verts = [(-hw, -hh, 0.0), (hw, -hh, 0.0), (hw, hh, 0.0), (-hw, hh, 0.0)]
    faces = [(0, 1, 2, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # UVs for the single quad (4 loops)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_loops = uv_layer.data
    # The single face produces the first 4 loops
    uv_loops[0].uv = (0.0, 0.0)
    uv_loops[1].uv = (1.0, 0.0)
    uv_loops[2].uv = (1.0, 1.0)
    uv_loops[3].uv = (0.0, 1.0)

    obj = bpy.data.objects.new(name, mesh)
    _require_scene().collection.objects.link(obj)
    return obj


def _make_emission_image_material(
    name: str, image_path: Path, alpha: float = 1.0, non_color: bool = False
) -> bpy.types.Material:
    """
    Emission material that shows the image; visible to camera only, transparent to all
    other rays so it doesn't light the scene or cast shadows. Alpha controls camera opacity.
    """
    mat = bpy.data.materials.new(name)
    nt = _get_node_tree(mat)
    if nt is None:
        raise RuntimeError(f"Node tree unavailable for emission material '{name}'")
    nodes = nt.nodes
    links = nt.links

    # Clear default nodes
    for n in list(nodes):
        nodes.remove(n)

    # Nodes
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (800, 0)
    mix_alpha = nodes.new("ShaderNodeMixShader")
    mix_alpha.location = (600, 0)
    mix_alpha.inputs["Fac"].default_value = float(alpha)

    mix_camera = nodes.new("ShaderNodeMixShader")
    mix_camera.location = (400, 0)
    emis = nodes.new("ShaderNodeEmission")
    emis.location = (200, 0)
    emis.inputs["Strength"].default_value = 1.0
    trans = nodes.new("ShaderNodeBsdfTransparent")
    trans.location = (200, -180)
    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (0, 0)
    lp = nodes.new("ShaderNodeLightPath")
    lp.location = (0, 200)

    # Image setup
    img = bpy.data.images.load(str(image_path), check_existing=True)
    if non_color and hasattr(img, "colorspace_settings"):
        cast(Any, img.colorspace_settings).name = "Non-Color"
    tex.image = img

    # Wire: Emission color from image
    links.new(tex.outputs["Color"], emis.inputs["Color"])

    # Wire: Only show emission for camera rays; transparent for all other rays
    # mix_camera: Fac = Is Camera Ray; Shader[1]=Transparent (non-camera), Shader[2]=Emission (camera)
    links.new(lp.outputs["Is Camera Ray"], mix_camera.inputs["Fac"])
    links.new(trans.outputs["BSDF"], mix_camera.inputs[1])
    links.new(emis.outputs["Emission"], mix_camera.inputs[2])

    # Wire: Apply constant alpha for camera (and keep fully transparent otherwise)
    links.new(trans.outputs["BSDF"], mix_alpha.inputs[1])  # base transparent
    links.new(mix_camera.outputs["Shader"], mix_alpha.inputs[2])  # camera-only emission over it
    links.new(mix_alpha.outputs["Shader"], out.inputs["Surface"])

    # Proper blending in Eevee; harmless in Cycles (version-safe toggles)
    if hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False
    # Disable shadowing if supported by this Blender version (Eevee-friendly)
    if hasattr(mat, "shadow_method"):
        cast(Any, mat).shadow_method = "NONE"
    elif hasattr(mat, "shadow_mode"):
        cast(Any, mat).shadow_mode = "NONE"

    # In Cycles, prevent sampling emissive material as a light source
    try:
        if hasattr(mat, "cycles") and hasattr(mat.cycles, "sample_as_light"):
            cast(Any, mat.cycles).sample_as_light = False
    except Exception:
        pass

    return mat


def add_camera_image_planes(
    cam_obj: bpy.types.Object,
    intrinsic: np.ndarray,
    width_px: int,
    height_px: int,
    rgb_path: Path | None = None,
    normals_path: Path | None = None,
    depth_path: Path | None = None,
    depth_m: float = 1.0,
    add_camera_body: bool = True,
    frustum_thickness_ratio: float = 0.005,
) -> list[bpy.types.Object]:
    """
    Creates one plane per available image (normals, depth, rgb), parented to cam_obj,
    positioned at z = -depth_m in camera local space, sized/offset from intrinsics so it
    matches the viewport. Also adds a small sphere as the camera body if requested and a
    wireframe frustum with adjustable thickness (via frustum_thickness_ratio * depth_m).
    Returns the created objects.
    """
    created = []
    attachments_coll = ensure_collection(f"{cam_obj.name}_attachments")

    plane_w, plane_h, off_x, off_y = _compute_plane_params_from_intrinsics(intrinsic, width_px, height_px, depth_m)

    # Small spacing to avoid z-fighting (farthest first)
    layers = [
        ("normals", normals_path, 0.5, 0.002, True),  # Non-Color for normals
        ("depth", depth_path, 0.5, 0.001, True),  # Non-Color for depth
        ("rgb", rgb_path, 1.0, 0.000, False),
    ]

    for name, img_path, alpha, z_eps, non_color in layers:
        if not img_path or not Path(img_path).is_file():
            continue
        plane = _make_plane_object(f"{cam_obj.name}_plane_{name}", plane_w, plane_h)

        # Parent to camera and use camera-local transforms (matrix_parent_inverse = Identity)
        plane.parent = cam_obj
        plane.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        plane.location = (off_x, off_y, -(depth_m + z_eps))
        plane.rotation_euler = (0.0, 0.0, 0.0)
        plane.scale = (1.0, 1.0, 1.0)
        _move_object_to_collection(plane, attachments_coll)

        # Material: camera-visible only, no lighting/shadows
        mat = _make_emission_image_material(f"{plane.name}_mat", Path(img_path), alpha=alpha, non_color=non_color)
        cast(Any, plane.data).materials.clear()
        cast(Any, plane.data).materials.append(mat)

        # Camera-only visibility: visible to camera rays, invisible to all others.
        # Blender 4.0+ uses obj.visible_* (replaced obj.cycles_visibility in 3.0).
        if hasattr(plane, "visible_camera"):
            plane.visible_camera = True
            for attr in (
                "visible_diffuse",
                "visible_glossy",
                "visible_transmission",
                "visible_volume_scatter",
                "visible_shadow",
            ):
                if hasattr(plane, attr):
                    setattr(plane, attr, False)
        elif hasattr(plane, "cycles_visibility"):
            try:
                vis = cast(Any, plane).cycles_visibility
                if hasattr(vis, "camera"):
                    vis.camera = True
                for attr in ("diffuse", "glossy", "transmission", "scatter", "volume_scatter", "shadow"):
                    if hasattr(vis, attr):
                        setattr(vis, attr, False)
            except Exception:
                pass

        created.append(plane)

    if add_camera_body:
        # Sizes anchored to the image plane distance
        size = 0.05 * depth_m  # baseline size used for sphere placement offset
        half = size / 2.0

        # -------------------------------
        # Camera body: small UV sphere
        # -------------------------------
        sphere_radius = 0.5 * size

        sphere_mesh = bpy.data.meshes.new(f"{cam_obj.name}_body_sphere_mesh")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=sphere_radius)
        bm.to_mesh(sphere_mesh)
        bm.free()

        body = bpy.data.objects.new(f"{cam_obj.name}_body_sphere", sphere_mesh)
        _require_scene().collection.objects.link(body)

        # Parent with camera-local offset (behind the pinhole; +Z in Blender cam space)
        body.parent = cam_obj
        body.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        body.location = (0.0, 0.0, +half * 1.2)
        body.rotation_euler = (0.0, 0.0, 0.0)
        body.scale = (1.0, 1.0, 1.0)
        _move_object_to_collection(body, attachments_coll)

        created.append(body)

        # -------------------------------
        # Frustum: pyramid → wireframe
        # -------------------------------
        # Base plane (rectangle) at z = -depth_m; apex at camera origin (0,0,0)
        hw = plane_w * 0.5
        hh = plane_h * 0.5

        apex = (0.0, 0.0, 0.0)
        c1 = (off_x - hw, off_y - hh, -depth_m)
        c2 = (off_x + hw, off_y - hh, -depth_m)
        c3 = (off_x + hw, off_y + hh, -depth_m)
        c4 = (off_x - hw, off_y + hh, -depth_m)

        frustum_verts = [apex, c1, c2, c3, c4]
        # Faces: 4 sides only (no base to avoid cross-beam in wireframe)
        frustum_faces = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 4),
            (0, 4, 1),
        ]
        # Explicit edges for the rectangular base outline
        frustum_edges = [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
        ]

        frustum_mesh = bpy.data.meshes.new(f"{cam_obj.name}_frustum_mesh")
        frustum_mesh.from_pydata(frustum_verts, frustum_edges, frustum_faces)
        frustum_mesh.update()

        frustum_obj = bpy.data.objects.new(f"{cam_obj.name}_frustum", frustum_mesh)
        _require_scene().collection.objects.link(frustum_obj)

        # Parent to camera, no transform baking
        frustum_obj.parent = cam_obj
        frustum_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        frustum_obj.location = (0.0, 0.0, 0.0)
        frustum_obj.rotation_euler = (0.0, 0.0, 0.0)
        frustum_obj.scale = (1.0, 1.0, 1.0)
        _move_object_to_collection(frustum_obj, attachments_coll)

        # Wireframe modifier for visible thickness
        wf = cast(Any, frustum_obj.modifiers.new(name="FrustumWire", type="WIREFRAME"))
        wf.thickness = float(frustum_thickness_ratio) * float(depth_m)
        wf.use_even_offset = True
        wf.use_relative_offset = False
        wf.use_replace = True  # show only the wireframe result

        mat = bpy.data.materials.new(name=f"{frustum_obj.name}_mat")
        mat_tree = _get_node_tree(mat)
        if mat_tree is not None:
            mat_nodes = mat_tree.nodes

            principled_bsdf_node = next(n for n in mat_nodes if "BsdfPrincipled" in n.bl_idname)
            principled_bsdf_node.inputs["Base Color"].default_value = [0, 0, 0, 1]
            principled_bsdf_node.inputs["Roughness"].default_value = 0.9

            cast(Any, frustum_obj.data).materials.append(mat)
            cast(Any, body.data).materials.append(mat)

        # Optional: avoid casting/receiving shadows (Eevee-friendly)
        if hasattr(frustum_obj, "visible_shadow"):
            frustum_obj.visible_shadow = False

        created.append(frustum_obj)

    return created


def set_world_background(color: Sequence[float] | None = None, strength: float | None = None):
    """Sets the color of blenders world background

    :param color: A three-dimensional list specifying the new color in floats.
    :param strength: The strength of the emitted background light.
    """
    world = cast(Any, _require_scene().world)
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    bg_node = nodes["Background"]

    # Unlink any incoming link that would overwrite the default value
    if bg_node.inputs["Color"].links:
        links.remove(bg_node.inputs["Color"].links[0])

    if strength is not None:
        bg_node.inputs["Strength"].default_value = strength
    if color is not None:
        bg_node.inputs["Color"].default_value = [*color, 1.0]


def set_world_background_hdr_img(path_to_hdr_file: str, strength: float = 1.0):
    world = cast(Any, _require_scene().world)
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # add a texture node and load the image and link it
    texture_node = nodes.new(type="ShaderNodeTexEnvironment")
    texture_node.image = bpy.data.images.load(path_to_hdr_file, check_existing=True)

    background_node = nodes["Background"]

    # link the new texture node to the background
    links.new(texture_node.outputs["Color"], background_node.inputs["Color"])

    # Set the brightness of the background
    background_node.inputs["Strength"].default_value = strength

    # add a mapping node and a texture coordinate node
    mapping_node = nodes.new("ShaderNodeMapping")
    tex_coords_node = nodes.new("ShaderNodeTexCoord")

    # link the texture coordinate node to mapping node
    links.new(tex_coords_node.outputs["Generated"], mapping_node.inputs["Vector"])

    # link the mapping node to the texture node
    links.new(mapping_node.outputs["Vector"], texture_node.inputs["Vector"])

    # mapping_node.inputs["Rotation"].default_value = rotation_euler


def find_stem(dir_path: Path, stem: str | None) -> str:
    if stem:
        return stem
    matches = sorted(dir_path.glob("*_camera.json"))
    if not matches:
        raise FileNotFoundError(f"No '*_camera.json' found in {dir_path}")
    return matches[0].name.replace("_camera.json", "")


def _ensure_cycles(
    samples: int = 384,
    denoise: bool = True,
    clamp_direct: float = 0.0,
    clamp_indirect: float = 2.5,
    preview_factor: float = 0.25,
    adaptive_threshold: float = 0.03,
    light_threshold: float = 0.01,
    blur_glossy: float = 0.0,
    max_bounces: int = 4,
    diffuse_bounces: int = 2,
    glossy_bounces: int = 2,
    transmission_bounces: int = 1,
    transparent_max_bounces: int = 4,
    use_caustics_reflective: bool = False,
    use_caustics_refractive: bool = False,
):
    scene = _require_scene()
    cast(Any, scene.render).engine = "CYCLES"

    # Keep user-configured Cycles device selection by default.
    # Explicit probing is opt-in because get_devices() can emit noisy HIP init
    # warnings on non-AMD setups.
    auto_probe = os.getenv("BLENDER_LOAD_AUTO_DEVICE_PROBE", "").strip().lower() in {"1", "true", "yes", "on"}
    if auto_probe:
        try:
            prefs = cast(Any, _context_any().preferences.addons["cycles"].preferences)
            device_types: list[str] = ["OPTIX", "CUDA"]
            try:
                has_hip_runtime = bool(ctypes.util.find_library("amdhip64") or ctypes.util.find_library("hiprtc"))
            except Exception:
                has_hip_runtime = False
            force_hip = os.getenv("BLENDER_LOAD_FORCE_HIP", "").strip().lower() in {"1", "true", "yes", "on"}
            if has_hip_runtime or force_hip:
                device_types.append("HIP")
            if sys.platform == "darwin":
                device_types.append("METAL")

            gpu_enabled = False
            for device_type in device_types:
                try:
                    prefs.compute_device_type = device_type
                    prefs.get_devices()
                    devices = prefs.devices
                    if not devices:
                        continue

                    non_cpu = [d for d in devices if getattr(d, "type", "") != "CPU"]
                    if not non_cpu:
                        continue

                    for d in devices:
                        d.use = d in non_cpu
                    cast(Any, scene.cycles).device = "GPU"
                    gpu_enabled = True
                    logger.debug(f"Cycles GPU enabled: {device_type} ({len(non_cpu)} device(s))")
                    break
                except Exception:
                    continue
            if not gpu_enabled:
                cast(Any, scene.cycles).device = "CPU"
                logger.warning("Cycles GPU backend not available; falling back to CPU")
        except Exception:
            pass

    cycles = cast(Any, scene.cycles)
    cycles.samples = int(samples)
    preview_samples = max(32, int(samples * preview_factor))
    cycles.preview_samples = preview_samples
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = float(adaptive_threshold)
    if hasattr(cycles, "light_sampling_threshold"):
        cycles.light_sampling_threshold = float(light_threshold)
    elif hasattr(cycles, "light_threshold"):
        cycles.light_threshold = float(light_threshold)
    cycles.blur_glossy = float(blur_glossy)
    if hasattr(cycles, "use_caustics_reflective"):
        cycles.use_caustics_reflective = bool(use_caustics_reflective)
    if hasattr(cycles, "use_caustics_refractive"):
        cycles.use_caustics_refractive = bool(use_caustics_refractive)
    cycles.max_bounces = int(max_bounces)
    cycles.diffuse_bounces = int(diffuse_bounces)
    cycles.glossy_bounces = int(glossy_bounces)
    cycles.transmission_bounces = int(transmission_bounces)
    cycles.transparent_max_bounces = int(transparent_max_bounces)
    cycles.sample_clamp_direct = float(clamp_direct)
    cycles.sample_clamp_indirect = float(clamp_indirect)

    # Denoise: prefer OptiX, fall back to OpenImageDenoise
    try:
        view = cast(Any, scene.view_layers[0])
        cast(Any, view.cycles).use_denoising = denoise
        if denoise:
            for denoiser in ("OPTIX", "OPENIMAGEDENOISE"):
                try:
                    cycles.denoiser = denoiser
                    cycles.preview_denoiser = denoiser
                    break
                except Exception:
                    continue
    except Exception:
        pass

    # World MIS helps HDRIs
    world = cast(Any, scene.world)
    if world and hasattr(world, "cycles"):
        if hasattr(world.cycles, "sample_as_light"):
            world.cycles.sample_as_light = True
        elif hasattr(world.cycles, "sampling_method"):
            world.cycles.sampling_method = "AUTOMATIC"


def _object_bounds_world(obj: bpy.types.Object):
    # Returns (center, radius) in world space
    mw = obj.matrix_world
    bb = [mw @ mathutils.Vector(corner) for corner in obj.bound_box]
    minv = mathutils.Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
    maxv = mathutils.Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
    center = (minv + maxv) * 0.5
    radius = (maxv - center).length
    return center, max(radius, 1e-3)


class ObjectBounds(NamedTuple):
    obj: bpy.types.Object
    min: mathutils.Vector
    max: mathutils.Vector
    center: mathutils.Vector
    radius: float


def _collect_object_bounds(
    objects: Iterable[bpy.types.Object],
) -> list[ObjectBounds]:
    """Return per-object bounds in world coordinates."""
    stats: list[ObjectBounds] = []
    for obj in objects:
        if _is_placeholder(obj):
            continue
        mw = obj.matrix_world
        bb = [mw @ mathutils.Vector(corner) for corner in obj.bound_box]
        minv = mathutils.Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
        maxv = mathutils.Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
        center = (minv + maxv) * 0.5
        radius = max((maxv - center).length, 1e-3)
        stats.append(ObjectBounds(obj, minv, maxv, center, radius))
    return stats


def _object_bbox_corners_world(obj: bpy.types.Object) -> list[mathutils.Vector]:
    """Return object's transformed bound-box corners in world coordinates."""
    mw = obj.matrix_world
    return [mw @ mathutils.Vector(corner) for corner in obj.bound_box]


def _object_frame_points_world(
    obj: bpy.types.Object,
    max_mesh_points: int = 2000,
) -> list[mathutils.Vector]:
    """Return representative world-space points for camera framing.

    Uses mesh vertices when available (subsampled for speed). Falls back to
    bound-box corners only when mesh vertices are unavailable.
    """
    points: list[mathutils.Vector] = []
    data = getattr(obj, "data", None)
    if getattr(obj, "type", None) == "MESH" and data is not None and hasattr(data, "vertices"):
        verts = getattr(data, "vertices", [])
        try:
            n = len(verts)
        except Exception:
            n = 0
        if n > 0:
            mw = obj.matrix_world
            if n <= max_mesh_points:
                indices = range(n)
            else:
                step = max(1, n // max_mesh_points)
                indices = range(0, n, step)
            for i in indices:
                try:
                    points.append(mw @ verts[i].co)
                except Exception:
                    continue
            # Always include exact mesh extrema to avoid missing limits from
            # coarse subsampling.
            extrema_indices: set[int] = set()
            try:
                min_x = min(range(n), key=lambda i: float(verts[i].co.x))
                max_x = max(range(n), key=lambda i: float(verts[i].co.x))
                min_y = min(range(n), key=lambda i: float(verts[i].co.y))
                max_y = max(range(n), key=lambda i: float(verts[i].co.y))
                min_z = min(range(n), key=lambda i: float(verts[i].co.z))
                max_z = max(range(n), key=lambda i: float(verts[i].co.z))
                extrema_indices.update((min_x, max_x, min_y, max_y, min_z, max_z))
            except Exception:
                extrema_indices.clear()

            for i in extrema_indices:
                try:
                    points.append(mw @ verts[i].co)
                except Exception:
                    continue
            if points:
                return points

    # Fallback for non-mesh objects or meshes with inaccessible vertex data.
    points.extend(_object_bbox_corners_world(obj))
    return points


def _collect_frame_point_cloud(
    objects: Iterable[bpy.types.Object],
    max_mesh_points_per_object: int = 2000,
) -> tuple[mathutils.Vector, list[mathutils.Vector]]:
    """Collect a world-space point cloud for camera framing.

    Returns (center, points), where center is the midpoint of the cloud AABB.
    """
    points: list[mathutils.Vector] = []
    for obj in objects:
        if _is_placeholder(obj):
            continue
        points.extend(_object_frame_points_world(obj, max_mesh_points=max_mesh_points_per_object))

    if not points:
        z = mathutils.Vector((0.0, 0.0, 0.0))
        return z, []

    bmin = mathutils.Vector(
        (
            min(c.x for c in points),
            min(c.y for c in points),
            min(c.z for c in points),
        )
    )
    bmax = mathutils.Vector(
        (
            max(c.x for c in points),
            max(c.y for c in points),
            max(c.z for c in points),
        )
    )
    center = (bmin + bmax) * 0.5
    return center, points


def _filter_outlier_bounds(stats: list[ObjectBounds]) -> list[ObjectBounds]:
    """Filter object-level outliers from bounds stats using MAD-based z-scores.

    Requires at least 4 items for stable statistics; smaller sets are returned
    unchanged.  Uses conservative z-score gates (6.0) to drop only clear
    outliers while retaining at least 60 % of the original set.
    """
    if len(stats) < 4:
        return stats

    centers = np.asarray([[b.center.x, b.center.y, b.center.z] for b in stats], dtype=np.float64)
    radii = np.asarray([b.radius for b in stats], dtype=np.float64)

    center_med = np.median(centers, axis=0)
    center_dist = np.linalg.norm(centers - center_med[None, :], axis=1)

    dist_med = float(np.median(center_dist))
    dist_mad = float(np.median(np.abs(center_dist - dist_med)))
    dist_scale = max(1.4826 * dist_mad, 1e-6)
    dist_z = np.abs(center_dist - dist_med) / dist_scale

    radius_med = float(np.median(radii))
    radius_mad = float(np.median(np.abs(radii - radius_med)))
    radius_scale = max(1.4826 * radius_mad, 1e-6)
    radius_z = np.abs(radii - radius_med) / radius_scale

    inlier_mask = (dist_z <= 6.0) & (radius_z <= 6.0)
    min_inliers = max(3, math.ceil(0.6 * len(stats)))
    if int(np.count_nonzero(inlier_mask)) >= min_inliers:
        return [stats[i] for i, keep in enumerate(inlier_mask) if bool(keep)]
    return stats


def _robust_bounds_from_stats(
    stats: list[ObjectBounds],
) -> tuple[mathutils.Vector, float]:
    """Compute robust scene bounds by filtering object-level outliers when safe."""
    if not stats:
        return mathutils.Vector((0.0, 0.0, 0.0)), 1.0

    filtered = _filter_outlier_bounds(stats)

    all_mins = [b.min for b in filtered]
    all_maxs = [b.max for b in filtered]
    global_min = mathutils.Vector((min(v.x for v in all_mins), min(v.y for v in all_mins), min(v.z for v in all_mins)))
    global_max = mathutils.Vector((max(v.x for v in all_maxs), max(v.y for v in all_maxs), max(v.z for v in all_maxs)))
    center = (global_min + global_max) * 0.5
    radius = max((global_max - center).length, 1e-3)
    return center, radius


def _scene_content_bounds(
    groups: dict[str, list[bpy.types.Object]],
) -> tuple[mathutils.Vector, float]:
    """Aggregate bounding boxes from all loaded objects to compute (center, radius)."""
    objects = [obj for objs in groups.values() for obj in objs]
    stats = _collect_object_bounds(objects)
    return _robust_bounds_from_stats(stats)


def _clamp_float(value: float, low: float, high: float) -> float:
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


def _solve_projected_mid_shift(
    lateral: Sequence[float],
    depths: Sequence[float],
) -> float:
    """Solve lateral camera-plane shift that centers projected min/max extent.

    Finds delta such that:
        min_i((lateral_i - delta)/depth_i) + max_i((lateral_i - delta)/depth_i) = 0
    using monotonic bisection.
    """
    if not lateral or not depths or len(lateral) != len(depths):
        return 0.0

    d = [max(float(v), 1e-6) for v in depths]
    lateral_values = [float(v) for v in lateral]

    span = max(max(lateral_values) - min(lateral_values), 1e-4)
    lo = min(lateral_values) - 8.0 * span
    hi = max(lateral_values) + 8.0 * span

    def _f(delta: float) -> float:
        vals = [(li - delta) / di for li, di in zip(lateral_values, d, strict=True)]
        return min(vals) + max(vals)

    f_lo = _f(lo)
    f_hi = _f(hi)
    if not math.isfinite(f_lo) or not math.isfinite(f_hi):
        return 0.0

    # Should bracket naturally (f decreases with delta); fallback if degenerate.
    if f_lo * f_hi > 0.0:
        vals = [li / di for li, di in zip(lateral_values, d, strict=True)]
        return 0.5 * (min(vals) + max(vals)) * float(np.median(d))

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f_mid = _f(mid)
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _estimate_auto_point_radius(
    obj: bpy.types.Object, point_count: int, scale: float, radius_min: float, radius_max: float
) -> float:
    _, bound_radius = _object_bounds_world(obj)
    diameter = max(bound_radius * 2.0, 1e-4)
    count = max(int(point_count), 1)
    spacing = diameter / max(count ** (1.0 / 3), 1.0)
    raw_radius = spacing * float(scale)
    return _clamp_float(raw_radius, float(radius_min), float(radius_max))


def _split_subsample_ratio(ratio: float | None) -> tuple[float | None, float | None]:
    if ratio is None or not (0.0 < ratio < 1.0):
        return None, None
    root = math.sqrt(ratio)
    if root <= 0.0:
        return None, None
    return root, ratio / root


def create_object_camera(
    obj: bpy.types.Object, reference_cam: bpy.types.Object, fill_factor: float = 0.90
) -> bpy.types.Object:
    """
    Create a camera for a specific object, matching the reference camera's angle
    but positioned to frame the object.

    :param obj: The object to frame
    :param reference_cam: Camera to copy orientation from
    :param fill_factor: How much of the frame the object should fill (0-1)
    :return: The created camera object
    """
    # Get object bounds
    center, radius = _object_bounds_world(obj)
    frame_points = _object_frame_points_world(obj, max_mesh_points=4000)

    # Create new camera
    cam_obj = new_camera(name=f"Camera_{obj.name}")
    cam = cast(Any, cam_obj.data)
    ref_cam_data = cast(Any, reference_cam.data)

    # Copy camera settings from reference
    cam.type = ref_cam_data.type
    cam.lens = ref_cam_data.lens
    cam.sensor_fit = ref_cam_data.sensor_fit
    cam.sensor_width = ref_cam_data.sensor_width
    cam.sensor_height = ref_cam_data.sensor_height
    # Keep clipping consistent with the reference camera for large scenes.
    if hasattr(ref_cam_data, "clip_start"):
        cam.clip_start = ref_cam_data.clip_start
    if hasattr(ref_cam_data, "clip_end"):
        cam.clip_end = ref_cam_data.clip_end
    # Per-object cameras use centered framing (no principal-point offset).
    cam.shift_x = 0.0
    cam.shift_y = 0.0

    # Get reference camera direction (from camera to scene)
    ref_rotation = reference_cam.matrix_world.to_quaternion()
    ref_forward = ref_rotation @ mathutils.Vector((0.0, 0.0, -1.0))

    # Calculate distance to frame object
    # For perspective camera: distance = radius / (tan(fov/2) * fill_factor)
    sensor_size = cam.sensor_width  # mm
    focal_length = cam.lens  # mm
    fov = 2.0 * math.atan(sensor_size / (2.0 * focal_length))
    distance = (radius / fill_factor) / math.tan(fov / 2.0)

    # Position camera: move back from object center along reference direction
    cam_obj.location = center - ref_forward * distance

    # Tighten from object bbox corners (avoids sphere-based underfill on
    # elongated/rotated objects while preserving GT-derived viewing direction).
    if frame_points:
        _reframe_camera_to_corners(cam_obj, center, frame_points, fill_factor=fill_factor)
    else:
        _aim_object_at(cam_obj, center)

    return cam_obj


def _aim_object_at(obj: bpy.types.Object, target: mathutils.Vector, up_axis: str = "Y"):
    direction = target - obj.location
    if direction.length < 1e-8:
        return
    track_axis = "-Z"  # Area lights emit along -Z
    quat = direction.normalized().to_track_quat(track_axis, up_axis)
    obj.rotation_euler = quat.to_euler()


def _camera_axes(cam_obj: bpy.types.Object):
    """
    Returns (right, up, forward) unit vectors of the camera in world space.
    For cameras in Blender, local -Z is forward and +Y is up.
    """
    R = cam_obj.matrix_world.to_3x3()
    right = (R @ mathutils.Vector((1.0, 0.0, 0.0))).normalized()
    up = (R @ mathutils.Vector((0.0, 1.0, 0.0))).normalized()
    forward = -(R @ mathutils.Vector((0.0, 0.0, 1.0))).normalized()
    return right, up, forward


def _configure_camera_dof_for_preset(
    cam_obj: bpy.types.Object | None,
    focus_point: mathutils.Vector,
    preset: str,
) -> None:
    """Enable subtle DOF only for the beauty preset."""
    if cam_obj is None or getattr(cam_obj, "type", None) != "CAMERA":
        return

    cam_data = getattr(cam_obj, "data", None)
    dof = getattr(cam_data, "dof", None)
    if dof is None:
        return

    use_dof = _normalize_compositor_preset(preset) == "beauty"
    try:
        dof.use_dof = bool(use_dof)
    except Exception:
        return

    if not use_dof:
        return

    # Keep focus on subject center and use a restrained aperture so DOF remains
    # aesthetic without obscuring qualitative geometry details.
    focus_distance = max((focus_point - cam_obj.location).length, 1e-3)
    try:
        dof.focus_object = None
    except Exception:
        pass
    try:
        dof.focus_distance = float(focus_distance)
    except Exception:
        pass
    try:
        dof.aperture_fstop = 8.0
    except Exception:
        pass


def _cam_space_offset(
    center: mathutils.Vector, cam_obj: bpy.types.Object, dist: float, azimuth_rad: float, elevation_rad: float
) -> mathutils.Vector:
    """
    Camera-oriented spherical offset:
    - azimuth rotates around camera UP, from camera FORWARD towards camera RIGHT.
    - elevation tilts from the horizontal great-circle up towards camera UP.
    Returns the world-space position center + offset(dist, az, el).
    """
    right, up, forward = _camera_axes(cam_obj)
    # Yaw around UP: move FORWARD toward RIGHT by azimuth
    horiz_dir = (math.cos(azimuth_rad) * forward + math.sin(azimuth_rad) * right).normalized()
    # Pitch by elevation toward UP
    dir_vec = (math.cos(elevation_rad) * horiz_dir + math.sin(elevation_rad) * up).normalized()
    return center + dir_vec * float(dist)


def _cam_rig_offset(
    cam_obj: bpy.types.Object, dist: float, azimuth_rad: float, elevation_rad: float
) -> mathutils.Vector:
    """Camera-attached rig offset in world space.

    Similar spherical parameterization as _cam_space_offset, but centered on the
    camera origin. This keeps the rig geometry fixed relative to the camera.
    """
    right, up, forward = _camera_axes(cam_obj)
    horiz_dir = (math.cos(azimuth_rad) * forward + math.sin(azimuth_rad) * right).normalized()
    dir_vec = (math.cos(elevation_rad) * horiz_dir + math.sin(elevation_rad) * up).normalized()
    return cam_obj.location + dir_vec * float(dist)


def _set_area_rect_size(light_data, sx: float, sy: float):
    """Best-effort area light sizing that tolerates API differences."""
    sx = max(float(sx), 1e-4)
    sy = max(float(sy), 1e-4)

    # Preferred rectangle sizing when attributes exist
    if hasattr(light_data, "shape"):
        try:
            light_data.shape = "RECTANGLE"
        except Exception:
            pass

    has_size_x = hasattr(light_data, "size_x")
    has_size_y = hasattr(light_data, "size_y")

    if has_size_x:
        light_data.size_x = sx
    if has_size_y:
        light_data.size_y = sy

    # Fallback to square size if rectangular controls are unavailable
    if not (has_size_x and has_size_y) and hasattr(light_data, "size"):
        light_data.size = max(sx, sy)


def _create_area_light(name: str, size: tuple[float, float], energy: float, color=(1.0, 1.0, 1.0)):
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_obj = bpy.data.objects.new(name, light_data)
    _require_scene().collection.objects.link(light_obj)
    light = cast(Any, light_data)

    _set_area_rect_size(light, size[0], size[1])

    # Energy and color exist across versions
    light.energy = float(energy)
    light.color = color

    if hasattr(light, "use_contact_shadow"):
        try:
            light.use_contact_shadow = True
        except Exception:
            pass
    if hasattr(light, "contact_shadow_distance"):
        light.contact_shadow_distance = 0.2
    if hasattr(light, "contact_shadow_bias"):
        light.contact_shadow_bias = 0.03

    return light_obj


def _ensure_area_light(name: str, size: tuple[float, float], energy: float, color=(1.0, 1.0, 1.0)):
    obj = bpy.data.objects.get(name)
    if obj is not None:
        light_data = getattr(obj, "data", None)
        if getattr(obj, "type", None) == "LIGHT" and getattr(light_data, "type", None) == "AREA":
            light_any = cast(Any, light_data)
            _set_area_rect_size(light_any, size[0], size[1])
            light_any.energy = float(energy)
            light_any.color = color
            if hasattr(light_any, "use_contact_shadow"):
                try:
                    light_any.use_contact_shadow = True
                except Exception:
                    pass
            if hasattr(light_any, "contact_shadow_distance"):
                light_any.contact_shadow_distance = 0.2
            if hasattr(light_any, "contact_shadow_bias"):
                light_any.contact_shadow_bias = 0.03
            if not getattr(obj, "users_collection", None):
                try:
                    _require_scene().collection.objects.link(obj)
                except Exception:
                    pass
            return obj
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

    return _create_area_light(name, size, energy, color=color)


def _ensure_shadow_catcher_material(plane: bpy.types.Object) -> None:
    try:
        mat = bpy.data.materials.get("ShadowCatcher_Mat")
        if mat is None:
            mat = bpy.data.materials.new("ShadowCatcher_Mat")
        nt = _get_node_tree(mat)
        if nt is None:
            return
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        trans = nt.nodes.new("ShaderNodeBsdfTransparent")
        nt.links.new(trans.outputs["BSDF"], out.inputs["Surface"])
        if hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND"
        if getattr(getattr(plane, "data", None), "materials", None) is not None:
            cast(Any, plane.data).materials.clear()
            cast(Any, plane.data).materials.append(mat)
    except Exception:
        pass


def _ensure_shadow_catcher_under(target_center: mathutils.Vector, radius: float, ground_z: float | None = None):
    plane = bpy.data.objects.get("ShadowCatcher")
    if plane is None or getattr(plane, "type", None) != "MESH":
        if plane is not None:
            try:
                bpy.data.objects.remove(plane, do_unlink=True)
            except Exception:
                pass
        return _add_shadow_catcher_under(target_center, radius, ground_z=ground_z)

    s = float(radius) * 8.0
    verts = [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)]
    try:
        mesh = cast(Any, plane.data)
        if len(getattr(mesh, "vertices", [])) == 4:
            for i, co in enumerate(verts):
                mesh.vertices[i].co = mathutils.Vector(co)
            mesh.update()
        else:
            new_mesh = bpy.data.meshes.new("ShadowCatcher_mesh")
            new_mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
            new_mesh.update()
            old_mesh = plane.data
            plane.data = new_mesh
            try:
                if old_mesh is not None and int(getattr(old_mesh, "users", 0)) == 0:
                    bpy.data.meshes.remove(cast(Any, old_mesh))
            except Exception:
                pass
    except Exception:
        try:
            bpy.data.objects.remove(plane, do_unlink=True)
        except Exception:
            pass
        return _add_shadow_catcher_under(target_center, radius, ground_z=ground_z)

    z_floor = float(ground_z) if ground_z is not None else (target_center.z - radius * 0.5)
    plane.location = (target_center.x, target_center.y, z_floor)
    plane.rotation_euler = (0.0, 0.0, 0.0)
    try:
        plane.display_type = "WIRE"
    except Exception:
        pass
    try:
        plane.hide_set(True)
    except Exception:
        pass
    try:
        if hasattr(plane, "is_shadow_catcher"):
            plane.is_shadow_catcher = True
        elif hasattr(cast(Any, plane).cycles, "is_shadow_catcher"):
            cast(Any, plane).cycles.is_shadow_catcher = True
    except Exception:
        pass
    try:
        if hasattr(plane, "hide_render"):
            plane.hide_render = False
    except Exception:
        pass
    _ensure_shadow_catcher_material(plane)
    return plane


def _remove_named_object_if_exists(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
    except Exception:
        pass


def _sync_view_layer_transforms() -> None:
    """Force transform evaluation so matrix_world reflects latest local edits."""
    try:
        view_layer = getattr(bpy.context, "view_layer", None)
        if view_layer is not None:
            view_layer.update()
    except Exception:
        pass


def _clear_parent_keep_world(obj: bpy.types.Object | None) -> None:
    if obj is None or getattr(obj, "parent", None) is None:
        return
    try:
        _sync_view_layer_transforms()
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
    except Exception:
        pass


def _set_parent_keep_world(obj: bpy.types.Object | None, parent: bpy.types.Object | None) -> None:
    if obj is None or parent is None:
        return
    try:
        _sync_view_layer_transforms()
        world = obj.matrix_world.copy()
        obj.parent = parent
        obj.matrix_parent_inverse = parent.matrix_world.inverted()
        obj.matrix_world = world
    except Exception:
        pass


def _name_matches_prefixes(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _remove_existing_lighting_rig() -> None:
    """Remove previously created rig objects so reruns stay deterministic."""
    object_prefixes = ("Key_Light", "Fill_Light", "Rim_Light", "Kicker", "ShadowCatcher", "Backdrop")
    light_prefixes = ("Key_Light", "Fill_Light", "Rim_Light", "Kicker")
    mesh_prefixes = ("ShadowCatcher_mesh", "Backdrop_mesh")
    material_prefixes = ("ShadowCatcher_Mat", "Backdrop_mat")

    try:
        objects = list(getattr(bpy.data, "objects", []))
        object_store = getattr(bpy.data, "objects", None)
        if object_store is not None:
            for obj in objects:
                try:
                    name = str(getattr(obj, "name", ""))
                    if not _name_matches_prefixes(name, object_prefixes):
                        continue
                    object_store.remove(obj, do_unlink=True)
                except Exception:
                    continue
    except Exception:
        pass

    def _remove_orphans(data_attr: str, prefixes: Sequence[str]) -> None:
        try:
            datablocks = getattr(bpy.data, data_attr, None)
            if datablocks is None:
                return
            for block in list(datablocks):
                try:
                    name = str(getattr(block, "name", ""))
                    if not _name_matches_prefixes(name, prefixes):
                        continue
                    if int(getattr(block, "users", 0)) > 0:
                        continue
                    datablocks.remove(block)
                except Exception:
                    continue
        except Exception:
            pass

    _remove_orphans("lights", light_prefixes)
    _remove_orphans("meshes", mesh_prefixes)
    _remove_orphans("materials", material_prefixes)


def _apply_material_ao_fallback(strength: float = 0.3, distance: float = 0.2) -> None:
    """Fallback AO when compositor node-tree is unavailable.

    Inserts a tiny node subgraph into mesh materials:
      base_color -> AO -> mix(original, ao_color, fac=strength) -> BSDF base color
    """
    ao_strength = _clamp_float(float(strength), 0.0, 1.0)
    if ao_strength <= 0.0:
        return

    for obj in getattr(_require_scene(), "objects", []):
        data = getattr(obj, "data", None)
        materials = getattr(data, "materials", None)
        if materials is None:
            continue
        for mat in materials:
            if mat is None:
                continue
            nt = _get_node_tree(mat)
            if nt is None:
                continue
            nodes = nt.nodes
            links = nt.links

            bsdf = next((n for n in nodes if "BsdfPrincipled" in getattr(n, "bl_idname", "")), None)
            if bsdf is None or "Base Color" not in bsdf.inputs:
                continue
            base_input = bsdf.inputs["Base Color"]

            ao_node = nodes.get("AO_Fallback_Node")
            if ao_node is None or getattr(ao_node, "bl_idname", "") != "ShaderNodeAmbientOcclusion":
                if ao_node is not None:
                    nodes.remove(ao_node)
                ao_node = nodes.new("ShaderNodeAmbientOcclusion")
                ao_node.name = "AO_Fallback_Node"
            ao_node.label = "AO_Fallback_Node"
            if "Distance" in ao_node.inputs:
                ao_node.inputs["Distance"].default_value = max(float(distance), 0.0)
            if hasattr(ao_node, "samples"):
                ao_node.samples = 8
            if hasattr(ao_node, "only_local"):
                ao_node.only_local = True

            mix_node = nodes.get("AO_Fallback_Mix")
            if mix_node is None or getattr(mix_node, "bl_idname", "") != "ShaderNodeMixRGB":
                if mix_node is not None:
                    nodes.remove(mix_node)
                mix_node = nodes.new("ShaderNodeMixRGB")
                mix_node.name = "AO_Fallback_Mix"
            mix_node.label = "AO_Fallback_Mix"
            mix_node.blend_type = "MIX"
            mix_node.inputs[0].default_value = ao_strength

            source_socket = None
            if base_input.links:
                from_socket = base_input.links[0].from_socket
                from_node_name = str(getattr(getattr(from_socket, "node", None), "name", ""))
                if from_node_name == mix_node.name and mix_node.inputs[1].links:
                    source_socket = mix_node.inputs[1].links[0].from_socket
                elif from_node_name != mix_node.name:
                    source_socket = from_socket
                while base_input.links:
                    links.remove(base_input.links[0])

            for sock in (ao_node.inputs["Color"], mix_node.inputs[1], mix_node.inputs[2]):
                while sock.links:
                    links.remove(sock.links[0])

            if source_socket is not None:
                links.new(source_socket, ao_node.inputs["Color"])
                links.new(source_socket, mix_node.inputs[1])
            else:
                base_color = tuple(base_input.default_value)
                ao_node.inputs["Color"].default_value = base_color
                mix_node.inputs[1].default_value = base_color

            links.new(ao_node.outputs["Color"], mix_node.inputs[2])
            links.new(mix_node.outputs["Color"], base_input)


def _configure_ambient_occlusion(strength: float = 0.3, distance: float = 0.2, preset: str = "paper_clean") -> None:
    """Configure AO and optional compositor post-processing presets.

    Presets:
      - off: AO only (if enabled)
      - paper_clean: mild contrast/saturation shaping for paper figures
      - beauty: paper_clean + subtle bloom
    """
    ao_strength = _clamp_float(float(strength), 0.0, 1.0)
    use_ao = ao_strength > 0.0
    compositor_preset = _normalize_compositor_preset(preset)
    use_grade = compositor_preset in {"paper_clean", "beauty"}
    use_bloom = compositor_preset == "beauty"
    if not use_ao and compositor_preset == "off":
        return

    scene = _require_scene()
    view_layer = None
    try:
        view_layer = scene.view_layers[0] if getattr(scene, "view_layers", None) else None
    except Exception:
        view_layer = None
    if view_layer is None:
        return

    if use_ao:
        try:
            if hasattr(view_layer, "use_pass_ambient_occlusion"):
                view_layer.use_pass_ambient_occlusion = True
        except Exception:
            pass
        try:
            if hasattr(view_layer, "ao_distance"):
                view_layer.ao_distance = max(float(distance), 0.0)
        except Exception:
            pass
        try:
            world = scene.world
            if world is not None and hasattr(world, "light_settings") and hasattr(world.light_settings, "distance"):
                world.light_settings.distance = max(float(distance), 0.0)
        except Exception:
            pass

    try:
        tree = _get_node_tree(scene)
        if tree is None:
            if use_ao:
                logger.debug("Scene node tree unavailable; using material AO fallback")
                _apply_material_ao_fallback(strength=ao_strength, distance=distance)
            elif compositor_preset != "off":
                logger.debug("Scene node tree unavailable; skipping compositor preset")
            return
        nodes = tree.nodes
        links = tree.links

        def _find_node(bl_idname: str) -> Any:
            return next((n for n in nodes if getattr(n, "bl_idname", "") == bl_idname), None)

        def _get_socket(sockets: Any, names: Sequence[str]) -> Any:
            for name in names:
                try:
                    return sockets[name]
                except Exception:
                    continue
            return None

        def _clear_links(socket: Any) -> None:
            if socket is None:
                return
            while socket.links:
                links.remove(socket.links[0])

        def _link(src: Any, dst: Any) -> bool:
            if src is None or dst is None:
                return False
            _clear_links(dst)
            links.new(src, dst)
            return True

        def _ensure_named_node(name: str, bl_idname: str) -> Any:
            node = nodes.get(name)
            if node is not None and getattr(node, "bl_idname", "") != bl_idname:
                nodes.remove(node)
                node = None
            if node is None:
                node = nodes.new(bl_idname)
                node.name = name
            node.label = name
            return node

        def _set_input_default(node: Any, names: Sequence[str], value: float) -> bool:
            for name in names:
                socket = _get_socket(node.inputs, (name,))
                if socket is not None:
                    cast(Any, socket).default_value = value
                    return True
            return False

        rlayers = _find_node("CompositorNodeRLayers")
        if rlayers is None:
            rlayers = nodes.new("CompositorNodeRLayers")
        composite = _find_node("CompositorNodeComposite")
        if composite is None:
            composite = nodes.new("CompositorNodeComposite")

        comp_image_input = _get_socket(composite.inputs, ("Image",))
        base_image = _get_socket(rlayers.outputs, ("Image",))
        alpha_socket = _get_socket(rlayers.outputs, ("Alpha",))
        if comp_image_input is None or base_image is None:
            return

        image_socket = base_image

        if use_ao:
            ao_socket = _get_socket(rlayers.outputs, ("AO", "Ambient Occlusion"))
            if ao_socket is None:
                logger.warning("AO pass output not available on Render Layers; skipping AO compositing")
            else:
                mix_node = _ensure_named_node("AO_Multiply", "CompositorNodeMixRGB")
                mix_node.blend_type = "MULTIPLY"
                if hasattr(mix_node, "use_clamp"):
                    mix_node.use_clamp = True
                if mix_node.inputs:
                    mix_node.inputs[0].default_value = ao_strength
                if len(mix_node.inputs) >= 3:
                    _link(image_socket, mix_node.inputs[1])
                    _link(ao_socket, mix_node.inputs[2])
                    image_socket = mix_node.outputs[0]

        if use_grade:
            grade = _ensure_named_node("PP_BrightContrast", "CompositorNodeBrightContrast")
            _set_input_default(grade, ("Bright", "Brightness"), -0.01)
            _set_input_default(grade, ("Contrast",), 0.16)
            _link(image_socket, _get_socket(grade.inputs, ("Image",)))
            image_socket = grade.outputs[0]

            hue_sat = _ensure_named_node("PP_HueSat", "CompositorNodeHueSat")
            _set_input_default(hue_sat, ("Fac", "Factor"), 1.0)
            _set_input_default(hue_sat, ("Hue",), 0.5)
            _set_input_default(hue_sat, ("Sat", "Saturation"), 1.03)
            _set_input_default(hue_sat, ("Val", "Value"), 1.0)
            _link(image_socket, _get_socket(hue_sat.inputs, ("Image",)))
            image_socket = hue_sat.outputs[0]

        if use_bloom:
            glare = _ensure_named_node("PP_Glare", "CompositorNodeGlare")
            try:
                glare.glare_type = "BLOOM"
            except Exception:
                try:
                    glare.glare_type = "FOG_GLOW"
                except Exception:
                    pass
            try:
                glare.quality = "MEDIUM"
            except Exception:
                pass
            if not _set_input_default(glare, ("Threshold",), 1.0) and hasattr(glare, "threshold"):
                glare.threshold = 1.0
            if not _set_input_default(glare, ("Size",), 6.0) and hasattr(glare, "size"):
                glare.size = 6
            if not _set_input_default(glare, ("Strength",), 0.08) and hasattr(glare, "mix"):
                glare.mix = 0.92  # Legacy equivalent for subtle bloom.
            _link(image_socket, _get_socket(glare.inputs, ("Image",)))
            image_socket = glare.outputs[0]

        final_socket = image_socket
        if alpha_socket is not None:
            set_alpha = _ensure_named_node("AO_SetAlpha", "CompositorNodeSetAlpha")
            if hasattr(set_alpha, "mode"):
                try:
                    set_alpha.mode = "REPLACE_ALPHA"
                except Exception:
                    pass
            _link(image_socket, _get_socket(set_alpha.inputs, ("Image",)))
            _link(alpha_socket, _get_socket(set_alpha.inputs, ("Alpha",)))
            final_socket = set_alpha.outputs[0]

        _link(final_socket, comp_image_input)

        viewer = _find_node("CompositorNodeViewer")
        if viewer is not None:
            _link(final_socket, _get_socket(viewer.inputs, ("Image",)))
    except Exception as exc:
        logger.warning(f"Failed to configure AO/compositor: {exc}")


def _add_shadow_catcher_under(target_center: mathutils.Vector, radius: float, ground_z: float | None = None):
    # Large ground plane set as Cycles shadow catcher
    mesh = bpy.data.meshes.new("ShadowCatcher_mesh")
    s = radius * 8.0
    verts = [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)]
    faces = [(0, 1, 2, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    plane = bpy.data.objects.new("ShadowCatcher", mesh)
    _require_scene().collection.objects.link(plane)
    z_floor = float(ground_z) if ground_z is not None else (target_center.z - radius * 0.5)
    plane.location = (target_center.x, target_center.y, z_floor)
    plane.rotation_euler = (0.0, 0.0, 0.0)
    # Keep the catcher unobtrusive in the viewport
    try:
        plane.display_type = "WIRE"
    except Exception:
        pass
    try:
        # Hide only in viewport; still render so it can catch shadows
        plane.hide_set(True)
    except Exception:
        pass
    # Mark as shadow catcher immediately after linking so Cycles treats it specially.
    # Blender 3.0 moved is_shadow_catcher from obj.cycles to obj directly.
    try:
        if hasattr(plane, "is_shadow_catcher"):
            plane.is_shadow_catcher = True
        elif hasattr(cast(Any, plane).cycles, "is_shadow_catcher"):
            cast(Any, plane).cycles.is_shadow_catcher = True
        # Force a depsgraph update so subsequent material/visibility tweaks see catcher state.
        try:
            cast(Any, _context_any().view_layer).update()
        except Exception:
            pass
    except Exception:
        pass

    # Ensure it renders (must NOT be hide_render) but stays invisible in Combined.
    try:
        if hasattr(plane, "hide_render"):
            plane.hide_render = False
    except Exception:
        pass

    _ensure_shadow_catcher_material(plane)
    return plane


def add_shadow_backdrop(
    subject_obj: bpy.types.Object | None,
    use_shadow_catcher: bool = True,
    size_scale: float = 8.0,
    offset_z: float = -0.05,
) -> bpy.types.Object | None:
    """Create a simple ground plane aligned under the subject.

    If use_shadow_catcher is True and Cycles supports it, the plane will be
    a shadow catcher (invisible in the Combined pass; only shadows preserved
    over transparency). It is hidden in the viewport for cleanliness.
    """
    try:
        center = mathutils.Vector((0.0, 0.0, 0.0))
        radius = 1.0
        min_z = 0.0
        if subject_obj is not None:
            center, radius = _object_bounds_world(subject_obj)
            bb_world = [subject_obj.matrix_world @ mathutils.Vector(corner) for corner in subject_obj.bound_box]
            min_z = min(v.z for v in bb_world) + float(offset_z)
    except Exception:
        pass

    size = radius * float(size_scale)
    mesh = bpy.data.meshes.new("Backdrop_mesh")
    verts = [(-size, -size, 0.0), (size, -size, 0.0), (size, size, 0.0), (-size, size, 0.0)]
    faces = [(0, 1, 2, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    plane = bpy.data.objects.new("Backdrop", mesh)
    _require_scene().collection.objects.link(plane)
    plane.location = (center.x, center.y, min_z)

    if use_shadow_catcher:
        # Minimal transparent material so Cycles treats plane cleanly.
        mat = bpy.data.materials.new("Backdrop_mat")
        nt = _get_node_tree(mat)
        if nt is None:
            return plane
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        trans = nt.nodes.new("ShaderNodeBsdfTransparent")
        nt.links.new(trans.outputs["BSDF"], out.inputs["Surface"])
        cast(Any, plane.data).materials.clear()
        cast(Any, plane.data).materials.append(mat)
        try:
            if hasattr(plane, "is_shadow_catcher"):
                plane.is_shadow_catcher = True
            elif hasattr(cast(Any, plane).cycles, "is_shadow_catcher"):
                cast(Any, plane).cycles.is_shadow_catcher = True
        except Exception:
            pass
        try:
            plane.hide_set(True)  # hide in viewport only
        except Exception:
            pass
        try:
            plane.hide_render = False  # must render to catch shadows
        except Exception:
            pass
    return plane


def add_three_point_rig(
    cam_obj: bpy.types.Object | None,
    subject_obj: bpy.types.Object | None,
    subject_center: mathutils.Vector | None = None,
    subject_radius: float | None = None,
    subject_min_z: float | None = None,
    hdr_strength: float = 0.2,
    world_ambient_scale: float = 0.9,
    samples: int = 384,
    light_scale: float = 0.5,
    distance_scale: float = 1.7,
    fill_distance_ratio: float = 1.28,
    rim_distance_ratio: float = 1.18,
    key_base_energy: float = 100.0,
    fill_energy_ratio: float = 1.0,
    rim_energy_ratio: float = 0.68,
    kicker_energy_ratio: float = 0.03,
    key_strength: float = 1.0,
    fill_strength: float = 1.0,
    rim_strength: float = 1.0,
    kicker_strength: float = 1.0,
    shadow_catcher: bool = True,
    ao_strength: float = 0.3,
    ao_distance: float = 0.2,
    compositor_preset: str = "paper_clean",
    cycles_denoise: bool = True,
    cycles_light_threshold: float = 0.01,
    cycles_preview_factor: float = 0.25,
    cycles_adaptive_threshold: float = 0.03,
    cycles_clamp_direct: float = 0.0,
    cycles_clamp_indirect: float = 2.5,
    cycles_blur_glossy: float = 0.0,
    cycles_max_bounces: int = 4,
    cycles_diffuse_bounces: int = 2,
    cycles_glossy_bounces: int = 2,
    cycles_transmission_bounces: int = 1,
    cycles_transparent_max_bounces: int = 4,
    cycles_caustics_reflective: bool = False,
    cycles_caustics_refractive: bool = False,
    rebuild_rig: bool = True,
):
    # Engine + denoise
    _ensure_cycles(
        samples=samples,
        denoise=cycles_denoise,
        clamp_direct=cycles_clamp_direct,
        clamp_indirect=cycles_clamp_indirect,
        preview_factor=cycles_preview_factor,
        adaptive_threshold=cycles_adaptive_threshold,
        light_threshold=cycles_light_threshold,
        blur_glossy=cycles_blur_glossy,
        max_bounces=cycles_max_bounces,
        diffuse_bounces=cycles_diffuse_bounces,
        glossy_bounces=cycles_glossy_bounces,
        transmission_bounces=cycles_transmission_bounces,
        transparent_max_bounces=cycles_transparent_max_bounces,
        use_caustics_reflective=cycles_caustics_reflective,
        use_caustics_refractive=cycles_caustics_refractive,
    )

    # In persistent mode, update lights/catcher in place instead of recreating.
    if rebuild_rig:
        _remove_existing_lighting_rig()
    _configure_ambient_occlusion(strength=ao_strength, distance=ao_distance, preset=compositor_preset)

    # Pick a subject: mesh > pcd > first visible mesh-like
    if subject_obj is None:
        mesh_candidates = [o for o in _require_scene().objects if getattr(o.data, "polygons", None)]
        subject_obj = mesh_candidates[0] if mesh_candidates else None

    # Fallback center/radius.
    center = subject_center.copy() if subject_center is not None else mathutils.Vector((0.0, 0.0, 0.0))
    radius = max(float(subject_radius), 1e-3) if subject_radius is not None else 1.0
    if subject_center is None or subject_radius is None:
        if subject_obj:
            center, radius = _object_bounds_world(subject_obj)

    # Gentle HDRI fill level.
    # Keep world_ambient_scale as a compatibility trim; hdr_strength is the
    # primary control and ambient scale defaults to neutral (1.0).
    world = cast(Any, _require_scene().world)
    if world and world.node_tree:
        try:
            bg = next(n for n in world.node_tree.nodes if "Background" in n.bl_idname)
            # Keep HDRI subtle, but slightly lift ambient fill to open shadows.
            bg.inputs["Strength"].default_value = float(hdr_strength) * max(float(world_ambient_scale), 0.0)
        except StopIteration:
            pass

    # Distances and sizes
    # Use one consistent lighting radius for distance/size/energy so tiny
    # objects don't get over-illuminated from near-field light placement.
    lighting_radius = _clamp_float(radius, 0.25, 4.0)
    dist = lighting_radius * max(float(distance_scale), 1e-3)

    # Slightly larger emitters for softer, more polished gradients.
    key_size = (lighting_radius * 1.45, lighting_radius * 1.15)
    fill_size = (lighting_radius * 2.2, lighting_radius * 1.6)
    rim_size = (lighting_radius * 1.0, lighting_radius * 0.75)

    # Scale energy with scene size so exposure stays more consistent across
    # object scales while preventing extreme outliers from exploding highlights.
    energy_radius_scale = lighting_radius**2
    key_energy = max(float(key_base_energy), 0.0) * light_scale * energy_radius_scale * max(float(key_strength), 0.0)
    # Keep clear key direction while avoiding overly crushed shadows.
    fill_energy = key_energy * max(float(fill_energy_ratio), 0.0) * max(float(fill_strength), 0.0)
    rim_energy = key_energy * max(float(rim_energy_ratio), 0.0) * max(float(rim_strength), 0.0)

    key = _ensure_area_light("Key_Light", key_size, key_energy, color=(1.0, 0.965, 0.93))
    fill = _ensure_area_light("Fill_Light", fill_size, fill_energy, color=(0.86, 0.93, 1.0))
    rim = _ensure_area_light("Rim_Light", rim_size, rim_energy, color=(0.95, 0.98, 1.0))
    _clear_parent_keep_world(key)
    _clear_parent_keep_world(fill)
    _clear_parent_keep_world(rim)

    # Rig semantics (camera-aware):
    # - key: camera-side/front
    # - fill: opposite of key, still camera-front
    # - rim: opposite camera relative to subject (back/side rim)
    if cam_obj is not None:
        cam_right, cam_up, cam_forward = _camera_axes(cam_obj)
        view_from_subject = cam_obj.location - center  # subject -> camera
        if view_from_subject.length < 1e-8:
            view_from_subject = -cam_forward
        view_from_subject = view_from_subject.normalized()
        key_side_sign = 1.0  # +1 => key on camera-right; flip for camera-left key

        def _dir(front_w: float, side_w: float, up_w: float) -> mathutils.Vector:
            vf = cast(mathutils.Vector, view_from_subject)
            cr = cast(mathutils.Vector, cam_right)
            cu = cast(mathutils.Vector, cam_up)
            vec = vf * float(front_w) + cr * float(side_w) + cu * float(up_w)
            if vec.length < 1e-8:
                return vf
            return vec.normalized()

        key_dir = _dir(front_w=1.00, side_w=key_side_sign * 0.82, up_w=0.62)
        fill_dir = _dir(front_w=1.00, side_w=-key_side_sign * 0.92, up_w=0.36)
        rim_dir = _dir(front_w=-1.00, side_w=key_side_sign * 0.34, up_w=0.54)

        key.location = center + key_dir * dist
        fill.location = center + fill_dir * (dist * max(float(fill_distance_ratio), 1e-3))
        rim.location = center + rim_dir * (dist * max(float(rim_distance_ratio), 1e-3))
    else:
        # Legacy world-oriented fallback
        def polar(offset_az, offset_el, r):
            return mathutils.Vector(
                (
                    r * math.cos(offset_el) * math.cos(offset_az),
                    r * math.cos(offset_el) * math.sin(offset_az),
                    r * math.sin(offset_el),
                )
            )

        key_az, key_el = math.radians(30), math.radians(30)
        fill_az, fill_el = math.radians(-58), math.radians(18)
        rim_az, rim_el = math.radians(78), math.radians(24)
        key.location = center + polar(key_az, key_el, dist)
        fill.location = center + polar(fill_az, fill_el, dist * max(float(fill_distance_ratio), 1e-3))
        rim.location = center + polar(rim_az, rim_el, dist * max(float(rim_distance_ratio), 1e-3))

    _aim_object_at(key, center)
    _aim_object_at(fill, center)
    _aim_object_at(rim, center)

    # Shadow catcher plane.
    if shadow_catcher:
        catcher = None
        if not rebuild_rig:
            existing = bpy.data.objects.get("ShadowCatcher")
            if existing is not None and getattr(existing, "type", None) == "MESH":
                catcher = existing

        if catcher is None:
            if subject_min_z is None and subject_obj is not None and getattr(subject_obj, "bound_box", None):
                try:
                    bb_world = [subject_obj.matrix_world @ mathutils.Vector(corner) for corner in subject_obj.bound_box]
                    subject_min_z = min(v.z for v in bb_world)
                except Exception:
                    subject_min_z = None
            # Keep catcher close to scene support height so contact shadows remain visible.
            if subject_min_z is not None:
                floor_z = float(subject_min_z) + max(float(radius) * 0.005, 0.0005)
            else:
                floor_z = float(center.z - radius * 0.5)
            catcher = _ensure_shadow_catcher_under(center, radius, ground_z=floor_z)
    else:
        if rebuild_rig:
            _remove_named_object_if_exists("ShadowCatcher")
        catcher = None

    # Optional: place a tiny "practical" light near the anchor camera for sparkle
    kicker = None
    if cam_obj is not None:
        kicker = _ensure_area_light(
            "Kicker",
            (lighting_radius * 0.35, lighting_radius * 0.3),
            key_energy * max(float(kicker_energy_ratio), 0.0) * max(float(kicker_strength), 0.0),
            color=(1.0, 0.985, 0.96),
        )
        _clear_parent_keep_world(kicker)
        right, up, forward = _camera_axes(cam_obj)
        kicker.location = (
            cam_obj.location
            + forward * (dist * 0.52)
            + right * (lighting_radius * 0.20)
            + up * (lighting_radius * 0.28)
        )
        _aim_object_at(kicker, center)
    elif rebuild_rig:
        _remove_named_object_if_exists("Kicker")

    # Final clamp: ensure all lights are above the ground/shadow plane using the actual catcher if present
    try:
        ground_z = (
            float(catcher.location.z)
            if catcher is not None
            else (float(subject_min_z) if subject_min_z is not None else float(center.z - radius * 0.5))
        )
        # Scale-aware clearance so lights stay above ground and don't point upward from below.
        base_clearance = max(float(radius) * 0.38, 0.12)

        min_light_z = ground_z + base_clearance
        if subject_min_z is not None:
            min_light_z = max(min_light_z, float(subject_min_z) + base_clearance)
        min_light_z = max(min_light_z, float(center.z) + float(radius) * 0.02)

        cam_forward = None
        view_from_subject = None
        min_front_common = 0.0
        min_front_kicker = 0.0
        front_subject_min = 0.0
        rim_back_subject_min = 0.0
        if cam_obj is not None:
            _right, _up, cam_forward = _camera_axes(cam_obj)
            view_from_subject = cam_obj.location - center
            if view_from_subject.length < 1e-8:
                view_from_subject = -cam_forward
            view_from_subject = view_from_subject.normalized()
            min_front_common = max(float(dist) * 0.10, float(radius) * 0.06, 0.03)
            min_front_kicker = max(float(dist) * 0.28, float(radius) * 0.14, 0.08)
            front_subject_min = max(float(radius) * 0.14, 0.04)
            rim_back_subject_min = max(float(radius) * 0.14, 0.04)

        def _constrain_light(_l: bpy.types.Object | None, role: str):
            if not _l:
                return
            changed = False

            if view_from_subject is not None:
                subject_side = float((_l.location - center).dot(view_from_subject))
                if role in {"key", "fill"} and subject_side < front_subject_min:
                    _l.location += view_from_subject * (front_subject_min - subject_side)
                    changed = True
                if role == "rim" and subject_side > -rim_back_subject_min:
                    _l.location -= view_from_subject * (subject_side + rim_back_subject_min)
                    changed = True

            if cam_obj is not None and cam_forward is not None:
                front = float((_l.location - cam_obj.location).dot(cam_forward))
                min_front = min_front_kicker if role == "kicker" else min_front_common
                if front < min_front:
                    _l.location += cam_forward * (min_front - front)
                    changed = True

            if _l.location.z <= min_light_z:
                _l.location.z = min_light_z
                changed = True

            if changed:
                _aim_object_at(_l, center)

        _constrain_light(key, "key")
        _constrain_light(fill, "fill")
        _constrain_light(rim, "rim")
        _constrain_light(kicker, "kicker")
    except Exception:
        pass

    if cam_obj is not None:
        for light_obj in (key, fill, rim, kicker):
            _set_parent_keep_world(light_obj, cam_obj)

    _configure_camera_dof_for_preset(cam_obj, center, compositor_preset)

    return {"key": key, "fill": fill, "rim": rim}


def configure_scene(
    hdri: Path | None, width: int, height: int, hdri_strength: float, view_exposure: float = 0.0
) -> None:
    """Reset the default scene and ensure renders are predictable."""
    modify_default_scene_objects()
    set_world_background(color=(1.0, 1.0, 1.0), strength=0.0)
    if hdri:
        logger.info(f"Using HDRI background: {hdri}")
        set_world_background_hdr_img(str(hdri), strength=float(hdri_strength))

    scene = _require_scene()
    set_scene_resolution(int(width), int(height))
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = float(view_exposure)
    # Set viewport to look through active camera with rendered shading
    screen = cast(Any, _context_any().screen)
    for area in screen.areas if screen else []:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.region_3d.view_perspective = "CAMERA"
                    space.shading.type = "RENDERED"
            break


def _align_default_camera(reference_cam: bpy.types.Object) -> None:
    default_cam = bpy.data.objects.get("Camera")
    if default_cam is None:
        logger.debug("Default camera not present; skipping alignment")
        return

    center = mathutils.Vector((0.0, 0.0, 0.0))
    offset = reference_cam.location - center
    opposite_position = center + mathutils.Vector((-4 * offset.x, -4 * offset.y, offset.z))
    default_cam.location = opposite_position
    _aim_object_at(default_cam, center)


def _apply_default_camera_transform(cam: bpy.types.Object | None, config: LoaderConfig) -> bpy.types.Object | None:
    """Apply CLI-provided camera transform to the default camera.

    Precedence:
    - If camera_look_at is provided, it overrides any rotation inputs (aims at point).
    - Else if quaternion provided, use it.
    - Else if euler provided, use it with camera_rotation_mode.
    Location is applied if provided.
    """
    if cam is None:
        return None

    try:
        if config.camera_location is not None:
            x, y, z = (
                float(config.camera_location[0]),
                float(config.camera_location[1]),
                float(config.camera_location[2]),
            )
            cam.location = (x, y, z)

        if config.camera_look_at is not None:
            target = mathutils.Vector(
                (float(config.camera_look_at[0]), float(config.camera_look_at[1]), float(config.camera_look_at[2]))
            )
            _aim_object_at(cam, target)
            return cam

        if config.camera_rotation_quat is not None:
            w, x, y, z = (
                float(config.camera_rotation_quat[0]),
                float(config.camera_rotation_quat[1]),
                float(config.camera_rotation_quat[2]),
                float(config.camera_rotation_quat[3]),
            )
            cam.rotation_mode = "QUATERNION"
            cam.rotation_quaternion = mathutils.Quaternion((w, x, y, z))
            return cam

        if config.camera_rotation_euler_deg is not None:
            rx, ry, rz = (
                math.radians(float(config.camera_rotation_euler_deg[0])),
                math.radians(float(config.camera_rotation_euler_deg[1])),
                math.radians(float(config.camera_rotation_euler_deg[2])),
            )
            # Validate/assign rotation mode if supported
            rotation_mode = str(config.camera_rotation_mode or "XYZ").upper()
            if rotation_mode not in {"QUATERNION", "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX", "AXIS_ANGLE"}:
                rotation_mode = "XYZ"
            cast(Any, cam).rotation_mode = rotation_mode
            cam.rotation_euler = mathutils.Euler((rx, ry, rz), cam.rotation_mode)
    except Exception:
        pass

    return cam


def setup_reference_camera(out_dir: Path, stem: str, config: LoaderConfig) -> bpy.types.Object | None:
    cam_json = out_dir / f"{stem}_camera.json"
    if not cam_json.exists():
        logger.warning(f"No camera metadata found at {cam_json}")
        return None

    with cam_json.open("r") as handle:
        cam_data = json.load(handle)

    intrinsic = np.asarray(cam_data["intrinsic"], dtype=np.float64)
    extrinsic = np_mat44(cam_data["extrinsic"])
    width = int(cam_data.get("width", 512))
    height = int(cam_data.get("height", 512))

    cam_obj = new_camera(name=cam_json.stem)
    set_camera_from_opencv_params(cam_obj, intrinsic, extrinsic, width, height)
    set_scene_resolution(width, height)
    try:
        cam_obj["_source_width"] = int(width)
        cam_obj["_source_height"] = int(height)
    except Exception:
        pass

    rgb_path = out_dir / f"{stem}_rgb.png"
    normals_path = out_dir / f"{stem}_normals.png"
    depth_path = out_dir / f"{stem}_depth.png"

    set_camera_background_images(
        cam_obj,
        rgb_path=rgb_path if rgb_path.exists() else None,
        normals_path=normals_path if normals_path.exists() else None,
        depth_path=depth_path if depth_path.exists() else None,
    )

    if config.as_planes:
        add_camera_image_planes(
            cam_obj,
            intrinsic=intrinsic,
            width_px=width,
            height_px=height,
            rgb_path=rgb_path if rgb_path.exists() else None,
            normals_path=normals_path if normals_path.exists() else None,
            depth_path=depth_path if depth_path.exists() else None,
            depth_m=config.plane_depth,
            add_camera_body=config.add_camera_body,
        )

        # Mirror visibility for attachments and group camera + attachments together
        # Per-object visibility for attachments is unnecessary; collection controls it

        try:
            attachments_coll = bpy.data.collections.get(f"{cam_obj.name}_attachments")
            if attachments_coll is not None:
                _move_object_to_collection(cam_obj, attachments_coll)
                # Prefer collection-level visibility so camera + attachments act as one unit
                try:
                    if hasattr(attachments_coll, "hide_render"):
                        attachments_coll.hide_render = True  # viewport-only for overlays by default
                    if hasattr(attachments_coll, "hide_viewport"):
                        attachments_coll.hide_viewport = False

                    # Sync active View Layers' LayerCollections (viewport visibility lives here)
                    def _set_layer_coll_visibility(layer_coll, target_coll, hide_flag: bool) -> bool:
                        if getattr(layer_coll, "collection", None) is target_coll:
                            if hasattr(layer_coll, "hide_viewport"):
                                layer_coll.hide_viewport = hide_flag
                            return True
                        for child in getattr(layer_coll, "children", []) or []:
                            if _set_layer_coll_visibility(child, target_coll, hide_flag):
                                return True
                        return False

                    scene = _require_scene()
                    for view_layer in getattr(scene, "view_layers", []) or []:
                        _set_layer_coll_visibility(view_layer.layer_collection, attachments_coll, False)
                except Exception:
                    pass
        except Exception:
            pass

        _set_visibility(cam_obj, visible="viewport")

    if not config.canonical_viewpoints:
        _align_default_camera(cam_obj)
    return cam_obj


def import_object_groups(out_dir: Path, stem: str, config: LoaderConfig) -> dict[str, list[bpy.types.Object]]:
    auto_kwargs = {
        "auto_tune_points": config.point_auto_enabled,
        "auto_target_points": config.point_auto_target,
        "auto_radius_scale": config.point_auto_radius_scale,
        "auto_radius_min": config.point_auto_radius_min,
        "auto_radius_max": config.point_auto_radius_max,
        "auto_use_nn": config.point_auto_use_nn,
        "auto_nn_sample_size": config.point_auto_nn_sample,
        "auto_nn_subsample_if_small": config.point_auto_nn_subsample_if_small,
    }
    specs: Iterable[tuple[str, Path, Callable[..., bpy.types.Object | None], dict, bool]] = [
        (
            "inputs",
            out_dir / f"{stem}_inputs.ply",
            import_pcd,
            {
                "point_radius": config.inputs_point_radius,
                "subsample_ratio": config.inputs_subsample,
                "color": config.inputs_color,
                "roughness": 0.9,
                "specular_level": 0.08,
                **auto_kwargs,
            },
            True,
        ),
        ("mesh", out_dir / f"{stem}_mesh.obj", import_mesh, {}, False),
        (
            "pcd",
            out_dir / f"{stem}_pointcloud.ply",
            import_pcd,
            {
                "point_radius": config.pcd_point_radius,
                "color": config.pcd_color,
                "roughness": 0.78,
                "specular_level": 0.12,
                **auto_kwargs,
            },
            True,
        ),
        (
            "occ",
            out_dir / f"{stem}_occ.ply",
            import_pcd,
            {
                "point_radius": config.occ_point_radius,
                "color": config.occ_color,
                "roughness": 0.78,
                "specular_level": 0.12,
                **auto_kwargs,
            },
            True,
        ),
        (
            "free",
            out_dir / f"{stem}_free.ply",
            import_pcd,
            {
                "point_radius": config.free_point_radius,
                "color": config.free_color,
                "roughness": 0.82,
                "specular_level": 0.06,
                "alpha": 0.1,
                **auto_kwargs,
            },
            True,
        ),
        (
            "logits",
            out_dir / f"{stem}_logits.ply",
            import_pcd,
            {
                "point_radius": config.logits_point_radius,
                "color": config.logits_color,
                "roughness": 0.76,
                "specular_level": 0.12,
                **auto_kwargs,
            },
            True,
        ),
        ("pred", out_dir / f"{stem}_pred_mesh.obj", import_mesh, {}, False),
    ]

    groups: dict[str, list[bpy.types.Object]] = {}
    for label, path, importer, kwargs, share_density in specs:
        if share_density and importer is import_pcd:
            objs = import_pcd_series_shared(path, **kwargs)
        else:
            objs = import_series(path, importer, **kwargs)
        if objs:
            groups[label] = objs
            logger.debug(f"Loaded {len(objs)} objects for {label} from {path.name}")
    return groups


def log_object_summary(groups: dict[str, list[bpy.types.Object]]) -> None:
    if not groups:
        logger.warning("No geometry loaded")
        return
    for label, objs in groups.items():
        logger.info(f"{label}: {[obj.name for obj in objs]}")


def assign_group_collections(stem: str, groups: dict[str, list[bpy.types.Object]]) -> None:
    mapping = {
        "inputs": f"{stem}_inputs",
        "mesh": f"{stem}_meshes",
        "pcd": f"{stem}_pointclouds",
        "occ": f"{stem}_occupied_points",
        "free": f"{stem}_free_points",
        "logits": f"{stem}_logits",
        "pred": f"{stem}_pred_meshes",
    }
    for label, collection_name in mapping.items():
        objs = groups.get(label, [])
        if objs:
            group_objects_into_collection(objs, collection_name)


def set_default_group_visibility(stem: str, groups: dict[str, list[bpy.types.Object]]) -> None:
    """Mirror previous per-object defaults at the collection level.

    Defaults:
      - inputs: visible in both
      - pred: visible in both
      - mesh/pcd/occ/free/logits: hidden by default
    """
    name_map = {
        "inputs": f"{stem}_inputs",
        "mesh": f"{stem}_meshes",
        "pcd": f"{stem}_pointclouds",
        "occ": f"{stem}_occupied_points",
        "free": f"{stem}_free_points",
        "logits": f"{stem}_logits",
        "pred": f"{stem}_pred_meshes",
    }

    desired: dict[str, Literal["viewport", "render", "both"] | None] = {
        "inputs": "both",
        "pred": "both",
        "mesh": None,
        "pcd": None,
        "occ": None,
        "free": None,
        "logits": None,
    }

    for label, objs in groups.items():
        if not objs:
            continue
        coll_name = name_map.get(label)
        if not coll_name:
            continue
        coll = ensure_collection(coll_name)
        try:
            _set_collection_visibility(coll, desired.get(label))
        except Exception:
            pass


def apply_group_visibility_overrides(
    stem: str, overrides: dict[str, Literal["viewport", "render", "both"] | None] | None
) -> None:
    if not overrides:
        return
    name_map = {
        "inputs": f"{stem}_inputs",
        "mesh": f"{stem}_meshes",
        "pcd": f"{stem}_pointclouds",
        "occ": f"{stem}_occupied_points",
        "free": f"{stem}_free_points",
        "logits": f"{stem}_logits",
        "pred": f"{stem}_pred_meshes",
    }
    for label, mode in overrides.items():
        coll_name = name_map.get(label)
        if not coll_name:
            continue
        coll = ensure_collection(coll_name)
        try:
            _set_collection_visibility(coll, mode)
        except Exception:
            pass


def create_object_cameras_for_targets(
    stem: str, default_cam: bpy.types.Object | None, groups: dict[str, list[bpy.types.Object]]
) -> None:
    if default_cam is None:
        logger.debug("No default camera available for per-object views")
        return
    gt_targets = [obj for obj in groups.get("mesh", []) if not _is_placeholder(obj)]
    pred_targets = [obj for obj in groups.get("pred", []) if not _is_placeholder(obj)]
    obj_cam_coll = ensure_collection(f"{stem}_object_cameras")
    # Shared GT/pred framing uses GT geometry only; keep a slightly looser fill
    # so pred deviations from GT do not clip at image borders.
    object_fill_factor = 0.85

    def _series_idx(obj: bpy.types.Object) -> int:
        """Extract stable object index from names like *_mesh_12(.001)."""
        # Blender can auto-append ".001" when names already exist in-session.
        # Strip that suffix first so index parsing stays stable across reruns.
        name = re.sub(r"\.\d{3}$", "", str(getattr(obj, "name", "")))

        # Prefer explicit mesh/pred_mesh index forms.
        for pat in (r"_pred_mesh_(\d+)$", r"_mesh_(\d+)$"):
            m = re.search(pat, name)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    continue

        # Fallback: any trailing underscore index.
        m = re.search(r"_(\d+)$", name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        return 0

    # When both GT and pred are present, enforce matched per-object viewpoints by
    # pairing objects through their series index and sharing one camera per pair.
    # Camera framing is always derived from GT; unmatched pred objects are skipped.
    if gt_targets and pred_targets:
        gt_by_idx: dict[int, list[bpy.types.Object]] = {}
        pred_by_idx: dict[int, list[bpy.types.Object]] = {}
        for obj in gt_targets:
            gt_by_idx.setdefault(_series_idx(obj), []).append(obj)
        for obj in pred_targets:
            pred_by_idx.setdefault(_series_idx(obj), []).append(obj)

        def _safe_bounds(obj: bpy.types.Object) -> tuple[mathutils.Vector, float]:
            try:
                return _object_bounds_world(obj)
            except Exception:
                return mathutils.Vector((0.0, 0.0, 0.0)), 1e-3

        # Always create one camera per GT object, with GT index/ordinal naming.
        key_to_anchor: dict[tuple[int, int], bpy.types.Object] = {}
        key_to_camera: dict[tuple[int, int], bpy.types.Object] = {}
        key_to_bounds: dict[tuple[int, int], tuple[mathutils.Vector, float]] = {}
        gt_keys: list[tuple[int, int]] = []
        for idx in sorted(gt_by_idx.keys()):
            gt_list = gt_by_idx[idx]
            for ordinal, gt_obj in enumerate(gt_list):
                key = (idx, ordinal)
                cam_suffix = f"{idx}" if ordinal == 0 else f"{idx}_{ordinal}"
                try:
                    obj_cam = create_object_camera(gt_obj, default_cam, fill_factor=object_fill_factor)
                    obj_cam.name = f"Camera_pair_{cam_suffix}"
                    _move_object_to_collection(obj_cam, obj_cam_coll)
                    gt_obj["_solo_camera"] = obj_cam.name
                    gt_obj["_solo_light_anchor"] = gt_obj.name
                    key_to_anchor[key] = gt_obj
                    key_to_camera[key] = obj_cam
                    key_to_bounds[key] = _safe_bounds(gt_obj)
                    gt_keys.append(key)
                    logger.info(f"Created shared camera {obj_cam.name} for {gt_obj.name}")
                except Exception as exc:
                    logger.warning(f"Failed to create shared camera for GT idx={idx} ord={ordinal}: {exc}")

        # Flatten pred entries in stable (idx, ordinal) order.
        pred_entries: list[tuple[int, int, bpy.types.Object]] = []
        for idx in sorted(pred_by_idx.keys()):
            pred_list = pred_by_idx[idx]
            for ordinal, pred_obj in enumerate(pred_list):
                pred_entries.append((idx, ordinal, pred_obj))

        # Prefer index pairing only when index-alignment quality is plausible.
        index_pairs: list[tuple[tuple[int, int], bpy.types.Object, float]] = []
        for idx, ordinal, pred_obj in pred_entries:
            key = (idx, ordinal)
            if key not in key_to_camera:
                continue
            pred_center, pred_radius = _safe_bounds(pred_obj)
            gt_center, gt_radius = key_to_bounds[key]
            denom = max(float(pred_radius + gt_radius), 1e-6)
            norm_dist = float((pred_center - gt_center).length / denom)
            index_pairs.append((key, pred_obj, norm_dist))

        use_index_pairs = False
        if len(index_pairs) >= 3:
            norms = sorted(pair[2] for pair in index_pairs)
            median_norm = norms[len(norms) // 2]
            use_index_pairs = median_norm <= 0.45
            if not use_index_pairs:
                logger.warning(
                    f"Pred/GT index alignment looks poor (median normalized center distance={median_norm:.2f}); "
                    f"using spatial pred->GT pairing for solo cameras"
                )

        assigned_pred_ids: set[int] = set()
        if use_index_pairs:
            for key, pred_obj, _norm_dist in index_pairs:
                cam = key_to_camera.get(key)
                anchor = key_to_anchor.get(key)
                if cam is None or anchor is None:
                    continue
                try:
                    pred_obj["_solo_camera"] = cam.name
                    pred_obj["_solo_light_anchor"] = anchor.name
                    assigned_pred_ids.add(id(pred_obj))
                except Exception:
                    continue
        else:
            # One-to-one greedy matching by normalized center distance.
            pred_bounds: list[tuple[int, bpy.types.Object, mathutils.Vector, float]] = []
            for rank, (_idx, _ordinal, pred_obj) in enumerate(pred_entries):
                c, r = _safe_bounds(pred_obj)
                pred_bounds.append((rank, pred_obj, c, r))

            gt_rank = {key: rank for rank, key in enumerate(sorted(gt_keys))}
            pair_candidates: list[tuple[float, int, int, tuple[int, int], bpy.types.Object]] = []
            for pred_rank, pred_obj, pred_center, pred_radius in pred_bounds:
                for key in sorted(gt_keys):
                    gt_center, gt_radius = key_to_bounds[key]
                    denom = max(float(pred_radius + gt_radius), 1e-6)
                    score = float((pred_center - gt_center).length / denom)
                    pair_candidates.append((score, pred_rank, gt_rank[key], key, pred_obj))
            pair_candidates.sort(key=lambda item: (item[0], item[1], item[2]))

            used_gt: set[tuple[int, int]] = set()
            for _score, _pred_rank, _gt_rank, key, pred_obj in pair_candidates:
                if key in used_gt or id(pred_obj) in assigned_pred_ids:
                    continue
                cam = key_to_camera.get(key)
                anchor = key_to_anchor.get(key)
                if cam is None or anchor is None:
                    continue
                try:
                    pred_obj["_solo_camera"] = cam.name
                    pred_obj["_solo_light_anchor"] = anchor.name
                    assigned_pred_ids.add(id(pred_obj))
                    used_gt.add(key)
                except Exception:
                    continue

        for _idx, _ordinal, pred_obj in pred_entries:
            if id(pred_obj) in assigned_pred_ids:
                continue
            # No GT match — create an individual camera for this pred object.
            try:
                obj_cam = create_object_camera(pred_obj, default_cam, fill_factor=object_fill_factor)
                obj_cam.name = f"Camera_{pred_obj.name}"
                _move_object_to_collection(obj_cam, obj_cam_coll)
                pred_obj["_solo_camera"] = obj_cam.name
                pred_obj["_solo_light_anchor"] = pred_obj.name
                logger.warning(f"No GT match for {pred_obj.name}; using individual camera {obj_cam.name}")
            except Exception as exc:
                logger.warning(f"Failed to create individual camera for {pred_obj.name}: {exc}")
        return

    target_objs: list[bpy.types.Object] = [*gt_targets, *pred_targets]
    if not target_objs:
        # Fallback: preserve previous behavior for non-mesh-only scenes.
        priority = ["logits", "occ", "pcd", "inputs"]
        for key in priority:
            objs = [obj for obj in groups.get(key, []) if not _is_placeholder(obj)]
            if objs:
                target_objs = objs
                break

    if not target_objs:
        return

    for target in target_objs:
        try:
            obj_cam = create_object_camera(target, default_cam, fill_factor=object_fill_factor)
            _move_object_to_collection(obj_cam, obj_cam_coll)
            try:
                target["_solo_camera"] = obj_cam.name
                target["_solo_light_anchor"] = target.name
            except Exception:
                pass
            logger.info(f"Created camera {obj_cam.name} for {target.name}")
        except Exception as exc:
            logger.warning(f"Failed to create camera for {target.name}: {exc}")


def _create_canonical_viewpoints(
    groups: dict[str, list[bpy.types.Object]],
    stem: str,
    reference_cam: bpy.types.Object | None,
    elevation_deg: float = 30.0,
    distance_scale: float = 2.5,
) -> None:
    """Create cameras at standard azimuth angles around the scene content."""
    # Anchor on GT meshes so canonical views are pred-invariant.
    gt_objs = [obj for obj in groups.get("mesh", []) if not _is_placeholder(obj)]
    if gt_objs:
        stats = _collect_object_bounds(gt_objs)
        center, radius = _robust_bounds_from_stats(stats)
    else:
        center, radius = _scene_content_bounds(groups)
    distance = radius * distance_scale
    elev_rad = math.radians(elevation_deg)

    viewpoints = [
        ("front", 0.0),
        ("front_right", 45.0),
        ("right", 90.0),
        ("back", 180.0),
        ("front_left", 315.0),
        ("top", None),  # handled separately
    ]

    coll = ensure_collection(f"{stem}_canonical_cameras")
    active_cam = None

    for name, azimuth_deg in viewpoints:
        cam_obj = new_camera(name=f"{stem}_canonical_{name}")
        cam = cast(Any, cam_obj.data)

        # Copy lens/sensor settings from reference camera if available.
        # Shifts are NOT copied: canonical viewpoints are synthetic and should
        # use centered intrinsics regardless of the reference sensor's offset.
        if reference_cam is not None:
            ref_cam_data = cast(Any, reference_cam.data)
            cam.type = ref_cam_data.type
            cam.lens = ref_cam_data.lens
            cam.sensor_fit = ref_cam_data.sensor_fit
            cam.sensor_width = ref_cam_data.sensor_width
            cam.sensor_height = ref_cam_data.sensor_height
            cam.shift_x = 0.0
            cam.shift_y = 0.0
            if hasattr(ref_cam_data, "clip_start"):
                cam.clip_start = ref_cam_data.clip_start
            if hasattr(ref_cam_data, "clip_end"):
                cam.clip_end = ref_cam_data.clip_end

        if name == "top":
            # Near-vertical view
            top_elev_rad = math.radians(85.0)
            cam_obj.location = (
                center.x,
                center.y - distance * math.cos(top_elev_rad),
                center.z + distance * math.sin(top_elev_rad),
            )
        else:
            az_rad = math.radians(azimuth_deg)
            cam_obj.location = (
                center.x + distance * math.sin(az_rad) * math.cos(elev_rad),
                center.y - distance * math.cos(az_rad) * math.cos(elev_rad),
                center.z + distance * math.sin(elev_rad),
            )

        _aim_object_at(cam_obj, center)
        _move_object_to_collection(cam_obj, coll)
        logger.info(f"Created canonical camera: {cam_obj.name}")

        if name == "front_right":
            active_cam = cam_obj

    if active_cam is not None:
        scene = _require_scene()
        scene.camera = active_cam
        logger.info(f"Set active camera to {active_cam.name}")


def _create_orbit_camera(
    groups: dict[str, list[bpy.types.Object]],
    stem: str,
    reference_cam: bpy.types.Object | None,
    frames: int = 120,
    elevation_deg: float = 25.0,
    distance_scale: float = 2.5,
) -> None:
    """Create an animated camera that orbits around the scene content.

    An Empty pivot is placed at the scene center with a camera parented to it.
    The pivot's Z-rotation is keyframed from 0 to 2*pi over *frames* frames.
    """
    # Anchor on GT meshes so orbit is pred-invariant.
    gt_objs = [obj for obj in groups.get("mesh", []) if not _is_placeholder(obj)]
    if gt_objs:
        stats = _collect_object_bounds(gt_objs)
        center, radius = _robust_bounds_from_stats(stats)
    else:
        center, radius = _scene_content_bounds(groups)
    distance = radius * distance_scale
    elev_rad = math.radians(elevation_deg)

    coll = ensure_collection(f"{stem}_orbit")

    # Pivot empty at scene center
    pivot = bpy.data.objects.new(f"{stem}_orbit_pivot", None)
    pivot.empty_display_type = "PLAIN_AXES"
    pivot.empty_display_size = radius * 0.25
    pivot.location = center
    scene = _require_scene()
    scene.collection.objects.link(pivot)
    _move_object_to_collection(pivot, coll)

    # Camera offset relative to center (azimuth=0, i.e. -Y direction in Blender)
    cam_obj = new_camera(name=f"{stem}_orbit_camera")
    cam = cast(Any, cam_obj.data)
    # Copy lens/sensor settings; shifts stay at zero (synthetic viewpoint).
    if reference_cam is not None:
        ref_cam_data = cast(Any, reference_cam.data)
        cam.type = ref_cam_data.type
        cam.lens = ref_cam_data.lens
        cam.sensor_fit = ref_cam_data.sensor_fit
        cam.sensor_width = ref_cam_data.sensor_width
        cam.sensor_height = ref_cam_data.sensor_height
        cam.shift_x = 0.0
        cam.shift_y = 0.0
        if hasattr(ref_cam_data, "clip_start"):
            cam.clip_start = ref_cam_data.clip_start
        if hasattr(ref_cam_data, "clip_end"):
            cam.clip_end = ref_cam_data.clip_end

    cam_obj.location = (
        center.x,
        center.y - distance * math.cos(elev_rad),
        center.z + distance * math.sin(elev_rad),
    )
    _aim_object_at(cam_obj, center)
    _move_object_to_collection(cam_obj, coll)

    # Parent camera to pivot (keep current world transform)
    cam_obj.parent = pivot
    cam_obj.matrix_parent_inverse = pivot.matrix_world.inverted()

    # Keyframe pivot Z-rotation for full 360-degree sweep
    scene = _require_scene()
    scene.frame_start = 1
    scene.frame_end = frames
    pivot.rotation_mode = "XYZ"

    angles = np.linspace(0.0, 2.0 * math.pi, frames + 1)[:-1]  # exclude duplicate last frame
    for i, angle in enumerate(angles):
        frame = i + 1
        pivot.rotation_euler[2] = angle
        pivot.keyframe_insert(data_path="rotation_euler", index=2, frame=frame)

    # Set all keyframes to LINEAR interpolation for constant angular velocity
    animation_data = cast(Any, pivot.animation_data)
    action = cast(Any, getattr(animation_data, "action", None))
    if action is None:
        scene.camera = cam_obj
        logger.info(f"Created orbit camera with {frames} frames at {elevation_deg} deg elevation")
        return
    try:
        fcurves = action.fcurves  # Blender < 4.4 (or 4.4 legacy proxy)
    except AttributeError:
        # Blender 4.4+/5.0: layered actions — fcurves live in channelbags
        slot = animation_data.action_slot
        channelbag = cast(Any, action.layers[0].strips[0]).channelbag(slot)
        fcurves = channelbag.fcurves
    for fcurve in fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"

    scene.camera = cam_obj
    logger.info(f"Created orbit camera with {frames} frames at {elevation_deg} deg elevation")


def _apply_render_visibility(
    stem: str,
    groups: dict[str, list[bpy.types.Object]],
    show_groups: dict[str, str],
    solo_object: bpy.types.Object | None = None,
) -> None:
    """Configure collection and object visibility for a single render pass.

    :param stem: Scene stem used for collection name mapping.
    :param groups: Label→objects dict from import_object_groups.
    :param show_groups: Label→visibility mode mapping (e.g. {"mesh": "both", "inputs": "both"}).
    :param solo_object: If set, hide all other objects in its parent collection.
    """
    name_map = {
        "inputs": f"{stem}_inputs",
        "mesh": f"{stem}_meshes",
        "pcd": f"{stem}_pointclouds",
        "occ": f"{stem}_occupied_points",
        "free": f"{stem}_free_points",
        "logits": f"{stem}_logits",
        "pred": f"{stem}_pred_meshes",
    }

    # Hide all geometry collections first
    for label, coll_name in name_map.items():
        coll = bpy.data.collections.get(coll_name)
        if coll is not None:
            _set_collection_visibility(coll, None)
            # Restore all objects to visible within their collection
            for obj in groups.get(label, []):
                try:
                    _set_visibility(obj, "both")
                except Exception:
                    pass

    # Show requested collections
    for label, mode in show_groups.items():
        coll_name = name_map.get(label)
        if not coll_name:
            continue
        coll = bpy.data.collections.get(coll_name)
        if coll is not None:
            _set_collection_visibility(coll, cast(Any, mode))

    # Solo mode: within the target's collection, hide everything except the target
    if solo_object is not None:
        for _label, objs in groups.items():
            if solo_object in objs:
                for obj in objs:
                    if obj is solo_object:
                        _set_visibility(obj, "both")
                    else:
                        _set_visibility(obj, None)
                break


def _objects_bounding_sphere(
    objects: list[bpy.types.Object],
) -> tuple[mathutils.Vector, float]:
    """Compute (center, radius) bounding sphere for a list of objects."""
    stats = _collect_object_bounds(objects)
    return _robust_bounds_from_stats(stats)


def _objects_aabb(
    objects: Iterable[bpy.types.Object],
) -> tuple[mathutils.Vector, mathutils.Vector, mathutils.Vector]:
    """Compute (center, min, max) axis-aligned bounding box for visible objects.

    No outlier filtering is applied here — callers are expected to pass a
    curated set of objects.  Use :func:`_robust_bounds_from_stats` (via
    :func:`_scene_content_bounds`) when operating on uncurated object sets.
    """
    stats = _collect_object_bounds(objects)
    if not stats:
        z = mathutils.Vector((0.0, 0.0, 0.0))
        return z, z.copy(), z.copy()
    all_mins = [b.min for b in stats]
    all_maxs = [b.max for b in stats]
    bmin = mathutils.Vector((min(v.x for v in all_mins), min(v.y for v in all_mins), min(v.z for v in all_mins)))
    bmax = mathutils.Vector((max(v.x for v in all_maxs), max(v.y for v in all_maxs), max(v.z for v in all_maxs)))
    return (bmin + bmax) * 0.5, bmin, bmax


def _reframe_camera(
    cam_obj: bpy.types.Object,
    center: mathutils.Vector,
    bounds_min: mathutils.Vector,
    bounds_max: mathutils.Vector,
    fill_factor: float = 0.95,
) -> None:
    """Reposition camera to tightly frame an AABB, preserving viewing angle.

    Projects all 8 AABB corners into camera space and computes the minimum
    distance along the current viewing direction that fits the content within
    *fill_factor* of the frame.  This avoids the over-conservative framing
    that a view-independent bounding sphere produces for elongated or flat
    scenes viewed at an angle.
    """
    # 8 AABB corners.
    corners = [
        mathutils.Vector((x, y, z))
        for x in (bounds_min.x, bounds_max.x)
        for y in (bounds_min.y, bounds_max.y)
        for z in (bounds_min.z, bounds_max.z)
    ]
    _reframe_camera_to_corners(cam_obj, center, corners, fill_factor=fill_factor)


def _reframe_camera_to_corners(
    cam_obj: bpy.types.Object,
    center: mathutils.Vector,
    corners: Sequence[mathutils.Vector],
    fill_factor: float = 0.95,
) -> None:
    """Reposition camera to frame an arbitrary world-space point cloud."""
    if not corners:
        return
    center = center.copy()
    cam = cast(Any, cam_obj.data)

    # Asymmetric frustum tangents, accounting for principal-point shifts and
    # render aspect ratio.  With sensor_fit='HORIZONTAL' the horizontal FOV
    # is fixed by sensor_width / lens; vertical scales by res_y / res_x.
    sw_over_lens = cam.sensor_width / cam.lens
    res_x, res_y = _effective_render_resolution(_require_scene())
    aspect = float(res_y) / max(float(res_x), 1e-6)
    shift_x = float(getattr(cam, "shift_x", 0.0))
    shift_y = float(getattr(cam, "shift_y", 0.0))
    # Keep a tiny pixel guard to avoid 1px edge clipping from rasterization.
    guard_px = 1.5
    guard_x = max(0.0, 1.0 - (2.0 * guard_px) / max(float(res_x), 1.0))
    guard_y = max(0.0, 1.0 - (2.0 * guard_px) / max(float(res_y), 1.0))

    # Per-edge fill tangents from optical axis to fill boundary.
    # Positive shift_x moves the principal point right in the image, which
    # extends the LEFT edge of the frustum and shrinks the RIGHT edge.
    fill_right = max(fill_factor * guard_x * (0.5 - shift_x) * sw_over_lens, 1e-6)
    fill_left = max(fill_factor * guard_x * (0.5 + shift_x) * sw_over_lens, 1e-6)
    fill_up = max(fill_factor * guard_y * (0.5 - shift_y) * sw_over_lens * aspect, 1e-6)
    fill_down = max(fill_factor * guard_y * (0.5 + shift_y) * sw_over_lens * aspect, 1e-6)

    # Viewing direction: from center toward current camera position.
    direction = cam_obj.location - center
    if direction.length < 1e-8:
        direction = mathutils.Vector((0.0, -1.0, 0.5))
    view_dir = direction.normalized()

    # Derive the actual camera-local right/up axes by computing the same
    # orientation that _aim_object_at will produce.  to_track_quat('-Z', 'Y')
    # aligns -Z with the aim direction and keeps the camera's local Y close
    # to world +Z (Blender's up convention).
    aim_dir = -view_dir
    quat = aim_dir.to_track_quat("-Z", "Y")
    rot_matrix = quat.to_matrix()
    right = (rot_matrix @ mathutils.Vector((1.0, 0.0, 0.0))).normalized()
    up = (rot_matrix @ mathutils.Vector((0.0, 1.0, 0.0))).normalized()

    def _solve_distance(local_center: mathutils.Vector) -> float:
        # For each corner Q, camera at P = center + D * view_dir:
        #   depth   = D - (Q - center) · view_dir        [positive = in front]
        #   lat_x   = (Q - center) · right               [signed: + = right]
        #   lat_y   = (Q - center) · up                   [signed: + = up]
        # Asymmetric constraint: lat_x must fit within fill_right (if positive)
        # or fill_left (if negative), and likewise for lat_y / fill_up / fill_down.
        distance = 0.0
        max_d_along = -float("inf")
        min_d_along = float("inf")
        for corner in corners:
            offset = corner - local_center
            d_along = offset.dot(view_dir)
            lat_x = offset.dot(right)
            lat_y = offset.dot(up)
            req_x = lat_x / fill_right if lat_x >= 0.0 else -lat_x / fill_left
            req_y = lat_y / fill_up if lat_y >= 0.0 else -lat_y / fill_down
            required = d_along + max(req_x, req_y)
            distance = max(distance, required)
            max_d_along = max(max_d_along, d_along)
            min_d_along = min(min_d_along, d_along)

        # Ensure all corners are in front of the camera.
        depth_span = max(max_d_along - min_d_along, 1e-6)
        front_margin = max(1e-6, 1e-4 * depth_span)
        return max(distance, max_d_along + front_margin)

    min_distance = _solve_distance(center)

    # Recenter by projected screen-space min/max (perspective-aware), then
    # recompute distance. Two iterations are enough for stable convergence.
    for _ in range(2):
        lateral_x: list[float] = []
        lateral_y: list[float] = []
        depths: list[float] = []
        for corner in corners:
            offset = corner - center
            d_along = offset.dot(view_dir)
            depths.append(max(min_distance - d_along, 1e-6))
            lateral_x.append(float(offset.dot(right)))
            lateral_y.append(float(offset.dot(up)))
        shift_x_world = _solve_projected_mid_shift(lateral_x, depths)
        shift_y_world = _solve_projected_mid_shift(lateral_y, depths)
        if abs(shift_x_world) + abs(shift_y_world) < 1e-7:
            break
        center += right * shift_x_world + up * shift_y_world
        min_distance = _solve_distance(center)

    # Ensure near clipping does not cut into tightly framed foreground.
    if hasattr(cam, "clip_start"):
        nearest_depth = min(max(min_distance - (c - center).dot(view_dir), 1e-8) for c in corners)
        required_clip_start = min(max(1e-6, 0.5 * nearest_depth), 0.99 * nearest_depth)
        try:
            if float(cam.clip_start) > required_clip_start:
                cam.clip_start = required_clip_start
        except Exception:
            pass

    # Ensure far clipping covers the full scene depth.
    max_scene_depth = min_distance + max(abs((c - center).dot(view_dir)) for c in corners)
    if hasattr(cam, "clip_end"):
        required_clip_end = max_scene_depth + 1.0
        if cam.clip_end < required_clip_end:
            cam.clip_end = required_clip_end

    cam_obj.location = center + view_dir * min_distance
    _aim_object_at(cam_obj, center)


def _iter_point_objects(
    groups: dict[str, list[bpy.types.Object]],
    labels: Iterable[str] | None = None,
) -> list[bpy.types.Object]:
    point_labels = {"inputs", "pcd", "occ", "free", "logits"}
    selected = point_labels if labels is None else {label for label in labels if label in point_labels}
    objs: list[bpy.types.Object] = []
    for label in selected:
        for obj in groups.get(label, []):
            if not _is_placeholder(obj):
                objs.append(obj)
    return objs


def _point_radius_socket(obj: bpy.types.Object):
    for modifier in getattr(obj, "modifiers", []) or []:
        try:
            if modifier.type != "NODES" or modifier.node_group is None:
                continue
            for node in modifier.node_group.nodes:
                if node.bl_idname == "GeometryNodeMeshToPoints" and "Radius" in node.inputs:
                    return node.inputs["Radius"]
        except Exception:
            continue
    return None


def _point_keep_socket(obj: bpy.types.Object):
    # Prefer explicitly labeled keep-compare node.
    for modifier in getattr(obj, "modifiers", []) or []:
        try:
            if modifier.type != "NODES" or modifier.node_group is None:
                continue
            for node in modifier.node_group.nodes:
                if node.bl_idname != "FunctionNodeCompare":
                    continue
                label = str(getattr(node, "label", "") or "")
                name = str(getattr(node, "name", "") or "")
                if label == "PCD_KEEP_COMPARE" or name.startswith("PCD_KEEP_COMPARE"):
                    if len(node.inputs) > 1:
                        return node.inputs[1]
        except Exception:
            continue

    # Backward-compatible fallback: detect compare node driving a POINT delete node.
    for modifier in getattr(obj, "modifiers", []) or []:
        try:
            if modifier.type != "NODES" or modifier.node_group is None:
                continue
            nodes = modifier.node_group.nodes
            for node in nodes:
                if node.bl_idname != "GeometryNodeDeleteGeometry":
                    continue
                if str(getattr(node, "domain", "") or "") not in {"", "POINT"}:
                    continue
                try:
                    selection_input = node.inputs["Selection"]
                except Exception:
                    continue
                for link in getattr(selection_input, "links", []) or []:
                    from_node = getattr(link, "from_node", None)
                    if from_node is None or from_node.bl_idname != "FunctionNodeCompare":
                        continue
                    if hasattr(from_node, "operation") and str(from_node.operation) != "GREATER_THAN":
                        continue
                    if len(from_node.inputs) > 1:
                        return from_node.inputs[1]
        except Exception:
            continue
    return None


def _ensure_point_keep_socket(obj: bpy.types.Object):
    socket = _point_keep_socket(obj)
    if socket is not None:
        return socket

    for modifier in getattr(obj, "modifiers", []) or []:
        try:
            if modifier.type != "NODES" or modifier.node_group is None:
                continue
            node_group = modifier.node_group
            nodes = node_group.nodes
            links = node_group.links

            mesh_to_points = None
            for node in nodes:
                if node.bl_idname == "GeometryNodeMeshToPoints" and "Mesh" in node.inputs:
                    mesh_to_points = node
                    break
            if mesh_to_points is None:
                continue

            mesh_input = mesh_to_points.inputs["Mesh"]
            upstream_socket = None
            existing_links = list(getattr(mesh_input, "links", []) or [])
            if existing_links:
                upstream_socket = existing_links[0].from_socket
                for link in existing_links:
                    try:
                        links.remove(link)
                    except Exception:
                        pass
            else:
                for node in nodes:
                    if node.bl_idname == "NodeGroupInput" and "Geometry" in node.outputs:
                        upstream_socket = node.outputs["Geometry"]
                        break
            if upstream_socket is None:
                continue

            delete_node = nodes.new(type="GeometryNodeDeleteGeometry")
            delete_node.domain = "POINT"
            rand_node = nodes.new(type="FunctionNodeRandomValue")
            rand_node.data_type = "FLOAT"
            rand_node.inputs["Min"].default_value = 0.0
            rand_node.inputs["Max"].default_value = 1.0
            compare_node = nodes.new(type="FunctionNodeCompare")
            compare_node.data_type = "FLOAT"
            compare_node.operation = "GREATER_THAN"
            compare_node.label = "PCD_KEEP_COMPARE"
            compare_node.name = "PCD_KEEP_COMPARE"
            compare_node.inputs[1].default_value = 1.0

            links.new(upstream_socket, delete_node.inputs["Geometry"])
            links.new(rand_node.outputs["Value"], compare_node.inputs[0])
            links.new(compare_node.outputs["Result"], delete_node.inputs["Selection"])
            links.new(delete_node.outputs["Geometry"], mesh_input)
            return compare_node.inputs[1]
        except Exception:
            continue
    return None


def _current_point_radius(obj: bpy.types.Object) -> float | None:
    socket = _point_radius_socket(obj)
    if socket is None:
        return None
    try:
        return float(socket.default_value)
    except Exception:
        return None


def _set_point_radius(obj: bpy.types.Object, radius: float) -> bool:
    socket = _point_radius_socket(obj)
    if socket is None:
        return False
    try:
        socket.default_value = float(radius)
        return True
    except Exception:
        return False


def _current_point_keep_ratio(obj: bpy.types.Object) -> float | None:
    socket = _point_keep_socket(obj)
    if socket is None:
        return None
    try:
        return _clamp_float(float(socket.default_value), 0.0, 1.0)
    except Exception:
        return None


def _set_point_keep_ratio(obj: bpy.types.Object, keep_ratio: float) -> bool:
    socket = _ensure_point_keep_socket(obj)
    if socket is None:
        return False
    try:
        socket.default_value = _clamp_float(float(keep_ratio), 0.0, 1.0)
        return True
    except Exception:
        return False


def _base_point_radius(obj: bpy.types.Object) -> float | None:
    base = _object_prop_float(obj, "_pcd_effective_radius")
    if base is not None and base > 0.0:
        return float(base)
    current = _current_point_radius(obj)
    if current is not None and current > 0.0:
        return float(current)
    return None


def _base_point_keep_ratio(obj: bpy.types.Object) -> float | None:
    base = _object_prop_float(obj, "_pcd_runtime_keep_ratio")
    if base is not None:
        return _clamp_float(float(base), 0.0, 1.0)
    current = _current_point_keep_ratio(obj)
    if current is not None:
        return _clamp_float(float(current), 0.0, 1.0)
    return None


def _compensated_keep_ratio_from_radius(
    base_keep_ratio: float,
    base_radius: float,
    active_radius: float,
) -> float:
    """Keep projected point coverage approximately constant.

    For screen-space discs, coverage scales roughly with N * r^2. If radius is
    increased by camera floor constraints, reduce keep ratio by (r0/r)^2.
    """
    base_keep = _clamp_float(float(base_keep_ratio), 0.0, 1.0)
    if base_keep <= 0.0:
        return 0.0
    r0 = max(float(base_radius), 1e-9)
    r = max(float(active_radius), 1e-9)
    scaled = base_keep * (r0 / r) ** 2
    return _clamp_float(scaled, 0.0, base_keep)


def _effective_render_resolution(scene: bpy.types.Scene) -> tuple[int, int]:
    scale = max(float(getattr(scene.render, "resolution_percentage", 100.0)) / 100.0, 1e-6)
    res_x = max(1, round(float(scene.render.resolution_x) * scale))
    res_y = max(1, round(float(scene.render.resolution_y) * scale))
    return res_x, res_y


def _point_radius_probe_points_world(obj: bpy.types.Object) -> list[mathutils.Vector]:
    """Return probe points used to estimate required point radius for a camera.

    Using only the cloud center can underestimate required radius at image
    corners where depth is larger. Probe with center + world-space bbox corners.
    """
    probes: list[mathutils.Vector] = []
    center: mathutils.Vector | None = None
    try:
        center, _ = _object_bounds_world(obj)
    except Exception:
        center = None

    if center is not None:
        probes.append(center)

    try:
        bound_box = getattr(obj, "bound_box", None)
        matrix_world = getattr(obj, "matrix_world", None)
        if bound_box:
            for corner in bound_box:
                v = mathutils.Vector(corner)
                if matrix_world is not None:
                    v = matrix_world @ v
                probes.append(v)
    except Exception:
        pass

    if not probes:
        probes.append(mathutils.Vector((0.0, 0.0, 0.0)))
    return probes


def _required_point_radius_for_camera(
    obj: bpy.types.Object,
    cam_obj: bpy.types.Object,
    min_px: float,
    scene: bpy.types.Scene,
) -> float:
    probes = _point_radius_probe_points_world(obj)
    required = 0.0
    for point_world in probes:
        floor_r = _min_world_radius_for_pixels(cam_obj, point_world, min_px=min_px, scene=scene)
        if floor_r > required:
            required = floor_r
    return required


def _min_world_radius_for_pixels(
    cam_obj: bpy.types.Object,
    point_world: mathutils.Vector,
    min_px: float,
    scene: bpy.types.Scene,
) -> float:
    cam = cast(Any, cam_obj.data)
    res_x, res_y = _effective_render_resolution(scene)

    if getattr(cam, "type", None) == "ORTHO":
        ortho_scale = max(float(getattr(cam, "ortho_scale", 1.0)), 1e-6)
        view_w = ortho_scale
        view_h = ortho_scale * (float(res_y) / max(float(res_x), 1e-6))
        px_per_meter = min(float(res_x) / max(view_w, 1e-6), float(res_y) / max(view_h, 1e-6))
        return float(min_px) / max(px_per_meter, 1e-6)

    # Default perspective case.
    depth = 1.0
    try:
        p_cam = cam_obj.matrix_world.inverted() @ point_world
        depth = max(-float(p_cam.z), 1e-6)
    except Exception:
        depth = max((cam_obj.location - point_world).length, 1e-6)

    focals: list[float] = []
    try:
        angle_x = float(cam.angle_x)
        if angle_x > 1e-6:
            focals.append(0.5 * float(res_x) / math.tan(angle_x * 0.5))
    except Exception:
        pass
    try:
        angle_y = float(cam.angle_y)
        if angle_y > 1e-6:
            focals.append(0.5 * float(res_y) / math.tan(angle_y * 0.5))
    except Exception:
        pass
    if not focals:
        lens = max(float(getattr(cam, "lens", 50.0)), 1e-6)
        sensor_w = max(float(getattr(cam, "sensor_width", 36.0)), 1e-6)
        sensor_h = max(float(getattr(cam, "sensor_height", 24.0)), 1e-6)
        focals = [lens / sensor_w * float(res_x), lens / sensor_h * float(res_y)]

    focal_px = max(min(focals), 1e-6)
    return float(min_px) * depth / focal_px


def _apply_point_radius_floor_for_camera(
    groups: dict[str, list[bpy.types.Object]],
    cam_obj: bpy.types.Object | None,
    min_px: float,
    labels: Iterable[str] | None = None,
) -> None:
    if cam_obj is None or min_px <= 0.0:
        return
    scene = cast(bpy.types.Scene, _require_scene())
    for obj in _iter_point_objects(groups, labels):
        base_radius = _base_point_radius(obj)
        if base_radius is None:
            continue
        min_radius = _required_point_radius_for_camera(obj, cam_obj, min_px=min_px, scene=scene)
        active_radius = max(base_radius, min_radius)
        _set_point_radius(obj, active_radius)

        base_keep = _base_point_keep_ratio(obj)
        if base_keep is None:
            continue
        keep_ratio = _compensated_keep_ratio_from_radius(base_keep, base_radius, active_radius)
        _set_point_keep_ratio(obj, keep_ratio)


def _restore_point_radii(groups: dict[str, list[bpy.types.Object]]) -> None:
    for obj in _iter_point_objects(groups):
        base_radius = _base_point_radius(obj)
        if base_radius is None:
            continue
        _set_point_radius(obj, base_radius)
        base_keep = _base_point_keep_ratio(obj)
        if base_keep is not None:
            _set_point_keep_ratio(obj, base_keep)


def _resolve_object_camera(obj: bpy.types.Object) -> bpy.types.Object | None:
    cam_name = _object_prop_str(obj, "_solo_camera")
    if cam_name:
        cam = bpy.data.objects.get(cam_name)
        if cam is not None:
            return cam
    return bpy.data.objects.get(f"Camera_{obj.name}")


def _apply_point_radius_floor_for_orbit(
    groups: dict[str, list[bpy.types.Object]],
    cam_obj: bpy.types.Object | None,
    min_px: float,
    labels: Iterable[str] | None = None,
) -> None:
    if cam_obj is None or min_px <= 0.0:
        return

    scene = _require_scene()
    objs = _iter_point_objects(groups, labels)
    if not objs:
        return

    saved_frame = int(scene.frame_current)
    max_floor: dict[str, float] = {obj.name: 0.0 for obj in objs}
    try:
        for frame in range(int(scene.frame_start), int(scene.frame_end) + 1):
            scene.frame_set(frame)
            for obj in objs:
                floor_r = _required_point_radius_for_camera(obj, cam_obj, min_px=min_px, scene=scene)
                if floor_r > max_floor[obj.name]:
                    max_floor[obj.name] = floor_r
    finally:
        scene.frame_set(saved_frame)

    for obj in objs:
        base_radius = _base_point_radius(obj)
        if base_radius is None:
            continue
        active_radius = max(base_radius, max_floor.get(obj.name, 0.0))
        _set_point_radius(obj, active_radius)
        base_keep = _base_point_keep_ratio(obj)
        if base_keep is None:
            continue
        keep_ratio = _compensated_keep_ratio_from_radius(base_keep, base_radius, active_radius)
        _set_point_keep_ratio(obj, keep_ratio)


def _save_camera_transforms(cameras: Iterable[bpy.types.Object]) -> dict[str, dict[str, Any]]:
    """Snapshot location, rotation, and scale for a set of cameras."""
    saved: dict[str, dict[str, Any]] = {}
    for cam in cameras:
        saved[cam.name] = {
            "location": cam.location.copy(),
            "rotation_mode": cam.rotation_mode,
            "rotation_euler": cam.rotation_euler.copy(),
            "rotation_quaternion": cam.rotation_quaternion.copy(),
            "rotation_axis_angle": tuple(cam.rotation_axis_angle),
            "scale": cam.scale.copy(),
        }
    return saved


def _restore_camera_transforms(cameras: Iterable[bpy.types.Object], saved: dict[str, dict[str, Any]]) -> None:
    """Restore cameras to previously saved transforms."""
    for cam in cameras:
        snap = saved.get(cam.name)
        if snap is None:
            continue
        cam.location = cast(Any, snap["location"]).copy()
        cam.scale = cast(Any, snap["scale"]).copy()
        cast(Any, cam).rotation_mode = snap["rotation_mode"]
        if cam.rotation_mode == "QUATERNION":
            cam.rotation_quaternion = cast(Any, snap["rotation_quaternion"]).copy()
        elif cam.rotation_mode == "AXIS_ANGLE":
            cam.rotation_axis_angle = cast(Any, snap["rotation_axis_angle"])
        else:
            cam.rotation_euler = cast(Any, snap["rotation_euler"]).copy()


def batch_render(
    stem: str,
    groups: dict[str, list[bpy.types.Object]],
    output_dir: Path,
    point_min_px: float = 1.0,
    render_orbit: bool = False,
    composite_resolution: int = 1024,
    solo_resolution: int = 512,
    config: LoaderConfig | None = None,
) -> None:
    """Render all cameras x visibility passes to disk.

    Composite passes (gt_inputs, pred) are rendered from canonical +
    reference cameras, reframed to fill the viewport.  Solo object passes are
    rendered from per-object cameras.  Output is a flat directory of PNGs.
    Point radii are floored per active camera to keep points visible at
    approximately point_min_px projected pixels. When lighting is enabled in
    config, a persistent light rig is updated per active camera while the
    scene geometry (including shadow catcher) stays static.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = _require_scene()

    # Save state for restoration
    orig_res_x = scene.render.resolution_x
    orig_res_y = scene.render.resolution_y

    # --- Discover cameras ---
    canonical_cams = sorted(
        [obj for obj in bpy.data.objects if obj.type == "CAMERA" and obj.name.startswith(f"{stem}_canonical_")],
        key=lambda c: c.name,
    )
    reference_cam = bpy.data.objects.get(f"{stem}_camera")
    reframed_reference_cam = bpy.data.objects.get(f"{stem}_camera_reframed")
    if reframed_reference_cam is None:
        reframed_reference_cam = _ensure_reframed_reference_camera(
            stem, groups, composite_resolution=composite_resolution, fill_factor=0.90
        )
    still_cameras: list[bpy.types.Object] = []
    if reference_cam is not None:
        still_cameras.append(reference_cam)
    if reframed_reference_cam is not None and reframed_reference_cam is not reference_cam:
        still_cameras.append(reframed_reference_cam)
    still_cameras.extend(canonical_cams)

    if not still_cameras:
        logger.warning("batch_render: no cameras found, skipping")
        return

    saved_transforms = _save_camera_transforms(still_cameras)

    objects_by_name: dict[str, bpy.types.Object] = {
        obj.name: obj for objs in groups.values() for obj in objs if obj is not None
    }
    # Solo object renders can expose hotter than scene composites; keep this
    # separately configurable so users can tune per-pass balance.
    solo_light_scale_multiplier = float(config.lighting_solo_scale) if config is not None else 0.9
    solo_fill_ratio_multiplier = float(config.lighting_solo_fill_ratio_scale) if config is not None else 1.6

    def _solo_lighting_anchor(obj: bpy.types.Object) -> bpy.types.Object:
        anchor_name = _object_prop_str(obj, "_solo_light_anchor")
        if anchor_name:
            anchor = objects_by_name.get(anchor_name) or bpy.data.objects.get(anchor_name)
            if anchor is not None and not _is_placeholder(anchor):
                return anchor
        return obj

    def _update_dynamic_lighting(
        cam_obj: bpy.types.Object | None,
        subject_obj: bpy.types.Object | None,
        subject_center: mathutils.Vector | None = None,
        subject_radius: float | None = None,
        subject_min_z: float | None = None,
        light_scale_multiplier: float = 1.0,
        fill_ratio_multiplier: float = 1.0,
        ambient_scale_multiplier: float = 1.0,
        rebuild_rig: bool = False,
    ) -> None:
        if config is None or not config.enable_lighting:
            return
        try:
            scaled_light = float(config.lighting_scale) * max(float(light_scale_multiplier), 0.0)
            scaled_fill_ratio = float(config.lighting_fill_ratio) * max(float(fill_ratio_multiplier), 0.0)
            scaled_ambient = float(config.lighting_world_ambient_scale) * max(float(ambient_scale_multiplier), 0.0)
            add_three_point_rig(
                cam_obj,
                subject_obj,
                subject_center=subject_center,
                subject_radius=subject_radius,
                subject_min_z=subject_min_z,
                hdr_strength=config.lighting_hdr_strength,
                world_ambient_scale=scaled_ambient,
                samples=config.cycles_samples,
                light_scale=scaled_light,
                distance_scale=config.lighting_distance_scale,
                fill_distance_ratio=config.lighting_fill_distance_ratio,
                rim_distance_ratio=config.lighting_rim_distance_ratio,
                key_base_energy=config.lighting_key_energy_base,
                fill_energy_ratio=scaled_fill_ratio,
                rim_energy_ratio=config.lighting_rim_ratio,
                kicker_energy_ratio=config.lighting_kicker_ratio,
                key_strength=config.lighting_key_strength,
                fill_strength=config.lighting_fill_strength,
                rim_strength=config.lighting_rim_strength,
                kicker_strength=config.lighting_kicker_strength,
                shadow_catcher=config.shadow_catcher,
                ao_strength=config.ao_strength,
                ao_distance=config.ao_distance,
                compositor_preset=config.compositor_preset,
                cycles_denoise=config.cycles_denoise,
                cycles_light_threshold=config.cycles_light_threshold,
                cycles_preview_factor=config.cycles_preview_factor,
                cycles_adaptive_threshold=config.cycles_adaptive_threshold,
                cycles_clamp_direct=config.cycles_clamp_direct,
                cycles_clamp_indirect=config.cycles_clamp_indirect,
                cycles_blur_glossy=config.cycles_blur_glossy,
                cycles_max_bounces=config.cycles_max_bounces,
                cycles_diffuse_bounces=config.cycles_diffuse_bounces,
                cycles_glossy_bounces=config.cycles_glossy_bounces,
                cycles_transmission_bounces=config.cycles_transmission_bounces,
                cycles_transparent_max_bounces=config.cycles_transparent_max_bounces,
                cycles_caustics_reflective=config.cycles_caustics_reflective,
                cycles_caustics_refractive=config.cycles_caustics_refractive,
                rebuild_rig=rebuild_rig,
            )
        except Exception as exc:
            cam_name = cam_obj.name if cam_obj is not None else "None"
            logger.warning(f"Dynamic lighting update skipped for {cam_name}: {exc}")

    # --- Define render passes ---
    has_pred = "pred" in groups and groups["pred"]

    composite_passes: list[tuple[str, dict[str, str]]] = [
        ("gt_inputs", {"mesh": "both", "inputs": "both"}),
    ]
    if has_pred:
        composite_passes.append(("pred", {"pred": "both"}))
        composite_passes.append(("pred_inputs", {"pred": "both", "inputs": "both"}))

    solo_passes: list[tuple[str, dict[str, str], bpy.types.Object, bpy.types.Object]] = []
    for obj in groups.get("mesh", []):
        obj_cam = _resolve_object_camera(obj)
        if obj_cam is not None:
            solo_passes.append((f"gt_{obj.name}", {"mesh": "both"}, obj, obj_cam))
    if has_pred:
        for obj in groups.get("pred", []):
            obj_cam = _resolve_object_camera(obj)
            if obj_cam is not None:
                solo_passes.append((f"pred_{obj.name}", {"pred": "both"}, obj, obj_cam))

    total = len(still_cameras) * len(composite_passes) + len(solo_passes)
    rendered = 0

    # --- Composite passes: reframed cameras ---
    scene.render.resolution_x = composite_resolution
    scene.render.resolution_y = composite_resolution

    def _composite_resolution_for_camera(cam_obj: bpy.types.Object) -> tuple[int, int]:
        """Return per-camera composite resolution.

        Keep the JSON source camera at native resolution; render synthetic
        reframed/canonical cameras at square composite resolution.
        """
        if reference_cam is not None and cam_obj is reference_cam:
            src_w = _object_prop_float(cam_obj, "_source_width")
            src_h = _object_prop_float(cam_obj, "_source_height")
            if src_w is not None and src_h is not None and src_w > 0.0 and src_h > 0.0:
                return round(src_w), round(src_h)
        return int(composite_resolution), int(composite_resolution)

    # Frame from GT meshes only so gt_inputs and pred use identical,
    # GT-derived camera placement for direct side-by-side comparison.
    framing_objs = [obj for obj in groups.get("mesh", []) if not _is_placeholder(obj)]
    if not framing_objs:
        # Prefer inputs (pred-invariant), then pred, then any visible objects.
        for fallback_label in ("inputs", "pred"):
            framing_objs = [obj for obj in groups.get(fallback_label, []) if not _is_placeholder(obj)]
            if framing_objs:
                break
        if not framing_objs:
            for _, vis in composite_passes:
                for label in vis:
                    framing_objs.extend(groups.get(label, []))
        logger.warning(f"batch_render: no GT meshes found for framing; falling back to {len(framing_objs)} objects")
    center, framing_corners = _collect_frame_point_cloud(framing_objs, max_mesh_points_per_object=2000)
    if not framing_corners:
        center, bounds_min, bounds_max = _objects_aabb(framing_objs)
        framing_corners = [
            mathutils.Vector((x, y, z))
            for x in (bounds_min.x, bounds_max.x)
            for y in (bounds_min.y, bounds_max.y)
            for z in (bounds_min.z, bounds_max.z)
        ]

    # Scene/canonical lighting must stay GT-anchored so gt/pred comparisons
    # use identical framing and illumination basis.
    scene_light_obj, scene_light_center, scene_light_radius, scene_light_min_z = _lighting_subject_bounds(
        {"mesh": [obj for obj in groups.get("mesh", []) if not _is_placeholder(obj)]}
    )
    if scene_light_obj is None:
        scene_light_obj, scene_light_center, scene_light_radius, scene_light_min_z = _lighting_subject_bounds(groups)

    for cam in still_cameras:
        # Keep the JSON camera untouched; render it as the true source view.
        if reference_cam is not None and cam is reference_cam:
            continue
        _reframe_camera_to_corners(cam, center, framing_corners, fill_factor=0.90)

    # Render camera-first so paired gt/pred passes share exactly the same light rig
    # and avoid any per-pass relighting drift.
    for cam in still_cameras:
        cam_res_x, cam_res_y = _composite_resolution_for_camera(cam)
        scene.render.resolution_x = cam_res_x
        scene.render.resolution_y = cam_res_y
        scene.camera = cam
        _update_dynamic_lighting(
            cam,
            scene_light_obj,
            subject_center=scene_light_center,
            subject_radius=scene_light_radius,
            subject_min_z=scene_light_min_z,
            light_scale_multiplier=float(config.lighting_scene_scale) if config is not None else 0.9,
            fill_ratio_multiplier=float(config.lighting_scene_fill_ratio_scale) if config is not None else 0.75,
            ambient_scale_multiplier=float(config.lighting_scene_ambient_scale) if config is not None else 1.0,
        )

        for pass_name, vis in composite_passes:
            rendered += 1
            _apply_render_visibility(stem, groups, vis)
            _apply_point_radius_floor_for_camera(groups, cam, point_min_px, labels=vis.keys())
            out_path = output_dir / f"{pass_name}_{cam.name}.png"
            logger.info(f"Rendering [{rendered}/{total}] {pass_name}_{cam.name}")

            scene.render.filepath = str(out_path)
            scene.render.image_settings.file_format = "PNG"
            bpy.ops.render.render(write_still=True)

    _restore_camera_transforms(still_cameras, saved_transforms)

    # --- Solo passes: per-object cameras ---
    scene.render.resolution_x = solo_resolution
    scene.render.resolution_y = solo_resolution

    for pass_name, vis, solo_obj, obj_cam in solo_passes:
        rendered += 1
        scene.camera = obj_cam
        light_anchor = _solo_lighting_anchor(solo_obj)
        _update_dynamic_lighting(
            obj_cam,
            light_anchor,
            light_scale_multiplier=solo_light_scale_multiplier,
            fill_ratio_multiplier=solo_fill_ratio_multiplier,
        )
        out_path = output_dir / f"{pass_name}.png"
        logger.info(f"Rendering [{rendered}/{total}] {pass_name}")

        _apply_render_visibility(stem, groups, vis, solo_object=solo_obj)
        _apply_point_radius_floor_for_camera(groups, obj_cam, point_min_px, labels=vis.keys())
        scene.render.filepath = str(out_path)
        scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(write_still=True)

    # Solo passes hide non-target objects per collection; restore object-level
    # visibility before any subsequent pass.
    for objs in groups.values():
        for obj in objs:
            try:
                _set_visibility(obj, "both")
            except Exception:
                pass

    # --- Orbit animation passes ---
    orbit_cam = bpy.data.objects.get(f"{stem}_orbit_camera")
    if render_orbit and orbit_cam is not None:
        scene.render.resolution_x = composite_resolution
        scene.render.resolution_y = composite_resolution
        scene.camera = orbit_cam

        orbit_passes: list[tuple[str, dict[str, str]]] = [
            ("gt_inputs", {"mesh": "both", "inputs": "both"}),
        ]
        if has_pred:
            orbit_passes.append(("pred", {"pred": "both"}))

        _update_dynamic_lighting(
            orbit_cam,
            scene_light_obj,
            subject_center=scene_light_center,
            subject_radius=scene_light_radius,
            subject_min_z=scene_light_min_z,
            light_scale_multiplier=float(config.lighting_scene_scale) if config is not None else 0.9,
            fill_ratio_multiplier=float(config.lighting_scene_fill_ratio_scale) if config is not None else 0.75,
            ambient_scale_multiplier=float(config.lighting_scene_ambient_scale) if config is not None else 1.0,
        )

        for pass_name, vis in orbit_passes:
            logger.info(f"Rendering orbit animation: {pass_name} ({scene.frame_end} frames)")
            _apply_render_visibility(stem, groups, vis)
            _apply_point_radius_floor_for_orbit(groups, orbit_cam, point_min_px, labels=vis.keys())
            scene.render.filepath = str(output_dir / f"orbit_{pass_name}_")
            scene.render.image_settings.file_format = "PNG"
            bpy.ops.render.render(animation=True)

    # --- Restore ---
    scene.render.resolution_x = orig_res_x
    scene.render.resolution_y = orig_res_y
    _restore_camera_transforms(still_cameras, saved_transforms)
    _restore_point_radii(groups)
    set_default_group_visibility(stem, groups)
    logger.info(f"Batch render complete: {output_dir}")


def choose_subject(groups: dict[str, list[bpy.types.Object]]) -> bpy.types.Object | None:
    subject, _center, _radius, _min_z = _lighting_subject_bounds(groups)
    return subject


def _lighting_subject_bounds(
    groups: dict[str, list[bpy.types.Object]],
) -> tuple[bpy.types.Object | None, mathutils.Vector | None, float | None, float | None]:
    """Pick lighting subjects and return (anchor, center, radius, min_z).

    For qualitative GT/pred comparisons, prefer mesh + pred objects and ignore
    inputs/auxiliary clouds for the lighting anchor.
    """
    preferred_keys = ("mesh", "pred")
    fallback_keys = ("pcd", "occ", "inputs", "free", "logits")

    candidates = [obj for key in preferred_keys for obj in groups.get(key, []) if not _is_placeholder(obj)]
    if not candidates:
        for key in fallback_keys:
            objs = [obj for obj in groups.get(key, []) if not _is_placeholder(obj)]
            if objs:
                candidates = objs
                break

    if not candidates:
        return None, None, None, None

    stats = _collect_object_bounds(candidates)
    if not stats:
        return None, None, None, None

    filtered = _filter_outlier_bounds(stats)
    all_mins = [b.min for b in filtered]
    all_maxs = [b.max for b in filtered]
    global_min = mathutils.Vector(
        (
            min(v.x for v in all_mins),
            min(v.y for v in all_mins),
            min(v.z for v in all_mins),
        )
    )
    global_max = mathutils.Vector(
        (
            max(v.x for v in all_maxs),
            max(v.y for v in all_maxs),
            max(v.z for v in all_maxs),
        )
    )
    center = (global_min + global_max) * 0.5
    radius = max((global_max - center).length, 1e-3)
    anchor = min(filtered, key=lambda b: (b.center - center).length).obj

    # Robust floor estimate: absolute min can be dominated by one problematic mesh.
    floor_z = float(global_min.z)
    if len(filtered) >= 4:
        try:
            per_obj_min_z = np.array([float(b.min.z) for b in filtered], dtype=np.float64)
            floor_z = float(np.quantile(per_obj_min_z, 0.2))
            # Keep catcher a touch below estimated support level to avoid z-fighting.
            floor_z -= max(float(radius) * 0.004, 5e-4)
        except Exception:
            floor_z = float(global_min.z)

    return anchor, center, radius, floor_z


def _ensure_reframed_reference_camera(
    stem: str,
    groups: dict[str, list[bpy.types.Object]],
    composite_resolution: int = 1024,
    fill_factor: float = 0.90,
) -> bpy.types.Object | None:
    """Create/update <stem>_camera_reframed from the JSON-loaded reference camera."""
    reference_cam = bpy.data.objects.get(f"{stem}_camera")
    if reference_cam is None or getattr(reference_cam, "type", None) != "CAMERA":
        return None

    reframed_name = f"{stem}_camera_reframed"
    reframed_cam = bpy.data.objects.get(reframed_name)
    if reframed_cam is None:
        reframed_cam = new_camera(name=reframed_name)
    elif getattr(reframed_cam, "type", None) != "CAMERA":
        logger.warning(f"Cannot create reframed camera '{reframed_name}': non-camera object exists")
        return None

    # Keep camera data independent so reframing doesn't mutate the JSON camera.
    try:
        old_data = cast(Any, reframed_cam.data)
        reframed_cam.data = cast(Any, reference_cam.data).copy()
        if old_data is not None and old_data.users == 0:
            bpy.data.cameras.remove(old_data)
    except Exception:
        pass

    try:
        reframed_cam.matrix_world = reference_cam.matrix_world.copy()
    except Exception:
        try:
            reframed_cam.location = reference_cam.location.copy()
            reframed_cam.rotation_mode = reference_cam.rotation_mode
            if reframed_cam.rotation_mode == "QUATERNION":
                reframed_cam.rotation_quaternion = reference_cam.rotation_quaternion.copy()
            elif reframed_cam.rotation_mode == "AXIS_ANGLE":
                reframed_cam.rotation_axis_angle = tuple(reference_cam.rotation_axis_angle)
            else:
                reframed_cam.rotation_euler = reference_cam.rotation_euler.copy()
        except Exception:
            pass
    try:
        reframed_cam.scale = reference_cam.scale.copy()
    except Exception:
        pass

    # Preserve source metadata for optional per-camera rendering policies.
    try:
        for key in ("_source_width", "_source_height"):
            if key in reference_cam.keys():
                reframed_cam[key] = reference_cam[key]
    except Exception:
        pass

    framing_objs = [obj for obj in groups.get("mesh", []) if not _is_placeholder(obj)]
    if not framing_objs:
        # Prefer inputs (pred-invariant), then pred, then any available.
        for fallback_label in ("inputs", "pred"):
            framing_objs = [obj for obj in groups.get(fallback_label, []) if not _is_placeholder(obj)]
            if framing_objs:
                break
    if not framing_objs:
        return reframed_cam

    center, framing_corners = _collect_frame_point_cloud(framing_objs, max_mesh_points_per_object=2000)
    if not framing_corners:
        center, bounds_min, bounds_max = _objects_aabb(framing_objs)
        framing_corners = [
            mathutils.Vector((x, y, z))
            for x in (bounds_min.x, bounds_max.x)
            for y in (bounds_min.y, bounds_max.y)
            for z in (bounds_min.z, bounds_max.z)
        ]

    scene = _require_scene()
    saved_res_x, saved_res_y = int(scene.render.resolution_x), int(scene.render.resolution_y)
    try:
        scene.render.resolution_x = int(composite_resolution)
        scene.render.resolution_y = int(composite_resolution)
        _reframe_camera_to_corners(reframed_cam, center, framing_corners, fill_factor=float(fill_factor))
    finally:
        scene.render.resolution_x = saved_res_x
        scene.render.resolution_y = saved_res_y

    logger.info(f"Prepared reframed camera: {reframed_cam.name} (source: {reference_cam.name})")
    return reframed_cam


def _set_initial_preview_to_gt_inputs(
    stem: str,
    groups: dict[str, list[bpy.types.Object]],
    config: LoaderConfig,
) -> None:
    """Match initial scene state to the gt_inputs_<stem>_camera render basis.

    This aligns active camera, visibility, and lighting with the composite
    gt_inputs camera pass so interactive setup matches saved output.
    """
    reference_cam = bpy.data.objects.get(f"{stem}_camera")
    reframed_cam = bpy.data.objects.get(f"{stem}_camera_reframed")
    preview_cam = reframed_cam if reframed_cam is not None else reference_cam
    if preview_cam is None:
        return

    scene = _require_scene()
    scene.camera = preview_cam

    gt_inputs_vis = {"mesh": "both", "inputs": "both"}
    _apply_render_visibility(stem, groups, gt_inputs_vis)
    _apply_point_radius_floor_for_camera(groups, preview_cam, float(config.point_min_px), labels=gt_inputs_vis.keys())

    if not config.enable_lighting:
        return

    # Keep scene/canonical illumination GT-anchored, identical to composite pass logic.
    scene_light_obj, scene_light_center, scene_light_radius, scene_light_min_z = _lighting_subject_bounds(
        {"mesh": [obj for obj in groups.get("mesh", []) if not _is_placeholder(obj)]}
    )
    if scene_light_obj is None:
        scene_light_obj, scene_light_center, scene_light_radius, scene_light_min_z = _lighting_subject_bounds(groups)

    try:
        add_three_point_rig(
            preview_cam,
            scene_light_obj,
            subject_center=scene_light_center,
            subject_radius=scene_light_radius,
            subject_min_z=scene_light_min_z,
            hdr_strength=config.lighting_hdr_strength,
            world_ambient_scale=float(config.lighting_world_ambient_scale) * float(config.lighting_scene_ambient_scale),
            samples=config.cycles_samples,
            light_scale=float(config.lighting_scale) * float(config.lighting_scene_scale),
            distance_scale=config.lighting_distance_scale,
            fill_distance_ratio=config.lighting_fill_distance_ratio,
            rim_distance_ratio=config.lighting_rim_distance_ratio,
            key_base_energy=config.lighting_key_energy_base,
            fill_energy_ratio=float(config.lighting_fill_ratio) * float(config.lighting_scene_fill_ratio_scale),
            rim_energy_ratio=config.lighting_rim_ratio,
            kicker_energy_ratio=config.lighting_kicker_ratio,
            key_strength=config.lighting_key_strength,
            fill_strength=config.lighting_fill_strength,
            rim_strength=config.lighting_rim_strength,
            kicker_strength=config.lighting_kicker_strength,
            shadow_catcher=config.shadow_catcher,
            ao_strength=config.ao_strength,
            ao_distance=config.ao_distance,
            compositor_preset=config.compositor_preset,
            cycles_denoise=config.cycles_denoise,
            cycles_light_threshold=config.cycles_light_threshold,
            cycles_preview_factor=config.cycles_preview_factor,
            cycles_adaptive_threshold=config.cycles_adaptive_threshold,
            cycles_clamp_direct=config.cycles_clamp_direct,
            cycles_clamp_indirect=config.cycles_clamp_indirect,
            cycles_blur_glossy=config.cycles_blur_glossy,
            cycles_max_bounces=config.cycles_max_bounces,
            cycles_diffuse_bounces=config.cycles_diffuse_bounces,
            cycles_glossy_bounces=config.cycles_glossy_bounces,
            cycles_transmission_bounces=config.cycles_transmission_bounces,
            cycles_transparent_max_bounces=config.cycles_transparent_max_bounces,
            cycles_caustics_reflective=config.cycles_caustics_reflective,
            cycles_caustics_refractive=config.cycles_caustics_refractive,
            rebuild_rig=False,
        )
    except Exception as exc:
        logger.warning(f"Initial gt_inputs preview lighting skipped: {exc}")


def load_scene(config: LoaderConfig) -> None:
    out_dir = config.directory
    if not out_dir.is_dir():
        raise NotADirectoryError(out_dir)

    configure_scene(
        config.hdri, config.render_width, config.render_height, config.hdri_strength, view_exposure=config.view_exposure
    )
    stem = find_stem(out_dir, config.stem)
    setup_reference_camera(out_dir, stem, config)

    # Apply any CLI camera overrides after reference camera alignment
    try:
        _apply_default_camera_transform(bpy.data.objects.get("Camera"), config)
    except Exception:
        pass

    groups = import_object_groups(out_dir, stem, config)
    assign_group_collections(stem, groups)
    # Apply collection-level visibility defaults per group
    set_default_group_visibility(stem, groups)
    # Apply CLI-driven visibility overrides if provided
    try:
        apply_group_visibility_overrides(stem, config.group_visibility_overrides)
    except Exception:
        pass
    log_object_summary(groups)

    default_cam = bpy.data.objects.get("Camera")
    if config.create_object_cameras:
        create_object_cameras_for_targets(stem, default_cam, groups)

    # GT-anchored lighting so different pred models produce identical illumination.
    subject_obj, lighting_center, lighting_radius, lighting_min_z = _lighting_subject_bounds(
        {"mesh": [obj for obj in groups.get("mesh", []) if not _is_placeholder(obj)]}
    )
    if subject_obj is None:
        subject_obj, lighting_center, lighting_radius, lighting_min_z = _lighting_subject_bounds(groups)
    if config.enable_lighting:
        try:
            add_three_point_rig(
                default_cam,
                subject_obj,
                subject_center=lighting_center,
                subject_radius=lighting_radius,
                subject_min_z=lighting_min_z,
                hdr_strength=config.lighting_hdr_strength,
                world_ambient_scale=config.lighting_world_ambient_scale,
                samples=config.cycles_samples,
                light_scale=config.lighting_scale,
                distance_scale=config.lighting_distance_scale,
                fill_distance_ratio=config.lighting_fill_distance_ratio,
                rim_distance_ratio=config.lighting_rim_distance_ratio,
                key_base_energy=config.lighting_key_energy_base,
                fill_energy_ratio=config.lighting_fill_ratio,
                rim_energy_ratio=config.lighting_rim_ratio,
                kicker_energy_ratio=config.lighting_kicker_ratio,
                key_strength=config.lighting_key_strength,
                fill_strength=config.lighting_fill_strength,
                rim_strength=config.lighting_rim_strength,
                kicker_strength=config.lighting_kicker_strength,
                shadow_catcher=config.shadow_catcher,
                ao_strength=config.ao_strength,
                ao_distance=config.ao_distance,
                compositor_preset=config.compositor_preset,
                cycles_denoise=config.cycles_denoise,
                cycles_light_threshold=config.cycles_light_threshold,
                cycles_preview_factor=config.cycles_preview_factor,
                cycles_adaptive_threshold=config.cycles_adaptive_threshold,
                cycles_clamp_direct=config.cycles_clamp_direct,
                cycles_clamp_indirect=config.cycles_clamp_indirect,
                cycles_blur_glossy=config.cycles_blur_glossy,
                cycles_max_bounces=config.cycles_max_bounces,
                cycles_diffuse_bounces=config.cycles_diffuse_bounces,
                cycles_glossy_bounces=config.cycles_glossy_bounces,
                cycles_transmission_bounces=config.cycles_transmission_bounces,
                cycles_transparent_max_bounces=config.cycles_transparent_max_bounces,
                cycles_caustics_reflective=config.cycles_caustics_reflective,
                cycles_caustics_refractive=config.cycles_caustics_refractive,
            )
        except Exception as exc:
            logger.warning(f"Lighting rig skipped: {exc}")

    reference_cam = bpy.data.objects.get(f"{stem}_camera")
    if config.canonical_viewpoints:
        _create_canonical_viewpoints(
            groups,
            stem,
            reference_cam,
            elevation_deg=config.canonical_elevation_deg,
            distance_scale=config.canonical_distance_scale,
        )

    if config.orbit_camera:
        _create_orbit_camera(
            groups,
            stem,
            reference_cam,
            frames=config.orbit_frames,
            elevation_deg=config.orbit_elevation_deg,
            distance_scale=config.orbit_distance_scale,
        )

    # Keep two explicit viewpoints for the reference camera:
    # - <stem>_camera: untouched JSON pose/intrinsics
    # - <stem>_camera_reframed: optimized square-composite framing
    _ensure_reframed_reference_camera(
        stem, groups, composite_resolution=config.render_composite_resolution, fill_factor=0.90
    )

    _set_initial_preview_to_gt_inputs(stem, groups, config)

    render_output = config.render_output
    if render_output is None and config.render:
        render_output = out_dir / "renders" / stem
    if render_output is not None:
        batch_render(
            stem,
            groups,
            render_output,
            point_min_px=config.point_min_px,
            render_orbit=config.render_orbit,
            composite_resolution=config.render_composite_resolution,
            solo_resolution=config.render_solo_resolution,
            config=config,
        )


def main(
    dir_str: str,
    stem: str | None,
    as_planes: bool = True,
    planes_depth: float = 1.0,
    add_camera_body: bool = True,
    hdri: Path | None = None,
    # Camera overrides
    camera_pos: Sequence[float] | None = None,
    camera_rot_euler: Sequence[float] | None = None,
    camera_rot_mode: str = "XYZ",
    camera_rot_quat: Sequence[float] | None = None,
    camera_look_at: Sequence[float] | None = None,
    enable_lighting: bool = True,
    lighting_samples: int = 384,
    light_scale: float = 0.5,
    inputs_subsample: float = 1.0,
    render_width: int = 1024,
    render_height: int = 768,
    hdri_strength: float = 0.06,
    create_object_cameras: bool = True,
    shadow_catcher: bool = True,
    lighting_hdr_strength: float = 0.06,
    lighting_world_ambient_scale: float = 1.0,
    lighting_distance_scale: float = 1.7,
    lighting_fill_distance_ratio: float = 1.28,
    lighting_rim_distance_ratio: float = 1.18,
    lighting_key_energy_base: float = 100.0,
    lighting_fill_ratio: float = 0.25,
    lighting_rim_ratio: float = 0.68,
    lighting_kicker_ratio: float = 0.03,
    lighting_key_strength: float = 1.0,
    lighting_fill_strength: float = 1.0,
    lighting_rim_strength: float = 1.0,
    lighting_kicker_strength: float = 1.0,
    lighting_scene_scale: float = 0.9,
    lighting_scene_fill_ratio_scale: float = 0.75,
    lighting_scene_ambient_scale: float = 1.0,
    lighting_solo_scale: float = 0.9,
    lighting_solo_fill_ratio_scale: float = 1.6,
    view_exposure: float = 0.0,
    ao_strength: float = 0.3,
    ao_distance: float = 0.2,
    compositor_preset: str = "paper_clean",
    inputs_point_radius: float | None = None,
    inputs_color: Sequence[float] | None = None,
    pcd_point_radius: float | None = None,
    pcd_color: Sequence[float] | None = None,
    occ_point_radius: float | None = None,
    occ_color: Sequence[float] | None = None,
    free_point_radius: float | None = None,
    free_color: Sequence[float] | None = (0.8, 0.8, 0.8),
    logits_point_radius: float | None = None,
    logits_color: Sequence[float] | None = None,
    point_auto_enabled: bool = True,
    point_auto_target: int | None = 100_000,
    point_auto_radius_scale: float = 0.2,
    point_auto_radius_min: float = 0.0012,
    point_auto_radius_max: float = 0.012,
    point_min_px: float = 1.0,
    cycles_samples: int | None = None,
    cycles_denoise: bool = True,
    cycles_light_threshold: float = 0.01,
    cycles_preview_factor: float = 0.25,
    cycles_adaptive_threshold: float = 0.03,
    cycles_clamp_direct: float = 0.0,
    cycles_clamp_indirect: float = 2.5,
    cycles_blur_glossy: float = 0.0,
    cycles_max_bounces: int = 4,
    cycles_diffuse_bounces: int = 2,
    cycles_glossy_bounces: int = 2,
    cycles_transmission_bounces: int = 1,
    cycles_transparent_max_bounces: int = 4,
    cycles_caustics_reflective: bool = False,
    cycles_caustics_refractive: bool = False,
    # Canonical viewpoints
    canonical_viewpoints: bool = True,
    canonical_elevation_deg: float = 30.0,
    canonical_distance_scale: float = 2.5,
    # Orbit camera
    orbit_camera: bool = True,
    orbit_frames: int = 120,
    orbit_elevation_deg: float = 25.0,
    orbit_distance_scale: float = 2.5,
    # Batch rendering
    render: bool = False,
    render_output: str | None = None,
    render_orbit: bool = False,
    scene_res: int = 1024,
    object_res: int = 512,
    # Visibility overrides (comma-separated label lists)
    show: str | None = None,
    show_viewport: str | None = None,
    show_render: str | None = None,
    hide: str | None = None,
):
    out_dir = Path(dir_str).expanduser().resolve()
    hdri_path = Path(hdri).expanduser().resolve() if hdri else None

    # Build per-group visibility overrides from CLI strings
    def _parse_labels(value: str | None) -> set[str]:
        if not value:
            return set()
        # support comma or whitespace separated lists
        parts = re.split(r"[\s,]+", value.strip())
        return {p.strip().lower() for p in parts if p.strip()}

    KNOWN = {"inputs", "mesh", "pcd", "occ", "free", "logits", "pred"}

    def _expand(labels: set[str]) -> set[str]:
        if "all" in labels:
            return set(KNOWN)
        return {label for label in labels if label in KNOWN}

    show_both = _expand(_parse_labels(show))
    show_vp = _expand(_parse_labels(show_viewport))
    show_r = _expand(_parse_labels(show_render))
    hide_both = _expand(_parse_labels(hide))

    overrides: dict[str, Literal["viewport", "render", "both"] | None] = {}
    for lbl in show_both:
        overrides[lbl] = "both"
    for lbl in show_vp:
        overrides[lbl] = "viewport"
    for lbl in show_r:
        overrides[lbl] = "render"
    # Hide last to win precedence if user specified conflicting flags
    for lbl in hide_both:
        overrides[lbl] = None

    config = LoaderConfig(
        directory=out_dir,
        stem=stem,
        hdri=hdri_path,
        as_planes=as_planes,
        plane_depth=float(planes_depth),
        add_camera_body=add_camera_body,
        enable_lighting=enable_lighting,
        lighting_samples=int(lighting_samples),
        cycles_samples=(int(cycles_samples) if cycles_samples is not None else int(lighting_samples)),
        lighting_scale=float(light_scale),
        inputs_subsample=float(inputs_subsample),
        render_width=int(render_width),
        render_height=int(render_height),
        hdri_strength=float(hdri_strength),
        create_object_cameras=create_object_cameras,
        shadow_catcher=shadow_catcher,
        lighting_hdr_strength=float(lighting_hdr_strength),
        lighting_world_ambient_scale=float(lighting_world_ambient_scale),
        lighting_distance_scale=float(lighting_distance_scale),
        lighting_fill_distance_ratio=float(lighting_fill_distance_ratio),
        lighting_rim_distance_ratio=float(lighting_rim_distance_ratio),
        lighting_key_energy_base=float(lighting_key_energy_base),
        lighting_fill_ratio=float(lighting_fill_ratio),
        lighting_rim_ratio=float(lighting_rim_ratio),
        lighting_kicker_ratio=float(lighting_kicker_ratio),
        lighting_key_strength=float(lighting_key_strength),
        lighting_fill_strength=float(lighting_fill_strength),
        lighting_rim_strength=float(lighting_rim_strength),
        lighting_kicker_strength=float(lighting_kicker_strength),
        lighting_scene_scale=float(lighting_scene_scale),
        lighting_scene_fill_ratio_scale=float(lighting_scene_fill_ratio_scale),
        lighting_scene_ambient_scale=float(lighting_scene_ambient_scale),
        lighting_solo_scale=float(lighting_solo_scale),
        lighting_solo_fill_ratio_scale=float(lighting_solo_fill_ratio_scale),
        view_exposure=float(view_exposure),
        ao_strength=float(ao_strength),
        ao_distance=float(ao_distance),
        compositor_preset=_normalize_compositor_preset(compositor_preset),
        camera_location=_normalize_vec3(camera_pos, "camera_pos"),
        camera_rotation_euler_deg=_normalize_vec3(camera_rot_euler, "camera_rot_euler"),
        camera_rotation_mode=str(camera_rot_mode or "XYZ"),
        camera_rotation_quat=_normalize_vec4(camera_rot_quat, "camera_rot_quat"),
        camera_look_at=_normalize_vec3(camera_look_at, "camera_look_at"),
        inputs_point_radius=(float(inputs_point_radius) if inputs_point_radius is not None else None),
        inputs_color=_normalize_color(inputs_color),
        pcd_point_radius=(float(pcd_point_radius) if pcd_point_radius is not None else None),
        pcd_color=_normalize_color(pcd_color),
        occ_point_radius=(float(occ_point_radius) if occ_point_radius is not None else None),
        occ_color=_normalize_color(occ_color),
        free_point_radius=(float(free_point_radius) if free_point_radius is not None else None),
        free_color=_normalize_color(free_color),
        logits_point_radius=(float(logits_point_radius) if logits_point_radius is not None else None),
        logits_color=_normalize_color(logits_color),
        point_auto_enabled=bool(point_auto_enabled),
        point_auto_target=(int(point_auto_target) if point_auto_target is not None else None),
        point_auto_radius_scale=float(point_auto_radius_scale),
        point_auto_radius_min=float(point_auto_radius_min),
        point_auto_radius_max=float(point_auto_radius_max),
        point_min_px=float(point_min_px),
        cycles_denoise=bool(cycles_denoise),
        cycles_light_threshold=float(cycles_light_threshold),
        cycles_preview_factor=float(cycles_preview_factor),
        cycles_adaptive_threshold=float(cycles_adaptive_threshold),
        cycles_clamp_direct=float(cycles_clamp_direct),
        cycles_clamp_indirect=float(cycles_clamp_indirect),
        cycles_blur_glossy=float(cycles_blur_glossy),
        cycles_max_bounces=int(cycles_max_bounces),
        cycles_diffuse_bounces=int(cycles_diffuse_bounces),
        cycles_glossy_bounces=int(cycles_glossy_bounces),
        cycles_transmission_bounces=int(cycles_transmission_bounces),
        cycles_transparent_max_bounces=int(cycles_transparent_max_bounces),
        cycles_caustics_reflective=bool(cycles_caustics_reflective),
        cycles_caustics_refractive=bool(cycles_caustics_refractive),
        group_visibility_overrides=(overrides if overrides else None),
        canonical_viewpoints=bool(canonical_viewpoints),
        canonical_elevation_deg=float(canonical_elevation_deg),
        canonical_distance_scale=float(canonical_distance_scale),
        orbit_camera=bool(orbit_camera),
        orbit_frames=int(orbit_frames),
        orbit_elevation_deg=float(orbit_elevation_deg),
        orbit_distance_scale=float(orbit_distance_scale),
        render=bool(render) or render_output is not None or bool(render_orbit),
        render_output=(Path(render_output).expanduser().resolve() if render_output is not None else None),
        render_orbit=bool(render_orbit),
        render_composite_resolution=int(scene_res),
        render_solo_resolution=int(object_res),
    )
    load_scene(config)


def _parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group("General")
    general.add_argument("--dir", required=True, help="Directory containing <stem>_* exported files")
    general.add_argument("--stem", help="Filename stem (prefix before _camera.json)")
    general.add_argument(
        "--log-level",
        default=os.getenv("BLENDER_LOAD_LOG_LEVEL", "INFO"),
        help="Set log verbosity (DEBUG, INFO, WARNING, ...)",
    )

    camera_group = parser.add_argument_group("Camera & Scene")
    camera_group.add_argument("--hdri", help="Path to HDRI file")
    camera_group.add_argument("--hdri-strength", type=float, default=0.06, help="Strength for the HDRI environment map")
    camera_group.add_argument("--render-width", type=int, default=1024, help="Render resolution width (pixels)")
    camera_group.add_argument("--render-height", type=int, default=768, help="Render resolution height (pixels)")
    camera_group.add_argument("--no-planes", action="store_true", help="Skip creating camera image planes")
    camera_group.add_argument("--planes-depth", type=float, default=1.0, help="Distance in meters for overlay planes")
    camera_group.add_argument("--no-camera-body", action="store_true", help="Skip camera body and frustum helpers")
    camera_group.add_argument(
        "--no-object-cameras", action="store_true", help="Skip creating per-target object cameras"
    )
    camera_group.add_argument(
        "--camera-pos",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override default camera position in meters",
    )
    camera_group.add_argument(
        "--camera-rot-euler",
        nargs=3,
        type=float,
        metavar=("RX", "RY", "RZ"),
        default=None,
        help="Override default camera rotation in degrees (Euler)",
    )
    camera_group.add_argument(
        "--camera-rot-mode",
        type=str,
        default="XYZ",
        help="Euler rotation mode for --camera-rot-euler (e.g., XYZ, XZY, ZYX)",
    )
    camera_group.add_argument(
        "--camera-rot-quat",
        nargs=4,
        type=float,
        metavar=("W", "X", "Y", "Z"),
        default=None,
        help="Override default camera rotation as quaternion (W X Y Z)",
    )
    camera_group.add_argument(
        "--camera-look-at",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Aim default camera at this world-space point (overrides rotation)",
    )

    viewpoint_group = parser.add_argument_group("Viewpoints")
    viewpoint_group.add_argument(
        "--no-canonical-viewpoints",
        action="store_true",
        help="Disable canonical viewpoint cameras (falls back to heuristic default camera)",
    )
    viewpoint_group.add_argument(
        "--canonical-elevation",
        type=float,
        default=30.0,
        help="Elevation angle in degrees for canonical viewpoints (default: 30)",
    )
    viewpoint_group.add_argument(
        "--canonical-distance-scale",
        type=float,
        default=2.5,
        help="Distance multiplier relative to scene radius for canonical cameras (default: 2.5)",
    )
    viewpoint_group.add_argument(
        "--no-orbit-camera",
        action="store_true",
        help="Disable the animated orbit camera that revolves around the scene",
    )
    viewpoint_group.add_argument(
        "--orbit-frames", type=int, default=120, help="Number of frames for a full 360-degree orbit (default: 120)"
    )
    viewpoint_group.add_argument(
        "--orbit-elevation", type=float, default=25.0, help="Elevation angle in degrees for orbit camera (default: 25)"
    )
    viewpoint_group.add_argument(
        "--orbit-distance-scale",
        type=float,
        default=2.5,
        help="Distance multiplier relative to scene radius for orbit camera (default: 2.5)",
    )

    # Render time scaling (approximate, GPU-dependent):
    #   --scene-res: quadratic - 2048^2 is about 4x slower than 1024^2 per frame
    #   --lighting-samples: linear upper bound, but adaptive sampling means
    #     most pixels converge early; 384->1024 is about 2-3x slower in practice
    #   --cycles-adaptive-threshold: lower = more samples in noisy regions;
    #     0.03 (default) -> 0.01 (HQ) adds ~50-100% time
    #   --cycles-*-bounces: sublinear; 4->8 max bounces is about 10-30% slower,
    #     mostly affects reflective/glass materials
    #   --cycles-clamp-indirect: no time impact (caps firefly brightness)
    #
    # Ballpark for 8-object tabletop scene (32 renders: 16 scene + 16 solo):
    #   1024² default quality:  ~1.5 min
    #   2048² default quality:  ~3.5 min
    #   2048² HQ (1024 spp, 0.01 thresh, 8 bounces): ~12 min
    render_group = parser.add_argument_group("Batch Rendering")
    render_group.add_argument(
        "--render", action="store_true", help="Enable batch rendering with default output at <dir>/renders/<stem>"
    )
    render_group.add_argument(
        "--render-output", type=str, default=None, help="Output directory for batch renders (implies --render)"
    )
    render_group.add_argument(
        "--render-orbit", action="store_true", help="Also render orbit animation sequences (implies --render)"
    )
    render_group.add_argument(
        "--scene-res", type=int, default=1024,
        help="Resolution for scene/composite renders in pixels (default: 1024). Time scales quadratically.",
    )
    render_group.add_argument(
        "--object-res", type=int, default=512,
        help="Resolution for per-object solo renders in pixels (default: 512). Time scales quadratically.",
    )

    lighting_group = parser.add_argument_group("Lighting")
    lighting_group.add_argument("--no-lighting", action="store_true", help="Disable procedural lighting rig")
    lighting_group.add_argument(
        "--lighting-samples", type=int, default=384, help="Cycles samples for lighting rig setup"
    )
    lighting_group.add_argument("--lighting-scale", type=float, default=0.5, help="Multiplier for area light energy")
    lighting_group.add_argument(
        "--lighting-hdr-strength",
        type=float,
        default=0.06,
        help="Background strength for the lighting rig's fill contribution",
    )
    lighting_group.add_argument(
        "--lighting-world-ambient-scale",
        type=float,
        default=1.0,
        help="Compatibility multiplier for HDR ambient contribution (1.0 = neutral)",
    )
    lighting_group.add_argument(
        "--lighting-distance-scale",
        type=float,
        default=1.7,
        help="Base camera-space light distance as scene_radius * scale",
    )
    lighting_group.add_argument(
        "--lighting-fill-distance-ratio",
        type=float,
        default=1.28,
        help="Fill-light distance relative to key-light distance",
    )
    lighting_group.add_argument(
        "--lighting-rim-distance-ratio",
        type=float,
        default=1.18,
        help="Rim-light distance relative to key-light distance",
    )
    lighting_group.add_argument(
        "--lighting-key-energy-base",
        type=float,
        default=100.0,
        help="Base key-light energy before scene-size and --lighting-scale factors",
    )
    lighting_group.add_argument(
        "--lighting-fill-ratio", type=float, default=0.25, help="Fill-light energy as a ratio of key-light energy"
    )
    lighting_group.add_argument(
        "--lighting-rim-ratio", type=float, default=0.68, help="Rim-light energy as a ratio of key-light energy"
    )
    lighting_group.add_argument(
        "--lighting-kicker-ratio", type=float, default=0.03, help="Kicker energy as a ratio of key-light energy"
    )
    lighting_group.add_argument(
        "--lighting-key-strength", type=float, default=1.0, help="Extra multiplier applied to key-light energy"
    )
    lighting_group.add_argument(
        "--lighting-fill-strength", type=float, default=1.0, help="Extra multiplier applied to fill-light energy"
    )
    lighting_group.add_argument(
        "--lighting-rim-strength", type=float, default=1.0, help="Extra multiplier applied to rim-light energy"
    )
    lighting_group.add_argument(
        "--lighting-kicker-strength", type=float, default=1.0, help="Extra multiplier applied to kicker-light energy"
    )
    lighting_group.add_argument(
        "--lighting-scene-scale",
        type=float,
        default=0.9,
        help="Extra light-scale multiplier used for scene/canonical/orbit renders",
    )
    lighting_group.add_argument(
        "--lighting-scene-fill-ratio-scale",
        type=float,
        default=0.75,
        help="Multiplier on --lighting-fill-ratio for scene/canonical/orbit renders",
    )
    lighting_group.add_argument(
        "--lighting-scene-ambient-scale",
        type=float,
        default=1.0,
        help="Multiplier on world ambient scale for scene/canonical/orbit renders",
    )
    lighting_group.add_argument(
        "--lighting-solo-scale",
        type=float,
        default=0.9,
        help="Extra light-scale multiplier used only for solo object renders",
    )
    lighting_group.add_argument(
        "--lighting-solo-fill-ratio-scale",
        type=float,
        default=1.6,
        help="Multiplier on --lighting-fill-ratio used only for solo object renders",
    )
    lighting_group.add_argument(
        "--view-exposure", type=float, default=0.0, help="Scene view exposure (applied with AgX view transform)"
    )
    lighting_group.add_argument(
        "--ao-strength", type=float, default=0.3, help="Ambient occlusion blend strength in compositor (0 disables)"
    )
    lighting_group.add_argument(
        "--ao-distance",
        type=float,
        default=0.2,
        help="Ambient occlusion distance if supported by current Blender build",
    )
    lighting_group.add_argument(
        "--compositor-preset",
        type=str,
        choices=("off", "paper_clean", "beauty"),
        default="paper_clean",
        help="Postprocess style: off (AO only), paper_clean, or beauty (adds subtle bloom)",
    )
    lighting_group.add_argument(
        "--no-shadow-catcher", action="store_true", help="Disable the ground shadow catcher under the subject"
    )

    inputs_group = parser.add_argument_group("Inputs")
    inputs_group.add_argument(
        "--inputs-subsample", type=float, default=1.0, help="Random subsample ratio for *_inputs.ply point clouds"
    )
    inputs_group.add_argument(
        "--inputs-point-radius",
        type=float,
        default=None,
        help="Sphere radius for *_inputs point clouds (meters). Omit for auto",
    )
    inputs_group.add_argument(
        "--inputs-color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=None,
        help="Override RGB color for *_inputs point clouds (0-1). Omit for attribute colors",
    )

    point_group = parser.add_argument_group("Point Cloud Overrides")
    point_group.add_argument(
        "--pcd-point-radius",
        type=float,
        default=None,
        help="Sphere radius for *_pointcloud data (meters). Omit for auto",
    )
    point_group.add_argument(
        "--pcd-color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=None,
        help="Override RGB color for *_pointcloud data (0-1)",
    )
    point_group.add_argument(
        "--occ-point-radius", type=float, default=None, help="Sphere radius for *_occ data (meters). Omit for auto"
    )
    point_group.add_argument(
        "--occ-color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=None,
        help="Override RGB color for *_occ data (0-1)",
    )
    point_group.add_argument(
        "--free-point-radius", type=float, default=None, help="Sphere radius for *_free data (meters). Omit for auto"
    )
    point_group.add_argument(
        "--free-color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=[0.8, 0.8, 0.8],
        help="Override RGB color for *_free data (0-1)",
    )
    point_group.add_argument(
        "--logits-point-radius",
        type=float,
        default=None,
        help="Sphere radius for *_logits data (meters). Omit for auto",
    )

    point_group.add_argument(
        "--point-auto-target",
        type=int,
        default=100_000,
        help="Upper bound on visible points per cloud before auto-downsampling (<=0 disables)",
    )
    point_group.add_argument(
        "--point-radius-scale",
        type=float,
        default=0.2,
        help="Scale factor used to convert spacing to sphere radius during auto sizing",
    )
    point_group.add_argument(
        "--point-radius-min", type=float, default=0.0012, help="Minimum sphere radius when auto sizing (meters)"
    )
    point_group.add_argument(
        "--point-radius-max", type=float, default=0.012, help="Maximum sphere radius when auto sizing (meters)"
    )
    point_group.add_argument(
        "--point-min-px",
        type=float,
        default=1.0,
        help="Minimum projected point radius in pixels during batch rendering (<=0 disables)",
    )
    point_group.add_argument(
        "--no-point-auto", action="store_true", help="Disable automatic point budgeting and radius heuristics"
    )
    point_group.add_argument(
        "--logits-color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=None,
        help="Override RGB color for *_logits data (0-1)",
    )

    cycles_group = parser.add_argument_group("Cycles")
    cycles_group.add_argument(
        "--cycles-samples",
        type=int,
        default=None,
        help="Render sample count (defaults to --lighting-samples when omitted)",
    )
    cycles_group.add_argument(
        "--cycles-preview-factor", type=float, default=0.25, help="Preview sample multiplier relative to render samples"
    )
    cycles_group.add_argument(
        "--cycles-adaptive-threshold", type=float, default=0.03, help="Adaptive sampling threshold"
    )
    cycles_group.add_argument(
        "--cycles-light-threshold",
        type=float,
        default=0.01,
        help="Light sampling threshold (use lower values to reduce fireflies)",
    )
    cycles_group.add_argument(
        "--cycles-clamp-direct", type=float, default=0.0, help="Clamp value for direct samples (0 disables)"
    )
    cycles_group.add_argument(
        "--cycles-clamp-indirect", type=float, default=2.5, help="Clamp value for indirect samples"
    )
    cycles_group.add_argument(
        "--cycles-blur-glossy", type=float, default=0.0, help="Blur glossy shader smoothing amount"
    )
    cycles_group.add_argument("--cycles-max-bounces", type=int, default=4, help="Maximum total light bounces")
    cycles_group.add_argument("--cycles-diffuse-bounces", type=int, default=2, help="Diffuse bounce count")
    cycles_group.add_argument("--cycles-glossy-bounces", type=int, default=2, help="Glossy bounce count")
    cycles_group.add_argument("--cycles-transmission-bounces", type=int, default=1, help="Transmission bounce count")
    cycles_group.add_argument("--cycles-transparent-bounces", type=int, default=4, help="Transparent bounce count")
    cycles_group.add_argument("--cycles-caustics-reflective", action="store_true", help="Enable reflective caustics")
    cycles_group.add_argument("--cycles-caustics-refractive", action="store_true", help="Enable refractive caustics")
    cycles_group.add_argument(
        "--no-cycles-denoise", action="store_true", help="Disable Cycles denoising on the active view layer"
    )

    visibility_group = parser.add_argument_group("Visibility Overrides")
    visibility_group.add_argument(
        "--show",
        type=str,
        default=None,
        help="Comma/space list of groups to show in viewport and render (e.g. 'mesh,pcd')",
    )
    visibility_group.add_argument("--show-viewport", type=str, default=None, help="Groups to show in viewport only")
    visibility_group.add_argument("--show-render", type=str, default=None, help="Groups to show in render only")
    visibility_group.add_argument("--hide", type=str, default=None, help="Groups to hide in both viewport and render")

    return parser.parse_args(argv)


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    args = _parse_cli_args(argv)
    _configure_logger(args.log_level)
    main(
        args.dir,
        args.stem,
        as_planes=not args.no_planes,
        planes_depth=args.planes_depth,
        add_camera_body=not args.no_camera_body,
        hdri=args.hdri,
        camera_pos=args.camera_pos,
        camera_rot_euler=args.camera_rot_euler,
        camera_rot_mode=args.camera_rot_mode,
        camera_rot_quat=args.camera_rot_quat,
        camera_look_at=args.camera_look_at,
        canonical_viewpoints=not args.no_canonical_viewpoints,
        canonical_elevation_deg=args.canonical_elevation,
        canonical_distance_scale=args.canonical_distance_scale,
        orbit_camera=not args.no_orbit_camera,
        orbit_frames=args.orbit_frames,
        orbit_elevation_deg=args.orbit_elevation,
        orbit_distance_scale=args.orbit_distance_scale,
        render=args.render,
        render_output=args.render_output,
        render_orbit=args.render_orbit,
        enable_lighting=not args.no_lighting,
        lighting_samples=args.lighting_samples,
        light_scale=args.lighting_scale,
        inputs_subsample=args.inputs_subsample,
        render_width=args.render_width,
        render_height=args.render_height,
        hdri_strength=args.hdri_strength,
        create_object_cameras=not args.no_object_cameras,
        shadow_catcher=not args.no_shadow_catcher,
        lighting_hdr_strength=args.lighting_hdr_strength,
        lighting_world_ambient_scale=args.lighting_world_ambient_scale,
        lighting_distance_scale=args.lighting_distance_scale,
        lighting_fill_distance_ratio=args.lighting_fill_distance_ratio,
        lighting_rim_distance_ratio=args.lighting_rim_distance_ratio,
        lighting_key_energy_base=args.lighting_key_energy_base,
        lighting_fill_ratio=args.lighting_fill_ratio,
        lighting_rim_ratio=args.lighting_rim_ratio,
        lighting_kicker_ratio=args.lighting_kicker_ratio,
        lighting_key_strength=args.lighting_key_strength,
        lighting_fill_strength=args.lighting_fill_strength,
        lighting_rim_strength=args.lighting_rim_strength,
        lighting_kicker_strength=args.lighting_kicker_strength,
        lighting_scene_scale=args.lighting_scene_scale,
        lighting_scene_fill_ratio_scale=args.lighting_scene_fill_ratio_scale,
        lighting_scene_ambient_scale=args.lighting_scene_ambient_scale,
        lighting_solo_scale=args.lighting_solo_scale,
        lighting_solo_fill_ratio_scale=args.lighting_solo_fill_ratio_scale,
        view_exposure=args.view_exposure,
        ao_strength=args.ao_strength,
        ao_distance=args.ao_distance,
        compositor_preset=args.compositor_preset,
        inputs_point_radius=args.inputs_point_radius,
        inputs_color=args.inputs_color,
        pcd_point_radius=args.pcd_point_radius,
        pcd_color=args.pcd_color,
        occ_point_radius=args.occ_point_radius,
        occ_color=args.occ_color,
        free_point_radius=args.free_point_radius,
        free_color=args.free_color,
        logits_point_radius=args.logits_point_radius,
        logits_color=args.logits_color,
        point_auto_enabled=not args.no_point_auto,
        point_auto_target=args.point_auto_target,
        point_auto_radius_scale=args.point_radius_scale,
        point_auto_radius_min=args.point_radius_min,
        point_auto_radius_max=args.point_radius_max,
        point_min_px=args.point_min_px,
        cycles_samples=args.cycles_samples,
        cycles_denoise=not args.no_cycles_denoise,
        cycles_light_threshold=args.cycles_light_threshold,
        cycles_preview_factor=args.cycles_preview_factor,
        cycles_adaptive_threshold=args.cycles_adaptive_threshold,
        cycles_clamp_direct=args.cycles_clamp_direct,
        cycles_clamp_indirect=args.cycles_clamp_indirect,
        cycles_blur_glossy=args.cycles_blur_glossy,
        cycles_max_bounces=args.cycles_max_bounces,
        cycles_diffuse_bounces=args.cycles_diffuse_bounces,
        cycles_glossy_bounces=args.cycles_glossy_bounces,
        cycles_transmission_bounces=args.cycles_transmission_bounces,
        cycles_transparent_max_bounces=args.cycles_transparent_bounces,
        cycles_caustics_reflective=args.cycles_caustics_reflective,
        cycles_caustics_refractive=args.cycles_caustics_refractive,
        show=args.show,
        show_viewport=args.show_viewport,
        show_render=args.show_render,
        hide=args.hide,
        scene_res=args.scene_res,
        object_res=args.object_res,
    )
