"""Visualize the generation process of diffusion and autoregressive models.

By default, this script decodes intermediate states to meshes and writes `.ply`
files for subsequent rendering via bproc-pubvis. With `vis.implicit_render=true`,
it instead renders intermediate states directly from the implicit field and
writes `.png` frames into an ``implicit/`` subdirectory (to avoid colliding with
bproc-rendered PNGs).

Output structure (mesh mode, default):
    output_dir/<category>/<obj_name>/
        diffusion/step_00.ply ... step_17.ply
        ar/token_001.ply ... token_512.ply
        input.ply
        gt.ply

Output structure (implicit mode, vis.implicit_render=true):
    output_dir/<category>/<obj_name>/
        diffusion/implicit/input.png, gt.png
        diffusion/implicit/step_00.png, step_00_normals.png, step_00_depth.png ...
        ar/implicit/input.png, gt.png
        ar/implicit/token_001.png, token_001_normals.png, token_001_depth.png ...
        input.ply
        gt.ply
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import lightning
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from dataset import get_dataset
from models import AutoregressiveModel, DiffusionModel, get_model, patch_attention
from models.src.dvr import DVR, DifferentiableRayCaster, RayMarchingConfig
from models.src.utils import get_normals
from utils import (
    binary_from_multi_class,
    convert_extrinsic,
    inv_trafo,
    look_at,
    resolve_save_dir,
    save_mesh,
    setup_config,
    setup_logger,
    suppress_known_optional_dependency_warnings,
    to_tensor,
)
from visualize import Generator

logger = setup_logger(__name__)

INPUT_COLOR = np.array([0.410603, 0.101933, 0.0683599], dtype=np.float32)
GT_COLOR = np.array([0.410603, 0.101933, 0.0683599], dtype=np.float32)


def _default_object_indices(dataset: Any, limit: int = 5) -> list[int]:
    """Select a deterministic default subset of dataset indices.

    If dataset metadata is available as ``dataset.objects`` with one entry per
    dataset item, sort by (category, name) to avoid filesystem-order variance.
    """
    total = len(dataset)
    if total <= 0:
        return []

    objects = getattr(dataset, "objects", None)
    if isinstance(objects, list) and len(objects) == total:
        keyed_indices: list[tuple[str, str, int]] = []
        for idx, obj in enumerate(objects):
            if isinstance(obj, dict):
                category = str(obj.get("category", ""))
                name = str(obj.get("name", obj.get("obj_name", "")))
            else:
                category = ""
                name = str(obj)
            keyed_indices.append((category, name, idx))
        keyed_indices.sort()
        return [idx for _category, _name, idx in keyed_indices[: min(limit, total)]]

    return list(range(min(limit, total)))


def _render_points_preview(
    points: np.ndarray,
    *,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_size: int,
    color: np.ndarray,
) -> Image.Image:
    """Render a simple z-buffered point preview in camera image space."""
    rgb = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    if points.size == 0:
        return Image.fromarray(rgb)

    points = np.asarray(points, dtype=np.float32)
    cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z = cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return Image.fromarray(rgb)

    cam = cam[valid]
    z = z[valid]
    proj = cam @ intrinsic.T
    u = np.rint(proj[:, 0] / np.maximum(proj[:, 2], 1e-6)).astype(np.int32)
    v = np.rint(proj[:, 1] / np.maximum(proj[:, 2], 1e-6)).astype(np.int32)

    in_bounds = (u >= 0) & (u < image_size) & (v >= 0) & (v < image_size)
    if not np.any(in_bounds):
        return Image.fromarray(rgb)

    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    order = np.argsort(z)[::-1]  # far -> near, so nearer points overwrite
    color_uint8 = np.clip(color * 255.0, 0, 255).astype(np.uint8)
    for idx in order:
        rgb[v[idx], u[idx]] = color_uint8
    return Image.fromarray(rgb)


def _save_implicit_reference_panels(
    method_dir: Path,
    implicit_renderer: "ImplicitRenderer",
    input_points: np.ndarray | None,
    gt_mesh: trimesh.Trimesh | None,
) -> None:
    """Save input/GT panels for implicit mode to keep strip composition consistent."""
    intrinsic = implicit_renderer.intrinsic[0].detach().cpu().numpy()
    extrinsic = implicit_renderer.extrinsic[0].detach().cpu().numpy()
    size = int(implicit_renderer.image_size)

    if input_points is not None and input_points.size > 0:
        input_img = _render_points_preview(
            input_points,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            image_size=size,
            color=INPUT_COLOR,
        )
        input_img.save(method_dir / "input.png")

    if gt_mesh is not None and len(gt_mesh.vertices) > 0:
        if gt_mesh.faces is not None and len(gt_mesh.faces) > 0:
            gt_points = np.asarray(gt_mesh.sample(20000), dtype=np.float32)
        else:
            gt_points = np.asarray(gt_mesh.vertices, dtype=np.float32)
        gt_img = _render_points_preview(
            gt_points,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            image_size=size,
            color=GT_COLOR,
        )
        gt_img.save(method_dir / "gt.png")


def decode_diffusion_feature(model: Any, latent: Tensor) -> Tensor:
    """Decode diffusion intermediate latent to implicit feature."""
    return cast(Tensor, model.decode(latent))


def decode_ar_feature(model: Any, partial_indices: Tensor, n_total: int) -> Tensor:
    """Decode AR intermediate (partial sequence) to implicit feature."""
    batch_size = partial_indices.size(0)
    current_len = partial_indices.size(1)
    if current_len < n_total:
        padding = torch.zeros(
            batch_size, n_total - current_len, dtype=partial_indices.dtype, device=partial_indices.device
        )
        padded = torch.cat([partial_indices, padding], dim=1)
    else:
        padded = partial_indices
    return cast(Tensor, model.decode(padded))


def logits_from_feature(
    predictor: Any,
    feature: Tensor,
    points: Tensor,
    points_batch_size: int | None = None,
    *,
    sdf: bool = False,
    sdf_tau: float = 0.01,
    sdf_iso: float = 0.0,
) -> Tensor:
    """Query implicit field logits with optional batching.

    For SDF decoders, map signed distance to occupancy-like logits around the
    zero level set so the ray marcher can stay in logit space.
    """
    if points_batch_size is None or points.size(1) <= points_batch_size:
        out = cast(dict[str, Tensor], predictor.decode(points=points, feature=feature))
        logits = out["logits"]
    else:
        out_list = [
            cast(dict[str, Tensor], predictor.decode(points=chunk, feature=feature))
            for chunk in torch.split(points, points_batch_size, dim=1)
        ]
        logits = torch.cat([o["logits"] for o in out_list], dim=1)

    if logits.ndim == 3:
        if logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        else:
            logits = binary_from_multi_class(logits)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape (B, N), got {tuple(logits.shape)}")

    if sdf:
        logits = -(logits - float(sdf_iso)) / max(float(sdf_tau), 1e-6)
    return logits.float()


def _implicit_predict(
    points: Tensor,
    feature: Tensor,
    predictor: Any,
    points_batch_size: int | None,
    sdf: bool,
    sdf_tau: float,
    sdf_iso: float,
    **_kwargs: Any,
) -> Tensor:
    """Ray-marcher callback: query implicit field at arbitrary points.

    Bound via ``functools.partial`` to avoid closure-capture pitfalls in loops.
    """
    return logits_from_feature(
        predictor,
        feature,
        points,
        points_batch_size,
        sdf=sdf,
        sdf_tau=sdf_tau,
        sdf_iso=sdf_iso,
    )


def decode_diffusion_intermediate(
    model: Any,
    latent: Tensor,
    generator: Generator,
    points_batch_size: int | None = None,
) -> trimesh.Trimesh:
    """Decode a diffusion intermediate latent to a mesh."""
    feature = decode_diffusion_feature(model, latent)
    vae = model._vae
    points = generator.query_points.unsqueeze(0).to(latent.device)
    # Keep decoder output domain unchanged for mesh extraction (raw logits or SDF).
    logits = logits_from_feature(vae, feature, points, points_batch_size)
    grid = logits.view(*generator.grid_shape).float().cpu().numpy()
    return generator.extract_mesh(grid)


# TODO: add alternative padding strategies for partial AR sequences:
#   - repeat last token
#   - mean codebook entry
#   - mask/ignore unfilled positions (needs decoder changes)
def decode_ar_intermediate(
    model: Any,
    partial_indices: Tensor,
    n_total: int,
    generator: Generator,
    points_batch_size: int | None = None,
) -> trimesh.Trimesh:
    """Decode AR intermediate (partial index sequence) to a mesh.

    Pads partial indices to full length with zeros before decoding.
    """
    feature = decode_ar_feature(model, partial_indices, n_total)
    discretizer = model._discretizer
    points = generator.query_points.unsqueeze(0).to(feature.device)
    # Keep decoder output domain unchanged for mesh extraction (raw logits or SDF).
    logits = logits_from_feature(discretizer, feature, points, points_batch_size)
    grid = logits.view(*generator.grid_shape).float().cpu().numpy()
    return generator.extract_mesh(grid)


@dataclass
class RenderResult:
    """All outputs from a single implicit render pass."""

    shading: Image.Image
    normals: Image.Image
    depth: Image.Image


def _filter_components_2d(
    mask: np.ndarray,
    *,
    keep_largest: bool,
    min_pixels: int,
) -> np.ndarray:
    """Filter disconnected foreground components in a 2D mask."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    if not keep_largest and min_pixels <= 1:
        return mask

    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    sizes: list[int] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                if cy > 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    stack.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    stack.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    stack.append((cy, cx + 1))

            components.append(coords)
            sizes.append(len(coords))

    if not components:
        return np.zeros_like(mask, dtype=bool)

    out = np.zeros_like(mask, dtype=bool)
    largest_idx = int(np.argmax(np.asarray(sizes, dtype=np.int64)))
    min_keep = max(int(min_pixels), 1)

    for comp_idx, coords in enumerate(components):
        size = len(coords)
        if keep_largest and comp_idx != largest_idx:
            continue
        if size < min_keep:
            continue
        for y, x in coords:
            out[y, x] = True
    return out


def _smooth_normals_map(
    normal_map: np.ndarray,
    mask_2d: np.ndarray,
    *,
    kernel_size: int,
    passes: int,
) -> np.ndarray:
    """Apply masked box filtering on a normal map to reduce banding artifacts."""
    if kernel_size <= 1 or passes <= 0:
        return normal_map
    if kernel_size % 2 == 0:
        kernel_size += 1

    normals_t = torch.from_numpy(normal_map).permute(2, 0, 1).unsqueeze(0).float()
    mask_t = torch.from_numpy(mask_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    area = float(kernel_size * kernel_size)

    for _ in range(passes):
        weighted = normals_t * mask_t
        n_sum = F.avg_pool2d(weighted, kernel_size=kernel_size, stride=1, padding=kernel_size // 2) * area
        w_sum = F.avg_pool2d(mask_t, kernel_size=kernel_size, stride=1, padding=kernel_size // 2) * area
        smoothed = n_sum / torch.clamp(w_sum, min=1e-6)
        normals_t = torch.where(w_sum > 0, smoothed, normals_t)
        normals_t = F.normalize(normals_t, dim=1, eps=1e-6)

    return normals_t.squeeze(0).permute(1, 2, 0).numpy()


class ImplicitRenderer:
    def __init__(
        self,
        device: torch.device,
        image_size: int,
        threshold: float,
        *,
        near: float = 0.0,
        far: float = 2.4,
        num_steps: int = 128,
        num_refine_steps: int = 12,
        refine_mode: Literal["secant", "bisection", "linear", "midpoint"] = "secant",
        cam_location: tuple[float, float, float] = (1.5, 0.0, 1.0),
        cam_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        cam_focal: float | None = None,
        keep_largest_component: bool = False,
        min_component_pixels: int = 0,
        normal_smooth_kernel: int = 0,
        normal_smooth_passes: int = 1,
    ) -> None:
        self.device = device
        self.image_size = image_size
        focal = float(cam_focal) if cam_focal is not None else float(image_size)
        self.keep_largest_component = keep_largest_component
        self.min_component_pixels = max(int(min_component_pixels), 0)
        self.normal_smooth_kernel = max(int(normal_smooth_kernel), 0)
        self.normal_smooth_passes = max(int(normal_smooth_passes), 1)

        self.intrinsic = torch.tensor(
            [[focal, 0.0, image_size / 2.0], [0.0, focal, image_size / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        pose = look_at(np.asarray(cam_location, dtype=np.float32), np.asarray(cam_target, dtype=np.float32))
        extrinsic_gl = cast(np.ndarray, inv_trafo(pose)).astype(np.float32)
        # `get_rays` uses OpenCV pinhole projection conventions (camera looks +Z).
        # Convert from OpenGL look_at pose to avoid flipped viewing direction.
        extrinsic_cv = cast(np.ndarray, convert_extrinsic(extrinsic_gl, "opengl", "opencv")).astype(np.float32)
        self.extrinsic = torch.from_numpy(extrinsic_cv).to(device).unsqueeze(0)

        ray_cfg = RayMarchingConfig(
            near=near,
            far=far,
            num_steps=num_steps,
            num_refine_steps=num_refine_steps,
            refine_mode=refine_mode,
            threshold=threshold,
            crop=True,
            padding=0.1,
        )
        self.ray_caster = DifferentiableRayCaster(ray_cfg)
        self.mask = torch.ones(1, image_size, image_size, dtype=torch.bool, device=device)
        self.ray0, self.ray_dirs, _, _ = DVR.get_rays(self.mask, self.intrinsic, self.extrinsic)
        self.base_color = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32, device=device)
        self.light_dir = F.normalize(torch.tensor([0.25, 0.35, 1.0], dtype=torch.float32, device=device), dim=0)
        self.ambient = 0.35

    def render(
        self,
        predict_fn: Any,
        feature: Tensor,
        points_batch_size: int | None = None,
    ) -> RenderResult:
        h = w = self.image_size
        d_pred, p_pred, mask_pred = self.ray_caster.perform_ray_marching(
            ray0=self.ray0,
            ray_dirs=self.ray_dirs,
            predict=predict_fn,
            feature=feature,
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic,
        )

        white = np.ones((h, w, 3), dtype=np.float32)

        if not mask_pred.any():
            blank = Image.fromarray((white * 255).astype(np.uint8))
            return RenderResult(shading=blank, normals=blank, depth=blank)

        mask_flat_orig = mask_pred[0].detach().cpu().numpy().astype(bool)  # (h*w,)
        mask_flat = mask_flat_orig
        if self.keep_largest_component or self.min_component_pixels > 1:
            filtered = _filter_components_2d(
                mask_flat_orig.reshape(h, w),
                keep_largest=self.keep_largest_component,
                min_pixels=self.min_component_pixels,
            )
            mask_flat = filtered.reshape(-1)
            if not mask_flat.any():
                blank = Image.fromarray((white * 255).astype(np.uint8))
                return RenderResult(shading=blank, normals=blank, depth=blank)
        mask_2d = mask_flat.reshape(h, w)
        mask_torch = torch.from_numpy(mask_flat).to(device=self.device, dtype=torch.bool).unsqueeze(0)

        # --- Depth ---
        depth_flat = d_pred[0].cpu().numpy()  # (h*w,)
        valid_d = depth_flat[mask_flat]
        d_min, d_max = valid_d.min(), valid_d.max()
        depth_rgb = white.copy()
        if d_max > d_min:
            depth_vis = 1.0 - (depth_flat.reshape(h, w) - d_min) / (d_max - d_min)
        else:
            depth_vis = np.ones((h, w), dtype=np.float32)
        for c in range(3):
            depth_rgb[:, :, c][mask_2d] = depth_vis[mask_2d]
        depth_img = Image.fromarray((depth_rgb * 255).astype(np.uint8))

        # --- Normals (via autograd) ---
        hit_points_all = p_pred[mask_pred]
        if not np.array_equal(mask_flat, mask_flat_orig):
            keep_hits = torch.from_numpy(mask_flat[mask_flat_orig]).to(device=hit_points_all.device, dtype=torch.bool)
            hit_points_all = hit_points_all[keep_hits]
        n_hits = hit_points_all.size(0)
        chunk = points_batch_size if points_batch_size else n_hits
        normals_chunks = []
        for start in range(0, n_hits, chunk):
            end = min(start + chunk, n_hits)
            hit_points = hit_points_all[start:end].unsqueeze(0).detach().requires_grad_(True)
            with torch.enable_grad():
                logits = predict_fn(
                    points=hit_points,
                    feature=feature,
                    intrinsic=self.intrinsic,
                    extrinsic=self.extrinsic,
                )
                n = get_normals(hit_points, logits)[0]
            normals_chunks.append(n.detach())
        normals = F.normalize(torch.cat(normals_chunks, dim=0), dim=-1).cpu().numpy()

        normal_map = np.zeros((h, w, 3), dtype=np.float32)
        normal_map[mask_2d] = normals
        normal_map = _smooth_normals_map(
            normal_map,
            mask_2d,
            kernel_size=self.normal_smooth_kernel,
            passes=self.normal_smooth_passes,
        )
        normals_filtered = normal_map[mask_2d]

        # Normal map: [-1,1] -> [0,1]
        normals_vis = (normals_filtered * 0.5 + 0.5).clip(0, 1)
        normal_rgb = white.copy().reshape(-1, 3)
        normal_rgb[mask_flat] = normals_vis
        normal_img = Image.fromarray((normal_rgb.reshape(h, w, 3) * 255).astype(np.uint8))

        # --- Shading ---
        normals_for_shading = torch.from_numpy(normals_filtered).to(device=self.device, dtype=torch.float32)
        diffuse = torch.clamp((normals_for_shading * self.light_dir.view(1, 3)).sum(-1, keepdim=True), min=0.0)
        shade = self.base_color.view(1, 3) * (self.ambient + (1.0 - self.ambient) * diffuse)
        shading_flat = torch.ones(h * w, 3, device=self.device)
        shading_flat[mask_torch[0]] = shade.clamp(0.0, 1.0)
        shading_img = Image.fromarray((shading_flat.view(h, w, 3).cpu().numpy() * 255).astype(np.uint8))

        return RenderResult(shading=shading_img, normals=normal_img, depth=depth_img)


def _prepare_conditioning(conditioning: Tensor | None, inputs: Tensor) -> Tensor | None:
    """Normalize conditioning shape for single-item visualization batches.

    - Class labels should stay 1D/2D (no extra spatial axis).
    - Pointcloud-like conditioning (N, C) is promoted to (1, N, C).
    """
    if conditioning is None:
        return None
    if conditioning.ndim == 0:
        return conditioning.unsqueeze(0)
    if conditioning.ndim == 2 and inputs.ndim == 3 and conditioning.size(-1) == inputs.size(-1):
        return conditioning.unsqueeze(0)
    return conditioning


def _select_input_points_for_visualization(
    inputs: Tensor,
    conditioning: Tensor | None,
    condition_key: str | None,
) -> Tensor:
    """Choose which point cloud to export as input.ply.

    For geometric conditioning (e.g. inputs.depth), show the conditioning cloud.
    Otherwise fall back to the model input cloud.
    """
    if condition_key and condition_key != "category.index" and conditioning is not None:
        if conditioning.ndim == 3 and conditioning.size(-1) == 3:
            return conditioning
        if conditioning.ndim == 2 and conditioning.size(-1) == 3:
            return conditioning.unsqueeze(0)
    return inputs


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg = setup_config(cfg)
    suppress_known_optional_dependency_warnings()

    split = cfg.test.split
    dataset = cast(Any, get_dataset(cfg, splits=(split,))[split])

    vis_cfg = cfg.get("vis", {})
    diff_steps = list(vis_cfg.get("diff_steps", list(range(18, 36))))  # second half of 36 steps
    ar_steps = list(vis_cfg.get("ar_steps", [32, 64, 128, 192, 256, 320, 352, 384, 416, 448, 480, 512]))
    obj_indices_cfg = vis_cfg.get("objects", None)
    if obj_indices_cfg is None:
        obj_indices = _default_object_indices(dataset, limit=5)
    elif isinstance(obj_indices_cfg, int):
        obj_indices = [obj_indices_cfg]
    else:
        obj_indices = [int(i) for i in list(obj_indices_cfg)]
    obj_indices = [i for i in obj_indices if 0 <= i < len(dataset)]
    if not obj_indices:
        raise ValueError("No valid object indices selected for visualization.")

    save_dir = resolve_save_dir(cfg) / "generation_process"
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.model.weights is None and cfg.model.checkpoint is None:
        cfg.model.load_best = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    model.requires_grad_(False)
    model = patch_attention(model, backend="torch")
    fabric = lightning.Fabric(precision=cfg.get(split, cfg.val).precision)
    model = cast(Any, fabric.setup_module(model))

    resolution = int(vis_cfg.get("resolution", 128))
    points_batch_size = int(vis_cfg.get("num_query_points", resolution**3))
    implicit_render = bool(vis_cfg.get("implicit_render", False))

    generator = Generator(
        cast(Any, model),
        points_batch_size=points_batch_size,
        threshold=cfg.implicit.threshold,
        padding=cfg.norm.padding,
        resolution=resolution,
        sdf=cfg.implicit.sdf,
        bounds=cfg.norm.bounds,
    )

    implicit_renderer = None
    if implicit_render:
        implicit_image_size = int(vis_cfg.get("implicit_image_size", 512))
        implicit_steps = int(vis_cfg.get("implicit_num_steps", 64))
        implicit_refine_steps = int(vis_cfg.get("implicit_num_refine_steps", 8))
        implicit_refine_mode_raw = str(vis_cfg.get("implicit_refine_mode", "secant"))
        valid_refine_modes = {"secant", "bisection", "linear", "midpoint"}
        if implicit_refine_mode_raw not in valid_refine_modes:
            raise ValueError(
                f"Unknown vis.implicit_refine_mode={implicit_refine_mode_raw}. "
                "Choose one of: secant,bisection,linear,midpoint"
            )
        implicit_refine_mode = cast(
            Literal["secant", "bisection", "linear", "midpoint"],
            implicit_refine_mode_raw,
        )
        implicit_near = float(vis_cfg.get("implicit_near", 0.0))
        implicit_far = float(vis_cfg.get("implicit_far", 2.4))
        implicit_points_batch_size = int(vis_cfg.get("implicit_points_batch_size", 65536))
        implicit_sdf_tau = float(vis_cfg.get("implicit_sdf_tau", 0.01))
        implicit_sdf_iso = float(vis_cfg.get("implicit_sdf_iso", cfg.implicit.threshold))
        implicit_keep_largest_component = bool(vis_cfg.get("implicit_keep_largest_component", False))
        implicit_min_component_pixels = int(vis_cfg.get("implicit_min_component_pixels", 0))
        implicit_normal_smooth_kernel = int(vis_cfg.get("implicit_normal_smooth_kernel", 0))
        implicit_normal_smooth_passes = int(vis_cfg.get("implicit_normal_smooth_passes", 1))
        # Camera in model's native Y-up space. 45° azimuth with moderate elevation.
        cam_location = tuple(float(x) for x in vis_cfg.get("cam_location", [1.0, 0.8, 1.0]))
        cam_target = tuple(float(x) for x in vis_cfg.get("cam_target", [0.0, 0.0, 0.0]))
        cam_focal = vis_cfg.get("cam_focal", None)
        # For SDF models the adapter -(sdf-iso)/tau already places the zero
        # crossing at 0 in logit space, so threshold=0.5 → log_threshold=0.
        # For occupancy models, use the configured threshold directly.
        implicit_threshold = 0.5 if cfg.implicit.sdf else float(cfg.implicit.threshold)
        implicit_renderer = ImplicitRenderer(
            device=device,
            image_size=implicit_image_size,
            threshold=implicit_threshold,
            near=implicit_near,
            far=implicit_far,
            num_steps=implicit_steps,
            num_refine_steps=implicit_refine_steps,
            refine_mode=implicit_refine_mode,
            cam_location=cast(tuple[float, float, float], cam_location),
            cam_target=cast(tuple[float, float, float], cam_target),
            cam_focal=float(cam_focal) if cam_focal is not None else None,
            keep_largest_component=implicit_keep_largest_component,
            min_component_pixels=implicit_min_component_pixels,
            normal_smooth_kernel=implicit_normal_smooth_kernel,
            normal_smooth_passes=implicit_normal_smooth_passes,
        )
        total_ray_queries = implicit_image_size * implicit_image_size * implicit_steps
        if total_ray_queries > 5_000_000:
            logger.warning(
                "Implicit render settings are expensive "
                f"(image_size^2 * num_steps = {total_ray_queries:,}). "
                "Consider reducing `vis.implicit_image_size` and/or `vis.implicit_num_steps`."
            )
        logger.info(f"Implicit render mode enabled: {implicit_image_size}px, {implicit_steps} steps")

    inner_model = getattr(model, "_forward_module", model)
    is_diffusion = isinstance(inner_model, DiffusionModel)
    is_ar = isinstance(inner_model, AutoregressiveModel)
    if not (is_diffusion or is_ar):
        raise ValueError(f"Model must be DiffusionModel or AutoregressiveModel, got {type(inner_model).__name__}")
    condition_key = getattr(model, "condition_key", None)
    is_unconditional = condition_key is None or condition_key == ""

    for idx in tqdm(obj_indices, desc="Processing objects"):
        item = dataset[idx]
        obj_name = str(item.get("inputs.name", f"obj_{idx}"))
        obj_category = str(item.get("category.id", "unknown"))

        # Avoid overwriting when multiple views/files of the same object are selected
        # (e.g., Automatica categories with several depth frames).
        view_suffix = ""
        file_idx = item.get("inputs.file")
        if isinstance(file_idx, (int, np.integer)):
            view_suffix = f"_f{int(file_idx):05d}"
        else:
            input_path = item.get("inputs.path")
            if input_path is not None:
                stem = Path(str(input_path)).stem
                if stem.isdigit():
                    view_suffix = f"_f{int(stem):05d}"
        if view_suffix:
            obj_name = f"{obj_name}{view_suffix}"

        out_dir = save_dir / obj_category / obj_name
        out_dir.mkdir(parents=True, exist_ok=True)

        inputs = to_tensor(item["inputs"], device=device)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)

        input_points_for_refs: np.ndarray | None = None
        conditioning = None
        if condition_key:
            conditioning = to_tensor(item.get(condition_key), device=device)
            conditioning = _prepare_conditioning(conditioning, inputs)

        # For unconditional models, there is no input to visualize.
        if not is_unconditional:
            input_points = _select_input_points_for_visualization(inputs, conditioning, condition_key)
            input_pts = input_points.squeeze(0).cpu().numpy()
            input_points_for_refs = input_pts
            trimesh.PointCloud(input_pts).export(out_dir / "input.ply")

        # Save GT mesh if available
        gt_mesh: trimesh.Trimesh | None = None
        if "mesh.vertices" in item and "mesh.triangles" in item:
            gt_mesh = trimesh.Trimesh(
                np.asarray(item["mesh.vertices"]),
                np.asarray(item["mesh.triangles"]),
                process=False,
            )
            gt_mesh.export(out_dir / "gt.ply")

        with fabric.autocast(), torch.no_grad():
            if is_diffusion:
                method_dir = out_dir / "diffusion"
                method_dir.mkdir(exist_ok=True)
                if implicit_render:
                    assert implicit_renderer is not None
                    implicit_dir = method_dir / "implicit"
                    implicit_dir.mkdir(exist_ok=True)
                    _save_implicit_reference_panels(
                        implicit_dir,
                        implicit_renderer,
                        input_points_for_refs,
                        gt_mesh,
                    )

                num_steps = int(vis_cfg.get("num_steps", 36))
                result = model.generate(
                    inputs=inputs if (not is_unconditional and inputs.size(-1) == 3) else None,
                    points=generator.query_points.unsqueeze(0).to(device),
                    conditioning=conditioning,
                    num_steps=num_steps,
                    points_batch_size=points_batch_size,
                    progress=True,
                    return_intermediates=True,
                )
                _logits, intermediates = result

                # Select and decode requested steps
                for step_idx in diff_steps:
                    if step_idx >= len(intermediates):
                        logger.warning(f"Step {step_idx} >= num intermediates {len(intermediates)}, skipping")
                        continue
                    latent = intermediates[step_idx]
                    if implicit_render:
                        assert implicit_renderer is not None
                        feature = decode_diffusion_feature(model, latent)
                        predict_fn = partial(
                            _implicit_predict,
                            predictor=model._vae,
                            points_batch_size=implicit_points_batch_size,
                            sdf=cfg.implicit.sdf,
                            sdf_tau=implicit_sdf_tau,
                            sdf_iso=implicit_sdf_iso,
                        )
                        result = implicit_renderer.render(
                            predict_fn, feature, points_batch_size=implicit_points_batch_size
                        )
                        stem = f"step_{step_idx:02d}"
                        result.shading.save(implicit_dir / f"{stem}.png")
                        result.normals.save(implicit_dir / f"{stem}_normals.png")
                        result.depth.save(implicit_dir / f"{stem}_depth.png")
                        logger.info(f"Saved diffusion step {step_idx} -> {implicit_dir / stem}.png")
                    else:
                        mesh = decode_diffusion_intermediate(model, latent, generator, points_batch_size)
                        if len(mesh.vertices) > 0:
                            out_path = method_dir / f"step_{step_idx:02d}.ply"
                            save_mesh(out_path, mesh.vertices, mesh.faces)
                            logger.info(f"Saved diffusion step {step_idx} -> {out_path}")
                        else:
                            logger.warning(f"Empty mesh at diffusion step {step_idx}")

            elif is_ar:
                method_dir = out_dir / "ar"
                method_dir.mkdir(exist_ok=True)
                if implicit_render:
                    assert implicit_renderer is not None
                    implicit_dir = method_dir / "implicit"
                    implicit_dir.mkdir(exist_ok=True)
                    _save_implicit_reference_panels(
                        implicit_dir,
                        implicit_renderer,
                        input_points_for_refs,
                        gt_mesh,
                    )

                temperature = float(vis_cfg.get("temperature", 1.0))
                result = model.generate(
                    inputs=inputs if (not is_unconditional and inputs.size(-1) == 3) else None,
                    points=generator.query_points.unsqueeze(0).to(device),
                    conditioning=conditioning,
                    temperature=temperature,
                    points_batch_size=points_batch_size,
                    progress=True,
                    return_intermediates=True,
                )
                _logits, intermediates = result

                n_total = model._autoregressor.n_block
                for token_count in ar_steps:
                    # intermediates[i] has i+1 tokens
                    inter_idx = token_count - 1
                    if inter_idx < 0 or inter_idx >= len(intermediates):
                        logger.warning(f"Token count {token_count} out of range [1, {len(intermediates)}], skipping")
                        continue
                    partial_seq = intermediates[inter_idx]
                    if implicit_render:
                        assert implicit_renderer is not None
                        feature = decode_ar_feature(model, partial_seq, n_total)
                        predict_fn = partial(
                            _implicit_predict,
                            predictor=model._discretizer,
                            points_batch_size=implicit_points_batch_size,
                            sdf=cfg.implicit.sdf,
                            sdf_tau=implicit_sdf_tau,
                            sdf_iso=implicit_sdf_iso,
                        )
                        result = implicit_renderer.render(
                            predict_fn, feature, points_batch_size=implicit_points_batch_size
                        )
                        stem = f"token_{token_count:03d}"
                        result.shading.save(implicit_dir / f"{stem}.png")
                        result.normals.save(implicit_dir / f"{stem}_normals.png")
                        result.depth.save(implicit_dir / f"{stem}_depth.png")
                        logger.info(f"Saved AR token {token_count} -> {implicit_dir / stem}.png")
                    else:
                        mesh = decode_ar_intermediate(model, partial_seq, n_total, generator, points_batch_size)
                        if len(mesh.vertices) > 0:
                            out_path = method_dir / f"token_{token_count:03d}.ply"
                            save_mesh(out_path, mesh.vertices, mesh.faces)
                            logger.info(f"Saved AR token {token_count} -> {out_path}")
                        else:
                            logger.warning(f"Empty mesh at AR token {token_count}")

    logger.info(f"Done. Output saved to {save_dir}")


if __name__ == "__main__":
    main()
