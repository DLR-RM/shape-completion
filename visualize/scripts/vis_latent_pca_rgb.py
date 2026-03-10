"""Visualize diffusion latent evolution by mapping PCA components to RGB.

This script runs diffusion generation with ``return_intermediates=True`` and exports
one colored point cloud per selected denoising step. Colors are obtained by fitting
PCA (3 components) jointly over all selected steps for each object.

Output structure:
    <base>/<category>/<object>/diffusion_pca_rgb/
        step_00_pca_rgb.ply
        step_06_pca_rgb.ply
        ...
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import lightning
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig
from PIL import Image
from sklearn.decomposition import PCA
from torch import Tensor
from tqdm import tqdm

from dataset import get_dataset
from libs import furthest_point_sample
from models import AutoregressiveModel, DiffusionModel, get_model, patch_attention
from utils import convert_extrinsic, inv_trafo, look_at, resolve_save_dir, setup_config, setup_logger, to_tensor

logger = setup_logger(__name__)


def _default_object_indices(dataset: Any, limit: int = 5) -> list[int]:
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


def _prepare_conditioning(conditioning: Tensor | None, inputs: Tensor) -> Tensor | None:
    if conditioning is None:
        return None
    if conditioning.ndim == 0:
        return conditioning.unsqueeze(0)
    if conditioning.ndim == 2 and inputs.ndim == 3 and conditioning.size(-1) == inputs.size(-1):
        return conditioning.unsqueeze(0)
    return conditioning


def _normalize_to_uint8(
    x: np.ndarray,
    *,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    chroma_boost: float = 1.15,
) -> np.ndarray:
    lo = np.percentile(x, clip_low, axis=0, keepdims=True)
    hi = np.percentile(x, clip_high, axis=0, keepdims=True)
    degenerate = (hi - lo) < 1e-8
    if np.any(degenerate):
        x_min = x.min(axis=0, keepdims=True)
        x_max = x.max(axis=0, keepdims=True)
        lo = np.where(degenerate, x_min, lo)
        hi = np.where(degenerate, x_max, hi)

    denom = np.maximum(hi - lo, 1e-8)
    y = (np.clip(x, lo, hi) - lo) / denom
    y = np.clip((y - 0.5) * float(chroma_boost) + 0.5, 0.0, 1.0)
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def _extract_query_points(inputs: Tensor, n_queries: int) -> np.ndarray:
    sampled = cast(Tensor, furthest_point_sample(inputs, n_queries))
    return sampled.squeeze(0).detach().cpu().numpy().astype(np.float32)


@dataclass
class StepLatents:
    steps: list[int]
    latents: list[np.ndarray]


def _collect_diffusion_step_latents(
    model: Any,
    inputs: Tensor,
    conditioning: Tensor | None,
    num_steps: int,
    selected_steps: list[int],
) -> StepLatents:
    # Keep decode-time points tiny; we only need intermediates for PCA.
    points_for_decode = inputs[:, : min(inputs.size(1), 64), :]
    _logits, intermediates = model.generate(
        inputs=inputs,
        points=points_for_decode,
        conditioning=conditioning,
        num_steps=num_steps,
        points_batch_size=points_for_decode.size(1),
        progress=True,
        return_intermediates=True,
    )

    steps: list[int] = []
    latents: list[np.ndarray] = []
    for step in selected_steps:
        if step < 0 or step >= len(intermediates):
            logger.warning(f"Skipping invalid step {step} (valid range: 0..{len(intermediates) - 1})")
            continue
        latent = intermediates[step].squeeze(0).detach().cpu().numpy().astype(np.float32)
        steps.append(step)
        latents.append(latent)

    return StepLatents(steps=steps, latents=latents)


def _decode_ar_feature(model: Any, partial_indices: Tensor, n_total: int) -> Tensor:
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


def _collect_ar_token_latents(
    model: Any,
    inputs: Tensor,
    conditioning: Tensor | None,
    selected_tokens: list[int],
    temperature: float,
) -> StepLatents:
    points_for_decode = inputs[:, : min(inputs.size(1), 64), :]
    _logits, intermediates = model.generate(
        inputs=inputs,
        points=points_for_decode,
        conditioning=conditioning,
        temperature=temperature,
        points_batch_size=points_for_decode.size(1),
        progress=True,
        return_intermediates=True,
    )

    n_total = int(model._autoregressor.n_block)
    steps: list[int] = []
    latents: list[np.ndarray] = []
    for token_count in selected_tokens:
        inter_idx = int(token_count) - 1
        if inter_idx < 0 or inter_idx >= len(intermediates):
            logger.warning(f"Skipping invalid token count {token_count} (valid range: 1..{len(intermediates)})")
            continue
        partial_seq = intermediates[inter_idx]
        feature = _decode_ar_feature(model, partial_seq, n_total)
        latent = feature.squeeze(0).detach().cpu().numpy().astype(np.float32)
        steps.append(int(token_count))
        latents.append(latent)

    return StepLatents(steps=steps, latents=latents)


def _write_colored_point_cloud(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    rgba = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    pcd = trimesh.PointCloud(vertices=points, colors=rgba)
    pcd.export(path)


def _write_token_strip(
    path: Path,
    rgb: np.ndarray,
    *,
    strip_height: int = 24,
    token_width: int = 4,
) -> None:
    h = max(int(strip_height), 1)
    w = max(int(token_width), 1)
    # One-row strip: token i is a solid rectangle with its RGB color.
    strip = np.repeat(rgb[np.newaxis, :, :], h, axis=0)              # (h, N, 3)
    strip = np.repeat(strip, w, axis=1)                              # (h, N*w, 3)
    Image.fromarray(strip, mode="RGB").save(path)


def _project_colored_points_to_image(
    points: np.ndarray,
    rgb: np.ndarray,
    *,
    image_size: int,
    cam_location: tuple[float, float, float],
    cam_target: tuple[float, float, float],
    cam_focal: float | None,
    point_radius: int,
) -> Image.Image:
    canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
    if points.size == 0:
        return Image.fromarray(canvas, mode="RGB")

    focal = float(cam_focal) if cam_focal is not None else float(image_size)
    intrinsic = np.array(
        [[focal, 0.0, image_size / 2.0], [0.0, focal, image_size / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    pose = look_at(np.asarray(cam_location, dtype=np.float32), np.asarray(cam_target, dtype=np.float32))
    extrinsic_gl = np.asarray(inv_trafo(pose), dtype=np.float32)
    extrinsic = np.asarray(convert_extrinsic(extrinsic_gl, "opengl", "opencv"), dtype=np.float32)

    cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z = cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return Image.fromarray(canvas, mode="RGB")

    cam = cam[valid]
    z = z[valid]
    colors = rgb[valid]

    proj = cam @ intrinsic.T
    u = np.rint(proj[:, 0] / np.maximum(proj[:, 2], 1e-6)).astype(np.int32)
    v = np.rint(proj[:, 1] / np.maximum(proj[:, 2], 1e-6)).astype(np.int32)

    in_bounds = (u >= 0) & (u < image_size) & (v >= 0) & (v < image_size)
    if not np.any(in_bounds):
        return Image.fromarray(canvas, mode="RGB")

    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    colors = colors[in_bounds]

    r = max(int(point_radius), 0)
    order = np.argsort(z)[::-1]  # far -> near
    for idx in order:
        cx = int(u[idx])
        cy = int(v[idx])
        color = colors[idx]
        x0 = max(cx - r, 0)
        x1 = min(cx + r + 1, image_size)
        y0 = max(cy - r, 0)
        y1 = min(cy + r + 1, image_size)
        canvas[y0:y1, x0:x1] = color

    return Image.fromarray(canvas, mode="RGB")


def _save_mp4(frames: list[Image.Image], path: Path, fps: float) -> bool:
    if shutil.which("ffmpeg") is None:
        logger.warning(f"ffmpeg not found; skipping MP4 export for {path}")
        return False

    with tempfile.TemporaryDirectory(prefix="vis_latent_pca_rgb_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for i, frame in enumerate(frames):
            frame.save(tmp_path / f"frame_{i:05d}.png")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.3f}",
            "-i",
            str(tmp_path / "frame_%05d.png"),
            "-vf",
            "format=yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err_lines = (result.stderr or "").strip().splitlines()
            tail = err_lines[-5:] if err_lines else ["unknown ffmpeg error"]
            logger.warning(f"ffmpeg failed for {path}:")
            for line in tail:
                logger.warning(f"  {line}")
            return False
    return True


def _export_projection_animations(
    frame_paths: list[Path],
    out_dir: Path,
    name_prefix: str,
    *,
    animation_format: str,
    gif_duration_ms: int,
    gif_loop: int,
    mp4_fps: float | None,
) -> None:
    if not frame_paths or animation_format == "none":
        return

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    try:
        if animation_format in {"gif", "both"}:
            gif_path = out_dir / f"{name_prefix}_pca_rgb_proj.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=gif_duration_ms,
                loop=gif_loop,
                disposal=2,
            )
            logger.info(f"Saved {gif_path}")

        if animation_format in {"mp4", "both"}:
            fps = float(mp4_fps) if mp4_fps is not None else (1000.0 / max(int(gif_duration_ms), 1))
            mp4_path = out_dir / f"{name_prefix}_pca_rgb_proj.mp4"
            if _save_mp4(frames, mp4_path, fps):
                logger.info(f"Saved {mp4_path}")
    finally:
        for frame in frames:
            frame.close()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)

    split = cfg.test.split
    dataset = cast(Any, get_dataset(cfg, splits=(split,))[split])

    vis_cfg = cfg.get("vis", {})
    num_steps = int(vis_cfg.get("num_steps", 36))
    pca_method = str(vis_cfg.get("pca_method", "both")).lower()
    selected_steps_cfg = vis_cfg.get("pca_rgb_steps", [0, 6, 12, 17, 24, 30, 35])
    selected_steps = [int(s) for s in list(selected_steps_cfg)]
    selected_tokens_cfg = vis_cfg.get("pca_rgb_ar_steps", [1, 64, 128, 192, 256, 320, 352, 384, 416, 448, 480, 512])
    selected_tokens = [int(t) for t in list(selected_tokens_cfg)]
    ar_temperature = float(vis_cfg.get("temperature", 1.0))
    pca_proj_size = int(vis_cfg.get("pca_proj_size", 512))
    pca_proj_radius = int(vis_cfg.get("pca_proj_radius", 2))
    pca_proj_cam_location = tuple(float(x) for x in vis_cfg.get("pca_proj_cam_location", [1.0, 0.8, 1.0]))
    pca_proj_cam_target = tuple(float(x) for x in vis_cfg.get("pca_proj_cam_target", [0.0, 0.0, 0.0]))
    pca_proj_cam_focal_cfg = vis_cfg.get("pca_proj_cam_focal", None)
    pca_proj_cam_focal = float(pca_proj_cam_focal_cfg) if pca_proj_cam_focal_cfg is not None else None
    pca_proj_animation_format = str(vis_cfg.get("pca_proj_animation_format", "both")).lower()
    pca_proj_gif_duration_ms = int(vis_cfg.get("pca_proj_gif_duration_ms", 120))
    pca_proj_gif_loop = int(vis_cfg.get("pca_proj_gif_loop", 0))
    pca_proj_mp4_fps_cfg = vis_cfg.get("pca_proj_mp4_fps", None)
    pca_proj_mp4_fps = float(pca_proj_mp4_fps_cfg) if pca_proj_mp4_fps_cfg is not None else None
    pca_rgb_clip_low = float(vis_cfg.get("pca_rgb_clip_low", 1.0))
    pca_rgb_clip_high = float(vis_cfg.get("pca_rgb_clip_high", 99.0))
    pca_rgb_chroma_boost = float(vis_cfg.get("pca_rgb_chroma_boost", 1.15))
    if pca_proj_animation_format not in {"none", "gif", "mp4", "both"}:
        raise ValueError(
            f"Unknown vis.pca_proj_animation_format={pca_proj_animation_format}. Choose one of: none,gif,mp4,both"
        )

    objects_cfg = vis_cfg.get("objects", None)
    if objects_cfg is None:
        object_indices = _default_object_indices(dataset, limit=5)
    elif isinstance(objects_cfg, int):
        object_indices = [int(objects_cfg)]
    else:
        object_indices = [int(i) for i in list(objects_cfg)]
    object_indices = [i for i in object_indices if 0 <= i < len(dataset)]
    if not object_indices:
        raise ValueError("No valid object indices selected.")

    if cfg.model.weights is None and cfg.model.checkpoint is None:
        cfg.model.load_best = True

    base_out = vis_cfg.get("pca_rgb_output_dir", None)
    save_dir = Path(str(base_out)) if base_out else resolve_save_dir(cfg) / "generation_process"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    model.requires_grad_(False)
    model = patch_attention(model, backend="torch")

    fabric = lightning.Fabric(precision=cfg.get(split, cfg.val).precision)
    model = cast(Any, fabric.setup_module(model))

    inner_model = getattr(model, "_forward_module", model)
    is_diffusion = isinstance(inner_model, DiffusionModel)
    is_ar = isinstance(inner_model, AutoregressiveModel)
    if not (is_diffusion or is_ar):
        raise ValueError(f"Expected DiffusionModel or AutoregressiveModel, got {type(inner_model).__name__}")
    if pca_method not in {"auto", "diffusion", "ar", "both"}:
        raise ValueError(f"Unknown vis.pca_method={pca_method}. Choose one of: auto,diffusion,ar,both")

    run_diffusion = (pca_method in {"auto", "diffusion", "both"}) and is_diffusion
    run_ar = (pca_method in {"auto", "ar", "both"}) and is_ar
    if not (run_diffusion or run_ar):
        raise ValueError(
            f"Requested vis.pca_method={pca_method} incompatible with model type {type(inner_model).__name__}"
        )

    condition_key = getattr(model, "condition_key", None)

    for idx in tqdm(object_indices, desc="PCA-RGB objects"):
        item = dataset[idx]
        obj_name = str(item.get("inputs.name", f"obj_{idx}"))
        obj_category = str(item.get("category.id", "unknown"))

        inputs = to_tensor(item["inputs"], device=device)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)

        conditioning = None
        if condition_key:
            conditioning = to_tensor(item.get(condition_key), device=device)
            conditioning = _prepare_conditioning(conditioning, inputs)

        modalities: list[tuple[str, StepLatents, str]] = []
        with fabric.autocast(), torch.no_grad():
            if run_diffusion:
                latents = _collect_diffusion_step_latents(
                    model=model,
                    inputs=inputs,
                    conditioning=conditioning,
                    num_steps=num_steps,
                    selected_steps=selected_steps,
                )
                modalities.append(("diffusion_pca_rgb", latents, "step"))
            if run_ar:
                latents = _collect_ar_token_latents(
                    model=model,
                    inputs=inputs,
                    conditioning=conditioning,
                    selected_tokens=selected_tokens,
                    temperature=ar_temperature,
                )
                modalities.append(("ar_pca_rgb", latents, "token"))

        for subdir, step_latents, name_prefix in modalities:
            out_dir = save_dir / obj_category / obj_name / subdir
            out_dir.mkdir(parents=True, exist_ok=True)

            if not step_latents.latents:
                logger.warning(f"No latents collected for object {obj_name} ({subdir})")
                continue

            cleaned_entries: list[tuple[int, np.ndarray, np.ndarray]] = []
            for step, latent in zip(step_latents.steps, step_latents.latents, strict=True):
                finite_mask = np.asarray(np.isfinite(latent).all(axis=1), dtype=bool)
                if finite_mask.ndim == 0:
                    finite_mask = np.full((latent.shape[0],), bool(finite_mask), dtype=bool)
                invalid_count = int((~finite_mask).sum())
                if invalid_count > 0:
                    logger.warning(f"{name_prefix} {step}: dropping {invalid_count} non-finite latent rows for {obj_name}")
                if not finite_mask.any():
                    logger.warning(f"{name_prefix} {step}: no finite latents left for {obj_name}, skipping")
                    continue
                cleaned_entries.append((step, latent[finite_mask], finite_mask))

            if not cleaned_entries:
                logger.warning(f"All selected {name_prefix}s became invalid for object {obj_name} ({subdir})")
                continue

            all_latents = np.concatenate([latent for _step, latent, _mask in cleaned_entries], axis=0)
            if all_latents.shape[0] < 3:
                logger.warning(
                    f"Not enough finite latent rows for PCA (need >=3, got {all_latents.shape[0]}) for object {obj_name} ({subdir})"
                )
                continue
            pca = PCA(n_components=3)
            projected_all = pca.fit_transform(all_latents)
            rgb_all = _normalize_to_uint8(
                projected_all,
                clip_low=pca_rgb_clip_low,
                clip_high=pca_rgb_clip_high,
                chroma_boost=pca_rgb_chroma_boost,
            )

            n_queries = step_latents.latents[0].shape[0]
            query_points = _extract_query_points(inputs, n_queries=n_queries)

            offset = 0
            proj_frames: list[Path] = []
            for step, latent, finite_mask in cleaned_entries:
                n = latent.shape[0]
                rgb = rgb_all[offset : offset + n]
                offset += n

                full_rgb = np.full((finite_mask.shape[0], 3), 127, dtype=np.uint8)
                full_rgb[finite_mask] = rgb

                strip_path = out_dir / f"{name_prefix}_{step:03d}_pca_rgb_strip.png"
                _write_token_strip(
                    strip_path,
                    full_rgb,
                    strip_height=int(vis_cfg.get("pca_strip_height", 48)),
                    token_width=int(vis_cfg.get("pca_strip_token_width", 4)),
                )
                logger.info(f"Saved {strip_path}")

                points = query_points[finite_mask][:n]
                spatial_path = out_dir / f"{name_prefix}_{step:03d}_pca_rgb.ply"
                _write_colored_point_cloud(spatial_path, points, rgb)
                logger.info(f"Saved {spatial_path}")

                proj_path = out_dir / f"{name_prefix}_{step:03d}_pca_rgb_proj.png"
                proj_img = _project_colored_points_to_image(
                    points,
                    rgb,
                    image_size=pca_proj_size,
                    cam_location=cast(tuple[float, float, float], pca_proj_cam_location),
                    cam_target=cast(tuple[float, float, float], pca_proj_cam_target),
                    cam_focal=pca_proj_cam_focal,
                    point_radius=pca_proj_radius,
                )
                proj_img.save(proj_path)
                logger.info(f"Saved {proj_path}")
                proj_frames.append(proj_path)

            _export_projection_animations(
                proj_frames,
                out_dir,
                name_prefix=name_prefix,
                animation_format=pca_proj_animation_format,
                gif_duration_ms=pca_proj_gif_duration_ms,
                gif_loop=pca_proj_gif_loop,
                mp4_fps=pca_proj_mp4_fps,
            )

    logger.info(f"Done. Outputs in {save_dir}")


if __name__ == "__main__":
    main()
