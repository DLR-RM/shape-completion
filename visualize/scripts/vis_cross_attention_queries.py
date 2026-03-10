"""Visualize input-point ownership by encoder cross-attention queries.

This script extracts cross-attention weights from the first `inputs_enc` block of
Shape3D2VecSet-style encoders and colors input points by the query with maximum
attention mass.

Output structure:
    <base>/<category>/<object>/cross_attention/
        query_ownership_input_points.ply
        uncertainty_input_points.ply
        queries_fps.ply
        query_ownership_input_points_proj.png
        uncertainty_input_points_proj.png
        attention_weights.npz
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
import lightning
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from dataset import get_dataset
from libs import furthest_point_sample
from models import get_model, patch_attention
from models.src.shape3d2vecset import Shape3D2VecSet
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


def _resolve_shape3d2vecset_encoder(model: Any) -> Shape3D2VecSet:
    candidates: list[Any] = []

    if hasattr(model, "_vae") and hasattr(model._vae, "ae"):
        candidates.append(model._vae.ae)
    if hasattr(model, "_discretizer") and hasattr(model._discretizer, "ae"):
        candidates.append(model._discretizer.ae)
    if hasattr(model, "ae"):
        candidates.append(model.ae)

    for cand in candidates:
        if isinstance(cand, Shape3D2VecSet):
            return cand

    raise ValueError("Could not find Shape3D2VecSet encoder in model (_vae.ae / _discretizer.ae / ae).")


def _first_inputs_enc_block(ae: Shape3D2VecSet) -> Any:
    if isinstance(ae.inputs_enc, torch.nn.ModuleList):
        return ae.inputs_enc[0]
    return ae.inputs_enc


def _to_query_colors(n_queries: int) -> np.ndarray:
    idx = np.arange(n_queries, dtype=np.float32)
    # Deterministic HSV-like wheel without extra dependencies.
    r = (0.5 + 0.5 * np.sin(2.0 * np.pi * idx / max(n_queries, 1) + 0.0))
    g = (0.5 + 0.5 * np.sin(2.0 * np.pi * idx / max(n_queries, 1) + 2.0943951))
    b = (0.5 + 0.5 * np.sin(2.0 * np.pi * idx / max(n_queries, 1) + 4.1887902))
    rgb = np.stack([r, g, b], axis=1)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def _write_colored_point_cloud(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    rgba = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    pcd = trimesh.PointCloud(vertices=points.astype(np.float32), colors=rgba)
    pcd.export(path)


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


def _uncertainty_colors_from_entropy(entropy_norm: np.ndarray) -> np.ndarray:
    # Low uncertainty -> dark blue, high uncertainty -> yellow.
    e = np.clip(entropy_norm.astype(np.float32), 0.0, 1.0)
    r = e
    g = e
    b = 1.0 - 0.7 * e
    rgb = np.stack([r, g, b], axis=1)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def _compute_attention_weights(
    ae: Shape3D2VecSet, inputs: Tensor
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    block = _first_inputs_enc_block(ae)
    cross_attn = block.cross_attn
    if cross_attn is None:
        raise RuntimeError("inputs_enc block has no cross-attention module")

    n_queries = int(ae.query_levels[0])
    inputs_fps = cast(Tensor, furthest_point_sample(inputs, n_queries))
    queries_nerf = ae.nerf_enc(inputs_fps)
    inputs_nerf = ae.nerf_enc(inputs)

    q_in = block.ln_2(queries_nerf)
    q = cross_attn.to_q(q_in)

    if hasattr(cross_attn, "to_kv"):
        k, _v = cross_attn.to_kv(inputs_nerf).chunk(2, dim=2)
    elif hasattr(cross_attn, "to_k"):
        k = cross_attn.to_k(inputs_nerf)
    else:
        raise RuntimeError("Unsupported cross-attn projection layout (expected to_kv or to_k)")

    h = int(cross_attn.n_head)
    d = q.shape[-1] // h
    qh = q.view(q.shape[0], q.shape[1], h, d).permute(0, 2, 1, 3)  # [B,H,Q,D]
    kh = k.view(k.shape[0], k.shape[1], h, d).permute(0, 2, 1, 3)  # [B,H,N,D]

    scale = float(d) ** -0.5
    attn_logits = torch.matmul(qh * scale, kh.transpose(-2, -1))
    attn = torch.softmax(attn_logits, dim=-1)  # [B,H,Q,N]
    attn_mean = attn.mean(dim=1)[0]  # [Q,N]

    owners = torch.argmax(attn_mean, dim=0).detach().cpu().numpy().astype(np.int32)
    probs = attn_mean.transpose(0, 1)  # [N,Q]
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=1)  # [N]
    entropy_norm = entropy / max(np.log(max(n_queries, 2)), 1e-12)
    entropy_norm_np = entropy_norm.detach().cpu().numpy().astype(np.float32)

    top2 = torch.topk(probs, k=min(2, probs.size(1)), dim=1).values
    if top2.size(1) == 2:
        margin = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy().astype(np.float32)
    else:
        margin = np.ones((probs.size(0),), dtype=np.float32)

    return (
        inputs.squeeze(0).detach().cpu().numpy().astype(np.float32),
        inputs_fps.squeeze(0).detach().cpu().numpy().astype(np.float32),
        attn_mean.detach().cpu().numpy().astype(np.float32),
    ), owners, entropy_norm_np, margin


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)

    split = cfg.test.split
    dataset = cast(Any, get_dataset(cfg, splits=(split,))[split])

    vis_cfg = cfg.get("vis", {})
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

    max_input_points = int(vis_cfg.get("max_input_points", 2048))
    proj_size = int(vis_cfg.get("cross_attn_proj_size", 512))
    proj_radius = int(vis_cfg.get("cross_attn_proj_radius", 2))
    proj_cam_location = tuple(float(x) for x in vis_cfg.get("cross_attn_proj_cam_location", [1.0, 0.8, 1.0]))
    proj_cam_target = tuple(float(x) for x in vis_cfg.get("cross_attn_proj_cam_target", [0.0, 0.0, 0.0]))
    proj_cam_focal_cfg = vis_cfg.get("cross_attn_proj_cam_focal", None)
    proj_cam_focal = float(proj_cam_focal_cfg) if proj_cam_focal_cfg is not None else None

    if cfg.model.weights is None and cfg.model.checkpoint is None:
        cfg.model.load_best = True

    base_out = vis_cfg.get("cross_attn_output_dir", None)
    save_dir = Path(str(base_out)) if base_out else resolve_save_dir(cfg) / "generation_process"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    model.requires_grad_(False)
    model = patch_attention(model, backend="torch")

    fabric = lightning.Fabric(precision=cfg.get(split, cfg.val).precision)
    model = cast(Any, fabric.setup_module(model))

    ae = _resolve_shape3d2vecset_encoder(model)
    ae.eval()

    for idx in tqdm(object_indices, desc="Cross-attention ownership"):
        item = dataset[idx]
        obj_name = str(item.get("inputs.name", f"obj_{idx}"))
        obj_category = str(item.get("category.id", "unknown"))

        out_dir = save_dir / obj_category / obj_name / "cross_attention"
        out_dir.mkdir(parents=True, exist_ok=True)

        inputs = to_tensor(item["inputs"], device=device)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)

        if inputs.size(1) > max_input_points:
            # Deterministic subsampling by uniform index spacing.
            sel = torch.linspace(0, inputs.size(1) - 1, steps=max_input_points, device=device).long()
            inputs = inputs[:, sel, :]

        with fabric.autocast(), torch.no_grad():
            (input_points, query_points, attn_mean), owners, entropy_norm, top1_top2_margin = _compute_attention_weights(
                ae, inputs
            )

        query_colors = _to_query_colors(query_points.shape[0])
        input_colors = query_colors[owners]
        uncertainty_colors = _uncertainty_colors_from_entropy(entropy_norm)

        _write_colored_point_cloud(out_dir / "query_ownership_input_points.ply", input_points, input_colors)
        _write_colored_point_cloud(out_dir / "queries_fps.ply", query_points, query_colors)
        _write_colored_point_cloud(out_dir / "uncertainty_input_points.ply", input_points, uncertainty_colors)

        ownership_proj = _project_colored_points_to_image(
            input_points,
            input_colors,
            image_size=proj_size,
            cam_location=cast(tuple[float, float, float], proj_cam_location),
            cam_target=cast(tuple[float, float, float], proj_cam_target),
            cam_focal=proj_cam_focal,
            point_radius=proj_radius,
        )
        ownership_proj_path = out_dir / "query_ownership_input_points_proj.png"
        ownership_proj.save(ownership_proj_path)

        uncertainty_proj = _project_colored_points_to_image(
            input_points,
            uncertainty_colors,
            image_size=proj_size,
            cam_location=cast(tuple[float, float, float], proj_cam_location),
            cam_target=cast(tuple[float, float, float], proj_cam_target),
            cam_focal=proj_cam_focal,
            point_radius=proj_radius,
        )
        uncertainty_proj_path = out_dir / "uncertainty_input_points_proj.png"
        uncertainty_proj.save(uncertainty_proj_path)

        np.savez_compressed(
            out_dir / "attention_weights.npz",
            attention_qn=attn_mean,
            owner_per_input=owners,
            uncertainty_entropy_norm=entropy_norm,
            uncertainty_top1_top2_margin=top1_top2_margin,
        )
        logger.info(f"Saved cross-attention ownership for {obj_category}/{obj_name}")

    logger.info(f"Done. Outputs in {save_dir}")


if __name__ == "__main__":
    main()
