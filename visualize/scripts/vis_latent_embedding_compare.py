"""Compare VAE vs VQ-VAE latent spaces using 2D projections.

For each selected dataset sample, this script encodes the input point cloud with
both a VAE and a VQ-VAE, mean-pools latents to one vector per shape, projects
to 2D (UMAP preferred, t-SNE fallback), and saves side-by-side scatter plots.

Output layout:
    <output>/latent_embedding_compare_<method>.png
    <output>/latent_embedding_compare_<method>.npz
    <output>/latent_embedding_compare_<method>.csv (optional)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, cast

import hydra
import lightning
import matplotlib
import numpy as np
import torch
from matplotlib.lines import Line2D
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor
from tqdm import tqdm

from dataset import get_dataset
from models import get_model, patch_attention
from utils import resolve_save_dir, setup_config, setup_logger, suppress_known_optional_dependency_warnings, to_tensor

matplotlib.use("Agg")
from matplotlib import pyplot as plt

logger = setup_logger(__name__)


def _default_object_indices(dataset: Any, limit: int | None = None) -> list[int]:
    total = len(dataset)
    if total <= 0:
        return []

    max_count = total if limit is None else min(limit, total)
    objects = getattr(dataset, "objects", None)
    if isinstance(objects, list) and len(objects) == total:
        keyed: list[tuple[str, str, int]] = []
        for idx, obj in enumerate(objects):
            if isinstance(obj, dict):
                category = str(obj.get("category", ""))
                name = str(obj.get("name", obj.get("obj_name", "")))
            else:
                category = ""
                name = str(obj)
            keyed.append((category, name, idx))
        keyed.sort()
        return [idx for _cat, _name, idx in keyed[:max_count]]

    return list(range(max_count))


def _resolve_indices(dataset: Any, vis_cfg: Any) -> list[int]:
    obj_indices_cfg = vis_cfg.get("objects", None)
    max_objects_cfg = vis_cfg.get("max_objects", 2000)
    max_objects = int(max_objects_cfg) if max_objects_cfg is not None else None

    if obj_indices_cfg is None:
        indices = _default_object_indices(dataset, limit=max_objects)
    elif isinstance(obj_indices_cfg, int):
        indices = [int(obj_indices_cfg)]
    else:
        indices = [int(i) for i in list(obj_indices_cfg)]

    indices = [idx for idx in indices if 0 <= idx < len(dataset)]
    if not indices:
        return []

    categories_cfg = vis_cfg.get("categories", None)
    if categories_cfg is None:
        return indices

    allowed = {str(c) for c in list(categories_cfg)}
    return [idx for idx in indices if str(dataset[idx].get("category.id", "unknown")) in allowed]


def _project_2d(
    features: np.ndarray,
    method: str,
    seed: int,
    umap_neighbors: int,
    umap_min_dist: float,
    tsne_perplexity: float,
) -> tuple[np.ndarray, str]:
    n = features.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float32), "identity"
    if n < 3:
        out = np.zeros((n, 2), dtype=np.float32)
        out[:, 0] = features[:, 0]
        return out, "identity"

    method_l = method.lower()

    def _run_umap() -> np.ndarray:
        from umap import UMAP  # pyright: ignore[reportMissingImports]

        reducer = UMAP(
            n_components=2,
            n_neighbors=min(max(2, umap_neighbors), n - 1),
            min_dist=float(umap_min_dist),
            metric="euclidean",
            random_state=seed,
        )
        return cast(np.ndarray, reducer.fit_transform(features))

    def _run_tsne() -> np.ndarray:
        perplexity = min(float(tsne_perplexity), float(max(2, n - 1)))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        return cast(np.ndarray, tsne.fit_transform(features))

    if method_l == "umap":
        try:
            return _run_umap().astype(np.float32), "umap"
        except ImportError as exc:
            raise ImportError("UMAP requested but package is unavailable") from exc

    if method_l == "tsne":
        return _run_tsne().astype(np.float32), "tsne"

    # auto: UMAP preferred, t-SNE fallback, PCA fallback if needed
    try:
        return _run_umap().astype(np.float32), "umap"
    except ImportError:
        logger.warning("UMAP not available. Falling back to t-SNE.")
        try:
            return _run_tsne().astype(np.float32), "tsne"
        except ValueError:
            logger.warning("t-SNE failed. Falling back to PCA.")
            pca = PCA(n_components=2, random_state=seed)
            return cast(np.ndarray, pca.fit_transform(features)).astype(np.float32), "pca"


def _category_color_map(categories: list[str]) -> tuple[dict[str, int], np.ndarray]:
    unique = sorted(set(categories))
    cmap = plt.get_cmap("tab20")
    mapping = {cat: i for i, cat in enumerate(unique)}
    colors = np.array([cmap(i % 20)[:3] for i in range(len(unique))], dtype=np.float32)
    return mapping, colors


def _save_csv(
    path: Path,
    sample_indices: list[int],
    object_names: list[str],
    categories: list[str],
    vae_xy: np.ndarray,
    vqvae_xy: np.ndarray,
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_idx", "object_name", "category", "vae_x", "vae_y", "vqvae_x", "vqvae_y"])
        for i in range(len(sample_indices)):
            writer.writerow(
                [
                    int(sample_indices[i]),
                    object_names[i],
                    categories[i],
                    float(vae_xy[i, 0]),
                    float(vae_xy[i, 1]),
                    float(vqvae_xy[i, 0]),
                    float(vqvae_xy[i, 1]),
                ]
            )


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    suppress_known_optional_dependency_warnings()

    vis_cfg = cfg.get("vis", {})
    seed = int(vis_cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    split = str(vis_cfg.get("embedding_split", cfg.test.split))
    dataset = cast(Any, get_dataset(cfg, splits=(split,))[split])
    indices = _resolve_indices(dataset, vis_cfg)
    if not indices:
        raise ValueError("No valid object indices selected for latent embedding comparison.")

    method = str(vis_cfg.get("projection", "auto"))
    tsne_perplexity = float(vis_cfg.get("tsne_perplexity", 30.0))
    umap_neighbors = int(vis_cfg.get("umap_neighbors", 15))
    umap_min_dist = float(vis_cfg.get("umap_min_dist", 0.1))
    save_csv = bool(vis_cfg.get("save_csv", True))
    marker_alpha = float(vis_cfg.get("embedding_marker_alpha", 0.95))
    marker_size_cfg = vis_cfg.get("embedding_marker_size", None)

    vae_arch = str(vis_cfg.get("vae_arch", "3dshape2vecset_vae"))
    vqvae_arch = str(vis_cfg.get("vqvae_arch", "3dshape2vecset_vqvae"))
    vae_weights = str(vis_cfg.get("vae_weights", "cvpr_2025_vae/pcd_long_new/model_best.pt"))
    vqvae_weights = str(vis_cfg.get("vqvae_weights", "cvpr_2025_vae/pcd_vqvae_16k_long/model_best.pt"))

    output_dir_cfg = vis_cfg.get("output_dir", None)
    output_dir = Path(str(output_dir_cfg)).expanduser() if output_dir_cfg else (resolve_save_dir(cfg) / "latent_embedding_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae_model = get_model(cfg, arch=vae_arch, weights_path=vae_weights).to(device)
    vqvae_model = get_model(cfg, arch=vqvae_arch, weights_path=vqvae_weights).to(device)
    vae_model.requires_grad_(False)
    vqvae_model.requires_grad_(False)
    vae_model = patch_attention(vae_model, backend="torch")
    vqvae_model = patch_attention(vqvae_model, backend="torch")

    fabric = lightning.Fabric(precision=cfg.get(split, cfg.val).precision)
    vae_model = cast(Any, fabric.setup_module(vae_model))
    vqvae_model = cast(Any, fabric.setup_module(vqvae_model))
    vae_model.eval()
    vqvae_model.eval()

    vae_features: list[np.ndarray] = []
    vqvae_features: list[np.ndarray] = []
    categories: list[str] = []
    object_names: list[str] = []

    for idx in tqdm(indices, desc="Encoding samples"):
        item = dataset[idx]
        inputs = to_tensor(item["inputs"], device=device)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)

        with torch.no_grad(), fabric.autocast():
            vae_latent = cast(Tensor, vae_model.encode(inputs)).mean(dim=1)
            vqvae_latent = cast(Tensor, vqvae_model.encode(inputs)).mean(dim=1)

        vae_features.append(vae_latent[0].detach().float().cpu().numpy())
        vqvae_features.append(vqvae_latent[0].detach().float().cpu().numpy())
        categories.append(str(item.get("category.id", "unknown")))
        object_names.append(str(item.get("inputs.name", f"obj_{idx}")))

    vae_array = np.stack(vae_features, axis=0).astype(np.float32)
    vqvae_array = np.stack(vqvae_features, axis=0).astype(np.float32)

    vae_xy, method_used = _project_2d(
        vae_array,
        method=method,
        seed=seed,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        tsne_perplexity=tsne_perplexity,
    )
    vqvae_xy, method_used_vq = _project_2d(
        vqvae_array,
        method=method,
        seed=seed,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        tsne_perplexity=tsne_perplexity,
    )

    if method_used_vq != method_used:
        logger.warning(f"Projection methods differ (VAE={method_used}, VQ-VAE={method_used_vq}).")

    cat_to_idx, cat_colors = _category_color_map(categories)
    color_idx = np.array([cat_to_idx[c] for c in categories], dtype=np.int32)
    point_colors = cat_colors[color_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    titles = [f"VAE ({method_used.upper()})", f"VQ-VAE ({method_used_vq.upper()})"]
    arrays = [vae_xy, vqvae_xy]
    n_samples = len(indices)
    marker_size = float(marker_size_cfg) if marker_size_cfg is not None else (80.0 if n_samples <= 50 else 18.0)
    edge_width = 0.8 if n_samples <= 50 else 0.2

    for axis, arr, title in zip(axes, arrays, titles, strict=True):
        axis.scatter(
            arr[:, 0],
            arr[:, 1],
            c=point_colors,
            s=marker_size,
            alpha=marker_alpha,
            edgecolors="black",
            linewidths=edge_width,
        )
        axis.set_title(title)
        axis.set_xlabel("Dim 1")
        axis.set_ylabel("Dim 2")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    unique_categories = sorted(cat_to_idx.keys())
    max_legend = int(vis_cfg.get("max_legend_categories", 20))
    if unique_categories and len(unique_categories) <= max_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cat_colors[cat_to_idx[cat]],
                markersize=6,
                label=cat,
            )
            for cat in unique_categories
        ]
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False)

    out_prefix = output_dir / f"latent_embedding_compare_{method_used}"
    fig.savefig(out_prefix.with_suffix(".png"), dpi=250)
    plt.close(fig)

    np.savez_compressed(
        out_prefix.with_suffix(".npz"),
        sample_indices=np.asarray(indices, dtype=np.int32),
        object_names=np.asarray(object_names),
        categories=np.asarray(categories),
        vae_features=vae_array,
        vqvae_features=vqvae_array,
        vae_xy=vae_xy,
        vqvae_xy=vqvae_xy,
        projection_method_vae=np.asarray(method_used),
        projection_method_vqvae=np.asarray(method_used_vq),
    )

    if save_csv:
        _save_csv(
            out_prefix.with_suffix(".csv"),
            sample_indices=indices,
            object_names=object_names,
            categories=categories,
            vae_xy=vae_xy,
            vqvae_xy=vqvae_xy,
        )

    logger.info(f"Done. Saved latent embedding comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
