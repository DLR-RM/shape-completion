import io
from contextlib import redirect_stderr, redirect_stdout
from functools import cached_property, partial
from subprocess import CalledProcessError
from typing import Any, cast

import mcubes
import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes
from torch import Tensor, autograd, nn, optim
from tqdm import tqdm, trange
from trimesh import PointCloud, Trimesh

from libs import MISE, simplify_mesh
from models import AutoregressiveModel, DiffusionModel, MCDropoutNet, Model, PSSNet, ShapeFormer, VAEModel
from utils import (
    PARTNET_COLORS,
    PLOTLY_COLORS,
    binary_from_multi_class,
    filter_dict,
    git_submodule_path,
    setup_logger,
    to_tensor,
)

logger = setup_logger(__name__)

SCRIPT_PATH = None
try:
    from process import apply_meshlab_filters, modify_simplify

    SCRIPT_PATH = git_submodule_path("process") / "assets" / "meshlab_filter_scripts" / "simplify.mlx"
except (ImportError, FileNotFoundError, CalledProcessError) as e:
    logger.warning(f"Unable to import process module: {e}. Using libsimplify for mesh simplification.")

FeatureTypes = Tensor | dict[str, Tensor] | list[Tensor]
MaybeFeatureTypes = FeatureTypes | None
ItemType = dict[str, Any]


def _log_debug_level_1(message: str) -> None:
    log_fn = getattr(logger, "debug_level_1", logger.debug)
    log_fn(message)


def _as_tensor(value: Any) -> Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor, got {type(value).__name__}")
    return cast(Tensor, value)


class Generator:
    def __init__(
        self,
        model: Model,
        points_batch_size: int | None = None,
        threshold: float = 0.5,
        extraction_class: int = 1,
        refinement_steps: int = 0,
        resolution: int = 128,
        upsampling_steps: int = 0,
        estimate_normals: bool = False,
        predict_colors: bool = False,
        padding: float = 0.1,
        scale_factor: float = 1.0,
        simplify: bool | int | float | None = None,
        use_skimage: bool = False,
        sdf: bool = False,
        bounds: tuple[float, float] | tuple[tuple[float, float, float], tuple[float, float, float]] = (-0.5, 0.5),
    ):
        assert not (extraction_class != 1 and upsampling_steps), "Non-binary data cannot be upsampled"
        self.model = model
        self.refinement_steps = refinement_steps
        self.threshold = threshold
        self.extraction_class = extraction_class
        self.resolution = resolution
        self.upsampling_steps = upsampling_steps
        self.estimate_normals = estimate_normals
        self.predict_colors = predict_colors
        self.padding = padding
        self.scale_factor = scale_factor
        self.simplify = simplify
        self.use_skimage = use_skimage
        self.sdf = sdf

        if isinstance(bounds[0], (int, float)) and isinstance(bounds[1], (int, float)):
            mins = np.array([float(bounds[0])] * 3, dtype=np.float32)
            maxs = np.array([float(bounds[1])] * 3, dtype=np.float32)
        else:
            mins = np.array(bounds[0], dtype=np.float32)
            maxs = np.array(bounds[1], dtype=np.float32)
        self._mins = mins
        self._maxs = maxs

        if self.upsampling_steps > 0:
            self.resolution = int(2 ** (np.log2(32) + upsampling_steps))
            _log_debug_level_1(f"Using MISE with resolution 2^(log2(32)+{upsampling_steps})={self.resolution}")

        self.points_batch_size = points_batch_size or int(np.prod(self.grid_shape))

    @property
    def device(self) -> torch.device:
        return self.model.device

    @cached_property
    def center(self) -> np.ndarray:
        return ((self._mins + self._maxs) / 2.0).astype(np.float32)  # (3,)

    @cached_property
    def axis_extent(self) -> np.ndarray:
        return (self._maxs - self._mins) * float(self.scale_factor)  # (3,)

    @cached_property
    def axis_box_size(self) -> np.ndarray:
        return self.axis_extent + float(self.padding)  # (3,)

    @cached_property
    def box_size(self) -> float:
        # Single scalar, the largest axis size; defines uniform voxel length
        return float(np.max(self.axis_box_size))

    @cached_property
    def voxel_size(self) -> float:
        # Uniform voxel length (dx=dy=dz)
        return self.box_size / float(self.resolution)

    @cached_property
    def grid_shape(self) -> tuple[int, int, int]:
        # Rectangular grid with uniform spacing; ensure at least 2 voxels per axis
        shape = np.maximum(2, np.round(self.axis_box_size / self.voxel_size).astype(int))
        return int(shape[0]), int(shape[1]), int(shape[2])

    @cached_property
    def query_points(self) -> Tensor:
        # Voxel centers on a uniform lattice, translated to 'center'
        vs = float(self.voxel_size)
        nx, ny, nz = self.grid_shape
        total = np.array([nx, ny, nz], dtype=np.float32) * vs
        offset = (total - vs) / 2.0
        x0, y0, z0 = (self.center - offset).tolist()
        xs = torch.linspace(x0, x0 + vs * (nx - 1), nx)
        ys = torch.linspace(y0, y0 + vs * (ny - 1), ny)
        zs = torch.linspace(z0, z0 + vs * (nz - 1), nz)
        gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
        return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    @torch.inference_mode()
    def generate_pcd(
        self, item: ItemType, key: str | None = None, threshold: float | None = None, show: bool = False
    ) -> PointCloud:
        item = {k: to_tensor(v, device=self.device) for k, v in item.items()}
        predict = partial(self.model.predict, points_batch_size=self.points_batch_size, key=key)
        threshold = threshold or self.threshold

        if isinstance(self.model, (DiffusionModel, AutoregressiveModel)):
            inputs = _as_tensor(item["inputs"])
            points = _as_tensor(item["points"])
            logits = self.model.generate(
                inputs=inputs if inputs.ndim == 3 and inputs.size(2) == 3 else None,
                points=points,
                threshold=threshold,
                show=show,
            )
        else:
            logits = predict(**item)

        if logits.ndim == 3:
            pred = logits.argmax(dim=-1).squeeze().cpu().float().numpy()
            if logits.size(-2) == _as_tensor(item["inputs"]).size(-2):
                points = _as_tensor(item["inputs"]).squeeze(0).cpu().numpy()
                colors = PARTNET_COLORS[pred]
            elif "points" in item and logits.size(-2) == _as_tensor(item["points"]).size(-2):
                occ = pred != logits.size(-1) - 1
                points = _as_tensor(item["points"]).squeeze(0).cpu().numpy()[occ]
                colors = PARTNET_COLORS[pred[occ]]
            else:
                raise NotImplementedError(f"Point cloud generation is not implemented for '{self.model.name}'")
        elif logits.ndim == 2:
            threshold = threshold if self.sdf else np.log(threshold / (1 - threshold))
            occ = logits.squeeze().cpu().float().numpy() >= threshold
            points = _as_tensor(item["points"]).squeeze(0).cpu().numpy()[occ]
            colors = None
        else:
            raise NotImplementedError(f"Invalid logits dimension. Expected 2 or 3, got {logits.dim()}")

        return PointCloud(points, colors=colors)

    @torch.no_grad()
    def generate_grid(
        self,
        item: ItemType,
        extraction_class: int | None = None,
        threshold: float | None = None,
        progress: bool = False,
        show: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray | list[np.ndarray] | dict[int, np.ndarray], np.ndarray, MaybeFeatureTypes]:
        feature = None
        item = {k: to_tensor(v, device=self.device) for k, v in item.items()}
        inputs = _as_tensor(item["inputs"])
        if isinstance(self.model, MCDropoutNet):
            logits_mean, probs_var = self.model.mc_sample(inputs)
            grid = [logits_mean.cpu().numpy(), probs_var.cpu().numpy()]
            points = self.query_points.numpy()
        elif isinstance(self.model, PSSNet):
            logits = self.model.predict_many(inputs)
            probs = torch.sigmoid(logits)
            probs_mean = probs.mean(dim=0)
            probs_var = probs.var(dim=0)
            logits_mean = torch.logit(probs_mean)
            grid = [logits_mean.cpu().numpy(), probs_var.cpu().numpy()]
            points = self.query_points.numpy()
        elif isinstance(self.model, ShapeFormer):
            grid, points = self.model.generate_grids(inputs)
        elif isinstance(self.model, (DiffusionModel, AutoregressiveModel)):
            points = self.query_points.unsqueeze(0).expand(inputs.size(0), -1, -1).to(self.device)
            logits = self.model.generate(
                inputs=inputs if inputs.ndim in [3, 4, 5] else None,
                points=points,
                conditioning=item.get(getattr(self.model, "condition_key", "conditioning")),
                threshold=threshold or self.threshold,
                points_batch_size=self.points_batch_size,
                show=show,
                num_steps=kwargs.get("steps", 18),  # Diffusion
                progress=progress,
            )
            grid = logits.view(-1, *self.grid_shape).cpu()
            grid = grid.squeeze(0).numpy() if grid.size(0) == 1 else [g.numpy() for g in grid]
            points = self.query_points.numpy()
        elif isinstance(self.model, VAEModel):
            sample = cast(int | None, kwargs.get("sample"))
            unconditional = cast(bool | None, kwargs.get("unconditional"))
            sample_unconditional = bool(sample) and bool(unconditional)
            points = self.query_points.unsqueeze(0).expand(inputs.size(0), -1, -1).to(self.device)
            logits = self.model.predict(
                inputs=None if sample_unconditional else inputs,
                points=points,
                points_batch_size=self.points_batch_size,
                sample=sample == 1 or sample_unconditional,
            )
            if sample is not None and sample > 1 and not unconditional and inputs.size(0) == 1:
                samples = [
                    self.model.predict(
                        inputs=inputs, points=points, points_batch_size=self.points_batch_size, sample=True
                    )
                    for _ in range(sample)
                ]
                logits = torch.stack([logits, *samples])
            grid = logits.view(-1, *self.grid_shape).float().cpu()
            grid = grid.squeeze(0).numpy() if len(grid) == 1 else [g.numpy() for g in grid]
            points = self.query_points.numpy()
        elif hasattr(self.model, "predict") and not (
            self.upsampling_steps or self.refinement_steps or self.estimate_normals or self.predict_colors
        ):
            align_to_gt = kwargs.get("align_to_gt", False)
            if (
                not align_to_gt
                or "points" not in item
                or _as_tensor(item["points"]).numel() != self.query_points.numel()
            ):
                item["points"] = self.query_points.unsqueeze(0).expand(inputs.size(0), -1, -1).to(self.device)
            logits = self.model.predict(
                **item,
                points_batch_size=self.points_batch_size,
                threshold=threshold or self.threshold,
                align_to_gt=align_to_gt,
                data=item,
                show=show,
            )
            if torch.is_tensor(logits):
                grid = logits.view(-1, *self.grid_shape).float().cpu()
            else:
                grid = []
                for logit in cast(list[Any], logits):
                    if len(logit):
                        grid.append(
                            torch.stack(
                                [value.view(*self.grid_shape).float().cpu() for value in cast(list[Tensor], logit)]
                            )
                        )
                if len(grid) == 1:
                    grid = grid[0]
            if torch.is_tensor(grid):
                grid = grid.squeeze(0).numpy() if grid.size(0) == 1 else [g.numpy() for g in grid]
            else:
                grid = [g.numpy() for g in grid]
            points = self.query_points.numpy()
        elif inputs.size(0) == 1:
            with torch.inference_mode(mode=not (self.estimate_normals or self.refinement_steps)):
                feature = cast(Any, self.model).encode(**item)
            grid, points = self.generate(feature, item, extraction_class)
        else:
            raise NotImplementedError(f"Grid generation is not implemented for '{self.model.name}'")
        return grid, points, feature

    @torch.inference_mode()
    def generate_grid_per_instance(
        self,
        item: ItemType,
        threshold: float | None = None,
        align_to_gt: bool = False,
        show: bool = False,
        return_meta: bool = True,
    ) -> list[np.ndarray | dict[str, Any]]:
        item = {k: to_tensor(v, device=self.device) for k, v in item.items()}
        if "points" not in item or _as_tensor(item["points"]).numel() != self.query_points.numel():
            item["points"] = self.query_points.unsqueeze(0).to(self.device)

        # 1) coarse lattice using the global query_points (resolution = self.resolution)
        nx, ny, nz = self.grid_shape
        vs_c = float(self.voxel_size)
        total = np.array([nx, ny, nz], dtype=np.float32) * vs_c
        offset = (total - vs_c) / 2.0
        x0, y0, z0 = (self.center - offset).tolist()

        thr = threshold or self.threshold
        thr_val = thr if self.sdf else float(np.log(thr / (1 - thr)))

        # 2) coarse predict
        feature = None
        if hasattr(self.model, "encode"):
            feature = cast(Any, self.model).encode(**item)
        logits = (
            self.model.predict(
                inputs=item.get("inputs"),
                points=item["points"],
                feature=feature,
                data=item,
                threshold=thr,
                align_to_gt=align_to_gt,
                points_batch_size=self.points_batch_size,
                show=show,
            )[0]
            .float()
            .cpu()
        )

        if logits.numel() == 0:
            # Use the actual rectangular global grid shape rather than assuming cubic resolution
            empty = np.zeros((nx, ny, nz), dtype=np.float32)
            if return_meta:
                return [{"grid": empty, "voxel_size": self.voxel_size, "center": self.center}]
            return [empty]

        # Helper for candidate selection (soft Dice + coarse consistency)
        def _select_best_candidate(
            out_preds, inside_mask: np.ndarray, coarse_idx_lin: np.ndarray, coarse_probs_flat: np.ndarray
        ) -> torch.Tensor:
            # Normalize inputs to list of tensors
            if isinstance(out_preds, (list, tuple)):
                candidates = out_preds
            elif torch.is_tensor(out_preds):
                candidates = list(out_preds) if out_preds.ndim >= 2 else [out_preds]
            else:
                candidates = [out_preds]

            m = inside_mask.astype(np.float32)

            # Stable sigmoid / occupancy mapping utilities
            def _sigmoid_np(x: np.ndarray) -> np.ndarray:
                # Numerically stable sigmoid without overflow warnings
                out = np.empty_like(x, dtype=np.float32)
                pos = x >= 0
                neg = ~pos
                out[pos] = 1.0 / (1.0 + np.exp(-np.clip(x[pos], -50, 50)))
                exp_x = np.exp(np.clip(x[neg], -50, 50))
                out[neg] = exp_x / (1.0 + exp_x)
                return out

            def _sdf_to_prob(x: np.ndarray, tau: float) -> np.ndarray:
                # Map (signed) SDF to occupancy probability around surface using stable logistic
                scaled = np.clip(x / tau, -50, 50)
                return 1.0 / (1.0 + np.exp(scaled))  # inside vs outside (negative inside if convention holds)

            eps = 1e-8
            scores = []
            for oj in candidates:
                v = oj.detach().float().cpu().numpy().reshape(-1)
                # Convert to probability domain
                if self.sdf:
                    tau = 0.02 * max(self.box_size, 1e-6)
                    p = _sdf_to_prob(v, tau)
                else:
                    p = _sigmoid_np(v)
                TP = (p * m).sum()
                FP = (p * (1 - m)).sum()
                FN = ((1 - p) * m).sum()
                dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
                fp_norm = FP / (len(p) + eps)
                # Coarse-fine consistency: average fine probabilities per coarse voxel
                coarse_sum = np.zeros_like(coarse_probs_flat)
                coarse_count = np.zeros_like(coarse_probs_flat)
                np.add.at(coarse_sum, coarse_idx_lin, p)
                np.add.at(coarse_count, coarse_idx_lin, 1)
                valid = coarse_count > 0
                coarse_avg = np.zeros_like(coarse_probs_flat)
                coarse_avg[valid] = coarse_sum[valid] / np.maximum(1, coarse_count[valid])
                mse = ((coarse_avg[valid] - coarse_probs_flat[valid]) ** 2).mean() if valid.any() else 0.0
                score = dice - 0.05 * fp_norm - 0.1 * mse
                scores.append(score)
            best_idx = int(np.argmax(scores))
            return candidates[best_idx]

        # Map world points -> coarse voxel index (nearest center)
        def _coarse_idx_from_world(pts_world: np.ndarray) -> np.ndarray:
            ijk = np.rint((pts_world - np.array([x0, y0, z0], dtype=np.float32)) / vs_c).astype(np.int32)
            ijk[:, 0] = np.clip(ijk[:, 0], 0, nx - 1)
            ijk[:, 1] = np.clip(ijk[:, 1], 0, ny - 1)
            ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)
            return ijk

        results: list[np.ndarray | dict[str, Any]] = []
        for slot_idx, logit in enumerate(logits):
            if logit.numel() == 0:
                continue

            vol = logit.view(nx, ny, nz).numpy()
            # Dual-threshold and simple dilation to protect thin structures
            if self.sdf:
                core_mask = vol >= thr  # SDF domain threshold
                fringe_mask = core_mask  # No second threshold for SDF; could extend based on |vol| < band
            else:
                core_mask = vol >= thr_val
                low_thr_logit = float(np.log(0.2 / 0.8))  # logit(0.2)
                fringe_mask = vol >= low_thr_logit

            # 6-neighborhood dilation (low-cost) on core_mask
            def _dilate6(mk: np.ndarray) -> np.ndarray:
                dm = mk.copy()
                shifts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                for dx, dy, dz in shifts:
                    shifted = np.zeros_like(mk)
                    xs = slice(max(0, dx), nx + min(0, dx))
                    ys = slice(max(0, dy), ny + min(0, dy))
                    zs = slice(max(0, dz), nz + min(0, dz))
                    xs_src = slice(max(0, -dx), nx + min(0, -dx))
                    ys_src = slice(max(0, -dy), ny + min(0, -dy))
                    zs_src = slice(max(0, -dz), nz + min(0, -dz))
                    shifted[xs, ys, zs] = mk[xs_src, ys_src, zs_src]
                    dm |= shifted
                return dm

            dilated_core = _dilate6(core_mask)
            mask = dilated_core | fringe_mask
            idx = np.argwhere(mask)
            if idx.size == 0:
                continue

            ix_min = idx.min(axis=0)
            ix_max = idx.max(axis=0)
            # world AABB from voxel index range (include full voxel extents)
            mins = np.array(
                [
                    x0 + vs_c * ix_min[0] - 0.5 * vs_c,
                    y0 + vs_c * ix_min[1] - 0.5 * vs_c,
                    z0 + vs_c * ix_min[2] - 0.5 * vs_c,
                ],
                dtype=np.float32,
            )
            maxs = np.array(
                [
                    x0 + vs_c * ix_max[0] + 0.5 * vs_c,
                    y0 + vs_c * ix_max[1] + 0.5 * vs_c,
                    z0 + vs_c * ix_max[2] + 0.5 * vs_c,
                ],
                dtype=np.float32,
            )

            # Instance size prior to padding for scaling
            axis_box_size_initial = maxs - mins
            box_size_initial = float(np.max(axis_box_size_initial))
            box_size_pad = max(0.1, self.padding) * axis_box_size_initial

            # Per-axis padding: scale by instance size; keep ≥1 coarse voxel
            pad_vec = np.maximum(box_size_pad / max(box_size_initial, 1e-8), vs_c).astype(np.float32)

            mins = mins - pad_vec
            maxs = maxs + pad_vec
            center_i = (mins + maxs) / 2.0
            axis_box_size_i = maxs - mins
            box_size_i = float(np.max(axis_box_size_i))

            # Resolution and uniform voxel size driven by largest axis
            vs_i = box_size_i / float(self.resolution)

            # Per-axis resolutions to preserve aspect ratio (>= 2)
            nxyz = np.maximum(2, np.round(axis_box_size_i / vs_i).astype(np.int32))
            nx_i, ny_i, nz_i = int(nxyz[0]), int(nxyz[1]), int(nxyz[2])

            # Build grid of voxel centers consistent with extract_mesh spacing
            total_i = np.array([nx_i, ny_i, nz_i], dtype=np.float32) * vs_i
            offset_i = (total_i - vs_i) / 2.0
            x0_i, y0_i, z0_i = (center_i - offset_i).tolist()

            xs = np.linspace(x0_i, x0_i + vs_i * (nx_i - 1), nx_i, dtype=np.float32)
            ys = np.linspace(y0_i, y0_i + vs_i * (ny_i - 1), ny_i, dtype=np.float32)
            zs = np.linspace(z0_i, z0_i + vs_i * (nz_i - 1), nz_i, dtype=np.float32)
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
            pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)

            # Coarse mask/sample at fine points
            coarse_idx = _coarse_idx_from_world(pts)
            inside_flat = mask[coarse_idx[:, 0], coarse_idx[:, 1], coarse_idx[:, 2]]
            coarse_idx_lin = coarse_idx[:, 0] * (ny * nz) + coarse_idx[:, 1] * nz + coarse_idx[:, 2]

            # Coarse probabilities for consistency term
            if self.sdf:
                tau_c = 0.02 * max(self.box_size, 1e-6)
                coarse_probs_flat = 1.0 / (1.0 + np.exp(np.clip(vol.reshape(-1) / tau_c, -50, 50)))
            else:
                # Stable sigmoid for coarse logits
                v_flat = vol.reshape(-1)
                pos = v_flat >= 0
                neg = ~pos
                coarse_probs_flat = np.empty_like(v_flat, dtype=np.float32)
                coarse_probs_flat[pos] = 1.0 / (1.0 + np.exp(-np.clip(v_flat[pos], -50, 50)))
                exp_x = np.exp(np.clip(v_flat[neg], -50, 50))
                coarse_probs_flat[neg] = exp_x / (1.0 + exp_x)

            # Predict fine
            pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(self.device)
            out = self.model.predict(
                inputs=item.get("inputs"),
                points=pts_t,
                feature=feature,
                data=item,
                threshold=thr,
                points_batch_size=self.points_batch_size,
                show=show,
            )[0]

            if out.numel():
                out_k = _select_best_candidate(out, inside_flat, coarse_idx_lin, coarse_probs_flat)
                fine_vals_flat = out_k.float().cpu().numpy().astype(np.float32)
            else:
                fine_vals_flat = np.zeros((nx_i * ny_i * nz_i,), dtype=np.float32)
            vals = fine_vals_flat
            dense = vals.reshape(nx_i, ny_i, nz_i).astype(np.float32)

            if return_meta:
                results.append({
                    "grid": dense,
                    "voxel_size": float(vs_i),
                    "center": center_i.astype(np.float32),
                    "instance_idx": slot_idx,
                })
            else:
                results.append(dense)

        return results

    def generate_mesh(
        self,
        item: ItemType,
        threshold: float | None = None,
        extraction_class: int | None = None,
        progress: bool = False,
        show: bool = False,
        **kwargs: Any,
    ) -> Trimesh | list[Trimesh]:
        grid, _, feature = self.generate_grid(
            item=item, extraction_class=extraction_class, threshold=threshold, progress=progress, show=show, **kwargs
        )
        return self.extract_meshes(
            grid=grid, feature=feature, item=item, threshold=threshold, extraction_class=extraction_class
        )

    @torch.no_grad()
    def generate(
        self, feature: MaybeFeatureTypes = None, item: ItemType | None = None, extraction_class: int | None = None
    ) -> tuple[np.ndarray | dict[int, np.ndarray], np.ndarray]:
        eval_points = partial(self.eval_points, feature=feature, item=item)
        if self.upsampling_steps == 0 or MISE is None:
            values = eval_points(points=self.query_points)
            if values.ndim == 2:
                if extraction_class is None:
                    value_grid = dict()
                    for i in torch.unique(torch.argmax(values, dim=1)):
                        if i != values.size(1) - 1:
                            binary_values = binary_from_multi_class(values, occ_label=i)
                            value_grid[i] = binary_values.numpy().reshape(self.grid_shape)
                else:
                    occ_label = None if extraction_class == -1 else extraction_class
                    binary_values = binary_from_multi_class(values, occ_label=occ_label)
                    value_grid = binary_values.numpy().reshape(self.grid_shape)
            else:
                value_grid = values.numpy().reshape(self.grid_shape)
            points = self.query_points.numpy()
        else:
            points_list = list()
            threshold = self.threshold if self.sdf else np.log(self.threshold / (1 - self.threshold))
            mise = MISE(resolution_0=32, depth=self.upsampling_steps, threshold=threshold)
            queries = mise.query()  # Initial queries at resolution 32 in range [0, 32]
            while len(queries):
                points = queries / mise.resolution - 0.5  # Normalize to [-0.5, 0.5]
                voxel_size = self.box_size / mise.resolution
                points = points * (self.axis_box_size - voxel_size) + self.center  # Scale to bounding box
                points_list.append(points)

                values = eval_points(points=torch.from_numpy(points).float())
                if values.ndim == 2:
                    values = binary_from_multi_class(values, occ_label=extraction_class)

                mise.update(queries, values.numpy().astype(np.float64))
                queries = mise.query()

            dense = mise.to_dense()  # (R,R,R)
            res = len(dense)
            nx, ny, nz = self.grid_shape
            sx = max(0, int(np.floor((res - nx) / 2.0)))
            sy = max(0, int(np.floor((res - ny) / 2.0)))
            sz = max(0, int(np.floor((res - nz) / 2.0)))
            value_grid = dense[sx : sx + nx, sy : sy + ny, sz : sz + nz]
            points = np.concatenate(points_list, axis=0)
        return value_grid, points

    def eval_points(
        self,
        points: Tensor | None = None,
        feature: MaybeFeatureTypes = None,
        item: ItemType | None = None,
        enable_grad: bool = False,
    ) -> Tensor:
        assert points is not None or (item is not None and "points" in item), "Points must be provided"
        if points is None:
            assert item is not None
            points = _as_tensor(item["points"])
        assert points.ndim == 2 and points.size(1) == 3, "Points must be of shape (N, 3)"

        # Split points into random batches
        indices = torch.randperm(len(points)) if self.points_batch_size < len(points) else torch.arange(len(points))
        points_split = torch.split(points[indices], self.points_batch_size)

        predictions = torch.tensor([], device=self.device if enable_grad else "cpu")
        for pi in tqdm(
            points_split, desc="Evaluating points", total=len(points_split), leave=False, disable=len(points_split) < 10
        ):
            pi = pi.unsqueeze(0).to(self.device)
            kwargs = dict() if item is None else filter_dict(item, remove={"points", "feature"})
            if enable_grad:
                with torch.enable_grad():
                    logits = cast(Any, self.model).decode(points=pi, feature=feature, **kwargs)
                if isinstance(logits, dict):
                    logits = logits["logits"]
            else:
                logits = self.model.predict(points=pi, feature=feature, **kwargs)
                if isinstance(logits, list):
                    logits = logits[0].squeeze(0)
                logits = logits.detach().float().cpu()
                if logits.ndim == 2:
                    logits = logits.sum(0)
            predictions = torch.cat([predictions, logits])

        return predictions[torch.argsort(indices)]

    def extract_mesh(
        self,
        predictions: np.ndarray,
        feature: MaybeFeatureTypes = None,
        item: ItemType | None = None,
        threshold: float | None = None,
        extraction_class: int | None = None,
        voxel_size: float | None = None,
        center: np.ndarray | None = None,
    ) -> Trimesh:
        threshold = self.threshold if threshold is None else threshold
        threshold = float(threshold if self.sdf else np.log(threshold / (1 - threshold)))
        vs = self.voxel_size if voxel_size is None else voxel_size
        center = self.center if center is None else center

        nx, ny, nz = predictions.shape
        total = np.array([nx, ny, nz], dtype=np.float32) * vs
        offset = (total - vs) / 2.0  # (3,)

        if self.use_skimage:
            try:
                gradient_direction = "descent" if self.sdf else "ascent"
                vertices, faces, normals, _ = marching_cubes(
                    volume=predictions,
                    level=threshold,
                    spacing=(vs, vs, vs),
                    gradient_direction=gradient_direction,
                    allow_degenerate=False,
                )

                mesh = Trimesh(
                    vertices,
                    faces,
                    vertex_normals=normals if self.estimate_normals else None,
                    process=False,
                    validate=False,
                )

                mesh.apply_translation((center - offset).tolist())
            except ValueError:
                return Trimesh()
        else:
            if self.sdf:
                predictions *= -1
                threshold *= -1
            # Make sure that mesh is watertight
            occ_hat_padded = np.pad(predictions, 1, "constant", constant_values=-1e6)
            vertices, triangles = mcubes.marching_cubes(occ_hat_padded, threshold)
            vertices -= 1  # Undo padding

            # Normalize to bounding box
            vertices = vertices * vs + (center - offset)

            mesh = Trimesh(vertices, triangles, process=False, validate=False)

        # Directly return if mesh is empty
        if len(mesh.vertices) == 0 and len(mesh.faces) == 0:
            return mesh

        if self.simplify:
            if SCRIPT_PATH is not None and SCRIPT_PATH.is_file():
                script_path = modify_simplify(SCRIPT_PATH, num_vertices=len(vertices))
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    mesh = Trimesh(*apply_meshlab_filters(mesh.vertices, mesh.faces, [script_path]))
            else:
                if isinstance(self.simplify, bool):
                    mesh = simplify_mesh(mesh, target_percent=10)
                elif isinstance(self.simplify, int):
                    mesh = simplify_mesh(mesh, target_faces=self.simplify)
                elif isinstance(self.simplify, float):
                    mesh = simplify_mesh(mesh, target_percent=int(100 * self.simplify))

        if self.refinement_steps or self.estimate_normals or self.predict_colors:
            if feature is None and isinstance(self.model.encoder, nn.Module):
                raise ValueError("Feature must be provided for mesh refinement and/or normal/color estimation")
            feature_for_post = cast(FeatureTypes, feature)

            if item:
                item = {k: to_tensor(v, device=self.device) for k, v in item.items()}

            if self.refinement_steps:
                mesh = self.refine_mesh(
                    mesh=mesh,
                    predictions=predictions,
                    feature=feature_for_post,
                    item=item,
                    threshold=threshold,
                    extraction_class=extraction_class,
                    normal_weight=0.01,
                )

            if self.estimate_normals:
                mesh = Trimesh(
                    mesh.vertices,
                    mesh.faces,
                    vertex_normals=self.estimate_vertex_normals(
                        vertices=mesh.vertices, feature=feature_for_post, item=item, extraction_class=extraction_class
                    ),
                    process=False,
                    validate=False,
                )

            if self.predict_colors:
                colors = self.predict_vertex_colors(vertices=mesh.vertices, feature=feature_for_post, item=item)
                cast(Any, mesh.visual).vertex_colors = (colors * 255).astype(np.uint8)
        return mesh

    def extract_meshes(
        self,
        grid: np.ndarray | dict[int, np.ndarray] | list[np.ndarray] | list[dict[str, Any]],
        feature: MaybeFeatureTypes = None,
        item: ItemType | None = None,
        threshold: float | None = None,
        extraction_class: int | None = None,
    ) -> Trimesh | list[Trimesh]:
        extract_mesh = partial(
            self.extract_mesh, feature=feature, item=item, threshold=threshold, extraction_class=extraction_class
        )
        if isinstance(grid, list):
            mesh = list()
            for c, g in enumerate(grid):
                # Use aligned instance index for color when available (preserves
                # correspondence with instseg PNGs after empty-instance compaction).
                if isinstance(g, dict):
                    inst_idx = g.get("instance_idx", c)
                    m = extract_mesh(predictions=g["grid"], voxel_size=g["voxel_size"], center=g["center"])
                else:
                    inst_idx = c
                    m = extract_mesh(predictions=g)
                color = (np.array([*list(PLOTLY_COLORS[inst_idx]), 1]) * 255).astype(np.uint8)
                cast(Any, m.visual).vertex_colors = color
                m.metadata["instance_idx"] = inst_idx
                mesh.append(m)
        elif isinstance(grid, dict):
            mesh = list()
            for c, g in grid.items():
                color = (np.array([*list(PLOTLY_COLORS[c]), 1]) * 255).astype(np.uint8)
                m = extract_mesh(predictions=g)
                cast(Any, m.visual).vertex_colors = color
                mesh.append(m)
            mesh = trimesh.util.concatenate(mesh)
        else:
            mesh = extract_mesh(predictions=grid, feature=feature)
        return mesh

    def extract_uncertain_regions(self):
        raise NotImplementedError("Extracting uncertain regions is not implemented here yet")

    @torch.enable_grad()
    def estimate_vertex_normals(
        self,
        vertices: np.ndarray,
        feature: FeatureTypes,
        item: ItemType | None = None,
        extraction_class: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        vertex_points = torch.from_numpy(vertices).float()
        vertex_points.requires_grad = True

        logits = self.eval_points(points=vertex_points, feature=feature, item=item, enable_grad=True)
        if logits.ndim == 2:
            logits = binary_from_multi_class(logits, occ_label=extraction_class)

        logits.sum().backward()
        if vertex_points.grad is None:
            raise RuntimeError("Failed to compute vertex normal gradients")
        normals = -vertex_points.grad

        if normalize:
            normals /= torch.norm(normals, dim=-1, keepdim=True)

        return normals.detach().cpu().numpy()

    @torch.enable_grad()
    def refine_mesh(
        self,
        mesh: Trimesh,
        predictions: Tensor | np.ndarray,
        feature: FeatureTypes,
        item: ItemType | None = None,
        threshold: float | None = None,
        extraction_class: int | None = None,
        normal_weight: float = 0.01,
    ) -> Trimesh:
        if threshold is None:
            threshold = self.threshold

        n_x, n_y, n_z = predictions.shape
        assert min(n_x, n_y, n_z) >= 2, "Grid must have at least 2 samples per axis"

        vertices = nn.Parameter(torch.from_numpy(mesh.vertices).float().to(self.device))
        faces = torch.from_numpy(mesh.faces).to(self.device)
        optimizer = optim.RMSprop([vertices], lr=1e-4)

        for _ in trange(self.refinement_steps, desc="Refining", leave=False):
            optimizer.zero_grad()

            face_vertices = vertices[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.size(0))  # Sample random points on faces
            eps = torch.from_numpy(eps).float().to(self.device)
            face_points = (face_vertices * eps[:, :, None]).sum(dim=1)

            logits = self.eval_points(points=face_points, feature=feature, item=item, enable_grad=True)

            if logits.ndim == 2:
                logits = binary_from_multi_class(logits, occ_label=extraction_class)

            face_value = torch.sigmoid(logits)
            loss = (face_value - threshold).pow(2).mean()  # MSE loss

            if normal_weight > 0:
                face_v1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
                face_v2 = face_vertices[:, 2, :] - face_vertices[:, 1, :]
                face_normal = torch.linalg.cross(face_v1, face_v2)
                face_normal = face_normal / (face_normal.norm(dim=1, keepdim=True) + 1e-10)

                normal_target = -autograd.grad(
                    [face_value.sum()],
                    [face_points],
                    retain_graph=True,
                    create_graph=True,
                )[0]
                normal_target = normal_target / (normal_target.norm(dim=1, keepdim=True) + 1e-10)

                loss_normal = (face_normal - normal_target).pow(2).sum(dim=1).mean()  # MSE loss
                loss += normal_weight * loss_normal

            loss.backward(retain_graph=True)
            optimizer.step()

        mesh.vertices = vertices.data.cpu().numpy()
        return mesh

    @torch.inference_mode()
    def predict_vertex_colors(
        self, vertices: np.ndarray, feature: FeatureTypes, item: ItemType | None = None
    ) -> np.ndarray:
        vertices_t = torch.from_numpy(vertices).float().unsqueeze(0).to(self.device)
        _item = dict() if item is None else {k: v for k, v in item.items() if k != "points"}
        colors = self.model.predict(points=vertices_t, feature=feature, key="colors", **_item)
        return torch.sigmoid(colors).squeeze(0).cpu().float().numpy()
