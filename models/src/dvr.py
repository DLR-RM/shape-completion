import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision.ops import focal_loss

from utils import DEBUG_LEVEL_1, adjust_intrinsic, apply_trafo, cosine_anneal, depth_to_image, inv_trafo, setup_logger

from .idr import RenderingNetwork
from .model import Model
from .utils import get_normal_loss, get_normals

logger = setup_logger(__name__)


def _require_tensor(data: dict[str, list[str] | Tensor], key: str) -> Tensor:
    value = data.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected tensor for `{key}`")
    return value


def _optional_tensor(value: list[str] | Tensor | None) -> Tensor | None:
    return value if isinstance(value, Tensor) else None


@dataclass
class RayMarchingConfig:
    """Configuration for ray marching parameters."""

    near: float = 0.0  # Near clipping plane
    far: float = 2.4  # Far clipping plane
    step_func: Literal["equal", "half", "cosine"] = "equal"
    num_steps: int | tuple[int, ...] = 16  # Number or range of ray marching steps
    num_pixels: int | None = 1024  # Number of pixels to sample
    num_points: int | None = 16384  # Maximum number of points to evaluate at once
    num_views: int = 1  # Number of views for multi-view supervision
    max_batch_size: int | None = 1024  # Maximum batch size for ray marching
    threshold: float = 0.5  # Occupancy threshold value
    refine_mode: Literal["secant", "bisection", "linear", "midpoint"] = "midpoint"  # Step refinement method
    num_refine_steps: int = 8  # Number of refinement steps
    crop: bool = False  # Whether to check for cube intersections
    padding: float = 0.1  # Padding for cube intersection
    debug: bool = False  # Enable debug mode

    def __post_init__(self):
        if self.near < 0:
            raise ValueError("Near clipping plane must be non-negative.")
        if self.near >= self.far:
            raise ValueError("Near clipping plane must be less than far clipping plane.")
        if self.far <= 0:
            raise ValueError("Far clipping plane must be positive.")

        if self.step_func not in ["equal", "half", "cosine"]:
            raise ValueError(f"Unknown step function: {self.step_func}. Must be one of 'equal', 'half', or 'cosine'.")
        if self.step_func == "cosine":
            if not isinstance(self.num_steps, tuple) or len(self.num_steps) != 2:
                raise ValueError("For cosine step function, num_steps must be a tuple of (min_steps, max_steps).")

        if not 0 <= self.threshold <= 1:
            raise ValueError(f"Threshold must be in range [0, 1], got {self.threshold}.")
        self.log_threshold = -math.log(1.0 / self.threshold - 1.0)

        if self.refine_mode not in ["secant", "bisection", "linear", "midpoint"]:
            raise ValueError(
                f"Unknown refinement mode: {self.refine_mode}. Must be one of 'secant', 'bisection', 'linear', or 'midpoint'."
            )


def _safe_denom(dc: Tensor, eps: Tensor) -> Tensor:
    return torch.copysign(torch.max(torch.abs(dc), eps), dc)


class DifferentiableRayCaster(nn.Module):
    """
    Differentiable ray casting implementation.

    This module performs ray marching to find surface intersections and
    supports backpropagation through the intersection process using
    implicit differentiation.
    """

    def __init__(self, config: RayMarchingConfig | None = None):
        """
        Initialize the ray caster with configuration parameters.

        Args:
            config: Configuration for ray marching parameters
        """
        super().__init__()
        config = config or RayMarchingConfig()
        self.config = config
        self.num_steps = config.num_steps if isinstance(config.num_steps, int) else config.num_steps[0]

    def forward(
        self,
        ray0: Tensor,
        ray_dirs: Tensor,
        predict: Callable,
        parameters: list[nn.Parameter],
        feature: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        intrinsic: Tensor | None = None,
        extrinsic: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        d_pred, p_pred, mask = self.perform_ray_marching(ray0, ray_dirs, predict, feature, intrinsic, extrinsic)

        inputs = [
            ray_dirs,
            d_pred,
            p_pred,
            mask,
            partial(predict, intrinsic=intrinsic, extrinsic=extrinsic),
            feature,
            *parameters,
        ]
        d_out = cast(Tensor, DepthFunction.apply(*inputs))
        p_out = ray0 + d_out.unsqueeze(-1) * ray_dirs

        return d_out, p_out, mask

    def check_ray_intersection_with_unit_cube(self, ray0: Tensor, ray_dirs: Tensor) -> tuple[Tensor, Tensor]:
        min_bound = -0.5 - self.config.padding / 2
        max_bound = 0.5 + self.config.padding / 2
        eps = torch.tensor(torch.finfo(ray0.dtype).eps).to(ray0)

        denom_x = _safe_denom(ray_dirs[..., 0], eps)
        denom_y = _safe_denom(ray_dirs[..., 1], eps)
        denom_z = _safe_denom(ray_dirs[..., 2], eps)

        # Calculate intersections with all planes using safe denominators
        t_min_x = (min_bound - ray0[..., 0]) / denom_x
        t_max_x = (max_bound - ray0[..., 0]) / denom_x
        t_min_y = (min_bound - ray0[..., 1]) / denom_y
        t_max_y = (max_bound - ray0[..., 1]) / denom_y
        t_min_z = (min_bound - ray0[..., 2]) / denom_z
        t_max_z = (max_bound - ray0[..., 2]) / denom_z

        # Get entry and exit points
        start = torch.max(
            torch.min(t_min_x, t_max_x), torch.max(torch.min(t_min_y, t_max_y), torch.min(t_min_z, t_max_z))
        )
        stop = torch.min(
            torch.max(t_min_x, t_max_x), torch.min(torch.max(t_min_y, t_max_y), torch.max(t_min_z, t_max_z))
        )

        mask = stop >= start
        start = torch.where(mask, start, torch.ones_like(start) * self.config.near)
        stop = torch.where(mask, stop, torch.ones_like(stop) * self.config.far)

        return start.unsqueeze(-1), stop.unsqueeze(-1)

    def sample_distances(
        self,
        n_points: int,
        n_steps: int | tuple[int, int] | None = None,
        n_batch: int = 1,
        ray0: Tensor | None = None,
        ray_dirs: Tensor | None = None,
        method: Literal["linear", "random"] = "linear",
    ) -> Tensor:
        n_steps_val = n_steps if n_steps is not None else self.num_steps
        if isinstance(n_steps_val, tuple):
            n_steps_i = int(torch.randint(*n_steps_val, (1,), device=ray0.device if ray0 is not None else None).item())
        else:
            n_steps_i = int(n_steps_val)

        steps: Tensor
        if method == "linear":
            steps = torch.linspace(0, 1, n_steps_i).view(1, 1, -1).expand(n_batch, n_points, -1)
        elif method == "random":
            steps = torch.rand(n_batch, n_points, n_steps_i)
            if n_steps_i > 1:
                steps, _ = torch.sort(steps, dim=2)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        start = torch.full_like(steps[..., :1], self.config.near)
        stop = torch.full_like(steps[..., :1], self.config.far)
        if self.config.crop and ray0 is not None and ray_dirs is not None:
            start, stop = self.check_ray_intersection_with_unit_cube(ray0, ray_dirs)
            steps = steps.to(start)

        return start + steps * (stop - start)

    @torch.no_grad()
    def perform_ray_marching(
        self,
        ray0: Tensor,
        ray_dirs: Tensor,
        predict: Callable,
        feature: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        intrinsic: Tensor | None = None,
        extrinsic: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, n, _ = ray0.shape
        n_steps = self.num_steps
        feature_tensor: Tensor | None = None
        if feature is not None:
            if not torch.is_tensor(feature):
                raise NotImplementedError("Non single tensor features not yet supported")
            feature_tensor = feature

        distances = self.sample_distances(n, n_batch=b, ray0=ray0, ray_dirs=ray_dirs).to(ray0)
        points = ray0.unsqueeze(2) + distances.unsqueeze(-1) * ray_dirs.unsqueeze(2)
        points = points.view(b, -1, 3)

        # Evaluate occupancy
        val = (
            predict(
                points=points,
                feature=feature_tensor,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
            )
            - self.config.log_threshold
        )
        val = val.view(b, n, n_steps)

        # Check if first point is not occupied
        mask_0_not_occupied = val[:, :, 0] < 0

        # Find sign changes (surface crossings)
        sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]), torch.ones(b, n, 1).to(ray0)], dim=-1)

        # Weight by distance from start (prefer earlier intersections)
        cost_matrix = sign_matrix * torch.arange(n_steps, 0, -1).float().to(ray0)

        # Get first sign change and relevant masks
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_neg_to_pos = val[torch.arange(b).unsqueeze(-1), torch.arange(n).unsqueeze(0), indices] < 0

        # Valid samples have sign change from negative to positive
        # and first point is not occupied
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied

        # Extract rays for valid points
        ray0_masked = ray0[mask]
        ray_dirs_masked = ray_dirs[mask]

        # Apply refinement method
        d_pred = torch.ones(ray_dirs_masked.size(0)).to(ray0)
        if mask.any():
            # Prepare for refinement method
            # Flatten for easier indexing
            m = b * n
            d_low = distances.view(m, n_steps, 1)[torch.arange(m), indices.view(m)].view(b, n)[mask]

            f_low = val.view(m, n_steps, 1)[torch.arange(m), indices.view(m)].view(b, n)[mask]

            # Get high points (clamp to stay in bounds)
            indices_high = torch.clamp(indices + 1, max=n_steps - 1)
            d_high = distances.view(m, n_steps, 1)[torch.arange(m), indices_high.view(m)].view(b, n)[mask]

            f_high = val.view(m, n_steps, 1)[torch.arange(m), indices_high.view(m)].view(b, n)[mask]

            feat_masked: Tensor | None = feature_tensor
            intrinsic_masked = intrinsic
            extrinsic_masked = extrinsic
            if b > 1:
                if feature_tensor is not None:
                    if feature_tensor.ndim == 2:  # Global feature vector (B, C)
                        feat_masked = feature_tensor.unsqueeze(1).expand(-1, n, -1)[mask]
                    elif feature_tensor.ndim == 3:  # Sequence features (B, T, C)
                        feat_masked = feature_tensor.unsqueeze(1).expand(-1, n, -1, -1)[mask]
                    elif feature_tensor.ndim == 4:  # Image features (B, C, H, W)
                        feat_masked = feature_tensor.unsqueeze(1).expand(-1, n, -1, -1, -1)[mask]
                    else:
                        raise NotImplementedError(f"Unknown feature shape: {feature_tensor.shape}")

                    assert feat_masked.size(0) == ray0_masked.size(0)

                if intrinsic is not None:
                    intrinsic_masked = intrinsic.unsqueeze(1).expand(-1, n, -1, -1)[mask]
                if extrinsic is not None:
                    extrinsic_masked = extrinsic.unsqueeze(1).expand(-1, n, -1, -1)[mask]

            if self.config.refine_mode == "secant":
                d_pred = self.run_secant_method(
                    f_low,
                    f_high,
                    d_low,
                    d_high,
                    ray0_masked,
                    ray_dirs_masked,
                    predict,
                    feat_masked,
                    intrinsic_masked,
                    extrinsic_masked,
                )
            elif self.config.refine_mode == "bisection":
                d_pred = self.run_bisection_method(
                    d_low,
                    d_high,
                    ray0_masked,
                    ray_dirs_masked,
                    predict,
                    feat_masked,
                    intrinsic_masked,
                    extrinsic_masked,
                )
            elif self.config.refine_mode == "linear":  # Same as single secant step
                d_pred = -f_low * (d_high - d_low) / (f_high - f_low) + d_low
            elif self.config.refine_mode == "midpoint":  # Same as single bisection step
                d_pred = 0.5 * (d_low + d_high)
            else:
                raise ValueError(f"Unknown refinement mode: {self.config.refine_mode}")

            # torch.cuda.empty_cache()

        d_pred_out = torch.ones(b, n).to(ray0)
        d_pred_out[mask_0_not_occupied] = 0
        d_pred_out[mask] = d_pred
        p_pred_out = ray0 + d_pred_out.unsqueeze(-1) * ray_dirs

        return d_pred_out, p_pred_out, mask

    @torch.no_grad()
    def run_secant_method(
        self,
        f_low: Tensor,
        f_high: Tensor,
        d_low: Tensor,
        d_high: Tensor,
        ray0_masked: Tensor,
        ray_dirs_masked: Tensor,
        predict: Callable,
        feature_masked: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        intrinsic_masked: Tensor | None = None,
        extrinsic_masked: Tensor | None = None,
    ) -> Tensor:
        d_pred = -f_low * (d_high - d_low) / (f_high - f_low) + d_low

        for _ in range(self.config.num_refine_steps):
            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_dirs_masked

            if feature_masked is not None:
                if not torch.is_tensor(feature_masked):
                    raise NotImplementedError
                if p_mid.size(0) != feature_masked.size(0):
                    if feature_masked.size(0) == 1:
                        p_mid = p_mid.unsqueeze(0)
                    else:
                        raise ValueError("Feature and point dimensions do not match")
                else:
                    p_mid = p_mid.unsqueeze(1)
            else:
                p_mid = p_mid.unsqueeze(0)

            f_mid = (
                predict(
                    points=p_mid,
                    feature=feature_masked,
                    intrinsic=intrinsic_masked,
                    extrinsic=extrinsic_masked,
                ).squeeze()
                - self.config.log_threshold
            )

            ind_low = f_mid < 0
            if ind_low.any():
                d_low[ind_low] = d_pred[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if (ind_low == 0).any():
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]

            d_pred = -f_low * (d_high - d_low) / (f_high - f_low) + d_low

        return d_pred

    @torch.no_grad()
    def run_bisection_method(
        self,
        d_low: Tensor,
        d_high: Tensor,
        ray0_masked: Tensor,
        ray_dirs_masked: Tensor,
        predict: Callable,
        feature_masked: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        intrinsic_masked: Tensor | None = None,
        extrinsic_masked: Tensor | None = None,
    ) -> Tensor:
        # Initial midpoint
        d_pred = 0.5 * (d_low + d_high)

        # Iterative refinement
        for _ in range(self.config.num_refine_steps):
            # Evaluate at midpoint
            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_dirs_masked

            if feature_masked is not None:
                if not torch.is_tensor(feature_masked):
                    raise NotImplementedError
                if p_mid.size(0) != feature_masked.size(0):
                    if feature_masked.size(0) == 1:
                        p_mid = p_mid.unsqueeze(0)
                    else:
                        raise ValueError("Feature and point dimensions do not match")
                else:
                    p_mid = p_mid.unsqueeze(1)
            else:
                p_mid = p_mid.unsqueeze(0)

            f_mid = (
                predict(
                    points=p_mid,
                    feature=feature_masked,
                    intrinsic=intrinsic_masked,
                    extrinsic=extrinsic_masked,
                ).squeeze()
                - self.config.log_threshold
            )

            # Update intervals based on midpoint value
            ind_low = f_mid < 0
            d_low[ind_low] = d_pred[ind_low]
            d_high[ind_low == 0] = d_pred[ind_low == 0]

            # Calculate new midpoint
            d_pred = 0.5 * (d_low + d_high)

        return d_pred


class DepthFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        ray_dirs, d_pred, p_pred, mask, predict, feature = inputs[:6]

        ctx.save_for_backward(ray_dirs, d_pred, p_pred, mask, feature)
        ctx.predict = predict
        ctx.parameters = list(inputs[6:])

        return d_pred

    @staticmethod
    def backward(ctx, grad_output):
        ray_dirs, _d_pred, p_pred, mask, feature = ctx.saved_tensors
        predict = ctx.predict
        parameters = ctx.parameters
        eps = torch.finfo(grad_output.dtype).eps

        if not mask.any():
            return_vals = [None] * 6 + [None] * len(parameters)
            return tuple(return_vals)

        # Prepare gradients
        with torch.enable_grad():
            # Make predicted points require gradients
            p_pred.requires_grad = True

            # Evaluate occupancy network at predicted points
            f_p = predict(points=p_pred, feature=feature)

            # Sum for gradient computation
            f_p_sum = f_p.sum()

            # Get gradient of occupancy with respect to points, i.e surface normals
            grad_p = torch.autograd.grad(f_p_sum, p_pred, retain_graph=True)[0]

            # Compute dot product with ray direction (needed for implicit differentiation)
            grad_p_dot_v = (grad_p * ray_dirs).sum(-1)

            # Replace invalid values with 1.0
            grad_p_dot_v[~mask] = 1.0

            # Avoid very small values in the denominator for numerical stability.
            condition_mask = torch.abs(grad_p_dot_v) < eps
            eps_tensor = torch.tensor(eps).to(grad_p_dot_v)
            grad_p_dot_v[condition_mask] = torch.copysign(eps_tensor, grad_p_dot_v[condition_mask])

            # Apply chain rule for implicit differentiation
            grad_outputs = -grad_output.squeeze(-1)
            grad_outputs = grad_outputs / grad_p_dot_v
            grad_outputs = grad_outputs * mask.float()

            # Compute gradients for feature
            grad_feat = None
            if feature is not None and feature.requires_grad:
                if isinstance(feature, dict):
                    feature = list(feature.values())
                if not isinstance(feature, list):
                    feature = [feature]

                all_grads = torch.autograd.grad(
                    f_p, feature + parameters, grad_outputs=grad_outputs, retain_graph=True, allow_unused=True
                )

                num_features = len(feature)
                grad_feats = all_grads[:num_features]
                grad_params = all_grads[num_features:]
                if len(grad_feats) == 0:
                    grad_feat = None
                elif len(grad_feats) == 1:
                    grad_feat = grad_feats[0]
                else:
                    grad_feat = list(grad_feats)
            else:
                grad_params = torch.autograd.grad(
                    f_p, parameters, grad_outputs=grad_outputs, retain_graph=True, allow_unused=True
                )

            if any(p is None for p in grad_params):
                logger.warning("Some parameters did not receive gradients.")

        return_vals = [None, None, None, None, None, grad_feat, *grad_params]

        return tuple(return_vals)


class DVR(nn.Module):
    def __init__(
        self,
        model: Model,
        config: RayMarchingConfig | None = None,
        depth_loss: Literal["l1", "smooth_l1", "l2", "mse"] | None = None,
        rgb_loss: Literal["l1", "smooth_l1", "l2", "mse"] | None = "l1",
        normal_loss: Literal["l1", "smooth_l1", "l2", "mse", "cos_sim", "eikonal", "field"] | None = None,
        sym_loss: Literal["tp", "pred", "field"] | None = None,
        feature_consistency_loss: bool = True,
        geometry_consistency_loss: bool = True,
        apply_loss_ratio: bool = True,
        learn_loss_weights: str | None = None,
        focal_loss_alpha: float | None = None,
        p_universal: bool = False,
    ):
        super().__init__()
        self.model = model
        self.config = config or RayMarchingConfig()
        self.ray_caster = DifferentiableRayCaster(self.config)
        self.renderer = None
        if rgb_loss:
            if p_universal:
                self.renderer = RenderingNetwork(feat_dim=256)
            else:
                self.renderer = nn.Linear(384, 3)
        self.p_universal = p_universal

        self.depth_loss = depth_loss
        self.rgb_loss = rgb_loss
        self.normal_loss = normal_loss
        self.sym_loss = sym_loss
        self.feature_consistency_loss = feature_consistency_loss
        self.geometry_consistency_loss = geometry_consistency_loss

        self.learn_loss_weights = learn_loss_weights
        self.apply_loss_ratio = apply_loss_ratio
        self.focal_loss_alpha = focal_loss_alpha
        if learn_loss_weights or rgb_loss:
            if hasattr(model, "optimizer"):
                self.optimizer = None
        if learn_loss_weights:
            if "surf" in learn_loss_weights:
                self.log_var_surf = nn.Parameter(torch.zeros(()))
            if "field" in learn_loss_weights or focal_loss_alpha:
                self.log_var_field = nn.Parameter(torch.zeros(()))
            else:
                if "occ" in learn_loss_weights:
                    self.log_var_occ = nn.Parameter(torch.zeros(()))
                if "free" in learn_loss_weights:
                    self.log_var_free = nn.Parameter(torch.zeros(()))
            if rgb_loss and "rgb" in learn_loss_weights:
                self.log_var_rgb = nn.Parameter(torch.zeros(()))
            if depth_loss and "depth" in learn_loss_weights:
                self.log_var_depth = nn.Parameter(torch.zeros(()))
            if normal_loss and "normal" in learn_loss_weights:
                self.log_var_normal = nn.Parameter(torch.zeros(()))
        self.depth_loss_weight = 1.0
        self.rgb_loss_weight = 1.0
        self.normal_loss_weight = 0.01
        self.sym_loss_weight = 0.01
        self.free_loss_weight = 1.0
        self.occ_loss_weight = 1.0

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @property
    def psnr(self) -> PeakSignalNoiseRatio:
        return PeakSignalNoiseRatio(data_range=1.0).to(self.device)

    @property
    def ssim(self) -> StructuralSimilarityIndexMeasure:
        return StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    @property
    def lpips(self) -> LearnedPerceptualImagePatchSimilarity:
        return LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

    @torch.no_grad()
    def compute_metrics(self, pred: Tensor, target: Tensor, prefix: str = ""):
        width, height = None, None
        if pred.ndim == 4:
            if pred.size(1) == 3:
                pred = pred.view(pred.size(0), 3, -1)
            elif pred.size(-1) == 3:
                pred = pred.view(pred.size(0), -1, 3).transpose(1, 2)
            else:
                raise ValueError
        if target.ndim == 4:
            if target.size(1) == 3:
                height, width = target.shape[2:]
                target = target.view(target.size(0), 3, -1)
            elif target.size(-1) == 3:
                height, width = target.shape[1:3]
                target = target.view(target.size(0), -1, 3).transpose(1, 2)
            else:
                raise ValueError

        if pred.size() != target.size():
            raise ValueError(f"Prediction and target must have the same shape: {pred.shape} != {target.shape}")

        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        if target.min() < 0:
            if target.size(1) == 3:
                target = target.transpose(1, 2)
            target = target * torch.tensor([0.229, 0.224, 0.225]).to(target)
            target = target + torch.tensor([0.485, 0.456, 0.406]).to(target)

        if pred.size(-1) == 3:
            pred = pred.transpose(1, 2)
        if target.size(-1) == 3:
            target = target.transpose(1, 2)

        self.model.log(f"{prefix}psnr", self.psnr(pred, target))
        if height is not None and width is not None:
            ssim = self.ssim(pred.view(-1, 3, height, width), target.view(-1, 3, height, width))
            lpips = self.lpips(pred.view(-1, 3, height, width), target.view(-1, 3, height, width))
            self.model.log(f"{prefix}ssim", ssim)
            self.model.log(f"{prefix}lpips", lpips)

    def get_rgb_loss(self, pred: Tensor, image: Tensor, mask: Tensor) -> Tensor:
        if pred.size() != image.transpose(1, 2).size():
            raise ValueError(f"Prediction and image must have the same shape: {pred.shape} != {image.shape}")
        rgb_pred_tp = torch.sigmoid(pred[mask])
        image_tp = image.transpose(1, 2)[mask]
        image_tp = image_tp * torch.tensor([0.229, 0.224, 0.225]).to(image_tp)
        image_tp = image_tp + torch.tensor([0.485, 0.456, 0.406]).to(image_tp)
        if image_tp.min() < 0 or image_tp.max() > 1:
            raise ValueError(f"Image must be in range [0, 1]: {image_tp.min().item(), image_tp.max().item()}")
        if self.rgb_loss == "smooth_l1":
            loss = F.smooth_l1_loss(rgb_pred_tp, image_tp, reduction="none")
        elif self.rgb_loss == "l2":
            loss = torch.linalg.norm(rgb_pred_tp - image_tp, dim=-1)
        elif self.rgb_loss == "mse":
            loss = F.mse_loss(rgb_pred_tp, image_tp, reduction="none")
        else:
            loss = F.l1_loss(rgb_pred_tp, image_tp, reduction="none")
        return loss

    def get_depth_loss(self, pred: Tensor, depth: Tensor, mask: Tensor, extrinsic: Tensor) -> Tensor:
        depth_tp = depth[mask]
        if self.depth_loss == "l2":
            return torch.linalg.norm(pred[mask] - depth_tp, dim=-1)

        p_pred_cam = cast(Tensor, apply_trafo(pred, extrinsic))
        z_pred_tp = p_pred_cam[..., -1][mask]
        if self.depth_loss == "smooth_l1":
            loss = F.smooth_l1_loss(z_pred_tp, depth_tp, reduction="none")
        elif self.depth_loss == "mse":
            loss = F.mse_loss(z_pred_tp, depth_tp, reduction="none")
        else:
            loss = F.l1_loss(z_pred_tp, depth_tp, reduction="none")
        return loss

    @staticmethod
    @torch.autocast(device_type="cuda", enabled=False)
    def get_rays(
        mask: Tensor, intrinsic: Tensor, extrinsic: Tensor, normalize: bool = False, num_samples: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b, h, w = mask.shape
        total_pixels = h * w
        num_pixels_used = total_pixels

        if num_samples is None or num_samples >= total_pixels:
            u, v = torch.meshgrid(
                torch.arange(w, device=mask.device), torch.arange(h, device=mask.device), indexing="xy"
            )
            u, v = u.flatten().unsqueeze(0).expand(b, -1).float(), v.flatten().unsqueeze(0).expand(b, -1).float()
            uv_coords = torch.stack((u, v), dim=-1)
            xyz = torch.cat((uv_coords, torch.ones(b, total_pixels, 1, device=mask.device)), dim=-1)
        else:
            indices = torch.randint(0, total_pixels, (b, num_samples), device=mask.device)
            u = (indices % w).float()
            v = (indices // w).float()
            uv_coords = torch.stack((u, v), dim=-1)
            xyz = torch.cat((uv_coords, torch.ones(b, num_samples, 1, device=mask.device)), dim=-1)
            num_pixels_used = num_samples

        inv_intrinsic = torch.inverse(intrinsic)
        inv_extrinsic_raw = inv_trafo(extrinsic)
        if torch.is_tensor(inv_extrinsic_raw):
            inv_extrinsic = inv_extrinsic_raw
        else:
            inv_extrinsic = torch.from_numpy(inv_extrinsic_raw).to(intrinsic)

        p_cam = xyz @ inv_intrinsic.transpose(-1, -2)
        p_world = cast(Tensor, apply_trafo(p_cam, inv_extrinsic))

        ray0 = inv_extrinsic[:, :3, 3].unsqueeze(1).expand(b, num_pixels_used, 3)
        ray_dirs = p_world - ray0

        if normalize:
            ray_dirs = ray_dirs / ray_dirs.norm(2, dim=-1, keepdim=True)

        return ray0, ray_dirs, u, v

    @torch.autocast(device_type="cuda", enabled=False)
    def get_points(self, item: dict[str, Tensor], num_points: int | None = None) -> tuple[Tensor, Tensor]:
        num_pixels = num_points or self.config.num_pixels

        inputs = item["inputs"]
        intrinsic = item["inputs.intrinsic"]
        extrinsic = item["inputs.extrinsic"]
        mask = item["inputs.mask"].bool()
        depth = item.get("inputs.depth")
        if depth is None and inputs.size() == mask.size():
            depth = item["inputs"]

        ray0, ray_dirs, u, v = self.get_rays(mask, intrinsic, extrinsic, num_samples=num_pixels)

        b, h, w = mask.shape
        total_pixels = h * w
        mask = mask.view(b, -1)
        if depth is not None:
            depth = depth.view(b, -1)

        if num_pixels is not None and num_pixels < total_pixels:
            index = (v.long() * w + u.long()).long()
            mask = torch.gather(mask, dim=1, index=index)
            if depth is not None:
                depth = torch.gather(depth, dim=1, index=index)

        d = (
            self.ray_caster.sample_distances(
                n_points=ray0.size(1), n_steps=1, n_batch=ray0.size(0), ray0=ray0, ray_dirs=ray_dirs, method="random"
            )
            .squeeze(-1)
            .to(ray0)
        )

        p = ray0 + d.unsqueeze(-1) * ray_dirs
        if depth is not None:
            uv_coords = torch.stack((u, v), dim=-1)
            xyz = torch.cat((uv_coords, torch.ones_like(u).unsqueeze(-1)), dim=-1)
            p_depth = xyz @ torch.inverse(intrinsic).transpose(-1, -2)
            p_depth *= depth.unsqueeze(-1)
            p_depth = cast(Tensor, apply_trafo(p_depth, inv_trafo(extrinsic)))
            p[mask] = p_depth[mask]

        return p, mask

    @torch.autocast(device_type="cuda", enabled=False)
    def step(
        self, data: dict[str, list[str] | Tensor], feature: Tensor | None = None, **kwargs
    ) -> dict[str, list[str] | Tensor]:
        inputs = _require_tensor(data, "inputs")
        intrinsic = _require_tensor(data, "inputs.intrinsic")
        extrinsic = _require_tensor(data, "inputs.extrinsic")
        mask_gt = _require_tensor(data, "inputs.mask").bool()
        image = _optional_tensor(data.get("inputs.image"))
        depth = _optional_tensor(data.get("inputs.depth"))
        normals = _optional_tensor(data.get("inputs.normals"))

        ray0, ray_dirs, u, v = self.get_rays(mask_gt, intrinsic, extrinsic, num_samples=self.config.num_pixels)

        b, h, w = mask_gt.shape
        total_pixels = h * w

        if depth is None and inputs.size() == mask_gt.size():
            depth = inputs
        if inputs.ndim == 4:
            image = inputs

        if image is None and depth is None:
            raise ValueError("Either image or depth must be provided for DVR.")

        # If no precomputed conditioning is provided, derive it from the current inputs.
        if feature is None:
            encode_kwargs = {k: v for k, v in {**data, **kwargs}.items()}
            feature = self.encode(**encode_kwargs)

        mask_gt = mask_gt.view(b, -1)
        depth_mask = torch.ones_like(mask_gt, dtype=torch.bool)
        if depth is not None:
            depth = depth.view(b, -1)
            depth_mask = depth.isfinite()
        if image is not None:
            if image.shape != (b, 3, h, w):
                raise ValueError(f"Image must have shape (B, 3, H, W): {image.shape}")
            if image.min() >= 0 and image.max() <= 1:
                raise ValueError(f"Image must be normalized: {image.min().item(), image.max().item()}")
            image = image.view(b, 3, -1)
        if normals is not None:
            normals = normals.view(b, -1, 3)

        if self.config.num_pixels is not None and self.config.num_pixels < total_pixels:
            index = (v.long() * w + u.long()).long()
            mask_gt = torch.gather(mask_gt, dim=1, index=index)
            if depth is not None:
                depth = torch.gather(depth, dim=1, index=index)
                depth_mask = torch.gather(depth_mask, dim=1, index=index)
            if image is not None:
                image = torch.gather(image, dim=2, index=index.unsqueeze(1).expand(-1, 3, -1))
            if normals is not None:
                normals = torch.gather(normals, dim=1, index=index.unsqueeze(2).expand(-1, -1, 3))

        decode_kwargs = {
            k: v for k, v in {**data, **kwargs}.items() if k not in ["points", "inputs.intrinsic", "inputs.extrinsic"]
        }
        decode_kwargs["return_point_feature"] = True
        predict = partial(self.predict, **decode_kwargs)

        _d_pred, p_pred, mask_pred = self.ray_caster(
            ray0=ray0,
            ray_dirs=ray_dirs,
            predict=predict,
            parameters=list(cast(Any, self.model).decoder.parameters()),
            feature=feature,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )

        mask_tp = mask_pred & mask_gt
        mask_fp = mask_pred & ~mask_gt
        mask_tn = ~mask_pred & ~mask_gt
        mask_fn = ~mask_pred & mask_gt
        mask_tp_and_finite = mask_tp
        mask_gt_and_finite = mask_gt

        p_depth = None
        if depth is not None:
            depth_tensor = depth
            mask_tp_and_finite = mask_tp & depth_mask
            mask_gt_and_finite = mask_gt & depth_mask
            depth_tensor[~depth_mask] = 0
            uv_coords = torch.stack((u, v), dim=-1)
            xyz = torch.cat((uv_coords, torch.ones_like(u).unsqueeze(-1)), dim=-1)
            p_depth = xyz @ torch.inverse(intrinsic).transpose(-1, -2)
            p_depth *= depth_tensor.unsqueeze(-1)
            p_depth = cast(Tensor, apply_trafo(p_depth, inv_trafo(extrinsic)))
            depth = depth_tensor

        rgb_loss = torch.zeros((), device=inputs.device)
        depth_loss = torch.zeros((), device=inputs.device)
        normal_loss = torch.zeros((), device=inputs.device)
        sym_loss = torch.zeros((), device=inputs.device)
        if mask_tp.any():
            if (
                self.rgb_loss
                or (self.normal_loss and self.normal_loss != "field")
                or (self.sym_loss and self.sym_loss != "field")
            ):
                enable_grad = self.training or self.normal_loss is not None or self.p_universal
                with torch.set_grad_enabled(enable_grad):
                    if enable_grad and not p_pred.requires_grad:
                        p_pred.requires_grad = True
                    out_pred = cast(
                        dict[str, Tensor],
                        predict(points=p_pred, feature=feature, intrinsic=intrinsic, extrinsic=extrinsic, key=None),
                    )

                normals_pred = None
                if self.normal_loss or self.p_universal:
                    normals_pred = get_normals(p_pred, out_pred["logits"], normalize=self.normal_loss != "eikonal")
                    if self.normal_loss:
                        normal_mode = cast(Literal["l1", "smooth_l1", "l2", "mse", "cos_sim"], self.normal_loss)
                        normal_loss = get_normal_loss(
                            normals_pred=normals_pred,
                            normals_gt=normals,
                            mask=mask_tp,
                            eikonal="eikonal" in self.normal_loss,
                            mode=normal_mode,
                        )
                        if self.apply_loss_ratio:
                            normal_loss = normal_loss.mean() * (mask_tp.sum() / mask_gt.numel())
                        else:
                            normal_loss = normal_loss.sum() / mask_gt.numel()

                if image is not None and self.rgb_loss:
                    if self.p_universal:
                        assert self.renderer is not None
                        view_dirs = -ray_dirs / ray_dirs.norm(2, dim=-1, keepdim=True)
                        pred_rgb = self.renderer(
                            points=p_pred,
                            feature=out_pred["feature"],
                            normals=cast(Tensor, normals_pred)
                            / cast(Tensor, normals_pred).norm(2, dim=-1, keepdim=True),
                            view_dirs=view_dirs,
                        )
                    else:
                        assert self.renderer is not None
                        pred_rgb = self.renderer(out_pred["feature"])

                    rgb_loss = self.get_rgb_loss(pred_rgb, image, mask_tp)
                    if self.apply_loss_ratio:
                        rgb_loss = rgb_loss.mean() * (mask_tp.sum() / mask_gt.numel())
                    else:
                        rgb_loss = rgb_loss.sum() / mask_gt.numel()

                    self.compute_metrics(
                        pred=pred_rgb[mask_tp].unsqueeze(0).detach(), target=image.transpose(1, 2)[mask_tp].unsqueeze(0)
                    )

                if self.sym_loss:
                    logits = out_pred["logits"]
                    p_sym = p_pred.detach().clone()
                    p_sym[..., 2] *= -1  # Assumes z-axis (xy-plane) symmetry
                    logits_sym = cast(
                        Tensor,
                        predict(points=p_sym, feature=feature, intrinsic=intrinsic, extrinsic=extrinsic),
                    )
                    if self.sym_loss == "tp":
                        mask_sym = mask_tp
                    elif self.sym_loss == "pred":
                        mask_sym = mask_pred

                    sym_loss = torch.abs(logits_sym[mask_sym] - logits[mask_sym].detach()).sum()

                    if self.apply_loss_ratio:
                        sym_loss = sym_loss * (mask_sym.sum() / mask_gt.numel())
                    else:
                        sym_loss = sym_loss / mask_gt.numel()

            if depth is not None and self.depth_loss:
                if self.depth_loss == "l2":
                    assert p_depth is not None
                    depth_gt = p_depth
                else:
                    depth_gt = depth
                depth_loss = self.get_depth_loss(p_pred, depth_gt, mask_tp_and_finite, extrinsic)
                if self.apply_loss_ratio:
                    depth_loss = depth_loss.mean() * (mask_tp_and_finite.sum() / mask_gt.numel())
                else:
                    depth_loss = depth_loss.sum() / mask_gt.numel()

        if self.normal_loss == "field":
            bounds_val = 0.5 + self.config.padding / 2
            with torch.enable_grad():
                p_field = torch.empty_like(p_pred).uniform_(-bounds_val, bounds_val)
                p_field.requires_grad = True
                logits = cast(
                    Tensor, predict(points=p_field, feature=feature, intrinsic=intrinsic, extrinsic=extrinsic)
                )
                normals_field = get_normals(p_field, logits, normalize=False)
            normal_loss = get_normal_loss(normals_field, eikonal=True, mode="l1").mean()

        if self.sym_loss == "field":
            bounds_val = 0.5 + self.config.padding / 2
            p1_field = torch.rand_like(p_pred)
            p1_field = p1_field * 2 * bounds_val - bounds_val
            p2_field = p1_field.clone()
            p2_field[..., 2] *= -1  # Assumes z-axis (xy-plane) symmetry
            p_field = torch.cat([p1_field, p2_field], dim=1)
            logits_field = cast(
                Tensor, predict(points=p_field, feature=feature, intrinsic=intrinsic, extrinsic=extrinsic)
            )
            logits1 = logits_field[:, : p1_field.size(1)]
            logits2 = logits_field[:, p1_field.size(1) :]
            sym_loss = torch.abs(logits1 - logits2).mean()

        d = (
            self.ray_caster.sample_distances(
                n_points=ray0.size(1), n_steps=1, n_batch=ray0.size(0), ray0=ray0, ray_dirs=ray_dirs, method="random"
            )
            .squeeze(-1)
            .to(ray0)
        )

        # d[mask_false_pos] = d_pred[mask_false_pos].detach()
        p = ray0 + d.unsqueeze(-1) * ray_dirs
        p[mask_fp] = p_pred[mask_fp].detach()
        if depth is not None:
            assert p_depth is not None
            p[mask_gt_and_finite] = p_depth[mask_gt_and_finite]

        logits = cast(Tensor, predict(points=p, feature=feature, intrinsic=intrinsic, extrinsic=extrinsic))
        logits_occ = logits[mask_gt]
        logits_free = logits[~mask_gt]

        occ_loss = torch.zeros((), device=inputs.device)
        if mask_fn.any():
            logits_fn = logits[mask_fn]
            occ_loss = F.binary_cross_entropy_with_logits(logits_fn, torch.ones_like(logits_fn), reduction="none")
            if self.apply_loss_ratio:
                occ_loss = occ_loss.mean() * (mask_fn.sum() / mask_gt.numel())
            else:
                occ_loss = occ_loss.sum() / mask_gt.numel()
        # if mask_gt.any():
        #     occ_loss = F.binary_cross_entropy_with_logits(logits_occ, torch.ones_like(logits_occ), reduction="sum") / numel

        free_loss = torch.zeros((), device=inputs.device)
        if (~mask_gt).any():
            free_loss = F.binary_cross_entropy_with_logits(logits_free, torch.zeros_like(logits_free), reduction="none")
            if self.apply_loss_ratio:
                free_loss = free_loss.mean() * ((~mask_gt).sum() / mask_gt.numel())
            else:
                free_loss = free_loss.sum() / mask_gt.numel()
        # if mask_false_pos.any():
        #     logits_false_pos = logits[mask_false_pos]
        #     free_loss = F.binary_cross_entropy_with_logits(logits_false_pos, torch.zeros_like(logits_false_pos), reduction="sum") / numel

        tp = mask_tp.sum().item()
        fp = mask_fp.sum().item()
        fn = mask_fn.sum().item()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0

        if self.rgb_loss:
            self.log("rgb_loss", rgb_loss.item())
        if self.depth_loss:
            self.log("depth_loss", depth_loss.item())
        if self.normal_loss:
            self.log("normal_loss", normal_loss.item())
        if self.sym_loss:
            self.log("sym_loss", sym_loss.item())
        self.log("occ_loss", occ_loss.item())
        self.log("free_loss", free_loss.item())

        self.log("mask_precision", precision, level=DEBUG_LEVEL_1)
        self.log("mask_recall", recall, level=DEBUG_LEVEL_1)
        self.log("mask_f1", f1, level=DEBUG_LEVEL_1)
        self.log("mask_iou", iou, level=DEBUG_LEVEL_1)

        if self.learn_loss_weights:
            if self.rgb_loss and "rgb" in self.learn_loss_weights:
                rgb_loss = rgb_loss * torch.exp(-self.log_var_rgb) + self.log_var_rgb
                self.log("rgb_weight", torch.exp(-self.log_var_rgb).item(), level=DEBUG_LEVEL_1)
            if self.depth_loss and "depth" in self.learn_loss_weights:
                depth_loss = depth_loss * torch.exp(-self.log_var_depth) + self.log_var_depth
                self.log("depth_weight", torch.exp(-self.log_var_depth).item(), level=DEBUG_LEVEL_1)
            if self.normal_loss and "normal" in self.learn_loss_weights:
                normal_loss = normal_loss * torch.exp(-self.log_var_normal) + self.log_var_normal
                self.log("normal_weight", torch.exp(-self.log_var_normal).item(), level=DEBUG_LEVEL_1)

        surf_loss = (
            self.rgb_loss_weight * rgb_loss
            + self.depth_loss_weight * depth_loss
            + self.normal_loss_weight * normal_loss
            + self.sym_loss_weight * sym_loss
        )
        field_loss = self.occ_loss_weight * occ_loss + self.free_loss_weight * free_loss

        if self.focal_loss_alpha:
            logits_fn = logits[mask_fn]
            all_logits = torch.cat([logits_fn, logits_free], dim=0)
            all_targets = torch.cat([torch.ones_like(logits_fn), torch.zeros_like(logits_free)], dim=0)
            field_loss = 2 * focal_loss.sigmoid_focal_loss(all_logits, all_targets, alpha=self.focal_loss_alpha)
            if self.apply_loss_ratio:
                field_loss = field_loss.mean() * ((mask_fn.sum() + (~mask_gt).sum()) / mask_gt.numel())
            else:
                field_loss = field_loss.sum() * mask_gt.numel()
            self.log("focal_loss", field_loss.item(), level=DEBUG_LEVEL_1)
        if self.learn_loss_weights:
            if "surf" in self.learn_loss_weights:
                surf_loss = surf_loss * torch.exp(-self.log_var_surf) + self.log_var_surf
                self.log("surf_weight", torch.exp(-self.log_var_surf).item(), level=DEBUG_LEVEL_1)
            if "field" in self.learn_loss_weights or self.focal_loss_alpha:
                field_loss = field_loss * torch.exp(-self.log_var_field) + self.log_var_field
                self.log("field_weight", torch.exp(-self.log_var_field).item(), level=DEBUG_LEVEL_1)
            else:
                if "occ" in self.learn_loss_weights:
                    occ_loss = occ_loss * torch.exp(-self.log_var_occ) + self.log_var_occ
                    self.log("occ_weight", torch.exp(-self.log_var_occ).item(), level=DEBUG_LEVEL_1)
                if "free" in self.learn_loss_weights:
                    free_loss = free_loss * torch.exp(-self.log_var_free) + self.log_var_free
                    self.log("free_weight", torch.exp(-self.log_var_free).item(), level=DEBUG_LEVEL_1)
                field_loss = self.occ_loss_weight * occ_loss + self.free_loss_weight * free_loss

        data["points"] = torch.cat((p[mask_gt], p[~mask_gt]), dim=0).unsqueeze(0)
        data["points.occ"] = torch.cat((torch.ones_like(logits_occ), torch.zeros_like(logits_free)), dim=0).unsqueeze(0)
        data["logits"] = torch.cat((logits_occ, logits_free), dim=0).unsqueeze(0)
        data["loss"] = surf_loss + field_loss

        if self.config.debug and inputs.size(0) == 1 and self.training:
            self.visualize_results(
                data,
                mask_gt,
                image,
                depth,
                normals,
                p,
                p_pred,
                p_depth,
                mask_tp,
                mask_fp,
                mask_tn,
                mask_fn,
                h,
                w,
                u,
                v,
                extrinsic,
            )

        return data

    @torch.no_grad()
    def render(
        self,
        data: dict[str, list[str] | Tensor],
        depth: bool = True,
        color: bool = True,
        normals: bool = True,
        targets: bool = True,
        max_size: int | None = None,
        **kwargs,
    ) -> dict[str, Image.Image]:
        data = data.copy()
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value[0:1]
            else:
                data[key] = [value[0]]

        intrinsic = _require_tensor(data, "inputs.intrinsic")
        extrinsic = _require_tensor(data, "inputs.extrinsic")
        mask_gt = _require_tensor(data, "inputs.mask").bool()
        _b, h, w = mask_gt.size()

        image_gt = _optional_tensor(data.get("inputs.image", data.get("inputs")))
        if image_gt is not None and image_gt.ndim != 4:
            image_gt = None
        depth_gt = _optional_tensor(data.get("inputs.depth", data.get("inputs")))
        if depth_gt is not None and depth_gt.ndim != 3:
            depth_gt = None
        normals_gt = _optional_tensor(data.get("inputs.normals"))

        if image_gt is not None:
            if image_gt.size(1) == 3:
                image_gt = image_gt[0].view(1, 3, -1).transpose(1, 2)
            elif image_gt.size(-1) == 3:
                image_gt = image_gt[0].view(1, -1, 3)
            else:
                raise ValueError(f"Image must have shape (B, 3, H, W) or (B, H, W, 3): {image_gt.shape}")
            if image_gt.min() < 0:
                image_gt = image_gt * torch.tensor([0.229, 0.224, 0.225]).to(image_gt)
                image_gt = image_gt + torch.tensor([0.485, 0.456, 0.406]).to(image_gt)
            elif image_gt.max() > 1:
                image_gt = image_gt / 255.0
        if normals_gt is not None:
            if normals_gt.size(1) == 3:
                normals_gt = normals_gt[0].view(1, 3, -1).transpose(1, 2)
            elif normals_gt.size(-1) == 3:
                normals_gt = normals_gt[0].view(1, -1, 3)
            else:
                raise ValueError(f"Normals must have shape (B, 3, H, W) or (B, H, W, 3): {normals_gt.shape}")
            if normals_gt.min() < 0:
                normals_gt = (normals_gt + 1) / 2
            if normals_gt.max() > 1:
                normals_gt = normals_gt / 255.0

        if max_size and max_size < min(h, w):
            scale_factor = min(max_size / h, max_size / w)
            mask_gt = (
                F.interpolate(mask_gt.float().unsqueeze(1), scale_factor=scale_factor, mode="nearest-exact")
                .squeeze(1)
                .bool()
            )
            intrinsic = adjust_intrinsic(intrinsic, w, h, size=max(mask_gt.size()))
            if not torch.is_tensor(intrinsic):
                intrinsic = torch.from_numpy(intrinsic).to(extrinsic)
            if image_gt is not None:
                image_gt = (
                    F.interpolate(image_gt.transpose(1, 2).view(1, 3, h, w), scale_factor=scale_factor, mode="bilinear")
                    .view(1, 3, -1)
                    .transpose(1, 2)
                )
            if targets:
                if depth_gt is not None:
                    depth_gt = F.interpolate(
                        depth_gt.unsqueeze(1), scale_factor=scale_factor, mode="nearest-exact"
                    ).squeeze(1)
                if normals_gt is not None:
                    normals_gt = (
                        F.interpolate(
                            normals_gt.transpose(1, 2).view(1, 3, h, w), scale_factor=scale_factor, mode="bilinear"
                        )
                        .view(1, 3, -1)
                        .transpose(1, 2)
                    )
            _b, h, w = mask_gt.size()

        encode_kwargs = {k: v for k, v in {**data, **kwargs}.items()}
        feature = self.encode(**encode_kwargs)

        decode_kwargs = {
            k: v for k, v in {**data, **kwargs}.items() if k not in ["points", "inputs.intrinsic", "inputs.extrinsic"]
        }
        decode_kwargs["return_point_feature"] = True  # FIXME: Hack to make ONet work
        predict = partial(self.predict, **decode_kwargs)

        ray0, ray_dirs, _u, _v = self.get_rays(mask_gt, intrinsic, extrinsic)
        _d_pred, p_pred, mask_pred = self.ray_caster(
            ray0=ray0,
            ray_dirs=ray_dirs,
            predict=predict,
            parameters=list(cast(Any, self.model).decoder.parameters()),
            feature=feature,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )

        def combine_pil_images(img1: Image.Image, img2: Image.Image) -> Image.Image:
            if img1.size != img2.size:
                raise ValueError("Images must have the same size to combine them.")
            if img1.mode != img2.mode:
                raise ValueError("Images must have the same mode to combine them.")
            dst = Image.new(img1.mode, (img1.width + img2.width, img1.height))
            dst.paste(img1, (0, 0))
            dst.paste(img2, (img1.width, 0))
            return dst

        out = dict()
        if depth:
            out["depth"] = Image.fromarray(torch.zeros(h, w, 3).byte().numpy())
            if mask_pred.any():
                p_pred_cam = cast(Tensor, apply_trafo(p_pred, extrinsic))
                z_pred = p_pred_cam[..., -1]
                z_pred[~mask_pred] = 0
                out["depth"] = depth_to_image(z_pred.view(h, w).cpu().numpy())
            if targets and depth_gt is not None:
                depth_gt_pil = depth_to_image(depth_gt.view(h, w).cpu().numpy())
                out["depth"] = combine_pil_images(out["depth"], depth_gt_pil)

        if color or normals:
            if not mask_pred.any():
                if color:
                    out["color"] = Image.fromarray((255 * torch.ones(h, w, 3)).byte().numpy())
                if normals:
                    out["normals"] = Image.fromarray((255 * torch.ones(h, w, 3)).byte().numpy())
                return out

            enable_grad = normals or self.p_universal
            with torch.set_grad_enabled(enable_grad):
                points = p_pred[mask_pred].unsqueeze(0)
                if enable_grad and not points.requires_grad:
                    points.requires_grad = True
                out_pred = cast(
                    dict[str, Tensor],
                    predict(points=points, feature=feature, intrinsic=intrinsic, extrinsic=extrinsic, key=None),
                )

                normal_map = torch.zeros(h * w, 3)
                if normals:
                    normal_map[mask_pred[0]] = get_normals(points, out_pred["logits"])[0].view(-1, 3).cpu()
                    out["normals"] = Image.fromarray(((normal_map.view(h, w, 3) + 1) / 2 * 255).byte().numpy())
                    if targets and normals_gt is not None:
                        if normals_gt.min() < 0 or normals_gt.max() > 1:
                            raise ValueError("Normals must be in range [0, 1] for visualization.")

                        normals_gt_pil = Image.fromarray((normals_gt.view(h, w, 3) * 255).byte().cpu().numpy())
                        out["normals"] = combine_pil_images(out["normals"], normals_gt_pil)
            if color and self.renderer is not None:
                if self.p_universal:
                    assert self.renderer is not None
                    ray_dirs = ray_dirs[mask_pred].unsqueeze(0)
                    view_dirs = -ray_dirs / ray_dirs.norm(2, dim=-1, keepdim=True)
                    if normals:
                        normals_tensor = normal_map.view(-1, 3)[mask_pred[0].cpu()].unsqueeze(0).to(points)
                    else:
                        normals_tensor = get_normals(points, out_pred["logits"])
                    pred_rgb = self.renderer(
                        points=points, feature=out_pred["feature"], normals=normals_tensor, view_dirs=view_dirs
                    )
                else:
                    assert self.renderer is not None
                    pred_rgb = self.renderer(out_pred["feature"])

                colors = torch.zeros(h * w, 3, device=pred_rgb.device)
                colors[mask_pred[0]] = torch.sigmoid(pred_rgb[0].detach().view(-1, 3))
                out["color"] = Image.fromarray((colors.view(h, w, 3) * 255).byte().cpu().numpy())
                if image_gt is not None:
                    if image_gt.min() < 0 or image_gt.max() > 1:
                        raise ValueError("Image must be in range [0, 1] for visualization.")

                    if targets:
                        image_gt_pil = Image.fromarray((image_gt.view(h, w, 3) * 255).byte().cpu().numpy())
                        out["color"] = combine_pil_images(out["color"], image_gt_pil)

                    image_masked = torch.zeros(h * w, 3, device=image_gt.device)
                    image_masked[mask_gt[0].view(-1)] = image_gt[0].view(-1, 3)[mask_gt[0].view(-1)]
                    self.compute_metrics(colors.view(1, h, w, 3), image_masked.view(1, h, w, 3), prefix="vis/")

        return out

    def visualize_results(
        self,
        data: dict[str, list[str] | Tensor],
        mask_gt: Tensor,
        image: Tensor | None,
        depth: Tensor | None,
        normals: Tensor | None,
        p: Tensor,
        p_pred: Tensor,
        p_depth: Tensor | None,
        mask_tp: Tensor,
        mask_fp: Tensor,
        mask_tn: Tensor,
        mask_fn: Tensor,
        h: int,
        w: int,
        u: Tensor,
        v: Tensor,
        extrinsic: Tensor,
    ) -> None:
        """Visualize ray casting results for debugging purposes."""
        import open3d as o3d

        print(self.model.get_log())

        total_pixels = h * w

        if self.config.num_pixels is None or self.config.num_pixels >= total_pixels:
            Image.fromarray((mask_gt.view(h, w).cpu().numpy() * 255).astype("uint8")).show(title="Mask")
            if image is not None:
                image = image.transpose(1, 2).cpu()
                image *= torch.tensor([0.229, 0.224, 0.225])
                image += torch.tensor([0.485, 0.456, 0.406])
                Image.fromarray((image.view(h, w, 3) * 255).byte().numpy()).show(title="Image")
            if depth is not None:
                depth_to_image(depth.view(h, w).cpu().numpy()).show(title="Depth")
            if normals is not None:
                Image.fromarray(normals.cpu().byte().numpy()).show(title="Normals")
        else:
            sampled_mask = torch.zeros((h, w))
            sampled_mask[v[0].long(), u[0].long()] = 1.0
            full_mask = _require_tensor(data, "inputs.mask").squeeze(0).cpu()
            overlay = torch.zeros((h, w, 3))
            overlay[..., 1] = full_mask
            overlay[..., 0] = sampled_mask
            Image.fromarray(overlay.numpy().astype("uint8") * 255).show(title="Mask + Samples")
            if image is not None:
                image = image.transpose(1, 2).cpu()
                image = image * torch.tensor([0.229, 0.224, 0.225])
                image = image + torch.tensor([0.485, 0.456, 0.406])
                image_vis = torch.zeros((h, w, 3))
                image_vis[v[0].long(), u[0].long()] = image
                Image.fromarray((image_vis * 255).byte().numpy()).show(title="Image")
            if depth is not None:
                depth_vis = torch.zeros((h, w))
                depth_vis[v[0].long(), u[0].long()] = depth[0].cpu()
                depth_to_image(depth_vis.numpy()).show(title="Depth")
            if normals is not None:
                normals = normals[0]
                if normals.min() < 0:
                    normals = (normals + 1) / 2
                if normals.max() > 1:
                    normals = normals / 255.0
                normals_vis = torch.zeros((h, w, 3))
                normals_vis[v[0].long(), u[0].long()] = normals.cpu()
                Image.fromarray((normals_vis * 255).byte().numpy()).show(title="Normals")

        p_free = p[~mask_gt]
        p_occ = p[mask_gt]
        p_tp = p_pred[mask_tp].detach()
        p_fp = p_pred[mask_fp].detach()
        p_tn = p[mask_tn]
        p_fn = p[mask_fn]

        pcd_depth = o3d.geometry.PointCloud()
        if p_depth is not None:
            pcd_depth = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_depth[mask_gt].view(-1, 3).cpu().numpy()))
        pcd_tp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_tp.view(-1, 3).cpu().numpy()))
        pcd_fp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_fp.view(-1, 3).cpu().numpy()))
        pcd_tn = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_tn.view(-1, 3).cpu().numpy()))
        pcd_fn = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_fn.view(-1, 3).cpu().numpy()))
        pcd_free = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_free.view(-1, 3).cpu().numpy()))
        pcd_occ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_occ.view(-1, 3).cpu().numpy()))

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-0.5 - self.config.padding / 2,) * 3, max_bound=(0.5 + self.config.padding / 2,) * 3
        )
        bbox.color = [0, 0, 0]
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(
            torch.inverse(extrinsic[0]).cpu().numpy()
        )

        o3d_module = cast(Any, o3d)
        o3d_module.visualization.draw_geometries(
            [
                pcd_free.paint_uniform_color((0.7,) * 3),
                pcd_occ.paint_uniform_color((0.3,) * 3),
                pcd_depth.paint_uniform_color((0, 0, 0)),
                pcd_depth,
                pcd_tp.paint_uniform_color((0, 1, 0)),
                pcd_fp.paint_uniform_color((1, 0, 0)),
                pcd_tn.paint_uniform_color((0, 0, 1)),
                pcd_fn.paint_uniform_color((1, 1, 0)),
                world,
                cam,
                bbox,
            ],
            window_name=cast(list[str], data["inputs.path"])[0],
        )

    def encode(self, *args, **kwargs) -> Tensor:
        inputs = kwargs.get("inputs")
        if isinstance(inputs, Tensor):
            is_pcd_or_img = inputs.ndim == 4 or (inputs.ndim == 3 and inputs.size(-1) == 3)
            if not is_pcd_or_img and isinstance(kwargs.get("inputs.image"), Tensor):
                kwargs["inputs"] = kwargs["inputs.image"]
        return cast(Tensor, cast(Any, self.model).encode(*args, **kwargs))

    def decode(self, *args, **kwargs) -> dict[str, Tensor]:
        return cast(dict[str, Tensor], cast(Any, self.model).decode(*args, **kwargs))

    def forward(self, *args, **kwargs) -> dict[str, Tensor]:
        return self.decode(*args, feature=self.encode(*args, **kwargs), **kwargs)

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        global_step: int | None = None,
        total_steps: int | None = None,
        **kwargs,
    ) -> Tensor:
        inputs = _require_tensor(data, "inputs")
        if global_step and total_steps and not isinstance(self.config.num_steps, int):
            if self.config.step_func == "cosine":
                start, stop = cast(tuple[int, int], self.config.num_steps)
                self.ray_caster.num_steps = int(cosine_anneal(start, stop, total_steps, global_step))
            else:
                steps = len(self.config.num_steps)
                if self.config.step_func == "half":
                    thresholds = [total_steps * (1 - 1 / (2**i)) for i in range(1, steps)]
                elif self.config.step_func == "equal":
                    thresholds = total_steps * torch.linspace(1 / steps, (steps - 1) / steps, steps - 1)
                idx = min(sum(global_step >= t.item() for t in thresholds), steps - 1)
                self.ray_caster.num_steps = self.config.num_steps[idx]
            self.model.log("num_steps", self.ray_caster.num_steps, level=DEBUG_LEVEL_1, train_only=True)

        if self.config.num_views > 1:  # Multi-view case
            # import numpy as np
            # import open3d as o3d
            # geometries = list()

            batch_size = inputs.shape[0]
            device = inputs.device
            loss = torch.zeros((), device=device)
            fc_loss = torch.zeros((), device=device)
            gc_loss = torch.zeros((), device=device)

            angle = torch.rand(batch_size, device=device) * 2 * torch.pi
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            trafo = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            trafo[:, 0, 0] = cos_a
            trafo[:, 0, 2] = sin_a
            trafo[:, 2, 0] = -sin_a
            trafo[:, 2, 2] = cos_a

            # trafo[:, :3, :3] = data["inputs.extrinsic"][:, 0, :3, :3]

            for i in range(self.config.num_views):
                item = {k: v[:, i] if "inputs" in k and torch.is_tensor(v) else v for k, v in data.items()}
                if i == 0 or self.feature_consistency_loss:
                    item_inputs = _require_tensor(item, "inputs")
                    if item_inputs.ndim == 3 and item_inputs.size(-1) == 3:
                        transformed = apply_trafo(item_inputs, inv_trafo(trafo))
                        item["inputs"] = cast(Tensor, transformed)
                        # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs.view(-1, 3).cpu().numpy()))
                        # geometries.append(pcd.paint_uniform_color((0, 0, 0)))

                    encode_kwargs = {k: v for k, v in {**item, **kwargs}.items()}
                    feature_i = self.encode(**encode_kwargs)
                intrinsic_i = _require_tensor(item, "inputs.intrinsic")
                extrinsic_i = _require_tensor(item, "inputs.extrinsic") @ trafo
                item["inputs.extrinsic"] = extrinsic_i

                points_i, mask_i = self.get_points(cast(dict[str, Tensor], item))
                occ_i = torch.zeros_like(mask_i).float()
                occ_i[mask_i] = 1.0
                """
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_i[mask_i].view(-1, 3).cpu().numpy()))
                geometries.append(pcd.paint_uniform_color(np.random.rand(3)))

                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                frame.transform(torch.inverse(extrinsic_i[0]).cpu().numpy())
                geometries.append(frame)
                """

                if i == 0:
                    feature = feature_i
                    intrinsic = intrinsic_i
                    extrinsic = extrinsic_i
                    points = points_i
                    occ = occ_i
                    loss = cast(Tensor, self.step(item, feature=feature, **kwargs)["loss"])
                    fc_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
                    gc_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)

                decode_kwargs = {
                    k: v
                    for k, v in {**item, **kwargs}.items()
                    if k not in ["points", "inputs.intrinsic", "inputs.extrinsic"]
                }

                if self.feature_consistency_loss:
                    logits = cast(
                        Tensor,
                        self.predict(
                            points=points,
                            feature=feature_i,
                            intrinsic=intrinsic_i,
                            extrinsic=extrinsic_i,
                            **decode_kwargs,
                        ),
                    )
                    if self.focal_loss_alpha:
                        fc_loss += 2 * focal_loss.sigmoid_focal_loss(
                            logits, occ_i, alpha=self.focal_loss_alpha, reduction="mean"
                        )
                    else:
                        fc_loss += F.binary_cross_entropy_with_logits(logits, occ)

                if self.geometry_consistency_loss:
                    logits = cast(
                        Tensor,
                        self.predict(
                            points=points_i, feature=feature, intrinsic=intrinsic, extrinsic=extrinsic, **decode_kwargs
                        ),
                    )
                    if self.focal_loss_alpha:
                        gc_loss += 2 * focal_loss.sigmoid_focal_loss(
                            logits, occ_i, alpha=self.focal_loss_alpha, reduction="mean"
                        )
                    else:
                        gc_loss += F.binary_cross_entropy_with_logits(logits, occ_i)

            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            # o3d.visualization.draw_geometries(geometries + [frame])

            if self.feature_consistency_loss:
                fc_loss /= self.config.num_views
                self.log("fc_loss", fc_loss.item(), level=DEBUG_LEVEL_1)
                loss += self.feature_consistency_loss * fc_loss
            if self.geometry_consistency_loss:
                gc_loss /= self.config.num_views
                self.log("gc_loss", gc_loss.item(), level=DEBUG_LEVEL_1)
                loss += self.geometry_consistency_loss * gc_loss
        else:
            encode_kwargs = {k: v for k, v in {**data, **kwargs}.items()}
            feature = self.encode(**encode_kwargs)
            loss = cast(Tensor, self.step(data, feature=feature, **kwargs)["loss"])
        return cast(Tensor, loss)

    @torch.no_grad()
    def evaluate(self, data: dict[str, list[str] | Tensor], **kwargs) -> dict[str, float | Tensor]:
        if "points" in data and "points.occ" in data:
            inputs = _require_tensor(data, "inputs")
            is_pcd_or_img = inputs.ndim == 4 or (inputs.ndim == 3 and inputs.size(-1) == 3)
            if not is_pcd_or_img and "inputs.image" in data:
                data["inputs"] = data["inputs.image"]
            return cast(dict[str, float | Tensor], cast(Any, self.model).evaluate(data, **kwargs))

        encode_kwargs = {k: v for k, v in {**data, **kwargs}.items()}
        feature = self.encode(**encode_kwargs)
        data = self.step(data, feature=feature, **kwargs)
        return cast(dict[str, float | Tensor], cast(Any, self.model).evaluate(data, **kwargs))

    @torch.autocast(device_type="cuda", enabled=False)
    def predict(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        feature: Tensor | list[Tensor] | dict[str, Tensor] | None = None,
        intrinsic: Tensor | None = None,
        extrinsic: Tensor | None = None,
        key: Literal["logits", "colors", "auto"] | str | None = "auto",
        points_batch_size: int | None = None,
        **kwargs,
    ) -> Tensor | dict[str, Tensor]:
        if intrinsic is not None:
            kwargs["inputs.intrinsic"] = intrinsic
        if extrinsic is not None:
            kwargs["inputs.extrinsic"] = extrinsic

        if points is None:
            raise ValueError("`points` must be provided for prediction")

        key_name = "logits" if key == "auto" else key

        points_per_call = points_batch_size if points_batch_size is not None else self.config.num_points
        if points_per_call is None:
            points_per_call = points.size(1)
        if points_per_call <= 0:
            raise ValueError(f"Invalid points batch size: {points_per_call}")

        if points.size(1) > points_per_call:
            p_split = torch.split(points, points_per_call, dim=1)
            out: dict[str, Tensor] = {}
            for i, pi in enumerate(p_split):
                out_i = self.decode(points=pi, feature=feature, inputs=inputs, **kwargs)
                if i == 0:
                    out.update(out_i)
                    continue

                for k, v in out_i.items():
                    if key_name is None or k == key_name:
                        out[k] = torch.cat((out[k], v), dim=1)
            return out if key_name is None else out[key_name]

        batch_size_per_call = self.config.max_batch_size if self.config.max_batch_size is not None else points.size(0)
        if batch_size_per_call <= 0:
            raise ValueError(f"Invalid batch size: {batch_size_per_call}")

        if points.size(0) > batch_size_per_call:
            p_split = torch.split(points, batch_size_per_call, dim=0)
            if feature is None:
                f_split: list[Tensor | list[Tensor] | dict[str, Tensor] | None] = [None] * len(p_split)
            elif torch.is_tensor(feature):
                f_split = list(torch.split(feature, batch_size_per_call, dim=0))
            else:
                f_split = [feature] * len(p_split)
            i_split = [None] * len(p_split)
            e_split = [None] * len(p_split)
            if "inputs.intrinsic" in kwargs:
                intrinsic_val = kwargs.pop("inputs.intrinsic")
                if isinstance(intrinsic_val, Tensor):
                    i_split = list(torch.split(intrinsic_val, batch_size_per_call, dim=0))
            if "inputs.extrinsic" in kwargs:
                extrinsic_val = kwargs.pop("inputs.extrinsic")
                if isinstance(extrinsic_val, Tensor):
                    e_split = list(torch.split(extrinsic_val, batch_size_per_call, dim=0))

            out: dict[str, Tensor] = {}
            for i, (pi, fi, ii, ei) in enumerate(zip(p_split, f_split, i_split, e_split, strict=False)):
                out_i = self.decode(
                    points=pi,
                    feature=cast(Tensor | list[Tensor] | dict[str, Tensor] | None, fi),
                    # inputs=inputs,  FIXME: implement batching for inputs
                    **{
                        **kwargs,
                        "inputs.intrinsic": ii,
                        "inputs.extrinsic": ei,
                    },
                )
                if i == 0:
                    out.update(out_i)
                    continue

                for k, v in out_i.items():
                    if key_name is None or k == key_name:
                        out[k] = torch.cat((out[k], v), dim=0)
            return out if key_name is None else out[key_name]

        out = self.decode(points=points, feature=feature, inputs=inputs, **kwargs)
        return out if key_name is None else out[key_name]
