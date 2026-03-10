from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mixins import MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model


def get_sphere_intersection(cam_loc: Tensor, ray_directions: Tensor, r: float = 1.0) -> tuple[Tensor, Tensor]:
    """
    Compute sphere intersections for ray tracing.

    Args:
        cam_loc: Camera locations (n_images x 3)
        ray_directions: Ray directions (n_images x n_rays x 3)
        r: Sphere radius

    Returns:
        Tuple of (sphere_intersections, mask_intersect)
    """
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape

    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot**2 - (cam_loc.norm(2, 1) ** 2 - r**2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0

    sphere_intersections = torch.zeros(n_imgs * n_pix, 2, device=cam_loc.device, dtype=torch.float32)
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.tensor(
        [-1, 1], device=cam_loc.device, dtype=torch.float32
    )
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect


class Embedder:
    """Positional encoding embedder for neural networks."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.embed_fns: list[Callable[[Tensor], Tensor]] = []
        self.out_dim: int = 0
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(torch.pi / 2 * x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: Tensor) -> Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(
    num_freqs: int, include_input: bool = True, input_dims: int = 3
) -> tuple[Callable[[Tensor], Tensor], int]:
    """
    Create embedder function and return embedding dimension.

    Args:
        multires: Number of frequency bands

    Returns:
        Tuple of (embed_function, output_dimension)
    """
    embed_kwargs = {
        "include_input": include_input,
        "input_dims": input_dims,
        "max_freq_log2": num_freqs - 1,
        "num_freqs": num_freqs,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


class IDRLoss(nn.Module):
    """Loss function for IDR training."""

    def __init__(self, eikonal_weight: float, mask_weight: float, alpha: float) -> None:
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction="sum")

    def get_rgb_loss(
        self, rgb_values: Tensor, rgb_gt: Tensor, network_object_mask: Tensor, object_mask: Tensor
    ) -> Tensor:
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_eikonal_loss(self, grad_theta: Tensor) -> Tensor:
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output: Tensor, network_object_mask: Tensor, object_mask: Tensor) -> Tensor:
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (
            (1 / self.alpha)
            * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction="sum")
            / float(object_mask.shape[0])
        )
        return mask_loss

    def forward(self, model_outputs: dict[str, Tensor], ground_truth: dict[str, Tensor]) -> dict[str, Tensor]:
        rgb_gt = ground_truth["rgb"].cuda()
        network_object_mask = model_outputs["network_object_mask"]
        object_mask = model_outputs["object_mask"]

        rgb_loss = self.get_rgb_loss(model_outputs["rgb_values"], rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs["sdf_output"], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs["grad_theta"])

        loss = rgb_loss + self.eikonal_weight * eikonal_loss + self.mask_weight * mask_loss

        return {
            "loss": loss,
            "rgb_loss": rgb_loss,
            "eikonal_loss": eikonal_loss,
            "mask_loss": mask_loss,
        }


class ImplicitNetwork(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    """Neural implicit network for SDF representation."""

    def __init__(
        self,
        feature_vector_size: int = 256,
        d_in: int = 3,
        d_out: int = 1,
        dims: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512, 512),
        geometric_init: bool = True,
        bias: float = 0.6,
        skip_in: tuple[int, ...] = (4,),
        weight_norm: bool = True,
        multires: int = 4,
        n_objs: int = 1,
    ):
        super().__init__()
        self.encoder = lambda *args, **kwargs: None
        if n_objs > 1:
            self.encoder = nn.Embedding(num_embeddings=n_objs, embedding_dim=max(dims))
            torch.nn.init.normal_(self.encoder.weight, 0.0, 0.01)

        dims_list = [d_in, *list(dims), d_out + feature_vector_size]
        self.embed_fn: Callable[[Tensor], Tensor] | None = None
        if multires > 0:
            """
            self.embed_fn = NeRFEncoding(in_dim=d_in,
                                         num_frequencies=multires,
                                         max_freq_exp=multires - 1,
                                         normalize_inputs=False,
                                         scale_inputs=False,
                                         implementation="torch")
            dims[0] = self.embed_fn.get_enc_dim()
            """

            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims_list[0] = input_ch

        if isinstance(self.encoder, nn.Embedding):
            dims_list[0] += self.encoder.embedding_dim
            for skip_idx in skip_in:
                dims_list[skip_idx] += self.encoder.embedding_dim

        self.num_layers = len(dims_list)
        self.skip_in = skip_in
        self.decoder = nn.ModuleList()
        for layer_idx in range(0, self.num_layers - 1):
            if layer_idx + 1 in self.skip_in:
                out_dim = dims_list[layer_idx + 1] - dims_list[0]
            else:
                out_dim = dims_list[layer_idx + 1]
            lin = nn.Linear(dims_list[layer_idx], out_dim)
            if geometric_init:
                if layer_idx == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims_list[layer_idx]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and layer_idx == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and layer_idx in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_list[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            setattr(self, "lin" + str(layer_idx), lin)
            self.decoder.append(lin)
        self.softplus = nn.Softplus(beta=100)

    def _forward(self, points: Tensor, feature: Tensor | None = None, **kwargs) -> Tensor:
        if self.embed_fn is not None:
            points = self.embed_fn(2 * points)
        x = points
        if feature is not None:
            feature = feature.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, feature], dim=-1) / np.sqrt(2)
        for layer_idx in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer_idx))
            if layer_idx in self.skip_in:
                x = torch.cat([x, points], dim=2) / np.sqrt(2)
                if feature is not None:
                    x = torch.cat([x, feature], dim=-1) / np.sqrt(2)
            x = lin(x)
            if layer_idx < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def forward(self, *args, **kwargs) -> dict[str, Tensor]:
        return self.decode(*args, feature=self.encode(*args, **kwargs), **kwargs)

    def encode(self, *args, **kwargs) -> Tensor | None:
        if isinstance(self.encoder, nn.Embedding):
            return self.encoder(kwargs["category.index"])
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> dict[str, Tensor]:
        x = self._forward(*args, **kwargs)
        if x.size(-1) == 1:
            return dict(logits=x[..., 0])
        elif x.size(-1) == 4:
            return dict(logits=x[..., 0], colors=x[..., 1:])
        return dict(logits=x[..., 0], feature=x[..., 1:])

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
        **kwargs,
    ) -> Tensor:
        return super().loss(data, regression, name, reduction, **kwargs)["occ_loss"]

    @torch.enable_grad()
    def normals(self, points: Tensor, logits: Tensor | None = None, normalize: bool = True) -> Tensor:
        if not points.requires_grad:
            points.requires_grad = True
        if logits is None:
            logits = self._forward(points)
        grad = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=points,
            create_graph=self.training,
            retain_graph=self.training,
            only_inputs=True,
        )[0]
        if normalize:
            return -grad / grad.norm(2, dim=-1, keepdim=True)
        return -grad


class RenderingNetwork(nn.Module):
    def __init__(
        self,
        feat_dim: int = 256,
        d_in: int = 3,
        d_out: int = 3,
        dims: tuple[int, ...] = (512, 512, 512, 512),
        view_dirs: bool = True,
        normals: bool = True,
        weight_norm: bool = True,
        multires_view: int = 4,
    ):
        super().__init__()

        dims_list = [d_in + feat_dim + 3 * sum([view_dirs, normals]), *list(dims), d_out]

        self.embedview_fn = None
        if view_dirs and multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims_list[0] += input_ch - d_in

        self.num_layers = len(dims_list)

        for layer_idx in range(self.num_layers - 1):
            lin = nn.Linear(dims_list[layer_idx], dims_list[layer_idx + 1])

            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)

            setattr(self, "lin" + str(layer_idx), lin)

        self.relu = nn.ReLU()

    def forward(
        self, points: Tensor, feature: Tensor, normals: Tensor | None = None, view_dirs: Tensor | None = None
    ) -> Tensor:
        if view_dirs is not None and self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        x = torch.cat([points, feature], dim=-1)
        if normals is not None:
            x = torch.cat([x, normals], dim=-1)
        if view_dirs is not None:
            x = torch.cat([x, view_dirs], dim=-1)

        for layer_idx in range(self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer_idx))

            x = lin(x)

            if layer_idx < self.num_layers - 2:
                x = self.relu(x)
        return x


class RayTracing(nn.Module):
    """Ray tracing module for surface intersection."""

    def __init__(
        self,
        object_bounding_sphere: float = 1.0,
        sdf_threshold: float = 5.0e-5,
        line_search_step: float = 0.5,
        line_step_iters: int = 1,
        sphere_tracing_iters: int = 10,
        n_steps: int = 100,
        n_secant_steps: int = 8,
    ) -> None:
        super().__init__()
        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps

    def forward(
        self, sdf: Callable[[Tensor], Tensor], cam_loc: Tensor, object_mask: Tensor, ray_directions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_pixels, _ = ray_directions.shape
        sphere_intersections, mask_intersect = get_sphere_intersection(
            cam_loc, ray_directions, r=self.object_bounding_sphere
        )
        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, _, _ = self.sphere_tracing(
            batch_size,
            num_pixels,
            sdf,
            cam_loc,
            ray_directions,
            mask_intersect,
            sphere_intersections,
        )
        network_object_mask = acc_start_dis < acc_end_dis
        sampler_mask = unfinished_mask_start
        if sampler_mask.sum() > 0:
            sampler_min_max = torch.zeros((batch_size, num_pixels, 2), device=cam_loc.device)
            sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[sampler_mask]
            sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]
            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(
                sdf,
                cam_loc,
                object_mask,
                ray_directions,
                sampler_min_max,
                sampler_mask,
            )
            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]
        return curr_start_points, network_object_mask, acc_start_dis

    def sphere_tracing(
        self,
        batch_size: int,
        num_pixels: int,
        sdf: Callable[[Tensor], Tensor],
        cam_loc: Tensor,
        ray_directions: Tensor,
        mask_intersect: Tensor,
        sphere_intersections: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        sphere_intersections_points = cam_loc.unsqueeze(1).unsqueeze(1) + sphere_intersections.unsqueeze(
            -1
        ) * ray_directions.unsqueeze(2)
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()

        curr_start_points = torch.zeros(batch_size * num_pixels, 3, device=cam_loc.device).float()
        if unfinished_mask_start.any():
            curr_start_points[unfinished_mask_start] = sphere_intersections_points[:, :, 0, :].reshape(-1, 3)[
                unfinished_mask_start
            ]
        acc_start_dis = torch.zeros(batch_size * num_pixels, device=cam_loc.device).float()
        if unfinished_mask_start.any():
            acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1, 2)[unfinished_mask_start, 0]

        acc_end_dis = torch.zeros(batch_size * num_pixels, device=cam_loc.device).float()
        if unfinished_mask_end.any():
            acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1, 2)[unfinished_mask_end, 1]

        iters = 0
        next_sdf_start = torch.zeros_like(acc_start_dis)
        if unfinished_mask_start.any():
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        while True:
            curr_sdf_start = torch.zeros_like(acc_start_dis)
            if unfinished_mask_start.any():
                curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            if unfinished_mask_start.sum() == 0 or iters == self.sphere_tracing_iters:
                break
            iters += 1
            acc_start_dis = acc_start_dis + curr_sdf_start
            curr_start_points = (
                cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions
            ).reshape(-1, 3)
            next_sdf_start = torch.zeros_like(acc_start_dis)  # Default to 0 for finished rays
            if unfinished_mask_start.any():  # Only compute SDF for unfinished rays
                next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])
            not_projected_start = next_sdf_start < 0
            not_proj_iters = 0
            while not_projected_start.sum() > 0 and not_proj_iters < self.line_step_iters:
                acc_start_dis[not_projected_start] -= (
                    (1 - self.line_search_step) / (2**not_proj_iters)
                ) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (
                    cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions
                ).reshape(-1, 3)[not_projected_start]
                if not_projected_start.any():  # Re-evaluate SDF
                    next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])
                else:  # Avoid re-evaluation if no points are in not_projected_start
                    next_sdf_start[not_projected_start] = 0  # or some other appropriate value
                not_projected_start = next_sdf_start < 0
                not_proj_iters += 1
            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)  # Ensure start is before end
        return (
            curr_start_points,
            unfinished_mask_start,
            acc_start_dis,
            acc_end_dis,
            acc_start_dis.clone(),
            acc_end_dis.clone(),
        )

    def ray_sampler(
        self,
        sdf: Callable[[Tensor], Tensor],
        cam_loc: Tensor,
        object_mask: Tensor,
        ray_directions: Tensor,
        sampler_min_max: Tensor,
        sampler_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_pixels, _ = ray_directions.shape
        n_total_pxl = batch_size * num_pixels
        sampler_pts = torch.zeros(n_total_pxl, 3, device=cam_loc.device).float()
        sampler_dists = torch.zeros(n_total_pxl, device=cam_loc.device).float()
        intervals_dist = torch.linspace(0, 1, steps=self.n_steps, device=cam_loc.device).view(1, 1, -1)
        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (
            sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]
        ).unsqueeze(-1)
        points = cam_loc.unsqueeze(1).unsqueeze(1) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        if mask_intersect_idx.numel() == 0:  # No rays to sample
            return sampler_pts, sampler_mask.clone(), sampler_dists

        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1, device=cam_loc.device).float().reshape(
            (1, self.n_steps)
        )
        sampler_pts_ind = torch.argmin(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        true_surface_pts = object_mask[sampler_mask]
        net_surface_pts = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = int(p_out_mask.sum().item())
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][
                torch.arange(n_p_out), out_pts_idx, :
            ]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][
                torch.arange(n_p_out), out_pts_idx
            ]
        sampler_net_obj_mask = sampler_mask.clone()
        if net_surface_pts.numel() > 0 and (~net_surface_pts).any():  # Make sure indices are valid
            if mask_intersect_idx[~net_surface_pts].numel() > 0:  # check if the indexing tensor is not empty
                sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        secant_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_secant_pts = int(secant_pts.sum().item())
        if n_secant_pts > 0:
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            cam_loc_secant = (
                cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            )
            ray_directions_secant = ray_directions.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)
            sampler_pts[mask_intersect_idx[secant_pts]] = (
                cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            )
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant
        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(
        self,
        sdf_low: Tensor,
        sdf_high: Tensor,
        z_low: Tensor,
        z_high: Tensor,
        cam_loc: Tensor,
        ray_directions: Tensor,
        sdf: Callable[[Tensor], Tensor],
    ) -> Tensor:
        z_pred = -sdf_low * (z_high - z_low) / (sdf_high - sdf_low + 1e-8) + z_low  # Added epsilon
        for _i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]
            z_pred = -sdf_low * (z_high - z_low) / (sdf_high - sdf_low + 1e-8) + z_low  # Added epsilon
        return z_pred


class SampleNetwork(nn.Module):
    """Network for sampling surface points."""

    def forward(
        self,
        surface_output: Tensor,
        surface_sdf_values: Tensor,
        surface_points_grad: Tensor,
        surface_dists: Tensor,
        surface_cam_loc: Tensor,
        surface_ray_dirs: Tensor,
    ) -> Tensor:
        surface_ray_dirs_0 = surface_ray_dirs.detach()
        surface_points_dot = torch.bmm(surface_points_grad.view(-1, 1, 3), surface_ray_dirs_0.view(-1, 3, 1)).squeeze(
            -1
        )
        # Avoid division by zero
        surface_points_dot = torch.where(
            torch.abs(surface_points_dot) < 1e-8,
            torch.sign(surface_points_dot) * 1e-8
            + (1e-8 * (surface_points_dot == 0).float()),  # add 1e-8 if it's exactly 0
            surface_points_dot,
        )

        surface_dists_theta = (
            surface_dists - (surface_output.squeeze(-1) - surface_sdf_values.squeeze(-1)) / surface_points_dot
        )
        surface_points_theta_c_v = surface_cam_loc + surface_dists_theta.unsqueeze(-1) * surface_ray_dirs
        return surface_points_theta_c_v


class DifferentiableSphereTracer(nn.Module):
    """Differentiable ray casting for neural rendering."""

    def __init__(self, ray_caster_config: dict | None = None, object_bounding_sphere: float = 1.0) -> None:
        super().__init__()
        if ray_caster_config is None:
            ray_caster_config = {}
        self.ray_caster = RayTracing(object_bounding_sphere=object_bounding_sphere, **ray_caster_config)
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = object_bounding_sphere

    def forward(
        self, ray0: Tensor, ray_direction: Tensor, object_mask: Tensor, model: Model
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_pixels, _ = ray_direction.shape
        cam_loc_for_tracer = ray0[
            :, 0, :
        ]  # Assumes ray origins are consistent for a batch, or uses first pixel's origin

        model.eval()
        with torch.no_grad():
            points_surface_raw, network_object_mask_flat, dists_flat = self.ray_caster(
                sdf=lambda points: model(points=points)["logits"],
                cam_loc=cam_loc_for_tracer,
                object_mask=object_mask.view(-1),
                ray_directions=ray_direction,
            )
        model.train()

        d_pred_no_grad = dists_flat.reshape(batch_size, num_pixels)
        network_object_mask = network_object_mask_flat.reshape(batch_size, num_pixels)
        x0 = points_surface_raw.reshape(batch_size, num_pixels, 3)  # x0 from sphere tracing
        t0 = d_pred_no_grad  # t0 from sphere tracing

        mask_for_diff_points = network_object_mask.reshape(-1)
        x0_flat = x0.reshape(-1, 3)[mask_for_diff_points]
        t0_flat = t0.reshape(-1)[mask_for_diff_points]
        ray_direction_flat = ray_direction.reshape(-1, 3)[mask_for_diff_points]

        d_out = torch.ones_like(t0) * self.object_bounding_sphere
        p_out = ray0 + d_out.unsqueeze(-1) * ray_direction

        if x0_flat.shape[0] > 0:
            x0_flat_nograd = (
                x0_flat.detach()
            )  # Use detached x0 for gradient calculation as per IDR paper's theta0 context
            x0_flat_nograd.requires_grad = True  # Enable grad for this detached copy for grad_at_x0

            # SDF and gradient at x0 (effectively with theta_0 for gradient calculation)
            implicit_output_at_x0_for_grad = model(points=x0_flat_nograd)["logits"]
            model_any = cast(Any, model)
            if hasattr(model_any, "gradient"):
                grad_at_x0_detached = model_any.gradient(x0_flat_nograd).squeeze(1).detach()
            else:
                grad_at_x0_detached = model_any.normals(
                    x0_flat_nograd, logits=implicit_output_at_x0_for_grad, normalize=False
                ).detach()

            # SDF at x0 with current theta for the numerator f(c+t0v; theta)
            # We use x0_flat which carries gradients from implicit_network(x0_flat)
            implicit_output_at_x0_for_sdf = model(points=x0_flat)[
                "logits"
            ]  # This x0_flat tracks grads to implicit_network
            sdf_at_x0_theta = implicit_output_at_x0_for_sdf

            grad_dot_ray = (grad_at_x0_detached * ray_direction_flat.detach()).sum(dim=-1)
            grad_dot_ray = torch.where(
                torch.abs(grad_dot_ray) < 1e-8,
                torch.sign(grad_dot_ray) * 1e-8 + (1e-8 * (grad_dot_ray == 0).float()),
                grad_dot_ray,
            )

            d_pred_flat_diff = t0_flat - sdf_at_x0_theta / grad_dot_ray

            d_pred_diff_full = torch.ones_like(t0) * self.object_bounding_sphere
            d_pred_diff_full.view(-1)[mask_for_diff_points] = d_pred_flat_diff
            p_pred_diff_full = ray0 + d_pred_diff_full.unsqueeze(-1) * ray_direction
            d_out = d_pred_diff_full
            p_out = p_pred_diff_full

        return d_out, p_out, network_object_mask
