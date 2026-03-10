from typing import Any, Literal, overload

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pointnet import GridEncoder


class PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int | None,
        radius: float | None,
        nsample: int | None,
        in_channel: int,
        mlp: list[int],
        group_all: bool,
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz: Tensor, points: Tensor | None) -> tuple[Tensor, Tensor]:
        xyz = xyz.transpose(1, 2)
        if points is not None:
            points = points.transpose(1, 2)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            if self.npoint is None or self.radius is None or self.nsample is None:
                raise ValueError("npoint, radius, and nsample must be set when group_all is False")
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.transpose(1, 2)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, mlp: list[int]) -> None:
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1: Tensor, xyz2: Tensor, points1: Tensor | None, points2: Tensor) -> Tensor:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)

        points2 = points2.transpose(1, 2)
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.transpose(1, 2)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose(1, 2)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNetPlusPlus(GridEncoder):
    def __init__(
        self,
        dim: int = 3,
        c_dim: int = 128,
        hidden_dim: int | None = None,
        scatter_type: str = "mean",
        unet: bool = False,
        unet_kwargs: dict[str, Any] | None = None,
        unet3d: bool = False,
        unet3d_kwargs: dict[str, Any] | None = None,
        plane_resolution: int | None = None,
        grid_resolution: int | None = None,
        feature_type: tuple[str, ...] = ("grid",),
        padding: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            c_dim,
            unet,
            unet_kwargs,
            unet3d,
            unet3d_kwargs,
            plane_resolution,
            grid_resolution,
            feature_type,
            scatter_type,
            padding,
            **kwargs,
        )

        if hidden_dim is None:
            dim_reduction = int(np.log2(128) - np.log2(c_dim))
            dim32 = 2 ** (5 - dim_reduction)
            dim64 = 2 ** (6 - dim_reduction)
            dim128 = 2 ** (7 - dim_reduction)
            dim256 = 2 ** (8 - dim_reduction)
            dim512 = 2 ** (9 - dim_reduction)
            dim1024 = 2 ** (10 - dim_reduction)
        else:
            dim32 = dim64 = dim128 = dim256 = dim512 = dim1024 = hidden_dim

        self.sa1 = PointNetSetAbstraction(
            npoint=dim512, radius=0.2, nsample=dim32, in_channel=2 * dim, mlp=[dim64, dim64, dim128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=dim128,
            radius=0.4,
            nsample=dim64,
            in_channel=dim128 + 3,
            mlp=[dim128, dim128, dim256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=dim256 + 3, mlp=[dim256, dim512, dim1024], group_all=True
        )
        self.fp3 = PointNetFeaturePropagation(in_channel=dim1024 + dim256, mlp=[dim256, dim256])
        self.fp2 = PointNetFeaturePropagation(in_channel=dim256 + dim128, mlp=[dim256, dim128])
        self.fp1 = PointNetFeaturePropagation(in_channel=dim128, mlp=[dim128, dim128, dim128])

        self.conv_out = nn.Conv1d(dim128, c_dim, 1)

    def forward(self, points: Tensor, **kwargs) -> dict[str, Tensor | list[Tensor]]:
        xyz = points.transpose(1, 2)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        feature = self.conv_out(l0_points)
        return self.generate_feature(feature, self.get_index_dict(points))


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


@overload
def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: Tensor,
    points: Tensor | None,
    returnfps: Literal[False] = False,
) -> tuple[Tensor, Tensor]: ...


@overload
def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: Tensor,
    points: Tensor | None,
    returnfps: Literal[True],
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: Tensor,
    points: Tensor | None,
    returnfps: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    B, _, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
