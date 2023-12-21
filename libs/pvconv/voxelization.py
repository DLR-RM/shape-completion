from typing import Tuple

import torch
from torch import nn, Tensor

from .functional.voxelization import avg_voxelize


def get_voxel_coords(points_t: Tensor,  # [B, 3, N]
                     resolution: int,
                     normalize: bool = False,
                     eps: float = 0.0) -> Tensor:
    points_t = points_t.detach()
    norm_coords = points_t - points_t.mean(2, keepdim=True)  # centering
    if normalize:
        norm_coords = norm_coords / (
                    norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + eps) + 0.5
    else:
        norm_coords = (norm_coords + 1) / 2.0  # [0, 1]
    return torch.clamp(norm_coords * resolution, 0, resolution - 1)  # [0, r - 1]


class Voxelization(nn.Module):
    def __init__(self, resolution: int, normalize: bool = False, eps: float = 0.0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features: Tensor, points: Tensor) -> Tuple[Tensor, Tensor]:
        voxel_coords = get_voxel_coords(points.transpose(1, 2), self.r, self.normalize, self.eps)
        return avg_voxelize(features, torch.round(voxel_coords).to(torch.int32), self.r), voxel_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
