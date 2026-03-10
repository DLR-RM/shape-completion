from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from .functional import ball_query, grouping

__all__ = ["BallQuery"]


class BallQuery(nn.Module):
    def __init__(self, radius, num_neighbors, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

    def forward(
        self,
        points_coords: Tensor,
        centers_coords: Tensor,
        temb: Tensor,
        points_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        points_coords = points_coords.contiguous()
        centers_coords = centers_coords.contiguous()
        neighbor_indices = ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = cast(Tensor, grouping(points_coords, neighbor_indices))
        neighbor_coordinates = neighbor_coordinates - centers_coords.unsqueeze(-1)

        if points_features is None:
            assert self.include_coordinates, "No Features For Grouping"
            neighbor_features = neighbor_coordinates
        else:
            neighbor_features = cast(Tensor, grouping(points_features, neighbor_indices))
            if self.include_coordinates:
                neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
        temb_grouped = cast(Tensor, grouping(temb, neighbor_indices))
        return neighbor_features, temb_grouped

    def extra_repr(self):
        return "radius={}, num_neighbors={}{}".format(
            self.radius, self.num_neighbors, ", include coordinates" if self.include_coordinates else ""
        )
