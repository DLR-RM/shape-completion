from typing import Tuple, Union
from torch import nn, Tensor

from .functional.devoxelization import trilinear_devoxelize
from .voxelization import Voxelization
from .shared_mlp import SharedMLP
from .se import SE3d


class PVConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 resolution: int,
                 with_se: bool = False,
                 normalize: bool = False,
                 eps: float = 0.0,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, features: Union[Tensor, Tuple[Tensor, Tensor]], inputs: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(features, Tensor):
            features, _ = features
        voxel_features, voxel_coords = self.voxelization(features, inputs)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features_samples = trilinear_devoxelize(voxel_features,
                                                      voxel_coords,
                                                      self.resolution,
                                                      self.training)
        points_features = self.point_features(features)
        fused_features = voxel_features_samples + points_features
        return fused_features, voxel_features
