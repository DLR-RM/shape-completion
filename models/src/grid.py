from abc import abstractmethod
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F
from pytorch3dunet.unet3d.model import UNet2D, UNet3D
from torch import Tensor, nn
from torch_scatter import scatter

from utils import adjust_intrinsic, coordinates_to_index, points_to_coordinates

from .utils import grid_sample_2d, grid_sample_3d, visualize_feature  # noqa: F401


def _as_tensor(value: Any, like: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _points_to_coordinates_tensor(points: Tensor, **kwargs: Any) -> Tensor:
    return cast(Tensor, points_to_coordinates(points, **kwargs))


def _coordinates_to_index_tensor(coordinates: Tensor, resolution: int) -> Tensor:
    return cast(Tensor, coordinates_to_index(coordinates, resolution))


class GridEncoder(nn.Module):
    def __init__(
        self,
        c_dim: int = 32,
        unet: bool = False,
        unet_kwargs: dict[str, Any] | None = None,
        unet3d: bool = False,
        unet3d_kwargs: dict[str, Any] | None = None,
        plane_resolution: int | None = None,
        grid_resolution: int | None = None,
        feature_type: tuple[str, ...] = ("grid",),
        scatter_type: str = "mean",
        padding: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if "grid" in feature_type and grid_resolution is None:
            raise ValueError("Grid resolution must be specified for grid feature type")
        elif ("uv" in feature_type or any(t in feature_type for t in ["x", "y", "z"])) and plane_resolution is None:
            raise ValueError("Plane resolution must be specified for plane feature types")

        self.plane_resolution = cast(int, plane_resolution)
        self.grid_resolution = cast(int, grid_resolution)
        self.feature_type = feature_type
        self.scatter_type = scatter_type
        self.padding = padding
        self.c_dim = c_dim

        self.unet: nn.Module | None = None
        self.unet3d: nn.Module | None = None

        if unet:
            self.unet = UNet2D(in_channels=c_dim, out_channels=c_dim, **(unet_kwargs or {}))
        if unet3d:
            self.unet3d = UNet3D(in_channels=c_dim, out_channels=c_dim, **(unet3d_kwargs or {}))

    def get_index_dict(
        self,
        points: Tensor,
        intrinsic: Tensor | None = None,
        extrinsic: Tensor | None = None,
        max_value: int | Tensor | None = None,
    ) -> dict[str, Tensor]:
        index_dict = dict()
        for feature_type in self.feature_type:
            if "grid" in feature_type:
                coordinates = _points_to_coordinates_tensor(points, max_value=1 + self.padding)
                index_dict[feature_type] = _coordinates_to_index_tensor(coordinates, self.grid_resolution).unsqueeze(1)
            else:
                p = points
                if extrinsic is not None:
                    extrinsic_t = _as_tensor(extrinsic, points)
                    rot = extrinsic_t[..., :3, :3]
                    trans = extrinsic_t[..., :3, 3:4]
                    if "uv" in feature_type:
                        p = torch.baddbmm(trans, rot, points.transpose(-1, -2)).transpose(-1, -2)
                    elif "xy" in feature_type:
                        p = torch.bmm(rot, points.transpose(-1, -2)).transpose(-1, -2)

                if "uv" in feature_type and intrinsic is not None:
                    uv_max_value = self.plane_resolution - 1 if max_value is None else max_value
                    coordinates = _points_to_coordinates_tensor(
                        p,
                        max_value=uv_max_value,
                        plane="uv",
                        intrinsic=_as_tensor(intrinsic, points),
                    )
                else:
                    coordinates = _points_to_coordinates_tensor(p, max_value=1 + self.padding, plane=feature_type[:2])
                index_dict[feature_type] = _coordinates_to_index_tensor(coordinates, self.plane_resolution).unsqueeze(1)
        return index_dict

    def generate_plane_feature(self, feature: Tensor, index: Tensor) -> Tensor | list[Tensor]:
        plane_feature = feature
        if plane_feature.dim() == 3:  # [B, C, N] -> [B, C, H, W]
            plane_feature = scatter(feature, index, dim_size=self.plane_resolution**2, reduce=self.scatter_type)
            plane_feature = plane_feature.view(
                feature.size(0), feature.size(1), self.plane_resolution, self.plane_resolution
            )

        if self.unet is not None:
            plane_feature = self.unet(plane_feature)

        return plane_feature

    def generate_grid_feature(self, feature: Tensor, index: Tensor) -> Tensor | list[Tensor]:
        grid_feature = feature
        if grid_feature.dim() == 3:  # [B, C, N] -> [B, C, D, H, W]
            grid_feature = scatter(feature, index, dim_size=self.grid_resolution**3, reduce=self.scatter_type)
            grid_feature = grid_feature.view(
                feature.size(0), feature.size(1), self.grid_resolution, self.grid_resolution, self.grid_resolution
            )

        if self.unet3d is not None:
            grid_feature = self.unet3d(grid_feature)

        return grid_feature

    def generate_feature(self, feature: Tensor, index_dict: dict[str, Tensor]) -> dict[str, Tensor | list[Tensor]]:
        feature_dict = dict()
        for feature_type, index in index_dict.items():
            if "grid" in feature_type:
                grid_feature = self.generate_grid_feature(feature, index)
                if isinstance(grid_feature, tuple) and isinstance(grid_feature[1], list):  # Final + intermediate
                    grid_feature = [grid_feature[0], *grid_feature[1]]
                feature_dict[feature_type] = grid_feature
            else:
                plane_feature = self.generate_plane_feature(feature, index)
                if isinstance(plane_feature, tuple) and isinstance(plane_feature[1], list):  # Final + intermediate
                    plane_feature = [plane_feature[0], *plane_feature[1]]
                feature_dict[feature_type] = plane_feature
        return feature_dict

    def pool_local(self, feature: Tensor, index_dict: dict[str, Tensor], scatter_type: str = "max") -> Tensor:
        c_out = torch.zeros_like(feature)  # [B, C, N]
        for feature_type, index in index_dict.items():
            # Put max of features with cell index i into cell at index i (local max pool)
            # i.e. c_out[:, :, i] = max(feature[:, :, j]) for all j with index[j] == i
            # e.g. c_out[0, :, 0] = torch.max(feature[0, :, index[0, 0] == index[0, 0, 0]], dim=1)[0]
            if "grid" in feature_type:
                feat = scatter(
                    feature, index, dim_size=self.grid_resolution**3, reduce=scatter_type
                )  # [B, C, grid_res ** 3]
            else:
                feat = scatter(
                    feature, index, dim_size=self.plane_resolution**2, reduce=scatter_type
                )  # [B, C, grid_res ** 2]
            # Extract per-cell max-pooled features from grid according to index
            c_out += feat.gather(dim=2, index=index.expand_as(feature))  # [B, C, N]
        return c_out

    @abstractmethod
    def forward(self, points: Tensor, **kwargs) -> dict[str, Tensor]:
        raise NotImplementedError("GridEncoder is an abstract class")


class GridDecoder(nn.Module):
    def __init__(
        self,
        c_dim: int,
        padding: float = 0.1,
        sample_mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        grid_sample: Callable[..., Tensor] = F.grid_sample,
        **kwargs,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.padding = padding
        self.sample_mode = sample_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.grid_sample = grid_sample

    @staticmethod
    def get_plane_coordinates(
        points: Tensor,
        max_value: int | float | Tensor,
        plane: str,
        intrinsic: Any | None = None,
        extrinsic: Any | None = None,
    ) -> Tensor:
        if extrinsic is not None:
            extrinsic_t = _as_tensor(extrinsic, points)
            rot = extrinsic_t[..., :3, :3]
            trans = extrinsic_t[..., :3, 3:4]
            if plane == "uv":
                points = torch.baddbmm(trans, rot, points.transpose(-1, -2)).transpose(-1, -2)
            elif plane == "xy":
                points = torch.bmm(rot, points.transpose(-1, -2)).transpose(-1, -2)

        if plane == "uv" and intrinsic is not None:
            coordinates = _points_to_coordinates_tensor(
                points, max_value=max_value, plane="uv", intrinsic=_as_tensor(intrinsic, points)
            )
        else:
            coordinates = _points_to_coordinates_tensor(points, max_value=max_value, plane=plane)
        return coordinates

    def sample_plane_feature(self, feature: Tensor, coordinates: Tensor) -> Tensor:
        grid = 2 * coordinates - 1  # grid sample expects range (-1, 1)
        grid = grid[:, :, None]  # (B, N, 1, 2)
        sampled = cast(
            Tensor,
            self.grid_sample(
                feature,  # (B, C, H, W)
                grid=grid,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                mode=self.sample_mode,
            ),
        )
        return sampled.squeeze(-1)  # (B, C, N)

    def sample_grid_feature(self, feature: Tensor, coordinates: Tensor) -> Tensor:
        grid = 2 * coordinates - 1  # grid sample expects range of -1/1
        grid = grid[:, :, None, None]  # (B, N, 1, 1, 3)
        sampled = cast(
            Tensor,
            self.grid_sample(
                feature,  # (B, C, D, H, W)
                grid=grid,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
                mode=self.sample_mode,
            ),
        )
        return sampled.squeeze(-1).squeeze(-1)  # (B, C, N)

    def sample_feature(
        self, points: Tensor, feature_dict: dict[str, Tensor | list[Tensor]], **kwargs
    ) -> tuple[Tensor, Tensor]:
        feature = points.new_zeros((points.size(0), self.c_dim, points.size(1)))
        feature_list: list[Tensor] = []
        coordinates = _points_to_coordinates_tensor(points, max_value=1 + self.padding)
        for key, value in feature_dict.items():
            values = list(value) if isinstance(value, (tuple, list)) else [value]
            # Check if feature need to be concatenated
            cat = "cat" in key or (len(values) > 1 and len({v.size(1) for v in values}) != 1)
            if "grid" in key:
                coordinates = _points_to_coordinates_tensor(points, max_value=1 + self.padding)
                if cat:
                    for val in values:
                        feature_list.append(self.sample_grid_feature(feature=val, coordinates=coordinates))
                else:
                    for val in values:
                        feature += self.sample_grid_feature(feature=val, coordinates=coordinates)
            else:
                intrinsic: Tensor | None = None
                resize = kwargs.get("resize_intrinsic", False)
                if "uv" in key:
                    width = cast(Tensor, kwargs["inputs.width"])
                    height = cast(Tensor, kwargs["inputs.height"])
                    intrinsic = _as_tensor(kwargs["inputs.intrinsic"], points)
                    width_eq_height = torch.equal(width, height)
                    all_eq_size = width.unique().numel() == 1 and height.unique().numel() == 1
                    if resize and not (all_eq_size or width_eq_height):
                        raise ValueError("All inputs must have the same size OR be square")
                    else:
                        if all_eq_size:
                            max_value = max(width[0].item(), height[0].item()) - 1
                        elif width_eq_height:
                            max_value = height - 1
                        else:
                            raise ValueError("All inputs must have the same size OR be square")
                        coordinates = self.get_plane_coordinates(
                            points,
                            max_value=max_value,
                            plane=key[:2],
                            intrinsic=intrinsic,
                            extrinsic=kwargs.get("inputs.extrinsic"),
                        )
                else:
                    coordinates = self.get_plane_coordinates(
                        points, max_value=1 + self.padding, plane=key[:2], extrinsic=kwargs.get("inputs.extrinsic")
                    )

                if cat:
                    for val in values:
                        if "uv" in key and resize:
                            _b, _c, h, w = val.size()
                            assert h == w, "Intrinsic resize for non-square feature maps not supported yet"
                            intrinsic = _as_tensor(
                                adjust_intrinsic(kwargs["inputs.intrinsic"], width, height, size=h), points
                            )
                            max_value = h - 1
                            coordinates = self.get_plane_coordinates(
                                points,
                                max_value=max_value,
                                plane=key[:2],
                                intrinsic=intrinsic,
                                extrinsic=kwargs.get("inputs.extrinsic"),
                            )

                        feat = self.sample_plane_feature(feature=val, coordinates=coordinates)

                        if kwargs.get("show", False):
                            for c, v, f in zip(coordinates, val, feat, strict=False):
                                _, h, w = v.size()
                                _u = (c[:, 0] * w).long().clamp(0, w - 1)
                                _v = (c[:, 1] * h).long().clamp(0, h - 1)
                                image = torch.zeros_like(v)
                                image[:, _v, _u] = f
                                visualize_feature(
                                    feature=image,
                                    name=f"decoder {key} ({v.shape})",
                                    batched=False,
                                    padding=self.padding,
                                )

                        if feat.size(1) != 3:
                            feature_list.append(feat)
                else:
                    for val in values:
                        if "uv" in key and resize:
                            _b, _c, h, w = val.size()
                            assert h == w, "Intrinsic resize for non-square feature maps not supported yet"
                            if intrinsic is None:
                                raise ValueError("Intrinsic is required for uv resize")
                            intrinsic = _as_tensor(adjust_intrinsic(intrinsic, width, height, size=h), points)
                            max_value = h - 1
                            coordinates = self.get_plane_coordinates(
                                points,
                                max_value=max_value,
                                plane=key[:2],
                                intrinsic=intrinsic,
                                extrinsic=kwargs.get("inputs.extrinsic"),
                            )

                        feat = self.sample_plane_feature(feature=val, coordinates=coordinates)

                        if kwargs.get("show", False):
                            for c, v, f in zip(coordinates, val, feat, strict=False):
                                _, h, w = v.size()
                                _u = (c[:, 0] * w).long().clamp(0, w - 1)
                                _v = (c[:, 1] * h).long().clamp(0, h - 1)
                                image = torch.zeros_like(v)
                                image[:, _v, _u] = f
                                visualize_feature(
                                    feature=image,
                                    name=f"decoder {key} ({v.shape})",
                                    batched=False,
                                    padding=self.padding,
                                )

                        feature += feat
        if feature_list:
            return torch.cat(feature_list, dim=1), coordinates
        return feature, coordinates
