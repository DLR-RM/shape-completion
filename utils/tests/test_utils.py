import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import torch_scatter
from PIL import Image

from ..src.utils import (
    adjust_intrinsic,
    convert_coordinates,
    coordinates_to_index,
    crop_and_resize_image,
    depth_to_points,
    get_git_root,
    git_submodule_path,
    make_3d_grid,
    points_to_coordinates,
    points_to_depth,
    points_to_uv,
    resolve_path,
)
from ..src.voxelizer import Voxelizer


def _to_tensor(array_or_tensor: Any) -> torch.Tensor:
    if torch.is_tensor(array_or_tensor):
        return array_or_tensor
    if isinstance(array_or_tensor, np.ndarray):
        return torch.from_numpy(array_or_tensor)
    raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(array_or_tensor)!r}")


def test_make_3d_grid():
    bounds = (-0.5, 0.5)
    padding = 0.1
    box_size = 1 + padding
    resolution = 32
    voxel_size = box_size / resolution
    size = box_size - voxel_size

    grid1 = size * make_3d_grid(bounds[0], bounds[1], resolution)
    grid2 = make_3d_grid(-size / 2, size / 2, resolution)

    assert torch.allclose(grid1, grid2)


def test_voxelizer():
    resolution = 32
    padding = 0
    points = (1 + padding) * np.random.rand(1000, 3) - (1 + padding) / 2
    points = np.concatenate(
        [points, (np.ones(3) * ((1 - padding) / 2)).reshape(1, 3), (np.ones(3) * (-(1 - padding) / 2)).reshape(1, 3)],
        axis=0,
    )

    voxelizer = Voxelizer(method="simple", padding=padding, resolution=resolution, round=False)
    _, simple_indices = voxelizer(points)

    voxelizer = Voxelizer(method="open3d", padding=padding, resolution=resolution, round=False)
    _, o3d_indices = voxelizer(points)

    voxelizer = Voxelizer(method="kdtree", padding=padding, resolution=resolution, round=False)
    _, kdtree_indices = voxelizer(points)

    assert len(np.unique(simple_indices)) == len(np.unique(o3d_indices)) == len(np.unique(kdtree_indices)), (
        f"{len(np.unique(simple_indices))} != {len(np.unique(o3d_indices))} != {len(np.unique(kdtree_indices))}"
    )
    for idx1, idx2, idx3 in zip(
        np.unique(simple_indices), np.unique(o3d_indices), np.unique(kdtree_indices), strict=False
    ):
        assert idx1 == idx2 == idx3, f"{idx1} != {idx2} != {idx3}"


def test_adjust_intrinsic():
    intrinsic = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]]).astype(np.float32)

    new_intrinsic = adjust_intrinsic(intrinsic, 640, 480)
    assert np.array_equal(new_intrinsic, intrinsic)

    new_intrinsic = adjust_intrinsic(intrinsic, 640, 480, size=640)
    assert np.array_equal(new_intrinsic, intrinsic)

    new_intrinsic = adjust_intrinsic(intrinsic, 640, 480, box=(0, 0, 640, 480))
    assert np.array_equal(new_intrinsic, intrinsic)

    new_intrinsic = adjust_intrinsic(intrinsic, 640, 480, box=(80, 0, 640 - 80, 480))
    assert new_intrinsic[0, 2] == new_intrinsic[1, 2]

    new_intrinsic = adjust_intrinsic(intrinsic, 640, 480, size=320)
    assert np.allclose(new_intrinsic[:2], intrinsic[:2] / 2)

    intrinsic = np.tile(intrinsic.reshape(1, 3, 3), (2, 1, 1))
    new_intrinsic = adjust_intrinsic(intrinsic, 640, 480, size=320)
    assert np.allclose(new_intrinsic[:, :2], intrinsic[:, :2] / 2)

    new_intrinsic = adjust_intrinsic(intrinsic, np.array([640, 640]), np.array([480, 480]), size=320)
    assert np.allclose(new_intrinsic[:, :2], intrinsic[:, :2] / 2)

    intrinsic[1, :2] = intrinsic[0, :2] / 2
    new_intrinsic = adjust_intrinsic(intrinsic, np.array([640, 320]), np.array([320, 160]), size=160)
    assert np.allclose(new_intrinsic[0, :2], intrinsic[0, :2] / 4)
    assert np.allclose(new_intrinsic[1, :2], intrinsic[1, :2] / 2)


def test_resolve_path():
    user_home = Path(os.path.expanduser("~"))
    path = "~/Documents"
    resolved_path = resolve_path(path)
    assert resolved_path == user_home / "Documents"


def test_get_git_path():
    assert get_git_root().is_dir()
    assert git_submodule_path("utils").is_dir()
    assert git_submodule_path(Path(__file__)).is_dir()
    assert git_submodule_path(Path(__file__).parent).is_dir()
    assert git_submodule_path("utils") == git_submodule_path(Path(__file__))


class TestConvertCoordinates:
    def test_conversion_opengl_to_opencv(self):
        points_opengl = np.array([[1, 2, 3]])
        expected_opencv = np.array([[1, -2, -3]])
        assert np.allclose(convert_coordinates(points_opengl, "opengl", "opencv"), expected_opencv)

    def test_conversion_opencv_to_opengl(self):
        points_opencv = np.array([[1, 2, 3]])
        points_opengl = convert_coordinates(points_opencv, "opencv", "opengl")
        expected_opengl = np.array([[1, -2, -3]])
        assert np.allclose(points_opengl, expected_opengl)

        points_opengl = points_opencv.copy()
        points_opengl[:, 1] *= -1
        points_opengl[:, 2] *= -1
        assert np.allclose(points_opengl, expected_opengl)

    def test_conversion_opengl_to_blender(self):
        points_opengl = np.array([[1, 2, 3]])
        expected_blender = np.array([[1, -3, 2]])
        assert np.allclose(convert_coordinates(points_opengl, "opengl", "blender"), expected_blender)

    def test_conversion_opencv_to_blender(self):
        points_opencv = np.array([[1, 2, 3]])
        expected_blender = np.array([[1, 3, -2]])
        assert np.allclose(convert_coordinates(points_opencv, "opencv", "blender"), expected_blender)

    def test_conversion_invalid_input_format(self):
        points = np.array([[1, 2, 3]])
        with pytest.raises(AssertionError):
            convert_coordinates(points, "invalid_format", "opengl")

    def test_conversion_invalid_output_format(self):
        points = np.array([[1, 2, 3]])
        with pytest.raises(AssertionError):
            convert_coordinates(points, "opengl", "invalid_format")

    def test_tensor_input(self):
        points_opengl = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        expected_opencv = torch.tensor([[1, -2, -3]], dtype=torch.float32)
        points_opencv = _to_tensor(convert_coordinates(points_opengl, "opengl", "opencv"))
        assert torch.allclose(points_opencv, expected_opencv)

    def test_invalid_point_dimensions(self):
        points = np.array([1, 2, 3])  # Not a 2D or 3D array
        with pytest.raises(AssertionError):
            convert_coordinates(points, "opengl", "opencv")


class TestCropAndResizeImage:
    @pytest.fixture
    def sample_image(self) -> Image.Image:
        return Image.new("RGB", (640, 480), color="red")

    def test_no_crop_no_resize(self, sample_image):
        result = crop_and_resize_image(sample_image)
        assert result.size == sample_image.size
        assert isinstance(result, type(sample_image))

        width, height = sample_image.size
        result = crop_and_resize_image(sample_image, box=(0, 0, width, height))
        assert result.size == sample_image.size

        result = crop_and_resize_image(sample_image, size=max(width, height))
        assert result.size == sample_image.size

    def test_only_crop(self, sample_image):
        _width, height = sample_image.size
        crop_coords = (80, 0, height + 80, height)
        result = crop_and_resize_image(sample_image, box=crop_coords)
        assert result.size == (height, height)

    def test_only_resize(self, sample_image):
        size = 50
        result = crop_and_resize_image(sample_image, size=size)
        width, height = cast(tuple[int, int], result.size)
        assert max(width, height) == size

    def test_crop_and_resize(self, sample_image):
        crop_coords = (10, 10, 80, 80)
        size = 40
        result = crop_and_resize_image(sample_image, box=crop_coords, size=size)
        width, height = cast(tuple[int, int], result.size)
        assert max(width, height) == size

    def test_out_of_bounds_crop_coords(self, sample_image):
        crop_coords = (-10, -10, 110, 110)
        result = crop_and_resize_image(sample_image, box=crop_coords)
        # Out-of-bounds boxes are clamped and padded instead of raising.
        assert result.size == (120, 120)

    def test_invalid_size(self, sample_image):
        size = -10
        with pytest.raises(ValueError):
            crop_and_resize_image(sample_image, size=size)


@pytest.fixture
def padding():
    return 0.1


@pytest.fixture
def points_2d():
    points_ll = torch.tensor([[-0.3, -0.2], [-0.2, -0.3]])
    points_lr = torch.tensor([[0.25, -0.25], [0.4, -0.1]])
    points_ul = torch.tensor([[-0.4, 0.4], [-0.25, 0.25], [-0.4, 0.1]])
    points_ur = torch.tensor([[0.25, 0.25]])

    points = torch.cat([points_ll, points_lr, points_ul, points_ur], dim=0).unsqueeze(0)
    return points


@pytest.fixture
def points_3d(padding):
    max_val = (1 + padding) / 2
    min_val = (-1 - padding) / 2
    points = torch.tensor(
        [[min_val, min_val, min_val], [-0.5, -0.5, -0.5], [0, 0, 0], [0.5, 0.5, 0.5], [max_val, max_val, max_val]]
    ).unsqueeze(0)
    return points


@pytest.fixture
def corner_points_2d():
    points = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]).unsqueeze(0)
    return points


@pytest.fixture
def corner_points_3d():
    points = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]
    ).unsqueeze(0)
    return points


class TestNormalizeCoordinate:
    def test_normalize_coordinate_2d(self, points_3d, padding):
        coordinates_xz = _to_tensor(points_to_coordinates(points_3d, max_value=1 + padding, plane="xz"))
        coordinates_xy = _to_tensor(points_to_coordinates(points_3d, max_value=1 + padding, plane="xy"))
        coordinates_yz = _to_tensor(points_to_coordinates(points_3d, max_value=1 + padding, plane="yz"))

        assert torch.equal(coordinates_xz, coordinates_xy)
        assert torch.equal(coordinates_xz, coordinates_yz)
        assert torch.allclose(
            coordinates_xz,
            torch.tensor(
                [[0, 0], [-0.5 / (1 + padding) + 0.5] * 2, [0.5, 0.5], [0.5 / (1 + padding) + 0.5] * 2, [1, 1]]
            ),
        )

    def test_normalize_coordinate_3d(self, points_3d, padding):
        coordinates = _to_tensor(points_to_coordinates(points_3d, max_value=1 + padding))

        assert torch.allclose(
            coordinates,
            torch.tensor(
                [
                    [0, 0, 0],
                    [-0.5 / (1 + padding) + 0.5] * 3,
                    [0.5, 0.5, 0.5],
                    [0.5 / (1 + padding) + 0.5] * 3,
                    [1, 1, 1],
                ]
            ),
        )


class TestCoordinate2Index:
    def test_coordinate2index_2d(self, corner_points_2d):
        normalized_corner_points_2d = _to_tensor(points_to_coordinates(corner_points_2d, plane="xy"))
        index = _to_tensor(coordinates_to_index(normalized_corner_points_2d, resolution=2))

        assert torch.equal(index.view(4), torch.arange(4))

    def test_coordinate2index_2d_real(self, points_2d):
        normalized_points_2d = _to_tensor(points_to_coordinates(points_2d, plane="xy"))
        index = _to_tensor(coordinates_to_index(normalized_points_2d, resolution=2))

        assert torch.equal(index, torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3]]))

    def test_coordinate2index_3d(self, corner_points_3d):
        normalized_test_points = _to_tensor(points_to_coordinates(corner_points_3d))
        index = _to_tensor(coordinates_to_index(normalized_test_points, resolution=2))

        assert torch.equal(index.view(8), torch.arange(8))


class TestScatter:
    def test_scatter_2d(self, corner_points_2d, show: bool = False):
        coordinates = _to_tensor(points_to_coordinates(corner_points_2d, plane="xy"))

        # Channels last
        index = _to_tensor(coordinates_to_index(coordinates, resolution=2)).unsqueeze(-1)
        feature = torch_scatter.scatter_max(corner_points_2d, index, dim=1, dim_size=4)[0]

        assert torch.equal(feature, corner_points_2d)

        # Channels first
        index = _to_tensor(coordinates_to_index(coordinates, resolution=2)).unsqueeze(1)
        feature = torch_scatter.scatter_max(corner_points_2d.transpose(1, 2), index, dim_size=4)[0]

        assert torch.equal(feature.transpose(1, 2), corner_points_2d)

        if show:
            import matplotlib.pyplot as plt

            plt.scatter(corner_points_2d[0, :, 0], corner_points_2d[0, :, 1], c="b")
            plt.scatter(feature[0, :, 0], feature[0, :, 1], c="r")
            plt.show()

    def test_scatter_2d_real(self, points_2d, show: bool = False):
        coordinates = _to_tensor(points_to_coordinates(points_2d, plane="xy"))
        index = _to_tensor(coordinates_to_index(coordinates, resolution=2)).unsqueeze(1)

        feature = torch_scatter.scatter_mean(points_2d.transpose(1, 2), index, dim_size=4)
        feature = feature.transpose(1, 2).squeeze(0)

        assert torch.allclose(feature[0], points_2d[:, :2, :].mean(dim=1))
        assert torch.allclose(feature[1], points_2d[:, 2:4, :].mean(dim=1))
        assert torch.allclose(feature[2], points_2d[:, 4:7, :].mean(dim=1))
        assert torch.allclose(feature[3], points_2d[:, 7, :])

        if show:
            import matplotlib.pyplot as plt

            plt.scatter(points_2d[0, :, 0], points_2d[0, :, 1], c="b")
            plt.scatter(feature[:, 0], feature[:, 1], c="r")
            plt.show()

    def test_scatter_3d(self, corner_points_3d, show: bool = False):
        coordinates = _to_tensor(points_to_coordinates(corner_points_3d))

        # Channels last
        index = _to_tensor(coordinates_to_index(coordinates, resolution=2)).unsqueeze(-1)
        feature = torch_scatter.scatter_max(corner_points_3d, index, dim=1, dim_size=8)[0]

        assert torch.equal(feature, corner_points_3d)

        # Channels first
        index = _to_tensor(coordinates_to_index(coordinates, resolution=2)).unsqueeze(1)
        feature = torch_scatter.scatter_max(corner_points_3d.transpose(1, 2), index, dim_size=8)[0]
        feature = feature.transpose(1, 2)

        assert torch.equal(feature, corner_points_3d)

        if show:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = cast(Any, fig.add_subplot(111, projection="3d"))
            points_np = corner_points_3d[0].cpu().numpy()
            feature_np = feature[0].cpu().numpy()
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c="b")
            ax.scatter(feature_np[:, 0], feature_np[:, 1], feature_np[:, 2], c="r")
            plt.show()

    def test_scatter_3d_real(self, points_3d, padding, show: bool = False):
        coordinates = _to_tensor(points_to_coordinates(points_3d, max_value=1 + padding))
        index = _to_tensor(coordinates_to_index(coordinates, resolution=2)).unsqueeze(1)

        feature = torch_scatter.scatter_mean(points_3d.transpose(1, 2), index, dim_size=8)
        feature = feature.transpose(1, 2).squeeze(0)

        assert torch.allclose(feature[0], points_3d[:, :2, :].mean(dim=1))
        assert all(torch.equal(f, torch.zeros(3)) for f in feature[1:-1])
        assert torch.allclose(feature[-1], points_3d[:, 2:, :].mean(dim=1))

        if show:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = cast(Any, fig.add_subplot(111, projection="3d"))
            points_np = points_3d[0].cpu().numpy()
            feature_np = feature.cpu().numpy()
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c="b")
            ax.scatter(feature_np[:, 0], feature_np[:, 1], feature_np[:, 2], c="r")
            plt.show()


class TestGridSample:
    def test_grid_sample_2d(self, corner_points_2d):
        coordinates = _to_tensor(points_to_coordinates(corner_points_2d, plane="xy"))
        feature = _to_tensor(coordinates_to_index(coordinates, resolution=2)).view(1, 1, 2, 2).float()
        grid = (2 * coordinates - 1).view(1, 4, 1, 2)
        feature_sampled = torch.nn.functional.grid_sample(feature, grid, align_corners=True)

        assert torch.equal(feature.view(4), feature_sampled.view(4))

        feature_sampled = torch.nn.functional.grid_sample(
            feature.permute(0, 1, 3, 2),
            (2 * torch.flip(coordinates, dims=[-1]) - 1).view(1, 4, 1, 2),
            align_corners=True,
        )

        assert torch.equal(feature.view(4), feature_sampled.view(4))

        feature_sampled = torch.nn.functional.grid_sample(feature, grid / 2, align_corners=False)

        assert torch.equal(feature.view(4), feature_sampled.view(4))

    def test_grid_sample_3d(self, corner_points_3d):
        coordinates = _to_tensor(points_to_coordinates(corner_points_3d))

        grid = (2 * coordinates - 1).view(1, 8, 1, 1, 3)
        feature = _to_tensor(coordinates_to_index(coordinates, resolution=2)).view(1, 1, 2, 2, 2).float()
        feature_sampled = torch.nn.functional.grid_sample(feature, grid, align_corners=True)

        assert torch.equal(feature.view(8), feature_sampled.view(8))

        feature_sampled = torch.nn.functional.grid_sample(feature, grid / 2, align_corners=False)

        assert torch.equal(feature.view(8), feature_sampled.view(8))


class TestScatterGridSample:
    def test_scatter_grid_sample_2d_real(self, points_2d, corner_points_2d):
        coordinates = _to_tensor(points_to_coordinates(points_2d, plane="xy"))
        index = _to_tensor(coordinates_to_index(coordinates, resolution=2)).unsqueeze(1)
        feature = torch_scatter.scatter_mean(points_2d.transpose(1, 2), index, dim_size=4).view(1, 2, 2, 2)

        coordinates = _to_tensor(points_to_coordinates(corner_points_2d, plane="xy"))
        grid = (2 * coordinates - 1).view(1, 4, 1, 2)

        feature_sampled = torch.nn.functional.grid_sample(feature, grid, align_corners=True).squeeze(-1)
        feature_sampled = feature_sampled.transpose(1, 2).squeeze(0)

        assert torch.allclose(feature_sampled[0], points_2d[:, :2, :].mean(dim=1))
        assert torch.allclose(feature_sampled[1], points_2d[:, 2:4, :].mean(dim=1))
        assert torch.allclose(feature_sampled[2], points_2d[:, 4:7, :].mean(dim=1))
        assert torch.allclose(feature_sampled[3], points_2d[:, 7, :].mean(dim=1))

        feature_sampled = torch.nn.functional.grid_sample(feature, grid / 2, align_corners=False).squeeze(-1)
        feature_sampled = feature_sampled.transpose(1, 2).squeeze(0)

        assert torch.allclose(feature_sampled[0], points_2d[:, :2, :].mean(dim=1))
        assert torch.allclose(feature_sampled[1], points_2d[:, 2:4, :].mean(dim=1))
        assert torch.allclose(feature_sampled[2], points_2d[:, 4:7, :].mean(dim=1))
        assert torch.allclose(feature_sampled[3], points_2d[:, 7, :].mean(dim=1))


class TestProjections:
    def test_depth_to_points(self):
        depth = np.array([[1, 2], [3, 4]], dtype=float)
        height, width = depth.shape
        intrinsic = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]])
        expected_points = np.array([[-0.5, -0.5, 1], [0, -1, 2], [-1.5, 0, 3], [0, 0, 4]])
        points = depth_to_points(depth, intrinsic, depth_scale=1.0)
        assert np.allclose(points, expected_points)

    def test_zero_depth(self):
        depth = np.zeros((2, 2), dtype=float)
        height, width = depth.shape
        intrinsic = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]])
        points = depth_to_points(depth, intrinsic, depth_scale=1.0)
        assert len(points) == 0  # Expecting no points for zero depth

    def test_points_to_uv(self):
        depth = np.array([[1, 2], [3, 4]])
        height, width = depth.shape
        intrinsic = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]])
        points = np.array([[-0.5, -0.5, 1], [0, -1, 2], [-1.5, 0, 3], [0, 0, 4]])
        u, v, _ = points_to_uv(points, intrinsic, width=width, height=height)

        _v, _u = np.nonzero(depth)
        assert np.all(u == _u)
        assert np.all(v == _v)

        index = np.vstack([u, v])
        assert points.shape[0] == index.shape[1]

    def test_points_to_depth(self):
        expected_depth = np.array([[1, 2], [3, 4]])
        height, width = expected_depth.shape
        intrinsic = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]])
        points = np.array([[-0.5, -0.5, 1], [0, -1, 2], [-1.5, 0, 3], [0, 0, 4]])
        depth = points_to_depth(points, intrinsic, width, height, depth_scale=1.0)
        np.testing.assert_array_equal(depth, expected_depth)

    def test_depth_to_points_to_depth(self):
        depth = np.array([[1, 2], [3, 4]], dtype=float)
        height, width = depth.shape
        intrinsic = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]])
        points = depth_to_points(depth, intrinsic, depth_scale=1.0)
        depth_reconstructed = points_to_depth(points, intrinsic, width, height, depth_scale=1.0)
        np.testing.assert_array_equal(depth, depth_reconstructed)

    def test_points_to_depth_to_points(self):
        points = np.array([[-0.5, -0.5, 1], [0, -1, 2], [-1.5, 0, 3], [0, 0, 4]])
        height, width = 2, 2
        intrinsic = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]])
        depth = points_to_depth(points, intrinsic, width, height, depth_scale=1.0)
        points_reconstructed = depth_to_points(depth, intrinsic, depth_scale=1.0)
        np.testing.assert_array_equal(points, points_reconstructed)
