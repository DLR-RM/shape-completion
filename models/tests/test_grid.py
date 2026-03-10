import copy
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pytest
import torch
from PIL import Image
from torch_scatter import scatter_mean
from torchvision.transforms import functional as F

from utils import (
    adjust_intrinsic,
    depth_to_points,
    inv_trafo,
    invert_intrinsic,
)
from utils import (
    coordinates_to_index as _coordinates_to_index,
)
from utils import (
    points_to_coordinates as _points_to_coordinates,
)
from utils import (
    points_to_uv as _points_to_uv,
)
from utils.src.voxelizer import Voxelizer

from ..src.grid import GridDecoder, GridEncoder

coordinates_to_index_any = cast(Any, _coordinates_to_index)
points_to_coordinates_any = cast(Any, _points_to_coordinates)
points_to_uv_any = cast(Any, _points_to_uv)
GridEncoderAny = cast(Any, GridEncoder)
GridDecoderAny = cast(Any, GridDecoder)


def _o3d_visualization() -> Any:
    visualization = getattr(o3d, "visualization", None)
    if visualization is None:
        pytest.skip("open3d visualization backend unavailable")
    return cast(Any, visualization)


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Missing test asset: {path}")


def _sample_plane(
    decoder: Any,
    points: torch.Tensor,
    feature: torch.Tensor,
    plane: str,
    max_value: int | float | torch.Tensor | None = None,
    intrinsic: torch.Tensor | None = None,
    extrinsic: torch.Tensor | None = None,
) -> torch.Tensor:
    max_value = (1 + decoder.padding) if max_value is None else max_value
    coordinates = decoder.get_plane_coordinates(
        points=points, max_value=max_value, plane=plane, intrinsic=intrinsic, extrinsic=extrinsic
    )
    return decoder.sample_plane_feature(feature=feature, coordinates=coordinates)


@pytest.fixture(scope="module")
def show() -> bool:
    return False


@pytest.fixture(scope="module")
def use_points() -> bool:
    return False


@pytest.fixture(scope="module")
def project() -> bool:
    return False


def test_scatter(shapenet_data_root: Path, show: bool):
    from dataset import PointCloudField

    if not shapenet_data_root.is_dir():
        pytest.skip("'data_root' is not a directory")
    dataset = "occupancy_networks"
    synthset = "02958343"
    obj_id = "1a1de15e572e039df085b75b20c2db33"
    _skip_if_missing(shapenet_data_root / dataset / synthset / obj_id / "pointcloud.npz")

    pcd_field = PointCloudField()
    pcd_data = pcd_field.load(shapenet_data_root / dataset / synthset / obj_id, index=0)

    padding = 0.1
    res = 128

    points = pcd_data[None]
    voxelizer = Voxelizer(resolution=res, padding=padding)
    grid_voxelized, _grid_index = voxelizer(points)

    x, y, z = np.where(grid_voxelized == 1)
    points_voxelized = np.stack((x, y, z), axis=-1)

    points_torch = torch.from_numpy(points).unsqueeze(0).float()

    coords = points_to_coordinates_any(points_torch, max_value=1 + padding)
    index = coordinates_to_index_any(coords, res).unsqueeze(1)
    grid_scattered = scatter_mean(src=points_torch.transpose(1, 2), index=index, dim_size=res**3)
    grid_scattered = grid_scattered.view(1, 3, res, res, res)

    x, y, z = np.where(np.any(grid_scattered.squeeze(0).numpy(), axis=0))
    points_scattered = np.stack((x, y, z), axis=-1)
    points_scattered = np.flip(points_scattered, axis=-1)  # zyx -> xyz

    if not sorted(points_voxelized.tolist()) == sorted(points_scattered.tolist()):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).paint_uniform_color([0, 0, 1])
        pcd_voxelized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_voxelized)).paint_uniform_color(
            [1, 0, 0]
        )
        pcd_scattered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_scattered)).paint_uniform_color(
            [0, 1, 0]
        )

        max_dist = np.asarray(pcd_voxelized.compute_point_cloud_distance(pcd_scattered)).max()
        print(max_dist)

        if show:
            _o3d_visualization().draw_geometries(
                [pcd_voxelized, pcd_scattered, pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]
            )

    grid_voxelized = torch.from_numpy(grid_voxelized[None, ...]).unsqueeze(0).float()
    coords = points_to_coordinates_any(points_torch, max_value=1 + padding)
    coords = torch.flip(coords, dims=[-1])  # xyz -> zyx
    grid = 2 * coords - 1
    sample_value = torch.nn.functional.grid_sample(
        input=grid_voxelized, grid=grid[:, :, None, None], mode="nearest", padding_mode="zeros", align_corners=True
    )
    sample_value = sample_value.view(1, 1, -1).squeeze().numpy()
    samples = points[sample_value > 0]
    pcd_voxelized_sampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(samples)).paint_uniform_color([1, 0, 0])
    assert len(samples) > 0

    coords = points_to_coordinates_any(points_torch, max_value=1 + padding)
    grid = 2 * coords - 1
    samples = torch.nn.functional.grid_sample(
        input=grid_scattered, grid=grid[:, :, None, None], mode="nearest", padding_mode="zeros", align_corners=True
    )
    samples = samples.view(1, 3, -1).transpose(1, 2).squeeze(0).numpy()
    pcd_scattered_sampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(samples)).paint_uniform_color([0, 1, 0])

    max_dist = np.asarray(pcd_voxelized_sampled.compute_point_cloud_distance(pcd_scattered_sampled)).max()
    if max_dist > 0.1:
        print(max_dist)
        if show:
            _o3d_visualization().draw_geometries(
                [
                    pcd_scattered_sampled,
                    pcd_voxelized_sampled,
                    pcd,
                    o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
                ]
            )
    assert max_dist <= 0.1


def test_project():
    w = 137
    fx = fy = w
    cx = cy = w / 2
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    inv_intrinsic = np.array([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1]], dtype=np.float32)

    assert np.allclose(inv_intrinsic, np.linalg.inv(intrinsic))

    points_uv = np.array([[0, 0, 1], [w, w, 1]])

    points_cam = (inv_intrinsic @ points_uv.T).T
    assert points_cam[:, :2].min() == -0.5 and np.isclose(points_cam[:, :2].max(), 0.5)


def test_grid_encoder(shapenet_data_root: Path, show: bool):
    from dataset import DepthField

    if not shapenet_data_root.is_dir():
        pytest.skip("'data_root' is not a directory")
    dataset = "ShapeNetCore.v1.fused.simple"
    synthset = "02958343"
    obj_id = "1a1de15e572e039df085b75b20c2db33"
    _skip_if_missing(shapenet_data_root / dataset / synthset / obj_id)

    depth_field = DepthField()
    depth_data = depth_field.load(shapenet_data_root / dataset / synthset / obj_id, index=0)
    depth = depth_data[None]
    normals = depth_data["normals"]
    intrinsic = depth_data["intrinsic"]
    extrinsic = depth_data["extrinsic"]
    width, height = depth_data["width"], depth_data["height"]

    inputs_world = depth
    normals = ((normals / 2 + 0.5) * 255).astype(np.uint8)
    plane_resolution = height // 2

    inputs_world = torch.from_numpy(inputs_world).unsqueeze(0).float()
    feature = torch.from_numpy(normals).unsqueeze(0).transpose(1, 2).float()
    intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float()
    extrinsic = torch.from_numpy(extrinsic).unsqueeze(0).float()

    feature_type = ("uv", "xy", "yz", "xz")
    _, C, _ = feature.size()
    encoder = GridEncoderAny(c_dim=C, plane_resolution=plane_resolution, feature_type=feature_type, scatter_type="mean")

    adjusted_intrinsic = adjust_intrinsic(
        intrinsic.squeeze(0), width, height, box=(80, 0, width - 80, height), size=plane_resolution
    )
    if isinstance(adjusted_intrinsic, np.ndarray):
        adjusted_intrinsic = torch.from_numpy(adjusted_intrinsic)
    adjusted_intrinsic = adjusted_intrinsic.unsqueeze(0)
    index_dict1 = encoder.get_index_dict(inputs_world, adjusted_intrinsic, extrinsic)

    adjusted_intrinsic = adjust_intrinsic(intrinsic.squeeze(0), width, height, box=(80, 0, width - 80, height))
    if isinstance(adjusted_intrinsic, np.ndarray):
        adjusted_intrinsic = torch.from_numpy(adjusted_intrinsic)
    adjusted_intrinsic = adjusted_intrinsic.unsqueeze(0)
    index_dict2 = encoder.get_index_dict(inputs_world, adjusted_intrinsic, extrinsic, max_value=height - 1)

    u, v, _ = points_to_uv_any(inputs_world, adjusted_intrinsic, extrinsic)
    image_uv = np.zeros((height, height, C), dtype=np.uint8)
    image_uv[v, u] = normals
    image_uv = np.asarray(Image.fromarray(image_uv).resize((plane_resolution, plane_resolution)))

    if show:
        plt.imshow(image_uv)
        plt.title("normals image")
        plt.show()

    if feature_type == "uv" and plane_resolution == height:
        assert torch.equal(index_dict1["uv"], index_dict2["uv"])

    cmap = plt.get_cmap("bwr")
    image1: np.ndarray | None = None
    image2: np.ndarray | None = None
    for key, value in index_dict1.items():
        plane = encoder.generate_plane_feature(feature, value)

        if show:
            image1_arr = cast(np.ndarray, plane.squeeze(0).numpy().transpose(1, 2, 0).astype(np.uint8))
            image1 = image1_arr
            plt.imshow(image1_arr)
            plt.title(f"{key} (w/ resize)")
            plt.show()

            if key == "uv":
                assert image1 is not None
                error_image = (image_uv.astype(np.float32) - image1.astype(np.float32)).sum(axis=2)
                error_image /= np.abs(error_image).max()
                error_image = (error_image + 1) / 2
                plt.imshow(cmap(error_image))
                plt.title(f"difference {key} (w/ resize): {np.linalg.norm(image_uv - image1)}")
                plt.show()

    for key, value in index_dict2.items():
        plane = encoder.generate_plane_feature(feature, value)

        if show:
            image2_arr = cast(np.ndarray, plane.squeeze(0).numpy().transpose(1, 2, 0).astype(np.uint8))
            image2 = image2_arr
            plt.imshow(image2_arr)
            plt.title(f"{key} (w/o resize)")
            plt.show()

            if key == "uv":
                assert image2 is not None
                error_image = (image_uv.astype(np.float32) - image2.astype(np.float32)).sum(axis=2)
                error_image /= np.abs(error_image).max()
                error_image = (error_image + 1) / 2
                plt.imshow(cmap(error_image))
                plt.title(f"difference {key} (w/o resize): {np.linalg.norm(image_uv - image2)}")
                plt.show()

    if show and "uv" in feature_type and image1 is not None and image2 is not None:
        error_image = (image1.astype(np.float32) - image2.astype(np.float32)).sum(axis=2)
        error_image /= np.abs(error_image).max()
        error_image = (error_image + 1) / 2
        plt.imshow(cmap(error_image))
        plt.title(f"difference (w/ vs. w/o resize): {np.linalg.norm(image1 - image2)}")
        plt.show()


def test_grid_decoder(shapenet_data_root: Path, show: bool):
    from dataset import DepthField

    if not shapenet_data_root.is_dir():
        pytest.skip("'data_root' is not a directory")
    dataset = "ShapeNetCore.v1.fused.simple"
    synthset = "02958343"
    obj_id = "1a1de15e572e039df085b75b20c2db33"
    _skip_if_missing(shapenet_data_root / dataset / synthset / obj_id)

    depth_field = DepthField(crop=True)
    depth_data = depth_field.load(shapenet_data_root / dataset / synthset / obj_id, index=0)
    depth = depth_data[None]
    normals = depth_data["normals"]
    intrinsic = depth_data["intrinsic"]
    extrinsic = depth_data["extrinsic"]
    width, height = depth_data["width"], depth_data["height"]

    inputs_world = depth
    normals = ((normals / 2 + 0.5) * 255).astype(np.uint8)
    size = height // 2

    u, v, _ = points_to_uv_any(inputs_world, intrinsic, extrinsic)
    normals_uv = np.zeros((height, height, 3), dtype=np.uint8)
    normals_uv[v, u] = normals
    normals_uv = np.array(Image.fromarray(normals_uv).resize((size, size)), copy=True)

    if show:
        plt.imshow(normals_uv)
        plt.show()

    inputs = torch.from_numpy(inputs_world).unsqueeze(0).float()
    feature = torch.from_numpy(normals_uv.transpose(2, 0, 1)).unsqueeze(0).float()
    intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float()
    extrinsic = torch.from_numpy(extrinsic).unsqueeze(0).float()

    decoder = GridDecoderAny(c_dim=3, sample_mode="nearest", align_corners=True)

    feature_dict = {"uv": feature}
    kwargs = {
        "inputs.width": width * torch.ones(1),
        "inputs.height": height * torch.ones(1),
        "inputs.intrinsic": intrinsic,
        "inputs.extrinsic": extrinsic,
        "show": False,
    }
    plane, coordinates = decoder.sample_feature(inputs, feature_dict, resize_intrinsic=True, **kwargs)
    if "uv" in feature_dict and size == height:
        assert np.array_equal(plane.transpose(1, 2).squeeze(0).numpy().astype(np.uint8), normals_uv[v, u])

    if show:
        uv = (coordinates * size).long().clamp(0, size - 1).numpy()
        u = uv[0, :, 0]
        v = uv[0, :, 1]
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[v, u] = plane.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

        plt.imshow(image)
        plt.title("sampled")
        plt.show()

        cmap = plt.get_cmap("bwr")
        error_image = (normals_uv.astype(np.float32) - image.astype(np.float32)).sum(axis=2)
        error_image /= np.abs(error_image).max()
        error_image = (error_image + 1) / 2
        plt.imshow(cmap(error_image))
        plt.title(f"difference: {np.linalg.norm(normals_uv - image)}")
        plt.show()


def test_sample_from_image(shapenet_data_root: Path, show: bool):
    from dataset import ImageField

    if not shapenet_data_root.is_dir():
        pytest.skip("'data_root' is not a directory")
    dataset = "occupancy_networks"
    synthset = "02958343"
    obj_id = "1a1de15e572e039df085b75b20c2db33"
    _skip_if_missing(shapenet_data_root / dataset / synthset / obj_id / "img_choy2016" / "cameras.npz")

    image_field = ImageField(index=(0, 24))
    image_data = image_field.load(shapenet_data_root / dataset / synthset / obj_id, index=0)
    image = np.asarray(image_data[None], dtype=np.uint8)

    if show:
        plt.imshow(image)
        plt.title("image")
        plt.show()

    intrinsic = image_data["intrinsic"]
    extrinsic = image_data["extrinsic"]
    inv_intrinsic = invert_intrinsic(intrinsic)
    inv_extrinsic = inv_trafo(extrinsic)

    intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float()
    extrinsic = torch.from_numpy(extrinsic).unsqueeze(0).float()
    inv_intrinsic = torch.from_numpy(inv_intrinsic).unsqueeze(0).float()
    inv_extrinsic = torch.from_numpy(inv_extrinsic).unsqueeze(0).float()
    rot = intrinsic @ extrinsic[:, :3, :3]
    trans = intrinsic @ extrinsic[:, :3, 3:4]

    feature = F.pil_to_tensor(image_data[None]).unsqueeze(0).float()
    _, C, H, W = feature.shape

    v, u = np.mgrid[:H, :W]
    u = u.ravel()
    v = v.ravel()
    z = np.ones_like(u).ravel()

    points_pixel = torch.from_numpy(np.column_stack([u, v, z])).unsqueeze(0).float()
    points_cam = torch.bmm(points_pixel, inv_intrinsic.transpose(1, 2))
    points_world = points_cam @ inv_extrinsic[:, :3, :3].transpose(1, 2) + inv_extrinsic[:, :3, 3]

    pp2 = torch.baddbmm(trans, rot, points_world.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(points_pixel, pp2, atol=1e-4)

    u1, v1, _ = points_to_uv_any(points_cam, intrinsic, width=W, height=H)
    torch.equal(u1, points_pixel[:, :, 0])
    torch.equal(v1, points_pixel[:, :, 1])

    u2, v2, _ = points_to_uv_any(points_world, intrinsic, extrinsic, width=W, height=H)
    torch.equal(u2, points_pixel[:, :, 0])
    torch.equal(v2, points_pixel[:, :, 1])

    points_norm = points_pixel / (W - 1) - 0.5
    coords1 = points_to_coordinates_any(points_norm, plane="xy")
    coords2 = points_to_coordinates_any(points_cam, max_value=W - 1, plane="uv", intrinsic=intrinsic)
    assert torch.allclose(coords1, coords2, atol=1e-5), torch.norm(coords1 - coords2)

    encoder = GridEncoderAny(c_dim=C, plane_resolution=W, feature_type=("xy",), padding=0)
    scattered_feature = encoder.generate_plane_feature(
        feature.view(1, 3, -1), encoder.get_index_dict(points_norm)["xy"]
    )
    scattered_image = scattered_feature.squeeze(0).numpy().transpose(1, 2, 0).astype(np.uint8)

    if show:
        plt.imshow(scattered_image)
        plt.title("scattered")
        plt.show()

    assert np.array_equal(image, scattered_image), np.linalg.norm(image - scattered_image)

    decoder = GridDecoderAny(c_dim=C, padding=0, sample_mode="nearest", align_corners=True)
    plane = _sample_plane(decoder, points_norm, feature, plane="xy")

    image_xy = plane.transpose(1, 2).squeeze(0).numpy().astype(np.uint8).reshape(H, W, C)

    if show:
        plt.imshow(image_xy)
        plt.title("sampled")
        plt.show()

    assert np.array_equal(image, image_xy), np.linalg.norm(image - image_xy)

    plane_cam = _sample_plane(decoder, points_cam, feature, plane="uv", max_value=W - 1, intrinsic=intrinsic)

    image_cam = plane_cam.transpose(1, 2).squeeze(0).numpy().astype(np.uint8).reshape(H, W, C)
    assert np.array_equal(image, image_cam), np.linalg.norm(image - image_cam)

    plane_world = _sample_plane(
        decoder, points_world, feature, plane="uv", max_value=W - 1, intrinsic=intrinsic, extrinsic=extrinsic
    )

    image_world = plane_world.transpose(1, 2).squeeze(0).numpy().astype(np.uint8).reshape(H, W, C)
    assert np.array_equal(image, image_world), np.linalg.norm(image - image_world)


def test_random_sample_from_image(shapenet_data_root: Path, show: bool):
    from dataset import ImageField

    if not shapenet_data_root.is_dir():
        pytest.skip("'data_root' is not a directory")
    dataset = "occupancy_networks"
    synthset = "02958343"
    obj_id = "1a1de15e572e039df085b75b20c2db33"
    _skip_if_missing(shapenet_data_root / dataset / synthset / obj_id / "img_choy2016" / "cameras.npz")

    image_field = ImageField(index=(0, 24))
    image_data = image_field.load(shapenet_data_root / dataset / synthset / obj_id, index=0)
    pil_image = image_data[None]
    if show:
        plt.imshow(pil_image)
        plt.show()

    intrinsic = image_data["intrinsic"]
    extrinsic = image_data["extrinsic"]
    inv_extrinsic = inv_trafo(extrinsic)
    intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float()
    extrinsic = torch.from_numpy(extrinsic).unsqueeze(0).float()
    inv_extrinsic = torch.from_numpy(inv_extrinsic).unsqueeze(0).float()

    feature = F.pil_to_tensor(pil_image).unsqueeze(0).float()
    _, C, H, W = feature.shape

    points = torch.rand(1, 10000, 3)
    points_world = points - 0.5
    points_cam = torch.baddbmm(extrinsic[:, :3, 3:4], extrinsic[:, :3, :3], points_world.transpose(1, 2)).transpose(
        1, 2
    )

    decoder = GridDecoderAny(c_dim=C, padding=0)
    plane = _sample_plane(decoder, points_cam, feature, plane="xy")

    coords = points_to_coordinates_any(points_cam, plane="xy")
    u, v = (coords[0, :, 0] * W).long().clamp(0, W - 1).numpy(), (coords[0, :, 1] * H).long().clamp(0, H - 1).numpy()

    image_xy = np.zeros((H, W, C), dtype=np.uint8)
    image_xy[v, u] = plane.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

    if show:
        plt.imshow(image_xy)
        plt.show()

    decoder = GridDecoderAny(c_dim=C, padding=0)
    plane = _sample_plane(decoder, points_cam, feature, plane="uv", max_value=W - 1, intrinsic=intrinsic)

    coords = points_to_coordinates_any(points_cam, max_value=W - 1, plane="uv", intrinsic=intrinsic)
    u, v = (coords[0, :, 0] * W).long().clamp(0, W - 1).numpy(), (coords[0, :, 1] * H).long().clamp(0, H - 1).numpy()

    image_uv = np.zeros((H, W, C), dtype=np.uint8)
    image_uv[v, u] = plane.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

    if show:
        plt.imshow(image_xy)
        plt.show()


def test_object_sample_from_image(shapenet_data_root: Path, show: bool, use_points: bool = False):
    from dataset import ImageField, PointCloudField, PointsField

    if not shapenet_data_root.is_dir():
        pytest.skip("'data_root' is not a directory")
    dataset = "occupancy_networks"
    synthset = "02958343"
    obj_id = "1a1de15e572e039df085b75b20c2db33"
    index = 0
    _skip_if_missing(shapenet_data_root / dataset / synthset / obj_id / "img_choy2016" / "cameras.npz")

    image_field = ImageField(index=(0, 24))
    image_data = image_field.load(shapenet_data_root / dataset / synthset / obj_id, index=index)
    pil_image = image_data[None]

    if show:
        plt.imshow(pil_image)
        plt.show()

    intrinsic = image_data["intrinsic"]
    extrinsic = image_data["extrinsic"]
    intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float()
    extrinsic = torch.from_numpy(extrinsic).unsqueeze(0).float()

    feature = F.pil_to_tensor(pil_image).unsqueeze(0).float()
    _, C, H, W = feature.shape

    if use_points:
        points_field = PointsField(padding=0)
        points_data = points_field.load(shapenet_data_root / dataset / synthset / obj_id, index=index)
        points = points_data[None]
        occ = points_data["occ"]
        points_world = torch.from_numpy(points[occ]).unsqueeze(0).float()
    else:
        pcd_field = PointCloudField()
        pcd_data = pcd_field.load(shapenet_data_root / dataset / synthset / obj_id, index=index)
        points = pcd_data[None]
        points_world = torch.from_numpy(points).unsqueeze(0).float()

    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:4]
    points_cam = torch.baddbmm(trans, rot, points_world.transpose(1, 2)).transpose(1, 2)

    decoder = GridDecoderAny(c_dim=C, padding=0)
    plane = _sample_plane(decoder, points_cam, feature, plane="xy")

    coords = points_to_coordinates_any(points_cam, plane="xy")
    u, v = (coords[0, :, 0] * W).long().clamp(0, W - 1).numpy(), (coords[0, :, 1] * H).long().clamp(0, H - 1).numpy()

    image_xy = np.zeros((H, W, C), dtype=np.uint8)
    image_xy[v, u] = plane.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

    if show:
        plt.imshow(image_xy)
        plt.show()

    plane = _sample_plane(
        decoder, points_world, feature, plane="uv", max_value=W - 1, intrinsic=intrinsic, extrinsic=extrinsic
    )

    coords = points_to_coordinates_any(points_cam, max_value=W - 1, plane="uv", intrinsic=intrinsic)
    u, v = (coords[0, :, 0] * W).long().clamp(0, W - 1).numpy(), (coords[0, :, 1] * H).long().clamp(0, H - 1).numpy()

    image_uv = np.zeros((H, W, C), dtype=np.uint8)
    image_uv[v, u] = plane.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

    if show:
        plt.imshow(image_uv)
        plt.show()


def test_depth_sample_from_image(shapenet_data_root: Path, show: bool, use_points: bool, project: bool):
    from dataset import DepthField, PointCloudField, PointsField

    if not shapenet_data_root.is_dir():
        pytest.skip("'data_root' is not a directory")
    dataset = "ShapeNetCore.v1.fused.simple"
    synthset = "02958343"
    obj_id = "1a1de15e572e039df085b75b20c2db33"
    _skip_if_missing(shapenet_data_root / dataset / synthset / obj_id)

    depth_field = DepthField(project=project, crop=True)
    depth_data = depth_field.load(shapenet_data_root / dataset / synthset / obj_id, index=0)
    depth = depth_data[None]
    normals = depth_data["normals"]
    intrinsic = depth_data["intrinsic"]
    extrinsic = depth_data["extrinsic"]
    inv_extrinsic = inv_trafo(extrinsic)
    scale = depth_data["scale"]
    W, H, C = depth_data["width"], depth_data["height"], 3

    if project:
        inputs_world = depth
        inputs_cam = inputs_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        normals = ((normals / 2 + 0.5) * 255).astype(np.uint8)
        intrinsic = adjust_intrinsic(intrinsic, W, H, box=(80, 0, W - 80, H))

        # o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inputs_cam)),
        #                                    o3d.geometry.TriangleMesh.create_coordinate_frame()])

        assert inputs_cam.shape == normals.shape

        u, v, _ = points_to_uv_any(inputs_cam, intrinsic)
        normals_uv = np.zeros((H, H, C), dtype=np.uint8)
        normals_uv[v, u] = normals

        H, W, C = normals_uv.shape
        assert H == W
        S = H

        if show:
            plt.imshow(normals_uv)
            plt.title("normals uv")
            plt.show()

        inputs = torch.from_numpy(inputs_cam).unsqueeze(0).float()
        feature = torch.from_numpy(normals).unsqueeze(0).transpose(1, 2).float()
        intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float()

        coords = points_to_coordinates_any(inputs, max_value=S - 1, plane="uv", intrinsic=intrinsic)
        u1, v1, _ = points_to_uv_any(inputs, intrinsic)
        u2, v2 = (
            (coords[0, :, 0] * S).long().clamp(0, S - 1).numpy(),
            (coords[0, :, 1] * S).long().clamp(0, S - 1).numpy(),
        )
        assert np.array_equal(u1.squeeze(0).numpy(), u2)
        assert np.array_equal(v1.squeeze(0).numpy(), v2)

        padding = 0
        plane = "xy"
        xy_to_uv = padding == 0

        if plane != "uv":
            if xy_to_uv:
                u, v, _ = points_to_uv_any(inputs[0], intrinsic[0])
                points_pixel = torch.from_numpy(np.column_stack([u, v, np.ones_like(u)])).unsqueeze(0).float()
                inputs = points_pixel / (S - 1) - 0.5
            coords = points_to_coordinates_any(inputs, max_value=1 + padding, plane=plane)
        else:
            coords = points_to_coordinates_any(inputs, max_value=S - 1, plane=plane, intrinsic=intrinsic)
        uv = (coords * S).long().clamp(0, S - 1).numpy()
        u = uv[0, :, 0]
        v = uv[0, :, 1]
        image_xy = np.zeros((S, S, C), dtype=np.uint8)
        image_xy[v, u] = feature.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

        if plane == "uv" or xy_to_uv:
            uv_unique = np.unique(np.column_stack([u, v]), axis=0)
            assert len(uv_unique) == len(inputs[0]), f"{len(uv_unique)} != {len(inputs[0])}"

        if not np.array_equal(image_xy, normals_uv):
            if show:
                plt.imshow(image_xy - normals_uv)
                plt.title("xy vs. uv")
                plt.show()

            if plane == "uv" or xy_to_uv:
                raise AssertionError(np.linalg.norm(image_xy - normals_uv))

        index = coordinates_to_index_any(coords, resolution=S)
        image_index = np.zeros((S, S, C), dtype=np.uint8).reshape(-1, C)
        image_index[index[0].numpy()] = feature.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)
        image_index = image_index.reshape(S, S, C)

        assert np.array_equal(image_xy, image_index), np.linalg.norm(image_xy - image_index)

        encoder = GridEncoderAny(c_dim=C, plane_resolution=S, feature_type=(plane,), padding=padding)

        if plane == "uv" or xy_to_uv:
            scatter_index = index.unsqueeze(1)
        else:
            scatter_index = encoder.get_index_dict(inputs)[plane]
            assert torch.equal(index, scatter_index.squeeze(1))

        scattered_feature = encoder.generate_plane_feature(feature, scatter_index)
        image_scattered = scattered_feature.squeeze(0).numpy().transpose(1, 2, 0).astype(np.uint8)

        if not np.array_equal(image_scattered, image_xy):
            if show:
                plt.imshow(image_scattered - image_xy)
                plt.title("scattered vs. xy")
                plt.show()

            if plane == "uv" or xy_to_uv:
                raise AssertionError(np.linalg.norm(image_scattered - image_xy))

        if not np.array_equal(image_scattered, normals_uv):
            if show:
                plt.imshow(image_scattered - normals_uv)
                plt.title("scattered vs. uv")
                plt.show()

            if plane == "uv" or xy_to_uv:
                raise AssertionError(np.linalg.norm(image_scattered - normals_uv))

        if show:
            plt.imshow(image_scattered)
            plt.title("scattered")
            plt.show()

        decoder = GridDecoderAny(c_dim=C, padding=padding, sample_mode="nearest")
        plane = _sample_plane(decoder, inputs, scattered_feature, plane=plane, intrinsic=intrinsic)

        image_sampled = np.zeros((S, S, C), dtype=np.uint8)
        image_sampled[v, u] = plane.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

        if show:
            plt.imshow(image_sampled)
            plt.title("sampled")
            plt.show()

        if not np.array_equal(image_sampled, image_scattered):
            if show:
                plt.imshow(image_sampled - image_scattered)
                plt.title("sampled vs. scattered")
                plt.show()

            if plane == "uv" or xy_to_uv:
                raise AssertionError(np.linalg.norm(image_sampled - image_scattered))
    else:
        normals_image = Image.fromarray(normals.astype(np.uint8))

        if show:
            plt.imshow(normals_image)
            plt.title("normals uv")
            plt.show()

        inputs_cam = depth_to_points(depth, intrinsic, depth_scale=1.0, depth_trunc=6.0) / scale
        inputs_world = inputs_cam @ inv_extrinsic[:3, :3].T + inv_extrinsic[:3, 3]

        inputs_cam = torch.from_numpy(inputs_cam).unsqueeze(0).float()
        inputs_world = torch.from_numpy(inputs_world).unsqueeze(0).float()
        intrinsic = torch.from_numpy(intrinsic).unsqueeze(0).float()
        extrinsic = torch.from_numpy(extrinsic).unsqueeze(0).float()
        scale = torch.from_numpy(scale).unsqueeze(0).float()

        feature = torch.from_numpy(normals[depth > 0]).unsqueeze(0).transpose(1, 2).float()
        image_feature = F.pil_to_tensor(normals_image).unsqueeze(0).float()
        _, C, H, W = image_feature.shape

        v, u = np.nonzero(depth)
        u1, v1, _ = points_to_uv_any(inputs_cam, intrinsic)
        assert np.array_equal(u, u1.squeeze(0).long().numpy())
        assert np.array_equal(v, v1.squeeze(0).long().numpy())

        if use_points:
            points_field = PointsField(file="surface_random.npz", padding=0)
            points_data = points_field.load(shapenet_data_root / dataset / synthset / obj_id / "samples", index=0)
            points = points_data[None]
            occ = points_data["occ"]
            _points_world = torch.from_numpy(points[occ]).unsqueeze(0).float()
        else:
            pcd_field = PointCloudField(file="surface.npz")
            pcd_data = pcd_field.load(shapenet_data_root / dataset / synthset / obj_id / "samples", index=0)
            points = pcd_data[None]
            _points_world = torch.from_numpy(points).unsqueeze(0).float()

        decoder = GridDecoderAny(c_dim=C)
        plane = _sample_plane(
            decoder,
            inputs_world,
            image_feature,
            plane="uv",
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            max_value=max(W, H) - 1,
        )

        coords = points_to_coordinates_any(inputs_cam, max_value=max(W, H) - 1, plane="uv", intrinsic=intrinsic)
        u, v = (coords[0, :, 0] * W).long().numpy(), (coords[0, :, 1] * H).long().numpy()

        normals_uv = 255 * np.ones((H, W, C), dtype=np.uint8)
        normals_uv[v, u] = plane.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

        if show:
            plt.imshow(normals_uv)
            plt.title("normals uv (plane)")
            plt.show()

        plane = _sample_plane(decoder, inputs_world, image_feature, plane="xy", extrinsic=extrinsic)

        coords = points_to_coordinates_any(inputs_cam, plane="xy")
        u, v = (
            (coords[0, :, 0] * W).long().clamp(0, W - 1).numpy(),
            (coords[0, :, 1] * H).long().clamp(0, H - 1).numpy(),
        )

        image_xy = 255 * np.ones((H, W, C), dtype=np.uint8)
        image_xy[v, u] = plane.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

        if show:
            plt.imshow(image_xy)
            plt.title("image xy")
            plt.show()

        coords = points_to_coordinates_any(inputs_cam, max_value=max(W, H) - 1, plane="uv", intrinsic=intrinsic)
        u, v = (
            (coords[0, :, 0] * W).long().clamp(0, W - 1).numpy(),
            (coords[0, :, 1] * H).long().clamp(0, H - 1).numpy(),
        )
        normals_uv = 255 * np.ones((H, W, C), dtype=np.uint8)
        normals_uv[v, u] = feature.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

        if show:
            plt.imshow(normals_uv)
            plt.title("normals (feature)")
            plt.show()

        coords = points_to_coordinates_any(inputs_cam, plane="xy")
        u, v = (
            (coords[0, :, 0] * W).long().clamp(0, W - 1).numpy(),
            (coords[0, :, 1] * H).long().clamp(0, H - 1).numpy(),
        )

        image_xy = 255 * np.ones((H, W, C), dtype=np.uint8)
        image_xy[v, u] = feature.transpose(1, 2).squeeze(0).numpy().astype(np.uint8)

        if show:
            plt.imshow(image_xy)
            plt.title("image xy (feature)")
            plt.show()


def test_projection_with_transformations(shapenet_data_root: Path, show: bool):
    from dataset import Affine, DepthField, Normalize, Rotate

    if not shapenet_data_root.is_dir():
        pytest.skip("'data_root' is not a directory")
    dataset = "ShapeNetCore.v1.fused.simple"
    synthset = "02958343"
    obj_id = "1a1de15e572e039df085b75b20c2db33"
    _skip_if_missing(shapenet_data_root / dataset / synthset / obj_id)

    depth_field = DepthField()
    depth_data = depth_field.load(shapenet_data_root / dataset / synthset / obj_id, index=0)
    normals = depth_data["normals"]
    intrinsic = depth_data["intrinsic"]
    W = depth_data["width"]
    H = depth_data["height"]
    C = 3

    normals = ((normals / 2 + 0.5) * 255).astype(np.uint8)
    intrinsic = adjust_intrinsic(intrinsic, W, H, box=(80, 0, W - 80, H))
    depth_data["intrinsic"] = intrinsic

    inputs_world = depth_data[None]
    extrinsic = depth_data["extrinsic"]
    inputs_cam_orig = inputs_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]

    u1, v1, _ = points_to_uv_any(inputs_cam_orig, intrinsic)
    if show:
        normals_uv = np.zeros((H, H, C), dtype=np.uint8)
        normals_uv[v1, u1] = normals

        plt.imshow(normals_uv)
        plt.show()

    transform = Affine(replace=False)
    rotate = Rotate(axes="x", from_inputs=True)
    norm = Normalize(center=True, scale=True)

    data = copy.deepcopy(depth_data)
    data["inputs"] = data[None]
    data["inputs.extrinsic"] = data["extrinsic"]

    data = transform(data)
    data = rotate(data)
    data = norm(data)

    inputs_net = data["inputs"]
    extrinsic = data["inputs.extrinsic"]
    inputs_cam = inputs_net @ extrinsic[:3, :3].T + extrinsic[:3, 3]

    u2, v2, _ = points_to_uv_any(inputs_cam, intrinsic)
    if show:
        normals_uv = np.zeros((H, H, C), dtype=np.uint8)
        normals_uv[v2, u2] = normals

        plt.imshow(normals_uv)
        plt.show()

    assert np.array_equal(u1, u2)
    assert np.array_equal(v1, v2)

    inputs_cam *= data["inputs.norm_scale"]
    u3, v3, _ = points_to_uv_any(inputs_cam, intrinsic)
    if show:
        normals_uv = np.zeros((H, H, C), dtype=np.uint8)
        normals_uv[v3, u3] = normals

        plt.imshow(normals_uv)
        plt.show()

    assert np.array_equal(u1, u3)
    assert np.array_equal(v1, v3)
    orig_rays = inputs_cam_orig[:, :2] / inputs_cam_orig[:, 2:3]
    new_rays = inputs_cam[:, :2] / inputs_cam[:, 2:3]
    assert np.allclose(orig_rays, new_rays)
