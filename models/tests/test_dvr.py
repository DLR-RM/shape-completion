import time
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from utils import apply_trafo, depth_to_points

from ..src.dinov2 import DinoRGB
from ..src.dvr import DVR, DifferentiableRayCaster, RayMarchingConfig

pytestmark = pytest.mark.filterwarnings("ignore:xFormers is not available.*:UserWarning")


def test_get_rays_identity():
    # One-pixel image, identity intrinsics/extrinsics: ray0 at origin, direction = +z.
    mask = torch.zeros(1, 1, 1)
    intrinsic = torch.eye(3).unsqueeze(0)
    extrinsic = torch.eye(4).unsqueeze(0)
    ray0, ray_dir, _u, _v = DVR.get_rays(mask, intrinsic, extrinsic, normalize=False, num_samples=None)
    # shapes
    assert ray0.shape == (1, 1, 3)
    assert ray_dir.shape == (1, 1, 3)
    # origin at (0,0,0)
    assert torch.allclose(ray0, torch.zeros_like(ray0))
    # direction along +z
    expected = torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3)
    assert torch.allclose(ray_dir, expected)


def test_secant_refinement_on_sphere_sdf():
    # sphere SDF of radius R=2
    R = 1.3465

    def sphere_sdf(points, feature=None, **kwargs):
        return torch.norm(points, dim=-1) - R

    # set up a single ray from origin along +x
    ray0_masked = torch.zeros(1, 3)
    ray_dir_masked = torch.tensor([[1.0, 0.0, 0.0]])
    # initial bracket [d_low, d_high] = [0, 4]
    d_low = torch.zeros(1)
    d_high = torch.full((1,), 4.0)
    # compute f at the two ends
    f_low = sphere_sdf(ray0_masked)
    f_high = sphere_sdf(ray0_masked + 4.0 * ray_dir_masked)

    print("linear", -f_low * (d_high - d_low) / (f_high - f_low) + d_low)
    print("midpoint", 0.5 * (d_low + d_high))

    # Run secant-refinement.
    cfg = RayMarchingConfig(num_refine_steps=7)
    caster = DifferentiableRayCaster(cfg)

    d_pred = caster.run_bisection_method(
        d_low=d_low,
        d_high=d_high,
        ray0_masked=ray0_masked,
        ray_dirs_masked=ray_dir_masked,
        predict=sphere_sdf,
        feature_masked=None,
    )
    print("bisection", d_pred)
    assert torch.isclose(d_pred, torch.tensor([R]), atol=0.1)

    d_pred = caster.run_secant_method(
        f_low=f_low,
        f_high=f_high,
        d_low=d_low,
        d_high=d_high,
        ray0_masked=ray0_masked,
        ray_dirs_masked=ray_dir_masked,
        predict=sphere_sdf,
        feature_masked=None,
    )
    print("secant", d_pred)
    # mask_pred would be True since f_low<0 and f_high>0
    # check that the refined root ≈ R
    assert torch.isclose(d_pred, torch.tensor([R]), atol=0.01)


@pytest.mark.integration
def test_ray_sampling(shapenet_data_root: Path, o3d_viz_enabled: bool):
    import numpy as np
    import open3d as o3d

    from dataset import DepthField, Normalize, Scale

    config = RayMarchingConfig(num_pixels=10_000, crop=False, refine_mode="secant")
    dvr = DVR(model=DinoRGB(freeze=False, condition="cls", init=False, nerf_enc="torch"), config=config)

    if not shapenet_data_root.is_dir():
        pytest.skip("'data_root' is not a directory")
    dataset = "ShapeNetCore.v1.fused.simple"
    # synthset = "02958343"
    synthset = "02691156"
    # obj_id = "1a1de15e572e039df085b75b20c2db33"
    obj_id = "1a04e3eab45ca15dd86060f189eb133"

    field = DepthField(project=False, crop=True, resize=224)
    data = field.load(shapenet_data_root / dataset / synthset / obj_id, index=0)
    data = {"inputs." + k if k else "inputs": v for k, v in data.items()}
    data["pointcloud"] = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    data = Scale(from_inputs=True, multiplier=1)(data)
    # data = Affine(replace=True)(data)  # "cam" frame
    data = Normalize(center=True, reference="pointcloud")(data)
    mask_gt = torch.from_numpy(data["inputs.mask"]).unsqueeze(0).bool()
    depth = torch.from_numpy(data["inputs.depth"]).unsqueeze(0).float()
    image = torch.from_numpy(data["inputs.image"]).permute(2, 0, 1).unsqueeze(0).float()
    intrinsic = torch.from_numpy(data["inputs.intrinsic"]).unsqueeze(0).float()
    extrinsic = torch.from_numpy(data["inputs.extrinsic"]).unsqueeze(0).float()
    b, c, h, w = image.size()

    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(
        torch.inverse(extrinsic).squeeze(0).numpy()
    )

    ray0, ray_direction, u, v = dvr.get_rays(mask_gt, intrinsic, extrinsic, num_samples=config.num_pixels)

    mask_gt = mask_gt.view(b, -1)
    image = image.view(b, c, -1)
    _depth = depth.view(b, -1)
    if config.num_pixels is not None and config.num_pixels < h * w:
        index = (v.long() * w + u.long()).long()
        mask_gt = torch.gather(mask_gt, dim=1, index=index)
        _depth = torch.gather(depth.view(b, -1), dim=1, index=index)
        image = torch.gather(image, dim=2, index=index.unsqueeze(1).expand(-1, c, -1))

    d = dvr.ray_caster.sample_distances(
        n_points=ray0.size(1), n_steps=1, n_batch=ray0.size(0), ray0=ray0, ray_dirs=ray_direction, method="random"
    ).squeeze(-1)
    p = ray0 + d.unsqueeze(-1) * ray_direction

    uv_coords = torch.stack((u, v), dim=-1)
    xyz = torch.cat((uv_coords, torch.ones(b, ray0.size(1), 1)), dim=-1)
    p_depth = xyz @ torch.inverse(intrinsic).transpose(-1, -2)
    p_depth *= _depth.unsqueeze(-1)
    p_depth = apply_trafo(p_depth, torch.inverse(extrinsic))
    p_depth = p_depth[mask_gt]

    p_depth_cam = cast(torch.Tensor, apply_trafo(p_depth, extrinsic))
    assert torch.allclose(p_depth_cam[..., -1], _depth[mask_gt], atol=1e-4)

    p_free = p[~mask_gt]
    p_occ = p[mask_gt]

    box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-0.5 - config.padding / 2,) * 3, max_bound=(0.5 + config.padding / 2,) * 3
    )
    box.color = np.zeros(3)

    p_occ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_occ.view(-1, 3).numpy()))
    p_free = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_free.view(-1, 3).numpy()))
    p_depth = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_depth.view(-1, 3).numpy()))
    p_cam = depth_to_points(depth.squeeze(0).numpy(), intrinsic.squeeze(0).numpy())
    p_world = apply_trafo(p_cam, torch.inverse(extrinsic).squeeze(0).numpy())
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_world))

    print(np.asarray(pcd.compute_point_cloud_distance(p_depth)).mean())

    visualization: Any = getattr(o3d, "visualization", None)
    if visualization is None:
        pytest.skip("open3d visualization module is unavailable")

    if o3d_viz_enabled:
        visualization.draw_geometries(
            [
                p_free.paint_uniform_color((0.7,) * 3),
                p_occ.paint_uniform_color((0, 0, 1)),
                p_depth.paint_uniform_color((0, 1, 0)),
                pcd.paint_uniform_color((0, 0, 0)),
                world,
                cam,
                box,
            ]
        )


# @torch.autocast("cuda", dtype=torch.bfloat16, enabled=True)
@pytest.mark.integration
def test_forward():
    torch.manual_seed(1234)
    config = RayMarchingConfig(near=1, far=2.4, crop=True, refine_mode="secant")
    dvr = DVR(
        model=DinoRGB(
            backbone="dinov2_vits14",
            freeze=False,
            condition="cls",
            init=False,
            nerf_enc="torch",
        ),
        config=config,
        depth_loss="l2",
        rgb_loss=None,
    ).cuda()

    batch_size = 2
    width, height = 224, 224
    image = torch.randn(batch_size, 3, height, width)
    depth = (1.0 + torch.randn(batch_size, height, width)).clamp(0)
    intrinsic = torch.tensor(
        [
            [
                [width, 0.0, width / 2],
                [0.0, width, height / 2],
                [0.0, 0.0, 1.0],
            ]
        ]
    ).expand(batch_size, -1, -1)
    extrinsic = torch.eye(4, 4).unsqueeze(0).expand(batch_size, -1, -1)
    extrinsic[0, 2, 3] = 2.0
    item: dict[str, list[str] | torch.Tensor] = {
        "inputs": image.cuda(),
        "inputs.depth": depth.cuda(),
        "inputs.mask": (depth > 0).cuda(),
        "inputs.intrinsic": intrinsic.cuda(),
        "inputs.extrinsic": extrinsic.cuda(),
        "inputs.width": 2 * intrinsic[:, 0, 2].cuda(),
        "inputs.height": 2 * intrinsic[:, 1, 2].cuda(),
    }

    start = time.perf_counter()
    data = dvr.step(item)
    loss = cast(torch.Tensor, data["loss"])
    points = cast(torch.Tensor, data["points"])
    points_occ = cast(torch.Tensor, data["points.occ"])
    print("forward time:", time.perf_counter() - start)

    start = time.perf_counter()
    loss.backward()
    print("backward time:", time.perf_counter() - start)
    assert torch.isfinite(loss)
    assert "points" in data and "points.occ" in data
    assert points.shape[-1] == 3
    assert points_occ.ndim == 2


@pytest.mark.integration
def test_backward():
    config = RayMarchingConfig(near=1, far=2.4, crop=True, refine_mode="secant")
    dvr = DVR(model=DinoRGB(backbone="dinov2_vits14", freeze=False, condition="cls", init=False), config=config).cuda()

    batch_size = 1
    width, height = 28, 28
    image = torch.empty(batch_size, 3, height, width).uniform_(-1.0, 1.0)
    depth = (1.0 + torch.randn(batch_size, height, width)).clamp(0)
    intrinsic = torch.tensor(
        [
            [
                [width, 0.0, width / 2],
                [0.0, width, height / 2],
                [0.0, 0.0, 1.0],
            ]
        ]
    ).expand(batch_size, -1, -1)
    extrinsic = torch.eye(4, 4).unsqueeze(0).expand(batch_size, -1, -1)
    extrinsic[0, 2, 3] = 2.0

    image.requires_grad = True
    depth.requires_grad = True

    item: dict[str, list[str] | torch.Tensor] = {
        "inputs": image.cuda(),
        "inputs.depth": depth.cuda(),
        "inputs.mask": (depth > 0).cuda(),
        "inputs.intrinsic": intrinsic.cuda(),
        "inputs.extrinsic": extrinsic.cuda(),
    }
    out = dvr.step(item)
    loss = cast(torch.Tensor, out["loss"])
    loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()
