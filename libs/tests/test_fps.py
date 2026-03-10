from time import perf_counter

import numpy as np
import pytest
import torch

from .. import furthest_point_sample


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_furthest_point_sample():
    pytest.importorskip("torch_cluster", reason="torch_cluster is required for backend parity checks")

    n_runs = 1
    B = 32
    N = 10 * 2048
    C = 3
    S = 2048
    points = torch.rand((B, N, C)).cuda()

    start = perf_counter()
    for _ in range(n_runs):
        fps1 = furthest_point_sample(points, S, backend="pvconv")
    print("libpvconv", perf_counter() - start)

    start = perf_counter()
    for _ in range(n_runs):
        fps2 = furthest_point_sample(points, S, backend="pointnet")
    print("libpointnet", perf_counter() - start)

    start = perf_counter()
    for _ in range(n_runs):
        fps3 = furthest_point_sample(points, S, backend="torch_cluster")
    print("torch-cluster", perf_counter() - start)

    start = perf_counter()
    for _ in range(n_runs):
        fps_gt = furthest_point_sample(points[0].cpu().numpy(), S, backend="open3d")
    print("open3d", perf_counter() - start)

    fps1 = fps1[0].cpu().numpy().astype(np.float32)
    fps2 = fps2[0].cpu().numpy().astype(np.float32)
    fps3 = fps3[0].cpu().numpy().astype(np.float32)
    if torch.is_tensor(fps_gt):
        fps_gt = fps_gt.cpu().numpy()
    fps_gt = fps_gt.astype(np.float32)

    # All backends should sample valid, unique source points.
    points_set = {tuple(np.round(p, 7)) for p in points[0].cpu().numpy()}
    for sampled in (fps1, fps2, fps3):
        sampled_set = {tuple(np.round(p, 7)) for p in sampled}
        assert len(sampled_set) == S
        assert sampled_set.issubset(points_set)

    # Backends are not guaranteed to return identical subsets; compare FPS quality instead.
    def min_neighbor_stats(sampled: np.ndarray) -> tuple[float, float]:
        sampled_t = torch.from_numpy(sampled)
        dist = torch.cdist(sampled_t[None], sampled_t[None]).squeeze(0)
        dist.fill_diagonal_(1e9)
        nn = dist.min(dim=1).values
        return nn.mean().item(), nn.min().item()

    gt_mean, gt_min = min_neighbor_stats(fps_gt)
    for sampled in (fps1, fps2, fps3):
        sampled_mean, sampled_min = min_neighbor_stats(sampled)
        assert np.isclose(sampled_mean, gt_mean, rtol=1e-3, atol=1e-4)
        assert np.isclose(sampled_min, gt_min, rtol=1e-3, atol=1e-4)
