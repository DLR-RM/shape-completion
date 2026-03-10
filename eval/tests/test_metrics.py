import pytest
import torch


def test_chamfer():
    from pytorch3d.loss import chamfer_distance

    from libs import ChamferDistanceL2

    x = torch.rand(1, 2048, 3).cuda()
    y = torch.rand(1, 2048, 3).cuda()

    chamfer = ChamferDistanceL2()
    d1, d2 = chamfer(x, y)
    print("L1:", d1.size(), d1.sqrt().sum().item(), d2.sqrt().sum().item())
    print("L2:", d1.size(), d1.sum().item(), d2.sum().item())

    d1, d2 = chamfer_distance(x, y, batch_reduction=None, point_reduction=None, norm=2)[0]
    print("L1:", d1.size(), d1.sqrt().sum().item(), d2.sqrt().sum().item())
    print("L2:", d1.size(), d1.sum().item(), d2.sum().item())
    d1, d2 = chamfer_distance(x, y, batch_reduction=None, point_reduction=None, norm=1)[0]
    print("L1:", d1.size(), d1.sum().item(), d2.sum().item())


@torch.no_grad()
def test_emd():
    from time import perf_counter

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    ot = pytest.importorskip("ot")
    SamplesLoss = pytest.importorskip("geomloss").SamplesLoss

    def earth_movers_distance(cloud1: np.ndarray, cloud2: np.ndarray, metric: str = "sqeuclidean") -> float:
        """
        Compute Earth Mover's Distance between two 3D point clouds.

        Args:
            cloud1: (N, 3) array of points
            cloud2: (N, 3) array of points

        Returns:
            float: EMD between the point clouds
        """
        diffs = cloud1[:, None, :] - cloud2[None, :, :]
        if metric == "sqeuclidean":
            distances = np.sum(diffs * diffs, axis=-1)
        elif metric == "euclidean":
            distances = np.linalg.norm(diffs, axis=-1)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Solve optimal transport problem using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distances)

        # Sum up the distances for optimal assignments
        emd = distances[row_ind, col_ind].sum() / len(cloud1)

        return emd

    def ot_emd(point_cloud_1, point_cloud_2, reg=0.01, eps=0.002, iters=10_000, sinkhorn=False):
        """
        Computes the approximate Earth Mover's Distance between two point clouds using the Sinkhorn algorithm.

        Parameters:
        - point_cloud_1: numpy array of shape (N, 3)
        - point_cloud_2: numpy array of shape (M, 3)
        - reg: Regularization parameter for the Sinkhorn algorithm

        Returns:
        - emd_value: The approximated Earth Mover's Distance between the two point clouds.
        """

        if not torch.is_tensor(point_cloud_1):
            point_cloud_1 = torch.from_numpy(point_cloud_1).cuda()
        if not torch.is_tensor(point_cloud_2):
            point_cloud_2 = torch.from_numpy(point_cloud_2).cuda()

        # Compute the cost matrix (pairwise distances between points)
        cost_matrix = ot.dist(point_cloud_1, point_cloud_2, p=2, metric="euclidean").cpu().numpy()

        # Number of points in each point cloud
        N = point_cloud_1.shape[0]
        M = point_cloud_2.shape[0]

        # Uniform weights for the point clouds (assuming equal mass)
        a = np.ones(N) / N  # Source distribution weights
        b = np.ones(M) / M  # Target distribution weights

        if sinkhorn:
            # Compute the Sinkhorn transport plan
            transport = ot.sinkhorn(a, b, cost_matrix, reg=reg, stopThr=eps, numItermax=iters)
        else:
            transport = ot.emd(a, b, cost_matrix, numThreads="max")

        # Compute the Sinkhorn distance (approximated EMD)
        emd_value = np.sum(transport * cost_matrix)

        return emd_value**2

    def compute_batch_sinkhorn_geomloss(batch_point_cloud_1, batch_point_cloud_2, blur=0.01):
        """
        Computes the approximate Earth Mover's Distance between batches of point clouds using GeomLoss.

        Parameters:
        - batch_point_cloud_1: torch.Tensor of shape (B, N, D)
        - batch_point_cloud_2: torch.Tensor of shape (B, M, D)
        - blur: Regularization parameter for the Sinkhorn algorithm

        Returns:
        - emd_values: torch.Tensor of shape (B,) containing the approximated EMD for each batch.
        """
        # Use SamplesLoss from GeomLoss
        loss = SamplesLoss("sinkhorn", p=2, blur=1.0, scaling=0.5, debias=False)

        # Compute the Sinkhorn loss for each batch
        emd_values = loss(batch_point_cloud_1, batch_point_cloud_2)

        return emd_values

    def batched_sinkhorn(batch_point_cloud_1, batch_point_cloud_2, reg=0.01, n_iters=100):
        """
        Batched Sinkhorn algorithm for computing the approximate EMD between distributions.

        Parameters:
        - epsilon: Regularization parameter
        - n_iters: Number of iterations

        Returns:
        - pi: Optimal transport plans of shape (B, N, M)
        - sinkhorn_dist: Sinkhorn distances of shape (B,)
        """
        B, N, _D = batch_point_cloud_1.size()
        B, M, _D = batch_point_cloud_2.size()
        a = torch.ones(B, N).cuda() / N
        if N == M:
            b = a
        else:
            b = torch.ones(B, M).cuda() / M
        M = torch.cdist(batch_point_cloud_1, batch_point_cloud_2, p=2)  # Cost matrices

        K = torch.exp(-M / reg)  # Kernel
        u = torch.ones_like(a) / a.shape[1]
        v = torch.ones_like(b) / b.shape[1]

        for _ in range(n_iters):
            u = a / (K @ v.unsqueeze(-1)).squeeze(-1)
            v = b / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)

        pi = u.unsqueeze(-1) * K * v.unsqueeze(1)
        return torch.sum(pi * M, dim=[1, 2])

    # Create two sample point clouds
    cloud1 = np.random.rand(2048, 3)
    cloud2 = np.random.rand(2048, 3)

    start = perf_counter()
    distance = earth_movers_distance(cloud1, cloud2)
    print(f"Squared Earth Mover's Distance: {distance:.4f} ({perf_counter() - start:.4f}s)")
    distance = earth_movers_distance(cloud1, cloud2, metric="euclidean")
    print(f"Earth Mover's Distance: {distance:.4f} ({perf_counter() - start:.4f}s)")

    start = perf_counter()
    distance = ot_emd(cloud1, cloud2)
    print(f"OT Earth Mover's Distance: {distance:.4f} ({perf_counter() - start:.4f}s)")

    start = perf_counter()
    distance = ot_emd(cloud1, cloud2, reg=0.01, eps=0.002, iters=10_000, sinkhorn=True)
    print(f"Sinkhorn Earth Mover's Distance: {distance:.4f} ({perf_counter() - start:.4f}s)")

    cloud1 = torch.from_numpy(cloud1).float().cuda().unsqueeze(0)
    cloud2 = torch.from_numpy(cloud2).float().cuda().unsqueeze(0)

    # start = perf_counter()
    # distance = compute_batch_sinkhorn_geomloss(cloud1, cloud2, blur=0.5)
    # print(f"GeomLoss Earth Mover's Distance: {distance.item():.4f} ({perf_counter() - start:.4f}s)")

    start = perf_counter()
    distance = batched_sinkhorn(cloud1, cloud2, reg=0.01, n_iters=10_000)
    print(distance.shape)
    print(f"Squared Sinkhorn Earth Mover's Distance: {(distance**2).item():.4f} ({perf_counter() - start:.4f}s)")
    print(f"Batched Sinkhorn Earth Mover's Distance: {distance.item():.4f} ({perf_counter() - start:.4f}s)")

    from libs import EarthMoversDistance

    emd = EarthMoversDistance(eps=0.002, iters=10_000)
    start = perf_counter()
    distance = emd(cloud1, cloud2)[0]
    print(distance.shape)
    print(f"Earth Mover's Distance: {distance.mean(-1).item():.4f} ({perf_counter() - start:.4f}s)")

    emd = EarthMoversDistance(eps=0.01, iters=100)
    start = perf_counter()
    distance = emd(cloud1, cloud2)[0].mean(-1)
    print(f"Earth Mover's Distance: {distance.item():.4f} ({perf_counter() - start:.4f}s)")

    from libs import ChamferDistanceL2

    chamfer = ChamferDistanceL2()
    start = perf_counter()
    d1, d2 = chamfer(cloud1, cloud2)
    distance = (d1 / 2 + d2 / 2).mean(-1)
    print(f"Chamfer Distance: {distance.item():.4f} ({perf_counter() - start:.4f}s)")
