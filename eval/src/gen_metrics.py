from collections.abc import Callable
from enum import Enum
from typing import Any, cast

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import Tensor
from tqdm import tqdm

from libs import ChamferDistanceL2, EarthMoversDistance

chamfer_distance = ChamferDistanceL2()
earth_movers_distance = EarthMoversDistance(eps=0.002, iters=100)


class DistanceMetrics(Enum):
    CHAMFER = "chamfer"
    EMD = "emd"
    F1 = "f1"
    FEAT = "feat"
    FPD = "fpd"


def directed_hausdorff(point_cloud1: Tensor, point_cloud2: Tensor) -> Tensor:
    # Adapted from https://github.com/ChrisWu1997/Multimodal-Shape-Completion/blob/master/evaluation/completeness.py#L11
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2))  # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1))  # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1))  # (B, N, M)
    shortest_dist, _ = torch.min(l2_dist, dim=2)
    hausdorff_dist, _ = torch.max(shortest_dist, dim=1)  # (B, )

    return torch.mean(hausdorff_dist)


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


def cov_mmd(distances: np.ndarray | Tensor, num_points: int | None = None) -> tuple[float, float]:
    """
    Compute Coverage (COV) and Minimum Matching Distance (MMD) metrics.

    Parameters:
    -----------
    distances : Union[np.ndarray, torch.Tensor]
        A 2D matrix of pairwise distances. The first dimension should represent
        generated samples, and the second dimension should represent reference (real) samples.
    num_points : int, optional
        The number of points in each sample. If provided, the MMD will be normalized.

    Returns:
    --------
    Tuple[float, float]
        A tuple containing (COV, MMD)

    Note:
    -----
    COV measures the fraction of reference samples that are matched to at least
    one generated sample. MMD measures the average distance between a reference
    sample and its nearest generated sample.
    """

    if isinstance(distances, np.ndarray):
        distances = torch.from_numpy(distances)
    _n_gen, n_ref = distances.shape[:2]

    # Coverage
    _, min_idx = distances.min(dim=1)
    cov = min_idx.unique().numel() / n_ref

    # Minimum Matching Distance
    min_val, _ = distances.min(dim=0)
    mmd = min_val.mean().item() / (num_points or 1)

    return cov, mmd


@torch.inference_mode()
def distance_fn(
    cloud1: np.ndarray | Tensor,
    cloud2: np.ndarray | Tensor,
    metric: DistanceMetrics = DistanceMetrics.CHAMFER,
    show: bool = False,
) -> float | np.ndarray:
    if isinstance(cloud1, np.ndarray):
        cloud1 = torch.from_numpy(cloud1)
        if cloud1.ndim == 2:
            cloud1 = cloud1.unsqueeze(0)
    cloud1 = cloud1.float().cuda()

    if isinstance(cloud2, np.ndarray):
        cloud2 = torch.from_numpy(cloud2).float()
        if cloud2.ndim == 2:
            cloud2 = cloud2.unsqueeze(0)
    cloud2 = cloud2.float().cuda()

    if metric in [DistanceMetrics.CHAMFER, DistanceMetrics.FEAT, DistanceMetrics.F1]:
        if cloud1.size(-1) != 3 or cloud2.size(-1) != 3:
            from pytorch3d.loss import chamfer_distance as cd_pytorch3d

            d1, d2 = cd_pytorch3d(cloud1, cloud2, batch_reduction=None, point_reduction=None)[0]
        else:
            d1, d2 = chamfer_distance(cloud1, cloud2)
        if metric in [DistanceMetrics.CHAMFER, DistanceMetrics.FEAT]:
            d = d1.sum(-1) + d2.sum(-1)  # As defined in PSGN: https://arxiv.org/abs/1612.00603
        else:
            precision = (d1 < 0.001).mean(-1)
            recall = (d2 < 0.001).mean(-1)
            f1 = 2 * (precision * recall / (precision + recall))
            d = -f1

    elif metric == DistanceMetrics.EMD:
        d = earth_movers_distance(cloud1, cloud2)[0].sum(-1)
        # d = batched_sinkhorn(cloud1.cuda(), cloud2.cuda(), reg=0.01, n_iters=100) ** 2
    else:
        raise ValueError(f"Invalid metric: {metric}")

    if show:
        import open3d as o3d

        print(f"Chamfer distance: {d.mean().cpu().item()}")
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(cloud1[0].cpu().numpy())
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(cloud2[0].cpu().numpy())
        cast(Any, o3d).visualization.draw_geometries(
            [pcd1.paint_uniform_color([1, 0, 0]), pcd2.paint_uniform_color([0, 0, 1])]
        )

    if d.size(0) == 1:
        return d.cpu().item()
    return d.cpu().numpy()


def paired_distances(
    generated_clouds: list[np.ndarray | Tensor],
    reference_clouds: list[np.ndarray | Tensor],
    distance_fn: Callable = distance_fn,
    metric: DistanceMetrics = DistanceMetrics.CHAMFER,
    equal_size: bool = True,
    progress: bool = True,
    show: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    all_clouds = reference_clouds + generated_clouds
    n_batch = 1 if all_clouds[0].ndim == 2 else all_clouds[0].shape[0]

    n_ref = len(reference_clouds) * n_batch
    n_gen = len(generated_clouds) * n_batch
    n_total = n_ref + n_gen
    if equal_size and n_ref != n_gen:
        raise ValueError("Number of generated and reference clouds must be equal")

    distances = np.zeros((n_total, n_total))
    n_steps_simple = n_total * (n_total - 1) // 2
    n_pairs = len(all_clouds) * (len(all_clouds) + 1) // 2
    n_steps_batched = n_batch * n_pairs

    def simple():
        pbar = tqdm(desc=f"Paired distance (simple, {metric.value})", total=n_steps_simple, disable=not progress)
        for i in range(n_total):
            for j in range(i + 1, n_total):
                distance = distance_fn(all_clouds[i], all_clouds[j], metric, show=show and verbose)
                distances[i, j] = distance
                distances[j, i] = distance
                pbar.update(1)
        pbar.close()

    def batched():
        pbar = tqdm(desc=f"Paired distance (batched, {metric.value})", total=n_steps_batched, disable=not progress)
        for i1, b1 in enumerate(all_clouds):
            if not torch.is_tensor(b1):
                b1 = torch.from_numpy(b1)
            b1 = b1.float().cuda()
            for i2, b2 in enumerate(all_clouds):
                if i1 > i2:
                    continue
                if not torch.is_tensor(b2):
                    b2 = torch.from_numpy(b2)
                b2 = b2.float().cuda()

                if b1.size(0) != b2.size(0):
                    raise ValueError("Batch sizes must be equal")

                for row, cloud in enumerate(b2):
                    distance = distance_fn(b1, cloud.unsqueeze(0).expand_as(b1), metric, show=show and verbose)
                    distances[(i2 * n_batch) + row, i1 * n_batch : (i1 + 1) * n_batch] = distance
                    pbar.update(1)
        pbar.close()

    if n_batch > 1:
        batched()
        distances = np.tril(distances)
        distances = distances + distances.T
    else:
        simple()

    if verbose:
        print("Pairwise distances:")
        with np.printoptions(precision=3, suppress=True):
            print(distances)

    return distances


def one_nn_accuracy(distances: np.ndarray, verbose: bool = False) -> float:
    """
    Compute the 1-Nearest Neighbor (1-NN) Accuracy metric for evaluating generative models.

    This metric assesses the similarity between two distributions: one of real samples (S_r)
    and one of generated samples (S_g). It is based on a leave-one-out 1-NN classifier and
    measures how often a sample's nearest neighbor belongs to the same set (real or generated).

    Parameters:
    -----------
    distances : np.ndarray
        A 2D square matrix of pairwise distances between samples. The first half of the
        rows/columns should correspond to reference (real) samples, the second half to generated samples.

    verbose : bool, optional
        If True, print additional information during computation. Default is False.

    Returns:
    --------
    float
        The 1-NN Accuracy score, a value between 0 and 1.

    Interpretation:
    ---------------
    - A score of 0.5 indicates that the generated distribution closely matches the real
      distribution. The classifier cannot distinguish between real and generated samples
      better than random guessing.
    - A score higher than 0.5 suggests that samples are more likely to find their nearest
      neighbor in their own set, indicating that the generated samples are distinguishable
      from real samples.
    - When real samples are from the test set: A score lower than 0.5 indicates that generated
      samples are more similar to real test samples than real samples are to each other. This could
      mean the generator has captured the underlying distribution very well, or it's producing very
      generic samples.
    - When real samples are from the training set: A score lower than 0.5 may indicate overfitting
      to the training data. In the extreme case of perfect memorization of the training set, the score
      would approach 0.
    - For a good generator, as the number of samples increases, the score should converge to 0.5.

    References:
    -----------
    1. Xu, Q., Huang, G., Yuan, Y., Guo, C., Sun, Y., Wu, F., & Weinberger, K. (2018).
       An empirical study on evaluation metrics of generative adversarial networks.
       arXiv preprint arXiv:1806.07755.
    2. Yang, G., Huang, X., Hao, Z., Liu, M. Y., Belongie, S., & Hariharan, B. (2019).
       PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows.
       arXiv preprint arXiv:1906.12320.
    """

    if distances.ndim != 2:
        raise ValueError("Distance matrix must be 2D")
    if distances.shape[0] != distances.shape[1]:
        raise ValueError("Distance matrix must be square")

    np.fill_diagonal(distances, np.inf)
    nearest_neighbors = np.argmin(distances, axis=1)

    if verbose:
        print("Nearest neighbors:")
        print(nearest_neighbors)

    # Create boolean arrays for real and generated samples
    is_real = np.zeros(len(nearest_neighbors), dtype=bool)
    is_real[: len(nearest_neighbors) // 2] = True

    # Count how many times the nearest neighbor is in the same set
    same_set_count = np.sum(is_real[nearest_neighbors] == is_real)

    # Compute accuracy
    accuracy = same_set_count / len(nearest_neighbors)
    return accuracy


def two_sample_test(dist_gen_ref, dist_gen_gen, dist_ref_ref, k: int = 10):
    # From https://github.com/GregorKobsik/Octree-Transformer/blob/master/evaluation/evaluation.py
    n_A, n_B = dist_gen_ref.shape[0], dist_gen_ref.shape[1]
    n = n_A + n_B

    dist = np.empty([n_A + n_B, n_A + n_B])
    dist[:n_A, :n_A] = dist_ref_ref
    dist[:n_A, n_A:] = dist_gen_ref
    dist[n_A:, :n_A] = dist_gen_ref.transpose()
    dist[n_A:, n_A:] = dist_gen_gen

    n_edges = (n - 1) * k
    valence = np.zeros(n)
    inner_A = 0
    inner_B = 0
    edge_AB = 0

    for _i in range(k):
        Tcsr = minimum_spanning_tree(dist)
        tree = Tcsr.toarray()
        indices = np.array(np.nonzero(tree))
        dist[indices[0], indices[1]] = 1e9
        dist[indices[1], indices[0]] = 1e9
        np.add.at(valence, indices[0], 1)
        np.add.at(valence, indices[1], 1)
        edges = (indices < n_A).sum(0)
        inner_A += (edges == 2).sum()
        inner_B += (edges == 0).sum()
        edge_AB += (edges == 1).sum()

    expected_A = n_edges * n_A * (n_A - 1) / (n * (n - 1))
    expected_B = n_edges * n_B * (n_B - 1) / (n * (n - 1))
    c = (valence**2).sum() / 2 - n_edges

    variance_A = (
        expected_A * (1 - expected_A)
        + (2 * c * n_A * (n_A - 1) * (n_A - 2) / (n * (n - 1) * (n - 2)))
        + (
            (n_edges * (n_edges - 1) - 2 * c)
            * (n_A * (n_A - 1) * (n_A - 2) * (n_A - 3))
            / (n * (n - 1) * (n - 2) * (n - 3))
        )
    )

    variance_B = (
        expected_B * (1 - expected_B)
        + (2 * c * n_B * (n_B - 1) * (n_B - 2) / (n * (n - 1) * (n - 2)))
        + (
            (n_edges * (n_edges - 1) - 2 * c)
            * (n_B * (n_B - 1) * (n_B - 2) * (n_B - 3))
            / (n * (n - 1) * (n - 2) * (n - 3))
        )
    )

    covariance = (n_edges * (n_edges - 1) - 2 * c) * (n_A * n_B * (n_A - 1) * (n_B - 1)) / (
        n * (n - 1) * (n - 2) * (n - 3)
    ) - expected_A * expected_B

    cov = np.array([[variance_A, covariance], [covariance, variance_B]])
    cov_inverse = np.linalg.inv(cov)

    diff_vector = np.array([inner_A - expected_A, inner_B - expected_B])
    score = diff_vector.dot(cov_inverse.dot(diff_vector))

    return score
