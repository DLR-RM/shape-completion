# pyright: reportMissingImports=false

from time import perf_counter

import ot
import torch

from libs import ChamferDistanceL2, EarthMoversDistance


def test_ot_emd():
    a = torch.randn(4, 1024, 3).cuda()
    b = torch.randn(4, 1024, 3).cuda()
    start = perf_counter()
    dist = ot.emd2(
        [], [], ot.dist(a.view(-1, 3), b.view(-1, 3), metric="sqeuclidean"), numItermax=10_000, numThreads="max"
    )
    print(dist.item())
    print(f"OT EMD: {perf_counter() - start:.3f}s")

    start = perf_counter()
    emd = EarthMoversDistance(eps=1e-10, iters=10_000)
    dist = emd(a, b)[0]
    print(dist.sqrt().mean().item())
    print(f"EMD: {perf_counter() - start:.3f}s")

    start = perf_counter()
    chamfer = ChamferDistanceL2()
    d1, d2 = chamfer(a, b)
    dist = (d1.mean() + d2.mean()) / 2
    print(dist.item())
    print(f"Chamfer: {perf_counter() - start:.3f}s")
