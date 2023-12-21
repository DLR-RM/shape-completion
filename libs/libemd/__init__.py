import os
import logging
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.autograd import Function


try:
    import emd
except ImportError:
    from pathlib import Path
    from torch.utils.cpp_extension import load

    logging.info("Unable to load Earth Mover's Distance CUDA kernels. JIT compiling.")
    logging.info("Consider installing the CUDA kernels with `python libs/libmanager.py install emd`.")
    cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if cuda_arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX"
    try:
        emd = load(name="emd",
                   sources=[str(Path(__file__).parent / "emd.cpp"),
                            str(Path(__file__).parent / "emd_cuda.cu")],
                   build_directory=str(Path(__file__).parent))
    except OSError:
        logging.error("Unable to JIT compile Earth Mover's Distance CUDA kernels.")


class EMDFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert (n == m)
        assert (xyz1.size()[0] == xyz2.size()[0])
        assert (n % 1024 == 0)
        assert (batchsize <= 512)

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx,
                    unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None


class EarthMoversDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1: Tensor, input2: Tensor, eps: float = 0.005, iters: int = 50) -> Tuple[Tensor, Tensor]:
        return EMDFunction.apply(input1, input2, eps, iters)
