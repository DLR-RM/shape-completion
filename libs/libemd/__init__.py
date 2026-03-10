import logging
import os
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Function

logger = logging.getLogger(__name__)
_ext: Any | None = None

try:
    import emd_ext as _ext
except (ImportError, ModuleNotFoundError):
    from pathlib import Path

    from torch.utils.cpp_extension import load

    logger.warning("Unable to load Earth Mover's Distance CUDA kernels. JIT compiling...")
    logger.info("Consider installing the CUDA kernels with `python libs/libmanager.py install emd`")

    cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if cuda_arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX"

    try:
        _ext = load(
            name="emd_ext",
            sources=[str(Path(__file__).parent / "emd.cpp"), str(Path(__file__).parent / "emd_cuda.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
        )
    except (OSError, RuntimeError, IndexError) as e:
        logger.error(f"Unable to JIT compile Earth Mover's Distance CUDA kernels: {e}")

_ext_backend = cast(Any, _ext)


class EMDFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters) -> tuple[Tensor, Tensor]:
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert n == m
        assert xyz1.size()[0] == xyz2.size()[0]
        assert n % 1024 == 0
        assert batchsize <= 512

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device="cuda").contiguous()
        assignment = torch.zeros(batchsize, n, device="cuda", dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device="cuda", dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device="cuda").contiguous()
        bid = torch.zeros(batchsize, n, device="cuda", dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device="cuda").contiguous()
        max_increments = torch.zeros(batchsize, m, device="cuda").contiguous()
        unass_idx = torch.zeros(batchsize * n, device="cuda", dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device="cuda", dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device="cuda").contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device="cuda").contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device="cuda").contiguous()

        _ext_backend.forward(
            xyz1,
            xyz2,
            dist,
            assignment,
            price,
            assignment_inv,
            bid,
            bid_increments,
            max_increments,
            unass_idx,
            unass_cnt,
            unass_cnt_sum,
            cnt_tmp,
            max_idx,
            eps,
            iters,
        )

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx) -> tuple[Tensor, Tensor, None, None]:
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device="cuda").contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device="cuda").contiguous()

        _ext_backend.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None


class EarthMoversDistance(nn.Module):
    def __init__(self, eps: float = 0.005, iters: int = 50):
        super().__init__()
        self.eps = eps
        self.iters = iters

    def forward(
        self, input1: Tensor, input2: Tensor, eps: float | None = None, iters: int | None = None
    ) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], EMDFunction.apply(input1, input2, eps or self.eps, iters or self.iters))


def emd_fn_cpu(x: Tensor, y: Tensor) -> Tensor:
    from scipy.optimize import linear_sum_assignment

    bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
    if npts != mpts:
        raise ValueError("EMD is only defined for equal number of points")
    dim = x.shape[-1]
    x = x.reshape(bs, npts, 1, dim)
    y = y.reshape(bs, 1, mpts, dim)
    dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)

    emd_lst = []
    dist_np = dist.cpu().detach().numpy()
    for i in range(bs):
        d_i = dist_np[i]
        r_idx, c_idx = linear_sum_assignment(d_i)
        emd_i = d_i[r_idx, c_idx].mean()
        emd_lst.append(emd_i)
    emd = np.stack(emd_lst).reshape(-1)
    return torch.from_numpy(emd).to(x)
