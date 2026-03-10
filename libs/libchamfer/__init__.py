import logging
import os
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.autograd import Function

logger = logging.getLogger(__name__)
_ext: Any | None = None

try:
    import chamfer_ext as _ext
except (ImportError, ModuleNotFoundError):
    from pathlib import Path

    from torch.utils.cpp_extension import load

    logger.warning("Unable to load Chamfer Distance CUDA kernels. JIT compiling...")
    logger.info("Consider installing the CUDA kernels with `python libs/libmanager.py install chamfer`.")

    cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if cuda_arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX"

    try:
        _ext = load(
            name="chamfer_ext",
            sources=[str(Path(__file__).parent / "chamfer_cuda.cpp"), str(Path(__file__).parent / "chamfer.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
        )
    except (OSError, RuntimeError, IndexError) as e:
        logger.error(f"Unable to JIT compile Chamfer Distance CUDA kernels: {e}")

_ext_backend = cast(Any, _ext)


class ChamferFunction(Function):
    @staticmethod
    def forward(ctx: Any, xyz1: Tensor, xyz2: Tensor) -> tuple[Tensor, Tensor]:
        dist1, dist2, idx1, idx2 = _ext_backend.forward(xyz1.float().contiguous(), xyz2.float().contiguous())
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx: Any, grad_dist1: Tensor, grad_dist2: Tensor) -> tuple[Tensor, Tensor]:
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = _ext_backend.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferFunctionWithIndices(Function):
    @staticmethod
    def forward(ctx: Any, xyz1: Tensor, xyz2: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dist1, dist2, idx1, idx2 = _ext_backend.forward(xyz1.float().contiguous(), xyz2.float().contiguous())
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx: Any, grad_dist1: Tensor, grad_dist2: Tensor) -> tuple[Tensor, Tensor]:
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = _ext_backend.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(nn.Module):
    def __init__(self, return_indices: bool = False):
        super().__init__()
        self.return_indices = return_indices

    def forward(self, xyz1: Tensor, xyz2: Tensor) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.return_indices:
            dist1, dist2, idx1, idx2 = cast(tuple[Tensor, Tensor, Tensor, Tensor], ChamferFunctionWithIndices.apply(xyz1, xyz2))
        else:
            dist1, dist2 = cast(tuple[Tensor, Tensor], ChamferFunction.apply(xyz1, xyz2))
        # if torch.all(dist1 == 0) and torch.all(dist2 == 0):
        #     raise RuntimeError("Chamfer distance is zero for all points.")
        if self.return_indices:
            return dist1, dist2, idx1, idx2
        return dist1, dist2


def chamfer_fn_cpu(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    x, y = a, b
    _bs, num_points, _points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]
