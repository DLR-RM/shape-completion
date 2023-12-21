import os
import logging
from typing import Tuple, Any, Union

import torch
from torch import nn, Tensor
from torch.autograd import Function


try:
    import chamfer
except ImportError:
    from pathlib import Path
    from torch.utils.cpp_extension import load

    logging.info("Unable to load Chamfer Distance CUDA kernels. JIT compiling.")
    logging.info("Consider installing the CUDA kernels with `python libs/libmanager.py install chamfer`.")
    cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if cuda_arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX"
    try:
        chamfer = load(name="chamfer",
                       sources=[str(Path(__file__).parent / "chamfer_cuda.cpp"),
                                str(Path(__file__).parent / "chamfer.cu")],
                       build_directory=str(Path(__file__).parent))
    except OSError:
        logging.error("Unable to JIT compile Chamfer Distance CUDA kernels.")


class ChamferFunction(Function):
    @staticmethod
    def forward(ctx: Any, xyz1: Tensor, xyz2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1.float(), xyz2.float())
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx: Any, grad_dist1: Tensor, grad_dist2: Tensor) -> Tuple[Tensor, Tensor]:
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferFunctionWithIndices(Function):
    @staticmethod
    def forward(ctx: Any, xyz1: Tensor, xyz2: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1.float(), xyz2.float())
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx: Any, grad_dist1: Tensor, grad_dist2: Tensor) -> Tuple[Tensor, Tensor]:
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(nn.Module):
    def __init__(self, return_indices: bool = False):
        super().__init__()
        self.return_indices = return_indices

    def forward(self,
                xyz1: Tensor,
                xyz2: Tensor) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        if self.return_indices:
            dist1, dist2, idx1, idx2 = ChamferFunctionWithIndices.apply(xyz1, xyz2)
        else:
            dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        if torch.all(dist1 == 0) and torch.all(dist2 == 0):
            raise RuntimeError("Chamfer distance is zero for all points.")
        if self.return_indices:
            return dist1, dist2, idx1, idx2
        return dist1, dist2
