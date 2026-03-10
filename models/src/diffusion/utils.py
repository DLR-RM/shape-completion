from inspect import isfunction
from typing import Any

from torch import Size, Tensor, nn

from utils import unsqueeze_as


def get_convnd(ndim: int) -> type[nn.Conv1d | nn.Conv2d | nn.Conv3d]:
    try:
        return getattr(nn, f"Conv{ndim}d")
    except AttributeError as err:
        raise NotImplementedError(f"Conv{ndim}d not implemented") from err


def exists(x: Any) -> bool:
    return x is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d() if isfunction(d) else d


def num_to_groups(num: int, divisor: int) -> list[int]:
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def extract(tensor: Tensor, timestep: Tensor, sample_shape: Size | tuple[int, ...]) -> Tensor:
    return unsqueeze_as(tensor.gather(-1, timestep), sample_shape)
