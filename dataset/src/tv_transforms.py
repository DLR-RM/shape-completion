from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F


class NormalizeDepth(T.Transform):
    """
    Normalizes a depth map to a given range.

    This transform scales the depth so that its values fall within the
    range [a, b]. The min and max values for scaling can either be the
    actual min/max values of the depth map or can be based on percentiles.

    Args:
        a (float): The lower bound of the target range. Default is 0.0.
        b (float): The upper bound of the target range. Default is 1.0.
        p_min (Optional[float]): The lower percentile to use as the minimum value
            for scaling. Must be in [0.0, 1.0). If None, the actual minimum
            value of the tensor is used. Default is None.
        p_max (Optional[float]): The upper percentile to use as the maximum value
            for scaling. Must be in (0.0, 1.0]. If None, the actual maximum
            value of the tensor is used. Default is None.
    """

    def __init__(
        self,
        a: float = 0.0,
        b: float = 1.0,
        p_min: float | None = None,
        p_max: float | None = None,
    ):
        super().__init__()
        if p_min is not None and not (0.0 <= p_min < 1.0):
            raise ValueError(f"p_min must be in the range [0.0, 1.0), but got {p_min}.")
        if p_max is not None and not (0.0 < p_max <= 1.0):
            raise ValueError(f"p_max must be in the range (0.0, 1.0], but got {p_max}.")
        if p_min is not None and p_max is not None and p_min >= p_max:
            raise ValueError(f"p_min must be strictly less than p_max, but got p_min={p_min} and p_max={p_max}.")

        self.a = a
        self.b = b
        self.p_min = p_min
        self.p_max = p_max

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, tv_tensors.Mask):
            return inpt

        # Promote image to float for normalization calculations
        output = inpt.to(dtype=torch.float32)

        if not torch.isfinite(output).all():
            raise ValueError("Input image tensor contains non-finite values.")
        if not output.any():  # If the tensor is all zeros
            return output

        min_val = output.min()
        max_val = output.max()

        if self.p_min is not None:
            min_val = torch.quantile(output, self.p_min)
        if self.p_max is not None:
            max_val = torch.quantile(output, self.p_max)

        # Clip values to the determined min/max range
        output = torch.clamp(output, min_val, max_val)

        if min_val.item() == max_val.item():
            # Handle the case where all values are the same
            if max_val.item() != 0:
                # Scale to `a` if the constant value is not zero
                return (output / max_val) * self.a
            else:
                # If all values are zero, the result is zero
                return output
        else:
            # Perform min-max normalization to the range [a, b]
            scale = (self.b - self.a) / (max_val - min_val)
            return (output - min_val) * scale + self.a


class CenterPad(T.Transform):
    """
    Pads an image, bounding boxes, masks, and camera intrinsics so that
    spatial dimensions are a multiple of a given number.
    """

    def __init__(self, multiple: int = 14):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size: int) -> tuple[int, int]:
        new_size = ((size + self.multiple - 1) // self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_1 = pad_size // 2
        pad_size_2 = pad_size - pad_size_1
        return pad_size_1, pad_size_2

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, (tv_tensors.Image, tv_tensors.Mask, tv_tensors.BoundingBoxes, CameraIntrinsic)):
            h, w = F.get_size(inpt)
            pad_top, pad_bottom = self._get_pad(h)
            pad_left, pad_right = self._get_pad(w)
            return F.pad(inpt, [pad_left, pad_top, pad_right, pad_bottom])
        return inpt


class CameraIntrinsic(tv_tensors.TVTensor):
    """
    A TVTensor for a 3x3 camera intrinsic matrix.
    """

    canvas_size: tuple[int, int]

    @classmethod
    def _wrap(cls, tensor: torch.Tensor, *, canvas_size: tuple[int, int]) -> CameraIntrinsic:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-2:] != (3, 3):
            raise ValueError(f"Intrinsic matrix must be of shape (*, 3, 3), but got {tensor.shape}.")
        intrinsic = tensor.as_subclass(cls)
        intrinsic.canvas_size = canvas_size
        return intrinsic

    def __new__(
        cls,
        data: Any,
        *,
        canvas_size: tuple[int, int],
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> CameraIntrinsic:  # type: ignore[override]
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor, canvas_size=canvas_size)

    @property
    def fx(self) -> torch.Tensor:
        return self[..., 0, 0]

    @property
    def fy(self) -> torch.Tensor:
        return self[..., 1, 1]

    @property
    def cx(self) -> torch.Tensor:
        return self[..., 0, 2]

    @property
    def cy(self) -> torch.Tensor:
        return self[..., 1, 2]

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr(canvas_size=self.canvas_size)


@F.register_kernel(functional=F.get_size, tv_tensor_cls=CameraIntrinsic)
def get_size_intrinsic(intrinsic: CameraIntrinsic) -> tuple[int, int]:
    return intrinsic.canvas_size


@F.register_kernel(functional=F.resize, tv_tensor_cls=CameraIntrinsic)
def resize_intrinsic(intrinsic: CameraIntrinsic, size: int | Sequence[int] | None, **kwargs: Any) -> CameraIntrinsic:
    h, w = intrinsic.canvas_size

    if isinstance(size, (list, tuple)) and len(size) == 2:
        # Case: size is a sequence like (h, w). Output size is matched directly.
        new_h, new_w = int(size[0]), int(size[1])
    else:
        # Case: size is an int, a sequence of length 1, or None.
        if isinstance(size, (list, tuple)):
            s = size[0]
        else:  # size is an int or None
            s = size

        max_size = kwargs.get("max_size")

        if s is None:
            # Case: size is None. The longer edge is matched to max_size.
            if max_size is None:
                # This should not be reached if called from v2.Resize, which raises an error.
                # We'll just return the input without changes.
                return intrinsic

            if h > w:
                new_h = max_size
                new_w = int(max_size * w / h)
            else:
                new_w = max_size
                new_h = int(max_size * h / w)
        else:
            # Case: size is an int. The smaller edge is matched to this number.
            if isinstance(s, Sequence):
                if len(s) == 0:
                    return intrinsic
                s = int(s[0])
            else:
                s = int(s)
            if h < w:
                new_h = s
                new_w = int(s * w / h)
            else:
                new_w = s
                new_h = int(s * h / w)

            # Sub-case: max_size constraint is applied if necessary.
            if max_size is not None and max(new_h, new_w) > max_size:
                if new_h > new_w:
                    new_w = int(max_size * new_w / new_h)
                    new_h = max_size
                else:
                    new_h = int(max_size * new_h / new_w)
                    new_w = max_size

    # If the size hasn't changed, return the original tensor to avoid computation.
    if (new_h, new_w) == (h, w):
        return intrinsic

    # Calculate scaling factors
    scale_w = new_w / w
    scale_h = new_h / h

    # Apply scaling to a clone of the intrinsic matrix
    output_intrinsic = intrinsic.clone()
    output_intrinsic[..., 0, 0] *= scale_w  # fx
    output_intrinsic[..., 0, 2] *= scale_w  # cx
    output_intrinsic[..., 1, 1] *= scale_h  # fy
    output_intrinsic[..., 1, 2] *= scale_h  # cy

    return CameraIntrinsic._wrap(output_intrinsic, canvas_size=(new_h, new_w))


@F.register_kernel(functional=F.resized_crop, tv_tensor_cls=CameraIntrinsic)
def resized_crop_intrinsic(
    intrinsic: CameraIntrinsic,
    top: int,
    left: int,
    height: int,
    width: int,
    size: list[int],
    **kwargs: Any,
) -> CameraIntrinsic:
    """
    Kernel for the resized_crop operation on camera intrinsics.
    This combines a crop and a resize operation.
    """
    # Step 1: Apply the crop transformation
    # We adjust the principal point based on the top-left corner of the crop.
    output_intrinsic = intrinsic.clone()
    output_intrinsic[..., 0, 2] -= left
    output_intrinsic[..., 1, 2] -= top

    # Step 2: Apply the resize transformation
    # The "original" size for the resize is the crop size (height, width).
    # The target size is given by the 'size' parameter.
    new_h, new_w = size
    scale_w = new_w / width
    scale_h = new_h / height

    # Apply scaling to the already-cropped intrinsic
    output_intrinsic[..., 0, 0] *= scale_w  # fx
    output_intrinsic[..., 0, 2] *= scale_w  # cropped cx
    output_intrinsic[..., 1, 1] *= scale_h  # fy
    output_intrinsic[..., 1, 2] *= scale_h  # cropped cy

    return CameraIntrinsic._wrap(output_intrinsic, canvas_size=(new_h, new_w))


@F.register_kernel(functional=F.crop, tv_tensor_cls=CameraIntrinsic)
def crop_intrinsic(intrinsic: CameraIntrinsic, top: int, left: int, height: int, width: int) -> CameraIntrinsic:
    output_intrinsic = intrinsic.clone()
    output_intrinsic[..., 0, 2] -= left
    output_intrinsic[..., 1, 2] -= top

    return CameraIntrinsic._wrap(output_intrinsic, canvas_size=(height, width))


@F.register_kernel(functional=F.pad, tv_tensor_cls=CameraIntrinsic)
def pad_intrinsic(intrinsic: CameraIntrinsic, padding: list[int], **kwargs) -> CameraIntrinsic:
    left, top, right, bottom = padding
    h, w = intrinsic.canvas_size
    new_h, new_w = h + top + bottom, w + left + right

    output_intrinsic = intrinsic.clone()
    output_intrinsic[..., 0, 2] += left
    output_intrinsic[..., 1, 2] += top

    return CameraIntrinsic._wrap(output_intrinsic, canvas_size=(new_h, new_w))


@F.register_kernel(functional=F.hflip, tv_tensor_cls=CameraIntrinsic)
def hflip_intrinsic(intrinsic: CameraIntrinsic, *args, **kwargs) -> CameraIntrinsic:
    h, w = intrinsic.canvas_size
    output_intrinsic = intrinsic.clone()
    output_intrinsic[..., 0, 2] = w - output_intrinsic[..., 0, 2]
    return CameraIntrinsic._wrap(output_intrinsic, canvas_size=(h, w))


@F.register_kernel(functional=F.vflip, tv_tensor_cls=CameraIntrinsic)
def vflip_intrinsic(intrinsic: CameraIntrinsic, *args, **kwargs) -> CameraIntrinsic:
    h, w = intrinsic.canvas_size
    output_intrinsic = intrinsic.clone()
    output_intrinsic[..., 1, 2] = h - output_intrinsic[..., 1, 2]
    return CameraIntrinsic._wrap(output_intrinsic, canvas_size=(h, w))


@F.register_kernel(functional=F.affine, tv_tensor_cls=CameraIntrinsic)
def affine_intrinsic(
    intrinsic: CameraIntrinsic, angle: float, translate: Sequence[float], scale: float, shear: Sequence[float], **kwargs
) -> CameraIntrinsic:
    h, w = intrinsic.canvas_size
    center = [w / 2, h / 2]

    matrix_2x3_inv = cast(Any, F.affine)(center, angle, list(translate), scale, list(shear))
    matrix_3x3_inv = torch.tensor(matrix_2x3_inv, dtype=intrinsic.dtype, device=intrinsic.device).view(2, 3)
    matrix_3x3_inv = torch.cat(
        [matrix_3x3_inv, torch.tensor([[0.0, 0.0, 1.0]], dtype=intrinsic.dtype, device=intrinsic.device)]
    )

    transform_matrix = torch.inverse(matrix_3x3_inv)

    output_intrinsic = transform_matrix @ intrinsic
    return CameraIntrinsic._wrap(output_intrinsic, canvas_size=(h, w))
