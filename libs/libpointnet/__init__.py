import logging
from typing import cast

from torch import Tensor

try:
    from .pointnet2_ops.pointnet2_utils import furthest_point_sampling as furthest_point_sampling
    from .pointnet2_ops.pointnet2_utils import gather as gather
    from .pointnet2_ops.pointnet2_utils import group as group

    __all__ = ["furthest_point_sample", "furthest_point_sampling", "gather", "group"]

    def furthest_point_sample(points: Tensor, num_samples: int) -> Tensor:
        indices = furthest_point_sampling(points[..., :3], num_samples)
        gathered = cast(Tensor, gather(points.transpose(1, 2).contiguous(), indices))
        return gathered.transpose(1, 2)

except ImportError:
    logging.warning(
        "The `pointnet` library is not installed. Please install it using `python libs/libmanager.py install pointnet`"
    )
    raise
