from typing import Any, cast

import pytest

from models import ConvONet

from ..src.callbacks import VisualizeCallback
from ..src.model import LitModel

pytestmark = pytest.mark.filterwarnings(
    "ignore:The `srun` command is available on your system but is not used.*:UserWarning"
)


def test_init():
    VisualizeCallback()


def test_setup():
    vis_callback = VisualizeCallback()
    vis_callback.setup(cast(Any, None), LitModel("conv_onet_grid", "", ConvONet()), cast(Any, None))
