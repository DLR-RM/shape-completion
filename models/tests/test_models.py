import pytest
import torch

from ..src import (
    PCN,
    PSGN,
    SHAPEFORMER_DEFAULT_KWARGS,
    VQDIF,
    VQDIF_DEFAULT_KWARGS,
    ConvONet,
    DMTet,
    IFNet,
    MCDropoutNet,
    ONet,
    PSSNet,
    RealNVP,
    ShapeFormer,
    SnowflakeNet,
)


@pytest.mark.parametrize(
    "model",
    [ONet, ConvONet, IFNet, MCDropoutNet, RealNVP, PSSNet, PCN, SnowflakeNet, PSGN, VQDIF, DMTet, ShapeFormer],
)
def test_init(model):
    if model is RealNVP and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if issubclass(model, VQDIF):
        model(**VQDIF_DEFAULT_KWARGS)
    elif issubclass(model, ShapeFormer):
        model(**SHAPEFORMER_DEFAULT_KWARGS)
    else:
        model()
