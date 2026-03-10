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


def test_init():
    for model in [
        ONet,
        ConvONet,
        IFNet,
        MCDropoutNet,
        RealNVP,
        PSSNet,
        PCN,
        SnowflakeNet,
        PSGN,
        VQDIF,
        DMTet,
        ShapeFormer,
    ]:
        if issubclass(model, VQDIF):
            model(**VQDIF_DEFAULT_KWARGS)
        elif issubclass(model, ShapeFormer):
            model(**SHAPEFORMER_DEFAULT_KWARGS)
        else:
            model()
