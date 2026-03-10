import logging

from .grid import GridDiffusionModel as GridDiffusionModel
from .latent import LatentDiffusionModel as LatentDiffusionModel
from .model import DiffusionModel as DiffusionModel
from .model import bit2int as bit2int
from .model import int2bit as int2bit
from .pcd import PVDModel as PVDModel
from .shape3d2vecset import EDMPrecond as EDMPrecond
from .transformer import EDMTransformer as EDMTransformer
from .unet import UNetModel as UNetModel

__all__ = [
    "DiffusionModel",
    "EDMPrecond",
    "EDMTransformer",
    "GridDiffusionModel",
    "LatentDiffusionModel",
    "PVDModel",
    "UNetModel",
    "bit2int",
    "int2bit",
]

try:
    from .diffusers import DiffusersModel as DiffusersModel
except ImportError as err:
    logging.warning("Unable to import Diffusers models: %s", err)
    logging.info("Consider installing the 'diffusers' package")
else:
    __all__.append("DiffusersModel")
