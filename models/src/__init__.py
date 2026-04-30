# pyright: reportUnsupportedDunderAll=false

import logging

from . import utils as _utils
from .autoregression import AutoregressiveModel, LatentAutoregressiveModel, LatentGPT
from .completr import CompleTr
from .conv_onet import ConvONet

try:
    from .diffusion import (
        DiffusersModel,
        DiffusionModel,
        EDMPrecond,
        EDMTransformer,
        GridDiffusionModel,
        LatentDiffusionModel,
        PVDModel,
        UNetModel,
        bit2int,
        int2bit,
    )
except (ModuleNotFoundError, ImportError) as err:
    logging.warning("Unable to import diffusion models: %s", err)
    logging.info("Install with: uv sync --extra diffusion")
    DiffusersModel = None
    DiffusionModel = None
    EDMPrecond = None
    EDMTransformer = None
    GridDiffusionModel = None
    LatentDiffusionModel = None
    PVDModel = None
    UNetModel = None
    bit2int = None
    int2bit = None
from .dinov2 import (
    Dino3D,
    DinoCls,
    DinoInst3D,
    DinoInstSeg,
    DinoInstSeg3D,
    DinoInstSegRGBD,
    DinoRGB,
    DinoRGBD,
    InstOccPipeline,
)
from .dmtet import DMTet
from .dvr import DVR, RayMarchingConfig
from .idr import ImplicitNetwork, RenderingNetwork
from .if_net import IFNet
from .mc_dropout_net import MCDropoutNet
from .model import Model
from .onet import ONet
from .pcn import PCN
from .pifu import PIFu
from .point_transformer import PointTransformer
from .pointnet import PointNetCls, PointNetSeg
from .psgn import PSGN
from .pssnet import PSSNet
from .realnvp import RealNVP
from .shape3d2vecset import Shape3D2VecSet, Shape3D2VecSetCls, Shape3D2VecSetVAE, Shape3D2VecSetVQVAE
from .shapeformer import DEFAULT_KWARGS as SHAPEFORMER_DEFAULT_KWARGS
from .shapeformer import ShapeFormer
from .snowflakenet import SnowflakeNet
from .transformer import TCNN_EXISTS, Attention, NeRFEncoding
from .utils import classification_loss, get_activation, patch_attention, probs_from_logits, regression_loss
from .vae import VAEModel, VQVAEModel
from .vqdif import DEFAULT_KWARGS as VQDIF_DEFAULT_KWARGS
from .vqdif import VQDIF

try:
    from .mask_rcnn import MaskRCNN
except (ModuleNotFoundError, ImportError):
    MaskRCNN = None

_UTILS_EXPORTS = [name for name in dir(_utils) if not name.startswith("_")]
globals().update({name: getattr(_utils, name) for name in _UTILS_EXPORTS})

__all__ = [
    "Attention",
    "AutoregressiveModel",
    "CompleTr",
    "ConvONet",
    "Dino3D",
    "DinoCls",
    "DinoInst3D",
    "DinoInstSeg",
    "DinoInstSeg3D",
    "DinoInstSegRGBD",
    "DinoRGB",
    "DinoRGBD",
    "DMTet",
    "DVR",
    "DiffusersModel",
    "DiffusionModel",
    "EDMPrecond",
    "EDMTransformer",
    "GridDiffusionModel",
    "IFNet",
    "ImplicitNetwork",
    "InstOccPipeline",
    "LatentAutoregressiveModel",
    "LatentDiffusionModel",
    "LatentGPT",
    "MCDropoutNet",
    "MaskRCNN",
    "Model",
    "NeRFEncoding",
    "ONet",
    "PCN",
    "PIFu",
    "PSGN",
    "PSSNet",
    "PVDModel",
    "PointNetCls",
    "PointNetSeg",
    "PointTransformer",
    "RayMarchingConfig",
    "RealNVP",
    "RenderingNetwork",
    "SHAPEFORMER_DEFAULT_KWARGS",
    "Shape3D2VecSet",
    "Shape3D2VecSetCls",
    "Shape3D2VecSetVAE",
    "Shape3D2VecSetVQVAE",
    "ShapeFormer",
    "SnowflakeNet",
    "TCNN_EXISTS",
    "UNetModel",
    "VAEModel",
    "VQDIF",
    "VQDIF_DEFAULT_KWARGS",
    "VQVAEModel",
    "bit2int",
    "classification_loss",
    "get_activation",
    "int2bit",
    "patch_attention",
    "probs_from_logits",
    "regression_loss",
    *_UTILS_EXPORTS,
]
