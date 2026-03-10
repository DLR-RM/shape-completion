from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from vector_quantize_pytorch import ResidualVQ

from utils import monkey_patch, resolve_checkpoint_path, resolve_path, resolve_weights_path, setup_logger

from .src import *

logger = setup_logger(__name__)


def get_model(cfg: DictConfig, arch: str | None = None, weights_path: str | Path | None = None, **kwargs) -> Model:
    arch = str(arch or cfg.model.arch)
    weights_path = weights_path or cfg.model.weights
    load_weights = weights_path or cfg.train.skip or not cfg.train.resume
    strict = True

    if cfg.inputs.nerf or cfg.points.nerf:
        nerf_enc = NeRFEncoding(
            in_dim=cfg.inputs.dim,
            padding=cfg.norm.padding,
            scale_inputs="conv_onet" not in arch and arch not in ["if_net", "vqdif"],
        )

    model: Any = None
    if "conv_onet" in arch:
        model = ConvONet(
            arch=arch,
            dim=cfg.inputs.dim,
            padding=cfg.norm.padding,
            inputs_type=cfg.inputs.type,
            custom_grid_sampling=cfg.vis.refinement_steps,
            num_classes=cfg.cls.num_classes,
            norm=cfg.model.norm,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
            condition=cfg.get("condition", "add"),
            sample_mode=cfg.get("sample_mode", "bilinear"),
            padding_mode=cfg.get("padding_mode", "zeros"),
            align_corners=cfg.get("align_corners", True),
            resize_intrinsic=cfg.get("resize_intrinsic", False),
            encoder_show=cfg.get("encoder_show", False),
            decoder_show=cfg.get("decoder_show", False),
        )
    elif arch == "if_net":
        model = IFNet(
            padding=cfg.norm.padding,
            displacements=cfg.get("displacements", True),
            multires=cfg.get("multires", True),
            pvconv=cfg.get("pvconv", False),
            xdconv=cfg.get("rxconv", False),
            points=cfg.get("rxconv_points", True),
            planes=cfg.get("rxconv_planes", ["xy", "xz", "yz"]),
            grid=cfg.get("rxconv_grid", True),
            scatter_type=cfg.get("scatter_type", "mean"),
            fuse_point_feature=cfg.get("fuse_point_feature", True),
            downsample=cfg.get("downsample", None),
            vanilla=cfg.get("vanilla", False),
        )
    elif "onet" in arch:
        model = ONet(
            arch=arch,
            inputs_type=cfg.inputs.type,
            dim=nerf_enc.get_enc_dim() if cfg.inputs.nerf or cfg.points.nerf else cfg.inputs.dim,
            norm=cfg.model.norm,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
        )
    elif "dropout" in arch:
        model = MCDropoutNet(input_res=cfg.inputs.voxelize, **kwargs)
    elif arch == "realnvp":
        model = RealNVP(**kwargs)
    elif arch == "pssnet":
        flow_path: str | Path | None = resolve_path("out/realnvp/mugs/paper/realnvp/model_best.pt")
        if weights_path or cfg.model.checkpoint or cfg.model.load_best:
            flow_path = None
        model = PSSNet(path_to_pretrained_flow=str(flow_path) if flow_path is not None else None)
    elif arch == "pcn":
        model = PCN(**kwargs)
    elif arch == "snowflakenet":
        model = SnowflakeNet(**kwargs)
    elif arch == "psgn":
        model = PSGN(**kwargs)
    elif arch == "vqdif":
        if cfg.inputs.nerf:
            VQDIF_DEFAULT_KWARGS["encoder_opt"]["dim"] = nerf_enc.get_enc_dim()
        if cfg.points.nerf:
            VQDIF_DEFAULT_KWARGS["decoder_opt"]["dim"] = nerf_enc.get_enc_dim()
        VQDIF_DEFAULT_KWARGS["decoder_opt"]["sample_mode"] = cfg.get("sample_mode", "bilinear")
        VQDIF_DEFAULT_KWARGS["decoder_opt"]["padding_mode"] = cfg.get("padding_mode", "zeros")
        VQDIF_DEFAULT_KWARGS["decoder_opt"]["align_corners"] = cfg.get("align_corners", True)
        model = VQDIF(
            **VQDIF_DEFAULT_KWARGS,
            cls_num_classes=cfg.cls.num_classes,
            seg_num_classes=cfg.seg.num_classes,
            custom_grid_sampling=cfg.vis.refinement_steps,
        )
    elif "dmtet" in arch:
        model = DMTet(
            scatter=cfg.get("scatter", "scatter" in arch),
            fuse_point_feature=cfg.get("fuse_point_feature", True),
            fuse_voxel_feature=cfg.get("fuse_voxel_feature", "fuse_voxel" in arch),
            **kwargs,
        )
    elif arch == "shapeformer":
        SHAPEFORMER_DEFAULT_KWARGS["representer_opt"]["vqdif_opt"]["weights_path"] = cfg.get("vqdif_weights_path")
        model = ShapeFormer(**SHAPEFORMER_DEFAULT_KWARGS)
    elif arch == "completr":
        encoder_type = cfg.get("encoder", "unetxd")
        encoder_kwargs = cfg.get("encoder_kwargs", {})
        encoder_kwargs["points"] = cfg.get("points_type", "self_attn")
        encoder_kwargs["planes"] = cfg.get("planes", ("xy", "xz", "yz"))
        encoder_kwargs["fuse_points"] = cfg.get("fuse_points", True)

        decoder_type = cfg.get("decoder", "transformer")
        decoder_kwargs = cfg.get("decoder_kwargs", {})
        decoder_kwargs["self_attn"] = cfg.get("self_attn", False)
        decoder_kwargs["cross_attn"] = cfg.get("cross_attn", True)
        decoder_kwargs["hidden_layer_multiplier"] = cfg.get("mlp_factor", 1)
        decoder_kwargs["n_layer"] = cfg.get("n_layer", 5)
        decoder_kwargs["n_head"] = cfg.get("n_head", 4)
        decoder_kwargs["dropout"] = cfg.get("dropout", 0)
        decoder_kwargs["bias"] = cfg.get("bias", False)
        if "hidden_dim" in cfg:
            decoder_kwargs["hidden_dim"] = cfg["hidden_dim"]

        model = CompleTr(
            dim=nerf_enc.get_enc_dim() if cfg.inputs.nerf else cfg.inputs.dim,
            padding=cfg.norm.padding,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
            multires=cfg.get("multires", True),
            custom_grid_sampling=cfg.vis.refinement_steps,
        )
    elif "point_tr" in arch:
        model = PointTransformer(
            dim=cfg.inputs.dim,
            n_embd=cfg.get("n_embd", 512),
            n_layer=cfg.get("n_layer", 8),
            n_head=cfg.get("n_head", 8),
            dropout=cfg.get("dropout", 0),
            bias=cfg.get("bias", False),
            enc_type=cfg.get("enc_type", "enc"),
            dec_type=cfg.get("dec_type", "dec"),
            pcd_enc_type=cfg.get("pcd_enc_type", "nerf" if cfg.inputs.nerf else "linear"),
            shared_pcd_enc=cfg.get("shared_pcd_enc", False),
            pos_enc_type=cfg.get("pos_enc_type"),
            shared_pos_enc=cfg.get("shared_pos_enc", True),
            implementation=cfg.get("implementation", "karpathy"),
            padding=cfg.norm.padding,
            occupancy=cfg.cls.occupancy,
            aux_loss=cfg.get("aux_loss", False),
            supervise_inputs=cfg.get("supervise_inputs"),
            cls_num_classes=cfg.cls.num_classes,
            seg_num_classes=cfg.seg.num_classes,
            use_linear_attn=cfg.get("use_linear_attn", False),
            n_fps=cfg.inputs.fps.num_points,
        )
    elif arch == "pointnet":
        assert not cfg.cls.occupancy, "PointNet does not support occupancy classification."
        if cfg.cls.num_classes:
            model = PointNetCls(dim=nerf_enc.get_enc_dim() if cfg.inputs.nerf else 3, num_classes=cfg.cls.num_classes)
        elif cfg.seg.num_classes:
            model = PointNetSeg(dim=nerf_enc.get_enc_dim() if cfg.inputs.nerf else 3, num_classes=cfg.seg.num_classes)
        else:
            raise ValueError("Either 'cls.num_classes' or 'seg.num_classes' must be set for PointNet.")
    elif arch == "pifu":
        model = PIFu()
    elif arch == "diffusers" or arch == "unet":
        model_cls = DiffusersModel
        if arch == "unet":
            model_cls = partial(UNetModel, pos_embd=cfg.get("pos_embd"), supervise_inputs=cfg.get("supervise_inputs"))
        model = model_cls(
            scheduler=cfg.get("scheduler", "ddpm"),
            num_train_timesteps=cfg.get("num_train_timesteps", 1000),
            num_inference_steps=cfg.get("num_inference_steps", 100),
            num_eval_steps=cfg.get("num_eval_steps", 1),
            beta_schedule=cfg.get("beta_schedule", "linear"),
            clip_sample=cfg.get("clip_sample", False),
            truncate_sample=cfg.get("truncate_sample", False),
            prediction_type=cfg.get("prediction_type", "epsilon"),
            timestep_spacing=cfg.get("timestep_spacing", "leading"),
            resolution=cfg.points.voxelize or 32,
            loss=cfg.train.loss,
            self_condition=cfg.get("self_condition", False),
            self_cond_on_prev_step=cfg.get("self_cond_prev", False),
            self_cond_grad=cfg.get("self_cond_grad", False),
            stop_steps=cfg.get("stop_steps", 0),
            threshold=cfg.implicit.threshold,
            zero_snr=cfg.get("zero_snr", False),
            min_snr_gamma=cfg.get("min_snr_gamma"),
            use_attention=cfg.model.attn or True,
            rescale_skip=cfg.get("rescale_skip", False),
            use_weight_standardization=cfg.model.use_ws or False,
            use_rms_norm=False if cfg.model.norm is None else "rms" in cfg.model.norm,
            downsample=cfg.model.down or "spd",
            upsample=cfg.model.up or "nearest",
            use_xdconv=cfg.get("xdconv", False),
            points=cfg.get("points_type", "single"),
            grid=cfg.get("grid_type", "single"),
            voxel=cfg.points.voxelize,
            dropout=cfg.model.dropout,
        )
    elif "3dshape2vecset" in arch:
        model_cls = Shape3D2VecSet
        if "vae" in arch:
            model_cls = Shape3D2VecSetVQVAE if "vqvae" in arch else Shape3D2VecSetVAE
            n_latent = (
                int(cfg.log.name.split("@")[-1].split("_")[0])
                if "vae@" in cfg.log.name
                else 512
                if "vqvae" in arch
                else 32
            )
            model_cls = partial(model_cls, n_latent=cfg.get("n_latent", n_latent))
            if "vqvae" in arch:
                model_cls = partial(
                    model_cls,
                    n_code=cfg.get("n_code", 16384),
                    n_enc_layer=cfg.get("n_enc_layer", 0),
                    n_feat_layer=cfg.get("n_feat_layer"),
                    quantizer=cfg.get("quantizer", "vq"),
                    learnable_codebook=cfg.get("learnable_codebook", False),
                    vq_loss_weight=cfg.get("vq_loss_weight", 1.0),
                    decay=cfg.get("decay", 0.8),
                    kmeans_init=cfg.get("kmeans_init", True),
                    use_cosine_sim=cfg.get("use_cosine_sim", False),
                    threshold_ema_dead_code=cfg.get("threshold_ema_dead_code", 0),
                    entropy_loss_weight=cfg.get("entropy_loss_weight"),
                    commitment_loss_weight=cfg.get("commitment_loss_weight"),
                    use_cross_entropy=cfg.get("use_cross_entropy", False),
                    sample_codes=cfg.get("sample_codes", False),
                    n_quantizer=cfg.get("n_quantizer", 1),
                    shared_codebook=cfg.get("shared_codebook", False),
                    quantize_soft=cfg.get("quantize_soft", False),
                )
            elif "vae" in arch:
                model_cls = partial(
                    model_cls, n_feat_layer=cfg.get("n_feat_layer"), kl_loss_weight=cfg.get("kl_loss_weight", 1e-2)
                )
        elif "cls" in arch:
            model_cls = partial(Shape3D2VecSetCls, n_classes=cfg.cls.num_classes or 57)
        suffix = ""
        if weights_path:
            suffix = "_vae" if "vae" in arch else "_cond"
        model = model_cls(
            n_layer=cfg.get(f"n_layer{suffix}", 24),
            n_embd=cfg.get(f"n_embd{suffix}", 512),
            n_head=cfg.get(f"n_head{suffix}", cfg.get(f"n_embd{suffix}", 512) // 64),
            n_queries=cfg.inputs.fps.num_points or 512,
            activation=cfg.get(f"activation{suffix}", "geglu"),
            learnable_queries=cfg.get("learnable_queries", False),
            padding=cfg.norm.padding,
            nerf_freqs=cfg.get(f"nerf_freqs{suffix}", 6),
        )
    elif arch in ["ldm", "latent_diffusion"]:
        vae = cast(
            VAEModel | VQVAEModel,
            get_model(cfg, arch=cfg.get("vae_arch", "3dshape2vecset_vae"), weights_path=cfg["vae_weights"], **kwargs),
        )
        conditioner = None
        if cfg.get("cond_key") and cfg["cond_key"] != "category.index":
            conditioner = cast(
                Model,
                get_model(
                    cfg, arch=cfg.get("cond_arch", "3dshape2vecset"), weights_path=cfg.get("cond_weights"), **kwargs
                ),
            )

        strict = False
        ldm_arch = cfg.get("ldm_arch", "transformer")
        if ldm_arch == "precond":
            model = EDMPrecond(vae=cast(VAEModel, vae))
        else:
            if ldm_arch == "transformer":
                n_latent = int(cast(Any, vae.n_latent))
                if isinstance(vae, VQVAEModel) and cfg.get("bit_diffusion", False):
                    n_code = int(cast(Any, vae.n_code))
                    n_latent = n_code if cfg.train.loss == "ce" else int(np.ceil(np.log2(n_code)))
                    if isinstance(vae.vq, ResidualVQ):
                        num_quantizers = int(vae.vq.num_quantizers or 1)
                        n_latent = num_quantizers * int(np.ceil(np.log2(n_code // num_quantizers)))
                n_latent = int(n_latent)
                denoise_fn = EDMTransformer(
                    n_latent=n_latent,
                    n_layer=cfg.get("n_layer", 24),
                    n_embd=cfg.get("n_embd", 512),
                    n_head=cfg.get("n_head", cfg.get("n_embd", 512) // 64),
                    n_classes=cfg.cls.num_classes,
                    bias=cfg.model.bias,
                    drop_path=cfg.get("drop_path", 0.0),
                    no_self_attn=cfg.get("no_self_attn", False),
                    activation=cfg.get("activation", "geglu"),
                    sigma_data=cfg.get("sigma_data", 1.0),
                )
            else:
                raise NotImplementedError(f"Latent Diffusion model architecture '{ldm_arch}' not implemented.")
            condition_key = cfg.get("cond_key", "category.index" if cfg.cls.num_classes else None)
            model = LatentDiffusionModel(
                vae=vae,
                denoise_fn=denoise_fn,
                conditioner=conditioner,
                vae_freeze=cfg.get("vae_freeze", True),
                conditioner_freeze=cfg.get("cond_freeze", True),
                condition_key=condition_key,
                bit_diffusion=cfg.get("bit_diffusion", False),
                loss_type=cfg.train.loss or "mse",
                pre_loss_fn=cfg.get("pre_loss_fn"),
                use_stats=cfg.get("use_stats", True),
                requantize=cfg.get("requantize", False),
                pos_enc=cfg.get("pos_enc", False),
            )
    elif arch in ["larm", "latent_ar_model", "latent_autoregressive_model"]:
        vae = cast(
            VAEModel | VQVAEModel,
            get_model(
                cfg, arch=cfg.get("vae_arch", "3dshape2vecset_vqvae"), weights_path=cfg.get("vae_weights"), **kwargs
            ),
        )
        conditioner = None
        if cfg.get("cond_key") and cfg["cond_key"] != "category.index":
            conditioner = cast(
                Model,
                get_model(
                    cfg, arch=cfg.get("cond_arch", "3dshape2vecset"), weights_path=cfg.get("cond_weights"), **kwargs
                ),
            )

        is_vae = isinstance(vae, VAEModel)
        strict = False
        ar_arch = cfg.get("ar_arch", "transformer")
        if ar_arch == "transformer":
            from_latents = cfg.get("from_latents", True)
            voc_enc = cast(
                bool | Callable[..., Any],
                False if is_vae else vae.vq.get_codes_from_indices if from_latents else True,
            )
            autoregressor = LatentGPT(
                n_vocab=int(2 * int(cast(Any, vae.n_latent)) if is_vae else int(cast(Any, vae.n_code))),
                n_block=int(cast(Any, vae.n_queries)),
                n_latent=int(cast(Any, vae.n_latent)) if from_latents else None,
                n_layer=cfg.get("n_layer", 24),
                n_embd=cfg.get("n_embd", 512),
                n_head=cfg.get("n_head", cfg.get("n_embd", 512) // 64),
                cond=cfg.get("cond", cfg.cls.num_classes or "cond_key" in cfg),
                drop_path=cfg.get("drop_path", 0.0),
                activation=cfg.get("activation", "geglu"),
                pos_enc=cfg.get("pos_enc"),
                voc_enc=voc_enc,
                bos_embd=cfg.get("bos_embd", True),
            )
        else:
            raise NotImplementedError(f"Latent Autoregressive model architecture '{ar_arch}' not implemented.")
        condition_key = cfg.get("cond_key", "category.index" if cfg.cls.num_classes else None)
        model = LatentAutoregressiveModel(
            discretizer=vae,
            autoregressor=autoregressor,
            conditioner=conditioner,
            discretizer_freeze=cfg.get("vae_freeze", True),
            conditioner_freeze=cfg.get("cond_freeze", True),
            condition_key=condition_key,
            loss_type=cfg.train.loss or ("nll" if is_vae else "ce"),
            objective=cfg.get("objective", "causal"),
        )
    elif arch in ["pvd", "pcd_diffusion", "point_diffusion"]:
        pvd_model_cls = cast(Any, PVDModel)
        model = pvd_model_cls()
    elif arch == "grid_diffusion":
        model = GridDiffusionModel(
            ndim=cfg.get("ndim", 3),
            channels=cfg.get("channels", 1),
            resolution=cfg.get("resolution", cfg.points.voxelize),
            rescale_skip=cfg.get("rescale_skip", True),
            use_rms_norm=False if cfg.model.norm is None else "rms" in cfg.model.norm,
            dropout=cfg.model.dropout,
        )
    elif arch == "mask_rcnn":
        return MaskRCNN()
    elif "dino_inst" in arch:
        if cfg.get("load_3d") and cfg.get("collate_3d") != "stack":
            model = DinoInst3D(
                num_objs=cfg.get("n_objs", 100),
                head=cfg.get("head", "grid+linear"),
                freeze=cfg.get("freeze", False),
                backbone=cfg.get("backbone", "dinov2_vits14"),
                bias=cfg.model.bias,
                dropout=cfg.model.dropout,
                n_dec_layers=cfg.get("n_dec"),
                mlp_heads=cfg.get("mlp_heads", False),
                match_cls=cfg.get("match_cls", True),
                match_3d=cfg.get("match_3d", True),
                separate_match_3d=cfg.get("separate_match_3d", False),
                pred_2d=cfg.get("pred_2d", True),
                sample=cfg.get("sample", False),
                nerf_freqs=cfg.get("nerf_freqs", 6),
                mask_weight=cfg.get("mask_weight", 1.0),
                occ_weight=cfg.get("occ_weight", 1.0),
                cls_weight=cfg.get("cls_weight", 1.0),
                aux_weight=cfg.get("aux_weight", 1.0),
                loss_weight_2d=cfg.get("weight_2d", 1.0),
                loss_weight_3d=cfg.get("weight_3d", 1.0),
                learn_loss_weights=cfg.get("learn_weights", False),
            )
        else:
            if cfg.inputs.type in ["depth", "kinect", "kinect_sim"] and cfg.inputs.project:
                model = DinoInstSeg3D(
                    dim=cfg.inputs.dim,
                    num_objs=cfg.get("n_objs", 100),
                    num_queries=cfg.get("n_queries", 1024),
                    num_enc_layers=cfg.get("l_enc", 1),
                    num_query_layers=cfg.get("l_query"),
                    backbone=cfg.get("backbone", "dinov2_vits14"),
                    bias=cfg.model.bias,
                    dropout=cfg.model.dropout,
                    drop_path=cfg.get("drop_path", 0.1),
                    mlp_ratio=cfg.get("mlp_ratio", 4),
                    cls_token=cfg.get("cls_token", False),
                    cat_feat=cfg.get("cat_feat", True),
                    cat_point_feat=cfg.get("cat_point_feat", False),
                    points_dec=cfg.get("point_dec"),
                    pred_cls=cfg.get("pred_cls", "quality+objectness"),
                    detach_cls=cfg.get("detach_cls", False),
                    queries_from_feat=cfg.get("queries_from_feat", "detach"),
                    cos_sim=cfg.get("cos_sim", False),
                    logit_scale=cfg.get("logit_scale", False),
                    embd_lvls=cfg.get("embd_lvls", False),
                    mlp_heads=cfg.get("mlp_heads", False),
                    match_cls=cfg.get("match_cls", True),
                    anneal_cls=cfg.get("anneal_cls"),
                    pad_targets=cfg.get("pad_targets", False),
                    sample=cfg.get("sample"),
                    loss_name=cfg.train.loss or "dice+bce",
                    mask_weight=cfg.get("mask_weight", 1.0),
                    mask_pos_weight=cfg.get("mask_pos_weight"),
                    bce_focal_weight=cfg.get("bce_focal_weight", 5.0),
                    dice_weight=cfg.get("dice_weight", 2.0),
                    cls_weight=cfg.get("cls_weight", 0.5),
                    cls_pos_weight=cfg.get("cls_pos_weight"),
                    aux_weight=cfg.get("aux_weight"),
                    init_weights=cfg.get("init_weights", False),
                    learn_loss_weights=cfg.get("learn_weights", False),
                    multitask=cfg.get("multitask", False),
                    inputs_weight=cfg.get("inputs_weight", 1.0),
                    points_weight=cfg.get("points_weight", 1.0),
                    focal_alpha=cfg.get("focal_alpha", 0.25),
                    focal_gamma=cfg.get("focal_gamma", 2.0),
                    cls_threshold=cfg.get("cls_threshold", 0.5),
                    min_mask_size=cfg.get("min_mask_size", 64),
                    apply_filter=cfg.get("apply_filter", True),
                    nerf_enc=cfg.get("nerf_enc", "tcnn" if TCNN_EXISTS else "torch"),
                    nerf_freqs=cfg.get("nerf_freqs", 6),
                )
            elif cfg.inputs.type == "rgbd":
                model = DinoInstSegRGBD(
                    num_objs=cfg.get("n_objs", 100),
                    backbone=cfg.get("backbone", "dinov2_vits14"),
                    dropout=cfg.model.dropout,
                    bias=cfg.model.bias,
                )
            else:
                dino_inst_mode = cast(Any, arch.replace("dino_inst", "").strip("_") or "conv")
                model = DinoInstSeg(
                    num_objs=cfg.get("n_objs", 100),
                    mode=dino_inst_mode,
                    freeze=cfg.get("freeze", False),
                    backbone=cfg.get("backbone", "dinov2_vits14"),
                    dropout=cfg.model.dropout,
                    bias=cfg.model.bias,
                    pred_det=cfg.get("pred_det"),
                    queries_from_feat=cfg.get("queries_from_feat", "cls"),
                    detach_queries=cfg.get("detach_queries", True),
                    straight_through=cfg.get("straight_through", False),
                    combine_feat=cfg.get("combine_feat"),
                    pred_cls=cfg.get("pred_cls", "objectness"),
                    num_dec_layers=cfg.get("n_dec"),
                    mlp_heads=cfg.get("mlp_heads", False),
                    aux_depth_loss=cfg.get("depth_loss", False),
                    match_cls=cfg.get("match_cls", True),
                    match_det=cfg.get("match_det", True),
                    global_match=cfg.get("global_match", False),
                    pad_targets=cfg.get("pad_targets", False),
                    sample=cfg.get("sample"),
                    loss_name=cfg.train.loss or "dice+bce",
                    focal_alpha=cfg.get("focal_alpha", 0.5),
                    focal_gamma=cfg.get("focal_gamma", 2.0),
                    mask_weight=cfg.get("mask_weight", 1.0),
                    cls_weight=cfg.get("cls_weight", 1.0),
                    det_weight=cfg.get("det_weight", 1.0),
                    depth_weight=cfg.get("depth_weight", 1.0),
                    aux_weight=cfg.get("aux_weight", "auto"),
                    learn_loss_weights=cfg.get("learn_weights", False),
                )
    elif "dino" in arch:
        backbone = "dinov2_vits14" if arch == "dino" else arch
        second_order = cfg.get("normal_loss", cfg.vis.refinement_steps) or not TCNN_EXISTS
        if cfg.inputs.type in ["image", "rgb", "color", "shading", "normals"] or not cfg.inputs.project:
            model = DinoRGB(
                padding=cfg.norm.padding,
                freeze=cfg.get("freeze", False),
                backbone=backbone,
                condition=cfg.get("condition", "all"),
                head=cfg.get("head", "mlp"),
                mask=cfg.get("mask"),
                bias=cfg.model.bias,
                init=cfg.get("init", not cfg.implicit.dvr),
                nerf_enc=cfg.get("nerf_enc", "torch" if second_order else "tcnn"),
                nerf_freqs=cfg.get("nerf_freqs", 6),
            )
        else:
            if cfg.inputs.type == "rgbd":
                model = DinoRGBD(
                    n_queries=cfg.get("n_queries", "auto"),
                    inputs_enc=cfg.get("inputs_enc", "both"),
                    patches_enc=cfg.get("patches_enc", "both"),
                    condition=cfg.get("condition", "all"),
                    mask=cfg.get("mask"),
                    padding=cfg.norm.padding,
                    freeze=cfg.get("freeze", False),
                    backbone=backbone,
                    bias=cfg.get("bias", False),
                )
            else:
                model = Dino3D(
                    num_queries=cfg.get("n_queries", 256),
                    backbone=backbone,
                    cls_token=cfg.get("cls_token", False),
                    cat_feat=cfg.get("cat_feat", True),
                    num_classes=cfg.cls.num_classes,
                    bias=cfg.model.bias,
                    dropout=cfg.model.dropout,
                    drop_path=cfg.get("drop_path", 0.1),
                    init_weights=cfg.get("init_weights", True),
                    loss_name=cfg.train.loss or "dice+bce",
                    bce_weight=cfg.get("bce_weight", 5.0),
                    dice_focal_weight=cfg.get("dice_focal_weight", 2.0),
                    focal_alpha=cfg.get("focal_alpha", 0.25),
                    focal_gamma=cfg.get("focal_gamma", 2.0),
                    nerf_enc=cfg.get("nerf_enc", "torch" if second_order else "tcnn"),
                    nerf_freqs=cfg.get("nerf_freqs", 6),
                    sample=cfg.get("sample"),
                    learn_loss_weights=cfg.get("learn_weights", False),
                )
    elif arch == "idr":
        model = ImplicitNetwork(
            feature_vector_size=256, geometric_init=False, weight_norm=True, multires=4, n_objs=len(cfg.data.categories)
        )
    elif "inst" in arch and "pipe" in arch:
        cfg.model.load_best = False
        inst_weights = cfg.get("inst_weights")
        inst = cast(DinoInstSeg3D, get_model(cfg, arch="dino_inst", weights_path=inst_weights, **kwargs))
        occ = cast(Dino3D, get_model(cfg, arch="dino", weights_path=cfg.get("occ_weights"), **kwargs))
        return InstOccPipeline(inst, occ, masks=cfg.get("masks", "inst" if inst_weights else "gt"))
    else:
        raise ValueError(f"Invalid model architecture '{arch}'")

    if not isinstance(model, Model):
        raise TypeError(f"Model '{arch}' must be an instance of 'Model'.")
    model = cast(Model, model)

    if hasattr(model, "reduction"):
        model.reduction = cfg.model.reduction

    if cfg.inputs.nerf and not isinstance(
        model,
        (
            PointTransformer,
            Shape3D2VecSet,
            Shape3D2VecSetCls,
            Shape3D2VecSetVAE,
            Shape3D2VecSetVQVAE,
            LatentDiffusionModel,
            EDMPrecond,
            LatentAutoregressiveModel,
            Dino3D,
            DinoRGBD,
        ),
    ):
        assert hasattr(model, "encode"), f"Model '{model.name}' does not support monkey patching of NeRF encoding."
        old_encode = cast(Callable[..., Tensor], model.encode)

        def new_encode(self, inputs: Tensor, **kwargs):
            nerf_enc.to(inputs.device)
            return old_encode(nerf_enc(inputs), **kwargs)

        logger.debug_level_1(f"Monkey patching NeRF encoding into {model.name}.{old_encode.__name__}")
        monkey_patch(model, old_encode.__name__, new_encode)

    if cfg.points.nerf and not isinstance(
        model,
        (
            PointTransformer,
            Shape3D2VecSet,
            Shape3D2VecSetCls,
            Shape3D2VecSetVAE,
            Shape3D2VecSetVQVAE,
            LatentDiffusionModel,
            EDMPrecond,
            LatentAutoregressiveModel,
            Dino3D,
            DinoRGB,
            DinoRGBD,
        ),
    ):
        assert hasattr(model, "decode"), f"Model '{model.name}' does not support monkey patching of NeRF encoding."
        old_decode = cast(Callable[..., Tensor], model.decode)

        def new_decode(self, points: Tensor, feature: Tensor, **kwargs) -> Tensor:
            nerf_enc.to(points.device)
            return old_decode(nerf_enc(points), feature, **kwargs)

        logger.debug_level_1(f"Monkey patching NeRF encoding into {model.name}.{old_decode.__name__}")
        monkey_patch(model, old_decode.__name__, new_decode)

    model = patch_attention(model, mode=cfg.model.attn_mode, backend=cfg.model.attn_backend)

    weights = None
    if load_weights:
        if weights_path:
            path = resolve_weights_path(cfg, weights_path)
            if path is None:
                raise FileNotFoundError(f"Could not resolve weights path '{weights_path}'.")
            logger.info(f"Loading weights from {path}")
            weights = torch.load(path, map_location="cpu", weights_only=False)
            cfg.model.weights = str(path)
        elif cfg.model.load_best or cfg.train.skip:
            name = f"model_{'ema' if cfg.model.average == 'ema' else 'best'}.pt"
            try:
                path = resolve_weights_path(cfg, name)
                if path is None:
                    raise FileNotFoundError(f"Could not resolve weights path '{name}'.")
                logger.info(f"Loading {'EMA' if cfg.model.average == 'ema' else 'best'} weights from {path}")
                weights = torch.load(path, map_location="cpu", weights_only=False)
                cfg.model.weights = str(path)
            except FileNotFoundError:
                cfg.model.checkpoint = "last"
                logger.warning(
                    f"Could not find {'EMA' if cfg.model.average == 'ema' else 'best'} weights. "
                    f"Trying to load last checkpoint instead."
                )

    if cfg.model.checkpoint and weights is None:
        path = resolve_checkpoint_path(cfg)
        if path is None:
            raise FileNotFoundError("Could not resolve checkpoint path from config.")
        logger.info(f"Loading checkpoint from {path}")
        weights = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        cfg.model.checkpoint = str(path)

    if weights is not None:
        model.load_state_dict(weights, strict=strict)
    model.setup()

    if cfg.implicit.dvr:
        model = DVR(
            model,
            RayMarchingConfig(
                near=cfg.implicit.near,
                far=cfg.implicit.far,
                step_func=cfg.implicit.step_func,
                num_steps=cfg.implicit.num_steps,
                num_pixels=cfg.implicit.num_pixels,
                num_points=cfg.vis.num_query_points,
                num_views=cfg.train.num_views,
                max_batch_size=cfg.implicit.max_batch_size,
                threshold=cfg.implicit.threshold,
                refine_mode=cfg.implicit.refine_mode,
                num_refine_steps=cfg.implicit.num_refine_steps,
                crop=cfg.implicit.crop,
                padding=cfg.norm.padding,
                debug=cfg.log.verbose > 2,
            ),
            depth_loss=cfg.get("depth_loss"),
            rgb_loss=cfg.get("rgb_loss", "l1"),
            normal_loss=cfg.get("normal_loss"),
            sym_loss=cfg.get("sym_loss"),
            learn_loss_weights=cfg.get("loss_weights"),
            focal_loss_alpha=cfg.get("focal_loss"),
            feature_consistency_loss=cfg.get("feature_consistency", True),
            geometry_consistency_loss=cfg.get("geometry_consistency", True),
            p_universal=cfg.get("p_universal", False),
        )

    return model
