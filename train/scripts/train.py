import os
from functools import partial
from typing import Any, cast

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from dataset import get_dataset
from models import assign_params_groups, get_model  # pyright: ignore[reportAttributeAccessIssue]
from train import get_collate_fn
from utils import (
    get_num_workers,
    log_optional_dependency_summary,
    resolve_checkpoint_path,
    resolve_path,
    resolve_save_dir,
    setup_config,
    setup_logger,
    suppress_known_optional_dependency_warnings,
)

from ..src.data_module import LitDataModule
from ..src.model import LitModel
from ..src.schedulers import LinearWarmupCosineAnnealingLR
from ..src.utils import get_test_dataset, save_best_model, save_ema_model

logger = setup_logger(__name__)


@rank_zero_only
def update_wandb_experiment_config(wandb_logger: WandbLogger, cfg: DictConfig):
    wandb_logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), allow_val_change=True
    )


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run(cfg)


def run(cfg: DictConfig) -> float:
    cfg = setup_config(cfg, seed_workers=True)
    suppress_known_optional_dependency_warnings()
    log_optional_dependency_summary(logger, cfg)

    save_dir = resolve_save_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.debug_level_1(f"Saving to {save_dir}")
    project = cfg.log.project or HydraConfig.get().job.config_name
    name = cfg.log.name or cfg.model.arch

    test_dataset = None
    load_test_split = False
    if cfg.test.run:
        if cfg.test.dir is not None and cfg.test.filename is not None:
            test_dataset = get_test_dataset(cfg)
        elif cfg.test.split == "test":
            load_test_split = True

    dataset = get_dataset(cfg, splits=("train", "val", "test") if load_test_split else ("train", "val"))

    num_workers = get_num_workers(cfg.load.num_workers)
    datamodule = LitDataModule(
        train=dataset["train"],
        val=dataset["val"],
        test=dataset["test"] if load_test_split else test_dataset,
        batch_size=cfg.train.batch_size,
        batch_size_val=cfg.val.batch_size,
        num_workers=num_workers,
        num_workers_val=num_workers,
        # shuffle_val=cfg.val.visualize or cfg.val.mesh,
        overfit=cfg.train.overfit_batches,
        prefetch_factor=cfg.load.prefetch_factor,
        pin_memory=cfg.load.pin_memory,
        weighted=cfg.load.weighted,
        seed=cfg.misc.seed,
        collate_fn=get_collate_fn(cfg),
        cache=cfg.data.cache,
        hash_items=cfg.data.hash_items or cfg.data.num_files.train,
        share_memory=cfg.data.share_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cast(Any, get_model(cfg)).to(device)
    scheduler = None
    monitor = cfg.train.model_selection_metric
    mode = "min" if "loss" in monitor else "max"
    patience = max(5, min(int(np.ceil(cfg.train.patience_factor * cfg.val.freq)), cfg.train.epochs // 2))
    has_key = hasattr(model, "condition_key")
    condition_key = cast(Any, getattr(model, "condition_key", None))
    no_cond = has_key and (not condition_key or "category" in str(condition_key))
    vis_inputs = cfg.vis.inputs and cfg.inputs.type and not (has_key and no_cond)
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    step_scale = cfg.train.batch_size * cfg.train.accumulate_grad_batches * torch.cuda.device_count() * num_nodes
    steps = (len(cast(Any, dataset["train"])) * cfg.train.epochs) // step_scale
    logger.debug_level_1(f"Training for {steps} steps")
    clip_16bit = "16" in cfg.train.precision and cfg.train.gradient_clip_val
    torch_cuda = int(torch.__version__[0]) >= 2 and torch.cuda.is_available()
    fused = cfg.get("fused", torch_cuda and not clip_16bit)

    optimizer: Any = None
    if cfg.train.hypergradients:
        try:
            from gradient_descent_the_ultimate_optimizer import gdtuo

            optimizer = gdtuo.Adam(optimizer=cast(Any, gdtuo.SGD(1e-5)))
            model = gdtuo.ModuleWrapper(model, optimizer=cast(Any, optimizer))
            model.initialize()
        except ImportError:
            logger.warning(
                "Could not import 'gradient_descent_the_ultimate_optimizer'. Using PyTorch optimizer instead."
            )
            cfg.train.hypergradients = False

    if optimizer is None:
        lr = cfg.train.lr
        min_lr = cfg.train.min_lr or lr / 10
        if cfg.train.scale_lr:
            lr *= step_scale
            min_lr *= step_scale
            logger.debug_level_1(f"Using learning rate {lr}")
        if hasattr(model, "optimizer") and model.optimizer is not None:
            optimizer = model.optimizer(
                lr=lr,
                weight_decay=cfg.train.weight_decay or 0,
                factor=cfg.get("lr_factor", 0.1),
                fused=fused,
                foreach=not fused,
            )
        if optimizer is None:
            params = model.parameters()
            if cfg.train.weight_decay:
                logger.debug_level_1(f"Using weight decay {cfg.train.weight_decay}")
                params_decay, params_no_decay = cast(Any, assign_params_groups)(model)
                params = [
                    {"params": params_decay, "weight_decay": cfg.train.weight_decay},
                    {"params": params_no_decay, "weight_decay": 0.0},
                ]

            if "8bit" in cfg.train.optimizer:
                try:
                    import bitsandbytes as bnb

                    optimizer = getattr(bnb.optim, cfg.train.optimizer)
                except ImportError:
                    logger.warning("Could not import 'bitsandbytes'. Using PyTorch optimizer instead.")
                    cfg.train.optimizer = cfg.train.optimizer.replace("8bit", "")

            if optimizer is None:
                optimizer = getattr(torch.optim, cfg.train.optimizer)
                optimizer = partial(optimizer, fused=fused, foreach=not fused)
            optimizer = optimizer(params, lr=lr, betas=tuple(cfg.train.betas))
            # eps=1e-7 if "16" in cfg.train.precision else 1e-8)
        logger.debug_level_1(f"Using '{optimizer.__class__.__name__}' optimizer")
        if cfg.train.scheduler in ["ReduceLROnPlateau", "StepLR", "LinearWarmupCosineAnnealingLR"]:
            if cfg.train.scheduler == "ReduceLROnPlateau" and patience < cfg.train.epochs:
                if cfg.val.freq > 1:
                    logger.warning("ReduceLROnPlateau scheduler is not supported with validation frequency > 1. ")
                else:
                    factor = cfg.train.lr_reduction_factor
                    logger.debug_level_1(
                        f"Learning rate will be reduced by {factor} if '{monitor}' does not "
                        f"{'increase' if mode == 'max' else 'decrease'} for {patience} epochs"
                    )
                    scheduler = ReduceLROnPlateau(
                        optimizer, mode=mode, factor=factor, patience=patience, threshold=1e-4, min_lr=min_lr
                    )
            elif cfg.train.scheduler == "StepLR":
                step_size = cfg.train.lr_step_size
                gamma = cfg.train.lr_gamma
                logger.debug_level_1(f"Learning rate will be reduced by {gamma} every {step_size} epochs")
                scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif cfg.train.scheduler == "LinearWarmupCosineAnnealingLR":
                warmup_steps = int(steps * cfg.train.warmup_frac)
                warmup_epochs = (warmup_steps * step_scale) // len(cast(Any, dataset["train"]))
                logger.debug_level_1(
                    f"LR warmup for {warmup_steps} steps/{warmup_epochs} epochs, then decay to {min_lr}"
                )
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer, warmup_iters=warmup_steps, total_iters=steps, min_lr=min_lr
                )

                # warmup_scheduler = LinearLR(optimizer, start_factor=min_lr / lr, total_iters=warmup_steps)
                # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=steps - warmup_steps, eta_min=min_lr)
                # scheduler = SequentialLR(optimizer,
                #                         schedulers=[warmup_scheduler, cosine_scheduler],
                #                         milestones=[warmup_steps])
            else:
                raise NotImplementedError(f"Scheduler '{cfg.train.scheduler}' not yet implemented")

    if cfg.model.compile:
        logger.debug_level_1("Compiling model")
        model = torch.compile(cast(Any, model), mode="max-autotune")

    ema_decay = None
    if cfg.model.average == "ema":
        round_to = int(np.pow(10, int(np.log10(steps))))
        steps_rounded = round_to if steps < 5 * round_to else 10 * round_to
        ema_decay = cfg.model.ema_decay or np.clip(1 - 100 / steps_rounded, 0.9, 0.9999)
        logger.debug_level_1(f"Using EMA with decay {ema_decay}")

    model = LitModel(
        name=cfg.model.arch,
        output_dir=save_dir,
        model=cast(Any, model),
        optimizer=optimizer,
        scheduler=scheduler,
        interval="step" if cfg.train.scheduler == "LinearWarmupCosineAnnealingLR" else "epoch",
        hypergradients=cfg.train.hypergradients,
        monitor=monitor,
        metrics=cfg.log.metrics,
        threshold=cfg.implicit.threshold,
        regression=cfg.implicit.sdf,
        loss=cfg.train.loss,
        reduction=cfg.train.reduction,
        points_batch_size=cfg.vis.num_query_points,
        sync_dist=num_nodes * torch.cuda.device_count() > 1,
        ema=ema_decay,
    )

    filename = "epoch={epoch}-step={step}-loss={train/loss:.2f}"
    if "val" in monitor:
        filename = "epoch={epoch}-step={step}-val_loss={val/loss:.2f}"
        if monitor != "val/loss":
            filename += f"-{monitor.replace('/', '_').lower()}=" + "{" + monitor + ":.2f}"
    callbacks = [
        ModelCheckpoint(
            filename=filename,
            monitor=monitor,
            verbose=cfg.log.verbose > 1,
            save_last=True,
            save_top_k=cfg.log.top_k,
            mode=mode,
            auto_insert_metric_name=False,
            every_n_epochs=cfg.train.epochs // 5 if cfg.log.top_k == -1 else None,
        ),
        RichModelSummary(max_depth=cfg.log.summary_depth + (1 if cfg.model.compile else 0) + cfg.log.verbose),
    ]
    if cfg.train.scheduler:
        callbacks.append(LearningRateMonitor())
    if cfg.log.progress == "rich":
        callbacks.append(RichProgressBar())
    if cfg.train.early_stopping and 3 * patience < cfg.train.epochs:
        patience = 3 * patience if cfg.train.scheduler else patience
        logger.debug_level_1(
            f"Training will stop early if '{monitor}' does not {'increase' if mode == 'max' else 'decrease'} for {patience} epochs"
        )
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=int(np.ceil(patience / cfg.val.freq)),
                verbose=cfg.log.verbose > 1,
                log_rank_zero_only=True,
            )
        )

    if cfg.model.average == "ema":
        try:
            # callbacks.append(EMACallback(decay=ema_decay))
            pass
        except ImportError:
            logger.exception("Could not import EMACallback. Disabling EMA.")
            cfg.model.average = None
    elif cfg.model.average == "swa":
        swa_lrs = cfg.model.swa_lr or 100 * optimizer.defaults["lr"]
        logger.debug_level_1(f"Using Stochastic Weight Averaging (SWA) with learning rate {swa_lrs}")
        callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lrs))
    elif cfg.model.average not in [None, ""]:
        raise ValueError(f"Unknown model averaging method '{cfg.model.average}'")

    if cfg.val.visualize:
        from ..src.callbacks import VisualizeCallback

        n_samples = f"{cfg.val.vis_n_category} per category" if cfg.val.vis_n_category else cfg.val.vis_n
        logger.debug_level_1(f"Visualizing (#{n_samples}) every {cfg.val.vis_n_eval} evaluations")
        if not vis_inputs:
            logger.debug_level_2("Input visualization is disabled")
        callbacks.append(
            VisualizeCallback(
                every_n_evals=cfg.val.vis_n_eval,
                n_per_category=cfg.val.vis_n_category,
                n_total=cfg.val.vis_n,
                meshes=cfg.vis.mesh or "mesh" in cfg.vis.render,
                inputs=vis_inputs,
                render=cfg.vis.render,
                upload_to_wandb=cfg.log.wandb,
                points_batch_size=cfg.vis.num_query_points or cfg.val.num_query_points,
                threshold=cfg.implicit.threshold,
                padding=cfg.norm.padding,
                resolution=cfg.vis.resolution,
                width=256 if cfg.log.wandb else 512,
                height=256 if cfg.log.wandb else 512,
                show=cfg.vis.show,
                precision=cfg.train.precision,
                progress=bool(cfg.log.progress),
                estimate_normals=cfg.vis.normals,
                predict_colors=cfg.vis.colors,
                simplify=cfg.vis.simplify,
                refinement_steps=cfg.vis.refinement_steps,
            )
        )

    if cfg.val.mesh:
        from ..src.callbacks import EvalMeshesCallback

        stats_name = None
        if any(m in cfg.val.mesh for m in ["fid", "kid", "all"]):
            if cfg.data.categories and len(cfg.data.categories) == 1:
                stats_name = f"{cfg.data.train_ds[0]}_{cfg.data.categories[0]}_train_icosphere"
            elif not cfg.data.categories:
                stats_name = f"{cfg.data.train_ds[0]}_train_icosphere"
            else:
                logger.warning("Multiple categories found. FID calculation will be disabled.")

        logger.debug_level_1(f"Evaluating meshes (metrics: {cfg.val.mesh}) every {cfg.val.vis_n_eval} evaluations")
        callbacks.append(
            EvalMeshesCallback(
                every_n_evals=cfg.val.vis_n_eval,
                upload_to_wandb=cfg.log.wandb,
                points_batch_size=cfg.vis.num_query_points or cfg.val.num_query_points,
                threshold=cfg.implicit.threshold,
                padding=cfg.norm.padding,
                resolution=cfg.vis.resolution,
                precision=cfg.train.precision,
                progress=bool(cfg.log.progress),
                num_workers=cfg.load.num_workers,
                fid_stats_name=stats_name,
                metrics=cfg.val.mesh,
            )
        )

    if cfg.log.wandb:
        pl_logger = WandbLogger(
            name=name,
            save_dir=save_dir.parent.parent,  # 'project' is added automatically
            version=cfg.log.id or save_dir.name,
            offline=cfg.log.offline,
            project=project,
            log_model=cfg.log.model,
        )
        update_wandb_experiment_config(pl_logger, cfg)
        if cfg.log.gradients or cfg.log.parameters or cfg.log.graph:
            if cfg.log.gradients and cfg.log.parameters:
                log = "all"
            elif cfg.log.gradients:
                log = "gradients"
            elif cfg.log.parameters:
                log = "parameters"
            else:
                log = None
            log_freq_factor = 1 if cfg.log.verbose > 1 else 10 if cfg.log.verbose else 100
            pl_logger.watch(model, log=log, log_freq=log_freq_factor * cfg.log.freq, log_graph=cfg.log.graph)
    else:
        try:
            pl_logger = TensorBoardLogger(save_dir=save_dir.parent, name=name, version=cfg.log.version)
        except ImportError as e:
            logger.warning(f"TensorBoardLogger import failed: {e}")
            pl_logger = True

    self_cond = cfg.get("self_condition", False)
    self_cond_grad = cfg.get("self_cond_grad", False)
    no_freeze = not cfg.get("vae_freeze", True) or not cfg.get("cond_freeze", True)
    strategy = "ddp_find_unused_parameters_true" if (self_cond and not self_cond_grad) or no_freeze else "auto"
    trainer = pl.Trainer(
        devices="auto",  # i.e. --gres=gpu:4 --ntasks-per-node=4 in sbatch/salloc
        accelerator="auto",
        strategy=strategy,
        num_nodes=num_nodes,  # i.e. --nodes=2 in sbatch/salloc
        logger=pl_logger,
        default_root_dir=save_dir,
        callbacks=callbacks,
        enable_model_summary=False,  # Already logged using RichModelSummary
        enable_progress_bar=None if cfg.log.progress == "rich" else cfg.log.progress,
        overfit_batches=cfg.train.overfit_batches,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        gradient_clip_algorithm="norm",
        benchmark=get_collate_fn(cfg, split="train") is None,  # Disable benchmarking if custom collate_fn is used
        detect_anomaly=cfg.train.detect_anomaly,  # Detect NaN/Inf in 16-bit training
        max_epochs=cfg.train.epochs,
        limit_train_batches=cfg.train.num_batches,
        limit_val_batches=cfg.val.num_batches,
        check_val_every_n_epoch=1 if cfg.val.freq < 1 else int(cfg.val.freq),
        num_sanity_val_steps=cfg.val.num_sanity,
        fast_dev_run=cfg.train.fast_dev_run,
        val_check_interval=None if cfg.val.freq >= 1 else cfg.val.freq,
        log_every_n_steps=cfg.log.freq,
        precision=cfg.train.precision,
        profiler="simple" if cfg.log.profile or cfg.log.verbose > 1 else None,
        plugins=SLURMEnvironment() if os.environ.get("SLURM_JOB_NAME") else None,
    )

    if not cfg.train.skip:
        if cfg.train.find_batch_size or cfg.train.find_lr:
            tuner = Tuner(trainer)
            if cfg.train.find_batch_size:
                tuner.scale_batch_size(model, datamodule=datamodule)
            if cfg.train.find_lr:
                tuner.lr_find(model, datamodule=datamodule)

        checkpoint_path = resolve_checkpoint_path(cfg)
        trainer.fit(model, datamodule, ckpt_path=checkpoint_path)

        state_dict = None
        checkpoint_callback = cast(Any, trainer.checkpoint_callback)
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path is None:
            best_model_path = checkpoint_callback.last_model_path
            if best_model_path is None:
                best_model_path = checkpoint_path
        if best_model_path is not None:
            best_model_path = resolve_path(best_model_path)
            if not best_model_path.is_file():
                best_model_path = resolve_save_dir(cfg) / best_model_path.name
            if best_model_path.is_file():
                logger.debug_level_1(f"Resolved best model path to {best_model_path}")
                state_dict = save_best_model(best_model_path, save_dir)
            else:
                logger.warning("Best model path could not be resolved")
        save_ema_model(trainer, model, save_dir, state_dict)

    best_score = cast(Any, trainer.checkpoint_callback).best_model_score
    if best_score is not None:
        logger.info(f"Best {monitor}: {best_score:.3f}")

    if datamodule.test is not None:
        try:
            from ..src.callbacks import TestMeshesCallback

            callbacks.append(
                TestMeshesCallback(
                    inputs=vis_inputs,
                    upload_to_wandb=cfg.log.wandb,
                    points_batch_size=cfg.vis.num_query_points or cfg.val.num_query_points,
                    threshold=cfg.implicit.threshold,
                    padding=cfg.norm.padding,
                    resolution=cfg.vis.resolution,
                    show=cfg.vis.show,
                    precision=cfg.train.precision,
                )
            )
            test_trainer = pl.Trainer(
                devices=1,
                logger=pl_logger,
                default_root_dir=save_dir,
                callbacks=callbacks,
                enable_model_summary=False,
                enable_progress_bar=None if cfg.log.progress == "rich" else cfg.log.progress,
                precision=cfg.train.precision,
            )
            if not cfg.train.skip:
                cast(Any, test_trainer.checkpoint_callback).best_model_path = str(best_model_path)
            test_trainer.test(
                model,
                datamodule,
                ckpt_path=None if cfg.train.skip else "best" if "val" in monitor else "last",
                verbose=cfg.log.verbose > 1,
            )
        except Exception as e:
            logger.exception(f"Testing failed: {e}")

    if cfg.log.wandb:
        try:
            import wandb

            wandb.finish()
        except ImportError:
            pass

    return float(best_score) if best_score is not None else 0.0


if __name__ == "__main__":
    main()
