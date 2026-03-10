from collections.abc import Callable, Sequence
from logging import DEBUG
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import _METRIC
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from eval import eval_pointcloud
from models import PCN, PSGN, Model, PSSNet, ShapeFormer, SnowflakeNet
from utils import DEBUG_LEVEL_1, DEBUG_LEVEL_2, setup_logger, to_numpy

from .utils import weight_norm

try:
    import wandb
except ImportError:
    wandb = None

logger = setup_logger(__name__)


class LitModel(pl.LightningModule):
    def __init__(
        self,
        name: str,
        output_dir: str | Path,
        model: Model,
        learning_rate: float | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        interval: str = "epoch",
        frequency: int = 1,
        hypergradients: bool = False,
        monitor: str | None = None,
        metrics: list[str] | None = None,
        threshold: float = 0.5,
        regression: bool = False,
        loss: str | None = None,
        reduction: str | None = "mean",
        points_batch_size: int | None = None,
        sync_dist: bool = False,
        ema: float | None = None,
    ):
        super().__init__()
        assert interval in ["epoch", "step"], f"Interval {interval} not supported."
        self.name = name
        self.output_dir = output_dir
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.interval = interval
        self.frequency = frequency
        self.hypergradients = hypergradients
        self.monitor = monitor
        self.metrics = metrics
        self.threshold = threshold
        self.regression = regression
        self.loss = loss
        self.reduction = reduction
        self.points_batch_size = points_batch_size
        self.sync_dist = sync_dist
        self.ema_model = None
        if ema:
            self.ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema))
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
        self.save_hyperparameters(ignore=["model", "ema_model", "optimizer", "scheduler"])

        self._logged_metrics = set()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def do_log(self, log_factor: float = 1.0) -> bool:
        trainer = cast(Any, self.trainer)
        log_every_n_steps = max(1, int(getattr(trainer, "log_every_n_steps", 1)))
        return self.global_step % int(log_factor * log_every_n_steps) == 0

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = self.model.orig_mod.state_dict(*args, **kwargs)
        return {"model." + k: v for k, v in state_dict.items()}

    def load_state_dict(self, state_dict: dict[str, Any], *args, **kwargs):
        strict = kwargs.pop("strict", True)
        for name in ["_vae", "_discretizer", "_conditioner"]:
            module = getattr(self.model.orig_mod, name, None)
            if module is not None:
                if not module.training and not any(p.requires_grad for p in module.parameters()):
                    strict = False
                    break
        self.model.orig_mod.load_state_dict(state_dict, *args, strict=strict, **kwargs)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = cast(Any, self.ema_model.module.orig_mod).state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        ema_state = checkpoint.get("ema_state_dict", None)
        if ema_state is not None and self.ema_model is not None:
            strict = True
            for name in ["_vae", "_discretizer", "_conditioner"]:
                module = getattr(self.ema_model.module.orig_mod, name, None)
                if module is not None:
                    if not module.training and not any(p.requires_grad for p in module.parameters()):
                        strict = False
                        break
            cast(Any, self.ema_model.module.orig_mod).load_state_dict(ema_state, strict=strict)

    def forward(self, x: dict[str, Any]) -> Tensor:
        if not self.model.orig_mod.training and self.ema_model is not None:
            return self.ema_model(**x)
        return self.model(**x)

    def log(
        self,
        name: str,
        value: _METRIC,
        prog_bar: bool = False,
        logger: bool | None = None,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        reduce_fx: str | Callable = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Any | None = None,
        add_dataloader_idx: bool = True,
        batch_size: int | None = None,
        metric_attribute: str | None = None,
        rank_zero_only: bool = False,
    ) -> None:
        self._logged_metrics.add(name)
        super().log(
            name,
            value,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            metric_attribute,
            rank_zero_only,
        )

    def on_before_backward(self, loss: Tensor) -> None:
        if not torch.isfinite(loss).all():
            raise ValueError(f"Loss is not finite: {loss}")

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        kwargs["create_graph"] = self.hypergradients
        super().backward(loss, *args, **kwargs)

    def on_after_backward(self):
        if self.global_step == 0 or logger.isEnabledFor(DEBUG_LEVEL_2):
            for name, p in self.named_parameters():
                if p.requires_grad and p.grad is None:
                    logger.warning(f"Parameter {name} has no gradient")

    def on_before_optimizer_step(self, optimizer: Optimizer):
        if self.do_log(log_factor=1 if logger.isEnabledFor(DEBUG_LEVEL_2) else 10):
            if logger.isEnabledFor(DEBUG_LEVEL_1):
                gn = grad_norm(self.model, norm_type=2)
                wn = weight_norm(self.model, norm_type=2)
                if gn:
                    self.log("train/grad_norm", gn["grad_2.0_norm_total"])
                if wn:
                    self.log("train/weight_norm", wn["weight_2.0_norm_total"])
        if logger.isEnabledFor(DEBUG_LEVEL_2) and self.do_log(log_factor=10):
            grad_norms = [float(v.cpu().item()) if isinstance(v, Tensor) else float(v) for k, v in gn.items() if k != "grad_2.0_norm_total"]
            weight_norms = [
                float(v.cpu().item()) if isinstance(v, Tensor) else float(v) for k, v in wn.items() if k != "weight_2.0_norm_total"
            ]
            self.log_histogram("gradients/grad_hist", grad_norms)
            self.log_histogram("gradients/weight_hist", weight_norms)

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        if self.ema_model is not None:
            self.ema_model.update_parameters(self.model)

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        if self.hypergradients:
            cast(Any, self.model).begin()

    def training_step(self, batch: dict, batch_idx: int):
        assert self.model.orig_mod.training, "Model should be in train mode during training"
        if hasattr(self.model, "loss"):
                loss = self.model.loss(
                batch,
                regression=self.regression,
                name=self.loss,
                reduction=self.reduction,
                points_batch_size=self.points_batch_size,
                    log_freq=int(getattr(cast(Any, self.trainer), "log_every_n_steps", 1)),
                global_step=self.global_step,
                total_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            out = self(batch)
            if isinstance(self.model.orig_mod, PSSNet):
                occ = batch["points.occ"]
                logits, z, mean, log_var, z_box, mean_box, log_var_box = out
                loss = self.model.loss(logits, occ, z, mean, log_var, z_box, mean_box, log_var_box)
            elif isinstance(self.model.orig_mod, PSGN):
                pcd = batch["pointcloud"]
                loss = self.model.loss(out, pcd, emd=bool(self.loss and "emd" in self.loss))
            elif isinstance(self.model.orig_mod, PCN):
                pcd = batch["pointcloud"]
                loss = self.model.loss(out, pcd, self.global_step, emd=bool(self.loss and "emd" in self.loss))
            elif isinstance(self.model.orig_mod, SnowflakeNet):
                inputs = batch["inputs"]
                pcd = batch["pointcloud"]
                loss = self.model.loss(inputs, out, pcd)
            elif isinstance(self.model.orig_mod, ShapeFormer):
                logits, targets = out
                loss = self.model.loss(logits, targets)
            else:
                raise NotImplementedError(f"Training for {self.model.name} not implemented yet")

        batch_size = batch["inputs"].size(0)
        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        for key, value_level in cast(dict[str, Any], self.model.get_log()).items():
            value, level = value_level
            if logger.isEnabledFor(int(level)):
                if isinstance(value, (int, float)) or (
                    isinstance(value, (np.ndarray, Tensor)) and np.prod(value.shape) == 1
                ):
                    self.log(
                        f"train/{key.removeprefix('train/')}",
                        cast(_METRIC, value if not isinstance(value, np.ndarray) else torch.as_tensor(value).float()),
                        prog_bar=logger.isEnabledFor(DEBUG_LEVEL_1),
                        batch_size=batch_size,
                    )
                elif isinstance(value, (np.ndarray, Tensor)):
                    if self.do_log(log_factor=10):
                        hist_value = to_numpy(value)
                        if isinstance(hist_value, np.ndarray):
                            self.log_histogram(f"train/{key.removeprefix('train/')}", hist_value.reshape(-1).tolist())
                        else:
                            self.log_histogram(f"train/{key.removeprefix('train/')}", cast(Sequence, hist_value))
                else:
                    logger.warning(f"Cannot log {key} of type {type(value)}")
        self.model.clear_log()
        return loss

    def log_histogram(self, name: str, value: Sequence):
        if self.do_log():
            pl_logger: Any = cast(Any, self.logger).experiment
            if hasattr(pl_logger, "add_histogram"):
                pl_logger.add_histogram(name, np.array(value), self.global_step)
            elif wandb:
                pl_logger.log({name: wandb.Histogram(value)})
            else:
                logger.warning("No logger available to log histogram")

    def log_result(
        self,
        data: dict[str, Any],
        keys_in_prog_bar: Sequence[str] = ("loss", "iou", "f1", "precision", "recall", "auprc"),
        **kwargs,
    ):
        keys_in_prog_bar_list = [*list(keys_in_prog_bar), self.monitor]
        for key, value in data.items():
            metric = key.split("/")[-1]  # Remove any prefix (e.g. "train/", "val/" or "test/")
            if self.metrics is None or metric in self.metrics or metric in keys_in_prog_bar_list:
                if isinstance(value, (int, float)) or (isinstance(value, Tensor) and value.numel() == 1):
                    self.log(
                        key,
                        value,
                        prog_bar=metric in keys_in_prog_bar_list + (self.metrics or []) or logger.isEnabledFor(DEBUG),
                        sync_dist=self.sync_dist,
                        **kwargs,
                    )

    @torch.inference_mode()
    def validation_step(self, batch: dict, batch_idx: int) -> dict[str, float]:
        assert not self.model.orig_mod.training, "Model should be in eval mode during validation"
        if hasattr(self.model, "evaluate"):
            evaluate = cast(Any, self.model.evaluate if self.ema_model is None else self.ema_model.module.evaluate)
            result = evaluate(
                batch,
                name=self.loss,
                threshold=self.threshold,
                regression=self.regression,
                reduction=self.reduction,
                metrics=self.metrics,
                points_batch_size=self.points_batch_size,
                global_step=self.global_step,
                total_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            out = self(batch)
            result: dict[str, Any] = dict()

            if isinstance(self.model.orig_mod, (PSGN, PCN, SnowflakeNet)):
                pcd = batch["pointcloud"]
                prefix = "val/"

                out_pcd = out
                if isinstance(self.model.orig_mod, (PCN, SnowflakeNet)):
                    out_pcd = out[-1]

                full_pcd_result = eval_pointcloud(out_pcd, pcd)
                loss = full_pcd_result["chamfer-l1"]
                pcd_result = {
                    "loss": loss,
                    "f1": full_pcd_result["f1"],
                    "precision": full_pcd_result["precision"],
                    "recall": full_pcd_result["recall"],
                }
                pcd_result = {f"{prefix}{k}": v for k, v in pcd_result.items()}
                result.update(pcd_result)
            elif isinstance(self.model.orig_mod, ShapeFormer):
                logits, targets = out
                loss = self.model.loss(logits, targets)
                result["val/loss"] = loss
            else:
                raise NotImplementedError(f"Validation for {self.model.name} not implemented yet.")

        result = {f"val/{k.removeprefix('val/')}": to_numpy(v) for k, v in result.items()}
        val_log: dict[str, Any] = {
            f"val/{k.removeprefix('val/')}": to_numpy(v[0])
            for k, v in cast(dict[str, Any], self.model.get_log()).items()
            if logger.isEnabledFor(int(v[1]))
        }
        result.update(val_log)
        self.log_result(result, batch_size=batch["inputs"].size(0))
        self.model.clear_log()
        return result

    def on_validation_epoch_end(self):
        result = self.model.on_validation_epoch_end()
        if result is not None:
            result = {f"val/{k.removeprefix('val/')}": to_numpy(v) for k, v in cast(dict[str, Any], result).items()}
            self.log_result(result)

    def on_validation_end(self) -> None:
        if self.monitor and self.monitor not in self._logged_metrics:
            raise ValueError(f"Metric {self.monitor} not in {self._logged_metrics}")

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        pass  # Potentially handled in callbacks

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError

    def configure_optimizers(self, lr: float | None = None, **kwargs) -> Optimizer | dict[str, Any]:
        if self.optimizer is None:
            lr_value = float(self.learning_rate or lr or 1e-4)
            self.optimizer = AdamW(self.parameters(), lr=lr_value, **kwargs)

        if self.scheduler is not None:
            if self.learning_rate or lr:
                raise NotImplementedError("Setting learning rate with scheduler not implemented yet.")

            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": self.interval,
                    "frequency": self.frequency,
                    "monitor": self.monitor,
                    "strict": True,
                },
            }

        if self.learning_rate or lr:
            lr = self.learning_rate or lr
            lrs = [group["lr"] for group in self.optimizer.param_groups]
            if np.unique(lrs).size == 1:
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
            elif np.unique(lrs).size == 2:
                factor = min(lrs) / max(lrs)
                for group in self.optimizer.param_groups:
                    if group["lr"] == max(lrs):
                        group["lr"] = lr
                    else:
                        group["lr"] = lr * factor
            else:
                raise NotImplementedError("Modifying more than two different learning rates not implemented yet.")

        return self.optimizer
