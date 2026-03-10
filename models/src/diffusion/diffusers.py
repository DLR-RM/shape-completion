import math
from functools import cached_property
from random import random
from typing import Any, cast

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_edm_euler import EDMEulerScheduler
from matplotlib import cm
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from utils import setup_logger, unsqueeze_as

from ..mixins import MultiEvalMixin
from .latent import EDMScheduler
from .model import DiffusionModel
from .unet import UNet
from .utils import extract

logger = setup_logger(__name__)


class DummyScheduler:
    pass


try:
    from diffusers.schedulers.scheduling_vdm import VDMScheduler
except ImportError:
    logger.debug("VDMScheduler not found, using DummyScheduler")
    VDMScheduler = DummyScheduler

try:
    from diffusers.schedulers import DiscreteStateScheduler  # pyright: ignore[reportAttributeAccessIssue]
except ImportError:
    logger.debug("DiscreteStateScheduler not found, using DummyScheduler")
    DiscreteStateScheduler = DummyScheduler


class DiffusersModel(MultiEvalMixin, DiffusionModel):
    def __init__(
        self,
        scheduler: str = "ddpm",
        num_train_timesteps: int | None = 1000,
        num_inference_steps: int = 100,
        num_eval_steps: int = 1,
        beta_schedule: str = "linear",
        clip_sample: bool | float = False,
        truncate_sample: bool = False,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
        resolution: int = 32,
        loss: str | None = None,
        self_condition: bool = False,
        self_cond_on_prev_step: bool = False,
        self_cond_grad: bool = False,
        zero_snr: bool = False,
        min_snr_gamma: float = 0,
        stop_steps: int | None = None,
        threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        if truncate_sample:
            assert prediction_type == "sample", "Truncation only supported for 'sample' prediction type"

        loss = loss or "mse"
        if "edm" not in scheduler:
            if loss != "mse":
                if prediction_type != "sample":
                    logger.warning("Setting prediction type to 'sample' if loss is not 'mse'")
                prediction_type = "sample"

            if prediction_type != "sample":
                if loss != "mse":
                    logger.warning("Setting loss to 'mse' if prediction type is not 'sample'")
                loss = "mse"

        if scheduler == "discrete":
            if prediction_type != "sample":
                logger.warning("Setting prediction type to 'sample' for DiscreteStateScheduler")
            prediction_type = "sample"
            if loss is None:
                loss = "bce"
        elif scheduler in ["vdm", "karras"]:
            num_train_timesteps = None

        self.resolution = resolution
        dim_mults = (1, 2) if resolution == 16 else (1, 2, 4) if resolution == 32 else (1, 2, 4, 8)
        self.denoise_fn = UNet(
            dim=resolution,
            channels=1,
            dim_mults=dim_mults,
            resolution=resolution,
            self_condition=self_condition,
            **kwargs,
        )

        self.scheduler_type = scheduler
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.truncate_sample = truncate_sample
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing
        self.loss_type = loss
        self.num_eval_steps = num_eval_steps
        self.zero_snr = zero_snr
        self.min_snr_gamma = min_snr_gamma  # 5 is the default value in the paper
        self.stop_steps = stop_steps
        self.threshold = math.log(threshold / (1 - threshold))

        self._self_condition = None
        self.self_cond_on_prev_step = self_cond_on_prev_step
        self.self_cond_grad = self_cond_grad
        self.antithetic_time_sampling = False

        logger.debug_level_1(f"Using scheduler: {self.scheduler.__class__.__name__}")
        logger.debug_level_1(f"Using prediction type: {self.prediction_type.capitalize()}")
        logger.debug_level_1(f"Using loss type: {self.loss_type.upper()}")

    def __len__(self) -> int:
        return len(self.scheduler)

    @property
    def self_condition(self) -> bool:
        if self.denoise_fn.self_condition:
            if self.training:
                if self._self_condition is None:
                    self._self_condition = False
                    return False  # Disable self-conditioning for first training step (prevents gradient check error)
                return random() < 0.5  # Self-condition 50% of the time during training
            return True  # Always self-condition during evaluation
        return False  # Never self-condition if the denoiser doesn't support it

    @cached_property
    def scheduler(self) -> Any:
        if self.scheduler_type in ["ddpm", "ddim", "dpm++"]:
            return DDPMScheduler(
                num_train_timesteps=int(self.num_train_timesteps or 1000),
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule=self.beta_schedule,
                clip_sample=bool(self.clip_sample),
                clip_sample_range=float(self.clip_sample) if isinstance(self.clip_sample, (int, float)) else 1.0,
                prediction_type=self.prediction_type,
                timestep_spacing=self.timestep_spacing,
                rescale_betas_zero_snr=self.zero_snr,
            )
        elif "edm" in self.scheduler_type:
            assert self.timestep_spacing == "leading", "EDM only supports leading timestep spacing"
            assert self.beta_schedule == "linear", "EDM only supports linear beta schedule"
            return EDMEulerScheduler(  # sigma_max=200,
                # sigma_schedule="exponential",
                num_train_timesteps=int(self.num_train_timesteps or 1000),
                prediction_type=self.prediction_type,
            )
        elif self.scheduler_type == "karras":
            return EDMScheduler(sigma_max=80)
        elif self.scheduler_type == "vdm":
            vdm_cls = cast(Any, VDMScheduler)
            return vdm_cls(
                num_train_timesteps=self.num_train_timesteps,
                beta_schedule=self.beta_schedule,
                clip_sample=bool(self.clip_sample),
                clip_sample_range=float(self.clip_sample) if isinstance(self.clip_sample, (int, float)) else 1.0,
                prediction_type=self.prediction_type,
                timestep_spacing=self.timestep_spacing,
            )
        elif self.scheduler_type == "discrete":
            discrete_cls = cast(Any, DiscreteStateScheduler)
            return discrete_cls(
                num_train_timesteps=self.num_train_timesteps,
                beta_schedule=self.beta_schedule,
                timestep_spacing=self.timestep_spacing,
                implementation="log" if self.loss_type in ["kl", "hybrid"] else "simple",
                rescale_betas_zero_snr=self.zero_snr,
            )
        else:
            raise ValueError(f"Invalid scheduler type: {self.scheduler_type}")

    @cached_property
    def solver(self) -> Any:
        if self.scheduler_type == "dpm++":
            common_kwargs = set(DPMSolverMultistepScheduler().config.keys()) & set(self.scheduler.config.keys())
            init_kwargs = {key: value for key, value in self.scheduler.config.items() if key in common_kwargs}
            return DPMSolverMultistepScheduler(**init_kwargs, solver_order=3, lower_order_final=True)
        elif self.scheduler_type == "ddim":
            common_kwargs = set(DDIMScheduler().config.keys()) & set(self.scheduler.config.keys())
            init_kwargs = {key: value for key, value in self.scheduler.config.items() if key in common_kwargs}
            return DDIMScheduler(**init_kwargs)
        elif self.scheduler_type == "edm++":
            common_kwargs = set(EDMDPMSolverMultistepScheduler().config.keys()) & set(self.scheduler.config.keys())
            init_kwargs = {key: value for key, value in self.scheduler.config.items() if key in common_kwargs}
            return EDMDPMSolverMultistepScheduler(
                **init_kwargs,
                solver_order=3,
                thresholding=bool(self.clip_sample),
                dynamic_thresholding_ratio=1,
                # solver_type="heun",
                lower_order_final=True,
            )

        return self.scheduler

    def forward(
        self,
        x: Tensor | tuple[Tensor, Tensor] | dict[str, Tensor] | None = None,
        t: Tensor | None = None,
        x_self_cond: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if isinstance(x, dict):
            t = x.get("t")
            x = x.get("x")

        if x is None:
            o = torch.zeros((1, 1, self.resolution**3), device=self.device, dtype=self.dtype)
            p = torch.randn((1, self.resolution**3, 3), device=self.device, dtype=self.dtype)
            x = (o, p)
        elif isinstance(x, Tensor):
            x = (x[..., -1].unsqueeze(1), x[..., :-1])

        if t is None:
            t = torch.randint(0, len(self), (1 if x is None else x[0].size(0),), device=self.device)
            if isinstance(self.scheduler, VDMScheduler) and hasattr(self.scheduler, "log_snr"):
                t = t / len(self)
        if isinstance(self.scheduler, VDMScheduler) and hasattr(self.scheduler, "log_snr"):
            t = cast(Any, self.scheduler).log_snr(t)

        if self.denoise_fn.self_condition:
            return self.denoise_fn(x, t, x_self_cond)
        return self.denoise_fn(x, t)

    def encode(self, inputs: Tensor, **kwargs) -> Tensor:
        return inputs

    def decode(self, data: dict[str, Any], **kwargs) -> dict[str, Tensor]:
        return {"logits": cast(Tensor, self.predict(**data))}

    def _predict_step(
        self,
        x: Tensor | tuple[Tensor, Tensor],
        timesteps: Tensor,
        noise: Tensor,
        x_cond_mask: Tensor | None = None,
        x_self_cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if torch.is_tensor(x):
            inputs, points = x, None
        else:
            inputs, points = x

        if isinstance(self.scheduler, EDMScheduler):
            timesteps = unsqueeze_as(timesteps, inputs)
            sigmas = self.scheduler.precondition_timesteps(timesteps)
            noisy_inputs = self.scheduler.add_noise(inputs, noise, sigmas)
            if x_cond_mask is not None:
                noisy_inputs[x_cond_mask.view_as(noisy_inputs)] = 1
            cond_noisy_inputs = self.scheduler.precondition_inputs(noisy_inputs, sigmas)
            cond_sigmas = self.scheduler.precondition_noise(sigmas)
            predictions = self((cond_noisy_inputs, points), cond_sigmas.flatten(), x_self_cond)
            predictions = self.scheduler.precondition_outputs(noisy_inputs, predictions, sigmas)
        elif isinstance(self.scheduler, EDMEulerScheduler):
            sigmas = extract(self.scheduler.sigmas.to(inputs.device), timesteps, inputs.shape)
            timesteps = self.scheduler.timesteps.to(inputs.device)[timesteps]
            noisy_inputs = self.scheduler.add_noise(inputs, noise, timesteps)
            if x_cond_mask is not None:
                noisy_inputs[x_cond_mask.view_as(noisy_inputs)] = 1
            cond_noisy_inputs = self.scheduler.precondition_inputs(noisy_inputs, sigmas)
            predictions = self((cond_noisy_inputs, points), timesteps, x_self_cond)
            predictions = self.scheduler.precondition_outputs(noisy_inputs, predictions, sigmas)
        else:
            noisy_inputs = self.scheduler.add_noise(inputs, noise, timesteps)
            if x_cond_mask is not None:
                noisy_inputs[x_cond_mask.view_as(noisy_inputs)] = 1
            predictions = self((noisy_inputs, points), timesteps, x_self_cond)

        return predictions, noisy_inputs

    def _predict_self_cond(
        self, x: tuple[Tensor, Tensor], timesteps: Tensor, noise: Tensor, x_cond: Tensor | None = None
    ) -> Tensor:
        inputs, points = x

        prev_timesteps = timesteps
        if self.self_cond_on_prev_step:
            prev_timesteps = (timesteps - 1).clamp(min=0)
            if isinstance(self.scheduler, VDMScheduler):
                prev_timesteps = (timesteps - 1 / len(self)).clamp(0, 1)

        enable_grad = self.training and self.self_cond_grad
        with torch.set_grad_enabled(enable_grad):
            prev_predictions = self._predict_step((inputs, points), prev_timesteps, noise, x_cond)[0]
            if not enable_grad:
                prev_predictions = prev_predictions.detach()  # TODO: Is this necessary?

        if self.self_cond_on_prev_step:
            prev_t = unsqueeze_as(prev_timesteps, prev_predictions)
            prev_predictions = torch.where(prev_t == 0, torch.zeros_like(prev_predictions), prev_predictions)

        return prev_predictions

    @staticmethod
    def _mask_sample(sample: Tensor, mask: Tensor, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if mask.ndim == 3 and mask.size(2) == 3:
            batch_size, num_points = mask.shape[:2]
            points = torch.cat([points, mask], dim=1)
            mask_occ = torch.ones(batch_size, num_points, device=mask.device)
            if sample.ndim == 2:
                sample = torch.cat([sample, mask_occ], dim=1)
            elif sample.ndim == 3:
                sample = torch.cat([sample, mask_occ.unsqueeze(1)], dim=2)
            mask = torch.zeros_like(sample, dtype=torch.bool)
            mask[..., -num_points:] = True
        sample[mask.view_as(sample)] = 1
        return sample, mask, points

    def _extract_data(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor | None]:
        occupancies = data["points.occ"].float()  # B, N
        points = data["points"]  # B, N, 3

        mask = None
        inputs = data["inputs"]  # B or B, M, 3 or B, R, R, R or B, C, R, R, R
        if inputs.ndim in [3, 4, 5]:
            occupancies, mask, points = self._mask_sample(occupancies, inputs, points)

        inputs = occupancies.unsqueeze(1)
        if not isinstance(self.scheduler, DiscreteStateScheduler):
            inputs = 2 * inputs - 1  # [0, 1] -> [-1, 1]
        return inputs, points, mask

    def _get_timesteps(self, inputs: Tensor) -> Tensor:
        continuous_time = isinstance(self.scheduler, (VDMScheduler, EDMScheduler)) and self.num_train_timesteps is None
        if self.antithetic_time_sampling:  # Timesteps are arranged around single random timestep
            if isinstance(self.scheduler, EDMScheduler):
                raise ValueError("Antithetic time sampling not supported for EDM")
            timesteps = (torch.rand(1) + torch.arange(1, inputs.size(0) + 1) / inputs.size(0)) % 1
            timesteps = timesteps.to(inputs.device)
            if not continuous_time:
                timesteps = (timesteps * int(self.num_train_timesteps or 1)).long()
        else:
            if continuous_time:
                if isinstance(self.scheduler, EDMScheduler):
                    timesteps = torch.randn((inputs.size(0),), device=inputs.device)
                else:
                    timesteps = torch.rand((inputs.size(0),), device=inputs.device)
            else:
                if self.num_train_timesteps is None:
                    raise ValueError("num_train_timesteps must be set for discrete timesteps.")
                timesteps = torch.randint(0, self.num_train_timesteps, (inputs.size(0),), device=inputs.device)
        return timesteps

    def _get_noise(self, inputs: Tensor) -> Tensor:
        if isinstance(self.scheduler, DiscreteStateScheduler):
            return torch.rand(
                *inputs.shape, self.scheduler.config.num_classes, dtype=inputs.dtype, device=inputs.device
            )
        return torch.randn_like(inputs)

    def _train_val_step(
        self, data: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        inputs, points, mask = self._extract_data(data)
        noise = self._get_noise(inputs)
        timesteps = self._get_timesteps(inputs)

        preds_are_logits = self.prediction_type == "sample" or isinstance(self.scheduler, EDMEulerScheduler)

        cond_preds = None
        if self.self_condition:
            cond_preds = self._predict_self_cond((inputs, points), timesteps, noise, mask)
            if preds_are_logits:
                if not isinstance(self.scheduler, EDMScheduler):
                    cond_preds = torch.tanh(cond_preds)  # [-inf, inf] -> [-1, 1]: 2 * sigmoid(x) - 1
        predictions, noisy_inputs = self._predict_step((inputs, points), timesteps, noise, mask, cond_preds)

        logits = None
        targets = noise
        if preds_are_logits:
            logits = predictions
            targets = inputs
            if self.loss_type == "mse":
                if not isinstance(self.scheduler, EDMScheduler):
                    predictions = torch.tanh(predictions)
                if isinstance(self.scheduler, DiscreteStateScheduler):
                    targets = 2 * targets - 1  # [0, 1] -> [-1, 1]
            elif not isinstance(self.scheduler, DiscreteStateScheduler):
                targets = targets / 2 + 0.5  # [-1, 1] -> [0, 1]
        elif self.prediction_type == "v_prediction":
            targets = self.scheduler.get_velocity(inputs, noise, timesteps)

        return predictions, targets, timesteps, noisy_inputs, logits, mask

    @torch.no_grad()
    def predict(self, data: dict[str, Tensor], return_loss: bool = False, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        predictions, targets, timesteps, noisy_inputs, logits, mask = self._train_val_step(data)

        if logits is None or self.num_eval_steps > 1:
            scheduler_any = cast(Any, self.scheduler)

            def _step_fn_discrete(_preds, _times, _inputs):
                out = scheduler_any.step(_preds, _times, _inputs)
                if self.num_eval_steps > 1:
                    return out.prev_sample
                return out.pred_original_sample

            def _step_fn_continuous(_preds, _times, _inputs):
                if self.num_eval_steps > 1:
                    return torch.stack(
                        [
                            scheduler_any.step(p, t, i).prev_sample
                            for p, t, i in zip(_preds, _times, _inputs, strict=False)
                        ]
                    )
                return torch.stack(
                    [
                        scheduler_any.step(p, t, i).pred_original_sample
                        for p, t, i in zip(_preds, _times, _inputs, strict=False)
                    ]
                )

            step_fn = (
                _step_fn_discrete
                if isinstance(self.scheduler, (VDMScheduler, DiscreteStateScheduler))
                else _step_fn_continuous
            )

            if self.num_eval_steps > 1:
                step_size = len(self) // self.num_eval_steps
                if isinstance(self.scheduler, VDMScheduler):
                    step_size = 1 / self.num_eval_steps
                logits = predictions
                sample = noisy_inputs
                t_mask = timesteps > 0
                while t_mask.sum() > 0:
                    sample[t_mask] = step_fn(predictions[t_mask], timesteps[t_mask], sample[t_mask])

                    if self.prediction_type != "sample":
                        logits[t_mask] = sample[t_mask].to(logits.dtype)

                    timesteps = (timesteps - step_size).clamp(0)
                    t_mask = timesteps > 0

                    if t_mask.sum() > 0:
                        if mask is not None:
                            sample[mask.view_as(sample)] = 1

                        model_in = cast(Tensor, scheduler_any.scale_model_input(sample))
                        predictions[t_mask] = self(
                            (model_in[t_mask], data["points"][t_mask]), timesteps[t_mask], predictions[t_mask]
                        )

                        if self.prediction_type == "sample":
                            logits[t_mask] = predictions[t_mask]
                            if not isinstance(self.solver, DiscreteStateScheduler):
                                predictions[t_mask] = torch.tanh(predictions[t_mask])
            else:
                if hasattr(self.scheduler.config, "clip_sample"):
                    clip = self.scheduler.config.clip_sample
                    self.scheduler.config.clip_sample = False
                logits = step_fn(predictions, timesteps, noisy_inputs)
                if hasattr(self.scheduler.config, "clip_sample"):
                    self.scheduler.config.clip_sample = clip

        if mask is not None and mask.ndim < 4:  # PVD-style: Filter logits for masked points
            logits = logits[~mask.view_as(logits)].view(logits.size(0), 1, -1)

        if return_loss:
            if self.loss_type == "mse":
                loss = F.mse_loss(predictions, targets, reduction="none")
            else:
                loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
            if isinstance(self.scheduler, EDMScheduler):
                sigmas = self.scheduler.precondition_timesteps(timesteps)
                loss = self.scheduler.precondition_loss(loss, unsqueeze_as(sigmas, loss))

            if mask is not None and mask.ndim < 4:  # PVD-style: Filter loss on masked points
                loss = loss[~mask.view_as(loss)].view(loss.size(0), -1)

            return logits, loss
        return logits

    @torch.no_grad()
    def evaluate(self, data: dict[str, Tensor], **kwargs) -> dict[str, float]:
        logits = data.get("logits")
        loss = data.get("loss")
        if logits is None or loss is None:
            logits, loss = self.predict(data, return_loss=True, **kwargs)

        data.update({"logits": logits.squeeze(1), "loss": loss.mean()})
        return super().evaluate(cast(Any, data), **kwargs)

    def _check_inputs(self, inputs: Tensor | None = None, points: Tensor | None = None) -> tuple[int, int]:
        if inputs is not None:
            if inputs.ndim in [4, 5]:
                if not all(r == self.resolution for r in inputs.shape[-3:]):
                    raise ValueError(f"Invalid resolution: {inputs.shape[-3:]}")
                # return inputs.size(0), inputs.size(-1) ** 3
            elif inputs.ndim == 3:
                if inputs.size(2) != 3:
                    raise ValueError("Inputs shape != (B, N, 3)")
                # return inputs.shape[:2]
            else:
                raise ValueError("Inputs must be a 3D point cloud or a 4/5D voxel grid.")

        if points is None:
            return 1, self.resolution**3

        if points.ndim != 3 or points.size(2) != 3:
            raise ValueError("Points shape != (B, N, 3)")

        return int(points.shape[0]), int(points.shape[1])

    def _get_gen_timesteps(self) -> Tensor:
        solver = cast(Any, self.solver)
        solver.set_timesteps(num_inference_steps=self.num_inference_steps, device=self.device)
        timesteps = cast(Tensor, solver.timesteps)

        if self.stop_steps:
            timesteps = timesteps[: self.stop_steps]

        return timesteps

    def _get_gen_noise(self, batch_size: int, num_points: int) -> Tensor:
        if isinstance(self.scheduler, DiscreteStateScheduler):
            return torch.rand(
                batch_size,
                1,
                num_points,
                cast(Any, self.scheduler.config).num_classes,
                dtype=self.dtype,
                device=self.device,
            ).argmax(dim=-1)
        return torch.randn(batch_size, 1, num_points, dtype=self.dtype, device=self.device)

    @torch.inference_mode()
    def generate(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        threshold: float | None = None,
        points_batch_size: int | None = None,
        show: bool = False,
        render: bool = False,
        **kwargs,
    ) -> Tensor:
        batch_size, num_points = self._check_inputs(inputs, points)
        timesteps = self._get_gen_timesteps()
        solver = cast(Any, self.solver)
        sample = solver.init_noise_sigma * self._get_gen_noise(batch_size, num_points)
        if inputs is not None:
            if points is None:
                raise ValueError("Points must be provided when conditioning with inputs.")
            sample, in_mask, points = self._mask_sample(sample, inputs, points)
            assert points is not None
            assert in_mask is not None

        scheduler_info = f"{self.scheduler_type.upper()} ({self.loss_type}, {self.prediction_type})"
        if show:
            if points is None:
                raise ValueError("Points must be provided for visualization")
            threshold = math.log(threshold / (1 - threshold)) if threshold else self.threshold
            step_info = f"{len(timesteps)} {self.beta_schedule} steps"
            window_name = f"{scheduler_info}: {step_info}"
            vis = cast(Any, o3d).visualization.Visualizer()
            vis.create_window(window_name=window_name, width=800, height=600, left=100, top=200)
            query_points = points[0].cpu().numpy()
            occ_mask = sample[0].squeeze(0).cpu().numpy() > threshold
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_points[occ_mask]))
            vis.add_geometry(pcd)

        predictions = None
        p_bar = tqdm(timesteps, desc=scheduler_info, leave=False)
        for i, t in enumerate(p_bar):
            p_bar.set_postfix(t=t.item())

            if inputs is not None:
                assert points is not None and in_mask is not None
                sample, in_mask, points = self._mask_sample(sample, in_mask, points)

            model_in = cast(Tensor, solver.scale_model_input(sample, t))
            assert points is not None
            if points_batch_size is None or points_batch_size >= points.size(1):
                predictions = self((model_in, points), t.unsqueeze(0), predictions)
            elif inputs is None and not self.denoise_fn.self_condition:
                idx = torch.randperm(points.size(1))
                i_split = torch.split(model_in[..., idx], points_batch_size, dim=2)
                p_split = torch.split(points[:, idx], points_batch_size, dim=1)
                predictions = [self((ii, pi), t.unsqueeze(0)) for ii, pi in zip(i_split, p_split, strict=False)]
                predictions = torch.cat(predictions, dim=2)[..., torch.argsort(idx)]
            else:
                raise ValueError("Inputs can't be split when conditioning")

            if self.prediction_type == "sample":
                logits = predictions.squeeze(1)
                if not isinstance(self.solver, DiscreteStateScheduler):  # Logits to discrete state in step
                    predictions = torch.tanh(predictions)

            if isinstance(self.solver, DDIMScheduler):
                output = solver.step(predictions, t, sample, eta=0, use_clipped_model_output=True)
            else:
                output = solver.step(predictions, t, sample)

            if self.prediction_type != "sample":
                logits = cast(Tensor, output.prev_sample).squeeze(1)
                if hasattr(output, "pred_original_sample"):
                    logits = cast(Tensor, output.pred_original_sample).squeeze(1)

            if isinstance(self.solver, EDMEulerScheduler):
                if isinstance(self.solver, EDMScheduler):
                    orig_sample = cast(Tensor, output.pred_original_sample).clip(-1, 1)
                else:
                    orig_sample = torch.tanh(cast(Tensor, output.pred_original_sample))
                step_index = int(solver.step_index) - 1
                sigma = solver.sigmas[step_index]
                derivative = (sample - orig_sample) / sigma
                dt = solver.sigmas[step_index + 1] - sigma
                sample = (sample + derivative * dt).type(predictions.dtype)
            else:
                sample = cast(Tensor, output.prev_sample)

            if show:
                colors = logits
                if inputs is not None:
                    threshold_val = float(self.threshold if threshold is None else threshold)
                    colors[in_mask.view_as(colors)] = threshold_val
                colors = colors[0].float().cpu().numpy()
                occ_mask = colors >= threshold
                if occ_mask.sum() > query_points.shape[0] // 1:  # Downsample if occupancy >10%
                    true_indices = np.where(occ_mask)[0]
                    selected_indices = np.random.choice(true_indices, size=query_points.shape[0] // 10, replace=False)
                    occ_mask.fill(False)
                    occ_mask[selected_indices] = True
                pcd_points = query_points[occ_mask]
                pcd.points = o3d.utility.Vector3dVector(pcd_points)
                pcd.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, -math.pi / 4, -math.pi / 8)), center=(0, 0, 0))

                colors = colors[occ_mask]
                threshold_mask = colors == float(self.threshold if threshold is None else threshold)
                if any(colors) and colors.min() != colors.max():
                    norm_colors = (colors - colors.min()) / (colors.max() - colors.min())
                    colormap = cm.get_cmap("magma")
                    colors = colormap(norm_colors)[:, :3]
                    colors[threshold_mask, :] = [0, 1, 0]
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                if render:
                    color = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                    image = Image.fromarray((color * 255).astype(np.uint8))
                    image.save(f"diffusion_debug/img{i:04d}.png")

        if inputs is not None and in_mask is not None and in_mask.ndim < 4:
            logits = logits[~in_mask.view_as(logits)].view(logits.size(0), -1)  # TODO: Is this correct?

        return cast(Tensor, logits).float()

    def _min_snr_loss(self, loss: Tensor, timesteps: Tensor) -> Tensor:
        # From "Efficient Diffusion Training via Min-SNR Weighting Strategy" (https://arxiv.org/abs/2303.09556)
        if isinstance(self.scheduler, VDMScheduler):
            log_snr = cast(Any, self.scheduler).log_snr(timesteps)
            snr = log_snr.exp()
        else:
            alpha = self.scheduler.alphas_cumprod[timesteps]
            snr = alpha / (1 - alpha)

        loss_weight = snr.clamp(min=self.min_snr_gamma)
        if self.prediction_type == "epsilon" or isinstance(self.scheduler, EDMEulerScheduler):
            loss_weight /= snr
        elif self.prediction_type == "v_prediction":
            loss_weight /= snr + 1
        loss = unsqueeze_as(loss_weight, loss) * loss
        return loss

    def _vb_loss(self, predictions: Tensor, targets: Tensor, timesteps: Tensor, noisy_inputs: Tensor) -> Tensor:
        prev_logits = self.scheduler.step(targets.long(), timesteps, noisy_inputs).prev_log_probs
        pred_prev_logits = self.scheduler.step(predictions, timesteps, noisy_inputs).prev_log_probs

        kl = (
            F.kl_div(
                F.log_softmax(pred_prev_logits.view(-1, pred_prev_logits.size(-1)), dim=-1),
                F.softmax(prev_logits.view(-1, prev_logits.size(-1)), dim=-1),
                reduction="none",
            )
            .sum(dim=-1)
            .view_as(targets)
            .clamp(min=0)
        )
        nll = F.cross_entropy(
            pred_prev_logits.view(-1, pred_prev_logits.size(-1)), targets.view(-1).long(), reduction="none"
        ).view_as(targets)

        # Cross-entropy equivalent to:
        # targets_one_hot = F.one_hot(targets.long(), num_classes=self.scheduler.config.num_classes).float()
        # nll = -(F.log_softmax(pred_prev_logits, dim=-1) * targets_one_hot).sum(dim=-1)

        # Equivalently for binary logits:
        # binary_logits = pred_prev_logits[..., 1] - pred_prev_logits[..., 0]
        # nll = F.binary_cross_entropy_with_logits(binary_logits, targets, reduction="none")

        return torch.where(unsqueeze_as(timesteps, kl) == 0, nll, kl)

    def loss(self, data: dict[str, Tensor], **kwargs) -> Tensor:
        predictions, targets, timesteps, noisy_inputs, _logits, mask = self._train_val_step(data)

        if self.loss_type == "mse":
            loss = F.mse_loss(predictions, targets, reduction="none")
        elif self.loss_type in ["bce", "hybrid", "kl"]:
            if self.loss_type in ["bce", "hybrid"]:
                loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
                if self.loss_type == "hybrid":
                    vb_loss = self._vb_loss(predictions, targets, timesteps, noisy_inputs)
                    loss = (vb_loss + 0.001 * loss) / math.log(2)
            else:
                loss = self._vb_loss(predictions, targets, timesteps, noisy_inputs)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        if isinstance(self.scheduler, EDMScheduler):
            sigmas = self.scheduler.precondition_timesteps(timesteps)
            loss = self.scheduler.precondition_loss(loss, unsqueeze_as(sigmas, loss))
        elif self.min_snr_gamma:
            loss = self._min_snr_loss(loss, timesteps)

        if mask is not None and mask.ndim == 3 and mask.size(2) == 3:
            loss = loss[~mask.view_as(loss)]

        return loss.mean()
