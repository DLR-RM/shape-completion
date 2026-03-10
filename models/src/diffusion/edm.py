import itertools
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_edm_euler import EDMEulerScheduler, EDMEulerSchedulerOutput
from torch import Tensor, nn
from tqdm import tqdm

from ..utils import check_precision, check_range


def edm_sampler(
    net: nn.Module,
    latents: Tensor,
    cond: Tensor | None = None,
    randn_like: Callable = torch.randn_like,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: float = 7,
    s_churn: float = 0,
    s_min: float = 0,
    s_max: float = float("inf"),
    s_noise: float = 1,
    progress: bool = True,
    return_intermediates: bool = False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, getattr(net, "sigma_min", 0))
    sigma_max = min(sigma_max, getattr(net, "sigma_max", float("inf")))

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    intermediates = [] if return_intermediates else None
    x_next = latents.to(torch.float64) * t_steps[0]
    p_bar = tqdm(itertools.pairwise(t_steps), desc="Sampling", total=num_steps, leave=False, disable=not progress)
    for i, (t_cur, t_next) in enumerate(p_bar):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(s_churn / num_steps, np.sqrt(2) - 1) if s_min <= t_cur <= s_max else 0
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * s_noise * randn_like(x_cur)
        p_bar.set_postfix(sigma=t_hat.item())

        # Euler step.
        denoised = net(x_hat.float(), t_hat.float(), cond).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next.float(), t_next.float(), cond).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        if intermediates is not None:
            intermediates.append(x_next.float().clone())

    if intermediates is not None:
        return x_next.float(), intermediates
    return x_next.float()


class EDMScheduler(EDMEulerScheduler):
    @register_to_config
    def __init__(
        self,
        num_steps: int = 18,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
    ):
        super().__init__(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            # sigma_schedule="exponential",
            rho=rho,
        )
        self.set_timesteps(num_steps)

    @check_precision(torch.float32)
    @check_range(torch.float32)
    def precondition_timesteps(self, timesteps: Tensor):
        config = cast(Any, self.config)
        sigmas = (timesteps * config.p_std + config.p_mean).exp()
        return sigmas

    @check_precision(torch.float32)
    @check_range(torch.float32)
    def precondition_noise(self, sigma: Tensor) -> Tensor:
        return sigma.log() / 4

    @check_precision(torch.float32)
    @check_range(torch.float32)
    def precondition_inputs(self, samples: Tensor, sigmas: Tensor) -> Tensor:
        # Divide by combined std dev of data and noise
        config = cast(Any, self.config)
        c_in = 1 / (config.sigma_data**2 + sigmas**2).sqrt()
        return c_in * samples

    @check_precision(torch.float32)
    @check_range(torch.float32)
    def precondition_outputs(self, samples: Tensor, model_output: Tensor, sigmas: float | Tensor) -> Tensor:
        config = cast(Any, self.config)
        sigma_tensor = torch.as_tensor(sigmas, dtype=samples.dtype, device=samples.device)
        c_skip = config.sigma_data**2 / (sigma_tensor**2 + config.sigma_data**2)
        c_out = sigma_tensor * config.sigma_data / (sigma_tensor**2 + config.sigma_data**2).sqrt()
        return c_skip * samples + c_out * model_output

    @check_precision(torch.float32)
    @check_range(torch.float32)
    def precondition_loss(self, loss: Tensor, sigmas: Tensor) -> Tensor:
        config = cast(Any, self.config)
        weight = (sigmas**2 + config.sigma_data**2) / (sigmas * config.sigma_data) ** 2
        return weight * loss

    @check_precision(torch.float32)
    @check_range(torch.float32)
    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor, **kwargs) -> Tensor:
        return original_samples + noise * timesteps

    @check_precision(torch.float32)
    @check_range(torch.float32)
    def step(
        self,
        model_output: Tensor,
        timestep: float | Tensor,
        sample: Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> EDMEulerSchedulerOutput | tuple:
        pred_original_sample = self.precondition_outputs(sample, model_output, timestep)
        prev_sample = sample

        if not return_dict:
            return (prev_sample,)

        return EDMEulerSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
