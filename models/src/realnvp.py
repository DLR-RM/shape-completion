from typing import Any, cast

import torch
from torch import Tensor, nn

"""
'batch_size': 1500,
'dim': 24,
'num_masked': 12,
'learning_rate': 1e-5,
'translation_pixel_range_x': 10,
'translation_pixel_range_y': 10,
'translation_pixel_range_z': 10,
'num_bijectors': 8,
'train_iters': 2e5,
'num_epochs' = 1000
"""


class BatchNormFlow(nn.Module):
    """PyTorch implementation of TensorFlow's BatchNorm bijector.
    https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/BatchNormalization
    """

    def __init__(self, num_inputs: int, momentum: float = 0.99, eps: float = 1e-3):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_inputs))
        self.register_buffer("running_var", torch.ones(num_inputs))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        mean = cast(Tensor, self.running_mean)
        var = cast(Tensor, self.running_var)
        beta = cast(Tensor, self.beta)
        gamma = cast(Tensor, self.gamma)

        x_hat = (inputs - beta) / gamma
        y = x_hat * var.sqrt() + mean
        return y, -(gamma.log() - 0.5 * var.log()).sum()

    def inverse(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if self.training:
            mean = inputs.mean(0)
            var = (inputs - mean).pow(2).mean(0) + self.eps

            running_mean = cast(Tensor, self.running_mean)
            running_var = cast(Tensor, self.running_var)
            running_mean.mul_(self.momentum)
            running_var.mul_(self.momentum)

            running_mean.add_(mean.data * (1 - self.momentum))
            running_var.add_(var.data * (1 - self.momentum))
        else:
            mean = cast(Tensor, self.running_mean)
            var = cast(Tensor, self.running_var)

        gamma = torch.relu(cast(Tensor, self.gamma)) + 1e-6
        beta = cast(Tensor, self.beta)
        x_hat = (inputs - mean) / var.sqrt()
        y = gamma * x_hat + beta
        return y, (gamma.log() - 0.5 * var.log()).sum()


class ScaleShiftLayer(nn.Module):
    """PyTorch implementation of TensorFlow's `real_nvp_default_template` and `RealNVP`.
    https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP#real_nvp_default_template
    https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP
    """

    def __init__(
        self,
        num_inputs: int,
        num_masked: int,
        hidden_layers: list[int],
        shift_only: bool = False,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.shift_only = shift_only
        self.num_masked = num_masked

        num_inputs -= num_masked
        layers = list()
        for index, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(num_inputs if index == 0 else hidden_dim, hidden_dim))
            layers.append(activation(inplace=True) if hasattr(activation, "inplace") else activation())
        layers.append(nn.Linear(hidden_layers[-1], (1 if shift_only else 2) * num_inputs))
        self.layers = nn.Sequential(*layers)

    def shift_log_scale(self, inputs: Tensor) -> tuple[Tensor, Tensor | None]:
        x = self.layers(inputs)
        if self.shift_only:
            return x, None
        shift, log_scale = torch.split(x, x.size(1) // 2, dim=1)
        return shift, log_scale

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor | None]:
        x = inputs
        x0, x1 = x[..., : self.num_masked], x[..., self.num_masked :]
        shift, log_scale = self.shift_log_scale(x0)

        if log_scale is None:
            y1 = x1 + shift
            logdet = inputs.new_zeros(inputs.size(0))
        else:
            scale = torch.exp(log_scale)
            y1 = x1 * scale + shift
            logdet = log_scale.sum(dim=1)

        return torch.cat([x0, y1], dim=-1), logdet

    def inverse(self, inputs: Tensor) -> tuple[Tensor, Tensor | None]:
        x = inputs
        x0, x1 = x[..., : self.num_masked], x[..., self.num_masked :]
        shift, log_scale = self.shift_log_scale(x0)

        if log_scale is None:
            y1 = x1 - shift
            logdet = inputs.new_zeros(inputs.size(0))
        else:
            scale = torch.exp(-log_scale)
            y1 = (x1 - shift) * scale
            logdet = -log_scale.sum(dim=1)

        return torch.cat([x0, y1], dim=-1), logdet


class Permute(nn.Module):
    def __init__(self, permutation: list[int]):
        super().__init__()
        self.perm = permutation
        self.inv_perm = [0] * len(permutation)
        for i, v in enumerate(self.perm):
            self.inv_perm[v] = i

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        return inputs[:, self.perm], inputs.new_zeros(inputs.size(0))

    def inverse(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        return inputs[:, self.inv_perm], inputs.new_zeros(inputs.size(0))


class FlowSequential(nn.Sequential):
    """A sequential container for flows.
    In addition to a forward pass it implements an inverse pass, computes log probabilities and samples.
    # Adapted from https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
    """

    def __init__(self, *args, prior: torch.distributions.Distribution):
        super().__init__(*args)
        self.prior = prior

    def forward(self, inputs: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        inputs = inputs.view(inputs.size(0), -1)
        logdets = inputs.new_zeros(inputs.size(0))

        for module in self:
            flow_module = cast(Any, module)
            inputs, logdet = flow_module(inputs)
            if logdet is not None:
                logdets += logdet

        return inputs, logdets

    def inverse(self, inputs: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        inputs = inputs.view(inputs.size(0), -1)
        logdets = inputs.new_zeros(inputs.size(0))

        for module in reversed(self):
            flow_module = cast(Any, module)
            inputs, logdet = flow_module.inverse(inputs)
            if logdet is not None:
                logdets += logdet

        return inputs, logdets

    def log_prob(self, inputs: Tensor) -> Tensor:
        z, log_jacob = self.inverse(inputs)
        return self.prior.log_prob(z) + log_jacob

    def sample(self, num_samples: int = 1) -> Tensor:
        return self(self.prior.sample((num_samples,)))[0]


class RealNVP(FlowSequential):
    """PyTorch implementation of RealNVP [1] as used in [2], adapted from [3].
    [1] Density estimation using Real NVP, ICLR 2017.
    [2] Diverse Plausible Shape Completions from Ambiguous Depth Images, CVPR 2020.
    [3] # https://github.com/UM-ARM-Lab/probabilistic_shape_completion/blob/main/shape_completion_training/src/shape_completion_training/model/flow.py
    """

    def __init__(self, dim: int = 24, hidden_dim: int = 512, n_blocks: int = 8, batchnorm: bool = True):
        modules = list()
        for i in range(n_blocks):
            modules.append(ScaleShiftLayer(num_inputs=dim, num_masked=dim // 2, hidden_layers=[hidden_dim, hidden_dim]))

            if batchnorm:
                if i % 3 == 0:
                    modules.append(BatchNormFlow(dim))

            modules.append(Permute([i for i in reversed(range(dim))]))
        modules = list(reversed(modules[:-1]))

        prior = torch.distributions.MultivariateNormal(torch.zeros(dim).cuda(), torch.eye(dim).cuda())
        super().__init__(*modules, prior=prior)
