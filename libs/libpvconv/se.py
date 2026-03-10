import torch
from torch import Tensor, nn


class Swish(nn.Module):
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class SE3d(nn.Module):
    def __init__(self, channel: int, reduction: int = 8, use_relu: bool = True):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(True) if use_relu else Swish(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)
