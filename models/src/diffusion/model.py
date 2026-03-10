from abc import ABC, abstractmethod

import torch
from torch import Tensor

from ..model import Model


def int2bit(x: Tensor, n: int = 8):
    mask = 2 ** torch.arange(n - 1, -1, -1).to(x.device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bit2int(x: Tensor):
    weights = 2 ** torch.arange(x.size(-1) - 1, -1, -1).to(x.device)
    return (x * weights).sum(-1).long()


class DiffusionModel(Model, ABC):
    @torch.inference_mode()
    @abstractmethod
    def generate(self, **kwargs):
        raise NotImplementedError(f"{self.name}.generate method is not implemented")
