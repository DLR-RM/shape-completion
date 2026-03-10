from abc import ABC, abstractmethod

import torch

from models.src.model import Model


class AutoregressiveModel(Model, ABC):
    @torch.inference_mode()
    @abstractmethod
    def generate(self, **kwargs):
        raise NotImplementedError(f"{self.name}.generate method is not implemented")
