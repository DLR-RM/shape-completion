import math
from collections.abc import Iterable
from typing import Any, Protocol, cast

import lightning
import lightning.pytorch as pl
from torch import Tensor
from trimesh import Trimesh

from utils import default_on_exception, setup_logger, stdout_redirected
from visualize import Generator

from .every_n import EveryNCallback

logger = setup_logger(__name__)


class _GeneratorLike(Protocol):
    model: Any
    query_points: Tensor
    predict_colors: bool
    estimate_normals: bool

    def generate_mesh(self, batch: dict[str, list[str] | Tensor], **kwargs: Any) -> Trimesh | list[Trimesh]: ...


class GenerateMeshesCallback(EveryNCallback):
    def __init__(
        self,
        every_n_evals: int | Iterable[int] | None = 1,
        resolution: int = 128,
        padding: float = 0.1,
        points_batch_size: int | None = None,
        threshold: float = 0.5,
        precision: str | int | None = None,
        **generator_kwargs: Any,
    ) -> None:
        super().__init__(n_evals=every_n_evals)
        self.resolution = resolution
        self.points_batch_size = points_batch_size
        self.padding = padding
        self.threshold = threshold
        self.generator_kwargs = generator_kwargs
        self._generator: _GeneratorLike | None = None

        with stdout_redirected():
            # Keep runtime flexibility for legacy precision values while satisfying static checks.
            self.fabric = lightning.Fabric(precision=cast(Any, precision))

    @property
    def generator(self) -> _GeneratorLike:
        if self._generator is None:
            raise RuntimeError("GenerateMeshesCallback.setup must run before generate_batch.")
        return self._generator

    @default_on_exception(default=[Trimesh()])
    def generate_batch(self, batch: dict[str, list[str] | Tensor], **kwargs: Any) -> list[Trimesh]:
        with self.fabric.autocast():
            mesh = self.generator.generate_mesh(batch, **kwargs)
        return [mesh] if isinstance(mesh, Trimesh) else mesh

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        module = cast(Any, pl_module)
        model = module.model.orig_mod
        ema_model = getattr(module, "ema_model", None)
        if ema_model is not None:
            model = ema_model.module.orig_mod

        model_resolution = getattr(model, "resolution", None)
        if isinstance(model_resolution, int):
            self.resolution = model_resolution
            self.points_batch_size = self.resolution**3

        upsampling_steps = int(math.log2(self.resolution) - math.log2(32))
        upsample = self.resolution > 128 or (
            self.points_batch_size is not None and self.points_batch_size < self.resolution**3
        )
        self._generator = cast(
            _GeneratorLike,
            Generator(
                resolution=self.resolution,
                padding=self.padding,
                points_batch_size=self.points_batch_size,
                threshold=self.threshold,
                model=model,
                upsampling_steps=upsampling_steps if upsample else 0,
                **self.generator_kwargs,
            ),
        )
        super().setup(trainer, pl_module, stage)
