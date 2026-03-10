from collections.abc import Iterable
from typing import Any

import lightning.pytorch as pl


class EveryNCallback(pl.Callback):
    def __init__(
        self,
        n_steps: int | None = None,
        n_epochs: int | None = None,
        n_evals: int | Iterable[int] | None = None,
        first: bool = True,
        last: bool = True,
    ) -> None:
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        if isinstance(n_evals, int):
            self.n_evals: int | list[int] | None = n_evals
        elif n_evals is None:
            self.n_evals = None
        else:
            self.n_evals = list(n_evals)
        self.first = first
        self.last = last
        self.eval_count = 0

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.eval_count = 0

    def run(self, trainer: pl.Trainer) -> bool:
        if self.n_steps and trainer.global_step % self.n_steps == 0:
            if not self.first and trainer.global_step == 0:
                return False
            if not self.last and trainer.global_step == trainer.max_steps:
                return False
            return True
        if self.n_epochs and trainer.current_epoch % self.n_epochs == 0:
            if not self.first and trainer.current_epoch == 0:
                return False
            if not self.last and trainer.current_epoch == trainer.max_epochs:
                return False
            return True
        if self.n_evals:
            if isinstance(self.n_evals, int) and self.eval_count % self.n_evals == 0:
                if not self.first and self.eval_count == 0:
                    return False
                return True
            elif isinstance(self.n_evals, list) and self.eval_count in self.n_evals:
                if not self.first and self.eval_count == 0:
                    return False
                return True
        return False

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.eval_count += 1

    def state_dict(self) -> dict[str, int]:
        return {"eval_count": self.eval_count}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.eval_count = int(state_dict["eval_count"])
