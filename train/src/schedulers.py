import math

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """Adapted from: https://github.com/karpathy/nanoGPT/blob/master/train.py#L227C1-L239C53"""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int,
        total_iters: int,
        warmup_start_lr: float = 0.0,
        min_lr: float | list[float] = 0.0,
        last_epoch: int = -1,  # More like "current_iter"
    ) -> None:
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.warmup_start_lr = warmup_start_lr
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

        if isinstance(min_lr, list):
            if len(min_lr) != len(self.initial_lrs):
                raise ValueError(f"Expected {len(self.initial_lrs)} values for min_lr, but got {len(min_lr)}")
            self.min_lrs = min_lr
        else:
            self.min_lrs = []
            if self.initial_lrs:
                min_lr_ratio = min_lr / max(self.initial_lrs)
                self.min_lrs = [base_lr * min_lr_ratio for base_lr in self.initial_lrs]

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        # linear warmup for warmup_iters steps
        if self.last_epoch < self.warmup_iters:
            if self.warmup_iters > 0:
                lr_scale = self.last_epoch / self.warmup_iters
                return [
                    self.warmup_start_lr + lr_scale * (base_lr - self.warmup_start_lr) for base_lr in self.initial_lrs
                ]
            else:  # warmup_iters is 0, so we are at the initial learning rate
                return self.initial_lrs

        # return min learning rate at the end of training
        if self.last_epoch >= self.total_iters:
            return self.min_lrs

        # 2. cosine annealing decay
        # Handle case where total_iters == warmup_iters to avoid division by zero
        if self.total_iters == self.warmup_iters:
            # At the end of warmup, LR should be base_lr, which is what a decay_ratio of 0 gives.
            # But since we are also at total_iters, it should be min_lr. Let's return min_lr.
            return self.min_lrs

        decay_ratio = (self.last_epoch - self.warmup_iters) / (self.total_iters - self.warmup_iters)
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        new_lrs = []
        for base_lr, min_lr in zip(self.initial_lrs, self.min_lrs, strict=True):
            new_lr = min_lr + coeff * (base_lr - min_lr)
            new_lrs.append(new_lr)
        return new_lrs
