from collections.abc import Callable, Sequence, Sized
from functools import partial
from typing import Any, Protocol, cast

import lightning.pytorch as pl
import torch
from joblib import cpu_count
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataset import SharedDataLoader
from utils import setup_logger

logger = setup_logger(__name__)


class WeightedDataset(Protocol):
    category_weights: Sequence[float] | torch.Tensor

    def __len__(self) -> int: ...


def _debug_level_1(message: str) -> None:
    debug_level_1 = cast(Any, getattr(logger, "debug_level_1", None))
    if callable(debug_level_1):
        debug_level_1(message)
    else:
        logger.debug(message)


def _debug_level_2(message: str) -> None:
    debug_level_2 = cast(Any, getattr(logger, "debug_level_2", None))
    if callable(debug_level_2):
        debug_level_2(message)
    else:
        logger.debug(message)


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: Dataset,
        val: Dataset | None = None,
        test: Dataset | None = None,
        batch_size: int = 32,
        batch_size_val: int = 32,
        num_workers: int = cpu_count(),
        num_workers_val: int = cpu_count(),
        shuffle_val: bool = False,
        overfit: bool = False,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        weighted: bool = False,
        seed: int = 0,
        collate_fn: Callable[..., Any] | None = None,
        cache: bool = False,
        hash_items: bool = False,
        share_memory: bool = False,
    ):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.num_workers_val = num_workers_val
        self.shuffle_val = shuffle_val
        self.overfit = overfit
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.weighted = weighted
        self.generator = torch.Generator().manual_seed(seed)
        self.collate_fn = collate_fn
        self.cache = cache
        self.hash_items = hash_items
        self.share_memory = share_memory

    def train_dataloader(self):
        sampler: WeightedRandomSampler | None = None
        if self.weighted:
            weighted_train = cast(WeightedDataset, self.train)
            sampler = WeightedRandomSampler(
                weights=cast(Any, weighted_train.category_weights),
                num_samples=len(cast(Sized, weighted_train)),
                generator=self.generator,
            )
        if sampler is not None:
            _debug_level_1("Using weighted sampler for training")
        loader: Callable[..., Any] = DataLoader
        if self.cache:
            loader = cast(
                Callable[..., Any],
                partial(SharedDataLoader, hash_items=self.hash_items, share_arrays=self.share_memory),
            )
        return loader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False if self.weighted else True,
            sampler=sampler,
            pin_memory=self.pin_memory,
            generator=self.generator,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            persistent_workers=True if self.num_workers else False,
        )

    def val_dataloader(self):
        dataset = self.train if self.overfit else self.val
        if dataset is None:
            raise ValueError("Validation dataset is required when overfit=False.")
        loader: Callable[..., Any] = DataLoader
        if self.cache:
            loader = cast(
                Callable[..., Any],
                partial(SharedDataLoader, hash_items=self.hash_items, share_arrays=self.share_memory),
            )
        return loader(
            dataset,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers_val,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle_val,
            prefetch_factor=self.prefetch_factor if self.num_workers_val else None,
            pin_memory=self.pin_memory,
            generator=self.generator,
            persistent_workers=True if self.num_workers_val else False,
        )

    def test_dataloader(self):
        if self.test is None:
            raise ValueError("Test dataset is required for test_dataloader.")
        prefetch_factor = 1 if self.test.__class__.__name__ == "TestDataset" else self.prefetch_factor
        _debug_level_2(f"Using prefetch factor {prefetch_factor} for testing")
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=self.num_workers,
            prefetch_factor=prefetch_factor if self.num_workers else None,
            pin_memory=self.pin_memory,
            generator=self.generator,
        )
