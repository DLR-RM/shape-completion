from typing import Any

import pytest
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ..src import data_module as data_module_module
from ..src.data_module import LitDataModule


class ToyDataset(Dataset[int]):
    def __init__(self, size: int) -> None:
        self._items = list(range(size))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> int:
        return self._items[index]


class WeightedToyDataset(ToyDataset):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.category_weights = torch.ones(size, dtype=torch.float32)


class TestDataset(ToyDataset):
    __test__ = False

    pass


class FakeLoader:
    def __init__(self, dataset: Dataset[int], **kwargs: Any) -> None:
        self.dataset = dataset
        self.kwargs = kwargs
        self.prefetch_factor = kwargs.get("prefetch_factor")


def test_train_dataloader_uses_weighted_sampler() -> None:
    dm = LitDataModule(
        train=WeightedToyDataset(6),
        weighted=True,
        batch_size=2,
        num_workers=0,
    )

    loader = dm.train_dataloader()

    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert loader.batch_size == 2


def test_val_dataloader_uses_train_in_overfit_mode() -> None:
    train = ToyDataset(4)
    val = ToyDataset(3)
    dm = LitDataModule(
        train=train,
        val=val,
        overfit=True,
        batch_size_val=3,
        num_workers_val=0,
    )

    loader = dm.val_dataloader()

    assert loader.dataset is train
    assert loader.batch_size == 3


def test_val_dataloader_requires_val_dataset() -> None:
    dm = LitDataModule(train=ToyDataset(4), val=None, overfit=False, num_workers_val=0)

    with pytest.raises(ValueError, match="Validation dataset is required"):
        dm.val_dataloader()


def test_test_dataloader_requires_test_dataset() -> None:
    dm = LitDataModule(train=ToyDataset(4), test=None, num_workers=0)

    with pytest.raises(ValueError, match="Test dataset is required"):
        dm.test_dataloader()


def test_test_dataloader_prefetch_for_named_test_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module_module, "DataLoader", FakeLoader)
    dm = LitDataModule(train=ToyDataset(4), test=TestDataset(2), num_workers=2, prefetch_factor=5)

    loader = dm.test_dataloader()

    assert isinstance(loader, FakeLoader)
    assert loader.prefetch_factor == 1


def test_cache_loader_uses_shared_data_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_shared_data_loader(
        dataset: Dataset[int],
        *,
        hash_items: bool = False,
        share_arrays: bool = False,
        **kwargs: Any,
    ) -> FakeLoader:
        calls["dataset"] = dataset
        calls["hash_items"] = hash_items
        calls["share_arrays"] = share_arrays
        calls["kwargs"] = kwargs
        return FakeLoader(dataset, **kwargs)

    monkeypatch.setattr(data_module_module, "SharedDataLoader", fake_shared_data_loader)
    dm = LitDataModule(
        train=ToyDataset(4),
        cache=True,
        hash_items=True,
        share_memory=True,
        num_workers=0,
    )

    loader = dm.train_dataloader()

    assert isinstance(loader, FakeLoader)
    assert calls["dataset"] is dm.train
    assert calls["hash_items"] is True
    assert calls["share_arrays"] is True
