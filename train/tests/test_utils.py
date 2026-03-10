from collections.abc import Iterable, Sized
from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from tqdm import tqdm

from ..src.utils import get_test_dataset

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def show() -> bool:
    return True


@pytest.fixture(scope="session")
def config_name() -> str:
    return "automatica_2023_kinect.yaml"


@pytest.fixture(scope="session")
def dirs() -> str:
    return "default"


@pytest.fixture(scope="session")
def cfg(config_name: str, dirs: str) -> DictConfig:
    with initialize(version_base=None, config_path="../../conf"):
        return compose(config_name, overrides=[f"dirs={dirs}"])


def test_get_test_dataset(cfg: DictConfig, show: bool):
    cfg.vis.show = show
    try:
        dataset = cast(Sized, get_test_dataset(cfg))
    except AssertionError as exc:
        if "No" in str(exc) and "files found" in str(exc):
            pytest.skip(str(exc))
        raise
    assert dataset is not None
    assert len(dataset) > 0
    for item in tqdm(cast(Iterable[dict[str, Any]], dataset)):
        if cfg.inputs.project:
            assert item["inputs"].ndim == 2 and item["inputs"].shape[1] == 3
        else:
            assert item["inputs"].ndim in [2, 3]
