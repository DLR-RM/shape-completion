from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from ..src import tabletop as tabletop_module
from ..src.tabletop import TableTop


def test_exclusion_manifest_preserves_original_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "tabletop.pile"
    split_dir = data_dir / "train"
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    for index in range(3):
        (images_dir / f"{index}.png").touch()
        (split_dir / f"{index}.hdf5").touch()

    class FakeCocoInstanceSegmentation:
        def __init__(self, data_dir: Path, **_: Any):
            self.root = str(data_dir / "train")

        def __len__(self) -> int:
            return 3

    monkeypatch.setattr(
        tabletop_module,
        "CocoInstanceSegmentation",
        FakeCocoInstanceSegmentation,
    )
    manifest = tmp_path / "empty_kinect_sim.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "tabletop.pile",
                "excluded": {"train": ["train/1.hdf5"]},
            }
        ),
        encoding="utf-8",
    )

    dataset = cast(Any, TableTop)(
        data_dir=data_dir,
        split="train",
        exclude_manifest=manifest,
    )

    assert len(dataset) == 2
    assert dataset._included_indices == [(0, 0, 0), (0, 2, 2)]


def test_exclusion_manifest_rejects_unknown_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "tabletop.pile"
    split_dir = data_dir / "train"
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "0.png").touch()
    (split_dir / "0.hdf5").touch()

    class FakeCocoInstanceSegmentation:
        def __init__(self, data_dir: Path, **_: Any):
            self.root = str(data_dir / "train")

        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(
        tabletop_module,
        "CocoInstanceSegmentation",
        FakeCocoInstanceSegmentation,
    )
    manifest = tmp_path / "empty_kinect_sim.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "tabletop.pile",
                "excluded": {"train": ["train/9.hdf5"]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown train frames"):
        cast(Any, TableTop)(
            data_dir=data_dir,
            split="train",
            exclude_manifest=manifest,
        )
