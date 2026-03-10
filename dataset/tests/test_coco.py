from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torchvision.datasets as datasets
from pycocotools.coco import COCO
from torchvision.transforms import v2 as T

from ..src.coco import CocoInstanceSegmentation

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def coco_root() -> Path:
    root = os.environ.get("COCO_ROOT")
    if not root:
        pytest.skip("Set COCO_ROOT to run COCO integration tests.")
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        pytest.skip(f"COCO_ROOT does not exist: {path}")
    return path


def test_coco_detection_loads(coco_root: Path):
    image_path = coco_root / "val2017"
    annotation_path = coco_root / "annotations" / "instances_val2017.json"
    if not image_path.is_dir() or not annotation_path.is_file():
        pytest.skip("COCO val2017 images/annotations not found under COCO_ROOT.")

    coco_dataset = datasets.CocoDetection(root=str(image_path), annFile=str(annotation_path))
    assert len(coco_dataset) > 0
    image, target = coco_dataset[0]
    assert image is not None
    assert isinstance(target, list)


def test_coco_api_category_lookup(coco_root: Path):
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    if not ann_file.is_file():
        pytest.skip("COCO annotations not found under COCO_ROOT.")

    coco = COCO(str(ann_file))
    cat_ids = coco.getCatIds(catNms=["cat"])
    img_ids = coco.getImgIds(catIds=cat_ids)
    assert isinstance(img_ids, list)


def test_coco_instance_segmentation_dataset(coco_root: Path):
    transforms = T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = CocoInstanceSegmentation(data_dir=coco_root, split="val", transforms=transforms)
    if len(dataset) == 0:
        pytest.skip("COCO dataset wrapper returned an empty split.")

    item = dataset[0]
    assert "inputs" in item
    assert "inputs.masks" in item
    assert "inputs.labels" in item
