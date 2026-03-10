from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import default_collate
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection

from .tv_transforms import CameraIntrinsic

COCO_NUM_CLASSES = 80
COCO_CATEGORIES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]
COCO_ID_TO_CLASS_IDX = {category_id: i for i, category_id in enumerate(COCO_CATEGORIES)}


def max_size_center_pad(
    images: list[tv_tensors.Image], targets: dict[str, list[Any]]
) -> tuple[Tensor, dict[str, list[Any]]]:
    """
    Pads images, masks and boxes to the maximum size in the batch.

    Returns a tuple of stacked tensors: (padded_images, padded_masks, padded_boxes).
    """
    # Check if all images in the batch have the same size. If so, no padding is needed.
    if all(img.shape == images[0].shape for img in images):
        return torch.stack([cast(Tensor, img) for img in images]), targets

    # If sizes differ, find the max dimensions and pad
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded_images: list[Tensor] = []
    for i, img in enumerate(images):
        h, w = img.shape[-2:]

        pad_w = max_w - w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        pad_h = max_h - h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        padded_images.append(cast(Tensor, F.pad(img, padding)))

        for key, values in targets.items():
            value = values[i]
            if isinstance(value, (tv_tensors.Mask, tv_tensors.BoundingBoxes, CameraIntrinsic)):
                targets[key][i] = F.pad(value, padding)

    return torch.stack(padded_images), targets


def coco_collate(batch: list[dict[str, Any]], list_keys: set[str] | None = None) -> dict[str, Any]:
    """
    Custom collate function that uses a helper to pad variable-sized images and masks.
    """
    list_keys = {
        "category.id",
        "category.name",
        "category.index",
        "inputs.info",
        "inputs.boxes",
        "inputs.masks",
    } | (list_keys or set())
    if torch.is_tensor(batch[0]["inputs"]):
        list_keys.add("inputs")

    default_collate_part = []
    list_part = {key: [] for key in list_keys if key in batch[0]}

    for item in batch:
        collated_sample = {}
        for key, value in item.items():
            if key in list_keys:
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                list_part[key].append(value)
            else:
                collated_sample[key] = value
        default_collate_part.append(collated_sample)

    try:
        final_batch = default_collate(default_collate_part)
    except RuntimeError:
        for item in default_collate_part:
            for key, value in item.items():
                if isinstance(value, (Tensor, np.ndarray)):
                    print(key, value.shape, type(value))
                elif isinstance(value, list):
                    print(key, len(value), type(value[0]))
                else:
                    print(key, value)
        raise

    if "inputs" in list_part:
        images = list_part.pop("inputs")
        images, list_part = max_size_center_pad(images=images, targets=list_part)
        final_batch["inputs"] = images
    final_batch.update(list_part)

    return final_batch


class CocoInstanceSegmentation(CocoDetection):
    def __init__(
        self,
        data_dir: Path,
        split: Literal["train", "val", "test"] = "train",
        train_dir: str = "train2017",
        val_dir: str = "val2017",
        test_dir: str = "val2017",
        ann_dir: str = "annotations",
        train_ann_file: str = "instances_train2017.json",
        val_ann_file: str = "instances_val2017.json",
        test_ann_file: str = "instances_val2017.json",
        transforms: Callable | None = None,
    ):
        self.name = self.__class__.__name__

        if split == "train":
            root = data_dir / train_dir
            annFile = data_dir / ann_dir / train_ann_file
        elif split == "val":
            root = data_dir / val_dir
            annFile = data_dir / ann_dir / val_ann_file
        elif split == "test":
            root = data_dir / test_dir
            annFile = data_dir / ann_dir / test_ann_file
        else:
            raise ValueError(f"Invalid split: {split}. Choose from 'train', 'val', or 'test'.")

        super().__init__(root=str(root), annFile=str(annFile), transforms=transforms)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # 1. Load image and annotations from the base class
        id = self.ids[index]
        images = self.coco.loadImgs(id)
        image = images[0]
        path = Path(self.root) / image["file_name"]
        image = tv_tensors.Image(Image.open(path).convert("RGB"))
        height, width = image.shape[-2:]
        target_list = self._load_target(id)

        # 2. Convert COCO annotations to the required format
        masks = []
        boxes = []
        labels = []
        info = []
        category_ids = []

        for ann in target_list:
            # Segmentation mask
            masks.append(self.coco.annToMask(ann))

            # Bounding box [x, y, width, height]
            boxes.append(ann["bbox"])

            # Label
            category_id = ann["category_id"]
            category_ids.append(category_id)
            labels.append(COCO_ID_TO_CLASS_IDX.get(category_id, category_id))

            # Additional information
            info.append({k: v for k, v in ann.items() if k not in ["bbox", "category_id", "segmentation"]})

        # 3. Create tv_tensors for boxes and masks
        if boxes:
            # Create BoundingBoxes object
            bounding_boxes = cast(Any, tv_tensors.BoundingBoxes)
            boxes_tensor = bounding_boxes(
                torch.tensor(boxes, dtype=torch.float32),
                format="XYWH",
                canvas_size=(height, width),
            )
            # Create Mask object
            masks_tensor = tv_tensors.Mask(torch.from_numpy(np.array(masks, dtype=bool)))
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            # Handle cases with no annotations
            bounding_boxes = cast(Any, tv_tensors.BoundingBoxes)
            boxes_tensor = bounding_boxes(
                torch.empty((0, 4), dtype=torch.float32),
                format="XYWH",
                canvas_size=(height, width),
            )
            masks_tensor = tv_tensors.Mask(torch.empty((0, height, width), dtype=torch.bool))
            labels_tensor = torch.empty((0,), dtype=torch.long)
            info = [{}]  # Empty info for no annotations

        # 4. Assemble the final target dictionary
        target = {
            "boxes": boxes_tensor,
            "masks": masks_tensor,
            "labels": labels_tensor,
        }

        # 5. Apply transforms to image, target, and update info
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            height, width = image.shape[-2:]
            for i, elem in enumerate(info):
                if "area" in elem:
                    elem["area"] = target["masks"][i].sum().item()

        category_names = [cat["name"] for cat in self.coco.loadCats([int(cat_id) for cat_id in category_ids])]

        return {
            "index": index,
            "category.id": category_ids,
            "category.name": category_names,
            "category.index": labels,
            "inputs.name": path.name,
            "inputs.path": str(path),
            "inputs": image,
            "inputs.width": width,
            "inputs.height": height,
            "inputs.boxes": target["boxes"],
            "inputs.masks": target["masks"],
            "inputs.labels": target["labels"],
            "inputs.info": info,
            "inputs.skip": not boxes,
        }
