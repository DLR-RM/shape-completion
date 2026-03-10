from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from trimesh import Trimesh

from utils import resolve_dtype, setup_logger

from .transforms import Transform, apply_transforms

logger = setup_logger(__name__)


def sample_pointcloud(mesh: Trimesh, args: Any) -> tuple[np.ndarray, np.ndarray]:
    points, face_idx = mesh.sample(args.num_points, return_index=True)
    normals = mesh.face_normals[face_idx]

    dtype = resolve_dtype(args.precision)
    points = points.astype(dtype)
    normals = normals.astype(dtype)

    return points, normals


class ModelNet(Dataset):
    def __init__(
        self,
        split: str,
        data_dir: str | Path,
        categories: list[str] | None = None,
        transforms: dict[str, list[Transform]] | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.name = self.__class__.__name__

        data_dir = Path(data_dir)
        if categories is None:
            categories = [c.name for c in Path(data_dir).iterdir() if c.is_dir()]
        self.categories = categories
        logger.debug(f"Categories: {categories}")

        self.objects = list()
        self.metadata = dict()
        for index, category in enumerate(categories):
            if split == "train":
                for s in ["train", "val"]:
                    category_path = data_dir / category / s
                    objects = [o.name for o in category_path.iterdir() if o.is_file()]
                    self.objects.extend(
                        [
                            {"category": category, "name": obj, "path": category_path / obj}
                            for obj in objects
                            if ".npz" in obj
                        ]
                    )
            else:
                category_path = data_dir / category / split
                objects = [o.name for o in category_path.iterdir() if o.is_file()]
                self.objects.extend(
                    [
                        {"category": category, "name": obj, "path": category_path / obj}
                        for obj in objects
                        if ".npz" in obj
                    ]
                )

            self.metadata[category] = {"index": index}
            self.metadata[category]["name"] = category
            self.metadata[category]["size"] = len(objects)

        self.verbose = verbose
        self.data_dir = data_dir
        self.split = split

        inputs_trafos = transforms.get("inputs") if transforms else None
        self.transform = transforms.get("data") if transforms else None
        self.augmentation = transforms.get("aug") if transforms else None

        self.fields = dict()
        self.fields["inputs"] = lambda path: apply_transforms(
            data={None: np.load(path)["points"]}, transforms=inputs_trafos
        )[None]

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index: int) -> dict[str, Any]:
        obj_name = self.objects[index]["name"]
        obj_category = self.objects[index]["category"]
        obj_path = self.objects[index]["path"]
        data = {
            "index": index,
            "inputs.name": obj_name,
            "inputs.path": obj_path,
            "category.id": obj_category,
            "category.name": self.metadata[obj_category]["name"],
            "category.index": self.metadata[obj_category]["index"],
        }

        for name, field in self.fields.items():
            field_data = field((obj_path.parent / obj_path.stem).with_suffix(".npz"))

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[name] = v
                    else:
                        data[f"{name}.{k}"] = v
            else:
                data[name] = field_data

        data = apply_transforms(data, self.transform)

        return data
