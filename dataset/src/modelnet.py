import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple, Type
from time import time

from torch.utils.data import Dataset
from trimesh import Trimesh
import numpy as np
from argparse import Namespace

from utils import setup_logger, load_mesh
from .fields import PointCloudField, MeshField
from .transforms import Transform, KeysToKeep, apply_transform, SubsamplePointcloud

logger = setup_logger(__name__)


def resolve_dtype(precision: int,
                  integer: bool = False,
                  unsigned: bool = False) -> Type[np.dtype]:
    if precision == 8:
        return np.uint8 if unsigned else np.int8
    elif precision == 16:
        return np.float16 if not integer else np.uint16 if unsigned else np.int16
    elif precision == 32:
        return np.float32 if not integer else np.uint32 if unsigned else np.int32
    elif precision == 64:
        return np.float64 if not integer else np.uint64 if unsigned else np.int64
    else:
        raise Exception(f"Invalid precision: {precision}.")


def sample_pointcloud(mesh: Trimesh, args: Any) -> Tuple[np.ndarray, np.ndarray]:
    points, face_idx = mesh.sample(args.num_points, return_index=True)
    normals = mesh.face_normals[face_idx]

    dtype = resolve_dtype(args.precision)
    points = points.astype(dtype)
    normals = normals.astype(dtype)

    return points, normals


class ModelNet(Dataset):
    def __init__(self,
                 split: str,
                 data_dir: Union[str, Path],
                 categories: List[str] = None,
                 cache: bool = False,
                 cache_inputs: bool = False,
                 cache_pointcloud: bool = False,
                 cache_mesh: bool = False,
                 transforms: Dict[str, List[Transform]] = None,
                 verbose: bool = False):
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
                    self.objects.extend([{"category": category,
                                          "name": obj,
                                          "path": category_path / obj} for obj in objects if ".npz" in obj])
            else:
                category_path = data_dir / category / split
                objects = [o.name for o in category_path.iterdir() if o.is_file()]
                self.objects.extend([{"category": category,
                                      "name": obj,
                                      "path": category_path / obj} for obj in objects if ".npz" in obj])

            self.metadata[category] = {"index": index}
            self.metadata[category]["name"] = category
            self.metadata[category]["size"] = len(objects)

        self.verbose = verbose
        self.data_dir = data_dir
        self.split = split
        self.cache = cache

        inputs_trafos = transforms.get("inputs") if transforms else None
        pcd_trafos = transforms.get("pointcloud") if transforms else None
        mesh_trafos = transforms.get("mesh") if transforms else None
        self.transform = transforms.get("data") if transforms else None
        self.augmentation = transforms.get("aug") if transforms else None

        self.fields = dict()
        self.fields["inputs"] = lambda path: apply_transform(data={None: np.load(path)["points"]},
                                                             transform=inputs_trafos)[None]

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        obj_name = self.objects[index]["name"]
        obj_category = self.objects[index]["category"]
        obj_path = self.objects[index]["path"]
        data = {"index": index,
                "inputs.name": obj_name,
                "inputs.path": obj_path,
                "category.id": obj_category,
                "category.name": self.metadata[obj_category]["name"],
                "category.index": self.metadata[obj_category]["index"]}

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

        data = apply_transform(data, self.transform, self.cache)

        return data
