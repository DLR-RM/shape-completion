"""
BOP Mugs:
---------

HB:
    - obj_000006.ply
LM:
    - obj_000007.ply (bad mesh)
TYO-L:
    - obj_000003.ply
    - obj_000004.ply
    - obj_000005.ply
    - obj_000006.ply
    - obj_000007.ply (no handle)
    - obj_000020.ply (weird depth)
    - obj_000021.ply
YCB-V:
    - obj_000014.ply (scenes 48 and 55)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from torch.utils.data import Dataset

from utils import setup_logger

from .fields import BOPField, MeshField, PointCloudField, PointsField
from .transforms import KeysToKeep, Transform, apply_transforms

logger = setup_logger(__name__)


class _BOPObject(TypedDict):
    category: int
    name: str


class BOP(Dataset):
    def __init__(
        self,
        name: str,
        split: str,
        data_dir: str,
        mesh_dir: str | None = None,
        sdf: bool = False,
        tsdf: float = 0,
        pointcloud_file: str | None = None,
        points_file: str | list[str] | None = None,
        normals_file: str | None = None,
        mesh_file: str | None = None,
        pose_file: str | None = None,
        objects: tuple[str, ...] | tuple[int, ...] | None = None,
        scene: str | None = None,
        camera: str | None = None,
        max_outlier_std: float = 0,
        max_correspondence_distance: float = 0,
        padding: float = 0.1,
        transforms: dict[str, list[Transform]] | None = None,
        verbose: bool = False,
    ):
        assert name in ["hb", "lm", "tyol", "ycbv"], f"Unknown BOP dataset: {name}."
        self.name = name
        self.split = split
        self.data_dir = data_dir
        self.camera = camera
        self.verbose = verbose

        if mesh_dir is None:
            raise ValueError("`mesh_dir` must be provided for BOP dataset initialization.")

        if objects is None:
            resolved_objects: list[_BOPObject] = [
                {"category": int(o[4:10].lstrip("0")), "name": o[4:10]}
                for o in os.listdir(os.path.join(data_dir, name, mesh_dir))
                if o.endswith(".ply")
            ]
        else:
            resolved_objects = [{"category": int(str(o).lstrip("0")), "name": str(o).zfill(6)} for o in objects]

        self.split = "_".join([split, camera]) if camera is not None else split
        self.scene = scene

        self.objects = list()
        file_ids = list()
        for obj in resolved_objects:
            obj_name = obj["name"]
            depth_dir = os.path.join(data_dir, name, self.split, obj_name if scene is None else scene, "depth")
            num_files = len([file for file in os.listdir(depth_dir) if file.endswith(".png")])
            start = 1 if name == "ycbv" else 0
            stop = num_files + 1 if name == "ycbv" else num_files
            file_ids.extend(np.arange(start, stop).tolist())
            self.objects.extend([obj] * num_files)

        self.metadata = {
            obj["category"]: {"index": index, "name": obj["name"]} for index, obj in enumerate(self.objects)
        }
        self.categories = [obj["category"] for obj in self.objects]

        inputs_trafos = transforms.get("inputs") if transforms else None
        points_trafos = transforms.get("points") if transforms else None
        data_trafos = transforms.get("data") if transforms else None
        points_source = points_file if points_file else mesh_file
        if points_source is None:
            points_source = "points.npz"
        pointcloud_source = pointcloud_file if pointcloud_file else mesh_file
        if pointcloud_source is None:
            pointcloud_source = "pointcloud.npz"

        self.fields = dict()
        self.fields["inputs"] = BOPField(
            file_ids=file_ids,
            camera=camera,
            max_outlier_std=max_outlier_std,
            max_correspondence_distance=max_correspondence_distance,
        )

        self.fields["points"] = PointsField(
            file=points_source,
            data_dir=None if points_file else mesh_dir,
            pose_file=f"mesh/{pose_file}",
            padding=padding,
            occ_from_sdf=not sdf,
            tsdf=tsdf,
            normalize=False,
            load_all_files=True,
        )

        self.fields["pointcloud"] = PointCloudField(
            file=pointcloud_source,
            data_dir="" if pointcloud_file else mesh_dir,
            normals_file=normals_file,
            load_normals=normals_file is not None,
            pose_file=f"mesh/{pose_file}",
        )

        if mesh_dir is not None:
            self.fields["mesh"] = MeshField(
                file=mesh_file, file_prefix="obj_", data_dir=mesh_dir, pose_file=pose_file, process=False
            )

        self.field_transforms: dict[str, list[Transform] | None] = {
            "inputs": inputs_trafos,
            "points": points_trafos,
            "pointcloud": None if transforms is None else transforms.get("pointcloud"),
            "mesh": None if transforms is None else transforms.get("mesh"),
        }

        self.transform = data_trafos

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item_time = time.perf_counter()
        obj_name = self.objects[index]["name"]
        category = self.objects[index]["category"]
        data = {"index": index, "obj_name": obj_name}

        for name, field in self.fields.items():
            field_data = field.load(
                obj_dir=os.path.join(
                    self.data_dir, self.name, self.split, obj_name if self.scene is None else self.scene
                ),
                index=index,
                category=category,
            )
            field_data = apply_transforms(field_data, self.field_transforms.get(name))

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[name] = v
                    else:
                        data[f"{name}.{k}"] = v
            else:
                data[name] = field_data

        transforms = self.transform
        if transforms is not None and bool(data.get("inputs.skip", False)):
            transforms = transforms[-2:]
        data = apply_transforms(data, transforms)

        if self.verbose:
            item = obj_name
            if "inputs.path" in data:
                item = "/".join(Path(str(data["inputs.path"])).parts[-4:-2])
            if self.transform and len(self.transform) >= 2 and isinstance(self.transform[-2], KeysToKeep):
                keys_to_keep = self.transform[-2].keys
            else:
                keys_to_keep = KeysToKeep().keys
            size_in_bytes = sum([sys.getsizeof(v) for k, v in data.items() if k in keys_to_keep])
            size_in_mb = round(size_in_bytes / 1024 / 1024, 2)
            logger.debug(
                f"Item {index}: {item} takes {time.perf_counter() - item_time:.4f}s (size: {size_in_mb:.2f} MB)."
            )
        return data
