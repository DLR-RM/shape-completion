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

import os
import sys
from pathlib import Path
from time import time
from typing import Tuple, Union, Dict, Any, List, Optional

import numpy as np
from torch.utils.data import Dataset

from utils import setup_logger
from .fields import PointCloudField, PointsField, BOPField, MeshField
from .transforms import Transform, KeysToKeep, apply_transform

logger = setup_logger(__name__)


class BOP(Dataset):
    def __init__(self,
                 name: str,
                 split: str,
                 data_dir: str,
                 mesh_dir: Optional[str] = None,
                 sdf: bool = False,
                 tsdf: float = 0,
                 pointcloud_file: Optional[str] = None,
                 points_file: Union[str, List[str]] = None,
                 normals_file: Optional[str] = None,
                 mesh_file: Optional[str] = None,
                 pose_file: Optional[str] = None,
                 objects: Union[Tuple[str], Tuple[int]] = None,
                 scene: Optional[str] = None,
                 camera: Optional[str] = None,
                 max_outlier_std: float = 0,
                 max_correspondence_distance: float = 0,
                 padding: float = 0.1,
                 cache: bool = True,
                 transforms: Dict[str, List[Transform]] = None,
                 verbose: bool = False):
        assert name in ["hb", "lm", "tyol", "ycbv"], f"Unknown BOP dataset: {name}."
        self.name = name
        self.split = split
        self.data_dir = data_dir
        self.camera = camera
        self.verbose = verbose
        self.cache = cache

        if objects is None:
            objects = [{"category": int(o[4:10].lstrip('0')), "name": o[4:10]} for o in
                       os.listdir(os.path.join(data_dir, name, mesh_dir)) if o.endswith(".ply")]
        else:
            objects = [{"category": int(str(o).lstrip('0')), "name": str(o).zfill(6)} for o in objects]

        self.split = '_'.join([split, camera]) if camera is not None else split
        self.scene = scene

        self.objects = list()
        file_ids = list()
        for obj in objects:
            obj_name = obj["name"]
            depth_dir = os.path.join(data_dir, name, self.split, obj_name if scene is None else scene, "depth")
            num_files = len([file for file in os.listdir(depth_dir) if file.endswith(".png")])
            start = 1 if name == "ycbv" else 0
            stop = num_files + 1 if name == "ycbv" else num_files
            file_ids.extend(np.arange(start, stop).tolist())
            self.objects.extend([obj] * num_files)

        self.metadata = {obj["category"]: {"index": index, "name": obj["name"]} for index, obj in
                         enumerate(self.objects)}
        self.categories = [obj["category"] for obj in self.objects]

        inputs_trafos = transforms.get("inputs") if transforms else None
        points_trafos = transforms.get("points") if transforms else None
        data_trafos = transforms.get("data") if transforms else None

        self.fields = dict()
        self.fields["inputs"] = BOPField(file_ids=file_ids,
                                         camera=camera,
                                         max_outlier_std=max_outlier_std,
                                         max_correspondence_distance=max_correspondence_distance,
                                         transform=inputs_trafos)

        self.fields["points"] = PointsField(file=points_file if points_file else mesh_file,
                                            data_dir=None if points_file else mesh_dir,
                                            pose_file=f"mesh/{pose_file}",
                                            padding=padding,
                                            occ_from_sdf=not sdf,
                                            tsdf=tsdf,
                                            cache=cache,
                                            normalize=False,
                                            load_all_files=True,
                                            transform=points_trafos)

        self.fields["pointcloud"] = PointCloudField(file=pointcloud_file if pointcloud_file else mesh_file,
                                                    data_dir="" if pointcloud_file else mesh_dir,
                                                    normals_file=normals_file,
                                                    load_normals=normals_file is not None,
                                                    pose_file=f"mesh/{pose_file}",
                                                    cache=cache,
                                                    transform=None if transforms is None else transforms.get(
                                                        "pointcloud"))

        if mesh_dir is not None:
            self.fields["mesh"] = MeshField(file=mesh_file,
                                            file_prefix="obj_",
                                            data_dir=mesh_dir,
                                            pose_file=pose_file,
                                            cache=cache,
                                            process=False,
                                            transform=None if transforms is None else transforms.get("mesh"))

        self.transform = data_trafos

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_time = time()
        obj_name = self.objects[index]["name"]
        category = self.objects[index]["category"]
        data = {"index": index, "obj_name": obj_name}

        for name, field in self.fields.items():
            field_data = field.load(obj_dir=os.path.join(self.data_dir,
                                                         self.name,
                                                         self.split,
                                                         obj_name if self.scene is None else self.scene),
                                    index=index,
                                    category=category)

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[name] = v
                    else:
                        data[f"{name}.{k}"] = v
            else:
                data[name] = field_data

        transform = self.transform[-2:] if data["inputs.skip"] else self.transform
        data = apply_transform(data, transform, self.cache)

        if self.verbose:
            item = obj_name
            if "inputs.path" in data:
                item = '/'.join(Path(data["inputs.path"]).parts[-4:-2])
            if self.transform and isinstance(self.transform[-2], KeysToKeep):
                keys_to_keep = self.transform[-2].keys
            else:
                keys_to_keep = KeysToKeep().keys
            size_in_bytes = sum([sys.getsizeof(v) for k, v in data.items() if k in keys_to_keep])
            size_in_mb = round(size_in_bytes / 1024 / 1024, 2)
            logger.debug(f"Item {index}: {item} takes {time() - item_time:.4f}s (size: {size_in_mb:.2f} MB).")
        return data
