import json
import os
import sys
from pathlib import Path
from time import time
from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import setup_logger
from .fields import (PointCloudField, PointsField, VoxelsField, BlenderProcRGBDField, ImageField, MeshField, EmptyField,
                     DepthField, PartNetField)
from .transforms import Transform, KeysToKeep, apply_transform

logger = setup_logger(__name__)


CORE_CATEGORIES = ["02691156",
                   "02828884",
                   "02933112",
                   "02958343",
                   "03001627",
                   "03211117",
                   "03636649",
                   "03691459",
                   "04090263",
                   "04256520",
                   "04379243",
                   "04401088",
                   "04530566"]


class ShapeNet(Dataset):
    def __init__(self,
                 split: str,
                 data_dir: Path,
                 partnet_dir: Optional[Path] = None,
                 inputs_dir: Optional[Path] = None,
                 points_dir_name: Optional[str] = None,
                 pointcloud_dir_name: Optional[str] = None,
                 mesh_dir_name: Optional[str] = None,
                 image_dir_name: Optional[str] = None,
                 cam_dir_name: Optional[str] = None,
                 categories: List[str] = None,
                 num_shards: int = 1,
                 files_per_shard: int = 1,
                 unscale: bool = False,
                 undistort: bool = False,
                 unrotate: bool = False,
                 sdf: bool = False,
                 tsdf: float = 0,
                 split_file: Optional[str] = None,
                 pointcloud_file: Optional[str] = None,
                 points_file: Optional[str] = None,
                 voxels_file: Optional[str] = None,
                 normals_file: Optional[str] = None,
                 mesh_file: Optional[str] = None,
                 pose_file: Optional[str] = None,
                 load_random_inputs: bool = False,
                 load_pointcloud: Union[bool, str] = False,
                 load_normals: bool = False,
                 load_points: bool = True,
                 load_all_points: bool = False,
                 load_random_points: bool = True,
                 load_surface_points: Optional[Union[bool, float]] = None,
                 load_meshes: bool = False,
                 load_cam: bool = False,
                 input_type: str = "pointcloud",
                 weighted: Union[bool, str] = False,
                 padding: float = 0.1,
                 cls_weight: float = 0,
                 seg_weight: float = 0,
                 precision: int = 16,
                 cache: bool = False,
                 cache_inputs: bool = False,
                 cache_points: bool = False,
                 cache_pointcloud: bool = False,
                 cache_mesh: bool = False,
                 from_hdf5: bool = False,
                 transforms: Dict[str, List[Transform]] = None,
                 verbose: bool = False):
        assert data_dir.is_dir(), f"{data_dir} is not a directory."
        if isinstance(load_pointcloud, str):
            assert load_pointcloud == "min_max_only", "load_pointcloud can only be 'min_max_only' if it is a string."

        self.verbose = verbose
        self._category_weights = None
        self._obj_weights = None
        if isinstance(weighted, str):
            self._obj_weights = weighted

        self.name = "ShapeNetCore.v2" if "v2" in data_dir.name else "ShapeNetCore.v1"
        if categories is None:
            if partnet_dir is None:
                categories = sorted([c.name for c in data_dir.iterdir() if c.is_dir()])
            else:
                categories = sorted([c.name for c in partnet_dir.iterdir() if c.is_dir()])
        self.categories = categories
        logger.debug(f"Categories: {categories}")

        with open(data_dir / "taxonomy.json", 'r') as f:
            taxonomy = json.load(f)

        self.metadata = dict()
        for index, category in enumerate(categories):
            self.metadata[category] = {"index": index}
            category_dict = next(item for item in taxonomy if item["synsetId"] == category)
            self.metadata[category]["name"] = category_dict["name"]
            self.metadata[category]["size"] = category_dict["numInstances"]

        self.objects = list()
        for category in categories:
            category_path = data_dir / category if partnet_dir is None else partnet_dir / category
            split_path = category_path / split_file if split_file else category_path / (split + ".lst")
            logger.debug(f"Loading split {split} for category {category} from {split_path}")
            with open(split_path, 'r') as f:
                objects = f.read().split('\n')

            if "" in objects:
                objects.remove("")

            logger.debug(f"Objects in {category}: {len(objects)}")

            self.objects.extend([{"category": category, "name": obj} for obj in objects if "#" not in obj])

        self.data_dir = data_dir
        self.mesh_dir = mesh_dir_name
        self.mesh_file = mesh_file
        self.split = split
        self.files_per_shard = files_per_shard

        num_objects = len(self.objects)
        num_images = 24
        if (input_type == "image" or load_cam) and self.split == "train":
            index = (0, num_images)
            self.objects *= num_images
        else:
            index = 0
            len_multiplier = 1 if load_random_inputs else files_per_shard * num_shards
            self.objects *= len_multiplier

        inputs_trafos = transforms.get("inputs") if transforms else None
        points_trafos = transforms.get("points") if transforms else None
        pcd_trafos = transforms.get("pointcloud") if transforms else None
        mesh_trafos = transforms.get("mesh") if transforms else None
        data_trafos = transforms.get("data") if transforms else None
        aug_trafos = transforms.get("aug") if transforms else None

        self.fields = dict()
        if input_type in ["pointcloud", "partial", "depth_like"]:
            self.fields["inputs"] = PointCloudField(file=pointcloud_file if pointcloud_file else mesh_file,
                                                    data_dir=inputs_dir if pointcloud_file else mesh_dir_name,
                                                    cam_dir=cam_dir_name if load_cam else None,
                                                    index=index,
                                                    num_cams=num_images,
                                                    normals_file=normals_file,
                                                    load_normals=load_normals,
                                                    cache=cache and cache_inputs,
                                                    from_hdf5=from_hdf5,
                                                    transform=inputs_trafos)
        elif input_type in ["render_depth", "render_pcd"]:
            self.fields["inputs"] = EmptyField()
        elif input_type in ["depth", "kinect"]:
            file_weights = self.obj_weights if split == "train" and isinstance(weighted, str) else None
            self.fields["inputs"] = DepthField(data_dir=inputs_dir,
                                               unscale=unscale,
                                               unrotate=unrotate,
                                               load_normals=load_normals,
                                               kinect=input_type == "kinect",
                                               path_suffix="models" if "v2" in self.name else "",
                                               num_objects=num_objects,
                                               num_files=files_per_shard,
                                               random_file=load_random_inputs,
                                               file_weights=file_weights,
                                               precision=precision,
                                               cache=cache and cache_inputs,
                                               from_hdf5=from_hdf5,
                                               transform=inputs_trafos)
        elif input_type in ["image", "rgbd", "depth_bp"]:
            if input_type == "image":
                image_field = ImageField(data_dir=image_dir_name,
                                         transform=inputs_trafos if input_type == "image" else None,
                                         index=index,
                                         num_images=num_images)
                self.fields["inputs"] = image_field
            elif input_type in ["rgbd", "depth_bp"]:
                self.fields["inputs"] = BlenderProcRGBDField(data_dir=os.path.join("" if inputs_dir is None else inputs_dir,
                                                                                   "blenderproc",
                                                                                   self.split),
                                                             unscale=unscale,
                                                             undistort=undistort,
                                                             num_objects=num_objects,
                                                             num_shards=num_shards,
                                                             files_per_shard=files_per_shard,
                                                             random_file=load_random_inputs,
                                                             random_shard=load_random_inputs,
                                                             input_type=input_type,
                                                             load_fixed=False,
                                                             cache=cache and cache_inputs,
                                                             transform=inputs_trafos)
        elif input_type == "none":
            pass
        else:
            raise ValueError(f"Unknown input type {input_type}")

        if load_points:
            sigmas = [0.001, 0.0015, 0.0025, 0.01, 0.015, 0.025, 0.05, 0.1, 0.15, 0.25]
            self.fields["points"] = PointsField(file=points_file if points_file else mesh_file,
                                                data_dir=points_dir_name if points_file else mesh_dir_name,
                                                params_dir=inputs_dir,
                                                path_suffix="models" if "v2" in self.name else "",
                                                padding=padding,
                                                occ_from_sdf=not sdf,
                                                tsdf=tsdf,
                                                sigmas=sigmas,
                                                load_surface_file=load_surface_points,
                                                normalize=False,
                                                load_all_files=load_all_points,
                                                load_random_file=load_random_points,
                                                cache=cache and cache_points,
                                                from_hdf5=from_hdf5,
                                                transform=points_trafos)

        if load_pointcloud:
            min_max_only = load_pointcloud == "min_max_only"
            self.fields["pointcloud"] = PointCloudField(file=pointcloud_file if pointcloud_file else mesh_file,
                                                        data_dir=pointcloud_dir_name if pointcloud_file else mesh_dir_name,
                                                        normals_file=normals_file,
                                                        load_normals=load_normals,
                                                        min_max_only=min_max_only,
                                                        cache=(cache and cache_pointcloud) or min_max_only,
                                                        from_hdf5=from_hdf5,
                                                        transform=pcd_trafos)

        if voxels_file is not None:
            self.fields["voxels"] = VoxelsField(file=voxels_file)

        if load_meshes or input_type in ["render_depth", "render_pcd"]:
            self.fields["mesh"] = MeshField(file=mesh_file,
                                            data_dir=mesh_dir_name,
                                            pose_file=pose_file,
                                            cache=cache and cache_mesh,
                                            from_hdf5=from_hdf5,
                                            transform=mesh_trafos)

        if partnet_dir is not None:
            self.fields["partnet"] = PartNetField(data_dir=partnet_dir,
                                                  transform=None)

        self.transform = data_trafos
        self.augmentation = aug_trafos
        self.start = 0
        self.end = len(self.objects)
        self.iteration = self.start - 1
        self.cache = cache
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight

    def __len__(self):
        return len(self.objects)

    def __iter__(self):
        return self

    def __next__(self):
        self.iteration += 1
        if self.iteration == self.end:
            self.iteration = self.start - 1
            raise StopIteration
        try:
            return self.__getitem__(self.iteration)
        except (FileNotFoundError, ValueError):
            return next(self)

    @property
    def category_weights(self) -> np.ndarray:
        if self._category_weights is None:
            inv_freqs = np.array([1 / self.metadata[category]["size"] for category in self.categories])
            inv_freqs /= inv_freqs.sum()
            category_weights = np.array([inv_freqs[self.categories.index(obj["category"])] for obj in self.objects])
            self._category_weights = category_weights
            logger.debug(f"Category weights: {set(self._category_weights)}")
        return self._category_weights

    @property
    def obj_weights(self) -> Dict[str, Dict[str, np.ndarray]]:
        if isinstance(self._obj_weights, str):
            path = Path(self._obj_weights).expanduser().resolve()
            logger.info(f"Loading object weights from {path}")

            def normalize(series):
                return series / series.sum()

            def sharpen(prob_vector, temperature: float = 0.1):
                powered_vector = np.power(prob_vector, 1 / temperature)
                return powered_vector / np.sum(powered_vector)

            df = pd.read_pickle(path)
            df_sorted = df.sort_values(by='path')
            df_sorted['loss'] = df_sorted.groupby(['object category', 'object name'])['loss'].transform(normalize)
            result_dict = df_sorted.groupby(['object category', 'object name'])['loss'].apply(list).to_dict()

            weights_dict = {}
            for (object_category, object_name), losses in result_dict.items():
                if object_category not in weights_dict:
                    weights_dict[object_category] = {}
                weights_dict[object_category][object_name] = sharpen(losses)

            """
            for object_category, inner_dict in weights_dict.items():
                for object_name, losses in inner_dict.items():
                    original_rows = df_sorted[(df_sorted['object category'] == object_category) &
                                              (df_sorted['object name'] == object_name) &
                                              (df_sorted['loss'].isin(losses))]

                    original_paths = original_rows['path'].values
                    for i, (loss, original_path) in enumerate(zip(losses, original_paths)):
                        path_number = original_path.split('/')[-1].split('.')[0]
                        assert str(i).zfill(5) == path_number, f"Path {original_path} does not end with '{str(i).zfill(5)}.png'"
            """

            self._obj_weights = weights_dict
        return self._obj_weights

    def get_category(self, index: int) -> str:
        return self.objects[index]["category"]

    def get_category_index(self, index: int, category: Optional[str] = None) -> int:
        if category is None:
            category = self.get_category(index)
        return self.metadata[category]["index"]

    def get_category_name(self, index: int, category: Optional[str] = None) -> str:
        if category is None:
            category = self.get_category(index)
        return self.metadata[category]["name"]

    def get_obj_name(self, index: int) -> str:
        return self.objects[index]["name"]

    def get_obj_dir(self, index: int, category: Optional[str] = None, obj_name: Optional[str] = None) -> str:
        if category is None:
            category = self.get_category(index)
        if obj_name is None:
            obj_name = self.get_obj_name(index)
        return os.path.join(self.data_dir, category, obj_name)

    def get_inputs_path(self,
                        index: int,
                        obj_dir: Optional[str] = None,
                        category: Optional[str] = None,
                        obj_name: Optional[str] = None) -> str:
        if obj_dir is None:
            if category is None:
                category = self.get_category(index)
            if obj_name is None:
                obj_name = self.get_obj_name(index)
            obj_dir = self.get_obj_dir(index, category, obj_name)
        if isinstance(self.fields["inputs"], (DepthField, BlenderProcRGBDField)):
            return self.fields["inputs"].get_depth_path(index, obj_dir)
        else:
            return obj_dir

    def _get_item(self, index: int) -> Dict[str, Any]:
        item_time = time()
        obj_name = self.get_obj_name(index)
        category = self.get_category(index)
        data = {"index": index,
                "category.id": category,
                "category.name": self.metadata[category]["name"],
                "category.index": self.metadata[category]["index"]}
        if self.cls_weight:
            data["cls_weight"] = self.cls_weight
        if self.seg_weight:
            data["seg_weight"] = self.seg_weight

        for name, field in self.fields.items():
            load_timer = time()
            if not self.cache or (self.cache and field.cache is not None):
                field_data = field.load(obj_dir=self.get_obj_dir(index, category, obj_name),
                                        index=index,
                                        category=data.get("inputs.file", self.get_category_index(index, category)))
                logger.debug(f"Field {field.name} ({name}) takes {time() - load_timer:.4f}s.")

                if isinstance(field_data, dict):
                    for k, v in field_data.items():
                        if k is None:
                            data[name] = v
                        else:
                            data[f"{name}.{k}"] = v
                else:
                    data[name] = field_data

        data = apply_transform(data, self.transform, self.cache)

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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            return self._get_item(index)
        except Exception as e:
            path = Path(self.get_inputs_path(index)).relative_to(self.data_dir)
            logger.error(f"Caught exception '{e}' for item {index}: {path}")
            logger.exception(e)
            raise e
