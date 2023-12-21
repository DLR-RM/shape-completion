import os
import sys
from pathlib import Path
from time import time
from typing import Union, Tuple, Dict, List, Any, Optional

from torch.utils.data import Dataset

from utils import setup_logger
from .fields import (PointCloudField, PointsField, VoxelsField, BlenderProcRGBDField, ImageField, MeshField, EmptyField,
                     DepthField)
from .transforms import Transform, KeysToKeep, apply_transform

logger = setup_logger(__name__)


class MeshDataset(Dataset):
    def __init__(self,
                 split: str,
                 data_dir: Path,
                 index_offset: int = 0,
                 inputs_dir: Optional[Path] = None,
                 points_dir: Optional[str] = None,
                 pointcloud_dir: Optional[str] = None,
                 mesh_dir: Optional[str] = None,
                 image_dir: Optional[str] = None,
                 cam_dir: Optional[str] = None,
                 objects: Union[Tuple[str], Tuple[int], Tuple[Path]] = None,
                 id_length: int = 3,
                 num_shards: int = 1,
                 files_per_shard: int = 1,
                 unscale: bool = False,
                 undistort: bool = False,
                 unrotate: bool = False,
                 sdf: bool = False,
                 tsdf: float = 0,
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
                 padding: float = 0.1,
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
        assert not (objects is None and id_length is None), "Either objects or 'id_length' must be specified."
        assert not (isinstance(objects, (tuple, list)) and isinstance(objects[0], int) and id_length is None),\
            "'id_length' must be specified if objects are ints."
        if isinstance(load_pointcloud, str):
            assert load_pointcloud == "min_max_only", "load_pointcloud can only be 'min_max_only' if it is a string."

        self.name = data_dir.name
        self.verbose = verbose
        if objects is None:
            self.objects = [{"category": obj[:id_length],
                             "name": obj} for obj in os.listdir(data_dir) if (data_dir / obj).is_dir()]
        else:
            if id_length is None:
                self.objects = list()
                id_length = len(str(len(objects)))
                for i, obj in enumerate(objects):
                    category = str(i).zfill(id_length)
                    name = Path(obj).parent.stem
                    mesh_file = Path(obj).name
                    self.objects.append({"category": category,
                                         "name": name})
            else:
                assert id_length > 0, "id_length must be greater than 0."
                selection = list()
                all_objects = os.listdir(data_dir)
                all_ids = [obj[:id_length] for obj in all_objects]
                for obj in objects:
                    if isinstance(obj, str):
                        if obj in all_objects:
                            selection.append(obj)
                        elif obj.zfill(id_length) in all_ids:
                            selection.append(all_objects[all_ids.index(obj.zfill(id_length))])
                    elif isinstance(obj, int) and str(obj).zfill(id_length) in all_ids:
                        selection.append(all_objects[all_ids.index(str(obj).zfill(id_length))])
                self.objects = [{"category": obj[:id_length], "name": obj} for obj in selection]

        self.metadata = dict()
        for i, obj in enumerate(self.objects):
            name = obj["name"] if id_length is None else obj["name"][id_length + 1:]
            self.metadata[obj["category"]] = {"index": i, "name": name}
        self.categories = [obj["category"] for obj in self.objects]

        self.data_dir = data_dir
        self.mesh_dir = mesh_dir
        self.mesh_file = mesh_file
        self.split = split
        self.files_per_shard = files_per_shard
        self.index_offset = index_offset

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
        data_trafos = transforms.get("data") if transforms else None
        aug_trafos = transforms.get("aug") if transforms else None

        self.fields = dict()
        if input_type in ["pointcloud", "partial", "depth_like"]:
            self.fields["inputs"] = PointCloudField(file=pointcloud_file if pointcloud_file else mesh_file,
                                                    data_dir=inputs_dir if pointcloud_file else mesh_dir,
                                                    cam_dir=cam_dir if load_cam else None,
                                                    index=index,
                                                    num_cams=num_images,
                                                    normals_file=normals_file,
                                                    load_normals=load_normals,
                                                    cache=cache and (cache_inputs or cache_pointcloud),
                                                    transform=inputs_trafos)
        elif input_type in ["render_depth", "render_pcd"]:
            self.fields["inputs"] = EmptyField()
        elif input_type in ["depth", "kinect"]:
            self.fields["inputs"] = DepthField(data_dir=inputs_dir,
                                               unscale=unscale,
                                               unrotate=unrotate,
                                               load_normals=load_normals,
                                               kinect=input_type == "kinect",
                                               num_objects=num_objects,
                                               num_files=files_per_shard,
                                               file_offset=index_offset,
                                               random_file=load_random_inputs,
                                               precision=precision,
                                               cache=cache and cache_inputs,
                                               from_hdf5=from_hdf5,
                                               transform=inputs_trafos)
        elif input_type in ["image", "rgbd", "depth_bp"]:
            if input_type == "image":
                image_field = ImageField(data_dir=image_dir,
                                         transform=inputs_trafos if input_type == "image" else None,
                                         index=index,
                                         num_images=num_images)
                self.fields["inputs"] = image_field
            elif input_type in ["rgbd", "depth_bp"]:
                inputs_dir = os.path.join(inputs_dir, "blenderproc", self.split) if inputs_dir else os.path.join(
                    "blenderproc", self.split)
                self.fields["inputs"] = BlenderProcRGBDField(data_dir=inputs_dir,
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
                                                data_dir=points_dir if points_file else mesh_dir,
                                                params_dir=inputs_dir,
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
                                                        data_dir=pointcloud_dir if pointcloud_file else mesh_dir,
                                                        normals_file=normals_file,
                                                        load_normals=load_normals,
                                                        min_max_only=min_max_only,
                                                        cache=(cache and cache_pointcloud) or min_max_only,
                                                        from_hdf5=from_hdf5,
                                                        transform=transforms.get("pointcloud"))

        if voxels_file is not None:
            self.fields["voxels"] = VoxelsField(file=voxels_file)

        if load_meshes or input_type in ["render_depth", "render_pcd"]:
            self.fields["mesh"] = MeshField(file=mesh_file,
                                            data_dir=mesh_dir,
                                            pose_file=pose_file,
                                            cache=cache and cache_mesh,
                                            from_hdf5=from_hdf5,
                                            transform=transforms.get("mesh"))

        self.transform = data_trafos
        self.augmentation = aug_trafos
        self.cache = cache

    def __len__(self):
        return len(self.objects)

    def get_category(self, index: int) -> str:
        return self.objects[index]["category"]

    def get_category_index(self, index: int, category: Optional[str] = None) -> int:
        if category is None:
            category = self.get_category(index)
        return self.metadata[category]["index"]

    def get_obj_name(self, index: int) -> str:
        return self.objects[index]["name"]

    def get_obj_dir(self, index: int, obj_name: Optional[str] = None) -> str:
        if obj_name is None:
            obj_name = self.get_obj_name(index)
        return os.path.join(self.data_dir, obj_name)

    def get_inputs_path(self, index: int, obj_dir: Optional[str] = None, obj_name: Optional[str] = None) -> str:
        if obj_dir is None:
            if obj_name is None:
                obj_name = self.get_obj_name(index)
            obj_dir = self.get_obj_dir(index, obj_name)
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

        for name, field in self.fields.items():
            load_timer = time()
            if not self.cache or (self.cache and field.cache is not None):
                field_data = field.load(obj_dir=self.get_obj_dir(index, obj_name),
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
            if self.transform and isinstance(self.transform[-2], KeysToKeep):
                keys_to_keep = self.transform[-2].keys
            else:
                keys_to_keep = KeysToKeep().keys
            size_in_bytes = sum([sys.getsizeof(v) for k, v in data.items() if k in keys_to_keep])
            size_in_mb = round(size_in_bytes / 1024 / 1024, 2)
            logger.debug(f"Item {index}: {obj_name} takes {time() - item_time:.4f}s (size: {size_in_mb:.2f} MB).")
        return data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            return self._get_item(index)
        except Exception as e:
            path = Path(self.get_inputs_path(index)).relative_to(self.data_dir)
            logger.error(f"Caught exception '{e}' for item {index}: {path}")
            logger.exception(e)
            raise e
