import copy
import json
import os
import sys
import time
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import setup_logger

from .fields import (
    BlenderProcRGBDField,
    DepthField,
    DTUField,
    EmptyField,
    Field,
    ImageField,
    IndexField,
    MeshField,
    MixedField,
    PartNetField,
    PointCloudField,
    PointsField,
    RandomField,
    VoxelsField,
)
from .transforms import KeysToKeep, Transform, apply_transforms

logger = setup_logger(__name__)


def _log_debug_level_1(message: str) -> None:
    log_fn = getattr(logger, "debug_level_1", None)
    if callable(log_fn):
        log_fn(message)
        return
    logger.debug(message)


def _log_debug_level_2(message: str) -> None:
    log_fn = getattr(logger, "debug_level_2", None)
    if callable(log_fn):
        log_fn(message)
        return
    logger.debug(message)


CORE_CATEGORIES = [
    "02691156",
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
    "04530566",
]

CATEGORIES_MAP = {
    "02691156": 0,  # airplane
    "02747177": 1,
    "02773838": 2,
    "02801938": 3,
    "02808440": 4,
    "02818832": 5,
    "02828884": 6,
    "02843684": 7,
    "02871439": 8,
    "02876657": 9,
    "02880940": 10,
    "02924116": 11,
    "02933112": 12,
    "02942699": 13,
    "02946921": 14,
    "02954340": 15,
    "02958343": 16,  # car
    "02992529": 17,
    "03001627": 18,  # chair
    "03046257": 19,
    "03085013": 20,
    "03207941": 21,
    "03211117": 22,
    "03261776": 23,
    "03325088": 24,
    "03337140": 25,
    "03467517": 26,
    "03513137": 27,
    "03593526": 28,
    "03624134": 29,
    "03636649": 30,  # lamp
    "03642806": 31,
    "03691459": 32,
    "03710193": 33,
    "03759954": 34,
    "03761084": 35,
    "03790512": 36,
    "03797390": 37,
    "03928116": 38,
    "03938244": 39,
    "03948459": 40,
    "03991062": 41,
    "04004475": 42,
    "04074963": 43,
    "04090263": 44,  # rifle
    "04099429": 45,
    "04225987": 46,
    "04256520": 47,  # sofa
    "04330267": 48,
    "04379243": 49,  # table
    "04401088": 50,
    "04460130": 51,
    "04468005": 52,
    "04530566": 53,
    "04554684": 54,
    # ShapeNet v1
    "02834778": 55,
    "02858304": 56,
}


def _parse_shapenet(
    data_dir: Path,
    split: str,
    split_file: str | None = None,
    partnet_dir: Path | None = None,
    categories: Sequence[str | int] | None = None,
) -> tuple[str, list[str], dict[str, dict[str, Any]], list[dict[str, str]]]:
    name = "ShapeNetCore.v2" if "v2" in data_dir.name else "ShapeNetCore.v1"
    category_ids: list[str]
    if categories is None:
        if partnet_dir is None:
            category_ids = sorted([c.name for c in data_dir.iterdir() if c.is_dir()])
        else:
            category_ids = sorted([c.name for c in partnet_dir.iterdir() if c.is_dir()])
        _log_debug_level_2(f"Categories: {category_ids}")
    else:
        normalized_categories: list[str] = []
        for category in categories:
            if isinstance(category, int):
                normalized_categories.append(next(key for key, val in CATEGORIES_MAP.items() if val == category))
            else:
                normalized_categories.append(category)
        category_ids = normalized_categories
        _log_debug_level_1(f"Categories: {category_ids}")

    with open(data_dir / "taxonomy.json") as f:
        taxonomy = json.load(f)

    metadata = dict()
    for category in category_ids:
        metadata[category] = {"index": CATEGORIES_MAP[category]}
        category_dict = next(item for item in taxonomy if item["synsetId"] == category)
        metadata[category]["name"] = category_dict["name"]
        metadata[category]["size"] = category_dict["numInstances"]

    objects = list()
    for category in category_ids:
        category_path = data_dir / category if partnet_dir is None else partnet_dir / category
        split_path = category_path / split_file if split_file else category_path / (split + ".lst")
        _log_debug_level_2(f"Loading split {split} for category {category} from {split_path}")
        with open(split_path) as f:
            object_ids = f.read().split("\n")

        if "" in object_ids:
            object_ids.remove("")

        logger.debug(f"Objects in {category}: {len(object_ids)}")
        objects.extend([{"category": category, "name": obj_id} for obj_id in object_ids if "#" not in obj_id])

    return name, category_ids, metadata, objects


def _parse_mesh_dataset(
    data_dir: Path, objects: Sequence[str | int | Path] | None = None, id_length: int | None = None
) -> tuple[str, list[str], dict[str, dict[str, Any]], list[dict[str, str]]]:
    assert not (objects is None and id_length is None), "Either 'objects' or 'id_length' must be specified."
    assert not (isinstance(objects, (tuple, list)) and isinstance(objects[0], int) and id_length is None), (
        "'id_length' must be specified if 'objects' are integers."
    )

    if objects is None:
        object_dicts = [
            {"category": obj[:id_length], "name": obj} for obj in os.listdir(data_dir) if (data_dir / obj).is_dir()
        ]
    else:
        if id_length is None:
            object_dicts = list()
            id_length = len(str(len(objects)))
            for i, obj in enumerate(objects):
                category = str(i).zfill(id_length)
                name = Path(str(obj)).parent.stem
                object_dicts.append({"category": category, "name": name})
        elif id_length > 0:
            selection = list()
            all_objects = os.listdir(data_dir)
            all_names = [obj[id_length + 1 :] for obj in all_objects]
            all_ids = [obj[:id_length] for obj in all_objects]
            for obj in objects:
                if isinstance(obj, str):
                    if obj in all_names:
                        selection.append(all_objects[all_names.index(obj)])
                    elif obj.zfill(id_length) in all_ids:
                        selection.append(all_objects[all_ids.index(obj.zfill(id_length))])
                elif isinstance(obj, int) and str(obj).zfill(id_length) in all_ids:
                    selection.append(all_objects[all_ids.index(str(obj).zfill(id_length))])
            object_dicts = [{"category": obj[:id_length], "name": obj} for obj in selection]
        else:
            raise ValueError("id_length < 0.")
    logger.debug(f"Objects: {object_dicts}")

    metadata = dict()
    for i, obj in enumerate(object_dicts):
        name = obj["name"] if id_length is None else obj["name"][id_length + 1 :]
        metadata[obj["category"]] = {"index": i, "name": name}
    categories = [obj["category"] for obj in object_dicts]

    _log_debug_level_1(f"Categories: {categories}")
    _log_debug_level_2(f"Metadata: {metadata}")

    return data_dir.name, categories, metadata, object_dicts


def get_inputs_field(
    cam_dir_name,
    convention,
    crop,
    file_offset,
    files_per_shard,
    from_hdf5,
    image_dir_name,
    index,
    inputs_dir,
    inputs_type,
    load_cam,
    load_normals,
    load_random_inputs,
    normals_file,
    num_images,
    num_objects,
    num_shards,
    name,
    obj_weights,
    pointcloud_file,
    precision,
    project,
    resize,
    split,
    undistort,
    unrotate,
    unscale,
    weighted,
) -> Field:
    if inputs_type in ["idx", None]:
        return IndexField()
    if "render" in inputs_type:
        return EmptyField()
    if "dtu" in inputs_type:
        return DTUField(
            load_depth="depth" in inputs_type or "rgbd" in inputs_type,
            random_view=False,
            ignore_image_idx=(3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39),
        )

    # Fixme: Remove this hack to deal with parts of the data coming from different directories
    suffix = ""
    if inputs_dir:
        if inputs_dir.suffix == ".hdf5":
            suffix = inputs_dir.suffix
            inputs_dir = inputs_dir.parent / inputs_dir.stem
        if "shapenet" in str(inputs_dir).lower() and ".fused.simple" not in inputs_dir.name:
            inputs_dir = inputs_dir.parent / (inputs_dir.name + ".fused.simple")

    if inputs_type in ["pointcloud", "partial", "depth_like"]:
        if inputs_dir and inputs_dir.name.endswith(".kinect"):
            inputs_dir = inputs_dir.parent / inputs_dir.name.removesuffix(".kinect")
        return PointCloudField(
            file=pointcloud_file or "pointcloud.npz",
            data_dir=inputs_dir.parent / (inputs_dir.name + suffix) if inputs_dir else None,
            cam_dir=cam_dir_name if load_cam else None,
            index=index,
            num_cams=num_images,
            normals_file=normals_file,
            load_normals=load_normals,
            from_hdf5=from_hdf5,
        )

    if inputs_type == "image":
        return ImageField(
            data_dir=inputs_dir,
            path_suffix=image_dir_name,
            cam_filename="cameras.npz" if inputs_dir is None else "rendering_metadata.txt",
            extension=".jpg" if inputs_dir is None else ".png",
            index=index,
            num_images=num_images,
            convention=convention,
        )

    if inputs_type == "depth_bp":
        return BlenderProcRGBDField(
            data_dir=os.path.join("" if inputs_dir is None else inputs_dir, "blenderproc", split),
            unscale=unscale,
            undistort=undistort,
            num_objects=num_objects,
            num_shards=num_shards,
            files_per_shard=files_per_shard,
            random_file=load_random_inputs,
            random_shard=load_random_inputs,
            input_type=inputs_type,
            load_fixed=False,
        )

    if (
        inputs_type == "kinect"
        and inputs_dir
        and "shapenet" in str(inputs_dir).lower()
        and not inputs_dir.name.endswith(".kinect")
    ):
        inputs_dir = inputs_dir.parent / (inputs_dir.name + ".kinect")
    return DepthField(
        data_dir=inputs_dir.parent / (inputs_dir.name + suffix) if inputs_dir else None,
        unscale=unscale,
        unrotate=unrotate,
        load_depth=inputs_type not in ["color", "rgb", "normals", "shading"],
        load_normals=load_normals,
        load_colors="rgb" in inputs_type or "color" in inputs_type,
        kinect="kinect" in inputs_type,
        path_suffix="models" if "v2" in name.lower() else None,
        num_objects=num_objects,
        num_files=files_per_shard,
        file_offset=file_offset,
        random_file=load_random_inputs,
        file_weights=obj_weights if split == "train" and isinstance(weighted, str) else None,
        precision=precision,
        from_hdf5=from_hdf5,
        project=project,
        crop=crop,
        resize=resize,
        convention=convention,
    )


class ShapeNet(Dataset):
    def __init__(
        self,
        split: str,
        data_dir: Path,
        partnet_dir: Path | None = None,
        inputs_dir: Path | None = None,
        points_dir: Path | None = None,
        pointcloud_dir: Path | None = None,
        mesh_dir: Path | None = None,
        image_dir_name: str | None = None,
        cam_dir_name: str | None = None,
        categories: list[str | int] | None = None,
        num_shards: int = 1,
        files_per_shard: int = 1,
        num_views: int = 1,
        unscale: bool = False,
        undistort: bool = False,
        unrotate: bool = False,
        project: bool = True,
        crop: bool = False,
        resize: int | None = None,
        convention: str = "opencv",
        sdf: bool = False,
        tsdf: float = 0,
        split_file: str | None = None,
        pointcloud_file: str | None = None,
        points_file: str | None = None,
        voxels_file: str | None = None,
        normals_file: str | None = None,
        mesh_file: str | None = None,
        pose_file: str | None = None,
        load_random_inputs: bool = False,
        load_pointcloud: bool | str = False,
        load_normals: bool = False,
        load_points: bool = True,
        load_all_points: bool = False,
        load_random_points: bool = True,
        load_surface_points: bool | float | None = None,
        load_meshes: bool | str = False,
        load_cam: bool = False,
        inputs_type: str | list[str] = "pointcloud",
        type_p: list[float] | None = None,
        weighted: bool | str = False,
        padding: float = 0.1,
        cls_weight: float = 0,
        seg_weight: float = 0,
        precision: int = 16,
        from_hdf5: bool = False,
        transforms: list[Transform] | None = None,
        verbose: bool = False,
    ):
        assert data_dir.is_dir(), f"{data_dir} is not a directory."
        if isinstance(load_pointcloud, str):
            assert load_pointcloud == "min_max_only", "load_pointcloud can only be 'min_max_only' if it is a string."

        self.verbose = verbose
        self.weighted = weighted
        self.num_views = num_views

        file_offset = 0
        if "shapenet" in str(data_dir).lower():
            self.name, self.categories, self.metadata, self.objects = _parse_shapenet(
                data_dir, split, split_file, partnet_dir, categories
            )
        else:
            self.name, self.categories, self.metadata, self.objects = _parse_mesh_dataset(
                data_dir, objects=categories, id_length=3
            )
            if "automatica" in self.name and "render" not in inputs_type:
                file_offset = 700 if split == "val" else 800 if split == "test" else 0

        self.data_dir = data_dir
        self.categories: list[str]
        self.metadata: dict[str, dict[str, Any]]
        self.objects: list[dict[str, str]]
        self.mesh_dir = mesh_dir
        self.mesh_file = mesh_file
        self.split = split
        self.files_per_shard = files_per_shard

        num_objects = len(self.objects)
        num_images = 36 if inputs_dir is not None and "DISN" in str(inputs_dir) else 24
        if (
            (inputs_type in ["image", "rgb", "rgbd"] or load_cam)
            and self.split == "train"
            and not self.name.lower() == "dtu"
        ):
            index = -1
            if not load_random_inputs:
                index = (0, num_images)
                self.objects *= num_images
        else:
            index = 0
            self.objects *= 1 if load_random_inputs and "render" not in inputs_type else (files_per_shard * num_shards)

        self.fields: dict[str, Field] = dict()
        if isinstance(inputs_type, str):
            self.fields["inputs"] = get_inputs_field(
                cam_dir_name,
                convention,
                crop,
                file_offset,
                files_per_shard,
                from_hdf5,
                image_dir_name,
                index,
                None if self.name.lower() == "automatica" else inputs_dir,
                "dtu_" + inputs_type if self.name.lower() == "dtu" else inputs_type,
                load_cam,
                load_normals,
                load_random_inputs,
                normals_file,
                num_images,
                num_objects,
                num_shards,
                self.name,
                self.obj_weights,
                pointcloud_file,
                precision,
                project,
                resize,
                split,
                undistort,
                unrotate,
                unscale,
                weighted,
            )
        else:
            fields = [
                get_inputs_field(
                    cam_dir_name,
                    convention,
                    crop,
                    file_offset,
                    files_per_shard,
                    from_hdf5,
                    image_dir_name,
                    index,
                    None if self.name.lower() == "automatica" else inputs_dir,
                    t,
                    load_cam,
                    load_normals,
                    load_random_inputs,
                    normals_file,
                    num_images,
                    num_objects,
                    num_shards,
                    self.name,
                    self.obj_weights,
                    pointcloud_file,
                    precision,
                    project,
                    resize,
                    split,
                    undistort,
                    unrotate,
                    unscale,
                    weighted,
                )
                for t in inputs_type
            ]
            base_type = type(fields[0])
            if all(isinstance(field, base_type) for field in fields):
                if split == "train":
                    self.fields["inputs"] = RandomField(fields, p=type_p)
                else:
                    self.fields["inputs"] = fields[0]
            else:
                merge_keys = [None, *inputs_type[1:]]
                if "kinect" in merge_keys:
                    merge_keys[merge_keys.index("kinect")] = "depth"
                self.fields["inputs"] = MixedField(fields, merge_keys=cast(Any, merge_keys), p=type_p)

        if load_points:
            file = points_file if points_file else mesh_file if mesh_file else "points.npz"
            sigmas = [0.001, 0.0015, 0.0025, 0.01, 0.015, 0.025, 0.05, 0.1, 0.15, 0.25]
            points_data_dir: str | None = None
            if "automatica" not in self.name:
                base_points_dir = points_dir if points_file else mesh_dir
                if base_points_dir is not None:
                    points_data_dir = str(base_points_dir)
            params_dir = None if "automatica" in self.name.lower() else (str(inputs_dir) if inputs_dir else None)
            self.fields["points"] = PointsField(
                file=file,
                data_dir=points_data_dir,
                params_dir=params_dir,
                path_suffix="models" if "v2" in self.name.lower() else "",
                padding=padding,
                occ_from_sdf=not sdf,
                tsdf=tsdf,
                sigmas=sigmas,
                load_surface_file=load_surface_points,
                load_all_files=load_all_points,
                load_random_file=load_random_points,
                from_hdf5=from_hdf5,
            )

        if load_pointcloud and "pointcloud" not in self.fields:
            file = pointcloud_file if pointcloud_file else mesh_file if mesh_file else "pointcloud.npz"
            min_max_only = load_pointcloud == "min_max_only"
            pointcloud_data_dir: str | None = None
            if "automatica" not in self.name:
                base_pointcloud_dir = pointcloud_dir if pointcloud_file else mesh_dir
                if base_pointcloud_dir is not None:
                    pointcloud_data_dir = str(base_pointcloud_dir)
            self.fields["pointcloud"] = PointCloudField(
                file=file,
                data_dir=pointcloud_data_dir,
                normals_file=normals_file,
                load_normals=not min_max_only,
                min_max_only=min_max_only,
                from_hdf5=from_hdf5,
            )

        bool_true = not isinstance(load_meshes, str) and load_meshes
        valid_str = isinstance(load_meshes, str) and load_meshes in ["path_only", f"{split}_only"]
        if bool_true or valid_str or "render" in inputs_type:
            mesh_data_dir = None if "automatica" in self.name.lower() else (str(mesh_dir) if mesh_dir else None)
            self.fields["mesh"] = MeshField(
                file=mesh_file,
                data_dir=mesh_data_dir,
                pose_file=pose_file,
                from_hdf5=from_hdf5,
                path_only=load_meshes == "path_only",
                geometry_only=False if "automatica" in self.name.lower() else True,
            )

        if voxels_file is not None:
            self.fields["voxels"] = VoxelsField(file=voxels_file)

        if partnet_dir is not None:
            self.fields["partnet"] = PartNetField(data_dir=partnet_dir)

        self.fields = dict(sorted(self.fields.items()))
        self.transforms = transforms
        self.start = 0
        self.end = len(self.objects)
        self.iteration = self.start - 1
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight

    def __repr__(self):
        if self.transforms:
            trafo_str = "Transforms:\n"
            for trafo in self.transforms:
                if trafo.args:
                    trafo_str += f"  {trafo.name}:\n"
                    for k, v in trafo.args.items():
                        if isinstance(v, np.ndarray):
                            with np.printoptions(precision=3, suppress=True):
                                if v.ndim == 1:
                                    trafo_str += f"    {k}: {v}\n"
                                else:
                                    trafo_str += f"    {k}: array({v.shape})\n"
                        elif isinstance(v, Sequence):
                            if not isinstance(v, str) and len(v) > 6:
                                list_str = f"[{v[0]}, {v[1]}, {v[2]}, ..., {v[-3]}, {v[-2]}, {v[-1]}]"
                                trafo_str += f"    {k}: {list_str} (len={len(v)})\n"
                            else:
                                trafo_str += f"    {k}: {v}\n"
                        else:
                            trafo_str += f"    {k}: {v}\n"
                else:
                    trafo_str += f"  {trafo.name}\n"
            return trafo_str
        return super().__repr__()

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

    @cached_property
    def category_weights(self) -> np.ndarray:
        inv_freqs = np.array([1 / self.metadata[category]["size"] for category in self.categories])
        inv_freqs /= inv_freqs.sum()
        category_weights = np.array([inv_freqs[self.categories.index(obj["category"])] for obj in self.objects])
        logger.debug(f"Category weights: {set(category_weights)}")
        return category_weights

    @cached_property
    def obj_weights(self) -> dict[str, dict[str, np.ndarray]] | None:
        if not self.weighted or not isinstance(self.weighted, str):
            return None

        path = Path(self.weighted).expanduser().resolve()
        _log_debug_level_1(f"Loading object weights from {path}")

        def normalize(series):
            return series / series.sum()

        def sharpen(prob_vector, temperature: float = 0.1):
            powered_vector = np.power(prob_vector, 1 / temperature)
            return powered_vector / np.sum(powered_vector)

        df = pd.read_pickle(path)
        df_sorted = cast(Any, df).sort_values("path")
        df_sorted["loss"] = df_sorted.groupby(["object category", "object name"])["loss"].transform(normalize)
        result_dict = df_sorted.groupby(["object category", "object name"])["loss"].apply(list).to_dict()

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
        return weights_dict

    def get_category(self, index: int) -> str:
        return self.objects[index]["category"]

    def get_category_index(self, index: int, category: str | None = None) -> int:
        if category is None:
            category = self.get_category(index)
        return self.metadata[category]["index"]

    def get_category_name(self, index: int, category: str | None = None) -> str:
        if category is None:
            category = self.get_category(index)
        return self.metadata[category]["name"]

    def get_obj_name(self, index: int) -> str:
        return self.objects[index]["name"]

    def get_obj_dir(self, index: int, category: str | None = None, obj_name: str | None = None) -> Path:
        if category is None:
            category = self.get_category(index)
        if obj_name is None:
            obj_name = self.get_obj_name(index)
        obj_dir = self.data_dir / category / obj_name
        # Support for ShapeNet (w/ or w/o per-object hdf5 file) and one object per directory
        return obj_dir if obj_dir.is_dir() or obj_dir.with_suffix(".hdf5").is_file() else self.data_dir / obj_name

    def get_inputs_path(
        self, index: int, obj_dir: str | Path | None = None, category: str | None = None, obj_name: str | None = None
    ) -> Path:
        if obj_dir is None:
            if category is None:
                category = self.get_category(index)
            if obj_name is None:
                obj_name = self.get_obj_name(index)
            obj_dir = self.get_obj_dir(index, category, obj_name)
        assert obj_dir is not None
        if isinstance(self.fields["inputs"], (DepthField, BlenderProcRGBDField)):
            inputs_field = cast(Any, self.fields["inputs"])
            return Path(inputs_field.get_depth_path(index, obj_dir))
        return Path(obj_dir)

    def load_field(self, name: str, field: Field, data: dict[str, Any]):
        load_timer = time.perf_counter()
        file_index = data.get("inputs.file", -1) if isinstance(field, PointsField) else data["category.index"]
        field_data = field.load(
            obj_dir=self.get_obj_dir(index=data["index"], category=data["category.id"]),
            index=data["index"],
            category=file_index,
        )
        logger.debug(f"Field {field.name} ({name}) takes {time.perf_counter() - load_timer:.4f}s.")
        if isinstance(field_data, dict):
            for k, v in field_data.items():
                if k is None:
                    data[name] = v
                else:
                    data[f"{name}.{k}"] = v
        else:
            data[name] = field_data

    def init_item(self, index: int) -> dict[str, Any]:
        category = self.get_category(index)
        category_name = self.get_category_name(index, category)
        category_index = self.get_category_index(index, category)
        data = {
            "index": index,
            "category.id": category,
            "category.name": category_name,
            "category.index": category_index,
        }
        if self.cls_weight:
            data["cls_weight"] = self.cls_weight
        if self.seg_weight:
            data["seg_weight"] = self.seg_weight

        return data

    def _get_item(self, index: int) -> dict[str, Any]:
        item_time = time.perf_counter()
        item = self.init_item(index)

        for name, field in self.fields.items():
            self.load_field(name, field, item)

        if self.num_views > 1:
            _item = copy.deepcopy(item)
            item = cast(dict[str, Any], apply_transforms(item, self.transforms))
            for view in range(self.num_views - 1):
                item_i = copy.deepcopy(_item)
                self.load_field("inputs", self.fields["inputs"], item_i)
                item_i = cast(dict[str, Any], apply_transforms(item_i, self.transforms))
                for key, value in item_i.items():
                    if not isinstance(value, str):
                        if not torch.is_tensor(value):
                            value = torch.as_tensor(value)
                        existing = item[key]
                        if not torch.is_tensor(existing):
                            existing = torch.as_tensor(existing)
                        if view == 0:
                            item[key] = torch.stack((existing, value))
                        else:
                            item[key] = torch.cat((existing, value[None, ...]))
        else:
            item = apply_transforms(item, self.transforms)

        if self.verbose:
            item_name = "/".join([self.get_category(index), self.get_obj_name(index)])
            if self.transforms and len(self.transforms) > 2 and isinstance(self.transforms[-2], KeysToKeep):
                maybe_keys_to_keep = self.transforms[-2].keys
                keys_to_keep = list(item.keys()) if maybe_keys_to_keep is None else maybe_keys_to_keep
            else:
                keys_to_keep = list(item.keys())
            size_in_bytes = sum([sys.getsizeof(v) for k, v in item.items() if k in keys_to_keep])
            size_in_mb = round(size_in_bytes / 1024 / 1024, 2)
            logger.debug(
                f"Item {index}: {item_name} takes {time.perf_counter() - item_time:.4f}s (size: {size_in_mb:.2f} MB)."
            )
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            return self._get_item(index)
        except Exception as e:
            path = Path(self.get_inputs_path(index)).relative_to(self.data_dir)
            logger.error(f"Caught exception '{e}' for item {index}: {path}")
            logger.exception(e)
            raise e
