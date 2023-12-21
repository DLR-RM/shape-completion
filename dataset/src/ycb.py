import os
from typing import Union, Tuple, Dict, List, Any, Optional

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .fields import RGBDField, PointsField, PointCloudField, MeshField, BlenderProcRGBDField, VoxelsField
from .transforms import Transform, RotatePointcloud


class YCB(Dataset):
    def __init__(self,
                 data_dir: str,
                 mesh_dirname: str = "google_16k",
                 sdf: bool = False,
                 pointcloud_file: Optional[str] = None,
                 points_file: Optional[str] = None,
                 voxels_file: Optional[str] = None,
                 normals_file: Optional[str] = None,
                 mesh_file: str = "nontextured.ply",
                 pose_file: str = "pose.npy",
                 load_pointcloud: bool = False,
                 load_points: bool = True,
                 load_meshes: bool = False,
                 load_real_data: bool = True,
                 objects: Union[Tuple[str], Tuple[int]] = (6, 19, 21),
                 split: str = "train",
                 rgb: bool = False,
                 high_res_rgb: bool = False,
                 mask: bool = False,
                 crop: float = 0.2,
                 cam_id_range: Tuple[int, int] = (1, 5),
                 merge_cam_id_range: Tuple[int, int] = None,
                 merge_angles: int = 0,
                 stride: int = 1,
                 filter_discontinuities: int = None,
                 input_type: Optional[str] = None,
                 padding: float = 0.1,
                 transforms: Dict[str, List[Transform]] = None):
        self.name = self.__class__.__name__
        assert os.path.isdir(data_dir)
        if cam_id_range:
            assert max(cam_id_range) <= 5 and min(cam_id_range) >= 1
        if merge_cam_id_range:
            assert merge_cam_id_range == cam_id_range
            num_merged_cams = max(merge_cam_id_range) - min(merge_cam_id_range) + 1
        else:
            num_merged_cams = 1

        if objects is None:
            self.objects = [{"category": obj[:3],
                             "name": obj} for obj in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, obj))]
        else:
            selection = list()
            all_objects = os.listdir(data_dir)
            all_ids = [obj[:3] for obj in all_objects]
            for obj in objects:
                if isinstance(obj, str):
                    if obj in all_objects:
                        selection.append(obj)
                    elif obj.zfill(3) in all_ids:
                        selection.append(all_objects[all_ids.index(obj.zfill(3))])
                elif isinstance(obj, int) and str(obj).zfill(3) in all_ids:
                    selection.append(all_objects[all_ids.index(str(obj).zfill(3))])
            self.objects = [{"category": obj[:3], "name": obj} for obj in selection]

        self.metadata = {obj["category"]: {"index": index,
                                           "name": obj["name"][4:]} for index, obj in enumerate(self.objects)}
        self.categories = [obj["category"] for obj in self.objects]

        self.data_dir = data_dir
        self.mesh_dirname = mesh_dirname
        self.mesh_file = mesh_file

        self.rotation_angle = 3
        self.num_rotations = 360 // self.rotation_angle
        self.max_index = self.num_rotations * max(cam_id_range)
        self.min_index = self.num_rotations * (min(cam_id_range) - 1)
        self.num_images = 1024 if split == "train" else 32
        if load_real_data:
            self.num_images = (self.max_index - self.min_index) // num_merged_cams

        inputs_trafos = transforms.get("inputs") if transforms else None
        points_trafos = transforms.get("points") if transforms else None
        data_trafos = transforms.get("data") if transforms else None

        self.fields = dict()
        if load_real_data:
            self.fields["inputs"] = RGBDField(rgb=rgb or input_type in ["image", "rgbd"],
                                              high_res_rgb=high_res_rgb,
                                              mask=mask,
                                              crop=crop,
                                              cam_id_range=cam_id_range,
                                              merge_cam_id_range=merge_cam_id_range,
                                              merge_angles=merge_angles,
                                              filter_discontinuities=filter_discontinuities,
                                              stride=stride,
                                              transform=Compose(inputs_trafos) if inputs_trafos else None)
        else:
            if input_type != "image":
                if not load_real_data:
                    if inputs_trafos is None:
                        inputs_trafos = [RotatePointcloud(axes='x', angles=(-90,))]
                    else:
                        inputs_trafos.insert(0, RotatePointcloud(axes='x', angles=(-90,)))
                self.fields["inputs"] = PointCloudField(file=pointcloud_file if pointcloud_file else mesh_file,
                                                        data_dir=mesh_dirname,
                                                        cache=True,
                                                        transform=Compose(inputs_trafos) if inputs_trafos else None)
            if input_type in ["image", "rgbd"]:
                self.fields["inputs"] = BlenderProcRGBDField(unscale=True,
                                                             undistort=True,
                                                             num_shards=5,
                                                             files_per_shard=5,
                                                             input_type=input_type,
                                                             transform=inputs_trafos)

        if load_points:
            if not load_real_data:
                if points_trafos is None:
                    points_trafos = [RotatePointcloud(axes='x', angles=(-90,))]
                else:
                    points_trafos.insert(0, RotatePointcloud(axes='x', angles=(-90,)))
            self.fields["points"] = PointsField(file=points_file if points_file else mesh_file,
                                                data_dir=mesh_dirname,
                                                padding=padding,
                                                occ_from_sdf=not sdf,
                                                tsdf=None,
                                                cache=True,
                                                # sigmas=[0.01, 0.015, 0.1, 0.15] if split == "train" else None,
                                                transform=Compose(points_trafos) if points_trafos else None)

        if load_pointcloud:
            transform = None if load_real_data else RotatePointcloud(axes='x', angles=(-90,))
            self.fields["pointcloud"] = PointCloudField(file=pointcloud_file if pointcloud_file else mesh_file,
                                                        data_dir=mesh_dirname,
                                                        normals_file=normals_file,
                                                        load_normals=True,
                                                        cache=True,
                                                        transform=transform)

        if voxels_file is not None:
            self.fields["voxels"] = VoxelsField(file=voxels_file)

        if load_meshes:
            self.fields["mesh"] = MeshField(file=mesh_file,
                                            data_dir=mesh_dirname,
                                            pose_file=pose_file,
                                            cache=True)
        self.transform = Compose(data_trafos) if data_trafos else None
        self.load_real_data = load_real_data

    def __len__(self):
        return len(self.objects) * self.num_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.load_real_data:
            category = self.objects[index // self.num_images]["category"]
            obj_name = self.objects[index // self.num_images]["name"]
        else:
            category = self.objects[index % len(self.objects)]["category"]
            obj_name = self.objects[index % len(self.objects)]["name"]

        category_index = self.metadata[category]["index"]
        obj_dir = os.path.join(self.data_dir, obj_name)

        data = {"index": index, "obj_name": obj_name}

        for name, field in self.fields.items():
            field_data = field.load(obj_dir, index, category_index)

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[name] = v
                    else:
                        data[f"{name}.{k}"] = v
            else:
                data[name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data
