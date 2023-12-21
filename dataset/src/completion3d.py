import math
import os
import random
from typing import List, Dict, Any

import h5py
import numpy as np
import pandas as pd
import transforms3d
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .fields import PointsField, PointCloudField
from .transforms import Transform


def load_h5(path):
    f = h5py.File(path, "r")
    cloud_data = np.array(f["data"])
    f.close()
    return cloud_data.astype(np.float64)


def pad_cloudN(P, Nin):
    N = P.shape[0]
    P = P[:].astype(np.float32)

    rs = np.random.random.__self__
    choice = np.arange(N)
    if N > Nin:
        ii = rs.choice(N, Nin)
        choice = ii
    elif N < Nin:
        ii = rs.choice(N, Nin - N)
        choice = np.concatenate([range(N), ii])
    P = P[choice, :]

    return P


def augment_cloud(Ps, pc_augm_scale, pc_augm_rot, pc_augm_mirror_prob, pc_augm_jitter):
    M = transforms3d.zooms.zfdir2mat(1)
    if pc_augm_scale > 1:
        s = random.uniform(1 / pc_augm_scale, pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if pc_augm_rot:
        angle = random.uniform(0, 2 * math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), M)  # y=upright assumption
    if pc_augm_mirror_prob > 0:  # mirroring x&z, not y
        if random.random() < pc_augm_mirror_prob / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if random.random() < pc_augm_mirror_prob / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), M)
    result = []
    for P in Ps:
        P[:, :3] = np.dot(P[:, :3], M.T)

        if pc_augm_jitter:
            sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
            P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
        result.append(P)
    return result


class Completion3D(Dataset):

    def __init__(self,
                 data_dir: str,
                 shapenet_dir: str,
                 categories: List[str] = None,
                 split: str = "train",
                 transforms: Dict[str, List[Transform]] = None,
                 load_pointcloud: bool = False,
                 load_points: bool = True):
        self.name = self.__class__.__name__
        self.data_dir = data_dir
        self.shapenet_dir = shapenet_dir
        self.split = split.lower()
        self.load_pointcloud = load_pointcloud
        self.load_points = load_points

        classmap = pd.read_csv(os.path.join(data_dir, "synsetoffset2category.txt"),
                               dtype=str,
                               delim_whitespace=True,
                               header=None)
        self.classmap = dict(zip(classmap.values[:, 1], classmap.values[:, 0]))

        if categories is None:
            categories = list(self.classmap.keys())
        self.categories = categories

        self.metadata = dict()
        for index, (class_id, class_name) in enumerate(self.classmap.items()):
            self.metadata[class_id] = {"index": index, "name": class_name, "id": class_id}

        self.data_paths = sorted([os.path.join(data_dir, self.split, "partial", k.rstrip() + ".h5") for k in
                                  open(os.path.join(data_dir, f"{self.split}.list")).readlines() if
                                  k.split('/')[-2] in categories])
        self.objects = list()
        for path in self.data_paths:
            self.objects.append({"category": path.split('/')[-2], "name": path.split('/')[-1].split('.')[0]})

        points_trafos = Compose(transforms["points"]) if transforms and transforms.get("points") else None
        self.inputs_trafos = Compose(transforms["inputs"]) if transforms and transforms.get("inputs") else None
        self.transform = Compose(transforms["data"]) if transforms and transforms.get("data") else None

        self.points = PointsField(transform=points_trafos)
        self.pointcloud = PointCloudField(load_normals=True)

    def load_data(self, filename: str):
        category, obj_name = filename.split('/')[-2:]
        obj_name = obj_name.split('.')[0]

        inputs = load_h5(filename)
        if self.inputs_trafos:
            inputs = self.inputs_trafos({None: inputs})[None]

        if self.split == "test":
            pcd2048 = inputs
        else:
            pcd2048 = load_h5(filename.replace("partial", "gt"))

        if self.load_pointcloud:
            pointcloud = self.pointcloud.load(obj_dir=os.path.join(self.shapenet_dir, category, obj_name),
                                              index=0,
                                              category=category)
        else:
            pointcloud = None
        # if self.split == "train":
        #     pointcloud, inputs = augment_cloud([pointcloud, inputs], 0, 0, 0, 0)

        if self.load_points:
            points = self.points.load(obj_dir=os.path.join(self.shapenet_dir, category, obj_name),
                                      index=0,
                                      category=category)
        else:
            points = None
        return inputs, points, pointcloud, pcd2048, category, obj_name

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        inputs, points, pointcloud, pcd2048, category, obj_name = self.load_data(self.data_paths[index])
        data = {"index": index,
                "obj_name": obj_name,
                "inputs.path": self.data_paths[index],
                "inputs": inputs.astype(np.float32),
                "pointcloud": pcd2048.astype(np.float32)}
        if self.load_pointcloud:
            data["pcd100k"] = pointcloud[None].astype(np.float32)
            data["pcd100k.normals"] = pointcloud["normals"].astype(np.float32)
        if self.load_points:
            data["points"] = points[None].astype(np.float32)
            data["points.occ"] = points["occ"].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data
