from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
from easy_o3d.utils import DownsampleTypes, OutlierTypes
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.transforms import v2 as T

from utils import Voxelizer

from .src.bop import BOP
from .src.coco import CocoInstanceSegmentation
from .src.completion3d import Completion3D
from .src.fields import (
    DepthField as DepthField,
)
from .src.fields import (
    ImageField as ImageField,
)
from .src.fields import (
    MeshField as MeshField,
)
from .src.fields import (
    PointCloudField as PointCloudField,
)
from .src.fields import (
    PointsField as PointsField,
)
from .src.graspnet import GraspNetEval
from .src.modelnet import ModelNet
from .src.shapenet import ShapeNet
from .src.shared import SharedDataLoader as SharedDataLoader
from .src.shared import SharedDataset as SharedDataset
from .src.tabletop import TableTop
from .src.transforms import (
    BPS,
    AddGaussianNoise,
    Affine,
    AngleOfIncidenceRemoval,
    ApplyPose,
    AxesCutPointcloud,
    BoundingBox,
    BoundingBoxJitter,
    CheckDtype,
    CropPointcloud,
    CropPointcloudWithMesh,
    CropPoints,
    DepthLikePointcloud,
    DepthToPointcloud,
    EdgeNoise,
    ImageBorderNoise,
    ImageToTensor,
    InputsNormalsFromPointcloud,
    KeysToKeep,
    LoadUncertain,
    MinMaxNumPoints,
    Normalize,
    NormalizeMesh,
    NormalsCameraCosineSimilarity,
    Permute,
    PointcloudFromMesh,
    PointsFromMesh,
    PointsFromPointcloud,
    ProcessPointcloud,
    RandomApply,
    RefinePose,
    RefinePosePerInstance,
    Render,
    RenderPointcloud,
    Return,
    Rotate,
    RotateMesh,
    Scale,
    SdfFromOcc,
    SegmentationFromPartNet,
    ShadingImageFromNormals,
    SphereCutPointcloud,
    SphereMovePointcloud,
    SplitData,
    SubsamplePointcloud,
    SubsamplePoints,
    Transform,
    Translate,
    VoxelizePointcloud,
    VoxelizePoints,
)
from .src.transforms import (
    FindUncertainPoints as FindUncertainPoints,
)
from .src.transforms import (
    RenderDepthMaps as RenderDepthMaps,
)
from .src.transforms import (
    SaveData as SaveData,
)
from .src.transforms import (
    Visualize as Visualize,
)
from .src.tv_transforms import CenterPad
from .src.utils import TorchvisionDatasetWrapper, get_file, logger
from .src.ycb import YCB

logger = cast(Any, logger)


def get_transformations(cfg: DictConfig, split: str) -> list[Transform]:
    train_aug = split == "train" and not cfg.train.no_aug
    val_aug = split == "val" and not cfg.val.no_aug
    test_aug = split == "test" and not cfg.test.no_aug
    apply_aug = train_aug or val_aug or test_aug

    input_has_depth = any(t in cfg.inputs.type for t in ["depth", "kinect", "rgbd"])
    input_has_pcd = cfg.inputs.type in [
        "pointcloud",
        "partial",
        "depth_like",
    ] or (input_has_depth and cfg.inputs.project)

    transformations = list()

    if cfg.mesh.norm:
        transformations.append(NormalizeMesh())
    if cfg.mesh.rot:
        transformations.append(RotateMesh(axes="xyz", angles=cfg.mesh.rot))

    if cfg.pointcloud.from_mesh:
        transformations.append(PointcloudFromMesh(num_points=cfg.pointcloud[split].num_points or int(1e5)))

    if cfg.inputs.type == "partial":
        transformations.append(
            AxesCutPointcloud(axes="z", cut_ratio=(0.4, 0.6), rotate_object=cfg.aug.rotate, upper_hemisphere=True)
        )
    elif cfg.inputs.type == "depth_like":
        transformations.extend(
            [
                SubsamplePointcloud(apply_to="inputs", num_samples=50000),
                DepthLikePointcloud(
                    rotate_object="" if cfg.aug.rotate == "cam" else cfg.aug.rotate,
                    upper_hemisphere=cfg.aug.upper_hemisphere,
                    rot_from_inputs=cfg.aug.rotate == "cam",
                    # cam_from_inputs=cfg.data.frame == "cam"
                ),
            ]
        )
    elif "render" in cfg.inputs.type:
        if any(m in cfg.inputs.type for m in ["rgb", "color", "depth", "normals", "rgbd"]):
            color = "color" in cfg.inputs.type or "rgb" in cfg.inputs.type
            normals = (
                cfg.aug.remove_angle
                or cfg.aug.edge_noise
                or cfg.inputs.normals
                or "normals" in cfg.inputs.type
                or "shading" in cfg.inputs.type
            )
            depth = "depth" in cfg.inputs.type or "rgbd" in cfg.inputs.type
            transformations.append(
                Render(
                    width=cfg.inputs.width or 640,
                    height=cfg.inputs.height or 480,
                    offscreen=not cfg.vis.show,
                    method="pyrender",
                    remove_mesh=cfg.inputs.cache and not (cfg.points.from_mesh or (cfg.vis.show and cfg.vis.mesh)),
                    render_color=color,
                    render_depth=depth,
                    render_normals=normals,
                    sample_cam=cfg.inputs.load_random,
                    cache=cfg.inputs.cache,
                )
            )
            if cfg.inputs.project:
                transformations.append(DepthToPointcloud(cache=cfg.inputs.cache or not cfg.inputs.load_random))
        elif "render_pcd" in cfg.inputs.type:
            transformations.append(RenderPointcloud())

    if input_has_pcd:
        if cfg.aug.downsample:
            transformations.append(
                ProcessPointcloud(
                    apply_to="inputs", downsample=DownsampleTypes.VOXEL, downsample_factor=cfg.aug.downsample
                )
            )
        if cfg.aug.cut_plane and apply_aug:
            transformations.append(
                RandomApply(AxesCutPointcloud(apply_to="inputs", axes="y", cut_ratio=cfg.aug.cut_plane))
            )
        if cfg.aug.cut_sphere and apply_aug:
            transformations.extend(
                [
                    SphereCutPointcloud(apply_to="inputs", radius=0.2, num_spheres=2),
                    SphereCutPointcloud(apply_to="inputs", radius=0.1, num_spheres=4),
                    SphereCutPointcloud(apply_to="inputs", radius=0.05, num_spheres=10),
                ]
            )
        if cfg.aug.move_sphere and apply_aug:
            transformations.extend(
                [
                    SphereMovePointcloud(
                        apply_to="inputs", radius=0.2, num_spheres=2, offset_amount=cfg.aug.move_sphere
                    ),
                    SphereMovePointcloud(
                        apply_to="inputs", radius=0.1, num_spheres=4, offset_amount=cfg.aug.move_sphere
                    ),
                    SphereMovePointcloud(
                        apply_to="inputs", radius=0.05, num_spheres=10, offset_amount=cfg.aug.move_sphere
                    ),
                ]
            )

    if "shading" in cfg.inputs.type:
        transformations.append(
            ShadingImageFromNormals(
                light_dir="random",
                ambient="random",
                diffuse="random",
                specular="random",
                shininess="random",
                multi_light=True,
                remove_normals="normals" not in cfg.inputs.type,
                replace=not input_has_depth,
                cache=cfg.inputs.cache,
            )
        )

    # INPUTS
    if (cfg.aug.remove_angle or cfg.aug.edge_noise) and apply_aug:
        if cfg.inputs.type in ["depth_like", "rgbd", "depth_bp"]:
            vis_pcd = cfg.vis.show and cfg.vis.pointcloud
            remove_pointcloud = not vis_pcd and not cfg.norm.reference == "pointcloud" and not cfg.data.sdf_from_occ
            transformations.append(InputsNormalsFromPointcloud(remove_pointcloud=remove_pointcloud))
        transformations.append(NormalsCameraCosineSimilarity(remove_normals=not cfg.vis.show))
    if input_has_pcd:
        if cfg.aug.remove_angle and apply_aug:
            cos_sim_threshold = None if isinstance(cfg.aug.remove_angle, bool) else cfg.aug.remove_angle
            transformations.append(
                AngleOfIncidenceRemoval(cos_sim_threshold=cos_sim_threshold, remove_cos_sim=not cfg.aug.edge_noise)
            )
        if cfg.aug.edge_noise and apply_aug:
            transformations.append(EdgeNoise(stddev=cfg.aug.edge_noise, remove_cos_sim=True))
        if cfg.inputs.num_points or (cfg.inputs.fps.num_points and cfg.inputs.fps.method != "gpu"):
            num_samples: int | tuple[float, float]
            if cfg.inputs.num_points == "random":
                num_samples = (0.01, 0.99)
            elif isinstance(cfg.inputs.num_points, Sequence):
                normalized = tuple(float(x) for x in cfg.inputs.num_points)
                if len(normalized) >= 2:
                    num_samples = (normalized[0], normalized[1])
                elif len(normalized) == 1:
                    num_samples = int(normalized[0])
                else:
                    num_samples = int(1e5)
            else:
                num_samples = int(cfg.inputs.num_points)
            if isinstance(num_samples, tuple) and not apply_aug:
                num_samples = int(num_samples[1] * 1e5)
            fps = bool(cfg.inputs.fps.num_points and cfg.inputs.fps.method != "gpu")
            transformations.append(
                SubsamplePointcloud(apply_to="inputs", num_samples=num_samples, fps=fps, cachable=fps)
            )
        if cfg.aug.noise and apply_aug:
            transformations.append(AddGaussianNoise(stddev=cfg.aug.noise, clip=cfg.aug.clip_noise))

    # POINTS
    load_points = cfg.files.points[split] or cfg[split].num_query_points
    if load_points:
        if cfg.points.from_pointcloud:
            transformations.append(PointsFromPointcloud())
        elif cfg.points.from_mesh:
            sigmas = None
            if cfg.points.load_surface or cfg[split].load_surface_points:
                sigmas = [0.001, 0.0015, 0.0025, 0.01, 0.015, 0.025, 0.05, 0.1, 0.15, 0.25]
            spheres = None
            if cfg.data.rotate or cfg.aug.rotate or cfg.data.frame or abs(cfg.data.rot_x) not in [90, 180, 270]:
                spheres = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
            points = None
            cache = True
            if cfg.points.voxelize:
                points = Voxelizer(resolution=cfg.points.voxelize, padding=cfg.norm.padding).grid_points
                cache = cfg.points.voxelize**3 <= int(3e5)
            transformations.append(
                PointsFromMesh(padding=cfg.norm.padding, sigmas=sigmas, spheres=spheres, points=points, cache=cache)
            )
        if cfg.points.load_uncertain:
            transformations.append(LoadUncertain())

    # Pre-subsample points for faster data loading
    if load_points and cfg.points.subsample and isinstance(cfg[split].num_query_points, int):
        if 5 * cfg[split].num_query_points < int(1e5):
            if not (cfg.inputs.cache and cfg.points.cache):
                transformations.append(SubsamplePoints(num_samples=5 * cfg[split].num_query_points))

    # POINTCLOUD
    if cfg.pointcloud.num_points:
        num_samples = cfg.pointcloud[split].num_points
        transformations.append(
            SubsamplePointcloud(
                apply_to="pointcloud",
                num_samples=(0.1, 1.0) if num_samples == "random" else num_samples,
                fps=cfg.pointcloud.fps.num_points,
                cachable=cfg.pointcloud.fps.num_points,
            )
        )
    if cfg.pointcloud.permute:
        transformations.append(Permute(apply_to="pointcloud"))

    # SCALE
    if cfg.data.scale:
        transformations.append(Scale(from_inputs=True, multiplier=cfg.data.scale_multiplier))
    if cfg.aug.scale and apply_aug:
        if isinstance(cfg.aug.scale, (int, float)):
            transformations.append(Scale(amount=cfg.aug.scale, random=True))
        elif isinstance(cfg.aug.scale, Sequence):
            transformations.append(Scale(amount=list(cfg.aug.scale), random=True))
        elif isinstance(cfg.aug.scale, str):
            transformations.append(Scale(axes=cfg.aug.scale, amount=0.2, random=True))
        elif isinstance(cfg.aug.scale, bool):
            transformations.append(Scale(amount=0.2, random=True))

    # ROTATE
    if cfg.data.rotate:
        transformations.append(Rotate(from_inputs=True))
    elif cfg.aug.rotate and cfg.inputs.type != "depth_like" and apply_aug:
        transformations.append(
            Rotate(
                axes=cfg.aug.rotate,
                angles=(90.0, 180.0, 270.0) if cfg.aug.principal_rotations else None,
                upper_hemisphere=cfg.aug.upper_hemisphere,
                angle_from_index=split == "test" or cfg.aug.angle_from_index,
                choose_random=cfg.aug.principal_rotations,
            )
        )

    if cfg.data.frame != "world":
        transformations.append(Affine(replace=cfg.data.frame == "cam"))
        if cfg.data.frame == "net":
            transformations.append(Rotate(axes="x", from_inputs=True))

    if cfg.data.rot:
        transformations.append(Rotate(axes="xyz", angles=cfg.data.rot))
    if cfg.data.rot_x:
        transformations.append(Rotate(axes="x", angles=(cfg.data.rot_x,)))  # +90: flip yz
    if cfg.data.rot_y:
        transformations.append(Rotate(axes="y", angles=(cfg.data.rot_y,)))  # +90: front x to -z, -90: front -z to x
    if cfg.data.rot_z:
        transformations.append(Rotate(axes="z", angles=(cfg.data.rot_z,)))

    # NORMALIZE
    if cfg.norm.center or cfg.norm.scale or cfg.norm.to_front or cfg.norm.offset or cfg.norm.true_height:
        transformations.append(
            Normalize(
                center=cfg.norm.center,
                to_front=cfg.norm.to_front,
                scale=cfg.norm.scale,
                offset=cfg.norm.offset,
                true_height=cfg.norm.true_height,
                reference=cfg.norm.reference,
                scale_method=cfg.norm.method,
            )
        )

    ####################################################################################################################

    # INPUTS
    if cfg.inputs.permute:
        transformations.append(Permute(apply_to="inputs"))
    if cfg.inputs.voxelize:
        num_points = int(1e5) if cfg.inputs.num_points == "random" else cfg.inputs.num_points
        cache = bool(cfg.inputs.type == "pointcloud" and num_points and cfg.inputs.voxelize**3 <= num_points)
        transformations.append(
            VoxelizePointcloud(
                apply_to="inputs", resolution=cfg.inputs.voxelize, padding=cfg.norm.padding, cachable=cache
            )
        )
    elif cfg.inputs.bps.num_points:
        transformations.append(
            BPS(
                num_points=cfg.inputs.bps.num_points,
                resolution=cfg.inputs.bps.resolution,
                padding=cfg.norm.padding,
                method=cfg.inputs.bps.method,
                feature=cfg.inputs.bps.feature,
                basis=cfg.inputs.bps.basis,
            )
        )

    if any(t in cfg.inputs.type for t in ["image", "rgb", "color", "normals", "shading", "render"]):
        apply_to = "inputs"
        format = "torchvision"  # "ResNet18_Weights"
        if input_has_depth:
            apply_to = "inputs.image"

        if cfg.aug.border_noise and apply_aug:
            transformations.append(ImageBorderNoise(apply_to=apply_to))

        transformations.append(
            ImageToTensor(
                apply_to=apply_to,
                resize=cfg.inputs.resize,
                crop=cfg.inputs.crop if cfg.inputs.crop > 1 else None,
                format=format,
            )
        )

    # POINTS
    if load_points:
        if cfg.points.crop:
            transformations.append(CropPoints(padding=cfg.norm.padding, cache=cfg.inputs.cache and cfg.points.cache))
        if cfg[split].num_query_points and cfg.points.subsample:
            num_samples = (0.1, 1.0) if cfg[split].num_query_points == "random" else cfg[split].num_query_points
            transformations.append(
                SubsamplePoints(
                    num_samples=num_samples, in_out_ratio=cfg.points.in_out_ratio if split == "train" else None
                )
            )
        if cfg.points.permute:
            transformations.append(Permute(apply_to="points"))
        if cfg.seg.num_classes:
            apply_to = list()
            if cfg.seg.inputs:
                apply_to.append("inputs")
            if cfg.seg.points:
                apply_to.append("points")
            if cfg.seg.pointcloud:
                apply_to.append("pointcloud")
            if cfg.seg.mesh:
                apply_to.append("mesh.vertices")
            assert len(apply_to) > 0, "No input to apply segmentation to."
            transformations.append(SegmentationFromPartNet(apply_to=apply_to, num_classes=cfg.seg.num_classes))
        if cfg.points.voxelize and not cfg.points.from_mesh:
            transformations.append(
                VoxelizePoints(
                    resolution=cfg.points.voxelize,
                    padding=cfg.norm.padding,
                    cachable=cfg.points.voxelize**3 <= int(3e5),
                )
            )

        if cfg.data.sdf_from_occ:
            remove_pointcloud = not (cfg.vis.show and cfg.vis.pointcloud)
            transformations.append(SdfFromOcc(tsdf="tsdf" in str(cfg.train.loss), remove_pointcloud=remove_pointcloud))

    # MISC
    bbox = cfg.inputs.bbox or cfg.points.bbox or cfg.pointcloud.bbox or cfg.mesh.bbox
    if bbox:
        bbox_ref = (
            "inputs"
            if cfg.inputs.bbox
            else "points"
            if cfg.points.bbox
            else "pointcloud"
            if cfg.pointcloud.bbox
            else "mesh.vertices"
        )
        remove_ref = (
            not (cfg.vis.show and cfg.vis.pointcloud)
            and not cfg.norm.reference == "pointcloud"
            and not (cfg.aug.remove_angle or cfg.aug.edge_noise)
        )
        transformations.append(BoundingBox(reference=bbox_ref, remove_reference=remove_ref))
    if bbox and cfg.aug.bbox_jitter and apply_aug:
        transformations.append(BoundingBoxJitter(max_jitter=cfg.aug.bbox_jitter))

    min_num_points = dict()
    max_num_points = dict()
    fixed_size = cfg.inputs.voxelize or cfg.inputs.fps.num_points or cfg.inputs.bps.num_points
    if cfg.inputs.project and not fixed_size and cfg.inputs.type not in ["image", "idx", None]:
        min_num_points["inputs"] = cfg.inputs.min_num_points
        max_num_points["inputs"] = cfg.inputs.max_num_points
    if load_points and not cfg.points.voxelize:
        min_num_points["points"] = cfg.points.min_num_points
    if not cfg.pointcloud.voxelize and not cfg.pointcloud.fps.num_points:
        min_num_points["pointcloud"] = cfg.pointcloud.min_num_points
    if min_num_points or max_num_points:
        transformations.append(MinMaxNumPoints(min_num_points=min_num_points, max_num_points=max_num_points))

    if cfg.load.keys_to_keep and not cfg.vis.show:
        keys_to_keep = KeysToKeep(keys=cfg.load.keys_to_keep)
        transformations.append(keys_to_keep)
    exclude = list()
    if cfg.inputs.voxelize or cfg.inputs.bps.num_points:
        exclude.append("inputs")
    if load_points and cfg.points.voxelize:
        exclude.append("points")
    transformations.append(CheckDtype(exclude=exclude, dither=split == "train" and cfg.data.dither))

    return transformations


def get_shapenet(
    cfg: DictConfig,
    split: str,
    data_dir: Path,
    load_pointcloud: bool | str,
    load_points: bool,
    load_mesh: bool | str,
    load_cam: bool,
    load_normals: bool,
) -> ShapeNet:
    partnet = Path(cfg.dirs.partnet) if cfg.seg.num_classes or cfg.get("partnet", False) else None
    return ShapeNet(
        split=split,
        data_dir=data_dir,
        partnet_dir=partnet,
        inputs_dir=Path(cfg.dirs[cfg.inputs.data_dir]) if cfg.inputs.data_dir else data_dir,
        points_dir=Path(cfg.dirs[cfg.points.data_dir]) if cfg.points.data_dir else data_dir,
        pointcloud_dir=Path(cfg.dirs[cfg.pointcloud.data_dir]) if cfg.pointcloud.data_dir else data_dir,
        mesh_dir=Path(cfg.dirs[cfg.mesh.data_dir]) if cfg.mesh.data_dir else data_dir,
        image_dir_name=cfg.inputs.image_dir,
        cam_dir_name=cfg.inputs.cam_dir,
        categories=cfg.data.categories or cfg.data.objects,
        num_shards=cfg.data.num_shards[split],
        files_per_shard=cfg.data.num_files[split],
        num_views=cfg.train.num_views if split == "train" else 1,
        unscale=cfg.data.unscale,
        undistort=cfg.data.undistort,
        unrotate=cfg.data.unrotate,
        project=cfg.inputs.project,
        crop=cfg.inputs.crop == 1,
        resize=cfg.inputs.resize,
        convention=cfg.data.convention,
        sdf=cfg.implicit.sdf,
        tsdf="tsdf" in str(cfg.train.loss),
        split_file=cfg.files.split[split],
        pointcloud_file=cfg.files.pointcloud,
        points_file=cfg.files.points[split],
        voxels_file=cfg.files.voxels,
        normals_file=cfg.files.normals,
        mesh_file=cfg.files.mesh,
        load_random_inputs=cfg.inputs.load_random if split != "test" else False,
        load_pointcloud=load_pointcloud,
        load_normals=load_normals,
        load_points=load_points,
        load_all_points=cfg.points.load_all,
        load_random_points=cfg.points.load_random if split != "test" else False,
        load_surface_points=cfg.points.load_surface or cfg[split].load_surface_points,
        load_meshes=load_mesh,
        load_cam=load_cam,
        inputs_type=cfg.inputs.type,
        type_p=cfg.inputs.type_p,
        weighted=cfg.load.weighted,
        padding=cfg.norm.padding,
        cls_weight=cfg.cls.weight,
        seg_weight=cfg.seg.weight,
        from_hdf5=cfg.load.hdf5,
        transforms=get_transformations(cfg, split),
        verbose=cfg.log.verbose > 1,
    )


def get_ycb(cfg: DictConfig, split: str, load_pointcloud: bool, load_voxels: bool) -> YCB:
    if split in ["train", "val", "test"]:
        return YCB(
            data_dir=cfg.dirs.ycb,
            sdf=cfg.implicit.sdf,
            pointcloud_file="samples/nontextured/surface.npy" if cfg.implicit.sdf else None,
            points_file="samples/nontextured/deepsdf.npy" if cfg.implicit.sdf and split == "train" else None,
            voxels_file="google_16k/sdf/model.binvox" if cfg.implicit.sdf and load_voxels else None,
            normals_file="samples/nontextured/normals.npy" if cfg.implicit.sdf else None,
            mesh_file="mesh/nontextured.off" if cfg.implicit.sdf else "nontextured.ply",
            # pointcloud_file="samples/nontextured/surface.npy",
            # points_file="samples/nontextured/if_net.npy" if split == "train" else None,
            # voxels_file="google_16k/sdf/model.binvox",
            # normals_file="samples/nontextured/normals.npy",
            # mesh_file="mesh/nontextured.off",
            load_pointcloud=load_pointcloud,
            load_points=True,
            load_meshes=cfg.norm.reference == "automatica",
            load_real_data=False,
            objects=(6, 19, 21),
            split=split,
            padding=cfg.norm.padding,
            transforms={"data": get_transformations(cfg, split)},
        )
    else:
        print("YCB transforms:")
        transforms: dict[str, list[Transform]] = {"inputs": list(), "data": list()}
        transforms["inputs"].append(CropPointcloudWithMesh(apply_to="inputs"))
        if cfg.aug.remove_outlier:
            transforms["inputs"].append(
                ProcessPointcloud(
                    apply_to="inputs",
                    remove_outlier=OutlierTypes.STATISTICAL,
                    outlier_std_ratio=1,
                )
            )
        if cfg.num_input_points:
            transforms["inputs"].append(SubsamplePointcloud(apply_to="inputs", num_samples=cfg.num_input_points))
        if cfg.data.frame:
            transforms["data"].append(Affine(replace=cfg.data.frame == "cam"))
            if cfg.data.frame == "net":
                transforms["data"].append(Rotate(axes="x", from_inputs=True))
        if cfg.inputs.type == "justin":
            if cfg.aug.remove_outlier:
                transforms["inputs"].append(
                    ProcessPointcloud(
                        apply_to="inputs",
                        downsample=DownsampleTypes.VOXEL,
                        downsample_factor=cfg.aug.voxel_size,
                        remove_outlier=OutlierTypes.STATISTICAL,
                        outlier_std_ratio=2,
                    )
                )
            else:
                transforms["inputs"].append(
                    ProcessPointcloud(
                        apply_to="inputs", downsample=DownsampleTypes.VOXEL, downsample_factor=cfg.aug.voxel_size
                    )
                )
        transforms["data"].append(
            Normalize(
                center="xyz" if "center" in cfg.norm.normalize else "",
                scale="scale" in cfg.norm.normalize,
                reference=cfg.norm.reference,
            )
        )
        print()
        return YCB(
            cfg.dirs.ycb,
            load_pointcloud=load_pointcloud,
            load_meshes=cfg.norm.reference == "automatica",
            cam_id_range=(2, 3) if cfg.inputs.type == "justin" else (1, 5),
            merge_angles=7 if cfg.inputs.type == "justin" else 0,
            stride=2 if cfg.inputs.type == "justin" else 1,
            filter_discontinuities=7 if cfg.inputs.type == "justin" else None,
            padding=cfg.norm.padding,
            transforms=transforms,
        )


def get_bop(cfg: DictConfig, ds: str):
    name = ds.split("_")[1]
    if name == "hb":
        objects = (6,)
    elif name == "lm":
        objects = (7,)
    elif name == "tyol":
        objects = (3, 4, 5, 6, 20, 21)
    elif name == "ycbv":
        objects = (14,)
    else:
        raise NotImplementedError(f"Unknown BOP dataset {name}.")
    objects = objects if cfg.data.objects is None else cfg.data.objects

    camera = None
    if "kinect" in ds:
        camera = "kinect"
    elif "primesense" in ds:
        camera = "primesense"

    scene = None
    if name == "ycbv" and "55" in ds:
        scene = "000055"
    elif name == "ycbv" and "48" in ds:
        scene = "000048"

    if cfg.load.keys_to_keep:
        keys_to_keep = KeysToKeep(keys=cfg.load.keys_to_keep)
    else:
        keys_to_keep = KeysToKeep()
    if cfg.vis.show:
        keys_to_keep = KeysToKeep()

    transforms: dict[str, list[Transform]] = {
        "inputs": [],
        "points": [Scale(amount=0.001)],
        "pointcloud": [Scale(amount=0.001)],
        "mesh": [Scale(amount=0.001)],
        "data": [  # Normalize(reference="mesh"),
            # PointsFromMesh(cache=True),
            # PointcloudFromMesh(cache=True),
            # Normalize(reference="mesh", reverse=True),
            ApplyPose(apply_to=("points", "pointcloud", "mesh.vertices", "bbox", "partnet.points")),
            Rotate(axes="x", from_inputs=True),
            Rotate(axes="x", angles=(-90,)),
            Normalize(
                center=cfg.norm.center,
                to_front=cfg.norm.to_front,
                scale=cfg.norm.scale,
                offset=cfg.norm.offset,
                true_height=cfg.norm.true_height,
                reference=cfg.norm.reference,
                scale_method=cfg.norm.method,
            ),
            LoadUncertain() if cfg.points.load_uncertain else Return(),
            CropPoints(padding=cfg.norm.padding),
        ],
    }
    if cfg.inputs.voxelize:
        transforms["inputs"].append(
            VoxelizePointcloud(apply_to="inputs", resolution=cfg.inputs.voxelize, padding=cfg.norm.padding)
        )
    transforms["data"].extend([keys_to_keep, CheckDtype()])

    return BOP(
        name=name,
        split="val" if name == "hb" else "test",
        data_dir=cfg.dirs["bop"],
        sdf=cfg.implicit.sdf,
        pointcloud_file=cfg.files.pointcloud,
        points_file=cfg.files.points.test,
        normals_file=cfg.files.normals,
        mesh_dir="mesh",  # f"../../models_eval"
        mesh_file="obj_000014.ply" if name == "ycbv" else None,
        pose_file="transform.npz",  # None
        objects=objects,
        scene=scene,
        max_outlier_std=1,
        max_correspondence_distance=0.005,
        camera=camera,
        padding=cfg.norm.padding,
        transforms=transforms,
        verbose=cfg.log.verbose > 1,
    )


def get_tabletop_transforms(
    cfg: DictConfig,
    split: str,
    load_3d: bool = False,
    patch_size: int | None = 14,
    scale: float = 0.5,
    normalize: bool = True,
) -> T.Transform | None:
    to_tensor = [T.ToImage(), T.ToDtype(torch.float32)]
    if cfg.inputs.type not in ["depth", "kinect", "kinect_sim"]:
        to_tensor = [T.ToImage(), T.ToDtype(torch.float32, scale=normalize)]
        if normalize:
            to_tensor.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    size = int(patch_size * 36 * scale) if patch_size else int(512 * scale)
    antialias = cfg.inputs.type == "image"

    transforms = list()
    if isinstance(cfg.inputs.crop, int) and cfg.inputs.crop > 1:
        transforms.append(T.CenterCrop(size=cfg.inputs.crop))

    if split == "train":
        if cfg.aug.resize:
            if cfg.aug.resize == "strong":
                transforms.append(
                    [
                        T.RandomChoice(
                            [
                                T.Compose(
                                    [
                                        T.ScaleJitter(
                                            target_size=(size, size), scale_range=(0.6, 1.4), antialias=antialias
                                        ),
                                        T.RandomResizedCrop(
                                            size=size, scale=(0.9, 1.0), ratio=(0.9, 1.1), antialias=antialias
                                        ),
                                    ]
                                ),
                                T.Compose(
                                    [
                                        T.ScaleJitter(
                                            target_size=(size, size), scale_range=(0.8, 1.2), antialias=antialias
                                        ),
                                        T.RandomResizedCrop(
                                            size=size, scale=(0.2, 0.9), ratio=(0.75, 1.33), antialias=antialias
                                        ),
                                    ]
                                ),
                            ],
                            p=[0.5, 0.5],
                        ),
                    ]
                )
            else:
                transforms.append(
                    [
                        T.RandomChoice(
                            [
                                T.Compose(
                                    [
                                        T.ScaleJitter(
                                            target_size=(size, size), scale_range=(0.9, 1.1), antialias=antialias
                                        ),
                                        T.RandomResizedCrop(
                                            size=size, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=antialias
                                        ),
                                    ]
                                ),
                                T.Compose(
                                    [
                                        T.ScaleJitter(
                                            target_size=(size, size), scale_range=(0.95, 1.05), antialias=antialias
                                        ),
                                        T.RandomResizedCrop(
                                            size=size, scale=(0.75, 0.95), ratio=(0.9, 1.1), antialias=antialias
                                        ),
                                    ]
                                ),
                            ],
                            p=[0.8, 0.2],
                        ),
                    ]
                )
        elif scale != 1.0 or patch_size:
            transforms.append(T.Resize(size=size, max_size=int(1333 * scale), antialias=antialias))
        if not load_3d and cfg.aug.flip:
            # TODO: Add height/width transpose transform which works with 3D data
            transforms.append(T.RandomHorizontalFlip())
        if cfg.inputs.type not in ["depth", "kinect", "kinect_sim"]:
            transforms.append(T.RandomPhotometricDistort())
    elif split == "val":
        if scale != 1.0 or patch_size:
            transforms.extend(
                [
                    T.Resize(size=size, max_size=int(1333 * scale), antialias=antialias),
                    T.CenterCrop(size=size),
                ]
            )
    elif split == "test":
        if scale != 1.0:
            transforms.append(T.Resize(size=size, max_size=int(1333 * scale), antialias=antialias))
        if patch_size:
            transforms.append(CenterPad(multiple=patch_size))
    if transforms + to_tensor:
        return T.Compose(transforms + to_tensor)


def get_tabletop(cfg: DictConfig, split: str, ds: str = "tabletop") -> TableTop:
    load_3d = cfg.get("load_3d")
    inputs_3d = cfg.inputs.type in ["depth", "kinect", "kinect_sim", "rgbd"] and cfg.inputs.project
    collate_3d = cfg.get("collate_3d")
    transforms = get_tabletop_transforms(
        cfg=cfg,
        split=split,
        load_3d=load_3d or inputs_3d,
        patch_size=cfg.get("patch_size", 14),
        scale=cfg.get("scale", 0.5),
    )

    transforms_3d: list[Transform] = [KeysToKeep(cfg.load.keys_to_keep)]
    if load_3d or inputs_3d:
        subsample_points = cfg.points.subsample and cfg[split].num_query_points
        min_num_points = dict()
        max_num_points = dict()
        apply_to: set[str] = set()
        if inputs_3d:
            apply_to.add("inputs")
            min_num_points["inputs"] = cfg.inputs.min_num_points
            max_num_points["inputs"] = cfg.inputs.max_num_points
        if load_3d or cfg.get("filter", False):
            apply_to.add("points")
            min_num_points["points"] = cfg.points.min_num_points
            if cfg.points.max_num_points:
                max_num_points["points"] = cfg.points.max_num_points
        transforms_3d.append(MinMaxNumPoints(min_num_points, apply_to=apply_to))
        if cfg.data.frame != "world":
            transforms_3d.append(Affine(replace=cfg.data.frame == "cam", remove_roll=split != "train"))
            if cfg.data.frame == "net":
                transforms_3d.append(Rotate(axes="x", from_inputs=True))
        if split == "train":
            if cfg.aug.rotate:
                transforms_3d.append(Rotate(axes="z"))
            if cfg.aug.scale:
                scale: float | tuple[float, float] | list[float] = (0.0, 0.1)
                if isinstance(cfg.aug.scale, float):
                    scale = cfg.aug.scale
                elif isinstance(cfg.aug.scale, Sequence):
                    scale = [float(x) for x in cfg.aug.scale]
                transforms_3d.append(Scale(amount=scale, random=True))
            if cfg.aug.translate:
                translate = 0.1
                if isinstance(cfg.aug.translate, float):
                    translate = cfg.aug.translate
                transforms_3d.append(Translate(axes="xy", amount=translate, random=True))
            if cfg.aug.noise and inputs_3d:
                noise: float | tuple[float, float] = (0.0, 0.002)
                clip_noise = 10 * noise[1]
                if isinstance(cfg.aug.noise, float):
                    noise = cfg.aug.noise
                    clip_noise = cfg.aug.clip_noise if cfg.aug.clip_noise else 10 * noise
                elif isinstance(cfg.aug.noise, Sequence):
                    seq = [float(x) for x in cfg.aug.noise]
                    if len(seq) >= 2:
                        noise = (seq[0], seq[1])
                        clip_noise = 10 * seq[1]
                    elif len(seq) == 1:
                        noise = seq[0]
                        clip_noise = cfg.aug.clip_noise if cfg.aug.clip_noise else 10 * noise
                transforms_3d.append(AddGaussianNoise(stddev=noise, clip=clip_noise))
        if cfg.norm.center or cfg.norm.scale:
            transforms_3d.append(
                Normalize(
                    center=cfg.norm.center,
                    scale=cfg.norm.scale,
                    scale_factor=cfg.norm.scale_factor,
                    scale_quantiles=(0.02, 0.98),
                    center_method="median",
                    reference=cfg.norm.reference,
                )
            )
        if inputs_3d:
            if cfg.inputs.crop:
                transforms_3d.append(
                    CropPointcloud(
                        apply_to="inputs",
                        mode=cast(Literal["cube", "sphere", "frustum"], cfg.inputs.crop)
                        if isinstance(cfg.inputs.crop, str)
                        else "cube",
                        padding=cfg.norm.padding,
                        scale_factor=cfg.norm.scale_factor,
                    )
                )
        if load_3d or cfg.get("filter", False):
            if cfg.points.crop:
                transforms_3d.append(
                    CropPoints(
                        # cfg values are constrained in config; cast to transform literal for static typing
                        mode=cast(Literal["cube", "sphere", "frustum"], cfg.points.crop)
                        if isinstance(cfg.points.crop, str)
                        else "cube",
                        padding=cfg.norm.padding,
                        scale_factor=cfg.norm.scale_factor,
                    )
                )
        transforms_3d.append(MinMaxNumPoints(min_num_points, max_num_points, apply_to))
        if inputs_3d:
            if cfg.inputs.num_points:
                transforms_3d.append(SubsamplePointcloud(apply_to="inputs", num_samples=cfg.inputs.num_points))
        if load_3d or cfg.get("filter", False):
            if subsample_points:
                transforms_3d.append(
                    SubsamplePoints(
                        cfg[split].num_query_points,
                        in_out_ratio=cfg.points.in_out_ratio if split == "train" else None,
                        padding=cfg.norm.padding,
                        scale_factor=cfg.norm.scale_factor,
                        voxel_res=16 if (split == "train" and cfg.points.subsample == "uniform") else None,
                        per_voxel_cap=2,
                    )
                )
            elif cfg.points.voxelize:
                transforms_3d.append(
                    VoxelizePoints(
                        resolution=cfg.points.voxelize,
                        padding=cfg.norm.padding,
                        scale_factor=cfg.norm.scale_factor,
                        bounds=cfg.norm.bounds,
                    )
                )
    transforms_3d.append(CheckDtype(dither=split == "train" and cfg.data.dither))

    if cfg.data.split:
        transforms_3d.append(SplitData(split_mesh=cfg.files.mesh, split_pointcloud=cfg.files.pointcloud))

    points_fraction = cfg.get("points_fraction", "auto")
    if split == "test" or not cfg.points.subsample or collate_3d not in ["stack", "cat"]:
        points_fraction = None
    tabletop_cls = cast(Any, TableTop)
    return tabletop_cls(
        data_dir=Path(cfg.dirs[ds]),
        load_color=cfg.inputs.type not in ["depth", "kinect", "kinect_sim"],
        load_depth=cfg.inputs.type
        if cfg.inputs.type in ["kinect", "kinect_sim"]
        else cfg.inputs.type in ["depth", "rgbd"],
        load_normals=cfg.inputs.normals or cfg.pointcloud.normals or cfg.points.from_pointcloud,
        apply_filter=cfg.get("filter", False),
        project=cfg.inputs.project,
        data_dir_3d=Path(cfg.dirs["shapenet_v1_fused"]) if load_3d else None,
        split=split,
        mesh_file=cfg.files.mesh,
        pcd_file=cfg.files.pointcloud,
        points_file=cfg.files.points[split],
        # crop_points=cfg.points.crop if collate_3d in ["stack", "cat"] else None,
        sample_free_points=cfg.get("sample_free", "cube") if split == "train" else "cube",
        points_fraction=points_fraction,
        append_pcd_to_points=cfg.points.from_pointcloud and split == "train",
        near=cfg.implicit.near,
        far=cfg.implicit.far,
        padding=cfg.norm.padding,
        scale_factor=cfg.norm.scale_factor,
        collate_3d=collate_3d,
        stack_2d=cfg.get("stack_2d", False),
        apply_pose=cfg.get("apply_pose", True),
        from_hdf5=cfg.load.hdf5,
        cache_points=cfg.points.cache,
        transforms=transforms,
        transforms_3d=transforms_3d,
    )


def get_graspnet(cfg: DictConfig, ds: str, split: str) -> GraspNetEval:
    """
    Construct the GraspNet evaluation dataset from the raw directory layout.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration node with directory and dataset options.
    ds : str
        Dataset identifier from cfg.data.{train,val,test}_ds (e.g., "graspnet", "graspnet_test_novel").
    split : str
        Requested split from the get_dataset caller ("train", "val", or "test").
    """
    # Derive the GraspNet split to use (train/val/test_seen/test_similar/test_novel/...)
    suffix = None
    if "_" in ds:
        prefix, maybe_suffix = ds.split("_", 1)
        if prefix == "graspnet" and maybe_suffix:
            suffix = maybe_suffix

    requested_split = cfg.data.get("graspnet_split")
    if requested_split is None:
        requested_split = suffix if suffix is not None else ("test_novel" if split == "test" else split)

    # Camera stream (kinect or realsense)
    camera = cfg.data.get("graspnet_camera", cfg.data.get("camera", "kinect"))

    # Optional overrides
    depth_scale = cfg.data.get("graspnet_depth_scale", 1000.0)
    scene_ids = cfg.data.get("graspnet_scene_ids")
    if scene_ids is not None:
        scene_ids = [int(s) for s in scene_ids]

    mesh_dir = None
    if "graspnet_models" in cfg.dirs:
        mesh_dir = cfg.dirs["graspnet_models"]
    elif hasattr(cfg.dirs, "get"):
        mesh_dir = cfg.dirs.get("graspnet_models", None)
    mesh_dir_path = Path(mesh_dir) if mesh_dir is not None else None

    transforms_3d: list[Transform] = []
    if cfg.norm.center or cfg.norm.scale:
        transforms_3d.append(
            Normalize(
                center=cfg.norm.center,
                scale=cfg.norm.scale,
                scale_factor=cfg.norm.scale_factor,
                scale_quantiles=(0.02, 0.98),
                center_method="median",
                reference=cfg.norm.reference,
            )
        )
    if cfg.inputs.crop:
        transforms_3d.append(
            CropPointcloud(
                apply_to="inputs",
                mode=cast(Literal["cube", "sphere", "frustum"], cfg.inputs.crop)
                if isinstance(cfg.inputs.crop, str)
                else "cube",
                padding=cfg.norm.padding,
                scale_factor=cfg.norm.scale_factor,
            )
        )
    if cfg.points.crop:
        transforms_3d.append(
            CropPoints(
                mode=cast(Literal["cube", "sphere", "frustum"], cfg.points.crop)
                if isinstance(cfg.points.crop, str)
                else "cube",
                padding=cfg.norm.padding,
                scale_factor=cfg.norm.scale_factor,
            )
        )
    if cfg.inputs.num_points:
        transforms_3d.append(SubsamplePointcloud(apply_to="inputs", num_samples=cfg.inputs.num_points))
    if cfg.points.subsample and cfg[split].num_query_points:
        transforms_3d.append(
            SubsamplePoints(
                cfg[split].num_query_points,
                in_out_ratio=cfg.points.in_out_ratio if split == "train" else None,
                padding=cfg.norm.padding,
                scale_factor=cfg.norm.scale_factor,
                voxel_res=16 if (split == "train" and cfg.points.subsample == "uniform") else None,
                per_voxel_cap=2,
            )
        )
    elif cfg.points.voxelize:
        transforms_3d.append(
            VoxelizePoints(
                resolution=cfg.points.voxelize,
                padding=cfg.norm.padding,
                scale_factor=cfg.norm.scale_factor,
                bounds=cfg.norm.bounds,
            )
        )
    if cfg.get("refine_pose"):
        transforms_3d.extend(
            [
                RefinePose(point_to_plane=True, remove_outlier=True),
                RefinePosePerInstance(point_to_plane=True, remove_outlier=True),
            ]
        )

    keys_to_kep = cfg.load.keys_to_keep
    if cfg.data.split:
        transforms_3d.append(SplitData(split_text=False))
    else:
        keys_to_kep = [key for key in keys_to_kep if not key.startswith("mesh")]
    transforms_3d.extend([KeysToKeep(keys_to_kep), CheckDtype(dither=split == "train" and cfg.data.dither)])

    return GraspNetEval(
        root=Path(cfg.dirs["graspnet"]),
        split=requested_split,
        camera=camera,
        # camera="realsense",
        depth_scale=depth_scale,
        load_mesh=True,
        mesh_dir=mesh_dir_path,
        load_pcd=cfg.vis.show or cfg.vis.save,
        scene_ids=scene_ids,
        project=cfg.inputs.project,
        crop_to_mesh=True,
        generate_points=True,
        collate_points=cfg.get("collate_3d"),
        load_points=True,
        sample_scene_points=True,
        load_label=True,
        stack_2d=cfg.get("stack_2d", False),
        one_view_per_scene=cfg.get("single_view", False),
        filter_background=0.01,
        transforms=transforms_3d,
    )


def get_torchvision_dataset(cfg: DictConfig, ds: str, split: str) -> TorchvisionDatasetWrapper:
    if ds == "mnist":
        data = MNIST(Path(cfg.dirs[ds]).parent, transform=ToTensor(), train=split == "train", download=True)
    elif ds == "fmnist":
        data = FashionMNIST(Path(cfg.dirs[ds]).parent, transform=ToTensor(), train=split == "train", download=True)
    elif ds == "cifar10":
        data = CIFAR10(Path(cfg.dirs[ds]), transform=ToTensor(), train=split == "train", download=True)
    else:
        raise NotImplementedError(f"Unknown dataset {ds}.")
    return TorchvisionDatasetWrapper(data)


def get_coco_transforms(
    cfg: DictConfig, split: str, patch_size: int | None = 14, scale: float = 0.25, normalize: bool = True
) -> T.Transform:
    to_tensor = [T.ToImage(), T.ToDtype(torch.float32, scale=normalize)]
    if normalize:
        to_tensor.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    size = int(patch_size * 72 * scale) if patch_size else int(1024 * scale)
    transforms = list()
    if patch_size:
        transforms.append(CenterPad(multiple=patch_size))
    transforms.extend(
        [
            # T.RandomApply([T.RandomResizedCrop(size=s, scale=(0.1 / scale, 2.0)) for s in sizes]),
            T.RandomResizedCrop(size=size, scale=(0.1 / scale, 2.0)),
            # T.Resize(size=size, max_size=int(1333 * scale)),
            T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.RandomPhotometricDistort(),
            # T.ScaleJitter(target_size=(size, size), scale_range=(0.1 / scale, 2.0)),
            # T.RandomPerspective(interpolation=T.InterpolationMode.BILINEAR, fill=0),
            # T.RandomRotation(degrees=(0, 180), interpolation=T.InterpolationMode.BILINEAR, fill=0),
            # T.RandomAffine(degrees=(30, 70),
            #                translate=(0.1, 0.3),
            #                scale=(0.5, 0.75),
            #                interpolation=T.InterpolationMode.BILINEAR,
            #               fill=0),
            # T.AutoAugment(interpolation=T.InterpolationMode.BILINEAR, fill=0),
            # T.AugMix(interpolation=T.InterpolationMode.BILINEAR, fill=0),
            # T.RandAugment(),
            # T.TrivialAugmentWide(),
        ]
    )
    # if patch_size:
    #     transforms.append(CenterPad(multiple=patch_size))
    if split == "val":
        transforms = [
            T.Resize(size=size, max_size=int(1333 * scale)),
            T.CenterCrop(size=size),
        ]
    elif split == "test":
        transforms = list()
        if normalize:
            transforms.append(T.Resize(size=int(800 * scale), max_size=int(1333 * scale)))
        if patch_size:
            transforms.append(CenterPad(multiple=patch_size))
    return T.Compose(transforms + to_tensor)


def get_dataset(
    cfg: DictConfig,
    splits: tuple[str, ...] = ("train", "val"),
    load_points: bool = False,
    load_pointcloud: bool = False,
    load_mesh: bool | str = False,
) -> dict[str, Dataset]:
    if cfg.data.val_ds is None and "val" in splits:
        cfg.data.val_ds = cfg.data.train_ds
    if cfg.data.test_ds is None and "test" in splits:
        cfg.data.test_ds = cfg.data.val_ds
    train_ds = [cfg.data.train_ds] if isinstance(cfg.data.train_ds, str) else cfg.data.train_ds
    val_ds = [cfg.data.val_ds] if isinstance(cfg.data.val_ds, str) else cfg.data.val_ds
    test_ds = [cfg.data.test_ds] if isinstance(cfg.data.test_ds, str) else cfg.data.test_ds
    train_data = None
    val_data = None
    test_data = None

    vis_points = cfg.vis.show and (cfg.vis.occupancy or cfg.vis.points)
    vis_pointcloud = cfg.vis.show and cfg.vis.pointcloud
    vis_mesh = cfg.vis.show and cfg.vis.mesh
    vis_voxels = cfg.vis.show and cfg.vis.voxels

    cls_only = (cfg.cls.num_classes is not None or cfg.seg.num_classes is not None) and not cfg.cls.occupancy
    is_pcd_net = cfg.model.arch in ["realnvp", "psgn", "pcn", "snowflakenet", "shapeformer", "pointnet"]

    load_points = (
        not (cfg.points.from_mesh or is_pcd_net or cls_only or cfg.points.from_pointcloud) or vis_points or load_points
    )
    load_voxels = vis_voxels
    load_normals = "normals" in cfg.inputs.type or cfg.inputs.normals
    cfg.pointcloud.normals = cfg.pointcloud.normals or (
        cfg.aug.remove_angle
        or cfg.aug.edge_noise
        or cfg.aug.move_sphere
        or (cfg.inputs.type == "shading" and not cfg.inputs.normals)
    )
    load_pointcloud_mode: bool | str = cast(
        bool | str,
        (
            not cls_only
            and (
                is_pcd_net
                or cfg.pointcloud.bbox
                or cfg.pointcloud.normals
                or cfg.data.sdf_from_occ
                or cfg.aug.remove_angle
                or cfg.aug.edge_noise
                or cfg.points.from_pointcloud
            )
        )
        or vis_pointcloud
        or load_pointcloud,
    )
    load_mesh = (
        load_mesh
        or vis_mesh
        or cfg.pointcloud.from_mesh
        or cfg.points.from_mesh
        or cfg.norm.reference == "mesh"
        or cfg.mesh.bbox
    )
    load_cam = False

    if (cfg.val.mesh == "all" or "pcd" in str(cfg.val.mesh)) and not load_pointcloud_mode:
        getattr(logger, "debug_level_2", logger.debug)("Loading pointclouds for validation")
        load_pointcloud_mode = True
    elif (cfg.val.mesh == "all" or "mesh" in str(cfg.val.mesh)) and not load_mesh:
        getattr(logger, "debug_level_2", logger.debug)("Loading meshes for validation")
        load_mesh = "val_only" if cfg.val.batch_size == 1 else "path_only"
        if load_mesh == "path_only":
            logger.warning("Transformations will not be applied to meshes")

    if (cfg.norm.true_height or cfg.norm.reference == "pointcloud") and not load_pointcloud_mode:
        load_pointcloud_mode = True
        if cfg.data.frame == "world" and not (cfg.data.rot or cfg.aug.rotate):
            getattr(logger, "debug_level_1", logger.debug)("Loading point cloud min/max values")
            load_pointcloud_mode = "min_max_only"
    if cfg.pointcloud.from_mesh:
        load_mesh = load_mesh or True
        load_pointcloud_mode = False

    if "train" in splits:
        for ds in train_ds:
            if "shapenet" in ds:
                load_pcd_train = (
                    False if (cfg.val.mesh == "all" or "pcd" in str(cfg.val.mesh)) else load_pointcloud_mode
                )
                load_mesh_train = False if (cfg.val.mesh == "all" or "mesh" in str(cfg.val.mesh)) else load_mesh
                data = get_shapenet(
                    cfg=cfg,
                    split="train",
                    data_dir=Path(cfg.dirs[ds]),
                    load_pointcloud=load_pcd_train,
                    load_points=load_points and cfg.files.points.train,
                    load_mesh=load_mesh_train,
                    load_cam=load_cam,
                    load_normals=load_normals,
                )
            elif ds == "completion3d":
                data = Completion3D(
                    data_dir=cfg.dirs[ds],
                    shapenet_dir=cfg.dirs.onet,
                    categories=cfg.data.categories,
                    split="train",
                    transforms=None,
                    load_pointcloud=bool(load_pointcloud_mode),
                    load_points=load_points,
                )
            elif ds == "ycb":
                data = get_ycb(
                    cfg=cfg, split="train", load_pointcloud=bool(load_pointcloud_mode), load_voxels=load_voxels
                )
            elif ds == "modelnet40":
                data = ModelNet(
                    split="train",
                    data_dir=Path(cfg.dirs[ds]),
                    transforms={"inputs": get_transformations(cfg, split="train")},
                    verbose=cfg.log.verbose > 1,
                )
            elif ds in ["mnist", "fmnist", "cifar10"]:
                data = get_torchvision_dataset(cfg, ds, "train")
            elif ds == "coco":
                transforms = get_coco_transforms(
                    cfg, split="train", patch_size=cfg.get("patch_size", 14), scale=cfg.get("coco_scale", 0.25)
                )
                data = CocoInstanceSegmentation(data_dir=Path(cfg.dirs[ds]), split="train", transforms=transforms)
            elif "tabletop" in ds:
                data = get_tabletop(cfg, split="train", ds=ds)
            elif ds.startswith("graspnet"):
                data = get_graspnet(cfg, ds, split="train")
            else:
                objects = cfg.data.objects
                if objects is not None:
                    raise NotImplementedError("Objects are not implemented for custom datasets.")
                if ds in ["bunny", "armadillo", "torus", "sphere"]:
                    raise NotImplementedError(f"Unknown dataset {ds}.")
                    objects = (get_file(ds),)
                    data_dir = objects[0].parent.parent
                else:
                    data_dir = Path(cfg.dirs[ds])
                data = get_shapenet(
                    cfg,
                    "train",
                    data_dir,
                    load_pointcloud_mode,
                    load_points and cfg.files.points.train,
                    load_mesh,
                    load_cam,
                    load_normals,
                )
            if train_data is None:
                train_data = data
            else:
                train_data += data
    if "val" in splits:
        for ds in val_ds:
            if "shapenet" in ds:
                data = get_shapenet(
                    cfg,
                    "val",
                    Path(cfg.dirs[ds]),
                    load_pointcloud_mode,
                    load_points and cfg.files.points.val,
                    load_mesh,
                    load_cam,
                    load_normals,
                )
            elif ds == "completion3d":
                data = Completion3D(
                    data_dir=cfg.dirs[ds],
                    shapenet_dir=cfg.dirs.onet,
                    categories=cfg.data.categories,
                    split="val",
                    transforms=None,
                    load_pointcloud=bool(load_pointcloud_mode),
                    load_points=load_points,
                )
            elif ds == "ycb":
                data = get_ycb(
                    cfg=cfg, split="val", load_pointcloud=bool(load_pointcloud_mode), load_voxels=load_voxels
                )
            elif ds == "modelnet40":
                data = ModelNet(
                    split="test",  # ModelNet has no validation set
                    data_dir=Path(cfg.dirs[ds]),
                    transforms={"inputs": get_transformations(cfg, split="test")},
                    verbose=cfg.log.verbose > 1,
                )
            elif ds in ["mnist", "fmnist", "cifar10"]:
                data = get_torchvision_dataset(cfg, ds, "val")
            elif ds == "coco":
                transforms = get_coco_transforms(
                    cfg, split="val", patch_size=cfg.get("patch_size", 14), scale=cfg.get("coco_scale", 0.25)
                )
                data = CocoInstanceSegmentation(data_dir=Path(cfg.dirs[ds]), split="val", transforms=transforms)
            elif "tabletop" in ds:
                data = get_tabletop(cfg, split="val", ds=ds)
            elif ds.startswith("graspnet"):
                data = get_graspnet(cfg, ds, split="val")
            else:
                objects = cfg.data.objects
                if objects is not None:
                    raise NotImplementedError("Objects are not implemented for custom datasets.")
                if ds in ["bunny", "armadillo", "torus", "sphere"]:
                    raise NotImplementedError(f"Unknown dataset {ds}.")
                    objects = (get_file(ds),)
                    data_dir = objects[0].parent.parent
                else:
                    data_dir = Path(cfg.dirs[ds])
                data = get_shapenet(
                    cfg,
                    "val",
                    data_dir,
                    load_pointcloud_mode,
                    load_points and cfg.files.points.val,
                    load_mesh,
                    load_cam,
                    load_normals,
                )
            if val_data is None:
                val_data = data
            else:
                val_data += data
    if "test" in splits:
        for ds in test_ds:
            if "shapenet" in ds:
                data = get_shapenet(
                    cfg,
                    "test",
                    Path(cfg.dirs[ds]),
                    load_pointcloud_mode,
                    load_points and cfg.files.points.test,
                    load_mesh,
                    load_cam,
                    load_normals,
                )
            elif ds == "completion3d":
                data = Completion3D(
                    data_dir=cfg.dirs[ds],
                    shapenet_dir=cfg.dirs.onet,
                    categories=cfg.data.categories,
                    split="val",  # Completion3D has no test set
                    transforms=None,
                    load_pointcloud=bool(load_pointcloud_mode),
                    load_points=load_points,
                )
            elif ds == "ycb":
                data = get_ycb(cfg, split="test", load_pointcloud=bool(load_pointcloud_mode), load_voxels=load_voxels)
            elif "bop" in ds:
                data = get_bop(cfg, ds)
            elif ds == "modelnet40":
                data = ModelNet(
                    split="test",
                    data_dir=Path(cfg.dirs[ds]),
                    transforms={"inputs": get_transformations(cfg, split="test")},
                    verbose=cfg.log.verbose > 1,
                )
            elif ds in ["mnist", "fmnist", "cifar10"]:
                data = get_torchvision_dataset(cfg, ds, "test")
            elif ds == "coco":
                transforms = get_coco_transforms(
                    cfg,
                    split="test",
                    scale=1.0 if cfg.model.arch == "mask_rcnn" else cfg.get("coco_scale", 0.25),
                    patch_size=None if cfg.model.arch == "mask_rcnn" else cfg.get("patch_size", 14),
                    normalize=cfg.model.arch != "mask_rcnn",
                )
                data = CocoInstanceSegmentation(data_dir=Path(cfg.dirs[ds]), split="test", transforms=transforms)
            elif "tabletop" in ds:
                data = get_tabletop(cfg, split="test", ds=ds)
            elif ds.startswith("graspnet"):
                data = get_graspnet(cfg, ds, split="test")
            else:
                objects = cfg.data.objects
                if objects is not None:
                    raise NotImplementedError("Objects are not implemented for custom datasets.")
                if ds in ["bunny", "armadillo", "torus", "sphere"]:
                    raise NotImplementedError(f"Unknown dataset {ds}.")
                    objects = (get_file(ds),)
                    data_dir = objects[0].parent.parent
                else:
                    data_dir = Path(cfg.dirs[ds])
                data = get_shapenet(
                    cfg,
                    "test",
                    data_dir,
                    load_pointcloud_mode,
                    load_points and cfg.files.points.test,
                    load_mesh,
                    load_cam,
                    load_normals,
                )
            if test_data is None:
                test_data = data
            else:
                test_data += data

    for data in [train_data, val_data, test_data]:
        if data is not None and hasattr(data, "fields"):
            for name, field in data.fields.items():
                if cfg[name].cache is not None:
                    field.cachable = cfg[name].cache

    data = dict()
    if "train" in splits:
        data["train"] = train_data
    if "val" in splits:
        data["val"] = val_data
    if "test" in splits:
        data["test"] = test_data

    print_dataset_info(data, splits)

    return data


@rank_zero_only
def print_dataset_info(data, splits):
    print()
    for split in splits:
        if isinstance(data[split], (tuple, list)):
            print(f"{split.upper()} datasets:")
            for d in data[split]:
                print(d)
                print("_" * len(d.__repr__()))
        else:
            print(f"{split.upper()} dataset: {data[split]}")
