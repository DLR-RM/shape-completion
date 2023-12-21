from typing import Dict, List, Tuple, Union
from pathlib import Path
from urllib.request import urlretrieve
import tarfile
import gzip
import shutil
import tempfile

from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from easy_o3d.utils import DownsampleTypes, OutlierTypes

from utils import setup_logger
from .bop import BOP
from .completion3d import Completion3D
from .mesh import MeshDataset
from .shapenet import ShapeNet
from .transforms import *
from .ycb import YCB
from .modelnet import ModelNet

logger = setup_logger(__name__)


def get_transforms(cfg: DictConfig,
                   split: str,
                   ds: str) -> Dict[str, List[Transform]]:
    logger.info(f"{split.upper()} transforms:")
    transforms = {"inputs": list(),
                  "points": list(),
                  "pointcloud": list(),
                  "data": list()}

    val_aug = split == "val" and not cfg.val.no_aug
    test_aug = split == "test" and not cfg.test.no_aug
    apply_aug = split == "train" or val_aug or test_aug

    if cfg.mesh.load or cfg.norm.reference == "mesh" or (cfg.vis.mesh and cfg.vis.show):
        transforms["mesh"] = [NormalizeMesh()]

    if cfg.inputs.type == "partial":
        transforms["inputs"].append(AxesCutPointcloud(axes='z',
                                                      cut_ratio=(0.4, 0.6),
                                                      rotate_object=cfg.aug.rotate,
                                                      upper_hemisphere=True))
    elif cfg.inputs.type == "depth_like":
        transforms["inputs"].extend([SubsamplePointcloud(num_samples=50000),
                                     DepthLikePointcloud(
                                         rotate_object="" if cfg.aug.rotate == "cam" else cfg.aug.rotate,
                                         upper_hemisphere=cfg.aug.upper_hemisphere,
                                         rot_from_inputs=cfg.aug.rotate == "cam",
                                         # cam_from_inputs=cfg.inputs.frame == "cam"
                                     )])
    elif cfg.inputs.type in ["render_depth", "render_pcd"]:
        if cfg.inputs.type == "render_depth":
            transforms["data"].extend([RenderDepthMap(width=640,
                                                      height=480,
                                                      offscreen=not cfg.vis.show,
                                                      method="pyrender",
                                                      remove_mesh=not cfg.vis.mesh,
                                                      render_normals=cfg.aug.remove_angle or cfg.aug.edge_noise,
                                                      verbose=cfg.log.verbose),
                                       DepthToPointcloud()])
        elif cfg.inputs.type == "render_pcd":
            transforms["data"].append(RenderPointcloud())
    if cfg.inputs.type != "image":
        if cfg.aug.downsample:
            transforms["inputs"].append(ProcessPointcloud(downsample=DownsampleTypes.VOXEL,
                                                          downsample_factor=cfg.aug.downsample))
        if cfg.aug.cut_plane:
            transforms["inputs"].append(RandomApply(AxesCutPointcloud(axes='y', cut_ratio=cfg.aug.cut_plane)))
        if cfg.aug.cut_sphere:
            transforms["inputs"].extend([SphereCutPointcloud(radius=0.2, num_spheres=2),
                                         SphereCutPointcloud(radius=0.05, num_spheres=5),
                                         SphereCutPointcloud(radius=0.02, num_spheres=10)])

    if cfg.pointcloud.num_points:
        num_samples = cfg.pointcloud.num_points if split == "train" else int(1e5)
        transforms["pointcloud"].append(SubsamplePointcloud(num_samples=num_samples))

    # INPUTS
    if (cfg.aug.remove_angle or cfg.aug.edge_noise) and apply_aug:
        if cfg.inputs.type in ["depth_like", "rgbd", "depth_bp"]:
            vis_pcd = cfg.vis.show and cfg.vis.pointcloud
            remove_pointcloud = not vis_pcd and not cfg.norm.reference == "pointcloud" and not cfg.data.sdf_from_occ
            transforms["data"].append(InputNormalsFromPointcloud(cache_kdtree=cfg.pointcloud.cache and cfg.inputs.cache,
                                                                 cache_normals=cfg.inputs.cache,
                                                                 remove_pointcloud=remove_pointcloud,
                                                                 verbose=cfg.log.verbose))
        transforms["data"].append(NormalsCameraCosineSimilarity(remove_normals=not cfg.vis.show))
    if cfg.inputs.type in ["image", "rgbd"]:
        transforms["data"].append(ImageToTensor())
    if cfg.aug.remove_angle and apply_aug:
        cos_sim_threshold = None if isinstance(cfg.aug.remove_angle, bool) else cfg.aug.remove_angle
        transforms["data"].append(AngleOfIncidenceRemoval(cos_sim_threshold=cos_sim_threshold,
                                                          remove_cos_sim=not cfg.aug.edge_noise))
    if cfg.aug.edge_noise and apply_aug:
        transforms["data"].append(EdgeNoise(stddev=cfg.aug.edge_noise,
                                            remove_cos_sim=True))
    if cfg.inputs.num_points:
        transforms["data"].append(SubsamplePointcloud(num_samples=cfg.inputs.num_points))
    if cfg.aug.noise and apply_aug:
        transforms["data"].append(PointcloudNoise(stddev=cfg.aug.noise,
                                                  clip=cfg.aug.clip_noise))

    # POINTS
    if cfg.points.from_mesh:
        sigmas = [0.001, 0.0015, 0.0025, 0.01, 0.015, 0.025, 0.1, 0.15, 0.25]
        num_samples = cfg.train.num_query_points if split == "train" else cfg.val.num_query_points
        transforms["data"].append(PointsFromMesh(padding=cfg.norm.padding,
                                                 sigmas=sigmas if split == "train" else None,
                                                 num_samples=num_samples))
    if cfg.points.load_uncertain:
        transforms["data"].append(LoadUncertain())
    if cfg.points.subsample and split == "train" and 10 * cfg.train.num_query_points < int(1e5):
        transforms["data"].append(SubsamplePoints(num_samples=10 * cfg.train.num_query_points))
    if cfg.data.unscale:
        transforms["data"].append(Scale(from_inputs=True,
                                        multiplier=cfg.data.scale_multiplier))

    # SCALE
    if cfg.aug.scale and apply_aug:
        if isinstance(cfg.aug.scale, (int, float)):
            transforms["data"].append(Scale(amount=cfg.aug.scale,
                                            random=True))
        elif isinstance(cfg.aug.scale, ListConfig):
            transforms["data"].append(Scale(amount=list(cfg.aug.scale),
                                            random=True))
        elif isinstance(cfg.aug.scale, str):
            transforms["data"].append(Scale(axes=cfg.aug.scale,
                                            amount=0.2,
                                            random=True))

    # ROTATE
    if cfg.data.unrotate:
        transforms["data"].append(Rotate(from_inputs=True))
    if cfg.inputs.frame:
        transforms["data"].append(Rotate(to_world_frame=cfg.inputs.frame == "world",
                                         to_cam_frame=cfg.inputs.frame == "cam"))
    elif cfg.aug.rotate and cfg.inputs.type != "depth_like" and apply_aug:
        transforms["data"].append(Rotate(axes=cfg.aug.rotate,
                                         angles=(90.0, 180.0, 270.0) if cfg.aug.principal_rotations else None,
                                         upper_hemisphere=cfg.aug.upper_hemisphere,
                                         angle_from_index=split == "test" or cfg.aug.angle_from_index,
                                         choose_random=cfg.aug.principal_rotations))

    # NORMALIZE
    if cfg.norm.center or cfg.norm.scale or cfg.norm.to_front or cfg.norm.offset or cfg.norm.true_height:
        transforms["data"].append(Normalize(center=cfg.norm.center,
                                            to_front=cfg.norm.to_front,
                                            scale=cfg.norm.scale,
                                            offset=cfg.norm.offset,
                                            true_height=cfg.norm.true_height,
                                            reference=cfg.norm.reference,
                                            method=cfg.norm.method,
                                            padding=cfg.norm.padding))

    ####################################################################################################################

    # INPUTS
    if cfg.inputs.voxelize:
        transforms["data"].append(VoxelizeInputs(resolution=cfg.inputs.voxelize,
                                                 padding=cfg.norm.padding,
                                                 method="kdtree"))
    elif cfg.inputs.bps.apply:
        transforms["data"].append(BPS(num_points=cfg.inputs.bps.num_points,
                                      resolution=cfg.inputs.bps.resolution,
                                      padding=cfg.norm.padding,
                                      method=cfg.inputs.bps.method,
                                      feature=cfg.inputs.bps.feature,
                                      basis=cfg.inputs.bps.basis))

    # POINTS
    if cfg.points.crop:
        transforms["data"].append(CropPoints(padding=cfg.norm.padding,
                                             # cache=(cfg.points.cache and cfg.inputs.cache),
                                             verbose=cfg.log.verbose))
    if cfg.points.subsample:
        num_query_points = cfg.train.num_query_points if split == "train" else cfg.val.num_query_points if split == "val" else cfg.test.num_query_points
        transforms["data"].append(SubsamplePoints(num_samples=num_query_points,
                                                  in_out_ratio=cfg.points.in_out_ratio if split == "train" else None))
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
        transforms["data"].append(SegmentationFromPartNet(apply_to=apply_to,
                                                          num_classes=cfg.seg.num_classes))
    if cfg.points.voxelize:
        transforms["data"].append(VoxelizePoints(resolution=cfg.points.voxelize,
                                                 padding=cfg.norm.padding,
                                                 method="kdtree"))
    if cfg.data.sdf_from_occ:
        remove_pointcloud = not (cfg.vis.show and cfg.vis.pointcloud)
        transforms["data"].append(SdfFromOcc(tsdf="tsdf" in str(cfg.train.loss),
                                             remove_pointcloud=remove_pointcloud))

    # MISC
    bbox = cfg.inputs.bbox or cfg.points.bbox or cfg.pointcloud.bbox or cfg.mesh.bbox
    if bbox:
        bbox_ref = "inputs" if cfg.inputs.bbox else "points" if cfg.points.bbox else "pointcloud" if cfg.pointcloud.bbox else "mesh.vertices"
        remove_ref = not (
                cfg.vis.show and cfg.vis.pointcloud) and not cfg.norm.reference == "pointcloud" and not (
                cfg.aug.remove_angle or cfg.aug.edge_noise)
        transforms["data"].append(BoundingBox(reference=bbox_ref,
                                              remove_reference=remove_ref,
                                              cache=cfg.pointcloud.cache,
                                              verbose=cfg.log.verbose))
    if bbox and cfg.aug.bbox_jitter and apply_aug:
        transforms["data"].append(BoundingBoxJitter(max_jitter=cfg.aug.bbox_jitter))

    """
    if cfg.inputs.nerf or cfg.points.nerf or cfg.pointcloud.nerf:
        apply_to = list()
        if cfg.inputs.nerf:
            apply_to.append("inputs")
        if cfg.points.nerf:
            apply_to.append("points")
        if cfg.pointcloud.nerf:
            apply_to.append("pointcloud")
        transforms["data"].append(NeRFEncoding(apply_to=apply_to,
                                               replace_key=not cfg.vis.show,
                                               padding=cfg.norm.padding))
    """

    min_num_points = {"pointcloud": cfg.pointcloud.min_num_points}
    max_num_points = dict()
    if not cfg.inputs.voxelize:
        min_num_points["inputs"] = cfg.inputs.min_num_points
        max_num_points["inputs"] = cfg.inputs.max_num_points
    if not cfg.points.voxelize:
        min_num_points["points"] = cfg.points.min_num_points
    if min_num_points or max_num_points:
        transforms["data"].append(MinMaxNumPoints(min_num_points=min_num_points,
                                                  max_num_points=max_num_points))

    if cfg.load.keys_to_keep:
        keys_to_keep = KeysToKeep(keys=cfg.load.keys_to_keep)
    else:
        keys_to_keep = KeysToKeep(preset="train")
    if cfg.vis.show:
        keys_to_keep = KeysToKeep(preset="visualize")
        if cfg.load.keys_to_keep:
            keys_to_keep.keys = list(set(keys_to_keep.keys + cfg.load.keys_to_keep))
    transforms["data"].extend([keys_to_keep, CheckDtype()])

    # DEBUG
    if cfg.val.verify:
        threshold = 0 if cfg.implicit.sdf and cfg.implicit.threshold == 0.5 else cfg.implicit.threshold
        transforms["data"].append(Visualize(show_inputs=cfg.vis.inputs,
                                            show_occupancy=cfg.vis.occupancy,
                                            show_points=cfg.vis.points,
                                            show_frame=cfg.vis.frame,
                                            show_pointcloud=cfg.vis.pointcloud,
                                            show_mesh=cfg.vis.mesh,
                                            show_box=cfg.vis.box,
                                            show_bbox=cfg.vis.bbox,
                                            show_cam=cfg.vis.cam,
                                            threshold=threshold,
                                            sdf=cfg.implicit.sdf,
                                            padding=cfg.norm.padding,
                                            n_calls=1 if cfg.load.num_workers else 3,
                                            cam_forward=(0, 0, -1) if cfg.inputs.frame == "cam" else (0, 0, 1),
                                            cam_up=(0, -1, 0) if cfg.inputs.frame == "cam" else (0, 1, 0)))
    print()
    return transforms


def get_ycb(cfg: DictConfig,
            split: str,
            load_pointcloud: bool,
            load_voxels: bool) -> YCB:
    if split in ["train", "val", "test"]:
        return YCB(data_dir=cfg.dirs.ycb,
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
                   transforms=get_transforms(cfg, split, ds="ycb"))
    else:
        print("YCB transforms:")
        transforms = {"inputs": list(),
                      "data": list()}
        transforms["inputs"].append(CropPointcloud())
        if cfg.aug.remove_outlier:
            transforms["inputs"].append(ProcessPointcloud(remove_outlier=OutlierTypes.STATISTICAL,
                                                          outlier_std_ratio=1))
        if cfg.num_input_points:
            transforms["inputs"].append(SubsamplePointcloud(num_samples=cfg.num_input_points))
        if cfg.inputs.frame:
            transforms["data"].append(Rotate(to_world_frame=cfg.inputs.frame == "world",
                                             to_cam_frame=cfg.inputs.frame == "cam"))
        if cfg.inputs.type == "justin":
            transforms["data"].append(ProcessInputs(downsample=DownsampleTypes.VOXEL,
                                                    downsample_factor=cfg.aug.voxel_size,
                                                    remove_outlier=OutlierTypes.STATISTICAL if cfg.aug.remove_outlier else None,
                                                    outlier_std_ratio=2))
        transforms["data"].append(Normalize(center="xyz" if "center" in cfg.norm.normalize else "",
                                            scale="scale" in cfg.norm.normalize,
                                            reference=cfg.norm.reference))
        if cfg.val.verify:
            transforms["data"].append(Visualize(n_calls=1 if cfg.val.batch_size > 1 else 3))
        print()
        return YCB(cfg.dirs.ycb,
                   load_pointcloud=load_pointcloud,
                   load_meshes=cfg.norm.reference == "automatica",
                   cam_id_range=(2, 3) if cfg.inputs.type == "justin" else (1, 5),
                   merge_angles=7 if cfg.inputs.type == "justin" else 0,
                   stride=2 if cfg.inputs.type == "justin" else 1,
                   filter_discontinuities=7 if cfg.inputs.type == "justin" else None,
                   padding=cfg.norm.padding,
                   transforms=transforms)


def get_bop(cfg: DictConfig, ds: str):
    name = ds.split('_')[1]
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
        keys_to_keep = KeysToKeep(preset="train")
    if cfg.vis.show:
        keys_to_keep = KeysToKeep(preset="visualize")

    transforms = {
        "points": [ScalePoints(0.001)],
        "pointcloud": [ScalePoints(0.001)],
        "mesh": [ScaleMesh(0.001)],
        "data": [  # Normalize(reference="mesh"),
            # PointsFromMesh(cache=True),
            # PointcloudFromMesh(cache=True),
            # Normalize(reference="mesh", reverse=True),
            ApplyPose(exclude=("inputs", "inputs.normals")),
            Rotate(to_world_frame=True),
            Rotate(axes='x', angles=(-90,)),
            Normalize(center=cfg.norm.center,
                      to_front=cfg.norm.to_front,
                      scale=cfg.norm.scale,
                      offset=cfg.norm.offset,
                      true_height=cfg.norm.true_height,
                      reference=cfg.norm.reference,
                      method=cfg.norm.method,
                      padding=cfg.norm.padding),
            LoadUncertain() if cfg.points.load_uncertain else ReturnTransform(),
            CropPoints(padding=cfg.norm.padding)]}
    if cfg.inputs.voxelize:
        transforms["data"].append(VoxelizeInputs(resolution=cfg.inputs.voxelize,
                                                 padding=cfg.norm.padding,
                                                 method="kdtree"))
    transforms["data"].extend([keys_to_keep, CheckDtype()])

    return BOP(name=name,
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
               cache=cfg.data.cache,
               transforms=transforms,
               verbose=cfg.log.verbose)


def get_file(name_or_url: Union[str, Path]) -> Path:
    if name_or_url == "bunny":
        url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
    elif name_or_url == "armadillo":
        url = "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
    elif name_or_url == "torus":
        raise NotImplementedError("Torus is not available for download.")
    elif name_or_url == "sphere":
        raise NotImplementedError("Sphere is not available for download.")
    else:
        url = name_or_url

    file = Path(tempfile.gettempdir()) / Path(url).name

    if name_or_url == "bunny":
        extract_file = file.parent / file.name.split('.')[0] / "reconstruction" / "bun_zipper.ply"
        out_file = file.parent / "bunny" / "mesh.ply"
        out_file.parent.mkdir(parents=True, exist_ok=True)

    elif name_or_url == "armadillo":
        extract_file = file.parent / "Armadillo.ply"
        out_file = file.parent / "armadillo" / "mesh.ply"
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise NotImplementedError

    if not out_file.exists():
        if url is not None:
            logger.debug(f"Downloading {name_or_url} from {url} to {file}")
            urlretrieve(str(url), file)
            if ".tar" in file.suffixes:
                with tarfile.open(file) as tar:
                    tar.extractall(file.parent)
            elif file.suffix == ".gz":
                with gzip.open(file) as f_in:
                    with open(file.parent / file.stem, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            shutil.move(extract_file, out_file)
            logger.debug(f"Extracted file to {out_file}")
    return out_file


def get_dataset(cfg: DictConfig,
                load_points: bool = True,
                load_pointcloud: bool = False,
                load_mesh: bool = False,
                load_voxels: bool = False,
                load_normals: bool = False,
                splits: Tuple[str, ...] = ("train", "val")) -> Dict[str, Dataset]:
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

    """
    if "completion3d" in train_ds + val_ds + test_ds:
        if "completion3d" in train_ds:
            print("TRAIN transforms:")
        elif "completion3d" in val_ds:
            print("VAL transforms:")
        elif "completion3d" in test_ds:
            print("TEST transforms:")
        completion3d_transforms = {"points": [SubsamplePoints(cfg.points.num_points)],
                                   "data": [Normalize(reference="pointcloud")]}
        if cfg.val.verify:
            completion3d_transforms["data"].append(Visualize(show_pointcloud=True,
                                                             n_calls=1 if cfg.load.num_workers else 3))
        print()
    """
    completion3d_transforms = None

    load_points = load_points or (cfg.vis.show and (cfg.vis.occupancy or cfg.vis.points))
    load_voxels = load_voxels or (cfg.vis.show and cfg.vis.voxels)
    load_normals = load_normals or cfg.aug.remove_angle or cfg.aug.edge_noise
    load_pointcloud = load_pointcloud or cfg.norm.reference == "pointcloud" or cfg.pointcloud.bbox or cfg.pointcloud.normals or cfg.data.sdf_from_occ or cfg.aug.remove_angle or cfg.aug.edge_noise or (cfg.vis.show and cfg.vis.pointcloud)
    load_pointcloud = "min_max_only" if cfg.norm.true_height and not load_pointcloud else load_pointcloud
    load_mesh = load_mesh or cfg.mesh.load or cfg.norm.reference == "mesh" or cfg.mesh.bbox or (cfg.vis.show and cfg.vis.mesh)
    partnet = Path(cfg.dirs.partnet) if cfg.seg.num_classes or cfg.get("partnet", False) else None
    cfg.mesh.load = load_mesh
    if "train" in splits:
        for ds in train_ds:
            if "shapenet" in ds:
                data = ShapeNet(split="train",
                                data_dir=Path(cfg.dirs[ds]),
                                partnet_dir=partnet,
                                inputs_dir=Path(cfg.dirs[cfg.inputs.data_dir]) if cfg.inputs.data_dir else None,
                                points_dir_name=cfg.points.data_dir,
                                pointcloud_dir_name=cfg.pointcloud.data_dir,
                                mesh_dir_name=cfg.mesh.data_dir,
                                image_dir_name=cfg.inputs.image_dir,
                                cam_dir_name=cfg.inputs.cam_dir,
                                categories=cfg.data.categories,
                                num_shards=cfg.data.num_shards.train,
                                files_per_shard=cfg.data.num_files.train,
                                unscale=cfg.data.unscale,
                                undistort=cfg.data.undistort,
                                unrotate=cfg.data.unrotate,
                                sdf=cfg.implicit.sdf,
                                tsdf="tsdf" in str(cfg.train.loss),
                                split_file=cfg.files.split.train,
                                pointcloud_file=cfg.files.pointcloud,
                                points_file=cfg.files.points.train,
                                voxels_file=cfg.files.voxels,
                                normals_file=cfg.files.normals,
                                mesh_file=cfg.files.mesh,
                                load_random_inputs=cfg.inputs.load_random,
                                load_pointcloud=load_pointcloud,
                                load_normals=load_normals,
                                load_points=load_points,
                                load_all_points=cfg.points.load_all,
                                load_random_points=cfg.points.load_random,
                                load_surface_points=cfg.points.load_surface or cfg.train.load_surface_points,
                                load_meshes=load_mesh,
                                load_cam=cfg.inputs.type in ["image", "rgbd"],
                                input_type=cfg.inputs.type,
                                weighted=cfg.load.weighted,
                                padding=cfg.norm.padding,
                                cls_weight=cfg.cls.weight,
                                seg_weight=cfg.seg.weight,
                                cache=cfg.data.cache,
                                cache_inputs=cfg.inputs.cache,
                                cache_points=cfg.points.cache,
                                cache_mesh=cfg.mesh.cache,
                                cache_pointcloud=cfg.pointcloud.cache,
                                from_hdf5=cfg.load.hdf5,
                                transforms=get_transforms(cfg, split="train", ds=ds),
                                verbose=cfg.log.verbose)
            elif ds == "completion3d":
                data = Completion3D(data_dir=Path(cfg.dirs[ds]),
                                    shapenet_dir=cfg.dirs.onet,
                                    categories=cfg.data.categories,
                                    split="train",
                                    transforms=completion3d_transforms,
                                    load_pointcloud=load_pointcloud,
                                    load_points=load_points)
            elif ds == "ycb":
                data = get_ycb(cfg=cfg,
                               split="train",
                               load_pointcloud=load_pointcloud,
                               load_voxels=load_voxels)
            elif ds == "modelnet40":
                data = ModelNet(split="train",
                                data_dir=Path(cfg.dirs[ds]),
                                cache=cfg.data.cache,
                                cache_inputs=cfg.inputs.cache,
                                cache_mesh=cfg.mesh.cache,
                                cache_pointcloud=cfg.pointcloud.cache,
                                transforms=get_transforms(cfg, split="train", ds=ds),
                                verbose=cfg.log.verbose)
            else:
                objects = cfg.data.objects
                if ds in ["bunny", "armadillo", "torus", "sphere"]:
                    objects = (get_file(ds),)
                    data_dir = objects[0].parent.parent
                else:
                    data_dir = Path(cfg.dirs[ds])
                data = MeshDataset(split="train",
                                   data_dir=data_dir,
                                   inputs_dir=Path(cfg.dirs[cfg.inputs.data_dir]) if cfg.inputs.data_dir else None,
                                   points_dir=cfg.points.data_dir,
                                   pointcloud_dir=cfg.pointcloud.data_dir,
                                   mesh_dir=cfg.mesh.data_dir,
                                   image_dir=cfg.inputs.image_dir,
                                   cam_dir=cfg.inputs.cam_dir,
                                   objects=objects,
                                   id_length=3 if ds == "automatica" else None,
                                   num_shards=cfg.data.num_shards.train,
                                   files_per_shard=cfg.data.num_files.train,
                                   unscale=cfg.data.unscale,
                                   undistort=cfg.data.undistort,
                                   unrotate=cfg.data.unrotate,
                                   sdf=cfg.implicit.sdf,
                                   tsdf="tsdf" in str(cfg.train.loss),
                                   pointcloud_file=cfg.files.pointcloud,
                                   points_file=cfg.files.points.train,
                                   voxels_file=cfg.files.voxels,
                                   normals_file=cfg.files.normals,
                                   mesh_file=cfg.files.mesh,
                                   load_random_inputs=cfg.inputs.load_random,
                                   load_pointcloud=load_pointcloud,
                                   load_normals=load_normals,
                                   load_points=load_points,
                                   load_all_points=cfg.points.load_all,
                                   load_random_points=cfg.points.load_random,
                                   load_surface_points=cfg.points.load_surface or cfg.train.load_surface_points,
                                   load_meshes=load_mesh,
                                   load_cam=cfg.inputs.type in ["image", "rgbd"],
                                   input_type=cfg.inputs.type,
                                   padding=cfg.norm.padding,
                                   cache=cfg.data.cache,
                                   cache_inputs=cfg.inputs.cache,
                                   cache_points=cfg.points.cache,
                                   cache_mesh=cfg.mesh.cache,
                                   cache_pointcloud=cfg.pointcloud.cache,
                                   from_hdf5=cfg.load.hdf5,
                                   transforms=get_transforms(cfg, split="train", ds=ds),
                                   verbose=cfg.log.verbose)
            if train_data is None:
                train_data = data
            else:
                train_data += data
    if "val" in splits:
        for ds in val_ds:
            if "shapenet" in ds:
                data = ShapeNet(split="val",
                                data_dir=Path(cfg.dirs[ds]),
                                partnet_dir=partnet,
                                inputs_dir=Path(cfg.dirs[cfg.inputs.data_dir]) if cfg.inputs.data_dir else None,
                                points_dir_name=cfg.points.data_dir,
                                pointcloud_dir_name=cfg.pointcloud.data_dir,
                                mesh_dir_name=cfg.mesh.data_dir,
                                image_dir_name=cfg.inputs.image_dir,
                                cam_dir_name=cfg.inputs.cam_dir,
                                categories=cfg.data.categories,
                                num_shards=cfg.data.num_shards.val,
                                files_per_shard=cfg.data.num_files.val,
                                unscale=cfg.data.unscale,
                                undistort=cfg.data.undistort,
                                unrotate=cfg.data.unrotate,
                                sdf=cfg.implicit.sdf,
                                tsdf="tsdf" in str(cfg.train.loss),
                                split_file=cfg.files.split.val,
                                pointcloud_file=cfg.files.pointcloud,
                                points_file=cfg.files.points.val,
                                voxels_file=cfg.files.voxels,
                                normals_file=cfg.files.normals,
                                mesh_file=cfg.files.mesh,
                                load_random_inputs=cfg.inputs.load_random,
                                load_pointcloud=load_pointcloud,
                                load_normals=load_normals,
                                load_points=load_points,
                                load_all_points=cfg.points.load_all,
                                load_random_points=False,
                                load_surface_points=cfg.points.load_surface or cfg.val.load_surface_points,
                                load_meshes=load_mesh,
                                load_cam=cfg.inputs.type in ["image", "rgbd"],
                                input_type=cfg.inputs.type,
                                weighted=cfg.load.weighted,
                                precision=cfg.inputs.precision,
                                padding=cfg.norm.padding,
                                cls_weight=cfg.cls.weight,
                                seg_weight=cfg.seg.weight,
                                cache=cfg.data.cache,
                                cache_inputs=cfg.inputs.cache,
                                cache_points=cfg.points.cache,
                                cache_mesh=cfg.mesh.cache,
                                cache_pointcloud=cfg.pointcloud.cache,
                                from_hdf5=cfg.load.hdf5,
                                transforms=get_transforms(cfg, split="val", ds=ds),
                                verbose=cfg.log.verbose)
            elif ds == "completion3d":
                data = Completion3D(data_dir=Path(cfg.dirs[ds]),
                                    shapenet_dir=cfg.dirs.onet,
                                    categories=cfg.data.categories,
                                    split="val",
                                    transforms=completion3d_transforms,
                                    load_pointcloud=load_pointcloud,
                                    load_points=load_points)
            elif ds == "ycb":
                data = get_ycb(cfg=cfg,
                               split="val",
                               load_pointcloud=load_pointcloud,
                               load_voxels=load_voxels)
            elif ds == "modelnet40":
                data = ModelNet(split="test",
                                data_dir=Path(cfg.dirs[ds]),
                                cache=cfg.data.cache,
                                cache_inputs=cfg.inputs.cache,
                                cache_mesh=cfg.mesh.cache,
                                cache_pointcloud=cfg.pointcloud.cache,
                                transforms=get_transforms(cfg, split="test", ds=ds),
                                verbose=cfg.log.verbose)
            else:
                objects = cfg.data.objects
                if ds in ["bunny", "armadillo", "torus", "sphere"]:
                    objects = (get_file(ds),)
                    data_dir = objects[0].parent.parent
                else:
                    data_dir = Path(cfg.dirs[ds])
                data = MeshDataset(split="val",
                                   data_dir=data_dir,
                                   index_offset=cfg.data.num_files.train,
                                   inputs_dir=Path(cfg.dirs[cfg.inputs.data_dir]) if cfg.inputs.data_dir else None,
                                   points_dir=cfg.points.data_dir,
                                   pointcloud_dir=cfg.pointcloud.data_dir,
                                   mesh_dir=cfg.mesh.data_dir,
                                   image_dir=cfg.inputs.image_dir,
                                   cam_dir=cfg.inputs.cam_dir,
                                   objects=objects,
                                   id_length=3 if ds == "automatica" else None,
                                   num_shards=cfg.data.num_shards.val,
                                   files_per_shard=cfg.data.num_files.val,
                                   unscale=cfg.data.unscale,
                                   undistort=cfg.data.undistort,
                                   unrotate=cfg.data.unrotate,
                                   sdf=cfg.implicit.sdf,
                                   tsdf="tsdf" in str(cfg.train.loss),
                                   pointcloud_file=cfg.files.pointcloud,
                                   points_file=cfg.files.points.val,
                                   voxels_file=cfg.files.voxels,
                                   normals_file=cfg.files.normals,
                                   mesh_file=cfg.files.mesh,
                                   load_random_inputs=cfg.inputs.load_random,
                                   load_pointcloud=load_pointcloud,
                                   load_normals=load_normals,
                                   load_points=load_points,
                                   load_all_points=cfg.points.load_all,
                                   load_random_points=False,
                                   load_surface_points=cfg.points.load_surface or cfg.val.load_surface_points,
                                   load_meshes=load_mesh,
                                   load_cam=cfg.inputs.type in ["image", "rgbd"],
                                   input_type=cfg.inputs.type,
                                   precision=cfg.inputs.precision,
                                   padding=cfg.norm.padding,
                                   cache=cfg.data.cache,
                                   cache_inputs=cfg.inputs.cache,
                                   cache_points=cfg.points.cache,
                                   cache_mesh=cfg.mesh.cache,
                                   cache_pointcloud=cfg.pointcloud.cache,
                                   from_hdf5=cfg.load.hdf5,
                                   transforms=get_transforms(cfg, split="val", ds=ds),
                                   verbose=cfg.log.verbose)
            if val_data is None:
                val_data = data
            else:
                val_data += data
    if "test" in splits or (cfg.test.run and cfg.test.split == "test"):
        for ds in test_ds:
            if "shapenet" in ds:
                data = ShapeNet(split="test",
                                data_dir=Path(cfg.dirs[ds]),
                                partnet_dir=partnet,
                                inputs_dir=Path(cfg.dirs[cfg.inputs.data_dir]) if cfg.inputs.data_dir else None,
                                points_dir_name=cfg.points.data_dir,
                                pointcloud_dir_name=cfg.pointcloud.data_dir,
                                mesh_dir_name=cfg.mesh.data_dir,
                                image_dir_name=cfg.inputs.image_dir,
                                cam_dir_name=cfg.inputs.cam_dir,
                                categories=cfg.data.categories,
                                num_shards=cfg.data.num_shards.test,
                                files_per_shard=cfg.data.num_files.test,
                                unscale=cfg.data.unscale,
                                undistort=cfg.data.undistort,
                                unrotate=cfg.data.unrotate,
                                sdf=cfg.implicit.sdf,
                                tsdf="tsdf" in str(cfg.train.loss),
                                split_file=cfg.files.split.test,
                                pointcloud_file=cfg.files.pointcloud,
                                points_file=cfg.files.points.test,
                                voxels_file=cfg.files.voxels,
                                normals_file=cfg.files.normals,
                                mesh_file=cfg.files.mesh,
                                load_random_inputs=False,
                                load_pointcloud=load_pointcloud,
                                load_normals=load_normals,
                                load_points=load_points,
                                load_all_points=cfg.points.load_all,
                                load_random_points=False,
                                load_surface_points=cfg.points.load_surface or cfg.test.load_surface_points,
                                load_meshes=load_mesh,
                                load_cam=cfg.inputs.type in ["image", "rgbd"],
                                input_type=cfg.inputs.type,
                                weighted=cfg.load.weighted,
                                padding=cfg.norm.padding,
                                cls_weight=cfg.cls.weight,
                                seg_weight=cfg.seg.weight,
                                precision=cfg.inputs.precision,
                                cache=cfg.data.cache,
                                cache_inputs=cfg.inputs.cache,
                                cache_points=cfg.points.cache,
                                cache_mesh=cfg.mesh.cache,
                                cache_pointcloud=cfg.pointcloud.cache,
                                from_hdf5=cfg.load.hdf5,
                                transforms=get_transforms(cfg, split="test", ds=ds),
                                verbose=cfg.log.verbose)
            elif ds == "completion3d":
                data = Completion3D(data_dir=Path(cfg.dirs[ds]),
                                    shapenet_dir=cfg.dirs.onet,
                                    categories=cfg.data.categories,
                                    split="val",
                                    transforms=completion3d_transforms,
                                    load_pointcloud=load_pointcloud,
                                    load_points=load_points)
            elif ds == "ycb":
                data = get_ycb(cfg,
                               split="test",
                               load_pointcloud=load_pointcloud,
                               load_voxels=load_voxels)
            elif "bop" in ds:
                data = get_bop(cfg, ds)
            elif ds == "modelnet40":
                data = ModelNet(split="test",
                                data_dir=Path(cfg.dirs[ds]),
                                cache=cfg.data.cache,
                                cache_inputs=cfg.inputs.cache,
                                cache_mesh=cfg.mesh.cache,
                                cache_pointcloud=cfg.pointcloud.cache,
                                transforms=get_transforms(cfg, split="test", ds=ds),
                                verbose=cfg.log.verbose)
            else:
                objects = cfg.data.objects
                if ds in ["bunny", "armadillo", "torus", "sphere"]:
                    objects = (get_file(ds),)
                    data_dir = objects[0].parent.parent
                else:
                    data_dir = Path(cfg.dirs[ds])
                data = MeshDataset(split="test",
                                   data_dir=data_dir,
                                   index_offset=cfg.data.num_files.train + cfg.data.num_files.val,
                                   inputs_dir=Path(cfg.dirs[cfg.inputs.data_dir]) if cfg.inputs.data_dir else None,
                                   points_dir=cfg.points.data_dir,
                                   pointcloud_dir=cfg.pointcloud.data_dir,
                                   mesh_dir=cfg.mesh.data_dir,
                                   image_dir=cfg.inputs.image_dir,
                                   cam_dir=cfg.inputs.cam_dir,
                                   objects=objects,
                                   id_length=3 if ds == "automatica" else None,
                                   num_shards=cfg.data.num_shards.test,
                                   files_per_shard=cfg.data.num_files.test,
                                   unscale=cfg.data.unscale,
                                   undistort=cfg.data.undistort,
                                   unrotate=cfg.data.unrotate,
                                   sdf=cfg.implicit.sdf,
                                   tsdf="tsdf" in str(cfg.train.loss),
                                   pointcloud_file=cfg.files.pointcloud,
                                   points_file=cfg.files.points.test,
                                   voxels_file=cfg.files.voxels,
                                   normals_file=cfg.files.normals,
                                   mesh_file=cfg.files.mesh,
                                   load_random_inputs=False,
                                   load_pointcloud=load_pointcloud,
                                   load_normals=load_normals,
                                   load_points=load_points,
                                   load_all_points=cfg.points.load_all,
                                   load_random_points=False,
                                   load_surface_points=cfg.points.load_surface or cfg.test.load_surface_points,
                                   load_meshes=load_mesh,
                                   load_cam=cfg.inputs.type in ["image", "rgbd"],
                                   input_type=cfg.inputs.type,
                                   padding=cfg.norm.padding,
                                   precision=cfg.inputs.precision,
                                   cache=cfg.data.cache,
                                   cache_inputs=cfg.inputs.cache,
                                   cache_points=cfg.points.cache,
                                   cache_mesh=cfg.mesh.cache,
                                   cache_pointcloud=cfg.pointcloud.cache,
                                   from_hdf5=cfg.load.hdf5,
                                   transforms=get_transforms(cfg, split="test", ds=ds),
                                   verbose=cfg.log.verbose)
            if test_data is None:
                test_data = data
            else:
                test_data += data

    data = dict()
    if "train" in splits:
        data["train"] = train_data
    if "val" in splits:
        data["val"] = val_data
    if "test" in splits:
        data["test"] = test_data
    return data
