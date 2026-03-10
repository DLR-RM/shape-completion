import os
import shutil
from functools import cache, partial
from pathlib import Path
from pprint import pprint
from random import shuffle
from typing import Any, cast

import hydra
import lightning
import numpy as np
import torch
from cleanfid import fid
from joblib import Parallel, delayed
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from trimesh import Trimesh

from dataset import (
    BOP,
    CheckDtype,
    MinMaxNumPoints,
    NormalizeMesh,
    PointcloudFromMesh,
    SubsamplePointcloud,
    get_dataset,
)
from libs import furthest_point_sample
from models import Model, Shape3D2VecSetVAE, resolve_weights_path
from utils import (
    get_num_workers,
    load_mesh,
    log_optional_dependency_summary,
    normalize_mesh,
    resolve_path,
    resolve_save_dir,
    setup_config,
    setup_logger,
    stdout_redirected,
    suppress_known_optional_dependency_warnings,
    to_tensor,
    tqdm_joblib,
)

from ..src.gen_metrics import DistanceMetrics, cov_mmd, one_nn_accuracy, paired_distances, two_sample_test
from ..src.prd import compute_prd_from_embedding
from ..src.prdc import compute_prdc
from ..src.utils import render_for_fid

logger = setup_logger(__name__)
CloudBatch = np.ndarray | torch.Tensor


def _debug_level_1(msg: str) -> None:
    debug_fn = getattr(logger, "debug_level_1", logger.debug)
    debug_fn(msg)


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, np.floating)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise TypeError(f"{name} must be an int-compatible value, got {type(value).__name__}")


def _to_numpy_cloud(value: Any) -> np.ndarray:
    if isinstance(value, tuple):
        value = value[0]
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _to_tensor_cloud(value: Any, *, device: torch.device) -> torch.Tensor:
    tensor = to_tensor(value, unsqueeze=False, device=device)
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor-convertible value, got {type(tensor).__name__}")
    return tensor


@cache
def get_feature_extractor(cfg: DictConfig) -> Model:
    project, name = cfg.log.project, cfg.log.name
    cfg.log.project = "cvpr_2025_vae"  # "cvpr_2025_depth_cls"
    cfg.log.name = "pcd_long_new"  # "pcd@10k_dist_focal"
    path = resolve_weights_path(cfg, "model_best.pt")
    if path is None:
        raise FileNotFoundError("Could not resolve model weights path for Shape3D2VecSet feature extractor.")
    cfg.log.project = project
    cfg.log.name = name
    weights = torch.load(path, map_location="cpu", weights_only=False)
    # model = Shape3D2VecSetCls(n_classes=57, n_layer=12, activation="new_geglu")
    model = Shape3D2VecSetVAE(n_layer=12, activation=cast(Any, "new_geglu"), n_latent=32)
    model.load_state_dict(weights)
    return model.eval()


def duplicate_views(save_dir: Path, n_views: int, min_images: int = 2048):
    for view in range(n_views):
        view_dir = save_dir / f"view_{view}"
        images = list(view_dir.glob("*.png"))
        n_images = len(images)

        if n_images < min_images:
            n_copies = int(np.ceil(min_images / n_images))
            for i in range(1, n_copies):
                for img_path in images:
                    new_path = view_dir / f"{img_path.stem}_{i}{img_path.suffix}"
                    shutil.copy2(img_path, new_path)


def render_gen(
    cfg: DictConfig,
    save_dir_gen: Path,
    generated_meshes: list[Path],
    views: str | int,
    n_views: int,
    num_workers: int,
    progress: bool = True,
):
    _render_for_fid = partial(render_for_fid, views=views)
    if cfg.test.overwrite:
        shutil.rmtree(save_dir_gen, ignore_errors=True)
    save_dir_gen.mkdir(parents=True, exist_ok=True)
    for view in range(n_views):
        (save_dir_gen / f"view_{view}").mkdir(exist_ok=True)
    with tqdm_joblib(tqdm(total=len(generated_meshes), desc="Rendering generated meshes", disable=not progress)):
        with Parallel(n_jobs=len(generated_meshes) <= num_workers or num_workers) as p:
            p(delayed(_render_for_fid)(path, save_dir_gen / path.stem) for path in generated_meshes)
    duplicate_views(save_dir_gen, n_views)


def render_ref(
    cfg: DictConfig,
    save_dir_ref: Path,
    reference_meshes: list[Path | tuple[Path, Trimesh]],
    views: str | int,
    n_views: int,
    num_workers: int,
    progress: bool = True,
):
    _render_for_fid = partial(render_for_fid, views=views)
    if cfg.test.overwrite:
        shutil.rmtree(save_dir_ref, ignore_errors=True)
    save_dir_ref.mkdir(parents=True, exist_ok=True)
    for view in range(n_views):
        (save_dir_ref / f"view_{view}").mkdir(exist_ok=True)
    with tqdm_joblib(tqdm(total=len(reference_meshes), desc="Rendering reference meshes", disable=not progress)):
        with Parallel(n_jobs=len(reference_meshes) <= num_workers or num_workers) as p:
            if cfg.load.hdf5 or cfg.get("load_mesh", False):
                p(
                    delayed(_render_for_fid)(mesh, save_dir_ref / path.parent.stem)
                    for entry in reference_meshes
                    if isinstance(entry, tuple)
                    for path, mesh in [entry]
                )
            else:
                p(
                    delayed(_render_for_fid)(
                        entry if isinstance(entry, Path) else entry[0],
                        save_dir_ref / (entry if isinstance(entry, Path) else entry[0]).parent.stem,
                    )
                    for entry in reference_meshes
                )
    duplicate_views(save_dir_ref, n_views)


def update_metrics_file(save_file: Path, results: dict[str, float]):
    """Update metrics file with new results.

    Args:
        save_file: Path to metrics file
        results: Dict of new metrics to add
    """
    # Read existing results before writing
    existing_metrics = {}
    if save_file.exists():
        with open(save_file) as existing_f:
            for line in existing_f:
                key, value = line.strip().split(": ")
                existing_metrics[key] = float(value)

    with open(save_file, "w") as f:
        for key, value in {**existing_metrics, **results}.items():
            f.write(f"{key}: {value}\n")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg = setup_config(cfg)
    suppress_known_optional_dependency_warnings()
    log_optional_dependency_summary(logger, cfg)

    if not cfg.vis.show:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    split = cfg.test.split
    n_points = _as_int(cfg.pointcloud.num_points or 2048, "pointcloud.num_points")
    if not cfg.pointcloud.num_points:
        cfg.pointcloud.num_points = 100_000
    cfg.pointcloud[split].num_points = _as_int(cfg.pointcloud.num_points, "pointcloud.num_points")
    metrics = ["chamfer", "fid"] if cfg.test.metrics is None else list(cfg.test.metrics)
    n_batch = _as_int(cfg.test.batch_size or (512 if "emd" in metrics else 1024), "test.batch_size")
    num_workers = get_num_workers(cfg.load.num_workers)
    stats_only = bool(cfg.get("stats_only", False))
    no_stats = bool(cfg.get("no_stats", False))
    assert not (stats_only and no_stats), "Cannot set both stats_only and no_stats to True"
    stats_exist = partial(fid.test_stats_exists, mode="clean")
    views_cfg = cfg.get("views", "sdfstylegan")
    views: str | int = views_cfg if isinstance(views_cfg, (str, int)) else str(views_cfg)
    n_views: int
    file_suffix = stats_suffix = f"{views}_views"
    if isinstance(views, str):
        file_suffix = stats_suffix = views
        if views == "icosphere":
            n_views = 12
        elif views in ["dodecahedron", "sdfstylegan"]:
            n_views = 20
        elif views in ["3dilg", "3dshape2vecset"]:
            views = "sdfstylegan"
            n_views = 10
            stats_suffix = views
        else:
            raise ValueError(f"Invalid view distribution: {views}")
    else:
        n_views = views

    load_clouds = any(metric.upper() in DistanceMetrics.__members__ for metric in metrics)
    assert not (no_stats and any("prd" in metric for metric in metrics)), "Cannot compute PRD without stats"

    ds = cfg.data[f"{split}_ds"][0]
    category = f"_{cfg.data.categories[0]}_" if cfg.data.categories and len(cfg.data.categories) == 1 else "_"
    mesh_dir = resolve_save_dir(cfg) / "generation" / split / "meshes" / category.strip("_")
    save_file = mesh_dir.parent / f"{ds}{category}{split}_gen_eval_{cfg.implicit.threshold:.2f}_{file_suffix}.txt"

    stats_name_root = f"{ds}_{split}"
    if cfg.data.categories and len(cfg.data.categories) == 1:
        stats_name_root = f"{ds}_{cfg.data.categories[0]}_{split}"
    stats_name = f"{stats_name_root}_{stats_suffix}"
    all_stats_exist = all(stats_exist(f"{stats_name}_view_{i}") for i in range(n_views))
    if any("clip" in metric for metric in metrics):
        all_clip_stats_exist = all(
            stats_exist(f"{stats_name}_view_{i}", model_name="clip_vit_b_32") for i in range(n_views)
        )
        all_stats_exist = all_stats_exist and all_clip_stats_exist
    load_ref_meshes = not all_stats_exist or stats_only or no_stats or cfg.test.overwrite
    if not any("fid" in metric or "kid" in metric or "prd" in metric for metric in metrics):
        load_ref_meshes = False
    stats_dir = resolve_path(cfg.dirs.log) / "fid" / ds / category.strip("_") / split / stats_suffix

    if stats_exist(stats_name) and all_stats_exist and stats_only and not cfg.test.overwrite:
        logger.info(f"Stats for {stats_name} already exist")
        return

    generated_meshes: list[Path] = [path for pattern in ("*.ply", "*.obj") for path in mesh_dir.rglob(pattern)]
    shuffle(generated_meshes)
    if cfg.test.num_instances:
        generated_meshes = generated_meshes[: cfg.test.num_instances]
    n_mesh = len(generated_meshes)
    logger.info(f"Found {n_mesh} generated meshes in {mesh_dir}")
    if n_mesh < n_batch:
        n_batch = n_mesh

    new_shape = (n_mesh // n_batch, n_batch, n_points, 3)
    generated_clouds: list[CloudBatch] = []
    if load_clouds:
        assert n_mesh >= n_batch, f"Number of generated meshes ({n_mesh}) must be at least the batch size ({n_batch})"
        generated_meshes = generated_meshes[: n_mesh - (n_mesh % n_batch)]
        n_mesh = len(generated_meshes)
        logger.info(f"Using {n_mesh} generated meshes for evaluation")

        def load_cloud(path: Path) -> np.ndarray:
            mesh = Trimesh(*load_mesh(path), process=False)
            if mesh.is_empty:
                return np.zeros((n_points, 3))
            if path.with_suffix(".npy").exists():
                trafo = np.load(path.with_suffix(".npy"))
                mesh.apply_transform(trafo)
            # components = mesh.split(only_watertight=False)
            # mesh = max(components, key=lambda c: c.volume)
            mesh = normalize_mesh(mesh)
            pcd = cast(np.ndarray, mesh.sample(_as_int(cfg.pointcloud.num_points, "pointcloud.num_points")))
            if pcd.shape[0] > n_points:
                fps_out = furthest_point_sample(torch.from_numpy(pcd).unsqueeze(0).cuda(), n_points)
                pcd = _to_numpy_cloud(fps_out.squeeze(0) if torch.is_tensor(fps_out) else fps_out)
            return pcd

        with tqdm_joblib(tqdm(total=n_mesh, desc="Loading generated clouds", disable=not cfg.log.progress)):
            with Parallel(n_jobs=n_mesh < num_workers or num_workers) as p:
                generated_clouds = cast(list[CloudBatch], p(delayed(load_cloud)(path) for path in generated_meshes))

        generated_clouds_np = np.concatenate(cast(list[np.ndarray], generated_clouds)).reshape(new_shape)
        generated_clouds = [cloud for cloud in generated_clouds_np]

    reference_clouds: list[CloudBatch] = []
    reference_meshes: list[Path | tuple[Path, Trimesh]] = list()
    if load_clouds or load_ref_meshes:
        _debug_level_1(
            f"Loading {split} reference data because: "
            f"load_clouds={load_clouds}, "
            f"load_ref_meshes={load_ref_meshes}, "
            f"stats_name={stats_name}, "
            f"all_stats_exist={all_stats_exist}, "
            f"stats_only={stats_only}, "
            f"no_stats={no_stats}, "
            f"overwrite={cfg.test.overwrite}"
        )
        with stdout_redirected(enabled=cfg.log.verbose < 2):
            load_meshes_default: bool | str = load_ref_meshes if cfg.load.hdf5 or load_clouds else "path_only"
            load_meshes = cast(bool | str, cfg.get("load_mesh", load_meshes_default))
            dataset = cast(
                Any, get_dataset(cfg, splits=(split,), load_pointcloud=load_clouds, load_mesh=load_meshes)[split]
            )
        dataset.fields = {k: v for k, v in dataset.fields.items() if k in ["pointcloud", "mesh"]}
        dataset.transformations = [
            t
            for t in dataset.transformations
            if isinstance(t, (CheckDtype, MinMaxNumPoints, PointcloudFromMesh, NormalizeMesh, SubsamplePointcloud))
        ]

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0 if isinstance(dataset, BOP) else num_workers,
            pin_memory=load_clouds,
            generator=torch.Generator().manual_seed(cfg.misc.seed),
        )

        for i, item in enumerate(tqdm(loader, desc=f"Loading {split} reference data", disable=not cfg.log.progress)):
            if load_clouds and i < n_mesh:
                pcd = cast(torch.Tensor, item["pointcloud"])
                if pcd.size(1) > n_points:
                    pcd = cast(torch.Tensor, furthest_point_sample(pcd.cuda(non_blocking=True), n_points))
                reference_clouds.append(pcd.squeeze(0).cpu().numpy())
            if load_ref_meshes:
                if cfg.load.hdf5:
                    reference_meshes.append(
                        (
                            Path(item["mesh.path"][0]),
                            Trimesh(item["mesh.vertices"][0].numpy(), item["mesh.triangles"][0].numpy(), process=False),
                        )
                    )
                else:
                    reference_meshes.append(Path(item["mesh.path"][0]))
        if reference_clouds:
            reference_clouds_np = np.concatenate(cast(list[np.ndarray], reference_clouds)).reshape(new_shape)
            reference_clouds = [cloud for cloud in reference_clouds_np]

    results = {}
    for metric in metrics:
        if metric.upper() in DistanceMetrics.__members__:
            if split != "test":
                logger.warning("Evaluation is NOT done on the TEST set!")
            all_clouds = reference_clouds + generated_clouds
            if metric in ["emd", "chamfer", "feat"]:
                if metric == "feat":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    fabric = lightning.Fabric(precision=cfg[split].precision)
                    model = cast(Model, get_feature_extractor(cfg).to(device))
                    sample_posterior = cast(Any, model.sample_posterior)

                    with fabric.autocast(), torch.inference_mode():
                        generated_clouds = [
                            cast(torch.Tensor, sample_posterior(_to_tensor_cloud(pcd, device=device), sample=False))
                            for pcd in generated_clouds
                        ]
                        reference_clouds = [
                            cast(torch.Tensor, sample_posterior(_to_tensor_cloud(pcd, device=device), sample=False))
                            for pcd in reference_clouds
                        ]
                distances = paired_distances(
                    generated_clouds,
                    reference_clouds,
                    metric=DistanceMetrics[metric.upper()],
                    progress=cfg.log.progress,
                    show=cfg.vis.show,
                    verbose=cfg.log.verbose > 1,
                )

                if cfg.vis.show:
                    import open3d as o3d

                    o3d_any = cast(Any, o3d)

                    n_batch = 1 if all_clouds[0].ndim == 2 else all_clouds[0].shape[0]
                    n_ref = len(reference_clouds) * n_batch

                    np.fill_diagonal(distances, np.inf)
                    nearest_neighbors = np.argmin(distances, axis=1)
                    cat_clouds = np.concatenate([_to_numpy_cloud(cloud) for cloud in all_clouds])

                    for i, nn in enumerate(
                        nearest_neighbors[:n_ref]
                    ):  # Iterate over reference clouds and their nearest neighbors
                        cloud1 = cat_clouds[i]
                        cloud2 = cat_clouds[nn]
                        pcd1 = o3d_any.geometry.PointCloud(o3d_any.utility.Vector3dVector(cloud1)).paint_uniform_color(
                            [0, 0, 1]
                        )
                        pcd2 = o3d_any.geometry.PointCloud(o3d_any.utility.Vector3dVector(cloud2))
                        if nn < n_ref:  # Nearest neighbor is a reference cloud
                            pcd2.paint_uniform_color([1, 0, 0])
                        else:  # Nearest neighbor is a generated cloud
                            pcd2.paint_uniform_color([0, 1, 0])
                        o3d_any.visualization.draw_geometries([pcd1, pcd2])

                accuracy = one_nn_accuracy(distances, verbose=cfg.log.verbose > 1)
                logger.info(f"{split.upper()} LOO 1-NN Accuracy ({metric}): {accuracy:.3f}")
                results[f"LOO 1-NN Accuracy ({metric})"] = accuracy

                dist_gen_ref = distances[n_mesh:, :n_mesh]  # Lower left block, distance generated <-> reference
                cov, mmd = cov_mmd(dist_gen_ref, num_points=n_points)
                logger.info(
                    f"{split.upper()} Coverage ({metric}): {cov:.3f}"
                )  # Fraction of references matched to at least one generation
                logger.info(f"{split.upper()} MMD ({metric}): {mmd:.3f}")  # Average NN distance generated <-> reference
                results[f"Coverage ({metric})"] = cov
                results[f"Minimum Matching Distance ({metric})"] = mmd

                dist_gen_gen = distances[n_mesh:, n_mesh:]  # Lower right block, distance generated <-> generated
                dist_ref_ref = distances[:n_mesh, :n_mesh]  # Upper left block, distance reference <-> reference
                ecd = two_sample_test(dist_gen_ref, dist_gen_gen, dist_ref_ref)
                logger.info(f"{split.upper()} ECD ({metric}): {ecd:.3f}")  # Edge Count Difference
                results[f"Edge Count Difference ({metric})"] = ecd
            elif metric == "fpd":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                fabric = lightning.Fabric(precision=cfg[split].precision)
                model = cast(Model, get_feature_extractor(cfg).to(device))
                encode = cast(Any, model.encode)

                with fabric.autocast(), torch.inference_mode():
                    gen_feat = torch.cat(
                        [
                            cast(torch.Tensor, encode(_to_tensor_cloud(pcd, device=device))).cpu()
                            for pcd in generated_clouds
                        ]
                    )
                    ref_feat = torch.cat(
                        [
                            cast(torch.Tensor, encode(_to_tensor_cloud(pcd, device=device))).cpu()
                            for pcd in reference_clouds
                        ]
                    )

                gen_feat = gen_feat.mean(dim=1).double().numpy()
                ref_feat = ref_feat.mean(dim=1).double().numpy()
                min_len = min(len(ref_feat), len(gen_feat))

                fpd = fid.fid_from_feats(gen_feat, ref_feat)
                logger.info(f"{split.upper()} FPD: {fpd:.3f}")
                results["FPD"] = fpd

                kpd = fid.kernel_distance(ref_feat, gen_feat)
                logger.info(f"{split.upper()} KPD: {kpd:.3f}")
                results["KPD"] = kpd

                prec, rec = compute_prd_from_embedding(gen_feat[:min_len], ref_feat[:min_len])
                logger.info(f"{split.upper()} Precision: {prec.mean():.3f}, Recall: {rec.mean():.3f}")
                results["Precision (3D)"] = prec.mean()
                results["Recall (3D)"] = rec.mean()

                prdc = compute_prdc(ref_feat[:min_len], gen_feat[:min_len], nearest_k=5, n_jobs=num_workers)
                logger.info(
                    f"{split.upper()} "
                    f"Precision: {prdc['precision']:.3f}, "
                    f"Recall: {prdc['recall']:.3f}, "
                    f"Density: {prdc['density']:.3f}, "
                    f"Coverage: {prdc['coverage']:.3f}"
                )
                results.update({k.capitalize() + " (3D,improved)": v for k, v in prdc.items()})
        elif "fid" in metric or "kid" in metric or "prd" in metric:
            if split != "train":
                logger.warning("Evaluation is NOT done on the TRAIN set!")
            make_custom_stats = partial(
                fid.make_custom_stats, num_workers=num_workers, batch_size=128, verbose=cfg.log.verbose > 1
            )
            compute_fid = partial(
                fid.compute_fid,
                num_workers=num_workers,
                batch_size=128,
                dataset_split="custom",
                verbose=cfg.log.verbose > 1,
            )
            compute_kid = partial(
                fid.compute_kid,
                num_workers=num_workers,
                batch_size=128,
                dataset_split="custom",
                verbose=cfg.log.verbose > 1,
            )

            render_views = file_suffix if file_suffix == "3dshape2vecset" else views
            save_dir_gen = mesh_dir / "fid"
            if not stats_only and (not save_dir_gen.exists() or cfg.test.overwrite):
                render_gen(
                    cfg, save_dir_gen, generated_meshes, render_views, n_views, num_workers, progress=cfg.log.progress
                )
            if load_ref_meshes and (not stats_dir.exists() or cfg.test.overwrite):
                render_ref(
                    cfg, stats_dir, reference_meshes, render_views, n_views, num_workers, progress=cfg.log.progress
                )

            if stats_name and not no_stats:
                if not stats_exist(stats_name) or cfg.test.overwrite:
                    if stats_exist(stats_name) and cfg.test.overwrite:
                        fid.remove_custom_stats(stats_name)
                    logger.info(f"Computing FID stats for {stats_name}")
                    make_custom_stats(stats_name, str(stats_dir))
                for view in trange(n_views, desc="Computing FID stats", disable=cfg.log.verbose):
                    stats_name_view = f"{stats_name}_view_{view}"
                    save_dir_ref_view = str(stats_dir / f"view_{view}")
                    if not stats_exist(stats_name_view) or cfg.test.overwrite:
                        if stats_exist(stats_name_view) and cfg.test.overwrite:
                            fid.remove_custom_stats(stats_name_view)
                        _debug_level_1(f"Computing FID stats for {stats_name_view}")
                        make_custom_stats(stats_name_view, save_dir_ref_view)
                    if "clip" in metric:
                        clip_stats_exit = stats_exist(stats_name_view, model_name="clip_vit_b_32")
                        if not clip_stats_exit or cfg.test.overwrite:
                            if clip_stats_exit and cfg.test.overwrite:
                                fid.remove_custom_stats(stats_name_view, model_name="clip_vit_b_32")
                            _debug_level_1(f"Computing FID stats for {stats_name_view} (CLIP)")
                            make_custom_stats(stats_name_view, save_dir_ref_view, model_name="clip_vit_b_32")
                if stats_only:
                    return

            if "clip" in metric:
                fid_score = 0.0
                for view in trange(n_views, desc="Computing FID (CLIP)", disable=cfg.log.verbose):
                    stats_name_view = f"{stats_name}_view_{view}"
                    save_dir_gen_view = str(save_dir_gen / f"view_{view}")
                    save_dir_ref_view = str(stats_dir / f"view_{view}")
                    if stats_exist(stats_name_view, model_name="clip_vit_b_32"):
                        _fid_score = compute_fid(
                            save_dir_gen_view, dataset_name=stats_name_view, model_name="clip_vit_b_32"
                        )
                        _debug_level_1(f"{split.upper()} FID (CLIP view {view}): {_fid_score}")
                        fid_score += _fid_score
                    else:
                        _fid_score = compute_fid(save_dir_gen_view, save_dir_ref_view, model_name="clip_vit_b_32")
                        _debug_level_1(f"{split.upper()} FID (CLIP view {view}): {_fid_score}")
                        fid_score += _fid_score
                fid_score /= n_views
                logger.info(f"{split.upper()} FID (CLIP): {fid_score:.3f}")
                results["FID (CLIP)"] = fid_score

            if "prd" in metric:
                precision = 0.0
                recall = 0.0
                prdc_metrics: list[dict[str, float]] = []
                for view in trange(n_views, desc="Computing PRD", disable=cfg.log.verbose):
                    stats_name_view = f"{stats_name}_view_{view}"
                    save_dir_gen_view = str(save_dir_gen / f"view_{view}")
                    if stats_exist(stats_name_view):
                        ref_feat_raw = fid.get_reference_statistics(
                            stats_name_view, res=1024, split="custom", metric="KID"
                        )
                        gen_feat_raw = fid.get_folder_features(
                            save_dir_gen_view,
                            model=fid.build_feature_extractor(mode="clean"),
                            num_workers=num_workers,
                            batch_size=128,
                            description=f"PRD(C) {save_dir_gen.name} : ",
                            verbose=cfg.log.verbose > 1,
                        )
                        if ref_feat_raw is None or gen_feat_raw is None:
                            raise ValueError(f"Missing PRD features for {stats_name_view}")
                        ref_feat = np.asarray(ref_feat_raw[0] if isinstance(ref_feat_raw, tuple) else ref_feat_raw)
                        gen_feat = np.asarray(gen_feat_raw)
                        min_len = min(len(ref_feat), len(gen_feat))
                        prec, rec = compute_prd_from_embedding(gen_feat[:min_len], ref_feat[:min_len])
                        _debug_level_1(
                            f"{split.upper()} Precision: {prec.mean():.3f}, Recall: {rec.mean():.3f} (view {view})"
                        )
                        precision += prec.mean()
                        recall += rec.mean()
                        if "prdc" in metric:
                            prdc = compute_prdc(ref_feat[:min_len], gen_feat[:min_len], nearest_k=5, n_jobs=num_workers)
                            _debug_level_1(
                                f"{split.upper()} "
                                f"Precision: {prdc['precision']:.3f}, "
                                f"Recall: {prdc['recall']:.3f}, "
                                f"Density: {prdc['density']:.3f}, "
                                f"Coverage: {prdc['coverage']:.3f} (view {view})"
                            )
                            prdc_metrics.append(prdc)
                    else:
                        raise ValueError(f"Stats for {stats_name_view} do not exist")
                precision /= n_views
                recall /= n_views
                logger.info(f"{split.upper()} Precision: {precision:.3f}, Recall: {recall:.3f}")
                results["Precision"] = precision
                results["Recall"] = recall
                if "prdc" in metric:
                    prdc_metrics_mean = {k: np.mean([m[k] for m in prdc_metrics]) for k in prdc_metrics[0]}
                    logger.info(f"{split.upper()} PRDC:")
                    pprint(prdc_metrics_mean)
                    results.update({k.capitalize() + " (improved)": v for k, v in prdc_metrics_mean.items()})

            if "fid" in metric:
                if stats_exist(stats_name):
                    fid_score = compute_fid(str(save_dir_gen), dataset_name=stats_name)
                else:
                    fid_score = compute_fid(str(save_dir_gen), str(stats_dir))
                logger.info(f"{split.upper()} FID (global): {fid_score:.3f}")

            fid_score = 0.0
            kid_score = 0.0
            for view in trange(n_views, desc="Computing FID", disable=cfg.log.verbose):
                stats_name_view = f"{stats_name}_view_{view}"
                save_dir_gen_view = str(save_dir_gen / f"view_{view}")
                save_dir_ref_view = str(stats_dir / f"view_{view}")
                if stats_exist(stats_name_view):
                    if "fid" in metric:
                        _fid_score = compute_fid(save_dir_gen_view, dataset_name=stats_name_view)
                        _debug_level_1(f"{split.upper()} FID (view {view}): {_fid_score:.3f}")
                        fid_score += _fid_score
                    if "kid" in metric:
                        _kid_score = compute_kid(save_dir_gen_view, dataset_name=stats_name_view)
                        _debug_level_1(f"{split.upper()} KID (view {view}): {_kid_score:.3f}")
                        kid_score += _kid_score
                else:
                    if "fid" in metric:
                        _fid_score = compute_fid(save_dir_gen_view, save_dir_ref_view)
                        _debug_level_1(f"{split.upper()} FID (view {view}): {_fid_score:.3f}")
                        fid_score += _fid_score
                    if "kid" in metric:
                        _kid_score = compute_kid(save_dir_gen_view, save_dir_ref_view)
                        _debug_level_1(f"{split.upper()} KID (view {view}): {_kid_score:.3f}")
                        kid_score += _kid_score
            fid_score /= n_views
            kid_score /= n_views
            if "fid" in metric:
                logger.info(f"{split.upper()} FID: {fid_score:.3f}")
                results["FID"] = fid_score
            if "kid" in metric:
                logger.info(f"{split.upper()} KID: {kid_score:.3f}")
                results["KID"] = kid_score
        else:
            logger.error(f"Invalid metric: {metric}")

        update_metrics_file(save_file, results)
