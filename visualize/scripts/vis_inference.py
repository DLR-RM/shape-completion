import copy
from joblib import cpu_count
from pathlib import Path
from pprint import PrettyPrinter
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import lightning
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import DictConfig, OmegaConf
from pykdtree.kdtree import KDTree
from torch import Tensor
from tqdm import trange, tqdm
from trimesh import Trimesh

from dataset import get_dataset, Visualize, VoxelizeInputs, ShapeNet, BOP
from eval import eval_pointcloud, eval_occupancy as get_metrics, EMPTY_RESULTS_DICT
from libs import check_mesh_contains
from models import ConvONet, IFNet, MCDropoutNet, PSSNet, get_model, probs_from_logits
from utils import make_3d_grid, resolve_save_dir, to_tensor, normalize, setup_logger, setup_config
from visualize import Generator

logger = setup_logger(__name__)


def occupancy_contour_plot(grid: np.ndarray,
                           threshold: float = 0.5,
                           bounds: Tuple[float, float] = (-0.5, 0.5),
                           scale: Optional[float] = None,
                           slice: Optional[int] = None,
                           sdf: bool = False,
                           padding: float = 0.1) -> Tuple[mpl.figure.Figure, List[mpl.figure.Axes]]:
    res = grid.shape[0]
    box_size = (bounds[1] - bounds[0]) + padding
    points = (box_size * make_3d_grid((bounds[0],) * 3,
                                      (bounds[1],) * 3,
                                      (res,) * 3)).numpy()
    values = grid.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    slice_min = -res / 2
    slice = res / 2 if slice is None else slice
    thickness = box_size / (grid.shape[0] - 1)
    mask_x = (x >= thickness * (slice_min + slice)) & (x < thickness * (slice_min + slice + 1))
    mask_y = (y >= thickness * (slice_min + slice)) & (y < thickness * (slice_min + slice + 1))
    mask_z = (z >= thickness * (slice_min + slice)) & (z < thickness * (slice_min + slice + 1))

    for i, mask in enumerate([mask_z, mask_x, mask_y]):
        x = y = np.linspace(box_size * bounds[0], box_size * bounds[1], res)
        z = values[mask].reshape((res, res))
        if i == 0:
            z = z.T
        if scale is not None and sdf:
            z /= scale
        center = threshold if sdf else np.log(threshold / (1 - threshold))
        levels = np.union1d(np.linspace(z.min(), center, np.clip(np.abs(z.min()) / 4, 8, 12).astype(int)),
                            np.linspace(center, z.max(), np.clip(z.max() / 2, 4, 6).astype(int)))
        handle = axes[i].contourf(x, y, z,
                                  cmap="bwr",
                                  levels=levels,
                                  norm=TwoSlopeNorm(vcenter=center, vmin=z.min(), vmax=z.max()))
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(handle, cax=cax)
        axes[i].contour(handle,
                        levels=handle.levels,
                        colors="black",
                        linewidths=0.5,
                        linestyles="solid")

    for ax in axes:
        ax.set_frame_on(True)
        ax.tick_params(right=False,
                       left=False,
                       top=False,
                       bottom=False,
                       labelright=False,
                       labelleft=False,
                       labeltop=False,
                       labelbottom=False)
        ax.set_aspect("equal")
    return fig, axes


def evaluate(probs: Tensor,
             occ: Tensor,
             points: np.ndarray = None,
             mesh: Trimesh = None,
             threshold: float = 0.5,
             verbose: bool = False) -> Dict[str, float]:
    if (probs >= threshold).sum() == 0 and occ.sum() == 0:
        results = EMPTY_RESULTS_DICT
    else:
        results = get_metrics(probs, occ, threshold)
    if mesh is not None and points is not None:
        occ_pred = check_mesh_contains(mesh, points)
        if occ_pred.sum() == 0 and occ.sum() == 0:
            mesh_results = EMPTY_RESULTS_DICT
        else:
            mesh_results = get_metrics(to_tensor(occ_pred), occ)
        mesh_results = {f"m_{k}": v for k, v in mesh_results.items()}
        results.update(mesh_results)
    if verbose:
        PrettyPrinter().pprint(results)
    return results


def process_mesh(mesh: Trimesh, min_num_triangles: int = 1000) -> Trimesh:
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                     o3d.utility.Vector3iVector(mesh.faces))
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    not_major = cluster_n_triangles[triangle_clusters] < 0.5 * max(cluster_n_triangles)
    too_small = cluster_n_triangles[triangle_clusters] < min_num_triangles
    triangles_to_remove = not_major | too_small
    if np.all(triangles_to_remove):
        mesh.remove_triangles_by_mask(not_major)
    else:
        mesh.remove_triangles_by_mask(triangles_to_remove)
    return Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    setup_config(cfg)

    if "shapenet" in cfg.data.test_ds[0]:
        save_dir = resolve_save_dir(cfg) / "generation" / "meshes" / cfg.data.categories[0]
    else:
        save_dir = resolve_save_dir(cfg) / "generation" / "meshes"
    save_dir.mkdir(parents=True, exist_ok=True)

    split = cfg.test.split
    dataset = get_dataset(cfg,
                          load_points=True,
                          load_pointcloud=cfg.vis.pointcloud,
                          load_mesh=True,
                          load_voxels=cfg.vis.voxels,
                          splits=(split,))[split]
    dataset.transform[-2].add_keys(["inputs.rot",
                                    "inputs.frame",
                                    "inputs.pose",
                                    "inputs.scale",
                                    "inputs.norm_scale",
                                    "inputs.norm_offset",
                                    "mesh.scale"])

    plot_contour = "plot_contour" in cfg.vis and cfg.vis.plot_contour

    cfg.model.load_best = cfg.model.weights is None and cfg.model.checkpoint is None
    model = get_model(cfg).eval().cuda()
    fabric = lightning.Fabric(precision=cfg.train.precision)
    model = fabric.setup_module(model)

    resolution = cfg.vis.resolution
    points_batch_size = int(cfg.vis.num_query_points) if cfg.vis.num_query_points else None
    if points_batch_size is None:
        if isinstance(model, ConvONet):
            points_batch_size = int(1e6) if resolution > 128 else int(5e6)
        elif isinstance(model, IFNet):
            points_batch_size = 250000
    if resolution <= 64:
        points_batch_size = resolution ** 3
    generator = Generator(model,
                          points_batch_size=points_batch_size,
                          threshold=cfg.implicit.threshold,
                          refinement_steps=cfg.vis.refinement_steps,
                          padding=cfg.norm.padding,
                          simplify_nfaces=cfg.vis.simplify,
                          resolution0=resolution,
                          with_normals=cfg.vis.normals,
                          sdf=cfg.implicit.sdf)

    threshold = 0 if cfg.implicit.sdf and cfg.implicit.threshold == 0.5 else cfg.implicit.threshold
    vis = Visualize(show_inputs=cfg.vis.inputs,
                    show_occupancy=cfg.vis.occupancy,
                    show_points=cfg.vis.points,
                    show_frame=cfg.vis.frame,
                    show_pointcloud=cfg.vis.pointcloud,
                    show_mesh=cfg.vis.mesh,
                    show_box=cfg.vis.box,
                    show_cam=cfg.vis.cam,
                    threshold=threshold,
                    padding=cfg.norm.padding,
                    sdf=cfg.implicit.sdf,
                    cam_forward=(0, 0, -1) if cfg.inputs.frame == "cam" else (0, 0, 1))

    rng = np.random.default_rng(cfg.misc.seed)
    check_list = list()
    if isinstance(dataset, BOP):
        if dataset.name in ["hb", "lm", "tyol"]:
            check_list = rng.integers(len(dataset), size=min(len(dataset), 100))
        elif dataset.name == "ycbv":
            check_list = rng.integers(len(dataset), size=min(len(dataset), 50))
    elif isinstance(dataset, ShapeNet):
        obj_counter = defaultdict(int)
        # check_list = rng.integers(len(dataset), size=min(len(dataset), 1000))
        # check_list = np.random.randint(len(dataset) // 2, len(dataset), size=min(len(dataset), 1000))
    # check_list.sort()
    # check_list = [141, 24]  # HB Primesense (trinary: 80/20, binary: 50/1)
    # check_list = [102, 153]  # HB Kinect
    # check_list = [29, 79]  # TYOL 3
    # check_list = [67]  # TYOL 4
    # check_list = [20]  # TYOL 5 (binary: 50/5)
    # check_list = [71, 72]  # TYOL 6 (binary: 50/5)
    # check_list = [4]  # TYOL 20 (binary: 50/5)
    # check_list = [6, 46]  # TYOL 21 (binary: 50/5)
    # check_list = [493, 1102]  # LM (binary: 50/5)
    # check_list = [537, 639]  # YCBV 48 (binary: 25/1, trinary: 25/20)
    # check_list = [115, 423]  # YCBV 55 (binary: 25/1, trinary: 25/20)

    if len(check_list) > 0:
        check_list.sort()
        tqdm_range = tqdm(check_list)
    else:
        tqdm_range = trange(len(dataset))

    save = "save" in cfg.vis and cfg.vis.save
    eval_dicts = []
    for index in tqdm_range:
        item = dataset[index]

        if isinstance(dataset, BOP):
            if item.get("inputs.skip", False):
                continue
        elif isinstance(dataset, ShapeNet):
            obj_dict = dataset.objects[index]
            obj_category = obj_dict["category"]
            instance = obj_counter[obj_category]
            if instance >= max(2, 10 // len(dataset.categories)):
                continue
            obj_counter[obj_category] += 1

        basename = f"{item['inputs.name']}_{item['index']}"

        """
        if save and not cfg.vis.show:
            if os.path.exists(os.path.join(out_dir, mesh_filename)):
                if os.path.exists(os.path.join(out_dir, mesh_filename.replace(".ply", "_inputs.ply"))):
                    if os.path.exists(os.path.join(out_dir, mesh_filename.replace(".ply", ".npz"))):
                        continue
        """

        if cfg.vis.show:
            vis(item)

        inputs = item["inputs"]
        if inputs.ndim == 3:
            voxel_grid = (1 + cfg.norm.padding) * make_3d_grid((-0.5,) * 3, (0.5,) * 3, inputs.shape).numpy()
            inputs = voxel_grid[inputs.ravel() == 1]
        mesh_dict = {"occ": None,
                     "gt": None,
                     "unc": None,
                     "unc_gt": None,
                     "inputs": Trimesh(inputs, process=False)}
        if cfg.model.checkpoint or cfg.model.weights or cfg.model.load_best:
            mesh_dict["gt"] = Trimesh(item["mesh.vertices"], item["mesh.triangles"], process=False)
        # mesh_list = [simplify_mesh(mesh, f_target=5000)]
        # mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(item["mesh.vertices"]),
        #                                  o3d.utility.Vector3iVector(item["mesh.triangles"]))
        # mesh = mesh.simplify_quadric_decimation(10000)
        # mesh_list = [trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))]

        points = item["points"]
        occ = item["points.occ"]
        inputs = item["inputs"]

        grid, grid_points, c = generator.generate_grid({"inputs": inputs})
        if isinstance(model, (MCDropoutNet, PSSNet)):
            grid, grid_var = grid[0], grid[1]

            voxelize_inputs = next(x for x in dataset.transform if isinstance(x, VoxelizeInputs))
            voxel_grid = voxelize_inputs.voxelizer.grid_points
            kdtree = KDTree(voxel_grid)

            if not cfg.points.voxelize:
                _, indices = kdtree.query(points)
                logits = to_tensor(grid.ravel()[indices])
                logits_var = to_tensor(grid_var.ravel()[indices])
            else:
                logits = to_tensor(grid.ravel())
                logits_var = to_tensor(grid_var.ravel())

            if np.any([dim != cfg.vis.resolution for dim in grid.shape]):
                _, indices = kdtree.query(grid_points)
                grid = grid.ravel()[indices].reshape((cfg.vis.resolution,) * 3)
                grid_var = grid_var.ravel()[indices].reshape((cfg.vis.resolution,) * 3)
        """
        elif isinstance(model, PSSNet):
            meshes = list()
            model.eval().cuda()
            logits = model.predict_many(to_tensor(inputs), num_predictions=100)
            for g in logits.cpu().numpy():
                # log_g = np.log((g + 1e-32) / (1 - g + 1e-32))
                mesh = generator.extract_mesh(g, c, threshold=cfg.implicit.threshold)
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                                 o3d.utility.Vector3iVector(mesh.faces))
                mesh.paint_uniform_color(np.random.uniform(0, 1, 3))
                mesh.compute_vertex_normals()
                meshes.append(mesh)
            o3d.visualization.draw_geometries(meshes)
            continue

            thresh = cfg.implicit.threshold
            log_thresh = np.log(thresh / (1 - thresh))
            uncertain_thresh = cfg.implicit.uncertain_threshold
            log_uncertain_thresh = np.log(uncertain_thresh / (1 - uncertain_thresh))

            grid_mean = grid[0]
            grid = grid[1]

            mesh = generator.extract_mesh(grid_mean, c, threshold=thresh)
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                             o3d.utility.Vector3iVector(mesh.faces))
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            mesh.compute_vertex_normals()

            always_thresh = 0.00001 * uncertain_thresh
            log_always_thresh = np.log(always_thresh / (1 - always_thresh))
            mask = (grid > log_always_thresh).sum(0)
            always = mask == len(grid)
            never = mask == 0
            uncertain = ~always & ~never

            grid_always = grid.copy()
            grid_always[:, ~always] = grid_always[:, ~always].min(0)
            grid_always[:, always] = grid_always[:, always].max(0)

            mesh = generator.extract_mesh(grid_always.mean(0), c, threshold=always_thresh)
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                             o3d.utility.Vector3iVector(mesh.faces))
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            mesh.compute_vertex_normals()

            grid_uncertain = grid.copy()
            grid_uncertain[:, ~uncertain] = grid_uncertain[:, ~uncertain].min(0)
            grid_uncertain[:, uncertain] = grid_uncertain[:, uncertain].max(0)

            mesh_unc = generator.extract_mesh(grid_uncertain.max(0), c, threshold=uncertain_thresh)
            mesh_unc = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_unc.vertices),
                                                 o3d.utility.Vector3iVector(mesh_unc.faces))
            mesh_unc.paint_uniform_color([1, 0, 0])
            mesh_unc.compute_vertex_normals()

            o3d.visualization.draw_geometries([mesh, mesh_unc])
            continue
        """

        mesh = generator.extract_mesh(grid, c)
        if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
            mesh_dict["occ"] = mesh

        if occ.max() == 2:
            kdtree = KDTree(grid_points)
            _, indices = kdtree.query(points[occ == 2])
            u_grid = np.zeros(len(grid_points))
            u_grid[indices] = 1
            u_mesh = generator.extract_mesh(u_grid.reshape((cfg.vis.resolution,) * 3), threshold=0.5)
            if len(u_mesh.vertices) > 0 and len(u_mesh.faces) > 0:
                u_mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(u_mesh.vertices),
                                                       o3d.utility.Vector3iVector(u_mesh.faces))

                u_mesh_o3d.compute_vertex_normals()
                pcd = u_mesh_o3d.sample_points_uniformly(10000)

                radii = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
                u_mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                                                                             o3d.utility.DoubleVector(
                                                                                                 radii))

                u_mesh_o3d.compute_vertex_normals()
                pcd = u_mesh_o3d.sample_points_uniformly(10000)
                u_mesh_o3d = \
                    o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5, n_threads=cpu_count())[0]

                u_mesh_test = Trimesh(np.asarray(u_mesh_o3d.vertices), np.asarray(u_mesh_o3d.triangles), process=True)
                mesh_dict["unc_gt"] = u_mesh_test

        # geometries = [o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)]
        geometries = []

        if "metrics" in cfg.val and cfg.val.metrics:
            if not save:
                print("OCCUPIED METRICS:")
            obj_dict = dataset.objects[index]
            obj_name = obj_dict["name"]
            obj_category = obj_dict["category"]
            category_name = dataset.metadata[obj_category]["name"]
            category_name = category_name.split(',')[0] if ',' in category_name else category_name
            eval_dict = {
                "index": index,
                "object category": obj_category,
                "category name": category_name,
                "object name": obj_name
            }
            eval_dicts.append(eval_dict)

            if not isinstance(model, (MCDropoutNet, PSSNet)):
                with torch.no_grad():
                    with generator.fabric.autocast():
                        logits = model.decode(to_tensor(points), c)
            probs = probs_from_logits(logits)

            if not save:
                print("Volume/Points metrics:")
            eval_dict.update(evaluate(probs,
                                      to_tensor(occ == 1),
                                      threshold=threshold,
                                      verbose=not save))

            if cfg.vis.show:
                prob_np = probs.squeeze(0).float().cpu().numpy()
                tp_mask = (occ == 1) & (prob_np >= threshold)
                fp_mask = (occ == 0) & (prob_np >= threshold)
                tn_mask = (occ == 0) & (prob_np < threshold)
                fn_mask = (occ == 1) & (prob_np < threshold)
                tp = points[tp_mask]
                tn = points[tn_mask]
                fp = points[fp_mask]
                fn = points[fn_mask]
                tp_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tp))
                tp_pcd.paint_uniform_color((0, 0, 0))
                fp_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fp))
                fp_pcd.paint_uniform_color((0, 0, 1))
                tn_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tn))
                tn_pcd.paint_uniform_color((0.8, 0.8, 0.8))
                fn_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fn))
                fn_pcd.paint_uniform_color((1, 0, 0))
                o3d.visualization.draw_geometries([tp_pcd, fp_pcd, tn_pcd, fn_pcd])

            if not save:
                print("Mesh/Grid metrics:")
            grid_preds = grid.ravel() >= np.log(threshold / (1 - threshold))
            grid_occ = check_mesh_contains(mesh_dict["gt"], grid_points)
            grid_results = evaluate(to_tensor(grid_preds),
                                    to_tensor(grid_occ),
                                    None if mesh_dict["occ"] is None else grid_points,
                                    mesh_dict["occ"],
                                    threshold,
                                    verbose=not save)
            grid_results = {f"g_{k}": v for k, v in grid_results.items()}
            eval_dict.update(grid_results)

            if cfg.vis.show:
                with torch.no_grad():
                    with generator.fabric.autocast():
                        logits = model.decode(to_tensor(grid_points), c)

                probs = probs_from_logits(logits)
                prob_np = probs.squeeze(0).float().cpu().numpy()
                tp_mask = (grid_occ == 1) & (prob_np >= threshold)
                fp_mask = (grid_occ == 0) & (prob_np >= threshold)
                tn_mask = (grid_occ == 0) & (prob_np < threshold)
                fn_mask = (grid_occ == 1) & (prob_np < threshold)
                tp = grid_points[tp_mask]
                tn = grid_points[tn_mask]
                fp = grid_points[fp_mask]
                fn = grid_points[fn_mask]
                tp_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tp))
                tp_pcd.paint_uniform_color((0, 0, 0))
                fp_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fp))
                fp_pcd.paint_uniform_color((0, 0, 1))
                tn_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tn))
                tn_pcd.paint_uniform_color((0.8, 0.8, 0.8))
                fn_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fn))
                fn_pcd.paint_uniform_color((1, 0, 0))
                o3d.visualization.draw_geometries([tp_pcd, fp_pcd, tn_pcd, fn_pcd])

            if mesh_dict["occ"] is not None:
                mesh = mesh_dict["occ"]
                pointcloud, pcd_index = mesh.sample(int(1e5), return_index=True)
                normals = mesh.face_normals[pcd_index]
                mesh_gt = mesh_dict["gt"]
                pointcloud_gt, pcd_index_gt = mesh_gt.sample(int(1e5), return_index=True)
                normals_gt = mesh_gt.face_normals[pcd_index_gt]
                pcd_results = eval_pointcloud(pointcloud, pointcloud_gt, normals, normals_gt)
                if not save:
                    print("Pointcloud/Surface metrics:")
                    PrettyPrinter().pprint(pcd_results)
                eval_dict.update(pcd_results)

        if "uncertain_threshold" in cfg.implicit:
            uncertain_threshold = cfg.implicit.uncertain_threshold
            mesh_thresh = threshold
            log_mesh_thresh = np.log(mesh_thresh / (1 - mesh_thresh))

            if cfg.cls.num_classes == 3:
                uncertain_grid, _, uncertain_feature = generator.generate_grid({"inputs": inputs}, extraction_class=2)
                mesh = generator.extract_mesh(uncertain_grid,
                                              uncertain_feature,
                                              threshold=uncertain_threshold,
                                              extraction_class=2)
                if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                    mesh_dict["unc"] = mesh
            elif isinstance(model, (MCDropoutNet, PSSNet)):
                uncertain_grid = copy.deepcopy(grid)

                log_uncertain_thresh = np.log(uncertain_threshold / (1 - uncertain_threshold))

                free_space_mask = (uncertain_grid > 0.5 * log_uncertain_thresh) | (grid_var < grid_var.mean())
                uncertain_grid[free_space_mask] = log_uncertain_thresh - 0.1

                mesh = generator.extract_mesh(uncertain_grid, c, uncertain_threshold)
                if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                    mesh = process_mesh(mesh, min_num_triangles=1000)
                    if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                        mesh_dict["unc"] = mesh

                if cfg.vis.show:
                    import matplotlib as mpl
                    # mask = grid_var.ravel() >= grid_var.mean()
                    mask = grid_var.ravel() >= grid_var.mean()

                    large_grid_var = grid_var.ravel()[mask]

                    cmap = mpl.cm.get_cmap('plasma')
                    grid_var_norm = normalize(large_grid_var, large_grid_var.min(), large_grid_var.max())
                    colors = np.array([cmap(x)[:3] for x in grid_var_norm])

                    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points[mask]))
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    o3d.visualization.draw_geometries([pcd])

                    # mesh_dict["inputs"] = Trimesh(grid_points[mask])
            else:
                uncertain_grid = copy.deepcopy(grid)
                # grad = np.array(np.gradient(grid)).reshape((3, -1)).T

                if plot_contour:
                    fig, axes = occupancy_contour_plot(grid=uncertain_grid,
                                                       threshold=cfg.implicit.uncertain_threshold,
                                                       bounds=generator.bounds,
                                                       slice=cfg.vis.slice if "slice" in cfg.vis else None,
                                                       sdf=cfg.implicit.sdf,
                                                       padding=cfg.norm.padding)
                    if save:
                        fig.savefig(Path(save_dir) / f"{basename}_unc_contour.png")

                grad = generator.estimate_normals(grid_points, c, normalize=False)
                grad_norm = np.linalg.norm(grad, axis=1).reshape((generator.resolution0,) * 3)

                if plot_contour:
                    fig, axes = occupancy_contour_plot(grid=-grad_norm,
                                                       threshold=-grad_norm.mean(),
                                                       bounds=generator.bounds,
                                                       slice=cfg.vis.slice if "slice" in cfg.vis else None,
                                                       sdf=True,
                                                       padding=cfg.norm.padding)
                    if save:
                        fig.savefig(Path(save_dir) / f"{basename}_grad_contour.png")
                    else:
                        plt.show()

                log_uncertain_thresh = np.log(uncertain_threshold / (1 - uncertain_threshold))

                free_space_mask = (uncertain_grid > log_mesh_thresh) | (grad_norm > grad_norm.mean())
                uncertain_grid[free_space_mask] = log_uncertain_thresh - 0.1

                mesh = generator.extract_mesh(uncertain_grid, c, uncertain_threshold)
                if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                    mesh = process_mesh(mesh, min_num_triangles=1000)
                    if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                        mesh_dict["unc"] = mesh

            if "metrics" in cfg.val and cfg.val.metrics:
                if not save:
                    print("UNCERTAIN METRICS:")
                # uncertain_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[occ == 2]))
                # uncertain_pcd.paint_uniform_color((0.67, 0.39, 0.98))
                # geometries.append(uncertain_pcd)
                uncertain_occ = to_tensor(occ == 2)

                # with torch.no_grad():
                #   logits = model.decode(to_tensor(points), c)

                if cfg.cls.num_classes == 3:
                    probs = torch.softmax(logits, dim=1)[:, 2]
                elif isinstance(model, (MCDropoutNet, PSSNet)):
                    logits = logits.squeeze(0).cpu().numpy()
                    logits_var = logits_var.squeeze(0).cpu().numpy()

                    free_space_mask = (logits > 0.5 * log_uncertain_thresh) | (logits_var < logits_var.mean())
                    logits[free_space_mask] = log_uncertain_thresh - 0.1
                    probs = to_tensor(logits >= log_uncertain_thresh)
                else:
                    grad = generator.estimate_normals(points, c, normalize=False)
                    grad_norm = np.linalg.norm(grad, axis=1)

                    logits = logits.squeeze(0).cpu().numpy()
                    free_space_mask = (logits > log_mesh_thresh) | (grad_norm > grad_norm.mean())
                    logits[free_space_mask] = log_uncertain_thresh - 0.1
                    probs = to_tensor(logits >= log_uncertain_thresh)

                if not save:
                    print("Volume/Points metrics:")
                uncertain_results = evaluate(probs, uncertain_occ, threshold=uncertain_threshold, verbose=not save)
                uncertain_results = {f"u_{k}": v for k, v in uncertain_results.items()}
                eval_dict.update(uncertain_results)
                if mesh_dict["unc"] is not None:
                    if not save:
                        print("Mesh/Grid metrics:")
                    probs = to_tensor(check_mesh_contains(mesh_dict["unc"], points))
                    uncertain_results = evaluate(probs, uncertain_occ, verbose=not save)
                    uncertain_results = {f"u_m_{k}": v for k, v in uncertain_results.items()}
                    eval_dict.update(uncertain_results)

                    if mesh_dict["unc_gt"] is not None:
                        mesh = mesh_dict["unc"]
                        pointcloud, pcd_index = mesh.sample(int(1e5), return_index=True)
                        normals = mesh.face_normals[pcd_index]
                        mesh_gt = mesh_dict["unc_gt"]
                        pointcloud_gt, pcd_index_gt = mesh_gt.sample(int(1e5), return_index=True)
                        normals_gt = mesh_gt.face_normals[pcd_index_gt]
                        uncertain_results = eval_pointcloud(pointcloud, pointcloud_gt, normals, normals_gt)
                        uncertain_results = {f"u_{k}": v for k, v in uncertain_results.items()}
                        if not save:
                            print("Pointcloud/Surface metrics:")
                            PrettyPrinter().pprint(uncertain_results)
                        eval_dict.update(uncertain_results)

        color_dict = {"gt": (0, 1, 0),
                      "occ": (0.7, 0.7, 0.7),
                      "unc": (1, 0, 0),
                      "unc_gt": (171 / 255, 99 / 255, 250 / 255),
                      "inputs": None}

        for name in ["occ", "gt", "unc", "unc_gt", "inputs"]:
            if mesh_dict[name] is not None:
                mesh = mesh_dict[name]
                if len(mesh.faces) > 0:
                    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                                     o3d.utility.Vector3iVector(mesh.faces))
                else:
                    mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh.vertices))

                """
                scale = item.get("inputs.norm_scale")
                if scale is not None:
                    mesh.scale(scale, center=(0, 0, 0))
                offset = item.get("inputs.norm_offset")
                if offset is not None:
                    mesh.translate(offset)
                if cfg.aug.scale is not None:
                    mesh.scale(1 / cfg.aug.scale, center=(0, 0, 0))
                """

                if save:
                    out_path = Path(save_dir)
                    params_path = out_path / f"{basename}_params.npz"
                    gt_path = out_path / f"{basename}_gt.ply"
                    input_path = out_path / f"{basename}_inputs.ply"
                    mesh_path = out_path / f"{basename}_mesh.ply"
                    unc_path = out_path / f"{basename}_uncertain.ply"
                    unc_gt_path = out_path / f"{basename}_uncertain_gt.ply"
                    if isinstance(mesh, o3d.geometry.PointCloud):
                        o3d.io.write_point_cloud(str(input_path), mesh)
                    else:
                        mesh.triangle_normals = o3d.utility.Vector3dVector([])
                        mesh.vertex_normals = o3d.utility.Vector3dVector([])
                        mesh.vertex_colors = o3d.utility.Vector3dVector([])

                        if name == "gt":
                            if isinstance(dataset, BOP):
                                scale = item["mesh.scale"]
                                # Applies ICP rotation -> cam/world rotation (x angle = 0) -> z up to y up rotation
                                rot = item["inputs.frame"] @ item["inputs.rot"] @ item["inputs.pose"][:3, :3]
                            else:
                                scale = item["inputs.scale"]
                                rot = item["inputs.frame"]  # Applies to cam or to world rotation

                            pose = np.eye(4)
                            pose[:3, :3] = rot
                            np.savez(params_path, pose=pose, scale=scale)
                            o3d.io.write_triangle_mesh(str(gt_path), mesh)
                        elif name == "occ":
                            o3d.io.write_triangle_mesh(str(mesh_path), mesh)
                        elif name == "unc":
                            o3d.io.write_triangle_mesh(str(unc_path), mesh)
                        elif name == "unc_gt":
                            o3d.io.write_triangle_mesh(str(unc_gt_path), mesh)
                else:
                    if color_dict[name] is not None:
                        mesh.paint_uniform_color(color_dict[name])
                    if isinstance(mesh, o3d.geometry.TriangleMesh):
                        mesh.compute_vertex_normals()
                    geometries.append(mesh)
        if not save and not plot_contour:
            # for i in range(len(geometries)):
            # geometry = geometries[i]
            # geometry.paint_uniform_color((0.7, 0.7, 0.7))
            # if i == len(geometries) - 1:
            # geometry = [o3d.geometry.TriangleMesh().create_sphere(radius=0.0005, resolution=10).paint_uniform_color((0.7, 0.7, 0.7)).compute_vertex_normals().translate(point, relative=False) for point in np.asarray(geometry.points)]
            # else:
            #     geometry = [geometry]
            o3d.visualization.draw_geometries(  # [geometries[0], geometries[2], geometries[-1]],
                geometries,
                width=800,
                height=800,
                window_name=basename,
                mesh_show_back_face=True,
                front=np.array([-0.0018263876147377365, 0.6859960776851548, 0.72760294509358869]),
                lookat=[0.0, 0.0, 0.0],
                up=np.array([-0.0049469628816617098, 0.72758905737237078, -0.6859954016975861]),
                zoom=0.8)

        if plot_contour:
            occupancy_contour_plot(grid=grid,
                                   threshold=threshold,
                                   bounds=generator.bounds,
                                   scale=cfg.aug.scale,
                                   slice=cfg.vis.slice if "slice" in cfg.vis else None,
                                   sdf=cfg.implicit.sdf,
                                   padding=cfg.norm.padding)
            if save:
                plt.savefig(Path(save_dir) / f"{basename}_contour.png")
            else:
                plt.show()

    if "metrics" in cfg.val and cfg.val.metrics:
        metrics_save_dir = save_dir.parent / "metrics"
        metrics_save_dir.mkdir(parents=True, exist_ok=True)

        save_file = metrics_save_dir / f"{split}_eval_full_{cfg.implicit.threshold:.2f}.pkl"
        save_file_class = metrics_save_dir / f"{split}_eval_{cfg.implicit.threshold:.2f}.csv"

        if "uncertain_threshold" in cfg.implicit:
            save_file = Path(str(save_file).replace(".pkl", f"_{cfg.implicit.uncertain_threshold:.2f}.pkl"))
            save_file_class = Path(str(save_file_class).replace(".csv", f"_{cfg.implicit.uncertain_threshold:.2f}.csv"))

        eval_df = pd.DataFrame(eval_dicts)
        eval_df.set_index(["index"], inplace=True)
        if save:
            eval_df.to_pickle(save_file)

        eval_df_class = eval_df.groupby(by=["category name"]).mean(numeric_only=True)
        eval_df_class.loc["mean"] = eval_df_class.mean()
        if save:
            eval_df_class.to_csv(save_file_class)
        print(eval_df_class)


if __name__ == "__main__":
    main()
