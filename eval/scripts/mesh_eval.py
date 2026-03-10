from pathlib import Path
from pprint import pprint
from typing import Any, cast

import hydra
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tabulate import tabulate
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from trimesh import Trimesh

from dataset import BOP, YCB, get_dataset
from libs import check_mesh_contains, furthest_point_sample
from utils import (
    get_num_workers,
    load_mesh,
    log_optional_dependency_summary,
    resolve_save_dir,
    setup_config,
    setup_logger,
    stdout_redirected,
    suppress_known_optional_dependency_warnings,
    to_numpy,
    to_tensor,
    tqdm_joblib,
)

from ..src.gen_metrics import directed_hausdorff, paired_distances
from ..src.utils import PCD_ERROR_DICT, eval_occupancy, eval_pointcloud, overwrite_results

logger = setup_logger(__name__)


def _debug_level_2(msg: str) -> None:
    debug_fn = getattr(logger, "debug_level_2", logger.debug)
    debug_fn(msg)


def _to_numpy_array(value: list[str] | Tensor | np.ndarray) -> np.ndarray:
    return np.asarray(to_numpy(cast(Any, value)))


def _to_tensor_value(value: list[str] | Tensor | np.ndarray) -> Tensor:
    tensor = to_tensor(cast(Any, value))
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor-convertible value, got {type(tensor).__name__}")
    return tensor


def eval_mesh(
    mesh: Trimesh, item: dict[str, list[str] | Tensor | np.ndarray], n_points: int = int(1e5)
) -> dict[str, float]:
    out_dict = PCD_ERROR_DICT.copy()
    if mesh.is_empty:
        _debug_level_2("Empty mesh")
    else:
        if "pointcloud" in item:
            pointcloud_gt = _to_numpy_array(item["pointcloud"])
            pointcloud, index = mesh.sample(n_points, return_index=True)
            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[index]

            normals_gt = item.get("pointcloud.normals")
            normals_gt_np: np.ndarray | None = None
            if normals_gt is not None:
                normals_gt_np = _to_numpy_array(normals_gt)
                out_dict["normals"] = 0

            pcd_result = eval_pointcloud(pointcloud, pointcloud_gt, normals, normals_gt_np, method="kdtree")
            out_dict.update(
                {
                    "chamfer-l1": pcd_result["chamfer-l1"],
                    "chamfer-l2": pcd_result["chamfer-l2"],
                    "pcd_f1": pcd_result["f1"],
                    "pcd_precision": pcd_result["precision"],
                    "pcd_recall": pcd_result["recall"],
                }
            )
            if "normals" in pcd_result:
                out_dict["normals"] = pcd_result["normals"]

        if "points" in item and "points.occ" in item:
            points_gt = _to_numpy_array(item["points"])
            occ_gt = _to_numpy_array(item["points.occ"])

            occ = check_mesh_contains(mesh, points_gt).astype(np.float32)
            if occ_gt.min() < 0:
                occ_gt = (occ_gt <= 0).astype(np.float32)

            occ_result = eval_occupancy(_to_tensor_value(occ == 1).int(), _to_tensor_value(occ_gt == 1))
            out_dict.update(
                {
                    "iou": occ_result["iou"],
                    "f1": occ_result["f1"],
                    "precision": occ_result["precision"],
                    "recall": occ_result["recall"],
                }
            )
    return out_dict


def single_eval(
    item: dict[str, list[str] | Tensor],
    save_dir: Path,
    dataset: Dataset,
    num_files: int = 1,
    num_shards: int = 1,
    show: bool = False,
) -> dict[str, int | float | str] | None:
    dataset_any = cast(Any, dataset)
    index = int(cast(Tensor, item["index"]).item())
    path = cast(list[str], item["inputs.path"])[0]
    if isinstance(dataset_any, YCB):
        if dataset_any.load_real_data:
            obj_dict = dataset_any.objects[index // dataset_any.num_images]
        else:
            obj_dict = dataset_any.objects[index % len(dataset_any.objects)]
        obj_name = obj_dict["name"] + "_" + str(3 * index)
    else:
        obj_dict = dataset_any.objects[index]
        obj_name = obj_dict["name"]
        if num_files > 1 or num_shards > 1:
            obj_name = obj_dict["name"] + "_" + str(index)

    obj_category = obj_dict["category"]
    category_name = dataset_any.metadata[obj_category]["name"]
    category_name = category_name.split(",")[0] if "," in category_name else category_name

    mesh_path = save_dir / "meshes" / obj_category / f"{obj_name}.ply"
    if mesh_path.is_file():
        mesh = Trimesh(*load_mesh(mesh_path), process=False)
        results = eval_mesh(mesh, cast(dict[str, list[str] | Tensor | np.ndarray], item))
        if (mesh_path.parent / (mesh_path.stem + "_1.ply")).exists():
            pcds = [mesh.sample(2048)]
            for i in range(1, 10):
                path_i = mesh_path.parent / (mesh_path.stem + f"_{i}.ply")
                if path_i.exists():
                    mesh_i = Trimesh(*load_mesh(path_i), process=False)
                    pcds.append(mesh_i.sample(2048))
                    results_i = eval_mesh(mesh_i, cast(dict[str, list[str] | Tensor | np.ndarray], item))
                    if results_i["pcd_f1"] > results["pcd_f1"]:
                        mesh = mesh_i
                        results = results_i

            n_pcds = len(pcds)
            pcds = np.concatenate(pcds).reshape(n_pcds, 2048, 3)
            distances = paired_distances([pcds], [pcds], progress=False)
            distances = distances[:n_pcds, :n_pcds]
            distances = distances[np.triu_indices(n_pcds, -1)]
            results["tmd"] = 2 * (distances / 2048).sum() / (n_pcds - 1)

            inputs_src = cast(Tensor, item.get("inputs.depth", item["inputs"]))
            inputs = cast(Tensor, furthest_point_sample(inputs_src.cuda(), num_samples=1024))
            inputs = inputs.repeat(n_pcds, 1, 1).transpose(1, 2)
            pcds = cast(Tensor, torch.from_numpy(pcds).float().transpose(1, 2).cuda())
            with torch.autocast(device_type="cuda", dtype=torch.float16), torch.inference_mode():
                results["uhd"] = directed_hausdorff(inputs, pcds).mean().cpu().item()

        if show:
            import open3d as o3d

            o3d_any = cast(Any, o3d)

            pprint(results)
            inputs = cast(Tensor, item.get("inputs.depth", item["inputs"])[0]).numpy()
            vertices = cast(Tensor, item["mesh.vertices"][0]).numpy()
            triangles = cast(Tensor, item["mesh.triangles"][0]).numpy()

            inputs_o3d = o3d_any.geometry.PointCloud(o3d_any.utility.Vector3dVector(inputs))
            mesh_gt_o3d = cast(Any, Trimesh(vertices, triangles, process=False)).as_open3d.compute_vertex_normals()
            mesh_o3d = cast(Any, mesh).as_open3d.compute_vertex_normals()

            material_pcd = o3d_any.visualization.rendering.MaterialRecord()
            material_pcd.shader = "defaultUnlit"
            material_pcd.base_color = [0.8, 0, 0, 1]
            material_pcd.point_size = 4

            material = o3d_any.visualization.rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.base_color = [0.2, 0.5, 0.4, 1]

            material_gt = o3d_any.visualization.rendering.MaterialRecord()
            material_gt.shader = "defaultLitTransparency"
            material_gt.base_color = [0.4, 0.4, 0.4, 0.7]

            o3d_any.visualization.draw(
                [
                    {"name": "mesh_gt", "geometry": mesh_gt_o3d, "material": material_gt},
                    {"name": "mesh", "geometry": mesh_o3d, "material": material},
                    {"name": "inputs", "geometry": inputs_o3d, "material": material_pcd},
                ],
                show_skybox=False,
                bg_color=[0.0, 0.0, 0.0, 1.0],
            )

            """
            vis_item = {"inputs": item.get("inputs.depth", item["inputs"])[0].numpy(),
                        "inputs.path": item["inputs.path"][0],
                        "mesh.vertices": mesh.vertices,
                        "mesh.triangles": mesh.faces}
            if "points" in item and "points.occ" in item:
                vis_item.update({"points": item["points"][0].numpy(),
                                 "points.occ": item["points.occ"][0].numpy()})
            if "pointcloud" in item:
                vis_item.update({"pointcloud": item["pointcloud"][0].numpy()})
            Visualize(show_occupancy="points" in vis_item,
                      show_pointcloud="pointcloud" in vis_item,
                      show_mesh=True)(vis_item)
            """

        eval_dict = {
            "index": index,
            "path": path,
            "object category": obj_category,
            "category name": category_name,
            "object name": obj_name,
        }
        eval_dict.update(results)
        return eval_dict
    else:
        logger.warning(f"Mesh file {mesh_path} does not exist")
    return None


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg = setup_config(cfg)
    suppress_known_optional_dependency_warnings()
    log_optional_dependency_summary(logger, cfg)
    split = cfg.test.split
    save_dir = resolve_save_dir(cfg) / "generation" / split
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = cfg.data[f"{split}_ds"][0]
    save_file = save_dir / f"{ds}_{split}_mesh_eval_full_{cfg.implicit.threshold:.2f}{cfg.files.suffix}.pkl"
    save_file_class = save_dir / f"{ds}_{split}_mesh_eval_{cfg.implicit.threshold:.2f}{cfg.files.suffix}.csv"
    if not cfg.test.merge:
        if not overwrite_results(cfg, save_file):
            return

    cfg.aug.noise = 0
    cfg.aug.edge_noise = 0
    cfg.aug.remove_angle = False
    cfg.pointcloud.normals = True
    with stdout_redirected(enabled=cfg.log.verbose < 2):
        dataset = get_dataset(cfg, splits=(split,), load_mesh=cfg.vis.show)[split]

    num_workers = get_num_workers(cfg.load.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0 if isinstance(dataset, BOP) else num_workers,
        prefetch_factor=cfg.load.prefetch_factor,
        pin_memory=cfg.load.pin_memory,
        generator=torch.Generator().manual_seed(cfg.misc.seed),
    )

    if split != "test":
        logger.warning("Evaluation is NOT done on the TEST set!")

    with tqdm_joblib(tqdm(total=len(loader), desc=f"Evaluating {cfg.log.name} meshes", disable=not cfg.log.progress)):
        with Parallel(n_jobs=1 if cfg.vis.show else max(1, num_workers)) as p:
            eval_dicts = p(
                delayed(single_eval)(
                    item, save_dir, dataset, cfg.data.num_files.test, cfg.data.num_shards.test, cfg.vis.show
                )
                for item in loader
            )

    eval_dicts = [d for d in eval_dicts if d is not None]
    assert len(eval_dicts), "No mesh evaluation results were generated"

    eval_df = cast(Any, pd.DataFrame(eval_dicts))
    eval_df.set_index(["index"], inplace=True)

    if not cfg.test.overwrite and save_file.is_file():
        logger.info(f"Merging with existing results from {save_file}")
        eval_df = cast(Any, pd.concat([eval_df, pd.read_pickle(save_file)], join="inner", ignore_index=True))
        eval_df = eval_df.drop_duplicates(subset="path", keep="first").reset_index(drop=True)
    eval_df.to_pickle(save_file)

    eval_df_class = cast(Any, eval_df.groupby(by=["category name"]).mean(numeric_only=True))
    eval_df_class.loc["mean (macro)"] = eval_df_class.mean(numeric_only=True)
    eval_df_class.loc["mean (micro)"] = eval_df.mean(numeric_only=True)
    eval_df_class.to_csv(save_file_class)
    print(eval_df_class)

    m = str(cfg.train.model_selection_metric).split("/")[-1]
    if m in eval_df_class.columns:
        if "loss" in m:
            logger.info(
                f"\nBest class ({m}): {eval_df_class[m].idxmin()} ({eval_df_class[m].min()})\n"
                f"Worst class ({m}): {eval_df_class[m].idxmax()} ({eval_df_class[m].max()})"
            )
        else:
            logger.info(
                f"\nBest class ({m}): {eval_df_class[m].idxmax()} ({eval_df_class[m].max()})\n"
                f"Worst class ({m}): {eval_df_class[m].idxmin()} ({eval_df_class[m].min()})"
            )

    table = tabulate(eval_df_class, headers="keys")
    with open(save_file_class.parent / save_file_class.name.replace(".csv", ".txt"), "w") as f:
        f.write(table)


if __name__ == "__main__":
    main()
