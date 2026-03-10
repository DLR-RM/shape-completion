import json
from typing import Any, cast

import hydra
import lightning
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pykdtree.kdtree import KDTree
from tabulate import tabulate
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary
from tqdm import tqdm

from dataset import BOP, YCB, CocoInstanceSegmentation, GraspNetEval, SaveData, TableTop, get_dataset
from libs import EarthMoversDistance
from models import (
    AutoregressiveModel,
    DiffusionModel,
    DinoInstSeg,
    DinoInstSeg3D,
    GridDiffusionModel,
    InstOccPipeline,
    classification_loss,
    get_model,
    probs_from_logits,
    regression_loss,
)
from utils import (
    config_hash,
    get_num_workers,
    log_optional_dependency_summary,
    make_3d_grid,
    resolve_save_dir,
    setup_config,
    setup_logger,
    suppress_known_optional_dependency_warnings,
    to_scalar,
    to_tensor,
)
from visualize import Generator

from ..src.utils import EMPTY_RESULTS_DICT, eval_occupancy, eval_pointcloud, overwrite_results

logger = setup_logger(__name__)


def _debug_level_1(message: str) -> None:
    debug_fn = getattr(logger, "debug_level_1", logger.debug)
    debug_fn(message)


def _to_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    if isinstance(value, list):
        return _to_int(value[0])
    return int(value)


@torch.no_grad()
def test_step(
    model: nn.Module,
    fabric: lightning.Fabric,
    item: dict[str, Any],
    threshold: float = 0.5,
    sdf: bool = False,
    basic: bool = True,
    uncertain: float | None = None,
    metrics: list[str] | None = None,
    points_batch_size: int | None = None,
    return_logits: bool = False,
    **kwargs,
) -> dict[str, float]:
    model_any = cast(Any, model)
    data = {k: to_tensor(v, unsqueeze=False) for k, v in item.items()}
    if hasattr(model_any, "evaluate"):
        with fabric.autocast(), torch.inference_mode():
            eval_results = model_any.evaluate(
                data=data,
                threshold=threshold,
                regression=sdf,
                prefix="",
                metrics=metrics,
                points_batch_size=points_batch_size,
                return_logits=return_logits,
                **kwargs,
            )
            return cast(dict[str, float], eval_results)

    if model_any.name not in ("PSGN", "PCN", "SnowflakeNet"):
        points = cast(torch.Tensor, item["points"]).to(model_any.device)
        occ = cast(torch.Tensor, item["points.occ"]).to(model_any.device)
        if occ.min() < 0:
            occ = occ <= 0

    probs: Any = None
    out: Any = None
    out_var: Any = None
    c: Any = None
    results: dict[str, float] = dict()
    if model_any.name not in ("PSGN", "PCN", "SnowflakeNet", "MCDropoutNet", "PSSNet"):
        if model_any.name == "VQDIF":
            c, quant_ind, quant_diff = model_any.encode(**data, return_quant=True)
            mode = quant_ind.view(-1).mode()[0]
            results["non-empty"] = (quant_ind != mode).sum().cpu().item()
            results["masked"] = (
                model_any.get_mask(
                    data["inputs"],
                    resolution=16,
                    padding=model_any.encoder.padding,
                )
                .sum()
                .cpu()
                .item()
            )
            results["quant-diff"] = quant_diff.cpu().item()
            out = model_any.decode(points, c)
        else:
            out = model_any(**data)

        if sdf:
            probs = -out
            results["loss"] = cast(Any, regression_loss)(out, occ, reduction="sum_points").cpu().item()
            occ = occ <= 0
        else:
            probs = probs_from_logits(out)
            results["loss"] = cast(Any, classification_loss)(out, occ, reduction="sum_points").cpu().item()
    elif model_any.name in ("MCDropoutNet", "PSSNet"):
        inputs = data["inputs"]
        if model_any.name == "MCDropoutNet":
            out, out_var = model_any.mc_sample(inputs)
        else:
            logits = model_any.predict_many(inputs)
            probs = torch.sigmoid(logits)
            probs_mean = probs.mean(dim=0)
            out = torch.logit(probs_mean)
            out_var = probs.var(dim=0)

        grid_shape = cast(tuple[int, int, int], tuple(inputs.shape[1:4]))
        voxel_grid = 2.5 * make_3d_grid((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), grid_shape).numpy()
        _, indices = KDTree(voxel_grid).query(points.squeeze(0).cpu().numpy(), k=1, eps=0)
        out = out.view(1, -1)[:, indices.astype(int)]
        out_var = out_var.view(1, -1)[:, indices.astype(int)]

        if sdf:
            probs = -out
            results["loss"] = cast(Any, regression_loss)(out, occ, reduction="sum_points").cpu().item()
            occ = occ <= 0
        else:
            probs = probs_from_logits(out)
            results["loss"] = cast(Any, classification_loss)(out, occ, reduction="sum_points").cpu().item()
    elif model_any.name in ("PSGN", "PCN", "SnowflakeNet"):
        out = model_any(**data)
        out_pcd = out[-1]
        indices = np.random.randint(out_pcd.size(1), size=1024)
        pcd = cast(torch.Tensor, item["pointcloud"]).to(model_any.device)
        """
        if batch["index"][0] % 1000 == 0:
            vis_inputs = out[-1].squeeze(0).cpu().numpy()
            vis_inputs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vis_inputs[indices]))
            vis_inputs.paint_uniform_color([1, 0, 0])
            vis_pcd = pcd.squeeze(0).cpu().numpy()
            vis_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vis_pcd[:len(indices)]))
            vis_pcd.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_geometries([vis_inputs, vis_pcd])
        """

        full_pcd_result = eval_pointcloud(out_pcd, pcd)
        pcd_result = {
            "loss": full_pcd_result["chamfer-l1"],
            "f1": full_pcd_result["f1"],
            "precision": full_pcd_result["precision"],
            "recall": full_pcd_result["recall"],
        }
        if probs is None:
            pcd_result = {f"pcd_{k}": v for k, v in pcd_result.items()}
        results.update(pcd_result)
        if not basic:
            emd = EarthMoversDistance().to(model_any.device)
            emd = emd(out_pcd[:, indices], pcd[:, : len(indices)], eps=0.002, iters=1000)[0].sqrt().mean()
            results["emd"] = emd.cpu().item()
    else:
        raise ValueError(f"Unknown model type {type(model)}")

    if probs is None:
        return results

    # Eval occupied
    occ_prob = probs if out.dim() == 2 else probs[:, 1]
    occ_pos_pred = cast(torch.Tensor, probs >= threshold)
    occ_pos = cast(torch.Tensor, occ == 1)

    unc_prob = probs if out.dim() == 2 else probs[:, 2]
    unc_threshold = float(uncertain if uncertain is not None else threshold)
    unc_pos_pred = cast(torch.Tensor, probs >= unc_threshold)
    unc_pos = cast(torch.Tensor, occ == 2)

    if occ_pos_pred.sum() == 0 and occ_pos.sum() == 0:
        results.update(EMPTY_RESULTS_DICT)
    else:
        results.update(eval_occupancy(occ_prob, occ_pos, threshold=threshold))
    """
    # New results: Ignore occ/unc confusion
    occ_prob[unc_pos] = 0
    new_results = eval_occupancy(occ_prob, occ_pos, threshold=threshold)
    new_results = {f"n_{k}": v for k, v in new_results.items()}
    results.update(new_results)
    """

    # Eval uncertain
    if uncertain:
        if out.dim() == 3:
            if unc_pos_pred.sum() == 0 and unc_pos.sum() == 0:
                uncertain_results = EMPTY_RESULTS_DICT
            else:
                uncertain_results = eval_occupancy(unc_prob, unc_pos, threshold=uncertain)

            unc_prob[occ_pos] = 0
            # new_uncertain_results = eval_occupancy(unc_prob, unc_pos, threshold=threshold)
        else:
            log_unc_threshold = np.log(uncertain / (1 - uncertain))
            if model_any.name in ("MCDropoutNet", "PSSNet"):
                logits = out.squeeze(0).cpu().numpy()
                logits_var = out_var.squeeze(0).cpu().numpy()

                free_space_mask = (logits > 0.5 * log_unc_threshold) | (logits_var < logits_var.mean())
                logits[free_space_mask] = log_unc_threshold - 0.1
            else:
                log_threshold = np.log(threshold / (1 - threshold))

                points_np = points.squeeze(0).cpu().numpy()
                grad = Generator(cast(Any, model_any), points_batch_size).estimate_vertex_normals(
                    points_np, c, normalize=False
                )
                grad_norm = np.linalg.norm(grad, axis=1)

                logits = out.squeeze(0).cpu().numpy()
                free_space_mask = (logits > log_threshold) | (grad_norm > grad_norm.mean())
                logits[free_space_mask] = log_unc_threshold - 0.1
            unc_pos_pred = to_tensor(logits >= log_unc_threshold)

            if unc_pos_pred.sum() == 0 and unc_pos.sum() == 0:
                uncertain_results = EMPTY_RESULTS_DICT
            else:
                uncertain_results = eval_occupancy(unc_pos_pred, unc_pos, threshold=uncertain)

            # New results: Ignore unc/occ confusion
            unc_pos_pred[occ_pos] = False
            # new_uncertain_results = eval_occupancy(unc_pos_pred, unc_pos, threshold=threshold)

        uncertain_results = {f"u_{k}": v for k, v in uncertain_results.items()}
        # new_uncertain_results = {f"n_u_{k}": v for k, v in new_uncertain_results.items()}
        results.update(uncertain_results)
        # results.update(new_uncertain_results)
    if not basic:
        import torchmetrics.functional as M

        auprc = cast(torch.Tensor, M.average_precision(occ_prob, occ_pos, task="binary"))
        ece = cast(torch.Tensor, M.calibration_error(occ_prob, occ_pos, task="binary"))
        results.update({"auprc": auprc.cpu().numpy().item(), "ece": ece.cpu().numpy().item()})
    return results


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg = setup_config(cfg)
    suppress_known_optional_dependency_warnings()

    split = cfg.test.split

    save_dir = resolve_save_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    if cfg.vis.save:
        save_data = SaveData(
            save_dir / "data" / split,
            save_points=False,
            save_pointcloud=False,
            save_mesh=False,
            threshold=cfg.implicit.threshold,
            sdf=cfg.implicit.sdf,
        )

    ds = cfg.data[f"{split}_ds"][0]
    cfg_id = config_hash(cfg, length=10)
    cfg_tag = f"{cfg_id}{cfg.files.suffix}"
    logger.info(f"Config ID: {cfg_tag}")
    log_optional_dependency_summary(logger, cfg)
    save_file = save_dir / f"{ds}_{split}_eval_full_{cfg_tag}.pkl"
    if not overwrite_results(cfg, save_file):
        return

    uncertain_threshold = 0
    if cfg.points.load_uncertain and "uncertain_threshold" in cfg.implicit:
        uncertain_threshold = cfg.implicit.uncertain_threshold

    dataset = get_dataset(cfg, splits=(split,))[split]
    num_workers = get_num_workers(cfg.load.num_workers)
    collate_fn = None
    try:
        from train import get_collate_fn

        collate_fn = get_collate_fn(cfg, split, cfg.test.batch_size or 1)
    except ImportError as e:
        logger.warning(f"Unable to import train module, custom collate functions won't be available: {e}")

    sampler = None
    indices = cfg.vis.index
    if indices is not None:
        if isinstance(indices, int):
            idx_list = [indices]
        elif isinstance(indices, str):
            idx_list = [int(s) for s in indices.split(",")]
        else:
            idx_list = [int(i) for i in indices]
        sampler = SubsetRandomSampler(idx_list)
        _debug_level_1(f"Using SubsetRandomSampler for indices: {idx_list}")

    loader = DataLoader(
        dataset,
        batch_size=1 if isinstance(dataset, BOP) else cfg.test.batch_size or 1,
        shuffle=cfg[split].shuffle and sampler is None,
        sampler=sampler,
        num_workers=0 if isinstance(dataset, BOP) else num_workers,
        collate_fn=collate_fn,
        prefetch_factor=cfg.load.prefetch_factor,
        persistent_workers=True if num_workers else False,
        pin_memory=cfg.load.pin_memory,
        generator=torch.Generator().manual_seed(cfg.misc.seed),
    )

    if cfg.model.weights is None and cfg.model.checkpoint is None:
        _debug_level_1("No weights or checkpoint specified. Trying to load best model.")
        cfg.model.load_best = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).eval().to(device)
    model_any = cast(Any, model)
    fabric = lightning.Fabric(precision=cfg[split].precision)

    summary_batch_size = int(loader.batch_size or 1)
    item = {k: v.to(device) for k, v in next(iter(loader)).items() if torch.is_tensor(v)}
    if cfg.vis.num_query_points:
        item["points"] = torch.zeros(summary_batch_size, int(cfg.vis.num_query_points), 3, device=device)
    with fabric.autocast():
        summary(model, input_data=item, depth=3 + cfg.log.verbose)

    if split != "test":
        logger.warning("Evaluation is NOT done on the TEST set!")

    batch_iou = list()
    eval_dicts = list()
    for batch in tqdm(loader, desc=f"Evaluating {cfg.log.name}", disable=not cfg.log.progress):
        if all(batch.get("inputs.skip", [False])):
            continue

        if len(batch["index"]) > 1 and hasattr(model_any, "predict"):
            data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with fabric.autocast(), torch.inference_mode():
                if isinstance(model_any, (DiffusionModel, AutoregressiveModel)):
                    logits, loss = model_any.predict(data, return_loss=True, points_batch_size=cfg.vis.num_query_points)
                    batch["logits"] = logits
                    batch["loss"] = loss
                elif isinstance(model_any, (DinoInstSeg, DinoInstSeg3D, InstOccPipeline)):
                    batch.update(model_any(**data, points_batch_size=cfg.vis.num_query_points))
                else:
                    logits = model_any.predict(
                        **data,
                        sample=cfg.get("sample", False),  # VAE
                        points_batch_size=cfg.vis.num_query_points,
                    )
                    batch["logits"] = logits
                    occ = data["points.occ"].long() == 1
                    probs = probs_from_logits(logits).float()
                    occ_result = eval_occupancy(probs, occ, cfg.implicit.threshold)
                    batch_iou.append(occ_result["iou"])

        for i in range(len(batch["index"])):
            item = {k: (v[i : i + 1] if torch.is_tensor(v) else [v[i]]) for k, v in batch.items()}
            index = _to_int(item["index"])

            if item.get("inputs.skip", False):
                continue

            if isinstance(model_any, GridDiffusionModel) and model_any.denoise_fn.ndim == 2:
                eval_dicts.append(
                    {
                        "index": index,
                        "category name": _to_int(item["inputs.labels"]),
                        "loss": float(cast(torch.Tensor, item["loss"]).mean().item()),
                        "sigma": float(cast(torch.Tensor, item["logits"]).item()),
                    }
                )
                print(eval_dicts[-1])
                continue

            eval_data = cast(
                dict[str, Any],
                test_step(
                    model,
                    fabric,
                    item,
                    threshold=cfg.implicit.threshold,
                    sdf=cfg.implicit.sdf,
                    basic=cfg.test.basic,
                    uncertain=uncertain_threshold,
                    metrics=cfg.log.metrics,
                    points_batch_size=cfg.vis.num_query_points,  # maybe subsample val.num_query_points
                    return_logits=cfg.vis.save,
                    sample=cfg.get("sample", False),  # VAE
                    align_to_gt=cfg.get("align_to_gt", False),
                    mask_iou=cfg.get("mask_iou", False),  # InstSeg
                    show=cfg.vis.show,
                ),
            )

            if cfg.vis.save:
                item = {k: v[0] for k, v in item.items()}
                item["logits"] = eval_data["logits"]
                item["inputs.logits"] = eval_data.get("inputs.logits", eval_data.get("inputs/logits"))
                save_data(item)
            eval_data = {k: v for k, v in eval_data.items() if "logits" not in k}

            log_data: Any = {}
            if hasattr(model_any, "get_log"):
                log_data = cast(Any, model_any.get_log())
                if split == "train":
                    if isinstance(log_data, dict) and "latent_hist" in log_data:
                        eval_data["stats"] = model_any.get_log("latent_hist")
                    elif isinstance(log_data, dict) and "quantized_hist" in log_data:
                        eval_data["stats"] = model_any.get_log("quantized_hist")
                if isinstance(log_data, dict):
                    eval_data.update({k: v for k, (v, _) in log_data.items() if k != "stats"})
                if hasattr(model_any, "clear_log"):
                    model_any.clear_log()

            if isinstance(dataset, (CocoInstanceSegmentation, TableTop, GraspNetEval)):
                category_name = item["category.name"]
                obj_category = item["category.id"]
                obj_name = item["inputs.name"]
                if isinstance(category_name, list):
                    category_name = category_name[0]
                if isinstance(obj_category, list):
                    obj_category = obj_category[0]
            else:
                if isinstance(dataset, YCB):
                    if dataset.load_real_data:
                        obj_dict = cast(Any, dataset).objects[index // dataset.num_images]
                    else:
                        obj_dict = cast(Any, dataset).objects[index % len(cast(Any, dataset).objects)]
                    obj_name = obj_dict["name"] + "_" + str(3 * index)
                else:
                    obj_dict = cast(Any, dataset).objects[index]
                    obj_name = obj_dict["name"]
                    if cfg.data.num_files.test > 1 or cfg.data.num_shards.test > 1:
                        obj_name = obj_dict["name"] + "_" + str(index)
                obj_category = obj_dict["category"]
                category_name = cast(Any, dataset).metadata[obj_category]["name"]
                category_name = category_name.split(",")[0] if "," in category_name else category_name

            eval_dict = {
                "index": index,
                "path": item["inputs.path"],
                "object category": obj_category,
                "category name": category_name,
                "object name": obj_name,
            }
            eval_dict.update(eval_data)
            eval_dicts.append(eval_dict)

    epoch_metrics = cast(dict[str, Any] | None, model_any.on_validation_epoch_end())

    stats = [d.pop("stats") for d in eval_dicts if "stats" in d]
    if stats and cfg.model.weights:
        stats = np.concatenate(stats)
        n_latent = _to_int(cast(Any, model_any.n_latent))
        stats = dict(
            mean=torch.from_numpy(stats.reshape(-1, n_latent).mean(0)),
            std=torch.from_numpy(stats.reshape(-1, n_latent).std(0)),
        )
        model_stats = cast(dict[str, Any], model_any.stats or dict())
        model_stats.update(stats)
        state_dict = cast(dict[str, Any], model_any.state_dict())
        state_dict["stats"] = model_stats
        if not getattr(model_any, "bit_diffusion", cfg.get("bit_diffusion", False)):
            torch.save(dict(model=state_dict), cfg.model.weights)
        _debug_level_1(
            f"Latent stats: "
            f"mean={state_dict['stats']['mean'].mean().item():.3f}, "
            f"std={state_dict['stats']['std'].mean().item():.3f} "
            f"({state_dict['stats']['std'].min().item():.3f},"
            f"{state_dict['stats']['std'].max().item():.3f})"
        )
    """
    from matplotlib import pyplot as plt
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d

    sigmas = np.array([d["sigma"] for d in eval_dicts])
    losses = np.array([d["loss"] for d in eval_dicts])

    # Sort sigmas and losses
    sorted_indices = np.argsort(sigmas)
    sigmas = sigmas[sorted_indices]
    losses = losses[sorted_indices]

    # Remove duplicates if any
    unique_sigmas, unique_indices = np.unique(sigmas, return_index=True)
    unique_losses = losses[unique_indices]

    # Create a function for interpolation (using 'linear' to handle potential close values)
    interp_func = interp1d(unique_sigmas, unique_losses, kind='linear', fill_value='extrapolate')

    # Create a dense set of points for smooth curve
    smooth_sigmas = np.logspace(np.log10(sigmas.min()), np.log10(sigmas.max()), 1000)
    smooth_losses = interp_func(smooth_sigmas)

    # Apply Gaussian smoothing
    smooth_losses = gaussian_filter1d(smooth_losses, sigma=5)

    plt.plot(smooth_sigmas, smooth_losses, label='Loss', color='blue')
    plt.show()
    """

    eval_df = cast(Any, pd.DataFrame(eval_dicts))
    eval_df.set_index(["index"], inplace=True)

    if not cfg.test.overwrite and save_file.is_file():
        logger.info(f"Merging with existing results from {save_file}")
        eval_df = pd.concat([eval_df, pd.read_pickle(save_file)], join="inner", ignore_index=True)
        eval_df_any = cast(Any, eval_df)
        eval_df = cast(Any, eval_df_any.drop_duplicates(subset=["path"], keep="first").reset_index(drop=True))
    if epoch_metrics:
        epoch_metrics_clean = {k: to_scalar(v) for k, v in epoch_metrics.items()}
        eval_df.attrs["epoch"] = epoch_metrics_clean
        with open(save_file.with_suffix(".json"), "w") as f:
            json.dump(epoch_metrics_clean, f, indent=2)
    eval_df.to_pickle(save_file)

    eval_df_class = cast(Any, eval_df.groupby(by=["category name"]).mean(numeric_only=True))
    eval_df_class.loc["mean (macro)"] = eval_df_class.mean(numeric_only=True)
    eval_df_class.loc["mean (micro)"] = eval_df.mean(numeric_only=True)
    eval_df_class.to_csv(save_file.with_suffix(".csv"))

    print(eval_df_class)
    m = cfg.train.model_selection_metric.split("/")[-1]
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
    if len(batch_iou) > 0:
        logger.info(f"Mean IoU (batched): {np.mean(batch_iou)}")
    if epoch_metrics:
        for k, v in epoch_metrics.items():
            logger.info(f"{k.upper()}: {v}")

    table = tabulate(cast(Any, eval_df_class), headers="keys")
    with open(save_file.with_suffix(".txt"), "w") as f:
        f.write(table)
        if len(batch_iou) > 0:
            f.write(f"\nMean IoU (batched): {np.mean(batch_iou)}\n")
        if epoch_metrics:
            f.write("\n\nEPOCH METRICS\n")
            for k, v in epoch_metrics.items():
                f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
