import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import open3d as o3d
import pandas as pd
import pyrender
import torch
import torch.nn.functional as F
import trimesh
from easy_o3d.utils import eval_transformation_data
from matplotlib import offsetbox
from matplotlib import pyplot as plt
from numpy import ndarray as array
from omegaconf import DictConfig
from PIL import Image
from pykdtree.kdtree import KDTree
from torch import Tensor
from torchmetrics import Metric
from trimesh import Trimesh

from dataset import NormalizeMesh
from libs import ChamferDistanceL2, check_mesh_contains
from utils import get_points, inv_trafo, load_mesh, look_at, normalize_mesh, setup_logger, to_tensor

logger = setup_logger(__name__)

# Maximum values for bounding box [-0.5, 0.5]^3
PCD_ERROR_DICT = {
    # "completeness": np.sqrt(3),
    # "accuracy": np.sqrt(3),
    # "completeness2": 3,
    # "accuracy2": 3,
    "chamfer-l1": 2 * np.sqrt(3),
    "chamfer-l2": 6,
    "pcd_f1": 0,
    "pcd_precision": 0,
    "pcd_recall": 0,
}

EMPTY_RESULTS_DICT = {
    "iou": np.nan,
    "f1": np.nan,
    "precision": np.nan,
    "recall": np.nan,
    "acc": 1.0,
    "tp": np.nan,
    "fp": np.nan,
    "tn": np.nan,
    "fn": np.nan,
}


def get_threshold_percentage(dist: np.ndarray, thresholds: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray([(dist <= t).mean() for t in thresholds])


@torch.no_grad()
def distance_p2p(
    points: np.ndarray | Tensor,
    points_gt: np.ndarray | Tensor,
    normals: np.ndarray | None = None,
    normals_gt: np.ndarray | None = None,
    method: str = "tensor",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    start = time.perf_counter()
    if method != "tensor":
        if torch.is_tensor(points):
            points = points.squeeze(0).cpu().numpy()
        if torch.is_tensor(points_gt):
            points_gt = points_gt.squeeze(0).cpu().numpy()

    if method == "tensor":
        points = to_tensor(points)
        points_gt = to_tensor(points_gt)
        chamfer_l2 = ChamferDistanceL2(return_indices=True).to(points.device)
        dist1, dist2, idx1, idx2 = chamfer_l2(points, points_gt)
        accuracy = dist1.squeeze(0).sqrt().cpu().numpy()
        completeness = dist2.squeeze(0).sqrt().cpu().numpy()
        accuracy_idx = idx1.squeeze(0).cpu().numpy()
        completeness_idx = idx2.squeeze(0).cpu().numpy()
    elif method == "kdtree":
        points = np.asarray(points)
        points_gt = np.asarray(points_gt)
        kdtree = KDTree(points_gt)
        accuracy, accuracy_idx = kdtree.query(points, k=1, eps=0)
        kdtree = KDTree(points)
        completeness, completeness_idx = kdtree.query(points_gt, k=1, eps=0)
    elif method == "open3d":
        gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_gt))
        pr = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        accuracy = np.asarray(gt.compute_point_cloud_distance(pr))
        completeness = np.asarray(pr.compute_point_cloud_distance(gt))
        # Fixme: this is not correct
        accuracy_idx = np.arange(len(points))
        completeness_idx = np.arange(len(points_gt))
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    logger.debug(f"Distance calculation with {method} method took {time.perf_counter() - start:.2f}s")

    if normals is not None and normals_gt is not None:
        if method == "open3d":
            raise NotImplementedError("Open3D is not supported for normal evaluation yet")

        normals = cast(np.ndarray, normals / np.linalg.norm(normals, axis=-1, keepdims=True))
        normals_gt = cast(np.ndarray, normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True))
        accuracy_idx_arr = cast(np.ndarray, accuracy_idx)
        completeness_idx_arr = cast(np.ndarray, completeness_idx)

        normals_accuracy = np.abs((normals_gt[accuracy_idx_arr] * normals).sum(axis=-1))
        normals_completeness = np.abs((normals[completeness_idx_arr] * normals_gt).sum(axis=-1))
        return accuracy, completeness, normals_accuracy, normals_completeness
    return accuracy, completeness, None, None


def eval_pointcloud(
    pointcloud: np.ndarray | Tensor,
    pointcloud_gt: np.ndarray | Tensor,
    normals: np.ndarray | None = None,
    normals_gt: np.ndarray | None = None,
    threshold: float = 0.01,
    method: str = "tensor",
) -> dict[str, float]:
    """
    Evaluates a pointcloud against a ground truth pointcloud using several metrics.

    This function computes the completeness, accuracy, Chamfer distance (both L1 and L2), precision,
    recall, and F1 score. If normal vectors are provided for both pointclouds, it also evaluates the correctness
    of the normals.

    Args:
        pointcloud (Union[np.ndarray, Tensor]): The pointcloud to be evaluated.
        pointcloud_gt (Union[np.ndarray, Tensor]): The ground truth pointcloud.
        normals (Optional[np.ndarray], optional): Normal vectors for the input pointcloud. Defaults to None.
        normals_gt (Optional[np.ndarray], optional): Normal vectors for the ground truth pointcloud. Defaults to None.
        threshold (float, optional): Threshold for precision and recall computation. Defaults to 0.01.
        method (str, optional): Method to use for pointcloud distance computation. Defaults to "tensor".

    Returns:
        Dict[str, float]: A dictionary containing the following keys:
            - "completeness": The completeness score.
            - "accuracy": The accuracy score.
            - "chamfer-l1": The L1 Chamfer distance.
            - "chamfer-l2": The L2 Chamfer distance.
            - "f1": The F1 score.
            - "precision": The precision.
            - "recall": The recall.
            - "normals": The normal correctness (only if normal vectors were provided for both pointclouds).

    Raises:
        ValueError: If the pointclouds or normal vectors (if provided) don't have the same shape.
    """
    accuracy, completeness, accuracy_normals, completeness_normals = distance_p2p(
        pointcloud, pointcloud_gt, normals, normals_gt, method
    )

    precision = (accuracy < threshold).mean()
    recall = (completeness < threshold).mean()
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall / (precision + recall))

    accuracy2 = (accuracy**2).mean()
    completeness2 = (completeness**2).mean()
    accuracy = accuracy.mean()
    completeness = completeness.mean()

    # Chamfer distance
    normals_correctness = None
    if completeness_normals is not None and accuracy_normals is not None:
        normals_correctness = 0.5 * (completeness_normals + accuracy_normals)
    chamfer_l2 = 0.5 * (completeness2 + accuracy2)
    chamfer_l1 = 0.5 * (completeness + accuracy)

    out_dict = {
        "completeness": completeness,
        "accuracy": accuracy,
        "chamfer-l1": chamfer_l1,
        "chamfer-l2": chamfer_l2,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
    if normals_correctness is not None:
        out_dict["normals"] = normals_correctness.mean()
    return out_dict


def eval_occupancy(probabilities: Tensor, occupancy: Tensor, threshold: float = 0.5) -> dict[str, float]:
    """
    Evaluates predicted occupancy against ground truth occupancy using several metrics.

    This function computes a confusion matrix based on the input probabilities and true occupancy. It then calculates
    various metrics including intersection over union (IoU), F1 score, precision, recall, and accuracy.

    Args:
        probabilities (Tensor): Predicted occupancy probabilities.
        occupancy (Tensor): Ground truth occupancy.
        threshold (float, optional): Threshold to binarize the predicted occupancy probabilities. Defaults to 0.5.

    Returns:
        Dict[str, float]: A dictionary containing the following keys:
            - "iou": Intersection over Union score.
            - "f1": F1 score.
            - "precision": Precision.
            - "recall": Recall.
            - "acc": Accuracy.
            - "tp": True positives count.
            - "fp": False positives count.
            - "tn": True negatives count.
            - "fn": False negatives count.

    Raises:
        ValueError: If the predicted probabilities and true occupancy don't have the same shape.
    """
    import torchmetrics.functional as M

    MCM = M.confusion_matrix(
        probabilities,
        occupancy.long(),
        task="binary",
        threshold=threshold if threshold > 0 else threshold + 1e-10,
        num_classes=2,
    )

    #       is,pred
    TP = MCM[1, 1]
    FP = MCM[0, 1]
    TN = MCM[0, 0]
    FN = MCM[1, 0]

    # Fraction of correctly labeled examples
    acc = (TP + TN) / (TP + TN + FP + FN)
    # Fraction of correctly predicted positives (TP) out of all positives (predicted and actual)
    iou = TP / (TP + FP + FN)
    # Fraction of correctly predicted positives (TP) out of all predicted positives
    precision = TP / (TP + FP)
    # Fraction of correctly predicted positives (TP) out of all actual positives
    recall = TP / (TP + FN)
    # Harmonic mean of precision and recall, i.e. high f1 score = few false positives and few false negatives
    squared_beta = 1  # Recall importance factor (recall is beta times as important as precision)
    f1 = (1 + squared_beta) * (precision * recall / (squared_beta * precision + recall))

    results = {
        "iou": torch.nan_to_num(iou).cpu().item(),
        "f1": torch.nan_to_num(f1).cpu().item(),
        "precision": torch.nan_to_num(precision).cpu().item(),
        "recall": torch.nan_to_num(recall).cpu().item(),
        "acc": torch.nan_to_num(acc.cpu()).item(),
        "tp": TP.cpu().item(),
        "fp": FP.cpu().item(),
        "tn": TN.cpu().item(),
        "fn": FN.cpu().item(),
    }
    return results


def accuracy(probabilities: array, labels: array, threshold: float = 0.5) -> float:
    """Computes the top 1 accuracy of the predicted class probabilities in percent.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        threshold: Determines at what point a prediction is considered belonging into the positive class (binary classifcation case).

    Returns:
        The top 1 accuracy in percent.
    """

    if len(probabilities.shape) == 2:
        return 100.0 * np.mean(np.argmax(probabilities, axis=1) == labels)
    else:
        return 100 * np.mean((probabilities >= threshold) == labels)
        # return 100 * np.mean(labels)


def confidence(probabilities: array, mean: bool = True) -> float | array:
    """The confidence of a prediction is the maximum of the predicted class probabilities.

    Args:
        probabilities: The predicted class probabilities.
        mean: If True, returns the average confidence over all provided predictions.

    Returns:
        The confidence.
    """
    if len(probabilities.shape) == 2:
        if mean:
            return np.mean(np.max(probabilities, axis=1))
        return np.max(probabilities, axis=1)
    else:
        conf = np.abs(2 * (probabilities - 0.5))
        if mean:
            return float(np.mean(conf))
        return conf
        # if mean:
        #     return np.mean(probabilities)
        # return probabilities


def expected_calibration_error(
    probabilities: array, labels: array, bins: int = 10
) -> tuple[float, array, array, array]:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.

    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` equally-spaced bins.

    Source: `On Calibration of Modern Neural Networks <https://arxiv.org/pdf/1706.04599.pdf)?>`_.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.

    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """
    conf = confidence(probabilities, mean=False)
    edges = np.linspace(0, 1, bins + 1)
    bin_ace = list()
    bin_accuracy = list()
    bin_confidence = list()
    ece = 0
    for i in range(bins):
        mask = np.logical_and(conf > edges[i], conf <= edges[i + 1])
        if any(mask):
            bin_acc = accuracy(probabilities[mask], labels[mask]) / 100
            bin_conf = cast(np.ndarray, conf)[mask].mean()
            ace = bin_conf - bin_acc
            ece += mask.mean() * np.abs(ace)

            bin_ace.append(ace)
            bin_accuracy.append(bin_acc)
            bin_confidence.append(bin_conf)
        else:
            bin_ace.append(0)
            bin_accuracy.append(0)
            bin_confidence.append(0)
    return ece, np.array(bin_ace), np.array(bin_accuracy), np.array(bin_confidence)


def reliability_diagram(probabilities, labels, bins=10, axis=None):
    ece, bin_aces, bin_accs, _bin_confs = expected_calibration_error(probabilities, labels, bins=bins)
    if axis is None:
        text = offsetbox.AnchoredText(
            f"ECE: {(ece * 100):.2f}%\nAccuracy: {accuracy(probabilities, labels):.2f}%\nConfidence: {100 * confidence(probabilities):.2f}%",
            loc="upper left",
            frameon=False,
            prop=dict(fontsize=12),
        )
        _, ax = plt.subplots(figsize=(9, 9), tight_layout=True)
        ax.add_artist(text)
    else:
        ax = axis
    ax.bar(
        x=np.arange(0, 1, 1 / bins),
        height=bin_accs,
        width=1 / bins,
        linewidth=1,
        edgecolor="black",
        align="edge",
        color="dodgerblue",
    )
    ax.bar(
        x=np.arange(0, 1, 1 / bins),
        height=bin_aces,
        bottom=bin_accs,
        width=1 / bins,
        linewidth=1,
        edgecolor="crimson",
        align="edge",
        color="crimson",
        fill=False,
        hatch="/",
    )
    ax.bar(
        x=np.arange(0, 1, 1 / bins),
        height=bin_aces,
        bottom=bin_accs,
        width=1 / bins,
        linewidth=1,
        edgecolor="crimson",
        align="edge",
        color="crimson",
        alpha=0.3,
    )

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), color="black", linestyle="dashed", linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", labelsize=12, right=False, top=False)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xlabel("Confidence", fontsize=14)
    plt.show()


def eval_mesh(
    mesh: str | Path | Trimesh,
    mesh_gt: str | Path | Trimesh,
    query_points: np.ndarray | None = None,
    offset: float | np.ndarray | None = 0,
    scale: float | None = 1,
    pose: str | Path | np.ndarray | None = None,
    normalize: bool = False,
) -> dict[str, float]:
    if not isinstance(mesh, Trimesh):
        mesh = Trimesh(*load_mesh(mesh), process=False, validate=False)
    if not isinstance(mesh_gt, Trimesh):
        mesh_gt = Trimesh(*load_mesh(mesh_gt), process=False, validate=False)
        if normalize:
            mesh_gt = Trimesh(
                NormalizeMesh()({"mesh.vertices": mesh_gt.vertices, "mesh.triangles": mesh_gt.faces})["mesh.vertices"],
                mesh_gt.faces,
                process=False,
                validate=False,
            )

    # Transform ground truth mesh to match input
    trafo: np.ndarray | None = None
    if pose is not None:
        trafo_data: np.ndarray | str | Path = pose
        if isinstance(pose, (str, Path)):
            trafo_data = np.load(pose)
        trafo = cast(np.ndarray, eval_transformation_data(cast(Any, trafo_data)))

    if trafo is not None:
        mesh_gt = cast(Trimesh, mesh_gt.apply_transform(trafo))
    offset = np.array([offset] * 3) if not isinstance(offset, np.ndarray) else offset
    init_scale = 1.0 if scale is None else float(scale)
    mesh_gt = cast(Trimesh, mesh_gt.apply_translation(-offset))
    mesh_gt = cast(Trimesh, mesh_gt.apply_scale(1 / init_scale))

    # Normalize generated and ground truth mesh for evaluation
    scale_norm = float((mesh_gt.bounds[1] - mesh_gt.bounds[0]).max())
    offset_norm = (mesh_gt.bounds[1] + mesh_gt.bounds[0]) / 2

    mesh_gt = cast(Trimesh, mesh_gt.apply_translation(-offset_norm))
    mesh_gt = cast(Trimesh, mesh_gt.apply_scale(1 / scale_norm))

    mesh = mesh.copy()
    mesh = cast(Trimesh, mesh.apply_translation(-offset_norm))
    mesh = cast(Trimesh, mesh.apply_scale(1 / scale_norm))

    pointcloud, index = mesh.sample(int(1e5), return_index=True)
    normals = mesh.face_normals[index]

    pointcloud_gt, index_gt = mesh_gt.sample(int(1e5), return_index=True)
    normals_gt = mesh_gt.face_normals[index_gt]

    result = PCD_ERROR_DICT.copy()
    pcd_result = eval_pointcloud(pointcloud, pointcloud_gt, normals, normals_gt)
    result.update(
        {
            "chamfer-l1": pcd_result["chamfer-l1"],
            "chamfer-l2": pcd_result["chamfer-l2"],
            "pcd_f1": pcd_result["f1"],
            "pcd_precision": pcd_result["precision"],
            "pcd_recall": pcd_result["recall"],
        }
    )

    if query_points is not None:
        query_points_arr = query_points.copy()
        query_points_arr -= offset_norm
        query_points_arr /= scale_norm

        occ = check_mesh_contains(mesh, query_points_arr)
        occ_gt = check_mesh_contains(mesh_gt, query_points_arr)

        occ_result = eval_occupancy(to_tensor(occ == 1).int(), to_tensor(occ_gt == 1))
        result.update(
            {
                "iou": occ_result["iou"],
                "f1": occ_result["f1"],
                "precision": occ_result["precision"],
                "recall": occ_result["recall"],
            }
        )

    return result


def iou_per_class(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    # pred and target shapes are (B, N) and (B, N, C)
    # One hot encode the target labels to have the same shape as predictions
    target_one_hot = F.one_hot(target, num_classes)  # (B, N, C)

    ious = []
    for cls in range(num_classes):
        pred_inds = pred[:, :, cls]
        target_inds = target_one_hot[:, :, cls]
        intersection = (pred_inds & target_inds).sum(dim=1).float()  # Shape (B,)
        union = (pred_inds | target_inds).sum(dim=1).float()  # Shape (B,)
        # Avoid division by zero
        union = torch.where(union == 0, torch.ones_like(union), union)
        # IoU per class, per batch item
        iou = intersection / union
        valid_iou = iou[union > 0]  # Avoid classes that do not appear in the batch
        ious.append(valid_iou.mean())  # Mean IoU for this class across the batch
    return torch.stack(ious)  # Stack to get shape (C,)


def batched_miou(preds: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    """
    preds: A tensor of predictions, shape (B, N, C)
    targets: A tensor of labels, shape (B, N)
    num_classes: Integer specifying number of classes
    """
    # Convert predictions to one hot format and take argmax
    preds_one_hot = torch.nn.functional.one_hot(preds.argmax(dim=-1), num_classes)
    ious = iou_per_class(preds_one_hot, targets, num_classes)
    # Mean IoU across all classes, ignoring nan values
    return torch.nanmean(ious)


def calculate_iou(preds, labels, num_classes):
    # preds: (B,N,C) after softmax or similar activation
    # labels: (B,N) ground truth labels
    batch_size = preds.shape[0]

    ious = torch.zeros(batch_size, num_classes)
    for batch in range(batch_size):
        for class_id in range(num_classes):
            if class_id not in preds[batch]:
                # Point 2: Handle empty ground truth for class
                ious[batch, class_id] = 1
                continue

            # Point 1: Exclude non-category labels from argmax
            target_class = preds[batch, :, class_id]
            target_class = torch.where(labels[batch] == class_id, target_class, torch.tensor(float("-inf")))
            pred_for_class = torch.argmax(target_class, dim=-1)

            intersection = ((pred_for_class == class_id) & (labels[batch] == class_id)).sum().float()
            union = ((pred_for_class == class_id) | (labels[batch] == class_id)).sum().float()

            # Calculate IoU, avoid division by zero
            ious[batch, class_id] = intersection / union if union != 0 else 1

    # Point 7: Calculate weighted IoU
    iou_weighted_sum = ious.sum(1)  # Sum over all classes
    iou_weights = (ious != 0).sum(1).float()  # Count non-zero IoUs for averaging
    miou = (iou_weighted_sum / iou_weights).mean()  # Mean IoU over the batch

    return miou


def eval_cls_seg(
    logits: Tensor, targets: Tensor, metrics: list[str] | None = None, prefix: str = ""
) -> dict[str, float | np.ndarray]:
    import torchmetrics.functional as M

    targets = targets.long()
    results = {}

    def calculate_metric(metric_name: str, average: Literal["micro", "macro", "weighted", "none"]) -> Tensor | None:
        avg_common = cast(Literal["micro", "macro", "weighted", "none"], average)
        avg_roc = cast(Literal["macro", "weighted", "none"], average)
        if metric_name.startswith("acc_"):
            return cast(
                Tensor, M.accuracy(logits, targets, task="multiclass", average=avg_common, num_classes=logits.size(-1))
            ).cpu()
        elif metric_name.startswith("iou_"):
            return cast(
                Tensor,
                M.jaccard_index(logits, targets, task="multiclass", average=avg_common, num_classes=logits.size(-1)),
            ).cpu()
        elif metric_name.startswith("f1_"):
            return cast(
                Tensor, M.f1_score(logits, targets, task="multiclass", average=avg_common, num_classes=logits.size(-1))
            ).cpu()
        elif metric_name.startswith("precision_"):
            return cast(
                Tensor, M.precision(logits, targets, task="multiclass", average=avg_common, num_classes=logits.size(-1))
            ).cpu()
        elif metric_name.startswith("recall_"):
            return cast(
                Tensor, M.recall(logits, targets, task="multiclass", average=avg_common, num_classes=logits.size(-1))
            ).cpu()
        elif metric_name.startswith("auroc_"):
            return cast(
                Tensor, M.auroc(logits, targets, task="multiclass", average=avg_roc, num_classes=logits.size(-1))
            ).cpu()
        elif metric_name.startswith("auprc_"):
            return cast(
                Tensor,
                M.average_precision(logits, targets, task="multiclass", average=avg_roc, num_classes=logits.size(-1)),
            ).cpu()
        elif metric_name == "ece":
            return cast(
                Tensor, M.calibration_error(logits, targets, task="multiclass", num_classes=logits.size(-1))
            ).cpu()
        else:
            return None

    # List of all possible metrics
    all_metrics = [
        "acc_micro",
        "acc_macro",
        "acc_weighted",
        "acc_per_class",
        "iou_micro",
        "iou_macro",
        "iou_weighted",
        "iou_per_class",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "f1_per_class",
        "precision_micro",
        "precision_macro",
        "precision_weighted",
        "precision_per_class",
        "recall_micro",
        "recall_macro",
        "recall_weighted",
        "recall_per_class",
        "auroc_macro",
        "auroc_weighted",
        "auroc_per_class",
        "auprc_macro",
        "auprc_weighted",
        "auprc_per_class",
        "ece",
    ]

    # Determine the metrics to be calculated
    metrics_to_calculate = metrics if metrics is not None else all_metrics

    for metric in metrics_to_calculate:
        metric = metric.removeprefix(prefix)
        if metric in all_metrics:
            average_type = "none" if "per_class" in metric else metric.split("_")[-1]
            result = calculate_metric(metric, cast(Literal["micro", "macro", "weighted", "none"], average_type))
            if result is not None:
                results[f"{prefix}{metric}"] = result.item() if "per_class" not in metric else result.numpy()

    return results


def overwrite_results(cfg: DictConfig, save_path: Path, set_overwrite: bool = True) -> bool:
    if not cfg.test.overwrite and save_path.exists():
        logger.info(f"Results already exist at {save_path}")
        if save_path.is_file():
            if save_path.suffix == ".pkl":
                eval_df = cast(Any, pd.read_pickle(save_path))
                eval_df_class = cast(Any, eval_df.groupby(by=["category name"]).mean(numeric_only=True))
                eval_df_class.loc["mean (macro)"] = eval_df_class.mean(numeric_only=True)
                eval_df_class.loc["mean (micro)"] = eval_df.mean(numeric_only=True)
                print(eval_df_class)
                if "epoch" in eval_df.attrs:
                    for k, v in eval_df.attrs["epoch"].items():
                        logger.info(f"{k.upper()}: {v}")
            elif save_path.suffix == ".txt":
                with open(save_path) as file:
                    for line in file:
                        print(line, end="")
            else:
                logger.warning(f"Results file {save_path} is not supported")
        response = input("Do you want to overwrite the results? [y/N]: ")
        if response.lower() == "y":
            cfg.test.overwrite = set_overwrite
            return True
        return False
    return True


def render_mesh(
    mesh: Trimesh,
    extrinsic: np.ndarray,
    size: int = 299,
    light: bool = True,
    intensity: float = 3.0,
    renderer: pyrender.OffscreenRenderer | None = None,
    camera: pyrender.Camera | None = None,
) -> np.ndarray:
    _renderer = renderer or pyrender.OffscreenRenderer(size, size)

    scene = pyrender.Scene(bg_color=None)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    _camera = camera or pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    scene.add(_camera, pose=inv_trafo(extrinsic))

    if light:
        direct_l = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        spot_l = pyrender.SpotLight(
            color=np.ones(3), intensity=intensity, innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6
        )
        point_l = pyrender.PointLight(color=np.ones(3), intensity=2 * intensity)
        scene.add(direct_l, pose=inv_trafo(extrinsic))
        scene.add(point_l, pose=inv_trafo(extrinsic))
        scene.add(spot_l, pose=inv_trafo(extrinsic))

    render_out = _renderer.render(
        scene,
        flags=(pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.OFFSCREEN)
        if light
        else (pyrender.RenderFlags.FLAT | pyrender.RenderFlags.OFFSCREEN),
    )
    render_pair = cast(tuple[np.ndarray, np.ndarray], render_out)
    color = render_pair[0]

    if not renderer:
        _renderer.delete()
    return color


def render_for_fid(
    mesh: Trimesh | Path,
    path: Path,
    size: int = 299,
    views: str | int = "dodecahedron",
    mkdirs: bool = False,
    fix_winding: bool = False,
):
    if isinstance(mesh, Path):
        trafo = np.eye(4)
        if mesh.with_suffix(".npy").exists():
            trafo = np.load(mesh.with_suffix(".npy"))
        mesh = Trimesh(*load_mesh(mesh), process=False)
        mesh.apply_transform(trafo)

    if not mesh.is_empty:
        if fix_winding:
            mesh_inv = mesh.copy()
            mesh_inv.invert()
            mesh = mesh + mesh_inv
        mesh = normalize_mesh(mesh, cube_or_sphere="sphere")

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    renderer = pyrender.OffscreenRenderer(size, size)

    if isinstance(views, int):
        points = 2 * get_points(views)
    elif views == "dodecahedron":
        points = 2.6 * trimesh.creation.icosphere(subdivisions=0).triangles.mean(axis=1)
    elif views == "icosphere":
        points = 2 * trimesh.creation.icosphere(subdivisions=0).vertices
    elif views in ["sdfstylegan", "3dilg", "3dshape2vecset"]:
        # From https://github.com/Zhengxinyang/SDF-StyleGAN/blob/main/utils/render/render.py#L10
        points = 2 * np.array(
            [
                [0.52573, 0.38197, 0.85065],
                [-0.20081, 0.61803, 0.85065],
                [-0.64984, 0.00000, 0.85065],
                [-0.20081, -0.61803, 0.85065],
                [0.52573, -0.38197, 0.85065],
                [0.85065, -0.61803, 0.20081],
                [1.0515, 0.00000, -0.20081],
                [0.85065, 0.61803, 0.20081],
                [0.32492, 1.00000, -0.20081],
                [-0.32492, 1.00000, 0.20081],
                [-0.85065, 0.61803, -0.20081],
                [-1.0515, 0.00000, 0.20081],
                [-0.85065, -0.61803, -0.20081],
                [-0.32492, -1.00000, 0.20081],
                [0.32492, -1.00000, -0.20081],
                [0.64984, 0.00000, -0.85065],
                [0.20081, 0.61803, -0.85065],
                [-0.52573, 0.38197, -0.85065],
                [-0.52573, -0.38197, -0.85065],
                [0.20081, -0.61803, -0.85065],
            ]
        )
        points = np.concatenate([points[::2], points[1::2]])
        if views in ["3dilg", "3dshape2vecset"]:
            points = points[:10]
    else:
        raise ValueError(f"Unknown view type '{views}'")

    if mkdirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    for i, point in enumerate(points):
        extrinsic = cast(np.ndarray, inv_trafo(look_at(point, np.zeros(3))))
        image = render_mesh(mesh, extrinsic, renderer=renderer, camera=camera)
        view_dir = path.parent / f"view_{i}"
        view_dir.mkdir(exist_ok=True)
        Image.fromarray(image).convert("L").save(view_dir / f"{path.stem}.png")

    renderer.delete()


class MeanAveragePrecision3D(Metric):
    """
    Class-agnostic COCO-style mAP/mAR for 3D masks.

    Expects per-sample dicts:
      - preds[i] = { "masks": (Pi, M) bool tensor, "scores": (Pi,) tensor }
      - targets[i] = { "masks": (Gi, M) bool tensor }
    Where masks are binarized internally (> 0) so inputs can be 0/1 or real-valued.
    """

    def __init__(
        self,
        iou_thresholds: tuple[float, ...] = tuple(torch.linspace(0.5, 0.95, 10).tolist()),
        max_detection_thresholds: tuple[int, ...] = (1, 10, 100),
        area_ranges: dict[str, tuple[float, float]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.add_state("iou_thresholds", default=torch.tensor(iou_thresholds))
        self.add_state("max_detection_thresholds", default=torch.tensor(max_detection_thresholds))
        if area_ranges is None:
            area_ranges = {
                "map_small": (0.0, 32.0**2),
                "map_medium": (32.0**2, 96.0**2),
                "map_large": (96.0**2, float("inf")),
            }
        self.area_ranges = area_ranges

        # per-image lists of tensors; torchmetrics will cat them
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("gt_areas", default=[], dist_reduce_fx="cat")
        self.add_state("pred_areas", default=[], dist_reduce_fx="cat")
        self.add_state("num_gts_per_sample", default=[], dist_reduce_fx="cat")
        # Store the matched GT area for each detection/IoU threshold (-1 for unmatched)
        self.add_state("matched_gt_areas", default=[], dist_reduce_fx="cat")
        self.add_state("num_preds_per_sample", default=[], dist_reduce_fx="cat")
        # self.update = torch.compile(self.update, mode="reduce-overhead")

    @torch.no_grad()
    def update(self, preds: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]):
        device = self.device
        max_detection_thresholds = cast(Tensor, self.max_detection_thresholds)
        iou_thresholds = cast(Tensor, self.iou_thresholds)
        max_k = int(max_detection_thresholds.max().item())

        thresholds = iou_thresholds.to(device)  # (T,)
        T = thresholds.numel()
        num_gts_per_sample = cast(list[Tensor], self.num_gts_per_sample)
        num_preds_per_sample = cast(list[Tensor], self.num_preds_per_sample)
        gt_areas = cast(list[Tensor], self.gt_areas)
        pred_areas = cast(list[Tensor], self.pred_areas)
        scores = cast(list[Tensor], self.scores)
        matched_gt_areas = cast(list[Tensor], self.matched_gt_areas)

        for p, t in zip(preds, targets, strict=False):
            pred_masks = p["masks"].to(device)
            pred_scores = p["scores"].to(device)
            gt_masks = t["masks"].to(device)

            # Top-K truncate (COCO style)
            if pred_scores.numel() > 0:
                topk_idx = torch.argsort(pred_scores, descending=True)[:max_k]
                pred_masks = pred_masks[topk_idx]
                pred_scores = pred_scores[topk_idx]

            num_preds, num_gts = pred_masks.shape[0], gt_masks.shape[0]
            num_gts_per_sample.append(torch.tensor([num_gts], device=device))
            num_preds_per_sample.append(torch.tensor([num_preds], device=device))

            if num_gts > 0:
                gt_areas.append(gt_masks.sum(dim=1).float())
            if num_preds > 0:
                pred_areas.append(pred_masks.sum(dim=1).float())
            else:
                pred_areas.append(torch.empty(0, device=device))

            scores.append(pred_scores)

            sample_matched_gt_areas = torch.full((num_preds, T), -1.0, dtype=torch.float, device=device)

            # Early exit if no preds or gts
            if num_preds == 0 or num_gts == 0:
                matched_gt_areas.append(sample_matched_gt_areas)
                continue

            # IoU matrix (P x G); chunk over flattened dimension to reduce peak mem
            P, M = pred_masks.shape
            G = gt_masks.shape[0]

            pred_f = pred_masks.float()
            gt_f = gt_masks.float()

            areas_p = pred_f.sum(dim=1)  # (P,)
            areas_g = gt_f.sum(dim=1)  # (G,)

            intersection = torch.zeros((P, G), dtype=torch.float32, device=device)
            chunk = max(1, min(M, 262144))
            for start in range(0, M, chunk):
                end = min(M, start + chunk)
                intersection += pred_f[:, start:end] @ gt_f[:, start:end].T

            union = areas_p[:, None] + areas_g[None, :] - intersection
            iou = intersection / (union + 1e-6)  # (P, G)

            # Drop predictions that can never match any threshold (all IoUs < min_threshold)
            min_thr = thresholds.min()
            pred_viable = (iou >= min_thr).any(dim=1)
            if pred_viable.count_nonzero().item() < P:
                # Mask out dead preds (kept unmatched)
                viable_idx = torch.nonzero(pred_viable, as_tuple=False).squeeze(1)
                iou_viable = iou[pred_viable]
                # We still need stable mapping back into sample_matched_gt_areas:
                # We'll iterate only viable preds but write into their original rows.
                sorted_score_indices = torch.argsort(pred_scores[pred_viable], descending=True)
                # Map sorted indices back to original pred indices
                sorted_score_indices = viable_idx[sorted_score_indices]
            else:
                iou_viable = iou
                sorted_score_indices = torch.argsort(pred_scores, descending=True)

            # Precompute comparison matrix for all thresholds to avoid per-loop comparisons
            # Shape: (T, P_viable, G)
            iou_ge = iou_viable.unsqueeze(0) >= thresholds.view(-1, 1, 1)

            neg_inf = iou.new_tensor(float("-inf"))

            # For each IoU threshold, greedy per-prediction matching in score order
            # (matching sets are independent per threshold, as before)
            for ti in range(T):
                gt_taken = torch.zeros(G, dtype=torch.bool, device=device)
                # Iterate predictions in descending score order (subset if filtered)
                for pred_idx in sorted_score_indices:
                    pred_idx_i = int(pred_idx.item())
                    # Row index into viability-reduced iou (if reduced) needs remap:
                    row_i = (
                        pred_idx_i
                        if pred_viable.all()
                        else int(torch.nonzero(viable_idx == pred_idx, as_tuple=False).item())
                    )

                    # Candidates ≥ threshold not already taken
                    cand = iou_ge[ti, row_i] & ~gt_taken
                    if not cand.any():
                        continue

                    # Mask out non-candidates for argmax
                    row = iou[pred_idx_i]
                    # Fast path: if only one candidate
                    if cand.count_nonzero().item() == 1:
                        best_gt_idx = torch.nonzero(cand, as_tuple=False).squeeze(1)[0]
                    else:
                        # Replace invalid with -inf (avoid allocation of masked tensor)
                        masked_row = row.masked_fill(~cand, neg_inf)
                        best_gt_idx = torch.argmax(masked_row)

                    # Record match
                    sample_matched_gt_areas[pred_idx_i, ti] = areas_g[best_gt_idx]
                    gt_taken[best_gt_idx] = True  # in-place, no one-hot alloc

            matched_gt_areas.append(sample_matched_gt_areas)

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
        device = self.device
        iou_thresholds = cast(Tensor, self.iou_thresholds)
        max_detection_thresholds = cast(Tensor, self.max_detection_thresholds)

        # Handle state that may be a list (pre-sync) or a Tensor (post-sync)
        def _cat_state(state, empty: Tensor) -> Tensor:
            if isinstance(state, list):
                return torch.cat(state) if len(state) > 0 else empty
            if torch.is_tensor(state):
                return state
            return empty

        scores = _cat_state(self.scores, torch.empty(0, device=device))
        matched_gt_areas = _cat_state(
            self.matched_gt_areas,
            torch.empty(0, len(iou_thresholds), dtype=torch.float, device=device),
        )
        gt_areas = _cat_state(self.gt_areas, torch.empty(0, device=device))
        pred_areas = _cat_state(self.pred_areas, torch.empty(0, device=device))
        num_gts_per_sample = _cat_state(self.num_gts_per_sample, torch.empty(0, dtype=torch.long, device=device))
        num_preds_per_sample = _cat_state(self.num_preds_per_sample, torch.empty(0, dtype=torch.long, device=device))

        num_total_gts = int(num_gts_per_sample.sum().item()) if num_gts_per_sample.numel() > 0 else 0

        # Shape consistency checks
        n_preds = int(scores.numel())
        if n_preds == 0 or num_total_gts == 0:
            return self._empty_results()

        if matched_gt_areas.shape[0] != n_preds:
            raise ValueError(
                f"Shape mismatch: matched_gt_areas.shape[0]={matched_gt_areas.shape[0]} vs scores={n_preds}"
            )
        if pred_areas.shape[0] != n_preds:
            raise ValueError(f"Shape mismatch: pred_areas.shape[0]={pred_areas.shape[0]} vs scores={n_preds}")
        if gt_areas.shape[0] != num_total_gts:
            raise ValueError(
                f"Shape mismatch: gt_areas.shape[0]={gt_areas.shape[0]} vs sum(num_gts_per_sample)={num_total_gts}"
            )

        sorted_score_indices = torch.argsort(scores, descending=True)

        # Build a boolean match matrix (n_preds x n_iou)
        is_match_matrix = matched_gt_areas >= 0

        # Overall mAP (no area filtering; detections already truncated to maxDets in update)
        aps = self._calculate_ap(sorted_score_indices, is_match_matrix, matched_gt_areas, num_positives=num_total_gts)
        results = self._format_results(aps)

        # Area-range mAP (filter both dets and GTs)
        for name, (min_area, max_area) in self.area_ranges.items():
            gt_mask = (gt_areas >= min_area) & (gt_areas < max_area)
            num_gts_in_range = int(gt_mask.sum().item())
            if num_gts_in_range == 0:
                results[name] = 0.0
                continue

            det_mask = (pred_areas >= min_area) & (pred_areas < max_area)
            if det_mask.numel() != n_preds:
                raise ValueError(f"det_mask length {det_mask.numel()} does not match number of detections {n_preds}.")

            aps_area = self._calculate_ap(
                sorted_score_indices,
                is_match_matrix,
                matched_gt_areas,
                num_positives=num_gts_in_range,
                det_mask=det_mask,
                area_range=(min_area, max_area),
            )
            results[name] = aps_area.mean().item()

        # Mean Average Recall (global, ignore images with 0 GTs)
        recalls = []
        # Reconstruct per-image num_preds from self.scores
        pred_counts = [int(x.item()) for x in num_preds_per_sample]
        pred_cum = [0]
        for c in pred_counts:
            pred_cum.append(pred_cum[-1] + c)

        # Reconstruct per-image GT spans for area MAR
        gt_counts = [int(x.item()) for x in num_gts_per_sample]
        gt_cum = [0]
        for c in gt_counts:
            gt_cum.append(gt_cum[-1] + c)

        for k in max_detection_thresholds:
            k = int(k.item())
            recalls_k = []
            for i in range(len(iou_thresholds)):
                tp_total = 0
                gt_total = 0
                for img_idx, num_gts in enumerate(num_gts_per_sample):
                    num_gts_i = int(num_gts.item())
                    if num_gts_i == 0:
                        continue  # ignore images with no GTs
                    s0, s1 = pred_cum[img_idx], pred_cum[img_idx + 1]
                    if s1 == s0:
                        gt_total += num_gts_i
                        continue
                    # top-k in this image
                    img_scores = scores[s0:s1]
                    topk_local = torch.argsort(img_scores, descending=True)[:k]
                    topk_idx = s0 + topk_local
                    # count matches
                    is_match_topk = is_match_matrix[topk_idx, i]
                    tp_total += int(is_match_topk.sum().item())
                    gt_total += num_gts_i
                recalls_k.append(
                    torch.tensor(0.0, device=device)
                    if gt_total == 0
                    else torch.tensor(tp_total / gt_total, device=device)
                )
            recalls.append(torch.stack(recalls_k).mean())
        recalls = torch.stack(recalls)

        for i, k in enumerate(max_detection_thresholds):
            results[f"mar_{int(k.item())}"] = recalls[i].item()

        # Area-range MAR at maxDets (mirror torchvision: mar_small/medium/large with maxDets=100)
        max_k = int(max_detection_thresholds.max().item())
        mar_area_results: dict[str, float] = {}
        for name, (min_area, max_area) in self.area_ranges.items():
            # map_* -> mar_* key
            mar_key = f"mar_{name.split('map_', 1)[1]}" if name.startswith("map_") else f"mar_{name}"

            recalls_k_area = []
            for i in range(len(iou_thresholds)):
                tp_total = 0
                gt_total = 0
                for img_idx, num_gts in enumerate(num_gts_per_sample):
                    num_gts_i = int(num_gts.item())
                    if num_gts_i == 0:
                        continue
                    # GT denom in-range for this image
                    g0, g1 = gt_cum[img_idx], gt_cum[img_idx + 1]
                    gt_in_range_i = ((gt_areas[g0:g1] >= min_area) & (gt_areas[g0:g1] < max_area)).sum().item()
                    if gt_in_range_i == 0:
                        continue  # no positives in this image for this area

                    # Select top-k detections in this image restricted to det area-range
                    s0, s1 = pred_cum[img_idx], pred_cum[img_idx + 1]
                    if s1 == s0:
                        # no preds
                        gt_total += int(gt_in_range_i)
                        continue
                    img_scores = scores[s0:s1]
                    det_in_range_local = (pred_areas[s0:s1] >= min_area) & (pred_areas[s0:s1] < max_area)
                    if det_in_range_local.any():
                        # map back to absolute indices
                        local_kept = torch.nonzero(det_in_range_local, as_tuple=False).squeeze(1)
                        img_scores_kept = img_scores[local_kept]
                        topk_local = local_kept[torch.argsort(img_scores_kept, descending=True)[:max_k]]
                        topk_idx = s0 + topk_local
                        # TP: matched and matched-GT is in the same area range
                        mgt = matched_gt_areas[topk_idx, i]
                        is_match_topk = mgt >= 0
                        is_match_gt_in_area = (mgt >= min_area) & (mgt < max_area)
                        tp_total += int((is_match_topk & is_match_gt_in_area).sum().item())

                    gt_total += int(gt_in_range_i)

                recalls_k_area.append(
                    torch.tensor(0.0, device=device)
                    if gt_total == 0
                    else torch.tensor(tp_total / gt_total, device=device)
                )
            mar_area_results[mar_key] = torch.stack(recalls_k_area).mean().item()

        results.update(mar_area_results)

        return results

    def _calculate_ap(
        self,
        sorted_score_indices: Tensor,
        is_match_matrix: Tensor,
        matched_gt_areas: Tensor,
        num_positives: int,
        det_mask: Tensor | None = None,
        area_range: tuple[float, float] | None = None,
    ) -> Tensor:
        """
        Compute AP across IoU thresholds given:
          - global score-sorted detection indices
          - per-detection match flags
          - optional detection area filter (det_mask)
          - optional matched-GT area filter (area_range)
        num_positives is the denominator (#GTs considered), already area-filtered if area_range is provided.
        """
        aps = []
        for i in range(is_match_matrix.shape[1]):
            idx = sorted_score_indices

            # Filter detections by area range if provided
            if det_mask is not None and det_mask.numel() > 0:
                if det_mask.numel() != is_match_matrix.shape[0]:
                    raise ValueError(
                        f"det_mask length {det_mask.numel()} does not match number of detections {is_match_matrix.shape[0]}"
                    )
                keep = det_mask[idx]
                idx = idx[keep]

            is_match_sorted = is_match_matrix[idx, i]

            if area_range is not None:
                min_area, max_area = area_range
                mgt_sorted = matched_gt_areas[idx, i]  # -1 for unmatched
                in_range_gt = (mgt_sorted >= min_area) & (mgt_sorted < max_area)
                true_positives = is_match_sorted & in_range_gt
                denom = max(1, int(num_positives))
            else:
                true_positives = is_match_sorted
                denom = max(1, int(num_positives))

            tp = torch.cumsum(true_positives, dim=0)
            fp = torch.cumsum(~true_positives, dim=0)

            recall = tp / (denom + 1e-6)
            precision = tp / (tp + fp + 1e-6)

            ap = self._compute_interpolated_ap(recall, precision)
            aps.append(ap)
        return torch.stack(aps)

    def _compute_interpolated_ap(self, recall: Tensor, precision: Tensor) -> Tensor:
        recall_thresholds = torch.linspace(0, 1, 101, device=self.device)
        precision = torch.cat([torch.tensor([1.0], device=self.device), precision])
        recall = torch.cat([torch.tensor([0.0], device=self.device), recall])
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = torch.maximum(precision[i], precision[i + 1])
        indices = torch.clamp(torch.searchsorted(recall, recall_thresholds, right=True) - 1, 0)
        return precision[indices].mean()

    def _empty_results(self) -> dict[str, float]:
        iou_thresholds = cast(Tensor, self.iou_thresholds)
        max_detection_thresholds = cast(Tensor, self.max_detection_thresholds)
        results = self._format_results(torch.zeros_like(iou_thresholds))
        for k in max_detection_thresholds:
            results[f"mar_{int(k.item())}"] = 0.0
        # Area mAP keys
        for name in self.area_ranges:
            results[name] = 0.0
        # Area MAR keys (at maxDets)
        for name in self.area_ranges:
            mar_key = f"mar_{name.split('map_', 1)[1]}" if name.startswith("map_") else f"mar_{name}"
            results[mar_key] = 0.0
        return results

    def _format_results(self, aps: Tensor) -> dict[str, float]:
        results = {"map": aps.mean().item()}

        # Make thresholds 1D and align length/device with aps for DDP-safety
        iou_thresholds_state = cast(Tensor, self.iou_thresholds)
        iou_thresholds = iou_thresholds_state.detach().to(aps.device).reshape(-1)[: aps.shape[0]]

        for thr, key in [(0.5, "map_50"), (0.75, "map_75")]:
            mask = torch.isclose(iou_thresholds, iou_thresholds.new_tensor(thr))
            results[key] = aps[mask].mean().item() if mask.any() else 0.0

        return results


class PanopticQuality3D(Metric):
    """
    Fast Panoptic Quality (PQ) for class-agnostic 3D/voxel masks with optional size bins.

    Performance focus:
      - Single greedy 1:1 matching (no second pass) with two selectable strategies:
          * 'argmax': iterative global argmax (great when P,G <= ~150; avoids sorting)
          * 'candidates': threshold filter + sort (better only if threshold high AND P,G large)
        'auto' picks the cheaper heuristic per batch.

    Args:
        iou_thresh: IoU threshold for a match.
        size_bin_edges: Ordered dict {label: lower_edge}. Final ∞ appended if missing.
        track_bin_iou: Whether to compute SQ per bin (adds scatter cost).
        match_strategy: 'auto' | 'argmax' | 'candidates'.
        candidate_sort_limit: Hard cap to fall back to argmax if candidate count exceeds this.
    """

    def __init__(
        self,
        iou_thresh: float = 0.5,
        size_bin_edges: dict[str, float] | None = None,
        track_bin_iou: bool = True,
        match_strategy: str = "auto",
        candidate_sort_limit: int = 50_000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.iou_thresh = float(iou_thresh)
        self.track_bin_iou = track_bin_iou
        self.match_strategy = match_strategy
        self.candidate_sort_limit = int(candidate_sort_limit)

        if size_bin_edges is None:
            size_bin_edges = {"small": 0.0, "medium": 32.0**2, "large": 96.0**2}
        self.size_bin_edges_dict = size_bin_edges

        edges = torch.as_tensor(list(size_bin_edges.values()), dtype=torch.float32)
        if torch.any(edges[1:] < edges[:-1]):
            raise ValueError("size_bin_edges must be ascending.")
        if not torch.isinf(edges[-1]):
            edges = torch.cat([edges, torch.tensor([float("inf")], dtype=edges.dtype)])
        if edges.numel() < 2:
            raise ValueError("Need ≥2 edges.")
        self.register_buffer("size_bin_edges", edges, persistent=False)
        num_bins = edges.numel() - 1

        # Global
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("iou_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # Bins
        self.add_state("tp_bins", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("fp_bins", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("fn_bins", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        if track_bin_iou:
            self.add_state("iou_sum_bins", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.update = torch.compile(self.update)

    def _bucketize(self, areas: Tensor) -> Tensor:
        if areas.numel() == 0:
            return areas.new_empty((0,), dtype=torch.long)
        size_bin_edges = cast(Tensor, self.size_bin_edges)
        idx = torch.bucketize(areas, size_bin_edges, right=False) - 1
        return idx.clamp_min(0)

    def _match_argmax(self, iou: Tensor, thr: float) -> tuple[Tensor, Tensor, Tensor]:
        """
        Greedy argmax without candidate list sorting.
        O(K * P * G) but very fast for moderate sizes due to simple kernels.
        """
        P, G = iou.shape
        work = iou  # we are allowed to modify a copy outside
        # Make a copy so caller retains original (needed for bins & IoU sum)
        work = work.clone()
        neg_inf = work.new_full((), float("-inf"))
        matched_pi = []
        matched_gi = []
        matched_vals = []
        K = min(P, G)
        flat = work.view(-1)
        for _ in range(K):
            best_val, flat_idx = flat.max(dim=0)
            if best_val < thr:
                break
            pi = flat_idx // G
            gi = flat_idx % G
            matched_pi.append(pi)
            matched_gi.append(gi)
            matched_vals.append(best_val)
            # Invalidate row & column
            work[pi, :] = neg_inf
            work[:, gi] = neg_inf
        if not matched_pi:
            empty_long = iou.new_empty((0,), dtype=torch.long)
            return empty_long, empty_long, iou.new_empty((0,))
        return torch.stack(matched_pi), torch.stack(matched_gi), torch.stack(matched_vals)

    def _match_candidates(self, iou: Tensor, thr: float) -> tuple[Tensor, Tensor, Tensor]:
        """
        Candidate threshold + sort approach.
        """
        mask = iou >= thr
        if not mask.any():
            empty_long = iou.new_empty((0,), dtype=torch.long)
            return empty_long, empty_long, iou.new_empty((0,))
        pi_idx, gi_idx = mask.nonzero(as_tuple=True)
        vals = iou[pi_idx, gi_idx]
        C = vals.numel()
        # If huge candidate set, fall back to argmax (sorting too expensive)
        if C > self.candidate_sort_limit:
            return self._match_argmax(iou, thr)
        order = torch.argsort(vals, descending=True)
        P, G = iou.shape
        taken_p = torch.zeros(P, dtype=torch.bool, device=iou.device)
        taken_g = torch.zeros(G, dtype=torch.bool, device=iou.device)
        matched_pi = []
        matched_gi = []
        matched_vals = []
        for o in order:
            p = pi_idx[o]
            g = gi_idx[o]
            if taken_p[p] or taken_g[g]:
                continue
            taken_p[p] = True
            taken_g[g] = True
            matched_pi.append(p)
            matched_gi.append(g)
            matched_vals.append(vals[o])
        if not matched_pi:
            empty_long = iou.new_empty((0,), dtype=torch.long)
            return empty_long, empty_long, iou.new_empty((0,))
        return torch.stack(matched_pi), torch.stack(matched_gi), torch.stack(matched_vals)

    def _choose_strategy(self, P: int, G: int, thr: float) -> str:
        if self.match_strategy != "auto":
            return self.match_strategy
        # Heuristic: argmax usually faster when matrix small/moderate or threshold low (many candidates).
        size = P * G
        if size <= 80 * 80:
            return "argmax"
        if thr < 0.45:  # low threshold => many candidates => prefer argmax
            return "argmax"
        return "candidates"

    @torch.no_grad()
    def update(self, preds: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]):
        tp_state = cast(Tensor, self.tp)
        fp_state = cast(Tensor, self.fp)
        fn_state = cast(Tensor, self.fn)
        iou_sum_state = cast(Tensor, self.iou_sum)
        tp_bins_state = cast(Tensor, self.tp_bins)
        fp_bins_state = cast(Tensor, self.fp_bins)
        fn_bins_state = cast(Tensor, self.fn_bins)
        iou_sum_bins_state = cast(Tensor, self.iou_sum_bins) if self.track_bin_iou else None
        size_bin_edges = cast(Tensor, self.size_bin_edges)
        device = tp_state.device
        thr = float(self.iou_thresh)
        bins_active = bool(self.size_bin_edges_dict) and self.track_bin_iou
        num_bins = size_bin_edges.numel() - 1 if bins_active else 0

        for p, t in zip(preds, targets, strict=False):
            pred_masks = p["masks"].to(device).float()
            gt_masks = t["masks"].to(device).float()
            P, _M = pred_masks.shape
            G = gt_masks.shape[0]

            # Empty cases
            if P == 0 and G == 0:
                continue
            if P == 0:
                self.fn += float(G)
                if bins_active and G > 0:
                    gt_bins = self._bucketize(gt_masks.sum(dim=1))
                    self.fn_bins += torch.bincount(gt_bins, minlength=num_bins)
                continue
            if G == 0:
                self.fp += float(P)
                if bins_active and P > 0:
                    pred_bins = self._bucketize(pred_masks.sum(dim=1))
                    self.fp_bins += torch.bincount(pred_bins, minlength=num_bins)
                continue

            # Areas
            areas_p = pred_masks.sum(dim=1)  # (P,)
            areas_g = gt_masks.sum(dim=1)  # (G,)
            pm = pred_masks
            gm = gt_masks
            intersection = pm @ gm.T  # (P,G)

            union = areas_p[:, None] + areas_g[None, :] - intersection
            iou = intersection / (union + 1e-6)

            # Matching
            strategy = self._choose_strategy(P, G, thr)
            if strategy == "argmax":
                matched_pi, matched_gi, matched_vals = self._match_argmax(iou, thr)
            else:
                matched_pi, matched_gi, matched_vals = self._match_candidates(iou, thr)

            tp_local = matched_vals.numel()
            fp_local = P - tp_local
            fn_local = G - tp_local

            tp_state += float(tp_local)
            fp_state += float(fp_local)
            fn_state += float(fn_local)
            iou_sum_state += matched_vals.sum()

            if bins_active:
                gt_bins = self._bucketize(areas_g)
                pred_bins = self._bucketize(areas_p)

                if tp_local > 0:
                    matched_gt_bins = gt_bins[matched_gi]
                    tp_bins_state += torch.bincount(matched_gt_bins, minlength=num_bins)
                    if self.track_bin_iou:
                        assert iou_sum_bins_state is not None
                        iou_sum_bins_state += torch.zeros_like(iou_sum_bins_state).scatter_add(
                            0, matched_gt_bins, matched_vals
                        )

                if fn_local > 0:
                    unmatched_g = torch.ones(G, dtype=torch.bool, device=device)
                    if tp_local > 0:
                        unmatched_g[matched_gi] = False
                    fn_bins_state += torch.bincount(gt_bins[unmatched_g], minlength=num_bins)

                if fp_local > 0:
                    unmatched_p = torch.ones(P, dtype=torch.bool, device=device)
                    if tp_local > 0:
                        unmatched_p[matched_pi] = False
                    fp_bins_state += torch.bincount(pred_bins[unmatched_p], minlength=num_bins)

        self.tp = tp_state
        self.fp = fp_state
        self.fn = fn_state
        self.iou_sum = iou_sum_state
        self.tp_bins = tp_bins_state
        self.fp_bins = fp_bins_state
        self.fn_bins = fn_bins_state
        if iou_sum_bins_state is not None:
            self.iou_sum_bins = iou_sum_bins_state

    @torch.no_grad()
    def compute(self) -> dict[str, float]:
        tp = float(cast(Tensor, self.tp).item())
        fp = float(cast(Tensor, self.fp).item())
        fn = float(cast(Tensor, self.fn).item())
        iou_sum = float(cast(Tensor, self.iou_sum).item())
        denom = tp + 0.5 * fp + 0.5 * fn
        if denom == 0:
            return {"pq": 0.0, "rq": 0.0, "sq": 0.0}

        rq = tp / denom
        sq = (iou_sum / tp) if tp > 0 else 0.0
        pq = iou_sum / denom
        results = {"pq": pq, "rq": rq, "sq": sq}

        if self.size_bin_edges_dict and self.track_bin_iou:
            for b, tag in enumerate(self.size_bin_edges_dict.keys()):
                tp_b = self.tp_bins[b].item()
                fp_b = self.fp_bins[b].item()
                fn_b = self.fn_bins[b].item()
                denom_b = tp_b + 0.5 * fp_b + 0.5 * fn_b
                if denom_b == 0:
                    pq_b = rq_b = sq_b = 0.0
                else:
                    rq_b = tp_b / denom_b
                    if tp_b > 0:
                        if self.track_bin_iou:
                            sq_b = self.iou_sum_bins[b].item() / tp_b
                            pq_b = self.iou_sum_bins[b].item() / denom_b
                        else:
                            sq_b = 0.0
                            pq_b = rq_b * sq_b
                    else:
                        sq_b = 0.0
                        pq_b = 0.0
                results[f"pq_{tag}"] = pq_b
                results[f"rq_{tag}"] = rq_b
                results[f"sq_{tag}"] = sq_b
        return results
