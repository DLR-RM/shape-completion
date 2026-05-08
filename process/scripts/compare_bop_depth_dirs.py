from __future__ import annotations

import csv
import json
import logging
import math
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


METRIC_COLUMNS = [
    "dataset",
    "split",
    "scene_id",
    "frame_id",
    "num_pixels",
    "sensor_valid_pixels",
    "pred_valid_pixels",
    "common_valid_pixels",
    "sensor_valid_ratio",
    "pred_valid_ratio",
    "common_valid_ratio",
    "pred_over_sensor_valid_ratio",
    "missing_sensor_valid_ratio",
    "hallucinated_valid_ratio",
    "mae_m",
    "rmse_m",
    "median_abs_error_m",
    "bias_m",
    "abs_rel",
    "delta1",
    "pcd_mae_m",
    "pcd_rmse_m",
    "median_scale_to_sensor",
]


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {path}: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to read JSON file {path}: {e}") from e


def _load_bop_depth_m(path: Path, depth_scale: float) -> np.ndarray:
    raw = np.asarray(Image.open(path), dtype=np.float32)
    return raw * depth_scale / 1000.0


def _scene_dirs(split_dir: Path, scene_ids: list[int] | None) -> list[Path]:
    scene_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and (p / "scene_camera.json").exists())
    if scene_ids is not None:
        wanted = {f"{scene_id:06d}" for scene_id in scene_ids}
        scene_dirs = [p for p in scene_dirs if p.name in wanted]
    if not scene_dirs:
        raise FileNotFoundError(f"No BOP scene directories found in {split_dir}.")
    return scene_dirs


def _frame_ids(scene_dir: Path, pred_depth_dir: str, frame_ids: list[int] | None) -> list[str]:
    pred_dir = scene_dir / pred_depth_dir
    if not pred_dir.exists():
        logger.warning("Predicted depth directory is missing: %s", pred_dir)
        return []
    available = sorted(p.stem for p in pred_dir.glob("*.png"))
    if frame_ids is None:
        return available
    wanted = {f"{frame_id:06d}" for frame_id in frame_ids}
    return [frame_id for frame_id in available if frame_id in wanted]


def _safe_float(value: float | np.floating[Any]) -> float:
    value_f = float(value)
    if math.isfinite(value_f):
        return value_f
    return float("nan")


def _metric_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": _safe_float(np.mean(finite)),
        "std": _safe_float(np.std(finite)),
        "min": _safe_float(np.min(finite)),
        "max": _safe_float(np.max(finite)),
    }


def _pcd_error_m(diff: np.ndarray, intrinsic: np.ndarray, common: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(diff.shape, dtype=np.float32)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    ray_norm = np.sqrt(((xx - cx) / fx) ** 2 + ((yy - cy) / fy) ** 2 + 1.0)
    return np.abs(diff[common]) * ray_norm[common]


def _frame_metrics(
    *,
    dataset: str,
    split: str,
    scene_id: int,
    frame_id: int,
    sensor: np.ndarray,
    pred: np.ndarray,
    intrinsic: np.ndarray,
) -> dict[str, float | int | str]:
    sensor_valid = np.isfinite(sensor) & (sensor > 0.0)
    pred_valid = np.isfinite(pred) & (pred > 0.0)
    common = sensor_valid & pred_valid
    hallucinated = pred_valid & ~sensor_valid
    missing = sensor_valid & ~pred_valid
    num_pixels = int(sensor.size)
    sensor_count = int(sensor_valid.sum())
    pred_count = int(pred_valid.sum())
    common_count = int(common.sum())

    row: dict[str, float | int | str] = {
        "dataset": dataset,
        "split": split,
        "scene_id": scene_id,
        "frame_id": frame_id,
        "num_pixels": num_pixels,
        "sensor_valid_pixels": sensor_count,
        "pred_valid_pixels": pred_count,
        "common_valid_pixels": common_count,
        "sensor_valid_ratio": sensor_count / num_pixels,
        "pred_valid_ratio": pred_count / num_pixels,
        "common_valid_ratio": common_count / num_pixels,
        "pred_over_sensor_valid_ratio": common_count / sensor_count if sensor_count else float("nan"),
        "missing_sensor_valid_ratio": int(missing.sum()) / sensor_count if sensor_count else float("nan"),
        "hallucinated_valid_ratio": int(hallucinated.sum()) / num_pixels,
    }

    if common_count == 0:
        for key in (
            "mae_m",
            "rmse_m",
            "median_abs_error_m",
            "bias_m",
            "abs_rel",
            "delta1",
            "pcd_mae_m",
            "pcd_rmse_m",
            "median_scale_to_sensor",
        ):
            row[key] = float("nan")
        return row

    diff = pred[common] - sensor[common]
    abs_error = np.abs(diff)
    rel = abs_error / np.maximum(sensor[common], 1e-8)
    ratio = np.maximum(pred[common] / np.maximum(sensor[common], 1e-8), sensor[common] / np.maximum(pred[common], 1e-8))
    pcd_error = _pcd_error_m(pred - sensor, intrinsic, common)

    row.update(
        {
            "mae_m": _safe_float(np.mean(abs_error)),
            "rmse_m": _safe_float(np.sqrt(np.mean(diff**2))),
            "median_abs_error_m": _safe_float(np.median(abs_error)),
            "bias_m": _safe_float(np.mean(diff)),
            "abs_rel": _safe_float(np.mean(rel)),
            "delta1": _safe_float(np.mean(ratio < 1.25)),
            "pcd_mae_m": _safe_float(np.mean(pcd_error)),
            "pcd_rmse_m": _safe_float(np.sqrt(np.mean(pcd_error**2))),
            "median_scale_to_sensor": _safe_float(np.median(sensor[common] / np.maximum(pred[common], 1e-8))),
        }
    )
    return row


def _normalize(values: np.ndarray, valid: np.ndarray, max_value: float) -> np.ndarray:
    norm = np.zeros(values.shape, dtype=np.float32)
    if max_value > 0:
        norm[valid] = np.clip(values[valid] / max_value, 0.0, 1.0)
    return norm


def _gray_image(values: np.ndarray, valid: np.ndarray, max_value: float) -> np.ndarray:
    norm = _normalize(values, valid, max_value)
    image = np.repeat((norm[..., None] * 255.0).astype(np.uint8), 3, axis=2)
    image[~valid] = 0
    return image


def _error_image(abs_error: np.ndarray, valid: np.ndarray, max_value: float) -> np.ndarray:
    norm = _normalize(abs_error, valid, max_value)
    image = np.zeros((*abs_error.shape, 3), dtype=np.uint8)
    image[..., 0] = (norm * 255.0).astype(np.uint8)
    image[..., 1] = (np.sqrt(norm) * 160.0).astype(np.uint8)
    return image


def _add_label(image: Image.Image, label: str) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + 18), "black")
    out.paste(image, (0, 18))
    draw = ImageDraw.Draw(out)
    draw.text((4, 2), label, fill="white")
    return out


def _write_visualization(
    *,
    path: Path,
    rgb_path: Path,
    sensor: np.ndarray,
    pred: np.ndarray,
) -> None:
    rgb = Image.open(rgb_path).convert("RGB").resize((sensor.shape[1], sensor.shape[0]))
    sensor_valid = np.isfinite(sensor) & (sensor > 0.0)
    pred_valid = np.isfinite(pred) & (pred > 0.0)
    common = sensor_valid & pred_valid
    depth_max = float(np.percentile(np.concatenate([sensor[sensor_valid], pred[pred_valid]]), 99)) if (sensor_valid.any() or pred_valid.any()) else 1.0
    abs_error = np.zeros(sensor.shape, dtype=np.float32)
    abs_error[common] = np.abs(pred[common] - sensor[common])
    error_max = float(np.percentile(abs_error[common], 95)) if common.any() else 1.0
    if error_max <= 0.0:
        error_max = 1.0

    panels = [
        _add_label(rgb, "rgb"),
        _add_label(Image.fromarray(_gray_image(sensor, sensor_valid, depth_max)), "sensor depth"),
        _add_label(Image.fromarray(_gray_image(pred, pred_valid, depth_max)), "pred depth"),
        _add_label(Image.fromarray(_error_image(abs_error, common, error_max)), "abs error"),
    ]
    out = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "black")
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in METRIC_COLUMNS:
        values = []
        for row in rows:
            value = row[key]
            if isinstance(value, int | float):
                values.append(float(value))
        if values:
            summary[key] = _metric_stats(np.asarray(values, dtype=np.float64))
    return summary


def run(args: Namespace) -> dict[str, Any]:
    split_dir = Path(args.bop_root) / args.dataset / args.split
    out_dir = Path(args.out_dir)
    rows: list[dict[str, float | int | str]] = []
    visualizations_written = 0
    max_visualizations = int(args.num_visualizations)
    missing_pred_scenes = 0

    for scene_dir in _scene_dirs(split_dir, args.scene_ids):
        camera = _load_json(scene_dir / "scene_camera.json")
        scene_frame_ids = _frame_ids(scene_dir, args.pred_depth_dir, args.frame_ids)
        if not scene_frame_ids:
            missing_pred_scenes += 1
        for frame_id in scene_frame_ids:
            frame_camera = camera[str(int(frame_id))]
            depth_scale = float(frame_camera.get("depth_scale", 1.0))
            pred_depth_scale = depth_scale if args.pred_depth_scale is None else float(args.pred_depth_scale)
            intrinsic = np.asarray(frame_camera["cam_K"], dtype=np.float32).reshape(3, 3)
            sensor_path = scene_dir / args.sensor_depth_dir / f"{frame_id}.png"
            pred_path = scene_dir / args.pred_depth_dir / f"{frame_id}.png"
            rgb_path = scene_dir / "rgb" / f"{frame_id}.png"
            sensor = _load_bop_depth_m(sensor_path, depth_scale)
            pred = _load_bop_depth_m(pred_path, pred_depth_scale)
            if sensor.shape != pred.shape:
                raise ValueError(f"Depth shape mismatch for {pred_path}: sensor={sensor.shape}, pred={pred.shape}")

            row = _frame_metrics(
                dataset=args.dataset,
                split=args.split,
                scene_id=int(scene_dir.name),
                frame_id=int(frame_id),
                sensor=sensor,
                pred=pred,
                intrinsic=intrinsic,
            )
            rows.append(row)

            if visualizations_written < max_visualizations and rgb_path.exists():
                vis_path = (
                    out_dir
                    / "visualizations"
                    / f"{args.dataset}_{args.split}_scene{int(scene_dir.name):06d}_frame{int(frame_id):06d}.png"
                )
                _write_visualization(path=vis_path, rgb_path=rgb_path, sensor=sensor, pred=pred)
                visualizations_written += 1

    if not rows:
        raise FileNotFoundError(f"No predicted depth frames found under {split_dir}/*/{args.pred_depth_dir}.")

    summary = _summarize(rows)
    csv_path = out_dir / f"{args.dataset}_{args.split}_{args.pred_depth_dir}_metrics.csv"
    summary_path = out_dir / f"{args.dataset}_{args.split}_{args.pred_depth_dir}_summary.json"
    _write_csv(csv_path, rows)
    summary_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "pred_depth_dir": args.pred_depth_dir,
                "sensor_depth_dir": args.sensor_depth_dir,
                "pred_depth_scale": args.pred_depth_scale,
                "num_frames": len(rows),
                "missing_pred_scenes": missing_pred_scenes,
                "summary": summary,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    return {
        "num_frames": len(rows),
        "missing_pred_scenes": missing_pred_scenes,
        "summary": summary,
        "csv_path": csv_path,
        "summary_path": summary_path,
    }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Compare a generated BOP depth directory against the original sensor depth.")
    parser.add_argument("--bop-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scene-ids", type=int, nargs="*", default=None)
    parser.add_argument("--frame-ids", type=int, nargs="*", default=None)
    parser.add_argument("--sensor-depth-dir", default="depth")
    parser.add_argument("--pred-depth-dir", default="depth_da3")
    parser.add_argument(
        "--pred-depth-scale",
        type=float,
        default=None,
        help="Override the predicted depth raw-unit scale. Defaults to the per-frame BOP scene depth_scale.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("bop_depth_compare"))
    parser.add_argument("--num-visualizations", type=int, default=12)
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps({"num_frames": summary["num_frames"], "summary_path": str(summary["summary_path"])}, indent=2))


if __name__ == "__main__":
    main()
