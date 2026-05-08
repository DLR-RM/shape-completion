from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PredictDepths = Callable[[list[Path], list[np.ndarray]], list[np.ndarray]]
logger = logging.getLogger(__name__)


def depth_m_to_bop_uint16(depth_m: np.ndarray, depth_scale: float = 1.0) -> np.ndarray:
    if depth_scale <= 0.0:
        raise ValueError(f"depth_scale must be positive, got {depth_scale}")
    raw_depth = np.rint(np.asarray(depth_m, dtype=np.float32) * 1000.0 / depth_scale)
    valid = np.isfinite(raw_depth) & (raw_depth > 0)
    encoded = np.zeros(raw_depth.shape, dtype=np.uint16)
    if valid.any() and np.any(raw_depth[valid] > np.iinfo(np.uint16).max):
        max_depth_m = float(depth_scale * np.iinfo(np.uint16).max / 1000.0)
        logger.warning(
            "Clipping generated BOP depth values above uint16 range for depth_scale=%s (max %.3f m).",
            depth_scale,
            max_depth_m,
        )
    encoded[valid] = np.clip(raw_depth[valid], 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return encoded


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {path}: {e}") from e
    except OSError as e:
        raise OSError(f"Failed to read JSON file {path}: {e}") from e


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _scene_dirs(split_dir: Path, scene_ids: list[int] | None) -> list[Path]:
    scene_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and (p / "rgb").exists())
    if scene_ids is not None:
        wanted = {f"{scene_id:06d}" for scene_id in scene_ids}
        scene_dirs = [p for p in scene_dirs if p.name in wanted]
    if not scene_dirs:
        raise FileNotFoundError(f"No BOP scene directories with rgb/ found in {split_dir}.")
    return scene_dirs


def _frame_ids(scene_dir: Path) -> list[str]:
    return sorted(p.stem for p in (scene_dir / "rgb").glob("*.png"))


def _select_frame_ids(scene_dir: Path, frame_ids: list[int] | None) -> list[str]:
    available = _frame_ids(scene_dir)
    if frame_ids is None:
        return available
    wanted = {f"{frame_id:06d}" for frame_id in frame_ids}
    return [frame_id for frame_id in available if frame_id in wanted]


def _frame_camera(camera: dict[str, Any], scene_dir: Path, frame_id: str) -> dict[str, Any]:
    frame_key = str(int(frame_id))
    try:
        frame_camera = camera[frame_key]
    except KeyError as e:
        raise KeyError(f"Missing frame {frame_key} in {scene_dir / 'scene_camera.json'}") from e
    return frame_camera


def _intrinsic_for(camera: dict[str, Any], scene_dir: Path, frame_id: str) -> np.ndarray:
    return np.asarray(_frame_camera(camera, scene_dir, frame_id)["cam_K"], dtype=np.float32).reshape(3, 3)


def _sensor_depth_m(scene_dir: Path, frame_id: str, camera_data: dict[str, Any]) -> np.ndarray:
    depth_scale = float(camera_data.get("depth_scale", 1.0))
    raw = np.asarray(Image.open(scene_dir / "depth" / f"{frame_id}.png"), dtype=np.float32)
    return raw * depth_scale / 1000.0


def _resize_to(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth.astype(np.float32, copy=False)
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    return np.asarray(image.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR), dtype=np.float32)


def _scale_factor(pred: np.ndarray, sensor: np.ndarray, mode: str) -> float:
    if mode == "none":
        return 1.0
    valid = np.isfinite(pred) & np.isfinite(sensor) & (pred > 0.0) & (sensor > 0.0)
    if not valid.any():
        return 1.0
    pred_valid = pred[valid].astype(np.float64)
    sensor_valid = sensor[valid].astype(np.float64)
    if mode == "median":
        return float(np.median(sensor_valid / np.maximum(pred_valid, 1e-8)))
    if mode == "least_squares":
        denom = float(pred_valid @ pred_valid)
        if denom <= 1e-12:
            return 1.0
        return float((pred_valid @ sensor_valid) / denom)
    raise ValueError(f"Unknown scale alignment mode: {mode}")


def _apply_valid_mask(pred: np.ndarray, sensor: np.ndarray, mode: str) -> np.ndarray:
    if mode == "all":
        out = pred.copy()
        out[~np.isfinite(out) | (out <= 0.0)] = 0.0
        return out
    if mode == "sensor":
        out = pred.copy()
        keep = np.isfinite(out) & (out > 0.0) & np.isfinite(sensor) & (sensor > 0.0)
        out[~keep] = 0.0
        return out
    raise ValueError(f"Unknown valid-mask mode: {mode}")


def _resolve_metric_intrinsics(
    pred_intrinsics: Any,
    input_intrinsics: list[np.ndarray],
    num_depths: int,
) -> list[np.ndarray]:
    if len(input_intrinsics) != num_depths:
        raise RuntimeError(f"Got {len(input_intrinsics)} input intrinsics for {num_depths} predicted depths.")

    if pred_intrinsics is None:
        return input_intrinsics

    intrinsics = np.asarray(pred_intrinsics, dtype=np.float32)
    if intrinsics.ndim == 3 and intrinsics.shape == (num_depths, 3, 3):
        return [intrinsic for intrinsic in intrinsics]
    if intrinsics.ndim == 2 and intrinsics.shape == (3, 3) and num_depths == 1:
        return [intrinsics]
    return input_intrinsics


def _convert_metric_depths(
    depths: np.ndarray,
    pred_intrinsics: Any,
    input_intrinsics: list[np.ndarray],
) -> list[np.ndarray]:
    depth_batch = np.asarray(depths, dtype=np.float32)
    if depth_batch.ndim == 2:
        depth_batch = depth_batch[None]

    intrinsics = _resolve_metric_intrinsics(pred_intrinsics, input_intrinsics, len(depth_batch))
    converted: list[np.ndarray] = []
    for depth, intrinsic in zip(depth_batch, intrinsics, strict=True):
        focal = float((intrinsic[0, 0] + intrinsic[1, 1]) / 2.0)
        # DA3 metric checkpoints emit depths normalized to a 300 px focal basis.
        converted.append((depth * focal / 300.0).astype(np.float32))
    return converted


def _load_da3_predictor(args: Namespace) -> PredictDepths:
    try:
        api_spec = find_spec("depth_anything_3.api")
    except ModuleNotFoundError:
        api_spec = None
    if api_spec is not None:
        from depth_anything_3.api import DepthAnything3
    else:
        logger.info("depth_anything_3.api not found; falling back to depth_anything_3.DepthAnything3.")
        from depth_anything_3 import DepthAnything3  # type: ignore[no-redef]

    model_name = str(args.model)
    if "/" in model_name:
        model = DepthAnything3.from_pretrained(model_name)
    else:
        model = DepthAnything3(model_name=model_name)
    model = model.to(args.device)

    def predict(images: list[Path], input_intrinsics: list[np.ndarray]) -> list[np.ndarray]:
        prediction = model.inference([str(path) for path in images])
        depths = np.asarray(prediction.depth, dtype=np.float32)
        if depths.ndim == 2:
            depths = depths[None]
        if args.metric_conversion == "da3metric" or (
            args.metric_conversion == "auto" and "metric" in model_name.lower()
        ):
            return _convert_metric_depths(depths, getattr(prediction, "intrinsics", None), input_intrinsics)
        return [depth.astype(np.float32) for depth in depths]

    return predict


def run(args: Namespace, predict_depths: PredictDepths | None = None) -> dict[str, Any]:
    split_dir = Path(args.bop_root) / args.dataset / args.split
    predict = predict_depths or _load_da3_predictor(args)
    batch_size = int(getattr(args, "batch_size", 1))
    metadata: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "dst_depth_dir": args.dst_depth_dir,
        "valid_mask": args.valid_mask,
        "align_scale": args.align_scale,
        "frames": [],
    }
    num_written = 0
    num_skipped = 0

    for scene_dir in _scene_dirs(split_dir, args.scene_ids):
        camera = _load_json(scene_dir / "scene_camera.json")
        frame_ids = _select_frame_ids(scene_dir, getattr(args, "frame_ids", None))
        for start in range(0, len(frame_ids), batch_size):
            batch_frame_ids = frame_ids[start : start + batch_size]
            images = [scene_dir / "rgb" / f"{frame_id}.png" for frame_id in batch_frame_ids]
            intrinsics = [_intrinsic_for(camera, scene_dir, frame_id) for frame_id in batch_frame_ids]
            predictions = predict(images, intrinsics)
            if len(predictions) != len(batch_frame_ids):
                raise RuntimeError(f"Predictor returned {len(predictions)} depths for {len(batch_frame_ids)} images.")

            for frame_id, image_path, pred in zip(batch_frame_ids, images, predictions, strict=True):
                camera_data = _frame_camera(camera, scene_dir, frame_id)
                depth_scale = float(camera_data.get("depth_scale", 1.0))
                sensor = _sensor_depth_m(scene_dir, frame_id, camera_data)
                pred = _resize_to(np.asarray(pred, dtype=np.float32), sensor.shape)
                scale = _scale_factor(pred, sensor, args.align_scale)
                pred = _apply_valid_mask(pred * scale, sensor, args.valid_mask)

                out_path = scene_dir / args.dst_depth_dir / f"{frame_id}.png"
                if out_path.exists() and not args.overwrite:
                    num_skipped += 1
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(depth_m_to_bop_uint16(pred, depth_scale=depth_scale)).save(out_path)
                num_written += 1
                metadata["frames"].append(
                    {
                        "scene_id": int(scene_dir.name),
                        "frame_id": int(frame_id),
                        "rgb_path": str(image_path),
                        "depth_path": str(out_path),
                        "depth_scale": depth_scale,
                        "scale": scale,
                    }
                )

    metadata["num_written"] = num_written
    metadata["num_skipped"] = num_skipped
    _write_json(split_dir / f"{args.dst_depth_dir}_meta.json", metadata)
    return {"num_written": num_written, "num_skipped": num_skipped, "metadata": metadata}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate BOP depth images from Depth Anything 3 RGB inference.")
    parser.add_argument("--bop-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scene-ids", type=int, nargs="*", default=None)
    parser.add_argument("--frame-ids", type=int, nargs="*", default=None)
    parser.add_argument("--dst-depth-dir", default="depth_da3")
    parser.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metric-conversion", choices=["auto", "none", "da3metric"], default="auto")
    parser.add_argument("--align-scale", choices=["none", "median", "least_squares"], default="none")
    parser.add_argument("--valid-mask", choices=["all", "sensor"], default="sensor")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps({"num_written": summary["num_written"], "num_skipped": summary["num_skipped"]}, indent=2))


if __name__ == "__main__":
    main()
