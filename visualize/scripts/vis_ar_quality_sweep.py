"""Evaluate AR generation quality across multiple random seeds.

This script provides a reproducible workflow for issue #23:
- optionally run AR generation-process export for multiple seeds
- score one final AR candidate mesh per seed/object
- pick the best seed per object using an explicit criterion
- emit auditable metadata to diagnose indexing-vs-quality failures

Outputs:
    <output_dir>/report.json
    <output_dir>/summary.csv

Example:
    .venv/bin/python visualize/scripts/vis_ar_quality_sweep.py \
      --generation-root /run/media/.../train/cvpr_2025 \
      --base-log-name pcd_larm_kinect_net_bos_v1 \
      --seeds 0,1,2,3,4 \
      --objects 19,41,44,50 \
      --run-generate \
      --override model.arch=larm \
      --override model.average=ema \
      --override model.ema_decay=0.999 \
      --override ++vae_arch=3dshape2vecset_vqvae \
      --override +vae_weights=cvpr_2025_vae/pcd_vqvae_16k_long/model_best.pt \
      --override model.compile=False \
      --override train.batch_size=64 \
      --override train.epochs=2000 \
      --override train.lr=3.90625e-07
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass
class CandidateMetrics:
    seed: int
    run_name: str
    object_key: str
    mesh_path: str
    gt_path: str
    cd_l1: float
    cd_l2: float
    fscore_01: float
    precision_01: float
    recall_01: float
    component_count: int
    largest_component_ratio: float


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def stable_hash_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def resolve_repo_root() -> Path:
    # .../shape-completion/visualize/scripts/vis_ar_quality_sweep.py
    return Path(__file__).resolve().parents[2]


def run_generation_for_seed(
    *,
    python_executable: str,
    repo_root: Path,
    config_name: str,
    base_log_name: str,
    seed: int,
    objects: list[int],
    overrides: list[str],
) -> tuple[list[str], int, str]:
    run_name = f"{base_log_name}_s{seed}"
    objects_override = f"+vis.objects=[{','.join(str(v) for v in objects)}]"
    cmd: list[str] = [
        python_executable,
        "visualize/scripts/vis_generation_process.py",
        "-cn",
        config_name,
        f"log.name={run_name}",
        f"misc.seed={seed}",
        objects_override,
        "+vis.ar_steps=[512]",
    ]
    cmd.extend(overrides)

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    stderr_tail = "\n".join(proc.stderr.splitlines()[-20:]) if proc.stderr else ""
    return cmd, proc.returncode, stderr_tail


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry in scene: {path}")
        return trimesh.util.concatenate(tuple(geoms))
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise ValueError(f"Unsupported mesh type for {path}: {type(loaded).__name__}")


def sample_points(mesh: trimesh.Trimesh, n_points: int, seed: int) -> np.ndarray:
    if len(mesh.vertices) == 0:
        raise ValueError("Mesh has no vertices")

    np.random.seed(seed)
    if mesh.faces is not None and len(mesh.faces) > 0:
        return np.asarray(mesh.sample(n_points), dtype=np.float32)

    rng = np.random.default_rng(seed)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    idx = rng.integers(0, len(verts), size=n_points)
    return verts[idx]


def min_distances(a: np.ndarray, b: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError(f"Expected Nx3 arrays, got {a.shape} and {b.shape}")

    out = np.empty(a.shape[0], dtype=np.float64)
    for start in range(0, a.shape[0], chunk_size):
        end = min(start + chunk_size, a.shape[0])
        block = a[start:end]
        diff = block[:, None, :] - b[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        out[start:end] = np.sqrt(np.min(d2, axis=1))
    return out


def mesh_fragmentation(mesh: trimesh.Trimesh) -> tuple[int, float]:
    parts = mesh.split(only_watertight=False)
    if not parts:
        return 0, 0.0
    vertex_counts = [len(part.vertices) for part in parts]
    total = int(sum(vertex_counts))
    largest = int(max(vertex_counts)) if vertex_counts else 0
    ratio = float(largest / total) if total > 0 else 0.0
    return len(parts), ratio


def compute_metrics(
    pred_mesh_path: Path,
    gt_mesh_path: Path,
    *,
    points: int,
    sample_seed: int,
) -> tuple[float, float, float, float, float, int, float]:
    pred_mesh = load_mesh(pred_mesh_path)
    gt_mesh = load_mesh(gt_mesh_path)

    pred_points = sample_points(pred_mesh, points, sample_seed)
    gt_points = sample_points(gt_mesh, points, sample_seed + 1)

    d_pred_to_gt = min_distances(pred_points, gt_points)
    d_gt_to_pred = min_distances(gt_points, pred_points)

    cd_l1 = float(d_pred_to_gt.mean() + d_gt_to_pred.mean())
    cd_l2 = float((d_pred_to_gt**2).mean() + (d_gt_to_pred**2).mean())

    thr = 0.01
    precision = float(np.mean(d_pred_to_gt <= thr))
    recall = float(np.mean(d_gt_to_pred <= thr))
    if precision + recall <= 0.0:
        fscore = 0.0
    else:
        fscore = float(2.0 * precision * recall / (precision + recall))

    n_comp, largest_ratio = mesh_fragmentation(pred_mesh)
    return cd_l1, cd_l2, fscore, precision, recall, n_comp, largest_ratio


def collect_object_dirs(run_generation_dir: Path, token_name: str) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    if not run_generation_dir.exists():
        return result

    for category_dir in sorted(run_generation_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for object_dir in sorted(category_dir.iterdir()):
            if not object_dir.is_dir():
                continue
            pred_path = object_dir / "ar" / token_name
            gt_path = object_dir / "gt.ply"
            if not pred_path.exists() or not gt_path.exists():
                continue
            key = f"{category_dir.name}/{object_dir.name}"
            result[key] = {
                "pred": pred_path,
                "gt": gt_path,
            }
    return result


def choose_best(candidates: list[CandidateMetrics], criterion: str) -> CandidateMetrics:
    if criterion == "cd_l1":
        return min(candidates, key=lambda c: c.cd_l1)
    if criterion == "fscore_01":
        return max(candidates, key=lambda c: c.fscore_01)
    raise ValueError(f"Unknown criterion: {criterion}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AR quality sweep over random seeds.")
    parser.add_argument("--generation-root", type=Path, required=True,
                        help="Directory containing run folders (e.g., .../train/cvpr_2025).")
    parser.add_argument("--base-log-name", type=str, required=True,
                        help="Base run name; seed suffix `_s<seed>` is appended automatically.")
    parser.add_argument("--seeds", type=str, required=True,
                        help="Comma-separated seeds, e.g. 0,1,2,3,4.")
    parser.add_argument("--objects", type=str, required=True,
                        help="Comma-separated dataset object indices to process; stored for auditability.")
    parser.add_argument("--config-name", type=str, default="cvpr_2025")
    parser.add_argument("--python", type=str, default=".venv/bin/python",
                        help="Python executable used when --run-generate is set.")
    parser.add_argument("--run-generate", action="store_true",
                        help="Run vis_generation_process.py for every seed before scoring.")
    parser.add_argument("--override", action="append", default=[],
                        help="Extra Hydra override for generation command; repeatable.")
    parser.add_argument("--token-name", type=str, default="token_512.ply",
                        help="Final AR token mesh file to score.")
    parser.add_argument("--points", type=int, default=4096,
                        help="Number of points sampled on pred/gt meshes for metrics.")
    parser.add_argument("--criterion", choices=["cd_l1", "fscore_01"], default="cd_l1",
                        help="Seed selection criterion per object.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for report outputs (defaults under generation-root).")
    args = parser.parse_args()

    seeds = sorted(set(parse_int_list(args.seeds)))
    objects = sorted(set(parse_int_list(args.objects)))
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")
    if not objects:
        raise ValueError("No objects parsed from --objects")

    generation_root = args.generation_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (generation_root / f"{args.base_log_name}_quality_audit")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = resolve_repo_root()

    command_log: list[dict[str, Any]] = []
    run_to_objects: dict[str, list[str]] = {}
    run_to_paths: dict[str, dict[str, dict[str, str]]] = {}

    if args.run_generate:
        for seed in seeds:
            cmd, return_code, stderr_tail = run_generation_for_seed(
                python_executable=args.python,
                repo_root=repo_root,
                config_name=args.config_name,
                base_log_name=args.base_log_name,
                seed=seed,
                objects=objects,
                overrides=list(args.override),
            )
            command_log.append(
                {
                    "seed": seed,
                    "command": cmd,
                    "return_code": return_code,
                    "stderr_tail": stderr_tail,
                }
            )
            if return_code != 0:
                raise RuntimeError(
                    f"Generation failed for seed={seed} (return code {return_code}). "
                    "Inspect report command_log for stderr tail."
                )

    for seed in seeds:
        run_name = f"{args.base_log_name}_s{seed}"
        run_generation_dir = generation_root / run_name / "generation_process"
        object_map = collect_object_dirs(run_generation_dir, args.token_name)
        run_to_objects[run_name] = sorted(object_map.keys())
        run_to_paths[run_name] = {
            key: {"pred": str(paths["pred"]), "gt": str(paths["gt"])}
            for key, paths in object_map.items()
        }

    run_key_sets = [set(keys) for keys in run_to_objects.values()]
    if not run_key_sets:
        raise RuntimeError("No runs found for evaluation")

    common_keys = sorted(set.intersection(*run_key_sets))
    union_keys = sorted(set.union(*run_key_sets))

    if not common_keys:
        raise RuntimeError(
            "No evaluable objects found: expected both prediction mesh "
            f"('{args.token_name}') and gt.ply per object across runs. "
            "This usually means the generation output lacks gt.ply for the selected dataset/config."
        )

    missing_by_run: dict[str, list[str]] = {}
    for run_name, keys in run_to_objects.items():
        missing = sorted(set(union_keys) - set(keys))
        missing_by_run[run_name] = missing

    candidate_rows: list[CandidateMetrics] = []
    for object_key in common_keys:
        for seed in seeds:
            run_name = f"{args.base_log_name}_s{seed}"
            path_info = run_to_paths[run_name][object_key]
            pred_path = Path(path_info["pred"])
            gt_path = Path(path_info["gt"])

            sample_seed = stable_hash_int(f"{object_key}:{seed}") % (2**31)
            cd_l1, cd_l2, fscore, precision, recall, comp_count, comp_ratio = compute_metrics(
                pred_path,
                gt_path,
                points=args.points,
                sample_seed=sample_seed,
            )

            candidate_rows.append(
                CandidateMetrics(
                    seed=seed,
                    run_name=run_name,
                    object_key=object_key,
                    mesh_path=str(pred_path),
                    gt_path=str(gt_path),
                    cd_l1=cd_l1,
                    cd_l2=cd_l2,
                    fscore_01=fscore,
                    precision_01=precision,
                    recall_01=recall,
                    component_count=comp_count,
                    largest_component_ratio=comp_ratio,
                )
            )

    grouped: dict[str, list[CandidateMetrics]] = {}
    for row in candidate_rows:
        grouped.setdefault(row.object_key, []).append(row)

    selection: dict[str, dict[str, Any]] = {}
    for object_key, rows in grouped.items():
        best = choose_best(rows, args.criterion)
        selection[object_key] = {
            "criterion": args.criterion,
            "best_seed": best.seed,
            "best_run_name": best.run_name,
            "best_metrics": asdict(best),
            "candidates": [asdict(r) for r in sorted(rows, key=lambda r: r.seed)],
        }

    report = {
        "base_log_name": args.base_log_name,
        "generation_root": str(generation_root),
        "objects_requested": objects,
        "seeds": seeds,
        "criterion": args.criterion,
        "points": args.points,
        "token_name": args.token_name,
        "commands": command_log,
        "runs": run_to_objects,
        "missing_objects_by_run": missing_by_run,
        "common_object_keys": common_keys,
        "selection": selection,
        "diagnostics": {
            "common_object_count": len(common_keys),
            "union_object_count": len(union_keys),
            "object_identity_consistent_across_seeds": len(common_keys) == len(union_keys),
            "note": (
                "If object_identity_consistent_across_seeds is false, failures may stem from data/index "
                "selection mismatches rather than pure generation quality."
            ),
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "object_key",
                "best_seed",
                "criterion",
                "cd_l1",
                "cd_l2",
                "fscore_01",
                "precision_01",
                "recall_01",
                "component_count",
                "largest_component_ratio",
            ]
        )
        for object_key in sorted(selection.keys()):
            metrics = selection[object_key]["best_metrics"]
            writer.writerow(
                [
                    object_key,
                    selection[object_key]["best_seed"],
                    args.criterion,
                    metrics["cd_l1"],
                    metrics["cd_l2"],
                    metrics["fscore_01"],
                    metrics["precision_01"],
                    metrics["recall_01"],
                    metrics["component_count"],
                    metrics["largest_component_ratio"],
                ]
            )

    print(f"Saved report: {report_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
