import shutil
import tempfile
from collections.abc import Iterable
from functools import cached_property, partial
from pathlib import Path
from pprint import pprint
from typing import cast

import lightning.pytorch as pl
import numpy as np
from cleanfid import fid
from joblib import Parallel, delayed
from lightning.pytorch.utilities import rank_zero_only
from tqdm import tqdm
from trimesh import Trimesh

from eval import eval_mesh, eval_mesh_pcd, render_for_fid
from utils import DEBUG_LEVEL_2, default_on_exception, get_num_workers, setup_logger, tqdm_joblib

from .visualize import VisualizeCallback

logger = setup_logger(__name__)
debug_level_2 = getattr(logger, "debug_level_2", logger.debug)

_MeshGT = Trimesh | str
_ItemValue = str | np.ndarray | Trimesh
_EvalItem = dict[str, _ItemValue]


class EvalMeshesCallback(VisualizeCallback):
    def __init__(
        self,
        every_n_evals: int | Iterable[int] | None = 1,
        n_per_category: int | None = None,
        n_total: int | None = 57 * 8,
        upload_to_wandb: bool = False,
        points_batch_size: int | None = None,
        threshold: float = 0.5,
        padding: float = 0.1,
        resolution: int = 128,
        precision: str | int | None = None,
        progress: bool = False,
        num_workers: int | None = None,
        fid_stats_name: str | None = None,
        metrics: str = "all",
        prefix: str = "val/mesh/",
    ) -> None:
        super().__init__(
            every_n_evals=every_n_evals,
            n_per_category=n_per_category,
            n_total=n_total,
            upload_to_wandb=upload_to_wandb,
            resolution=resolution,
            padding=padding,
            points_batch_size=points_batch_size,
            threshold=threshold,
            precision=precision,
            progress=progress,
        )

        if fid_stats_name:
            assert n_total is not None and n_total * 12 >= 2048, (
                "n_total * 12 must at least be 2048 for render FID evaluation"
            )  # TODO: Change to 20 when using dodecahedron

        assert not ("pcd" in metrics and "mesh" in metrics), "Pointcloud and mesh metrics are equivalent"

        self.fid_stats_name = fid_stats_name
        self.metrics = metrics
        self.prefix = prefix
        self.num_workers = get_num_workers(num_workers=num_workers)

    @cached_property
    def fid_dir(self) -> Path:
        return Path(tempfile.mkdtemp())

    @cached_property
    def parallel(self) -> Parallel:
        return Parallel(n_jobs=self.num_workers)

    @staticmethod
    def _mean_results(values: list[dict[str, float]]) -> dict[str, float]:
        return {key: sum(value[key] for value in values) / len(values) for key in values[0]}

    @rank_zero_only
    def _eval_meshes(self, pl_module: pl.LightningModule) -> dict[str, float]:
        if not self.items:
            return {}

        results: dict[str, float] = {}
        sorted_items = sorted(cast(list[_EvalItem], self.items), key=lambda item: str(item["category.name"]))

        pcd_inputs: list[tuple[Trimesh, _EvalItem]] = []
        mesh_inputs: list[tuple[Trimesh, _MeshGT, np.ndarray | None]] = []
        fid_inputs: list[tuple[str, Trimesh]] = []

        for item in sorted_items:
            input_name = item.get("inputs.name")
            mesh = item.get("mesh")
            if not isinstance(input_name, str) or not isinstance(mesh, Trimesh):
                continue
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                continue

            fid_inputs.append((input_name, mesh))

            pointcloud = item.get("pointcloud")
            if isinstance(pointcloud, np.ndarray):
                pcd_inputs.append((mesh, item))

            points = item.get("points")
            query_points = points if isinstance(points, np.ndarray) else None

            mesh_gt: _MeshGT | None = None
            if "mesh.vertices" in item and "mesh.triangles" in item:
                mesh_gt = Trimesh(
                    cast(np.ndarray, item["mesh.vertices"]),
                    cast(np.ndarray, item["mesh.triangles"]),
                    process=False,
                )
            elif "mesh.path" in item and isinstance(item["mesh.path"], str):
                mesh_gt = cast(str, item["mesh.path"])
            if mesh_gt is not None:
                mesh_inputs.append((mesh, mesh_gt, query_points))

        if pcd_inputs and any(metric in self.metrics for metric in ("pcd", "all")):
            with tqdm_joblib(
                tqdm(
                    total=len(pcd_inputs),
                    desc="Evaluating pointclouds",
                    leave=True,
                    disable=not self.progress,
                )
            ):
                pcd_results_raw = cast(
                    list[dict[str, float] | None],
                    self.parallel(delayed(eval_mesh_pcd)(mesh, item) for mesh, item in pcd_inputs),
                )
            pcd_results = [result for result in pcd_results_raw if result is not None]
            if pcd_results:
                mean_results = self._mean_results(pcd_results)
            else:
                mean_results = {}
            if logger.isEnabledFor(DEBUG_LEVEL_2):
                print("Pointcloud evaluation results:")
                pprint(mean_results)
            results.update(mean_results)
        elif mesh_inputs and any(metric in self.metrics for metric in ("mesh", "all")):
            _eval_mesh = partial(eval_mesh, normalize=not isinstance(mesh_inputs[0][1], Trimesh))
            with tqdm_joblib(
                tqdm(
                    total=len(mesh_inputs),
                    desc="Evaluating meshes",
                    leave=True,
                    disable=not self.progress,
                )
            ):
                mesh_results_raw = cast(
                    list[dict[str, float] | None],
                    self.parallel(
                        delayed(_eval_mesh)(mesh, mesh_gt, query_points) for mesh, mesh_gt, query_points in mesh_inputs
                    ),
                )
            mesh_results = [result for result in mesh_results_raw if result is not None]
            if mesh_results:
                mean_results = self._mean_results(mesh_results)
            else:
                mean_results = {}
            if logger.isEnabledFor(DEBUG_LEVEL_2):
                print("Mesh evaluation results:")
                pprint(mean_results)
            results.update(mean_results)

        if self.fid_stats_name and any(metric in self.metrics for metric in ("fid", "kid", "all")):
            metric_names: list[str] = []
            if "fid" in self.metrics or "all" in self.metrics:
                metric_names.append("FID")
            if "kid" in self.metrics or "all" in self.metrics:
                metric_names.append("KID")
            model_names = ["inception_v3"]
            if "clip" in self.metrics or "all" in self.metrics:
                model_names.append("clip_vit_b_32")

            if len(fid_inputs) * 12 < 2048:  # TODO: Change to 20 when using dodecahedron
                logger.warning(f"Not enough images for FID evaluation ({len(fid_inputs) * 12} < 2048)")
                for metric in metric_names:
                    for model_name in model_names:
                        results[f"{self.prefix}{model_name.split('_')[0]}_{metric}"] = float("nan")
            elif not fid.test_stats_exists(self.fid_stats_name, mode="clean"):
                logger.error(f"FID stats for {self.fid_stats_name} not found")
            else:
                shutil.rmtree(self.fid_dir, ignore_errors=True)
                self.fid_dir.mkdir(exist_ok=True, parents=True)
                for view in range(12):
                    (self.fid_dir / f"view_{view}").mkdir(exist_ok=True)

                _render_for_fid = partial(render_for_fid, views="icosphere")  # TODO: Change to dodecahedron
                with tqdm_joblib(
                    tqdm(
                        total=len(fid_inputs),
                        desc="Rendering meshes",
                        leave=True,
                        disable=not self.progress,
                    )
                ):
                    self.parallel(delayed(_render_for_fid)(mesh, self.fid_dir / name) for name, mesh in fid_inputs)

                for metric in metric_names:
                    for model_name in model_names:
                        score = float("nan")
                        if fid.test_stats_exists(
                            self.fid_stats_name, mode="clean", model_name=model_name, metric=metric
                        ):
                            if metric == "KID" and model_name == "inception_v3":
                                score = fid.compute_kid(
                                    str(self.fid_dir),
                                    dataset_name=self.fid_stats_name,
                                    dataset_split="custom",
                                    num_workers=self.num_workers,
                                    device=pl_module.device,
                                    verbose=self.progress,
                                )
                                debug_level_2(f"{metric} ({model_name.split('_')[0].upper()}): {score}")
                            elif metric == "FID":
                                score = fid.compute_fid(
                                    str(self.fid_dir),
                                    dataset_name=self.fid_stats_name,
                                    dataset_split="custom",
                                    model_name=model_name,
                                    num_workers=self.num_workers,
                                    device=pl_module.device,
                                    verbose=self.progress,
                                )
                                debug_level_2(f"{metric} ({model_name.split('_')[0].upper()}): {score}")
                        else:
                            logger.error(f"{metric} stats ({model_name}) not found")
                        results[f"{self.prefix}{model_name.split('_')[0]}_{metric}"] = score
        return results

    @default_on_exception()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        results = self._eval_meshes(pl_module)
        if results is None:
            return
        pl_module.log_dict(trainer.strategy.broadcast(results))

    @rank_zero_only
    def __del__(self) -> None:
        fid_dir = self.__dict__.get("fid_dir")
        if isinstance(fid_dir, Path):
            debug_level_2(f"Removing FID directory: {fid_dir}")
            shutil.rmtree(fid_dir, ignore_errors=True)
