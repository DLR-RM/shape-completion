from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from PIL import Image
from tabulate import tabulate
from torch import Tensor
from trimesh import PointCloud, Trimesh

from eval import eval_mesh
from utils import DEBUG_LEVEL_1, default_on_exception, load_mesh, resolve_path, setup_logger, stack_images

from .visualize import VisualizeCallback

logger = setup_logger(__name__)
debug_level_1 = getattr(logger, "debug_level_1", logger.debug)
debug_level_2 = getattr(logger, "debug_level_2", logger.debug)


class TestMeshesCallback(VisualizeCallback):
    def __init__(
        self,
        test: bool = True,
        meshes: bool = True,
        inputs: bool = True,
        front: bool = True,
        back: bool = True,
        upload_to_wandb: bool = False,
        points_batch_size: int | None = None,
        threshold: float = 0.5,
        padding: float = 0.1,
        resolution: int = 128,
        width: int = 512,
        height: int = 512,
        show: bool = False,
        precision: str | int | None = None,
    ) -> None:
        super().__init__(
            meshes=meshes,
            inputs=inputs,
            front=front or True,
            back=back,
            upload_to_wandb=upload_to_wandb,
            points_batch_size=points_batch_size,
            threshold=threshold,
            padding=padding,
            resolution=resolution,
            width=width,
            height=height,
            show=show,
            precision=precision,
        )

        if not test and not (front or back):
            raise ValueError("Either testing or visualization must be enabled")

        self.test = test
        self.front = front
        self.back = back
        self.stats: list[dict[str, dict[str, float]]] = []

    @default_on_exception()
    def _eval_mesh(
        self,
        in_path: Path,
        mesh: Trimesh,
        offset: float | np.ndarray | None = 0,
        scale: float | None = 1,
        pose: np.ndarray | str | Path | None = None,
    ) -> dict[str, dict[str, float]] | None:
        debug_level_2(f"Starting mesh evaluation for '{in_path}'.")
        pose = pose if isinstance(pose, np.ndarray) else np.load(pose) if isinstance(pose, (str, Path)) else None
        if pose is None:
            for file in in_path.parent.glob("*.npy"):
                if in_path.stem.replace("object_", "") in file.name and "pose" in file.name:
                    debug_level_2(f"Found pose file '{file.name}' in input directory.")
                    pose = np.load(file)
                    break

        if pose is None:
            logger.warning(f"Mesh evaluation for '{in_path}' failed. Could not find pose file.")
            return

        gt_mesh_path = None
        gt_mesh_paths = open(in_path.parent / "gt_mesh_paths.txt").readlines()
        for path in gt_mesh_paths:
            gt_mesh_path = resolve_path(path.strip())
            if gt_mesh_path.is_file():
                debug_level_2(f"Found gt mesh file '{gt_mesh_path}'.")
                break

        if gt_mesh_path is None or not gt_mesh_path.is_file():
            logger.warning(f"Mesh evaluation for '{in_path}' failed. Could not find gt mesh path.")
            return

        generator = cast(Any, self.generator)
        query_points = generator.query_points
        query_points_array = query_points.numpy() if hasattr(query_points, "numpy") else np.asarray(query_points)

        result = eval_mesh(
            mesh=mesh,
            mesh_gt=Trimesh(*load_mesh(gt_mesh_path)),
            query_points=query_points_array,
            offset=offset,
            scale=scale,
            pose=pose,
        )
        return {gt_mesh_path.parent.name: result}

    def _visualize_batch(self) -> None:
        if not self._batch:
            return

        item = cast(dict[str, Any], self._batch[0])
        mesh = item.get("mesh")
        inputs = item.get("inputs")
        if not isinstance(mesh, Trimesh):
            return

        pcd: PointCloud | None = None
        if isinstance(inputs, np.ndarray) and inputs.ndim == 2 and inputs.shape[1] == 3:
            pcd = PointCloud(inputs)

        pcds = [pcd] if self.inputs and pcd is not None else None
        self._render_meshes([mesh], pcds)

    def _eval_batch(self) -> None:
        if not self._batch:
            return

        item = cast(dict[str, Any], self._batch[0])
        path = item.get("inputs.path")
        if not isinstance(path, (str, Path)):
            debug_level_1("Could not find input path in batch item. Skipping mesh evaluation.")
            return

        mesh = item.get("mesh")
        if not isinstance(mesh, Trimesh):
            return

        loc_raw = item.get("inputs.norm_offset", np.zeros(3))
        loc = cast(float | np.ndarray, loc_raw)
        scale_raw = item.get("inputs.norm_scale", 1.0)
        if isinstance(scale_raw, (int, float, np.floating)):
            scale = float(scale_raw)
        else:
            scale = 1.0

        result = self._eval_mesh(resolve_path(path), mesh, loc, scale)
        if result is not None:
            self.stats.append(result)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: dict[str, list[str] | Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._gather_items(trainer.world_size)
        self._add_item({key: value[0] for key, value in batch.items()})
        self._process_batch(cast(Any, pl_module).model)
        if self.test:
            self._eval_batch()
        if self.front or self.back:
            self._visualize_batch()

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.world_size <= 1:
            return

        gathered_stats: list[list[dict[str, dict[str, float]]] | None] = [None] * trainer.world_size
        torch.distributed.all_gather_object(gathered_stats, self.stats)

        self.stats = [item for sublist in gathered_stats if sublist is not None for item in sublist]

    @staticmethod
    def _make_table_data(data: list[dict[str, dict[str, float]]]) -> tuple[list[list[Any]], list[str]]:
        table_data: list[list[Any]] = []
        for item in data:
            for name, metrics in item.items():
                row: list[Any] = [name]
                for metric_val in metrics.values():
                    row.append(metric_val)
                table_data.append(row)

        headers = ["name", *next(iter(data[0].values())).keys()]
        mean_values: list[Any] = ["mean"]
        for col in range(1, len(headers)):
            mean = np.nanmean([row[col] for row in table_data])
            mean_values.append(mean)
        table_data.append(mean_values)

        return table_data, headers

    @default_on_exception()
    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        out_dir = Path(trainer.default_root_dir) / "test"
        out_dir.mkdir(exist_ok=True, parents=True)

        stats = [item for item in self.stats if item is not None]
        table = ""
        headers: list[str] = []
        table_data: list[list[Any]] = []
        if stats:
            table_data, headers = self._make_table_data(stats)
            table = tabulate(table_data, headers=headers, tablefmt="presto", floatfmt=".4f")
            if logger.isEnabledFor(DEBUG_LEVEL_1):
                print(table)

        if self.upload_to_wandb:
            log_dict: dict[str, Any] = {"trainer/global_step": trainer.global_step}
            if self.front and self.images.get("front"):
                log_dict["vis/test/front"] = [wandb.Image(image) for image in self.images["front"]]
            if self.back and self.images.get("back"):
                log_dict["vis/test/back"] = [wandb.Image(image) for image in self.images["back"]]
            if stats:
                for key, value in zip(headers[1:], table_data[-1][1:], strict=True):
                    log_dict[f"test/{key}"] = value
            experiment = getattr(trainer.logger, "experiment", None)
            if experiment is not None:
                experiment.log(log_dict)
        else:
            if self.front:
                self.save_image(
                    out_dir / "front",
                    image=stack_images(cast(list[np.ndarray | Image.Image], self.images["front"])),
                )
            if self.back:
                self.save_image(
                    out_dir / "back",
                    image=stack_images(cast(list[np.ndarray | Image.Image], self.images["back"])),
                )
            if stats:
                with open(out_dir / "stats.txt", "w") as f:
                    f.write(table)

        for key in ("names", "front", "back", "inputs", "logits", "depth", "color", "normals"):
            values = self.images.get(key)
            if values is not None:
                values.clear()
        self.images.pop("stacked_front", None)
        self.images.pop("stacked_back", None)
        self.images.pop("stacked_logits", None)

        for key in ("categories", "meshes", "points", "logits", "inputs"):
            values = self.data.get(key)
            if values is not None:
                values.clear()
        self.items.clear()
        self.stats.clear()
