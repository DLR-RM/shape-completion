import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed
import trimesh
import wandb
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor
from torch.utils.data import default_collate
from trimesh import PointCloud, Trimesh

from utils import default_on_exception, inv_trafo, look_at, setup_logger, stack_images

from .generate_meshes import GenerateMeshesCallback

logger = setup_logger(__name__)


class _DataStore(TypedDict):
    categories: list[str]
    meshes: list[Trimesh]
    points: list[np.ndarray]
    logits: list[np.ndarray]
    inputs: list[PointCloud | np.ndarray]


class _ImageStore(TypedDict):
    names: list[str]
    front: list[np.ndarray]
    back: list[np.ndarray]
    inputs: list[np.ndarray]
    logits: list[np.ndarray]
    depth: list[np.ndarray]
    color: list[np.ndarray]
    normals: list[np.ndarray]
    stacked_front: NotRequired[list[tuple[str, np.ndarray]]]
    stacked_back: NotRequired[list[tuple[str, np.ndarray]]]
    stacked_logits: NotRequired[list[tuple[str, np.ndarray]]]


BatchItem = dict[str, Any]
BatchDict = dict[str, list[str] | Tensor]


class VisualizeCallback(GenerateMeshesCallback):
    def __init__(
        self,
        every_n_evals: int | Iterable[int] | None = 1,
        n_per_category: int | None = 4,
        n_total: int | None = None,
        meshes: bool = True,
        inputs: bool = True,
        logits: bool = False,
        render: Literal["color", "normals", "mesh"] | None = None,
        front: bool = True,
        back: bool = False,
        upload_to_wandb: bool = False,
        points_batch_size: int | None = None,
        threshold: float = 0.5,
        padding: float = 0.1,
        resolution: int = 128,
        width: int = 512,
        height: int = 512,
        show: bool = False,
        precision: str | int | None = None,
        progress: bool = False,
        **generator_kwargs,
    ):
        super().__init__(
            every_n_evals=every_n_evals,
            resolution=resolution,
            padding=padding,
            points_batch_size=points_batch_size,
            threshold=threshold,
            precision=precision,
            **generator_kwargs,
        )

        if n_per_category:
            assert n_per_category % 2 == 0 or n_per_category == 1, "n_per_category must be 1 or even"
        assert n_per_category or n_total, "At least one of n_per_category or n_total must be set"
        assert meshes or logits or inputs or render, "At least one of meshes, logits, inputs or render must be True"
        assert front or back, "At least one of front or back must be True"

        self.n_per_category = n_per_category
        self.n_total = n_total
        self.meshes = meshes
        self.inputs = inputs
        self.logits = logits
        self.render = render
        self.front = front
        self.back = back
        self.upload_to_wandb = upload_to_wandb
        self.width = width
        self.height = height
        self.progress = progress
        self.counter = Counter()

        self.handles = list()
        if show:
            n_vis = 0
            if meshes:
                n_vis += 1
            elif inputs:
                n_vis += 1
            if logits:
                n_vis += 1
            if render:
                n_vis += 1
            if render and "color" in render:
                n_vis += 1
            if render and "normals" in render:
                n_vis += 1
            self.fig, self.axes = plt.subplots(nrows=1, ncols=n_vis, figsize=(n_vis * 6, 6))
            if not isinstance(self.axes, np.ndarray):
                self.axes = [self.axes]
            for ax in self.axes:
                ax.axis("off")
                self.handles.append(ax.imshow(np.zeros((width, height, 3))))
            plt.tight_layout()
            plt.ion()
            plt.show(block=False)
            plt.pause(1)
        else:
            os.environ["PYOPENGL_PLATFORM"] = "egl"

        self.data: _DataStore = {
            "categories": list(),
            "meshes": list(),
            "points": list(),
            "logits": list(),
            "inputs": list(),
        }
        self.images: _ImageStore = {
            "names": list(),
            "front": list(),
            "back": list(),
            "inputs": list(),
            "logits": list(),
            "depth": list(),
            "color": list(),
            "normals": list(),
        }
        self.items: list[BatchItem] = []
        self._batch: list[BatchItem] = []
        self._done = False

    @staticmethod
    def _check_item(item: BatchItem) -> BatchItem:
        _item: BatchItem = {}
        for key, value in item.items():
            if torch.is_tensor(value):
                if torch.is_floating_point(value):  # Deal with 16-bit precision
                    value = value.float()
                _item[key] = value.cpu().numpy()
            else:
                _item[key] = value
        return _item

    def _add_item(self, item: BatchItem):
        self._batch.append(self._check_item(item))

    def add_items(self, items: list[BatchItem]):
        self._batch.extend([self._check_item(item) for item in items])

    def done(self, world_size: int = 1) -> bool:
        if not self._done:
            self._gather_items(world_size)
            if self.n_total and len(self.items) >= self.n_total:
                self._done = True
        return self._done

    @cached_property
    def renderer(self) -> Any:
        from visualize import Renderer

        return Renderer(
            method="pyrender",
            width=self.width,
            height=self.height,
            file_format="JPEG" if self.upload_to_wandb else "PNG",
        )

    @property
    def default_image(self) -> np.ndarray:
        return np.zeros((self.renderer.height, self.renderer.width, 3), dtype=np.uint8)

    @staticmethod
    def save_image(path: Path, image: np.ndarray | Image.Image, image_format: str | None = None):
        path.parent.mkdir(exist_ok=True, parents=True)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(path.with_suffix(image_format or ".png"))

    @staticmethod
    def save_mesh_or_pointcloud(
        path: Path, mesh_or_pointcloud: Trimesh | PointCloud | np.ndarray, mesh_format: str | None = None
    ):
        path.parent.mkdir(exist_ok=True, parents=True)
        if isinstance(mesh_or_pointcloud, np.ndarray):
            mesh_or_pointcloud = PointCloud(mesh_or_pointcloud)
        suffix = mesh_format or ".obj" if isinstance(mesh_or_pointcloud, Trimesh) else ".ply"
        mesh_or_pointcloud.export(path.with_suffix(suffix))

    @rank_zero_only
    def _upload(
        self,
        trainer: pl.Trainer,
        num_3d: int = 4,
        max_num_faces: int = int(1e4),
        logit_as_pointcloud: bool = False,
        max_num_points: int = int(1e4),
        prefix: str = "vis/",
    ):
        log_dict: dict[str, Any] = {"trainer/global_step": trainer.global_step}

        columns: list[str] = []
        if self.images["depth"]:
            columns.append("depth")
        if self.images["color"]:
            columns.append("color")
        if self.images["normals"]:
            columns.append("normals")

        if columns:
            dcn_table = wandb.Table(columns=["category", *columns])

            for index, category in enumerate(self.data["categories"]):
                data: list[Any] = [category]
                if self.images["depth"]:
                    data.append(wandb.Image(self.images["depth"][index], caption=f"Depth - {category}"))

                if self.images["color"]:
                    data.append(wandb.Image(self.images["color"][index], caption=f"Color - {category}"))

                if self.images["normals"]:
                    data.append(wandb.Image(self.images["normals"][index], caption=f"Normals - {category}"))

                dcn_table.add_data(*data)

            if dcn_table:
                log_dict[f"{prefix}renders"] = dcn_table

        images_front = self.images["front"]
        stacked_front = self.images.get("stacked_front")
        images_back = self.images["back"]
        stacked_back = self.images.get("stacked_back")
        logits_images = self.images["logits"]
        stacked_logits = self.images.get("stacked_logits")
        categories_for_lists = self.data["categories"]

        def create_image_list_for_upload(
            img_data_list: list[np.ndarray],
            stacked_img_data: list[tuple[str, np.ndarray]] | None,
            cats_list: list[str],
            img_type_name: str = "",
        ) -> list[Any]:
            image_list_output: list[Any] = []
            if stacked_img_data:
                for cat_name, img_content in stacked_img_data:
                    image_list_output.append(wandb.Image(img_content, caption=cat_name))
            elif img_data_list:
                for idx, img_content in enumerate(img_data_list):
                    cat_name = cats_list[idx] if idx < len(cats_list) else f"Item {idx}"
                    caption = f"{cat_name} {img_type_name} {idx}".strip()
                    image_list_output.append(wandb.Image(img_content, caption=caption))
            return image_list_output

        wandb_front_back_list = create_image_list_for_upload(images_front, stacked_front, categories_for_lists, "view")
        wandb_front_back_list.extend(
            create_image_list_for_upload(images_back, stacked_back, categories_for_lists, "view")
        )

        if wandb_front_back_list:
            log_dict[f"{prefix}images"] = wandb_front_back_list

        wandb_logits_img_list = create_image_list_for_upload(
            logits_images, stacked_logits, categories_for_lists, "logits"
        )
        if wandb_logits_img_list:
            log_dict[f"{prefix}logits"] = wandb_logits_img_list

        if self.meshes:
            wandb_meshes_data: list[Any] = []
            source_mesh_categories = self.data["categories"]
            source_mesh_objects = self.data["meshes"]

            effective_num_meshes = min(num_3d, len(source_mesh_objects), len(source_mesh_categories))
            for i in range(effective_num_meshes):
                category = source_mesh_categories[i]
                mesh_obj = source_mesh_objects[i]

                processed_mesh = mesh_obj
                if len(mesh_obj.faces) > max_num_faces:
                    try:
                        from libs import simplify_mesh  # Ensure this import is valid in your environment

                        processed_mesh = simplify_mesh(mesh_obj, max_num_faces)
                        logger.debug(f"Simplified mesh for {category}")
                    except ImportError:
                        logger.warning("libs.simplify_mesh not available. Using original mesh.")
                    except Exception as e:
                        logger.error(f"Error simplifying mesh for {category}: {e}")

                temp_fd, path = tempfile.mkstemp(suffix=".obj")
                os.close(temp_fd)  # Close the file descriptor
                processed_mesh.export(path)
                caption = f"{category} {i}" if effective_num_meshes > 1 and num_3d > 1 else category
                wandb_meshes_data.append(wandb.Object3D(str(path), caption=caption))

            if wandb_meshes_data:
                log_dict[f"{prefix}meshes"] = wandb_meshes_data

        if self.logits and logit_as_pointcloud:  # self.logits is a boolean flag
            wandb_logits_points_data: list[Any] = []
            source_lp_categories = self.data["categories"]
            source_lp_points = self.data["points"]

            effective_num_lp = min(num_3d, len(source_lp_points), len(source_lp_categories))
            for i in range(effective_num_lp):
                category = source_lp_categories[i]
                points_data = source_lp_points[i]
                if points_data.ndim == 2 and points_data.shape[1] == 3 and points_data.shape[0] > 0:
                    caption = f"{category} {i}" if effective_num_lp > 1 and num_3d > 1 else category
                    wandb_logits_points_data.append(wandb.Object3D(points_data[:max_num_points], caption=caption))
                elif points_data.shape[0] == 0:
                    logger.debug(
                        f"Skipping logits point cloud for {category} {i} as there are no points after filtering."
                    )
                else:
                    logger.warning(
                        f"Skipping logits point cloud for {category} {i} due to unexpected points data shape: {points_data.shape}"
                    )

            if wandb_logits_points_data:
                log_dict[f"{prefix}logits_3d"] = wandb_logits_points_data  # New key

        if self.logits and self.data["logits"]:
            valid_logit_arrays = [arr for arr in self.data["logits"] if arr.size > 0]
            if valid_logit_arrays:
                try:
                    concatenated_logits = np.concatenate(valid_logit_arrays)
                    log_dict[f"{prefix}logits_hist"] = wandb.Histogram(concatenated_logits.tolist())
                except ValueError:
                    logger.warning(
                        "Could not create logits histogram, possibly due to empty logit arrays after filtering."
                    )

        logger_obj = getattr(trainer, "logger", None)
        experiment = getattr(logger_obj, "experiment", None)
        if experiment is not None and hasattr(experiment, "log"):
            experiment.log(log_dict)

    def _log_image(self, trainer: pl.Trainer, image: np.ndarray | Tensor, tag: str = "images/", prefix: str = "vis/"):
        logger_obj = getattr(trainer, "logger", None)
        experiment = getattr(logger_obj, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_image"):
            return

        if torch.is_tensor(image):
            image_pt = image.detach().cpu()
        else:
            image_pt = torch.from_numpy(image.copy())
        if image_pt.ndim == 2:
            image_pt = image_pt.unsqueeze(0)
        elif image_pt.ndim == 3 and image_pt.shape[-1] in {1, 3, 4}:
            image_pt = image_pt.permute(2, 0, 1)
        experiment.add_image(f"{prefix}{tag}", image_pt, global_step=trainer.global_step)

    @rank_zero_only
    def _save(self, trainer: pl.Trainer):
        out_dir = Path(trainer.default_root_dir) / "vis" / f"step_{trainer.global_step}"

        for key in ("stacked_front", "stacked_back", "stacked_logits"):
            for category, image in self.images.get(key, []):
                suffix = key.split("_")[-1]
                self.save_image(out_dir / f"{category}_{suffix}", image)
                if not self.upload_to_wandb:
                    self._log_image(trainer, image, f"{category}_{suffix}")

        for key in ("front", "back", "inputs", "logits", "depth", "color", "normals"):
            for i, image in enumerate(self.images[key]):
                category = self.data["categories"][i] if i < len(self.data["categories"]) else f"item_{i}"
                name = self.images["names"][i] if i < len(self.images["names"]) else f"item_{i}"
                path = out_dir / category / "images" / f"{name}_{key}"
                self.save_image(path, image)
                if not self.upload_to_wandb:
                    self._log_image(trainer, image, f"{category}_{key}")

        for key, values in (("meshes", self.data["meshes"]), ("inputs", self.data["inputs"])):
            for i, item in enumerate(values):
                category = self.data["categories"][i] if i < len(self.data["categories"]) else f"item_{i}"
                name = self.images["names"][i] if i < len(self.images["names"]) else f"item_{i}"
                path = out_dir / category / key / name
                self.save_mesh_or_pointcloud(path, item)

    def _show(self, sleep: float = 0.1):
        images = list()
        if self.meshes:
            if self.front:
                images.append(self.images["front"][0])
            elif self.back:
                images.append(self.images["back"][0])
        elif self.inputs:
            images.append(self.images["inputs"][0])
        if self.logits:
            images.append(self.images["logits"][0])
        if self.render:
            images.append(self.images["depth"][0])
            if "color" in self.render:
                images.append(self.images["color"][0])
            if "normals" in self.render:
                images.append(self.images["normals"][0])

        for image, handle in zip(images, self.handles, strict=False):
            handle.set_data(image)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(sleep)

    def _render_logits(self, points: np.ndarray, logits: np.ndarray) -> np.ndarray:
        assert len(points) == len(logits), "Number of points and logits must be equal"
        if len(points) == 0:
            return self.default_image

        colors = logits
        max_color = float(colors.max()) if colors.size else 0.0
        normalized_colors = colors / max_color if max_color > 0 else np.zeros_like(colors, dtype=np.float32)
        colormap = plt.get_cmap("plasma")
        colors = colormap(normalized_colors)
        return self.renderer(points, colors=colors[:, :3])["color"]

    def _render_pointcloud(self, points: np.ndarray, colors: np.ndarray | None = None) -> np.ndarray:
        return self.renderer(vertices=points, colors=colors)["color"]

    def _render_mesh(self, mesh: Trimesh, colors: np.ndarray | None = None) -> np.ndarray:
        return self.renderer(vertices=mesh.vertices, faces=mesh.faces, colors=colors)["color"]

    def _render(self, mesh: Trimesh, pcd: PointCloud | None = None, z_angle: float | None = None) -> np.ndarray:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return self.default_image

        @default_on_exception(self.default_image)
        def run(_mesh, _pcd, _z_angle):
            _mesh = _mesh.copy()
            min_val = _mesh.vertices[:, 1].min()

            _mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            _mesh.apply_translation([0, 0, -min_val])

            if _pcd:
                _pcd = _pcd.copy()
                _pcd.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
                _pcd.apply_translation([0, 0, -min_val])

            if _z_angle is not None:
                _mesh.apply_transform(trimesh.transformations.rotation_matrix(_z_angle, [0, 0, 1]))
                if _pcd:
                    _pcd.apply_transform(trimesh.transformations.rotation_matrix(_z_angle, [0, 0, 1]))

            colors = None
            if self.generator.predict_colors:
                colors = [_mesh.visual.vertex_colors, _pcd.visual.vertex_colors] if _pcd else _mesh.visual.vertex_colors
            normals = None
            if self.generator.estimate_normals:
                normals = [_mesh.vertex_normals, _pcd.vertex_normals] if _pcd else _mesh.vertex_normals
            extrinsic = inv_trafo(look_at(np.array([0, -1.5, 1.5]), _mesh.centroid))
            return self.renderer(
                vertices=[_mesh.vertices, _pcd.vertices] if _pcd else _mesh.vertices,
                faces=[_mesh.faces, None] if _pcd else _mesh.faces,
                colors=colors,
                normals=normals,
                extrinsic=extrinsic,
            )["color"]

        return run(mesh, pcd, z_angle)

    def _render_meshes(self, meshes: list[Trimesh], pcds: list[PointCloud] | None = None):
        if pcds is not None:
            if len(meshes) != len(pcds):
                raise ValueError("Number of meshes and pointclouds must match.")
            for mesh, pcd in zip(meshes, pcds, strict=True):
                self.data["meshes"].append(mesh)
                self.data["inputs"].append(pcd)
                if self.front:
                    self.images["front"].append(self._render(mesh, pcd, z_angle=np.pi / 4))
                if self.back:
                    self.images["back"].append(self._render(mesh, pcd, z_angle=np.pi / 4 + np.pi))
        else:
            for mesh in meshes:
                self.data["meshes"].append(mesh)
                if self.front:
                    self.images["front"].append(self._render(mesh, z_angle=np.pi / 4))
                if self.back:
                    self.images["back"].append(self._render(mesh, z_angle=np.pi / 4 + np.pi))

    def _render_items(self, items: list[BatchItem]):
        for item in items:
            if self.inputs and not self.meshes:
                inputs = item.get("inputs")
                if not isinstance(inputs, np.ndarray):
                    continue
                self.data["inputs"].append(inputs)
                self.images["inputs"].append(self._render_pointcloud(inputs))
            if self.logits:
                points = item.get("points")
                logits = item.get("logits")
                if not isinstance(points, np.ndarray) or not isinstance(logits, np.ndarray):
                    continue
                mask = logits >= np.log(self.threshold / (1 - self.threshold))
                points = points[mask]
                logits = logits[mask]
                self.data["points"].append(points)
                self.data["logits"].append(logits)
                self.images["logits"].append(self._render_logits(points, logits))
            depth = item.get("depth")
            if isinstance(depth, np.ndarray):
                self.images["depth"].append(depth)
            color = item.get("color")
            if isinstance(color, np.ndarray):
                self.images["color"].append(color)
            normals = item.get("normals")
            if isinstance(normals, np.ndarray):
                self.images["normals"].append(normals)

    def _prepare_batch(self, batch: BatchDict):
        raw_categories = batch.get("category.name")
        raw_indices = batch.get("index")
        if not isinstance(raw_categories, list) or not torch.is_tensor(raw_indices):
            return

        category_names = [name.split(",")[0] if "," in name else name for name in raw_categories]
        batch["category.name"] = category_names

        for index in range(len(raw_indices)):
            category_name = category_names[index]
            total_selected = len(self.items) + len(self._batch)
            if self.n_total is not None and total_selected >= self.n_total:
                continue
            if self.n_per_category is not None and self.counter[category_name] >= self.n_per_category:
                continue

            item: BatchItem = {}
            for key, value in batch.items():
                if "loss" in key:
                    continue
                if torch.is_tensor(value):
                    item[key] = value[index]
                elif isinstance(value, list):
                    item[key] = value[index]
            self._add_item(item)
            self.counter[category_name] += 1

    def _process_batch(self, model: pl.LightningModule):
        if not self._batch:
            return

        _batch = default_collate(self._batch)

        render_fn = getattr(model, "render", None)
        if self.render and callable(render_fn):
            for index in range(len(_batch["index"])):
                item = {k: v[index : index + 1] for k, v in _batch.items()}
                item = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in item.items()}

                images = render_fn(
                    item,
                    color="color" in self.render,
                    normals="normals" in self.render,
                    targets=not self.upload_to_wandb,
                    max_size=256 if self.upload_to_wandb else 512,
                )
                if not isinstance(images, dict):
                    continue
                for k, v in images.items():
                    if isinstance(v, Image.Image):
                        self._batch[index][k] = np.asarray(v)
                    elif isinstance(v, np.ndarray):
                        self._batch[index][k] = v

        if self.meshes or self.inputs:
            meshes = self.generate_batch(_batch, progress=self.progress)
            inputs = _batch.get("inputs")
            if not torch.is_tensor(inputs):
                return
            batch_inputs = []
            if inputs.ndim == 4 and inputs.size(1) == inputs.size(2) == inputs.size(3):
                grid = self.generator.query_points.unsqueeze(0).expand(inputs.size(0), -1, -1)
                mask = inputs.view(inputs.size(0), -1) == 1
                batch_inputs = [grid[i][mask[i]].cpu().numpy() for i in range(inputs.size(0))]
            elif hasattr(self.generator.model, "fps"):
                from libs import furthest_point_sample

                model_device = getattr(model, "device", inputs.device)
                sampled_inputs = furthest_point_sample(inputs.to(model_device), self.generator.model.fps)
                if torch.is_tensor(sampled_inputs):
                    sampled_inputs = sampled_inputs.detach().cpu().numpy()
                else:
                    sampled_inputs = np.asarray(sampled_inputs)
                batch_inputs = [sampled_inputs[i] for i in range(sampled_inputs.shape[0])]
            else:
                batch_inputs = [inputs[i].detach().cpu().numpy() for i in range(inputs.size(0))]

            for item, mesh, item_inputs in zip(self._batch, meshes, batch_inputs, strict=True):
                item["mesh"] = mesh
                item["inputs"] = item_inputs

    def _gather_items(self, world_size: int = 1):
        if world_size <= 1:
            self.items.extend(self._batch)
            self._batch.clear()
            return

        gathered_items: list[list[BatchItem] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered_items, self._batch)

        self.items.extend([item for sublist in gathered_items for item in (sublist or [])])
        self._batch.clear()

    @rank_zero_only
    def _visualize_items(self):
        sorted_items = sorted(self.items, key=lambda x: str(x.get("category.name", "")))
        categories = [str(item.get("category.name", "")) for item in sorted_items]
        self.data["categories"].extend(categories)
        self.images["names"].extend(categories)
        if self.logits or (self.inputs and not self.meshes) or self.render:
            self._render_items(sorted_items)
        if self.meshes:
            pcds: list[PointCloud] | None = None
            first_inputs = sorted_items[0].get("inputs")
            if isinstance(first_inputs, np.ndarray) and first_inputs.ndim == 2 and first_inputs.shape[1] == 3:
                pcds = [PointCloud(cast(np.ndarray, item["inputs"])) for item in sorted_items]
            meshes = [cast(Trimesh, item["mesh"]) for item in sorted_items]
            self._render_meshes(meshes, pcds if self.inputs else None)
        if self.handles:
            self._show()

        if self.n_per_category and self.n_per_category >= 4:
            self.images["stacked_front"] = list()
            self.images["stacked_back"] = list()
            self.images["stacked_logits"] = list()
            for i in range(0, len(self.images["front"]), self.n_per_category):
                if i + self.n_per_category <= len(self.images["front"]):
                    images_to_stack = [
                        cast(np.ndarray | Image.Image, image)
                        for image in self.images["front"][i : i + self.n_per_category]
                    ]
                    stacked_image = stack_images(images_to_stack)
                    category = self.data["categories"][i]
                    self.images["stacked_front"].append((category, stacked_image))
            for i in range(0, len(self.images["back"]), self.n_per_category):
                if i + self.n_per_category <= len(self.images["back"]):
                    images_to_stack = [
                        cast(np.ndarray | Image.Image, image)
                        for image in self.images["back"][i : i + self.n_per_category]
                    ]
                    stacked_image = stack_images(images_to_stack)
                    category = self.data["categories"][i]
                    self.images["stacked_back"].append((category, stacked_image))
            for i in range(0, len(self.images["logits"]), self.n_per_category):
                if i + self.n_per_category <= len(self.images["logits"]):
                    images_to_stack = [
                        cast(np.ndarray | Image.Image, image)
                        for image in self.images["logits"][i : i + self.n_per_category]
                    ]
                    stacked_image = stack_images(images_to_stack)
                    category = self.data["categories"][i]
                    self.images["stacked_logits"].append((category, stacked_image))

    def run(self, trainer: pl.Trainer) -> bool:
        if super().run(trainer) and not self.done(trainer.world_size):
            return True
        return False

    @default_on_exception()
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: BatchDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.run(trainer):
            return

        self._prepare_batch(batch)
        self._process_batch(cast(Any, pl_module).model)

    @default_on_exception()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._gather_items(trainer.world_size)
        if self.items:
            self._visualize_items()
            if self.upload_to_wandb:
                self._upload(trainer)
            else:
                self._save(trainer)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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
        self.counter.clear()
        self.eval_count += 1
        self._done = False
