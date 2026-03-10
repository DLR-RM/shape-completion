from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from dataset.src import coco_collate
from utils import (
    ItemTypes,
    adjust_intrinsic,
    apply_trafo,
    bbox_from_mask,
    crop_and_resize_image,
    depth_to_image,
    points_to_depth,
    resolve_path,
    setup_logger,
)

logger = setup_logger(__name__)

PYTORCH3D_AVAILABLE = True
try:
    from pytorch3d.structures import Pointclouds
except ImportError:
    logger.warning("The 'PyTorch3D' module is not installed. Heterogeneous batching will not be available.")
    PYTORCH3D_AVAILABLE = False


def _debug_level_1(msg: str) -> None:
    debug_fn = getattr(logger, "debug_level_1", logger.debug)
    debug_fn(msg)


def _item_to_tensor(value: ItemTypes) -> Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    raise TypeError(f"Expected ndarray or Tensor, got {type(value).__name__}")


def save_best_model(checkpoint_path: str | Path, save_dir: Path) -> dict[str, Tensor]:
    model_best = torch.load(checkpoint_path, weights_only=False)
    state_dict = {k.replace("model.", ""): v for k, v in model_best["state_dict"].items()}
    _debug_level_1("Saving BEST weights")
    torch.save({"model": state_dict}, save_dir / "model_best.pt")
    return state_dict


def save_ema_model(
    trainer: pl.Trainer, model: pl.LightningModule, save_dir: Path, state_dict: dict[str, Tensor] | None = None
):
    for optimizer in trainer.optimizers:
        state_dict_ema = None
        optimizer_any = cast(Any, optimizer)
        if hasattr(optimizer_any, "swap_ema_weights"):
            _debug_level_1("Saving EMA weights")
            with optimizer_any.swap_ema_weights():
                state_dict_ema = model.state_dict()
        else:
            ema_model = getattr(cast(Any, model), "ema_model", None)
            if ema_model is not None:
                state_dict_ema = cast(dict[str, Tensor], ema_model.module.orig_mod.state_dict())

        if state_dict_ema is not None:
            state_dict_ema = {k.replace("model.", ""): v for k, v in state_dict_ema.items()}
            if state_dict is not None:
                for (k1, v1), (k2, v2) in zip(state_dict.items(), state_dict_ema.items(), strict=False):
                    assert k1 == k2, f"Model and EMA model have different keys: {k1} != {k2}"
                    if torch.equal(v1.cpu(), v2.cpu()):
                        logger.warning(
                            f"EMA and model params {k1} are equal: {v1.mean().item():.5f} == {v2.mean().item():.5f}"
                        )
            _debug_level_1("Saving EMA weights")
            torch.save({"model": state_dict_ema}, save_dir / "model_ema.pt")


def weight_norm(module: nn.Module, norm_type: float | int | str, group_separator: str = "/") -> dict[str, float]:
    """Compute each parameter's weight norm and their overall norm.

    The overall norm is computed over all weights together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the weight norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's weight and
            a special entry for the total p-norm of the weights viewed
            as a single vector.

    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    norms = {
        f"weight_{norm_type}_norm{group_separator}{name}": p.data.norm(norm_type).item()
        for name, p in module.named_parameters()
    }
    if norms:
        total_norm = torch.tensor(list(norms.values())).norm(norm_type).item()
        norms[f"weight_{norm_type}_norm_total"] = total_norm
    return norms


def common_collate(batch: list[dict[str, ItemTypes]]) -> dict[str, str | Tensor]:
    common_keys = set.intersection(*[set(item.keys()) for item in batch])
    try:
        return default_collate([{key: item[key] for key in common_keys} for item in batch])
    except RuntimeError:
        for item in batch:
            for k, v in item.items():
                if isinstance(v, (np.ndarray, Tensor)):
                    print(k, v.shape)
                elif isinstance(v, list):
                    print(k, len(v))
                else:
                    print(k, v)
        raise


def heterogeneous_collate(
    batch: list[dict[str, ItemTypes]], res_modifier: int = 1, keys: list[str] | None = None
) -> dict[str, str | Tensor]:
    if not PYTORCH3D_AVAILABLE:
        raise ImportError("The 'PyTorch3D' module is required for heterogeneous batching.")

    if keys is None or "inputs" in keys:
        inputs = [_item_to_tensor(item["inputs"]) for item in batch]
        normals: list[Tensor] | None = None
        normals_item = batch[0].get("inputs.normals")
        if isinstance(normals_item, (np.ndarray, Tensor)):
            normals = [_item_to_tensor(item["inputs.normals"]) for item in batch]

        features: list[Tensor] | None = None
        features_item = batch[0].get("inputs.labels")
        if isinstance(features_item, (np.ndarray, Tensor)) and len(features_item) == len(inputs[0]):
            features = [_item_to_tensor(item["inputs.labels"]).unsqueeze(1) for item in batch]

        colors: list[Tensor] | None = None
        colors_item = batch[0].get("inputs.colors")
        if isinstance(colors_item, (np.ndarray, Tensor)) and len(colors_item) == len(inputs[0]):
            colors = [_item_to_tensor(item["inputs.colors"]) for item in batch]
            if features is not None:
                features = [torch.cat((f, c), dim=1) for f, c in zip(features, colors, strict=False)]
            else:
                features = colors

        pcds = Pointclouds(inputs, normals=normals, features=features).subsample(
            640 // res_modifier * 480 // res_modifier
        )
        inputs_pad = cast(Tensor, pcds.points_padded())
        lengths_in = cast(Tensor, pcds.num_points_per_cloud())
        Nmax_in = inputs_pad.size(1)
        mask_in = torch.arange(Nmax_in, device=inputs_pad.device)[None, :] < lengths_in.to(inputs_pad.device)[:, None]

        if normals and features:
            normals_pad = cast(Tensor, pcds.normals_padded())
            features_pad = cast(Tensor, pcds.features_padded())
            for x, (i, n, f, m, L) in enumerate(
                zip(inputs_pad, normals_pad, features_pad, mask_in, lengths_in, strict=False)
            ):
                batch[x]["inputs"] = i
                batch[x]["inputs.normals"] = n
                batch[x]["inputs.mask"] = m
                batch[x]["inputs.lengths"] = L
                if f.size(1) == 1:
                    batch[x]["inputs.labels"] = f.squeeze(1)
                elif f.size(1) == 3:
                    batch[x]["inputs.colors"] = f
                else:
                    batch[x]["inputs.labels"] = f[:, 0].long()
                    batch[x]["inputs.colors"] = f[:, 1:]
        elif normals:
            normals_pad = cast(Tensor, pcds.normals_padded())
            for x, (i, n, m, L) in enumerate(zip(inputs_pad, normals_pad, mask_in, lengths_in, strict=False)):
                batch[x]["inputs"] = i
                batch[x]["inputs.normals"] = n
                batch[x]["inputs.mask"] = m
                batch[x]["inputs.lengths"] = L
        elif features:
            features_pad = cast(Tensor, pcds.features_padded())
            for x, (i, f, m, L) in enumerate(zip(inputs_pad, features_pad, mask_in, lengths_in, strict=False)):
                batch[x]["inputs"] = i
                batch[x]["inputs.mask"] = m
                batch[x]["inputs.lengths"] = L
                if f.size(1) == 1:
                    batch[x]["inputs.labels"] = f.squeeze(1)
                elif f.size(1) == 3:
                    batch[x]["inputs.colors"] = f
                else:
                    batch[x]["inputs.labels"] = f[:, 0].long()
                    batch[x]["inputs.colors"] = f[:, 1:]
        else:
            for x, (i, m, L) in enumerate(zip(inputs_pad, mask_in, lengths_in, strict=False)):
                batch[x]["inputs"] = i
                batch[x]["inputs.mask"] = m
                batch[x]["inputs.lengths"] = L

    if (keys is None or "points" in keys) and batch[0].get("points") is not None:
        raise NotImplementedError("Heterogeneous batching for 'points' is not supported yet.")
        points = [torch.from_numpy(item["points"]) for item in batch]
        occ = [torch.from_numpy(item["points.occ"]).unsqueeze(1) for item in batch]
        labels = batch[0].get("points.labels")
        if labels is not None and len(labels) == len(points[0]):
            occ = [torch.cat((o, label), dim=1) for o, label in zip(occ, labels, strict=False)]

        pcds = Pointclouds(points, features=occ)
        points_pad = pcds.points_padded()
        features_pad = pcds.features_padded()
        lengths = pcds.num_points_per_cloud()
        Nmax = points_pad.size(1)
        mask = torch.arange(Nmax, device=points_pad.device)[None, :] < lengths.to(points_pad.device)[:, None]

        for x, (p, f, m, L) in enumerate(zip(points_pad, features_pad, mask, lengths, strict=False)):
            batch[x]["points"] = p
            batch[x]["points.mask"] = m
            batch[x]["points.lengths"] = L
            if f.size(1) == 1:
                # Preserve dtype/values (0/1/2/etc.)
                occ_x = f.squeeze(1)
                # Optional: set ignore value on padded rows
                # occ_x = occ_x.masked_fill(~m, -1)  # if you want an explicit ignore sentinel
                batch[x]["points.occ"] = occ_x
            else:
                occ_x = f[:, 0]
                # Optional: set ignore for padded rows:
                # occ_x = occ_x.masked_fill(~m, -1)
                batch[x]["points.occ"] = occ_x
                batch[x]["points.labels"] = f[:, 1].long()

    return common_collate(batch)


def get_test_dataset(cfg: DictConfig) -> Dataset | None:
    try:
        from inference import get_input_data_from_point_cloud, get_point_cloud, get_rot_from_extrinsic, remove_plane
    except ImportError:
        logger.exception("Could not import inference functions. Is the `inference` submodule present?")
        return

    class TestDataset(Dataset):
        def __init__(self, show: bool = cfg.vis.show):
            test_dir = resolve_path(cfg.dirs[cfg.test.dir])
            paths = sorted(test_dir.rglob(cfg.test.filename))
            num_paths = int(np.floor(np.sqrt(len(paths))) ** 2)
            self.paths = paths[:num_paths]
            assert len(self.paths) > 0, f"No {cfg.test.filename} files found in {test_dir}"
            _debug_level_1(f"Found {len(self.paths)} {cfg.test.filename} files in {test_dir}")
            self.show = show

        def __getitem__(self, index: int) -> dict[str, str | int | float | np.ndarray]:
            path = self.paths[index]
            extrinsic = None
            camera_path = path.parent / "camera.npy"
            if camera_path.is_file():
                extrinsic = np.load(camera_path)
                if "depth" in path.stem and path.suffix == ".npy":
                    pcd, intrinsic, extrinsic = get_point_cloud(path, extrinsic=extrinsic)
                else:
                    pcd, intrinsic, _ = get_point_cloud(path)
                    rot_x, rot_y, _rot_z = get_rot_from_extrinsic(extrinsic)
                    extrinsic[0, 3] = 0
                    extrinsic[1, 3] = 0.25
                    extrinsic[2, 3] = 0.75
                    extrinsic[:3, :3] = (rot_x @ rot_y).T
            else:
                pcd, intrinsic, _ = get_point_cloud(path)

            pcds, plane_model = remove_plane(pcd)
            pcd = pcds[0]
            width, height = 640, 480
            inputs = np.asarray(pcd.points)

            offset = np.zeros(3)
            scale = 1.0
            if cfg.inputs.project:
                offset_y = 0
                if cfg.norm.true_height:
                    a, b, c, d = plane_model
                    points = np.asarray(pcd.points)
                    index = int(np.argmin(points[:, 1]))
                    point = points[index]
                    offset_y = (a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(a**2 + b**2 + c**2)
                scale = cfg.norm.scale if cfg.aug.scale is None else cfg.aug.scale
                inputs, offset, scale = get_input_data_from_point_cloud(
                    pcd,
                    num_input_points=cfg.inputs.num_points,
                    offset_y=offset_y,
                    center=cfg.norm.center,
                    scale=scale,
                    crop=(1 + cfg.norm.padding) / 2,
                    voxelize=cfg.inputs.voxelize,
                    padding=cfg.norm.padding,
                    show=self.show,
                )
            elif intrinsic is not None:
                if extrinsic is not None:
                    inputs = apply_trafo(inputs, extrinsic)
                depth = points_to_depth(inputs, intrinsic, width, height)
                depth = np.asarray(depth)
                height, width = depth.shape
                if cfg.inputs.crop:
                    bbox = bbox_from_mask(depth, padding=0.1)
                    depth = crop_and_resize_image(depth, box=bbox)
                    depth = np.asarray(depth)
                    intrinsic = adjust_intrinsic(intrinsic, width, height, box=bbox)
                    height, width = depth.shape
                if cfg.inputs.resize:
                    depth = crop_and_resize_image(depth, size=cfg.inputs.resize, interpolation="nearest")
                    depth = np.asarray(depth)
                    intrinsic = adjust_intrinsic(intrinsic, width, height, size=cfg.inputs.resize)
                    height, width = depth.shape
                inputs = depth
                if self.show:
                    image = depth_to_image(np.asarray(depth))
                    plt.imshow(image)
                    plt.show()

            data = {
                "inputs": inputs,
                "inputs.depth": inputs,
                "inputs.width": width,
                "inputs.height": height,
                "inputs.path": str(path),
                "inputs.norm_offset": offset,
                "inputs.norm_scale": scale,
            }
            if intrinsic is not None:
                data["inputs.intrinsic"] = np.asarray(intrinsic, dtype=np.float32)
            if extrinsic is not None:
                data["inputs.extrinsic"] = np.asarray(extrinsic, dtype=np.float32)
            return data

        def __len__(self) -> int:
            return len(self.paths)

    return TestDataset()


def get_collate_fn(cfg: DictConfig, split: str = "train", batch_size: int | None = None) -> Callable | None:
    if batch_size == 1:
        if cfg.data.split:
            return partial(coco_collate, list_keys={"inputs.name", "mesh.vertices", "mesh.triangles", "pointcloud"})
        return None

    collate = common_collate
    hb_keys = list()
    voxel_or_bps = cfg.inputs.voxelize or cfg.inputs.bps.num_points
    if cfg.inputs.project and not isinstance(cfg.inputs.num_points, int) and not voxel_or_bps:
        hb_keys.append("inputs")
    if (cfg[split].num_query_points in ["random", None] or not cfg.points.subsample) and not cfg.points.voxelize:
        hb_keys.append("points")
    if hb_keys:
        _debug_level_1(f"Using heterogeneous batching for {hb_keys}")
        collate = partial(heterogeneous_collate, res_modifier=cfg.load.res_modifier, keys=hb_keys)

    ds_name = cfg.data[f"{split}_ds"][0]
    collate_3d = cfg.get("load_3d") and cfg.get("collate_3d") in ["list", None]
    not_stack_2d = not cfg.get("stack_2d") or (cfg.get("stack_2d") and cfg.inputs.type == "rgbd")
    if ds_name == "coco" or ("tabletop" in ds_name and (not_stack_2d or collate_3d)):
        if collate_3d:
            collate = partial(coco_collate, list_keys={"points", "points.occ", "points.indices"})
        else:
            collate = coco_collate

    return collate
