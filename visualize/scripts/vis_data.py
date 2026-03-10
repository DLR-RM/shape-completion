from functools import partial
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm

from dataset import SaveData, SharedDataLoader, SharedDataset, Visualize, get_dataset
from utils import (
    get_num_workers,
    log_optional_dependency_summary,
    resolve_save_dir,
    setup_config,
    setup_logger,
    suppress_known_optional_dependency_warnings,
)

logger = setup_logger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg = setup_config(cfg, seed_workers=True if cfg.vis.split == "train" else False)
    suppress_known_optional_dependency_warnings()
    log_optional_dependency_summary(logger, cfg)

    split = cfg.vis.split
    if cfg.vis.show:
        logger.debug_level_1(f"Visualizing {split} split.")
    dataset = get_dataset(cfg, splits=(split,))[split]
    dataset_any = cast(Any, dataset)
    logger.debug_level_1(f"Dataset: name={dataset_any.name}, len={len(dataset_any)}")

    if cfg.vis.index is not None:
        if isinstance(cfg.vis.index, int):
            idx_list = [cfg.vis.index]
        elif isinstance(cfg.vis.index, str):
            idx_list = [int(s) for s in cfg.vis.index.split(",")]
        else:
            idx_list = [int(i) for i in cfg.vis.index]

    if cfg.vis.use_loader:
        num_workers = get_num_workers(cfg.load.num_workers)

        collate_fn = None
        try:
            from train import get_collate_fn

            collate_fn = get_collate_fn(cfg, split, cfg[split].batch_size or 1)
        except ImportError as e:
            logger.warning(f"Unable to import train module, heterogeneous batching won't be available: {e}")

        generator = torch.Generator().manual_seed(cfg.misc.seed)
        sampler = (
            WeightedRandomSampler(weights=dataset_any.category_weights, num_samples=len(dataset_any), generator=generator)
            if cfg.load.weighted
            else None
        )
        if sampler is not None:
            logger.debug_level_1("Using weighted sampler for training")
        loader = DataLoader
        if cfg.data.cache:
            loader = partial(
                SharedDataLoader,
                hash_items=cfg.data.hash_items or cfg.data.num_files[split],
                share_arrays=cfg.data.share_memory,
            )
        if cfg.train.fast_dev_run and hasattr(dataset_any, "objects"):
            dataset_any.objects = np.random.choice(dataset_any.objects, cfg.train.fast_dev_run, replace=False)

        if cfg.vis.index is not None:
            sampler = SubsetRandomSampler(idx_list)
            logger.debug_level_1(f"Using SubsetRandomSampler for indices: {idx_list}")

        loader = loader(
            dataset_any,
            batch_size=1 if split == "test" else cfg[split].batch_size,
            num_workers=0 if cfg.data.test_ds[0] == "bop" else num_workers,
            collate_fn=collate_fn,
            shuffle=cfg[split].shuffle and sampler is None,
            sampler=sampler,
            pin_memory=cfg.load.pin_memory,
            generator=generator,
            prefetch_factor=cfg.load.prefetch_factor if num_workers else None,
            persistent_workers=num_workers > 0,
        )

    if cfg.vis.show:
        has_points = (
            cfg.files.points[split]
            or (cfg.points.from_mesh and cfg[split].num_query_points)
            or cfg.get("filter", False)
        )
        vis = Visualize(
            show_inputs=cfg.vis.inputs if cfg.inputs.type is not None else False,
            show_occupancy=bool(cfg.vis.occupancy and has_points),
            show_points=bool(cfg.vis.points and has_points),
            show_frame=cfg.vis.frame,
            show_pointcloud=cfg.vis.pointcloud,
            show_mesh=cfg.vis.mesh,
            show_box=cfg.vis.box,
            show_bbox=cfg.vis.bbox,
            show_cam=cfg.vis.cam,
            threshold=cfg.implicit.threshold,
            sdf=cfg.implicit.sdf,
            padding=cfg.norm.padding,
            scale_factor=cfg.norm.scale_factor,
            frame=cfg.data.frame,
            convention=cfg.data.convention,
        )

    if cfg.vis.save:
        save_dir = resolve_save_dir(cfg) / "data" / split
        save_data = SaveData(save_dir, threshold=cfg.implicit.threshold, sdf=cfg.implicit.sdf)

    if cfg.vis.use_loader:
        for epoch in range(cfg.train.epochs):
            for batch in tqdm(
                loader, desc=f"Epoch {epoch}/{cfg.train.epochs}", disable=cfg.log.verbose > 2 or not cfg.log.progress
            ):
                if cfg.vis.show or cfg.vis.save:
                    for index in range(len(batch["index"])):
                        item = {k: v[index] for k, v in batch.items() if k != "points.path"}

                        if item.get("inputs.skip", False):
                            continue

                        if cfg.vis.save:
                            save_data(item)
                        if cfg.vis.show:
                            for k, v in item.items():
                                if torch.is_tensor(v):
                                    item[k] = v.numpy()
                            vis(item)
    else:
        if cfg.data.cache:
            shared_dict = dict()
            shared_hash_map = dict() if cfg.data.hash_items or cfg.data.num_files[split] else None
            dataset_any = SharedDataset(dataset_any, shared_dict, shared_hash_map)
            if cfg.data.share_memory:
                logger.warning("Shared memory is only supported with SharedDataLoader, not with SharedDataset.")

        names = np.random.permutation(np.arange(len(dataset_any))) if cfg[split].shuffle else range(len(dataset_any))
        names = names[: cfg.train.fast_dev_run] if cfg.train.fast_dev_run else names
        if cfg.vis.index is not None:
            names = idx_list
        for epoch in range(cfg.train.epochs):
            for _counter, index in enumerate(
                tqdm(
                    names, desc=f"Epoch {epoch}/{cfg.train.epochs}", disable=cfg.log.verbose > 2 or not cfg.log.progress
                )
            ):
                try:
                    item = dataset_any[index]
                    if item.get("inputs.skip", False):
                        continue

                    assert item is not None, f"None item at index {index}"
                    if any(value is None for value in item.values()):
                        logger.error(f"None value in item at index {index}")
                        print(item)

                    if cfg.vis.show:
                        for k, v in item.items():
                            if torch.is_tensor(v):
                                item[k] = v.numpy()

                        vis(item)
                    if cfg.vis.save:
                        save_data(item)
                except Exception as e:
                    logger.exception(e)
                    if cfg.vis.show:
                        raise e


if __name__ == "__main__":
    main()
