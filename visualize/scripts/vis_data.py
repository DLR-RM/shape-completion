import time
from functools import partial

import hydra
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import Visualize, get_dataset
from utils import setup_logger, save_mesh, get_num_workers, setup_config

logger = setup_logger(__name__)

TRAIN_AVAILABLE = True
try:
    from train import heterogeneous_batching, cache, preload
except ImportError:
    logger.warning("Unable to import train module. Train functionality is not available.")
    TRAIN_AVAILABLE = False


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    setup_config(cfg, workers=True if cfg.vis.split == "train" else False)

    split = cfg.vis.split
    logger.debug(f"Visualizing {split} split.")
    pcd_nets = ["realnvp", "psgn", "pcn", "snowflakenet", "shapeformer", "pointnet"]
    cls_only = (cfg.cls.num_classes is not None or cfg.seg.num_classes is not None) and not cfg.cls.occupancy
    load_points = cfg.model.arch not in pcd_nets and not cls_only
    if not load_points:
        cfg.points.subsample = False
        cfg.points.crop = False
    dataset = get_dataset(cfg,
                          load_points=load_points,
                          load_pointcloud=cfg.model.arch in pcd_nets,
                          splits=(split,))[split]

    if TRAIN_AVAILABLE:
        if cfg.data.cache:
            dataset = cache(cfg, dataset)
        if cfg.load.preload:
            dataset = preload(cfg, dataset)

    if cfg.vis.use_loader:
        collate = None
        if cfg.inputs.num_points == 0 and not (cfg.inputs.voxelize or cfg.inputs.bps.apply):
            if not TRAIN_AVAILABLE:
                raise ImportError("Unable to heterogeneous batching without `train` module."
                                  "Set `vis.use_loader` to False.")
            collate = partial(heterogeneous_batching, res_modifier=cfg.load.res_modifier)
        num_workers = get_num_workers(cfg.load.num_workers)
        generator = torch.Generator().manual_seed(cfg.misc.seed)
        sampler = WeightedRandomSampler(weights=dataset.category_weights,
                                        num_samples=len(dataset),
                                        generator=generator) if cfg.load.weighted else None
        if sampler is not None:
            logger.info(f"Using weighted sampler for training")
        loader = DataLoader(dataset,
                            batch_size=1 if split == "test" else cfg[split].batch_size,
                            num_workers=0 if cfg.data.test_ds[0] == "bop" else num_workers,
                            collate_fn=collate,
                            shuffle=True if split == "train" and not cfg.load.weighted else False,
                            sampler=sampler,
                            pin_memory=cfg.load.pin_memory,
                            generator=generator,
                            prefetch_factor=cfg.load.prefetch_factor if num_workers else None,
                            persistent_workers=True if num_workers and split != "test" else False)

    threshold = 0 if cfg.implicit.sdf and cfg.implicit.threshold == 0.5 else cfg.implicit.threshold
    vis = Visualize(show_inputs=cfg.vis.inputs,
                    show_occupancy=cfg.vis.occupancy,
                    show_points=cfg.vis.points,
                    show_frame=cfg.vis.frame,
                    show_pointcloud=cfg.vis.pointcloud,
                    show_mesh=cfg.vis.mesh,
                    show_box=cfg.vis.box,
                    show_bbox=cfg.vis.bbox,
                    show_cam=cfg.vis.cam,
                    threshold=threshold,
                    sdf=cfg.implicit.sdf,
                    padding=cfg.norm.padding)
    # cam_forward=(0, 0, -1) if cfg.inputs.frame == "cam" else (0, 0, 1),
    # cam_up=(0, -1, 0) if cfg.inputs.frame == "cam" else (0, 1, 0))

    if cfg.vis.use_loader:
        for epoch in range(cfg.train.epochs):
            for batch in tqdm(loader, disable=cfg.log.verbose, desc=f"Epoch {epoch}/{cfg.train.epochs}"):
                if cfg.vis.show:
                    for index in range(len(batch["index"])):
                        item = {k: v[index] for k, v in batch.items()}

                        if item.get("inputs.skip", False):
                            continue

                        for k, v in item.items():
                            if isinstance(v, torch.Tensor):
                                item[k] = v.numpy()
                        vis(item)
            time.sleep(3)
    else:
        iterator = np.random.permutation(np.arange(len(dataset))) if split == "train" else range(len(dataset))
        for epoch in range(cfg.train.epochs):
            for index in tqdm(iterator, desc=f"{split.capitalize()} split", disable=cfg.log.verbose):
                try:
                    item = dataset[index]

                    if item.get("inputs.skip", False):
                        continue

                    assert item is not None, f"None item at index {index}"
                    if any(value is None for value in item.values()):
                        logger.error(f"None value in item at index {index}")
                        print(item)

                    if cfg.vis.show:
                        logger.debug(f"\nIndex: {index}\nObject name:{item['inputs.name']}\nCategory ID: {item['category.id']}\nCategory name: {item['category.name']}\nCategory index: {item['category.index']}\n")
                        vis(item)
                    if cfg.vis.save:
                        save_mesh(f"~/{item['inputs.name']}_mesh.obj", item["mesh.vertices"], item["mesh.triangles"])
                        trimesh.PointCloud(item["points"][item["points.occ"] == 2]).export(f"~/{item['inputs.name']}_uncertain.ply")
                        trimesh.PointCloud(item["points"][item["points.occ"] == 1]).export(f"~/{item['inputs.name']}_occupied.ply")
                        trimesh.PointCloud(item["points"][item["points.occ"] == 0]).export(f"~/{item['inputs.name']}_free.ply")
                except Exception as e:
                    logger.exception(e)


if __name__ == "__main__":
    main()
