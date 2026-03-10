import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import hydra
import lightning
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchinfo import summary
from tqdm import tqdm, trange

from dataset import BOP, YCB, CocoInstanceSegmentation, GraspNetEval, SaveData, TableTop, get_dataset
from models import AutoregressiveModel, DiffusionModel, GridDiffusionModel, UNetModel, get_model, patch_attention
from utils import (
    get_num_workers,
    log_optional_dependency_summary,
    resolve_save_dir,
    save_mesh,
    setup_config,
    setup_logger,
    suppress_known_optional_dependency_warnings,
    to_numpy,
    to_tensor,
)
from visualize import Generator

from ..src.utils import overwrite_results

logger = setup_logger(__name__)


def _debug_level_1(message: str) -> None:
    debug_fn = getattr(logger, "debug_level_1", logger.debug)
    debug_fn(message)


def _to_int(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def _shorten_path_filename(path: Path, max_name_len: int = 200) -> Path:
    """
    Ensure the final path component (filename) does not exceed filesystem limits.
    If too long, truncate the visible name and append a short hash to preserve uniqueness.

    Only the filename (stem + suffix) is modified; directories remain unchanged.
    """
    name = path.stem
    suffix = path.suffix
    # Fast path if already within limit (approximate in characters, safe under 255 bytes)
    if len(name) + len(suffix) <= max_name_len:
        return path

    # Sanitize for filesystem friendliness while preserving most characters
    safe_name = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_"))
    if not safe_name:
        safe_name = "file"

    import hashlib

    digest = hashlib.sha1(name.encode("utf-8", errors="ignore")).hexdigest()[:10]
    keep = max(1, max_name_len - len(digest) - 1 - len(suffix))
    new_name = f"{safe_name[:keep]}_{digest}{suffix}"
    return path.with_name(new_name)


def process_item(
    cfg: DictConfig,
    item: dict[str, Any],
    feature: Tensor | list[Tensor] | dict[str, Tensor] | None,
    dataset: Any,
    generator: Generator,
    obj_counter: defaultdict[str, int],
    vis_dir: Path,
    meshes_dir: Path,
    inputs_dir: Path,
):
    index = _to_int(item["index"])
    if isinstance(dataset, (CocoInstanceSegmentation, TableTop, GraspNetEval)):
        category_name = item["category.name"]
        obj_category = item["category.id"]
        obj_name = item["inputs.name"]
        if isinstance(obj_category, float):
            obj_category = int(obj_category)
    else:
        if isinstance(dataset, YCB):
            if dataset.load_real_data:
                obj_dict = dataset.objects[index // dataset.num_images]
            else:
                obj_dict = dataset.objects[index % len(dataset.objects)]
            obj_name = obj_dict["name"] + "_" + str(3 * index)
        else:
            obj_dict = dataset.objects[index]
            obj_name = obj_dict["name"]
            if cfg.data.num_files.test > 1 or cfg.data.num_shards.test > 1:
                obj_name = obj_dict["name"] + "_" + str(index)
        obj_category = obj_dict["category"]
        category_name = dataset.metadata[obj_category]["name"]
        category_name = category_name.split(",")[0] if "," in category_name else category_name

    if isinstance(obj_category, list):
        obj_instances = obj_counter[str(obj_category[0])]
    else:
        obj_instances = obj_counter[str(obj_category)]
    n_categories = len(getattr(dataset, "categories", []))
    n_instances = _to_int(cfg.vis.num_instances) if cfg.vis.num_instances else max(2, 10 // max(n_categories, 1))
    copy_inputs = obj_instances < min(4, n_instances)

    # Skip if we only want to visualize a fixed number of instances per category
    if cfg.vis.num_instances and obj_instances >= n_instances:
        return

    category_vis_dir: Any = vis_dir
    category_inputs_dir: Any = inputs_dir
    category_meshes_dir: Any = meshes_dir

    if isinstance(obj_category, (str, int)) and isinstance(category_name, str):
        category_vis_dir = vis_dir / "_".join([str(obj_category), category_name])
        if not category_vis_dir.is_dir():
            category_vis_dir.mkdir()
    elif isinstance(obj_category, list) and isinstance(category_name, list):
        category_vis_dir = []
        for cat, name in zip(obj_category, category_name, strict=False):
            cat_dir = vis_dir / "_".join([cat, name])
            if not cat_dir.is_dir():
                cat_dir.mkdir()
            category_vis_dir.append(cat_dir)

    if isinstance(obj_category, (str, int)):
        category_inputs_dir = inputs_dir / str(obj_category)
        if not category_inputs_dir.is_dir():
            category_inputs_dir.mkdir()
    elif isinstance(obj_category, list):
        category_inputs_dir = []
        for cat in obj_category:
            cat_dir = inputs_dir / cat
            if not cat_dir.is_dir():
                cat_dir.mkdir()
            category_inputs_dir.append(cat_dir)

    if isinstance(obj_category, (str, int)):
        category_meshes_dir = meshes_dir / str(obj_category)
        if not category_meshes_dir.is_dir():
            category_meshes_dir.mkdir()
    elif isinstance(obj_category, list):
        category_meshes_dir = []
        for cat in obj_category:
            cat_dir = meshes_dir / cat
            if not cat_dir.is_dir():
                cat_dir.mkdir()
            category_meshes_dir.append(cat_dir)

    out_dict = dict()
    """
    pcd_path = category_pcds_dir / f"{obj_name}.ply"
    
    if not cfg.points.voxelize and not isinstance(model, (DiffusionModel, AutoregressiveModel)):
        key = "logits"
        if cfg.seg.num_classes is not None:
            key = "seg_logits"
            if cfg.seg.inputs and cfg.seg.points:
                key = "aux_logits"
        with fabric.autocast():
            generator.generate_pcd(item, key, show=cfg.vis.show).export(pcd_path)
        out_dict["pcd"] = pcd_path
    """

    inv_extrinsic = np.asarray(item.get("inputs.inv_extrinsic", np.eye(4)))
    if not np.allclose(inv_extrinsic, np.eye(4)):
        extrinsic = np.asarray(item.get("inputs.extrinsic", np.eye(4)))
        inv_extrinsic = inv_extrinsic @ extrinsic
        inv_extrinsic[:3, 3] = 0
        np.save(category_meshes_dir / f"{obj_name}.npy", inv_extrinsic)

    if cfg.vis.save:
        save_dir = resolve_save_dir(cfg) / "data" / cfg.test.split
        save_data = SaveData(save_dir)

    grid = cast(np.ndarray | dict[int, np.ndarray] | list[np.ndarray], item["grid"])
    mesh = generator.extract_meshes(grid=grid, feature=feature, item=cast(dict[str, Any], item))
    if isinstance(mesh, list):
        out_dict["mesh"] = list()
        for i, m in enumerate(mesh):
            # Use aligned instance index (from GT-aligned Hungarian matching) when
            # available so that mesh colors and filenames stay consistent with the
            # instseg PNGs, which preserve the aligned instance order.
            metadata = cast(dict[str, Any], getattr(m, "metadata", {}) or {})
            inst_idx = cast(int, metadata.get("instance_idx", i))
            if isinstance(obj_name, list):
                mp = category_meshes_dir[inst_idx] / f"{obj_name[inst_idx]}.ply"
            elif inst_idx:
                mp = category_meshes_dir / (f"{obj_name}_{inst_idx}.ply")
            else:
                mp = category_meshes_dir / f"{obj_name}.ply"
            mp = _shorten_path_filename(mp)
            visual = cast(Any, m.visual)
            vertex_colors = cast(np.ndarray | None, getattr(visual, "vertex_colors", None))
            has_vertex_colors = vertex_colors is not None and len(np.unique(vertex_colors[:, :3])) > 1
            colors = vertex_colors if has_vertex_colors else None
            normals = m.vertex_normals if generator.estimate_normals else None
            save_mesh(mp, m.vertices, m.faces, colors, normals)
            out_dict["mesh"].append(mp)
            if cfg.vis.show:
                if not has_vertex_colors:
                    cast(Any, m.visual).vertex_colors = (0, 127, 127)
                m.show(smooth=False, caption=category_name)
            if cfg.vis.save:
                filename = str(index).zfill(8) + "_pred"
                mesh_dict = {"mesh.vertices": m.vertices, "mesh.triangles": m.faces}
                if normals is not None:
                    mesh_dict["mesh.normals"] = normals
                if colors is not None:
                    mesh_dict["mesh.colors"] = cast(Any, colors[:, :3] / 255.0)
                save_data._save_mesh(mesh_dict, filename, index=inst_idx, suffix=".obj")
    else:
        mesh_path = category_meshes_dir / f"{obj_name}.ply"
        mesh_path = _shorten_path_filename(mesh_path)
        visual = cast(Any, mesh.visual)
        vertex_colors = cast(np.ndarray | None, getattr(visual, "vertex_colors", None))
        has_vertex_colors = vertex_colors is not None and len(np.unique(vertex_colors[:, :3])) > 1
        colors = vertex_colors if has_vertex_colors else None
        normals = mesh.vertex_normals if generator.estimate_normals else None
        save_mesh(mesh_path, mesh.vertices, mesh.faces, colors, normals)
        out_dict["mesh"] = mesh_path
        if cfg.vis.show:
            if not has_vertex_colors:
                cast(Any, mesh.visual).vertex_colors = (0, 127, 127)
            mesh.show(smooth=False, caption=category_name)
        if cfg.vis.save:
            filename = str(index).zfill(8)
            mesh_dict = {"mesh.vertices": mesh.vertices, "mesh.triangles": mesh.faces}
            if normals is not None:
                mesh_dict["mesh.normals"] = normals
            if colors is not None:
                mesh_dict["mesh.colors"] = cast(Any, colors[:, :3] / 255.0)
            save_data._save_mesh(mesh_dict, filename, suffix=".obj")

    if cfg.inputs.type == "pointcloud" or (cfg.inputs.type in ["depth", "kinect", "rgbd"] and cfg.inputs.project):
        if isinstance(obj_name, list):
            inputs_path = category_inputs_dir[0] / f"{obj_name[0]}.ply"
        else:
            inputs_path = category_inputs_dir / f"{obj_name}.ply"
        inputs_path = _shorten_path_filename(inputs_path)
        points = np.asarray(item["inputs"])
        if points.ndim == 3:
            grid = generator.query_points.numpy()
            points = grid[points.ravel() == 1]
        elif cfg.inputs.fps.num_points and len(points) > cfg.inputs.fps.num_points:
            from libs import furthest_point_sample

            points = furthest_point_sample(points, cfg.inputs.fps.num_points, backend="open3d")
        trimesh.PointCloud(points).export(inputs_path)
        out_dict["inputs"] = inputs_path
    else:
        out_dict["inputs"] = Path(str(item["inputs.path"]))

    if isinstance(obj_category, str):
        obj_instances = obj_counter[obj_category]

    if copy_inputs:
        for k, filepath in out_dict.items():
            if isinstance(filepath, list):
                if len(filepath) == 0:
                    continue

                if isinstance(category_vis_dir, list):
                    if isinstance(obj_category, list):
                        obj_instances = obj_counter[str(obj_category[0])]
                    out_file = (category_vis_dir[0] / f"{obj_instances:02d}_{k}").with_suffix(filepath[0].suffix)
                else:
                    out_file = (category_vis_dir / f"{obj_instances:02d}_{k}").with_suffix(filepath[0].suffix)
                shutil.copyfile(filepath[0], out_file)
                for i, fp in enumerate(filepath[1:]):
                    if isinstance(category_vis_dir, list):
                        if isinstance(obj_category, list):
                            obj_instances = obj_counter[str(obj_category[i + 1])]
                        out_file = (category_vis_dir[i + 1] / f"{obj_instances:02d}").with_suffix(fp.suffix)
                    else:
                        out_file = (category_vis_dir / f"{obj_instances:02d}_{k}_{i + 1}").with_suffix(fp.suffix)
                    shutil.copyfile(fp, out_file)
            else:
                if isinstance(category_vis_dir, list):
                    out_file = (category_vis_dir[0] / f"{obj_instances:02d}_{k}").with_suffix(filepath.suffix)
                else:
                    out_file = (category_vis_dir / f"{obj_instances:02d}_{k}").with_suffix(filepath.suffix)
                shutil.copyfile(filepath, out_file)

    if isinstance(obj_category, str):
        obj_counter[str(obj_category)] += 1
    elif isinstance(obj_category, list):
        for cat in obj_category:
            obj_counter[str(cat)] += 1


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg = setup_config(cfg)
    suppress_known_optional_dependency_warnings()
    log_optional_dependency_summary(logger, cfg)
    split = cfg.test.split
    save_dir = resolve_save_dir(cfg) / "generation" / split
    category = ""
    if cfg.data.categories and len(cfg.data.categories) == 1:
        category = cfg.data.categories[0]
    if not overwrite_results(cfg, save_dir / "meshes" / str(category), set_overwrite=False):
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = save_dir / "meshes"
    inputs_dir = save_dir / "inputs"
    vis_dir = save_dir / "vis"
    for d in [meshes_dir, inputs_dir, vis_dir]:
        if cfg.test.overwrite:
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(exist_ok=True)

    dataset = get_dataset(cfg, splits=(split,))[split]
    num_workers = get_num_workers(cfg.load.num_workers)
    collate_fn = None
    try:
        from train import get_collate_fn

        collate_fn = get_collate_fn(cfg, split, cfg.test.batch_size or 1)
    except ImportError as e:
        logger.warning(f"Unable to import train module, custom collate functions won't be available: {e}")

    sampler = None
    indices = cfg.vis.index
    if indices is not None:
        if isinstance(indices, int):
            idx_list = [indices]
        elif isinstance(indices, str):
            idx_list = [int(s) for s in indices.split(",")]
        else:
            idx_list = [int(i) for i in indices]
        sampler = SubsetRandomSampler(idx_list)
        _debug_level_1(f"Using SubsetRandomSampler for indices: {idx_list}")

    loader = DataLoader(
        dataset,
        batch_size=1 if isinstance(dataset, BOP) else cfg.test.batch_size or 1,
        shuffle=cfg[split].shuffle,
        sampler=sampler,
        num_workers=0 if isinstance(dataset, BOP) else num_workers,
        collate_fn=collate_fn,
        prefetch_factor=cfg.load.prefetch_factor,
        persistent_workers=True if num_workers else False,
        pin_memory=cfg.load.pin_memory,
        generator=torch.Generator().manual_seed(cfg.misc.seed),
    )

    if cfg.model.weights is None and cfg.model.checkpoint is None:
        _debug_level_1("No weights or checkpoint specified. Trying to load best model.")
        cfg.model.load_best = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).eval().to(device)
    if cfg.vis.refinement_steps or "16" not in cfg.val.precision:
        model = patch_attention(model, mode="math" if cfg.vis.refinement_steps else None, backend="torch")
    fabric = lightning.Fabric(precision=cfg[split].precision)

    resolution = cfg.vis.resolution
    upsampling_steps = cfg.vis.upsampling_steps
    if upsampling_steps is None:
        upsampling_steps = 0
        if resolution > 128 and not isinstance(model, (DiffusionModel, AutoregressiveModel, UNetModel)):
            upsampling_steps = int(np.log2(resolution) - np.log2(32))
    if upsampling_steps:
        resolution = int(2 ** (np.log2(32) + upsampling_steps))
    points_batch_size = int(cfg.vis.num_query_points or resolution**3)

    summary_batch_size = int(loader.batch_size or 1)
    item = {k: v.to(device) for k, v in next(iter(loader)).items() if torch.is_tensor(v)}
    item["points"] = torch.zeros(summary_batch_size, points_batch_size, 3, device=device)
    if model.name == "ShapeFormer":
        item["pointcloud"] = torch.zeros(summary_batch_size, 32768, 3, device=device)
    with fabric.autocast():
        summary(model, input_data=item, depth=3 + cfg.log.verbose)

    if split != "test":
        logger.warning("Evaluation is NOT done on the TEST set!")

    generator = Generator(
        cast(Any, model),
        points_batch_size=points_batch_size,
        threshold=cfg.implicit.threshold,
        refinement_steps=cfg.vis.refinement_steps,
        padding=cfg.norm.padding,
        scale_factor=cfg.norm.scale_factor,
        resolution=resolution,
        upsampling_steps=upsampling_steps,
        estimate_normals=cfg.vis.normals,
        predict_colors=cfg.vis.colors,
        simplify=cfg.vis.simplify,
        sdf=cfg.implicit.sdf,
        bounds=cfg.norm.bounds,
    )

    obj_counter = defaultdict(int)
    desc = (
        f"Generating {cfg.log.name} "
        f"(res={resolution}, "
        f"up_steps={upsampling_steps}, "
        f"points_bs={points_batch_size}, "
        f"thresh={cfg.implicit.threshold})"
    )
    desc += f"{f' {cfg.vis.num_instances} class instances' if cfg.vis.num_instances else ''}"
    for batch_idx, batch in enumerate(tqdm(loader, desc=desc, disable=not cfg.log.progress)):
        if batch.get("inputs.skip", False):
            continue

        if isinstance(model, GridDiffusionModel) and model.denoise_fn.ndim == 2:
            fid_dir = save_dir / "fid"
            fid_dir.mkdir(exist_ok=True)

            batch_size = int(cfg.test.batch_size or 1)
            images = model.generate(inputs=torch.zeros((batch_size,)), show=cfg.vis.show)
            images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

            for img_idx, img in enumerate(images):
                img_path = fid_dir / f"{batch_idx}_{img_idx}.png"
                if img.shape[2] == 1:
                    Image.fromarray(img[:, :, 0], "L").save(img_path)
                else:
                    Image.fromarray(img, "RGB").save(img_path)

            obj_counter["fid"] += batch_size
            if cfg.vis.num_instances and obj_counter["fid"] >= cfg.vis.num_instances:
                break
            continue

        if cfg.vis.num_instances and all(obj_counter[c] >= cfg.vis.num_instances for c in set(batch["category.id"])):
            continue

        data = {str(k): to_tensor(v, unsqueeze=False, device=device) for k, v in batch.items()}
        with fabric.autocast():
            feature = None
            sample = _to_int(cfg.get("sample", 0) or 0)
            if len(batch["index"]) == 1 and sample > 1:
                data["inputs"] = data["inputs"].repeat(sample, 1, 1)
                if "inputs.depth" in data:
                    data["inputs.depth"] = data["inputs.depth"].repeat(sample, 1, 1)
            align_to_gt = bool(cfg.get("align_to_gt", False))
            if cfg.predict.instances:
                grid = generator.generate_grid_per_instance(
                    cast(dict[str, Any], data),
                    align_to_gt=align_to_gt,  # InstSeg
                    show=cfg.vis.show,
                )
            else:
                grid, _points, feature = generator.generate_grid(
                    cast(dict[str, Any], data),
                    progress=cfg.log.progress,
                    show=cfg.vis.show,
                    sample=sample,  # VAE, Diffusion, Autoregressive
                    unconditional=cfg.get("unconditional", False),  # VAE
                    steps=cfg.get("steps", 18),  # Diffusion
                    align_to_gt=align_to_gt,  # InstSeg
                )
            batch["grid"] = [grid]

        for i in trange(
            _to_int(len(batch["index"])),
            desc="Processing batch",
            leave=False,
            disable=len(batch["index"]) == 1 or not cfg.log.progress,
        ):
            item = {str(k): to_numpy(v[i]) for k, v in batch.items()}
            process_item(
                cfg,
                cast(dict[str, Any], item),
                feature,
                dataset,
                generator,
                obj_counter,
                vis_dir,
                meshes_dir,
                inputs_dir,
            )


if __name__ == "__main__":
    main()
