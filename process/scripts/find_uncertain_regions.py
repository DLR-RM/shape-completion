import copy
import os
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import open3d as o3d
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from dataset import BOP, FindUncertainPoints, RenderDepthMaps, ShapeNet, Visualize, get_dataset
from utils import (
    inv_trafo,
    log_optional_dependency_summary,
    setup_logger,
    suppress_known_optional_dependency_warnings,
    tqdm_joblib,
)

logger = setup_logger(__name__)


def get_offset(points: np.ndarray, show: bool = False) -> np.ndarray:
    _min = points.min(axis=0)
    _max = points.max(axis=0)
    bounds = _max - _min

    pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cuts = list()

    for cut in np.linspace(0, 1, 11):
        offset = 0.005 * bounds[1]
        if cut in [0, 1]:
            offset = 0.03 * bounds[1]
        pcd_cut = pointcloud.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(-1, _min[1] + cut * bounds[1] - offset, -1),
                max_bound=(1, _min[1] + cut * bounds[1] + offset, 1),
            )
        )
        if pcd_cut.has_points():
            cuts.append(pcd_cut)

    if show:
        o3d_visualization = cast(Any, o3d).visualization
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d_visualization.draw_geometries([*cuts, frame])

    symmetric = 1
    best_cut = None
    for cut in cuts:
        _size = cut.get_max_bound() - cut.get_min_bound()
        if all(s == 0 for s in _size):
            continue
        if abs(_size[2] - _size[0]) < symmetric:
            symmetric = abs(_size[2] - _size[0])
            best_cut = cut
    if best_cut is None:
        return np.zeros(3, dtype=np.float32)
    offset = -(best_cut.get_max_bound() + best_cut.get_min_bound()) / 2
    return offset


def process(
    dataset: ShapeNet | BOP,
    index: int,
    parallel: bool = False,
    overwrite: bool = False,
    debug: bool = False,
    show: bool = False,
) -> bool:
    if isinstance(dataset, ShapeNet):
        inputs_path = Path(dataset.get_inputs_path(index))
    elif isinstance(dataset, BOP):
        item = dataset[index]
        if item.get("inputs.skip", False):
            if debug:
                print(
                    "Up:",
                    item["inputs.up"],
                    "Visib. fract.:",
                    item["inputs.visib_fract"],
                    "Num. input points:",
                    len(item["inputs"]),
                )
            return False
        inputs_path = Path(item["inputs.path"])

    done = True
    points_field = cast(Any, dataset.fields["points"])
    point_files = cast(list[str], points_field.file)
    for file in point_files:
        path = inputs_path.parent.joinpath("_".join([inputs_path.stem, Path(file).stem, "uncertain.npy"]))
        if not path.is_file():
            done = False
            break
    if done and not overwrite:
        if debug:
            print("Files already exist.")
        return True
    elif isinstance(dataset, ShapeNet):
        item = dataset[index]

    if debug and show:
        Visualize(show_frame=False, show_box=False, show_mesh=True)(item)

    item_copy = copy.deepcopy(item)
    extrinsic = item_copy["inputs.extrinsic"].copy()
    inv_extrinsic = inv_trafo(extrinsic)

    # Hack 1: Cameras facing the mugs handle (x > 0) are discarded
    if inv_extrinsic[0, 3] > 0:
        if debug:
            print(inv_extrinsic[0, 3])
        return False

    offset = get_offset(item_copy["pointcloud"], show=debug and show)
    offset = np.asarray([offset[0], offset[2], 0])
    rot_x_90 = np.array([1, 0, 0, 0, 0, -1, 0, 1, 0]).reshape(3, 3)

    # Hack 2: Views with partially visible handle are discarded
    scale = 1.0
    if dataset.name == "tyol":
        scale = 0.8
    elif dataset.name in ["hb", "lm"]:
        scale = 0.9
    inputs_offset = -(item_copy["inputs"].max(axis=0) + item_copy["inputs"].min(axis=0)) / 2
    if inputs_offset[0] < scale * offset[0]:
        if debug:
            print("Offsets:", inputs_offset[0], scale * offset[0])
        return False

    vertices = item_copy["mesh.vertices"].copy()
    vertices = (rot_x_90 @ vertices.T).T
    vertices += offset
    item_copy["mesh.vertices"] = vertices

    points = item_copy["points"].copy()
    points = (rot_x_90 @ points.T).T
    points += offset
    item_copy["points"] = points

    inv_extrinsic_arr = np.asarray(inv_extrinsic).copy()
    inv_extrinsic_arr[:3, 3] += offset
    extrinsic = inv_trafo(inv_extrinsic_arr)
    item_copy["inputs.extrinsic"] = extrinsic

    item_copy["inputs.look_at"] = offset

    render_cls = cast(Any, RenderDepthMaps)
    render = render_cls(
        step=5 if dataset.name in ["tyol", "shapenet"] else 10,
        min_angle=90 if "shapenet" in dataset.name.lower() else 0,
        mapitch=275 if "shapenet" in dataset.name.lower() else 360,
        width=640 if "shapenet" in dataset.name.lower() else item["inputs.width"],
        height=480 if "shapenet" in dataset.name.lower() else item["inputs.height"],
        inplane_rot=90 if "shapenet" in dataset.name.lower() else 0,
        offscreen=not show,
        print_message=debug,
    )
    depth_list, _extrinsic_list, angle_list = render(item_copy)
    if debug and show:
        render.show()

    item_copy = copy.deepcopy(item)
    offset = np.asarray([offset[0], 0, offset[1]])

    vertices = item_copy["mesh.vertices"].copy()
    vertices += offset
    item_copy["mesh.vertices"] = vertices

    points = item_copy["points"].copy()
    points += offset
    item_copy["points"] = points

    inputs = item_copy["inputs"].copy()
    inputs += offset
    item_copy["inputs"] = inputs

    item_copy["inputs.offset"] = offset

    max_chamfer_dist = 0.01
    if dataset.name == "tyol":
        max_chamfer_dist = 0.02
    elif dataset.name == "hb":
        max_chamfer_dist = 0.025
    find_cls = cast(Any, FindUncertainPoints)
    find = find_cls(
        depth_list,
        angle_list,
        max_chamfer_dist=max_chamfer_dist,
        parallel=parallel,
        debug=debug,
        show=show,
        print_message=debug,
    )
    item_copy = find(item_copy)

    uncertain = item_copy.get("points.uncertain")
    if uncertain is not None:
        if show:
            points_occ = cast(Any, item["points.occ"])
            points_occ[uncertain.astype(bool)] = 2
            Visualize(show_frame=False, show_box=False)(item)
        assert len(uncertain) == len(item_copy["points.path"]) * int(1e5)
        for index, path in enumerate(item_copy["points.path"]):
            output_path = inputs_path.parent.joinpath("_".join([inputs_path.stem, Path(path).stem, "uncertain.npy"]))
            np.save(str(output_path), np.packbits(uncertain[index * int(1e5) : (index + 1) * int(1e5)].astype(bool)))
        return True
    return False


def protect(function: Any, **kwargs: Any) -> Any | Exception:
    try:
        return function(**kwargs)
    except Exception as e:
        logger.exception(e)
        return e


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    suppress_known_optional_dependency_warnings()
    log_optional_dependency_summary(logger, cfg)
    if not cfg.vis.show:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    cfg.load.keys_to_keep.extend(["inputs.intrinsic", "inputs.extrinsic", "inputs.rot", "inputs.pitch"])
    dataset = get_dataset(cfg, splits=(cfg.vis.split,))[cfg.vis.split]

    indices = range(len(cast(Sized, dataset)))
    with tqdm_joblib(tqdm(total=len(indices), desc="Processing views", disable=cfg.log.verbose)):
        with Parallel(n_jobs=1 if cfg.log.verbose > 1 else -1) as p:
            out = p(
                delayed(protect)(
                    process,
                    **dict(
                        dataset=dataset,
                        index=index,
                        overwrite=cfg.test.overwrite,
                        debug=cfg.log.verbose,
                        show=cfg.vis.show,
                    ),
                )
                for index in indices
            )

    exceptions = [view for view in out if isinstance(view, Exception)]
    exception_mask = np.array([isinstance(view, Exception) for view in out])
    exception_indices = exception_mask.nonzero()[0].tolist()
    index_exception_dict = dict(zip(exception_indices, exceptions, strict=False))

    uncertain_views = np.array([view for view in out if isinstance(view, bool)])
    print("Found", uncertain_views.sum(), "uncertain views.")

    if index_exception_dict:
        print()
        print("Exceptions caught during execution:")
        for index, exception in index_exception_dict.items():
            print(index, exception)


if __name__ == "__main__":
    main()
