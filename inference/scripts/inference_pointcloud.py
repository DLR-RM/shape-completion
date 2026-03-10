import logging
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, cast

import numpy as np
import open3d as o3d
import torch
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf
from tabulate import tabulate
from torch import nn
from tqdm import tqdm
from trimesh import Trimesh

from models import get_model
from utils import (
    eval_input,
    load_mesh,
    log_optional_dependency_summary,
    make_3d_grid,
    set_log_level,
    setup_logger,
    suppress_known_optional_dependency_warnings,
)

from ..src.utils import get_input_data_from_point_cloud, get_point_cloud, remove_plane

logger = setup_logger(__name__)


def load_model(args: Namespace) -> nn.Module:
    """
    This function loads a trained model based on the provided arguments. It creates a model configuration using the
    OmegaConf.create method and then calls the get_model function from the models module to create the model.
    The model is then set to evaluation mode and moved to the GPU.

    Parameters:
    args (Namespace): The arguments provided by the user, including settings for the model.

    Returns:
    model (nn.Module): The trained model ready for inference.
    """
    model = get_model(
        OmegaConf.create(
            {
                "model": {
                    "arch": args.model,
                    "weights": args.weights,
                    "checkpoint": args.checkpoint,
                    "num_classes": 2,
                    "norm": None,
                    "activation": "relu",
                    "dropout": False,
                    "load_best": args.weights is None and args.checkpoint is None,
                    "reduction": "mean",
                    "attn_backend": "torch",
                    "attn_mode": "flash",
                },
                "inputs": {
                    "type": "pointcloud",
                    "dim": 3,
                    "fps": {"num_points": None},
                    "nerf": "nerf" in str(args.weights),
                },
                "points": {"nerf": "nerf" in str(args.weights)},
                "implicit": {"sdf": args.sdf},
                "train": {"precision": "32-true", "resume": False},
                "vis": {"refinement_steps": 0},
                "cls": {"num_classes": None},
                "seg": {"num_classes": None},
                "norm": {"padding": args.padding},
                "misc": {"output_dir": ""},
                "dirs": {"log": "test", "backup": None},
                "log": {"name": "test", "project": "test", "id": None, "version": None},
            }
        ),
        freeze_encoder=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return model


def try_run(model: nn.Module, in_path: Path, args: Namespace):
    """
    This function attempts to run the inference process on a given input path with the provided model and arguments.
    If an exception occurs during the process, it logs the error and the exception.

    Parameters:
    model (nn.Module): The trained model to be used for inference.
    in_path (Path): The path to the input data for inference.
    args (Namespace): The arguments provided by the user, including settings for the inference process.
    """
    try:
        run(model, in_path, args)
    except Exception as e:
        logger.error(f"Error while processing '{in_path}': {e}")
        logger.exception(e)


def run(model: nn.Module, in_path: Path, args: Namespace):
    if args.show:
        try:
            from visualize import Generator
        except ImportError:
            logger.exception("Could not import from the 'visualize' submodule. Skipping visualization.")
            args.show = False
        try:
            from dataset import Visualize
        except ImportError:
            logger.exception("Could not import from the 'dataset' submodule. Skipping visualization.")
            args.show = False

    if args.eval:
        try:
            from eval import eval_mesh
        except ImportError:
            logger.exception("Could not import from the 'eval' submodule. Skipping evaluation.")
            args.eval = False

    out_dir: Path | None = None
    if args.output is not None:
        output_dir = cast(Path, args.output).expanduser().resolve()
        out_dir = output_dir / in_path.parent.stem
        out_dir.mkdir(exist_ok=True, parents=True)

    pcd, _intrinsic, _extrinsic = get_point_cloud(
        in_path,
        args.depth_scale,
        args.depth_trunc,
        args.intrinsic,
        args.extrinsic,
        args.pcd_crop,
        show=args.show and args.verbose,
    )

    plane_model = None
    offset_y = 0
    object_points = None
    if args.remove_plane:
        pcds, plane_model = remove_plane(
            pcd,
            distance_threshold=args.distance_threshold,
            num_iterations=args.ransac_iterations,
            outlier_neighbors=args.outlier_neighbors,
            outlier_radius=args.outlier_radius,
            outlier_std=args.outlier_std,
            cluster=args.cluster,
            num_cluster=1,
            crop=args.crop,
            crop_scale=args.crop_scale,
            crop_up_axis=args.up_axis,
            show=args.show and args.verbose,
        )
        pcd = pcds[0]
        object_points = np.asarray(pcd.points).copy()
        if args.on_plane:
            a, b, c, d = plane_model
            points = np.asarray(pcd.points)
            index = np.argmin(points[:, 1])
            point = points[index]
            offset_y = (a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(a**2 + b**2 + c**2)

    input_data, loc, scale = get_input_data_from_point_cloud(
        pcd,
        offset_y=offset_y,
        center=True,
        scale=args.scale if args.scale != 1 else True,
        crop=(1 + args.padding) / 2,
        voxelize=128 if model.name == "IFNet" else None,
        padding=args.padding,
        show=args.show and args.verbose,
    )

    # Predict
    device = next(model.parameters()).device
    mesh, pcd = None, None
    threshold = 0 if args.sdf and args.threshold == 0.5 else args.threshold
    if model.name in ["ONet", "ConvONet", "IFNet", "VQDIF", "Shape3D2VecSet", "Shape3D2VecSetVAE"]:
        inputs = torch.from_numpy(input_data).unsqueeze(0).to(device)
        points_batch_size = args.n_points if args.n_points > 0 else args.resolution**3
        if args.show:
            generator = Generator(
                cast(Any, model),
                points_batch_size=points_batch_size,
                threshold=threshold,
                refinement_steps=0,
                padding=args.padding,
                resolution=args.resolution,
                upsampling_steps=args.n_up if args.n_up >= 0 else int(np.log2(args.resolution) - np.log2(32)),
                # with_normals=not args.sdf,
                sdf=args.sdf,
            )
            start = time.perf_counter()
            grid, _, c = generator.generate_grid({"inputs": inputs})
            logger.debug(f"Grid generation took {time.perf_counter() - start:.2f}s")
            mesh = generator.extract_mesh(cast(np.ndarray, grid), c)
        else:
            box_size = 1 + args.padding
            voxel_size = box_size / args.resolution
            grid_points = (box_size - voxel_size) * make_3d_grid(-0.5, 0.5, args.resolution)
            query_points = grid_points.unsqueeze(0).to(inputs.device)
            logits = cast(Any, model).predict(inputs, query_points, points_batch_size=points_batch_size)
            logits = logits.squeeze(0).cpu().numpy()
    elif model.name in ["RealNVP", "MCDropoutNet", "PSSNet"]:
        raise NotImplementedError
    elif model.name in ["PCN", "SnowflakeNet"]:
        with torch.no_grad():
            pcd = model(torch.from_numpy(input_data).unsqueeze(0).to(device))[-1].squeeze(0).cpu().numpy()
    else:
        raise NotImplementedError(f"Evaluation of '{model.name}' not implemented.")

    if args.eval and args.mesh and mesh:
        pose = args.pose
        if pose is None:
            files = list(in_path.parent.glob("*.npy"))
            for file in files:
                if in_path.stem.replace("object_", "") in file.name and "pose" in file.name:
                    print(f"Found pose file '{file.name}' in input directory.")
                    pose = np.load(file)

        result = eval_mesh(
            mesh.copy(),
            Trimesh(*load_mesh(args.mesh)),
            cast(Any, generator).query_points().numpy(),
            loc,
            scale,
            pose,
        )

        table = tabulate([result], headers="keys", tablefmt="presto", floatfmt=".4f")
        print(table)
        if args.output is not None:
            assert out_dir is not None
            np.save(out_dir / "result.npy", np.array(result, dtype=object))
            with open(out_dir / "result.txt", "w") as f:
                f.write(table)

    if args.show:
        vis = Visualize(
            show_inputs=True,
            show_occupancy=False,
            show_mesh=True,
            show_pointcloud=mesh is not None and pcd is not None,
            sdf=args.sdf,
            padding=args.padding,
        )

        vis_data = {"inputs": input_data, "inputs.path": str(in_path)}
        if mesh is not None and pcd is not None:
            vis_data["mesh.vertices"] = mesh.vertices
            vis_data["mesh.triangles"] = mesh.faces
            vis_data["pointcloud"] = pcd
        elif mesh is not None:
            vis_data["mesh.vertices"] = mesh.vertices
            vis_data["mesh.triangles"] = mesh.faces
        elif pcd is not None:
            vis_data["mesh.vertices"] = pcd
            vis_data["mesh.triangles"] = None
        else:
            raise ValueError("Neither mesh nor pointcloud to visualize.")
        vis(vis_data)

    if args.output is not None and (mesh is not None or pcd is not None):
        assert out_dir is not None
        if plane_model is not None:
            np.save(out_dir / "plane.npy", plane_model)
        if object_points is not None:
            o3d.io.write_point_cloud(
                str(in_path.parent / f"object_{in_path.name}"),
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_points)),
            )

        if mesh is not None:
            if args.denormalize:
                mesh.apply_scale(scale)
                mesh.apply_translation(loc)
            mesh.export(out_dir / "mesh.ply")
        elif pcd is not None:
            o3d.io.write_point_cloud(str(out_dir / "pcd.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd)))

        o3d.io.write_point_cloud(
            str(out_dir / "inputs.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(input_data))
        )


def main():
    parser = ArgumentParser(description="Run inference of trained model on input data.")
    parser.add_argument("in_dir_or_path", type=Path, help="Path to input data.")
    parser.add_argument("model", type=str.lower, default="conv_onet_grid", help="Name of model to evaluate.")

    param_group = parser.add_mutually_exclusive_group(required=True)
    param_group.add_argument("-w", "--weights", type=Path, help="Path to model weights.")
    param_group.add_argument("-c", "--checkpoint", type=Path, help="Path to PyTorch Lightning checkpoint.")

    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold at which mesh is extracted (will be mapped to log space).",
    )
    gen_group.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Resolution at which to predict and extract SDF/occupancy voxel grid",
    )
    gen_group.add_argument("--n_points", type=int, default=-1, help="Number of query points for grid generation.")
    gen_group.add_argument("--n_up", type=int, default=-1, help="Number of upsampling steps for grid generation.")
    gen_group.add_argument("--padding", type=float, default=0.1, help="Padding used during data generation.")
    gen_group.add_argument(
        "--sdf", action="store_true", help="Model outputs SDF values rather than occupancy probabilities (i.e. logits)."
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("-f", "--in_format", type=str, default=".ply", help="Input file format.")
    io_group.add_argument("--recursion_depth", type=int, help="Depth of recursive glob pattern matching.")
    io_group.add_argument("-o", "--output", type=Path, help="Output path where results are stored.")
    io_group.add_argument("--sort", action="store_true", help="Sort files before processing.")
    io_group.add_argument("--denormalize", action="store_true", help="Denormalize output before saving.")

    cam_group = parser.add_argument_group("Camera")
    cam_group.add_argument(
        "--intrinsic",
        nargs="+",
        help="List of (4, 5 or 9) floats or filepath specifying the camera intrinsic parameters.",
    )
    cam_group.add_argument(
        "-e", "--extrinsic", nargs="+", help="List of (7, 9, or 12) floats or filepath specifying the camera pose."
    )
    cam_group.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="The scale of the depth data. 1000 means it is in mm and will be converted to m.",
    )
    cam_group.add_argument(
        "--depth_trunc",
        type=float,
        default=1.1,
        help="The distance at which to truncate the depth when creating the point cloud.",
    )

    process_group = parser.add_argument_group("Pre- and Post-Processing")
    process_group.add_argument(
        "--remove_plane", action="store_true", help="Remove planar surfaces from input data (e.g. ground, table, etc)."
    )
    process_group.add_argument(
        "--distance_threshold",
        type=float,
        default=0.006,
        help="Distance at which points are rejected as belonging to segmented plane.",
    )
    process_group.add_argument(
        "--ransac_iterations", type=int, default=1000, help="RANSAC plane segmentation iterations."
    )
    process_group.add_argument("--cluster", action="store_true", help="Cluster points after plane segmentation.")
    process_group.add_argument(
        "--outlier_neighbors",
        type=int,
        default=50,
        help="Number of neighbors to use for radius/statistical outlier removal.",
    )
    process_group.add_argument(
        "--outlier_radius", type=float, default=0.1, help="Radius to use for radius outlier removal."
    )
    process_group.add_argument(
        "--outlier_std", type=float, default=10.0, help="Standard deviation to use for statistical outlier removal."
    )
    process_group.add_argument("--on_plane", action="store_true", help="Place object on plane.")
    process_group.add_argument(
        "--crop",
        action="store_true",
        help="Retain points from plane that intersect with convex hull of non-plane points.",
    )
    process_group.add_argument("--crop_scale", type=float, default=1.0, help="Scale factor for convex hull.")
    process_group.add_argument(
        "--up_axis", type=int, default=1, help="Axis along which to crop plane points (0=x, 1=y, 2=z)."
    )
    process_group.add_argument(
        "--pcd_crop",
        nargs=6,
        type=float,
        default=[-0.4, -0.5, -0.5, 0.4, 0.5, 0.5],
        help="Crop point cloud to specified region (min_x, min_y, min_z, max_x, max_y, max_z).",
    )

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("-m", "--mesh", type=Path, help="Path to ground truth mesh file.")
    eval_group.add_argument(
        "-p",
        "-t",
        "--pose",
        "--transform",
        nargs="+",
        help="Transformation applied to the mesh to match the input data."
        "List of (3, 4, 7, 9 or 12) floats or filepath.",
    )
    eval_group.add_argument("--eval", action="store_true", help="Evaluate generated mesh against ground truth.")

    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--show", action="store_true", help="Visualize completion result.")
    misc_group.add_argument("--seed", type=int, default=0, help="Seed for random number generators.")
    misc_group.add_argument("--scale", type=float, default=1, help="Scale input points.")
    misc_group.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    suppress_known_optional_dependency_warnings()
    if args.verbose:
        set_log_level(logging.DEBUG)
    log_optional_dependency_summary(logger, model_arch=args.model, attn_backend="torch")

    seed_everything(args.seed)
    model = load_model(args)
    files = eval_input(args.in_dir_or_path, args.in_format, args.recursion_depth, args.sort)
    for file in tqdm(files, desc="Running inference", disable=args.verbose):
        try_run(model, file, args)


if __name__ == "__main__":
    main()
