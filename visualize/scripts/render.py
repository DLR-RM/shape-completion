import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import numpy as np
import open3d as o3d
import trimesh
from lightning.pytorch import seed_everything
from PIL import Image
from tqdm import tqdm
from trimesh import Trimesh

from utils import (
    inv_trafo,
    log_optional_dependency_summary,
    look_at,
    setup_logger,
    stack_images,
    suppress_known_optional_dependency_warnings,
)

from ..src.renderer import Renderer

logger = setup_logger(__name__)


def main():
    parser = ArgumentParser(description="Run inference of trained model on input data.")
    parser.add_argument("dir", type=Path, help="Path to input data directory.")
    parser.add_argument("--obj_type", type=str, choices=["mesh", "pcd"], default="mesh", help="Type of input data.")
    parser.add_argument("--individual", action="store_true", help="Render mesh and pointcloud separately.")
    parser.add_argument(
        "--look_at", type=str, choices=["centroid", "zero"], default="centroid", help="Look at centroid or zero."
    )
    parser.add_argument("--show", action="store_true", help="Show rendered images.")
    parser.add_argument("--verbose", action="store_true", help="Print additional information.")
    args = parser.parse_args()

    suppress_known_optional_dependency_warnings()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    log_optional_dependency_summary(logger, needs_blender=True)

    if args.show:
        mpl.use("GTK3Agg")

    seed_everything(1337)

    in_dir = args.dir.expanduser().resolve()
    if in_dir.is_file():
        paths = [in_dir]
    elif in_dir.is_dir():
        paths = sorted(in_dir.rglob(f"*_{args.obj_type}.ply"))
    else:
        raise ValueError(f"Invalid input path: {in_dir}")
    output_dir = in_dir.parent if in_dir.is_file() else in_dir
    logger.debug(f"Found {len(paths) if len(paths) > 1 else 'one'} object{'s' if len(paths) > 1 else ''}.")

    renderer = Renderer(
        method="blender",
        width=1024,
        height=1024,
        raytracing=True,
        transparent_background=args.obj_type == "pcd",
        show=args.show,
    )
    backdrop_path = Path(__file__).parent.parent / "assets" / "backdrop.ply"
    plane = trimesh.load(backdrop_path)
    plane_any = cast(Any, plane)

    num_images = int(np.floor(np.sqrt(len(paths))) ** 2)
    images = list()
    for obj_path in tqdm(paths[:num_images], desc=f"Rendering {args.obj_type.upper()}", disable=args.verbose):
        obj_name = obj_path.stem
        basename = obj_name.replace(f"_{args.obj_type}", "")
        pcd_name = obj_name.replace(f"_{args.obj_type}", "_input")
        mesh_gt_name = obj_name.replace(f"_{args.obj_type}", "_gt")

        obj = cast(Any, trimesh.load(obj_path, validate=False, process=False))
        is_mesh = isinstance(obj, Trimesh)
        pcd = o3d.io.read_point_cloud(str(obj_path).replace(f"_{args.obj_type}.ply", "_inputs.ply"))
        scale = np.max(obj.bounding_box.extents)

        mesh_gt = None
        scale_gt = 1
        extrinsic_gt = None
        if (obj_path.parent / obj_path.name.replace(f"_{args.obj_type}.ply", "_gt.ply")).is_file():
            mesh_gt = cast(Any, trimesh.load(str(obj_path).replace(f"_{args.obj_type}.ply", "_gt.ply"), process=False))
            scale_gt = np.max(mesh_gt.bounding_box.extents)

        # Reduce pcd density until its minimum neighbor distance is at least 1cm
        counter = 10
        while np.min(pcd.compute_nearest_neighbor_distance()) < 0.01 * scale:
            num_points = len(np.asarray(pcd.points))
            pcd = pcd.farthest_point_down_sample(num_points - max(10, int(num_points / counter)))
            counter *= 1.08
        pcd = trimesh.PointCloud(np.asarray(pcd.points))

        min_val = cast(np.ndarray, obj.vertices)[:, 1].min()
        obj.apply_transformations(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        obj.apply_translation([0, 0, -min_val])
        pcd.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        pcd.apply_translation([0, 0, -min_val])
        if mesh_gt is not None:
            mesh_gt.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            mesh_gt.apply_translation([0, 0, -min_val])
            center = mesh_gt.centroid if args.look_at == "centroid" else np.array([0, 0, 0])
            inv_extrinsic = look_at(np.array([0, -1.75 * scale_gt, 1.5 * scale_gt]), center)
            extrinsic_gt = np.asarray(inv_trafo(inv_extrinsic))

        center = obj.centroid if args.look_at == "centroid" else np.array([0, 0, 0])
        inv_extrinsic = look_at(np.array([0, -1.75 * scale, 1.5 * scale]), center)
        extrinsic = np.asarray(inv_trafo(inv_extrinsic))

        for index, rot_z_angle in enumerate([0, np.pi]):
            trafo = trimesh.transformations.rotation_matrix(rot_z_angle, [0, 0, 1])
            obj.apply_transformations(trafo)
            pcd.apply_transform(trafo)

            if args.individual:
                data = renderer(
                    vertices=obj.vertices,
                    faces=obj.faces if is_mesh else None,
                    colors=np.array([0.272, 0.272, 0.272]) if is_mesh else cast(Any, obj).colors,
                    extrinsic=cast(Any, extrinsic),
                )
                image = data["color"][0]
                image_path = obj_path.parent / f"{obj_name}_{'front' if index == 0 else 'back'}.png"
                Image.fromarray(image).save(image_path)
                logger.debug(f"Saved mesh image to {image_path}.")

                data = renderer(
                    vertices=pcd.vertices,
                    faces=None,
                    colors=np.array([0.126, 0.186, 0.8]),
                    extrinsic=cast(Any, extrinsic),
                )
                image = data["color"][0]
                image_path = obj_path.parent / f"{pcd_name}_{'front' if index == 0 else 'back'}.png"
                Image.fromarray(image).save(image_path)
                logger.debug(f"Saved pointcloud image to {image_path}.")

                if mesh_gt is not None:
                    mesh_gt.apply_transform(trafo)
                    data = renderer(
                        vertices=mesh_gt.vertices,
                        faces=mesh_gt.faces,
                        extrinsic=cast(Any, extrinsic_gt),
                    )
                    image = data["color"][0]
                    image_path = obj_path.parent / f"{mesh_gt_name}_{'front' if index == 0 else 'back'}.png"
                    Image.fromarray(image).save(image_path)
                    logger.debug(f"Saved ground truth mesh image to {image_path}.")

            if is_mesh:
                vertices = [cast(Any, obj).vertices, pcd.vertices, plane_any.vertices]
                faces = [cast(Any, obj).faces, None, plane_any.faces]
                colors = [renderer.default_mesh_color, renderer.default_pcd_color, np.ones(3)]
            else:
                vertices = [cast(Any, obj).vertices]
                faces = [None]
                colors = [cast(Any, obj).colors]

            data = renderer(
                vertices=cast(Any, vertices),
                faces=cast(Any, faces),
                colors=cast(Any, colors),
                extrinsic=cast(Any, extrinsic),
            )
            image = data["color"][0]
            images.append(image)
            image_path = obj_path.parent / f"{basename}_{'front' if index == 0 else 'back'}.png"
            image = Image.fromarray(image)
            white = Image.new("RGBA", image.size, (0, 0, 0))
            Image.alpha_composite(white, image).save(image_path)

    front_images = images[::2]
    back_images = images[1::2]
    front = Image.fromarray(stack_images(front_images))
    white = Image.new("RGBA", front.size, (255, 255, 255))
    Image.alpha_composite(white, front).save(output_dir / "renders_front.png")
    back = Image.fromarray(stack_images(back_images))
    white = Image.new("RGBA", back.size, (255, 255, 255))
    Image.alpha_composite(white, back).save(output_dir / "renders_back.png")


if __name__ == "__main__":
    main()
