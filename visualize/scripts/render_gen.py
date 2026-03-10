import logging
import os
from argparse import ArgumentParser
from logging import DEBUG, INFO, WARNING
from pathlib import Path
from typing import Any, cast

import numpy as np
import open3d as o3d
import trimesh
from lightning.pytorch import seed_everything
from PIL import Image
from trimesh import Trimesh

from utils import (
    DEBUG_LEVEL_1,
    DEBUG_LEVEL_2,
    inv_trafo,
    load_mesh,
    look_at,
    normalize_mesh,
    set_log_level,
)

from ..src.renderer import Renderer

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description="Render meshes and pointclouds.")
    # parser.add_argument("dir", type=Path, help="Path to input data directory.")
    # parser.add_argument("--obj_type", type=str, choices=["mesh", "pcd"], default="mesh",
    #                     help="Type of input data.")
    parser.add_argument("--individual", action="store_true", help="Render mesh and pointcloud separately.")
    parser.add_argument(
        "--look_at", type=str, choices=["centroid", "zero"], default="centroid", help="Look at centroid or zero."
    )
    parser.add_argument("--transparent", action="store_true", help="Render with transparent background.")
    parser.add_argument("--show", action="store_true", help="Show rendered images.")
    parser.add_argument("--verbose", action="store_true", help="Print additional information.")
    args = parser.parse_args()

    logger.setLevel(WARNING)
    if args.verbose > 0:
        logger.setLevel(INFO)
        if args.verbose == 1:
            set_log_level(DEBUG_LEVEL_1)
        elif args.verbose == 2:
            set_log_level(DEBUG_LEVEL_2)
        elif args.verbose == 3:
            set_log_level(DEBUG)

    # if args.show:
    #     mpl.use("GTK3Agg")

    seed_everything(1337)

    renderer = Renderer(
        method="blender",
        width=512,
        height=512,
        offscreen=not args.show,
        raytracing=True,
        transparent_background=args.transparent,
    )
    backdrop_path = Path(__file__).parent.parent / "assets" / "backdrop.ply"
    plane = trimesh.load(backdrop_path)
    plane_any = cast(Any, plane)

    data_dir = Path(os.environ.get("SHAPENET_ROOT", "/data/shapenet"))
    dataset_name = "ShapeNetCore.v1.fused.simple"
    dataset_dir = data_dir / dataset_name
    log_dir = Path(os.environ.get("LOG_ROOT", "/data/train"))
    project_name = "cvpr_2025"
    exp_name = "pcd_kinect_net_ft"
    obj_category = "03001627"  # "03001627" "02958343" "04379243"
    obj_name = "e6b0b43093f105277997a53584de8fa7"

    """
    e63546037077f0847dee16617fd6925f
    0 0.589242304693459
    1 0.5069466993454402
    2 0.5707770961173665
    3 0.4758546298996213
    4 0.5777848690109093
    5 0.3831791319530405
    6 0.43959758874444105
    7 0.46678837156256664
    8 0.47889474948145216
    9 0.47458185936265473
    [0, 8, 5]
    
    e6b0b43093f105277997a53584de8fa7
    0 0.6212998012920041
    1 0.7136728973272147
    2 0.6503237334645154
    3 0.7473485247875827
    4 0.7848194518157091
    5 0.7479277206211686
    6 0.7416107477315933
    7 0.739303608649511
    8 0.7527425381600797
    9 0.7535307288475057
    [4, 1]
    
    e779cf261cf04a77acd8c40fddcf9ca
    0 0.4270988795974937
    1 0.464138396000215
    2 0.4554890940785651
    3 0.4170171213587015
    4 0.5641212452793477
    5 0.6203159515882589
    6 0.48180120933982445
    7 0.42838112671794815
    8 0.4004880094008804
    9 0.46875341844788604
    [5, 4, 8]
    """

    """
    e64dd4ff16eab5752a9eb0f146e94477
    0 0.31486729098310856
    1 0.3050885312131349
    2 0.538160507542543
    3 0.3879337567146299
    4 0.3721368380198226
    5 0.4220414487709877
    6 0.35179872117543026
    7 0.39580476587644486
    8 0.34140858519299483
    9 0.31401648248420067
    [2, 7, 1]
    
    e9b2aebf673e516d1f0d59cde6139796
    0 0.3468951049757194
    1 0.279196752606368
    2 0.607327412050806
    3 0.5284246381280266
    4 0.3125679577880155
    5 0.4661067995743022
    6 0.4504416930230493
    7 0.4955881897716048
    8 0.4754040592834832
    9 0.5656031722201125
    [2, 9, 5]
    
    e6ec389325de7ae5becf71e2e014ff6f
    0 0.3709281163628542
    1 0.6080063098758131
    2 0.28923946781432114
    3 0.4753830750446135
    4 0.2994477348730101
    5 0.25758930143206427
    6 0.6482542988368258
    7 0.6344252918135833
    8 0.3787182381014816
    9 0.48583561199925934
    [6, 3, 5]
    """

    """
    e7abcb9f3d5876962b70eac6546e93fd
    0 0.9321469657209958
    1 0.8853038202044261
    2 0.9294365209540291
    3 0.8040079946029252
    4 0.7981785185785435
    5 0.828373443970914
    6 0.8851216596873411
    7 0.9475031084228164
    8 0.9423507306535642
    9 0.8796230559477685
    [7, 9, 3]
    """

    exp_dir = log_dir / project_name / exp_name
    test_dir = exp_dir / "generation" / "test"
    mesh_g_path = test_dir / "meshes" / obj_category / f"{obj_name}.ply"
    inputs_path = test_dir / "inputs" / obj_category / f"{obj_name}.ply"
    mesh_d_path = (
        log_dir
        / "cvpr_2025_depth"
        / "kinect_net"
        / "generation_shapenet"
        / "test"
        / "meshes"
        / obj_category
        / f"{obj_name}.ply"
    )
    mesh_gt_path = dataset_dir / obj_category / obj_name / "model.off"
    out_dir = Path(os.environ.get("FIGURE_ROOT", "figures")) / obj_category / obj_name
    out_dir.mkdir(parents=True, exist_ok=True)

    trafo = np.load(mesh_g_path.with_suffix(".npy"))
    mesh_d = Trimesh(*load_mesh(mesh_d_path), process=False, validate=False)
    mesh_d.apply_transform(trafo)
    offset = mesh_d.bounds.mean(axis=0)
    scale = mesh_d.extents.max()  # x1.1 for table
    mesh_d.apply_translation(-offset)
    mesh_d.apply_scale(1 / scale)

    inputs = o3d.io.read_point_cloud(str(inputs_path))
    inputs = trimesh.PointCloud(np.asarray(inputs.points))
    inputs.apply_transform(trafo)
    # inputs.apply_translation(-offset)
    inputs.apply_scale(1 / scale)

    mesh_gt = Trimesh(*load_mesh(mesh_gt_path), process=False, validate=False)
    mesh_gt = normalize_mesh(mesh_gt)
    # mesh_gt.apply_translation(-offset)
    mesh_gt.apply_scale(1 / scale)

    mesh_g_list = list()
    for i in np.arange(10):
        mesh = Trimesh(
            *load_mesh(mesh_g_path.parent / (f"{obj_name}.ply" if i == 0 else f"{obj_name}_{i}.ply")),
            process=False,
            validate=False,
        )
        mesh.apply_transform(trafo)
        mesh.apply_translation(-offset)
        mesh.apply_scale(1 / scale)
        mesh_g_list.append(mesh)

    plane.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    plane.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    plane.apply_translation([0, -0.505, 0])

    if args.show:
        frame = cast(Any, o3d).geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        cast(Any, o3d).visualization.draw_geometries(
            [
                cast(Any, o3d).geometry.PointCloud(cast(Any, o3d).utility.Vector3dVector(inputs)).paint_uniform_color(
                    renderer.default_pcd_color
                ),
                cast(Any, mesh_gt).as_open3d.compute_vertex_normals().paint_uniform_color([0.7, 0.7, 0.7]),
                cast(Any, mesh_d).as_open3d.compute_vertex_normals().paint_uniform_color([0.9, 0.9, 0.9]),
                cast(Any, mesh_g_list[0]).as_open3d.compute_vertex_normals().paint_uniform_color(renderer.default_mesh_color),
                cast(Any, plane).as_open3d.compute_vertex_normals().paint_uniform_color([0.5, 0.5, 0.5]),
                frame,
            ]
        )

    for name, obj in zip(["inputs", "gt", "disc", "gen"], [inputs, mesh_gt, mesh_d, mesh_g_list], strict=False):
        center = np.zeros(3)
        inv_extrinsic = look_at(np.array([-0.75, 0.75, -0.75]), center)  # 0.75, 0.75, 0.75 for table, also switch light
        extrinsic = np.asarray(inv_trafo(inv_extrinsic))

        if isinstance(obj, list):
            for o in obj:
                o.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        else:
            obj.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))

        if name == "inputs":
            obj_any = cast(Any, obj)
            image = renderer(
                vertices=cast(Any, [obj_any.vertices, plane_any.vertices]),
                faces=cast(Any, [None, plane_any.faces]),
                colors=cast(Any, [np.array([0.410603, 0.101933, 0.0683599]), "shadow"]),
                extrinsic=cast(Any, extrinsic),
            )["color"]
            if args.show:
                Image.fromarray(image).show()
            Image.fromarray(image).save((out_dir / name).with_suffix(".png"))
        elif name == "gen":
            for i, m in enumerate(cast(list[Any], obj)):
                image = renderer(
                    vertices=cast(Any, [m.vertices, plane_any.vertices]),
                    faces=cast(Any, [m.faces, plane_any.faces]),
                    colors=cast(Any, [np.array([0.165398, 0.558341, 0.416653]), "shadow"]),
                    extrinsic=cast(Any, extrinsic),
                )["color"]
                if args.show:
                    Image.fromarray(image).show()
                Image.fromarray(image).save((out_dir / f"{name}_{i}").with_suffix(".png"))
        else:
            obj_any = cast(Any, obj)
            image = renderer(
                vertices=cast(Any, [obj_any.vertices, plane_any.vertices]),
                faces=cast(Any, [obj_any.faces, plane_any.faces]),
                colors=cast(
                    Any,
                    [
                        np.array([0.410603, 0.101933, 0.0683599])
                        if name == "gt"
                        else np.array([0.502887, 0.494328, 0.456411]),
                        "shadow",
                    ],
                ),
                extrinsic=cast(Any, extrinsic),
            )["color"]
            if args.show:
                Image.fromarray(image).show()
            Image.fromarray(image).save((out_dir / name).with_suffix(".png"))


if __name__ == "__main__":
    main()
