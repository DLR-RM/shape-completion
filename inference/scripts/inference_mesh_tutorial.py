from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import numpy as np
import open3d as o3d
import pyrender
import torch
from pyrender.shader_program import ShaderProgramCache
from scipy.spatial.transform import Rotation as R
from trimesh import Trimesh

from models import ConvONet
from utils import inv_trafo, rot_from_euler
from visualize import Generator


def render(mesh: o3d.geometry.TriangleMesh, width: int = 640, height: int = 480) -> tuple[np.ndarray, np.ndarray]:
    trimesh_mesh = Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
    camera = pyrender.IntrinsicsCamera(fx=width, fy=width, cx=width // 2, cy=height // 2, znear=0.01, zfar=10.0)
    pose = np.eye(4)
    pose[2, 3] = np.clip(np.random.normal(1, 0.4), 0.7, 1.5)

    scene = pyrender.Scene()
    scene.add(pyrender_mesh)
    scene.add(camera, pose=pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    shader_dir = Path(__file__).parent.parent.parent / "utils" / "assets" / "shaders"
    renderer_backend = getattr(renderer, "_renderer", None)
    if renderer_backend is not None:
        cast(Any, renderer_backend)._program_cache = ShaderProgramCache(shader_dir=shader_dir)

    render_out = renderer.render(scene)
    if render_out is None:
        raise RuntimeError("pyrender.OffscreenRenderer.render returned None")
    normal_image, depth_image = render_out

    rgbd_image = o3d.geometry.RGBDImage().create_from_color_and_depth(
        o3d.geometry.Image(normal_image.astype(np.uint8)),
        o3d.geometry.Image(depth_image),
        depth_scale=1.0,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False,
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx=width, fy=width, cx=width // 2, cy=height // 2)
    extrinsic = np.asarray(inv_trafo(pose), dtype=np.float64).copy()
    extrinsic[1, :] *= -1
    extrinsic[2, :] *= -1
    pcd = o3d.geometry.PointCloud().create_from_rgbd_image(image=rgbd_image, intrinsic=intrinsic, extrinsic=extrinsic)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.colors) * 2 - 1
    return points, normals


def hidden_point_removal(
    mesh: o3d.geometry.TriangleMesh, number_of_points: int = 50000, distance_factor: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    pcd = mesh.sample_points_uniformly(number_of_points)
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    camera = np.array([0, 0, diameter])
    _, indices = pcd.hidden_point_removal(camera, distance_factor * diameter)
    pcd = pcd.select_by_index(indices)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    return points, normals


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path.expanduser().resolve()))
    mesh = mesh.compute_vertex_normals()
    return mesh


def normalize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    offset = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    max_size = (mesh.get_max_bound() - mesh.get_min_bound()).max()

    mesh = mesh.translate(-offset)
    mesh = mesh.scale(1 / max_size, center=(0, 0, 0))

    return mesh


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    offset = (points.max(axis=0) + points.min(axis=0)) / 2
    max_size = (points.max(axis=0) - points.min(axis=0)).max()
    return (points - offset) / max_size, offset, max_size


def scale_points(points: np.ndarray, scale: float = 1.5) -> tuple[np.ndarray, np.ndarray, float]:
    offset = (points.max(axis=0) + points.min(axis=0)) / 2
    max_size = scale
    return (points - offset) / max_size, offset, max_size


def augment_points(
    points: np.ndarray, normals: np.ndarray, rot: np.ndarray, noise: float = 0.002, edge_noise: float = 0.01
) -> np.ndarray:
    # Undo rotation
    points = points @ rot
    normals = normals @ rot

    # Compute cosine similarity between camera and normals
    cam = rot.T[:, 2]
    cos_sim = np.dot(normals, cam) / (np.linalg.norm(normals, axis=1) * np.linalg.norm(cam))

    # Remove points that are near perpendicular to camera
    indices = np.abs(cos_sim) >= (0.8 - 0.2) * np.random.random_sample(len(points)) + 0.2
    points = points[indices]
    cos_sim = cos_sim[indices]

    # Add noise to points that are near perpendicular to the camera
    indices = np.abs(cos_sim) >= (0.8 - 0.2) * np.random.random_sample(len(points)) + 0.2
    points[~indices] += edge_noise * np.random.randn(*points[~indices].shape)

    # Add noise to all points
    points += noise * np.random.randn(*points.shape)

    # Redo rotation
    points = points @ rot.T
    return points


def load_model(path: Path, padding: float = 0.1) -> ConvONet:
    state_dict = torch.load(path, weights_only=False)["model"]
    model = ConvONet(padding=padding)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return model


def visualize(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    mesh: o3d.geometry.TriangleMesh | None = None,
    mesh_pred: o3d.geometry.TriangleMesh | None = None,
    padding: float = 0.1,
):
    geometries = list()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(normals / 2 + 0.5)
    geometries.append(pcd)

    if mesh is None and mesh_pred is None:
        frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=(0, 0, 0))
        box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-0.5 - padding / 2,) * 3, max_bound=(0.5 + padding / 2,) * 3
        )
        box.color = np.zeros(3)
        geometries.extend([frame, box])
    else:
        if mesh is not None:
            geometries.append(mesh)
        if mesh_pred is not None:
            geometries.append(mesh_pred)
    cast(Any, o3d).visualization.draw_geometries(geometries)


def main():
    parser = ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to the input file")
    parser.add_argument("weights", type=Path, help="Path to the weights file")
    # parser.add_argument('output', type=Path, help='Path to the output file')
    parser.add_argument("-r", "--render", action="store_true", help="Render depth map")
    parser.add_argument("--resolution", type=int, default=48, help="Output resolution")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize the input and output")
    parser.add_argument("--flip_yz", action="store_true", help="Flip y and z axis")
    parser.add_argument("-s", "--scale", type=float, default=0, help="Apply fixed scaling to input")
    parser.add_argument("-p", "--padding", type=float, default=0.1, help="Padding around the input")
    args = parser.parse_args()

    # Load and normalize mesh to unit cube
    mesh = load_mesh(args.input)
    if args.flip_yz:
        rot = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        mesh.rotate(rot, center=(0, 0, 0))
    mesh = normalize_mesh(mesh)

    # Randomly rotate mesh
    rot, pitch = rot_from_euler(axes="yx", upper_hemisphere=True)
    mesh.rotate(rot, center=(0, 0, 0))

    # Randomly scale mesh
    scale = np.clip(np.random.normal(0.3, 0.2), 0.05, 0.5)
    mesh.scale(scale, center=(0, 0, 0))

    # Get input points and normals from mesh
    if args.render:
        points, normals = render(mesh)
    else:
        points, normals = hidden_point_removal(mesh)

    # Augment
    points = augment_points(points, normals, rot, noise=0.002 * scale, edge_noise=0.01 * scale)

    # Transform to robot world frame
    rot_x = R.from_euler("x", -pitch, degrees=True).as_matrix()
    inputs = points @ rot_x.T
    if args.scale:
        inputs, offset, max_size = scale_points(inputs, args.scale)
    else:
        inputs, offset, max_size = normalize_points(inputs)

    if args.vis:
        visualize(inputs)

    # Generate grid
    model = load_model(args.weights, args.padding)
    generator = Generator(model, resolution=args.resolution)
    grid = generator.generate_grid({"inputs": torch.from_numpy(inputs).float().unsqueeze(0).cuda()})[0]

    # Visualize output
    if args.vis:
        mesh_pred = generator.extract_mesh(cast(np.ndarray, grid))
        mesh_pred = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh_pred.vertices), o3d.utility.Vector3iVector(mesh_pred.faces)
        )
        mesh_pred.scale(max_size, center=(0, 0, 0))
        mesh_pred.translate(offset)
        mesh_pred.rotate(rot_x.T, center=(0, 0, 0))
        mesh_pred.compute_vertex_normals()
        mesh_pred.paint_uniform_color([0, 1, 0])

        visualize(points, normals, mesh, mesh_pred, padding=args.padding)


if __name__ == "__main__":
    main()
