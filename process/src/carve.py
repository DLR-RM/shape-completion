import copy
from pathlib import Path
from typing import Dict, Union
import os

import open3d as o3d
import numpy as np
import mcubes
from trimesh import Trimesh

from utils import setup_logger
from .utils import get_vertices_and_faces


logger = setup_logger(__name__)


def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)


def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def preprocess(model, return_center_scale=False):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    if return_center_scale:
        return model, center, scale
    return model


def voxel_carving(mesh,
                  output_filename,
                  camera_path,
                  cubic_size,
                  voxel_resolution,
                  w=300,
                  h=300,
                  use_depth=True,
                  surface_method='pointcloud'):
    mesh.compute_vertex_normals()
    camera_sphere = o3d.io.read_triangle_mesh(camera_path)

    # setup dense voxel grid
    vc = o3d.geometry.VoxelGrid().create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
        color=[1.0, 0.7, 0.0])

    # rescale geometry
    camera_sphere = preprocess(camera_sphere)
    mesh = preprocess(mesh)

    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    # vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # carve voxel grid
    pcd_agg = o3d.geometry.PointCloud()
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    previous_depth = None
    for cid, xyz in enumerate(camera_sphere.vertices):
        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        centers_pts[cid, :] = c[:3]
        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(do_render=True)
        if previous_depth is not None:
            assert np.allclose(depth, previous_depth)
        previous_depth = copy.deepcopy(np.asarray(depth))

        if cid == 50:
            o3d.visualization.draw_geometries([depth])

        pcd_agg += o3d.geometry.PointCloud().create_from_depth_image(
            o3d.geometry.Image(depth),
            param.intrinsic,
            param.extrinsic,
            depth_scale=1)

        # depth map carving method
        if use_depth:
            vc.carve_depth_map(o3d.geometry.Image(depth), param)
        else:
            vc.carve_silhouette(o3d.geometry.Image(depth), param)
        print("Carve view %03d/%03d" % (cid + 1, len(camera_sphere.vertices)))
    vis.destroy_window()

    # add voxel grid surface
    print('Surface voxel grid from %s' % surface_method)
    if surface_method == 'pointcloud':
        voxel_surface = o3d.geometry.VoxelGrid().create_from_point_cloud_within_bounds(
            pcd_agg,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    elif surface_method == 'mesh':
        voxel_surface = o3d.geometry.VoxelGrid().create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    else:
        raise Exception('invalid surface method')
    voxel_carving_surface = voxel_surface + vc

    return voxel_carving_surface, vc, voxel_surface


def voxel_carving_pipeline(mesh: Union[Trimesh, Dict[str, np.ndarray]],
                           cubic_size: float = 2.0,
                           voxel_resolution: float = 64.0) -> Dict[str, np.ndarray]:
    vertices, faces = get_vertices_and_faces(mesh)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    mesh, center, scale = preprocess(mesh, return_center_scale=True)

    vertices = np.asarray(mesh.vertices)
    min_val = vertices.min(axis=0)
    max_val = vertices.max(axis=0)
    print((min_val + max_val) / 2, (max_val - min_val).max())

    output_filename = os.path.abspath("../../test_data/voxelized.ply")
    camera_path = str(Path(__file__).parent.parent / "assets/sphere.ply")
    visualization = True
    cubic_size = 2.0
    voxel_resolution = 128.0

    voxel_grid, carved_voxels, voxel_surface = voxel_carving(mesh,
                                                             output_filename,
                                                             camera_path,
                                                             cubic_size,
                                                             voxel_resolution)
    print("Carved voxels ...")
    print(carved_voxels)

    voxel_indices = np.asarray([voxel.grid_index for voxel in carved_voxels.get_voxels()])

    points = voxel_indices / (voxel_resolution - 1)
    points = (points - 0.5) * cubic_size

    min_val = points.min(axis=0)
    max_val = points.max(axis=0)
    print((min_val + max_val) / 2, (max_val - min_val).max())

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.paint_uniform_color([0.7, 0.1, 0.1])

    num_points = int(voxel_resolution)
    occupancy = np.zeros((num_points,) * 3, dtype=bool)
    occupancy[tuple(voxel_indices.T)] = True
    # occ_hat_padded = np.pad(occupancy.astype(float), 1, "constant", constant_values=-1e6)
    vertices, triangles = mcubes.marching_cubes(occupancy, 0.5)
    # vertices -= 1
    vertices /= (voxel_resolution - 1)
    vertices = (vertices - 0.5) * cubic_size

    min_val = vertices.min(axis=0)
    max_val = vertices.max(axis=0)
    print((min_val + max_val) / 2, (max_val - min_val).max())

    carved_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    carved_mesh.compute_vertex_normals()
    carved_mesh.paint_uniform_color([0.1, 0.1, 0.7])

    o3d.visualization.draw_geometries([carved_mesh])

    return {"vertices": np.asarray(carved_mesh.vertices), "faces": np.asarray(carved_mesh.triangles)}
