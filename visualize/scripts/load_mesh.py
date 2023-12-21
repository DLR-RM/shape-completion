import argparse
import os

import numpy as np
import open3d as o3d
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the data directory.")
    parser.add_argument("-md", "--mesh_dir", type=str, required=False, help="Path to the data directory.")
    args = parser.parse_args()

    for file in os.listdir(args.data_dir):
        if file.endswith(".npz"):
            mesh_path = os.path.join(args.data_dir, file.replace(".npz", ".ply"))
            mesh_id = file.split('_')[0]

            # Load shape completion mesh
            mesh_list = [trimesh.load_mesh(mesh_path, process=False)]

            # Load uncertain region mesh (if available)
            uncertain_path = mesh_path.replace(".ply", "_uncertain.ply")
            if os.path.isfile(uncertain_path):
                mesh_list.append(trimesh.load_mesh(uncertain_path, process=False))

            if args.mesh_dir is not None:
                # Load ground truth mesh
                rotate_list = list()
                if "ShapeNetCore" in args.mesh_dir or "bop" in args.mesh_dir:
                    if "ShapeNetCore" in args.mesh_dir:
                        rotate_list = ["3c0467f96e26b8c6a93445a1757adf6",
                                       "1a1c0a8d4bad82169f0594e65f756cf5",
                                       "1f035aa5fc6da0983ecac81e09b15ea9",
                                       "68f4428c0b38ae0e2469963e6d044dfe",
                                       "f1866a48c2fc17f85b2ecd212557fda0"]
                    if "ShapeNetCore.v1" in args.mesh_dir:
                        gt_mesh_path = os.path.join(args.mesh_dir, mesh_id, "model.obj")
                    elif "ShapeNetCore.v2" in args.mesh_dir:
                        gt_mesh_path = os.path.join(args.mesh_dir, mesh_id, "models", "model_normalized.obj")
                    elif "bop" in args.mesh_dir:
                        gt_mesh_path = os.path.join(args.mesh_dir, args.data_dir.split('/')[-3].split('_')[0],
                                                    f"obj_{mesh_id}.ply")
                    else:
                        raise NotImplementedError

                    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
                    gt_mesh = trimesh.Trimesh(vertices=np.asarray(gt_mesh.vertices),
                                              faces=np.asarray(gt_mesh.triangles),
                                              process=False)
                    if "ShapeNetCore" in args.mesh_dir:
                        bbox = gt_mesh.bounding_box.bounds

                        loc = (bbox[0] + bbox[1]) / 2
                        scale = (bbox[1] - bbox[0]).max()

                        gt_mesh.apply_translation(-loc)
                        gt_mesh.apply_scale(1 / scale)

                        if mesh_id in rotate_list:
                            trafo = trimesh.transformations.euler_matrix(-np.pi, 0, 0)
                            gt_mesh.apply_transform(trafo)
                else:
                    gt_mesh_path = os.path.join(args.mesh_dir, mesh_id, "mesh", f"{mesh_id}.ply")
                    gt_mesh = trimesh.load_mesh(gt_mesh_path, process=False)

                # Load/apply rotation and scale
                mesh_data = np.load(os.path.join(args.data_dir, file), allow_pickle=True)

                scale = mesh_data.get("scale")
                if scale is not None:
                    gt_mesh.apply_scale(scale)

                pose = mesh_data.get("pose")
                if pose is not None and len(pose.shape) > 0:
                    gt_mesh.apply_transform(pose)

                mesh_list.append(gt_mesh)

                """
                rot_mat = trimesh.transformations.euler_matrix(np.pi / 2, 0, 0)
                for index in range(len(mesh_list)):
                    mesh_list[index].apply_transform(rot_mat)
                """

            geometries = [o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)]
            color_list = [(0.7, 0.7, 0.7), (1, 0, 0), (0, 1, 0)]
            for color, mesh in zip(color_list, mesh_list):
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                                 o3d.utility.Vector3iVector(mesh.faces))
                mesh.paint_uniform_color(color)
                mesh.compute_vertex_normals()
                geometries.append(mesh)
            o3d.visualization.draw_geometries(geometries,
                                              window_name=file,
                                              mesh_show_back_face=True,
                                              mesh_show_wireframe=False,
                                              zoom=0.75,
                                              lookat=(0, 0, 0),
                                              front=(0, 0, 1),
                                              up=(0, 1, 0))


if __name__ == "__main__":
    main()
