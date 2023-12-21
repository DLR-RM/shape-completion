import os
from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt

import pyrender
from ..libkinect import KinectSim


def load_mesh():
    f_kinect_world = np.array([[0.18889654314248513, -0.5491412049704937, 0.8141013875353555, -1.29464394785277],
                               [-0.9455020122657823, -0.325615105774899, -0.0002537971720279581, -0.009541282377521727],
                               [0.2652233842565174, -0.7696874527524885, -0.5807223538112247, 0.5834419201245094],
                               [0.0, 0.0, 0.0, 1.0]])
    x_world_detection_position = np.array([0.25, -0.8, 0.68])
    mesh = trimesh.load_mesh(Path(__file__).parent.parent / "libkinect/assets/mesh_gray_bowl.obj")

    # Move a bit towards cam
    mesh = mesh.apply_translation([0, 0.15, 0])

    # Add table
    table = trimesh.primitives.Box([1, 1, 0.6], trimesh.transformations.translation_matrix([0, 0, -0.3]))
    mesh = trimesh.util.concatenate(mesh, table)

    mesh = mesh.apply_translation(x_world_detection_position)
    mesh.apply_transform(f_kinect_world)

    return mesh


def test_kinect(show: bool = True):
    mesh = load_mesh()
    kinect_depth = KinectSim().simulate(mesh.vertices, mesh.faces, verbose=True)

    if show and os.environ.get("DISPLAY") is not None:
        mesh.apply_transform(trimesh.transformations.euler_matrix(np.pi, 0, 0))
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        camera = pyrender.IntrinsicsCamera(fx=582.6989, fy=582.6989, cx=320.7906, cy=245.2647, znear=0.5, zfar=6.0)

        scene = pyrender.Scene()
        scene.add(pyrender_mesh)
        scene.add(camera)

        renderer = pyrender.OffscreenRenderer(640, 480)
        pyrender_depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

        plt.subplots(1, 2, figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(kinect_depth)
        plt.subplot(1, 2, 2)
        plt.imshow(pyrender_depth)
        plt.tight_layout()
        plt.show()
