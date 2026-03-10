import tempfile
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import trimesh
from PIL import Image
from torch import Tensor
from trimesh import PointCloud, Trimesh

from models import Model

from .generator import Generator
from .renderer import Renderer


class Visualizer:
    def __init__(
        self,
        output_dir: Path | None = None,
        points_batch_size: int = int(1e6),
        threshold: float = 0.5,
        padding: float = 0.1,
        resolution: int = 128,
        sdf: bool = False,
        width: int = 512,
        height: int = 512,
        method: str = "auto",
        file_format: str = "PNG",
    ):
        assert file_format in ["PNG", "JPEG"]

        self.points_batch_size = points_batch_size
        self.threshold = threshold
        self.padding = padding
        self.resolution = resolution
        self.sdf = sdf
        self.width = width
        self.height = height
        self.method = method
        self._file_format = file_format

        self._generator: Generator | None = None

        if output_dir is not None:
            output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = output_dir

        self.default_mesh_format = ".obj"
        self.default_image_format = ".png"

    @property
    def generator(self) -> Generator:
        assert self._generator is not None, "Generator is not initialized. Assign a model first."
        return self._generator

    @generator.setter
    def generator(self, model: Model):
        if self._generator is None:
            upsampling_steps = int(np.log2(self.resolution) - np.log2(32)) if self.resolution > 128 else 0
            self._generator = Generator(
                model,
                points_batch_size=self.points_batch_size,
                threshold=self.threshold,
                padding=self.padding,
                resolution=self.resolution,
                upsampling_steps=upsampling_steps,
                sdf=self.sdf,
            )

    @cached_property
    def renderer(self) -> Renderer:
        return Renderer(method=self.method, width=self.width, height=self.height, file_format=self.file_format)

    @property
    def file_format(self):
        return self._file_format

    @file_format.setter
    def file_format(self, file_format: str):
        assert file_format in ["PNG", "JPEG"]
        self._file_format = file_format
        self.renderer.file_format = file_format

    def get_mesh(
        self, item: dict[str, str | int | float | np.ndarray | Tensor] | None = None, logits: np.ndarray | None = None
    ) -> Trimesh:
        assert item is not None or logits is not None, "Either inputs or logits must be provided"
        if item is None:
            assert logits is not None
            return self.generator.extract_mesh(logits)
        mesh = self.generator.generate_mesh(item)
        if isinstance(mesh, list):
            return trimesh.util.concatenate(mesh)
        return mesh

    def get_image(
        self,
        meshes_or_pointclouds: list[Trimesh | PointCloud | np.ndarray],
        colors: list[np.ndarray] | None = None,
        compress: bool = False,
    ) -> np.ndarray:
        vertices = list()
        faces = list()
        for mesh_or_pointcloud in meshes_or_pointclouds:
            if isinstance(mesh_or_pointcloud, Trimesh):
                vertices.append(mesh_or_pointcloud.vertices)
                faces.append(mesh_or_pointcloud.faces)
            elif isinstance(mesh_or_pointcloud, PointCloud):
                vertices.append(mesh_or_pointcloud.vertices)
                faces.append(None)
            else:
                vertices.append(mesh_or_pointcloud)
                faces.append(None)
        image = cast(np.ndarray, self.renderer(vertices, faces, colors)["color"])
        if compress:
            path = tempfile.mkstemp(suffix=".jpg")[1]
            Image.fromarray(image).save(path, optimize=True, quality=95)
            image = np.asarray(Image.open(path))
        return image

    def save(
        self,
        path: Path | None = None,
        mesh: Trimesh | None = None,
        image: np.ndarray | None = None,
        mesh_format: str | None = None,
        image_format: str | None = None,
    ):
        if path is None:
            assert self.output_dir is not None, "Not path provided and output directory not set"
            path = self.output_dir
        if mesh is not None:
            if mesh_format is None:
                mesh_format = self.default_mesh_format
            mesh.export(path.with_suffix(mesh_format))
        if image is not None:
            if image_format is None:
                image_format = self.default_image_format
            Image.fromarray(image).save(path.with_suffix(image_format))
