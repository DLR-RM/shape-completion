import tempfile
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
from PIL import Image
from torch import nn
from trimesh import Trimesh, PointCloud

from .generator import Generator
from .renderer import Renderer


class Visualizer:
    def __init__(self,
                 output_dir: Optional[Path] = None,
                 points_batch_size: int = int(1e6),
                 threshold: float = 0.5,
                 padding: float = 0.1,
                 resolution: int = 128,
                 sdf: bool = False,
                 width: int = 512,
                 height: int = 512,
                 method: str = "auto",
                 file_format: str = "PNG",
                 verbose: bool = False):
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
        self.verbose = verbose

        self._generator = None
        self._renderer = None

        if output_dir is not None:
            output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = output_dir

        self.default_mesh_format = ".obj"
        self.default_image_format = ".png"

    @property
    def generator(self) -> Generator:
        return self._generator

    @generator.setter
    def generator(self, model: nn.Module):
        if self._generator is None:
            resolution = 32 if self.resolution > 64 else self.resolution
            upsampling_steps = int(np.log2(self.resolution) - np.log2(32)) if self.resolution > 64 else 0
            self.points_batch_size = self.points_batch_size if self.resolution > 64 else resolution ** 3
            self._generator = Generator(model,
                                        points_batch_size=self.points_batch_size,
                                        threshold=self.threshold,
                                        padding=self.padding,
                                        resolution0=resolution,
                                        upsampling_steps=upsampling_steps,
                                        sdf=self.sdf)

    @property
    def renderer(self) -> Renderer:
        if self._renderer is None:
            self._renderer = Renderer(method=self.method,
                                      width=self.width,
                                      height=self.height,
                                      file_format=self.file_format,
                                      verbose=self.verbose)
        return self._renderer

    @property
    def file_format(self):
        return self._file_format

    @file_format.setter
    def file_format(self, file_format: str):
        assert file_format in ["PNG", "JPEG"]
        self._file_format = file_format
        self.renderer.file_format = file_format

    def get_mesh(self, inputs: Optional[np.ndarray] = None, logits: Optional[np.ndarray] = None) -> Trimesh:
        assert inputs is not None or logits is not None, "Either inputs or logits must be provided"
        if inputs is not None:
            return self.generator.generate_mesh({"inputs": inputs})
        else:
            return self.generator.extract_mesh(logits)

    def get_image(self,
                  meshes_or_pointclouds: List[Union[Trimesh, PointCloud, np.ndarray]],
                  colors: Optional[List[np.ndarray]] = None,
                  compress: bool = False) -> np.ndarray:
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
        image = self.renderer(vertices, faces, colors)["color"]
        if compress:
            path = tempfile.mkstemp(suffix=".jpg")[1]
            Image.fromarray(image).save(path, optimize=True, quality=95)
            image = np.asarray(Image.open(path))
        return image

    def save(self,
             path: Optional[Path] = None,
             mesh: Optional[Trimesh] = None,
             image: Optional[np.ndarray] = None,
             mesh_format: Optional[str] = None,
             image_format: Optional[str] = None):
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
