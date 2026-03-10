import numpy as np
import pytest
import torch
from trimesh import Trimesh

from models import ConvONet

from ..src.generator import Generator
from ..src.visualizer import Visualizer


def test_init():
    Visualizer()


def test_generator():
    vis = Visualizer()
    vis.generator = ConvONet()
    assert isinstance(vis.generator, Generator)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_mesh():
    vis = Visualizer()
    vis.generator = ConvONet().cuda().eval()
    inputs = np.random.rand(2048, 3).astype(np.float32)
    mesh = vis.get_mesh({"inputs": inputs})
    assert isinstance(mesh, Trimesh)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_image():
    vis = Visualizer()
    vis.generator = ConvONet().cuda().eval()
    inputs = np.random.rand(2048, 3).astype(np.float32)
    mesh = vis.get_mesh({"inputs": inputs})
    image = vis.get_image([mesh])
    assert isinstance(image, np.ndarray)
