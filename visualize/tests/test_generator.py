import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

from ..src.generator import Generator


class _DummyGeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "dummy"
        self.encoder = nn.Identity()
        self._device_param = nn.Parameter(torch.zeros(1))

    @property
    def device(self) -> torch.device:
        return self._device_param.device

    def encode(self, **item: Any) -> torch.Tensor:
        inputs = item.get("inputs")
        if isinstance(inputs, torch.Tensor):
            return inputs.mean(dim=-1, keepdim=True)
        return torch.zeros((1, 1, 1), device=self.device)

    def decode(self, points: torch.Tensor, **_: Any) -> torch.Tensor:
        # Signed function with analytic gradient for normal estimation tests.
        return points.squeeze(0)[..., 2]

    def predict(self, points: torch.Tensor, **_: Any) -> torch.Tensor:
        # Occupancy logit: points with x < 0 are inside.
        return -10.0 * points[..., 0]


class _StreamingGeneratorModel(_DummyGeneratorModel):
    def __init__(self):
        super().__init__()
        self.chunk_sizes: list[int] = []

    def predict(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("streaming surface extraction must not call dense predict()")

    def stream_instance_logits(
        self,
        *,
        points: torch.Tensor,
        points_batch_size: int,
        **_: Any,
    ):
        instance_indices = [3]
        yield {"instance_indices": instance_indices, "num_points": points.size(1)}
        for start in range(0, points.size(1), points_batch_size):
            end = min(start + points_batch_size, points.size(1))
            chunk = points[:, start:end]
            self.chunk_sizes.append(end - start)
            logits = -10.0 * chunk[0, :, 0].unsqueeze(0)
            yield {"start": start, "end": end, "instance_indices": instance_indices, "logits": logits}


class _CurvedStreamingGeneratorModel(_StreamingGeneratorModel):
    def stream_instance_logits(
        self,
        *,
        points: torch.Tensor,
        points_batch_size: int,
        **_: Any,
    ):
        instance_indices = [5]
        yield {"instance_indices": instance_indices, "num_points": points.size(1)}
        for start in range(0, points.size(1), points_batch_size):
            end = min(start + points_batch_size, points.size(1))
            chunk = points[:, start:end]
            self.chunk_sizes.append(end - start)
            x = chunk[0, :, 0]
            # Root is x = 0.2 ** (1 / 3). Linear edge interpolation on a coarse
            # grid is biased; bracketed secant refinement should query the model
            # and move points onto the true implicit surface.
            logits = (x**3 - 0.2).unsqueeze(0)
            yield {"start": start, "end": end, "instance_indices": instance_indices, "logits": logits}


class _NewtonStreamingGeneratorModel(_StreamingGeneratorModel):
    def stream_instance_logits(
        self,
        *,
        points: torch.Tensor,
        points_batch_size: int,
        **_: Any,
    ):
        instance_indices = [7]
        yield {"instance_indices": instance_indices, "num_points": points.size(1)}
        for start in range(0, points.size(1), points_batch_size):
            end = min(start + points_batch_size, points.size(1))
            chunk = points[:, start:end]
            self.chunk_sizes.append(end - start)
            x = chunk[0, :, 0]
            y = chunk[0, :, 1]
            z = chunk[0, :, 2]
            logits = (x.square() + 0.5 * y + 0.25 * z - 0.2).unsqueeze(0)
            yield {"start": start, "end": end, "instance_indices": instance_indices, "logits": logits}


def _make_generator(**kwargs: Any) -> Generator:
    return Generator(
        model=cast(Any, _DummyGeneratorModel()),
        resolution=kwargs.pop("resolution", 16),
        padding=kwargs.pop("padding", 0.0),
        upsampling_steps=kwargs.pop("upsampling_steps", 0),
        use_skimage=kwargs.pop("use_skimage", False),
        bounds=kwargs.pop("bounds", ((-1.0, -0.5, -0.25), (1.0, 0.5, 0.25))),
        **kwargs,
    )


def test_query_points_follow_rectangular_bounds():
    generator = _make_generator(resolution=16, padding=0.0)
    nx, ny, nz = generator.grid_shape
    assert (nx, ny, nz) == (16, 8, 4)
    assert generator.query_points.shape == (nx * ny * nz, 3)

    points = generator.query_points.numpy()
    expected_min = np.array([-0.9375, -0.4375, -0.1875], dtype=np.float32)
    expected_max = np.array([0.9375, 0.4375, 0.1875], dtype=np.float32)
    np.testing.assert_allclose(points.min(axis=0), expected_min, atol=1e-6)
    np.testing.assert_allclose(points.max(axis=0), expected_max, atol=1e-6)


def test_generate_grid_predict_path_returns_rectangular_grid():
    generator = _make_generator(resolution=12, padding=0.0)
    item = {"inputs": torch.zeros((1, 4, 3), dtype=torch.float32)}

    grid, points, feature = generator.generate_grid(item=cast(dict[str, Any], item))

    assert isinstance(grid, np.ndarray)
    assert grid.shape == generator.grid_shape
    assert points.shape == (np.prod(generator.grid_shape), 3)
    assert feature is None

    # Occupancy rule from dummy model: x < 0 -> positive logit.
    assert float(grid[0, 0, 0]) > 0
    assert float(grid[-1, 0, 0]) < 0


def test_generate_surface_per_instance_streaming_uses_chunked_logits():
    model = _StreamingGeneratorModel()
    generator = Generator(
        model=cast(Any, model),
        resolution=8,
        padding=0.0,
        points_batch_size=17,
        use_skimage=False,
        bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    )
    item = {"inputs": torch.zeros((1, 4, 3), dtype=torch.float32)}

    surfaces = generator.generate_surface_per_instance_streaming(item=cast(dict[str, Any], item))

    assert model.chunk_sizes
    assert max(model.chunk_sizes) <= 17
    assert len(surfaces) == 1
    surface = surfaces[0]
    assert surface["instance_idx"] == 3
    assert surface["points"].shape[1] == 3
    assert surface["normals"].shape == surface["points"].shape
    assert surface["points"].shape[0] > 0
    np.testing.assert_allclose(surface["points"][:, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(surface["normals"], axis=1), 1.0, atol=1e-6)


def test_generate_surface_per_instance_streaming_refines_edge_crossings():
    model = _CurvedStreamingGeneratorModel()
    generator = Generator(
        model=cast(Any, model),
        resolution=8,
        padding=0.0,
        points_batch_size=64,
        use_skimage=False,
        bounds=((-1.0, -0.5, -0.5), (1.0, 0.5, 0.5)),
        surface_refinement_steps=6,
        surface_refinement_mode="secant",
    )
    item = {"inputs": torch.zeros((1, 4, 3), dtype=torch.float32)}

    surfaces = generator.generate_surface_per_instance_streaming(item=cast(dict[str, Any], item))

    assert len(surfaces) == 1
    surface = surfaces[0]
    expected_x = np.cbrt(0.2)
    np.testing.assert_allclose(surface["points"][:, 0], expected_x, atol=1e-4)


def test_generate_surface_per_instance_streaming_newton_projects_to_field_zero():
    model = _NewtonStreamingGeneratorModel()
    item = {"inputs": torch.zeros((1, 4, 3), dtype=torch.float32)}
    base = Generator(
        model=cast(Any, model),
        resolution=8,
        padding=0.0,
        points_batch_size=128,
        use_skimage=False,
        bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        surface_refinement_mode="linear",
    )
    newton = Generator(
        model=cast(Any, model),
        resolution=8,
        padding=0.0,
        points_batch_size=128,
        use_skimage=False,
        bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        surface_refinement_mode="linear",
        surface_newton_steps=3,
    )

    base_surface = base.generate_surface_per_instance_streaming(item=cast(dict[str, Any], item))[0]
    newton_surface = newton.generate_surface_per_instance_streaming(item=cast(dict[str, Any], item))[0]

    def residual(points: np.ndarray) -> np.ndarray:
        return points[:, 0] ** 2 + 0.5 * points[:, 1] + 0.25 * points[:, 2] - 0.2

    assert np.abs(residual(base_surface["points"])).mean() > 1e-3
    assert np.abs(residual(newton_surface["points"])).max() < 1e-4


def test_generate_surface_per_instance_streaming_autograd_normals_match_field_gradient():
    model = _NewtonStreamingGeneratorModel()
    generator = Generator(
        model=cast(Any, model),
        resolution=8,
        padding=0.0,
        points_batch_size=128,
        use_skimage=False,
        bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        surface_refinement_mode="linear",
        surface_newton_steps=3,
        surface_normal_mode="autograd",
    )
    item = {"inputs": torch.zeros((1, 4, 3), dtype=torch.float32)}

    surface = generator.generate_surface_per_instance_streaming(item=cast(dict[str, Any], item))[0]

    points = surface["points"].astype(np.float64)
    expected = np.stack([2.0 * points[:, 0], np.full(points.shape[0], 0.5), np.full(points.shape[0], 0.25)], axis=1)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(surface["normals"], expected, atol=1e-4)
    assert surface["surface_newton_steps"] == 3
    assert surface["surface_normal_mode"] == "autograd"


def test_extract_mesh_is_empty_for_fully_negative_grid():
    generator = _make_generator(resolution=8, padding=0.0)
    grid = np.full(generator.grid_shape, -10.0, dtype=np.float32)

    mesh = generator.extract_mesh(grid)

    assert len(mesh.vertices) == 0
    assert len(mesh.faces) == 0


def test_extract_mesh_builds_surface_for_positive_center_cube():
    generator = _make_generator(resolution=16, padding=0.0, bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    grid = np.full(generator.grid_shape, -10.0, dtype=np.float32)
    grid[5:11, 5:11, 5:11] = 10.0

    mesh = generator.extract_mesh(grid)

    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_estimate_vertex_normals_returns_unit_vectors():
    generator = _make_generator(resolution=8, padding=0.0)
    vertices = np.array([[0.0, 0.0, 0.0], [0.5, -0.2, 0.7], [-0.1, 0.3, -0.4]], dtype=np.float32)

    normals = generator.estimate_vertex_normals(vertices=vertices, feature=torch.zeros((1, 1)))

    assert isinstance(normals, np.ndarray)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), np.ones(len(vertices)), atol=1e-5)
    np.testing.assert_allclose(normals, np.array([[0.0, 0.0, -1.0]] * len(vertices)), atol=1e-4)


@pytest.fixture(scope="module")
def shapenet_root() -> Path:
    root = os.getenv("SC_SHAPENET_ROOT")
    if root is None:
        pytest.skip("Set SC_SHAPENET_ROOT to run integration generator test")
    path = Path(root)
    if not path.is_dir():
        pytest.skip(f"SC_SHAPENET_ROOT does not exist: {path}")
    return path


def test_generate_integration(shapenet_root: Path):
    from dataset import MeshField, NormalizeMesh, PointCloudField
    from utils.src.voxelizer import Voxelizer

    dataset = "ShapeNetCore.v1.fused.simple"
    synthset = "02958343"
    obj_id = "1a1de15e572e039df085b75b20c2db33"

    pcd_field = PointCloudField(file="samples/surface.npz")
    pcd_data = pcd_field.load(shapenet_root / dataset / synthset / obj_id, index=0)

    mesh_field = MeshField(file="model.off")
    mesh_data_raw = mesh_field.load(shapenet_root / dataset / synthset / obj_id, index=0)
    mesh_data = NormalizeMesh()(cast(dict[str | None, Any], mesh_data_raw))

    padding = 0.1
    resolution = 128
    voxelizer = Voxelizer(resolution=resolution, padding=padding)
    occ, _ = voxelizer(pcd_data[None])

    generator = Generator(
        model=cast(Any, nn.Linear(10, 10)),
        padding=padding,
        resolution=resolution,
        upsampling_steps=0,
        use_skimage=False,
    )
    mesh = generator.extract_mesh(2 * occ - 1)

    assert np.allclose(voxelizer.grid_points, generator.query_points.numpy())
    assert len(mesh_data["vertices"]) > 0
    assert len(mesh.vertices) > 0
