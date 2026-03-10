from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from process.scripts import sdf_from_mesh as sfm


def test_uniform_grid_sampling_shape_and_bounds() -> None:
    grid = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    bounds = np.array([-1.0, -2.0, -3.0, 1.0, 2.0, 3.0], dtype=np.float32)

    samples = sfm.uniform_grid_sampling(grid, bounds, num_points=25)

    assert samples.shape == (25, 4)
    assert np.all(samples[:, 0] >= bounds[0]) and np.all(samples[:, 0] <= bounds[3])
    assert np.all(samples[:, 1] >= bounds[1]) and np.all(samples[:, 1] <= bounds[4])
    assert np.all(samples[:, 2] >= bounds[2]) and np.all(samples[:, 2] <= bounds[5])
    assert set(np.unique(samples[:, 3])).issubset(set(grid.flatten()))


def test_uniform_grid_sampling_with_mask_only_uses_masked_voxels() -> None:
    grid = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    bounds = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    mask = np.zeros_like(grid, dtype=bool)
    mask[0, 1, 2] = True
    mask[2, 0, 1] = True
    allowed = {grid[0, 1, 2], grid[2, 0, 1]}

    samples = sfm.uniform_grid_sampling(grid, bounds, num_points=40, mask=mask)

    assert samples.shape == (40, 4)
    assert set(np.unique(samples[:, 3])).issubset(allowed)


def test_uniform_random_sampling_clips_points_to_bounds() -> None:
    grid = np.zeros((3, 3, 3), dtype=np.float32)
    bounds = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    points = np.array(
        [
            [-5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )

    samples = sfm.uniform_random_sampling(grid, bounds, points=points, ignore_bounds=False)

    assert samples.shape == (3, 4)
    assert np.all(samples[:, 0] >= bounds[0]) and np.all(samples[:, 0] <= bounds[3])
    assert np.all(samples[:, 1] >= bounds[1]) and np.all(samples[:, 1] <= bounds[4])
    assert np.all(samples[:, 2] >= bounds[2]) and np.all(samples[:, 2] <= bounds[5])


def test_path_helpers_non_shapenet(tmp_path: Path) -> None:
    args = Namespace(shapenet=False, o=str(tmp_path), version=1)

    mesh_path = sfm.get_mesh_path("/data/object_a.obj", args)
    sdf_path = sfm.get_sdf_path(mesh_path, args)
    sample_path = sfm.get_sample_path(sdf_path, args)

    assert mesh_path == str(tmp_path / "mesh" / "object_a.obj")
    assert sdf_path == str(tmp_path / "sdf" / "object_a.dist")
    assert sample_path == str(tmp_path / "samples" / "object_a.npz")


def test_path_helpers_shapenet(tmp_path: Path) -> None:
    args = Namespace(shapenet=True, o=str(tmp_path), version=1)
    src_path = "/root/02876657/abcd/model.obj"

    mesh_path = sfm.get_mesh_path(src_path, args)
    sdf_path = sfm.get_sdf_path(mesh_path, args)
    sample_path = sfm.get_sample_path(sdf_path, args)

    assert mesh_path == str(tmp_path / "02876657" / "abcd" / "abcd.obj")
    assert sdf_path == str(tmp_path / "02876657" / "abcd" / "abcd.dist")
    assert sample_path == str(tmp_path / "02876657" / "abcd" / "abcd.npz")


def test_load_sdf_parses_binary_layout(tmp_path: Path) -> None:
    resolution = 2
    sdf_path = tmp_path / "sample.dist"
    ress = np.array([-resolution, resolution, resolution], dtype=np.int32)
    bounds = np.array([-1.0, -2.0, -3.0, 1.0, 2.0, 3.0], dtype=np.float64)
    values = np.arange((resolution + 1) ** 3, dtype=np.float32).reshape(
        resolution + 1,
        resolution + 1,
        resolution + 1,
    )
    sdf_path.write_bytes(ress.tobytes() + bounds.tobytes() + values.tobytes())

    sdf = sfm.load_sdf(str(sdf_path), resolution=resolution)
    parsed_values = np.asarray(sdf["values"])
    parsed_bounds = np.asarray(sdf["bounds"], dtype=np.float32)

    assert parsed_values.shape == (3, 3, 3)
    assert np.allclose(parsed_bounds, bounds.astype(np.float32))
    assert np.array_equal(parsed_values, values)


def test_mesh_from_sdf_vega_requires_output_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_compute_marching_cubes(sdf_path: str, output_path: str, **kwargs: object) -> bool:
        calls["sdf_path"] = sdf_path
        calls["output_path"] = output_path
        calls["kwargs"] = kwargs
        return True

    monkeypatch.setattr(sfm, "compute_marching_cubes", fake_compute_marching_cubes)

    assert sfm.mesh_from_sdf("in.dist", method="vega", o="out.ply", marker=7) is True
    assert calls["sdf_path"] == "in.dist"
    assert calls["output_path"] == "out.ply"
    assert calls["kwargs"] == {"method": "vega", "o": "out.ply", "marker": 7}

    with pytest.raises(ValueError, match="Missing output path"):
        sfm.mesh_from_sdf("in.dist", method="vega")


def test_run_skips_when_outputs_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = str(tmp_path / "input.obj")
    Path(mesh).write_text("dummy")
    root_out = tmp_path / "out"

    args = Namespace(
        verbose=False,
        o=str(root_out),
        shapenet=False,
        version=1,
        num_uniform_random=1,
        uniform_sphere_radii=[1.0, 2.0],
        samples=["uniform_random.npy", "uniform_sphere.npy"],
        overwrite=False,
    )

    stem = Path(mesh).stem
    output_base = root_out / stem
    mesh_path = output_base / "mesh" / f"{stem}.obj"
    sdf_path = output_base / "sdf" / f"{stem}.dist"
    sample_dir = output_base / "samples"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sdf_path.write_bytes(b"dist")
    (mesh_path.parent / f"{stem}.ply").write_bytes(b"ply")

    expected_num_samples = (
        args.num_uniform_random
        + len(args.uniform_sphere_radii)
        + len(set(args.samples) & {"uniform_sphere", "uniform_random"})
        + 2
    )
    for idx in range(expected_num_samples):
        (sample_dir / f"sample_{idx}.npy").write_bytes(b"npy")

    monkeypatch.setattr(
        sfm,
        "normalize_mesh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("normalize_mesh should not run")),
    )
    monkeypatch.setattr(
        sfm,
        "compute_distance_field",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compute_distance_field should not run")),
    )
    monkeypatch.setattr(
        sfm,
        "mesh_from_sdf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("mesh_from_sdf should not run")),
    )
    monkeypatch.setattr(
        sfm,
        "sample",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sample should not run")),
    )

    sfm.run(mesh, args)
