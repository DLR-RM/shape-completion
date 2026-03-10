from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from torchvision.transforms import Compose

from ..src import transforms as transforms_module
from ..src.transforms import (
    BPS,
    AddGaussianNoise,
    Affine,
    ApplyPose,
    CheckDtype,
    Compress,
    CropPointcloud,
    CropPoints,
    FindUncertainPoints,
    ImageToTensor,
    KeysToKeep,
    MinMaxNumPoints,
    Normalize,
    Permute,
    PointcloudFromMesh,
    ProcessPointcloud,
    RandomApply,
    RandomChoice,
    RefinePose,
    Render,
    RenderDepthMaps,
    RotateMesh,
    RotatePointcloud,
    SaveData,
    Scale,
    ShadingImageFromNormals,
    SphereCutPointcloud,
    SphereMovePointcloud,
    SubsamplePoints,
    Transform,
    Translate,
    Unpack,
    apply_transforms,
)


def _str_key_data(data: dict[str, Any]) -> dict[str | None, Any]:
    return cast(dict[str | None, Any], data)


class _AppendKeyTransform(Transform):
    def __init__(self, apply_to: str | list[str] | tuple[str, ...] | None = None):
        super().__init__(apply_to=apply_to)

    def apply(self, data: dict[str | None, Any], key: str | None) -> dict[str | None, Any]:
        data.setdefault("applied_keys", []).append(key)
        return data


class _AddTransform(Transform):
    def __init__(self, delta: int):
        super().__init__()
        self.delta = delta

    def apply(self, data: dict[str | None, Any], key: str | None) -> dict[str | None, Any]:
        data["value"] = int(data["value"]) + self.delta
        return data


def test_transform_call_without_apply_to_uses_none_key():
    transform = _AppendKeyTransform()
    data = {"value": 1}

    out = transform(_str_key_data(data))

    assert out["applied_keys"] == [None]


def test_transform_call_with_apply_to_applies_to_matching_keys_only():
    transform = _AppendKeyTransform(apply_to=("inputs", "points"))
    data = {"inputs": np.zeros((2, 3)), "points": np.zeros((2, 3)), "extra": 1}

    out = transform(_str_key_data(data))

    assert set(out["applied_keys"]) == {"inputs", "points"}
    assert len(out["applied_keys"]) == 2


def test_transform_invalid_apply_to_raises_assertion():
    with pytest.raises(AssertionError):
        _AppendKeyTransform(apply_to=("invalid-key",))


def test_apply_transforms_list_and_compose_paths():
    data = {"value": 0}
    transformed_data = cast(dict[Any, np.ndarray], data.copy())

    out_list = apply_transforms(transformed_data.copy(), transforms=[_AddTransform(1), _AddTransform(2)])
    assert out_list["value"] == 3

    out_single = apply_transforms(transformed_data.copy(), transforms=_AddTransform(4))
    assert out_single["value"] == 4

    compose = Compose([_AddTransform(3)])
    out_compose = apply_transforms(transformed_data.copy(), transforms=compose)
    assert out_compose["value"] == 3

    with pytest.raises(TypeError):
        apply_transforms(transformed_data.copy(), transforms="invalid")  # type: ignore[arg-type]


def test_random_choice_uses_selected_transform(monkeypatch: pytest.MonkeyPatch):
    first = _AddTransform(1)
    second = _AddTransform(10)
    transform = RandomChoice([first, second], p=[0.4, 0.6])

    def _fake_choice(options: int, p: list[float] | None = None) -> int:
        assert p == [0.4, 0.6]
        assert options == 2
        return 1

    monkeypatch.setattr(np.random, "choice", _fake_choice)
    out = transform(_str_key_data({"value": 0}))

    assert out["value"] == 10


def test_random_apply_respects_probability(monkeypatch: pytest.MonkeyPatch):
    transform = RandomApply(_AddTransform(5), p=0.5)

    monkeypatch.setattr(np.random, "rand", lambda: 0.1)
    out_apply = transform(_str_key_data({"value": 0}))
    assert out_apply["value"] == 5

    monkeypatch.setattr(np.random, "rand", lambda: 0.9)
    out_skip = transform(_str_key_data({"value": 0}))
    assert out_skip["value"] == 0


def test_permute_keeps_points_and_occ_aligned(monkeypatch: pytest.MonkeyPatch):
    indices = np.asarray([2, 0, 1], dtype=np.int64)
    monkeypatch.setattr(np.random, "permutation", lambda _: indices)

    data = {
        "points": np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        "points.occ": np.asarray([True, False, True]),
        "points.normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    }
    expected_points = data["points"][indices].copy()
    expected_occ = data["points.occ"][indices].copy()
    original_normals = data["points.normals"].copy()

    out = Permute(apply_to=("points",))(_str_key_data(data))

    np.testing.assert_array_equal(out["points"], expected_points)
    np.testing.assert_array_equal(out["points.occ"], expected_occ)
    np.testing.assert_array_equal(out["points.normals"], original_normals)


def test_permute_keeps_pointcloud_and_normals_aligned(monkeypatch: pytest.MonkeyPatch):
    indices = np.asarray([1, 2, 0], dtype=np.int64)
    monkeypatch.setattr(np.random, "permutation", lambda _: indices)

    data = {
        "pointcloud": np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        "pointcloud.normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    }
    expected_points = data["pointcloud"][indices].copy()
    expected_normals = data["pointcloud.normals"][indices].copy()

    out = Permute(apply_to=("pointcloud",))(_str_key_data(data))

    np.testing.assert_array_equal(out["pointcloud"], expected_points)
    np.testing.assert_array_equal(out["pointcloud.normals"], expected_normals)


def test_min_max_num_points_handles_empty_and_subsample(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(transforms_module, "subsample_indices", lambda array, num: np.arange(int(num)))

    transform = MinMaxNumPoints(
        min_num_points={"inputs": 3, "points": 2},
        max_num_points={"inputs": 2, "points": 3},
        apply_to=("inputs", "points"),
    )

    data = {
        "inputs": np.empty((0, 3), dtype=np.float32),
        "inputs.normals": np.empty((0, 3), dtype=np.float32),
        "inputs.colors": np.empty((0, 3), dtype=np.float32),
        "inputs.path": "dummy-inputs",
        "points": np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        "points.occ": np.asarray([0, 1, 0, 1], dtype=bool),
        "points.path": "dummy-points",
    }
    expected_points = data["points"][:3].copy()
    expected_occ = data["points.occ"][:3].copy()

    out = transform(_str_key_data(data))

    assert out["inputs.skip"] is True
    assert out["inputs"].shape == (3, 3)
    assert out["inputs.normals"].shape == (3, 3)
    assert out["inputs.colors"].shape == (3, 3)
    np.testing.assert_array_equal(out["points"], expected_points)
    np.testing.assert_array_equal(out["points.occ"], expected_occ)


def test_keys_to_keep_filter_and_mutation_methods():
    transform = KeysToKeep(keys=("a",))
    transform.add_key("b")
    transform.add_keys(["c"])
    transform.remove_key("c")

    data = {"a": 1, "b": 2, "c": 3}
    out = transform(_str_key_data(data))

    assert out == {"a": 1, "b": 2}


def test_keys_to_keep_none_keeps_all_data():
    transform = KeysToKeep(keys=None)
    data = {"a": 1, "b": 2}

    out = transform(_str_key_data(data))

    assert out == data


def test_check_dtype_casts_float_arrays_paths_and_sequences():
    transform = CheckDtype(exclude=("keep",), dither=False)
    data = {
        "keep": Path("relative/path.txt"),
        "float_array": np.asarray([1.0, 2.0], dtype=np.float64),
        "int_array": np.asarray([1, 2], dtype=np.int32),
        "path": Path("another/path.txt"),
        "tuple_data": (1, 2, 3),
    }

    out = transform(_str_key_data(data))

    assert isinstance(out["keep"], Path)
    assert isinstance(out["path"], str)
    assert isinstance(out["tuple_data"], list)
    assert out["tuple_data"] == [1, 2, 3]
    assert isinstance(out["float_array"], np.ndarray)
    assert out["float_array"].dtype == np.float32
    assert isinstance(out["int_array"], np.ndarray)
    assert out["int_array"].dtype == np.int32


def test_compress_and_unpack_roundtrip_for_bool_mask_and_float_array():
    compress = Compress(dtype=np.float16, packbits=True)
    unpack = Unpack()
    mask = np.asarray([True, False, True, True, False, False, True, False], dtype=bool)
    data = {
        "float_array": np.asarray([1.0, 2.0], dtype=np.float64),
        "mask": mask,
    }

    compressed = compress(_str_key_data(data))
    assert isinstance(compressed["float_array"], np.ndarray)
    assert compressed["float_array"].dtype == np.float16
    assert isinstance(compressed["mask"], np.ndarray)
    assert compressed["mask"].dtype == np.uint8

    unpacked = unpack(_str_key_data(compressed))
    assert isinstance(unpacked["mask"], np.ndarray)
    np.testing.assert_array_equal(unpacked["mask"][: mask.size], mask.astype(np.uint8))


def test_image_to_tensor_converts_numpy_image_to_tensor():
    transform = ImageToTensor(apply_to="inputs", resize=8, crop=8, normalize=True, format="torchvision")
    image = np.random.randint(0, 255, size=(12, 12, 3), dtype=np.uint8)

    out = transform(_str_key_data({"inputs": image}))

    assert isinstance(out["inputs"], torch.Tensor)
    assert tuple(out["inputs"].shape) == (3, 8, 8)


def test_shading_hdri_fallback_without_map_does_not_crash():
    transform = ShadingImageFromNormals(use_hdri=True, hdri_path=None, replace=False)
    normals = np.zeros((6, 6, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    out = transform(_str_key_data({"inputs.normals": normals}))

    assert "inputs.image" in out
    assert isinstance(out["inputs.image"], PILImage.Image)


def test_shading_hdri_supports_random_ambient_and_diffuse():
    transform = ShadingImageFromNormals(use_hdri=True, ambient="random", diffuse="random")
    transform.hdri = np.full((4, 8, 3), fill_value=128, dtype=np.uint8)
    normals = np.zeros((5, 7, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    shaded = transform._generate_hdri_shading(normals)

    assert shaded.shape == (5, 7, 3)
    assert np.isfinite(shaded).all()
    assert (shaded >= 0).all()
    assert (shaded <= 1).all()


def test_affine_applies_translation_and_sets_inverse_extrinsic():
    trafo = np.eye(4, dtype=np.float32)
    trafo[:3, 3] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    inputs = np.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]], dtype=np.float32)
    transform = Affine(apply_to=("inputs",), trafo=trafo)

    out = transform(_str_key_data({"inputs": inputs.copy()}))

    expected = np.asarray([[1.0, 0.0, 0.0], [3.0, 1.0, -1.0]], dtype=np.float32)
    np.testing.assert_allclose(out["inputs"], expected)
    assert "inputs.inv_extrinsic" in out
    np.testing.assert_allclose(out["inputs.inv_extrinsic"], np.linalg.inv(trafo))


def test_add_gaussian_noise_with_stddev_range_and_return_noise(monkeypatch: pytest.MonkeyPatch):
    value = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    transform = AddGaussianNoise(apply_to="inputs", stddev=(0.1, 0.3), return_noise=True)

    monkeypatch.setattr(np.random, "randn", lambda *shape: np.ones(shape, dtype=np.float32))
    monkeypatch.setattr(np.random, "uniform", lambda low, high: 0.2)

    out = transform(_str_key_data({"inputs": value.copy()}))

    expected_noise = np.ones_like(value) * 0.2
    expected_value = value + expected_noise * 3.0
    np.testing.assert_allclose(out["inputs"], expected_value)
    np.testing.assert_allclose(out["inputs.noise"], expected_noise)


def test_process_pointcloud_identity_path_preserves_arrays(monkeypatch: pytest.MonkeyPatch):
    transform = ProcessPointcloud(apply_to="pointcloud", downsample=None, remove_outlier=None)
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    normals = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    colors = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    monkeypatch.setattr(transforms_module, "process_point_cloud", lambda pcd, **_: pcd)
    out = transform(
        _str_key_data(
            {"pointcloud": points.copy(), "pointcloud.normals": normals.copy(), "pointcloud.colors": colors.copy()}
        )
    )

    np.testing.assert_allclose(out["pointcloud"], points)
    np.testing.assert_allclose(out["pointcloud.normals"], normals)
    np.testing.assert_allclose(out["pointcloud.colors"], colors)


def test_rotate_pointcloud_rotates_points_and_normals():
    transform = RotatePointcloud(axes="z", angles=(90.0,))
    points = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)

    out = transform(_str_key_data({"pointcloud": points.copy(), "pointcloud.normals": normals.copy()}))

    np.testing.assert_allclose(out["pointcloud"], np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64), atol=1e-6)
    np.testing.assert_allclose(out["pointcloud.normals"], np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float64), atol=1e-6)


def test_rotate_mesh_rotates_vertices_and_normals():
    transform = RotateMesh(axes="x", angles=(180.0,))
    vertices = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)
    normals = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)

    out = transform(_str_key_data({"mesh.vertices": vertices.copy(), "mesh.normals": normals.copy()}))

    np.testing.assert_allclose(out["mesh.vertices"], np.asarray([[0.0, -1.0, -2.0]], dtype=np.float64), atol=1e-6)
    np.testing.assert_allclose(out["mesh.normals"], np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64), atol=1e-6)


def test_apply_pose_applies_transform_to_points_and_not_normals():
    transform = ApplyPose()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)

    out = transform(_str_key_data({"inputs.pose": pose, "points": points.copy(), "points.normals": normals.copy()}))

    np.testing.assert_allclose(out["points"], np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(out["points.normals"], normals)
    np.testing.assert_allclose(out["pose"], pose)


def test_crop_pointcloud_frustum_uses_mask(monkeypatch: pytest.MonkeyPatch):
    transform = CropPointcloud(apply_to="inputs", mode="frustum")
    points = np.asarray([[0.0, 0.0, 0.5], [5.0, 5.0, 5.0]], dtype=np.float32)
    mask = np.asarray([True, False], dtype=bool)

    monkeypatch.setattr(transforms_module, "is_in_frustum", lambda **_: mask)
    data = {
        "inputs": points.copy(),
        "inputs.intrinsic": np.eye(3, dtype=np.float32),
        "inputs.extrinsic": np.eye(4, dtype=np.float32),
        "inputs.width": 640,
        "inputs.height": 480,
    }
    out = transform(_str_key_data(data))

    np.testing.assert_allclose(out["inputs"], points[:1])


def test_crop_points_frustum_uses_mask(monkeypatch: pytest.MonkeyPatch):
    transform = CropPoints(mode="frustum")
    points = np.asarray([[0.0, 0.0, 0.5], [5.0, 5.0, 5.0]], dtype=np.float32)
    occ = np.asarray([1, 0], dtype=np.uint8)
    mask = np.asarray([True, False], dtype=bool)

    monkeypatch.setattr(transforms_module, "is_in_frustum", lambda **_: mask)
    data = {
        "points": points.copy(),
        "points.occ": occ.copy(),
        "inputs.intrinsic": np.eye(3, dtype=np.float32),
        "inputs.extrinsic": np.eye(4, dtype=np.float32),
        "inputs.width": 640,
        "inputs.height": 480,
    }
    out = transform(_str_key_data(data))

    np.testing.assert_allclose(out["points"], points[:1])
    np.testing.assert_array_equal(out["points.occ"], occ[:1])


def test_sphere_cut_pointcloud_handles_non_random_radius(monkeypatch: pytest.MonkeyPatch):
    transform = SphereCutPointcloud(apply_to="inputs", radius=0.6, num_spheres=1, random=False, max_percent=1.0)
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    fake_randint = cast(
        Any,
        lambda low, high=None, size=None: 0 if size is None else np.zeros(size, dtype=np.int64),
    )
    monkeypatch.setattr(np.random, "randint", fake_randint)
    out = transform(_str_key_data({"inputs": points.copy(), "inputs.normals": normals.copy()}))

    np.testing.assert_allclose(out["inputs"], points[1:])
    np.testing.assert_allclose(out["inputs.normals"], normals[1:])


def test_sphere_move_pointcloud_handles_non_random_radius(monkeypatch: pytest.MonkeyPatch):
    transform = SphereMovePointcloud(
        apply_to="inputs",
        radius=0.6,
        num_spheres=1,
        random=False,
        offset_amount=0.5,
        inward_probability=0.0,
        max_percent=1.0,
    )
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    fake_randint = cast(
        Any,
        lambda low, high=None, size=None: 0 if size is None else np.zeros(size, dtype=np.int64),
    )
    monkeypatch.setattr(np.random, "randint", fake_randint)
    out = transform(_str_key_data({"inputs": points.copy(), "inputs.normals": normals.copy()}))

    expected = points.copy()
    expected[0, 0] += 0.5
    np.testing.assert_allclose(out["inputs"], expected)


def test_translate_random_scalar_respects_axes_and_updates_extrinsic(monkeypatch: pytest.MonkeyPatch):
    transform = Translate(axes="xy", amount=0.5, random=True)
    points = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)

    def _fake_uniform(low: float, high: float, size: int | None = None) -> np.ndarray:
        assert low == -0.5
        assert high == 0.5
        assert size == 3
        return np.asarray([0.1, -0.2, 0.3], dtype=np.float32)

    monkeypatch.setattr(np.random, "uniform", _fake_uniform)
    out = transform(_str_key_data({"points": points.copy(), "inputs.extrinsic": extrinsic.copy()}))

    np.testing.assert_allclose(out["points"], np.asarray([[1.1, 0.8, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(out["offset"], np.asarray([0.1, -0.2, 0.0], dtype=np.float32))
    np.testing.assert_allclose(out["inputs.extrinsic"][:3, 3], np.asarray([-0.1, 0.2, 0.0], dtype=np.float32))


def test_translate_reverse_uses_previous_offset():
    transform = Translate(reverse=True)
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    previous_offset = np.asarray([0.3, -0.5, 0.2], dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)

    out = transform(
        _str_key_data({"points": points.copy(), "offset": previous_offset, "inputs.extrinsic": extrinsic.copy()})
    )

    np.testing.assert_allclose(out["points"], np.asarray([[-0.3, 0.5, -0.2]], dtype=np.float32))
    np.testing.assert_allclose(out["offset"], -previous_offset)
    np.testing.assert_allclose(out["inputs.extrinsic"][:3, 3], previous_offset)


def test_subsample_points_frustum_volume_requires_camera_metadata():
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError):
        SubsamplePoints._sample_volume(num_samples=2, points=points, volume="frustum")


def test_subsample_points_frustum_volume_path_with_mocked_rays(monkeypatch: pytest.MonkeyPatch):
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)

    def _fake_get_rays(
        depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, num_samples: int | None = None, **_: Any
    ) -> tuple[np.ndarray, np.ndarray, None, None]:
        assert depth.shape == (2, 3)
        assert intrinsic.shape == (3, 3)
        assert extrinsic.shape == (4, 4)
        assert num_samples == 2
        ray0 = np.zeros((2, 3), dtype=np.float32)
        ray_dirs = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return ray0, ray_dirs, None, None

    monkeypatch.setattr(transforms_module, "get_rays", _fake_get_rays)
    monkeypatch.setattr(
        transforms_module, "sample_distances", lambda n_points, near, far: np.ones((n_points, 1), dtype=np.float32)
    )

    indices = SubsamplePoints._sample_volume(
        num_samples=2,
        points=points,
        volume="frustum",
        intrinsic=np.eye(3, dtype=np.float32),
        extrinsic=np.eye(4, dtype=np.float32),
        width=3,
        height=2,
    )

    assert isinstance(indices, np.ndarray)
    assert tuple(indices.shape) == (2,)
    assert (indices >= 0).all()
    assert (indices < len(points)).all()


def test_normalize_points_reference_uses_occupied_points_for_center():
    transform = Normalize(reference="points", center="xyz", scale=False)
    points = np.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]], dtype=np.float32)
    occ = np.asarray([0, 1], dtype=np.uint8)

    out = transform(_str_key_data({"points": points.copy(), "points.occ": occ.copy()}))

    np.testing.assert_allclose(out["points"], np.asarray([[-2.0, -1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32))


def test_normalize_true_height_without_pointcloud_is_safe():
    transform = Normalize(reference="points", center="xyz", scale=False, true_height=True)
    points = np.asarray([[1.0, 0.0, 0.0], [3.0, 2.0, 0.0]], dtype=np.float32)
    occ = np.asarray([1, 1], dtype=np.uint8)

    out = transform(_str_key_data({"points": points.copy(), "points.occ": occ.copy()}))

    np.testing.assert_allclose(out["points"], np.asarray([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32))


def test_refine_pose_skips_icp_when_too_few_points(monkeypatch: pytest.MonkeyPatch):
    transform = RefinePose(projective_icp=False)
    src = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    tgt = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    def _fail_process_point_cloud(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("process_point_cloud should not be called for small inputs")

    monkeypatch.setattr(transforms_module, "process_point_cloud", _fail_process_point_cloud)
    out = transform(
        _str_key_data(
            {"mesh.vertices": src.copy(), "inputs": tgt.copy(), "inputs.intrinsic": np.eye(3, dtype=np.float32)}
        )
    )

    np.testing.assert_allclose(out["mesh.vertices"], src)


def test_render_apply_normalizes_tensor_camera_params(monkeypatch: pytest.MonkeyPatch):
    transform = Render(method="open3d", render_color=False, render_depth=True, render_normals=False, sample_cam=False)
    captured: dict[str, np.ndarray] = {}

    def _fake_render_open3d(
        vertices: np.ndarray, triangles: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        captured["vertices"] = vertices
        captured["triangles"] = triangles
        captured["intrinsic"] = intrinsic
        captured["extrinsic"] = extrinsic
        return np.ones((4, 4), dtype=np.float32), None

    monkeypatch.setattr(transform, "render_open3d", _fake_render_open3d)
    out = transform(
        _str_key_data(
            {
                "mesh.vertices": torch.zeros((3, 3), dtype=torch.float32),
                "mesh.triangles": torch.zeros((1, 3), dtype=torch.int64),
                "mesh.path": "dummy_mesh.obj",
                "mesh.name": "dummy",
                "inputs.intrinsic": torch.eye(3, dtype=torch.float32),
                "inputs.extrinsic": torch.eye(4, dtype=torch.float32),
            }
        )
    )

    assert isinstance(captured["intrinsic"], np.ndarray)
    assert isinstance(captured["extrinsic"], np.ndarray)
    assert isinstance(captured["vertices"], np.ndarray)
    assert isinstance(captured["triangles"], np.ndarray)
    assert tuple(out["inputs"].shape) == (4, 4)
    assert out["inputs.width"] == transform.width
    assert out["inputs.height"] == transform.height


def test_bps_apply_with_numpy_inputs_and_normals():
    transform = BPS(num_points=8, method="kdtree", feature=["distance", "delta"], basis="sphere", seed=0)
    inputs = np.asarray(np.random.default_rng(0).random((16, 3)), dtype=np.float32)
    normals = np.asarray(np.random.default_rng(1).random((16, 3)), dtype=np.float32)

    out = transform(_str_key_data({"inputs": inputs.copy(), "inputs.normals": normals.copy()}))

    assert "bps.inputs" in out
    assert "bps.basis" in out
    assert out["inputs"].shape[0] == 8
    assert out["inputs.normals"].shape[0] == 8


def test_scale_from_inputs_multiplier_scales_per_axis():
    transform = Scale(from_inputs=True, multiplier=2.0)
    inputs = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    data = {"inputs": inputs.copy(), "inputs.scale": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)}

    out = transform(_str_key_data(data))

    np.testing.assert_allclose(out["inputs"], np.asarray([[2.0, 4.0, 6.0]], dtype=np.float32))


def test_pointcloud_from_mesh_accepts_tensor_vertices_and_triangles():
    transform = PointcloudFromMesh(num_points=8)
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    out = transform(_str_key_data({"mesh.vertices": vertices, "mesh.triangles": triangles}))

    assert isinstance(out["pointcloud"], np.ndarray)
    assert tuple(out["pointcloud"].shape) == (8, 3)


def test_save_data_labels_to_map_accepts_numpy_uv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    saver = SaveData(output_dir=tmp_path, save_inputs=False, save_pointcloud=False, save_mesh=False, save_points=False)
    data = {
        "inputs.labels": np.asarray([5, 7], dtype=np.uint16),
        "inputs": np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        "inputs.intrinsic": np.eye(3, dtype=np.float32),
        "inputs.extrinsic": np.eye(4, dtype=np.float32),
        "inputs.width": 3,
        "inputs.height": 3,
    }

    monkeypatch.setattr(
        transforms_module,
        "points_to_uv",
        lambda *_args, **_kwargs: (
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1, 0], dtype=np.int64),
            np.asarray([True, True], dtype=bool),
        ),
    )
    seg = saver._labels_to_map(data)

    assert seg.shape == (3, 3)
    assert seg[1, 0] == 5
    assert seg[0, 1] == 7


def test_save_data_preds_to_map_accepts_numpy_uv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    saver = SaveData(output_dir=tmp_path, save_inputs=False, save_pointcloud=False, save_mesh=False, save_points=False)
    data = {
        "inputs.logits": torch.tensor([[10.0, -10.0], [-10.0, 10.0]], dtype=torch.float32),
        "inputs": np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        "inputs.intrinsic": np.eye(3, dtype=np.float32),
        "inputs.extrinsic": np.eye(4, dtype=np.float32),
        "inputs.width": 3,
        "inputs.height": 3,
    }

    monkeypatch.setattr(
        transforms_module,
        "points_to_uv",
        lambda *_args, **_kwargs: (
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([True, True], dtype=bool),
        ),
    )
    seg = saver._preds_to_map(data)

    assert seg.shape == (3, 3)
    assert seg[0, 0] == 1
    assert seg[1, 1] == 2


def test_save_data_save_points_skips_missing_occ_chunk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    saver = SaveData(output_dir=tmp_path, save_inputs=False, save_pointcloud=False, save_mesh=False, save_points=False)
    writes: list[str] = []

    monkeypatch.setattr(transforms_module.logger, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        transforms_module.o3d.io,
        "write_point_cloud",
        lambda path, pcd: writes.append(str(path)),
    )
    data = {
        "points": [
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        ],
        "points.occ": [np.asarray([1, 0], dtype=np.uint8), None],
    }

    saver._save_points(data, "00000000")

    assert len(writes) == 2
    assert any("00000000_occ_0.ply" in path for path in writes)
    assert any("00000000_free_0.ply" in path for path in writes)


def test_find_uncertain_points_check_occupancy_filters_empty_and_short_lists():
    occ = np.asarray([True, False, True], dtype=bool)
    empty = np.asarray([], dtype=bool)

    kept = FindUncertainPoints.check_occupancy([occ, empty, occ, occ, occ])
    assert len(kept) == 4
    assert all(len(x) > 0 for x in kept)

    dropped = FindUncertainPoints.check_occupancy([occ, empty, occ])
    assert dropped == []


def test_find_uncertain_points_eval_single_returns_empty_array_on_size_skip(monkeypatch: pytest.MonkeyPatch):
    transform = FindUncertainPoints(
        depth_list=[np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)], angle_list=[0.0, 10.0]
    )
    transform.init_inputs = np.asarray([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)

    fake_pcd = type("FakePCD", (), {"points": np.asarray([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)})()
    monkeypatch.setattr(transforms_module, "convert_depth_image_to_point_cloud", lambda *args, **kwargs: fake_pcd)

    out = transform.eval_single(
        data={
            "inputs.intrinsic": np.eye(3, dtype=np.float32),
            "mesh.vertices": np.zeros((3, 3), dtype=np.float32),
            "mesh.triangles": np.zeros((1, 3), dtype=np.int64),
            "points": np.zeros((2, 3), dtype=np.float32),
        },
        scale=1.0,
        depth=np.ones((2, 2), dtype=np.float32),
        angle=0.0,
    )

    assert isinstance(out, np.ndarray)
    assert out.dtype == bool
    assert out.size == 0


def test_find_uncertain_points_get_uncertain_uses_supported_rotate_signature(monkeypatch: pytest.MonkeyPatch):
    transform = FindUncertainPoints(
        depth_list=[np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)], angle_list=[0.0, 10.0]
    )
    called: dict[str, Any] = {}

    class _FakeRotate:
        def __init__(self, **kwargs: Any):
            called.update(kwargs)

        def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
            return {"points": data["points"], "inputs": data["inputs"]}

    class _FakeSKDTree:
        def __init__(self, points: np.ndarray):
            self.points = points

        def query(self, query: np.ndarray, k: int = 1, workers: int = 1) -> tuple[np.ndarray, None]:
            q = np.asarray(query)
            if k == 2:
                return np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (len(q), 1)), None
            return np.ones((len(q),), dtype=np.float32) * 2.0, None

    class _FakePointCloud:
        def __init__(self, points: Any):
            self.points = points

        def remove_statistical_outlier(self, nb_neighbors: int, std_ratio: float) -> tuple[None, np.ndarray]:
            return None, np.asarray([0], dtype=np.int64)

    monkeypatch.setattr(transforms_module, "Rotate", _FakeRotate)
    monkeypatch.setattr(transforms_module, "SKDTree", _FakeSKDTree)
    monkeypatch.setattr(transforms_module.o3d.utility, "Vector3dVector", lambda x: x)
    monkeypatch.setattr(transforms_module.o3d.geometry, "PointCloud", _FakePointCloud)

    data = {
        "points": np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
        "inputs": np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32),
    }
    occupancy = [
        np.asarray([True, True, False], dtype=bool),
        np.asarray([True, True, True], dtype=bool),
        np.asarray([True, True, False], dtype=bool),
        np.asarray([True, True, True], dtype=bool),
    ]

    uncertain, always, sometimes = transform.get_uncertain(data, occupancy)

    assert called["axes"] == "x"
    assert called["from_inputs"] is True
    assert uncertain.shape == always.shape == sometimes.shape == (3,)


def test_render_depth_maps_update_camera_requires_state(monkeypatch: pytest.MonkeyPatch):
    class _FakeRenderer:
        def render(self, scene: Any, flags: Any) -> np.ndarray:
            return np.zeros((2, 2), dtype=np.float32)

        def delete(self) -> None:
            return None

    monkeypatch.setattr(transforms_module.pyrender, "OffscreenRenderer", lambda *args, **kwargs: _FakeRenderer())
    transform = RenderDepthMaps(step=90)

    with pytest.raises(ValueError):
        transform.update_camera(0.0)
