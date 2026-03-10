from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, cast

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import inference as inference_module
from dataset.src import coco_collate
from train.src import utils as script


class _FakeModel:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self._state = state

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state


class _SwapOptimizer:
    def __init__(self) -> None:
        self.swap_calls = 0

    @contextmanager
    def swap_ema_weights(self) -> Any:
        self.swap_calls += 1
        yield


class _PlainOptimizer:
    pass


class _FakePointCloud:
    def __init__(self, points: list[list[float]] | np.ndarray) -> None:
        self.points = np.asarray(points, dtype=np.float32)


def _base_cfg() -> Any:
    return OmegaConf.create(
        {
            "data": {"split": False, "train_ds": ["dummy_train"]},
            "inputs": {
                "voxelize": False,
                "project": False,
                "num_points": 2048,
                "type": "pointcloud",
                "bps": {"num_points": 0},
            },
            "train": {"num_query_points": 64},
            "points": {"subsample": True, "voxelize": False},
            "load": {"res_modifier": 2},
            "load_3d": False,
            "collate_3d": None,
            "stack_2d": True,
        }
    )


def _test_dataset_cfg(tmp_path: Path, *, filename: str, project: bool, show: bool = False) -> Any:
    return OmegaConf.create(
        {
            "vis": {"show": show},
            "dirs": {"test_root": str(tmp_path)},
            "test": {"dir": "test_root", "filename": filename},
            "inputs": {
                "project": project,
                "num_points": 8,
                "voxelize": False,
                "crop": False,
                "resize": None,
            },
            "norm": {"true_height": False, "scale": 1.5, "center": True, "padding": 0.2},
            "aug": {"scale": None},
        }
    )


def test_item_to_tensor_and_common_collate() -> None:
    tensor = torch.tensor([1.0, 2.0])
    array = np.array([3.0, 4.0], dtype=np.float32)

    assert script._item_to_tensor(tensor) is tensor
    assert torch.equal(script._item_to_tensor(array), torch.tensor([3.0, 4.0]))

    with pytest.raises(TypeError, match="Expected ndarray or Tensor"):
        script._item_to_tensor("bad")

    collated = script.common_collate(
        [
            {"shared": torch.tensor([1]), "only_a": "a"},
            {"shared": torch.tensor([2]), "only_b": "b"},
        ]
    )

    assert set(collated.keys()) == {"shared"}
    assert torch.equal(cast(torch.Tensor, collated["shared"]), torch.tensor([[1], [2]]))


def test_save_best_model_and_save_ema_model(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "checkpoint.ckpt"
    torch.save({"state_dict": {"model.weight": torch.tensor([1.0]), "model.bias": torch.tensor([2.0])}}, ckpt_path)

    saved = script.save_best_model(ckpt_path, tmp_path)
    best_payload = torch.load(tmp_path / "model_best.pt", weights_only=False)

    assert set(saved.keys()) == {"weight", "bias"}
    assert set(best_payload["model"].keys()) == {"weight", "bias"}

    swap_optimizer = _SwapOptimizer()
    ema_model = _FakeModel({"model.weight": torch.tensor([10.0]), "model.bias": torch.tensor([20.0])})
    trainer = type("Trainer", (), {"optimizers": [swap_optimizer]})()

    script.save_ema_model(cast(pl.Trainer, trainer), cast(pl.LightningModule, ema_model), tmp_path, state_dict=saved)

    ema_payload = torch.load(tmp_path / "model_ema.pt", weights_only=False)
    assert swap_optimizer.swap_calls == 1
    assert torch.equal(ema_payload["model"]["weight"], torch.tensor([10.0]))
    assert torch.equal(ema_payload["model"]["bias"], torch.tensor([20.0]))


def test_save_ema_model_uses_attached_ema_module(tmp_path: Path) -> None:
    orig_mod = _FakeModel({"model.weight": torch.tensor([7.0])})
    ema_model = type(
        "LightningModel",
        (),
        {"ema_model": type("EMAWrapper", (), {"module": type("Module", (), {"orig_mod": orig_mod})()})()},
    )()
    trainer = type("Trainer", (), {"optimizers": [_PlainOptimizer()]})()

    script.save_ema_model(cast(pl.Trainer, trainer), cast(pl.LightningModule, ema_model), tmp_path)

    ema_payload = torch.load(tmp_path / "model_ema.pt", weights_only=False)
    assert torch.equal(ema_payload["model"]["weight"], torch.tensor([7.0]))


def test_weight_norm_heterogeneous_collate_and_get_collate_fn(monkeypatch: Any) -> None:
    module = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[3.0, 4.0]]))
        module.bias.copy_(torch.tensor([12.0]))

    norms = script.weight_norm(module, 2)
    assert norms["weight_2.0_norm/weight"] == pytest.approx(5.0)
    assert norms["weight_2.0_norm/bias"] == pytest.approx(12.0)
    assert norms["weight_2.0_norm_total"] == pytest.approx(13.0)

    with pytest.raises(ValueError, match="must be a positive number"):
        script.weight_norm(module, 0)

    monkeypatch.setattr(script, "PYTORCH3D_AVAILABLE", False)
    with pytest.raises(ImportError, match="PyTorch3D"):
        script.heterogeneous_collate([{"inputs": np.zeros((2, 3), dtype=np.float32)}])

    cfg = _base_cfg()
    cfg.data.split = True
    collate_single = script.get_collate_fn(cfg, batch_size=1)
    assert isinstance(collate_single, partial)
    assert collate_single.func is coco_collate

    cfg = _base_cfg()
    cfg.inputs.project = True
    cfg.inputs.num_points = "variable"
    cfg.train.num_query_points = "random"
    cfg.points.subsample = False
    collate_hb = script.get_collate_fn(cfg)
    assert isinstance(collate_hb, partial)
    assert collate_hb.func is script.heterogeneous_collate
    assert collate_hb.keywords == {"res_modifier": 2, "keys": ["inputs", "points"]}

    cfg = _base_cfg()
    cfg.data.train_ds = ["coco"]
    cfg.load_3d = True
    cfg.collate_3d = "list"
    collate_coco = script.get_collate_fn(cfg)
    assert isinstance(collate_coco, partial)
    assert collate_coco.func is coco_collate
    assert collate_coco.keywords == {"list_keys": {"points", "points.occ", "points.indices"}}


def test_get_test_dataset_project_branch_uses_point_cloud_normalization(monkeypatch: Any, tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.ply"
    sample_path.touch()

    cfg = _test_dataset_cfg(tmp_path, filename="*.ply", project=True)
    fake_pcd = _FakePointCloud([[0.0, -0.5, 0.0], [0.5, 0.2, 0.3]])
    plane_model = np.array([0.0, 1.0, 0.0, -0.25], dtype=np.float32)
    normalized = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    offset = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(script, "resolve_path", lambda value: Path(str(value)))
    monkeypatch.setattr(inference_module, "get_point_cloud", lambda path, extrinsic=None: (fake_pcd, None, extrinsic))
    monkeypatch.setattr(inference_module, "remove_plane", lambda pcd: ([pcd], plane_model))

    def _fake_get_input_data_from_point_cloud(
        pcd: Any,
        *,
        num_input_points: int,
        offset_y: float,
        center: bool,
        scale: float,
        crop: float,
        voxelize: bool,
        padding: float,
        show: bool,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        calls.append(
            {
                "pcd": pcd,
                "num_input_points": num_input_points,
                "offset_y": offset_y,
                "center": center,
                "scale": scale,
                "crop": crop,
                "voxelize": voxelize,
                "padding": padding,
                "show": show,
            }
        )
        return normalized, offset, 2.5

    monkeypatch.setattr(inference_module, "get_input_data_from_point_cloud", _fake_get_input_data_from_point_cloud)

    dataset = cast(Any, script.get_test_dataset(cfg))
    assert dataset is not None
    assert len(dataset) == 1

    item = dataset[0]
    np.testing.assert_allclose(item["inputs"], normalized)
    np.testing.assert_allclose(item["inputs.depth"], normalized)
    np.testing.assert_allclose(item["inputs.norm_offset"], offset)
    assert item["inputs.norm_scale"] == pytest.approx(2.5)
    assert item["inputs.path"] == str(sample_path)
    assert item["inputs.width"] == 640
    assert item["inputs.height"] == 480
    assert "inputs.intrinsic" not in item
    assert "inputs.extrinsic" not in item
    assert calls == [
        {
            "pcd": fake_pcd,
            "num_input_points": 8,
            "offset_y": 0,
            "center": True,
            "scale": 1.5,
            "crop": 0.6,
            "voxelize": False,
            "padding": 0.2,
            "show": False,
        }
    ]


def test_get_test_dataset_camera_branch_projects_depth_and_updates_intrinsics(
    monkeypatch: Any, tmp_path: Path
) -> None:
    sample_path = tmp_path / "frame.png"
    sample_path.touch()
    extrinsic_path = tmp_path / "camera.npy"
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(extrinsic_path, extrinsic)

    cfg = _test_dataset_cfg(tmp_path, filename="*.png", project=False, show=True)
    cfg.inputs.crop = True
    cfg.inputs.resize = [3, 5]

    fake_pcd = _FakePointCloud([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    intrinsic = np.eye(3, dtype=np.float32)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    rot_y = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    transformed = np.array([[1.0, 1.5, 2.0], [2.0, 2.5, 3.0]], dtype=np.float32)
    initial_depth = np.arange(16, dtype=np.float32).reshape(4, 4)
    cropped_depth = np.arange(9, dtype=np.float32).reshape(3, 3)
    resized_depth = np.arange(15, dtype=np.float32).reshape(3, 5)
    bbox = (1, 2, 3, 4)
    imshow_calls: list[np.ndarray] = []
    show_calls: list[bool] = []
    adjust_calls: list[dict[str, Any]] = []
    crop_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(script, "resolve_path", lambda value: Path(str(value)))
    monkeypatch.setattr(inference_module, "get_point_cloud", lambda path: (fake_pcd, intrinsic.copy(), None))
    monkeypatch.setattr(inference_module, "get_rot_from_extrinsic", lambda value: (rot_x, rot_y, np.eye(3)))
    monkeypatch.setattr(inference_module, "remove_plane", lambda pcd: ([pcd], np.array([0.0, 1.0, 0.0, 0.0])))

    def _fake_apply_trafo(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        np.testing.assert_allclose(matrix[:3, 3], np.array([0.0, 0.25, 0.75], dtype=np.float32))
        np.testing.assert_allclose(matrix[:3, :3], (rot_x @ rot_y).T)
        np.testing.assert_allclose(points, fake_pcd.points)
        return transformed

    def _fake_adjust_intrinsic(
        matrix: np.ndarray, width: int, height: int, box: Any = None, size: Any = None
    ) -> np.ndarray:
        adjust_calls.append({"width": width, "height": height, "box": box, "size": size, "matrix": matrix.copy()})
        return matrix + len(adjust_calls)

    def _fake_crop_and_resize_image(
        image: np.ndarray, box: Any = None, size: Any = None, interpolation: str | None = None
    ) -> np.ndarray:
        crop_calls.append({"shape": image.shape, "box": box, "size": size, "interpolation": interpolation})
        if box is not None:
            return cropped_depth
        if size is not None:
            assert interpolation == "nearest"
            return resized_depth
        raise AssertionError("Unexpected crop/resize call")

    monkeypatch.setattr(script, "apply_trafo", _fake_apply_trafo)
    monkeypatch.setattr(script, "points_to_depth", lambda points, matrix, width, height: initial_depth)
    monkeypatch.setattr(script, "bbox_from_mask", lambda depth, padding=0.1: bbox)
    monkeypatch.setattr(script, "crop_and_resize_image", _fake_crop_and_resize_image)
    monkeypatch.setattr(script, "adjust_intrinsic", _fake_adjust_intrinsic)
    monkeypatch.setattr(script, "depth_to_image", lambda depth: depth + 100.0)
    monkeypatch.setattr(script.plt, "imshow", lambda image: imshow_calls.append(np.asarray(image)))
    monkeypatch.setattr(script.plt, "show", lambda: show_calls.append(True))

    dataset = cast(Any, script.get_test_dataset(cfg))
    assert dataset is not None

    item = dataset[0]
    np.testing.assert_allclose(item["inputs"], resized_depth)
    np.testing.assert_allclose(item["inputs.depth"], resized_depth)
    np.testing.assert_allclose(item["inputs.intrinsic"], intrinsic + 3)
    expected_extrinsic = extrinsic.copy()
    expected_extrinsic[0, 3] = 0
    expected_extrinsic[1, 3] = 0.25
    expected_extrinsic[2, 3] = 0.75
    expected_extrinsic[:3, :3] = (rot_x @ rot_y).T
    np.testing.assert_allclose(item["inputs.extrinsic"], expected_extrinsic)
    assert item["inputs.width"] == 5
    assert item["inputs.height"] == 3
    assert item["inputs.path"] == str(sample_path)
    assert crop_calls == [
        {"shape": (4, 4), "box": bbox, "size": None, "interpolation": None},
        {"shape": (3, 3), "box": None, "size": [3, 5], "interpolation": "nearest"},
    ]
    assert len(adjust_calls) == 2
    assert {k: v for k, v in adjust_calls[0].items() if k != "matrix"} == {
        "width": 4,
        "height": 4,
        "box": bbox,
        "size": None,
    }
    assert {k: v for k, v in adjust_calls[1].items() if k != "matrix"} == {
        "width": 3,
        "height": 3,
        "box": None,
        "size": [3, 5],
    }
    np.testing.assert_allclose(cast(np.ndarray, adjust_calls[0]["matrix"]), intrinsic)
    np.testing.assert_allclose(cast(np.ndarray, adjust_calls[1]["matrix"]), intrinsic + 1)
    assert len(imshow_calls) == 1
    np.testing.assert_allclose(imshow_calls[0], resized_depth + 100.0)
    assert show_calls == [True]


def test_get_test_dataset_depth_file_preserves_loader_extrinsic(monkeypatch: Any, tmp_path: Path) -> None:
    sample_path = tmp_path / "depth_view.npy"
    np.save(sample_path, np.ones((2, 2), dtype=np.float32))
    extrinsic_path = tmp_path / "camera.npy"
    original_extrinsic = np.eye(4, dtype=np.float32)
    original_extrinsic[:3, 3] = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    np.save(extrinsic_path, original_extrinsic)

    cfg = _test_dataset_cfg(tmp_path, filename="depth_*.npy", project=False)
    fake_pcd = _FakePointCloud([[0.0, 0.0, 1.0]])
    loaded_extrinsic = np.eye(4, dtype=np.float32) * 2.0
    intrinsic = np.eye(3, dtype=np.float32)
    apply_calls: list[np.ndarray] = []

    monkeypatch.setattr(script, "resolve_path", lambda value: Path(str(value)))

    def _fake_get_point_cloud(path: Path, extrinsic: np.ndarray | None = None) -> tuple[_FakePointCloud, np.ndarray, np.ndarray]:
        assert extrinsic is not None
        np.testing.assert_allclose(extrinsic, original_extrinsic)
        return fake_pcd, intrinsic, loaded_extrinsic

    monkeypatch.setattr(inference_module, "get_point_cloud", _fake_get_point_cloud)
    monkeypatch.setattr(inference_module, "remove_plane", lambda pcd: ([pcd], np.array([0.0, 1.0, 0.0, 0.0])))
    monkeypatch.setattr(script, "apply_trafo", lambda points, matrix: apply_calls.append(matrix.copy()) or points)
    monkeypatch.setattr(script, "points_to_depth", lambda points, matrix, width, height: np.array([[7.0]], dtype=np.float32))

    dataset = cast(Any, script.get_test_dataset(cfg))
    assert dataset is not None

    item = dataset[0]
    np.testing.assert_allclose(apply_calls[0], loaded_extrinsic)
    np.testing.assert_allclose(item["inputs"], np.array([[7.0]], dtype=np.float32))
    np.testing.assert_allclose(item["inputs.intrinsic"], intrinsic)
    np.testing.assert_allclose(item["inputs.extrinsic"], loaded_extrinsic)
