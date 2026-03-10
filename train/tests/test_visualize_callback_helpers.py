from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from trimesh import Trimesh

from ..src.callbacks.visualize import VisualizeCallback


def _callback() -> VisualizeCallback:
    callback = cast(VisualizeCallback, VisualizeCallback.__new__(VisualizeCallback))
    callback.n_total = 10
    callback.n_per_category = 1
    callback.items = []
    callback.counter = Counter()
    callback._batch = []
    return callback


def _mesh() -> Trimesh:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return Trimesh(vertices=vertices, faces=faces, process=False)


def test_prepare_batch_filters_loss_keys_and_limits_per_category() -> None:
    callback = _callback()
    callback.n_total = None
    batch: dict[str, Any] = {
        "index": torch.tensor([0, 1, 2]),
        "category.name": ["chair,foo", "chair,bar", "table,baz"],
        "inputs": torch.ones((3, 4, 3)),
        "train.loss": torch.tensor([0.1, 0.2, 0.3]),
    }

    callback._prepare_batch(batch)

    assert len(callback._batch) == 2
    assert callback.counter["chair"] == 1
    assert callback.counter["table"] == 1
    assert all("train.loss" not in item for item in callback._batch)
    assert batch["category.name"] == ["chair", "chair", "table"]


def test_prepare_batch_enforces_n_total_within_single_batch() -> None:
    callback = _callback()
    callback.n_total = 1
    callback.n_per_category = None
    batch: dict[str, Any] = {
        "index": torch.tensor([0, 1, 2]),
        "category.name": ["chair", "table", "lamp"],
        "inputs": torch.ones((3, 4, 3)),
    }

    callback._prepare_batch(batch)

    assert len(callback._batch) == 1


def test_prepare_batch_respects_per_category_even_with_total_limit() -> None:
    callback = _callback()
    callback.n_total = 10
    callback.n_per_category = 1
    batch: dict[str, Any] = {
        "index": torch.tensor([0, 1, 2]),
        "category.name": ["chair", "chair", "table"],
        "inputs": torch.ones((3, 4, 3)),
    }

    callback._prepare_batch(batch)

    assert len(callback._batch) == 2
    assert callback.counter["chair"] == 1
    assert callback.counter["table"] == 1


def test_log_image_accepts_tensor_and_uses_add_image() -> None:
    callback = _callback()
    calls: list[tuple[str, tuple[int, ...], int]] = []
    trainer = SimpleNamespace(
        global_step=7,
        logger=SimpleNamespace(
            experiment=SimpleNamespace(
                add_image=lambda tag, image, global_step: calls.append((tag, tuple(image.shape), global_step))
            )
        ),
    )

    callback._log_image(cast(Any, trainer), torch.zeros((3, 8, 8), dtype=torch.float32), tag="foo")

    assert calls == [("vis/foo", (3, 8, 8), 7)]


def test_log_image_converts_hwc_numpy_to_chw() -> None:
    callback = _callback()
    calls: list[tuple[str, tuple[int, ...], int]] = []
    trainer = SimpleNamespace(
        global_step=3,
        logger=SimpleNamespace(
            experiment=SimpleNamespace(
                add_image=lambda tag, image, global_step: calls.append((tag, tuple(image.shape), global_step))
            )
        ),
    )

    image = np.zeros((5, 6, 3), dtype=np.uint8)
    callback._log_image(cast(Any, trainer), image, tag="bar")

    assert calls == [("vis/bar", (3, 5, 6), 3)]


def test_log_image_is_noop_without_experiment_logger() -> None:
    callback = _callback()
    trainer = SimpleNamespace(global_step=1, logger=SimpleNamespace())

    callback._log_image(cast(Any, trainer), np.zeros((4, 4, 3), dtype=np.uint8), tag="noop")


def test_render_logits_handles_all_zero_logits_without_nan() -> None:
    callback = _callback()
    captured: dict[str, np.ndarray] = {}

    class _Renderer:
        width = 4
        height = 3

        def __call__(self, vertices, colors):
            captured["vertices"] = vertices
            captured["colors"] = colors
            return {"color": np.zeros((3, 4, 3), dtype=np.uint8)}

    callback.__dict__["renderer"] = _Renderer()
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    logits = np.array([0.0, 0.0], dtype=np.float32)

    image = callback._render_logits(points, logits)

    assert image.shape == (3, 4, 3)
    assert np.isfinite(captured["colors"]).all()


def test_on_validation_end_resets_state_and_increments_eval_count() -> None:
    callback = _callback()
    callback.images = {
        "names": [],
        "front": [np.zeros((2, 2, 3), dtype=np.uint8)],
        "back": [np.zeros((2, 2, 3), dtype=np.uint8)],
        "inputs": [],
        "logits": [],
        "depth": [],
        "color": [],
        "normals": [],
    }
    callback.data = {
        "categories": ["chair"],
        "meshes": [],
        "points": [],
        "logits": [],
        "inputs": [],
    }
    callback.items = [{"category.name": "chair"}]
    callback.counter = Counter({"chair": 2})
    callback.eval_count = 5
    callback._done = True

    callback.on_validation_end(cast(Any, SimpleNamespace()), cast(Any, SimpleNamespace()))

    assert callback.images["front"] == []
    assert callback.images["back"] == []
    assert callback.data["categories"] == []
    assert callback.items == []
    assert callback.counter == Counter()
    assert callback.eval_count == 6
    assert callback._done is False


def test_process_batch_voxel_inputs_are_extracted_per_sample() -> None:
    callback = _callback()
    callback.meshes = True
    callback.inputs = True
    callback.render = None
    callback.upload_to_wandb = False
    callback.progress = False
    callback._generator = cast(
        Any,
        SimpleNamespace(
            query_points=torch.arange(24, dtype=torch.float32).reshape(8, 3),
            model=SimpleNamespace(),
        ),
    )
    callback.generate_batch = cast(Any, lambda _batch, **_kwargs: [_mesh(), _mesh()])
    callback._batch = [
        {
            "index": np.array(0),
            "inputs": np.array(
                [
                    [[1, 0], [0, 0]],
                    [[1, 0], [0, 0]],
                ],
                dtype=np.float32,
            ),
        },
        {
            "index": np.array(1),
            "inputs": np.array(
                [
                    [[0, 0], [1, 0]],
                    [[1, 1], [0, 0]],
                ],
                dtype=np.float32,
            ),
        },
    ]

    callback._process_batch(cast(Any, SimpleNamespace(device=torch.device("cpu"))))

    first = cast(np.ndarray, callback._batch[0]["inputs"])
    second = cast(np.ndarray, callback._batch[1]["inputs"])
    assert first.shape == (2, 3)
    assert second.shape == (3, 3)


def test_process_batch_fps_path_uses_model_device(monkeypatch) -> None:
    callback = _callback()
    callback.meshes = True
    callback.inputs = True
    callback.render = None
    callback.upload_to_wandb = False
    callback.progress = False
    callback._generator = cast(
        Any,
        SimpleNamespace(
            query_points=torch.arange(24, dtype=torch.float32).reshape(8, 3),
            model=SimpleNamespace(fps=2),
        ),
    )
    callback.generate_batch = cast(Any, lambda _batch, **_kwargs: [_mesh(), _mesh()])
    callback._batch = [
        {"index": np.array(0), "inputs": np.ones((4, 3), dtype=np.float32)},
        {"index": np.array(1), "inputs": np.ones((4, 3), dtype=np.float32)},
    ]

    captured: dict[str, Any] = {}

    def _fake_fps(inputs: torch.Tensor, num_samples: int) -> torch.Tensor:
        captured["device"] = inputs.device.type
        captured["num_samples"] = num_samples
        return inputs[:, :num_samples, :]

    import libs

    monkeypatch.setattr(libs, "furthest_point_sample", _fake_fps, raising=False)
    callback._process_batch(cast(Any, SimpleNamespace(device=torch.device("cpu"))))

    assert captured["device"] == "cpu"
    assert captured["num_samples"] == 2
    assert cast(np.ndarray, callback._batch[0]["inputs"]).shape == (2, 3)
    assert cast(np.ndarray, callback._batch[1]["inputs"]).shape == (2, 3)


def test_save_mesh_or_pointcloud_accepts_numpy_points(tmp_path: Path) -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    out = tmp_path / "pointcloud"

    VisualizeCallback.save_mesh_or_pointcloud(out, points)

    assert out.with_suffix(".ply").is_file()


def test_save_persists_numpy_inputs_as_pointcloud_files(tmp_path: Path) -> None:
    callback = _callback()
    callback.upload_to_wandb = True
    callback.images = {
        "names": ["sample"],
        "front": [],
        "back": [],
        "inputs": [],
        "logits": [],
        "depth": [],
        "color": [],
        "normals": [],
    }
    callback.data = {
        "categories": ["chair"],
        "meshes": [],
        "points": [],
        "logits": [],
        "inputs": [np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)],
    }

    callback._save(cast(Any, SimpleNamespace(default_root_dir=str(tmp_path), global_step=0)))

    assert (tmp_path / "vis" / "step_0" / "chair" / "inputs" / "sample.ply").is_file()


def test_save_handles_missing_category_and_name_indices(tmp_path: Path) -> None:
    callback = _callback()
    callback.upload_to_wandb = False
    callback.images = {
        "names": [],
        "front": [np.zeros((2, 2, 3), dtype=np.uint8)],
        "back": [],
        "inputs": [],
        "logits": [],
        "depth": [],
        "color": [],
        "normals": [],
    }
    callback.data = {
        "categories": [],
        "meshes": [],
        "points": [],
        "logits": [],
        "inputs": [],
    }

    callback._save(
        cast(
            Any,
            SimpleNamespace(
                default_root_dir=str(tmp_path),
                global_step=0,
                logger=SimpleNamespace(),
            ),
        )
    )

    assert (tmp_path / "vis" / "step_0" / "item_0" / "images" / "item_0_front.png").is_file()


def test_render_meshes_raises_on_mismatched_pointcloud_count() -> None:
    callback = _callback()
    callback.data = {"categories": [], "meshes": [], "points": [], "logits": [], "inputs": []}
    callback.images = {
        "names": [],
        "front": [],
        "back": [],
        "inputs": [],
        "logits": [],
        "depth": [],
        "color": [],
        "normals": [],
    }
    callback.front = True
    callback.back = False
    callback._render = cast(Any, lambda *_args, **_kwargs: np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        callback._render_meshes([_mesh(), _mesh()], [])
