from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from eval.src import utils as script


class _FakeTrimesh:
    instances: ClassVar[list[_FakeTrimesh]] = []
    invert_calls: ClassVar[int] = 0
    add_calls: ClassVar[int] = 0

    def __init__(self, vertices: Any = None, faces: Any = None, process: bool = False, validate: bool = False) -> None:
        _ = process, validate
        self.vertices = np.asarray(vertices if vertices is not None else np.zeros((0, 3), dtype=np.float32), dtype=np.float32)
        self.faces = np.asarray(faces if faces is not None else np.zeros((0, 3), dtype=np.int32), dtype=np.int32)
        self.face_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (max(len(self.vertices), 2), 1))
        self.transform_calls: list[np.ndarray] = []
        self.translation_calls: list[np.ndarray] = []
        self.scale_calls: list[float] = []
        self.__class__.instances.append(self)

    @property
    def bounds(self) -> np.ndarray:
        return np.stack([self.vertices.min(axis=0), self.vertices.max(axis=0)], axis=0)

    @property
    def is_empty(self) -> bool:
        return len(self.vertices) == 0

    def apply_transform(self, trafo: np.ndarray) -> _FakeTrimesh:
        self.transform_calls.append(np.asarray(trafo, dtype=np.float32))
        self.vertices = self.vertices @ trafo[:3, :3].T + trafo[:3, 3]
        return self

    def apply_translation(self, offset: np.ndarray) -> _FakeTrimesh:
        offset_arr = np.asarray(offset, dtype=np.float32)
        self.translation_calls.append(offset_arr)
        self.vertices = self.vertices + offset_arr
        return self

    def apply_scale(self, factor: float) -> _FakeTrimesh:
        self.scale_calls.append(float(factor))
        self.vertices = self.vertices * float(factor)
        return self

    def copy(self) -> _FakeTrimesh:
        copied = _FakeTrimesh(self.vertices.copy(), self.faces.copy())
        copied.face_normals = self.face_normals.copy()
        return copied

    def invert(self) -> None:
        self.__class__.invert_calls += 1

    def __add__(self, other: _FakeTrimesh) -> _FakeTrimesh:
        self.__class__.add_calls += 1
        vertices = np.concatenate([self.vertices, other.vertices], axis=0)
        faces = self.faces if len(self.faces) > 0 else other.faces
        return _FakeTrimesh(vertices, faces)

    def sample(self, count: int, return_index: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        _ = count
        sample_points = self.vertices[:2].copy()
        indices = np.arange(len(sample_points), dtype=np.int64)
        if return_index:
            return sample_points, indices
        return sample_points


class _FakeNormalizeMesh:
    calls: ClassVar[list[dict[str, np.ndarray]]] = []

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.__class__.calls.append(data)
        return {"mesh.vertices": data["mesh.vertices"] + np.array([1.0, 0.0, 0.0], dtype=np.float32)}


class _FakeRenderFlags:
    ALL_SOLID = 1
    OFFSCREEN = 2
    FLAT = 4


class _FakeScene:
    def __init__(self, bg_color: Any = None) -> None:
        self.bg_color = bg_color
        self.added: list[tuple[Any, Any]] = []

    def add(self, obj: Any, pose: np.ndarray | None = None) -> None:
        self.added.append((obj, pose))


class _FakeRenderer:
    instances: ClassVar[list[_FakeRenderer]] = []

    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.render_calls: list[tuple[_FakeScene, int]] = []
        self.deleted = False
        self.__class__.instances.append(self)

    def render(self, scene: _FakeScene, flags: int) -> tuple[np.ndarray, np.ndarray]:
        self.render_calls.append((scene, flags))
        return np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8), np.zeros(self.size, dtype=np.float32)

    def delete(self) -> None:
        self.deleted = True


class _FakeCamera:
    def __init__(self, yfov: float, aspectRatio: float) -> None:
        self.yfov = yfov
        self.aspect_ratio = aspectRatio


class _FakeLight:
    def __init__(self, color: np.ndarray, intensity: float, **kwargs: Any) -> None:
        self.color = np.asarray(color, dtype=np.float32)
        self.intensity = intensity
        self.kwargs = kwargs


def test_get_threshold_percentage_accuracy_and_confidence() -> None:
    dist = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    thresholds = [0.15, 0.25]

    percentages = script.get_threshold_percentage(dist, thresholds)
    multi_acc = script.accuracy(np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32), np.array([1, 0], dtype=np.int64))
    binary_acc = script.accuracy(
        np.array([0.9, 0.1, 0.6, 0.4], dtype=np.float32),
        np.array([1, 0, 1, 0], dtype=np.int64),
    )
    multi_conf = script.confidence(np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32))
    binary_conf = script.confidence(
        np.array([0.9, 0.1, 0.6, 0.4], dtype=np.float32),
        mean=False,
    )

    assert np.allclose(percentages, np.array([1 / 3, 2 / 3], dtype=np.float64))
    assert multi_acc == 100.0
    assert binary_acc == 100.0
    assert multi_conf == pytest.approx(0.7)
    assert np.allclose(binary_conf, np.array([0.8, 0.8, 0.2, 0.2], dtype=np.float32))


def test_distance_p2p_kdtree_and_invalid_method() -> None:
    points = np.array([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]], dtype=np.float32)
    points_gt = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals_gt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    accuracy, completeness, accuracy_normals, completeness_normals = script.distance_p2p(
        points,
        points_gt,
        normals=normals,
        normals_gt=normals_gt,
        method="kdtree",
    )

    assert np.allclose(accuracy, np.array([0.0, 0.1], dtype=np.float64))
    assert np.allclose(completeness, np.array([0.0, 0.1], dtype=np.float64))
    assert accuracy_normals is not None and np.allclose(accuracy_normals, np.array([1.0, 1.0], dtype=np.float32))
    assert completeness_normals is not None and np.allclose(completeness_normals, np.array([1.0, 1.0], dtype=np.float32))

    with pytest.raises(NotImplementedError, match="Method bogus not implemented"):
        script.distance_p2p(points, points_gt, method="bogus")


def test_eval_pointcloud_and_eval_occupancy(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        script,
        "distance_p2p",
        lambda *args, **kwargs: (
            np.array([0.0, 0.2], dtype=np.float32),
            np.array([0.1, 0.3], dtype=np.float32),
            np.array([0.5, 1.0], dtype=np.float32),
            np.array([0.25, 0.75], dtype=np.float32),
        ),
    )

    pcd_metrics = script.eval_pointcloud(
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        threshold=0.15,
    )
    occ_metrics = script.eval_occupancy(
        torch.tensor([0.9, 0.8, 0.2, 0.1], dtype=torch.float32),
        torch.tensor([1, 0, 1, 0], dtype=torch.int64),
        threshold=0.5,
    )

    assert pcd_metrics == {
        "completeness": pytest.approx(0.2),
        "accuracy": pytest.approx(0.1),
        "chamfer-l1": pytest.approx(0.15),
        "chamfer-l2": pytest.approx(0.035),
        "f1": pytest.approx(0.5),
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(0.5),
        "normals": pytest.approx(0.625),
    }
    assert occ_metrics == {
        "iou": pytest.approx(1 / 3),
        "f1": pytest.approx(0.5),
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(0.5),
        "acc": pytest.approx(0.5),
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
    }


def test_calibration_helpers_and_reliability_diagram(monkeypatch: Any) -> None:
    probabilities = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int64)
    show_calls: list[bool] = []

    monkeypatch.setattr(script.plt, "show", lambda: show_calls.append(True))

    ece, ace, accs, confs = script.expected_calibration_error(probabilities, labels, bins=2)
    fig, ax = plt.subplots()
    script.reliability_diagram(probabilities, labels, bins=2, axis=ax)
    plt.close(fig)

    assert ece == pytest.approx(0.3)
    assert np.allclose(ace, np.array([0.0, -0.3], dtype=np.float64))
    assert np.allclose(accs, np.array([0.0, 1.0], dtype=np.float64))
    assert np.allclose(confs, np.array([0.0, 0.7], dtype=np.float64))
    assert len(ax.patches) == 6
    assert ax.get_xlabel() == "Confidence"
    assert ax.get_ylabel() == "Accuracy"
    assert show_calls == [True]


def test_iou_helpers_and_eval_cls_seg() -> None:
    pred_one_hot = torch.tensor([[[1, 0], [0, 1], [1, 0]]], dtype=torch.int64)
    targets = torch.tensor([[0, 1, 1]], dtype=torch.int64)
    logits_bnc = torch.tensor([[[4.0, 1.0], [1.0, 4.0], [4.0, 1.0]]], dtype=torch.float32)
    logits_bcn = torch.tensor([[[4.0, 1.0], [1.0, 4.0]]], dtype=torch.float32)
    metric_targets = torch.tensor([[0, 1]], dtype=torch.int64)

    per_class = script.iou_per_class(pred_one_hot, targets, num_classes=2)
    batched = script.batched_miou(logits_bnc, targets, num_classes=2)
    calculated = script.calculate_iou(logits_bnc, targets, num_classes=2)
    cls_metrics = script.eval_cls_seg(
        logits_bcn,
        metric_targets,
        metrics=["val/acc_micro", "val/iou_macro", "val/f1_per_class", "val/ece"],
        prefix="val/",
    )

    assert torch.allclose(per_class, torch.tensor([0.5, 0.5]))
    assert batched.item() == pytest.approx(0.5)
    assert calculated.item() == pytest.approx(5 / 6)
    assert cls_metrics["val/acc_micro"] == pytest.approx(1.0)
    assert cls_metrics["val/iou_macro"] == pytest.approx(1.0)
    assert np.allclose(cls_metrics["val/f1_per_class"], np.array([1.0, 1.0], dtype=np.float32))
    assert cls_metrics["val/ece"] >= 0.0


def test_eval_mesh_applies_normalization_pose_and_query_metrics(monkeypatch: Any, tmp_path: Path) -> None:
    pred_path = tmp_path / "pred.obj"
    gt_path = tmp_path / "gt.obj"
    pose_path = tmp_path / "pose.npy"
    mesh_map = {
        str(pred_path): (
            np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0, 1, 1]], dtype=np.int32),
        ),
        str(gt_path): (
            np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0, 1, 1]], dtype=np.int32),
        ),
    }
    pose = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    check_calls: list[np.ndarray] = []
    pointcloud_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def _fake_np_load(path: str | Path) -> np.ndarray:
        if Path(path) == pose_path:
            return pose
        raise AssertionError(path)

    def _fake_check_mesh_contains(mesh: _FakeTrimesh, points: np.ndarray) -> np.ndarray:
        _ = mesh
        check_calls.append(np.asarray(points, dtype=np.float32))
        return np.array([1, 0], dtype=np.int64) if len(check_calls) == 1 else np.array([1, 1], dtype=np.int64)

    def _fake_eval_pointcloud(
        pointcloud: np.ndarray,
        pointcloud_gt: np.ndarray,
        normals: np.ndarray,
        normals_gt: np.ndarray,
    ) -> dict[str, float]:
        _ = normals, normals_gt
        pointcloud_calls.append((np.asarray(pointcloud), np.asarray(pointcloud_gt)))
        return {
            "chamfer-l1": 0.125,
            "chamfer-l2": 0.25,
            "f1": 0.5,
            "precision": 0.6,
            "recall": 0.7,
        }

    def _fake_eval_occupancy(probabilities: torch.Tensor, occupancy: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
        _ = threshold
        assert torch.equal(probabilities.cpu(), torch.tensor([[1, 0]], dtype=torch.int32))
        assert torch.equal(occupancy.cpu(), torch.tensor([[1.0, 1.0]], dtype=torch.float32))
        return {"iou": 0.4, "f1": 0.5, "precision": 0.6, "recall": 0.7, "acc": 0.8, "tp": 1, "fp": 0, "tn": 0, "fn": 1}

    _FakeTrimesh.instances.clear()
    _FakeNormalizeMesh.calls.clear()
    monkeypatch.setattr(script, "Trimesh", _FakeTrimesh)
    monkeypatch.setattr(script, "load_mesh", lambda path: mesh_map[str(path)])
    monkeypatch.setattr(script.np, "load", _fake_np_load)
    monkeypatch.setattr(script, "NormalizeMesh", lambda: _FakeNormalizeMesh())
    monkeypatch.setattr(script, "eval_transformation_data", lambda value: value)
    monkeypatch.setattr(script, "check_mesh_contains", _fake_check_mesh_contains)
    monkeypatch.setattr(script, "eval_pointcloud", _fake_eval_pointcloud)
    monkeypatch.setattr(script, "eval_occupancy", _fake_eval_occupancy)

    result = script.eval_mesh(
        pred_path,
        gt_path,
        query_points=np.array([[1.5, 0.0, 0.0], [3.5, 0.0, 0.0]], dtype=np.float32),
        offset=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        scale=2.0,
        pose=pose_path,
        normalize=True,
    )

    assert np.allclose(_FakeNormalizeMesh.calls[0]["mesh.vertices"], np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(pointcloud_calls[0][0], np.array([[-0.25, 0.0, 0.0], [0.75, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(pointcloud_calls[0][1], np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32))
    assert np.allclose(check_calls[0], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
    assert result["chamfer-l1"] == pytest.approx(0.125)
    assert result["chamfer-l2"] == pytest.approx(0.25)
    assert result["pcd_f1"] == pytest.approx(0.5)
    assert result["pcd_precision"] == pytest.approx(0.6)
    assert result["pcd_recall"] == pytest.approx(0.7)
    assert result["iou"] == pytest.approx(0.4)
    assert result["f1"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.6)
    assert result["recall"] == pytest.approx(0.7)


def test_overwrite_results_pickle_and_text_paths(monkeypatch: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pkl_path = tmp_path / "results.pkl"
    txt_path = tmp_path / "results.txt"
    df = pd.DataFrame({"category name": ["chair", "table"], "score": [1.0, 3.0]})
    df.attrs["epoch"] = {"loss": 0.25}
    df.to_pickle(pkl_path)
    txt_path.write_text("already there\n")
    info_calls: list[str] = []

    monkeypatch.setattr(script.logger, "info", lambda msg: info_calls.append(str(msg)))

    cfg_pkl = OmegaConf.create({"test": {"overwrite": False}})
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert script.overwrite_results(cfg_pkl, pkl_path) is False
    assert cfg_pkl.test.overwrite is False
    pickle_stdout = capsys.readouterr().out
    assert "mean (macro)" in pickle_stdout
    assert "mean (micro)" in pickle_stdout
    assert any("LOSS: 0.25" in msg for msg in info_calls)

    cfg_txt = OmegaConf.create({"test": {"overwrite": False}})
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert script.overwrite_results(cfg_txt, txt_path) is True
    assert cfg_txt.test.overwrite is True
    assert "already there" in capsys.readouterr().out


def test_render_mesh_uses_renderer_camera_and_lights(monkeypatch: Any) -> None:
    fake_mesh_module = SimpleNamespace(from_trimesh=lambda mesh, smooth=False: {"mesh": mesh, "smooth": smooth})
    fake_pyrender = SimpleNamespace(
        OffscreenRenderer=_FakeRenderer,
        Scene=_FakeScene,
        Mesh=fake_mesh_module,
        PerspectiveCamera=_FakeCamera,
        DirectionalLight=_FakeLight,
        SpotLight=_FakeLight,
        PointLight=_FakeLight,
        RenderFlags=_FakeRenderFlags,
    )
    extrinsic = np.eye(4, dtype=np.float32)

    monkeypatch.setattr(script, "pyrender", fake_pyrender)
    monkeypatch.setattr(script, "inv_trafo", lambda value: value + 1.0)

    _FakeRenderer.instances.clear()
    shared_renderer = _FakeRenderer(16, 16)
    shared_camera = _FakeCamera(1.0, 1.0)
    color_flat = script.render_mesh(
        cast(Any, _FakeTrimesh([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
        extrinsic,
        light=False,
        renderer=cast(Any, shared_renderer),
        camera=cast(Any, shared_camera),
    )

    assert color_flat.shape == (16, 16, 3)
    assert shared_renderer.deleted is False
    flat_scene, flat_flags = shared_renderer.render_calls[0]
    assert flat_flags == (_FakeRenderFlags.FLAT | _FakeRenderFlags.OFFSCREEN)
    assert len(flat_scene.added) == 2
    assert np.allclose(flat_scene.added[1][1], extrinsic + 1.0)

    color_lit = script.render_mesh(
        cast(Any, _FakeTrimesh([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])),
        extrinsic,
        size=8,
        light=True,
    )

    assert color_lit.shape == (8, 8, 3)
    internal_renderer = _FakeRenderer.instances[-1]
    assert internal_renderer.deleted is True
    lit_scene, lit_flags = internal_renderer.render_calls[0]
    assert lit_flags == (_FakeRenderFlags.ALL_SOLID | _FakeRenderFlags.OFFSCREEN)
    assert len(lit_scene.added) == 5


def test_render_for_fid_path_views_and_output_dirs(monkeypatch: Any, tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text("mesh")
    mesh_path.with_suffix(".npy").write_text("pose")
    output_path = tmp_path / "renders" / "sample.png"

    pose = np.array(
        [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    render_calls: list[np.ndarray] = []
    normalize_modes: list[str] = []
    get_points_calls: list[int] = []

    class _TinyRenderer:
        instances: ClassVar[list[_TinyRenderer]] = []

        def __init__(self, width: int, height: int) -> None:
            self.size = (width, height)
            self.deleted = False
            self.__class__.instances.append(self)

        def delete(self) -> None:
            self.deleted = True

    def _fake_render_mesh(
        mesh: _FakeTrimesh,
        extrinsic: np.ndarray,
        size: int = 299,
        light: bool = True,
        intensity: float = 3.0,
        renderer: Any = None,
        camera: Any = None,
    ) -> np.ndarray:
        _ = mesh, light, intensity, renderer, camera
        render_calls.append(np.asarray(extrinsic, dtype=np.float32))
        return np.zeros((size, size, 3), dtype=np.uint8)

    monkeypatch.setattr(script, "Trimesh", _FakeTrimesh)
    monkeypatch.setattr(
        script,
        "pyrender",
        SimpleNamespace(
            PerspectiveCamera=lambda yfov, aspectRatio: ("camera", yfov, aspectRatio),
            OffscreenRenderer=_TinyRenderer,
        ),
    )
    monkeypatch.setattr(
        script,
        "load_mesh",
        lambda path: (
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0, 1, 1]], dtype=np.int32),
        ),
    )
    monkeypatch.setattr(script.np, "load", lambda path: pose if Path(path) == mesh_path.with_suffix(".npy") else None)
    monkeypatch.setattr(script, "normalize_mesh", lambda mesh, cube_or_sphere="sphere": normalize_modes.append(cube_or_sphere) or mesh)
    monkeypatch.setattr(
        script,
        "get_points",
        lambda num_points: get_points_calls.append(num_points)
        or np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(script, "look_at", lambda point, target: np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32))
    monkeypatch.setattr(script, "inv_trafo", lambda value: np.asarray(value, dtype=np.float32))
    monkeypatch.setattr(script, "render_mesh", _fake_render_mesh)

    _FakeTrimesh.instances.clear()
    _FakeTrimesh.invert_calls = 0
    _FakeTrimesh.add_calls = 0
    _TinyRenderer.instances.clear()

    script.render_for_fid(mesh_path, output_path, size=8, views=3, mkdirs=True, fix_winding=True)

    assert get_points_calls == [3]
    assert normalize_modes == ["sphere"]
    assert _FakeTrimesh.instances[0].transform_calls[0].shape == (4, 4)
    assert np.allclose(_FakeTrimesh.instances[0].transform_calls[0], pose)
    assert _FakeTrimesh.invert_calls == 1
    assert _FakeTrimesh.add_calls == 1
    assert len(render_calls) == 3
    assert (output_path.parent / "view_0" / "sample.png").exists()
    assert (output_path.parent / "view_1" / "sample.png").exists()
    assert (output_path.parent / "view_2" / "sample.png").exists()
    assert _TinyRenderer.instances[0].deleted is True


def test_render_for_fid_special_views_and_unknown_view(monkeypatch: Any, tmp_path: Path) -> None:
    render_calls: list[np.ndarray] = []

    class _TinyRenderer:
        def __init__(self, width: int, height: int) -> None:
            _ = width, height
            self.deleted = False

        def delete(self) -> None:
            self.deleted = True

    monkeypatch.setattr(
        script,
        "pyrender",
        SimpleNamespace(
            PerspectiveCamera=lambda yfov, aspectRatio: ("camera", yfov, aspectRatio),
            OffscreenRenderer=_TinyRenderer,
        ),
    )
    monkeypatch.setattr(script, "normalize_mesh", lambda mesh, cube_or_sphere="sphere": mesh)
    monkeypatch.setattr(script, "look_at", lambda point, target: np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32))
    monkeypatch.setattr(script, "inv_trafo", lambda value: np.asarray(value, dtype=np.float32))
    monkeypatch.setattr(
        script,
        "render_mesh",
        lambda mesh, extrinsic, size=299, light=True, intensity=3.0, renderer=None, camera=None: render_calls.append(
            np.asarray(extrinsic, dtype=np.float32)
        )
        or np.zeros((size, size, 3), dtype=np.uint8),
    )

    mesh = _FakeTrimesh([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    output_path = tmp_path / "special" / "mesh.png"

    script.render_for_fid(cast(Any, mesh), output_path, size=4, views="3dilg", mkdirs=True)

    assert len(render_calls) == 10
    assert (output_path.parent / "view_9" / "mesh.png").exists()

    with pytest.raises(ValueError, match="Unknown view type"):
        script.render_for_fid(cast(Any, mesh), output_path, views="bogus", mkdirs=True)
