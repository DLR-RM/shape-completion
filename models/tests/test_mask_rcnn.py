import types

import torch

from ..src import mask_rcnn as mask_rcnn_module


class _DummyBoxes:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


class _DummyInstances:
    def __init__(self, batch_idx: int, height: int, width: int) -> None:
        self.pred_boxes = _DummyBoxes(torch.tensor([[0.0, 0.0, float(width), float(height)]], dtype=torch.float32))
        self.pred_masks = torch.zeros(1, height, width, dtype=torch.bool)
        self.scores = torch.tensor([0.9], dtype=torch.float32)

    def to(self, *_args, **_kwargs) -> "_DummyInstances":
        return self


class _DummyDetector:
    def __init__(self) -> None:
        self.last_inputs = None

    def __call__(self, model_inputs):
        self.last_inputs = model_inputs
        out = []
        for i, item in enumerate(model_inputs):
            out.append({"instances": _DummyInstances(i, item["height"], item["width"])})
        return out


class _DummyMAP:
    def __init__(self, *args, **kwargs) -> None:
        self.updated = 0
        self.reset_called = False

    def update(self, preds, targets) -> None:
        assert len(preds) == len(targets)
        self.updated += 1

    def compute(self) -> dict[str, torch.Tensor]:
        return {"map": torch.tensor(0.5)}

    def reset(self) -> None:
        self.reset_called = True


class _DummyMetadataCatalog:
    @staticmethod
    def get(_name: str):
        return object()


class _Detectron2DataModule(types.ModuleType):
    MetadataCatalog: type[_DummyMetadataCatalog]


def _build_model(monkeypatch):
    detector = _DummyDetector()
    cfg = types.SimpleNamespace(DATASETS=types.SimpleNamespace(TRAIN=["dummy_train"]))

    monkeypatch.setattr(mask_rcnn_module, "load_mask_rcnn_model", lambda from_pretrained=True: (detector, cfg))
    monkeypatch.setattr(mask_rcnn_module, "MeanAveragePrecision", _DummyMAP)

    detectron2_data = _Detectron2DataModule("detectron2.data")
    detectron2_data.MetadataCatalog = _DummyMetadataCatalog
    monkeypatch.setitem(__import__("sys").modules, "detectron2", types.ModuleType("detectron2"))
    monkeypatch.setitem(__import__("sys").modules, "detectron2.data", detectron2_data)

    model = mask_rcnn_module.MaskRCNN()
    return model, detector


def test_mask_rcnn_forward_builds_detectron_inputs(monkeypatch):
    model, detector = _build_model(monkeypatch)
    inputs = torch.rand(2, 3, 8, 6)

    out = model(inputs)

    assert len(out) == 2
    assert detector.last_inputs is not None
    assert detector.last_inputs[0]["height"] == 8
    assert detector.last_inputs[0]["width"] == 6
    assert torch.equal(detector.last_inputs[0]["image"], torch.flip(inputs[0], dims=[0]))


def test_mask_rcnn_evaluate_updates_map_and_epoch_end_returns_metrics(monkeypatch):
    model, _ = _build_model(monkeypatch)
    data = {
        "inputs": torch.rand(2, 3, 8, 6),
        "inputs.boxes": torch.zeros(2, 1, 4),
        "inputs.masks": torch.zeros(2, 1, 8, 6, dtype=torch.bool),
        "inputs.labels": torch.zeros(2, 1, dtype=torch.long),
    }

    metrics = model.evaluate(data)
    epoch_metrics = model.on_validation_epoch_end()

    assert metrics == {}
    assert "map" in epoch_metrics
    assert model.map.reset_called
