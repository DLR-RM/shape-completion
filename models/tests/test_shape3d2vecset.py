import pytest
import torch

from ..src import shape3d2vecset
from ..src.shape3d2vecset import Shape3D2VecSet, Shape3D2VecSetCls

pytestmark = pytest.mark.filterwarnings("ignore:input must be a CUDA tensor, but isn't.*:UserWarning")


@pytest.fixture(autouse=True)
def patch_fps(monkeypatch: pytest.MonkeyPatch):
    def fake_fps(points: torch.Tensor, num_samples: int, **kwargs) -> torch.Tensor:
        return points[:, :num_samples].contiguous()

    monkeypatch.setattr(shape3d2vecset, "furthest_point_sample", fake_fps)


def test_shape3d2vecset_forward_smoke():
    model = Shape3D2VecSet(n_layer=2, n_embd=32, n_head=4, n_queries=32)
    inputs = torch.randn(2, 128, 3)
    points = torch.randn(2, 64, 3)

    out = model(inputs, points)

    assert "logits" in out
    assert out["logits"].shape == (2, 64)


def test_shape3d2vecset_hierarchical_queries_encode_inputs():
    model = Shape3D2VecSet(n_layer=1, n_embd=32, n_head=4, n_queries=(64, 32))
    inputs = torch.randn(2, 128, 3)

    encoded = model.encode_inputs(inputs)

    assert encoded.shape == (2, 32, 32)


def test_shape3d2vecset_cls_forward_smoke():
    model = Shape3D2VecSetCls(n_classes=7, n_layer=2, n_embd=32, n_head=4, n_queries=32)
    inputs = torch.randn(2, 128, 3)

    out = model(inputs)

    assert "cls_logits" in out
    assert out["cls_logits"].shape == (2, 7)
