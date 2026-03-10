import pytest
import torch

from ..src.dinov2 import DinoInst3D, DinoInstSeg

pytestmark = pytest.mark.filterwarnings("ignore:xFormers is not available.*:UserWarning")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dino_inst_seg_forward():
    torch.manual_seed(1337)
    model = DinoInstSeg().cuda()

    batch_size = 2
    height = 224
    width = 224
    inputs = torch.randn(batch_size, 3, height, width, device="cuda")

    out = model(inputs)
    assert isinstance(out, dict)
    assert "logits" in out
    assert out["logits"].shape[0] == batch_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dino_inst_3d_forward():
    torch.manual_seed(1337)
    model = DinoInst3D().cuda()

    batch_size = 2
    height = 224
    width = 224
    num_points = 128

    inputs = torch.randn(batch_size, 3, height, width, device="cuda")
    masks = [torch.randint(0, 2, (2, height, width), device="cuda", dtype=torch.bool) for _ in range(batch_size)]
    points = torch.randn(batch_size, num_points, 3, device="cuda")
    occ = torch.randint(0, 2, (batch_size, num_points), device="cuda", dtype=torch.bool)
    intrinsic = torch.rand(batch_size, 3, 3, device="cuda")
    extrinsic = torch.rand(batch_size, 4, 4, device="cuda")

    out = model(
        inputs=inputs,
        **{
            "inputs.masks": masks,
            "points": points,
            "points.occ": occ,
            "inputs.intrinsic": intrinsic,
            "inputs.extrinsic": extrinsic,
            "inputs.width": torch.full((batch_size, 1), width, device="cuda"),
            "inputs.height": torch.full((batch_size, 1), height, device="cuda"),
        },
    )
    assert isinstance(out, dict)
    assert "logits" in out or "occ_logits" in out
