import pytest
import torch
from torch import nn

from ..src.dinov2 import (
    Dino3D,
    DinoInst3D,
    DinoInstSeg,
    DinoInstSeg3D,
    _is_legacy_dino3d_state_dict,
    _is_legacy_dino_inst_seg_3d_state_dict,
    _remap_legacy_dino3d_state_dict,
    dino3d_legacy_config_overrides,
    dino_inst_seg_3d_legacy_config_overrides,
)
from ..src.model import Model

pytestmark = pytest.mark.filterwarnings("ignore:xFormers is not available.*:UserWarning")


def test_legacy_dino3d_state_dict_remaps_only_known_keys() -> None:
    state = {
        "nerf_enc.tcnn_encoding.params": torch.empty(0),
        "nerf_enc.mlp.weight": torch.empty(384, 63),
        "nerf_enc.mlp.bias": torch.empty(384),
        "inputs_enc.ln_1.weight": torch.empty(384),
        "inputs_enc.self_attn.to_qkv.weight": torch.empty(1152, 384),
        "inputs_enc.self_attn.to_qkv.bias": torch.empty(1152),
        "points_enc.cross_attn.to_q.weight": torch.empty(384, 384),
        "points_enc.cross_attn.to_kv.weight": torch.empty(768, 384),
        "occ_head.c_proj.weight": torch.empty(1, 1536),
    }

    assert _is_legacy_dino3d_state_dict(state)

    remapped = _remap_legacy_dino3d_state_dict(state)

    assert remapped["input_enc.ln_1.weight"] is state["inputs_enc.ln_1.weight"]
    assert remapped["input_enc.self_attn.to_q.weight"].shape == (384, 384)
    assert remapped["input_enc.self_attn.to_kv.weight"].shape == (768, 384)
    assert remapped["cross_attn.cross_attn.to_q.weight"] is state["points_enc.cross_attn.to_q.weight"]
    assert remapped["decoder.1.cross_attn.to_q.weight"] is state["points_enc.cross_attn.to_q.weight"]
    assert remapped["decoder.2.c_proj.weight"] is state["occ_head.c_proj.weight"]
    assert "points_enc.cross_attn.to_q.weight" not in remapped


def test_legacy_dino3d_checkpoint_uses_legacy_constructor_defaults() -> None:
    state = {
        "nerf_enc.mlp.weight": torch.empty(384, 63),
        "inputs_enc.ln_1.weight": torch.empty(384),
        "points_enc.cross_attn.to_q.weight": torch.empty(384, 384),
    }

    assert dino3d_legacy_config_overrides(state) == {
        "repo_or_dir": "facebookresearch/dinov2:81b2b6419385a321287de91e00282ef7cbd26f94",
        "num_queries": 512,
        "cls_token": True,
        "cat_feat": False,
        "nerf_freqs": 10,
        "init_weights": False,
        "normalize_inputs": True,
        "scale_inputs": True,
        "legacy_forward_features": True,
    }


def test_dino3d_apply_feature_accepts_concatenated_cls_token() -> None:
    class CrossAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.memory_shape: tuple[int, ...] | None = None

        def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
            self.memory_shape = tuple(memory.shape)
            return x

    model = Dino3D.__new__(Dino3D)
    nn.Module.__init__(model)
    cross_attn = CrossAttention()
    model.cross_attn = cross_attn

    x = torch.zeros(2, 4, 3)
    cls_feat = torch.ones(2, 1, 3)
    patch_feat = torch.ones(2, 5, 3)

    out = model.apply_feature(x, (cls_feat, patch_feat))

    assert out is x
    assert cross_attn.memory_shape == (2, 6, 3)


def test_legacy_dino_inst_seg_3d_checkpoint_uses_legacy_constructor_defaults() -> None:
    state = {
        "nerf_enc.mlp.weight": torch.empty(384, 39),
        "query_pos.weight": torch.empty(100, 384),
        "queries.0.weight": torch.empty(384, 384),
        "inputs_head.0.weight": torch.empty(768, 384),
        "cls_quality_head.0.weight": torch.empty(192, 384),
    }

    assert _is_legacy_dino_inst_seg_3d_state_dict(state)
    assert dino_inst_seg_3d_legacy_config_overrides(state) == {
        "repo_or_dir": "facebookresearch/dinov2:81b2b6419385a321287de91e00282ef7cbd26f94",
        "num_objs": 100,
        "num_queries": 1024,
        "mlp_heads": True,
        "multitask": "head",
        "pred_cls": "quality+objectness",
        "queries_from_feat": "detach",
        "init_weights": False,
        "nerf_freqs": 6,
    }


def test_legacy_dino_inst_seg_3d_checkpoint_disables_fuzzy_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {
        "nerf_enc.mlp.weight": torch.empty(384, 39),
        "query_pos.weight": torch.empty(100, 384),
        "queries.0.weight": torch.empty(384, 384),
        "inputs_head.0.weight": torch.empty(768, 384),
        "cls_quality_head.0.weight": torch.empty(192, 384),
    }
    captured: dict[str, object] = {}

    def fake_load_state_dict(self, state_dict, *args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(Model, "load_state_dict", fake_load_state_dict)
    model = DinoInstSeg3D.__new__(DinoInstSeg3D)

    model.load_state_dict(state)

    assert captured["fuzzy_match"] is False
    assert captured["shape_suffix_match"] is False


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
