from __future__ import annotations

from typing import Any, Literal, cast

import pytest
import torch
from torch import Tensor, nn

from ..src import dinov2


class TinyBackbone(nn.Module):
    """Stands in for a DINOv2 backbone on CPU.

    Serves two roles depending on caller:
    - point-token encoder: `get_intermediate_layers` returns per-layer (patch, cls).
    - RGB provider: `forward_features` returns a dict with `x_norm_patchtokens`.
    """

    def __init__(self, embed_dim: int = 8, num_heads: int = 2, n_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = 14
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.image_proj = nn.Linear(3, embed_dim)
        self.n_layers = n_layers
        self.patch_embed = None
        self.pos_embed = None
        self.mask_token = None
        self.last_forward_input: Tensor | None = None

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: list[int] | tuple[int, ...] | int,
        reshape: bool = False,
        return_class_token: bool = True,
        norm: bool = True,
    ) -> list[tuple[Tensor, Tensor]]:
        layers = n if isinstance(n, (list, tuple)) else range(self.n_layers - n, self.n_layers)
        patch = self.proj(x[:, 1:])
        cls = self.proj(x[:, 0])
        return [(patch + float(i), cls + float(i)) for i, _ in enumerate(layers)]

    def forward_features(self, x: Tensor) -> dict[str, Tensor]:
        # Pool to a 2x2 patch grid regardless of input size (matches /14 on a 28x28 image).
        self.last_forward_input = x.detach().clone()
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(2, 2))
        patch = self.image_proj(pooled.flatten(2).transpose(1, 2))
        cls = patch.mean(dim=1)
        return {
            "x_norm_clstoken": cls,
            "x_prenorm": patch,
            "x_norm_patchtokens": patch,
            "masks": torch.zeros(patch.shape[:2], dtype=torch.bool, device=x.device),
        }


class TinyFeatUpProvider(nn.Module):
    def __init__(self, repo: str, checkpoint: str):
        super().__init__()
        self.out_dim = 8
        self.proj = nn.Linear(3, self.out_dim, bias=False)

    def forward(self, image: Tensor, pixel_coords: Tensor) -> Tensor:
        pooled = torch.nn.functional.adaptive_avg_pool2d(image, output_size=(2, 2))
        fmap = self.proj(pooled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        height, width = image.shape[-2:]
        coords = pixel_coords.to(dtype=fmap.dtype)
        grid = torch.stack(
            (
                2 * coords[..., 1] / max(width - 1, 1) - 1,
                2 * coords[..., 0] / max(height - 1, 1) - 1,
            ),
            dim=-1,
        ).unsqueeze(2)
        return torch.nn.functional.grid_sample(fmap, grid, align_corners=True).squeeze(-1).transpose(1, 2)


@pytest.fixture(autouse=True)
def patch_dino_and_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dinov2, "_load_dino_backbone", lambda *args, **kwargs: TinyBackbone())

    def fake_fps(inputs: Tensor, num_samples: int, return_indices: bool = False) -> Tensor:
        if return_indices:
            return torch.arange(num_samples, device=inputs.device).unsqueeze(0).expand(len(inputs), -1)
        return inputs[:, :num_samples, :]

    monkeypatch.setattr(dinov2, "furthest_point_sample", fake_fps)


def build_batch(batch_size: int = 2, n_inputs: int = 7, n_points: int = 5) -> dict[str, Tensor]:
    torch.manual_seed(123)
    return {
        "inputs": torch.randn(batch_size, n_inputs, 3),
        "inputs.depth": torch.ones(batch_size, 28, 28),
        "inputs.colors": torch.rand(batch_size, n_inputs, 3),
        "inputs.image": torch.rand(batch_size, 3, 28, 28),
        # [row(height), col(width)] order, matching tabletop's colors[v, u] convention.
        "inputs.pixel_coords": torch.randint(0, 28, (batch_size, n_inputs, 2)),
        "inputs.intrinsic": torch.eye(3).repeat(batch_size, 1, 1),
        "inputs.extrinsic": torch.eye(4).repeat(batch_size, 1, 1),
        "inputs.width": torch.full((batch_size, 1), 28),
        "inputs.height": torch.full((batch_size, 1), 28),
        "points": torch.randn(batch_size, n_points, 3),
        "points.occ": torch.randint(0, 2, (batch_size, n_points), dtype=torch.float32),
        "points.labels": torch.randint(0, 3, (batch_size, n_points), dtype=torch.long),
    }


def make_depth_model(
    cat_feat: bool = False,
    multitask: bool | Literal["head", "dec"] | None = None,
    **kwargs: Any,
) -> dinov2.DinoInstSeg3D:
    return dinov2.DinoInstSeg3D(
        dim=3,
        num_objs=3,
        num_queries=4,
        num_query_layers=1,
        cat_feat=cat_feat,
        pred_cls="objectness",
        match_cls=False,
        mlp_heads=False,
        init_weights=False,
        nerf_enc="torch",
        multitask=multitask,
        **kwargs,
    )


def test_local_depth_descriptor_maps_detect_depth_jump() -> None:
    depth = torch.tensor([[[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]]])

    descriptors = dinov2.DinoInstSeg3D._local_depth_descriptor_maps(depth)

    assert descriptors.shape == (1, 3, 3, 3)
    assert descriptors[0, 0, 1, 1] > descriptors[0, 0, 1, 0]
    assert torch.all((descriptors[:, 2] >= 0) & (descriptors[:, 2] <= 1))


def test_disabled_input_spatial_options_match_default_model() -> None:
    torch.manual_seed(7)
    expected_model = make_depth_model().eval()
    torch.manual_seed(7)
    actual_model = make_depth_model(
        depth_descriptor=None,
        voxel_anchor_resolutions=(),
        extent_weight=0.0,
        allow_branch_warmstart=False,
    ).eval()
    batch = build_batch()

    with torch.inference_mode():
        expected = expected_model(**batch)
        actual = actual_model(**batch)

    assert expected_model.state_dict().keys() == actual_model.state_dict().keys()
    assert torch.equal(expected["logits"], actual["logits"])


def test_depth_descriptor_forward_has_finite_branch_gradient() -> None:
    model = make_depth_model(depth_descriptor="local").train()
    batch = build_batch()

    output = model(**batch)
    output["logits"].sum().backward()

    projection = cast(nn.Linear, model.depth_descriptor_mlp[-1])
    assert output["logits"].shape == (2, 3, 5)
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()


def test_voxel_anchor_residual_starts_at_zero_and_receives_gradient() -> None:
    model = make_depth_model(voxel_anchor_resolutions=(4, 8)).train()
    inputs = torch.randn(2, 7, 3)
    fps_idx = torch.arange(4).unsqueeze(0).expand(2, -1)

    residual = model._voxel_anchor_residual(inputs, fps_idx)
    residual.sum().backward()

    projection = cast(nn.Linear, model.voxel_output[-1])
    assert residual.shape == (2, 4, model.n_embd)
    assert torch.count_nonzero(residual) == 0
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()
    assert torch.count_nonzero(projection.weight.grad) > 0


def test_extent_loss_is_finite_and_differentiable() -> None:
    model = make_depth_model(extent_weight=0.25)
    points = torch.tensor(
        [
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ]
        ]
    )
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]], requires_grad=True)
    targets = torch.ones(1, 8)

    loss, center_error, extent_error = model._extent_loss(points, logits, targets, torch.tensor([0]))
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(center_error)
    assert torch.isfinite(extent_error)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_extent_loss_is_wired_into_matched_model_loss() -> None:
    model = make_depth_model(extent_weight=0.25).train()
    points = torch.tensor(
        [
            [
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [0.0, -1.0, 1.0],
                [0.0, 1.0, -1.0],
                [0.1, -0.5, -0.5],
                [0.1, 0.5, 0.5],
                [1.0, -0.5, 0.5],
                [1.0, 0.5, -0.5],
            ]
        ]
    )
    logits = torch.randn(1, 3, 8, requires_grad=True)
    cls_logits = torch.randn(1, 3, requires_grad=True)
    data = {
        "logits": logits,
        "cls_logits": cls_logits,
        "points": points,
        "points.labels": torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]]),
    }

    loss = cast(Tensor, model.loss(data, global_step=1, log_freq=1))
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(cast(Tensor, model.get_log("extent/loss")))
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_extent_diagnostics_are_reported_during_validation() -> None:
    model = make_depth_model(extent_weight=0.25).eval()
    points = torch.tensor(
        [
            [
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [0.0, -1.0, 1.0],
                [0.0, 1.0, -1.0],
                [0.1, -0.5, -0.5],
                [0.1, 0.5, 0.5],
                [1.0, -0.5, 0.5],
                [1.0, 0.5, -0.5],
            ]
        ]
    )
    data = {
        "logits": torch.randn(1, 3, 8),
        "cls_logits": torch.randn(1, 3),
        "points": points,
        "points.labels": torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]]),
    }

    model.evaluate(data)

    assert torch.isfinite(cast(Tensor, model.get_log("extent/loss")))
    assert torch.isfinite(cast(Tensor, model.get_log("extent/center_l1")))
    assert torch.isfinite(cast(Tensor, model.get_log("extent/log_half_extent_l1")))


def test_branch_warmstart_accepts_only_enabled_branch_parameters() -> None:
    source = make_depth_model()
    target = make_depth_model(
        depth_descriptor="local",
        voxel_anchor_resolutions=(4,),
        allow_branch_warmstart=True,
    )
    branch_state = {
        key: value.clone()
        for key, value in target.state_dict().items()
        if key.startswith(("depth_descriptor_mlp.", "voxel_axis_embeddings.", "voxel_local_mlp.", "voxel_output."))
    }

    result = target.load_state_dict(source.state_dict(), strict=True)

    assert result.missing_keys
    assert all(
        key.startswith(("depth_descriptor_mlp.", "voxel_axis_embeddings.", "voxel_local_mlp.", "voxel_output."))
        for key in result.missing_keys
    )
    for key, value in branch_state.items():
        assert torch.equal(target.state_dict()[key], value)


def test_branch_checkpoint_remains_strict_without_warmstart_opt_in() -> None:
    source = make_depth_model()
    target = make_depth_model(depth_descriptor="local")

    with pytest.raises(RuntimeError, match="Missing key"):
        target.load_state_dict(source.state_dict(), strict=True)


@pytest.mark.parametrize("multitask", [None, "head"])
def test_empty_gt_has_finite_negative_only_loss(multitask: Literal["head"] | None) -> None:
    model = make_depth_model(multitask=multitask).train()
    model.pred_cls = "quality+objectness"
    points_logits = torch.randn(2, 3, 5, requires_grad=True)
    cls_logits = torch.randn(2, 3, requires_grad=True)
    cls_quality = torch.randn(2, 3, requires_grad=True)
    data: dict[str, Tensor] = {
        "logits": points_logits,
        "cls_logits": cls_logits,
        "cls_quality": cls_quality,
        "points.labels": torch.zeros(2, 5, dtype=torch.long),
    }
    inputs_logits = None
    if multitask:
        inputs_logits = torch.randn(2, 3, 7, requires_grad=True)
        data["inputs.logits"] = inputs_logits
        data["inputs.labels"] = torch.zeros(2, 7, dtype=torch.long)

    loss = cast(Tensor, model.loss(data))

    assert torch.isfinite(loss)
    loss.backward()
    assert cls_logits.grad is not None
    assert torch.count_nonzero(cls_logits.grad) > 0
    assert cls_quality.grad is not None
    assert torch.count_nonzero(cls_quality.grad) == 0
    assert points_logits.grad is not None
    assert torch.count_nonzero(points_logits.grad) == 0
    if inputs_logits is not None:
        assert inputs_logits.grad is not None
        assert torch.count_nonzero(inputs_logits.grad) == 0


def test_rgbd_none_matches_depth_model_outputs_in_eval_mode() -> None:
    torch.manual_seed(1)
    depth_model = make_depth_model()
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=depth_model, rgb_fusion="none")
    batch = build_batch()

    depth_model.train(False)
    wrapper.train(False)

    with torch.inference_mode():
        expected = depth_model(**batch)
        actual = wrapper(**batch)

    assert torch.allclose(actual["logits"], expected["logits"], atol=1e-6, rtol=1e-6)


def test_oracle_dice_scores_take_best_gt_match_per_query() -> None:
    logits = torch.tensor(
        [
            [
                [8.0, 8.0, -8.0, -8.0],
                [-8.0, -8.0, 8.0, 8.0],
                [8.0, -8.0, 8.0, -8.0],
            ]
        ]
    )
    targets = [
        torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
    ]

    scores = dinov2.DinoInstSeg3D._get_oracle_dice_scores(logits, targets)

    assert scores.shape == (1, 3)
    assert scores[0, 0] > 0.99
    assert scores[0, 1] > 0.99
    assert 0.49 < scores[0, 2] < 0.51


def test_rgbd_none_matches_depth_model_outputs_cat_feat() -> None:
    # The production checkpoint uses cat_feat=True; equivalence must hold there too.
    torch.manual_seed(2)
    depth_model = make_depth_model(cat_feat=True)
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=depth_model, rgb_fusion="none")
    batch = build_batch()

    depth_model.train(False)
    wrapper.train(False)

    with torch.inference_mode():
        expected = depth_model(**batch)
        actual = wrapper(**batch)

    assert torch.allclose(actual["logits"], expected["logits"], atol=1e-6, rtol=1e-6)


def test_depth_checkpoint_loads_into_wrapper() -> None:
    # An un-prefixed depth-only checkpoint must load into wrapper.depth_model.
    source = make_depth_model()
    depth_state = source.state_dict()

    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="raw_point")
    result = wrapper.load_state_dict(depth_state, strict=True)
    # routed to depth_model.load_state_dict, so it must fully match the depth params
    assert not getattr(result, "missing_keys", [])
    assert not getattr(result, "unexpected_keys", [])

    loaded = wrapper.depth_model.state_dict()
    for key, value in depth_state.items():
        assert torch.equal(loaded[key], value)


def test_lightning_prefixed_depth_checkpoint_loads_into_wrapper() -> None:
    source = make_depth_model()
    depth_state = {f"model.{key}": value for key, value in source.state_dict().items()}

    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="raw_point")
    result = wrapper.load_state_dict(depth_state, strict=True)

    assert not getattr(result, "missing_keys", [])
    assert not getattr(result, "unexpected_keys", [])
    loaded = wrapper.depth_model.state_dict()
    for key, value in source.state_dict().items():
        assert torch.equal(loaded[key], value)


def test_lightning_prefixed_rgbd_checkpoint_loads_into_wrapper() -> None:
    source = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="raw_point")
    rgbd_state = {f"model.{key}": value for key, value in source.state_dict().items()}

    target = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="raw_point")
    result = target.load_state_dict(rgbd_state, strict=True)

    assert not getattr(result, "missing_keys", [])
    assert not getattr(result, "unexpected_keys", [])
    loaded = target.state_dict()
    for key, value in source.state_dict().items():
        assert torch.equal(loaded[key], value)


@pytest.mark.parametrize("container_key", ["model", "state_dict"])
def test_container_wrapped_rgbd_checkpoint_loads_into_wrapper(container_key: str) -> None:
    source = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="raw_point")
    prefix = "model." if container_key == "state_dict" else ""
    wrapped_state = {container_key: {f"{prefix}{key}": value for key, value in source.state_dict().items()}}

    target = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="raw_point")
    result = target.load_state_dict(wrapped_state, strict=True)

    assert not getattr(result, "missing_keys", [])
    assert not getattr(result, "unexpected_keys", [])
    loaded = target.state_dict()
    for key, value in source.state_dict().items():
        assert torch.equal(loaded[key], value)


def test_container_wrapped_depth_checkpoint_still_routes_into_depth_model() -> None:
    source = make_depth_model()
    wrapped_state = {"model": source.state_dict()}

    target = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="raw_point")
    result = target.load_state_dict(wrapped_state, strict=True)

    assert not getattr(result, "missing_keys", [])
    assert not getattr(result, "unexpected_keys", [])
    for key, value in source.state_dict().items():
        assert torch.equal(target.depth_model.state_dict()[key], value)


def test_raw_point_requires_inputs_colors() -> None:
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="raw_point")
    batch = build_batch()
    batch.pop("inputs.colors")

    with pytest.raises(KeyError, match=r"inputs\.colors"):
        wrapper(**batch)


def test_raw_point_forward_uses_full_input_side_features() -> None:
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="raw_point")
    batch = build_batch(n_inputs=9)

    out = wrapper(**batch)

    assert out["logits"].shape == (2, 3, 5)
    # K/V-side fusion: rgb aligns to full input cloud, not the FPS subset.
    assert wrapper.last_rgb_feature_shape == (2, 9, 8)


def test_raw_decode_changes_only_input_segmentation_features() -> None:
    torch.manual_seed(3)
    depth_model = make_depth_model(multitask="head")
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=depth_model,
        rgb_fusion="raw_decode",
        fusion_gate_init_bias=-2.0,
    ).eval()
    batch = build_batch(n_inputs=9)

    with torch.inference_mode():
        expected = depth_model(**batch)
        actual = wrapper(**batch)

    assert torch.equal(actual["logits"], expected["logits"])
    assert not torch.equal(actual["inputs.logits"], expected["inputs.logits"])
    assert wrapper.last_rgb_feature_shape == (2, 9, 3)


def test_raw_decode_tracks_training_input_subsampling() -> None:
    depth_model = make_depth_model(multitask="head")
    depth_model.sample = (0, 2)
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=depth_model,
        rgb_fusion="raw_decode",
        fusion_gate_init_bias=-2.0,
    ).train()
    batch = build_batch(n_inputs=9)

    actual = wrapper(**batch)

    assert actual["inputs.logits"].shape[-1] == 6
    assert actual["inputs.index"].shape == (2, 6)


def test_featup_decode_starts_at_depth_identity_and_passes_projection_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dinov2, "FeatUpPointFeatureProvider", TinyFeatUpProvider)
    torch.manual_seed(5)
    depth_model = make_depth_model(multitask="head")
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=depth_model,
        rgb_fusion="featup_decode",
        fusion_mode="zero_add",
        train_mode="adapter",
        featup_repo="unused",
        featup_checkpoint="unused",
    ).train()
    batch = build_batch(n_inputs=9)

    with torch.no_grad():
        expected = depth_model(**batch)
    actual = wrapper(**batch)

    assert torch.equal(actual["logits"], expected["logits"])
    assert torch.equal(actual["inputs.logits"], expected["inputs.logits"])
    assert wrapper.last_rgb_feature_shape == (2, 9, 8)
    assert wrapper.rgb_provider is not None
    assert not wrapper.rgb_provider.training
    assert not any(param.requires_grad for param in wrapper.rgb_provider.parameters())

    actual["inputs.logits"].square().sum().backward()
    assert wrapper.point_fusion is not None
    assert isinstance(wrapper.point_fusion.rgb_proj, nn.Linear)
    assert wrapper.point_fusion.rgb_proj.weight.grad is not None
    assert torch.count_nonzero(wrapper.point_fusion.rgb_proj.weight.grad) > 0


def test_rgb_metric_adapter_normalizes_embeddings_and_passes_gradients() -> None:
    adapter = dinov2.RGBMetricAdapter(input_dim=8, hidden_dim=6, output_dim=4)
    features = torch.randn(2, 5, 8, requires_grad=True)

    embeddings = adapter(features)

    assert embeddings.shape == (2, 5, 4)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(2, 5), atol=1e-6)
    embeddings[..., 0].sum().backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_norm_bounded_point_fusion_is_zero_identity_with_hard_cap(dtype: torch.dtype) -> None:
    fusion = dinov2.NormBoundedPointFusion(dim=8, rgb_dim=4, max_ratio=0.1).to(dtype=dtype)
    depth = torch.randn(2, 5, 8, dtype=dtype)
    rgb = torch.randn(2, 5, 4, dtype=dtype)

    identity = fusion(depth, rgb)
    assert torch.equal(identity, depth)

    identity.sum().backward()
    assert fusion.rgb_proj.weight.grad is not None
    assert torch.count_nonzero(fusion.rgb_proj.weight.grad) > 0

    with torch.no_grad():
        fusion.rgb_proj.weight.fill_(100.0)
    fused = fusion(depth, rgb)
    ratio = (fused - depth).float().norm(dim=-1) / depth.float().norm(dim=-1)
    assert float(ratio.max()) <= 0.1001
    assert fusion.last_ratio_stats is not None
    assert fusion.last_ratio_stats["max"] <= 0.1001


def test_norm_bounded_query_fusion_is_zero_identity_and_passes_gradients() -> None:
    fusion = dinov2.NormBoundedQueryFusion(dim=8, rgb_dim=4, num_heads=2, max_ratio=0.1)
    queries = torch.randn(2, 3, 8)
    rgb = torch.randn(2, 5, 4)

    identity = fusion(queries, rgb)

    assert torch.equal(identity, queries)
    identity.sum().backward()
    assert fusion.cross_attn.out_proj.weight.grad is not None
    assert torch.count_nonzero(fusion.cross_attn.out_proj.weight.grad) > 0


def test_fusion_keep_masks_are_mutually_exclusive() -> None:
    rgb_keep, depth_keep = dinov2._fusion_keep_masks(
        torch.tensor([0.05, 0.20, 0.50]),
        rgb_residual_dropout=0.15,
        depth_fusion_dropout=0.15,
    )

    assert torch.equal(rgb_keep, torch.tensor([False, True, True]))
    assert torch.equal(depth_keep, torch.tensor([True, False, True]))
    assert torch.all(rgb_keep | depth_keep)


def test_rgb_feature_token_dropout_masks_whole_tokens_without_rescaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random_values = torch.tensor([[[0.05], [0.20], [0.90]], [[0.10], [0.80], [0.14]]])
    monkeypatch.setattr(dinov2.torch, "rand", lambda *args, **kwargs: random_values)
    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    dropped, keep_mask = dinov2._drop_rgb_feature_tokens(features, 0.15)

    expected_mask = random_values >= 0.15
    assert torch.equal(keep_mask, expected_mask)
    assert torch.equal(dropped[expected_mask.squeeze(-1)], features[expected_mask.squeeze(-1)])
    assert torch.count_nonzero(dropped[~expected_mask.squeeze(-1)]) == 0

    features.requires_grad_()
    dropped, keep_mask = dinov2._drop_rgb_feature_tokens(
        features,
        0.15,
        force_keep_samples=torch.tensor([True, False]),
    )
    dropped.sum().backward()
    assert torch.all(keep_mask[0])
    assert features.grad is not None
    assert torch.equal(features.grad, keep_mask.expand_as(features).to(dtype=features.dtype))


def test_wrapper_shares_token_dropped_features_between_deliveries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dinov2, "FeatUpPointFeatureProvider", TinyFeatUpProvider)
    dropped_calls = 0

    def deterministic_dropout(
        features: Tensor,
        probability: float,
        force_keep_samples: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        nonlocal dropped_calls
        dropped_calls += 1
        assert probability == 0.15
        assert force_keep_samples is None
        keep_mask = torch.ones((*features.shape[:2], 1), dtype=torch.bool, device=features.device)
        keep_mask[:, 1::2] = False
        return features * keep_mask, keep_mask

    monkeypatch.setattr(dinov2, "_drop_rgb_feature_tokens", deterministic_dropout)
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=make_depth_model(multitask="head"),
        rgb_fusion="featup_dual",
        featup_repo=".",
        featup_checkpoint="test-checkpoint",
        rgb_token_dropout=0.15,
        query_rgb_tokens=4,
    )
    assert isinstance(wrapper.point_fusion, dinov2.NormBoundedPointFusion)
    assert wrapper.query_fusion is not None
    point_rgb: list[Tensor] = []
    query_rgb: list[Tensor] = []
    point_hook = wrapper.point_fusion.register_forward_pre_hook(
        lambda module, args: point_rgb.append(args[1].detach().clone())
    )
    query_hook = wrapper.query_fusion.register_forward_pre_hook(
        lambda module, args: query_rgb.append(args[1].detach().clone())
    )
    batch = build_batch(n_inputs=9)

    wrapper.train()(**batch)

    assert dropped_calls == 1
    assert torch.count_nonzero(point_rgb[0][:, 1::2]) == 0
    assert torch.equal(query_rgb[0], point_rgb[0][:, :4])

    wrapper.eval()(**batch)
    assert dropped_calls == 1
    point_hook.remove()
    query_hook.remove()


def test_token_dropout_forces_rgb_present_when_depth_fusion_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dinov2, "FeatUpPointFeatureProvider", TinyFeatUpProvider)
    rgb_keep = torch.tensor([True, True])
    depth_keep = torch.tensor([False, True])
    monkeypatch.setattr(dinov2, "_fusion_keep_masks", lambda *args, **kwargs: (rgb_keep, depth_keep))
    observed_force_keep: list[Tensor] = []

    def capture_dropout(
        features: Tensor,
        probability: float,
        force_keep_samples: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        assert probability == 0.15
        assert force_keep_samples is not None
        observed_force_keep.append(force_keep_samples.detach().clone())
        keep_mask = force_keep_samples.view(-1, 1, 1).expand(*features.shape[:2], 1).clone()
        return features * keep_mask, keep_mask

    monkeypatch.setattr(dinov2, "_drop_rgb_feature_tokens", capture_dropout)
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=make_depth_model(multitask="head"),
        rgb_fusion="featup_dual",
        featup_repo=".",
        featup_checkpoint="test-checkpoint",
        rgb_residual_dropout=0.15,
        depth_fusion_dropout=0.15,
        rgb_token_dropout=0.15,
        query_rgb_tokens=4,
    )

    wrapper.train()(**build_batch(batch_size=2, n_inputs=9))

    assert len(observed_force_keep) == 1
    assert torch.equal(observed_force_keep[0], torch.tensor([True, False]))


def test_depth_fusion_mask_preserves_bounded_rgb_residual() -> None:
    fusion = dinov2.NormBoundedPointFusion(dim=8, rgb_dim=4, max_ratio=0.1)
    with torch.no_grad():
        fusion.rgb_proj.weight.fill_(1.0)
        fusion.rgb_proj.bias.fill_(1.0)
    depth = torch.randn(2, 5, 8)
    rgb = torch.randn(2, 5, 4)
    fusion.depth_sample_mask = torch.tensor([False, True])

    fused = fusion(depth, rgb)

    dropped_ratio = fused[0].float().norm(dim=-1) / depth[0].float().norm(dim=-1)
    kept_residual_ratio = (fused[1] - depth[1]).float().norm(dim=-1) / depth[1].float().norm(dim=-1)
    assert float(dropped_ratio.max()) <= 0.1001
    assert torch.count_nonzero(fused[0]) > 0
    assert float(kept_residual_ratio.max()) <= 0.1001


def test_query_depth_fusion_mask_preserves_bounded_rgb_residual() -> None:
    fusion = dinov2.NormBoundedQueryFusion(dim=8, rgb_dim=4, num_heads=2, max_ratio=0.1)
    with torch.no_grad():
        fusion.cross_attn.out_proj.weight.fill_(1.0)
        fusion.cross_attn.out_proj.bias.fill_(1.0)
    queries = torch.randn(2, 3, 8)
    rgb = torch.randn(2, 5, 4)
    fusion.depth_sample_mask = torch.tensor([False, True])

    fused = fusion(queries, rgb)

    dropped_ratio = fused[0].float().norm(dim=-1) / queries[0].float().norm(dim=-1)
    kept_residual_ratio = (fused[1] - queries[1]).float().norm(dim=-1) / queries[1].float().norm(dim=-1)
    assert float(dropped_ratio.max()) <= 0.1001
    assert torch.count_nonzero(fused[0]) > 0
    assert float(kept_residual_ratio.max()) <= 0.1001


def test_wrapper_shares_dropout_masks_and_disables_them_in_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dinov2, "FeatUpPointFeatureProvider", TinyFeatUpProvider)
    rgb_keep = torch.tensor([False, True])
    depth_keep = torch.tensor([True, False])
    monkeypatch.setattr(dinov2, "_fusion_keep_masks", lambda *args, **kwargs: (rgb_keep, depth_keep))
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=make_depth_model(multitask="head"),
        rgb_fusion="featup_dual",
        featup_repo=".",
        featup_checkpoint="test-checkpoint",
        rgb_residual_dropout=0.15,
        depth_fusion_dropout=0.15,
        query_rgb_tokens=4,
    )
    batch = build_batch(batch_size=2, n_inputs=9)

    wrapper.train()(**batch)

    assert isinstance(wrapper.point_fusion, dinov2.NormBoundedPointFusion)
    assert wrapper.query_fusion is not None
    assert wrapper.point_fusion.sample_mask is wrapper.query_fusion.sample_mask
    assert wrapper.point_fusion.depth_sample_mask is wrapper.query_fusion.depth_sample_mask
    assert wrapper.point_fusion.sample_mask is not None
    assert wrapper.point_fusion.depth_sample_mask is not None
    assert torch.equal(wrapper.point_fusion.sample_mask, rgb_keep)
    assert torch.equal(wrapper.point_fusion.depth_sample_mask, depth_keep)

    wrapper.eval()(**batch)
    assert wrapper.point_fusion.sample_mask is None
    assert wrapper.point_fusion.depth_sample_mask is None
    assert wrapper.query_fusion.sample_mask is None
    assert wrapper.query_fusion.depth_sample_mask is None


def test_fusion_dropout_probabilities_must_leave_joint_samples() -> None:
    with pytest.raises(ValueError, match="sum to less than 1"):
        dinov2.DinoInstSegRGBD3D(
            depth_model=make_depth_model(),
            rgb_fusion="featup_dual",
            rgb_residual_dropout=0.5,
            depth_fusion_dropout=0.5,
        )

    with pytest.raises(ValueError, match="rgb_token_dropout"):
        dinov2.DinoInstSegRGBD3D(
            depth_model=make_depth_model(),
            rgb_fusion="featup_dual",
            rgb_token_dropout=1.0,
        )

    with pytest.raises(ValueError, match="only with featup_dual"):
        dinov2.DinoInstSegRGBD3D(
            depth_model=make_depth_model(),
            rgb_fusion="raw_dual",
            rgb_token_dropout=0.15,
        )


def test_featup_dual_starts_at_depth_identity_and_selectively_unfreezes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setattr(dinov2, "FeatUpPointFeatureProvider", TinyFeatUpProvider)
    adapter = dinov2.RGBMetricAdapter(input_dim=8)
    checkpoint = tmp_path / "metric.pt"
    torch.save(
        {"model_state_dict": {f"adapter.{key}": value for key, value in adapter.state_dict().items()}},
        checkpoint,
    )
    torch.manual_seed(7)
    depth_model = make_depth_model(multitask="head")
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=depth_model,
        rgb_fusion="featup_dual",
        train_mode="task_aligned",
        featup_repo="unused",
        featup_checkpoint="unused",
        rgb_metric_checkpoint=str(checkpoint),
        fusion_max_ratio=0.1,
        query_rgb_tokens=4,
    ).eval()
    batch = build_batch(n_inputs=9)

    with torch.inference_mode():
        expected = depth_model(**batch)
        actual = wrapper(**batch)

    assert torch.equal(actual["logits"], expected["logits"])
    assert torch.equal(actual["inputs.logits"], expected["inputs.logits"])
    assert not any(parameter.requires_grad for parameter in depth_model.inputs_enc.parameters())
    assert not any(parameter.requires_grad for parameter in depth_model.points_head.parameters())
    assert any(parameter.requires_grad for parameter in depth_model.query_enc.parameters())
    assert any(parameter.requires_grad for parameter in depth_model.inputs_head.parameters())
    assert any(parameter.requires_grad for parameter in depth_model.cls_head.parameters())

    optimizer = wrapper.optimizer(lr=1e-3, factor=0.1)
    assert sorted(group["lr"] for group in optimizer.param_groups) == [1e-4, 1e-3]
    depth_parameter_ids = {id(parameter) for parameter in depth_model.parameters()}
    for group in optimizer.param_groups:
        is_depth_group = all(id(parameter) in depth_parameter_ids for parameter in group["params"])
        assert is_depth_group == (group["lr"] == 1e-4)

    control = dinov2.DinoInstSegRGBD3D(
        depth_model=make_depth_model(multitask="head"),
        rgb_fusion="none",
        train_mode="task_aligned",
    )
    control_optimizer = control.optimizer(lr=1e-3, factor=0.1)
    assert [group["lr"] for group in control_optimizer.param_groups] == [1e-4]

    actual = wrapper.train()(**batch)
    actual["inputs.logits"].square().sum().backward()
    assert isinstance(wrapper.point_fusion, dinov2.NormBoundedPointFusion)
    assert wrapper.point_fusion.rgb_proj.weight.grad is not None
    assert torch.count_nonzero(wrapper.point_fusion.rgb_proj.weight.grad) > 0
    assert wrapper.query_fusion is not None
    query_gradient = wrapper.query_fusion.cross_attn.out_proj.weight.grad
    assert query_gradient is not None
    assert torch.count_nonzero(query_gradient) > 0


@pytest.mark.parametrize(
    ("delivery", "has_point", "has_query"),
    [("point", True, False), ("query", False, True)],
)
def test_featup_single_delivery_supports_fresh_adapter_and_zero_identity(
    monkeypatch: pytest.MonkeyPatch,
    delivery: Literal["point", "query"],
    has_point: bool,
    has_query: bool,
) -> None:
    monkeypatch.setattr(dinov2, "FeatUpPointFeatureProvider", TinyFeatUpProvider)
    depth_model = make_depth_model(multitask="head").eval()
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=depth_model,
        rgb_fusion="featup_dual",
        train_mode="adapter",
        featup_repo="unused",
        featup_checkpoint="unused",
        rgb_metric_checkpoint=None,
        fusion_max_ratio=0.1,
        query_rgb_tokens=4,
        rgb_delivery=delivery,
    ).eval()
    batch = build_batch(n_inputs=9)

    with torch.inference_mode():
        expected = depth_model(**batch)
        actual = wrapper(**batch)

    assert torch.equal(actual["logits"], expected["logits"])
    assert torch.equal(actual["inputs.logits"], expected["inputs.logits"])
    assert (wrapper.point_fusion is not None) is has_point
    assert (wrapper.query_fusion is not None) is has_query


def test_raw_dual_starts_at_depth_identity() -> None:
    depth_model = make_depth_model(multitask="head").eval()
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=depth_model,
        rgb_fusion="raw_dual",
        train_mode="adapter",
        fusion_max_ratio=0.1,
        query_rgb_tokens=4,
    ).eval()
    batch = build_batch(n_inputs=9)

    with torch.inference_mode():
        expected = depth_model(**batch)
        actual = wrapper(**batch)

    assert torch.equal(actual["logits"], expected["logits"])
    assert torch.equal(actual["inputs.logits"], expected["inputs.logits"])
    assert wrapper.metric_adapter is not None
    assert wrapper.metric_adapter.input_dim == 3


def test_dino_token_cat_rejects_multi_point_decoder() -> None:
    depth_model = make_depth_model()
    depth_model.points_dec = nn.ModuleList([depth_model.points_dec])

    with pytest.raises(ValueError, match="single decoder"):
        dinov2.DinoInstSegRGBD3D(depth_model=depth_model, rgb_fusion="dino_token_cat")


def test_adapter_mode_freezes_depth_model_parameters() -> None:
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=make_depth_model(),
        rgb_fusion="raw_point",
        train_mode="adapter",
    )

    assert not any(param.requires_grad for param in wrapper.depth_model.parameters())
    assert any(param.requires_grad for name, param in wrapper.named_parameters() if not name.startswith("depth_model."))


def test_train_mode_keeps_frozen_modules_in_eval() -> None:
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=make_depth_model(),
        rgb_fusion="dino_point",
        train_mode="adapter",
        freeze_rgb_encoder=True,
    )
    wrapper.train()

    assert not wrapper.depth_model.training
    assert isinstance(wrapper.rgb_provider, dinov2.DinoPointFeatureProvider)
    rgb_encoder = cast(nn.Module, wrapper.rgb_provider.encoder)
    assert not rgb_encoder.training
    assert wrapper.point_fusion is not None and wrapper.point_fusion.training


def test_gate_init_bias_is_configurable_and_logged() -> None:
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=make_depth_model(),
        rgb_fusion="raw_point",
        fusion_gate_init_bias=-2.0,
    )
    batch = build_batch()

    wrapper(**batch)

    assert isinstance(wrapper.point_fusion, dinov2.RGBPointFusion)
    gate = cast(nn.Sequential, wrapper.point_fusion.gate)
    gate_bias = cast(nn.Linear, gate[0]).bias
    assert gate_bias is not None
    assert torch.allclose(gate_bias, torch.full_like(gate_bias, -2.0))
    logs = cast(dict[str, tuple[Any, int]], wrapper.get_log())
    assert 0.05 < logs["rgb_gate/mean"][0] < 0.25
    assert logs["rgb_gate/max"][0] > logs["rgb_gate/min"][0]


def test_default_gate_init_preserves_legacy_closed_gate() -> None:
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="raw_point")
    batch = build_batch()

    wrapper(**batch)

    logs = cast(dict[str, tuple[Any, int]], wrapper.get_log())
    assert logs["rgb_gate/mean"][0] < 1e-3


def test_zero_add_starts_at_depth_identity_and_passes_projection_gradients() -> None:
    fusion = dinov2.RGBPointFusion(dim=8, rgb_dim=3, mode="zero_add")
    depth = torch.randn(2, 7, 8)
    rgb = torch.randn(2, 7, 3)

    output = fusion(depth, rgb)
    output.square().sum().backward()

    assert torch.equal(output, depth)
    assert isinstance(fusion.rgb_proj, nn.Linear)
    assert fusion.rgb_proj.weight.grad is not None
    assert torch.count_nonzero(fusion.rgb_proj.weight.grad) > 0


def test_dino_point_forward_samples_features_for_each_input_point() -> None:
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="dino_point")
    batch = build_batch(n_inputs=6)

    out = wrapper(**batch)

    assert out["logits"].shape == (2, 3, 5)
    # one sampled feature per input point, fused on the K/V side
    assert wrapper.last_rgb_feature_shape == (2, 6, 8)


def test_dino_provider_does_not_double_normalize_preprocessed_image() -> None:
    provider = dinov2.DinoPointFeatureProvider()
    image = torch.randn(1, 3, 28, 28)
    pixel_coords = torch.tensor([[[0, 0], [14, 14]]], dtype=torch.long)

    provider(image, pixel_coords)

    assert isinstance(provider.encoder, TinyBackbone)
    assert provider.encoder.last_forward_input is not None
    assert torch.equal(provider.encoder.last_forward_input, image)


def test_dino_provider_keeps_explicit_imagenet_image_even_when_range_is_ambiguous() -> None:
    provider = dinov2.DinoPointFeatureProvider(image_normalization="imagenet")
    image = torch.full((1, 3, 28, 28), 0.25)
    pixel_coords = torch.tensor([[[0, 0], [14, 14]]], dtype=torch.long)

    provider(image, pixel_coords)

    assert isinstance(provider.encoder, TinyBackbone)
    assert provider.encoder.last_forward_input is not None
    assert torch.equal(provider.encoder.last_forward_input, image)


def test_dino_provider_normalizes_unit_range_image() -> None:
    provider = dinov2.DinoPointFeatureProvider()
    image = torch.rand(1, 3, 28, 28)
    pixel_coords = torch.tensor([[[0, 0], [14, 14]]], dtype=torch.long)

    provider(image, pixel_coords)

    assert isinstance(provider.encoder, TinyBackbone)
    assert provider.encoder.last_forward_input is not None
    assert not torch.equal(provider.encoder.last_forward_input, image)


def test_dino_provider_normalizes_uint8_rgb_image() -> None:
    provider = dinov2.DinoPointFeatureProvider()
    image = torch.randint(0, 256, (1, 3, 28, 28), dtype=torch.uint8)
    pixel_coords = torch.tensor([[[0, 0], [14, 14]]], dtype=torch.long)

    provider(image, pixel_coords)

    assert isinstance(provider.encoder, TinyBackbone)
    assert provider.encoder.last_forward_input is not None
    assert torch.is_floating_point(provider.encoder.last_forward_input)
    assert float(provider.encoder.last_forward_input.max()) < 3.0


def test_adapter_keeps_dino_encoder_frozen() -> None:
    # In adapter mode the pretrained DINO encoder must stay frozen (requires_grad=False),
    # while fusion/projection params remain trainable.
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=make_depth_model(),
        rgb_fusion="dino_point",
        train_mode="adapter",
        freeze_rgb_encoder=True,
    )

    assert isinstance(wrapper.rgb_provider, dinov2.DinoPointFeatureProvider)
    rgb_encoder = cast(nn.Module, wrapper.rgb_provider.encoder)
    assert not any(param.requires_grad for param in rgb_encoder.parameters())
    assert wrapper.point_fusion is not None
    assert any(param.requires_grad for param in wrapper.point_fusion.parameters())


def test_dino_token_cat_forward_appends_rgb_memory_tokens() -> None:
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(), rgb_fusion="dino_token_cat")
    batch = build_batch(n_inputs=4)

    out = wrapper(**batch)

    assert out["logits"].shape == (2, 3, 5)
    # 2x2 patch grid on a 28x28 image => 4 RGB memory tokens, projected to n_embd=8.
    assert wrapper.last_rgb_feature_shape == (2, 4, 8)


def test_dino_token_cat_forward_with_cat_feat() -> None:
    # Single decoder + cat_feat=True is the production layout; appended RGB tokens
    # must reach the decoder (else branch attends to the full memory).
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=make_depth_model(cat_feat=True), rgb_fusion="dino_token_cat")
    batch = build_batch(n_inputs=4)

    out = wrapper(**batch)

    assert out["logits"].shape == (2, 3, 5)
    assert wrapper.last_rgb_feature_shape == (2, 4, 8)


def test_factory_builds_rgbd3d_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    from omegaconf import OmegaConf

    from .. import get_model

    monkeypatch.setattr(dinov2, "FeatUpPointFeatureProvider", TinyFeatUpProvider)
    cfg = OmegaConf.create(
        {
            "model": {
                "arch": "dino_inst_rgbd3d",
                "bias": True,
                "dropout": 0.0,
                "weights": None,
                # Post-construction keys required by get_model's shared epilogue:
                "attn_mode": None,
                "attn_backend": None,
                "reduction": None,
                "checkpoint": None,
            },
            "inputs": {"dim": 3, "type": "rgbd", "project": True, "nerf": False},
            "points": {"nerf": False},
            # skip=False + resume=True → load_weights=False; avoids checkpoint resolution.
            "train": {"loss": None, "skip": False, "resume": True},
            "cls": {"num_classes": None},
            "norm": {"padding": 0.0},
            "vis": {"refinement_steps": 0},
            "implicit": {"dvr": False},
            "rgb_fusion": "featup_dual",
            "featup_repo": ".",
            "featup_checkpoint": "test-checkpoint",
            "fusion_mode": "gate",
            "fusion_gate_init_bias": -2.0,
            "rgbd_train_mode": "finetune",
            "rgb_image_normalization": "rgb",
            "depth_fusion_dropout": 0.15,
            "rgb_token_dropout": 0.20,
            "n_objs": 3,
            "n_queries": 4,
            "l_enc": 1,
            "l_query": 1,
            "cat_feat": False,
            "pred_cls": None,
            "match_cls": False,
            "nerf_enc": "torch",
        }
    )

    model = get_model(cfg)

    assert isinstance(model, dinov2.DinoInstSegRGBD3D)
    assert model.rgb_image_normalization == "rgb"
    assert model.fusion_gate_init_bias == -2.0
    assert model.depth_fusion_dropout == 0.15
    assert model.rgb_token_dropout == 0.20


def test_factory_propagates_input_spatial_branch_controls() -> None:
    from omegaconf import OmegaConf

    from .. import get_model

    cfg = OmegaConf.create(
        {
            "model": {
                "arch": "dino_inst_mask",
                "bias": True,
                "dropout": 0.0,
                "weights": None,
                "attn_mode": None,
                "attn_backend": None,
                "reduction": None,
                "checkpoint": None,
            },
            "inputs": {"dim": 3, "type": "depth", "project": True, "nerf": False},
            "points": {"nerf": False},
            "train": {"loss": None, "skip": False, "resume": True},
            "cls": {"num_classes": None},
            "norm": {"padding": 0.0},
            "vis": {"refinement_steps": 0},
            "implicit": {"dvr": False},
            "n_objs": 3,
            "n_queries": 4,
            "l_enc": 1,
            "l_query": 1,
            "cat_feat": False,
            "pred_cls": None,
            "match_cls": False,
            "nerf_enc": "torch",
            "depth_descriptor": "local",
            "voxel_anchor_resolutions": [4, 8],
            "extent_weight": 0.25,
            "extent_quantile_temperature": 0.03,
            "allow_branch_warmstart": True,
        }
    )

    model = get_model(cfg)

    assert isinstance(model, dinov2.DinoInstSeg3D)
    assert model.depth_descriptor == "local"
    assert model.voxel_anchor_resolutions == (4, 8)
    assert model.extent_weight == pytest.approx(0.25)
    assert model.extent_quantile_temperature == pytest.approx(0.03)
    assert model.allow_branch_warmstart is True


def test_factory_rgbd3d_arch_wins_over_kinect_input_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from omegaconf import OmegaConf

    from .. import get_model

    cfg = OmegaConf.create(
        {
            "model": {
                "arch": "dino_inst_rgbd3d",
                "bias": True,
                "dropout": 0.0,
                "weights": None,
                "attn_mode": None,
                "attn_backend": None,
                "reduction": None,
                "checkpoint": None,
            },
            "inputs": {"dim": 3, "type": "kinect", "project": True, "nerf": False},
            "points": {"nerf": False},
            "train": {"loss": None, "skip": False, "resume": True},
            "cls": {"num_classes": None},
            "norm": {"padding": 0.0},
            "vis": {"refinement_steps": 0},
            "implicit": {"dvr": False},
            "rgb_fusion": "none",
            "fusion_mode": "gate",
            "rgbd_train_mode": "adapter",
            "n_objs": 3,
            "n_queries": 4,
            "l_enc": 1,
            "l_query": 1,
            "cat_feat": False,
            "pred_cls": None,
            "match_cls": False,
            "nerf_enc": "torch",
        }
    )

    model = get_model(cfg)

    assert isinstance(model, dinov2.DinoInstSegRGBD3D)
    assert not any(param.requires_grad for param in model.depth_model.parameters())


def test_finetune_optimizer_uses_lower_lr_for_depth_params() -> None:
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=make_depth_model(),
        rgb_fusion="raw_point",
        train_mode="finetune",
    )

    optimizer = wrapper.optimizer(lr=1e-4, weight_decay=0.01, factor=0.1, foreach=True)

    assert optimizer is not None
    lrs = sorted({group["lr"] for group in optimizer.param_groups})
    assert lrs == [1e-5, 1e-4]


def test_adapter_optimizer_has_single_group() -> None:
    wrapper = dinov2.DinoInstSegRGBD3D(
        depth_model=make_depth_model(),
        rgb_fusion="raw_point",
        train_mode="adapter",
    )

    optimizer = wrapper.optimizer(lr=1e-4, weight_decay=0.01, factor=0.1, foreach=True)

    lrs = {group["lr"] for group in optimizer.param_groups}
    assert lrs == {1e-4}


def test_predict_with_data_containing_inputs_does_not_collide() -> None:
    # Regression: predict(**batch, data=batch) must not raise TypeError from a
    # duplicate `inputs` kwarg when encoding (generator.py:264 call pattern).
    depth_model = make_depth_model()
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=depth_model, rgb_fusion="raw_point")
    batch = build_batch()
    wrapper.train(False)
    with torch.inference_mode():
        out = wrapper.predict(**batch, data=batch)
    assert out is not None


def test_predict_accepts_precomputed_feature() -> None:
    # Regression: predict(points=..., feature=...) must pass the feature through
    # without re-encoding or raising (generator.py:524/648/898 call pattern).
    depth_model = make_depth_model()
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=depth_model, rgb_fusion="raw_point")
    batch = build_batch()
    wrapper.train(False)
    with torch.inference_mode():
        # encode() takes inputs as explicit arg; strip it from kwargs to avoid collision.
        enc_kwargs = {k: v for k, v in batch.items() if k != "inputs"}
        feature = wrapper.encode(inputs=batch["inputs"], **enc_kwargs)
        out = wrapper.predict(points=batch["points"], feature=feature, data=batch)
    assert out is not None


def test_predict_none_matches_depth_model() -> None:
    # Drop-in: with rgb_fusion="none" the wrapper's predict must equal the bare model's.
    torch.manual_seed(7)
    depth_model = make_depth_model()
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=depth_model, rgb_fusion="none")
    batch = build_batch()
    depth_model.train(False)
    wrapper.train(False)
    with torch.inference_mode():
        expected = cast(Any, depth_model).predict(**batch, data=batch)
        actual = cast(Any, wrapper).predict(**batch, data=batch)
    exp = expected[0] if isinstance(expected, tuple) else expected
    act = actual[0] if isinstance(actual, tuple) else actual
    assert len(act) == len(exp)
    for a, e in zip(act, exp, strict=True):
        assert torch.equal(a, e)


def test_wrapper_forwards_depth_model_logs_and_clears_them() -> None:
    depth_model = make_depth_model()
    wrapper = dinov2.DinoInstSegRGBD3D(depth_model=depth_model, rgb_fusion="raw_point")

    depth_model.log("train/iou", 0.5)
    wrapper.log("wrapper_metric", 1.0)

    logs = wrapper.get_log()

    assert cast(dict[str, tuple[Any, int]], logs)["train/iou"][0] == 0.5
    assert cast(dict[str, tuple[Any, int]], logs)["wrapper_metric"][0] == 1.0

    wrapper.clear_log()

    assert wrapper.get_log() == {}
    assert depth_model.get_log() == {}
