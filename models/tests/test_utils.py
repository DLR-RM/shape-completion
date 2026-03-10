import builtins

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import grid_sample
from torch_scatter import scatter_mean

from utils import coordinates_to_index, points_to_coordinates

from ..src import utils as utils_module
from ..src.utils import check_finite_context, dice_loss, grid_sample_2d, grid_sample_3d


@pytest.fixture
def padding() -> float:
    return 0.1


def test_dice_loss():
    preds = torch.rand(7, 224, 224)
    targets = torch.randint(0, 2, preds.shape).float()
    loss = dice_loss(preds, targets)
    assert torch.isfinite(loss)


def test_check_finite():
    tensor = torch.tensor([[1.0, 2.0], [torch.inf, 4.0]])
    with check_finite_context(tensor, do_raise=False) as t:
        tensor[t] = 3.0

    assert torch.equal(tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    try:
        with check_finite_context(tensor, name="test"):
            tensor[0, 0] = torch.nan
    except ValueError as e:
        print(f"Exception: {e}")


def test_custom_grid_sample_2d(padding: float):
    resolution = 16
    inputs = torch.rand(1, 300, 3) - 0.5
    feature = torch.rand(1, 32, 300)
    points = (1 + padding) * (torch.rand(1, 2048, 3) - 0.5)

    coordinates = points_to_coordinates(inputs, max_value=1 + padding, plane="xy")
    assert torch.is_tensor(coordinates)
    index = coordinates_to_index(coordinates, resolution).unsqueeze(1)
    feature = scatter_mean(feature, index, dim_size=resolution**2).view(
        feature.size(0), feature.size(1), resolution, resolution
    )
    feature.requires_grad_()

    for plane in ["xy", "xz", "yz"]:
        coordinates = points_to_coordinates(points, plane=plane)  # Don't normalize to get out-of-bounds values
        assert torch.is_tensor(coordinates)
        grid = 2 * coordinates - 1
        grid = grid[:, :, None]

        for padding_mode in ["zeros", "border"]:
            feature_sampled = grid_sample(feature, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
            loss_standard = feature_sampled.sum()
            loss_standard.backward()

            assert feature.grad is not None
            gradients = feature.grad.clone()
            feature.grad.zero_()

            feature_sampled_custom = grid_sample_2d(feature, grid, padding_mode=padding_mode)
            loss_custom = feature_sampled_custom.sum()
            loss_custom.backward()

            assert feature.grad is not None
            gradients_custom = feature.grad.clone()
            feature.grad.zero_()

            assert torch.allclose(feature_sampled, feature_sampled_custom)
            assert torch.allclose(gradients, gradients_custom)


def test_custom_grid_sample_3d(padding):
    resolution = 16
    inputs = torch.rand(1, 300, 3) - 0.5
    feature = torch.rand(1, 32, 300)
    points = (1 + padding) * (torch.rand(1, 2048, 3) - 0.5)

    coordinates = points_to_coordinates(inputs, max_value=1 + padding)
    assert torch.is_tensor(coordinates)
    index = coordinates_to_index(coordinates, resolution).unsqueeze(1)
    feature = scatter_mean(feature, index, dim_size=resolution**3).view(*feature.shape[:2], *(resolution,) * 3)
    feature.requires_grad_()

    coordinates = points_to_coordinates(points)  # Don't normalize to get out-of-bounds values
    assert torch.is_tensor(coordinates)
    grid = 2 * coordinates - 1
    grid = grid[:, :, None, None]

    for padding_mode in ["zeros", "border"]:
        feature_sampled = grid_sample(feature, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
        loss_standard = feature_sampled.sum()
        loss_standard.backward()

        assert feature.grad is not None
        gradients = feature.grad.clone()
        feature.grad.zero_()

        feature_sampled_custom = grid_sample_3d(feature, grid, padding_mode=padding_mode)
        loss_custom = feature_sampled_custom.sum()
        loss_custom.backward()

        assert feature.grad is not None
        gradients_custom = feature.grad.clone()
        feature.grad.zero_()

        assert torch.allclose(feature_sampled, feature_sampled_custom)
        assert torch.allclose(gradients, gradients_custom)


def test_loss_helpers_cover_classification_regression_and_reduction():
    logits = torch.tensor([[0.0, 1.0], [1.5, -0.5]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    expected_sum = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
    assert torch.allclose(utils_module.classification_loss(logits, targets, reduction="sum"), expected_sum)
    assert torch.allclose(utils_module.get_loss(logits, targets, reduction="sum"), expected_sum)

    for name in ["weighted_bce", "inv_freq_bce", "pifuhd_bce", "focal", "dice", "focal_dice"]:
        loss = utils_module.classification_loss(logits, targets, name=name, reduction="mean")
        assert torch.isfinite(loss)

    class_logits = torch.tensor([[[2.0, -1.0, 0.5], [0.1, 1.2, -0.3]]], dtype=torch.float32)
    class_targets = torch.tensor([[0, 2]], dtype=torch.long)
    assert torch.isfinite(utils_module.classification_loss(class_logits, class_targets, name="weighted_ce"))

    sdf_pred = torch.tensor([[-0.2, 0.0, 0.3], [0.1, -0.5, 0.02]], dtype=torch.float32)
    sdf_target = torch.tensor([[-0.1, 0.01, 0.0], [0.2, -0.2, -0.01]], dtype=torch.float32)
    for name in ["l1", "smooth_l1", "mse", "shape_l1", "inv_weight_l1", "inv_l1", "sign_l1", "disn"]:
        loss = utils_module.regression_loss(
            sdf_pred,
            sdf_target,
            tsdf=0.25 if name == "l1" else None,
            name=name,
            reduction="mean",
        )
        assert torch.isfinite(loss)

    point_loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    assert utils_module.reduce_loss(point_loss, reduction="sum_points").item() == pytest.approx(5.0)

    with pytest.raises(NotImplementedError, match="Reduction"):
        utils_module.reduce_loss(point_loss, reduction="bad")
    with pytest.raises(NotImplementedError, match="Loss 'bad'"):
        utils_module.regression_loss(sdf_pred, sdf_target, name="bad")


def test_activation_norm_prob_and_dim_helpers():
    assert utils_module.reduce_dim(torch.arange(24, dtype=torch.float32).view(1, 2, 3, 4)).shape == (1, 24)
    assert utils_module.reduce_dim(torch.arange(48, dtype=torch.float32).view(1, 2, 3, 4, 2)).shape == (1, 48)

    with pytest.raises(ValueError, match="Unsupported input dimension"):
        utils_module.reduce_dim(torch.ones(1, dtype=torch.float32))

    probs_2d = utils_module.probs_from_logits(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
    assert torch.allclose(probs_2d, torch.tensor([[0.5, torch.sigmoid(torch.tensor(1.0)).item()]], dtype=torch.float32))

    logits_3d = torch.tensor([[[0.0, 1.0, 2.0], [1.0, 0.0, -1.0]]], dtype=torch.float32)
    probs_3d = utils_module.probs_from_logits(logits_3d)
    assert probs_3d.shape == logits_3d.shape
    assert torch.allclose(probs_3d.sum(dim=-1), torch.ones_like(probs_3d.sum(dim=-1)))

    with pytest.raises(ValueError, match="Unsupported input dimension"):
        utils_module.probs_from_logits(torch.ones((1, 1, 1, 1, 1, 1), dtype=torch.float32))

    assert isinstance(utils_module.get_activation("relu", inplace=True), torch.nn.ReLU)
    assert isinstance(utils_module.get_activation("new_gelu", jit=False), utils_module.NewGelu)
    assert isinstance(utils_module.get_activation("geglu"), utils_module.GEGLU)
    assert isinstance(utils_module.get_activation("softplus", beta=2, threshold=3), torch.nn.Softplus)
    assert isinstance(utils_module.get_activation("silu"), torch.nn.SiLU)
    assert isinstance(utils_module.get_activation("selu"), torch.nn.SELU)
    with pytest.raises(NotImplementedError, match="Activation"):
        utils_module.get_activation("bad")  # pyright: ignore[reportArgumentType]

    assert isinstance(utils_module.get_norm("batch", 4, dim=2), torch.nn.BatchNorm2d)
    assert isinstance(utils_module.get_norm("instance", 4, dim=3), torch.nn.InstanceNorm3d)
    assert isinstance(utils_module.get_norm("layer", 4, dim=1), torch.nn.LayerNorm)
    assert isinstance(utils_module.get_norm("group", 4, dim=2, num_groups=2), torch.nn.GroupNorm)
    with pytest.raises(AssertionError, match="LayerNorm"):
        utils_module.get_norm("layer", 4, dim=2)
    with pytest.raises(AssertionError, match="Invalid dimension"):
        utils_module.get_norm("batch", 4, dim=0)
    with pytest.raises(NotImplementedError, match="Normalization"):
        utils_module.get_norm("bad", 4)  # pyright: ignore[reportArgumentType]

    inv = utils_module.inverse_sigmoid(torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32))
    assert torch.isfinite(inv).all()
    assert inv[1].item() == pytest.approx(0.0, abs=1e-6)


def test_geometry_mask_and_score_helpers():
    boxes_cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.4]], dtype=torch.float32)
    boxes_xyxy = utils_module.box_cxcywh_to_xyxy(boxes_cxcywh)
    boxes_xywh = utils_module.box_cxcywh_to_xywh(boxes_cxcywh)
    assert torch.allclose(boxes_xyxy, torch.tensor([[0.4, 0.3, 0.6, 0.7]], dtype=torch.float32))
    assert torch.allclose(utils_module.box_xyxy_to_cxcywh(boxes_xyxy), boxes_cxcywh)
    assert torch.allclose(utils_module.box_xyxy_to_xywh(boxes_xyxy), boxes_xywh)
    assert torch.allclose(utils_module.box_xywh_to_xyxy(boxes_xywh), boxes_xyxy)
    assert torch.allclose(utils_module.box_xywh_to_cxcywh(boxes_xywh), boxes_cxcywh)

    preds = torch.tensor([[1.0, 1.0, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    iou, union = utils_module.box_iou(preds, targets)
    assert iou.shape == union.shape == (2, 1)
    assert iou[0, 0].item() == pytest.approx(0.0)
    assert iou[1, 0].item() == pytest.approx(1.0)
    assert torch.isfinite(utils_module.generalized_box_iou(preds, targets)).all()

    masks = torch.zeros((2, 4, 4), dtype=torch.float32)
    masks[0, 1:3, 1:4] = 1
    boxes_from_masks = utils_module.get_boxes_from_masks(masks)
    circles_from_masks = utils_module.get_circles_from_masks(masks)
    assert boxes_from_masks.shape == (2, 4)
    assert circles_from_masks.shape == (2, 3)
    assert torch.equal(boxes_from_masks[1], torch.zeros(4))
    assert circles_from_masks[0, 2] > 0

    pixel_box = torch.tensor([[1.0, 1.0, 3.0, 3.0]], dtype=torch.float32)
    circles = utils_module.get_circles_from_boxes(pixel_box, size=(4, 4), method="inscribed")
    boxes_back = utils_module.get_boxes_from_circles(circles, size=(4, 4), method="inscribed")
    assert circles.shape == (1, 3)
    assert boxes_back.shape == (1, 4)
    with pytest.raises(Exception, match="Unknown method"):
        utils_module.get_circles_from_boxes(pixel_box, method="bad")
    with pytest.raises(Exception, match="Unknown method"):
        utils_module.get_boxes_from_circles(circles, method="bad")

    pairwise_giou = utils_module.generalized_circle_iou(
        torch.tensor([[0.5, 0.5, 0.2]], dtype=torch.float32),
        torch.tensor([[0.5, 0.5, 0.2], [0.8, 0.8, 0.1]], dtype=torch.float32),
        mode="pairwise",
    )
    elementwise_iou = utils_module.generalized_circle_iou(
        torch.tensor([[0.5, 0.5, 0.2]], dtype=torch.float32),
        torch.tensor([[0.5, 0.5, 0.2]], dtype=torch.float32),
        return_iou=True,
    )
    assert pairwise_giou.shape == (1, 2)
    assert elementwise_iou.shape == (1,)
    assert elementwise_iou[0].item() > 0.99
    with pytest.raises(Exception, match="same shape"):
        utils_module.generalized_circle_iou(
            torch.tensor([[0.5, 0.5, 0.2]], dtype=torch.float32),
            torch.tensor([[0.5, 0.5, 0.2], [0.8, 0.8, 0.1]], dtype=torch.float32),
            mode="elementwise",
        )

    label_map = utils_module.labels_from_masks(
        torch.tensor(
            [
                [[1, 0], [0, 0]],
                [[0, 1], [1, 0]],
            ],
            dtype=torch.bool,
        ),
        template=torch.zeros((2, 2), dtype=torch.long),
    )
    assert torch.equal(label_map, torch.tensor([[1, 2], [2, 0]], dtype=torch.long))

    logits = torch.tensor(
        [
            [
                [[5.0, -5.0], [1.0, -1.0]],
                [[1.0, 4.0], [0.0, -2.0]],
            ]
        ],
        dtype=torch.float32,
    )
    labels = utils_module.labels_from_logits(logits, threshold=0.6)
    assert torch.equal(labels, torch.tensor([[[1, 2], [1, 0]]], dtype=torch.long))

    dense_masks = utils_module.masks_from_labels(labels, reindex=False, drop_background=True)
    reindexed_masks = utils_module.masks_from_labels(labels, reindex=True, drop_background=False)
    assert len(dense_masks) == len(reindexed_masks) == 1
    assert dense_masks[0].shape == (2, 2, 2)
    assert reindexed_masks[0].shape == (3, 2, 2)

    inst_masks = torch.tensor(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    inst_scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    filtered_masks, filtered_scores, keep_idx = utils_module.filter_instance_masks(
        inst_masks,
        inst_scores,
        min_size=2,
        nms_iou=0.5,
        threshold=0.5,
    )
    assert filtered_masks.shape[0] == filtered_scores.shape[0] == keep_idx.shape[0] == 1
    assert keep_idx.tolist() == [0]

    cls_scores_topk = utils_module.cls_probs_from_logits(logits, method="top_k")
    cls_scores_ensemble = utils_module.cls_probs_from_logits(logits, method="ensemble")
    cls_scores_gated = utils_module.cls_probs_from_logits(logits, method="mean", min_size=10)
    assert cls_scores_topk.shape == cls_scores_ensemble.shape == cls_scores_gated.shape == (1, 2)
    assert (cls_scores_topk >= 0).all() and (cls_scores_topk <= 1).all()
    assert torch.equal(cls_scores_gated, torch.zeros_like(cls_scores_gated))


def test_matcher_and_sinkhorn_helpers():
    target_masks = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    mask_logits = target_masks.unsqueeze(0) * 8.0 - 4.0
    occ_logits = torch.tensor([[[8.0, -8.0, -8.0, -8.0], [-8.0, -8.0, -8.0, 8.0]]], dtype=torch.float32)
    occ_targets = [torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)]
    dets = utils_module.get_boxes_from_masks(target_masks).unsqueeze(0)
    cls_preds = torch.tensor([[0.9, 0.1]], dtype=torch.float32)

    pairwise_dice = utils_module.batch_dice_loss(mask_logits[0], target_masks)
    pairwise_bce = utils_module.batch_binary_ce_or_focal_loss(mask_logits[0], target_masks)
    pairwise_focal = utils_module.batch_binary_ce_or_focal_loss(mask_logits[0], target_masks, focal=True)
    assert pairwise_dice.shape == pairwise_bce.shape == pairwise_focal.shape == (2, 2)
    assert pairwise_dice[0, 0] < pairwise_dice[0, 1]
    assert torch.isfinite(pairwise_bce).all()
    assert torch.isfinite(pairwise_focal).all()

    match, avg_cost = utils_module.hungarian_matcher(
        1,
        mask_logits=mask_logits,
        mask_tgt=[target_masks],
        occ_logits=occ_logits,
        occ_tgt=occ_targets,
        dets=dets,
        cls_preds=cls_preds,
        det_weight=1.0,
        cls_weight=1.0,
    )
    assert len(match) == 1
    assert torch.equal(match[0][0], torch.tensor([0, 1]))
    assert torch.equal(match[0][1], torch.tensor([0, 1]))
    assert isinstance(avg_cost, float)

    batch_idx, src_idx = utils_module.index_from_match(match)
    assert torch.equal(batch_idx, torch.tensor([0, 0]))
    assert torch.equal(src_idx, torch.tensor([0, 1]))

    sinkhorn = utils_module.sinkhorn_segmentation_loss(mask_logits, [target_masks], reduction="none")
    batched_sinkhorn = utils_module.batched_sinkhorn_segmentation_loss(mask_logits, [target_masks], reduction="none")
    assert sinkhorn.shape == batched_sinkhorn.shape == (1,)
    assert torch.allclose(sinkhorn, batched_sinkhorn, atol=1e-5)

    zero_targets = [torch.empty((0, 2, 2), dtype=torch.float32)]
    assert utils_module.sinkhorn_segmentation_loss(mask_logits, zero_targets).item() == pytest.approx(0.0)
    assert utils_module.batched_sinkhorn_segmentation_loss(mask_logits, zero_targets).item() == pytest.approx(0.0)

    with pytest.raises(ValueError, match="No valid matching costs"):
        utils_module.hungarian_matcher(1)


def test_query_sampling_and_mask_building_helpers(monkeypatch: pytest.MonkeyPatch):
    logits = torch.tensor([[0.0, 3.0, -4.0, 0.1]], dtype=torch.float32)
    uncertain = utils_module.sample_uncertain(logits, 2, importance_sample_ratio=1.0)
    assert uncertain.shape == (1, 2)
    assert set(uncertain[0].tolist()) == {0, 3}
    assert torch.equal(utils_module.sample_uncertain(logits, 10), torch.tensor([[0, 1, 2, 3]]))

    inst_logits = torch.tensor(
        [
            [
                [[5.0, -5.0], [5.0, -5.0]],
                [[4.0, 4.0], [-5.0, -5.0]],
            ]
        ],
        dtype=torch.float32,
    )
    masks = utils_module.masks_from_logits(
        inst_logits,
        scores=torch.tensor([[0.2, 0.9]], dtype=torch.float32),
        threshold=0.5,
        apply_filter=False,
    )
    assert len(masks) == 1
    assert masks[0].shape == (2, 2, 2)
    assert torch.equal(masks[0][0], inst_logits[0, 1])
    assert torch.equal(masks[0][1], inst_logits[0, 0])

    queries = torch.tensor([[[1.0, 0.0], [0.0, 3.0], [2.0, 2.0], [1.0, 1.0]]], dtype=torch.float32)
    scores = torch.tensor([[0.1, 0.8, 0.6, 0.2]], dtype=torch.float32)

    hard = utils_module.queries_from_feat(queries, scores=scores, num_queries=2)
    assert torch.equal(hard, torch.tensor([[[0.0, 3.0], [2.0, 2.0]]], dtype=torch.float32))

    soft = utils_module.queries_from_feat(queries, scores=scores, num_queries=2, gather="soft", tau=0.01)
    assert soft.shape == (1, 2, 2)
    assert torch.allclose(soft, hard, atol=1e-4)

    torch.manual_seed(0)
    gumbel = utils_module.queries_from_feat(queries, scores=scores, num_queries=2, gather="gumbel", ste=True)
    assert gumbel.shape == (1, 2, 2)
    assert torch.isfinite(gumbel).all()

    assert utils_module.queries_from_feat(torch.empty((1, 0, 2)), num_queries=3).shape == (1, 0, 2)

    def _fake_fps(points: torch.Tensor, num_samples: int, return_indices: bool = True) -> torch.Tensor:
        assert return_indices is True
        assert points.shape == (1, 4, 3)
        assert num_samples == 2
        return torch.tensor([[0, 2]], device=points.device, dtype=torch.long)

    monkeypatch.setattr(utils_module, "furthest_point_sample", _fake_fps)
    fps_selected = utils_module.queries_from_feat(queries, num_queries=2, select="fps", gather="hard")
    assert torch.equal(fps_selected, torch.tensor([[[1.0, 0.0], [2.0, 2.0]]], dtype=torch.float32))

    with pytest.raises(ValueError, match="not supported when select='fps'"):
        utils_module.queries_from_feat(queries, num_queries=2, select="fps", gather="soft")
    with pytest.raises(ValueError, match="Unknown select method"):
        utils_module.queries_from_feat(queries, num_queries=2, select=None)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="Unknown gather method"):
        utils_module.queries_from_feat(queries, scores=scores, num_queries=2, gather="bad")  # pyright: ignore[reportArgumentType]


def test_basic_helper_modules_and_permute():
    torch.manual_seed(0)
    tensor = torch.arange(6, dtype=torch.float32).view(2, 3)
    permuted_tensor = utils_module.permute(tensor, dim=1)
    assert torch.is_tensor(permuted_tensor)
    sorted_tensor, _ = torch.sort(permuted_tensor, dim=1)
    assert torch.equal(sorted_tensor, tensor)

    np.random.seed(0)
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    permuted_array = utils_module.permute(array, dim=1)
    assert np.array_equal(np.sort(permuted_array, axis=1), array)

    with pytest.raises(TypeError, match="Unsupported type"):
        utils_module.permute([1, 2, 3], dim=0)  # pyright: ignore[reportArgumentType]

    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(utils_module.new_gelu(x), utils_module.new_gelu_jit(x))
    assert torch.allclose(utils_module.NewGelu(jit=True)(x), utils_module.new_gelu(x))
    assert torch.allclose(utils_module.NewGelu(jit=False)(x), utils_module.new_gelu(x))

    geglu_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    expected_geglu = geglu_input[:, :2] * F.gelu(geglu_input[:, 2:])
    assert torch.allclose(utils_module.GEGLU()(geglu_input), expected_geglu)

    assert torch.equal(utils_module.drop_path(tensor, drop_prob=0.0, training=True), tensor)
    assert torch.equal(utils_module.drop_path(tensor, drop_prob=0.5, training=False), tensor)
    assert torch.equal(utils_module.drop_path(tensor, drop_prob=1.0, training=True), torch.zeros_like(tensor))

    drop_module = utils_module.DropPath(drop_prob=1.0)
    drop_module.train()
    assert torch.equal(drop_module(tensor), torch.zeros_like(tensor))
    drop_module.eval()
    assert torch.equal(drop_module(tensor), tensor)
    assert drop_module.extra_repr() == "drop_prob=1.000"


def test_wrapper_precision_and_range_helpers(monkeypatch: pytest.MonkeyPatch):
    warnings: list[str] = []
    exceptions: list[str] = []
    breakpoints: list[bool] = []

    monkeypatch.setattr(utils_module.logger, "warning", warnings.append)
    monkeypatch.setattr(utils_module.logger, "exception", exceptions.append)
    monkeypatch.setattr(builtins, "breakpoint", lambda: breakpoints.append(True))

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = "initial"
            self.backend = "legacy"

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()
            self.linear = nn.Linear(2, 2)

    model = Wrapper()
    assert utils_module.patch_attention(model) is model
    patched = utils_module.patch_attention(model, mode="flash", backend="torch")
    assert patched is model
    assert model.attn.mode == "flash"
    assert model.attn.backend == "torch"

    @utils_module.breakpoint_on_exception
    def explode() -> None:
        raise RuntimeError("boom")

    assert explode() is None
    assert breakpoints == [True]
    assert len(exceptions) == 1 and "explode" in exceptions[0]

    @utils_module.check_precision(torch.float32)
    def add_with_extra(x: torch.Tensor, y: torch.Tensor, extra: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        return x + y + extra["bonus"][0]

    ones = torch.ones(2, dtype=torch.float32)
    assert torch.equal(add_with_extra(ones, ones, extra={"bonus": [ones]}), torch.full((2,), 3.0))
    with pytest.raises(ValueError, match=r"Expected dtype torch\.float32"):
        add_with_extra(ones, ones.double(), extra={"bonus": [ones]})

    @utils_module.check_range(torch.float16)
    def echo(x: torch.Tensor) -> torch.Tensor:
        return x

    near_limit = torch.tensor([60000.0], dtype=torch.float16)
    assert torch.equal(echo(near_limit), near_limit)
    assert any("approaching the limits" in warning for warning in warnings)

    with pytest.raises(ValueError, match="out of range"):
        echo(torch.tensor([float("inf")], dtype=torch.float16))


def test_call_group_and_segmentation_loss_helpers():
    @utils_module.count_calls
    def record_count(*, n_calls: int) -> int:
        return n_calls

    assert record_count() == 1
    assert record_count() == 2

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)
            self.norm = nn.LayerNorm(2)
            self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)

    model = SmallModel()
    decay, no_decay = utils_module.assign_params_groups(model)
    decay_ids = {id(param) for param in decay}
    no_decay_ids = {id(param) for param in no_decay}
    assert id(model.linear.weight) in decay_ids
    assert id(model.linear.bias) in no_decay_ids
    assert id(model.norm.weight) in no_decay_ids
    assert id(model.norm.bias) in no_decay_ids
    assert id(model.frozen) not in decay_ids | no_decay_ids

    logits = torch.tensor(
        [
            [[6.0, -6.0], [-6.0, 6.0]],
            [[-8.0, -8.0], [-8.0, -8.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    dice_score = utils_module.calculate_dice_score(targets.flatten(1), targets.flatten(1))
    assert torch.allclose(dice_score, torch.ones_like(dice_score))
    assert utils_module.tversky_loss(logits[:1], targets[:1]).item() < 0.01

    dice_none = utils_module.dice_loss(logits[:1], targets[:1], reduction="none")
    assert dice_none.shape == (1,)
    assert torch.isfinite(utils_module.dice_loss_instr(logits, targets, power=0.3, neg_weight=0.5))

    empty_logits = torch.empty((0, 2, 2), dtype=torch.float32)
    empty_targets = torch.empty((0, 2, 2), dtype=torch.float32)
    assert utils_module.dice_loss(empty_logits, empty_targets, reduction="sum").item() == pytest.approx(0.0)
    assert utils_module.dice_loss_instr(empty_logits, empty_targets).item() == pytest.approx(0.0)
    with pytest.raises(Exception, match="Both predictions and targets are empty"):
        utils_module.dice_loss(empty_logits, empty_targets, reduction="none")
