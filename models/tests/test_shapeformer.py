import os
from typing import Any, cast

import pytest
import torch

from ..src.shapeformer import DEFAULT_KWARGS, ShapeFormer


@pytest.fixture(scope="session")
def root_dir() -> str:
    return os.environ.get("PRETRAINED_ROOT", "/path/to/pretrained/models")


@pytest.fixture
def path_to_pretrained_shapeformer(root_dir) -> str:
    return os.path.join(root_dir, "out/shapeformer/shapeformer/shapeformer_new/model_best.pt")


class TestShapeFormer:
    def test_init(self):
        ShapeFormer(**DEFAULT_KWARGS)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_representer(self):
        shapeformer = ShapeFormer(**DEFAULT_KWARGS).eval().cuda()

        inputs = torch.randn(2, 16384, 3).cuda() - 0.5
        quant_ind, _mode = shapeformer.representer.vqdif.quantize_cloud(inputs)

        max_length = cast(int | None, shapeformer.representer.max_length)
        input_end_tokens = cast(tuple[int, int], shapeformer.representer.input_end_tokens)
        sparse_packed, mode = shapeformer.representer.batch_dense2sparse(
            quant_ind, unpack=False, max_length=max_length, end_tokens=input_end_tokens
        )

        assert _mode == mode

        new_quant_ind = shapeformer.representer.batch_sparse2dense(
            sparse_packed, empty_ind=mode, dense_res=quant_ind.size(-1), return_flattened=False, dim=3
        )

        assert torch.allclose(quant_ind, new_quant_ind)

        quantizer = cast(Any, shapeformer.representer.vqdif.quantizer)
        quant_feat = quantizer.embedding(quant_ind).permute(0, 4, 1, 2, 3)
        new_quant_feat = quantizer.embedding(new_quant_ind).permute(0, 4, 1, 2, 3)

        assert torch.equal(quant_feat, new_quant_feat)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward(self):
        shapeformer = ShapeFormer(**DEFAULT_KWARGS).cuda()

        inputs = torch.randn(2, 16384, 3).cuda() - 0.5
        pointcloud = torch.rand(2, 32768, 3).cuda() - 0.5

        _logits, _targets = shapeformer(inputs, pointcloud)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss(self):
        shapeformer = ShapeFormer(**DEFAULT_KWARGS).cuda()

        inputs = torch.randn(1, 16384, 3).cuda() - 0.5
        pointcloud = torch.rand(1, 32768, 3).cuda() - 0.5

        logits, targets = shapeformer(inputs, pointcloud)
        shapeformer.loss(logits, targets)

    def test_optimizer(self):
        shapeformer = ShapeFormer(**DEFAULT_KWARGS)
        shapeformer.optimizer()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_generate_grid(self, root_dir):
        vqdif_weights = DEFAULT_KWARGS["representer_opt"]["vqdif_opt"]["weights_path"]
        path_to_pretrained_vqdif = os.path.join(root_dir, vqdif_weights)
        if not os.path.exists(path_to_pretrained_vqdif):
            pytest.skip("Pretrained model not found")

        shapeformer = ShapeFormer(**DEFAULT_KWARGS).eval().cuda()
        inputs = torch.randn(1, 16384, 3).cuda() - 0.5
        _grids, _grid_points = shapeformer.generate_grids(inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTrainedShapeFormer:
    def test_forward(self, root_dir, path_to_pretrained_shapeformer):
        vqdif_weights = DEFAULT_KWARGS["representer_opt"]["vqdif_opt"]["weights_path"]
        path_to_pretrained_vqdif = os.path.join(root_dir, vqdif_weights)
        if not os.path.exists(path_to_pretrained_shapeformer) or not os.path.exists(path_to_pretrained_vqdif):
            pytest.skip("Pretrained model not found")

        shapeformer = ShapeFormer(**DEFAULT_KWARGS)
        shapeformer.load_state_dict(torch.load(path_to_pretrained_shapeformer, weights_only=False)["model"])
        shapeformer.cuda()

        inputs = torch.randn(2, 16384, 3).cuda() - 0.5
        pointcloud = torch.rand(2, 32768, 3).cuda() - 0.5

        shapeformer(inputs, pointcloud)

    def test_generate_grid(self, root_dir, path_to_pretrained_shapeformer):
        vqdif_weights = DEFAULT_KWARGS["representer_opt"]["vqdif_opt"]["weights_path"]
        path_to_pretrained_vqdif = os.path.join(root_dir, vqdif_weights)
        if not os.path.exists(path_to_pretrained_shapeformer) or not os.path.exists(path_to_pretrained_vqdif):
            pytest.skip("Pretrained model not found")

        shapeformer = ShapeFormer(**DEFAULT_KWARGS)
        shapeformer.load_state_dict(torch.load(path_to_pretrained_shapeformer, weights_only=False)["model"])
        shapeformer.eval().cuda()

        inputs = torch.randn(1, 16384, 3).cuda() - 0.5
        _grids, _grid_points = shapeformer.generate_grids(inputs)
