import os

import pytest
import torch

from ..src.vqdif import DEFAULT_KWARGS, VQDIF


@pytest.fixture(scope="session")
def root_dir() -> str:
    return os.environ.get("PRETRAINED_ROOT", "/path/to/pretrained/models")


@pytest.fixture
def path_to_pretrained_vqdif(root_dir) -> str:
    return os.path.join(root_dir, "out/vqdif/shapeformer/vqdif/model_best.pt")


class TestVQDIF:
    def test_init(self):
        VQDIF(**DEFAULT_KWARGS)

    def test_forward(self):
        vqdif = VQDIF(**DEFAULT_KWARGS).cuda()

        inputs = torch.randn(1, 3000, 3).cuda() - 0.5
        points = 1.1 * torch.randn(1, 2048, 3).cuda() - 0.55

        vqdif(inputs, points)

    def test_mask(self):
        vqdif = VQDIF(**DEFAULT_KWARGS)

        inputs = torch.randn(1, 3000, 3) - 0.5
        mask = vqdif.get_mask(inputs, resolution=16, padding=vqdif.encoder.padding)
        assert mask.dtype == torch.bool
        assert tuple(mask.shape) == (1, 16, 16, 16)
        assert mask.any()


class TestTrainedVQDIF:
    def test_forward(self, path_to_pretrained_vqdif):
        if not os.path.isfile(path_to_pretrained_vqdif):
            pytest.skip("Pretrained model not found")

        vqdif = VQDIF(**DEFAULT_KWARGS)
        vqdif.load_state_dict(torch.load(path_to_pretrained_vqdif, weights_only=False)["model"])
        vqdif.eval().cuda()

        inputs = torch.randn(1, 3000, 3).cuda() - 0.5
        points = 1.1 * torch.randn(1, 2048, 3).cuda() - 0.55

        vqdif(inputs, points)
