from typing import Any, cast

import torch

from ..src.vae import VQVAEModel


def test_quantize():
    vae_cls = cast(Any, VQVAEModel)
    model = vae_cls(ae=None, n_hidden=32, n_code=256)
    x = torch.randn(2, 8, 32)
    q, i, _loss = model.quantize(x)
    assert torch.equal(i, model.quantize(q)[1])
    assert torch.allclose(q, model.quantize(q)[0], atol=1e-3)
