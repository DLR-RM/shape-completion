from typing import Any, cast

import pytest
import torch
from torch import nn

from models import get_activation

tcnn = pytest.importorskip("tinycudann")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tinycudann():
    n_embd = 256
    hidden_layer_multiplier = 2
    bias = False
    activation = "relu"

    n_neurons = hidden_layer_multiplier * n_embd
    config = {
        "otype": "FullyFusedMLP" if n_neurons <= 128 else "CutlassMLP",
        "activation": activation,
        "output_activation": "None",
        "n_neurons": n_neurons,
        "n_hidden_layers": 1,
    }
    mlp_tcnn = tcnn.Network(n_input_dims=n_embd, n_output_dims=n_embd, network_config=config)

    linear1 = nn.Linear(n_embd, n_neurons, bias)
    linear2 = nn.Linear(n_neurons, n_embd, bias)
    activation_module = cast(nn.Module, cast(Any, get_activation(activation)))
    mlp_torch = nn.Sequential(linear1, activation_module, linear2)

    params = torch.cat(
        [
            linear1.weight.data.flatten(),
            linear2.weight.data.flatten(),
        ]
    ).half()
    mlp_tcnn.params.data[...] = params

    x = torch.randn(4, 3000, n_embd)

    if torch.cuda.is_available():
        x = x.cuda()
        mlp_tcnn.cuda()
        mlp_torch.cuda()

    B, N, C = x.size()
    y_tcnn = mlp_tcnn(x.reshape(-1, C)).view(B, N, n_embd).to(x.dtype)
    y_torch = mlp_torch(x)

    assert torch.allclose(y_tcnn, y_torch, rtol=0.01, atol=0.01)
