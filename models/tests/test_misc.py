import torch


def test_conv1x1_vs_fc():
    conv = torch.nn.Conv1d(3, 8, 1)
    fc = torch.nn.Linear(3, 8)
    assert conv.bias is not None
    assert fc.bias is not None
    conv.weight.data = fc.weight.data.unsqueeze(-1)
    conv.bias.data = fc.bias.data
    x = torch.randn(2, 10, 3)
    assert torch.allclose(conv(x.transpose(1, 2)).transpose(1, 2), fc(x), atol=1e-6, rtol=1e-5)
