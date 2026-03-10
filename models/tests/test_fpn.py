import torch

from ..src.fpn import FPN, SimpleConvDecoder, _expand


def test_expand_repeats_batch_dimension():
    tensor = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    expanded = _expand(tensor, length=3)

    assert expanded.shape == (6, 3, 2, 2)
    assert torch.equal(expanded[0], tensor[0])
    assert torch.equal(expanded[2], tensor[0])
    assert torch.equal(expanded[3], tensor[1])
    assert torch.equal(expanded[5], tensor[1])


def test_fpn_forward_with_skip_batch_expansion():
    model = FPN(dim=64, fpn_dims=[256, 128, 64], context_dim=512, num_groups=8)
    x = torch.randn(4, 64, 4, 4)
    fpns = [
        torch.randn(2, 256, 4, 4),
        torch.randn(2, 128, 8, 8),
        torch.randn(2, 64, 16, 16),
    ]

    out = model(x, fpns)
    assert out.shape == (4, 1, 16, 16)


def test_simple_conv_decoder_output_size_override():
    model = SimpleConvDecoder(input_dim=768, num_masks=7, decoder_channels=(512, 256, 128, 64))
    x = torch.randn(2, 768, 14, 14)

    out_default = model(x)
    out_override = model(x, size=(32, 48))

    assert out_default.shape == (2, 7, 112, 112)
    assert out_override.shape == (2, 7, 32, 48)
