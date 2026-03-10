import torch

from ..src.dpt import ConvModule, DPTHead, ReassembleBlocks


def _build_reassemble_inputs(batch_size: int = 1, embed_dims: int = 16):
    return [
        (torch.randn(batch_size, embed_dims, 1, 1), torch.randn(batch_size, embed_dims)),
        (torch.randn(batch_size, embed_dims, 2, 2), torch.randn(batch_size, embed_dims)),
        (torch.randn(batch_size, embed_dims, 4, 4), torch.randn(batch_size, embed_dims)),
        (torch.randn(batch_size, embed_dims, 8, 8), torch.randn(batch_size, embed_dims)),
    ]


def test_conv_module_forward_without_norm():
    module = ConvModule(
        in_channels=8, out_channels=8, kernel_size=3, padding=1, norm_layer=None, act_layer=torch.nn.ReLU
    )
    out = module(torch.randn(2, 8, 8, 8))
    assert out.shape == (2, 8, 8, 8)


def test_reassemble_blocks_deconv_shape_alignment():
    block = ReassembleBlocks(in_channels=16, out_channels=(8, 8, 8, 8), readout_type="project", resize_type="deconv")
    out = block(_build_reassemble_inputs(batch_size=2, embed_dims=16))

    assert len(out) == 4
    for feature in out:
        assert feature.shape == (2, 8, 4, 4)


def test_dpt_head_forward_shape():
    head = DPTHead(
        channels=16,
        embed_dims=16,
        post_process_channels=(8, 8, 8, 8),
        readout_type="project",
        resize_type="deconv",
        conv_type="conv",
    )

    out = head(_build_reassemble_inputs(batch_size=1, embed_dims=16))
    assert out.shape == (1, 16, 64, 64)
