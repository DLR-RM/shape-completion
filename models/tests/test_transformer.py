import time

import pytest
import torch

from ..src.transformer import (
    TCNN_EXISTS,
    XFORMERS_EXISTS,
    Attention,
    MultiHeadAttention,
    NeRFEncoding,
    PyTorchMultiHeadAttention,
)


def test_permutation():
    b, n, c = 2, 16, 8
    q = torch.randn(b, n, c)
    k = v = torch.randn(b, n, c)
    index = torch.randperm(n)
    attn = Attention()

    # Attention is equivariant to query permutations
    assert torch.allclose(attn(q[:, index], k, v), attn(q, k, v)[:, index], atol=1e-6)

    # Attention is invariant to key/value permutations
    assert torch.allclose(attn(q, k[:, index], v[:, index]), attn(q, k, v), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attention():
    attn1 = MultiHeadAttention(32, 4).eval().cuda()
    attn2 = PyTorchMultiHeadAttention(32, 4).eval().cuda()

    with torch.no_grad():
        attn2.attn.in_proj_weight.copy_(attn1.to_qkv.weight)
        attn2.attn.out_proj.weight.copy_(attn1.to_out.weight)

    q = torch.randn(2, 128, 32, device="cuda")

    backends = ["einops", "torch", "xformers"] if XFORMERS_EXISTS else ["einops", "torch"]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        start = time.perf_counter()
        for _ in range(100):
            y2 = attn2(q)
        print(attn2.__class__.__name__, time.perf_counter() - start)

        start = time.perf_counter()
        for _ in range(100):
            attn1(q)
        print(attn1.__class__.__name__, time.perf_counter() - start)

        for backend in backends:
            attn1.attn.backend = backend
            start = time.perf_counter()
            for _ in range(100):
                y1 = attn1(q)
            print(f"{attn1.__class__.__name__} ({attn1.attn.backend}):", time.perf_counter() - start)

            assert y1.size() == y2.size()
            assert torch.allclose(y1, y2, atol=1e-3)


class TestNeRFEncoding:
    def test_overlap(self):
        dim = 3
        nerf_enc = NeRFEncoding(dim, include_inputs=False, implementation="torch")
        for p1 in torch.linspace(0, 1, 11):
            for p2 in torch.linspace(0, 1, 11):
                if p1 != p2:
                    points = torch.tensor([[p1.item()] * dim, [p2.item()] * dim]).view([1, 2, dim]) - 0.5
                    points_enc = nerf_enc(points)
                    dist = torch.linalg.norm(points_enc[0, 0] - points_enc[0, 1])
                    print(points[0, 0], points[0, 1], dist)
                    assert not torch.allclose(
                        points_enc[0, 0], points_enc[0, 1], atol=1e-4
                    )  # Periodicity maps min/max values to same value

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not TCNN_EXISTS, reason="tinycudann not available")
    def test_vis(self):
        import matplotlib.pyplot as plt

        def plot_nerf_encoding_outputs(encoding_module, input_tensor, title_prefix=""):
            """
            Plots the output of a NeRFEncoding module for a given input tensor.
            Assumes input_tensor is effectively 1D for plotting (e.g., [N,1]),
            or if [N,D] (D>1), it plots against the first dimension of the input.
            """
            if not isinstance(input_tensor, torch.Tensor):
                print(f"Error: input_tensor is not a torch.Tensor. Got type: {type(input_tensor)}")
                return

            with torch.no_grad():  # Disable gradient calculations for inference
                y_encoded = encoding_module(input_tensor)

            plt.figure(figsize=(15, 10))

            num_output_dims = y_encoded.shape[-1]

            # Use the first dimension of input_tensor for the x-axis if it's multi-dimensional
            plot_x_values_np = (
                input_tensor[:, 0].cpu().numpy() if input_tensor.shape[-1] > 1 else input_tensor.cpu().numpy().squeeze()
            )
            x_label_suffix = " (varying dim 0)" if input_tensor.shape[-1] > 1 else ""

            # Plot each output dimension
            # For more specific labels (sin/cos/input), you'd need to know the exact
            # configuration of encoding_module (include_inputs, implementation, in_dim).
            for i in range(num_output_dims):
                plt.plot(plot_x_values_np, y_encoded[:, i].cpu().numpy(), label=f"Output Dim {i}")

            plt.xlabel(f"Input Value{x_label_suffix}")
            plt.ylabel("Encoded Value")

            title_components = [title_prefix]
            if hasattr(encoding_module, "implementation"):
                title_components.append(f"impl: {encoding_module.implementation}")
            if hasattr(encoding_module, "num_frequencies"):
                title_components.append(f"freqs: {encoding_module.num_frequencies}")
            if hasattr(encoding_module, "in_dim"):
                title_components.append(f"in_dim: {encoding_module.in_dim}")
            if hasattr(encoding_module, "mlp") and encoding_module.mlp is not None:
                title_components.append("MLP active")
            else:
                title_components.append("Raw encoding (no MLP)")

            plt.title("NeRFEncoding Output: " + ", ".join(filter(None, title_components)))
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=(1 if num_output_dims <= 15 else 2))
            plt.grid(True)
            plt.tight_layout(rect=(0.0, 0.0, 0.8 if num_output_dims > 15 else 0.85, 1.0))  # Adjust for legend
            plt.close()

        points = torch.linspace(-1, 1, 1000).view(-1, 1).cuda()  # 1D input
        nerf_enc = NeRFEncoding(
            in_dim=1,
            num_frequencies=4,
            max_freq_exp=3,
            include_inputs=True,
            normalize_inputs=False,
            scale_inputs=False,
            padding=0.1,
            implementation="tcnn",
        )
        # nerf_enc, _ = get_embedder(num_freqs=4, include_input=True, input_dims=1)
        plot_nerf_encoding_outputs(nerf_enc, points, title_prefix="NeRF Encoding")
