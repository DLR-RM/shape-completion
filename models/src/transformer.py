import math
from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Literal, TypeAlias, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from utils import get_device_info, points_to_coordinates, setup_logger

from .utils import DropPath, get_activation, get_norm

logger = setup_logger(__name__)

ActivationName: TypeAlias = Literal[
    "relu",
    "leaky",
    "leakyrelu",
    "elu",
    "gelu",
    "new_gelu",
    "geglu",
    "new_geglu",
    "softplus",
    "silu",
    "selu",
]
NormName: TypeAlias = Literal["batch", "instance", "layer", "group"]
GetActivationName: TypeAlias = Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu", "softplus"]
TripleBool: TypeAlias = tuple[bool, bool, bool]
TripleFloat: TypeAlias = tuple[float, float, float]
CausalPair: TypeAlias = tuple[bool, bool]
ScaleInitTriple: TypeAlias = tuple[int | None, int | None, int | None]


def _log_debug_level_2(msg: str) -> None:
    debug_level_2 = getattr(logger, "debug_level_2", logger.debug)
    debug_level_2(msg)


def _to_get_activation_name(name: str) -> GetActivationName:
    if name == "new_geglu":
        return "geglu"
    if name == "leakyrelu":
        return "leaky"
    return cast(GetActivationName, name)


def _as_triple_bool(value: bool | tuple[bool, bool, bool]) -> TripleBool:
    if isinstance(value, tuple):
        return value
    return (value, value, value)


def _as_triple_float(value: float | tuple[float, float, float]) -> TripleFloat:
    if isinstance(value, tuple):
        return (float(value[0]), float(value[1]), float(value[2]))
    scalar = float(value)
    return (scalar, scalar, scalar)


def _as_causal_pair(value: bool | tuple[bool, bool]) -> CausalPair:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _as_scale_init_triple(
    value: int | tuple[int | None, int | None, int | None] | list[int | None] | None,
) -> ScaleInitTriple:
    if value is None:
        return (None, None, None)
    if isinstance(value, tuple):
        return (value[0], value[1], value[2])
    if isinstance(value, list):
        if len(value) != 3:
            raise ValueError(f"Expected 3 scale_init_out entries, got {len(value)}")
        return (value[0], value[1], value[2])
    scalar = int(value)
    return (scalar, scalar, scalar)


CUDA_DEVICE_CAPABILITY = get_device_info()[0]

SDP_BACKEND_EXISTS = True
try:
    from torch.nn.attention import SDPBackend
except ImportError:
    SDP_BACKEND_EXISTS = False

XFORMERS_EXISTS = CUDA_DEVICE_CAPABILITY >= 7
try:
    logger.debug("Importing XFormers")
    from xformers import ops as xops  # pyright: ignore[reportMissingImports]
except ImportError as e:
    logger.debug(f"XFormers import failed. Some functionality won't be available: {e}")
    XFORMERS_EXISTS = False

TCNN_EXISTS = True
try:
    logger.debug("Importing TinyCudaNN")
    import tinycudann as tcnn
except (ImportError, OSError) as e:
    logger.debug(f"TinyCudaNN import failed. Some functionality won't be available: {e}")
    TCNN_EXISTS = False


class Attention(nn.Module):
    def __init__(
        self,
        causal: bool = False,
        dropout: float = 0.0,
        mode: str = "flash" if CUDA_DEVICE_CAPABILITY >= 8 else "efficient",
        backend: Literal["torch", "xformers", "einops"] = "torch",
        linear: bool = False,
    ):
        super().__init__()
        assert mode in ["flash", "efficient", "math"], f"Unknown attention mode {mode}"

        self.causal = causal
        self.dropout = dropout
        self.mode = mode
        self.linear = linear
        self.warned = False

        self.backends = ["torch", "einops"]
        if XFORMERS_EXISTS:
            self.backends.append("xformers")
        self.backend = backend

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, value: str):
        if value not in self.backends:
            raise ValueError(f"Backend `{value}` not supported. Supported backends: {self.backends}")

        backend = value

        if value == "einops":
            if self.linear:
                if self.causal:
                    raise NotImplementedError("Linear causal attention not implemented yet")
                if self.dropout:
                    raise NotImplementedError("Linear attention with dropout not implemented yet")

        if value == "xformers":
            if self.mode == "math":
                logger.warning("Math mode not supported with xformers backend. Switching to torch.")
                backend = "torch"
            if self.linear:
                logger.warning("Linear attention not supported with xformers backend. Switching to einops.")
                backend = "einops"

        if value == "torch":
            if self.linear:
                logger.warning("Linear attention not supported with torch backend. Switching to einops.")
                backend = "einops"

        if backend == value:
            backend_info = f"Using {backend} attention backend"
            if backend != "einops":
                backend_info += f" in {self.mode} mode"
            logger.debug(backend_info)

        self._backend = backend

    def _torch(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
        is_16bit = q.dtype in [torch.float16, torch.bfloat16]
        enable_flash = self.mode == "flash" and is_16bit
        enable_math = self.mode == "math" or not is_16bit
        enable_mem_efficient = self.mode == "efficient" or (self.mode == "flash" and not is_16bit) or q.size(-1) > 256

        try:
            if SDP_BACKEND_EXISTS:
                backends = []
                if enable_flash:
                    backends.append(SDPBackend.FLASH_ATTENTION)
                if enable_math:
                    backends.append(SDPBackend.MATH)
                if enable_mem_efficient:
                    backends.append(SDPBackend.EFFICIENT_ATTENTION)
                with torch.nn.attention.sdpa_kernel(backends):
                    return F.scaled_dot_product_attention(
                        q, k, v, dropout_p=self.dropout, is_causal=self.causal
                    )  # [B, H, T, C]
            else:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=enable_flash, enable_math=enable_math, enable_mem_efficient=enable_mem_efficient
                ):
                    return F.scaled_dot_product_attention(
                        q, k, v, dropout_p=self.dropout, is_causal=self.causal
                    )  # [B, H, T, C]
        except torch.cuda.OutOfMemoryError:
            logger.exception(f"CUDA OOM with input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
            raise

    def _xformers(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
        in_dtype = q.dtype
        out_dtype = in_dtype
        is_16bit = in_dtype in [torch.float16, torch.bfloat16]
        op = None
        if self.mode == "flash" and is_16bit:
            op = xops.MemoryEfficientAttentionFlashAttentionOp
        elif self.mode == "efficient" or (self.mode == "flash" and not is_16bit):
            op = xops.MemoryEfficientAttentionOp
            in_dtype = torch.float32
        attn_bias = xops.fmha.attn_bias.LowerTriangularMask() if self.causal else None
        return (
            xops.memory_efficient_attention(
                q.transpose(1, 2).to(in_dtype),
                k.transpose(1, 2).to(in_dtype),
                v.transpose(1, 2).to(in_dtype),
                attn_bias=attn_bias,
                p=self.dropout,
                op=op,
            )
            .transpose(1, 2)
            .to(out_dtype)
        )  # [B, H, C, T] before T

    def _einops(
        self, q: Tensor, k: Tensor, v: Tensor | None = None, linear: bool = False, efficient: bool = False, **kwargs
    ) -> Tensor:
        h = q.size(1)
        scale = q.size(-1) ** -0.5
        if v is None:
            raise ValueError("Value tensor must not be None for attention computation")
        value = v

        if linear or self.linear:
            q = q.softmax(dim=-2)
            k = k.softmax(dim=-1)
            context = torch.einsum("b h t d, b h t e -> b h d e", k, value)
            return torch.einsum("b h e d, b h t d -> b h t e", context, q * scale)

        # Q @ K.T
        if efficient or self.mode == "efficient":
            q_e = rearrange(q, "b h t d -> (b h) t d")
            k_e = rearrange(k, "b h t d -> (b h) t d")
            v_e = rearrange(value, "b h t d -> (b h) t d")
            sim = torch.einsum("b i d, b j d -> b i j", q_e * scale, k_e)
        else:
            sim = torch.einsum("b h i d, b h j d -> b h i j", q * scale, k)

        sim = sim - sim.detach().amax(dim=-1, keepdim=True)
        if self.causal:
            mask = torch.tril(torch.ones(sim.size(-1), sim.size(-1), device=sim.device)).view(1, 1, *sim.shape[-2:])
            sim = sim.masked_fill(mask == 0, float("-inf"))
        attn = sim.softmax(dim=-1)
        if self.dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        if efficient or self.mode == "efficient":
            out = torch.einsum("b i j, b j d -> b i d", attn, v_e)
            return rearrange(out, "(b h) n d -> b h n d", h=h)
        return torch.einsum("b h i j, b h j d -> b h i d", attn, value)  # [B, H, T, C]

    def forward(self, q: Tensor, k: Tensor | None = None, v: Tensor | None = None, **kwargs) -> Tensor:
        k_tensor = q if k is None else k
        v_tensor = q if v is None else v
        if self.backend == "torch":
            return self._torch(q, k_tensor, v_tensor, **kwargs)
        if self.backend == "xformers":
            return self._xformers(q, k_tensor, v_tensor, **kwargs)
        if self.backend == "einops":
            return self._einops(q, k_tensor, v_tensor, **kwargs)
        raise RuntimeError(f"Unsupported attention backend: {self.backend}")


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        in_channels: int | None = None,
        out_channels: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool = False,
        k_dim: int | None = None,
        v_dim: int | None = None,
        chunk: bool = True,
        scale_init_out: int | None = None,
        **kwargs,
    ):
        super().__init__()
        assert n_embd % n_head == 0, f"Embedding dimension {n_embd} must be divisible by number of heads {n_head}"
        if chunk and k_dim is None and v_dim is None:
            self.to_qkv = nn.Linear(in_channels or n_embd, 3 * n_embd, bias=bias)
        elif chunk and k_dim == v_dim:
            self.to_q = nn.Linear(in_channels or n_embd, n_embd, bias=bias)
            self.to_kv = nn.Linear(in_channels or n_embd, 2 * n_embd, bias=bias)
        else:
            self.to_q = nn.Linear(in_channels or n_embd, n_embd, bias=bias)
            if k_dim:
                self.to_k = nn.Linear(in_channels or n_embd, k_dim, bias=bias)
            if v_dim:
                self.to_v = nn.Linear(in_channels or n_embd, v_dim, bias=bias)
        self.to_out = nn.Linear(n_embd, out_channels or n_embd, bias=bias)
        self.residual_dropout = nn.Dropout(dropout, inplace=False)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal
        self.attn = Attention(causal, dropout, **kwargs)

        if scale_init_out:
            nn.init.normal_(self.to_out.weight, std=0.02 / math.sqrt(2 * scale_init_out))

    def forward(self, q: Tensor, k: Tensor | None = None, v: Tensor | None = None, **kwargs) -> Tensor:
        unfold = "b t (h d) -> b h t d"
        assemble = "b h t d -> b t (h d)"
        if hasattr(self, "to_qkv"):
            assert k is None and v is None, "Cannot use to_qkv with k or v"
            q_proj, k_proj, v_proj = self.to_qkv(q).chunk(3, dim=2)
            q = rearrange(q_proj, unfold, h=self.n_head)
            k = rearrange(k_proj, unfold, h=self.n_head)
            v = rearrange(v_proj, unfold, h=self.n_head)
        elif hasattr(self, "to_kv"):
            assert v is None, "Cannot use to_kv with v"
            k = q if k is None else k
            k_proj, v_proj = self.to_kv(k).chunk(2, dim=2)
            k = rearrange(k_proj, unfold, h=self.n_head)
            v = rearrange(v_proj, unfold, h=self.n_head)
            q = rearrange(self.to_q(q), unfold, h=self.n_head)
        else:
            k = q if k is None else k
            v = q if v is None else v
            q = rearrange(self.to_q(q), unfold, h=self.n_head)
            k = rearrange(self.to_k(k), unfold, h=self.n_head)
            v = rearrange(self.to_v(v), unfold, h=self.n_head)

        y = rearrange(self.attn(q, k, v, **kwargs), assemble)
        return self.residual_dropout(self.to_out(y))


class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool = False, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd, num_heads=n_head, dropout=dropout, bias=bias, batch_first=True
        )
        self.causal = causal

    def forward(self, q: Tensor, k: Tensor | None = None, v: Tensor | None = None) -> Tensor:
        k = q if k is None else k
        v = q if v is None else v
        return self.attn(q, k, v, need_weights=False, is_causal=self.causal)[0]


class PyTorchEncoderBlock(nn.TransformerEncoderLayer):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool = True,  # Fixme: PyTorch doesn't support bias=False yet
        dropout: float = 0.0,
        causal: bool = False,
        hidden_layer_multiplier: int = 4,
    ):
        if not bias:
            logger.warning("PyTorch doesn't support bias=False yet")
        super().__init__(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=hidden_layer_multiplier * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn.forward = partial(self.self_attn.forward, is_causal=causal)


class PyTorchEncoder(nn.TransformerEncoder):
    def __init__(
        self,
        n_layer: int,
        n_embd: int,
        n_head: int,
        bias: bool = True,
        dropout: float = 0.0,
        causal: bool = False,
        hidden_layer_multiplier: int = 4,
    ):
        layer = PyTorchEncoderBlock(
            n_embd=n_embd,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            causal=causal,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )
        super().__init__(encoder_layer=layer, num_layers=n_layer)


class PyTorchDecoderBlock(nn.TransformerDecoderLayer):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool = True,  # Fixme: PyTorch doesn't support bias=False yet
        dropout: float = 0.0,
        causal: bool = False,
        hidden_layer_multiplier: int = 4,
    ):
        if not bias:
            logger.warning("PyTorch doesn't support bias=False yet")
        super().__init__(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=hidden_layer_multiplier * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn.forward = partial(self.self_attn.forward, is_causal=causal)


class PyTorchDecoder(nn.TransformerDecoder):
    def __init__(
        self,
        n_layer: int,
        n_embd: int,
        n_head: int,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool = False,
        hidden_layer_multiplier: int = 4,
    ):
        layer = PyTorchDecoderBlock(
            n_embd=n_embd,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            causal=causal,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )
        super().__init__(decoder_layer=layer, num_layers=n_layer)


class MLP(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_out: int | None = None,
        n_hidden: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
        hidden_layer_multiplier: int = 4,
        activation: ActivationName = "gelu",
        backend: Literal["torch", "tcnn"] = "tcnn" if TCNN_EXISTS else "torch",
        scale_init_out: int | None = None,
    ):
        super().__init__()
        assert backend in ["torch", "tcnn"], f"Unknown MLP backend {backend}"

        hidden_dim = n_hidden or hidden_layer_multiplier * n_embd
        if backend == "tcnn" and (
            bias
            or activation
            not in ["none", "relu", "leakyrelu", "exponential", "sine", "sigmoid", "squareplus", "softplus", "tanh"]
        ):
            _log_debug_level_2(f"{self.__class__.__name__}: Switching to 'torch' backend")
            backend = "torch"

        if backend == "tcnn":
            config = {
                "otype": "FullyFusedMLP" if hidden_dim <= 128 else "CutlassMLP",
                "activation": activation,
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 1,
                "dropout": dropout,
            }
            self.net = tcnn.Network(n_input_dims=n_embd, n_output_dims=n_out or n_embd, network_config=config)
            self.forward = self.forward_tcnn
        elif backend == "torch":
            self.c_fc = nn.Linear(n_embd, 2 * hidden_dim if activation == "geglu" else hidden_dim, bias)
            self.c_proj = nn.Linear(hidden_dim // 2 if activation == "new_geglu" else hidden_dim, n_out or n_embd, bias)
            self.dropout = nn.Dropout(dropout, inplace=False)
            self.actvn = get_activation(_to_get_activation_name(activation))
            self.forward = self.forward_torch

            if scale_init_out:
                nn.init.normal_(self.c_proj.weight, std=0.02 / math.sqrt(2 * scale_init_out))
        else:
            raise ValueError(f"Unknown backend {backend}")

    def forward_torch(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.actvn(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def forward_tcnn(self, x: Tensor) -> Tensor:
        return self.net(x.view(-1, x.shape[-1])).view(*x.shape[:-1], -1).to(dtype=x.dtype, device=x.device)


class LambdaLayer(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn
        self.name = self.__class__.__name__
        if hasattr(fn, "__name__") and fn.__name__:
            self.name = fn.__name__

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class MixerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        seq_len: int,
        n_out: int | None = None,
        n_hidden: int = 128,
        bias: bool = False,
        dropout: float = 0.0,
        hidden_layer_multiplier: int = 4,
        activation: ActivationName = "gelu",
        norm: NormName = "layer",
    ):
        super().__init__()
        self.ln_1 = get_norm(norm, n_embd, bias=bias)
        self.ln_2 = get_norm(norm, n_embd, bias=bias)
        self.token_mixing = MLP(n_embd=seq_len, n_hidden=n_hidden, bias=bias, dropout=dropout, activation=activation)
        self.channel_mixing = MLP(
            n_embd=n_embd,
            n_out=n_out,
            bias=bias,
            dropout=dropout,
            hidden_layer_multiplier=hidden_layer_multiplier,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.ln_1(x)
        y = self.token_mixing(y.transpose(1, 2)).transpose(1, 2)
        x = x + y
        y = self.ln_2(x)
        return x + self.channel_mixing(y)


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_out: int | None = None,
        n_hidden: int | None = None,
        bias: bool | tuple[bool, bool, bool] = False,
        dropout: float | tuple[float, float, float] = 0.0,
        drop_path: float | tuple[float, float, float] = 0.0,
        causal: bool | tuple[bool, bool] = False,
        hidden_layer_multiplier: int = 4,
        self_attn: bool = False,
        cross_attn: bool = False,
        projection: Literal["mlp", "linear"] | None = None,
        activation: ActivationName = "gelu",
        norm: NormName | None = "layer",
        chunks: int = 1,
        scale_init_out: int | tuple[int | None, int | None, int | None] | list[int | None] | None = None,
        **kwargs,
    ):
        super().__init__()
        bias_values = _as_triple_bool(bias)
        dropout_values = _as_triple_float(dropout)
        drop_path_values = _as_triple_float(drop_path)
        causal_values = _as_causal_pair(causal)
        scale_init_values = _as_scale_init_triple(scale_init_out)

        self.ln_1 = get_norm(norm, n_embd, bias=bias_values[0]) if norm and self_attn else nn.Identity()
        self.ln_2 = get_norm(norm, n_embd, bias=bias_values[1]) if norm and cross_attn else nn.Identity()
        self.ln_3 = get_norm(norm, n_embd, bias=bias_values[2]) if norm and projection else nn.Identity()

        self.cross_attn = (
            MultiHeadAttention(
                n_embd=n_embd,
                n_head=n_head,
                bias=bias_values[1],
                dropout=dropout_values[1],
                causal=causal_values[1],
                # Cross-attention needs separate q/k(v) projections by default.
                k_dim=kwargs.pop("k_dim", n_embd if chunks in [1, 2] else None),
                v_dim=kwargs.pop("v_dim", n_embd if chunks in [1, 2] else None),
                chunk=chunks < 3,
                scale_init_out=scale_init_values[1],
                **kwargs,
            )
            if cross_attn
            else None
        )

        self.self_attn = (
            MultiHeadAttention(
                n_embd=n_embd,
                n_head=n_head,
                bias=bias_values[0],
                dropout=dropout_values[0],
                causal=causal_values[0],
                k_dim=n_embd if chunks > 1 else None,
                v_dim=n_embd if chunks > 1 else None,
                chunk=chunks < 3,
                scale_init_out=scale_init_values[0],
                **kwargs,
            )
            if self_attn
            else None
        )

        self.projection = None
        if projection:
            if projection == "mlp":
                self.projection = MLP(
                    n_embd=n_embd,
                    n_out=n_out,
                    n_hidden=n_hidden,
                    bias=bias_values[2],
                    dropout=dropout_values[2],
                    hidden_layer_multiplier=hidden_layer_multiplier,
                    activation=activation,
                    scale_init_out=scale_init_values[2],
                )
            elif projection == "linear":
                self.projection = nn.Linear(n_embd, n_out or n_embd, bias=bias_values[2])
            else:
                raise ValueError(f"Unknown projection type {projection}")

        self.dp_1 = DropPath(drop_path_values[0]) if drop_path_values[0] and self_attn else nn.Identity()
        self.dp_2 = DropPath(drop_path_values[1]) if drop_path_values[1] and cross_attn else nn.Identity()
        self.dp_3 = DropPath(drop_path_values[2]) if drop_path_values[2] and projection else nn.Identity()

    def forward(self, x: Tensor, mem: Tensor | None = None) -> Tensor:
        if self.self_attn is not None:
            x = x + self.dp_1(self.self_attn(self.ln_1(x)))
        if self.cross_attn is not None:
            x = x + self.dp_2(self.cross_attn(self.ln_2(x), mem))
        if self.projection is not None:
            x = x + self.dp_3(self.projection(self.ln_3(x)))
        return x


class EncoderBlock(Block):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_out: int | None = None,
        n_hidden: int | None = None,
        bias: bool | tuple[bool, bool, bool] = False,
        dropout: float | tuple[float, float, float] = 0.0,
        drop_path: float | tuple[float, float, float] = 0.0,
        causal: bool = False,
        hidden_layer_multiplier: int = 4,
        no_self_attn: bool = False,
        activation: ActivationName = "gelu",
        norm: NormName | None = "layer",
        **kwargs,
    ):
        super().__init__(
            n_embd=n_embd,
            n_head=n_head,
            n_out=n_out,
            n_hidden=n_hidden,
            bias=bias,
            dropout=dropout,
            drop_path=drop_path,
            causal=causal,
            hidden_layer_multiplier=hidden_layer_multiplier,
            self_attn=not no_self_attn,
            cross_attn=False,
            projection="mlp",
            activation=activation,
            norm=norm,
            **kwargs,
        )


class DecoderBlock(Block):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_out: int | None = None,
        n_hidden: int | None = None,
        bias: bool | tuple[bool, bool, bool] = False,
        dropout: float | tuple[float, float, float] = 0.0,
        drop_path: float | tuple[float, float, float] = 0.0,
        causal: bool | tuple[bool, bool] = False,
        hidden_layer_multiplier: int = 4,
        no_self_attn: bool = False,
        no_projection: bool = False,
        activation: ActivationName = "gelu",
        norm: NormName | None = "layer",
        **kwargs,
    ):
        super().__init__(
            n_embd=n_embd,
            n_head=n_head,
            n_out=n_out,
            n_hidden=n_hidden,
            bias=bias,
            dropout=dropout,
            drop_path=drop_path,
            causal=causal,
            hidden_layer_multiplier=hidden_layer_multiplier,
            self_attn=not no_self_attn,
            cross_attn=True,
            projection=None if no_projection else "mlp",
            activation=activation,
            norm=norm,
            **kwargs,
        )


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer: int,
        n_embd: int,
        n_head: int,
        bias: bool | tuple[bool, bool, bool] = False,
        dropout: float | tuple[float, float, float] = 0.0,
        drop_path: float | tuple[float, float, float] = 0.0,
        drop_path_rate: float | None = None,
        causal: bool = False,
        hidden_layer_multiplier: int = 4,
        no_self_attn: bool = False,
        activation: ActivationName = "gelu",
        norm: NormName | None = "layer",
        **kwargs,
    ):
        super().__init__()
        if drop_path_rate is not None:
            dpr = torch.linspace(0.0, float(drop_path_rate), steps=n_layer).tolist()
        else:
            dpr = [float(drop_path) if isinstance(drop_path, (int, float)) else float(drop_path[0])] * n_layer

        self.layers = nn.Sequential(
            *[
                EncoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    bias=bias,
                    dropout=dropout,
                    drop_path=(dpr[i], dpr[i], dpr[i]),
                    causal=causal,
                    hidden_layer_multiplier=hidden_layer_multiplier,
                    no_self_attn=no_self_attn,
                    activation=activation,
                    norm=norm,
                    **kwargs,
                )
                for i in range(n_layer)
            ]
        )
        self.forward = self.layers.forward


class Decoder(nn.Module):
    def __init__(
        self,
        n_layer: int,
        n_embd: int,
        n_head: int,
        bias: bool | tuple[bool, bool, bool] = False,
        dropout: float | tuple[float, float, float] = 0.0,
        drop_path: float | tuple[float, float, float] = 0.0,
        drop_path_rate: float | None = None,
        causal: bool | tuple[bool, bool] = False,
        hidden_layer_multiplier: int = 4,
        no_self_attn: bool = False,
        activation: ActivationName = "gelu",
        norm: NormName | None = "layer",
        **kwargs,
    ):
        super().__init__()
        if drop_path_rate is not None:
            dpr = torch.linspace(0.0, float(drop_path_rate), steps=n_layer).tolist()
        else:
            dpr = [float(drop_path) if isinstance(drop_path, (int, float)) else float(drop_path[0])] * n_layer

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    bias=bias,
                    dropout=dropout,
                    drop_path=(dpr[i], dpr[i], dpr[i]),
                    causal=causal,
                    hidden_layer_multiplier=hidden_layer_multiplier,
                    no_self_attn=no_self_attn,
                    activation=activation,
                    norm=norm,
                    **kwargs,
                )
                for i in range(n_layer)
            ]
        )

    def forward(self, x: Tensor, mem: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mem)
        return x


class PositionalEncoding(nn.Module):
    """
    [1] https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
    [2] https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    [3] http://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding
    [4] https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, dim: int, max_len: int, theta: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(theta) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        pe = cast(Tensor, self.pe)
        return x + pe[:, : x.size(1)]


# From https://docs.nerf.studio/_modules/nerfstudio/field_components/encodings.html#NeRFEncoding


def expected_sin(x_means: Tensor, x_vars: Tensor) -> Tensor:
    """Computes the expected value of sin(y) where y ~ N(x_means, x_vars)

    Args:
        x_means: Mean values.
        x_vars: Variance of values.

    Returns:
        torch.Tensor: The expected value of sin.
    """
    return torch.exp(-0.5 * x_vars) * torch.sin(x_means)


class FieldComponent(nn.Module):
    """Field modules that can be combined to store and compute the fields.

    Args:
        in_dim: Input dimension to module.
        out_dim: Output dimension to module.
    """

    def __init__(self, in_dim: int | None = None, out_dim: int | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build_nn_modules(self):
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int):
        """Sets input dimension of encoding

        Args:
            in_dim: input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_enc_dim(self) -> int:
        """Calculates output dimension of encoding."""
        if self.out_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.out_dim

    @abstractmethod
    def forward(self, in_tensor: Tensor) -> Tensor:
        """
        Returns processed tensor

        Args:
            in_tensor: Input tensor to process
        """
        raise NotImplementedError


class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int):
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @classmethod
    def get_tcnn_encoding_config(cls, **kwargs) -> dict:
        """Get the encoding configuration for tcnn if implemented"""
        raise NotImplementedError("Encoding does not have a TCNN implementation")

    @abstractmethod
    def forward(self, in_tensor: Tensor) -> Tensor:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class NeRFEncoding(Encoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_inputs: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int | None = None,
        hidden_dim: int | None = None,
        num_frequencies: int = 6,
        min_freq_exp: float = 0,
        max_freq_exp: float = 5,
        include_inputs: bool = True,
        normalize_inputs: bool = True,
        scale_inputs: bool = True,
        padding: float = 0.1,
        implementation: Literal["tcnn", "torch"] = "tcnn" if TCNN_EXISTS else "torch",
    ):
        super().__init__(in_dim)
        assert not (scale_inputs and not normalize_inputs), "Inputs need to be normalized for scaling"

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_inputs = include_inputs
        self.normalize_inputs = normalize_inputs
        self.scale_inputs = scale_inputs
        self.padding = padding

        self.mlp = None
        self.tcnn_encoding = None
        if implementation == "tcnn":
            assert min_freq_exp == 0, "tcnn only supports min_freq_exp = 0"
            assert max_freq_exp == num_frequencies - 1, "tcnn only supports max_freq_exp = num_frequencies - 1"
            config = {"otype": "Frequency", "n_frequencies": num_frequencies}
            self.tcnn_encoding = tcnn.Encoding(n_input_dims=in_dim, encoding_config=config)

        if out_dim is not None:
            if hidden_dim is not None:
                self.mlp = MLP(
                    n_embd=self.get_enc_dim(),
                    n_out=out_dim,
                    n_hidden=hidden_dim,
                    activation="leakyrelu",
                    backend=implementation,
                )
            else:
                self.mlp = nn.Linear(self.get_enc_dim(), out_dim)

    def get_enc_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_inputs:
            out_dim += self.in_dim
        return out_dim

    def _torch(self, in_tensor: Tensor, covs: Tensor | None = None) -> Tensor:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        in_tensor = torch.pi * in_tensor
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies, device=in_tensor.device)
        scaled_inputs = in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        return encoded_inputs

    def _tcnn(self, x: Tensor) -> Tensor:
        if self.tcnn_encoding is None:
            raise RuntimeError("TCNN encoding backend is unavailable")
        return self.tcnn_encoding(x.reshape(-1, x.size(-1))).view(*x.shape[:-1], -1).to(x)

    def forward(self, in_tensor: Tensor, covs: Tensor | None = None) -> Tensor:
        if in_tensor.ndim not in [2, 3] or in_tensor.shape[-1] != self.in_dim:
            raise ValueError(f"Input tensor should have shape (B, (C), {self.in_dim}) not {in_tensor.shape}")
        x = in_tensor
        if self.normalize_inputs:
            x = cast(Tensor, points_to_coordinates(in_tensor, max_value=1 + self.padding))  # scale to [0, 1]

        if self.tcnn_encoding is None:
            x = self._torch(x, covs)
        else:
            assert covs is None, "tcnn does not support covariances"
            x = self._tcnn(x)

        if self.include_inputs:
            # inputs = 2 * (in_tensor / (1 + self.padding)) if self.scale_inputs else in_tensor  # scale to [-1, 1]
            inputs = 2 * in_tensor if self.scale_inputs else in_tensor  # scale to [-1, 1]
            x = torch.cat([inputs, x], dim=-1)

        if self.mlp is None:
            return x
        return self.mlp(x)


class HashGridEncoding(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.5,
        normalize_inputs: bool = True,
    ):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.encoding = tcnn.Encoding(
            n_input_dims=in_dim,
            encoding_config=dict(
                otype="HashGrid",
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                per_level_scale=per_level_scale,
                interpolation="Linear",
            ),
        )
        raw_dim = n_levels * n_features_per_level
        self.proj = nn.Linear(raw_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.normalize_inputs:
            x = (x + 1) / 2  # scale from [-1, 1] to [0, 1]
        B, N, D = x.size()
        x_flat = x.reshape(-1, D).contiguous()
        enc = self.encoding(x_flat)  # (B*N, raw_dim)
        enc = self.proj(enc)
        return enc.view(B, N, -1)
