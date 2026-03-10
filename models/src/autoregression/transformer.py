import math
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
import torch.distributions as dist
from torch import Tensor, nn
from tqdm import trange

from utils import setup_logger

from ..transformer import Decoder, LambdaLayer, PositionalEncoding
from ..utils import DropPath, get_norm

logger = setup_logger(__name__)


def _debug_level_2(message: str) -> None:
    debug_level_2 = getattr(cast(Any, logger), "debug_level_2", None)
    if callable(debug_level_2):
        debug_level_2(message)
        return
    logger.debug(message)


class GPT(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_block: int,
        n_layer: int = 12,
        n_embd: int = 768,
        n_head: int = 12,
        bias: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        cross_attn: bool = False,
        activation: Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu"] = "gelu",
        norm: Literal["batch", "instance", "layer", "group"] | None = "layer",
        pos_enc: Literal["learned", "freq"] | None = "learned",
        voc_enc: bool | Callable = True,
        bos_embd: bool = True,
    ):
        super().__init__()
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self.n_vocab = int(n_vocab)
        self.n_block = int(n_block)
        self.n_embd = int(n_embd)
        self.layer = Decoder(
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            drop_path=drop_path,
            causal=(True, False) if cross_attn else True,
            no_self_attn=not cross_attn,
            activation=activation,
            norm=norm,
        )
        self.pos_enc: Callable[[Tensor], Tensor] | None = None
        if pos_enc:
            if pos_enc == "learned":
                self._pos_enc = nn.Embedding(n_block, n_embd)
                self.pos_enc = lambda x: x + self._pos_enc.weight[: x.size(1)].expand_as(x)
            elif "freq" in pos_enc:
                self.pos_enc = PositionalEncoding(n_embd, n_block)
            else:
                raise ValueError(f"pos_enc: {pos_enc} not supported")

        self.drop = DropPath(drop_path) if drop_path else nn.Dropout(dropout, inplace=False)
        self.ln = get_norm(norm, n_embd) if norm else nn.Identity()
        self.pred = nn.Linear(n_embd, n_vocab, bias=False)
        # nn.init.zeros_(self.pred.weight)

        self.voc_enc: nn.Module = nn.Identity()
        if voc_enc:
            if isinstance(voc_enc, nn.Module):
                self.voc_enc = voc_enc
            elif callable(voc_enc):
                self.voc_enc = LambdaLayer(voc_enc)
            else:
                self.voc_enc = nn.Embedding(n_vocab, n_embd)
                self.voc_enc.weight = self.pred.weight

        self.bos_embd: nn.Embedding | None = None
        if bos_embd:
            self.bos_embd = nn.Embedding(1, n_embd)

    def setup(self, *args, **kwargs):
        num_params = sum(p.numel() for p in self.parameters())
        if isinstance(self.pos_enc, nn.Parameter):
            num_params -= self.pos_enc.numel()
        _debug_level_2(f"Model {self.__class__.__name__} has {num_params:,} parameters")

    def forward(
        self, x: Tensor | None = None, c: Tensor | None = None, batch_size: int | None = None, **kwargs
    ) -> Tensor:
        if x is None:
            if self.bos_embd is None and c is not None:
                x = c
            elif self.bos_embd is not None:
                x = self.bos_embd.weight.expand(batch_size or 1, -1, -1)  # initialize with BOS token (B, 1, n_embd)
            else:
                raise ValueError("x must be provided if no BOS token is used")

        if x.ndim == 2 or x.size(2) != self.n_embd:
            x = self.voc_enc(x)
        x = cast(Tensor, x)
        if self.bos_embd is None and c is not None:
            x_shifted = x[:, :-1] if x.size(1) == self.n_block else x
            x = torch.cat((c, x_shifted), dim=1)
        elif self.bos_embd is not None:
            bos_embd = self.bos_embd.weight.expand_as(x[:, :1])  # (1, n_embd) -> (B, 1, n_embd)
            x_shifted = x[:, :-1] if x.size(1) == self.n_block else x
            x = torch.cat((bos_embd, x_shifted), dim=1)

        if self.pos_enc is not None:
            x = self.pos_enc(x)

        x = self.drop(x)
        x = cast(Tensor, self.layer(x, None if self.bos_embd is None else c))
        x = cast(Tensor, self.ln(x))

        if x.size(1) == self.n_block:
            return self.pred(x)
        elif x.size(1) > self.n_block:
            return self.pred(x[:, -self.n_block :])
        return self.pred(x[:, [-1]])

    @torch.inference_mode()
    def generate(
        self,
        x: Tensor | None = None,
        c: Tensor | None = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        topk: int | bool | None = None,
        progress: bool = True,
        return_intermediates: bool = False,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        intermediates = [] if return_intermediates else None
        progress_bar = trange(
            max_new_tokens or self.n_block,
            desc=f"Autoregressive sampling (temp={temperature})",
            leave=False,
            disable=not progress,
        )
        for step in progress_bar:
            x_cond = x
            if x is not None:
                # if the sequence context is growing too long we must crop it at block_size
                x_cond = x if x.size(1) <= self.n_block else x[:, -self.n_block :]

            # forward the model to get the logits for the index in the sequence
            logits = self(x_cond, c, batch_size=batch_size)
            # pluck the logits at the final step
            logits = logits[:, [-1]]  # (B, 1, n_vocab)
            # optionally crop the logits to only the top k options
            if topk:
                tk = (max_new_tokens or self.n_block) - step if isinstance(topk, bool) else topk
                progress_bar.set_description(f"Autoregressive sampling (temp={temperature}, topk={tk})")
                v, _ = torch.topk(logits, k=min(tk, logits.size(2)), dim=2)
                logits[logits < v[..., [-1]]] = -float("inf")

            if isinstance(self.voc_enc, nn.Linear):
                mean, logstd = logits.chunk(2, dim=2)
                scaled_logstd = logstd + 0.5 * math.log(temperature)
                x_next = dist.Normal(mean, scaled_logstd.exp()).sample()
            else:
                # Scale by desired temperature and sample from the distribution
                x_next = dist.Categorical(logits=logits / temperature).sample()

            # initialize the sequence if not provided or append sampled index to the running sequence and continue
            x = x_next if x is None else torch.cat((x, x_next), dim=1)

            if intermediates is not None:
                intermediates.append(x.clone())

        if intermediates is not None:
            return x, intermediates
        return x


class LatentGPT(GPT):
    def __init__(
        self,
        n_vocab: int,
        n_block: int,
        n_latent: int | None = None,
        n_layer: int = 24,
        n_embd: int = 512,
        n_head: int = 8,
        cond: bool | int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        activation: Literal["relu", "leaky", "elu", "gelu", "new_gelu", "geglu"] = "geglu",
        norm: Literal["batch", "instance", "layer", "group"] | None = "layer",
        pos_enc: Literal["learned", "freq"] | None = None,
        voc_enc: bool | Callable = False,
        bos_embd: bool = True,
    ):
        super().__init__(
            n_vocab=n_vocab,
            n_block=n_block,
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            drop_path=drop_path,
            cross_attn=bool(cond and bos_embd),
            activation=activation,
            norm=norm,
            pos_enc=pos_enc,
            voc_enc=voc_enc,
            bos_embd=bos_embd,
        )
        self.n_latent = n_latent
        self.category_enc: nn.Embedding | None = None
        if n_latent:
            if callable(voc_enc):
                self.voc_enc = nn.Sequential(LambdaLayer(voc_enc), nn.Linear(n_latent, n_embd))
            elif voc_enc:
                self.voc_enc = nn.Sequential(nn.Linear(n_latent, n_embd), self.voc_enc)
            else:
                self.voc_enc = nn.Linear(n_latent, n_embd)
        if isinstance(cond, int) and not isinstance(cond, bool):
            self.category_enc = nn.Embedding(cond, n_embd)

    def forward(self, x: Tensor | None = None, cond: Tensor | None = None, **kwargs) -> Tensor:
        cond_embd = cond
        if cond is not None and not torch.is_floating_point(cond):
            if self.category_enc is None:
                raise ValueError("`cond` provided as categorical indices but `category_enc` is not initialized")
            cond_embd = self.category_enc(cond).unsqueeze(1)
        return super().forward(x, cond_embd, **kwargs)
