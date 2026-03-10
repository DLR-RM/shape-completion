import inspect
import math
import os
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.special import log_softmax
from torch import Tensor, nn
from tqdm.auto import tqdm

from utils import make_3d_grid, resolve_path

from .transformer import EncoderBlock, PyTorchEncoder
from .vqdif import DEFAULT_KWARGS as VQDIF_DEFAULT_KWARGS
from .vqdif import VQDIF

DEFAULT_KWARGS = {
    "voxel_res": 16,
    "end_tokens": [4096, 4096],
    "vocab_sizes": [4097, 4097],
    "extra_vocab_sizes": [4097],
    "block_size": 812,
    "tuple_n": 2,
    "top_k": 100,
    "top_p": 0.4,
    "sample_n": 4,
    "sample_max_step": 512,
    "temperature": 1,
    "representer_opt": {
        "voxel_res": 16,
        "block_size": 812,
        "end_tokens": [4096, 4096],
        "random_cind_masking": True,
        "mask_invalid": True,
        "mask_invalid_completion": True,
        "vqdif_opt": {"weights_path": "out/vqdif/shapeformer/vqdif/model_best.pt"},
    },
    "transformer_opt": {
        "tuple_n": 2,
        "vocab_sizes": [4097, 4097],
        "extra_vocab_sizes": [4097],
        "n_layers": [20, 4],
        "block_size": 812,
        "n_head": 16,
        "n_embd": 1024,
        "attn_pdrop": 0.01,
        "resid_pdrop": 0.01,
        "embd_pdrop": 0.01,
    },
}


class CondTupleGPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        vocab_sizes: list[int],
        extra_vocab_sizes: list[int],
        block_size: int,
        tuple_n: int,
        n_layers: tuple[int, ...] | list[int] = (20, 4),
        n_head: int = 16,
        n_embd: int = 1024,
        embd_pdrop: float = 0.01,
        resid_pdrop: float = 0.01,
        attn_pdrop: float = 0.01,
        bias: bool = True,
        implementation: str = "native",
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())
        tuple_n_int = int(tuple_n)
        self.tuple_n = tuple_n_int
        self.tok_embs, self.extra_tok_embs = nn.ModuleList([]), nn.ModuleList([])
        self.drops, self.blocks, self.heads = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
        assert tuple_n_int == len(vocab_sizes)
        assert tuple_n_int == len(n_layers)
        self.extra_tuple_n = len(extra_vocab_sizes)
        for i in range(tuple_n_int):
            vocab_size = vocab_sizes[i]
            n_layer = n_layers[i]
            # input embedding stem
            self.tok_embs.append(nn.Embedding(vocab_size, n_embd))
            self.drops.append(nn.Dropout(embd_pdrop))

            # transformer
            if implementation == "native":
                assert attn_pdrop == resid_pdrop, "Native implementation does not support different dropout rates"
                self.blocks.append(
                    nn.Sequential(
                        *[
                            EncoderBlock(
                                n_embd=n_embd,
                                n_head=n_head,
                                bias=bias,
                                dropout=attn_pdrop,
                                causal=True,
                            )
                            for _ in range(n_layer)
                        ]
                    )
                )
            elif implementation == "torch":
                assert attn_pdrop == resid_pdrop, "PyTorch implementation does not support different dropout rates"
                assert bias, "PyTorch implementation does not support bias=False"
                self.blocks.append(PyTorchEncoder(n_layer, n_embd, n_head, bias, dropout=attn_pdrop, causal=True))
            else:
                raise NotImplementedError(f"Unknown implementation {implementation}")

            # decoder heads
            self.heads.append(nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, vocab_size, bias=False)))

        for i in range(self.extra_tuple_n):
            vocab_size = extra_vocab_sizes[i]
            self.extra_tok_embs.append(nn.Embedding(vocab_size, n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.block_size = block_size

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for n_layer, blocks in zip(n_layers, self.blocks, strict=False):
            for pn, p in blocks.named_parameters():
                if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_pos_embeddings(self, L_cond, L_gen):
        cond_position_embeddings = self.cond_pos_emb[:, :L_cond, :]
        shape_position_embeddings = self.pos_emb[:, :L_gen, :]
        pos_embd = torch.cat([cond_position_embeddings, shape_position_embeddings], dim=1)
        return pos_embd

    @staticmethod
    def get_token_embeddings(tok_embs, idx):
        token_embeddings = 0
        for i, tok_emb in enumerate(tok_embs):
            token_embeddings += tok_emb(idx[..., i])
        return token_embeddings

    def get_embeddings(self, idx, extra_idx, L_c):
        """idx: (B, L, tuple_n) cond_idx: idx[:,:L_cond,:]
        extra_idx: (B, L, extra_tuple_n)
        """
        assert idx.size(-1) == self.tuple_n, f"{idx.size(-1)}!={self.tuple_n}"
        assert extra_idx.size(-1) == self.extra_tuple_n, f"{extra_idx.size(-1)}!={self.extra_tuple_n}"
        L = idx.size(1)  # L_c + L_z
        L_z = L - L_c
        assert L <= self.block_size, "Cannot forward, model block size is exhausted"
        # each index maps to a (learnable) vector
        token_embeddings = self.get_token_embeddings(self.tok_embs, idx)
        extra_token_embeddings = self.get_token_embeddings(self.extra_tok_embs, extra_idx)
        pos_embd = self.get_pos_embeddings(L_c, L_z)
        # (B, L, embd_dim)
        x = token_embeddings + extra_token_embeddings + pos_embd
        return x

    def compute_logits(self, x, targets):
        """x: embeddings,  (B, L, embd_dim)
        targets:        (B, L, tuple_n)"""
        logits = []  # list of logits
        for i in range(self.tuple_n):
            x = self.blocks[i](self.drops[i](x))
            # (B, L, vocab_size) <- (B, L, embd_dim)
            logits.append(self.heads[i](x))
            x = x + self.tok_embs[i](targets[..., i])  # targets = idx shifted to left
        return logits

    @torch.no_grad()
    def sample_next_tuple(self, idx, extra_idx=None, L_cond=1):
        # x: (B, L, embd_dim) <- (B, L_cond, tuple_n), (B, L_gen, tuple_n)
        x = self.get_embeddings(idx, extra_idx, L_cond)
        logits = list()
        for i in range(self.tuple_n):
            x = self.blocks[i](self.drops[i](x))
            logits.append(self.heads[i](x))

            if i == self.tuple_n - 1:
                break

            target_i = yield logits[-1]
            # Add coordinate transformer features
            x = x + self.tok_embs[i](target_i)
        return logits

    def forward(self, idx, extra_idx=None, L_cond=1, target_idx=None):
        """idx: (B, L, tuple_n) cond_idx: idx[:,:L_cond,:]
        extra_idx: (B, L, extra_tuple_n)
        target_idx: (B, L, tuple_n)
        """
        x = self.get_embeddings(idx, extra_idx, L_cond)
        logits = self.compute_logits(x, target_idx)
        return logits


class ShapeRepresenter(nn.Module):
    def __init__(
        self,
        voxel_res: int = 16,
        end_tokens: tuple[int, int] | None = None,
        input_end_tokens: tuple[int, int] | None = None,
        block_size: int | None = None,
        vqdif_opt: dict[str, Any] | None = None,
        random_cind_masking: bool = True,
        mask_invalid: bool = True,
        mask_invalid_completion: bool = True,
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())
        if end_tokens is None:
            msg = "end_tokens must be provided"
            raise ValueError(msg)
        if block_size is None:
            msg = "block_size must be provided"
            raise ValueError(msg)

        self.end_tokens = end_tokens
        if input_end_tokens is None:
            self.input_end_tokens: tuple[int, int] = self.end_tokens
        else:
            self.input_end_tokens = input_end_tokens
        self.block_size = block_size
        self.max_length: int = self.block_size // 2

        self.vqdif = VQDIF(**VQDIF_DEFAULT_KWARGS)

        vqdif_opt = {} if vqdif_opt is None else vqdif_opt
        vqdif_weights_path = vqdif_opt.get("weights_path")
        if vqdif_weights_path is not None:
            vqdif_weights_path = resolve_path(vqdif_weights_path)
            if os.path.isfile(vqdif_weights_path):
                print(f"loading pre-trained VQDIF weights from {vqdif_weights_path}")
                self.vqdif.load_state_dict(torch.load(vqdif_weights_path, weights_only=False)["model"])
            else:
                print(f"vqdif weights not found at {vqdif_weights_path}")

        self.vqdif.eval()
        cast(Any, self.vqdif).train = lambda mode=True: self.vqdif

        for p in self.vqdif.parameters():
            p.requires_grad = False

    @staticmethod
    def ravel_index(index: Tensor, shape: tuple[int, ...]) -> Tensor:
        """Return a flattened version of the input index tensor in row-major (C-style) order.
        (row major, or raster scan order, where the image representation is unrolled from top left to bottom right)

        Parameters
        ----------
        index : Tensor
            A tensor of indices of shape (batch size, ..., num_indices) where each entry is an index into a tensor of
            shape (batch size, resolution, resolution, resolution)
        shape : Tuple[int, int, int]
            A tuple of three integers representing the shape of the tensor being indexed into

        Returns
        -------
        Tensor
            A flattened version of the input index tensor that can be used to index into a tensor of shape
            (batch size, resolution, resolution, resolution)
        """
        if index.size(-1) == 2:
            raveled = index[..., 0] * shape[1] + index[..., 1]
        elif index.size(-1) == 3:
            raveled = index[..., 0] * shape[1] * shape[2] + index[..., 1] * shape[2] + index[..., 2]
        else:
            raise ValueError("shape must be 2 or 3 dimensional")
        return raveled

    @staticmethod
    def unravel_index(index: Tensor, shape: tuple[int, ...]) -> Tensor:
        unraveled = torch.zeros(*index.shape, len(shape)).type_as(index)
        if len(shape) == 2:
            unraveled[..., 0] = index // shape[1]
            unraveled[..., 1] = index % shape[1]
        elif len(shape) == 3:
            s12 = shape[1] * shape[2]
            unraveled[..., 0] = index // s12
            unraveled[..., 1] = index % s12 // shape[2]
            unraveled[..., 2] = index % s12 % shape[2]
        else:
            raise ValueError("shape must be 2 or 3 dimensional")
        return unraveled

    @staticmethod
    def unpack_sparse(sparse: Tensor, max_length: int | None = None, end_tokens: tuple[int, int] = (100, 200)):
        """Converts a sparse tensor to a dense tensor by unpacking it.

        Parameters
        ----------
        sparse : Tensor
            A sparse tensor of shape (N, 3) where N is the total number of non-zero entries. The first column contains
            the batch indices, the second column contains the raveled indices, and the third column contains the values.
        max_length : int, optional
            The maximum length of the sequence. If the unpacked tensor exceeds this length, it will be truncated to the
            maximum length, by default None
        end_tokens : Tuple[int, int], optional
            A tuple of two integers representing the end tokens used to indicate the end of a sequence.
            The default is (100, 200).

        Returns
        -------
        Tensor
            A dense tensor of shape (B, K, 2) where B is the number of batches and K is the length of the longest
            sequence. The first column of the tensor contains the raveled indices and the second column contains the
            values.
        """

        # Get the batch indices, raveled indices, and values from the sparse tensor
        batch_ind = sparse[:, 0]
        raveled_ind = sparse[:, 1]
        val = sparse[:, 2]

        # Eliminate consecutive duplicates in the batch indices and count remaining occurrences
        # Example: [0, 0, 0, 1, 1, 2, 2, 2, 2] -> [0, 1, 2], [3, 2, 4]
        _, counts = torch.unique_consecutive(batch_ind, return_counts=True)

        # Repeat the counts to match the size of the sparse tensor
        # Example: [3, 2, 4] -> [3, 3, 3, 2, 2, 4, 4, 4, 4]
        repeated_counts = counts.repeat_interleave(counts)

        # Compute the cumulative sum of the counts and repeat it to match the size of the sparse tensor
        # Example: [3, 2, 4] -> [3, 5, 9] -> [3, 3, 3, 5, 5, 9, 9, 9, 9]
        repeated_cum = torch.cumsum(counts, dim=0).repeat_interleave(counts)

        # Compute the indices relative to the repeated cumulative sum and the repeated counts
        # Example: [3, 3, 3, 5, 5, 9, 9, 9, 9] -> [0, 1, 2, 0, 1, 0, 1, 2, 3]
        arange = torch.arange(len(repeated_cum)).type_as(sparse)
        repeated_arange = arange - (repeated_cum - repeated_counts)

        # Create the end tokens tensor
        end_tokens_tensor = torch.tensor(end_tokens, device=sparse.device, dtype=sparse.dtype)[None, None, :]

        # Get the number of tokens in the end tokens tensor (i.e 2)
        tuple_n = end_tokens_tensor.size(-1)

        # Create the target tensor with the end tokens
        target = end_tokens_tensor + torch.zeros(
            len(counts),
            int(counts.max().item()) + 1,
            tuple_n,
            device=sparse.device,
            dtype=sparse.dtype,
        )

        # Fill the target tensor with the values from the sparse tensor
        target[batch_ind, repeated_arange, 0] = raveled_ind
        target[batch_ind, repeated_arange, 1] = val

        # Truncate the target tensor if necessary
        if max_length is not None and target.size(1) > max_length:
            target = target[:, :max_length, :]
            target[:, max_length - 1, :] = end_tokens_tensor[:, 0, :]

        # Return the target tensor
        return target

    def batch_dense2sparse(
        self,
        dense_indices: Tensor,
        unpack: bool = True,
        max_length: int | None = None,
        end_tokens: tuple[int, int] = (100, 200),
    ) -> tuple[Tensor, Tensor]:
        """Extracts grid cell and codebook indices.

        Parameters
        ----------
        dense_indices : Codebook index grid (B, R, R, R)
        unpack : Convert sequence of batched grid and codebook indices to batched sequences of grid and codebook indices
        max_length : Maximum length of the sequence
        end_tokens : End token embedding index used to indicate the end of a sequence

        B: batch size
        D: depth
        H: height
        W: width
        L: sequence length (flattened non-empty grid cells)
        N: flattened non-empty grid cell positions (grid indices)
        V: flattened non-empty grid cell values (codebook indices)
        """

        # Find the mode of the tensor
        mode = dense_indices.view(-1).mode()[0]

        # Get the sparse grid indices (where dense indices are not equal to the mode, i.e. non-empty)
        sparse_indices = (dense_indices != mode).nonzero(as_tuple=False)  # (L, 4) indexing into B, D, H, W

        # Get the batch indices and ravelled indices in row-major (C-style) order
        batch_indices = sparse_indices[:, 0]  # i.e. which batch the non-empty grid cell belongs to
        dense_shape = tuple(int(dim) for dim in dense_indices.shape[1:])
        raveled_indices = self.ravel_index(sparse_indices[:, 1:], shape=dense_shape)

        # Get the codebook indices
        values = dense_indices[
            sparse_indices[:, 0], sparse_indices[:, 1], sparse_indices[:, 2], sparse_indices[:, 3]
        ].reshape(-1)

        # Pack the sparse tensor
        packed_sparse = torch.stack((batch_indices, raveled_indices, values), dim=-1)  # (L, 3) indexing into B, N, V

        if unpack:
            # (B, L, 2) indexing into N, V
            unpacked = self.unpack_sparse(packed_sparse, max_length=max_length, end_tokens=end_tokens)
            return unpacked, mode
        else:
            return packed_sparse, mode

    def batch_sparse2dense(
        self, sparse_indices: Tensor, empty_ind: Tensor, dense_res: int, return_flattened: bool = False, dim: int = 3
    ):
        # sparse: (K, 3) should be packed
        batch_ind = sparse_indices[:, 0]
        raveled_ind = sparse_indices[:, 1]
        val = sparse_indices[:, 2]
        unique_batch_ind, _counts = torch.unique_consecutive(batch_ind, return_counts=True)
        batch_size = len(unique_batch_ind)

        dense_shape = tuple([dense_res] * dim)
        dense = empty_ind + torch.zeros(
            batch_size,
            *dense_shape,
            device=sparse_indices.device,
            dtype=sparse_indices.dtype,
        )
        # (K, dim)
        grid_ind = self.unravel_index(raveled_ind, shape=dense_shape)
        # fill in the values
        dense[(batch_ind[:, None], *grid_ind.split(1, dim=-1))] = val[:, None]
        if return_flattened:
            return dense.reshape(batch_size, -1)
        return dense

    def encode_cloud(self, cloud: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encodes a pointcloud using the vqdif model.

        Parameters
        ----------
        cloud : Pointcloud to encode (B, N, 3)

        Returns
        -------
        quant_ind: Codebook index grid for the given pointcloud (B, R, R, R)
        mode: The most common predicted codebook index (1,)
        sparse_unpacked: Codebook indices in (grid index, codebook index) representation (B, L, 2)
        """
        quant_ind = self.vqdif.quantize_cloud(cloud)[0]
        sparse_unpacked, mode = self.batch_dense2sparse(
            quant_ind, max_length=self.max_length, end_tokens=self.input_end_tokens
        )
        return sparse_unpacked, mode

    def get_indices(
        self, inputs: Tensor, pointcloud: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """
        Computes indices of the quantized (partial) input pointcloud and (optionally) the quantized complete pointcloud.

        Parameters
        ----------
        inputs : The (partial) input pointcloud.
        pointcloud : The (optional) complete pointcloud.

        Returns
        -------
        c_indices : Grid and codebook indices of the input pointcloud (B, L_c, 2)
        z_indices : Grid and codebook indices of the complete pointcloud (B, L_z, 2)
        extra_indices :
        """

        # encode the (partial) input pointcloud into quantized tensor with discrete indices
        c_indices, mode = self.encode_cloud(inputs)

        # if Xbd is None, z_indices is an empty tensor
        if pointcloud is None:
            z_indices = c_indices[:, :0, :]
        else:
            # encode the complete pointcloud into quantized tensor with discrete indices
            z_indices = self.encode_cloud(pointcloud)[0]

        others = dict(empty_index=mode, origin_c_indices=c_indices, origin_z_indices=z_indices)

        # mask random indices
        if self.training and self.random_cind_masking and c_indices.size(1) >= 1:
            max_num = c_indices.size(1) - 1
            select_num = np.random.randint(0, max_num + 1)
            selected_ind = np.sort(np.random.choice(max_num, select_num, replace=False))
            c_indices = torch.cat([c_indices[:, selected_ind, :], c_indices[:, -1:, :]], dim=1)

        extra_indices = self.get_extra_indices(c_indices, z_indices)
        return c_indices, z_indices, extra_indices, others

    @staticmethod
    def get_next_cond(c_pos_indices, z_pos_indices, end_token):
        if z_pos_indices.size(1) == 0:
            return z_pos_indices.clone()

        next_ids = torch.searchsorted(c_pos_indices.contiguous(), z_pos_indices.contiguous(), right=True)
        next_ids[z_pos_indices == end_token] = c_pos_indices.size(1) - 1
        next_cond_pos = torch.gather(c_pos_indices, dim=1, index=next_ids)
        next_cond_pos[z_pos_indices == end_token] = end_token
        return next_cond_pos

    def get_extra_indices(self, c_indices: Tensor, z_indices: Tensor) -> Tensor:
        c_extra = c_indices[..., 0].clone()  # Grid indices of input pointcloud (B, L_c)
        z_extra = self.get_next_cond(c_indices[..., 0], z_indices[..., 0], self.end_tokens[0])  # (B, L_z)
        extra_indices = torch.cat([c_extra, z_extra], dim=1)[..., None]  # (B, L_c+L_z, 1)
        return extra_indices

    def sampling_masker(
        self,
        logits: Tensor,
        idx: Tensor,
        extra_idx: Tensor | None = None,
        L_cond: int | None = None,
        step_j: int | None = None,
        tuple_i: int | None = None,
    ) -> Tensor:
        """logits: (B, vocab_size), idx[-1]: current position (to be sampled)"""
        # only apply to position index
        logits = logits.clone()
        latest_positions = idx[:, -2, 0]
        B = logits.shape[0]
        end_tokens = torch.tensor(self.end_tokens).type_as(idx)
        if tuple_i == 1:
            # if pos==end_token[0], then val should be end_token[1]
            end_mask = idx[:, -1, 0] == end_tokens[0]
            logits[end_mask, :] = -float("Inf")
            logits[end_mask, end_tokens[1]] = 1.0
            return logits
        positions = torch.arange(logits.shape[-1])[None, :].type_as(idx)
        if self.mask_invalid and step_j is not None and step_j > 0:
            # print("masking")
            # the latest sampled position is the largest
            # only keep the possibilities for index>largest, except for end token
            invalid_mask = positions <= latest_positions[:, None]
            invalid_mask[:, end_tokens[0]] = False
            logits[invalid_mask] = -float("Inf")  # mask out invalids
        if self.mask_invalid_completion and L_cond is not None:
            cond_pos_idx = idx[:, :L_cond, 0].contiguous()
            # (B, L_cond+1) append 1+end_tokens[0] to prevent corner cases
            cond_poses = torch.cat((cond_pos_idx, 1 + end_tokens[0][None, None].expand(B, -1)), dim=1)
            # find next position of condition
            next_ids = torch.searchsorted(cond_poses.contiguous(), latest_positions[:, None].contiguous(), right=True)
            next_poses = torch.gather(cond_poses, dim=1, index=next_ids)
            # print("Next poses", next_ids)
            # print("Current seq", idx[:,L_cond:L_cond+step_j,0])
            # mask the logits after the next cond
            invalid_mask = positions > next_poses
            logits[invalid_mask] = -float("Inf")
        return logits


class ShapeFormer(nn.Module):
    def __init__(
        self,
        tuple_n: int | None = None,
        block_size: int | None = None,
        end_tokens: tuple[int, int] | None = None,
        vocab_sizes: list[int] | None = None,
        extra_vocab_sizes: list[int] | None = None,
        voxel_res: int = 16,
        transformer_opt: dict[str, Any] | None = None,
        representer_opt: dict[str, Any] | None = None,
        top_k: int = 100,
        top_p: float = 0.4,
        sample_n: int = 4,
        sample_max_step: int = 512,
        temperature: float = 1,
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())
        if transformer_opt is None:
            msg = "transformer_opt must be provided"
            raise ValueError(msg)
        if representer_opt is None:
            msg = "representer_opt must be provided"
            raise ValueError(msg)
        if tuple_n is None:
            msg = "tuple_n must be provided"
            raise ValueError(msg)
        if end_tokens is None:
            msg = "end_tokens must be provided"
            raise ValueError(msg)
        self.tuple_n = int(tuple_n)
        self.end_tokens = end_tokens
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.sample_n = int(sample_n)
        self.sample_max_step = int(sample_max_step)
        self.temperature = float(temperature)

        self.transformer = CondTupleGPT(**transformer_opt)
        self.representer = ShapeRepresenter(**representer_opt)
        self.name = self.__class__.__name__
        self.optimizer = self._configure_optimizer
        if self.transformer.implementation == "xformers":
            self.optimizer = self._configure_optimizer_xformer

        # self.representer.max_length = sample_max_step

    @torch.no_grad()
    def generate_grids(self, inputs: Tensor, resolution: int = 128) -> tuple[list[np.ndarray], np.ndarray]:
        assert inputs.size(0) == 1, "Only batch size 1 supported"
        c_indices, z_indices, _extra_indices, others = self.representer.get_indices(inputs=inputs)
        sample_indices, logits_history = self.sample_indices(
            c_indices=c_indices.expand(self.sample_n, -1, -1),
            z_indices=z_indices.expand(self.sample_n, -1, -1),
            max_steps=self.sample_max_step,
            temperature=self.temperature,
            best_in_first=True,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        samples = sample_indices.cpu().numpy()
        logits_history = [[logit.cpu().numpy() for logit in lh] for lh in logits_history]
        log_prob = self.compute_log_probs(samples, logits_history)
        probs = [logp.sum() for logp in log_prob]
        order = np.argsort(np.array(probs))[::-1]

        grids = list()
        box_size = 1 + self.representer.vqdif.encoder.padding
        grid_shape = (resolution, resolution, resolution)
        grid_points = box_size * make_3d_grid((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), grid_shape)
        query_points = grid_points.unsqueeze(0).to(inputs.device)
        quantizer = cast(Any, self.representer.vqdif.quantizer)
        for i in range(len(samples)):
            origin_i = i
            i = order[origin_i]
            sample = samples[i]

            filtered = self.filter_end_tokens(sample, end_tokens=self.end_tokens)
            packed_sparse = torch.zeros(filtered.shape[0], 3).long()
            packed_sparse[:, 1:] = torch.from_numpy(filtered)
            voxel_vqind = self.representer.batch_sparse2dense(
                packed_sparse, others["empty_index"], 2**4, return_flattened=False, dim=3
            )[0]
            voxel_vqind = voxel_vqind.unsqueeze(0).long().to(inputs.device)

            quant_feat = quantizer.embedding(voxel_vqind)
            quant_feat = quant_feat.permute(0, 4, 1, 2, 3).contiguous()

            grid = cast(Tensor, self.representer.vqdif.decode(query_points, quant_feat))
            grids.append(grid.cpu().numpy().reshape((resolution,) * 3))
        return grids, grid_points.numpy()

    def forward(self, inputs: Tensor, pointcloud: Tensor | None = None, **kwargs) -> tuple[list[Tensor], Tensor]:
        # get indices from representer
        c_indices, z_indices, extra_indices, _others = self.representer.get_indices(inputs, pointcloud)
        cz_indices = torch.cat((c_indices, z_indices), dim=1)  # (B, L_c+L_z, 2)
        L_c = c_indices.size(1)
        # target includes all sequence elements (no need to handle first one differently because we are conditioning)
        targets = z_indices
        # make the prediction
        logits = self.transformer(
            idx=cz_indices[:, :-1, :], extra_idx=extra_indices[:, :-1, :], L_cond=L_c, target_idx=cz_indices[:, 1:, :]
        )
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        for i in range(self.tuple_n):
            # (B, L, vocab_sizes[i])
            logits[i] = logits[i][..., L_c - 1 :, :]
            # targets[i] = z_indices[i] #targets[i][:, cond_L-1:, :]
        return logits, targets

    @staticmethod
    def loss(logits: list[Tensor], targets: Tensor) -> Tensor:
        if not logits:
            msg = "logits must not be empty"
            raise ValueError(msg)
        loss = logits[0].new_zeros(())
        for i in range(len(logits)):
            logi = rearrange(logits[i], "B L vocab_size -> (B L) vocab_size")
            targ = rearrange(targets[..., i], "B L -> (B L)")
            loss += F.cross_entropy(logi, targ)
        return loss / len(logits)

    def _configure_optimizer(self, weight_decay=1e-2, learning_rate=1e-5, betas=(0.9, 0.95), device_type="cuda"):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, _p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith(("weight", "in_proj_weight")) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif fpn in ["pos_emb", "cond_pos_emb"]:
                    # special case the position embedding parameter in the root GPT module as not decayed
                    no_decay.add(fpn)
                    continue

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurrence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        # decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params!s} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {param_dict.keys() - union_params!s} were not separated into either decay/no_decay set!"
        )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def _configure_optimizer_xformer(self, lr: float = 1e-5, **kwargs) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight", "pos_emb", "cond_pos_emb"]
        params_decay = [p for n, p in self.transformer.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.transformer.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": 0.01},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))
        return optimizer

    @staticmethod
    def filter_end_tokens(indices, end_tokens=(8192, 4096)):
        # indices: (L, tuple_n)
        end_tokens = np.array(end_tokens)[None, ...]
        valids = (indices != end_tokens).all(axis=1)
        indices = indices[valids, :]
        return indices

    @staticmethod
    def compute_log_probs(samples, logits_history):
        """sample: (S, L, tuple_n), logits_history: [(S, L,vocab_size),]*tuple_n
        output: (S, L, tuple_n)
        """
        sample_n, sample_L, tuple_n = samples.shape
        slog_p = np.zeros(samples.shape)
        for si in range(sample_n):
            for ti in range(tuple_n):
                sample_ti = samples[si, :, ti]
                log_prob = log_softmax(logits_history[ti][si], axis=-1)
                slog_p[si, :, ti] = log_prob[np.arange(sample_L), sample_ti]
        return slog_p

    @staticmethod
    def filter_sampling_logits(logits, top_k, top_p, temperature, filter_value=-float("Inf")):
        # batch size 1 for now - could be updated for more but the code would be less clear
        assert logits.dim() == 1
        top_k = min(top_k, logits.size(-1))  # Safety check
        logits = logits / temperature
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def sample_logits(self, logits, num_samples=1, **filter_kwargs):
        # logits: (B, vocab_size)
        # apply softmax to convert to probabilities
        filtered_logits = [self.filter_sampling_logits(logits[i], **filter_kwargs) for i in range(logits.shape[0])]
        filtered_logits = torch.stack(filtered_logits)
        probs = F.softmax(filtered_logits, dim=-1)
        # print("probs\n",probs.numpy())
        ix_sampled = torch.multinomial(probs, num_samples=num_samples, replacement=True)
        # (B, num_samples)
        return ix_sampled

    @torch.no_grad()
    def sample_indices(
        self,
        c_indices: Tensor,
        z_indices: Tensor,
        max_steps: int,
        best_in_first: bool = False,
        top_k: int = 100,
        top_p: float = 0.4,
        temperature: float = 1,
    ) -> tuple[Tensor, list[Tensor]]:
        # x may start as shape (B, 0, tuple_n)
        # c: (B, L_c, tuple_n)
        assert not self.transformer.training
        B, L_c, tuple_n = c_indices.shape
        L_z = z_indices.shape[1]
        L = L_c + L_z
        end_tokens = torch.tensor(self.end_tokens).type_as(z_indices)
        block_size = self.transformer.block_size

        sampled = torch.zeros(B, L + max_steps, tuple_n).type_as(z_indices)
        seq_head, seq_tail = 0, L
        sampled[:, seq_head:seq_tail, :] = torch.cat((c_indices, z_indices), dim=1)
        logits_history: list[list[Tensor]] = [[] for _ in range(tuple_n)]

        for j in tqdm(range(max_steps), desc="Sampling"):
            if seq_tail - seq_head >= block_size:
                # crop generated shape if needed
                sampled[L_c : seq_tail - 1] = sampled[L_c + 1 : seq_tail]
                seq_tail -= 1
            c_indices, z_indices = sampled[:, :L_c, :].contiguous(), sampled[:, L_c:seq_tail, :].contiguous()
            idx = sampled[:, :seq_tail, :]
            extra_indices = self.representer.get_extra_indices(c_indices, z_indices)
            # print("extra ",extra_indices[0, L_c:seq_tail, 0])
            # print("idx ", idx[0, L_c:seq_tail, 0])
            # print("i", extra_indices)

            sample_gen = self.transformer.sample_next_tuple(idx, extra_idx=extra_indices, L_cond=L_c)
            # (B, vocab_size)
            logits = next(sample_gen)[:, -1, :]  # get newest predict's logit
            for i in range(0, tuple_n):  # start from 1, self.tuple_n-1 items in total
                # masking invalid logits
                logits = self.representer.sampling_masker(
                    logits, sampled[:, : seq_tail + 1, :], extra_indices, L_cond=L_c, step_j=j, tuple_i=i
                )
                logits_history[i].append(logits.detach().cpu())
                # new_ind: (B)
                new_ind = self.sample_logits(
                    logits, top_k=top_k, top_p=top_p, temperature=temperature, num_samples=1
                ).reshape(-1)
                new_best_ind = self.sample_logits(
                    logits, top_k=1, top_p=0.001, temperature=temperature, num_samples=1
                ).reshape(-1)
                if best_in_first:
                    new_ind[0] = new_best_ind[0]
                # fill in the newly generated sample
                sampled[:, seq_tail, i] = new_ind
                if i == tuple_n - 1:
                    break
                # (B, seq_head:seq_tail+1)
                target_i = sampled[:, seq_head + 1 : seq_tail + 1, i]
                # get newest predict's logit
                logits = sample_gen.send(target_i)[:, -1, :]
            seq_tail += 1
            # If all batch encountered stop token, then exit
            no_stop_token = (sampled[:, seq_tail - 1, :] != end_tokens[None, :]).all(dim=-1)
            if no_stop_token.long().sum() == 0:
                break
        # logits_history: [(B, vocab_size),]*tuple_n
        stacked_logits_history = [
            cast(Tensor, rearrange(history, "L B vocab_size -> B L vocab_size")) for history in logits_history
        ]
        # cut off conditioning
        x = sampled[:, L_c:seq_tail, :]
        # convert indices to output
        return x, stacked_logits_history
