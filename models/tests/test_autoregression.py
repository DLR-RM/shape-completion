from types import MethodType
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from torch import Tensor, nn

from ..src.autoregression.latent import (
    LatentAutoregressiveModel,
    hungarian_set_classification_loss,
    set_cross_entropy,
)
from ..src.autoregression.transformer import LatentGPT
from ..src.model import Model
from ..src.vae import VAEModel, VQVAEModel


def test_hungarian_set_classification_loss():
    logits = torch.randn(2, 32, 128)
    targets = torch.randint(0, 128, (2, 32))
    loss = hungarian_set_classification_loss(logits, targets, causal=False)
    loss_ce = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    idx = torch.rand(logits.shape[:2]).to(logits).argsort()
    logits_perm = logits.gather(1, idx.unsqueeze(-1).expand_as(logits))
    loss_perm = hungarian_set_classification_loss(logits_perm, targets, causal=False)
    loss_ce_perm = torch.nn.functional.cross_entropy(logits_perm.view(-1, logits_perm.size(-1)), targets.view(-1))

    assert torch.allclose(loss, loss_perm)
    assert not torch.allclose(loss_ce, loss_ce_perm)

    loss_causal = hungarian_set_classification_loss(logits, targets, causal=True)
    targets_perm = targets.gather(1, idx)
    loss_causal_perm = hungarian_set_classification_loss(logits_perm, targets_perm, causal=True)
    loss_ce_perm = torch.nn.functional.cross_entropy(logits_perm.view(-1, logits_perm.size(-1)), targets_perm.view(-1))

    assert torch.allclose(loss_causal, loss_causal_perm)
    assert torch.allclose(loss_ce, loss_ce_perm)


@torch.no_grad()
def test_permutation_loss():
    seed_everything(1337)

    predictions = torch.randn(8, 512, 4096)
    targets = torch.randint(0, 4096, (8, 512))
    targets = targets.gather(dim=1, index=torch.rand_like(targets.float()).argsort())

    one_hot_targets = F.one_hot(targets, num_classes=predictions.size(2)).float()
    cumulative_targets = torch.cummax(one_hot_targets.flip(dims=[1]), dim=1).values.flip(dims=[1])

    n_pos = cumulative_targets.sum(dim=2, keepdim=True)
    weight = predictions.size(1) / n_pos  # Accounts for sample imbalance
    loss = F.binary_cross_entropy_with_logits(predictions, cumulative_targets, reduction="none")
    loss_w = F.binary_cross_entropy_with_logits(predictions, cumulative_targets, weight=weight)
    assert torch.isclose((weight * loss).mean(), loss_w)

    pos_weight = (predictions.size(2) - n_pos) / n_pos  # Accounts for class imbalance
    loss_pos_w = F.binary_cross_entropy_with_logits(predictions, cumulative_targets, pos_weight=pos_weight)
    loss_full = F.binary_cross_entropy_with_logits(
        predictions, cumulative_targets, weight=weight, pos_weight=pos_weight
    )

    print(loss.mean(), loss_w, loss_pos_w, loss_full)


class _FakeCodebook(nn.Module):
    def __init__(self, n_latent: int) -> None:
        super().__init__()
        self.n_latent = n_latent
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def get_codes_from_indices(self, indices: Tensor) -> Tensor:
        return indices.float().unsqueeze(-1).expand(*indices.shape, self.n_latent)


class _FakeAutoencoder(Model):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))

    def setup(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs

    def encode(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        _ = kwargs
        return inputs

    def decode(self, points: Tensor, feature: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        _ = kwargs
        return {"logits": feature.sum(dim=-1) + points.sum(dim=-1) * 0}

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        _ = args, kwargs
        return {}

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        _ = args, kwargs
        return {}

    def predict(self, *args: Any, **kwargs: Any) -> Tensor:
        _ = args, kwargs
        return torch.zeros((1, 1))

    def loss(self, *args: Any, **kwargs: Any) -> Tensor:
        _ = args, kwargs
        return torch.tensor(0.0)


class _FakeConditionerModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.setup_calls = 0
        self.encode_calls: list[Tensor] = []

    def setup(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self.setup_calls += 1

    def encode(self, conditioning: Tensor, **kwargs: Any) -> Tensor:
        _ = kwargs
        self.encode_calls.append(conditioning.detach().clone())
        return conditioning + 3.0

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        _ = args, kwargs
        return {}

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        _ = args, kwargs
        return {}

    def predict(self, *args: Any, **kwargs: Any) -> Tensor:
        _ = args, kwargs
        return torch.zeros((1, 1))

    def loss(self, *args: Any, **kwargs: Any) -> Tensor:
        _ = args, kwargs
        return torch.tensor(0.0)


def _make_vq_discretizer(*, n_queries: int = 3, n_latent: int = 4) -> VQVAEModel:
    discretizer = cast(
        Any,
        VQVAEModel(ae=_FakeAutoencoder(), n_hidden=n_latent, n_code=2**n_latent, n_latent=n_latent, kmeans_init=False),
    )
    discretizer.n_queries = n_queries
    discretizer.setup_calls = 0
    discretizer.predict_calls = []
    discretizer.quantize_calls = []
    discretizer.vq = _FakeCodebook(n_latent)
    discretizer.latent_to_embd = nn.Identity()

    def _setup(self: Any, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self.setup_calls += 1

    def _quantize(self: Any, inputs: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor]:
        self.quantize_calls.append({"inputs": inputs.detach().clone(), "kwargs": kwargs})
        batch_size, queries, _ = inputs.shape
        indices = torch.arange(queries, device=inputs.device).view(1, queries).expand(batch_size, -1) % int(self.n_code)
        return inputs + 1.0, indices.long(), inputs.new_tensor(0.25)

    def _predict(
        self: Any,
        points: Tensor | None = None,
        feature: Tensor | None = None,
        points_batch_size: int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        assert points is not None and feature is not None
        self.predict_calls.append(
            {
                "points": points.detach().clone(),
                "feature": feature.detach().clone(),
                "points_batch_size": points_batch_size,
                "kwargs": kwargs,
            }
        )
        return feature.sum(dim=-1)

    def _latent_to_feat(self: Any, latents: Tensor, **kwargs: Any) -> Tensor:
        _ = kwargs
        return latents + 5.0

    discretizer.setup = MethodType(_setup, discretizer)
    discretizer.quantize = MethodType(_quantize, discretizer)
    discretizer.predict = MethodType(_predict, discretizer)
    discretizer.latent_to_feat = MethodType(_latent_to_feat, discretizer)
    return cast(VQVAEModel, discretizer)


def _make_vae_discretizer(*, n_queries: int = 3, n_latent: int = 4) -> VAEModel:
    discretizer = cast(Any, VAEModel(ae=_FakeAutoencoder(), n_embd=n_latent, n_latent=n_latent))
    discretizer.n_queries = n_queries
    discretizer.setup_calls = 0
    discretizer.predict_calls = []
    discretizer.stats = {
        "mean": torch.zeros((1, n_queries, n_latent)),
        "std": torch.ones((1, n_queries, n_latent)),
    }
    discretizer.latent_to_embd = nn.Identity()

    def _setup(self: Any, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self.setup_calls += 1

    def _get_mean_logstd(self: Any, inputs: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        _ = kwargs
        return inputs + 0.5, torch.full_like(inputs, -0.7)

    def _sample_posterior(
        self: Any, inputs: Tensor, sample: bool = True, return_moments: bool = False, **kwargs: Any
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        mean, logstd = _get_mean_logstd(self, inputs, **kwargs)
        z = mean + 1.0 if sample else mean
        if return_moments:
            return z, mean, logstd
        return z

    def _predict(
        self: Any,
        points: Tensor | None = None,
        feature: Tensor | None = None,
        points_batch_size: int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        assert points is not None and feature is not None
        self.predict_calls.append(
            {
                "points": points.detach().clone(),
                "feature": feature.detach().clone(),
                "points_batch_size": points_batch_size,
                "kwargs": kwargs,
            }
        )
        return feature.sum(dim=-1)

    discretizer.setup = MethodType(_setup, discretizer)
    discretizer.get_mean_logstd = MethodType(_get_mean_logstd, discretizer)
    discretizer.sample_posterior = MethodType(_sample_posterior, discretizer)
    discretizer.predict = MethodType(_predict, discretizer)
    return cast(VAEModel, discretizer)


def _make_autoregressor(
    *, n_vocab: int, n_block: int, n_latent: int | None = None, n_classes: int | None = None
) -> LatentGPT:
    autoregressor = cast(
        Any,
        LatentGPT(
            n_vocab=n_vocab,
            n_block=n_block,
            n_latent=n_latent,
            n_layer=1,
            n_embd=max(n_latent or 1, 4),
            n_head=1,
            cond=n_classes,
            bos_embd=False,
            pos_enc=None,
            voc_enc=False,
        ),
    )
    autoregressor.setup_calls = 0
    autoregressor.forward_calls = []
    autoregressor.generate_calls = []

    def _setup(self: Any, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self.setup_calls += 1

    def _forward(self: Any, x: Tensor, cond: Tensor | None = None, **kwargs: Any) -> Tensor:
        cond_record = None if cond is None else cond.detach().clone()
        self.forward_calls.append({"x": x.detach().clone(), "cond": cond_record, "kwargs": kwargs})
        cond_term = torch.zeros((x.size(0), 1, 1), device=x.device, dtype=torch.float32)
        if cond is not None:
            cond_term = cond.float().reshape(cond.size(0), -1).mean(dim=1).view(-1, 1, 1)
        if x.ndim == 3:
            mean = x + 0.25 + cond_term
            logstd = torch.full_like(x, -0.7)
            return torch.cat((mean, logstd), dim=2)
        vocab = torch.arange(int(self.n_vocab), device=x.device, dtype=torch.float32).view(1, 1, -1)
        return vocab + x.unsqueeze(-1).float() + cond_term

    def _generate(
        self,
        c: Tensor | None = None,
        batch_size: int | None = None,
        temperature: float = 1.0,
        topk: int | None = None,
        progress: bool = True,
        return_intermediates: bool = False,
        **kwargs: Any,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        _ = kwargs
        batch = batch_size or (c.size(0) if c is not None else 1)
        self.generate_calls.append(
            {
                "c": None if c is None else c.detach().clone(),
                "batch_size": batch_size,
                "temperature": temperature,
                "topk": topk,
                "progress": progress,
                "return_intermediates": return_intermediates,
            }
        )
        if self.n_latent is not None:
            seq = torch.full((batch, self.n_block, int(self.n_latent)), 1.5)
            if c is not None:
                seq = seq + c.float().reshape(batch, -1).mean(dim=1).view(batch, 1, 1)
        else:
            seq = torch.arange(self.n_block).view(1, self.n_block).expand(batch, -1) % int(self.n_vocab)
            if c is not None and not torch.is_floating_point(c):
                seq = (seq + c.long().view(batch, 1)) % int(self.n_vocab)
        if return_intermediates:
            intermediates = [seq[:, : i + 1].clone() for i in range(seq.size(1))]
            return seq, intermediates
        return seq

    autoregressor.setup = MethodType(_setup, autoregressor)
    autoregressor.forward = MethodType(_forward, autoregressor)
    autoregressor.generate = MethodType(_generate, autoregressor)
    return cast(LatentGPT, autoregressor)


def test_latent_autoregressive_init_setup_and_state_dict_respect_freeze_and_set_ce() -> None:
    discretizer = _make_vq_discretizer()
    conditioner = _FakeConditionerModel()
    autoregressor = _make_autoregressor(n_vocab=int(discretizer.n_code), n_block=3, n_classes=5)
    model = LatentAutoregressiveModel(
        discretizer=discretizer,
        autoregressor=autoregressor,
        conditioner=conditioner,
        discretizer_freeze=True,
        conditioner_freeze=True,
        condition_key="cond",
        loss_type="set_ce",
    )

    assert model.loss_fn is set_cross_entropy
    assert not any(parameter.requires_grad for parameter in discretizer.parameters())
    assert not any(parameter.requires_grad for parameter in conditioner.parameters())

    model.setup()
    state_dict = model.state_dict()

    assert cast(Any, discretizer).setup_calls >= 1
    assert conditioner.setup_calls >= 1
    assert cast(Any, autoregressor).setup_calls >= 1
    assert not any(key.startswith("_discretizer.") for key in state_dict)
    assert not any(key.startswith("_conditioner.") for key in state_dict)


def test_latent_autoregressive_vq_paths_cover_encoding_conditioning_predict_and_generate() -> None:
    seed_everything(1337)
    discretizer = _make_vq_discretizer()
    conditioner = _FakeConditionerModel()
    autoregressor = _make_autoregressor(n_vocab=int(discretizer.n_code), n_block=3, n_classes=7)
    model = LatentAutoregressiveModel(
        discretizer=discretizer,
        autoregressor=autoregressor,
        conditioner=conditioner,
        condition_key="cond",
        objective="permutation",
        loss_type="ce",
    )
    inputs = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    points = torch.randn(2, 3, 3)
    cond = torch.ones((2, 3, 4))
    data = {"inputs": inputs, "points": points, "points.occ": torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), "cond": cond}

    model.train()
    encoded = model.encode(inputs)
    expected = torch.arange(inputs.size(1)).view(1, -1).expand(inputs.size(0), -1)
    assert torch.equal(encoded.sort(dim=1).values, expected)

    conditioned = model.get_conditioning(cond)
    assert torch.allclose(cast(Tensor, conditioned), cond + 3.0)

    fallback_model = LatentAutoregressiveModel(
        discretizer=discretizer,
        autoregressor=autoregressor,
        conditioner=cast(Any, object()),
        loss_type="ce",
    )
    fallback_conditioning = fallback_model.get_conditioning(cond)
    assert torch.equal(cast(Tensor, fallback_conditioning), expected.float() + 5.0)

    model.eval()
    decoded = model.decode(torch.tensor([[0, 1, 2]]))
    assert torch.equal(decoded, torch.tensor([[[0.0] * 4, [1.0] * 4, [2.0] * 4]]))

    greedy = model._predictions_to_indices_or_latents(torch.tensor([[[0.0, 2.0, 1.0], [4.0, 3.0, 2.0]]]), sample=False)
    sampled = model._predictions_to_indices_or_latents(
        torch.tensor([[[0.0, 2.0, 1.0], [4.0, 3.0, 2.0]]]), sample=True, topk=1
    )
    assert torch.equal(greedy, torch.tensor([[1, 0]]))
    assert torch.equal(sampled, greedy)

    logits, token_loss = model.predict(data, return_loss=True, sample=False)
    result = model.evaluate({"logits": logits, "loss": token_loss, "points.occ": data["points.occ"]}, threshold=0.4)
    generated, intermediates = cast(
        tuple[Tensor, list[Tensor]],
        model.generate(points=points, conditioning=cond, return_intermediates=True, progress=False, topk=1),
    )
    model.forward(torch.zeros((2, 3), dtype=torch.long))
    category_cond = cast(Any, autoregressor).forward_calls[-1]["cond"]

    assert logits.shape == data["points.occ"].shape
    assert token_loss.ndim == 1 and token_loss.numel() == inputs.numel() // inputs.size(-1)
    assert "val/loss" in result
    assert generated.shape == data["points.occ"].shape
    assert len(intermediates) == 3
    assert torch.allclose(category_cond.float(), category_cond.float().round())

    callable_model = LatentAutoregressiveModel(
        discretizer=discretizer,
        autoregressor=autoregressor,
        conditioner=lambda x, **_: x + 4.0,
        condition_key="cond",
        loss_type=cast(Any, "bce_hybrid"),
    )
    callable_loss = callable_model.loss(data, hybrid_loss_weight=0.1)
    assert torch.isfinite(callable_loss)


def test_latent_autoregressive_vae_paths_cover_init_predict_generate_and_losses() -> None:
    inputs = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
    points = torch.randn(1, 3, 3)
    occ = torch.tensor([[1.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="NLL loss requires a VAE model"):
        LatentAutoregressiveModel(
            discretizer=_make_vq_discretizer(),
            autoregressor=_make_autoregressor(n_vocab=16, n_block=3),
            loss_type="nll",
        )

    discretizer = _make_vae_discretizer()
    autoregressor = _make_autoregressor(n_vocab=8, n_block=3, n_latent=4)
    model = LatentAutoregressiveModel(discretizer=discretizer, autoregressor=autoregressor, loss_type="nll")
    model.eval()

    encoded = model.encode(inputs, sample=False)
    decoded = model.decode(encoded)
    deterministic = model._predictions_to_indices_or_latents(
        torch.cat((inputs + 0.75, torch.full_like(inputs, -0.7)), dim=2),
        sample=False,
    )
    sampled = model._predictions_to_indices_or_latents(
        torch.cat((inputs + 0.75, torch.full_like(inputs, -0.7)), dim=2),
        sample=True,
    )
    logits, nll = model.predict({"inputs": inputs, "points": points, "points.occ": occ}, return_loss=True, sample=False)
    generated, intermediates = cast(
        tuple[Tensor, list[Tensor]],
        model.generate(points=points, return_intermediates=True, progress=False),
    )
    nll_loss = model.loss({"inputs": inputs})
    kl_model = LatentAutoregressiveModel(
        discretizer=_make_vae_discretizer(),
        autoregressor=_make_autoregressor(n_vocab=8, n_block=3, n_latent=4),
        loss_type="kl",
    )
    kl_loss = kl_model.loss({"inputs": inputs})

    assert torch.allclose(encoded, inputs + 0.5)
    assert torch.equal(decoded, encoded)
    assert torch.allclose(deterministic, inputs + 0.75)
    assert sampled.shape == inputs.shape
    assert logits.shape == occ.shape
    assert nll.shape == occ.shape
    assert generated.shape == occ.shape
    assert len(intermediates) == inputs.size(1)
    assert torch.isfinite(nll_loss)
    assert torch.isfinite(kl_loss)
