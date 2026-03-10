import math
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
import torch.distributions as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops import focal_loss

try:
    from geomloss import SamplesLoss  # pyright: ignore[reportMissingImports]
except ImportError:
    SamplesLoss = None

from utils import setup_logger

from ..mixins import MultiEvalMixin
from ..model import Model
from ..utils import classification_loss
from ..vae import VAEModel, VQVAEModel
from .model import AutoregressiveModel
from .transformer import LatentGPT

logger = setup_logger(__name__)


def sinkhorn_set_classification_loss(
    logits: Tensor,
    targets: Tensor,
    temperature: float = 0.1,
    max_iter: int = 50,
    causal: bool = True,
    reduction: str = "mean",
) -> Tensor:
    """
    Computes set-based classification loss using the Sinkhorn algorithm for approximate matching.

    Args:
        logits: Tensor of shape (B, N, C) containing the predicted logits for each class.
        targets: Tensor of shape (B, N) containing the true class indices.
        temperature: Regularization parameter for the Sinkhorn algorithm.
        max_iter: Maximum number of iterations for the Sinkhorn algorithm.
        causal: If True, prevents matching predictions to themselves or previous positions.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        A (scalar) tensor representing the (average) loss over the batch.
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits.float(), dim=2)  # (B, N, C)

    # Create cost matrix: (B, N, N)
    index = targets.unsqueeze(2).expand(-1, -1, targets.size(1)).transpose(1, 2)
    cost_matrix = -log_probs.gather(dim=2, index=index)

    if causal:
        # Create a mask to prevent matching prediction i to target j where j < i
        # Allow j >= i (including self-matching)
        mask = torch.triu(torch.ones_like(cost_matrix))  # (B, N, N)
        # Set cost to infinity where mask == 0 (i.e., j <= i)
        cost_matrix = cost_matrix.masked_fill(mask == 0, 10 * cost_matrix.max())

    # Initialize the joint distribution matrix
    P = (-cost_matrix / temperature).clamp(min=-88, max=88).exp()

    # Run the Sinkhorn algorithm
    for _ in range(max_iter):
        P = P / P.sum(dim=2, keepdim=True)
        P = P / P.sum(dim=1, keepdim=True)

    # Compute the loss
    loss = (P * cost_matrix).sum(dim=2).mean(dim=1)

    loss = loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss
    return loss.type(logits.dtype)


def hungarian_set_classification_loss(logits: Tensor, targets: Tensor, causal: bool = True) -> Tensor:
    """
     Computes set-based classification loss using the Hungarian algorithm.

    * Non-Differentiability of the Matching Process: While the loss is differentiable with respect to predictions,
      the matching process (Hungarian algorithm) itself is non-differentiable. This means that the gradients do not
      account for how small changes in predictions might alter the optimal assignment. The gradient will only reflect
      the loss given the current assignment.

    * Piecewise Differentiability: The loss function is effectively piecewise differentiable. Within each region where
      the assignment remains constant, the loss is a smooth function of predictions. However, at points where a small
      change in predictions would change the assignment, the gradient does not capture this effect.

     Args:
         logits: Tensor of shape (B, N, C) containing the predicted logits for each class.
         targets: Tensor of shape (B, N) containing the true class indices.
         causal: If True, prevents matching predictions to themselves or previous positions.

     Returns:
         A scalar tensor representing the average loss over the batch.
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=2)  # (B, N, C)

    loss = torch.tensor(0.0, device=logits.device)
    for b in range(logits.size(0)):
        # Extract log_probs and targets for the current batch
        log_prob = log_probs[b]  # (N, C)
        target_classes = targets[b]  # (N)

        # Create a cost matrix where cost[i][j] is the cross-entropy loss of assigning prediction i to target j
        # This is equivalent to -log_prob[i][target_classes[j]]
        cost_matrix = -log_prob[:, target_classes]  # (N, N)

        if causal:
            # Create a mask to prevent matching prediction i to target j where j < i
            # Allow j >= i (including self-matching)
            mask = torch.triu(torch.ones_like(cost_matrix))  # (N, N)
            # Set cost to infinity where mask == 0 (i.e., j <= i)
            cost_matrix = cost_matrix.masked_fill(mask == 0, float("inf"))

        # Convert to NumPy for the Hungarian algorithm
        cost_matrix_np = cost_matrix.detach().cpu().numpy()

        # Perform Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

        # Sum the assigned costs
        loss += cost_matrix[row_ind, col_ind].mean()

    # Average the loss over the batch
    return loss / logits.size(0)


def set_cross_entropy(
    logits: Tensor, targets: Tensor, causal: bool = True, temperature: float = 1.0, reduction: str = "mean"
) -> Tensor:
    """
    Calculate the Set Cross Entropy loss between two sets for multi-class classification in a batch,
    where the targets are class indices. It is the upper bound of NLL between sets.

    Args:
        logits (torch.Tensor): Predicted class logits of shape (B, N_pred, K),
                              where B is the batch size, N_pred is the number of predicted elements in the set,
                              and K is the number of classes.
        targets (torch.Tensor): Target class indices of shape (B, N_target),
                                where N_target is the number of target elements in the set.
        causal (bool): If True, prevents matching predictions to themselves or previous positions.
        temperature (float): Temperature parameter for the softmax function.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        torch.Tensor: The batched Set Cross Entropy loss.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)  # Shape: (B, N_pred, K)

    # Create cost matrix: (B, N, N)
    index = targets.unsqueeze(2).expand(-1, -1, targets.size(1)).transpose(1, 2)
    cost_matrix = -log_probs.gather(dim=2, index=index)

    if causal:
        # Create a mask to prevent matching prediction i to target j where j < i
        # Allow j >= i (including self-matching)
        mask = torch.triu(torch.ones_like(cost_matrix))
        # Set cost to infinity where mask == 0 (i.e., j <= i)
        cost_matrix = cost_matrix.masked_fill(mask == 0, float("inf"))

    # Apply logsumexp over the target elements to handle permutations (soft minimum)
    loss = -torch.logsumexp(-cost_matrix / temperature, dim=2).mean(dim=1)  # Shape: (B,)

    # Return the mean loss across the batch
    loss = loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss
    return loss.type(logits.dtype)


class LatentAutoregressiveModel(MultiEvalMixin, AutoregressiveModel):
    def __init__(
        self,
        discretizer: VQVAEModel | VAEModel,
        autoregressor: LatentGPT,
        conditioner: Model | Callable | None = None,
        discretizer_freeze: bool = True,
        conditioner_freeze: bool = True,
        condition_key: str | None = None,
        loss_type: Literal["ce", "set_ce", "hungarian", "sinkhorn", "bce", "nll", "kl"] = "ce",
        objective: Literal["causal", "denoise", "permutation"] = "causal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.discretizer_freeze = discretizer_freeze
        self._discretizer: VQVAEModel | VAEModel = discretizer
        if discretizer_freeze:
            self._discretizer.requires_grad_(False)
            self._discretizer.eval()

        self.condition_key = condition_key
        self.conditioner_freeze = conditioner_freeze
        self._conditioner: Model | Callable[..., Tensor] | None = conditioner
        if conditioner_freeze and isinstance(self._conditioner, nn.Module):
            self._conditioner.requires_grad_(False)
            self._conditioner.eval()

        self._autoregressor: LatentGPT = autoregressor
        self.loss_fn: Callable[..., Tensor] | type[dist.Normal]
        self.loss_type = loss_type
        if "bce" in loss_type:
            self.loss_fn = F.binary_cross_entropy_with_logits
        elif loss_type == "set_ce":
            self.loss_fn = set_cross_entropy
        elif loss_type == "sinkhorn":
            if SamplesLoss is None:
                logger.warning("The 'geomloss' package is not installed. Using custom implementation.")
                self.loss_fn = sinkhorn_set_classification_loss
            else:

                def cost_fn(x: Tensor, y: Tensor, causal: bool = True) -> Tensor:
                    log_probs = F.log_softmax(x.float(), dim=2)
                    targets = y.argmax(dim=2)
                    index = targets.unsqueeze(2).expand(-1, -1, targets.size(1)).transpose(1, 2)
                    cost_matrix = -log_probs.gather(dim=2, index=index)

                    if causal:
                        mask = torch.triu(torch.ones_like(cost_matrix))
                        cost_matrix = cost_matrix.masked_fill(mask == 0, float("inf"))

                    return cost_matrix

                self.loss_fn = SamplesLoss(
                    "sinkhorn", scaling=0.5, truncate=None, cost=cost_fn, debias=False, backend="tensorized"
                )
        elif loss_type == "hungarian":
            self.loss_fn = hungarian_set_classification_loss
        elif "ce" in loss_type:
            self.loss_fn = classification_loss
        elif loss_type in ["nll", "kl"]:
            if not isinstance(self._discretizer, VAEModel):
                raise ValueError("NLL loss requires a VAE model.")
            self.loss_fn = dist.Normal
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        self.objective = objective

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self._discretizer.setup(*args, **kwargs)
        conditioner = self._conditioner
        if conditioner is not None and hasattr(conditioner, "setup"):
            cast(Any, conditioner).setup(*args, **kwargs)
        if hasattr(self._autoregressor, "setup"):
            self._autoregressor.setup(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        exclude = ["_discretizer" if self.discretizer_freeze else "", "_conditioner" if self.conditioner_freeze else ""]
        return {
            k: v for k, v in super().state_dict(*args, **kwargs).items() if not any(ex in k for ex in exclude if ex)
        }

    def encode(self, inputs: Tensor, sample: bool = False, **kwargs) -> Tensor:
        with torch.enable_grad() if self._discretizer.training else torch.no_grad():
            if isinstance(self._discretizer, VQVAEModel):
                _, targets, _ = self._discretizer.quantize(inputs, **kwargs)
            elif isinstance(self._discretizer, VAEModel):
                targets = self._discretizer.sample_posterior(inputs, sample=self.training or sample, **kwargs)
            else:
                targets = cast(Tensor, self._discretizer.encode(inputs, **kwargs))
            targets = cast(Tensor, targets)
            if "perm" in self.objective and self.training:
                return targets.gather(dim=1, index=torch.rand_like(targets.float()).argsort())
            return targets

    def decode(self, indices_or_latents: Tensor, **kwargs) -> Tensor:
        with torch.enable_grad() if self._discretizer.training else torch.no_grad():
            if isinstance(self._discretizer, VQVAEModel):
                latents = cast(Tensor, cast(Any, self._discretizer).vq.get_codes_from_indices(indices_or_latents))
            elif isinstance(self._discretizer, VAEModel):
                latents = indices_or_latents
            else:
                latents = indices_or_latents
            discretizer = cast(Any, self._discretizer)
            return cast(Tensor, discretizer.latent_to_embd(latents))

    def get_conditioning(self, conditioning: Tensor, **kwargs) -> Tensor | None:
        conditioner = self._conditioner
        if conditioner is not None:
            if isinstance(conditioner, nn.Module):
                with torch.enable_grad() if conditioner.training else torch.no_grad():
                    if isinstance(conditioner, Model):
                        return cast(Tensor, cast(Any, conditioner).encode(conditioning, **kwargs))
                    return cast(Tensor, conditioner(conditioning, **kwargs))
            if callable(conditioner):
                return cast(Tensor, conditioner(conditioning, **kwargs))
            if hasattr(self._discretizer, "latent_to_feat"):
                discretizer = cast(Any, self._discretizer)
                with torch.enable_grad() if self._discretizer.training else torch.no_grad():
                    return cast(Tensor, discretizer.latent_to_feat(self.encode(conditioning, **kwargs)))
            return self.encode(conditioning, **kwargs)
        return conditioning

    def forward(self, targets: Tensor | None = None, conditioning: Tensor | None = None, **kwargs) -> Tensor:
        if targets is None:
            targets = self.encode(**kwargs)
        category_enc = getattr(self._autoregressor, "category_enc", None)
        if conditioning is None and isinstance(category_enc, nn.Embedding):
            conditioning = torch.randint(category_enc.num_embeddings, (targets.size(0),), device=targets.device)
        return self._autoregressor(targets, conditioning, **kwargs)

    def _predictions_to_indices_or_latents(
        self, predictions: Tensor, sample: bool = True, topk: int | None = None, temperature: float = 1.0
    ) -> Tensor:
        if sample:
            if isinstance(self._discretizer, VAEModel):
                mean, logstd = predictions.chunk(2, dim=2)
                scaled_logstd = logstd + 0.5 * math.log(temperature)
                return dist.Normal(mean, scaled_logstd.exp()).sample()
            if topk is not None:
                v, _ = torch.topk(predictions, k=min(topk, predictions.size(2)), dim=2)
                predictions[predictions < v[..., [-1]]] = -float("inf")
            return dist.Categorical(logits=predictions / temperature).sample()
        if isinstance(self._discretizer, VAEModel):
            return predictions.chunk(2, dim=2)[0]
        return predictions.argmax(dim=2)

    @torch.no_grad()
    def predict(
        self, data: dict[str, Tensor], return_loss: bool = False, sample: bool = True, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        targets = self.encode(inputs=data["inputs"], **kwargs)
        conditioning = self.get_conditioning(data[self.condition_key], **kwargs) if self.condition_key else None
        predictions = self(targets, conditioning, **kwargs)
        indices_or_latents = self._predictions_to_indices_or_latents(predictions, sample=sample)
        logits = cast(
            Tensor,
            self._discretizer.predict(
                points=data["points"],
                feature=self.decode(indices_or_latents, **kwargs),
                points_batch_size=kwargs.get("points_batch_size"),
            ),
        )
        self.log("bce_loss", F.binary_cross_entropy_with_logits(logits, data["points.occ"]).cpu().item())
        if return_loss:
            if isinstance(self._discretizer, VAEModel):
                mean, logstd = predictions.chunk(2, dim=2)
                loss = -dist.Normal(mean, logstd.exp()).log_prob(targets).mean(dim=2)
            else:
                loss = F.cross_entropy(
                    predictions.view(-1, predictions.size(-1)), targets.view(-1).long(), reduction="none"
                )
            return logits, loss
        return logits

    @torch.no_grad()
    def evaluate(self, data: dict[str, Tensor], **kwargs) -> dict[str, float]:
        logits = data.get("logits")
        loss = data.get("loss")
        if logits is None or loss is None:
            logits, loss = self.predict(data, return_loss=True, **kwargs)
        data.update({"logits": logits, "loss": loss})
        return super().evaluate(cast(dict[str, Tensor | list[str]], data), **kwargs)

    @torch.inference_mode()
    def generate(
        self,
        inputs: Tensor | None = None,
        points: Tensor | None = None,
        conditioning: Tensor | None = None,
        threshold: float | None = None,
        points_batch_size: int | None = None,
        temperature: float = 1.0,
        topk: int | None = None,
        progress: bool = True,
        return_intermediates: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        batch_size = inputs.size(0) if inputs is not None else points.size(0) if points is not None else 1
        category_enc = getattr(self._autoregressor, "category_enc", None)
        if isinstance(category_enc, nn.Embedding) and conditioning is None:
            conditioning = torch.randint(category_enc.num_embeddings, (batch_size,), device=self.device)
        elif self.condition_key and conditioning is not None:
            conditioning = self.get_conditioning(conditioning, **kwargs)

        gen_result = self._autoregressor.generate(
            c=conditioning, batch_size=batch_size, temperature=temperature, topk=topk,
            progress=progress, return_intermediates=return_intermediates,
        )

        if return_intermediates:
            indices_or_latents, intermediates = cast(tuple[Tensor, list[Tensor]], gen_result)
            logits = cast(
                Tensor,
                self._discretizer.predict(
                    points=points, feature=self.decode(indices_or_latents, **kwargs), points_batch_size=points_batch_size
                ),
            )
            return logits, intermediates

        indices_or_latents = cast(Tensor, gen_result)
        logits = cast(
            Tensor,
            self._discretizer.predict(
                points=points, feature=self.decode(indices_or_latents, **kwargs), points_batch_size=points_batch_size
            ),
        )
        return logits

    def loss(self, data: dict[str, Tensor], **kwargs) -> Tensor:
        targets = self.encode(inputs=data["inputs"], **kwargs)
        conditioning = self.get_conditioning(data[self.condition_key], **kwargs) if self.condition_key else None
        predictions = self(targets, conditioning, **kwargs)
        if self.loss_type in ["nll", "kl"]:
            mean, logstd = predictions.chunk(2, dim=2)
            p_z = cast(type[dist.Normal], self.loss_fn)(mean, logstd.exp())
            if self.loss_type == "nll":
                loss = -p_z.log_prob(targets).mean()
            else:
                with torch.no_grad():
                    target_mean, target_logstd = cast(
                        tuple[Tensor, Tensor],
                        cast(Any, self._discretizer).get_mean_logstd(inputs=data["inputs"], **kwargs),
                    )
                    q_z = dist.Normal(target_mean, target_logstd.exp())
                loss = dist.kl_divergence(q_z, p_z).mean()
        elif self.loss_type in ["set_ce", "hungarian"]:
            loss = cast(Callable[[Tensor, Tensor], Tensor], self.loss_fn)(predictions, targets)
        elif self.loss_type == "sinkhorn":
            if SamplesLoss is None:
                loss = cast(Callable[[Tensor, Tensor], Tensor], self.loss_fn)(predictions, targets)
            else:
                loss = cast(Callable[[Tensor, Tensor], Tensor], self.loss_fn)(
                    predictions, F.one_hot(targets, num_classes=predictions.size(2))
                ).mean()
        elif "bce" in self.loss_type:
            one_hot_targets = F.one_hot(targets, num_classes=predictions.size(2)).float()
            cumulative_targets = torch.cummax(one_hot_targets.flip(dims=[1]), dim=1).values.flip(dims=[1])

            weight = None
            pos_weight = None
            if "weight" in self.loss_type:
                n_pos = cumulative_targets.sum(dim=2, keepdim=True)
                weight = predictions.size(1) / n_pos
                if "pos" in self.loss_type:
                    pos_weight = (predictions.size(2) - n_pos) / n_pos

            if "focal" in self.loss_type:
                loss = focal_loss.sigmoid_focal_loss(predictions, cumulative_targets, reduction="none")
                if weight is not None:
                    loss *= weight
            else:
                loss = cast(Callable[..., Tensor], self.loss_fn)(
                    predictions, cumulative_targets, weight=weight, pos_weight=pos_weight, reduction="none"
                )
            loss = loss.min(dim=2).values.mean() if "min" in self.loss_type else loss.mean()
            if "hybrid" in self.loss_type:
                weight = kwargs.get("hybrid_loss_weight", 1e-3)
                loss += weight * F.cross_entropy(
                    predictions.view(-1, predictions.size(2)), targets.view(-1).long(), weight=pos_weight
                )
        else:
            loss = classification_loss(predictions, targets, name=self.loss_type)
        return loss
