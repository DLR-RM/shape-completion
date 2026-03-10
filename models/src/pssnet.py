import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal

from .model import Model
from .realnvp import RealNVP


def log_normal_pdf(sample, mean, log_var):
    log2pi = math.log(2 * np.pi)
    return torch.sum(-0.5 * ((sample - mean).pow(2) * torch.exp(-log_var) + log_var + log2pi), dim=1)


def compute_vae_loss(z, mean, log_var, sample_logit, labels):
    cross_ent = F.binary_cross_entropy_with_logits(sample_logit, labels, reduction="none")
    logpx_z = -torch.sum(cross_ent, dim=[1, 2, 3])
    logpz = log_normal_pdf(z, torch.zeros(1, device=z.device), torch.zeros(1, device=z.device))
    logqz_x = log_normal_pdf(z, mean, log_var)
    return -torch.mean(logpx_z + logpz - logqz_x)


def compute_box_loss(gt_latent_box, inferred_latent_box_mean, inferred_latent_box_logvar):
    losses = -log_normal_pdf(gt_latent_box, inferred_latent_box_mean, inferred_latent_box_logvar)
    return torch.mean(losses)


class PSSNet(Model):
    """PyTorch implementation of PSSNet as proposed in [1] adapted from [2].
    [1] Diverse Plausible Shape Completions from Ambiguous Depth Images, CVPR 2020.
    [2] https://github.com/UM-ARM-Lab/probabilistic_shape_completion/blob/main/shape_completion_training/src/shape_completion_training/model/pssnet.py
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        use_flow_during_inference: bool = False,
        path_to_pretrained_flow: str | None = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.box_latent_size = 24
        self.use_flow_during_inference = use_flow_during_inference
        self.encoder = make_encoder(hidden_dim)
        self.generator = make_generator(hidden_dim)

        self.flow = None
        if path_to_pretrained_flow:
            print(f"Loading flow from {path_to_pretrained_flow}")
            self.flow = RealNVP(self.box_latent_size)
            self.flow.load_state_dict(torch.load(path_to_pretrained_flow, weights_only=False)["model"])
            self.flow.eval()

    """
    def call(self, dataset_element, training=False, **kwargs):
        known = stack_known(dataset_element)
        mean, logvar = self.encode(known)
        sampled_features = self.sample_latent(mean, logvar)

        if self.use_flow_during_inference:
            sampled_features = self.apply_flow_to_latent_box(sampled_features)

        predicted_occ = self.decode(sampled_features, apply_sigmoid=True)
        output = {'predicted_occ': predicted_occ, 'predicted_free': 1 - predicted_occ,
                  'latent_mean': mean, 'latent_logvar': logvar, 'sampled_latent': sampled_features}
        return output
    """

    def train(self, mode: bool = True):
        super().train(mode)
        if self.flow is not None:
            self.flow.eval()
        return self

    """
    @property
    def flow(self) -> RealNVP:
        if self._flow is None:
            self._flow = 
            self._modules["flow"] = self._flow
            if self.flow_path:
                print(f"Loading flow from {self.flow_path}")
                self._flow.load_state_dict(torch.load(self.flow_path, weights_only=False)['model'])
                print("Done")
            else:
                raise ValueError("Flow requested but no path to pretrained flow provided")
        return self._flow
    """

    @staticmethod
    def box_loss(z_box: Tensor, mean_box: Tensor, log_var_box: Tensor) -> Tensor:
        return -Normal(mean_box, torch.exp(0.5 * log_var_box)).log_prob(z_box).sum(1)

    @staticmethod
    def rec_loss(logits: Tensor, occ: Tensor) -> Tensor:
        if logits.shape != occ.shape:
            logits = logits.reshape(occ.shape)
        log_pxz = F.binary_cross_entropy_with_logits(logits, occ, reduction="none")
        if log_pxz.dim() == 2:
            return log_pxz.sum(1)
        elif log_pxz.dim() in [4, 5]:
            if log_pxz.dim() == 5:
                log_pxz = log_pxz.squeeze(1)
            return log_pxz.sum([1, 2, 3])
        else:
            raise ValueError(f"Unexpected dimensionality of log_pxz: {log_pxz.dim()}")

    @staticmethod
    def kl_loss(z: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
        """Monte Carlo estimate of KL-Divergence KL(q(z|x) || p(z))"""
        p = Normal(torch.zeros_like(mean), torch.ones_like(log_var))
        q = Normal(mean, torch.exp(0.5 * log_var))

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qzx - log_pz
        return kl.sum(1)

    @staticmethod
    def kl_divergence(mean: Tensor, log_var: Tensor) -> Tensor:
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def elbo_loss(self, logits: Tensor, occ: Tensor, z: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
        rec_loss = self.rec_loss(logits, occ)
        kl_loss = self.kl_loss(z, mean, log_var)
        return rec_loss + kl_loss

    def loss(
        self,
        logits: Tensor,
        occ: Tensor,
        z: Tensor,
        mean: Tensor,
        log_var: Tensor,
        z_box: Tensor | None = None,
        mean_box: Tensor | None = None,
        log_var_box: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        elbo_loss = self.elbo_loss(logits, occ, z, mean, log_var).mean()
        if z_box is not None and mean_box is not None and log_var_box is not None:
            box_loss = self.box_loss(z_box, mean_box, log_var_box).mean()
            return elbo_loss + box_loss
        return elbo_loss

    @staticmethod
    def sample_z(mean: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick to sample from N(mean, std) using N(0,1)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mean

    def split_box(self, x):
        split = [self.hidden_dim - self.box_latent_size, self.box_latent_size]
        features, box = torch.split(x, split_size_or_sections=split, dim=1)
        return features, box

    def replace_true_box(self, z, z_box):
        z = self.split_box(z)[0]
        return torch.cat([z, z_box], dim=1)

    def apply_flow_to_latent_box(self, z):
        if self.flow is None:
            raise ValueError("Flow model is not initialized")
        z, z_box = self.split_box(z)
        return torch.cat([z, self.flow.forward(z_box)[0]], dim=1)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean_log_var = self.encoder(x)
        mean, log_var = torch.split(mean_log_var, split_size_or_sections=mean_log_var.size(1) // 2, dim=1)
        return mean, log_var

    def decode(self, z, apply_sigmoid: bool = False):
        logits = self.generator(z)
        if apply_sigmoid:
            probs = torch.sigmoid(logits)
            return probs
        return logits

    def evaluate(self, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, inputs: Tensor) -> Tensor:
        x = inputs
        if x.dim() == 4:
            x = inputs.unsqueeze(1)
        mean, log_var = self.encode(x)
        z = self.sample_z(mean, log_var)
        logits = self.decode(z)
        return logits.squeeze(1)

    @torch.no_grad()
    def predict_many(self, inputs: Tensor, num_predictions: int = 20) -> Tensor:
        predictions = list()
        for _ in range(num_predictions):
            predictions.append(self.predict(inputs))
        return torch.stack(predictions).squeeze(1)

    @torch.no_grad()
    def predict_box(self, inputs: Tensor) -> Tensor:
        x = inputs
        if x.dim() == 4:
            x = inputs.unsqueeze(1)
        mean, log_var = self.encode(x)
        z = self.sample_z(mean, log_var)
        z, z_box = self.split_box(z)
        if self.flow is None:
            raise ValueError("Flow model is not initialized")
        return self.flow.forward(z_box)[0]

    def forward(
        self, inputs: Tensor, bbox: Tensor | None = None, **kwargs
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        x = inputs
        if x.dim() == 4:
            x = x.unsqueeze(1)

        mean, log_var = self.encode(x)
        z = self.sample_z(mean, log_var)

        z_box = None
        mean_box = None
        log_var_box = None
        z_with_box = z

        if self.flow is not None:
            if self.training:
                if bbox is not None:
                    z_box = self.flow.inverse(bbox)[0].detach()
                    z_with_box = self.replace_true_box(z, z_box)

                    mean, mean_box = self.split_box(mean)
                    log_var, log_var_box = self.split_box(log_var)
                    z = self.split_box(z)[0]
            else:
                if self.use_flow_during_inference:
                    z_with_box = self.apply_flow_to_latent_box(z_with_box)

        logits = self.decode(z_with_box)

        return logits.squeeze(1), z, mean, log_var, z_box, mean_box, log_var_box

    """
    @tf.function
    def train_step(self, train_element):
        bb = tf.keras.layers.Flatten()(tf.cast(train_element['bounding_box'], tf.float32))
        gt_latent_box = self.flow.bijector.inverse(bb)
        gt_latent_box = tf.stop_gradient(gt_latent_box)
        # train_element['gt_latent_box'] = gt_latent_box
        with tf.GradientTape() as tape:
            # train_outputs = self.call(train_element, training=True)
            # train_losses = self.compute_loss(gt_latent_box, train_outputs)
            known = stack_known(train_element)
            mean, logvar = self.encode(known)
            sampled_latent = self.sample_latent(mean, logvar)
            corrected_latent = self.replace_true_box(sampled_latent, gt_latent_box)

            if self.use_flow_during_inference:
                corrected_latent = self.apply_flow_to_latent_box(corrected_latent)

            logits = self.decode(corrected_latent)

            mean_f, mean_box = self.split_box(mean)
            logvar_f, logvar_box = self.split_box(logvar)
            sampled_f, _ = self.split_box(sampled_latent)

            vae_loss = compute_vae_loss(sampled_f, mean_f, logvar_f, logits, train_element['gt_occ'])
            box_loss = compute_box_loss(gt_latent_box, mean_box, logvar_box)
            train_losses = {"loss/vae_loss": vae_loss, "loss/box_loss": box_loss,
                            "loss": vae_loss + box_loss}
            train_outputs = None

        gradient_metrics = self.apply_gradients(tape, train_element, train_outputs, train_losses)
        other_metrics = self.calculate_metrics(train_element, train_outputs)

        metrics = {}
        metrics.update(train_losses)
        metrics.update(gradient_metrics)
        metrics.update(other_metrics)

        return train_outputs, metrics
        """


def make_encoder(hidden_dim: int = 200):
    # Mimics Tensorflow's default padding='SAME' behavior
    return nn.Sequential(
        nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0),
        nn.Conv3d(1, 64, 2),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
        nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0),
        nn.Conv3d(64, 128, 2),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
        nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0),
        nn.Conv3d(128, 256, 2),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
        nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0),
        nn.Conv3d(256, 512, 2),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4 * 512, 2 * hidden_dim),
    )


def make_generator(hidden_dim: int = 200):
    return nn.Sequential(
        nn.Linear(hidden_dim, 4 * 4 * 4 * 512),
        nn.ReLU(inplace=True),
        nn.Unflatten(1, (512, 4, 4, 4)),
        nn.ConvTranspose3d(512, 256, 2, stride=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose3d(256, 128, 2, stride=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose3d(128, 64, 2, stride=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose3d(64, 32, 2, stride=2),
        nn.ReLU(inplace=True),
        # FIXME: This is a hack to make the output size match the input size
        nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0),
        nn.ConvTranspose3d(32, 1, 2, padding=1),
    )
