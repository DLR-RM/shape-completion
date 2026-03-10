from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .model import Model
from .utils import classification_loss, regression_loss


class MCDropoutNet(Model):
    def __init__(
        self,
        input_res: int = 40,
        latent_space: int = 2000,
        merge_mode: str = "concat",
        drop_rate: float = 0.2,
        dall: bool = True,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.input_res = input_res
        self.merge_mode = merge_mode

        if input_res < 40:
            c_e = [1, 32, 64, 128]
        elif input_res == 40:
            c_e = [1, 64, 128, 256]
        elif input_res > 40:
            c_e = [1, 64, 128, 256, 512]

        if use_batch_norm:
            e_convs: list[nn.Module] = [
                nn.Sequential(
                    nn.Conv3d(c_e[0], c_e[1], kernel_size=4, stride=2), nn.LeakyReLU(), nn.BatchNorm3d(c_e[1])
                )
            ]
        else:
            e_convs = [nn.Sequential(nn.Conv3d(c_e[0], c_e[1], kernel_size=4, stride=2), nn.LeakyReLU())]

        d_s = [2, 2, 2]
        d_p = [0, 1, 0]

        if "concat" in merge_mode:
            d_convs: list[nn.Module] = [
                nn.Sequential(
                    nn.ConvTranspose3d(2 * c_e[1], c_e[0], kernel_size=4, stride=d_s[0], output_padding=d_p[0])
                )
            ]
        else:
            d_convs = [
                nn.Sequential(nn.ConvTranspose3d(c_e[1], c_e[0], kernel_size=4, stride=d_s[0], output_padding=d_p[0]))
            ]
        # TODO: Fix batch norm
        for p in range(2, len(c_e)):
            if dall:
                e_convs.append(
                    nn.Sequential(
                        nn.Conv3d(c_e[p - 1], c_e[p], kernel_size=4, stride=2), nn.LeakyReLU(), nn.Dropout3d(drop_rate)
                    )
                )
            else:
                e_convs.append(nn.Sequential(nn.Conv3d(c_e[p - 1], c_e[p], kernel_size=4, stride=2), nn.LeakyReLU()))

            if "concat" in merge_mode:
                if dall:
                    d_convs.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(
                                2 * c_e[p], c_e[p - 1], kernel_size=4, stride=d_s[p - 1], output_padding=d_p[p - 1]
                            ),
                            nn.ReLU(),
                            nn.Dropout3d(drop_rate),
                        )
                    )
                else:
                    d_convs.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(
                                2 * c_e[p], c_e[p - 1], kernel_size=4, stride=d_s[p - 1], output_padding=d_p[p - 1]
                            ),
                            nn.ReLU(),
                        )
                    )
            else:
                if dall:
                    d_convs.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(
                                c_e[p], c_e[p - 1], kernel_size=4, stride=d_s[p - 1], output_padding=d_p[p - 1]
                            ),
                            nn.ReLU(),
                            nn.Dropout3d(drop_rate),
                        )
                    )
                else:
                    d_convs.append(
                        nn.Sequential(
                            nn.ConvTranspose3d(
                                c_e[p], c_e[p - 1], kernel_size=4, stride=d_s[p - 1], output_padding=d_p[p - 1]
                            ),
                            nn.ReLU(),
                        )
                    )
        d_convs.reverse()
        self.e_convs = nn.ModuleList(e_convs)
        self.d_convs = nn.ModuleList(d_convs)

        res = input_res
        for _ in range(len(self.e_convs)):
            res = self.conv_shape(res, k=4, s=2)
        hidden_dim = res**3 * c_e[-1]
        # print(f'Hidden dim: {hidden_dim}')

        fc: list[nn.Module] = [nn.Linear(hidden_dim, latent_space)]
        fc.append(nn.Dropout(drop_rate))
        fc.append(nn.ReLU())
        fc.append(nn.Linear(latent_space, hidden_dim))
        fc.append(nn.Dropout(drop_rate))
        fc.append(nn.ReLU())
        self.fc = nn.ModuleList(fc)

    @staticmethod
    def conv_shape(x, k=1, p=0, s=1, d=1):
        return int((x + 2 * p - d * (k - 1) - 1) / s + 1)

    def encode(self, x):
        x = x.unsqueeze(1)
        encoder_outs = []
        for i in range(len(self.e_convs)):
            x = self.e_convs[i](x)
            encoder_outs.append(x)
        return x, encoder_outs

    def latent(self, x):
        preshape = x.shape
        x = x.view(x.size(0), -1)
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        return x.view(preshape)

    def decode(self, x, encoder_outs):
        for i in range(len(self.d_convs)):
            if "concat" in self.merge_mode:
                skip_con = encoder_outs[-(i + 1)]
                x = torch.cat((x, skip_con), 1)
                x = self.d_convs[i](x)
            else:
                x = self.d_convs[i](x)
        return x.squeeze(1)

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        net, encoder_outs = self.encode(inputs)
        net = self.latent(net)
        net = self.decode(net, encoder_outs)
        return net

    def set_dropout_state(self, train: bool = True):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                if train:
                    module.train()
                else:
                    module.eval()

    def mc_sample(self, x: Tensor, num_samples: int = 20) -> tuple[Tensor, Tensor]:
        self.set_dropout_state(train=True)
        logits = list()
        for _ in range(num_samples):
            logits.append(self(x))
        logits = torch.stack(logits)
        if not self.training:
            self.set_dropout_state(train=False)
        probs = torch.sigmoid(logits)
        probs_mean = probs.mean(dim=0)
        probs_var = probs.var(dim=0)
        logits_mean = torch.logit(probs_mean)
        return logits_mean, probs_var

    def evaluate(self, **kwargs) -> dict[str, Tensor]:
        raise NotImplementedError

    def predict(self, **kwargs):
        raise NotImplementedError

    def loss(
        self,
        predictions: Tensor,
        occ_or_sdf: Tensor,
        sdf: str | None = None,
        tsdf: float | None = None,
        reduction: Literal["mean", "sum", "none"] | None = "mean",
    ) -> Tensor:
        reduction_mode: Literal["mean", "sum", "none"] = reduction or "mean"
        if sdf is not None:
            return regression_loss(predictions, occ_or_sdf, tsdf, name=sdf, reduction=reduction_mode)
        return classification_loss(predictions, occ_or_sdf, reduction=reduction_mode)
