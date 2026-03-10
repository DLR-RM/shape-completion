import torch
from torch import Tensor, nn

from libs import ChamferDistanceL2, EarthMoversDistance

from .model import Model


class PCN(Model):
    def __init__(self, num_pred: int = 16384, encoder_channel: int = 1024):
        super().__init__()
        self.number_fine = num_pred
        self.encoder_channel = encoder_channel
        grid_size = 4  # set default
        self.grid_size = grid_size
        assert self.number_fine % grid_size**2 == 0
        self.number_coarse = self.number_fine // (grid_size**2)
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Conv1d(512, self.encoder_channel, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.number_coarse),
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )
        a = (
            torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float)
            .view(1, grid_size)
            .expand(grid_size, grid_size)
            .reshape(1, -1)
        )
        b = (
            torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float)
            .view(grid_size, 1)
            .expand(grid_size, grid_size)
            .reshape(1, -1)
        )
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, grid_size**2)  # 1 2 S

        self.chamfer = ChamferDistanceL2()
        self.emd = EarthMoversDistance()

    def forward(self, inputs: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        feature, feature_global = self.encode(inputs)
        coarse, fine = self.decode(feature, feature_global)
        return coarse, fine

    def encode(self, inputs: Tensor, **kwargs):
        feature = self.first_conv(inputs.transpose(1, 2))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1, -1, inputs.size(1)), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 1024
        return feature, feature_global

    def decode(self, feature: Tensor, feature_global: Tensor, **kwargs):
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.number_coarse, 3)  # B M 3
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1)  # B M S 3
        point_feat = point_feat.reshape(-1, self.number_fine, 3).transpose(1, 2)  # B 3 N

        seed = (
            self.folding_seed.unsqueeze(2).expand(feature.size(0), -1, self.number_coarse, -1).to(feature.device)
        )  # B 2 M S
        seed = seed.reshape(feature.size(0), -1, self.number_fine)  # B 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.number_fine)  # B 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1)  # B C N

        fine = self.final_conv(feat) + point_feat  # B 3 N

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def predict(self, **kwargs):
        raise NotImplementedError

    def loss(
        self, coarse_and_fine_pcd: Tensor, gt_pcd: Tensor, step: int, emd: bool = True, **kwargs
    ) -> tuple[Tensor, Tensor]:
        coarse, fine = coarse_and_fine_pcd
        if emd:
            loss_coarse = self.emd(coarse, gt_pcd[:, : coarse.size(1)])[0].sqrt().mean()
        else:
            dist1, dist2 = self.chamfer(coarse, gt_pcd)
            loss_coarse = (dist1.sqrt().mean() + dist2.sqrt().mean()) / 2
        dist1, dist2 = self.chamfer(fine, gt_pcd)
        loss_fine = (dist1.sqrt().mean() + dist2.sqrt().mean()) / 2

        alpha = 0.01
        if step > 10000:
            alpha = 0.1
        if step > 20000:
            alpha = 0.5
        if step > 50000:
            alpha = 1

        return loss_coarse + alpha * loss_fine
