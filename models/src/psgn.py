from torch import Tensor, nn

from libs import ChamferDistanceL2, EarthMoversDistance

from .model import Model
from .pointnet import ResNetPointNet


class Decoder(nn.Module):
    r"""Simple decoder for the Point Set Generation Network.
    The simple decoder consists of 4 fully-connected layers, resulting in an
    output of 3D coordinates for a fixed number of points.
    Args:
        dim (int): The output dimension of the points (e.g. 3)
        c_dim (int): dimension of the input vector
        n_points (int): number of output points
    """

    def __init__(self, dim=3, c_dim=128, n_points=1024):
        super().__init__()
        # Attributes
        self.dim = dim
        self.c_dim = c_dim
        self.n_points = n_points

        # Submodules
        self.actvn = nn.ReLU(inplace=True)
        self.fc_0 = nn.Linear(c_dim, 512)
        self.fc_1 = nn.Linear(512, 512)
        self.fc_2 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, dim * n_points)

    def forward(self, c):
        batch_size = c.size(0)

        net = self.fc_0(c)
        net = self.fc_1(self.actvn(net))
        net = self.fc_2(self.actvn(net))
        points = self.fc_out(self.actvn(net))
        points = points.view(batch_size, self.n_points, self.dim)

        return points


class PSGN(Model):
    r"""The Point Set Generation Network.
    For the PSGN, the input image is first passed to the encoder network,
    e.g. restnet-18 or the CNN proposed in the original publication. Next,
    this latent code is then used as the input for the decoder network, e.g.
    the 2-Branch model from the PSGN paper.
    Args:
        decoder (nn.Module): The decoder network
        encoder (nn.Module): The encoder network
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = ResNetPointNet(c_dim=512, hidden_dim=512)
        self.decoder = Decoder(c_dim=512, n_points=16 * 1024)

        self.chamfer = ChamferDistanceL2()
        self.emd = EarthMoversDistance()

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        feature = self.encode(inputs)
        return self.decode(feature)

    def encode(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.encoder(inputs)

    def decode(self, feature: Tensor, **kwargs) -> Tensor:
        return self.decoder(feature)

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def predict(self, **kwargs):
        raise NotImplementedError

    def loss(self, pred_pcd: Tensor, gt_pcd: Tensor, emd: bool = False, **kwargs) -> Tensor:
        if emd:
            return self.emd(pred_pcd, gt_pcd)[0].sqrt().mean()
        dist1, dist2 = self.chamfer(pred_pcd, gt_pcd)
        return (dist1.sqrt().mean() + dist2.sqrt().mean()) / 2
