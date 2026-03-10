from torch import Tensor

from .conv_onet import SimpleGridDecoder
from .mixins import MultiEvalMixin, MultiLossMixin, PredictMixin
from .model import Model
from .pvcnn import PVCNN
from .xdconf import pytorch_scatter


class DMTet(MultiEvalMixin, MultiLossMixin, PredictMixin, Model):
    """Implementation of Deep Marching Tetrahedra: https://research.nvidia.com/labs/toronto-ai/DMTet"""

    def __init__(
        self,
        channels: tuple[int, ...] = (3, 64, 256, 512),
        resolutions: tuple[int, ...] = (32, 16, 8),
        padding: float = 0.1,
        scatter: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.resolutions = resolutions
        self.padding = padding
        self.scatter = scatter

        """
        Input Encoder Given an input point cloud x, we use PVCNN [11] to extract multi-scale 3D feature volumes of
        sizes: R^3_1 x C_1, R^3_2 x C_2 and R^3_3 x C_3, where the spatial resolution R_1 = 32, R_2 = 16, R_3 = 8 and
        the number of channels C_1 = 64, C_2 = 256, C_3 = 512. To extract the point-wise feature for a point location
        v ∈ R^3, we use trilinear interpolation to obtain the feature vector from each 3D feature volume, and
        concatenate these vectors together to form the final feature vector Fvol(v, x) ∈ R^832. Similarly to DefTet [5],
        we use two encoders: one provides features for initial prediction of SDF and the other provides features for
        surface refinement and surface subdivision.
        """
        self.encoder = PVCNN(channels, resolutions, padding=padding, **kwargs)

        """
        MLPs for Initial Prediction of SDF For the initial prediction of SDF, we employ a four-layer
        MLPs with hidden dimensions 256, 256, 128 and 64, respectively. We extract the activations before
        the output layer, denoted as f(v), and pass them to the surface refinement step.
        """
        self.decoder = SimpleGridDecoder(channels=(sum(channels[1:]), 256, 256, 128, 64, 1))

    def forward(self, inputs: Tensor, points: Tensor, **kwargs) -> dict[str, Tensor]:
        return self.decode(points, self.encode(inputs, **kwargs), **kwargs)

    def encode(self, inputs: Tensor, **kwargs) -> dict[str, list[Tensor]]:
        feature = self.encoder(inputs)
        if self.scatter:
            return {
                "grid": [
                    pytorch_scatter(f[0], inputs, r, self.padding)
                    for f, r in zip(feature, self.resolutions, strict=False)
                ]
            }
        return {"grid": [f[1] for f in feature]}

    def decode(self, points: Tensor, feature: dict[str, list[Tensor]], **kwargs) -> dict[str, Tensor]:
        logits = self.decoder(points, feature)
        return {"logits": logits}

    def loss(
        self,
        data: dict[str, list[str] | Tensor],
        regression: bool = False,
        name: str | None = None,
        reduction: str | None = "mean",
    ) -> Tensor:
        return super().loss(data, regression, name, reduction)["occ_loss"]
