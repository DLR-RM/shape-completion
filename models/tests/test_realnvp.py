import numpy as np
import pytest
import torch
from lightning.pytorch import seed_everything
from matplotlib import pyplot as plt
from sklearn import datasets

from ..src.realnvp import BatchNormFlow, Permute, RealNVP, ScaleShiftLayer


def test_scale_shift():
    scale_shift = ScaleShiftLayer(num_inputs=2, num_masked=1, hidden_layers=[2, 2])
    x = torch.randn(2, 2)
    fwd_x = scale_shift.forward(x)[0]
    assert torch.allclose(x, scale_shift.inverse(fwd_x)[0])


def test_permute():
    permute = Permute(permutation=[1, 0])
    x = torch.randn(2, 2)
    fwd_x = permute.forward(x)[0]
    assert torch.allclose(x, permute.inverse(fwd_x)[0])


def test_batch_norm_flow():
    bn_flow = BatchNormFlow(num_inputs=2).eval()
    x = torch.randn(2, 2)
    fwd_x = bn_flow.forward(x)[0]
    assert torch.allclose(x, bn_flow.inverse(fwd_x)[0])

    bn_flow.train()
    y = torch.distributions.MultivariateNormal(torch.ones(2), 2 * torch.eye(2)).sample((10000,))
    x = bn_flow.inverse(y)[0]

    assert torch.allclose(y.mean(dim=0), torch.ones(2), atol=1e-1)
    assert torch.allclose(y.var(dim=0), 2 * torch.ones(2), atol=1e-1)

    assert torch.allclose(x.mean(dim=0), torch.zeros(2), atol=1e-1)
    assert torch.allclose(x.var(dim=0), torch.ones(2), atol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_real_nvp():
    torch.manual_seed(0)
    nvp = RealNVP(dim=2).train().cuda()
    x = nvp.sample()
    log_prob_sample = nvp.log_prob(x)
    log_prob_zero = nvp.log_prob(torch.zeros((1, 2)).cuda())
    assert torch.isfinite(log_prob_sample).all()
    assert torch.isfinite(log_prob_zero).all()
    assert torch.allclose(log_prob_sample, log_prob_zero, atol=4e-1)


def plot(flow):
    flow.eval()

    noisy_moons = np.asarray(datasets.make_moons(n_samples=1000, noise=0.05)[0], dtype=np.float32)
    z = flow.inverse(torch.from_numpy(noisy_moons).cuda())[0].detach().cpu().numpy()
    plt.subplot(221)
    plt.scatter(z[:, 0], z[:, 1])
    plt.title(r"$z = f(X)$")

    z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)
    plt.subplot(222)
    plt.scatter(z[:, 0], z[:, 1])
    plt.title(r"$z \sim p(z)$")

    plt.subplot(223)
    x = np.asarray(datasets.make_moons(n_samples=1000, noise=0.05)[0], dtype=np.float32)
    plt.scatter(x[:, 0], x[:, 1], c="r")
    plt.title(r"$X \sim p(X)$")

    plt.subplot(224)
    x = flow.sample(1000).detach().cpu().numpy()
    plt.scatter(x[:, 0], x[:, 1], c="r")
    plt.title(r"$X = g(z)$")

    plt.show()


@pytest.mark.skip(reason="too slow")
def test_real_nvp_training():
    seed_everything(0)

    flow = RealNVP(dim=2).train().cuda()
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)

    for t in range(4501):
        noisy_moons = np.asarray(datasets.make_moons(n_samples=100, noise=0.05)[0], dtype=np.float32)
        loss = -flow.log_prob(torch.from_numpy(noisy_moons).cuda()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 500 == 0:
            print(f"iter {t}:", f"loss = {loss:.3f}")

    plot(flow)


if __name__ == "__main__":
    test_real_nvp_training()
