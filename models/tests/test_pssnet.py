import os
from typing import Any, cast

import open3d as o3d
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, kl_divergence

from ..src.pssnet import PSSNet, compute_box_loss, compute_vae_loss
from ..src.realnvp import RealNVP


@pytest.fixture
def root_dir() -> str:
    return os.environ.get("PRETRAINED_ROOT", "/path/to/pretrained/models")


@pytest.fixture
def path_to_pretrained_flow(root_dir) -> str:
    return os.path.join(root_dir, "/out/realnvp/mugs/paper/realnvp/model_best.pt")


@pytest.fixture
def path_to_pretrained_pssnet(root_dir) -> str:
    return os.path.join(root_dir, "/out/pssnet/mugs/paper/pssnet_split/model_best.pt")


def test_init(path_to_pretrained_flow):
    if not os.path.isfile(path_to_pretrained_flow):
        pytest.skip("Pretrained flow not found")

    model = PSSNet()
    assert model.flow is None

    if os.path.isfile(path_to_pretrained_flow):
        model = PSSNet(path_to_pretrained_flow=path_to_pretrained_flow)
        assert model.flow is not None

        model.train()
        assert model.training
        assert not model.flow.training


def test_forward():
    model = PSSNet()

    inputs = torch.rand(1, 64, 64, 64)
    logits = model(inputs=inputs)[0]
    assert logits.shape == inputs.shape


def test_loss():
    model = PSSNet()
    model.flow = RealNVP().eval()

    inputs = torch.rand(16, 64, 64, 64)
    occ = torch.rand(16, 64, 64, 64)
    occ = (occ > 0).float()
    box = torch.rand(16, 24)

    logits, z, mean, log_var, z_box, mean_box, log_var_box = model(inputs, box)

    assert model.rec_loss(logits, occ).mean() == model.rec_loss(logits, occ.view(occ.size(0), -1)).mean()

    log_pxz = -model.rec_loss(logits, occ).mean()
    log_pxz_test = Bernoulli(logits=logits).log_prob(occ).sum(dim=(1, 2, 3)).mean()
    log_pxz_alt = -F.binary_cross_entropy_with_logits(logits, occ, reduction="none").sum(dim=(1, 2, 3)).mean()

    assert torch.allclose(log_pxz, log_pxz_test)
    assert torch.allclose(log_pxz, log_pxz_alt)
    assert torch.allclose(log_pxz_test, log_pxz_alt)

    q_z = Normal(mean, torch.exp(0.5 * log_var))
    p_z = Normal(torch.zeros_like(mean), torch.ones_like(log_var))

    log_qzx = q_z.log_prob(z)
    log_pz = p_z.log_prob(z)

    kl_loss = model.kl_loss(z, mean, log_var).mean()
    kl_loss_test = (log_qzx - log_pz).sum(1).mean()

    assert torch.allclose(kl_loss, kl_loss_test)

    elbo = model.elbo_loss(logits, occ, z, mean, log_var).mean()
    elbo_test = kl_loss - log_pxz

    assert torch.allclose(elbo, elbo_test)

    box_loss = model.box_loss(
        z_box,
        mean_box,
        log_var_box,
    ).mean()
    box_loss_test = compute_box_loss(z_box, mean_box, log_var_box)

    assert torch.allclose(box_loss, box_loss_test)

    vae_loss = model.loss(logits, occ, z, mean, log_var)
    vae_loss_test = compute_vae_loss(z, mean, log_var, logits, occ)

    assert torch.allclose(vae_loss, vae_loss_test)

    loss = model.loss(logits, occ, z, mean, log_var, z_box, mean_box, log_var_box)
    loss_test = vae_loss_test + box_loss_test

    assert torch.allclose(loss, loss_test)

    kl_div = model.kl_divergence(mean, log_var)
    kl_div_test = kl_divergence(q_z, p_z).sum()

    assert torch.allclose(kl_div, kl_div_test)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(os.environ.get("DISPLAY") is None, reason="No display available")
def test_box(path_to_pretrained_flow, path_to_pretrained_pssnet, o3d_viz_enabled: bool):
    if not os.path.isfile(path_to_pretrained_flow) or not os.path.isfile(path_to_pretrained_pssnet):
        pytest.skip("Pretrained models not found")

    model = PSSNet(path_to_pretrained_flow=path_to_pretrained_flow)
    model.load_state_dict(torch.load(path_to_pretrained_pssnet, weights_only=False)["model"])
    model.eval().cuda()

    inputs = torch.rand(16, 64, 64, 64).cuda()
    bboxes = model.predict_box(inputs).detach().cpu().numpy().reshape(-1, 8, 3)

    for bbox in bboxes:
        bbox_points = o3d.utility.Vector3dVector(bbox)
        bbox_pcd = o3d.geometry.PointCloud(bbox_points)
        bbox_pcd.paint_uniform_color((1, 0, 0))

        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = bbox_points
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        if o3d_viz_enabled:
            cast(Any, o3d).visualization.draw_geometries([bbox_pcd, line_set])
