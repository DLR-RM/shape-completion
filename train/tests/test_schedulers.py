import math

import pytest
import torch

from ..src.schedulers import LinearWarmupCosineAnnealingLR


def test_run():
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    warmup_epochs = 5
    max_epochs = 20
    warmup_start_lr = 0
    eta_min = 1e-5

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs, max_epochs, warmup_start_lr, eta_min)

    for epoch in range(max_epochs + 5):
        optimizer.step()
        scheduler.step()
        print(epoch, scheduler.get_lr())


def test_linear_warmup_cosine_annealing_lr():
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    warmup_epochs = 5
    max_epochs = 20
    warmup_start_lr = 0.01
    eta_min = 0.001

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs, max_epochs, warmup_start_lr, eta_min)

    # Test during warmup phase
    for epoch in range(1, warmup_epochs):
        expected_lr = warmup_start_lr + (optimizer.defaults["lr"] - warmup_start_lr) * (epoch / warmup_epochs)
        optimizer.step()
        scheduler.step()
        assert all([group["lr"] == pytest.approx(expected_lr) for group in optimizer.param_groups])

    # Test during cosine annealing phase
    for epoch in range(warmup_epochs, max_epochs):
        optimizer.step()
        scheduler.step()
        decay_ratio = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        coeff = 0.5 * (1.0 + math.cos(torch.pi * decay_ratio))
        expected_lr = eta_min + coeff * (optimizer.defaults["lr"] - eta_min)
        assert all([group["lr"] == pytest.approx(expected_lr) for group in optimizer.param_groups])

    # Test post-maximum epochs
    for _epoch in range(max_epochs, max_epochs + 5):
        optimizer.step()
        scheduler.step()
        assert all([group["lr"] == pytest.approx(eta_min) for group in optimizer.param_groups])
