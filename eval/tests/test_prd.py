import numpy as np
import pytest

from eval.src import prd


def test_compute_prd_shape_and_range() -> None:
    eval_dist = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    ref_dist = np.array([0.1, 0.4, 0.5], dtype=np.float64)

    precision, recall = prd.compute_prd(eval_dist, ref_dist, num_angles=101)

    assert precision.shape == (101,)
    assert recall.shape == (101,)
    assert np.all((precision >= 0.0) & (precision <= 1.0))
    assert np.all((recall >= 0.0) & (recall <= 1.0))


def test_compute_prd_from_embedding_enforce_balance() -> None:
    eval_data = np.random.default_rng(seed=0).normal(size=(8, 4))
    ref_data = np.random.default_rng(seed=1).normal(size=(7, 4))

    with pytest.raises(ValueError, match="not equal"):
        prd.compute_prd_from_embedding(eval_data, ref_data, enforce_balance=True)


def test_prd_to_max_f_beta_pair_outputs_valid_scores() -> None:
    precision = np.linspace(0.1, 0.9, 9)
    recall = np.linspace(0.2, 1.0, 9)

    f_beta, f_beta_inv = prd.prd_to_max_f_beta_pair(precision, recall, beta=8)

    assert 0.0 <= f_beta <= 1.0
    assert 0.0 <= f_beta_inv <= 1.0
