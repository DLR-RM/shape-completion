import numpy as np

from eval.src import prdc


def test_compute_pairwise_distance_is_symmetric_for_same_input() -> None:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = prdc.compute_pairwise_distance(x, n_jobs=1)
    np.testing.assert_allclose(d, d.T, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(np.diag(d), np.zeros(3, dtype=np.float32), rtol=0.0, atol=1e-7)


def test_get_kth_value_matches_expected_second_smallest() -> None:
    x = np.array([[3.0, 1.0, 2.0], [10.0, 8.0, 9.0]], dtype=np.float32)
    # k=2 returns the max of the 2 smallest values per row.
    kth = prdc.get_kth_value(x, k=2, axis=-1)
    np.testing.assert_allclose(kth, np.array([2.0, 9.0], dtype=np.float32), rtol=0.0, atol=1e-7)


def test_compute_prdc_outputs_expected_keys_and_ranges() -> None:
    real = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    fake = np.array([[0.1, 0.0], [0.9, 0.0], [0.1, 1.0], [0.9, 1.0]], dtype=np.float32)
    out = prdc.compute_prdc(real, fake, nearest_k=1, n_jobs=1)

    assert set(out.keys()) == {"precision", "recall", "density", "coverage"}
    assert 0.0 <= float(out["precision"]) <= 1.0
    assert 0.0 <= float(out["recall"]) <= 1.0
    assert 0.0 <= float(out["coverage"]) <= 1.0
    assert np.isfinite(float(out["density"]))
    assert float(out["density"]) >= 0.0
