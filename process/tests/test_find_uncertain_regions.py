import numpy as np

from process.scripts.find_uncertain_regions import get_offset, protect


def test_protect_returns_value() -> None:
    out = protect(lambda value: value + 1, value=41)
    assert out == 42


def test_protect_returns_exception() -> None:
    out = protect(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert isinstance(out, ValueError)


def test_get_offset_returns_finite_vector() -> None:
    points = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    offset = get_offset(points, show=False)

    assert offset.shape == (3,)
    assert np.isfinite(offset).all()
