from argparse import ArgumentTypeError

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from ..scripts.generate_physics_poses import filter_matrices, show_types, simplify_types


def test_filter_matrices():
    rot = [R.from_euler("x", angle).as_matrix() for angle in [0, -np.pi / 2, np.pi / 2, np.pi]]
    rot += [R.from_euler("y", angle).as_matrix() for angle in [-np.pi / 2, np.pi / 2]]
    assert len(filter_matrices(rot)) == len(rot)

    rot += [R.from_euler("z", angle).as_matrix() for angle in np.random.rand(3) * 2 * np.pi]
    assert len(filter_matrices(rot)) == len(rot) - 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("True", True),
        ("False", False),
        ("42", 42),
        ("0.25", 0.25),
    ],
)
def test_simplify_types_valid(raw: str, expected: bool | int | float) -> None:
    assert simplify_types(raw) == expected


@pytest.mark.parametrize("raw", ["0", "100001", "1.0", "-0.1", "invalid"])
def test_simplify_types_invalid(raw: str) -> None:
    with pytest.raises(ArgumentTypeError):
        simplify_types(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("True", True),
        ("False", False),
        ("poses", "poses"),
    ],
)
def test_show_types_valid(raw: str, expected: bool | str) -> None:
    assert show_types(raw) == expected


def test_show_types_invalid() -> None:
    with pytest.raises(ArgumentTypeError):
        show_types("bad")
