from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset.src.fields import _load_points


def test_load_points_truncates_packbits_padding(tmp_path: Path) -> None:
    points = np.zeros((5, 3), dtype=np.float32)
    occupancy = np.array([True, False, True, False, True])
    path = tmp_path / "points.npz"
    np.savez_compressed(path, points=points, occupancies=np.packbits(occupancy))

    loaded_points, loaded_occupancy = _load_points(path)

    np.testing.assert_array_equal(loaded_points, points)
    np.testing.assert_array_equal(loaded_occupancy, occupancy)
