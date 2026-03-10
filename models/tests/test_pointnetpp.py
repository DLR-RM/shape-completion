import torch

from ..src.pointnetpp import PointNetPlusPlus, index_points, sample_and_group_all, square_distance


def test_square_distance_shape_and_values():
    src = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])  # (1, 2, 3)
    dst = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # (1, 2, 3)

    dist = square_distance(src, dst)

    assert dist.shape == (1, 2, 2)
    expected = torch.tensor([[[0.0, 1.0], [1.0, 2.0]]])
    assert torch.allclose(dist, expected)


def test_index_points_gathers_expected_rows():
    points = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        ]
    )  # (2, 3, 2)
    idx = torch.tensor([[2, 0], [1, 1]])  # (2, 2)

    gathered = index_points(points, idx)

    assert gathered.shape == (2, 2, 2)
    assert torch.equal(gathered[0, 0], points[0, 2])
    assert torch.equal(gathered[0, 1], points[0, 0])
    assert torch.equal(gathered[1, 0], points[1, 1])
    assert torch.equal(gathered[1, 1], points[1, 1])


def test_sample_and_group_all_shapes():
    xyz = torch.randn(2, 16, 3)
    points = torch.randn(2, 16, 5)

    new_xyz, new_points = sample_and_group_all(xyz, points)

    assert new_xyz.shape == (2, 1, 3)
    assert new_points.shape == (2, 1, 16, 8)


def test_pointnetplusplus_forward_grid_output():
    model = PointNetPlusPlus(c_dim=8, hidden_dim=8, grid_resolution=4, feature_type=("grid",))
    points = torch.randn(2, 16, 3)

    out = model(points)

    assert "grid" in out
    assert isinstance(out["grid"], torch.Tensor)
    assert out["grid"].shape == (2, 8, 4, 4, 4)
