from ..src.fields import EmptyField, PointsField, PointCloudField


def test_init():
    for field in [EmptyField, PointsField, PointCloudField]:
        field()
