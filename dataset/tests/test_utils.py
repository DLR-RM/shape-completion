from ..src.utils import get_file


def test_get_file():
    for name in ["bunny", "armadillo"]:
        file = get_file(name)
        assert file.exists()
