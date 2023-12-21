import os

from ..src.utils import resolve_path


def test_resolve_path():
    """Test for the resolve_path function.

    This test verifies that the resolve_path function correctly resolves
    a given file path, expanding any user symbols such as '~' and providing
    the real absolute path.

    The test takes a path string containing the user home symbol ('~') and
    passes it to the resolve_path function. It then compares the returned
    path with the expected absolute path constructed using Python's os module.

    The expected behavior is that the resolve_path function should replace
    the '~' symbol with the user's home directory path, and the test asserts
    that the returned path matches the expected absolute path.
    """
    user_home = os.path.expanduser('~')
    path = '~/Documents'
    resolved_path = resolve_path(path)
    assert resolved_path == os.path.join(user_home, 'Documents')
