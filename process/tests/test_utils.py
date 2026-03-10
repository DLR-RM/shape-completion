import numpy as np

from utils import git_submodule_path, stdout_redirected

from ..src.utils import apply_meshlab_filters, load_scripts


def test_apply_meshlab_filters():
    vertices = np.random.rand(100, 3)
    faces = np.random.randint(0, 100, (100, 3))
    scripts = load_scripts(
        git_submodule_path("process") / "assets" / "meshlab_filter_scripts", num_vertices=len(vertices)
    )
    with stdout_redirected():
        vertices, faces = apply_meshlab_filters(vertices, faces, scripts)
    assert vertices is not None and faces is not None
