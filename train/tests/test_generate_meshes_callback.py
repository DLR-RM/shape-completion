from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

from trimesh import Trimesh

from ..src.callbacks import generate_meshes as generate_meshes_module


class _FakeFabric:
    def __init__(self, *, precision: Any = None) -> None:
        self.precision = precision

    def autocast(self):
        return nullcontext()


def test_generate_batch_wraps_single_mesh(monkeypatch) -> None:
    monkeypatch.setattr(generate_meshes_module.lightning, "Fabric", _FakeFabric)
    callback = generate_meshes_module.GenerateMeshesCallback()
    callback._generator = cast(Any, SimpleNamespace(generate_mesh=lambda *_args, **_kwargs: Trimesh()))

    meshes = callback.generate_batch({"category.name": ["chair"]})

    assert len(meshes) == 1
    assert isinstance(meshes[0], Trimesh)


def test_generate_batch_preserves_mesh_list(monkeypatch) -> None:
    monkeypatch.setattr(generate_meshes_module.lightning, "Fabric", _FakeFabric)
    callback = generate_meshes_module.GenerateMeshesCallback()
    callback._generator = cast(Any, SimpleNamespace(generate_mesh=lambda *_args, **_kwargs: [Trimesh(), Trimesh()]))

    meshes = callback.generate_batch({"category.name": ["chair"]})

    assert len(meshes) == 2
    assert all(isinstance(mesh, Trimesh) for mesh in meshes)


def test_setup_prefers_ema_model_and_sets_upsampling(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _RecorderGenerator:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def generate_mesh(self, *_args: Any, **_kwargs: Any) -> Trimesh:
            return Trimesh()

    monkeypatch.setattr(generate_meshes_module.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(generate_meshes_module, "Generator", _RecorderGenerator)

    callback = generate_meshes_module.GenerateMeshesCallback(resolution=64, points_batch_size=64)
    ema_model = SimpleNamespace(resolution=256)
    pl_module = SimpleNamespace(
        model=SimpleNamespace(orig_mod=SimpleNamespace(resolution=64)),
        ema_model=SimpleNamespace(module=SimpleNamespace(orig_mod=ema_model)),
    )

    callback.setup(cast(Any, SimpleNamespace()), cast(Any, pl_module), "fit")

    assert callback.resolution == 256
    assert callback.points_batch_size == 256**3
    assert captured["model"] is ema_model
    assert captured["resolution"] == 256
    assert captured["points_batch_size"] == 256**3
    assert captured["upsampling_steps"] == 3
    assert isinstance(callback.generator, _RecorderGenerator)


def test_setup_uses_existing_resolution_when_model_has_none(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _RecorderGenerator:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def generate_mesh(self, *_args: Any, **_kwargs: Any) -> Trimesh:
            return Trimesh()

    monkeypatch.setattr(generate_meshes_module.lightning, "Fabric", _FakeFabric)
    monkeypatch.setattr(generate_meshes_module, "Generator", _RecorderGenerator)

    callback = generate_meshes_module.GenerateMeshesCallback(resolution=128, points_batch_size=16)
    base_model = SimpleNamespace()
    pl_module = SimpleNamespace(
        model=SimpleNamespace(orig_mod=base_model),
        ema_model=None,
    )

    callback.setup(cast(Any, SimpleNamespace()), cast(Any, pl_module), "fit")

    assert callback.resolution == 128
    assert callback.points_batch_size == 16
    assert captured["model"] is base_model
    assert captured["resolution"] == 128
    assert captured["points_batch_size"] == 16
    assert captured["upsampling_steps"] == 2
