from typing import TYPE_CHECKING

from .src.generator import Generator

if TYPE_CHECKING:
    from .src.renderer import Renderer as Renderer

__all__ = ["Generator", "Renderer"]


def __getattr__(name: str):
    if name == "Renderer":
        from .src.renderer import Renderer

        return Renderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
