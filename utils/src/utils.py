import hashlib
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import cache, partial, wraps
from glob import glob
from io import BytesIO
from logging import DEBUG
from pathlib import Path
from pprint import pprint
from random import shuffle
from typing import IO, Any, Literal, cast

import h5py
import lovely_tensors as lt
import numpy as np
import plotly.colors as plotly_colors
import torch
import torch.distributed as dist
import trimesh
from hydra.core.hydra_config import HydraConfig
from joblib import Parallel, cpu_count
from lightning import seed_everything
from lightning.fabric.utilities.seed import log
from matplotlib.cm import get_cmap
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch import Size, Tensor
from tqdm import tqdm
from trimesh import Trimesh

from .logging import DEBUG_LEVEL_1, DEBUG_LEVEL_2, set_log_level, setup_logger

logger = setup_logger(__name__)
ExceptionTypes = type[Exception] | tuple[type[Exception], ...] | list[type[Exception]]
ItemTypes = int | float | str | np.ndarray | Tensor
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PARTNET_COLORS = np.array(
    [
        [0.65, 0.95, 0.05],
        [0.35, 0.05, 0.35],
        [0.65, 0.35, 0.65],
        [0.95, 0.95, 0.65],
        [0.95, 0.65, 0.05],
        [0.35, 0.05, 0.05],
        [0.65, 0.05, 0.05],
        [0.65, 0.35, 0.95],
        [0.05, 0.05, 0.65],
        [0.65, 0.05, 0.35],
        [0.05, 0.35, 0.35],
        [0.65, 0.65, 0.35],
        [0.35, 0.95, 0.05],
        [0.05, 0.35, 0.65],
        [0.95, 0.95, 0.35],
        [0.65, 0.65, 0.65],
        [0.95, 0.95, 0.05],
        [0.65, 0.35, 0.05],
        [0.35, 0.65, 0.05],
        [0.95, 0.65, 0.95],
        [0.95, 0.35, 0.65],
        [0.05, 0.65, 0.95],
        [0.65, 0.95, 0.65],
        [0.95, 0.35, 0.95],
        [0.05, 0.05, 0.95],
        [0.65, 0.05, 0.95],
        [0.65, 0.05, 0.65],
        [0.35, 0.35, 0.95],
        [0.95, 0.95, 0.95],
        [0.05, 0.05, 0.05],
        [0.05, 0.35, 0.95],
        [0.65, 0.95, 0.95],
        [0.95, 0.05, 0.05],
        [0.35, 0.95, 0.35],
        [0.05, 0.35, 0.05],
        [0.05, 0.65, 0.35],
        [0.05, 0.95, 0.05],
        [0.95, 0.65, 0.65],
        [0.35, 0.95, 0.95],
        [0.05, 0.95, 0.35],
        [0.95, 0.35, 0.05],
        [0.65, 0.35, 0.35],
        [0.35, 0.95, 0.65],
        [0.35, 0.35, 0.65],
        [0.65, 0.95, 0.35],
        [0.05, 0.95, 0.65],
        [0.65, 0.65, 0.95],
        [0.35, 0.05, 0.95],
        [0.35, 0.65, 0.95],
        [0.35, 0.05, 0.65],
    ]
)

PLOTLY_COLORS = (
    np.array(
        [plotly_colors.unlabel_rgb(h) for h in plotly_colors.qualitative.Pastel]
        + [plotly_colors.unlabel_rgb(h) for h in plotly_colors.qualitative.Pastel1]
        + [plotly_colors.unlabel_rgb(h) for h in plotly_colors.qualitative.Pastel2]
        + [plotly_colors.hex_to_rgb(h) for h in plotly_colors.qualitative.Plotly]
    )
    / 255.0
)


class TrackingDictConfig(DictConfig):
    def __init__(self, content, parent=None, ref_type=None, key=None):
        super().__init__(content, parent=parent, ref_type=ref_type, key=key)
        object.__setattr__(self, "accessed_keys", set())
        object.__setattr__(self, "added_params", set())
        for override in HydraConfig.get().overrides.task:
            if override.startswith("+"):
                param_name = override[1:].split("=")[0]
                object.__getattribute__(self, "added_params").add(param_name)

    def __getitem__(self, key):
        object.__getattribute__(self, "accessed_keys").add(key)
        value = super().__getitem__(key)
        if isinstance(value, DictConfig):
            return TrackingDictConfig(value)
        return value

    def get(self, key, default_value=None):
        object.__getattribute__(self, "accessed_keys").add(key)
        value = super().get(key, default_value)
        if isinstance(value, DictConfig):
            return TrackingDictConfig(value)
        return value


def disable_multithreading():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TBB_NUM_THREADS"] = "1"


def unsqueeze_as(tensor: Tensor, other: Tensor | Size | tuple[int, ...]) -> Tensor:
    ndim = other.ndim if torch.is_tensor(other) else len(other)
    return tensor.view(-1, *([1] * (ndim - 1)))


def get_file_descriptor(file_or_fd: int | IO[Any]) -> int:
    """Returns the file descriptor of the given file.

    :param file_or_fd: Either a file or a file descriptor. If a file descriptor is given, it is returned directly.
    :return: The file descriptor of the given file.
    """
    if hasattr(file_or_fd, "fileno"):
        fd = cast(IO[Any], file_or_fd).fileno()
    else:
        fd = file_or_fd
    if not isinstance(fd, int):
        raise AttributeError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(to: int | IO[Any] | str = os.devnull, enabled: bool = True) -> Iterator[IO[Any]]:
    """Redirects all stdout to the given file.

    From https://stackoverflow.com/a/22434262.

    :param to: The file which should be the new target for stdout. Can be a path, file or file descriptor.
    :param enabled: If False, then this context manager does nothing.
    :return: The old stdout output.
    """
    if enabled:
        stdout = sys.stdout
        stdout_fd = get_file_descriptor(stdout)
        # copy stdout_fd before it is overwritten
        # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
        with os.fdopen(os.dup(stdout_fd), "w") as copied:
            stdout.flush()  # flush library buffers that dup2 knows nothing about
            if isinstance(to, (str, os.PathLike)):
                with open(to, "wb") as to_file:
                    os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
            else:
                os.dup2(get_file_descriptor(to), stdout_fd)  # $ exec >&to
            try:
                yield copied
            finally:
                # restore stdout to its previous value
                # NOTE: dup2 makes stdout_fd inheritable unconditionally
                stdout.flush()
                os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
    else:
        yield sys.stdout


def load_mesh(path: str | Path, load_with: str | None = None, **kwargs: Any) -> tuple[np.ndarray, np.ndarray | None]:
    assert load_with in [None, "trimesh", "open3d", "pymeshlab"], (
        f"'load_with' must be one of [None, 'trimesh', 'open3d', 'pymeshlab'], got {load_with}."
    )

    path = Path(path).expanduser().resolve()

    def load_trimesh() -> tuple[np.ndarray, np.ndarray | None]:
        import trimesh

        mesh = trimesh.load(
            path,
            force=kwargs.get("force", "mesh"),
            process=kwargs.get("process", False),
            validate=kwargs.get("validate", False),
        )
        if isinstance(mesh, trimesh.Trimesh):
            return mesh.vertices, mesh.faces
        elif isinstance(mesh, trimesh.PointCloud):
            return mesh.vertices, None
        elif isinstance(mesh, list):
            if not mesh:
                return np.empty((0, 3)), np.empty((0, 3))
            elif len(mesh) == 1:
                return mesh[0].vertices, mesh[0].faces
            return np.concatenate([m.vertices for m in mesh]), np.concatenate([m.faces for m in mesh])
        elif isinstance(mesh, trimesh.Scene):
            vertices = np.concatenate([m.vertices for m in mesh.geometry.values()])
            faces = np.concatenate([m.faces for m in mesh.geometry.values()])
            return vertices, faces
        else:
            raise ValueError(f"Unsupported mesh type: {type(mesh)}")

    def load_open3d() -> tuple[np.ndarray, np.ndarray]:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(
            str(path),
            enable_post_processing=kwargs.get("enable_post_processing", False),
            print_progress=kwargs.get("print_progress", False),
        )
        return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    def load_pymeshlab() -> tuple[np.ndarray, np.ndarray]:
        import pymeshlab

        ms = cast(Any, pymeshlab).MeshSet()
        ms.load_new_mesh(str(path))
        mesh = ms.current_mesh()
        return mesh.vertex_matrix(), mesh.face_matrix()

    if load_with == "trimesh":
        return load_trimesh()
    elif load_with == "open3d":
        return load_open3d()
    elif load_with == "pymeshlab":
        return load_pymeshlab()
    else:
        try:
            return load_trimesh()
        except (ValueError, ImportError) as e:
            if isinstance(e, ImportError):
                logger.warning("Could not import Trimesh. Falling back to Open3D.")
            elif isinstance(e, ValueError):
                logger.warning("Could not load mesh with Trimesh. Falling back to Open3D.")
            try:
                return load_open3d()
            except ImportError:
                logger.warning("Could not import Open3D. Falling back to PyMeshlab.")
                try:
                    return load_pymeshlab()
                except ImportError:
                    logger.warning("Could not import PyMeshLab.")
                    logger.error("Could not load mesh.")
                    raise


def save_mesh(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    save_with: str | None = None,
    **kwargs: Any,
):
    path = Path(path).expanduser().resolve()
    assert path.suffix in [".stl", ".off", ".ply", ".obj"]
    assert save_with in [None, "trimesh", "open3d", "pymeshlab"], (
        f"'save_with' must be one of [None, 'trimesh', 'open3d', 'pymeshlab'], got {save_with}."
    )

    def save_trimesh():
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            face_normals=None if normals is None or len(normals) != len(faces) else normals,
            vertex_normals=None if normals is None or len(normals) != len(vertices) else normals,
            face_colors=None if colors is None or len(colors) != len(faces) else colors,
            vertex_colors=None if colors is None or len(colors) != len(vertices) else colors,
            process=kwargs.get("process", False),
            validate=kwargs.get("validate", False),
        )
        if path.suffix == ".obj":
            mesh.export(
                str(path),
                include_normals=kwargs.get("include_normals", False if normals is None else True),
                include_color=kwargs.get("include_color", False if colors is None else True),
                include_texture=kwargs.get("include_texture", False),
                return_texture=kwargs.get("return_texture", False),
                write_texture=kwargs.get("write_texture", False),
                digits=kwargs.get("digits", 8),
            )
        elif path.suffix == ".ply":
            mesh.export(
                str(path),
                encoding=kwargs.get("encoding", "binary"),
                vertex_normal=kwargs.get("vertex_normal", False if normals is None else True),
                include_attributes=kwargs.get("include_attributes", False),
            )
        elif path.suffix == ".off":
            mesh.export(str(path), digits=kwargs.get("digits", 10))
        elif path.suffix == ".stl":
            mesh.export(str(path), mode=kwargs.get("mode", "binary"))

    def save_open3d():
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        o3d.io.write_triangle_mesh(
            str(path),
            mesh,
            write_ascii=kwargs.get("write_ascii", False),
            compressed=kwargs.get("compressed", True),
            write_vertex_normals=kwargs.get("write_vertex_normals", False),
            write_vertex_colors=kwargs.get("write_vertex_colors", False),
            write_triangle_uvs=kwargs.get("write_triangle_uvs", False),
        )

    def save_pymeshlab():
        import pymeshlab

        ms = cast(Any, pymeshlab).MeshSet()
        pymesh = cast(Any, pymeshlab).Mesh(vertex_matrix=vertices, face_matrix=faces)
        ms.add_mesh(pymesh)
        save_current_mesh = partial(ms.save_current_mesh, save_textures=kwargs.get("save_textures", False))
        if path.suffix == ".stl":
            save_current_mesh = partial(
                save_current_mesh,
                binary=kwargs.get("binary", True),
                save_face_color=kwargs.get("save_face_color", False),
            )
        if path.suffix in [".off", ".ply", ".obj"]:
            save_current_mesh = partial(
                save_current_mesh,
                save_vertex_color=kwargs.get("save_vertex_color", False),
                save_vertex_coord=kwargs.get("save_vertex_coord", False),
                save_face_color=kwargs.get("save_face_color", False),
                save_polygonal=kwargs.get("save_polygonal", False),
            )
        if path.suffix == ".ply":
            save_current_mesh = partial(
                save_current_mesh,
                binary=kwargs.get("binary", True),
                save_vertex_quality=kwargs.get("save_vertex_quality", False),
                save_vertex_normal=kwargs.get("save_vertex_normal", False),
                save_vertex_radius=kwargs.get("save_vertex_radius", False),
                save_face_quality=kwargs.get("save_face_quality", False),
                save_wedge_color=kwargs.get("save_wedge_color", False),
                save_wedge_texcoord=kwargs.get("save_wedge_texcoord", False),
                save_wedge_normal=kwargs.get("save_wedge_normal", False),
            )
        elif path.suffix == ".obj":
            save_current_mesh = partial(
                save_current_mesh,
                save_vertex_normal=kwargs.get("save_vertex_normal", False),
                save_wedge_texcoord=kwargs.get("save_wedge_texcoord", False),
                save_wedge_normal=kwargs.get("save_wedge_normal", False),
            )
        save_current_mesh(file_name=str(path))

    if save_with == "trimesh":
        save_trimesh()
    elif save_with == "open3d":
        save_open3d()
    elif save_with == "pymeshlab":
        save_pymeshlab()
    else:
        try:
            save_trimesh()
        except ImportError:
            logger.warning("Could not import Trimesh. Falling back to Open3D.")
            try:
                save_open3d()
            except ImportError:
                logger.warning("Could not import Open3D. Falling back to PyMeshlab.")
                try:
                    save_pymeshlab()
                except ImportError:
                    logger.warning("Could not import PyMeshLab.")
                    logger.error("Could not save mesh.")
                    raise


@contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Context manager to patch joblib to report into a tqdm progress bar.

    This function patches the `print_progress` method of joblib's `Parallel` class to update the
    provided tqdm progress bar object each time a job is completed. When the context is exited,
    the original `print_progress` method is restored and the tqdm progress bar is closed.

    Args:
        tqdm_object (tqdm): An instance of a tqdm progress bar.

    Yields:
        tqdm: The same tqdm object, updated with progress from the joblib parallel execution.

    Examples:
        >>> with tqdm_joblib(tqdm(total=10)) as pbar:
        >>>     Parallel(n_jobs=2)(delayed(some_function)(i) for i in range(10))
    """

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = Parallel.print_progress
    Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        Parallel.print_progress = original_print_progress
        tqdm_object.close()


def stack_images(images: list[np.ndarray | Image.Image]) -> np.ndarray:
    """
    Stacks a list of images into a square grid layout.

    If the list contains a single image, it is returned as is. Otherwise, the images are stacked
    into a square grid layout. The number of rows and columns in the grid is the square root of
    the number of images in the list (rounded to an integer).

    Args:
        images (List[np.ndarray]): A list of 2D or 3D numpy ndarrays representing images. All
        images should have the same dimensions.

    Returns:
        np.ndarray: An ndarray representing the grid of images. Its dimensions are
        (height*rows, width*cols, [3]) where height and width are the dimensions of a single image,
        and rows and cols are the numbers of rows and columns in the grid.

    Raises:
        ValueError: If any image in the list has a different shape than the others.

    Example:
        >>> image1 = np.random.rand(10, 10)
        >>> image2 = np.random.rand(10, 10)
        >>> result = stack_images([image1, image2])
        >>> print(result.shape)
        (20, 10)
    """
    if len(images) == 1:
        return np.asarray(images[0])
    images_np: list[np.ndarray] = [np.asarray(image) for image in images]
    num_rows_cols = int(np.sqrt(len(images_np)))
    rows: list[np.ndarray] = []
    for step in range(0, len(images_np), num_rows_cols):
        rows.append(np.hstack(images_np[step : step + num_rows_cols]))
    return np.vstack(rows)


def normalize(
    val: np.ndarray | Tensor, a: float = 0, b: float = 1, p_min: float | None = None, p_max: float | None = None
) -> np.ndarray | Tensor:
    is_tensor = torch.is_tensor(val)
    if is_tensor:
        if not val.isfinite().all():
            raise ValueError("Input tensor contains non-finite values.")
        if not val.any():
            return val
    else:
        if not np.isfinite(val).all():
            raise ValueError("Input array contains non-finite values.")
        if not np.any(val):
            return val

    if is_tensor:
        tensor = cast(Tensor, val)
        min_val_t = tensor.min()
        max_val_t = tensor.max()
        if p_min is not None:
            min_val_t = torch.quantile(tensor, p_min)
        if p_max is not None:
            max_val_t = torch.quantile(tensor, p_max)
        tensor = torch.clamp(tensor, min_val_t, max_val_t)
        if bool((min_val_t == max_val_t).item()):
            return (tensor / max_val_t) * a if bool((max_val_t != 0).item()) else tensor
        return (tensor - min_val_t) / (max_val_t - min_val_t) * (b - a) + a

    array = cast(np.ndarray, val)
    min_val_n = float(np.min(array))
    max_val_n = float(np.max(array))
    if p_min is not None:
        min_val_n = float(np.percentile(array, p_min * 100))
    if p_max is not None:
        max_val_n = float(np.percentile(array, p_max * 100))
    array = np.clip(array, min_val_n, max_val_n)
    if min_val_n == max_val_n:
        return (array / max_val_n) * a if max_val_n != 0 else array
    return (array - min_val_n) / (max_val_n - min_val_n) * (b - a) + a


def to_tensor(
    x: int | float | tuple | list | np.ndarray | Tensor,
    unsqueeze: bool = True,
    device: str | torch.device | None = DEFAULT_DEVICE,
) -> Any:
    if isinstance(x, (list, tuple)):
        return [to_tensor(item, unsqueeze=unsqueeze, device=device) for item in x]
    if not isinstance(x, (int, float, np.ndarray, Tensor)):
        return x
    if not torch.is_tensor(x):
        x = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else torch.tensor(x)
        if unsqueeze:
            x = x.unsqueeze(0)
    return x if device is None else x.to(device)


def to_numpy(x: int | float | str | np.ndarray | Tensor | list[str | np.ndarray | Tensor], squeeze: bool = True) -> Any:
    if isinstance(x, list):
        return [to_numpy(item, squeeze=squeeze) for item in x]
    if not torch.is_tensor(x):
        return x
    if squeeze:
        x = x.squeeze(0)
    x = x.float().detach().cpu()
    if x.ndim == 0:
        return x.item()
    return x.numpy()


def filter_dict(d: dict[Any, Any], keep: set[str] | None = None, remove: set[str] | None = None) -> dict[Any, Any]:
    if keep and remove:
        return {k: v for k, v in d.items() if k in keep and k not in remove}
    if keep:
        return {k: v for k, v in d.items() if k in keep}
    if remove:
        return {k: v for k, v in d.items() if k not in remove}
    return d


def rot_from_euler(axes: str, upper_hemisphere: bool) -> tuple[np.ndarray, float]:
    pitch = 0
    angles = np.random.uniform(0, 360, size=len(axes))
    if "x" in axes:
        pitch = angles[list(axes).index("x")]
        if upper_hemisphere:
            pitch = np.random.uniform(0, 90)
            angles[list(axes).index("x")] = pitch
    rot = R.from_euler(axes, angles[0] if len(angles) == 1 else angles, degrees=True).as_matrix()
    # rot = rotation_matrix(axes, angles)
    return rot, pitch


def generate_random_basis(n_points: int = 1024, n_dims: int = 3, radius: float = 0.5, seed: int | None = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_points, n_dims))
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape((-1, 1))
    x_unit = x / x_norms

    r = rng.uniform(size=(n_points, 1))
    u = np.power(r, 1.0 / n_dims)
    x = radius * x_unit * u

    return x


def subsample_indices(data: np.ndarray | list | tuple, num_samples: int, replace: bool = False) -> np.ndarray:
    sample_range = len(data)
    if sample_range == num_samples or num_samples == 0:
        return np.random.permutation(sample_range)
    if replace or sample_range < num_samples:
        return np.random.randint(sample_range, size=num_samples)
    return np.random.choice(sample_range, size=num_samples, replace=False)


def inv_trafo(trafo: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """
    Computes the inverse of a 4x4 homogeneous transformation matrix (or a batch of them).

    Args:
        trafo: A (4, 4) or (B, 4, 4) transformation matrix (PyTorch Tensor or NumPy ndarray).
               Assumes the bottom row is [0, 0, 0, 1].

    Returns:
        The inverse transformation matrix (or batch of matrices) with the same type as input.
    """
    is_tensor = torch.is_tensor(trafo)
    if not is_tensor and not isinstance(trafo, np.ndarray):
        raise TypeError(f"Input must be a torch.Tensor or np.ndarray, got {type(trafo)}")

    if trafo.shape[-2:] != (4, 4):
        raise ValueError(f"Input matrices must be of shape (..., 4, 4), got {trafo.shape}")

    if is_tensor:
        trafo_t = cast(Tensor, trafo)
        rot = trafo_t[..., :3, :3]
        trans = trafo_t[..., :3, 3]
        rot_t = rot.transpose(-1, -2)
        identity = torch.eye(3).to(rot)
        if rot.ndim == 3:
            identity = identity.unsqueeze(0).expand_as(rot)
        if not torch.allclose(rot @ rot_t, identity, atol=1e-6):
            return torch.inverse(trafo_t)
        if trafo_t.ndim == 3:
            inverse = torch.eye(4).unsqueeze(0).expand_as(trafo_t).to(trafo_t)
            inverse[..., :3, :3] = rot_t
            inverse[..., :3, 3] = (-rot_t @ trans.unsqueeze(-1)).squeeze(-1)
        elif trafo_t.ndim == 2:
            inverse = torch.eye(4).to(trafo_t)
            inverse[:3, :3] = rot_t
            inverse[:3, 3] = -rot_t @ trans
        else:
            raise ValueError(f"Input must have 2 or 3 dimensions, got {trafo_t.ndim}")
        return inverse
    else:
        trafo_n = cast(np.ndarray, trafo)
        rot = trafo_n[..., :3, :3]
        trans = trafo_n[..., :3, 3]
        rot_t = rot.swapaxes(-1, -2)
        identity = np.eye(3, dtype=rot.dtype)
        if rot.ndim == 3:
            identity = np.broadcast_to(identity, rot.shape)
        if not np.allclose(rot @ rot_t, identity, atol=1e-6):
            return np.linalg.inv(trafo_n)
        if trafo_n.ndim == 3:
            inverse = np.broadcast_to(np.eye(4), trafo.shape)
            inverse[..., :3, :3] = rot_t
            inverse[..., :3, 3] = (-rot_t @ trans[..., None]).squeeze(-1)
        elif trafo_n.ndim == 2:
            inverse = np.eye(4)
            inverse[:3, :3] = rot_t
            inverse[:3, 3] = -rot_t @ trans
        else:
            raise ValueError(f"Input must have 2 or 3 dimensions, got {trafo_n.ndim}")
        return inverse


def apply_trafo(points: np.ndarray | Tensor, trafo: np.ndarray | Tensor) -> np.ndarray | Tensor:
    assert points.ndim in [2, 3], "Points must be of shape (N, 3) or (B, N, 3)."
    assert points.shape[-1] == 3, "Points must be of shape (N, 3) or (B, N, 3)."
    assert trafo.ndim in [2, 3], "Trafo must be of shape (4, 4) or (B, 4, 4)."
    assert trafo.shape[-2:] == (4, 4), "Trafo must be of shape (4, 4) or (B, 4, 4)."

    if torch.is_tensor(points) and torch.is_tensor(trafo):
        points_t = cast(Tensor, points)
        trafo_t = cast(Tensor, trafo)
        rot = trafo_t[..., :3, :3]
        trans = trafo_t[..., :3, 3]
        # Equivalent to torch.baddbmm(trans.unsqueeze(-1), rot, points.transpose(-1, -2)).transpose(-1, -2)
        return points_t @ rot.transpose(-1, -2) + trans.unsqueeze(-2)
    points_n = cast(np.ndarray, points)
    trafo_n = cast(np.ndarray, trafo)
    rot = trafo_n[..., :3, :3]
    trans = trafo_n[..., :3, 3]
    return points_n @ rot.T + trans


def invert_intrinsic(intrinsic: np.ndarray | Tensor) -> np.ndarray | Tensor:
    if intrinsic[0, 1] == 0:
        inverse = torch.eye(3).to(intrinsic) if torch.is_tensor(intrinsic) else np.eye(3)
        inverse[0, 0] = 1 / intrinsic[0, 0]
        inverse[1, 1] = 1 / intrinsic[1, 1]
        inverse[0, 2] = -intrinsic[0, 2] / intrinsic[0, 0]
        inverse[1, 2] = -intrinsic[1, 2] / intrinsic[1, 1]
        return inverse
    return torch.linalg.inv(intrinsic) if torch.is_tensor(intrinsic) else np.linalg.inv(intrinsic)


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def resolve_save_dir(cfg: DictConfig) -> Path:
    if not cfg.dirs.get("log"):
        raise ValueError("No 'log' directory specified in the 'conf/dirs/default.yaml.")
    project = str(HydraConfig.get().job.config_name if cfg.log.project is None else cfg.log.project)
    name = str(cfg.log.id or cfg.log.name or cfg.model.arch)
    path = resolve_path(cfg.dirs.log)
    return path / project / name


def resolve_backup_dir(cfg: DictConfig) -> Path:
    save_dir = resolve_save_dir(cfg)
    if cfg.dirs.backup:
        return resolve_path(cfg.dirs.backup) / save_dir.relative_to(resolve_path(cfg.dirs.log))
    return save_dir


def resolve_weights_path(cfg: DictConfig, weights_path: str | Path | None = None) -> Path | None:
    weights_path = weights_path or cfg.model.weights
    if not weights_path:
        return None

    potential_paths = [
        resolve_path(weights_path),
        resolve_save_dir(cfg) / weights_path,
        resolve_path(cfg.dirs.log) / weights_path,
    ]

    if cfg.dirs.backup:
        potential_paths.extend([resolve_backup_dir(cfg) / weights_path, resolve_path(cfg.dirs.backup) / weights_path])

    for path in potential_paths:
        if path.is_file():
            return path

    raise FileNotFoundError(f"Could not find weights file '{weights_path}'")


def resolve_checkpoint_path(cfg: DictConfig) -> str | Path | None:
    if not cfg.model.checkpoint:
        return "last" if cfg.train.resume else None

    version = ""
    if cfg.log.version:
        version = cfg.log.version if isinstance(cfg.log.version, str) else f"version_{cfg.log.version}"

    checkpoint_name = Path(cfg.model.checkpoint).name
    potential_paths = [
        resolve_path(cfg.model.checkpoint),
        resolve_save_dir(cfg) / version / "checkpoints" / checkpoint_name,
        resolve_path(cfg.dirs.log) / checkpoint_name,
    ]

    if cfg.dirs.backup:
        potential_paths.extend(
            [
                resolve_backup_dir(cfg) / version / "checkpoints" / checkpoint_name,
                resolve_path(cfg.dirs.backup) / checkpoint_name,
            ]
        )

    for path in potential_paths:
        path = path.with_suffix(".ckpt")
        if path.exists():
            logger.debug_level_1(f"Resolved checkpoint path to {path}")
            return path

    raise FileNotFoundError(f"Could not find checkpoint file '{cfg.model.checkpoint}'")


def make_3d_grid(
    bb_min: float | int | list[float] | tuple[float, float, float],
    bb_max: float | int | list[float] | tuple[float, float, float],
    shape: int | list[int] | tuple[int, int, int],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    bb_min_xyz: tuple[float, float, float]
    if isinstance(bb_min, (int, float)):
        bb_min_xyz = (float(bb_min), float(bb_min), float(bb_min))
    elif isinstance(bb_min, list):
        bb_min_xyz = cast(tuple[float, float, float], tuple(bb_min))
    else:
        bb_min_xyz = bb_min

    bb_max_xyz: tuple[float, float, float]
    if isinstance(bb_max, (int, float)):
        bb_max_xyz = (float(bb_max), float(bb_max), float(bb_max))
    elif isinstance(bb_max, list):
        bb_max_xyz = cast(tuple[float, float, float], tuple(bb_max))
    else:
        bb_max_xyz = bb_max

    shape_xyz: tuple[int, int, int]
    if isinstance(shape, int):
        shape_xyz = (shape, shape, shape)
    elif isinstance(shape, list):
        shape_xyz = cast(tuple[int, int, int], tuple(shape))
    else:
        shape_xyz = shape

    bx0, bx1, bx2 = bb_min_xyz
    by0, by1, by2 = bb_max_xyz
    sx, sy, sz = shape_xyz
    size = sx * sy * sz

    pxs = torch.linspace(bx0, by0, sx, dtype=dtype)
    pys = torch.linspace(bx1, by1, sy, dtype=dtype)
    pzs = torch.linspace(bx2, by2, sz, dtype=dtype)

    pxs = pxs.view(-1, 1, 1).expand(sx, sy, sz).reshape(size)
    pys = pys.view(1, -1, 1).expand(sx, sy, sz).reshape(size)
    pzs = pzs.view(1, 1, -1).expand(sx, sy, sz).reshape(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def eval_input(
    inputs: Path | list[Path | str],
    in_format: str | None = None,
    recursion_depth: int | None = None,
    sort: bool = False,
) -> list[Path]:
    files: list[Path]
    if isinstance(inputs, list):
        files = [Path(f).expanduser().resolve() for f in inputs]
    else:
        inputs = inputs.expanduser().resolve()
        if inputs.is_file():
            files = [inputs]
        elif inputs.is_dir():
            assert in_format is not None, "If input is a directory, in_format must be set."
            logger.debug(f"Globbing paths from {inputs}.")
            if recursion_depth is None:
                files = list(inputs.rglob(f"*{in_format}"))
            else:
                pattern = f"{'/'.join(['*' for _ in range(recursion_depth)])}/*{in_format}"
                files = list(inputs.glob(pattern))
        else:
            files = [Path(p).expanduser().resolve() for p in glob(str(inputs))]
    shuffle(files)
    if sort:
        files = sorted(files)
    logger.debug(f"Found {len(files)} files.")
    return files


def count_gpus() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=True)
        return len(result.stdout.strip().split("\n"))
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to execute nvidia-smi to detect GPUs.") from e


def get_num_workers(num_workers: int | None = None):
    if num_workers is None or num_workers < 0:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    if num_workers > 1:
        disable_multithreading()
    logger.debug_level_1(f"Loading data with {num_workers} workers")
    return num_workers


def get_device_info() -> tuple[int, int]:
    major, minor = 100, 100
    for device_id in range(torch.cuda.device_count()):
        device = torch.cuda.get_device_name(device_id)
        big, small = torch.cuda.get_device_capability(device_id)
        logger.debug_level_1(f"Using GPU {device} with compute capability {big}.{small}")
        if big > 7:
            torch.set_float32_matmul_precision("high")
            logger.debug("Using high precision matrix multiplication")
        # Return lowest compute capability
        if big < major or (big == major and small < minor):
            major, minor = big, small
    return major, minor


def points_to_coordinates(
    points: Tensor | np.ndarray,
    max_value: int | float | Tensor | np.ndarray | None = None,
    plane: str | None = None,
    intrinsic: Tensor | np.ndarray | None = None,
    clip: bool | tuple[float, float] = False,
) -> Tensor | np.ndarray:
    """Converts points in R3 of range [-max_val / 2, max_val / 2] to plane, uv or grid coordinates of range [0, 1).

    :param points: The points to be converted.
    :param max_value: The maximum range of the coordinates. If None, the points are assumed to be in range [-0.5, 0.5].
    :param plane: The plane to project the points to. If None, 3D coordinates are returned.
    :param intrinsic: The intrinsic camera matrix to project the points to the image plane.
    :param clip: Whether to clip the coordinates of range [0, 1] to the range [0, 1).
    """
    coords = points[..., :3]

    if intrinsic is None:
        if plane is not None:
            if plane == "xy":
                coords = coords[..., [0, 1]]
            elif plane == "xz":
                coords = coords[..., [0, 2]]
            elif plane == "yz":
                coords = coords[..., [1, 2]]
            else:
                raise NotImplementedError(f"{plane} plane not implemented.")
        if max_value is not None:
            coords = coords / max_value
        coords = coords + 0.5
    elif plane == "uv":
        u, v, _ = points_to_uv(coords, intrinsic)
        if isinstance(coords, np.ndarray):
            coords = np.column_stack([u, v]).astype(np.float32)
        else:
            coords = torch.stack([cast(Tensor, u), cast(Tensor, v)], dim=-1).float()
        if max_value is not None:
            if isinstance(max_value, (Tensor, np.ndarray)) and len(max_value.shape) != len(coords.shape):
                max_value = max_value[..., None, None]
            coords = coords / max_value
    else:
        raise NotImplementedError(f"{plane} plane not implemented.")

    if clip:
        clip_min, clip_max = clip if isinstance(clip, (tuple, list)) else (0, 1 - 1e-6)
        if isinstance(coords, Tensor):
            return torch.clamp(coords, clip_min, clip_max)
        if isinstance(coords, np.ndarray):
            return np.clip(coords, clip_min, clip_max)
    return coords


def coordinates_to_index(coordinates: Tensor, resolution: int) -> torch.LongTensor:
    """Converts the given coordinates in range [0, 1] to indices in range [0, resolution ** x - 1] in row-major order."""
    grid_coords = (coordinates * resolution).long().clamp(0, resolution - 1)
    if coordinates.size(-1) == 1:
        index = grid_coords[:, :, 0]
    elif coordinates.size(-1) == 2:
        index = grid_coords[:, :, 0] + resolution * grid_coords[:, :, 1]
    elif coordinates.size(-1) == 3:
        index = grid_coords[:, :, 0] + resolution * (grid_coords[:, :, 1] + resolution * grid_coords[:, :, 2])
    else:
        raise NotImplementedError(f"{coordinates.size(-1)}D coordinates not implemented.")
    return cast(torch.LongTensor, index)


def load_from_binary_hdf5(
    obj_path: Path, file_names: list[str], file_dirs: list[str | None] | None = None
) -> list[np.ndarray | trimesh.Trimesh]:
    files = list()
    if file_dirs is None:
        file_dirs = [None for _ in file_names]
    with h5py.File(obj_path.with_suffix(".hdf5"), "r") as hdf5_file:
        for file_name, file_dir in zip(file_names, file_dirs, strict=False):
            data: Any = hdf5_file
            if file_dir is not None:
                data = hdf5_file[file_dir]
            raw = data[file_name][()]
            if isinstance(raw, np.ndarray):
                raw_bytes = raw.tobytes()
            elif isinstance(raw, (bytes, bytearray, memoryview)):
                raw_bytes = bytes(raw)
            else:
                raw_bytes = bytes(raw)
            data = BytesIO(raw_bytes)
            if file_name[-3:] in ["npy", "npz"]:
                data = np.load(data)
            elif file_name[-3:] in ["off", "ply", "obj"]:
                data = trimesh.load(data, file_type=file_name[-3:], force="mesh")
            elif file_name[-3:] in ["png", "jpg"]:
                data = Image.open(data, mode="r", formats=["PNG", "JPEG"])
                data = np.asarray(
                    data, dtype=np.uint16 if data.mode == "I;16" else np.uint8 if data.mode == "L" else None
                )
            else:
                raise NotImplementedError(f"Unknown file format {file_name}.")
            files.append(data)
    return files


def resolve_out_dir(in_path: Path, in_dir: Path, out_dir: Path, shard: int | None = None) -> Path:
    in_path = in_path.expanduser().resolve()
    in_dir = in_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    shard_name = "" if shard is None else f"{shard:04d}"
    out_dir = (out_dir / in_path.relative_to(in_dir)).parent / shard_name
    logger.debug(f"Resolved output directory to {out_dir}.")
    return out_dir


def monkey_patch(instance: Any, method_name: str, new_method: Any):
    original_method = getattr(instance, method_name, None)
    if original_method is None:
        raise AttributeError(f"Method {method_name} not found in the instance.")

    # Bind the new method to the instance
    bound_method = new_method.__get__(instance, instance.__class__)
    setattr(instance, method_name, bound_method)


def save_command_and_args_to_file(file_path: str | Path, args: Any | None = None):
    file_path = resolve_path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        file.write("Command: " + " ".join(sys.argv) + "\n")
        if args is not None:
            file.write("Arguments:\n")
            for arg, value in vars(args).items():
                file.write(f"{arg}: {value}\n")


def binary_from_multi_class(logits: Tensor, free_label: int | None = None, occ_label: int | None = None) -> Tensor:
    """
    Convert multi-class logits to binary logits where one class is defined as 'free'
    and all other classes are considered 'occupied'.

    Parameters:
    logits (Tensor): The input multi-class logits tensor of shape (..., num_classes).
    free_label (Optional[int]): The index of the 'free' class. If None, occ_label must be provided.
    occ_label (Optional[int]): The index of the 'occupied' class. If None, free_label must be provided.

    Returns:
    Tensor: The resulting binary logits tensor of shape (...).

    Note:
    - Only one of free_label or occ_label can be specified.
    - If both free_label and occ_label are None, the last class is considered 'free'.
    """

    assert free_label is None or occ_label is None, "Only one of free_label and occ_label can be specified."

    if free_label == -1 or (free_label is None and occ_label is None):
        # If free_label is -1 or none of the labels are specified
        free_logits = logits[..., -1]
        occ_logits = logits[..., :-1]
        binary_logits = torch.logsumexp(occ_logits, dim=-1) - free_logits
    elif free_label is None:
        # If only occ_label is specified
        assert occ_label is not None
        occ_logits = logits[..., occ_label]
        free_logits = torch.cat([logits[..., :occ_label], logits[..., occ_label + 1 :]], dim=-1)
        binary_logits = occ_logits - torch.logsumexp(free_logits, dim=-1)
    else:
        # If only free_label is specified
        free_logits = logits[..., free_label]
        occ_logits = torch.cat([logits[..., :free_label], logits[..., free_label + 1 :]], dim=-1)
        binary_logits = torch.logsumexp(occ_logits, dim=-1) - free_logits

    return binary_logits.squeeze(-1)


def check_cfg(cfg: DictConfig, **kwargs: Any):
    def replace_none(_cfg: DictConfig) -> None:
        for key, value in _cfg.items():
            if isinstance(value, DictConfig):
                replace_none(value)
            elif isinstance(value, str) and value.lower() == "none":
                _cfg[key] = None

    replace_none(cfg)

    assert cfg.model.average in [None, "", "none", "ema", "swa"], f"Unknown model averaging method {cfg.model.average}"
    load_surface = cfg.points.load_surface or any(cfg[split].load_surface_points for split in ["train", "val", "test"])
    assert not (cfg.points.load_all and (cfg.points.load_random or load_surface)), (
        "Cannot load all files and a random or surface file at the same time"
    )

    if isinstance(cfg.test.metrics, str):
        cfg.test.metrics = [cfg.test.metrics]

    if cfg.log.progress and cfg.log.verbose > 2:
        logger.debug_level_1("Disabling progress logging in verbose mode.")
        cfg.log.progress = False

    if cfg.log.progress and (cfg.val.visualize or cfg.val.mesh):
        logger.debug_level_1("Disabling 'rich' progress logging in visualization mode.")
        cfg.log.progress = True  # Disable 'rich' progress logging

    input_has_pcd = cfg.inputs.type in [
        "pointcloud",
        "partial",
        "depth_like",
    ] or (
        any(
            t in cfg.inputs.type
            for t in [
                "depth",
                "kinect",
                "rgbd",
            ]
        )
        and cfg.inputs.project
    )
    if not input_has_pcd:
        logger.debug_level_1("Input does not contain point cloud data. Disabling NeRF encoding.")
        cfg.inputs.nerf = False

    cls_only = (cfg.cls.num_classes is not None or cfg.seg.num_classes is not None) and not cfg.cls.occupancy
    if cls_only:
        logger.debug_level_1("Only classification or segmentation. Disabling NeRF encoding for points.")
        cfg.points.nerf = False

    if cfg.points.voxelize:
        logger.debug_level_1("Points are being voxelized. Disabling subsampling and cropping")
        cfg.points.subsample = False
        cfg.points.crop = False
        cfg.vis.resolution = cfg.points.voxelize

    if cfg.implicit.sdf and cfg.implicit.threshold == 0.5:
        logger.debug_level_1("Setting threshold to 0 for SDFs")
        cfg.implicit.threshold = 0

    if cfg.log.verbose == 2 and cfg.log.progress == "rich":
        cfg.log.progress = True

    if cfg.data.frame != "world":
        cfg.load.keys_to_keep.extend(["inputs.extrinsic", "inputs.inv_extrinsic"])
    if cfg.norm.center:
        cfg.load.keys_to_keep.append("inputs.norm_offset")
    if cfg.norm.scale:
        cfg.load.keys_to_keep.append("inputs.norm_scale")
    condition_key = cfg.get("cond_key")
    if condition_key:
        cfg.load.keys_to_keep.append(condition_key)


def setup_config(
    cfg: DictConfig, seed_workers: bool = False, functions: Iterable[Callable] | None = (check_cfg,), **kwargs: Any
) -> DictConfig:
    # cfg = TrackingDictConfig(cfg)

    major, minor = get_device_info()
    if major + minor / 10 < 7.0:
        if any("16" in str(cfg[split].precision) for split in ["train", "val", "test"]):
            cfg.train.precision = cfg.val.precision = cfg.test.precision = "32-true"
            logger.warning("Compute capability below 7.0. Using full precision")
    elif major + minor / 10 < 8.0:
        if any("bf" in str(cfg[split].precision) for split in ["train", "val", "test"]):
            cfg.train.precision = cfg.train.precision.strip("bf")
            cfg.val.precision = cfg.val.precision.strip("bf")
            cfg.test.precision = cfg.test.precision.strip("bf")
            logger.warning("Compute capability below 8.0. Using half precision")

    if cfg.log.pretty:
        lt.monkey_patch()
    log.setLevel(logging.WARNING)
    if cfg.log.verbose > 0:
        log.setLevel(logging.INFO)
        if cfg.log.verbose == 1:
            set_log_level(DEBUG_LEVEL_1)
        elif cfg.log.verbose == 2:
            set_log_level(DEBUG_LEVEL_2)
        elif cfg.log.verbose == 3:
            set_log_level(logging.DEBUG)
            print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.misc.seed, workers=seed_workers)

    if functions is not None:
        for function in functions:
            function(cfg, **kwargs)

    return cfg


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """
    Generates a look-at matrix for a camera (i.e. an inverse camera extrinsic matrix) in OpenGL convention.

    This function creates a transformation matrix that represents a camera looking at a specific target point in 3D space.
    The camera is located at the position specified by the 'eye' parameter, and it is oriented such that its up-vector aligns
    with the 'up' parameter.

    Parameters:
    eye (np.ndarray): A 1D numpy array of shape (3,) representing the 3D coordinates of the camera position.
    target (np.ndarray): A 1D numpy array of shape (3,) representing the 3D coordinates of the target point the camera is looking at.
    up (np.ndarray): A 1D numpy array of shape (3,) representing the up-vector of the camera. Default is np.array([0, 1, 0]) which represents the y-axis.

    Returns:
    np.ndarray: A 4x4 transformation matrix representing the camera's position and orientation (pose) in 3D space.

    """
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    up_vec = np.array([0, 1, 0]) if up is None else up
    x_axis = np.cross(up_vec, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    return np.array(
        [
            [x_axis[0], y_axis[0], z_axis[0], eye[0]],
            [x_axis[1], y_axis[1], z_axis[1], eye[1]],
            [x_axis[2], y_axis[2], z_axis[2], eye[2]],
            [0, 0, 0, 1],
        ]
    )


def convert_coordinates(points: np.ndarray | Tensor, input_format: str, output_format: str) -> np.ndarray | Tensor:
    if input_format.lower() == output_format.lower():
        return points

    # Transformation matrices from OpenGL to other formats
    transforms = {
        "opengl": np.eye(3),
        "opencv": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # Invert y and z axes
        "blender": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),  # Swap y and z axes, invert y axis
    }

    # Validate formats
    assert input_format.lower() in transforms, f"Unknown input format: {input_format}"
    assert output_format.lower() in transforms, f"Unknown output format: {output_format}"

    # Validate shape
    assert points.shape[-1] == 3, "Point coordinates must be 3D."
    assert points.ndim in [2, 3], "Point coordinates (N, 3) or (B, N, 3)."

    # Select transformation matrices
    in_trafo = transforms[input_format.lower()]
    out_trafo = transforms[output_format.lower()]

    # Convert to torch.Tensor if necessary
    if torch.is_tensor(points):
        in_trafo = torch.from_numpy(in_trafo).to(points.dtype).to(points.device)
        out_trafo = torch.from_numpy(out_trafo).to(points.dtype).to(points.device)

    # Convert from input format to OpenGL
    points_opengl = points @ in_trafo.T

    # Convert from OpenGL to output format
    return points_opengl @ out_trafo


def convert_extrinsic(extrinsic: np.ndarray, input_format: str, output_format: str) -> np.ndarray:
    inv_extrinsic = cast(np.ndarray, inv_trafo(extrinsic))
    inv_extrinsic[:3, :3] = cast(np.ndarray, convert_coordinates(inv_extrinsic[:3, :3], input_format, output_format))
    return cast(np.ndarray, inv_trafo(inv_extrinsic))


def adjust_intrinsic(
    intrinsic: np.ndarray | Tensor,
    width: int | np.ndarray | Tensor,
    height: int | np.ndarray | Tensor,
    box: tuple[int, int, int, int] | None = None,
    size: int | None = None,
) -> np.ndarray | Tensor:
    if box is None and size is None:
        return intrinsic

    if isinstance(width, int):
        width = torch.tensor(width)
    elif isinstance(width, np.ndarray):
        width = torch.from_numpy(width)
    if isinstance(height, int):
        height = torch.tensor(height)
    elif isinstance(height, np.ndarray):
        height = torch.from_numpy(height)

    if width.unique().numel() == 1 and height.unique().numel() == 1:  # All items in the batch have the same size
        w = width.item() if width.dim() == 0 else width[0].item()
        h = height.item() if height.dim() == 0 else height[0].item()
        if size is None and box == (0, 0, w, h):
            return intrinsic
        if box is None and size == max(w, h):
            return intrinsic

    adjusted_matrix = torch.from_numpy(intrinsic).clone() if isinstance(intrinsic, np.ndarray) else intrinsic.clone()
    if box is not None and adjusted_matrix.dim() == 3:
        raise ValueError("Box cannot be specified for batched intrinsic matrices.")

    crop_width, crop_height = width, height
    if box is not None:
        x1, y1, x2, y2 = box
        crop_width, crop_height = torch.tensor(x2 - x1), torch.tensor(y2 - y1)
        adjusted_matrix[..., 0, 2] -= x1
        adjusted_matrix[..., 1, 2] -= y1

    if size is not None:
        if size <= 0:
            raise ValueError("Size must be a positive integer.")
        max_dim = torch.max(crop_width, crop_height)
        scale = size / torch.clamp(max_dim, min=1).float()
        adjusted_matrix[..., 0, 0] *= scale
        adjusted_matrix[..., 1, 1] *= scale
        adjusted_matrix[..., 0, 2] *= scale
        adjusted_matrix[..., 1, 2] *= scale

    return adjusted_matrix.numpy() if isinstance(intrinsic, np.ndarray) else adjusted_matrix


def crop_and_resize_image(
    image: np.ndarray | Image.Image | Tensor,
    box: tuple[int, int, int, int] | None = None,
    size: int | None = None,
    color: Literal["black", "white"] = "black",
    interpolation: Literal["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"] | None = None,
) -> np.ndarray | Image.Image | Tensor:
    height, width = np.asarray(image).shape[:2]
    if box is None and size is None:
        return image
    elif size is None and box == (0, 0, width, height):
        return image
    elif box is None and size == max(width, height):
        return image

    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    elif isinstance(image, Tensor):
        pil_image = Image.fromarray(image.cpu().numpy())
    else:
        pil_image = image

    if box is not None:
        left, top, right, bottom = box
        crop_box = (max(0, left), max(0, top), min(width, right), min(height, bottom))
        pil_image = pil_image.crop(crop_box)

        box_width, box_height = right - left, bottom - top
        if pil_image.size != (box_width, box_height):
            canvas = Image.new(pil_image.mode, (box_width, box_height), color=color)
            paste_x, paste_y = max(0, -left), max(0, -top)
            canvas.paste(pil_image, (paste_x, paste_y))
            pil_image = canvas

    if size is not None:
        if size <= 0:
            raise ValueError("Size must be > 0.")
        resample_dict = {
            None: None,
            "nearest": Image.Resampling.NEAREST,
            "box": Image.Resampling.BOX,
            "bilinear": Image.Resampling.BILINEAR,
            "hamming": Image.Resampling.HAMMING,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }
        if interpolation not in resample_dict:
            raise ValueError(f"Unknown interpolation method {interpolation}.")
        width, height = cast(Image.Image, pil_image).size
        new_width, new_height = size, size
        # Warning: Larger edge will be set to size (unlike TorchVision's Resize)
        if width > height:
            new_height = int(size * height / width)
        elif height > width:
            new_width = int(size * width / height)
        pil_image = cast(Image.Image, pil_image).resize((new_width, new_height), resample_dict[interpolation])

    if isinstance(image, np.ndarray):
        return np.asarray(pil_image)
    elif isinstance(image, Tensor):
        return torch.from_numpy(np.asarray(pil_image)).to(image)
    return pil_image


def depth_to_image(
    depth_map: np.ndarray,
    cmap: str | None = None,
    a: float = 0.1,
    b: float = 1.0,
    p_min: float | None = 0.01,
    p_max: float | None = 0.99,
) -> Image.Image:
    depth_image = depth_map.copy()
    if depth_image.ndim > 2:
        depth_image = depth_image.squeeze()

    empty_mask = depth_image == 0
    finite_mask = np.isfinite(depth_image)
    valid_pixels_mask = ~empty_mask & finite_mask
    if np.any(valid_pixels_mask):
        depth_image[valid_pixels_mask] = normalize(depth_image[valid_pixels_mask], a=a, b=b, p_min=p_min, p_max=p_max)

    if cmap is None:
        depth_image = np.broadcast_to(depth_image[:, :, np.newaxis], (*depth_image.shape, 3))
    else:
        cmap_fn = get_cmap(cmap)
        depth_image = cmap_fn(depth_image)
        depth_image[empty_mask] = 0
    depth_image = Image.fromarray((depth_image[..., :3] * 255).astype(np.uint8))
    return depth_image


def bbox_from_mask(mask: np.ndarray | Tensor, padding: int | float = 0) -> tuple[int, int, int, int]:
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if np.all(mask == 0):
        height, width = mask.shape
        s = min(height, width)
        left = (width - s) // 2
        top = (height - s) // 2
        right = left + s
        bottom = top + s
        return left, top, right, bottom

    v, u = np.nonzero(mask)
    u_size, v_size = max(1, np.ptp(u)), max(1, np.ptp(v))
    size = max(u_size, v_size)

    size += 2 * size * padding if isinstance(padding, float) else 2 * padding

    u_center, v_center = u.min() + u_size / 2, v.min() + v_size / 2

    half_size = size / 2
    left = int(np.round(u_center - half_size))
    top = int(np.round(v_center - half_size))
    right = int(np.round(u_center + half_size))
    bottom = int(np.round(v_center + half_size))

    width_diff = right - left
    height_diff = bottom - top
    size_diff = width_diff - height_diff
    if size_diff < 0:  # height > width
        right += abs(size_diff)
    elif size_diff > 0:  # width > height
        bottom += size_diff

    return left, top, right, bottom


def depth_to_points(
    depth: np.ndarray | Tensor,
    intrinsic: np.ndarray | Tensor,
    depth_scale: float | None = None,
    depth_trunc: float | None = None,
    vectorized: bool = True,
) -> np.ndarray | Tensor:
    if torch.is_tensor(depth):
        depth_t = cast(Tensor, depth)
        intrinsic_t = cast(Tensor, intrinsic)
        z_t = torch.nan_to_num(depth_t, nan=0, posinf=0, neginf=0)
        v_t, u_t = torch.nonzero(z_t, as_tuple=True)

        z_t = z_t[v_t, u_t]
        if depth_scale:
            z_t = z_t / depth_scale
        if depth_trunc:
            mask_t = z_t <= depth_trunc
            u_t = u_t[mask_t]
            v_t = v_t[mask_t]
            z_t = z_t[mask_t]

        if vectorized:
            points_t = torch.stack(
                (
                    u_t.to(dtype=intrinsic_t.dtype),
                    v_t.to(dtype=intrinsic_t.dtype),
                    torch.ones_like(u_t, dtype=intrinsic_t.dtype),
                ),
                dim=-1,
            )
            inv_intr_t = cast(Tensor, invert_intrinsic(intrinsic_t)).transpose(-1, -2)
            points_t = points_t @ inv_intr_t
            points_t = points_t * z_t[:, None]
        else:
            f_x, f_y = intrinsic_t[0, 0], intrinsic_t[1, 1]
            c_x, c_y = intrinsic_t[0, 2], intrinsic_t[1, 2]
            x_t = (u_t - c_x) * z_t / f_x
            y_t = (v_t - c_y) * z_t / f_y
            points_t = torch.stack((x_t, y_t, z_t), dim=-1)
        return points_t

    depth_n = cast(np.ndarray, depth)
    intrinsic_n = cast(np.ndarray, intrinsic)
    z_n = np.nan_to_num(depth_n, nan=0, posinf=0, neginf=0)
    v_n, u_n = np.nonzero(z_n)

    z_n = z_n[v_n, u_n]
    if depth_scale:
        z_n = z_n / depth_scale
    if depth_trunc:
        mask_n = z_n <= depth_trunc
        u_n = u_n[mask_n]
        v_n = v_n[mask_n]
        z_n = z_n[mask_n]

    if vectorized:
        points_n = np.column_stack([u_n, v_n, np.ones_like(u_n)])
        inv_intr_t_n = cast(np.ndarray, invert_intrinsic(intrinsic_n)).T
        points_n = points_n @ inv_intr_t_n
        points_n = points_n * z_n[:, None]
    else:
        f_x, f_y = intrinsic_n[0, 0], intrinsic_n[1, 1]
        c_x, c_y = intrinsic_n[0, 2], intrinsic_n[1, 2]
        x_n = (u_n - c_x) * z_n / f_x
        y_n = (v_n - c_y) * z_n / f_y
        points_n = np.column_stack([x_n, y_n, z_n])
    return points_n


def points_to_uv(
    points: np.ndarray | Tensor,
    intrinsic: np.ndarray | Tensor,
    extrinsic: np.ndarray | Tensor | None = None,
    width: int | None = None,
    height: int | None = None,
    orthogonal: bool = False,
) -> tuple[np.ndarray | Tensor, np.ndarray | Tensor, np.ndarray | Tensor | None]:
    batched = points.ndim == 3
    is_tensor = torch.is_tensor(points)

    if not batched:
        points = points[None, ...]
    if intrinsic.ndim == 2:
        intrinsic = intrinsic[None, ...]
    if extrinsic is None:
        if is_tensor:
            points_t = cast(Tensor, points)
            extrinsic = torch.eye(4, dtype=points_t.dtype, device=points_t.device)
        else:
            extrinsic = np.eye(4)
    if extrinsic.ndim == 2:
        if is_tensor:
            points_t = cast(Tensor, points)
            extrinsic = cast(Tensor, extrinsic)[None, ...].expand(points_t.size(0), -1, -1).to(points_t)
        else:
            points_n = cast(np.ndarray, points)
            extrinsic = cast(np.ndarray, extrinsic)[None, ...].repeat(points_n.shape[0], axis=0)

    if is_tensor:
        points_t = cast(Tensor, points)
        intrinsic_t = cast(Tensor, intrinsic)
        extrinsic_t = cast(Tensor, extrinsic)
        points_h = torch.cat([points_t, torch.ones(*points_t.shape[:2], 1).to(points_t)], dim=-1)
        xyz = (intrinsic_t @ extrinsic_t[:, :3, :] @ points_h.transpose(-1, -2)).transpose(-1, -2)
    else:
        points_n = cast(np.ndarray, points)
        intrinsic_n = cast(np.ndarray, intrinsic)
        extrinsic_n = cast(np.ndarray, extrinsic)
        points_h = np.concatenate([points_n, np.ones((*points_n.shape[:2], 1))], axis=-1)
        xyz = (intrinsic_n @ extrinsic_n[:, :3, :] @ points_h.transpose(0, 2, 1)).transpose(0, 2, 1)

    if not batched:
        xyz = xyz.squeeze(0)

    uv = xyz[..., :2]
    if not orthogonal:
        uv = uv / xyz[..., 2:]

    if is_tensor:
        uv_t = cast(Tensor, uv)
        u_t = uv_t[..., 0].round().long()
        v_t = uv_t[..., 1].round().long()
        if width is not None and height is not None:
            mask_t = (u_t >= 0) & (u_t < width) & (v_t >= 0) & (v_t < height)
            u_t = u_t[mask_t]
            v_t = v_t[mask_t]
            return u_t, v_t, mask_t
        return u_t, v_t, None

    uv_n = cast(np.ndarray, uv)
    u_n = np.round(uv_n[..., 0]).astype(np.int64)
    v_n = np.round(uv_n[..., 1]).astype(np.int64)
    if width is not None and height is not None:
        mask_n = (u_n >= 0) & (u_n < width) & (v_n >= 0) & (v_n < height)
        u_n = u_n[mask_n]
        v_n = v_n[mask_n]
        return u_n, v_n, mask_n
    return u_n, v_n, None


def points_to_depth(
    points: np.ndarray | Tensor,
    intrinsic: np.ndarray | Tensor,
    width: int,
    height: int,
    depth_scale: float | None = None,
    depth_trunc: float | None = None,
) -> np.ndarray | Tensor:
    batched = points.ndim == 3
    if torch.is_tensor(points):
        points_t = cast(Tensor, points)
        u_raw, v_raw, mask_raw = points_to_uv(points_t, intrinsic, width=width, height=height)
        u_t = cast(Tensor, u_raw)
        v_t = cast(Tensor, v_raw)
        mask_t = cast(Tensor | None, mask_raw)

        z_t = points_t[..., 2]
        if mask_t is not None:
            z_t = z_t[mask_t]
        if depth_scale:
            z_t = z_t * depth_scale
        if depth_trunc:
            z_t = torch.clamp(z_t, max=depth_trunc)

        shape = (points_t.shape[0], height, width) if batched else (height, width)
        depth_t = torch.zeros(shape, dtype=points_t.dtype, device=points_t.device)

        if batched:
            u_t = u_t[None, ...]
            v_t = v_t[None, ...]
            z_t = z_t[None, ...]
            batch_index_t = torch.arange(depth_t.shape[0], device=depth_t.device).view(-1, 1)
            depth_t[batch_index_t, v_t, u_t] = z_t
        else:
            depth_t[v_t, u_t] = z_t
        return depth_t

    points_n = cast(np.ndarray, points)
    u_raw, v_raw, mask_raw = points_to_uv(points_n, intrinsic, width=width, height=height)
    u_n = cast(np.ndarray, u_raw)
    v_n = cast(np.ndarray, v_raw)
    mask_n = cast(np.ndarray | None, mask_raw)

    z_n = points_n[..., 2]
    if mask_n is not None:
        z_n = z_n[mask_n]
    if depth_scale:
        z_n = z_n * depth_scale
    if depth_trunc:
        z_n[z_n > depth_trunc] = depth_trunc

    shape = (points_n.shape[0], height, width) if batched else (height, width)
    depth_n = np.zeros(shape)

    if batched:
        u_n = u_n[None, ...]
        v_n = v_n[None, ...]
        z_n = z_n[None, ...]
        batch_index_n = np.arange(depth_n.shape[0]).reshape(-1, 1)
        depth_n[batch_index_n, v_n, u_n] = z_n
    else:
        depth_n[v_n, u_n] = z_n

    return depth_n


def draw_camera(
    intrinsic: np.ndarray,
    extrinsic_opencv: np.ndarray,
    width: int,
    height: int,
    scale: float = 1.0,
    color: tuple[float, float, float] | list[float] | np.ndarray | None = None,
):
    import open3d as o3d

    if color is None:
        color = [0.8, 0.2, 0.8]

    inv_extrinsic = np.linalg.inv(extrinsic_opencv)
    R = inv_extrinsic[:3, :3]
    t = inv_extrinsic[:3, 3]

    # camera model scale
    s = 1 / scale

    # intrinsics
    Ks = np.array(
        [[intrinsic[0, 0] * s, 0, intrinsic[0, 2]], [0, intrinsic[1, 1] * s, intrinsic[1, 2]], [0, 0, intrinsic[2, 2]]]
    )
    Ksinv = np.array([[1 / Ks[0, 0], 0, -Ks[0, 2] / Ks[0, 0]], [0, 1 / Ks[1, 1], -Ks[1, 2] / Ks[1, 1]], [0, 0, 1]])

    # axis
    axis = o3d.geometry.TriangleMesh().create_coordinate_frame(size=scale * 0.25).transform(inv_extrinsic)

    # points in pixel
    points_pixel = np.array(
        [
            [0, 0, 0],  # camera center
            [0, 0, 1],  # lower left
            [width, 0, 1],  # lower right
            [0, height, 1],  # upper left
            [width, height, 1],  # upper right
        ]
    )

    # pixel to camera coordinate system
    points = np.array([scale * Ksinv @ p for p in points_pixel])

    # image plane
    plane_width = points[4, 0] - points[1, 0]
    plane_height = points[4, 1] - points[1, 1]
    if isinstance(color, np.ndarray) and color.shape == (height, width, 3):
        v, u = np.mgrid[:height, :width]
        u = u / width * plane_width
        v = v / height * plane_height
        plane = np.stack((u.flatten(), v.flatten(), 1e-6 * np.ones(u.size)), axis=1)
        plane = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane))
        plane.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3) / 255)
    else:
        plane = o3d.geometry.TriangleMesh.create_box(plane_width, plane_height, depth=1e-6)
        plane.paint_uniform_color(color)
    plane.transform(inv_extrinsic)  # transform to world coordinates
    plane.translate(R @ [points[1][0], points[1][1], scale])

    # pyramid
    points_world = np.array([(R @ p + t) for p in points])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [4, 3], [3, 1], [1, 4], [2, 3]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_world), lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([np.zeros(3) for _ in range(len(lines))])
    points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_world))
    points.paint_uniform_color(np.zeros(3))

    return [axis, plane, line_set, points]


def pitch_from_trafo(trafo: np.ndarray, roll: float | None = None, degrees: bool = True) -> float:
    """
    Calculate the pitch angle (x-axis rotation) from the a homogeneous transformation matrix.

    This function calculates the pitch angle from a homogeneous transformation matrix. The pitch angle is the angle of rotation
    around the x-axis (pointing right). Ambiguities in the angle due to gimbal lock are resolved.
    Camera roll needs to be provided and is removed before calculating the pitch angle.

    Args:
        trafo (np.ndarray): The transformation matrix from which to calculate the pitch angle.
        roll (float, optional): The angle of rotation around the forward (+/- z) axis. Defaults to 0.
        degrees (bool, optional): Whether to expect and return angles in degrees. Defaults to True.

    Returns:
        float: The calculated pitch angle.
    """

    rot = trafo[:3, :3]
    # Remove camera roll (in-plane rotation, i.e. rotation around the forward (+/- z) axis)
    if roll:
        rot_z = R.from_euler("z", roll, degrees=degrees).as_matrix()
        rot = rot_z.T @ rot

    # Calculate the angle of rotation around the x-axis (pitch)
    pitch = np.degrees(np.arctan2(-rot.T[2, 1], rot.T[2, 2]))

    # Adjusting angle to be in the range [0, 180] union [-180, 0]
    if pitch < -90:
        pitch += 180
    elif pitch > 90:
        pitch -= 180

    return pitch if degrees else np.radians(pitch)


def resolve_dtype(precision: int, integer: bool = False, unsigned: bool = False) -> Any:
    if precision == 8:
        return np.uint8 if unsigned else np.int8
    elif precision == 16:
        return np.float16 if not integer else np.uint16 if unsigned else np.int16
    elif precision == 32:
        return np.float32 if not integer else np.uint32 if unsigned else np.int32
    elif precision == 64:
        return np.float64 if not integer else np.uint64 if unsigned else np.int64
    else:
        raise Exception(f"Invalid precision: {precision}.")


def git_show_toplevel(cwd: str | Path | None = None) -> Path:
    try:
        root_path = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=cwd or Path.cwd(), universal_newlines=True
        ).strip()
        return Path(root_path)
    except subprocess.CalledProcessError as e:
        raise FileNotFoundError("Not inside a git repository.") from e


def git_show_superproject_working_tree(cwd: str | Path | None = None) -> Path:
    try:
        root_path = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"], cwd=cwd or Path.cwd(), universal_newlines=True
        ).strip()
        return Path(root_path)
    except subprocess.CalledProcessError as e:
        raise FileNotFoundError("Not inside a git submodule.") from e


def get_git_root() -> Path:
    root_path = git_show_superproject_working_tree()
    if not root_path or root_path == Path("."):
        root_path = git_show_toplevel()
    return Path(root_path)


def git_submodule_path(submodule_name_or_path: str | Path) -> Path:
    if isinstance(submodule_name_or_path, str):
        project_root = get_git_root()
        submodule_path = project_root / submodule_name_or_path

        if not submodule_path.exists():
            raise FileNotFoundError(f"Submodule {submodule_name_or_path} does not exist")
    elif isinstance(submodule_name_or_path, Path):
        if submodule_name_or_path.exists():
            if submodule_name_or_path.is_dir():
                submodule_path = git_show_toplevel(submodule_name_or_path)
            else:
                submodule_path = git_show_toplevel(submodule_name_or_path.parent)
        else:
            raise FileNotFoundError(f"Path {submodule_name_or_path} does not exist")

        if not submodule_path:
            raise FileNotFoundError(f"File {submodule_name_or_path} not inside any git submodule")
    else:
        raise TypeError("`submodule_name_or_file_path` must be a string or a Path object")

    return Path(submodule_path)


def get_args(print_args: bool = False) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any] | tuple[Any, dict[str, Any]]:
            # Get the function's signature
            signature = inspect.signature(func)
            bound_arguments = signature.bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            # Convert bound arguments to a dictionary
            args_dict = dict(bound_arguments.arguments)
            self = args_dict.pop("self", None)

            if print_args:
                with np.printoptions(precision=3, suppress=True):
                    if self is not None:
                        if args_dict:
                            args_str = ", ".join(f"{k}={v}" for k, v in args_dict.items())
                            print(f"{self.__class__.__name__}: {args_str}")
                        else:
                            print(f"{self.__class__.__name__}")
                    else:
                        if args_dict:
                            pprint(args_dict, compact=True)

            # Call the original function
            result = func(*args, **kwargs)

            # Handle method (if first argument is 'self')
            if self is not None:
                if hasattr(self, "_args"):
                    self._args.update(args_dict)
                else:
                    self._args = args_dict
                return result

            # Handle regular function
            if result is None:
                return args_dict
            return result, args_dict

        return wrapper

    return decorator


def default_on_exception(default: Any | None = None, exceptions: ExceptionTypes | None = Exception) -> Callable:
    handled_exceptions: ExceptionTypes = Exception if exceptions is None else exceptions
    caught_exceptions = tuple(handled_exceptions) if isinstance(handled_exceptions, list) else handled_exceptions

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except caught_exceptions as e:
                logger.error(f"Caught exception in {func.__name__}: {e}")
                if not (isinstance(default, np.ndarray) or torch.is_tensor(default)):
                    logger.debug_level_1(f"Returning default value: {default}")
                else:
                    logger.debug_level_1(f"Returning default value: {type(default)} ({default.shape})")
                if logger.isEnabledFor(DEBUG_LEVEL_2):
                    logger.exception(e)
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                return default

        return wrapper

    return decorator


def measure_runtime(arg=None, *, log_level: int | None = None, name: str | None = None, cuda: bool = False):
    """
    Flexible timing utility usable as:
        1) Decorator (old style, fully supported):
            @measure_runtime()
            def f(): ...

            @measure_runtime(DEBUG_LEVEL_1)
            def g(): ...

            @measure_runtime(log_level=DEBUG_LEVEL_2)
            def h(): ...

        2) Bare decorator (no parentheses):
            @measure_runtime
            def foo(): ...

        3) Context manager:
            with measure_runtime():
                work()

            with measure_runtime("data loading"):
                load()

            with measure_runtime(name="stage 1", log_level=DEBUG_LEVEL_1):
                stage1()

        4) Context manager with explicit log level positional (discouraged but supported):
            with measure_runtime(DEBUG_LEVEL_1):
                stuff()

    Positional argument disambiguation:
        - callable -> decorating that function
        - int      -> treated as log_level
        - str      -> treated as name (label for log message)
        - None     -> just return context manager (optionally using keyword args)

    Precedence:
        Explicit keyword 'log_level' or 'name' override any inferred positional meaning.
    """
    # ------------------------------------------------------------------ #
    # Normalize inputs
    # ------------------------------------------------------------------ #
    _func = None
    _log_level = DEBUG if log_level is None else log_level
    _name = name

    if callable(arg):
        # Used as @measure_runtime
        _func = arg
    elif isinstance(arg, int) and log_level is None:
        _log_level = arg
    elif isinstance(arg, str) and name is None:
        _name = arg
    elif arg is not None and not (callable(arg) or isinstance(arg, (int, str))):
        raise TypeError("Unsupported first positional argument for measure_runtime().")

    log_fn_map = {
        DEBUG: logger.debug,
        DEBUG_LEVEL_1: logger.debug_level_1,
        DEBUG_LEVEL_2: logger.debug_level_2,
    }
    log_fn = log_fn_map.get(_log_level, logger.info)

    # ------------------------------------------------------------------ #
    # Context manager implementation
    # ------------------------------------------------------------------ #
    @contextmanager
    def _context(label: str | None):
        if logger.isEnabledFor(_log_level):
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                yield
            finally:
                if cuda:
                    torch.cuda.synchronize()
                dur = time.perf_counter() - start
                log_fn(f"{label or 'runtime'} took {dur:.4f}s")
        else:
            yield

    # ------------------------------------------------------------------ #
    # Decorator implementation
    # ------------------------------------------------------------------ #
    def _decorate(func: Callable):
        local_label = _name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not logger.isEnabledFor(_log_level):
                return func(*args, **kwargs)
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                if cuda:
                    torch.cuda.synchronize()
                dur = time.perf_counter() - start
                log_fn(f"{local_label} takes {dur:.4f}s")

        return wrapper

    # If used as decorator directly: @measure_runtime
    if _func is not None:
        return _decorate(_func)

    # If user wants decorator factory: measure_runtime(...)(func)
    def _factory(func: Callable):
        return _decorate(func)

    # Heuristic: if no function supplied and we are inside a 'with', Python expects a CM.
    # We can't know syntactically here, but returning an object that supports __call__ (decorator)
    # AND context manager semantics can be overkill. Simplicity: if *no* function passed,
    # return context manager. User can still get decorator via measure_runtime(...)(f).
    return _context(_name or "runtime")


def cosine_anneal(start: float, stop: float, steps: int, current_step: int) -> float:
    ratio = min(current_step / steps, 1.0)
    cosine_value = 0.5 * (1 + math.cos(math.pi * ratio))
    return stop + (start - stop) * cosine_value


@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def normalize_mesh(mesh: Trimesh, center: bool = True, scale: bool = True, cube_or_sphere: str = "cube") -> Trimesh:
    if center:
        if cube_or_sphere in ["cube", "sphere"]:
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
        # elif cube_or_sphere == "sphere":
        #     mesh.apply_translation(-mesh.centroid)  # Fixme: Should be -mesh.centroid?
        else:
            raise ValueError(f"Normalization shape '{cube_or_sphere}' not supported.")
    if scale:
        if cube_or_sphere == "cube":
            mesh.apply_scale(1 / mesh.extents.max())
        elif cube_or_sphere == "sphere":
            mesh.apply_scale(1 / np.max(np.linalg.norm(mesh.vertices, axis=1)))
        else:
            raise ValueError(f"Normalization shape '{cube_or_sphere}' not supported.")
    return mesh


def get_points(n_views: int = 100) -> np.ndarray:
    """See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere."""
    rnd = 1.0
    points = []
    offset = 2.0 / n_views
    increment = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(n_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % n_views) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)


def get_rays(
    mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    normalize: bool = False,
    num_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v_coords, u_coords = np.nonzero(mask)  # v for height (y), u for width (x)
    num_masked_pixels = len(u_coords)

    if num_masked_pixels == 0:
        # Return empty arrays if mask is all zeros
        return (np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0))

    if num_samples is not None:
        # Sample from the non-zero pixels if a specific number of samples is requested
        indices = np.random.choice(num_masked_pixels, num_samples, replace=num_samples > num_masked_pixels)
        u = u_coords[indices]
        v = v_coords[indices]
        num_pixels_used = num_samples
    else:
        # Use all non-zero pixels
        u = u_coords
        v = v_coords
        num_pixels_used = num_masked_pixels

    uv_coords = np.stack((u.astype(float), v.astype(float)), axis=-1)
    xyz = np.concatenate((uv_coords, np.ones((num_pixels_used, 1))), axis=-1)

    inv_intrinsic = np.linalg.inv(intrinsic)
    inv_extrinsic = inv_trafo(extrinsic)

    p_cam = xyz @ inv_intrinsic.T
    p_world = apply_trafo(p_cam, inv_extrinsic)

    ray0 = np.expand_dims(inv_extrinsic[:3, 3], 0).repeat(num_pixels_used, axis=0)
    ray_dirs = p_world - ray0

    if normalize:
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=-1, keepdims=True)

    return ray0, ray_dirs, u, v


def sample_distances(
    n_points: int,
    near: float,
    far: float,
    n_steps: int | tuple[int, int] | None = 1,
    method: Literal["linear", "random"] = "random",
    uniform_in_volume: bool = False,
) -> np.ndarray:
    """
    Sample distances along rays between near and far.

    Args:
        n_points: Number of rays.
        near: Near plane distance (> 0).
        far: Far plane distance (> near).
        n_steps: Number of samples per ray, or (min, max) to randomize per-ray steps.
        method: "linear" places steps evenly in [0,1]; "random" draws U[0,1] and (if n_steps>1) sorts per-ray.
        uniform_in_volume: If True, map steps via cubic CDF so that samples are uniform in frustum volume:
            d = (near^3 + u * (far^3 - near^3))^(1/3). If False, use linear depth: d = near + u * (far - near).

    Returns:
        distances: Array of shape (n_points, n_steps) with sampled depths.
    """
    if n_steps is None:
        n_steps = 1
    if isinstance(n_steps, (tuple, list)):
        if len(n_steps) != 2:
            raise ValueError("n_steps tuple must contain exactly two values (min, max).")
        n_steps = np.random.randint(n_steps[0], n_steps[1])

    if method == "linear":
        steps = np.linspace(0, 1, n_steps)
        steps = np.expand_dims(steps, 0).repeat(n_points, axis=0)
    elif method == "random":
        steps = np.random.rand(n_points, n_steps)
        if n_steps > 1:
            steps.sort(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'linear' or 'random'.")

    if uniform_in_volume:
        # Map uniform steps u in [0,1] to cubic depth CDF for uniform-in-volume sampling
        # d = (near^3 + u * (far^3 - near^3))^(1/3)
        start_c, stop_c = near**3, far**3
        return (start_c + steps * (stop_c - start_c)) ** (1.0 / 3.0)

    # Default: linear in depth
    return near + steps * (far - near)


@measure_runtime
def is_in_frustum(
    points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    width: int,
    height: int,
    near: float = 0.2,
    far: float = 2.4,
) -> np.ndarray:
    """
    Checks which points in an Nx3 array lie inside a camera frustum.

    Args:
        points: An (N, 3) NumPy array of points in world space.
        intrinsic: The (3, 3) camera intrinsics matrix.
        extrinsic: The (4, 4) world-to-camera extrinsics matrix (View Matrix).
        width: The image width in pixels.
        height: The image height in pixels.
        near: The distance to the near clipping plane.
        far: The distance to the far clipping plane.

    Returns:
        An (N,) boolean NumPy array (mask), True for points inside the frustum.
    """
    # 1. Build the 4x4 projection matrix from intrinsics
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    f_minus_n = far - near
    f_plus_n = far + near

    proj_matrix = np.array(
        [
            [2 * fx / width, 0, (2 * cx / width) - 1, 0],
            [0, 2 * fy / height, (2 * cy / height) - 1, 0],
            [0, 0, f_plus_n / f_minus_n, -2 * far * near / f_minus_n],
            [0, 0, 1.0, 0],
        ]
    )
    # Note: This direct construction matches your original matrix logic
    # and is slightly cleaner than building from np.zeros.

    # 2. Combine into a single View-Projection matrix
    view_proj_matrix = proj_matrix @ extrinsic

    # 3. Transform points to clip space without creating a temporary array
    # This is the main optimization.
    vp_matrix_t = view_proj_matrix.T
    clip_coords = points @ vp_matrix_t[:3, :] + vp_matrix_t[3, :]

    # 4. Perform the vectorized clip space test
    # This part of your original code is already highly efficient.
    w = clip_coords[:, 3]
    mask = np.all(np.abs(clip_coords[:, :3]) <= w[:, np.newaxis], axis=1)

    return mask


def config_hash(cfg: DictConfig, length: int = 10, ignore: list[str] | None = None) -> str:
    container = OmegaConf.to_container(cfg, resolve=True)

    # Convert dotted paths to tuples, e.g. "test.filename" -> ("test", "filename")
    ignore_paths = []
    if ignore:
        ignore_paths = [tuple(p.split(".")) for p in ignore]

    def should_drop(path_tuple: tuple) -> bool:
        # Drop if the path exactly matches any ignore path
        return any(path_tuple == ip for ip in ignore_paths)

    def drop(obj, path=()):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                new_path = (*path, k)
                # If this exact node should be dropped, skip it
                if should_drop(new_path):
                    continue
                # If a top-level prefix is ignored (e.g., ("hydra",)), skip the whole subtree
                if any(new_path[: len(ip)] == ip and len(ip) == 1 for ip in ignore_paths):
                    # skip entire subtree if the ignore path is a single-segment prefix like ("hydra",)
                    if new_path[0] == "hydra":
                        continue
                out[k] = drop(v, new_path)
            return out
        if isinstance(obj, list):
            return [drop(v, path) for v in obj]
        return obj

    # Default ignore list tuned to your config.yaml
    if not ignore:
        ignore_paths = [
            ("hydra",),  # all Hydra runtime info (timestamps, dirs, etc.)
            ("log", "id"),
            ("log", "version"),
            ("log", "wandb"),
            ("log", "offline"),
            ("log", "profile"),
            ("test", "overwrite"),  # doesn't change metrics, only write behavior
            ("test", "dir"),
            ("test", "filename"),
        ]

    pruned = drop(container)
    payload = json.dumps(pruned, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:length]


def to_scalar(x: np.ndarray | np.generic | Tensor | int | float | bool) -> Any:
    if hasattr(x, "item"):
        try:
            return cast(Any, x).item()
        except Exception:
            pass
    try:
        if isinstance(x, np.generic):
            return np.asarray(x).item()
    except Exception:
        pass
    return x
