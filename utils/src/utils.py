import contextlib
import logging
import os
import sys
from glob import glob
from io import BytesIO
from random import shuffle
from typing import Union, List, Tuple, Optional, Any, IO, Callable
from pathlib import Path
from contextlib import contextmanager
from functools import partial

import lightning
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import h5py
from PIL import Image

import numpy as np
import torch
import trimesh
from joblib import Parallel, cpu_count
from lightning.pytorch.utilities import rank_zero_only
from lightning import seed_everything
from scipy.spatial.transform import Rotation as R
from torch import Tensor, nn
from tqdm import tqdm
import lovely_tensors as lt


def get_partnet_colors() -> np.ndarray:
    return np.array([[0.65, 0.95, 0.05], [0.35, 0.05, 0.35], [0.65, 0.35, 0.65], [0.95, 0.95, 0.65],
                     [0.95, 0.65, 0.05], [0.35, 0.05, 0.05], [0.65, 0.05, 0.05], [0.65, 0.35, 0.95],
                     [0.05, 0.05, 0.65], [0.65, 0.05, 0.35], [0.05, 0.35, 0.35], [0.65, 0.65, 0.35],
                     [0.35, 0.95, 0.05], [0.05, 0.35, 0.65], [0.95, 0.95, 0.35], [0.65, 0.65, 0.65],
                     [0.95, 0.95, 0.05], [0.65, 0.35, 0.05], [0.35, 0.65, 0.05], [0.95, 0.65, 0.95],
                     [0.95, 0.35, 0.65], [0.05, 0.65, 0.95], [0.65, 0.95, 0.65], [0.95, 0.35, 0.95],
                     [0.05, 0.05, 0.95], [0.65, 0.05, 0.95], [0.65, 0.05, 0.65], [0.35, 0.35, 0.95],
                     [0.95, 0.95, 0.95], [0.05, 0.05, 0.05], [0.05, 0.35, 0.95], [0.65, 0.95, 0.95],
                     [0.95, 0.05, 0.05], [0.35, 0.95, 0.35], [0.05, 0.35, 0.05], [0.05, 0.65, 0.35],
                     [0.05, 0.95, 0.05], [0.95, 0.65, 0.65], [0.35, 0.95, 0.95], [0.05, 0.95, 0.35],
                     [0.95, 0.35, 0.05], [0.65, 0.35, 0.35], [0.35, 0.95, 0.65], [0.35, 0.35, 0.65],
                     [0.65, 0.95, 0.35], [0.05, 0.95, 0.65], [0.65, 0.65, 0.95], [0.35, 0.05, 0.95],
                     [0.35, 0.65, 0.95], [0.35, 0.05, 0.65]])


def disable_multithreading():
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TBB_NUM_THREADS'] = '1'


def get_file_descriptor(file_or_fd: Union[int, IO]) -> int:
    """ Returns the file descriptor of the given file.

    :param file_or_fd: Either a file or a file descriptor. If a file descriptor is given, it is returned directly.
    :return: The file descriptor of the given file.
    """
    if hasattr(file_or_fd, 'fileno'):
        fd = file_or_fd.fileno()
    else:
        fd = file_or_fd
    if not isinstance(fd, int):
        raise AttributeError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(to: Union[int, IO, str] = os.devnull, enabled: bool = True) -> IO:
    """ Redirects all stdout to the given file.

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
        with os.fdopen(os.dup(stdout_fd), 'w') as copied:
            stdout.flush()  # flush library buffers that dup2 knows nothing about
            try:
                os.dup2(get_file_descriptor(to), stdout_fd)  # $ exec >&to
            except AttributeError:  # filename
                with open(to, 'wb') as to_file:
                    os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
            try:
                yield copied
            finally:
                # restore stdout to its previous value
                # NOTE: dup2 makes stdout_fd inheritable unconditionally
                stdout.flush()
                os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
    else:
        yield sys.stdout


def load_mesh(path: Union[str, Path],
              load_with: Optional[str] = None,
              **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    assert load_with in [None, 'trimesh', 'open3d', 'pymeshlab'], \
        f"'load_with' must be one of [None, 'trimesh', 'open3d', 'pymeshlab'], got {load_with}."

    path = Path(path).expanduser().resolve()

    def load_trimesh() -> Tuple[np.ndarray, Optional[np.ndarray]]:
        import trimesh
        mesh = trimesh.load(path,
                            force=kwargs.get('force', 'mesh'),
                            process=kwargs.get('process', False),
                            validate=kwargs.get('validate', False))
        if isinstance(mesh, trimesh.PointCloud):
            return mesh.vertices, None
        return mesh.vertices, mesh.faces

    def load_open3d() -> Tuple[np.ndarray, np.ndarray]:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(path),
                                         enable_post_processing=kwargs.get('enable_post_processing', False),
                                         print_progress=kwargs.get('print_progress', False))
        return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    def load_pymeshlab() -> Tuple[np.ndarray, np.ndarray]:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(path))
        mesh = ms.current_mesh()
        return mesh.vertex_matrix(), mesh.face_matrix()

    if load_with == 'trimesh':
        return load_trimesh()
    elif load_with == 'open3d':
        return load_open3d()
    elif load_with == 'pymeshlab':
        return load_pymeshlab()
    else:
        try:
            return load_trimesh()
        except (ValueError, ImportError) as e:
            if isinstance(e, ImportError):
                logging.warning("Could not import Trimesh. Falling back to Open3D.")
            elif isinstance(e, ValueError):
                logging.warning("Could not load mesh with Trimesh. Falling back to Open3D.")
            try:
                return load_open3d()
            except ImportError:
                logging.warning("Could not import Open3D. Falling back to PyMeshlab.")
                try:
                    return load_pymeshlab()
                except ImportError:
                    logging.warning("Could not import PyMeshLab.")
                    logging.error("Could not load mesh.")
                    raise


def save_mesh(path: Union[str, Path],
              vertices: np.ndarray,
              faces: np.ndarray,
              colors: Optional[np.ndarray] = None,
              normals: Optional[np.ndarray] = None,
              save_with: Optional[str] = None,
              **kwargs: Any):
    path = Path(path).expanduser().resolve()
    assert path.suffix in ['.stl', '.off', '.ply', '.obj']
    assert save_with in [None, 'trimesh', 'open3d', 'pymeshlab'], \
        f"'save_with' must be one of [None, 'trimesh', 'open3d', 'pymeshlab'], got {save_with}."

    def save_trimesh():
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices,
                               faces=faces,
                               face_normals=None if normals is None or len(normals) != len(faces) else normals,
                               vertex_normals=None if normals is None or len(normals) != len(vertices) else normals,
                               face_colors=None if colors is None or len(colors) != len(faces) else colors,
                               vertex_colors=None if colors is None or len(colors) != len(vertices) else colors,
                               process=kwargs.get('process', False),
                               validate=kwargs.get('validate', False))
        if path.suffix == '.obj':
            mesh.export(str(path),
                        include_normals=kwargs.get('include_normals', False if normals is None else True),
                        include_color=kwargs.get('include_color', False if colors is None else True),
                        include_texture=kwargs.get('include_texture', False),
                        return_texture=kwargs.get('return_texture', False),
                        write_texture=kwargs.get('write_texture', False),
                        digits=kwargs.get('digits', 8))
        elif path.suffix == '.ply':
            mesh.export(str(path),
                        encoding=kwargs.get('encoding', 'binary'),
                        vertex_normal=kwargs.get('vertex_normal', False if normals is None else True),
                        include_attributes=kwargs.get('include_attributes', False))
        elif path.suffix == '.off':
            mesh.export(str(path),
                        digits=kwargs.get('digits', 10))
        elif path.suffix == '.stl':
            mesh.export(str(path),
                        mode=kwargs.get('mode', 'binary'))

    def save_open3d():
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
        o3d.io.write_triangle_mesh(str(path),
                                   mesh,
                                   write_ascii=kwargs.get('write_ascii', False),
                                   compressed=kwargs.get('compressed', True),
                                   write_vertex_normals=kwargs.get('write_vertex_normals', False),
                                   write_vertex_colors=kwargs.get('write_vertex_colors', False),
                                   write_triangle_uvs=kwargs.get('write_triangle_uvs', False))

    def save_pymeshlab():
        import pymeshlab
        ms = pymeshlab.MeshSet()
        pymesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
        ms.add_mesh(pymesh)
        save_current_mesh = partial(ms.save_current_mesh, save_textures=kwargs.get('save_textures', False))
        if path.suffix == ".stl":
            save_current_mesh = partial(save_current_mesh,
                                        binary=kwargs.get('binary', True),
                                        save_face_color=kwargs.get('save_face_color', False))
        if path.suffix in [".off", ".ply", ".obj"]:
            save_current_mesh = partial(save_current_mesh,
                                        save_vertex_color=kwargs.get('save_vertex_color', False),
                                        save_vertex_coord=kwargs.get('save_vertex_coord', False),
                                        save_face_color=kwargs.get('save_face_color', False),
                                        save_polygonal=kwargs.get('save_polygonal', False))
        if path.suffix == ".ply":
            save_current_mesh = partial(save_current_mesh,
                                        binary=kwargs.get('binary', True),
                                        save_vertex_quality=kwargs.get('save_vertex_quality', False),
                                        save_vertex_normal=kwargs.get('save_vertex_normal', False),
                                        save_vertex_radius=kwargs.get('save_vertex_radius', False),
                                        save_face_quality=kwargs.get('save_face_quality', False),
                                        save_wedge_color=kwargs.get('save_wedge_color', False),
                                        save_wedge_texcoord=kwargs.get('save_wedge_texcoord', False),
                                        save_wedge_normal=kwargs.get('save_wedge_normal', False))
        elif path.suffix == ".obj":
            save_current_mesh = partial(save_current_mesh,
                                        save_vertex_normal=kwargs.get('save_vertex_normal', False),
                                        save_wedge_texcoord=kwargs.get('save_wedge_texcoord', False),
                                        save_wedge_normal=kwargs.get('save_wedge_normal', False))
        save_current_mesh(file_name=str(path))

    if save_with == 'trimesh':
        save_trimesh()
    elif save_with == 'open3d':
        save_open3d()
    elif save_with == 'pymeshlab':
        save_pymeshlab()
    else:
        try:
            save_trimesh()
        except ImportError:
            logging.warning("Could not import Trimesh. Falling back to Open3D.")
            try:
                save_open3d()
            except ImportError:
                logging.warning("Could not import Open3D. Falling back to PyMeshlab.")
                try:
                    save_pymeshlab()
                except ImportError:
                    logging.warning("Could not import PyMeshLab.")
                    logging.error("Could not save mesh.")
                    raise


@contextlib.contextmanager
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


def stack_images(images: List[np.ndarray]) -> np.ndarray:
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
        return images[0]
    num_rows_cols = int(np.sqrt(len(images)))
    rows = list()
    for step in range(0, len(images), num_rows_cols):
        rows.append(np.hstack(images[step:step + num_rows_cols]))
    return np.vstack(rows)


# Global set to keep track of loggers created using the setup_logger function
_created_loggers = set()


def set_log_level(level: Union[int, str]):
    for logger_name in _created_loggers:
        logging.getLogger(logger_name).setLevel(level)


def setup_logger(name: str = __name__, check_exists: bool = False) -> logging.Logger:
    """Initializes a multi-GPU-friendly python command line logger.

    Args:
        name: Name of the logger
        check_exists: Whether to check if the logger already exists

    Returns:
        A python logger
    """
    # If the name is None or "__main__", we use the script's name
    if name is None or name == "__main__":
        # Get the current working directory
        cwd = os.getcwd()

        # Convert the script's absolute path to a relative path
        rel_path = os.path.relpath(sys.argv[0], start=cwd)

        # Convert the relative path to a module name
        name = os.path.splitext(rel_path)[0].replace(os.sep, '.')

    # check if requested logger already exists
    if check_exists:
        assert logging.getLogger(name).hasHandlers(), f"Logger {name} does not exist"

    # initialize logger
    logger = logging.getLogger(name)

    # If the logger already has handlers, it has been setup before
    # so we skip the rest of the setup to avoid duplicate handlers
    if not logger.hasHandlers():
        # Create a handler for the logger. Here we're using a StreamHandler to log to console
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(fmt='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
                                      datefmt='%d-%b-%y %H:%M:%S')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

        # Set logger level
        logger.setLevel(logging.INFO)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    # Record the logger's name
    _created_loggers.add(name)
    return logger


def normalize(val: Union[np.ndarray, Tensor],
              a: float = 0,
              b: float = 1) -> Union[np.ndarray, Tensor]:
    # Normalize a value between a and b
    return (val - val.min()) / (val.max() - val.min()) * (b - a) + a


def to_tensor(x: Union[np.ndarray, Tensor],
              unsqueeze: bool = True,
              device: Optional[Union[str, torch.device]] = "cuda") -> Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if unsqueeze:
            x = x.unsqueeze(0)
    if device is not None:
        x = x.to(device)
    return x


def rot_from_euler(axes: str, upper_hemisphere: bool) -> Tuple[np.ndarray, float]:
    x_angle = 0
    angles = np.random.uniform(0, 360, size=len(axes))
    if 'x' in axes:
        x_angle = angles[list(axes).index('x')]
        if upper_hemisphere:
            x_angle = np.random.uniform(0, 90)
            angles[list(axes).index('x')] = x_angle
    rot = R.from_euler(axes, angles[0] if len(angles) == 1 else angles, degrees=True).as_matrix()
    # rot = rotation_matrix(axes, angles)
    return rot, x_angle


def generate_random_basis(n_points: int = 1024,
                          n_dims: int = 3,
                          radius: float = 0.5,
                          seed: Optional[int] = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_points, n_dims))
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape((-1, 1))
    x_unit = x / x_norms

    r = rng.uniform(size=(n_points, 1))
    u = np.power(r, 1.0 / n_dims)
    x = radius * x_unit * u

    return x


def subsample(data: np.ndarray, num_samples: int) -> np.ndarray:
    """Return subsampled index into a given array of data."""
    if len(data) == num_samples or num_samples == 0:
        return np.arange(len(data))
    if len(data) < num_samples:
        return np.random.choice(len(data), size=num_samples)
    return np.random.randint(len(data), size=num_samples)


def inv_trafo(trafo: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = trafo[:3, :3].T
    inverse[:3, 3] = -trafo[:3, :3].T @ trafo[:3, 3]
    return inverse


def resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def resolve_save_dir(cfg: DictConfig) -> Path:
    assert "log" in cfg.dirs, "No log directory specified."
    project = HydraConfig.get().job.config_name if cfg.log.project is None else cfg.log.project
    name = cfg.model.arch if cfg.log.name is None else cfg.log.name
    return resolve_path(cfg.dirs["log"]) / project / name


def make_3d_grid(bb_min: Union[List[float], Tuple[float, float, float]],
                 bb_max: Union[List[float], Tuple[float, float, float]],
                 shape: Union[List[int], Tuple[int, int, int]]) -> Tensor:
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).reshape(size)
    pys = pys.view(1, -1, 1).expand(*shape).reshape(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).reshape(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def eval_input(inputs: Union[Path, List[Union[Path, str]]],
               in_format: Optional[str] = None,
               recursion_depth: Optional[int] = None,
               sort: bool = False) -> List[Path]:
    if isinstance(inputs, list):
        files = [Path(f).expanduser().resolve() for f in inputs]
    else:
        inputs = inputs.expanduser().resolve()
        if inputs.is_file():
            files = [inputs]
        elif inputs.is_dir():
            assert in_format is not None, "If input is a directory, in_format must be set."
            logging.debug(f"Globbing paths from {inputs}.")
            if recursion_depth is None:
                files = list(inputs.rglob(f"*{in_format}"))
            else:
                pattern = f"{'/'.join(['*' for _ in range(recursion_depth)])}/*{in_format}"
                files = list(inputs.glob(pattern))
        else:
            files = glob(str(inputs))
    shuffle(files)
    if sort:
        files = sorted(files)
    logging.debug(f"Found {len(files)} files.")
    return files


def get_num_workers(num_workers: Optional[int] = None):
    if num_workers is None or num_workers < 0:
        num_workers = cpu_count()
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        # Get the GPU IDs allocated to the job
        gpus_allocated = os.environ.get('SLURM_JOB_GPUS', '')

        # Count the number of GPUs by splitting the string and measuring its length
        total_gpus = len(gpus_allocated.split(',')) if gpus_allocated else 1

        # Total number of tasks per node
        tasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))

        # Number of GPUs per task
        gpus_per_task = total_gpus // tasks_per_node

        # Total number of CPUs allocated to the task
        total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

        # Calculate CPUs per GPU
        num_workers = total_cpus // gpus_per_task
    else:
        logging.debug("Not on a SLURM node.")
    if num_workers > 1:
        disable_multithreading()
    logging.info(f"Loading data with {num_workers} workers.")
    return num_workers


def get_device_info() -> Tuple[int, int]:
    major, minor = 0, 0
    for device_id in range(torch.cuda.device_count()):
        device = torch.cuda.get_device_name(device_id)
        major, minor = torch.cuda.get_device_capability(device_id)
        logging.info(f"Using GPU {device} with compute capability {major}.{minor}")
        if major > 7:
            torch.set_float32_matmul_precision("high")
            logging.debug("Using high precision matrix multiplication.")
    return major, minor


def points_to_coordinates(points: Union[Tensor, np.ndarray],
                          padding: float = 0.1,
                          plane: Optional[str] = None) -> Union[Tensor, np.ndarray]:
    """Converts the given points to coordinates in the range [0, 1]."""
    coordinates = points[:, :, :3].clone() if isinstance(points, Tensor) else points[:, :3].copy()

    # project to 2d
    if plane is not None:
        if plane == "xz":
            coordinates = coordinates[:, :, [0, 2]]
        elif plane == "xy":
            coordinates = coordinates[:, :, [0, 1]]
        elif plane == "yz":
            coordinates = coordinates[:, :, [1, 2]]
        else:
            raise NotImplementedError(f"{plane} plane not implemented.")

    # normalize to [0, 1]
    coordinates /= (1 + padding)
    coordinates += 0.5
    coordinates[coordinates > 1] = 1
    coordinates[coordinates < 0] = 0

    return coordinates


def coordinates_to_index(coordinates: Tensor, resolution: int) -> Tensor:
    """Converts the given coordinates to indices in the range [0, resolution - 1]."""
    grid_coords = torch.clamp(coordinates * resolution, 0, resolution - 1).long()
    if coordinates.size(-1) == 2:
        index = grid_coords[:, :, 0] + resolution * grid_coords[:, :, 1]
    elif coordinates.size(-1) == 3:
        index = grid_coords[:, :, 0] + resolution * (grid_coords[:, :, 1] + resolution * grid_coords[:, :, 2])
    else:
        raise NotImplementedError(f"{coordinates.size(-1)}D coordinates not implemented.")
    return index


def load_from_binary_hdf5(obj_path: Path,
                          file_names: List[str],
                          file_dirs: Optional[List[Optional[str]]] = None) -> List[Union[np.ndarray, trimesh.Trimesh]]:
    files = list()
    if file_dirs is None:
        file_dirs = [None] * len(file_names)
    with h5py.File(obj_path.with_suffix(".hdf5"), 'r') as hdf5_file:
        for file_name, file_dir in zip(file_names, file_dirs):
            data = hdf5_file
            if file_dir is not None:
                data = hdf5_file[file_dir]
            data = BytesIO(np.frombuffer(data[file_name][()], dtype=np.uint8))
            if file_name[-3:] in ['npy', 'npz']:
                data = np.load(data)
            elif file_name[-3:] in ['off', 'ply', 'obj']:
                data = trimesh.load(data, file_type=file_name[-3:], force='mesh')
            elif file_name[-3:] in ['png', 'jpg']:
                data = Image.open(data, mode='r', formats=['PNG', 'JPEG'])
                data = np.asarray(data, dtype=np.uint16 if data.mode == 'I' else np.uint8)
            else:
                raise NotImplementedError(f"Unknown file format {file_name}.")
            files.append(data)
    return files


def resolve_out_dir(in_path: Path,
                    in_dir: Path,
                    out_dir: Path,
                    shard: Optional[int] = None) -> Path:
    in_path = in_path.expanduser().resolve()
    in_dir = in_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    shard = '' if shard is None else f'{shard:04d}'
    out_dir = (out_dir / in_path.relative_to(in_dir)).parent / shard
    logging.debug(f"Resolved output directory to {out_dir}.")
    return out_dir


def monkey_patch(instance, method_name, new_method):
    """
    Monkey patches the method of the given instance.

    :param instance: The instance whose method is to be monkey patched.
    :param method_name: The name of the method to be patched.
    :param new_method: The new method to replace the original one.
    """
    original_method = getattr(instance, method_name, None)
    if not original_method:
        raise AttributeError(f"Method {method_name} not found in the instance.")

    # Bind the new method to the instance
    bound_method = new_method.__get__(instance, instance.__class__)
    setattr(instance, method_name, bound_method)


def save_command_and_args_to_file(file_path: Union[str, Path], args: Optional[Any] = None):
    file_path = resolve_path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('w', encoding='utf-8') as file:
        file.write('Command: ' + ' '.join(sys.argv) + '\n')
        if args is not None:
            file.write('Arguments:\n')
            for arg, value in vars(args).items():
                file.write(f"{arg}: {value}\n")


def binary_from_multi_class(logits: Tensor,
                            free_label: Optional[int] = None,
                            occ_label: Optional[int] = None) -> Tensor:
    assert free_label is None or occ_label is None, "Only one of free_label and occ_label can be specified."
    if free_label == -1 or (free_label is None and occ_label is None):
        occ_logits = logits[..., :-1]
        free_logits = logits[..., -1]
        binary_logits = occ_logits.max(dim=-1)[0] - free_logits
    elif free_label is None:
        occ_logits = logits[..., occ_label:occ_label + 1].squeeze(-1)
        free_logits = torch.cat([logits[..., :occ_label], logits[..., occ_label + 1:]], dim=-1)
        binary_logits = occ_logits - free_logits.max(dim=-1)[0]
    else:
        free_logits = logits[..., free_label:free_label + 1].squeeze(-1)
        occ_logits = torch.cat([logits[..., :free_label], logits[..., free_label + 1:]], dim=-1)
        binary_logits = occ_logits.max(dim=-1)[0] - free_logits
    return binary_logits.squeeze(-1)


def setup_config(cfg: DictConfig, functions: Optional[List[Callable]] = None, **kwargs: Any):
    major, minor = get_device_info()
    if major + minor / 10 < 7.0:
        if '16' in str(cfg.train.precision):
            cfg.train.precision = '32-true'
            logging.warning("Compute capability below 7.0. Using full precision.")
    elif major + minor / 10 < 8.0:
        if 'bf' in str(cfg.train.precision):
            cfg.train.precision = cfg.train.precision.strip('bf')
            logging.warning("Compute capability below 8.0. Using half precision.")

    with stdout_redirected(enabled=not cfg.log.verbose):
        seed_everything(cfg.misc.seed, workers=kwargs.get('workers', False))

    if cfg.log.pretty:
        lt.monkey_patch()
    if cfg.log.verbose:
        set_log_level(logging.DEBUG)
        print(OmegaConf.to_yaml(cfg))

    if functions is not None:
        for function in functions:
            function(cfg, **kwargs)


def set_precision(model: nn.Module, precision: Union[str, int]) -> nn.Module:
    fabric = lightning.Fabric(precision=precision)
    model = fabric.setup_module(model)
    if hasattr(model, "encoder"):
        if isinstance(model.encoder, nn.ModuleList):
            model.encoder = nn.ModuleList([fabric.setup_module(m) for m in model.encoder])
        else:
            model.encoder = fabric.setup_module(model.encoder)
    if hasattr(model, "decoder"):
        model.decoder = fabric.setup_module(model.decoder)
    return model
