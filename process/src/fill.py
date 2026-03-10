import importlib
import time
from pathlib import Path

import numpy as np
import torch
from pykdtree.kdtree import KDTree
from scipy import ndimage
from trimesh import Trimesh

from utils import setup_logger

from .utils import extract, get_vertices_and_faces

try:
    kaolin = importlib.import_module("kaolin")
except ImportError:
    kaolin = None

logger = setup_logger(__name__)


def kaolin_pipeline(
    mesh: Trimesh | dict[str, np.ndarray],
    resolution: int = 256,
    fill_holes: bool = True,
    eps: float = 1e-6,
    smoothing_iterations: int = 3,
    realign: bool = True,
    save_voxel_path: Path | None = None,
    try_cpu: bool = False,
) -> dict[str, np.ndarray]:
    if kaolin is None:
        raise ImportError("kaolin is required for kaolin_pipeline.")

    kaolin_mod = kaolin
    start = time.perf_counter()
    vertices, faces = get_vertices_and_faces(mesh)
    torch_vertices = torch.from_numpy(vertices).cuda()
    torch_faces = torch.from_numpy(faces).cuda()
    logger.debug(f"Loading mesh to GPU took {time.perf_counter() - start:.3f}s.")

    try:
        restart = time.perf_counter()
        voxel = kaolin_mod.ops.conversions.trianglemeshes_to_voxelgrids(
            torch_vertices.unsqueeze(0),
            torch_faces,
            resolution=resolution,
            origin=torch.zeros((1, 3)).cuda() - 0.5,
            scale=torch.ones(1).cuda(),
        )
        logger.debug(f"GPU voxelization took {time.perf_counter() - restart:.3f}s.")
    except torch.cuda.OutOfMemoryError:
        if try_cpu:
            logger.error("Out of memory error during voxelization on GPU. Trying CPU implementation.")
            restart = time.perf_counter()
            voxel = torch.from_numpy(voxelize(mesh, resolution=resolution)).unsqueeze(0).cuda()
            logger.debug(f"CPU voxelization took {time.perf_counter() - restart:.3f}s.")
        else:
            raise

    if fill_holes:
        restart = time.perf_counter()
        voxel = torch.from_numpy(ndimage.binary_fill_holes(voxel.squeeze(0).cpu().numpy())).unsqueeze(0).cuda()
        logger.debug(f"Hole filling took {time.perf_counter() - restart:.3f}s.")

    restart = time.perf_counter()
    voxel = kaolin_mod.ops.voxelgrid.extract_surface(voxel)
    logger.debug(f"Surface extraction took {time.perf_counter() - restart:.3f}s.")

    restart = time.perf_counter()
    odms = kaolin_mod.ops.voxelgrid.extract_odms(voxel)
    logger.debug(f"ODM extraction took {time.perf_counter() - restart:.3f}s.")

    restart = time.perf_counter()
    voxel = kaolin_mod.ops.voxelgrid.project_odms(odms)
    logger.debug(f"ODM projection took {time.perf_counter() - restart:.3f}s.")

    try:
        restart = time.perf_counter()
        vertices, faces = kaolin_mod.ops.conversions.voxelgrids_to_trianglemeshes(voxel)
        vertices = vertices[0] / (voxel.size(-1) + 1)
        vertices -= 0.5
        faces = faces[0]
        logger.debug(f"GPU Marching Cubes took {time.perf_counter() - restart:.3f}s.")
    except torch.cuda.OutOfMemoryError:
        if try_cpu:
            restart = time.perf_counter()
            logger.error("Out of memory error during Marching Cubes on GPU. Trying CPU implementation.")
            mesh = extract(voxel.squeeze(0).cpu().numpy(), level=0.5, resolution=resolution, pad=False)
            vertices, faces = get_vertices_and_faces(mesh)
            vertices = torch.from_numpy(vertices).cuda()
            faces = torch.from_numpy(faces).cuda()
            logger.debug(f"CPU Marching Cubes took {time.perf_counter() - restart:.3f}s.")
        else:
            raise

    voxel = voxel.squeeze(0).cpu().numpy()
    if save_voxel_path is not None:
        save_voxel_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(save_voxel_path), voxel=np.packbits(voxel))

    if smoothing_iterations > 0:
        restart = time.perf_counter()
        adj_sparse = kaolin_mod.ops.mesh.adjacency_matrix(len(vertices), faces, sparse=True)
        neighbor_num = torch.sparse.sum(adj_sparse, dim=1).to_dense().view(-1, 1)
        for _ in range(smoothing_iterations):
            neighbor_sum = torch.sparse.mm(adj_sparse, vertices)
            vertices = neighbor_sum / neighbor_num
        logger.debug(f"Smoothing took {time.perf_counter() - restart:.3f}s.")

    if realign:
        restart = time.perf_counter()
        src_min, src_max = vertices.min(0, keepdim=True)[0], vertices.max(0, keepdim=True)[0]
        tgt_min, tgt_max = torch_vertices.min(0, keepdim=True)[0], torch_vertices.max(0, keepdim=True)[0]
        vertices = ((vertices - src_min) / (src_max - src_min + eps)) * (tgt_max - tgt_min) + tgt_min
        logger.debug(f"Realigning took {time.perf_counter() - restart:.3f}s.")

    return {"vertices": vertices.cpu().numpy(), "faces": faces.cpu().numpy()}


def voxelize(
    mesh: Trimesh | dict[str, np.ndarray], min_value: float = -0.5, max_value: float = 0.5, resolution: int = 256
) -> np.ndarray:
    x = np.linspace(min_value, max_value, resolution)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    grid_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    vertices, faces = get_vertices_and_faces(mesh)
    sampled = Trimesh(vertices=vertices, faces=faces).sample(resolution**3)
    if isinstance(sampled, tuple):
        points = np.asarray(sampled[0])
    else:
        points = np.asarray(sampled)

    _, indices = KDTree(grid_points).query(points, k=1, eps=0)
    voxel = np.zeros(len(grid_points), dtype=bool)
    voxel[indices] = True

    return voxel.reshape((resolution,) * 3)
