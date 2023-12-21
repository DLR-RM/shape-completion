from pathlib import Path
from typing import Dict, Optional, Union
from time import time

from pykdtree.kdtree import KDTree
from trimesh import Trimesh
import numpy as np
import torch
import kaolin
from scipy import ndimage

from .utils import get_vertices_and_faces, extract
from utils import setup_logger

logger = setup_logger(__name__)


def kaolin_pipeline(mesh: Union[Trimesh, Dict[str, np.ndarray]],
                    resolution: int = 256,
                    fill_holes: bool = True,
                    eps: float = 1e-6,
                    smoothing_iterations: int = 3,
                    realign: bool = True,
                    save_voxel_path: Optional[Path] = None,
                    try_cpu: bool = False) -> Dict[str, np.ndarray]:
    start = time()
    vertices, faces = get_vertices_and_faces(mesh)
    torch_vertices = torch.from_numpy(vertices).cuda()
    torch_faces = torch.from_numpy(faces).cuda()
    logger.debug(f"Loading mesh to GPU took {time() - start:.3f}s.")

    try:
        restart = time()
        voxel = kaolin.ops.conversions.trianglemeshes_to_voxelgrids(torch_vertices.unsqueeze(0),
                                                                    torch_faces,
                                                                    resolution=resolution,
                                                                    origin=torch.zeros((1, 3)).cuda() - 0.5,
                                                                    scale=torch.ones(1).cuda())
        logger.debug(f"GPU voxelization took {time() - restart:.3f}s.")
    except torch.cuda.OutOfMemoryError as e:
        if try_cpu:
            logger.error("Out of memory error during voxelization on GPU. Trying CPU implementation.")
            restart = time()
            voxel = torch.from_numpy(voxelize(mesh, resolution=resolution)).unsqueeze(0).cuda()
            logger.debug(f"CPU voxelization took {time() - restart:.3f}s.")
        else:
            raise e

    if fill_holes:
        restart = time()
        voxel = torch.from_numpy(ndimage.binary_fill_holes(voxel.squeeze(0).cpu().numpy())).unsqueeze(0).cuda()
        logger.debug(f"Hole filling took {time() - restart:.3f}s.")

    restart = time()
    voxel = kaolin.ops.voxelgrid.extract_surface(voxel)
    logger.debug(f"Surface extraction took {time() - restart:.3f}s.")

    restart = time()
    odms = kaolin.ops.voxelgrid.extract_odms(voxel)
    logger.debug(f"ODM extraction took {time() - restart:.3f}s.")

    restart = time()
    voxel = kaolin.ops.voxelgrid.project_odms(odms)
    logger.debug(f"ODM projection took {time() - restart:.3f}s.")

    try:
        restart = time()
        vertices, faces = kaolin.ops.conversions.voxelgrids_to_trianglemeshes(voxel)
        vertices = vertices[0] / (voxel.size(-1) + 1)
        vertices -= 0.5
        faces = faces[0]
        logger.debug(f"GPU Marching Cubes took {time() - restart:.3f}s.")
    except torch.cuda.OutOfMemoryError as e:
        if try_cpu:
            restart = time()
            logger.error("Out of memory error during Marching Cubes on GPU. Trying CPU implementation.")
            mesh = extract(voxel.squeeze(0).cpu().numpy(), level=0.5, resolution=resolution, pad=False)
            vertices, faces = get_vertices_and_faces(mesh)
            vertices = torch.from_numpy(vertices).cuda()
            faces = torch.from_numpy(faces).cuda()
            logger.debug(f"CPU Marching Cubes took {time() - restart:.3f}s.")
        else:
            raise e

    voxel = voxel.squeeze(0).cpu().numpy()
    if save_voxel_path is not None:
        save_voxel_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(save_voxel_path), voxel=np.packbits(voxel))

    if smoothing_iterations > 0:
        restart = time()
        adj_sparse = kaolin.ops.mesh.adjacency_matrix(len(vertices), faces, sparse=True)
        neighbor_num = torch.sparse.sum(adj_sparse, dim=1).to_dense().view(-1, 1)
        for _ in range(smoothing_iterations):
            neighbor_sum = torch.sparse.mm(adj_sparse, vertices)
            vertices = neighbor_sum / neighbor_num
        logger.debug(f"Smoothing took {time() - restart:.3f}s.")

    if realign:
        restart = time()
        src_min, src_max = vertices.min(0, keepdim=True)[0], vertices.max(0, keepdim=True)[0]
        tgt_min, tgt_max = torch_vertices.min(0, keepdim=True)[0], torch_vertices.max(0, keepdim=True)[0]
        vertices = ((vertices - src_min) / (src_max - src_min + eps)) * (tgt_max - tgt_min) + tgt_min
        logger.debug(f"Realigning took {time() - restart:.3f}s.")

    return {"vertices": vertices.cpu().numpy(), "faces": faces.cpu().numpy()}


def voxelize(mesh: Union[Trimesh, Dict[str, np.ndarray]],
             min_value: float = -0.5,
             max_value: float = 0.5,
             resolution: int = 256) -> np.ndarray:
    x = np.linspace(min_value, max_value, resolution)
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    grid_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    vertices, faces = get_vertices_and_faces(mesh)
    points = Trimesh(vertices=vertices, faces=faces).sample(resolution ** 3)

    _, indices = KDTree(grid_points, leafsize=100).query(points)
    voxel = np.zeros(len(grid_points), dtype=bool)
    voxel[indices] = True

    return voxel.reshape((resolution,) * 3)
