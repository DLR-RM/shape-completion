"""Add libkinect depth simulation to existing HDF5 datasets.

This script post-processes existing HDF5 files to add the `kinect_sim` modality,
which simulates structured light Kinect v1 depth sensing with stereo matching
artifacts (IR projector occlusion shadows, missing depth at silhouettes).

Usage:
    python -m process.scripts.add_kinect_sim \
        --input-dir /path/to/hdf5s \
        --shapenet-dir /path/to/shapenet
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import trimesh
import tyro
from loguru import logger
from tqdm import tqdm

from libs.libkinect import KinectSimCython, NoiseType
from utils.src.runtime import optional_dependency_summary, suppress_known_optional_dependency_warnings


@dataclass
class Config:
    input_dir: Path
    """Directory containing HDF5 files."""
    shapenet_dir: Path | None = None
    """Path to ShapeNet directory for loading meshes."""
    overwrite: bool = False
    """Overwrite existing kinect_sim data."""
    noise: str = "perlin"
    """Noise type: none, gaussian, perlin, simplex."""
    verbose: bool = False
    """Enable verbose logging."""


def create_plane_mesh(size: float = 5.0) -> trimesh.Trimesh:
    """Create plane mesh matching BlenderProc's default PLANE primitive."""
    half = size / 2.0
    verts = np.array(
        [
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return trimesh.Trimesh(vertices=verts, faces=faces)


def _as_trimesh(mesh_obj: Any, source: Path) -> trimesh.Trimesh:
    if isinstance(mesh_obj, trimesh.Scene):
        mesh_obj = mesh_obj.dump(concatenate=True)
        if isinstance(mesh_obj, list):
            mesh_obj = trimesh.util.concatenate(mesh_obj)
    if isinstance(mesh_obj, trimesh.Trimesh):
        return mesh_obj
    raise TypeError(f"Expected trimesh.Trimesh for {source}, got {type(mesh_obj)}.")


def load_shapenet_mesh(shapenet_dir: Path, name: str) -> trimesh.Trimesh:
    """Load mesh from ShapeNet given object name like '02958343_abc123' or '02958343_abc123_1'.

    Args:
        shapenet_dir: Root ShapeNet directory.
        name: Object name in format 'category_objectid' or 'category_objectid_instance'.

    Returns:
        Loaded and merged trimesh object.
    """
    parts = name.split("_")
    category = parts[0]
    obj_id = parts[1]

    # Try common ShapeNet file locations
    suffixes = [
        "models/model_normalized.obj",
        "model.obj",
        "model_normalized.obj",
        "models/model.obj",
    ]

    for suffix in suffixes:
        path = shapenet_dir / category / obj_id / suffix
        if path.exists():
            mesh = trimesh.load(path, force="mesh", process=False)
            return _as_trimesh(mesh, path)

    # Try glTF
    for ext in [".glb", ".gltf"]:
        path = shapenet_dir / category / obj_id / f"model{ext}"
        if path.exists():
            mesh = trimesh.load(path, force="mesh", process=False)
            return _as_trimesh(mesh, path)

    raise FileNotFoundError(f"Could not find mesh for {name} in {shapenet_dir}")


def merge_meshes_to_camera(
    meshes: list[trimesh.Trimesh],
    poses: list[np.ndarray],
    scales: list[np.ndarray],
    extrinsic: np.ndarray,
    surface_mesh: trimesh.Trimesh | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge meshes into single mesh in camera coordinates.

    Args:
        meshes: List of trimesh objects (already normalized).
        poses: World poses for each mesh (4x4 matrices).
        scales: Scale factors for each mesh.
        extrinsic: World-to-camera transform (4x4, OpenCV convention).
        surface_mesh: Optional surface mesh (already in world coords).

    Returns:
        vertices: (N, 3) float32 in camera coords.
        faces: (M, 3) int32.
    """
    all_verts = []
    all_faces = []
    vert_offset = 0

    for mesh, pose, scale in zip(meshes, poses, scales, strict=False):
        verts = mesh.vertices.copy()

        # Apply scale
        scale = np.asarray(scale)
        if scale.ndim == 0:
            scale = np.array([scale, scale, scale])
        verts = verts * scale

        # Apply world pose
        pose = np.asarray(pose)
        verts_world = verts @ pose[:3, :3].T + pose[:3, 3]

        # Transform to camera coords
        verts_cam = verts_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]

        all_verts.append(verts_cam)
        all_faces.append(mesh.faces + vert_offset)
        vert_offset += len(verts_cam)

    if surface_mesh is not None:
        verts_cam = surface_mesh.vertices @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        all_verts.append(verts_cam)
        all_faces.append(surface_mesh.faces + vert_offset)

    return (
        np.vstack(all_verts).astype(np.float32),
        np.vstack(all_faces).astype(np.int32),
    )


def main(cfg: Config):
    logger.remove()
    level = "DEBUG" if cfg.verbose else "INFO"
    logger.add(lambda msg: tqdm.write(msg, end=""), level=level, colorize=True)
    suppress_known_optional_dependency_warnings()
    logger.info("Optional dependency summary:\n" + "\n".join(optional_dependency_summary()))

    noise_map = {
        "none": NoiseType.NONE,
        "gaussian": NoiseType.GAUSSIAN,
        "perlin": NoiseType.PERLIN,
        "simplex": NoiseType.SIMPLEX,
    }
    noise_type = noise_map.get(cfg.noise.lower(), NoiseType.PERLIN)

    sim = KinectSimCython()
    hdf5_files = sorted(cfg.input_dir.glob("*.hdf5"))

    if not hdf5_files:
        logger.warning(f"No HDF5 files found in {cfg.input_dir}")
        return

    logger.info(f"Processing {len(hdf5_files)} HDF5 files from {cfg.input_dir}")

    for hdf5_path in tqdm(hdf5_files, desc="Processing HDF5 files"):
        try:
            with h5py.File(hdf5_path, "r+") as f:
                if "kinect_sim" in f and not cfg.overwrite:
                    logger.debug(f"Skipping {hdf5_path.name} (kinect_sim exists)")
                    continue

                # Load metadata
                data_node = f["data"]
                if not isinstance(data_node, h5py.Dataset):
                    raise TypeError(f"Expected dataset 'data' in {hdf5_path.name}, got {type(data_node)}")
                data_raw = data_node[()]
                if isinstance(data_raw, bytes):
                    metadata_raw = data_raw.decode("utf-8")
                elif isinstance(data_raw, np.bytes_):
                    metadata_raw = data_raw.tobytes().decode("utf-8")
                else:
                    raise TypeError(f"Expected bytes payload in {hdf5_path.name}:data, got {type(data_raw)}")
                metadata = json.loads(metadata_raw)
                intrinsic = np.array(metadata["intrinsic"])
                extrinsic = np.array(metadata["extrinsic"])

                # Get image dimensions from depth
                if "depth" not in f:
                    logger.warning(f"No depth in {hdf5_path.name}, skipping")
                    continue
                depth_node = f["depth"]
                if not isinstance(depth_node, h5py.Dataset):
                    logger.warning(f"Dataset 'depth' in {hdf5_path.name} is not an HDF5 dataset, skipping")
                    continue
                depth_shape = depth_node.shape
                if len(depth_shape) < 2:
                    logger.warning(f"Dataset 'depth' in {hdf5_path.name} has invalid shape {depth_shape}, skipping")
                    continue
                height, width = int(depth_shape[0]), int(depth_shape[1])

                # Load meshes
                meshes = []
                poses = []
                scales = []

                names = list(metadata.get("names", []))
                meta_poses = list(metadata.get("poses", []))
                meta_scales = list(metadata.get("scales", []))

                if not names or cfg.shapenet_dir is None:
                    logger.warning(f"No object names or shapenet_dir not set for {hdf5_path.name}, skipping")
                    continue

                for name, pose, scale in zip(names, meta_poses, meta_scales, strict=False):
                    try:
                        mesh = load_shapenet_mesh(cfg.shapenet_dir, cast(str, name))
                        meshes.append(mesh)
                        poses.append(np.array(pose))
                        scales.append(np.array(scale))
                    except FileNotFoundError as e:
                        logger.warning(f"Mesh not found: {e}")
                        continue

                if not meshes:
                    logger.warning(f"No meshes loaded for {hdf5_path.name}, skipping")
                    continue

                # Create surface mesh if needed
                surface_mesh = None
                surface_type = metadata.get("surface")
                if surface_type == "plane":
                    surface_mesh = create_plane_mesh(size=5.0)

                # Merge meshes to camera coordinates
                vertices, faces = merge_meshes_to_camera(meshes, poses, scales, extrinsic, surface_mesh)

                # Run simulation
                fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
                cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])

                depth_sim = sim.simulate(
                    vertices,
                    faces,
                    width=width,
                    height=height,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    noise=noise_type,
                    verbose=cfg.verbose,
                )

                # Write to HDF5
                if "kinect_sim" in f:
                    del f["kinect_sim"]
                f.create_dataset("kinect_sim", data=depth_sim, compression="gzip")

                logger.debug(f"Added kinect_sim to {hdf5_path.name}")

        except Exception as e:
            logger.error(f"Error processing {hdf5_path.name}: {e}")
            if cfg.verbose:
                import traceback

                traceback.print_exc()

    logger.info("Done.")


if __name__ == "__main__":
    main(tyro.cli(Config))
