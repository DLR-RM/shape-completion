from typing import Any, Dict, Union, Tuple, List, Optional

import mcubes
import numpy as np
import torch
from numpy import ndarray
from skimage.measure import marching_cubes
from torch import nn, Tensor, autograd, optim
import trimesh
from trimesh import Trimesh, PointCloud
from tqdm import trange

from libs import MISE, simplify_mesh
from utils import make_3d_grid, to_tensor, setup_logger, binary_from_multi_class, get_partnet_colors

logger = setup_logger(__name__)


class Generator:
    def __init__(self,
                 model: nn.Module,
                 points_batch_size: int = int(1e6),
                 threshold: float = 0.5,
                 extraction_class: int = 1,
                 refinement_steps: int = 0,
                 resolution0: int = 128,
                 upsampling_steps: int = 0,
                 with_normals: bool = False,
                 padding: float = 0.1,
                 simplify_nfaces: int = None,
                 use_skimage: bool = False,
                 sdf: bool = False,
                 bounds: Tuple[float, float] = (-0.5, 0.5)):
        assert not (extraction_class != 1 and upsampling_steps), "Non-binary data cannot be upsampled."
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_steps
        self.threshold = threshold
        self.extraction_class = extraction_class
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.simplify_nfaces = simplify_nfaces
        self.use_skimage = use_skimage
        self.sdf = sdf
        self.bounds = bounds
        self.voxel_size = np.ones(3)

        if self.upsampling_steps > 0:
            resolution = int(2 ** (np.log2(resolution0) + upsampling_steps))
            logger.debug(f"Using MISE with resolution 2^(log2({resolution0})+{upsampling_steps})={resolution}.")

    @torch.no_grad()
    def generate_pcd(self,
                     item: Dict[str, Any],
                     key: Optional[str] = None,
                     threshold: Optional[float] = None) -> PointCloud:
        item = {k: to_tensor(v, device=self.device) if isinstance(v, (ndarray, Tensor)) else v for k, v in item.items()}
        logits = self.model.predict(**item, key=key).to(torch.float32)
        if logits.dim() == 3:
            pred = logits.argmax(dim=-1).squeeze().cpu().numpy()
            if logits.size(-2) == item["inputs"].size(-2):
                points = item["inputs"].squeeze(0).cpu().numpy()
                colors = get_partnet_colors()[pred]
            elif "points" in item and logits.size(-2) == item["points"].size(-2):
                occ = pred != logits.size(-1) - 1
                points = item["points"].squeeze(0).cpu().numpy()[occ]
                colors = get_partnet_colors()[pred[occ]]
            else:
                raise NotImplementedError(f"Point cloud generation is not implemented for '{self.model.name}'.")
        elif logits.dim() == 2:
            threshold = self.threshold if threshold is None else threshold
            threshold = threshold if self.sdf else np.log(threshold / (1 - threshold))
            occ = logits.squeeze().cpu().numpy() >= threshold
            points = item["points"].squeeze(0).cpu().numpy()[occ]
            colors = None
        else:
            raise NotImplementedError(f"Invalid logits dimension. Expected 2 or 3, got {logits.dim()}.")
        pcd = PointCloud(points, colors=colors)
        return pcd

    @torch.no_grad()
    def generate_grid(self,
                      item: Dict[str, Any],
                      extraction_class: Optional[int] = None) -> Tuple[Union[ndarray, List[ndarray]], ndarray, Union[Tensor, Dict[str, Tensor], List[Tensor]]]:
        feature = None
        latent = None
        item = {k: to_tensor(v, device=self.device) if isinstance(v, (ndarray, Tensor)) else v for k, v in item.items()}
        inputs = item["inputs"]
        if self.model.name == "MCDropoutNet":
            logits_mean, probs_var = self.model.mc_sample(inputs)
            grid = [logits_mean.cpu().numpy(), probs_var.cpu().numpy()]
            points = self.get_query_points().numpy()
        elif self.model.name == "PSSNet":
            logits = self.model.predict_many(inputs)

            """
            meshes = list()
            for g in logits:
                mesh = self.extract_mesh(g.cpu().numpy(), c, threshold=0.4)
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                                 o3d.utility.Vector3iVector(mesh.faces))
                mesh.paint_uniform_color(np.random.uniform(0, 1, 3))
                mesh.compute_vertex_normals()
                meshes.append(mesh)
            o3d.visualization.draw_geometries(meshes)
            """

            probs = torch.sigmoid(logits)
            probs_mean = probs.mean(dim=0)
            probs_var = probs.var(dim=0)
            logits_mean = torch.log((probs_mean + 1e-6) / (1 - probs_mean + 1e-6))

            # mask = (probs > 0.1).sum(0)
            # sometimes = mask != len(probs)
            # probs_var[~sometimes] = 0

            grid = [logits_mean.cpu().numpy(), probs_var.cpu().numpy()]

            points = self.get_query_points().numpy()
        elif self.model.name == "ShapeFormer":
            grid, points = self.model.generate_grids(inputs)
        else:
            feature = self.model.encode(inputs)
            if self.model.name == "ONet":
                latent = self.model.get_z_from_prior((1,)).to(self.device)
            grid, points = self.generate_from_latent(inputs=inputs,
                                                     feature=feature,
                                                     latent=latent,
                                                     extraction_class=extraction_class)
        return grid, points, feature

    def generate_mesh(self,
                      item: Dict[str, Any],
                      threshold: Optional[float] = None,
                      extraction_class: Optional[int] = None) -> Union[Trimesh, List[Trimesh]]:
        grid, _, feature = self.generate_grid(item, extraction_class)
        if isinstance(grid, list):
            mesh = list()
            for g in grid:
                mesh.append(self.extract_mesh(g, feature, threshold, extraction_class))
        elif isinstance(grid, dict):
            mesh = list()
            for c, g in grid.items():
                color = (np.array(list(get_partnet_colors()[c]) + [1]) * 255).astype(np.uint8)
                m = self.extract_mesh(g, feature, threshold, extraction_class)
                m.visual.vertex_colors = color
                mesh.append(m)
            mesh = trimesh.util.concatenate(mesh)
        else:
            mesh = self.extract_mesh(grid, feature, threshold, extraction_class)
        return mesh

    def get_box_size(self) -> float:
        return (self.bounds[1] - self.bounds[0]) + self.padding

    def get_query_points(self) -> Tensor:
        return self.get_box_size() * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (self.resolution0,) * 3)

    def generate_from_latent(self,
                             inputs: Optional[Tensor] = None,
                             feature: Optional[Union[Tensor, Dict[str, Tensor], List[Tensor]]] = None,
                             latent: Optional[Tensor] = None,
                             extraction_class: Optional[int] = None) -> Tuple[Union[ndarray, Dict[int, ndarray]], ndarray]:
        if self.upsampling_steps == 0:
            points = self.get_query_points()
            values = self.eval_points(inputs=inputs, points=points, feature=feature, latent=latent)
            if values.dim() == 2:
                if extraction_class is None:
                    value_grid = dict()
                    for i in torch.unique(torch.argmax(values, dim=1)):
                        if i != values.size(1) - 1:
                            binary_values = binary_from_multi_class(values, occ_label=i)
                            value_grid[i] = binary_values.to(torch.float32).numpy().reshape((self.resolution0,) * 3)
                else:
                    if extraction_class == -1:
                        binary_values = binary_from_multi_class(values)
                    else:
                        binary_values = binary_from_multi_class(values, occ_label=extraction_class)
                    value_grid = binary_values.to(torch.float32).numpy().reshape((self.resolution0,) * 3)
            else:
                value_grid = values.to(torch.float32).numpy().reshape((self.resolution0,) * 3)
            points = points.numpy()
        else:
            points_list = list()
            threshold = self.threshold if self.sdf else np.log(self.threshold / (1 - self.threshold))
            mesh_extractor = MISE(self.resolution0, self.upsampling_steps, threshold)
            queries = mesh_extractor.query()
            while queries.shape[0] != 0:
                points = queries / mesh_extractor.resolution
                points = self.get_box_size() * (points - 0.5)
                points_list.append(points)
                points = torch.as_tensor(points, dtype=torch.float32).to(self.device)
                values = self.eval_points(inputs=inputs, points=points, feature=feature, latent=latent)
                if values.dim() == 2:
                    values = binary_from_multi_class(values)
                values = values.to(torch.float64).numpy()
                mesh_extractor.update(queries, values)
                queries = mesh_extractor.query()
            value_grid = mesh_extractor.to_dense()
            points = np.concatenate(points_list, axis=0)
        return value_grid, points

    @torch.no_grad()
    def eval_points(self,
                    points: Tensor,
                    feature: Union[Tensor, Dict[str, Tensor], List[Tensor]],
                    inputs: Optional[Tensor] = None,
                    latent: Optional[Tensor] = None) -> Tensor:
        p_split = torch.split(points, self.points_batch_size)
        predictions = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            try:
                out = self.model.predict(points=pi, feature=feature, inputs=inputs)
            except (NotImplementedError, AttributeError):
                if self.model.name == "ONet":
                    out = self.model.decode(pi, latent, feature)
                else:
                    out = self.model.decode(pi, feature)
            predictions.append(out.squeeze(0).cpu())
        predictions = torch.cat(predictions)
        return predictions

    def extract_mesh(self,
                     predictions: ndarray,
                     feature: Optional[Union[Tensor, Dict[str, Tensor], List[Tensor]]] = None,
                     threshold: Optional[float] = None,
                     extraction_class: Optional[int] = None) -> Trimesh:
        threshold = self.threshold if threshold is None else threshold
        threshold = threshold if self.sdf else np.log(threshold / (1 - threshold))
        box_size = self.get_box_size()
        self.voxel_size = box_size / (np.array(predictions.shape) - 1)

        if self.use_skimage:
            try:
                gradient_direction = "descent" if self.sdf else "ascent"
                vertices, faces, normals, _ = marching_cubes(volume=predictions,
                                                             level=threshold,
                                                             spacing=self.voxel_size,
                                                             gradient_direction=gradient_direction,
                                                             allow_degenerate=False)

                mesh = Trimesh(vertices,
                               faces,
                               vertex_normals=normals if self.with_normals else None,
                               process=False)

                offsets = np.repeat(0.5 * box_size, 3)
                mesh.apply_translation(-offsets)
            except ValueError:
                return Trimesh()
        else:
            if self.sdf:
                predictions *= -1
                threshold *= -1
            # Make sure that mesh is watertight
            occ_hat_padded = np.pad(predictions, 1, "constant", constant_values=-1e6)
            vertices, triangles = mcubes.marching_cubes(occ_hat_padded, threshold)
            vertices -= 1  # Undo padding

            # Normalize to bounding box
            vertices /= (np.array(predictions.shape) - 1)
            vertices = box_size * (vertices - 0.5)

            # Create mesh
            mesh = Trimesh(vertices,
                           triangles,
                           process=False,
                           validate=False)

        # Directly return if mesh is empty
        if mesh.vertices.shape[0] == 0:
            return mesh

        if self.simplify_nfaces is not None:
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)

        # Refine mesh
        if self.refinement_step > 0 and feature is not None:
            mesh = self.refine_mesh(mesh, predictions, feature, extraction_class)

        # Estimate normals if needed
        if self.with_normals and feature is not None:
            mesh = Trimesh(mesh.vertices,
                           mesh.faces,
                           vertex_normals=self.estimate_normals(mesh.vertices, feature, extraction_class),
                           process=False,
                           validate=False)
        return mesh

    def extract_uncertain_regions(self):
        pass

    @torch.enable_grad()
    def estimate_normals(self,
                         vertices: ndarray,
                         feature: Union[Tensor, Dict[str, Tensor], List[Tensor]],
                         extraction_class: Optional[int] = None,
                         normalize: bool = True) -> ndarray:
        vertices = torch.as_tensor(vertices, dtype=torch.float32)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(self.device)
            vi.requires_grad_()

            if hasattr(self.model, "predict"):
                logits = self.model.predict(points=vi, feature=feature)
            else:
                logits = self.model.decode(points=vi, feature=feature)

            if logits.dim() == 3:
                logits = binary_from_multi_class(logits, occ_label=extraction_class)

            out = logits.sum()
            out.backward()
            ni = -vi.grad
            if normalize:
                ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)
        return np.concatenate(normals, axis=0)

    @torch.enable_grad()
    def refine_mesh(self,
                    mesh: Trimesh,
                    predictions: Tensor,
                    c: Union[Tensor, Dict[str, Tensor], List[Tensor]],
                    threshold: float = None,
                    extraction_class: int = None) -> Trimesh:
        if threshold is None:
            threshold = self.threshold

        # Some shorthands
        n_x, n_y, n_z = predictions.shape
        assert (n_x == n_y == n_z)

        # Vertex parameter
        vertices = torch.from_numpy(mesh.vertices.astype(np.float32)).to(self.device)
        vertices = torch.nn.Parameter(vertices.clone(), requires_grad=True)

        # Faces of mesh
        faces = torch.from_numpy(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([vertices], lr=1e-4)

        for _ in trange(self.refinement_step, desc="Refining"):
            optimizer.zero_grad()

            # Loss
            face_vertices = vertices[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0]).astype(np.float32)
            eps = torch.from_numpy(eps).to(self.device)
            face_point = (face_vertices * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
            face_v2 = face_vertices[:, 2, :] - face_vertices[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / (face_normal.norm(dim=1, keepdim=True) + 1e-10)

            if self.model.name == "ONet":
                z = self.model.get_z_from_prior((1,)).to(self.device)
                logits = self.model.decode(face_point.unsqueeze(0), z, c)
            else:
                logits = self.model.decode(face_point.unsqueeze(0), c)

            if logits.dim() == 3:
                if extraction_class is None:
                    extraction_class = self.extraction_class
                face_value = torch.softmax(logits, dim=1)[:, extraction_class]
            else:
                face_value = torch.sigmoid(logits)
            normal_target = -autograd.grad([face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = normal_target / (normal_target.norm(dim=1, keepdim=True) + 1e-6)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()
        mesh.vertices = vertices.data.cpu().numpy()
        return mesh
