import gc
import logging
import os
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
from trimesh import Trimesh
from trimesh.primitives import Sphere

from utils import inv_trafo, stack_images, normalize, setup_logger

logger = setup_logger(__name__)


PACKAGES = ["blender", "pyrender", "open3d", "pytorch3d", "plotly"]
try:
    import bpy
    from blenderproc.python.utility.Initializer import clean_up, _Initializer
    from blenderproc.python.types.MeshObjectUtility import create_primitive, create_with_empty_mesh
    from blenderproc.python.types.LightUtility import Light
    from blenderproc.python.utility.MathUtility import build_transformation_mat
    from blenderproc.python.camera.CameraUtility import set_intrinsics_from_K_matrix, add_camera_pose
    from blenderproc.python.renderer.RendererUtility import (render, set_denoiser, set_world_background,
                                                             set_max_amount_of_samples, set_output_format,
                                                             set_noise_threshold, set_light_bounces, set_render_devices,
                                                             set_simplify_subdivision_render)
    from blenderproc.python.utility.Utility import stdout_redirected
    from blenderproc.python.loader.BlendLoader import load_blend
except (ImportError, RuntimeError):
    logger.warning("Could not import BPY and/or BlenderProc, 'blender' method disabled.")
    PACKAGES.remove("blender")
try:
    import pyrender
    from pyrender.shader_program import ShaderProgramCache
except ImportError:
    logger.warning("Could not import Pyrender, 'pyrender' mode disabled.")
    PACKAGES.remove("pyrender")
try:
    import open3d as o3d
except ImportError:
    logger.warning("Could not import Open3D, 'open3d' mode disabled.")
    PACKAGES.remove("open3d")
try:
    import torch
    from pytorch3d.renderer import (FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer,
                                    look_at_view_transform,
                                    look_at_rotation, PerspectiveCameras)
except ImportError:
    logger.warning("Could not import PyTorch and/or PyTorch3D, 'pytorch3d' mode disabled.")
    PACKAGES.remove("pytorch3d")
assert len(PACKAGES) > 0, "No rendering packages available, exiting."
logger.debug(f"Enabled modes: {PACKAGES}")


class Renderer:
    def __init__(self,
                 method: str = "auto",
                 width: int = 512,
                 height: int = 512,
                 render_color: bool = True,
                 render_depth: bool = False,
                 render_normal: bool = False,
                 offscreen: bool = True,
                 differentiable: bool = False,
                 raytracing: bool = False,
                 file_format: str = "PNG",
                 transparent_background: bool = False,
                 show: bool = False,
                 verbose: bool = False):
        if verbose:
            logger.setLevel(logging.DEBUG)

        assert method in PACKAGES + ["auto"], f"Method {method} not available, choose from {PACKAGES}."
        assert not ((render_normal and method == "pytorch3d") or (render_normal and offscreen and method == "open3d")), \
            "Normal rendering not supported in PyTorch3D or Open3D offscreen mode."
        assert not (differentiable and method not in ["pytorch3d", "auto"]), \
            "Differentiable rendering only supported in PyTorch3D."
        assert not (raytracing and method not in ["blender", "auto"]), "Raytracing only supported in Blender."
        assert not (show and "open3d" not in PACKAGES), "Visualization requires Open3D."
        assert file_format in ["PNG", "JPEG", "EXR"], f"File format {file_format} not supported."
        assert not (file_format != "PNG" and transparent_background), "Only PNG supports transparent background."

        self._method = method
        self._width = width
        self._height = height
        self.render_color = render_color
        self.render_depth = render_depth
        self.render_normal = render_normal
        self._offscreen = offscreen
        self._differentiable = differentiable
        self._raytracing = raytracing
        self._file_format = file_format
        self._transparent_background = transparent_background
        self._show = show
        self.verbose = verbose

        self.default_intrinsic = self._get_intrinsic(width, height)

        self.default_extrinsic_opengl = inv_trafo(self.look_at(np.array([1, 0.5, 1]), np.zeros(3)))
        self.default_extrinsic_opencv = self.default_extrinsic_opengl.copy()
        self.default_extrinsic_opencv[1, :] *= -1
        self.default_extrinsic_opencv[2, :] *= -1

        self.default_mesh_color = np.array([0.333, 0.593, 0.666])
        self.default_pcd_color = np.array([1, 0.49, 0.435])
        self.default_background_color = np.array([0.27, 0.32, 0.37])
        self.default_light_color = np.array([1, 0.4, 0.18])

        self.pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM")

        self._renderer = None

    @staticmethod
    def _get_intrinsic(width: int, height: int):
        return np.array([[width, 0, width / 2],
                         [0, width, height / 2],
                         [0, 0, 1]])

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value: int):
        if value == self._width:
            return
        self._width = value
        self.default_intrinsic = self._get_intrinsic(value, self.height)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value: int):
        if value == self._height:
            return
        self._height = value
        self.default_intrinsic = self._get_intrinsic(self.width, value)

    @property
    def method(self):
        if self._method in [None, "auto"]:
            self.choose_renderer()
        return self._method

    @method.setter
    def method(self, value: str):
        if value == self._method:
            return

        if value not in PACKAGES + ["auto"]:
            logger.warning(f"Method {value} not available, choose from {PACKAGES}.")
            return

        if value != self._method:
            self._method = value
            if value == "auto":
                self.choose_renderer()
            if self.method != "blender":
                self.init_renderer()

    @property
    def offscreen(self):
        return self._offscreen

    @offscreen.setter
    def offscreen(self, value: bool):
        if value == self._offscreen:
            return
        self._offscreen = value

        if not value and "PYOPENGL_PLATFORM" in os.environ:
            os.environ.pop("PYOPENGL_PLATFORM")

        if self.method in ["pyrender", "open3d"]:
            self.init_renderer()  # Open3D/Pyrender renderer needs to be re-initialized after offscreen change

    @property
    def differentiable(self):
        return self._differentiable

    @differentiable.setter
    def differentiable(self, value: bool = False):
        if value == self._differentiable:
            return

        if value and "pytorch3d" not in PACKAGES:
            logger.warning("Differentiable rendering requires PyTorch3D which is not available. Not enabling.")
            return

        self._differentiable = value
        if value and self.method != "pytorch3d":
            self.method = "pytorch3d"

    @property
    def raytracing(self):
        return self._raytracing

    @raytracing.setter
    def raytracing(self, value: bool = False):
        if value == self._raytracing:
            return

        if value and "blender" not in PACKAGES:
            logger.warning("Raytracing requires Blender which is not available. Not enabling.")
            return

        self._raytracing = value
        self.method = "blender"  # Always re-initialize renderer as raytracing might require a change of render engine
        self._renderer = None

    @property
    def renderer(self):
        if self._renderer is None:
            if self.method == "auto":
                self.choose_renderer()
            if self.method != "blender":
                self.init_renderer()
        return self._renderer

    @renderer.setter
    def renderer(self, value):
        self._renderer = value

    @property
    def show(self):
        return self._show

    @show.setter
    def show(self, value: bool):
        if value == self._show:
            return
        self._show = value
        if value:
            os.environ.pop("PYOPENGL_PLATFORM")

    @property
    def file_format(self):
        return self._file_format

    @file_format.setter
    def file_format(self, value: str):
        if value != "PNG" and self.transparent_background:
            logger.warning("Transparent background only supported for PNG files.")
            return
        if value == self._file_format:
            return
        self._file_format = value

    @property
    def transparent_background(self):
        return self._transparent_background

    @transparent_background.setter
    def transparent_background(self, value: bool):
        if value == self._transparent_background:
            return
        if value and self.file_format != "PNG":
            logger.warning("Transparent background only supported for PNG files.")
            return
        self._transparent_background = value

    def choose_renderer(self):
        if self.differentiable:
            if "pytorch3d" in PACKAGES:
                self.method = "pytorch3d"
            else:
                raise NotImplementedError("Differentiable rendering requires PyTorch3D.")
        elif self.raytracing:
            if "blender" in PACKAGES:
                self.method = "blender"
            else:
                raise NotImplementedError("Raytracing requires Blender.")
        else:
            if "pyrender" in PACKAGES:
                self.method = "pyrender"
            elif "open3d" in PACKAGES:
                self.method = "open3d"
            elif "blender" in PACKAGES:
                self.method = "blender"
            elif "pytorch3d" in PACKAGES:
                self.method = "pytorch3d"
        logger.debug(f"Using renderer: {self.method}")

    def init_renderer(self):
        if self.method == "blender":
            self.renderer = None

            with stdout_redirected(enabled=not self.verbose):
                clean_up(clean_up_camera=True)
                bpy.context.scene.render.engine = "CYCLES"
                set_render_devices()

            _Initializer.set_default_parameters()
            set_output_format(file_format=self.file_format)
            set_world_background(list(self.default_background_color), strength=0.2)
            try:
                bpy.context.scene.view_settings.look = "High Contrast"  # Blender < 4.0
            except TypeError:
                bpy.context.scene.view_settings.look = "AgX - High Contrast"

            if self.raytracing:
                set_denoiser("OPTIX")
                set_noise_threshold(0.01)
                set_max_amount_of_samples(100)
            else:
                bpy.context.scene.render.engine = "BLENDER_EEVEE"
            logger.debug(f"Using {bpy.context.scene.render.engine} rendering engine.")
            if self.transparent_background:
                if self.file_format == "PNG":
                    set_output_format(enable_transparency=True)
                else:
                    logger.warning("Transparent background only supported for PNG files.")
        elif self.method == "pyrender":
            if self.offscreen and not self.show:
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
            else:
                self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        elif self.method == "open3d":
            if self.offscreen:
                assert not self.show, "Open3D offscreen renderer and visualization not supported at the same time yet."
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
            else:
                self.renderer = o3d.visualization.Visualizer()
                self.renderer.create_window(width=self.width, height=self.height, visible=False)
        elif self.method == "pytorch3d":
            intrinsic = self.default_intrinsic.astype(np.float32)
            cameras = PerspectiveCameras(focal_length=((intrinsic[0, 0], intrinsic[1, 1]),),
                                         principal_point=((intrinsic[0, 2], intrinsic[1, 2]),),
                                         R=torch.from_numpy(self.default_extrinsic_opencv[:3, :3]).cuda(),
                                         T=torch.from_numpy(self.default_extrinsic_opencv[:3, 3]).cuda(),
                                         device=torch.device("cuda:0"),
                                         in_ndc=False,
                                         image_size=((self.height, self.width),))
            raster_settings = RasterizationSettings(image_size=(self.height, self.width),
                                                    blur_radius=0.0,
                                                    faces_per_pixel=1,
                                                    bin_size=0)
            self.renderer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        else:
            raise NotImplementedError("Unknown rendering method.")

    @staticmethod
    def look_at(eye: np.ndarray,
                target: np.ndarray,
                up: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
        z_axis = eye - target
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        return np.array([
            [x_axis[0], y_axis[0], z_axis[0], eye[0]],
            [x_axis[1], y_axis[1], z_axis[1], eye[1]],
            [x_axis[2], y_axis[2], z_axis[2], eye[2]],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def draw_cam(K, R, t, width, height, scale=1, color=None):
        if color is None:
            color = [0.8, 0.2, 0.8]

        # camera model scale
        s = 1 / scale

        # intrinsics
        Ks = np.array([[K[0, 0] * s, 0, K[0, 2]],
                       [0, K[1, 1] * s, K[1, 2]],
                       [0, 0, K[2, 2]]])
        Kinv = np.linalg.inv(Ks)

        # 4x4 transformation
        T = np.column_stack((R, t))
        T = np.vstack((T, (0, 0, 0, 1)))

        # axis
        axis = o3d.geometry.TriangleMesh().create_coordinate_frame(size=scale * 0.5).transform(T)

        # points in pixel
        points_pixel = [
            [0, 0, 0],
            [0, 0, 1],
            [width, 0, 1],
            [0, height, 1],
            [width, height, 1],
        ]

        # pixel to camera coordinate system
        points = [scale * Kinv @ p for p in points_pixel]

        # image plane
        width = abs(points[1][0]) + abs(points[3][0])
        height = abs(points[1][1]) + abs(points[3][1])
        plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
        plane.paint_uniform_color(color)
        plane.transform(T)
        plane.translate(R @ [points[1][0], points[1][1], scale])

        # pyramid
        points_in_world = [(R @ p + t) for p in points]
        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
        ]
        colors = [color for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_in_world),
            lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return [axis, plane, line_set]

    def visualize_scene(self, scene: Dict[str, Any]):
        intrinsic = scene["intrinsic"]
        extrinsic = scene["extrinsic"]
        objects = scene["objects"]
        lights = scene.get("lights", list())

        cam = list()
        for ext in extrinsic:
            inv_extrinsic = inv_trafo(ext)
            cam += self.draw_cam(intrinsic, inv_extrinsic[:3, :3], inv_extrinsic[:3, 3], self.width, self.height)

        meshes = list()
        for obj in objects:
            vertices, faces = obj["vertices"], obj["faces"]
            if faces is None:
                mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
            else:
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(obj["vertices"]),
                                                 o3d.utility.Vector3iVector(obj["faces"]))
                mesh.compute_vertex_normals()
            mesh.paint_uniform_color(obj["color"])
            meshes.append(mesh)
        for l in lights:
            light = o3d.geometry.TriangleMesh().create_sphere(radius=l["size"])
            light.paint_uniform_color(l["color"])
            light.translate(l["location"])
            meshes.append(light)
        o3d.visualization.draw_geometries(cam + meshes + [o3d.geometry.TriangleMesh().create_coordinate_frame(0.5)])

    @staticmethod
    def visualize_result(result: Dict[str, List[np.ndarray]],
                         color: bool = True,
                         depth: bool = True,
                         normal: bool = True):
        if color and "color" in result:
            plt.imshow(stack_images(result["color"]))
            plt.show()
        if depth and "depth" in result:
            depth = stack_images(result["depth"])
            depth[depth == 0] = 1.02 * depth.max()
            depth_map = normalize(depth, depth.min(), depth.max())
            cmap = plt.get_cmap("Greys")
            depth = cmap(depth_map)
            plt.imshow(depth)
            plt.show()
        if normal and "normal" in result:
            plt.imshow(stack_images(result["normal"]))
            plt.show()

    def render_blender(self,
                       vertices: Union[np.ndarray, List[np.ndarray]],
                       faces: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                       colors: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                       intrinsic: Optional[np.ndarray] = None,
                       extrinsic: Optional[Union[np.ndarray, List[np.ndarray]]] = None) -> Dict[str, List[np.ndarray]]:
        self.init_renderer()

        for index, (v, f, c) in enumerate(zip(vertices, faces, colors)):
            obj = create_with_empty_mesh(f"Object.{index:03d}")
            if f is None:
                if c is None:
                    c = self.default_pcd_color

                obj.get_mesh().from_pydata(v, [], [])
                obj.get_mesh().validate()
                obj.set_shading_mode("SMOOTH")

                if len(c) == len(v):
                    mesh = obj.get_mesh()
                    color_attr_name = "point_color"  # Name of the custom attribute
                    mesh.attributes.new(name=color_attr_name, type='FLOAT_COLOR', domain='POINT')

                    color_data = mesh.attributes[color_attr_name].data
                    for i, color in enumerate(c):
                        color_data[i].color = color / 255.0

                    material = obj.new_material(f"Material.{index:03d}")
                    blender_mat = material.blender_obj
                    blender_mat.use_nodes = True
                    bsdf = blender_mat.node_tree.nodes["Principled BSDF"]
                    nodes = blender_mat.node_tree.nodes
                    links = blender_mat.node_tree.links

                    attribute_node = nodes.new(type="ShaderNodeAttribute")
                    attribute_node.attribute_name = color_attr_name
                    links.new(attribute_node.outputs['Color'], bsdf.inputs['Base Color'])
                elif len(c) == 3:
                    material = obj.new_material(f"Material.{index:03d}")
                    material.set_principled_shader_value("Roughness", 0.2)
                    material.set_principled_shader_value("Base Color", [*c, 1])
                else:
                    raise ValueError(f"Unsupported color format: {c}")

                # Geometry nodes setup for Cycles
                if bpy.context.scene.render.engine == "CYCLES":
                    logger.debug("Rendering pointcloud using nodes.")
                    bpy.context.view_layer.objects.active = obj.blender_obj
                    bpy.ops.node.new_geometry_nodes_modifier()
                    geometry_nodes = obj.blender_obj.modifiers[0]
                    node_group = geometry_nodes.node_group
                    nodes = node_group.nodes
                    links = node_group.links

                    for node in nodes:
                        nodes.remove(node)

                    group_input = nodes.new(type="NodeGroupInput")
                    group_output = nodes.new(type="NodeGroupOutput")
                    mesh_to_points = nodes.new(type="GeometryNodeMeshToPoints")
                    set_material = nodes.new(type="GeometryNodeSetMaterial")

                    mesh_to_points.inputs["Radius"].default_value = 0.005
                    set_material.inputs["Material"].default_value = material.blender_obj

                    links.new(group_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
                    links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
                    links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])
                elif bpy.context.scene.render.engine == "BLENDER_EEVEE":
                    logger.debug("Rendering pointcloud using particle system.")
                    sphere = create_primitive("SPHERE", radius=0.1, location=(5, 0, 0))
                    sphere.set_shading_mode("SMOOTH")

                    material = sphere.new_material(f"Material {index}")
                    material.set_principled_shader_value("Roughness", 0.2)
                    material.set_principled_shader_value("Base Color", [*c, 1])

                    bpy.context.view_layer.objects.active = obj.blender_obj
                    particle_system = bpy.context.object.modifiers.new(name="Particle System", type='PARTICLE_SYSTEM')
                    particle_system.particle_system.settings.count = len(v)
                    particle_system.particle_system.settings.frame_end = 0
                    particle_system.particle_system.settings.emit_from = 'VERT'
                    particle_system.particle_system.settings.use_emit_random = False
                    particle_system.particle_system.settings.render_type = 'OBJECT'
                    particle_system.particle_system.settings.instance_object = sphere.blender_obj
                    obj.blender_obj.show_instancer_for_render = False
                else:
                    raise ValueError(f"Unsupported render engine: {bpy.context.scene.render.engine}")
            else:
                if c is None:
                    c = self.default_mesh_color

                obj.get_mesh().from_pydata(v, [], f)
                obj.get_mesh().validate()
                obj.set_shading_mode("FLAT")

                material = obj.new_material(f"Material.{index:03d}")
                material.set_principled_shader_value("Roughness", 0.7)
                material.set_principled_shader_value("Base Color", [*c, 1])

        if intrinsic is None:
            intrinsic = self.default_intrinsic
        set_intrinsics_from_K_matrix(K=intrinsic, image_width=self.width, image_height=self.height)

        if isinstance(extrinsic, np.ndarray):
            add_camera_pose(inv_trafo(extrinsic))
        else:
            for ext in extrinsic:
                add_camera_pose(inv_trafo(ext))

        if self.render_color:
            if bpy.context.scene.render.engine == "CYCLES":
                key_light = Light(light_type="AREA", name="key_light")
                key_light.set_location([-3, 0, 3])
                key_light.set_scale([0.1, 0.1, 0.1])
                key_light.set_energy(10)
                key_light.set_color([1, 1, 1])
                key_light.set_rotation_euler([0, np.deg2rad(-45), 0])
                key_light.blender_obj.data.spread = np.deg2rad(60)

                rim_light = Light(light_type="POINT", name="rim_light")
                rim_light.set_location([0, 10, 5])
                rim_light.set_scale([1, 1, 1])
                rim_light.set_energy(1000)
                rim_light.set_color(list(self.default_light_color))
                # rim_light.set_rotation_euler([np.deg2rad(-90), 0, 0])
                # rim_light.blender_obj.data.spread = np.pi

                cam_light = Light(light_type="POINT", name="cam_light")
                cam_light.set_location(inv_trafo(extrinsic)[:3, 3])
                cam_light.set_energy(2.5)
                cam_light.set_color([1, 1, 1])
            elif bpy.context.scene.render.engine == "BLENDER_EEVEE":
                key_light = Light(light_type="AREA", name="key_light")
                key_light.set_location([0, -1, 2.5])
                key_light.set_scale([0.2, 0.2, 0.2])
                key_light.set_energy(20)
                key_light.set_color([1, 1, 1])
                key_light.set_rotation_euler([np.deg2rad(30), 0, 0])
                key_light.blender_obj.data.use_contact_shadow = True
            else:
                raise ValueError(f"Unsupported render engine: {bpy.context.scene.render.engine}")

        with stdout_redirected(enabled=not self.verbose):
            data = render(verbose=False)

        # Todo: Does this setup prevent memory leaks?
        result = dict()
        if self.render_color:
            result["color"] = data["colors"].copy()
        if self.render_depth:
            result["depth"] = data["depth"].copy()
        if self.render_normal:
            result["normal"] = data["normals"].copy()

        # del data
        # gc.collect()

        return result

    def render_pyrender(self,
                        vertices: Union[np.ndarray, List[np.ndarray]],
                        faces: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                        colors: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                        intrinsic: Optional[np.ndarray] = None,
                        extrinsic: Optional[np.ndarray] = None,
                        points_as_spheres: bool = False,
                        point_size: int = 2):
        flags = pyrender.renderer.RenderFlags.NONE
        if self.render_depth and not (self.render_color or self.render_normal):
            flags = pyrender.RenderFlags.DEPTH_ONLY
        if self.offscreen:
            flags |= pyrender.RenderFlags.OFFSCREEN

        scene = pyrender.Scene()

        for index, (v, f, c) in enumerate(zip(vertices, faces, colors)):
            if f is None:
                if c is None:
                    c = self.default_pcd_color
                if c.ndim == 1:
                    c = [c] * len(v)

                if points_as_spheres:
                    if len(v) > 2500:
                        v = v[np.random.randint(0, len(v), 2500)]
                    for _v, _c in zip(v, c):
                        point = Sphere(radius=point_size * 0.005, center=_v).to_mesh()
                        if _c is not None:
                            point.visual.vertex_colors = _c
                        scene.add(pyrender.Mesh.from_trimesh(point))
                else:
                    scene.add(pyrender.Mesh.from_points(v, colors=c))
            else:
                if c is None:
                    c = self.default_mesh_color
                if c.ndim == 1:
                    c = [c] * len(v)

                mesh = Trimesh(v, f, vertex_colors=c, process=False)
                scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        camera = pyrender.IntrinsicsCamera(intrinsic[0, 0],
                                           intrinsic[1, 1],
                                           intrinsic[0, 2],
                                           intrinsic[1, 2],
                                           znear=0.01,
                                           zfar=10.0)
        scene.add(camera, pose=inv_trafo(extrinsic))

        if self.render_color:
            light = pyrender.PointLight(intensity=30)
            light_pose = np.eye(4)
            light_pose[:3, 3] = [3, 1.5, 0]
            scene.add(light, pose=light_pose)

        self.renderer.point_size = point_size
        if self.render_color or self.render_normal:
            if self.render_color:
                color, depth = self.renderer.render(scene, flags=flags)
            if self.render_normal:
                program_cache = self.renderer._renderer._program_cache
                shader_dir = Path(__file__).parent.parent.parent / "utils" / "assets" / "shaders"
                self.renderer._renderer._program_cache = ShaderProgramCache(shader_dir=shader_dir)
                normal, depth = self.renderer.render(scene, flags=flags)
                normal /= 255
                self.renderer._renderer._program_cache = program_cache
        else:
            depth = self.renderer.render(scene, flags=flags)
        depth[depth >= 10.0] = 0

        result = dict()
        if self.render_color:
            result["color"] = color
        if self.render_depth:
            result["depth"] = depth
        if self.render_normal:
            result["normal"] = normal
        return result

    def render_open3d(self,
                      vertices: Union[np.ndarray, List[np.ndarray]],
                      faces: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                      colors: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                      intrinsic: Optional[np.ndarray] = None,
                      extrinsic: Optional[np.ndarray] = None,
                      points_as_spheres: bool = False,
                      point_size: int = 2):
        meshes = list()
        for index, (v, f, c) in enumerate(zip(vertices, faces, colors)):
            if c is not None and c.ndim == 1:
                c = [c] * len(v)
            if f is None:
                if points_as_spheres:
                    if len(v) > 2500:
                        v = v[np.random.randint(0, len(v), 2500)]
                    for _v, _c in zip(v, c):
                        point = o3d.geometry.TriangleMesh().create_sphere(radius=point_size * 0.005, resolution=10)
                        point.translate(_v)
                        if _c is not None:
                            point.paint_uniform_color(_c)
                        point.compute_vertex_normals()
                        meshes.append(point)
                else:
                    mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))
                    if c is not None:
                        mesh.colors = o3d.utility.Vector3dVector(c)
                    meshes.append(mesh)
            else:
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                                 o3d.utility.Vector3iVector(f))
                if c is not None:
                    mesh.vertex_colors = o3d.utility.Vector3dVector(c)
                if self.render_color or self.render_normal:
                    mesh.compute_vertex_normals()
                meshes.append(mesh)

        if self.offscreen:
            for index, mesh in enumerate(meshes):
                material = o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultLit"
                material.base_roughness = 0.5
                material.point_size = point_size
                self.renderer.scene.add_geometry(f"mesh_{index}", mesh, material)

            self.renderer.setup_camera(intrinsic, extrinsic, self.width, self.height)

            if self.render_color:
                self.renderer.scene.set_background(np.ones(4))
                self.renderer.scene.scene.enable_indirect_light(False)
                self.renderer.scene.scene.enable_sun_light(True)
                self.renderer.scene.scene.set_sun_light([-0.707, -0.707, -0.707], [1.0, 1.0, 1.0], 50000)
                """
                self.renderer.scene.scene.add_point_light("light",
                                                          np.ones(3),
                                                          np.array([3, 1.5, 0]),
                                                          75000,
                                                          10000,
                                                          True)
                """
                color = np.asarray(self.renderer.render_to_image())

            if self.render_normal:
                raise NotImplementedError("Normal rendering is not supported yet by Open3D offscreen renderer.")

            if self.render_depth:
                depth = np.asarray(self.renderer.render_to_depth_image(z_in_view_space=True))
                depth[depth == np.inf] = 0

            self.renderer.scene.clear_geometry()
        else:
            ctr = self.renderer.get_view_control()
            camera_parameters = ctr.convert_to_pinhole_camera_parameters()
            camera_parameters.intrinsic.intrinsic_matrix = intrinsic
            camera_parameters.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)

            for mesh in meshes:
                self.renderer.add_geometry(mesh)

            if self.render_color:
                color = np.asarray(self.renderer.capture_screen_float_buffer(do_render=True))
            if self.render_normal:
                self.renderer.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
                normal = np.asarray(self.renderer.capture_screen_float_buffer(do_render=True))
            if self.render_depth:
                depth = np.asarray(self.renderer.capture_depth_float_buffer(do_render=True))

            self.renderer.clear_geometries()

        result = dict()
        if self.render_color:
            result["color"] = color
        if self.render_depth:
            result["depth"] = depth
        if self.render_normal:
            result["normal"] = normal

        return result

    def render_pytorch3d(self,
                         vertices: Union[np.ndarray, List[np.ndarray]],
                         faces: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                         colors: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                         intrinsic: Optional[np.ndarray] = None,
                         extrinsic: Optional[Union[np.ndarray, List[np.ndarray]]] = None):
        raise NotImplementedError("Pytorch3D rendering is not implemented yet.")

    def render(self,
               vertices: Union[np.ndarray, List[np.ndarray]],
               faces: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
               colors: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
               intrinsic: Optional[np.ndarray] = None,
               extrinsic: Optional[Union[np.ndarray, List[np.ndarray]]] = None):
        return self(vertices, faces, colors, intrinsic, extrinsic)

    def __call__(self,
                 vertices: Union[np.ndarray, List[np.ndarray]],
                 faces: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                 colors: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                 intrinsic: Optional[np.ndarray] = None,
                 extrinsic: Optional[Union[np.ndarray, List[np.ndarray]]] = None) -> Dict[str, np.ndarray]:
        vertices = vertices if isinstance(vertices, list) else [vertices]
        faces = faces if isinstance(faces, list) else [faces]
        colors = colors if isinstance(colors, list) else [colors] * len(vertices)

        """
        if self.show:
            objects = list()
            for v, f, c in zip(vertices, faces, colors):
                objects.append({"vertices": v, "faces": f, "color": c})
            lights = [{"location": [0, 0.75, 1.5], "color": [1, 1, 0], "size": 0.1},
                      {"location": [0, 0.75, -1.5], "color": [1, 1, 0], "size": 0.1},
                      {"location": [1, 1, 0], "color": [1, 1, 0], "size": 0.5},
                      {"location": [-1, 1, 0], "color": [1, 1, 0], "size": 0.5}]
            
            extrinsic_opengl = list()
            for ext in extrinsic:
                ext = ext.copy()
                ext[1, :] *= -1
                ext[2, :] *= -1
                extrinsic_opengl.append(ext)
            scene = {"intrinsic": intrinsic,
                     "extrinsic": extrinsic_opengl,
                     "objects": objects,
                     "lights": lights}
            self.visualize_scene(scene)
        """

        if self.method == "blender":
            extrinsic = self.default_extrinsic_opengl if extrinsic is None else extrinsic
            result = self.render_blender(vertices, faces, colors, intrinsic, extrinsic)
        elif self.method == "pyrender":
            intrinsic = self.default_intrinsic if intrinsic is None else intrinsic
            extrinsic = self.default_extrinsic_opengl if extrinsic is None else extrinsic
            result = self.render_pyrender(vertices, faces, colors, intrinsic, extrinsic)
        elif self.method == "open3d":
            intrinsic = self.default_intrinsic if intrinsic is None else intrinsic
            extrinsic = self.default_extrinsic_opencv if extrinsic is None else extrinsic
            result = self.render_open3d(vertices, faces, colors, intrinsic, extrinsic)
        elif self.method == "pytorch3d":
            intrinsic = self.default_intrinsic if intrinsic is None else intrinsic
            extrinsic = self.default_extrinsic_opengl if extrinsic is None else extrinsic
            result = self.render_pytorch3d(vertices, faces, colors, intrinsic, extrinsic)
        else:
            raise ValueError(f"Unknown rendering method: {self.method}")

        if self.show:
            self.visualize_result(result)

        return result

    def __del__(self):
        if "PYOPENGL_PLATFORM" in os.environ:
            os.environ.pop("PYOPENGL_PLATFORM")
        elif self.pyopengl_platform is not None:
            os.environ["PYOPENGL_PLATFORM"] = self.pyopengl_platform

        if isinstance(self.renderer, pyrender.OffscreenRenderer):
            self.renderer.delete()
        elif isinstance(self.renderer, o3d.visualization.Visualizer):
            self.renderer.destroy_window()
            self.renderer.close()
