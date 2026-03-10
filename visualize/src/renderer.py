import os
from logging import DEBUG
from pathlib import Path
from typing import Any, cast

import numpy as np
from matplotlib import pyplot as plt
from pykdtree.kdtree import KDTree
from trimesh import Trimesh
from trimesh.primitives import Sphere

from utils import (
    DEBUG_LEVEL_1,
    DEBUG_LEVEL_2,
    convert_extrinsic,
    depth_to_image,
    draw_camera,
    inv_trafo,
    look_at,
    setup_logger,
    stack_images,
)

logger = setup_logger(__name__)

PACKAGES = ["blender", "pyrender", "open3d", "pytorch3d", "plotly"]
try:
    import bpy
    from blenderproc.python.camera.CameraUtility import (  # pyright: ignore[reportMissingImports]
        add_camera_pose,
        rotation_from_forward_vec,
        set_intrinsics_from_K_matrix,
    )
    from blenderproc.python.renderer.RendererUtility import (  # pyright: ignore[reportMissingImports]
        render,
        set_denoiser,
        set_max_amount_of_samples,
        set_noise_threshold,
        set_output_format,
        set_render_devices,
        set_world_background,
    )
    from blenderproc.python.types.LightUtility import Light  # pyright: ignore[reportMissingImports]
    from blenderproc.python.types.MeshObjectUtility import (  # pyright: ignore[reportMissingImports]
        create_primitive,
        create_with_empty_mesh,
    )
    from blenderproc.python.utility.Initializer import _Initializer, clean_up  # pyright: ignore[reportMissingImports]
    from blenderproc.python.utility.Utility import stdout_redirected  # pyright: ignore[reportMissingImports]
except (ImportError, RuntimeError) as e:
    logger.warning(f"BPY and/or BlenderProc import failed, 'blender' method disabled: {e}")
    PACKAGES.remove("blender")
try:
    import pyrender
    from pyrender.shader_program import ShaderProgramCache
except ImportError as e:
    logger.warning(f"Pyrender import failed, 'pyrender' mode disabled: {e}")
    PACKAGES.remove("pyrender")
try:
    import open3d as o3d
except ImportError as e:
    logger.warning(f"Open3D import failed, 'open3d' mode disabled: {e}")
    PACKAGES.remove("open3d")
try:
    import torch
    from pytorch3d.renderer import (
        Materials,
        MeshRasterizer,
        MeshRenderer,
        PerspectiveCameras,
        PointLights,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
    )
    from pytorch3d.structures import Meshes
except ImportError as e:
    logger.warning(f"PyTorch and/or PyTorch3D import failed, 'pytorch3d' mode disabled. {e}")
    PACKAGES.remove("pytorch3d")
assert len(PACKAGES) > 0, "No rendering packages available, exiting."
logger.debug(f"Enabled modes: {PACKAGES}")


class Renderer:
    def __init__(
        self,
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
    ):
        assert method in [*PACKAGES, "auto", "cycles", "eevee"], (
            f"Method {method} not available, choose from {PACKAGES}."
        )
        assert not (
            (render_normal and method == "pytorch3d") or (render_normal and offscreen and method == "open3d")
        ), "Normal rendering not supported in PyTorch3D or Open3D offscreen mode."
        assert not (differentiable and method not in ["pytorch3d", "auto"]), (
            "Differentiable rendering only supported in PyTorch3D."
        )
        assert not (raytracing and method not in ["blender", "cycles", "auto"]), "Raytracing only supported in Blender."
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

        self.default_intrinsic = self._get_intrinsic(width, height)
        self.default_extrinsic_opengl = inv_trafo(look_at(np.array([1, 0.5, 1]), np.zeros(3)))
        self.default_extrinsic_opencv = convert_extrinsic(
            cast(np.ndarray, self.default_extrinsic_opengl), "opengl", "opencv"
        )

        self.default_mesh_color = np.array([0.333, 0.593, 0.666])
        self.default_pcd_color = np.array([1, 0.49, 0.435])
        self.default_background_color = np.array([0.27, 0.32, 0.37])
        self.default_light_color = np.array([1, 0.4, 0.18])

        self.pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM")

        self._renderer = None

    @staticmethod
    def _get_intrinsic(width: int, height: int):
        return np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]])

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

        if value not in [*PACKAGES, "auto", "cycles", "eevee"]:
            logger.warning(f"Method {value} not available, choose from {PACKAGES}.")
            return

        self._method = value
        if value == "auto":
            self.choose_renderer()
        elif value == "cycles":
            self.raytracing = True
        self.init_renderer()

    @property
    def offscreen(self):
        return self._offscreen

    @offscreen.setter
    def offscreen(self, value: bool):
        if value == self._offscreen:
            return
        self._offscreen = value

        if not value:
            os.environ.pop("PYOPENGL_PLATFORM", None)

        self.init_renderer()

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

        if not torch.cuda.is_available():
            logger.warning("Differentiable rendering requires CUDA which is not available. Not enabling.")
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
        self.method = "cycles"  # Always re-initialize renderer as raytracing might require a change of render engine
        self._renderer = None

    @property
    def renderer(self):
        if self._renderer is None:
            if self.method == "auto":
                self.choose_renderer()
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
            os.environ.pop("PYOPENGL_PLATFORM", None)

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
            if "pytorch3d" in PACKAGES and torch.cuda.is_available():
                self.method = "pytorch3d"
            else:
                raise ImportError("Differentiable rendering requires PyTorch3D and CUDA.")
        elif self.raytracing:
            if "blender" in PACKAGES:
                self.method = "cycles"
            else:
                raise ImportError("Raytracing requires Blender.")
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
        self.renderer = None
        if self.method in ["blender", "cycles", "eevee"]:
            engine = "CYCLES" if self.method == "cycles" or self.raytracing else "BLENDER_EEVEE"

            with stdout_redirected(enabled=not logger.isEnabledFor(DEBUG_LEVEL_1)):
                clean_up(clean_up_camera=True)
                scene = cast(Any, bpy.context.scene)
                scene.render.engine = engine
                set_render_devices()

            _Initializer.set_default_parameters()
            set_output_format(file_format=self.file_format)
            set_world_background(list(self.default_background_color), strength=0.15)

            scene = cast(Any, bpy.context.scene)
            scene.view_settings.view_transform = "Filmic"
            scene.view_settings.look = "Medium Contrast"

            if engine == "CYCLES":
                set_denoiser("OPTIX")
                set_noise_threshold(0.01)
                set_max_amount_of_samples(100)
            else:
                if self.offscreen and not self.show:
                    os.environ["PYOPENGL_PLATFORM"] = "egl"
                scene.render.engine = "BLENDER_EEVEE"
            logger.debug(f"Using {scene.render.engine} rendering engine.")
            if self.transparent_background:
                if self.file_format == "PNG":
                    set_output_format(enable_transparency=True)
                else:
                    logger.warning("Transparent background only supported for PNG files.")
        elif self.method == "pyrender":
            if self.offscreen:
                os.environ["PYOPENGL_PLATFORM"] = "egl"
            self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        elif self.method == "open3d":
            if self.offscreen:
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                self.renderer = cast(Any, o3d).visualization.rendering.OffscreenRenderer(self.width, self.height)
            else:
                self.renderer = cast(Any, o3d).visualization.Visualizer()
                self.renderer.create_window(width=self.width, height=self.height, visible=False)
        elif self.method == "pytorch3d":
            logger.warning("PyTorch3D renderer not yet properly implemented.")
        else:
            raise NotImplementedError("Unknown rendering method.")

    def visualize_scene(self, scene: dict[str, Any]):
        intrinsic = scene["intrinsic"]
        extrinsic = scene["extrinsic"]
        objects = scene["objects"]
        lights = scene.get("lights", list())

        cam = list()
        for ext in extrinsic:
            cam += draw_camera(intrinsic, ext, self.width, self.height)

        meshes = list()
        for obj in objects:
            vertices, faces = obj["vertices"], obj["faces"]
            if faces is None:
                mesh = cast(Any, o3d).geometry.PointCloud(cast(Any, o3d).utility.Vector3dVector(vertices))
            else:
                mesh = cast(Any, o3d).geometry.TriangleMesh(
                    cast(Any, o3d).utility.Vector3dVector(obj["vertices"]),
                    cast(Any, o3d).utility.Vector3iVector(obj["faces"]),
                )
                mesh.compute_vertex_normals()
            mesh.paint_uniform_color(obj["color"])
            meshes.append(mesh)
        for light_spec in lights:
            light = cast(Any, o3d).geometry.TriangleMesh().create_sphere(radius=light_spec["size"])
            light.paint_uniform_color(light_spec["color"])
            light.translate(light_spec["location"])
            meshes.append(light)
        cast(Any, o3d).visualization.draw_geometries(
            cam + meshes + [cast(Any, o3d).geometry.TriangleMesh().create_coordinate_frame(0.5)]
        )

    @staticmethod
    def visualize_result(
        result: dict[str, np.ndarray | list[np.ndarray]],
        color: bool = True,
        depth: bool = True,
        normal: bool = True,
    ):
        if color and "color" in result:
            color_data = result["color"]
            plt.imshow(
                stack_images(cast(list[np.ndarray | Any], color_data)) if isinstance(color_data, list) else color_data
            )
            plt.axis("off")
            plt.show()
        if depth and "depth" in result:
            depth_map = result["depth"]
            if isinstance(depth_map, list):
                depth_img = stack_images([depth_to_image(d) for d in depth_map])
            else:
                depth_img = depth_to_image(depth_map)
            plt.imshow(depth_img)
            plt.axis("off")
            plt.show()
        if normal and "normal" in result:
            normal_data = result["normal"]
            plt.imshow(
                stack_images(cast(list[np.ndarray | Any], normal_data)) if isinstance(normal_data, list) else normal_data
            )
            plt.axis("off")
            plt.show()

    def render_blender(
        self,
        vertices: np.ndarray | list[np.ndarray],
        faces: np.ndarray | list[np.ndarray] | None = None,
        colors: np.ndarray | list[np.ndarray] | None = None,
        normals: np.ndarray | list[np.ndarray] | None = None,
        intrinsic: np.ndarray | None = None,
        extrinsic: np.ndarray | list[np.ndarray] | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        self.init_renderer()
        vertices_list: list[np.ndarray] = vertices if isinstance(vertices, list) else [vertices]
        faces_list = cast(list[np.ndarray | None], faces if isinstance(faces, list) else [faces] * len(vertices_list))
        colors_list = cast(
            list[np.ndarray | None], colors if isinstance(colors, list) else [colors] * len(vertices_list)
        )

        for index, (v, f, c) in enumerate(zip(vertices_list, faces_list, colors_list, strict=False)):
            obj = create_with_empty_mesh(f"Object.{index:03d}")
            if f is None:  # Render pointcloud
                # Compute average distance between points for point size
                kdtree = KDTree(v)
                distances, _ = kdtree.query(v, k=2)
                nearest_neighbor_distances = distances[:, 1]
                point_size = np.quantile(nearest_neighbor_distances, 0.8)

                if c is None:
                    c = self.default_pcd_color

                obj.get_mesh().from_pydata(v, [], [])
                obj.get_mesh().validate()
                obj.set_shading_mode("SMOOTH")

                if len(c) == len(v):
                    mesh = obj.get_mesh()
                    color_attr_name = "point_color"
                    mesh.attributes.new(name=color_attr_name, type="FLOAT_COLOR", domain="POINT")

                    color_data = mesh.attributes[color_attr_name].data
                    for i, color in enumerate(c):
                        color_data[i].color = np.array([*(color / 255.0), 1])

                    material = obj.new_material(f"Material.{index:03d}")
                    blender_mat = cast(Any, material.blender_obj)
                    blender_mat.use_nodes = True
                    bsdf = blender_mat.node_tree.nodes["Principled BSDF"]
                    nodes = blender_mat.node_tree.nodes
                    links = blender_mat.node_tree.links

                    attribute_node = nodes.new(type="ShaderNodeAttribute")
                    attribute_node.attribute_name = color_attr_name
                    links.new(attribute_node.outputs["Color"], bsdf.inputs["Base Color"])
                elif len(c) == 3:
                    material = obj.new_material(f"Material.{index:03d}")
                    material.set_principled_shader_value("Roughness", 0.9)
                    material.set_principled_shader_value("Base Color", [*c, 1])
                else:
                    raise ValueError(f"Unsupported color format: {c}")

                # Geometry nodes setup for Cycles
                scene = cast(Any, bpy.context.scene)
                if scene.render.engine == "CYCLES":
                    logger.debug_level_2("Rendering pointcloud using nodes")
                    cast(Any, bpy.context.view_layer).objects.active = obj.blender_obj
                    bpy.ops.node.new_geometry_nodes_modifier()
                    geometry_nodes = cast(Any, obj.blender_obj).modifiers[0]
                    node_group = geometry_nodes.node_group
                    nodes = node_group.nodes
                    links = node_group.links

                    for node in nodes:
                        nodes.remove(node)

                    group_input = nodes.new(type="NodeGroupInput")
                    group_output = nodes.new(type="NodeGroupOutput")
                    mesh_to_points = nodes.new(type="GeometryNodeMeshToPoints")
                    set_material = nodes.new(type="GeometryNodeSetMaterial")

                    mesh_to_points.inputs["Radius"].default_value = point_size
                    set_material.inputs["Material"].default_value = material.blender_obj

                    links.new(group_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
                    links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
                    links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])
                elif scene.render.engine == "BLENDER_EEVEE":
                    logger.debug_level_2("Rendering pointcloud using particle system")
                    sphere = create_primitive("SPHERE", radius=point_size, location=(5, 0, 0))
                    sphere.set_shading_mode("SMOOTH")

                    material = sphere.new_material(f"Material {index}")
                    material.set_principled_shader_value("Roughness", 0.9)
                    material.set_principled_shader_value("Base Color", [*c, 1])

                    cast(Any, bpy.context.view_layer).objects.active = obj.blender_obj
                    particle_system = cast(Any, bpy.context.object).modifiers.new(
                        name="Particle System", type="PARTICLE_SYSTEM"
                    )
                    particle_system.particle_system.settings.count = len(v)
                    particle_system.particle_system.settings.frame_end = 0
                    particle_system.particle_system.settings.emit_from = "VERT"
                    particle_system.particle_system.settings.use_emit_random = False
                    particle_system.particle_system.settings.render_type = "OBJECT"
                    particle_system.particle_system.settings.instance_object = sphere.blender_obj
                    obj.blender_obj.show_instancer_for_render = False
                else:
                    raise ValueError(f"Unsupported render engine: {scene.render.engine}")
            else:  # Render mesh
                if isinstance(c, str) and c == "shadow":
                    obj.get_mesh().from_pydata(v, [], f)
                    obj.get_mesh().validate()
                    obj.set_shading_mode("SMOOTH")

                    material = obj.new_material(f"Material.{index:03d}")
                    material.set_principled_shader_value("Roughness", 1)
                    material.set_principled_shader_value("Base Color", [1, 1, 1, 1])
                    material.set_principled_shader_value("Alpha", 0.7)
                    obj.blender_obj.is_shadow_catcher = True
                else:
                    if c is None:
                        c = self.default_mesh_color

                    obj.get_mesh().from_pydata(v, [], f)
                    obj.get_mesh().validate()
                    obj.set_shading_mode("FLAT")

                    material = obj.new_material(f"Material.{index:03d}")
                    material.set_principled_shader_value("Roughness", 0.5)
                    material.set_principled_shader_value("Base Color", [*c, 1])

            ao_node = material.new_node("ShaderNodeAmbientOcclusion")
            ao_node.inputs["Distance"].default_value = 0.2
            ao_node.samples = 8
            ao_node.only_local = True
            ao_node.inputs["Color"].default_value = material.get_principled_shader_value("Base Color")
            material.set_principled_shader_value("Base Color", ao_node.outputs["Color"])

        if intrinsic is None:
            intrinsic = self.default_intrinsic
        set_intrinsics_from_K_matrix(K=intrinsic, image_width=self.width, image_height=self.height)

        if isinstance(extrinsic, np.ndarray):
            add_camera_pose(inv_trafo(extrinsic))
        else:
            for ext in (extrinsic or []):
                add_camera_pose(inv_trafo(ext))

        if self.render_color:
            key_light = Light("AREA", name="key_light")
            key_light.set_location([-1, 2, -0.5])
            key_light.set_rotation_mat(rotation_from_forward_vec(obj.get_location() - key_light.get_location()))
            key_light.set_scale([1, 1, 1])
            key_light.set_energy(int(150 * 0.7))
            if cast(Any, bpy.context.scene).render.engine == "BLENDER_EEVEE":
                cast(Any, key_light.blender_obj.data).use_contact_shadow = True

        data = dict()
        with stdout_redirected(enabled=not logger.isEnabledFor(DEBUG_LEVEL_1)):
            data.update(render(verbose=logger.isEnabledFor(DEBUG_LEVEL_2)))

        result = dict()
        if self.render_color:
            result["color"] = data["colors"][0]
        if self.render_depth:
            result["depth"] = data["depth"][0]
        if self.render_normal:
            result["normal"] = data["normals"][0]

        return result

    def render_pyrender(
        self,
        vertices: np.ndarray | list[np.ndarray],
        faces: np.ndarray | list[np.ndarray] | None = None,
        colors: np.ndarray | list[np.ndarray] | None = None,
        normals: np.ndarray | list[np.ndarray] | None = None,
        intrinsic: np.ndarray | None = None,
        extrinsic: np.ndarray | None = None,
        points_as_spheres: bool = False,
        point_size: int = 5,
        znear: float = 0.01,
        zfar: float = 10.0,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        flags = pyrender.renderer.RenderFlags.NONE
        if self.render_depth and not (self.render_color or self.render_normal):
            flags = pyrender.RenderFlags.DEPTH_ONLY
        if self.offscreen:
            flags |= pyrender.RenderFlags.OFFSCREEN

        scene = pyrender.Scene()

        vertices_list: list[np.ndarray] = vertices if isinstance(vertices, list) else [vertices]
        faces_list = cast(list[np.ndarray | None], faces if isinstance(faces, list) else [faces] * len(vertices_list))
        colors_list = cast(
            list[np.ndarray | None], colors if isinstance(colors, list) else [colors] * len(vertices_list)
        )

        for _index, (v, f, c) in enumerate(zip(vertices_list, faces_list, colors_list, strict=False)):
            if f is None:
                if c is None:
                    c = self.default_pcd_color
                if c.ndim == 1:
                    c = [c] * len(v)

                if points_as_spheres:
                    if len(v) > 2048:
                        v = v[np.random.randint(0, len(v), 2048)]
                    for _v, _c in zip(v, c, strict=False):
                        point = Sphere(radius=0.0015 * point_size, center=_v).to_mesh()
                        if _c is not None:
                            cast(Any, point.visual).vertex_colors = _c
                        scene.add(pyrender.Mesh.from_trimesh(point))
                else:
                    scene.add(pyrender.Mesh.from_points(v, colors=c))
            else:
                if c is None:
                    c = self.default_mesh_color
                if c.ndim == 1:
                    c = [c] * len(v)

                mesh = Trimesh(v, f, vertex_normals=normals, vertex_colors=c, process=False, validate=False)
                scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

        assert intrinsic is not None
        assert extrinsic is not None
        camera = pyrender.IntrinsicsCamera(
            intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2], znear=znear, zfar=zfar
        )
        scene.add(camera, pose=inv_trafo(extrinsic))

        if self.render_color:
            light = pyrender.PointLight(intensity=7)
            scene.add(light, pose=inv_trafo(extrinsic))

        renderer = cast(Any, self.renderer)
        renderer.point_size = point_size
        if self.render_color or self.render_normal:
            if self.render_color:
                color, depth = renderer.render(scene, flags=flags)
            if self.render_normal:
                program_cache = renderer._renderer._program_cache
                shader_dir = Path(__file__).parent.parent.parent / "utils" / "assets" / "shaders"
                renderer._renderer._program_cache = ShaderProgramCache(shader_dir=shader_dir)
                normal, depth = renderer.render(scene, flags=flags)
                normal /= 255
                renderer._renderer._program_cache = program_cache
        else:
            depth = renderer.render(scene, flags=flags)
        depth[depth >= zfar] = 0

        result = dict()
        if self.render_color:
            result["color"] = color
        if self.render_depth:
            result["depth"] = depth
        if self.render_normal:
            result["normal"] = normal
        return result

    def render_open3d(
        self,
        vertices: np.ndarray | list[np.ndarray],
        faces: np.ndarray | list[np.ndarray] | None = None,
        colors: np.ndarray | list[np.ndarray] | None = None,
        normals: np.ndarray | list[np.ndarray] | None = None,
        intrinsic: np.ndarray | None = None,
        extrinsic: np.ndarray | None = None,
        points_as_spheres: bool = True,
        point_size: int = 5,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        vertices_list: list[np.ndarray] = vertices if isinstance(vertices, list) else [vertices]
        faces_list = cast(list[np.ndarray | None], faces if isinstance(faces, list) else [faces] * len(vertices_list))
        colors_list = cast(
            list[np.ndarray | None], colors if isinstance(colors, list) else [colors] * len(vertices_list)
        )
        objs: dict[str, tuple[Any, list[float] | np.ndarray]] = dict()
        index = 0
        for v, f, c in zip(vertices_list, faces_list, colors_list, strict=False):
            if f is None:
                if c is None:
                    c = self.default_pcd_color
                if c.ndim == 1:
                    c = [c] * len(v)

                if points_as_spheres or len(v) <= 1024:
                    if len(v) > 2048:
                        v = v[np.random.randint(0, len(v), 2048)]
                    for _v, _c in zip(v, c, strict=False):
                        point = cast(Any, o3d).geometry.TriangleMesh().create_sphere(
                            radius=0.0015 * point_size, resolution=10
                        )
                        point.translate(_v)
                        point.paint_uniform_color(_c)
                        point.compute_vertex_normals()
                        objs[f"pcd_{index}"] = (point, _c)
                        index += 1
                else:
                    pcd = cast(Any, o3d).geometry.PointCloud(cast(Any, o3d).utility.Vector3dVector(v))
                    pcd.colors = cast(Any, o3d).utility.Vector3dVector(c)
                    objs[f"pcd_{index}"] = (pcd, c[0])
                    index += 1
            else:
                if c is None:
                    c = self.default_mesh_color
                if c.ndim == 1:
                    c = [c] * len(v)

                mesh = cast(Any, o3d).geometry.TriangleMesh(
                    cast(Any, o3d).utility.Vector3dVector(v), cast(Any, o3d).utility.Vector3iVector(f)
                )
                mesh.vertex_colors = o3d.utility.Vector3dVector(c)
                if self.render_color or self.render_normal:
                    mesh.compute_vertex_normals()
                objs[f"mesh_{index}"] = (mesh, c[0])
                index += 1

        assert intrinsic is not None
        assert extrinsic is not None
        renderer = cast(Any, self.renderer)
        if self.offscreen:
            for i, (k, v) in enumerate(objs.items()):
                o, c = v
                material = cast(Any, o3d).visualization.rendering.MaterialRecord()
                material.base_color = [*c, 1.0]
                material.shader = "defaultLit"
                material.base_roughness = 0.2 if "pcd" in k else 0.5
                material.point_size = point_size
                renderer.scene.add_geometry(f"obj_{i}", o, material)

            renderer.setup_camera(intrinsic, extrinsic, self.width, self.height)

            if self.render_color:
                renderer.scene.set_background([*self.default_background_color, 1.0])
                renderer.scene.scene.set_sun_light([0.707, 0.0, -0.707], [1.0, 1.0, 1.0], 100_000)
                renderer.scene.scene.enable_sun_light(True)
                """
                self.renderer.scene.scene.add_point_light("cam_light",
                                                          np.ones(3),
                                                          inv_trafo(extrinsic)[:3, 3],
                                                          75000,
                                                          10000,
                                                          True)
                """

                color = np.asarray(renderer.render_to_image())

            if self.render_normal:
                raise NotImplementedError("Normal rendering is not supported yet by Open3D offscreen renderer.")

            if self.render_depth:
                depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
                depth[depth == np.inf] = 0

            renderer.scene.clear_geometry()
        else:
            for obj in objs.values():
                renderer.add_geometry(obj[0])

            # NEEDS to be called AFTER add_geometry :facepalm:
            ctr = renderer.get_view_control()
            camera_parameters = ctr.convert_to_pinhole_camera_parameters()
            camera_parameters.intrinsic.intrinsic_matrix = intrinsic
            camera_parameters.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)

            if self.render_color:
                color = np.asarray(renderer.capture_screen_float_buffer(do_render=True))
                color = (color * 255).astype(np.uint8)
            if self.render_normal:
                renderer.get_render_option().mesh_color_option = cast(Any, o3d).visualization.MeshColorOption.Normal
                normal = np.asarray(renderer.capture_screen_float_buffer(do_render=True))
            if self.render_depth:
                depth = np.asarray(renderer.capture_depth_float_buffer(do_render=True))

            renderer.clear_geometries()

        result = dict()
        if self.render_color:
            result["color"] = color
        if self.render_depth:
            result["depth"] = depth
        if self.render_normal:
            result["normal"] = normal

        return result

    @torch.no_grad()
    def render_pytorch3d(
        self,
        vertices: np.ndarray | list[np.ndarray],
        faces: np.ndarray | list[np.ndarray] | None = None,
        colors: np.ndarray | list[np.ndarray] | None = None,
        normals: np.ndarray | list[np.ndarray] | None = None,
        intrinsic: np.ndarray | None = None,
        extrinsic: np.ndarray | list[np.ndarray] | None = None,
        points_as_spheres: bool = False,
        point_size: int = 5,
        device: str = "cuda:0",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        torch_device = torch.device(device)
        assert intrinsic is not None
        assert extrinsic is not None
        intrinsic = intrinsic.astype(np.float32)
        extrinsic_arr = cast(np.ndarray, extrinsic)
        # calibration = np.eye(4)
        # calibration[:3, :3] = intrinsic
        cameras = PerspectiveCameras(
            focal_length=((intrinsic[0, 0], intrinsic[1, 1]),),
            principal_point=((intrinsic[0, 2], intrinsic[1, 2]),),
            R=torch.from_numpy(extrinsic_arr[:3, :3]).unsqueeze(0).float().to(torch_device),
            T=torch.from_numpy(extrinsic_arr[:3, 3]).unsqueeze(0).float().to(torch_device),
            # K=torch.from_numpy(calibration).unsqueeze(0).float().to(device),
            device=torch_device,
            in_ndc=False,
            image_size=((self.height, self.width),),
        )
        raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,  # Use naive rasterization
            max_faces_per_bin=None,
        )
        lights = PointLights(device=torch_device, location=[inv_trafo(extrinsic_arr)[:3, 3]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
        )

        v_list = list()
        f_list = list()
        n_list = list()
        t_list = list()
        vertices_list: list[np.ndarray] = vertices if isinstance(vertices, list) else [vertices]
        faces_list = cast(list[np.ndarray | None], faces if isinstance(faces, list) else [faces] * len(vertices_list))
        colors_list = cast(
            list[np.ndarray | None], colors if isinstance(colors, list) else [colors] * len(vertices_list)
        )
        for _index, (v, f, c) in enumerate(zip(vertices_list, faces_list, colors_list, strict=False)):
            if f is None:
                if c is None:
                    c = self.default_pcd_color
                if c.ndim == 1:
                    c = [c] * len(v)

                if points_as_spheres or len(v) <= 1024:
                    if len(v) > 2048:
                        v = v[np.random.randint(0, len(v), 2048)]
                    for _v, _c in zip(v, c, strict=False):
                        sphere = cast(Any, o3d).geometry.TriangleMesh().create_sphere(
                            radius=0.0015 * point_size, resolution=10
                        )
                        sphere.translate(_v)
                        sphere.compute_vertex_normals()
                        v_list.append(torch.from_numpy(np.asarray(sphere.vertices)))
                        f_list.append(torch.from_numpy(np.asarray(sphere.triangles)))
                        n_list.append(torch.from_numpy(np.asarray(sphere.vertex_normals)))
                        t_list.append(torch.from_numpy(np.stack([_c] * len(sphere.vertices), axis=0)))
                else:
                    raise NotImplementedError("Pointcloud rendering is not supported yet by PyTorch3D.")

            else:
                if c is None:
                    c = self.default_mesh_color
                if c.ndim == 1:
                    c = [c] * len(v)

                mesh = cast(Any, o3d).geometry.TriangleMesh(
                    cast(Any, o3d).utility.Vector3dVector(v), cast(Any, o3d).utility.Vector3iVector(f)
                )
                mesh.compute_vertex_normals()

                v_list.append(torch.from_numpy(v))
                f_list.append(torch.from_numpy(f))
                n_list.append(torch.from_numpy(np.asarray(mesh.vertex_normals)))
                t_list.append(torch.from_numpy(np.stack(cast(Any, c), axis=0)))

        textures = TexturesVertex(verts_features=torch.cat(t_list, dim=0).unsqueeze(0).float().to(torch_device))
        meshes = Meshes(
            verts=torch.cat(v_list, dim=0).unsqueeze(0).float().to(torch_device),
            faces=torch.cat(f_list, dim=0).unsqueeze(0).long().to(torch_device),
            verts_normals=torch.cat(n_list, dim=0).unsqueeze(0).float().to(torch_device),
            textures=textures,
        )
        materials = Materials(device=torch_device, shininess=10)
        images = renderer(meshes, lights=lights, materials=materials, cameras=cameras)

        result = dict()
        if self.render_color:
            result["color"] = images[0].cpu().numpy().astype(np.uint8)
        if self.render_depth:
            raise NotImplementedError("Depth rendering is not supported yet by PyTorch3D.")
        if self.render_normal:
            raise NotImplementedError("Normal rendering is not supported yet by PyTorch3D.")

        return result

    def render(
        self,
        vertices: np.ndarray | list[np.ndarray],
        faces: np.ndarray | list[np.ndarray] | None = None,
        colors: np.ndarray | list[np.ndarray] | None = None,
        normals: np.ndarray | list[np.ndarray] | None = None,
        intrinsic: np.ndarray | None = None,
        extrinsic: np.ndarray | list[np.ndarray] | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        vertices_list = vertices if isinstance(vertices, list) else [vertices]
        faces_list = cast(list[np.ndarray | None], faces if isinstance(faces, list) else [faces] * len(vertices_list))
        colors_list = cast(
            list[np.ndarray | None], colors if isinstance(colors, list) else [colors] * len(vertices_list)
        )
        intrinsic_arr = self.default_intrinsic if intrinsic is None else intrinsic
        extrinsic_input = self.default_extrinsic_opengl if extrinsic is None else extrinsic
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

        if self.method in ["blender", "cycles", "eevee"]:
            result = self.render_blender(
                vertices_list,
                cast(Any, faces_list),
                cast(Any, colors_list),
                normals,
                intrinsic_arr,
                cast(Any, extrinsic_input),
                **kwargs,
            )
        elif self.method == "pyrender":
            extrinsic_arr = extrinsic_input[0] if isinstance(extrinsic_input, list) else extrinsic_input
            result = self.render_pyrender(
                vertices_list,
                cast(Any, faces_list),
                cast(Any, colors_list),
                normals,
                intrinsic_arr,
                cast(np.ndarray, extrinsic_arr),
                **kwargs,
            )
        elif self.method == "open3d":
            extrinsic_arr = extrinsic_input[0] if isinstance(extrinsic_input, list) else extrinsic_input
            extrinsic_cv = convert_extrinsic(cast(np.ndarray, extrinsic_arr), "opengl", "opencv")
            result = self.render_open3d(
                vertices_list,
                cast(Any, faces_list),
                cast(Any, colors_list),
                normals,
                intrinsic_arr,
                extrinsic_cv,
                **kwargs,
            )
        elif self.method == "pytorch3d":
            extrinsic_arr = extrinsic_input[0] if isinstance(extrinsic_input, list) else extrinsic_input
            extrinsic_cv = convert_extrinsic(cast(np.ndarray, extrinsic_arr), "opengl", "opencv")
            result = self.render_pytorch3d(
                vertices_list,
                cast(Any, faces_list),
                cast(Any, colors_list),
                normals,
                intrinsic_arr,
                extrinsic_cv,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown rendering method: {self.method}")

        if self.show or logger.isEnabledFor(DEBUG):
            self.visualize_result(cast(dict[str, np.ndarray | list[np.ndarray]], result))

        return result

    def __call__(
        self,
        vertices: np.ndarray | list[np.ndarray],
        faces: np.ndarray | list[np.ndarray] | None = None,
        colors: np.ndarray | list[np.ndarray] | None = None,
        normals: np.ndarray | list[np.ndarray] | None = None,
        intrinsic: np.ndarray | None = None,
        extrinsic: np.ndarray | list[np.ndarray] | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        return self.render(vertices, faces, colors, normals, intrinsic, extrinsic, **kwargs)

    def __del__(self):
        os.environ.pop("PYOPENGL_PLATFORM", None)
        if self.pyopengl_platform is not None:
            os.environ["PYOPENGL_PLATFORM"] = self.pyopengl_platform

        if isinstance(self._renderer, pyrender.OffscreenRenderer):
            self._renderer.delete()
        elif isinstance(self._renderer, cast(Any, o3d).visualization.Visualizer):
            cast(Any, self._renderer).destroy_window()
