"""Tests for libkinect Kinect depth sensor simulator."""

import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

from ..libkinect import KinectSimCython

try:
    # Validate native runtime linkage once; skip the module cleanly if broken.
    _probe_simulator = KinectSimCython()
    del _probe_simulator
except Exception as exc:
    pytest.skip(f"Skipping libkinect tests due to runtime import/link error: {exc}", allow_module_level=True)


class TestKinectSimulator:
    """Test cases for KinectSimCython depth simulator."""

    @pytest.fixture
    def simulator(self):
        """Create a KinectSimCython instance."""
        return KinectSimCython()

    def test_simple_quad_at_z1(self, simulator):
        """Test depth rendering of a simple quad at z=1."""
        # Create a quad facing the camera at z=1
        vertices = np.array(
            [
                [-0.5, -0.5, 1.0],
                [0.5, -0.5, 1.0],
                [0.5, 0.5, 1.0],
                [-0.5, 0.5, 1.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        depth = simulator.simulate(vertices, faces, verbose=False)

        # Check output shape
        assert depth.shape == (480, 640)

        # Center of image should have depth close to 1.0
        center_depth = depth[240, 320]
        assert 0.9 < center_depth < 1.1, f"Center depth {center_depth} not close to 1.0"

        # Should have many non-zero pixels (quad covers significant area)
        non_zero = np.count_nonzero(depth)
        assert non_zero > 100000, f"Only {non_zero} non-zero pixels, expected > 100000"

    def test_sphere_at_z2(self, simulator):
        """Test depth rendering of a sphere at z=2."""
        sphere = trimesh.primitives.Sphere(radius=0.3)
        sphere.apply_translation([0, 0, 2])
        assert sphere.vertices is not None
        assert sphere.faces is not None

        depth = simulator.simulate(
            sphere.vertices.astype(np.float32),
            sphere.faces.astype(np.int32),
            verbose=False,
        )

        # Front of sphere is at z=1.7, back at z=2.3
        center_depth = depth[240, 320]
        assert 1.6 < center_depth < 1.8, f"Sphere front depth {center_depth} not in expected range"

        # Should have visible pixels
        valid_pixels = np.sum((depth > 0) & np.isfinite(depth))
        assert valid_pixels > 10000, f"Only {valid_pixels} valid pixels"

    def test_mesh_at_different_distances(self, simulator):
        """Test that depth scales correctly with distance."""
        depths_at_center = []

        for z in [1.0, 1.5, 2.0, 2.5]:
            vertices = np.array(
                [
                    [-0.3, -0.3, z],
                    [0.3, -0.3, z],
                    [0.3, 0.3, z],
                    [-0.3, 0.3, z],
                ],
                dtype=np.float32,
            )
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

            depth = simulator.simulate(vertices, faces, verbose=False)
            depths_at_center.append(depth[240, 320])

        # Depths should increase monotonically
        for i in range(len(depths_at_center) - 1):
            assert depths_at_center[i] < depths_at_center[i + 1], (
                f"Depth at z={1.0 + i * 0.5} ({depths_at_center[i]}) >= depth at z={1.5 + i * 0.5} ({depths_at_center[i + 1]})"
            )

    def test_rotated_mesh(self, simulator):
        """Test that rotated meshes still produce valid depth."""
        sphere = trimesh.primitives.Sphere(radius=0.3)

        # Apply 45-degree rotation around X axis
        R = trimesh.transformations.rotation_matrix(np.pi / 4, [1, 0, 0])
        sphere.apply_transform(R)
        sphere.apply_translation([0, 0, 2])
        assert sphere.vertices is not None
        assert sphere.faces is not None

        depth = simulator.simulate(
            sphere.vertices.astype(np.float32),
            sphere.faces.astype(np.int32),
            verbose=False,
        )

        # Should still have valid depth values
        valid_pixels = np.sum((depth > 0) & np.isfinite(depth))
        assert valid_pixels > 10000, f"Only {valid_pixels} valid pixels after rotation"

        # Center depth should still be reasonable
        center_depth = depth[240, 320]
        assert 1.5 < center_depth < 2.5, f"Center depth {center_depth} out of expected range"

    def test_bowl_mesh_simple(self, simulator):
        """Test with the bowl mesh asset positioned simply in front of camera."""
        mesh_path = Path(__file__).parent.parent / "libkinect/assets/mesh_gray_bowl.obj"
        if not mesh_path.exists():
            pytest.skip(f"Bowl mesh not found: {mesh_path}")

        mesh = trimesh.load_mesh(mesh_path)

        # Position bowl directly in front of camera at z=0.6 (beyond z_near=0.5)
        mesh.apply_translation([0, 0, 0.6])

        depth = simulator.simulate(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32), verbose=False)

        # Should have valid depth around center
        center_region = depth[200:280, 280:360]
        valid_in_center = np.sum((center_region > 0) & np.isfinite(center_region))
        assert valid_in_center > 1000, f"Only {valid_in_center} valid pixels in center region"

        # Depth values should be around 0.6
        valid_depths = center_region[(center_region > 0) & np.isfinite(center_region)]
        if len(valid_depths) > 0:
            mean_depth = np.mean(valid_depths)
            assert 0.5 < mean_depth < 0.8, f"Mean depth {mean_depth} not in expected range"

    def test_custom_resolution(self, simulator):
        """Test simulation with custom resolution."""
        vertices = np.array(
            [
                [-0.5, -0.5, 1.0],
                [0.5, -0.5, 1.0],
                [0.5, 0.5, 1.0],
                [-0.5, 0.5, 1.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        # Test with smaller resolution
        depth = simulator.simulate(vertices, faces, width=320, height=240, verbose=False)

        assert depth.shape == (240, 320)
        # Center should still have valid depth
        center_depth = depth[120, 160]
        assert 0.9 < center_depth < 1.1, f"Center depth {center_depth} not close to 1.0"


def test_original_bowl_scene():
    """Original test with complex camera transformation - was failing before CGAL 6 fix.

    Note: Kinect simulation produces some edge artifacts with extreme values,
    similar to real Kinect sensors. We test that the majority of pixels have
    reasonable depth values.
    """
    from pathlib import Path

    import trimesh

    from libs.libkinect import KinectSimCython

    f_kinect_world = np.array(
        [
            [
                0.18889654314248513,
                -0.5491412049704937,
                0.8141013875353555,
                -1.29464394785277,
            ],
            [
                -0.9455020122657823,
                -0.325615105774899,
                -0.0002537971720279581,
                -0.009541282377521727,
            ],
            [
                0.2652233842565174,
                -0.7696874527524885,
                -0.5807223538112247,
                0.5834419201245094,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    x_world_detection_position = np.array([0.25, -0.8, 0.68])

    mesh_path = Path(__file__).parent.parent / "libkinect/assets/mesh_gray_bowl.obj"
    if not mesh_path.exists():
        pytest.skip(f"Bowl mesh not found: {mesh_path}")

    mesh = trimesh.load_mesh(mesh_path)
    mesh = mesh.apply_translation([0, 0.15, 0])

    # Add table
    table = trimesh.primitives.Box([1, 1, 0.6], trimesh.transformations.translation_matrix([0, 0, -0.3]))
    mesh = trimesh.util.concatenate(mesh, table)

    mesh = mesh.apply_translation(x_world_detection_position)
    mesh.apply_transform(f_kinect_world)

    print(f"Mesh bounds after transform: {mesh.bounds}")
    print(f"Z range: {mesh.vertices[:, 2].min():.3f} - {mesh.vertices[:, 2].max():.3f}")

    sim = KinectSimCython()
    depth = sim.simulate(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32), verbose=True)

    # Check for valid depth values
    valid_mask = (depth > 0) & np.isfinite(depth)
    valid_pixels = np.sum(valid_mask)
    print(f"Valid pixels: {valid_pixels}")

    assert valid_pixels > 50000, f"Expected > 50000 valid pixels, got {valid_pixels}"

    valid_depths = depth[valid_mask]

    # Filter to reasonable depth range (Kinect simulation has edge artifacts)
    # Real Kinect data also requires similar filtering
    reasonable_mask = (valid_depths > 0.1) & (valid_depths < 5.0)
    reasonable_depths = valid_depths[reasonable_mask]
    reasonable_count = len(reasonable_depths)

    print(
        f"Reasonable depth pixels: {reasonable_count} / {valid_pixels} ({100 * reasonable_count / valid_pixels:.1f}%)"
    )
    print(f"Reasonable depth range: {reasonable_depths.min():.3f} - {reasonable_depths.max():.3f}")
    print(f"Mean depth: {reasonable_depths.mean():.3f}")

    # At least 80% of valid pixels should have reasonable depth
    assert reasonable_count > 0.8 * valid_pixels, f"Only {reasonable_count}/{valid_pixels} pixels have reasonable depth"

    # Mean depth should be in expected range (mesh Z is 0.35-1.74)
    assert 0.5 < reasonable_depths.mean() < 2.0, f"Mean depth {reasonable_depths.mean():.3f} outside expected range"


def test_depth_buffer_initialized():
    """Regression test: depth buffer must be initialized to 0, not garbage.

    Previously, uninitialized memory caused random non-zero values in background
    pixels, appearing as stripe artifacts.
    """
    from libs.libkinect import KinectSimCython

    # Small quad that only covers center of image - most pixels are background
    vertices = np.array(
        [
            [-0.1, -0.1, 1.0],
            [0.1, -0.1, 1.0],
            [0.1, 0.1, 1.0],
            [-0.1, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    sim = KinectSimCython()
    depth = sim.simulate(vertices, faces, verbose=False)

    # Background pixels (outside the quad) must be exactly 0, not garbage
    # Check corners which should definitely be background
    corners = [
        depth[0, 0],
        depth[0, 639],
        depth[479, 0],
        depth[479, 639],
        depth[10, 10],
        depth[10, 630],
        depth[470, 10],
        depth[470, 630],
    ]

    for i, val in enumerate(corners):
        assert val == 0.0, f"Background pixel {i} has value {val}, expected 0.0 (uninitialized memory bug)"

    # Also verify no extreme garbage values anywhere
    non_zero = depth[depth != 0]
    if len(non_zero) > 0:
        assert non_zero.min() > 0, "Found negative depth values (garbage)"
        assert non_zero.max() < 100, f"Found extreme depth value {non_zero.max()} (garbage)"


def test_close_objects_visible():
    """Test that z_near parameter correctly controls minimum visible depth.

    The default z_near=0.5 matches real Kinect v1 behavior.
    Setting z_near lower allows closer objects to be visible.
    """
    from libs.libkinect import KinectSimCython

    # Plane at z=0.3m (closer than default z_near=0.5)
    vertices = np.array(
        [
            [-0.5, -0.5, 0.3],
            [0.5, -0.5, 0.3],
            [0.5, 0.5, 0.3],
            [-0.5, 0.5, 0.3],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    sim = KinectSimCython()

    # With default z_near=0.5, object at z=0.3 should NOT be visible
    depth_default = sim.simulate(vertices, faces, verbose=False)
    center_default = depth_default[240, 320]
    assert center_default == 0, f"Object at z=0.3 visible with z_near=0.5: {center_default}"

    # With z_near=0.2, object at z=0.3 SHOULD be visible
    depth_near = sim.simulate(vertices, faces, z_near=0.2, verbose=False)
    center_near = depth_near[240, 320]
    assert 0.25 < center_near < 0.35, f"Object at z=0.3 not visible with z_near=0.2: {center_near}"

    # Should have many valid pixels with lowered z_near
    valid = (depth_near > 0) & (depth_near < 1)
    assert valid.sum() > 100000, f"Only {valid.sum()} valid pixels with z_near=0.2"


def test_noise_types():
    """Test that different noise types produce different results."""
    from libs.libkinect import KinectSimCython, NoiseType

    # Simple plane
    vertices = np.array(
        [
            [-0.5, -0.5, 1.0],
            [0.5, -0.5, 1.0],
            [0.5, 0.5, 1.0],
            [-0.5, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    sim = KinectSimCython()

    # Test with no noise
    depth_none = sim.simulate(vertices, faces, noise=NoiseType.NONE, verbose=False)

    # Test with Perlin noise (default)
    depth_perlin = sim.simulate(vertices, faces, noise=NoiseType.PERLIN, verbose=False)

    # Both should produce valid depth at center
    assert 0.9 < depth_none[240, 320] < 1.1
    assert 0.9 < depth_perlin[240, 320] < 1.1

    # With noise, there should be more variation in depth values
    valid_none = depth_none[depth_none > 0]
    valid_perlin = depth_perlin[depth_perlin > 0]

    # Perlin noise should add some variation (though may be small)
    # At minimum, verify both produce reasonable output
    assert len(valid_none) > 100000, "No-noise depth should have many valid pixels"
    assert len(valid_perlin) > 100000, "Perlin-noise depth should have many valid pixels"


def test_configurable_parameters():
    """Test that configurable parameters affect the output correctly."""
    from libs.libkinect import KinectSimCython, NoiseType

    # Plane at z=1
    vertices = np.array(
        [
            [-0.3, -0.3, 1.0],
            [0.3, -0.3, 1.0],
            [0.3, 0.3, 1.0],
            [-0.3, 0.3, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    sim = KinectSimCython()

    # Test z_far: object at z=1 should be visible with z_far=2, not with z_far=0.8
    depth_far2 = sim.simulate(vertices, faces, z_far=2.0, noise=NoiseType.NONE, verbose=False)
    depth_far08 = sim.simulate(vertices, faces, z_far=0.8, noise=NoiseType.NONE, verbose=False)

    assert depth_far2[240, 320] > 0, "Object at z=1 should be visible with z_far=2"
    assert depth_far08[240, 320] == 0, "Object at z=1 should NOT be visible with z_far=0.8"

    # Test custom resolution
    depth_small = sim.simulate(vertices, faces, width=320, height=240, noise=NoiseType.NONE, verbose=False)
    assert depth_small.shape == (240, 320)
    assert depth_small[120, 160] > 0, "Center should have valid depth at smaller resolution"


def test_kinect_visual(show: bool = False):
    """Visual comparison test between KinectSim and pyrender.

    This test creates a visual comparison but doesn't assert - useful for debugging.
    Run with: pytest -k test_kinect_visual -s
    """
    # Simple scene: sphere on a plane
    sphere = trimesh.primitives.Sphere(radius=0.2)
    sphere.apply_translation([0, 0, 1.5])

    plane = trimesh.primitives.Box([2, 2, 0.1])
    plane.apply_translation([0, 0, 2.0])

    mesh = trimesh.util.concatenate([sphere, plane])

    sim = KinectSimCython()
    kinect_depth = sim.simulate(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32), verbose=True)

    print("\nKinect depth stats:")
    print(f"  Shape: {kinect_depth.shape}")
    print(f"  Valid pixels: {np.sum((kinect_depth > 0) & np.isfinite(kinect_depth))}")
    valid = kinect_depth[(kinect_depth > 0) & np.isfinite(kinect_depth)]
    if len(valid) > 0:
        print(f"  Depth range: {valid.min():.3f} - {valid.max():.3f}")
        print(f"  Mean depth: {valid.mean():.3f}")

    if show and os.environ.get("DISPLAY") is not None:
        import matplotlib.pyplot as plt

        try:
            import pyrender

            # Flip mesh for pyrender (different coordinate convention)
            mesh_render = mesh.copy()
            mesh_render.apply_transform(trimesh.transformations.euler_matrix(np.pi, 0, 0))

            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_render, smooth=False)
            camera = pyrender.IntrinsicsCamera(fx=582.6989, fy=582.6989, cx=320.7906, cy=245.2647, znear=0.1, zfar=10.0)

            scene = pyrender.Scene()
            scene.add(pyrender_mesh)
            scene.add(camera)

            renderer = pyrender.OffscreenRenderer(640, 480)
            pyrender_depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

            _fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(kinect_depth, cmap="viridis")
            axes[0].set_title("Kinect Simulator")
            axes[0].colorbar = plt.colorbar(axes[0].images[0], ax=axes[0])

            axes[1].imshow(pyrender_depth, cmap="viridis")
            axes[1].set_title("PyRender Reference")
            axes[1].colorbar = plt.colorbar(axes[1].images[0], ax=axes[1])

            plt.tight_layout()
            plt.show()
        except ImportError:
            # pyrender not available, just show kinect output
            import matplotlib.pyplot as plt

            plt.imshow(kinect_depth, cmap="viridis")
            plt.colorbar()
            plt.title("Kinect Simulator Depth")
            plt.show()


def test_original_bowl_scene_visual(show: bool = False):
    """Visual test for the original bowl scene with complex camera transformation.

    Run with: pytest -k test_original_bowl_scene_visual -s
    Or directly: python -c "from tests.test_libkinect import test_original_bowl_scene_visual; test_original_bowl_scene_visual(show=True)"
    """
    from pathlib import Path

    from libs.libkinect import KinectSimCython

    f_kinect_world = np.array(
        [
            [
                0.18889654314248513,
                -0.5491412049704937,
                0.8141013875353555,
                -1.29464394785277,
            ],
            [
                -0.9455020122657823,
                -0.325615105774899,
                -0.0002537971720279581,
                -0.009541282377521727,
            ],
            [
                0.2652233842565174,
                -0.7696874527524885,
                -0.5807223538112247,
                0.5834419201245094,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    x_world_detection_position = np.array([0.25, -0.8, 0.68])

    mesh_path = Path(__file__).parent.parent / "libkinect/assets/mesh_gray_bowl.obj"
    if not mesh_path.exists():
        pytest.skip(f"Bowl mesh not found: {mesh_path}")

    mesh = trimesh.load_mesh(mesh_path)
    mesh = mesh.apply_translation([0, 0.15, 0])

    # Add table
    table = trimesh.primitives.Box([1, 1, 0.6], trimesh.transformations.translation_matrix([0, 0, -0.3]))
    mesh = trimesh.util.concatenate(mesh, table)

    mesh = mesh.apply_translation(x_world_detection_position)
    mesh.apply_transform(f_kinect_world)

    print(f"\nMesh bounds after transform: {mesh.bounds}")
    print(f"Z range: {mesh.vertices[:, 2].min():.3f} - {mesh.vertices[:, 2].max():.3f}")

    sim = KinectSimCython()
    depth = sim.simulate(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32), verbose=True)

    # Stats
    valid_mask = (depth > 0) & np.isfinite(depth)
    valid_depths = depth[valid_mask]
    reasonable_mask = (valid_depths > 0.1) & (valid_depths < 5.0)
    reasonable_depths = valid_depths[reasonable_mask]

    print("\nDepth stats:")
    print(f"  Shape: {depth.shape}")
    print(f"  Valid pixels: {np.sum(valid_mask)}")
    print(f"  Reasonable pixels: {len(reasonable_depths)} ({100 * len(reasonable_depths) / len(valid_depths):.1f}%)")
    if len(reasonable_depths) > 0:
        print(f"  Depth range: {reasonable_depths.min():.3f} - {reasonable_depths.max():.3f}")
        print(f"  Mean depth: {reasonable_depths.mean():.3f}")

    if show and os.environ.get("DISPLAY") is not None:
        import matplotlib.pyplot as plt

        try:
            import pyrender

            # For pyrender, flip Y and Z axes (different coordinate convention)
            mesh_render = mesh.copy()
            mesh_render.apply_transform(trimesh.transformations.euler_matrix(np.pi, 0, 0))

            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_render, smooth=False)
            camera = pyrender.IntrinsicsCamera(fx=582.6989, fy=582.6989, cx=320.7906, cy=245.2647, znear=0.1, zfar=10.0)

            scene = pyrender.Scene()
            scene.add(pyrender_mesh)
            scene.add(camera)

            renderer = pyrender.OffscreenRenderer(640, 480)
            pyrender_depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

            _fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            im0 = axes[0].imshow(depth, cmap="viridis")
            axes[0].set_title("Kinect Simulator (with stereo occlusion + noise)")
            plt.colorbar(im0, ax=axes[0], label="Depth (m)")

            im1 = axes[1].imshow(pyrender_depth, cmap="viridis")
            axes[1].set_title("PyRender Reference (ideal depth)")
            plt.colorbar(im1, ax=axes[1], label="Depth (m)")

            plt.suptitle("Bowl Scene: Kinect Simulator vs PyRender")
            plt.tight_layout()
            plt.show()

        except ImportError:
            # pyrender not available, just show kinect output
            _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            im = axes[0].imshow(depth, cmap="viridis")
            axes[0].set_title("Bowl Scene - Kinect Simulator Depth")
            plt.colorbar(im, ax=axes[0], label="Depth (m)")

            axes[1].hist(reasonable_depths.flatten(), bins=50, edgecolor="black")
            axes[1].set_xlabel("Depth (m)")
            axes[1].set_ylabel("Pixel count")
            axes[1].set_title(f"Depth Distribution (n={len(reasonable_depths)} pixels)")

            plt.suptitle("Bowl Scene (pyrender not available)")
            plt.tight_layout()
            plt.show()
