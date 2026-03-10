# pyright: reportMissingImports=false

from enum import IntEnum
from pathlib import Path

import numpy as np


class NoiseType(IntEnum):
    """Noise type for Kinect depth simulation."""

    NONE = 0
    GAUSSIAN = 1
    PERLIN = 2
    SIMPLEX = 3


# Default Kinect v1 intrinsics (640x480)
KINECT_V1_FX = 582.6989
KINECT_V1_FY = 582.6989
KINECT_V1_CX = 320.7906
KINECT_V1_CY = 245.2647
KINECT_V1_BASELINE = 0.075  # 75mm baseline between IR projector and camera
KINECT_V1_Z_NEAR = 0.5  # Kinect v1 min range ~0.5m (near mode ~0.4m)
KINECT_V1_Z_FAR = 4.0  # Kinect v1 max range ~4m


class KinectSimCython:
    """Kinect depth sensor simulator using structured light stereo matching.

    Simulates the depth sensing behavior of a Kinect-style sensor by:
    1. Casting rays from the IR camera through each pixel
    2. Finding mesh intersections
    3. Checking visibility from the IR projector (stereo matching)
    4. Computing disparity and converting to depth
    5. Applying realistic noise patterns

    Parameters can be customized to simulate different sensor configurations.
    """

    def __init__(self, dot_pattern_path: str | None = None):
        """Initialize the Kinect simulator.

        Args:
            dot_pattern_path: Path to the IR dot pattern image. If None, uses
                the default Kinect pattern included with the library.
        """
        import sys

        # Add libkinect directory to path for local import
        libkinect_path = str(Path(__file__).resolve().parent)
        if libkinect_path not in sys.path:
            sys.path.insert(0, libkinect_path)
        import kinect_ext

        self.kinect_ext = kinect_ext
        if dot_pattern_path is None:
            self.dot_pattern_path = str(Path(__file__).resolve().parent / "assets/kinect-pattern_3x3_no_iccp.png")
        else:
            self.dot_pattern_path = dot_pattern_path

    def simulate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        width: int = 640,
        height: int = 480,
        fx: float = KINECT_V1_FX,
        fy: float = KINECT_V1_FY,
        cx: float = KINECT_V1_CX,
        cy: float = KINECT_V1_CY,
        z_near: float = KINECT_V1_Z_NEAR,
        z_far: float = KINECT_V1_Z_FAR,
        baseline: float = KINECT_V1_BASELINE,
        noise: NoiseType = NoiseType.PERLIN,
        verbose: bool = False,
    ) -> np.ndarray:
        """Simulate Kinect depth measurement for a mesh.

        Args:
            vertices: Mesh vertices as (N, 3) float array in camera coordinates.
                      Z should be positive (in front of camera).
            faces: Mesh faces as (M, 3) int array of vertex indices.
            width: Image width in pixels.
            height: Image height in pixels.
            fx: Focal length in x (pixels).
            fy: Focal length in y (pixels).
            cx: Principal point x coordinate (pixels).
            cy: Principal point y coordinate (pixels).
            z_near: Minimum depth (meters). Objects closer are not visible.
                    Kinect v1: ~0.5m (default), near mode: ~0.4m
            z_far: Maximum depth (meters). Objects farther are not visible.
                   Kinect v1: ~4m (default)
            baseline: Distance between IR projector and camera (meters).
                      Kinect v1: 0.075m (75mm)
            noise: Type of noise to add. See NoiseType enum.
            verbose: Print debug information.

        Returns:
            Depth image as (height, width) float32 array in meters.
            Background pixels have value 0.
        """
        result, out_depth = self.kinect_ext.python_simulate(
            vertices.astype(np.float32),
            len(vertices),
            faces.astype(np.int32),
            len(faces),
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            z_near,
            z_far,
            baseline,
            int(noise),
            self.dot_pattern_path,
            verbose,
        )
        if result:
            raise RuntimeError("Error: Kinect simulation failed.")

        return out_depth


# Legacy alias
KinectSim = KinectSimCython
