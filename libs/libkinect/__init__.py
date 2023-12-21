from pathlib import Path
import ctypes
import numpy as np
from ..libkinect import kinectSimulator


class KinectSim:
    def __init__(self):
        self.lib_path = Path(__file__).parent / "lib"
        self.dot_path = self.lib_path.parent / "assets/kinect-pattern_3x3_no_iccp.png"
        # self.dot_pattern = str(self.dot_path)
        self.dot_pattern = str(self.dot_path).encode('utf-8')
        self.kinect_sim = np.ctypeslib.load_library("libkinectSim", self.lib_path)
        self.kinect_sim.simulate.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                                             ctypes.c_int,
                                             np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'),
                                             ctypes.c_int,
                                             np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                                             ctypes.c_int,
                                             ctypes.c_int,
                                             ctypes.c_float,
                                             ctypes.c_float,
                                             ctypes.c_float,
                                             ctypes.c_float,
                                             ctypes.c_char_p,
                                             ctypes.c_bool]
        self.kinect_sim.simulate.restype = int
        # self.kinect_sim = kinectSimulator

    def simulate(self,
                 vertices: np.ndarray,
                 faces: np.ndarray,
                 width: int = 640,
                 height: int = 480,
                 fx: float = 582.6989,
                 fy: float = 582.6989,
                 cx: float = 320.7906,
                 cy: float = 245.2647,
                 verbose: bool = False) -> np.ndarray:
        out_depth = np.zeros((height, width), dtype=np.float32)
        if self.kinect_sim.simulate(vertices.astype(np.float32),
                                    len(vertices),
                                    faces.astype(np.int32),
                                    len(faces),
                                    out_depth,
                                    width,
                                    height,
                                    fx,
                                    fy,
                                    cx,
                                    cy,
                                    self.dot_pattern,
                                    verbose):
            raise RuntimeError("Error: Kinect simulation failed.")
        """
        result, out_depth = self.kinect_sim.python_simulate(vertices.astype(np.float32),
                                                            len(vertices),
                                                            faces.astype(np.int32),
                                                            len(faces),
                                                            width,
                                                            height,
                                                            fx,
                                                            fy,
                                                            cx,
                                                            cy,
                                                            self.dot_pattern,
                                                            verbose)
        if result:
            raise RuntimeError("Error: Kinect simulation failed.")
        out_depth = np.array(out_depth).reshape((height, width))
        """
        return out_depth
