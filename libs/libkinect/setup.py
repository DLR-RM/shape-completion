import os
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

dependency_prefix = os.environ.get("KINECT_DEP_PREFIX")
dependency_include_dirs: list[str] = []
dependency_library_dirs: list[str] = []
if dependency_prefix:
    prefix = Path(dependency_prefix)
    dependency_include_dirs = [
        str(prefix / "include" / "opencv4"),
        str(prefix / "include" / "eigen3"),
        str(prefix / "include"),
    ]
    dependency_library_dirs = [str(prefix / "lib")]

# Define the Cython extension
extension = Extension(
    name="kinect_ext",
    sources=["simulate_wrapper.pyx", "src/kinectSimulator.cpp", "src/noiseutils.cpp"],
    include_dirs=[
        np.get_include(),
        *dependency_include_dirs,
        "include",
        "/usr/include/eigen3",
        "/usr/include",
        "/usr/include/opencv4",
        "lib/libnoise/include",
        "src",
    ],
    libraries=["assimp", "noise", "opencv_core", "opencv_highgui", "opencv_imgcodecs"],
    library_dirs=[
        *dependency_library_dirs,
        "lib/libnoise/lib",
    ],
    language="c++",
    extra_compile_args=["-std=c++17"],  # CGAL 5+ requires C++17
    extra_link_args=[],  # add any additional linking flags if needed
)

setup(name="kinect_ext", version="0.1.0", ext_modules=cythonize([extension]))
