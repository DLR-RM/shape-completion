
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extension = Extension(
    name="kinectSimulator",
    sources=["simulate_wrapper.pyx", "src/kinectSimulator.cpp", "src/noiseutils.cpp"],
    include_dirs=[
        np.get_include(),
        "include",
        "/usr/include/eigen3",
        "/usr/include",
        "/usr/include/opencv4",
        "lib/libnoise/include",
        "src"
    ],
    libraries=[
        "assimp",
        "noise",
        "opencv_core",
        "opencv_highgui",
        "opencv_imgcodecs"
    ],
    language="c++",
    extra_compile_args=["-std=c++14"],  # adjust as needed
    extra_link_args=[]  # add any additional linking flags if needed
)

setup(
    name="kinectSimulator",
    ext_modules=cythonize([extension]),
    zip_safe=False,
)
