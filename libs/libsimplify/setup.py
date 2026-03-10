import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

long_description = ""
try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    pass

extensions = [
    Extension(
        name="simplify_ext",
        sources=["simplify_mesh.pyx"],
        language="c++",
        libraries=["m"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="simplify_ext",
    version="1.0.1",
    description="A C++ and Cython extension for fast quadric mesh simplification.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sven Forstmann, Lars Mescheder",
    author_email="info@svenforstmann.com, LarsMescheder@gmx.net",
    url="https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification, "
    "https://github.com/autonomousvision/occupancy_networks/tree/master/im2mesh/utils/libsimplify",
    license="MIT",
    install_requires=[
        "numpy",
        "cython",
    ],
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
