import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

long_description = ''
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    pass

extensions = [
    Extension(
        name="intersect.triangle_hash",
        sources=["triangle_hash.pyx"],
        language="c++",
        libraries=['m'],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="intersect",
    version="1.0",
    description="A C++ and Cython extension for triangle hash computation.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lars Mescheder',
    author_email='LarsMescheder@gmx.net',
    url='https://github.com/autonomousvision/occupancy_networks/tree/master/im2mesh/utils/libmesh',
    license='MIT',
    install_requires=[
        'numpy',
        'cython',
    ],
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
