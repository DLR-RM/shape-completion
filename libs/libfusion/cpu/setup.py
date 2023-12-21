from pathlib import Path
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages
import numpy as np
import platform

extra_compile_args = ["-ffast-math", '-msse', '-msse2', '-msse3', '-msse4.2']
extra_link_args = []
if 'Linux' in platform.system():
    print('Added OpenMP')
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')

long_description = ''
try:
    with open(Path(__file__).parent.parent / 'README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    pass

extensions = [
    Extension(
        name='pyfusion_cpu',
        sources=['cyfusion.pyx', 'fusion.cpp'],
        language='c++',
        libraries=['m'],
        include_dirs=[np.get_include(), str(Path(__file__).parent)],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )
]

setup(
    name='pyfusion_cpu',
    version='1.0.1',
    description='A C++ and Cython extension for fast fusion of occupancy grids.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='David Stutz, Andreas Geiger',
    author_email='hello@davidstutz.de, a.geiger@uni-tuebingen.de',
    url='https://github.com/davidstutz/mesh-fusion',
    license='BSD',
    install_requires=[
        'numpy',
        'cython'
    ],
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6'
)
