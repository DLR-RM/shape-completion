import os
from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from Cython.Build import cythonize
from setuptools import setup, find_packages
import numpy as np
import platform

cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
if cuda_arch_list is None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX"

extra_compile_args = {'cxx': ["-ffast-math", '-msse', '-msse2', '-msse3', '-msse4.2']}
extra_link_args = []
if 'Linux' in platform.system():
    print('Added OpenMP')
    extra_compile_args['cxx'].append('-fopenmp')
    extra_link_args.append('-fopenmp')

long_description = ''
try:
    with open(Path(__file__).parent.parent / 'README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    pass

extensions = [
    CUDAExtension(
        name='pyfusion_gpu',
        sources=['cyfusion.pyx', 'fusion.cu', 'fusion_zach_tvl1.cu'],
        language='c++',
        libraries=['m'],
        include_dirs=[np.get_include(), str(Path(__file__).parent)],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )
]

setup(
    name='pyfusion_gpu',
    version='1.0.2',
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
    python_requires='>=3.6',
    cmdclass={'build_ext': BuildExtension}
)
