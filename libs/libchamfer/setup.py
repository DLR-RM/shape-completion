import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
if cuda_arch_list is None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX"

long_description = ''
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    pass

setup(
    name='chamfer',
    version='2.0.2',
    description='A CUDA extension for chamfer distance computation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Thibault Groueix',
    author_email='thibault.groueix.2012@polytechnique.org',
    url='https://github.com/ThibaultGROUEIX/ChamferDistancePytorch',
    license='MIT',
    install_requires=[
        'torch'
    ],
    ext_modules=[
        CUDAExtension('chamfer', [
            'chamfer_cuda.cpp',
            'chamfer.cu'
        ])
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.6',
)
