import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(osp.join(_ext_src_root, "src", "*.cu"))
# _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

try:
    from pointnet2_ops._version import __version__
except ImportError:
    __version__ = "unknown-version"

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
    name="pointnet2_ops",
    version=__version__,
    description="CUDA Operations for PointNet2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Erik Wijmans",
    url="https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib",
    license="Unlicense",
    packages=find_packages(),
    install_requires=["torch>=1.4"],
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: The Unlicense",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
