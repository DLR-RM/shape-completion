import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
if cuda_arch_list is None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX"

setup_dir = Path(__file__).resolve().parent
src_dir = setup_dir / "pointnet2_ops/_ext-src/src"
include_dirs = [str(setup_dir / "pointnet2_ops/_ext-src/include")]
sources = [str(p.relative_to(setup_dir)) for p in src_dir.glob("**/*") if p.suffix in [".cpp", ".cu"]]

__version__ = "0.0.0"
_version_file = setup_dir / "pointnet2_ops" / "_version.py"
if _version_file.exists():
    for _line in _version_file.read_text().splitlines():
        if _line.startswith("__version__"):
            __version__ = _line.split("=")[1].strip().strip('"').strip("'")
            break

long_description = ""
try:
    with (setup_dir / "README.md").open("r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    pass

setup(
    name="pointnet_ext",
    version=__version__,
    description="CUDA operations for PointNet2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Erik Wijmans",
    url="https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib",
    license="Unlicense",
    packages=find_packages(),
    install_requires=["torch>=1.4"],
    ext_modules=[
        CUDAExtension(
            name="pointnet_ext",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=include_dirs,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: The Unlicense",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
