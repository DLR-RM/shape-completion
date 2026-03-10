import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
if cuda_arch_list is None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX"

setup_dir = Path(__file__).resolve().parent
src_dir = setup_dir / "functional/src"
sources = [str(p.relative_to(setup_dir)) for p in src_dir.glob("**/*") if p.suffix in [".cpp", ".cu"]]

long_description = ""
try:
    with (setup_dir / "README.md").open("r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    pass

setup(
    name="pvconv_ext",
    version="1.1.0",
    description="CUDA operations for PVConv",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yujun Lin and Song Han",
    url="https://github.com/mit-han-lab/pvcnn/tree/master/modules",
    license="MIT",
    packages=find_packages(),
    install_requires=["torch>=1.4"],
    ext_modules=[
        CUDAExtension(
            name="pvconv_ext", sources=sources, extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": ["-O3"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
