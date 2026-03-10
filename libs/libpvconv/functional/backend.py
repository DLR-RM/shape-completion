import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import pvconv_ext as _ext
except (ImportError, ModuleNotFoundError):
    try:
        from torch.utils.cpp_extension import load

        logger.warning("Unable to load PVConv CUDA kernels. JIT compiling...")
        logger.info("Consider installing the CUDA kernels with `python libs/libmanager.py install pvconv`.")

        cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if cuda_arch_list is None:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0+PTX;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9+PTX"

        setup_dir = Path(__file__).parent
        src_dir = setup_dir / "src"
        sources = [str(p) for p in src_dir.glob("**/*") if p.suffix in [".cpp", ".cu"]]

        _ext = load(name="pvconv_ext", sources=sources, extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=["-O3"])
    except (OSError, RuntimeError, IndexError) as e:
        logger.error(f"Unable to JIT compile PVConv CUDA kernels {e}")
        _ext = None
