import logging

__all__ = ["PyViews", "tsdf_fusion"]

try:
    from pyfusion_gpu import PyViews as PyViews
    from pyfusion_gpu import tsdf_gpu as tsdf_fusion
except ModuleNotFoundError:
    logging.warning("Unable to load PyFusion CUDA kernels. Will fallback to CPU.")
    logging.info("Consider installing the CUDA kernels with `python libs/libmanager.py install fusion`")
    raise
