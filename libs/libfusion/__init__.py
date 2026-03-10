import logging

try:
    from .gpu import PyViews as PyViews
    from .gpu import tsdf_fusion as tsdf_fusion

    logging.debug("Using GPU fusion")
except ModuleNotFoundError:
    from .cpu import PyViews as PyViews
    from .cpu import tsdf_fusion as tsdf_fusion

    logging.debug("Using CPU fusion")
