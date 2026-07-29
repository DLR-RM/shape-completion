import logging

try:
    from .gpu import PyViews as PyViews  # pyright: ignore[reportAttributeAccessIssue]
    from .gpu import tsdf_fusion as tsdf_fusion  # pyright: ignore[reportAttributeAccessIssue]

    logging.debug("Using GPU fusion")
except ModuleNotFoundError:
    from .cpu import PyViews as PyViews  # pyright: ignore[reportAttributeAccessIssue]
    from .cpu import tsdf_fusion as tsdf_fusion  # pyright: ignore[reportAttributeAccessIssue]

    logging.debug("Using CPU fusion")
