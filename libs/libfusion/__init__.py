import logging

try:
    from .gpu import PyViews, tsdf_fusion
    logging.debug('Using GPU fusion')
except ModuleNotFoundError:
    from .cpu import PyViews, tsdf_fusion
    logging.debug('Using CPU fusion')
