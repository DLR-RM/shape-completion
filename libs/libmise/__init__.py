import logging

try:
    from mise_ext import MISE as MISE
except ImportError:
    logging.warning(
        "The `mise` library is not installed. Please install it using `python libs/libmanager.py install mise`"
    )
    raise
