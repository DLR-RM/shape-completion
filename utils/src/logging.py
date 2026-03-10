import logging
import os
import sys
from typing import Any, cast

from lightning.pytorch.utilities import rank_zero_only

# Configure logging
logging.basicConfig(level=logging.INFO)

# Custom debug levels
DEBUG_LEVEL_1 = 12
DEBUG_LEVEL_2 = 11

logging.addLevelName(DEBUG_LEVEL_1, "DEBUG_LEVEL_1")
logging.addLevelName(DEBUG_LEVEL_2, "DEBUG_LEVEL_2")

# Global set to keep track of loggers created using the setup_logger function
_created_loggers = set()


class ShapeCompletionLogger(logging.Logger):
    def debug_level_1(self, message: str, *args: object, **kwargs: object) -> None: ...
    def debug_level_2(self, message: str, *args: object, **kwargs: object) -> None: ...


@rank_zero_only
def debug_with_level(self, message: str, level: int = logging.DEBUG, *args, **kwargs) -> None:
    if self.isEnabledFor(level):
        self._log(level, message, args, **kwargs)


def setup_logger(name: str = __name__) -> ShapeCompletionLogger:
    if not name or name == "__main__":
        cwd = os.getcwd()
        rel_path = os.path.relpath(sys.argv[0], start=cwd)
        name = os.path.splitext(rel_path)[0].replace(os.sep, ".")

    logger = cast(ShapeCompletionLogger, logging.getLogger(name))

    if name not in _created_loggers:
        logger.setLevel(logging.INFO)
        _created_loggers.add(name)

        # Define logging levels to wrap
        logging_levels = ("debug", "info", "warning", "error", "fatal", "critical", "debug_level_1", "debug_level_2")

        # Add custom debug levels
        logger_any = cast(Any, logger)
        logger_any.debug_level_1 = lambda message, *args, **kwargs: debug_with_level(
            logger, message, DEBUG_LEVEL_1, *args, **kwargs
        )
        logger_any.debug_level_2 = lambda message, *args, **kwargs: debug_with_level(
            logger, message, DEBUG_LEVEL_2, *args, **kwargs
        )

        # Wrap existing log methods to pass through debug_with_level
        for level_name in logging_levels[:-2]:
            level = getattr(logging, level_name.upper())
            setattr(
                logger_any,
                level_name,
                lambda message, lvl=level, *args, **kwargs: debug_with_level(logger, message, lvl, *args, **kwargs),
            )

        # Handle exception method separately
        logger_any.exception = lambda message, *args, exc_info=True, **kwargs: debug_with_level(
            logger, message, logging.ERROR, *args, exc_info=exc_info, **kwargs
        )

    return logger


def set_log_level(level: int | str) -> None:
    for logger_name in _created_loggers:
        logger = cast(ShapeCompletionLogger, logging.getLogger(logger_name))
        logger.setLevel(level)
        logger.debug(f"Using log level {level} for logger {logger.name}")
