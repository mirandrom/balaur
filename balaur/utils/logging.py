import logging

from typing import *


def _get_library_name() -> str:
    return __name__.split(".")[0]


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the specified name.
    """
    name = name or _get_library_name()
    return logging.getLogger(name)