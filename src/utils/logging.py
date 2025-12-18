from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Consistent logger across the project.
    Logs to stdout so it works nicely in terminals, Docker, and CI later.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger