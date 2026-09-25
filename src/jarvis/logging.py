"""Structured logging setup (structlog)."""

import logging
import sys

import structlog
from structlog.typing import FilteringBoundLogger


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for readable console output on stderr.

    Args:
        level: Minimum level name to emit, e.g. "DEBUG" or "INFO".
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a logger bound to `name` (typically the caller's `__name__`).

    The logger stays lazy, so module-level loggers created at import time still
    pick up the configuration applied later by `configure_logging`.
    """
    return structlog.get_logger(logger_name=name)
