"""Centralized logging for Nitin Coding Agent.

Single configuration point for the whole application:

* one console handler that always writes to ``sys.stderr`` -
  stdout stays reserved for the CLI's JSON document,
* one optional file handler when ``Settings.log_file`` is set,
* idempotent setup: :func:`configure_logging` replaces the handlers
  this module installed on the previous call, so handlers never
  stack and no record is ever emitted twice.

Standard library only - no third-party logging dependency.

Secrets are never logged: callers pass ``Settings`` objects (whose
``__str__``/``repr`` mask API keys) instead of raw environment
values, and nothing in this module logs a full prompt or an
environment mapping.
"""

import logging
import sys

from agent.config.settings import Settings

# The logger this module configures. Modules under the ``agent``
# package obtain child loggers (``logging.getLogger("agent...")``)
# whose records propagate to this one.
LOGGER_NAME = "agent"

# Attribute marking handlers installed here, so a repeated call can
# replace exactly the handlers it owns and nothing else.
_MANAGED_HANDLER = "_nitin_managed_handler"

# Human-readable console format including the level so DEBUG/INFO/
# WARNING/ERROR records are distinguishable at a glance.
_CONSOLE_FORMAT = "%(levelname)s %(name)s: %(message)s"

# File format adds a timestamp for later diagnosis.
_FILE_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s: %(message)s"
)


def _resolve_level(value) -> int:
    """Translate a level name (or number) into a logging level.

    Unknown or empty values fall back to INFO so a typo in
    ``LOG_LEVEL`` can never break the run.
    """

    if isinstance(value, int):
        return value

    candidate = getattr(
        logging,
        str(value).strip().upper(),
        None,
    )

    if isinstance(candidate, int):
        return candidate

    return logging.INFO


def _is_known_level(value) -> bool:
    """True when ``value`` names (or is) a real logging level."""

    if isinstance(value, int):
        return True

    return isinstance(
        getattr(logging, str(value).strip().upper(), None),
        int,
    )


def configure_logging(
    settings: Settings | None = None,
) -> logging.Logger:
    """Configure application logging for the current run.

    Called once at the CLI boundary; safe to call repeatedly
    because handlers from the previous call are removed first.

    Console records go to stderr (never stdout, so logging can
    never corrupt the ``--json`` document); a file destination is
    added when ``settings.log_file`` is set. A broken log path
    degrades to console-only logging instead of crashing.

    :param settings: configuration snapshot. Defaults to
        ``Settings.from_env()``.
    :return: the centrally configured ``agent`` logger.
    """

    if settings is None:
        settings = Settings.from_env()

    logger = logging.getLogger(LOGGER_NAME)

    # Replace handlers from a previous call - never stack them.
    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER, False):
            logger.removeHandler(handler)
            handler.close()

    level = _resolve_level(settings.log_level)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter(_CONSOLE_FORMAT)
    )
    setattr(console, _MANAGED_HANDLER, True)
    logger.addHandler(console)

    if settings.log_file:
        try:
            file_handler = logging.FileHandler(
                settings.log_file,
                encoding="utf-8",
            )
        except OSError as exc:
            # A bad log path must degrade to console logging,
            # never crash the run.
            logger.warning(
                "log file unavailable (%s): %s",
                settings.log_file,
                exc,
            )
        else:
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter(_FILE_FORMAT)
            )
            setattr(
                file_handler, _MANAGED_HANDLER, True
            )
            logger.addHandler(file_handler)

    logger.setLevel(level)

    # Exactly one emission path: this logger's own handlers, so a
    # root-logger handler added by an embedding process can never
    # duplicate a line.
    logger.propagate = False

    if not _is_known_level(settings.log_level):
        logger.warning(
            "Unknown log level %r - using INFO.",
            settings.log_level,
        )

    return logger


def reset_logging() -> None:
    """Remove handlers installed by :func:`configure_logging`.

    Used by tests (and available at shutdown) so no handler stays
    bound to a stream that has since been closed.
    """

    logger = logging.getLogger(LOGGER_NAME)

    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER, False):
            logger.removeHandler(handler)
            handler.close()

    logger.propagate = True
