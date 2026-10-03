"""Central runtime configuration for Nitin Coding Agent.

Single source of truth for every environment variable the agent reads.
Providers must obtain their configuration from :class:`Settings` instead
of calling ``os.getenv`` directly.

Security rules for this module:

* Secret values are never printed, logged, or included in ``repr``/``str``.
* No default secret values exist - an unset key stays ``None`` so the caller
  keeps its existing "key is not set" error.
* Nothing is written back to the environment.
"""

import os
from dataclasses import dataclass

# -------------------------------------------------------------
# Environment variable names (unchanged from the original code)
# -------------------------------------------------------------
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_GEMINI_MODEL = "GEMINI_MODEL"

# -------------------------------------------------------------
# Logging environment variables (Batch 6)
# -------------------------------------------------------------
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_LOG_FILE = "LOG_FILE"

# -------------------------------------------------------------
# Default Gemini model (unchanged from the original code)
# -------------------------------------------------------------
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

# -------------------------------------------------------------
# Default logging level when LOG_LEVEL is unset or empty
# -------------------------------------------------------------
DEFAULT_LOG_LEVEL = "INFO"

MASK = "***"


def _mask(value: str | None) -> str:
    """Render a secret safely for display."""

    if value is None:
        return "None"

    return repr(MASK)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the agent's runtime configuration.

    Build it with :meth:`from_env`; the constructor exists mainly so tests
    can build a fixed instance directly.
    """

    openrouter_api_key: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL

    # Logging configuration (never secrets). LOG_LEVEL names the
    # console/file level; LOG_FILE optionally adds a file destination.
    log_level: str = DEFAULT_LOG_LEVEL
    log_file: str | None = None

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------

    @classmethod
    def from_env(cls, environ=None) -> "Settings":
        """Read configuration from the environment.

        Semantics are identical to the pre-centralization code:

        * keys use ``os.getenv(name)`` -> ``None`` when unset
        * the model uses ``os.getenv(name, DEFAULT_GEMINI_MODEL)`` ->
          the default only when the variable is completely absent

        Logging additions (Batch 6):

        * ``LOG_LEVEL`` falls back to ``INFO`` when unset *or* empty
        * ``LOG_FILE`` falls back to ``None`` when unset *or* empty

        :param environ: mapping to read from. Defaults to ``os.environ``.
            Tests can pass a plain ``dict`` to stay hermetic.
        """

        env = os.environ if environ is None else environ

        return cls(
            openrouter_api_key=env.get(ENV_OPENROUTER_API_KEY),
            gemini_api_key=env.get(ENV_GEMINI_API_KEY),
            gemini_model=env.get(
                ENV_GEMINI_MODEL,
                DEFAULT_GEMINI_MODEL,
            ),
            log_level=(
                env.get(ENV_LOG_LEVEL) or DEFAULT_LOG_LEVEL
            ),
            log_file=env.get(ENV_LOG_FILE) or None,
        )

    # ---------------------------------------------------------
    # Convenience readers
    # ---------------------------------------------------------

    @property
    def has_openrouter_key(self) -> bool:
        """Truthy when an OpenRouter key is present."""

        return bool(self.openrouter_api_key)

    @property
    def has_gemini_key(self) -> bool:
        """Truthy when a Gemini key is present."""

        return bool(self.gemini_api_key)

    # ---------------------------------------------------------
    # Secret-safe display
    # ---------------------------------------------------------

    def masked(self) -> dict:
        """Return a display-only view with secrets redacted."""

        return {
            "openrouter_api_key": _mask(self.openrouter_api_key),
            "gemini_api_key": _mask(self.gemini_api_key),
            "gemini_model": self.gemini_model,
        }

    def __repr__(self) -> str:
        """Never reveal secret values."""

        return (
            "Settings("
            f"openrouter_api_key={_mask(self.openrouter_api_key)}, "
            f"gemini_api_key={_mask(self.gemini_api_key)}, "
            f"gemini_model={self.gemini_model!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()
