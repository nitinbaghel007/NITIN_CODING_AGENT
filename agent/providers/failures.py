"""Deterministic provider failure categories.

A small classifier - not a framework. It maps a provider exception to
exactly one category using exception types and message markers, in the
same marker style already used by ``ErrorRecoveryManager`` and the
``*_RATE_LIMIT:`` provider messages.

The coding loop uses the category to decide - deterministically -
whether a failover to another configured provider is allowed:

* ``configuration``  missing/unusable API key or endpoint -> failover
* ``authentication`` invalid/unauthorized key             -> failover
* ``transient``      timeout/connection/5xx/rate bursts   -> failover
* ``response``       malformed/unexpected provider reply  -> failover
* ``rate_limit``     existing ``*_RATE_LIMIT:`` messages  -> the
                     pre-existing rate-limit failover path
* ``unknown``        programming errors, invalid internal
                     arguments, unexpected local bugs      -> NEVER
                     failover and NEVER retry

No secrets are read or logged here; the module only inspects the
exception it is handed.
"""

import requests

# -------------------------------------------------------------
# Categories
# -------------------------------------------------------------

CATEGORY_CONFIGURATION = "configuration"
CATEGORY_AUTHENTICATION = "authentication"
CATEGORY_TRANSIENT = "transient"
CATEGORY_RESPONSE = "response"
CATEGORY_RATE_LIMIT = "rate_limit"
CATEGORY_UNKNOWN = "unknown"

# Categories that justify switching to another configured provider.
# Rate limit is handled by the existing dedicated path; unknown is
# deliberately excluded (programming errors must not fail over).
FAILOVER_CATEGORIES = frozenset(
    {
        CATEGORY_CONFIGURATION,
        CATEGORY_AUTHENTICATION,
        CATEGORY_TRANSIENT,
        CATEGORY_RESPONSE,
    }
)

# -------------------------------------------------------------
# Detection rules
# -------------------------------------------------------------

# Existing provider message conventions (checked verbatim, in the
# exact shape CodingLoop already recognizes).
RATE_LIMIT_PREFIXES = (
    "GEMINI_RATE_LIMIT:",
    "OPENROUTER_RATE_LIMIT:",
)

CONFIGURATION_MARKERS = (
    "API_KEY IS NOT SET",
    "PROVIDER NOT AVAILABLE",
    # Permanent request rejections against this provider:
    # 400/404/422 mean the request or endpoint is unusable there.
    "400",
    "404",
    "422",
)

AUTHENTICATION_MARKERS = (
    "401",
    "403",
    "UNAUTHORIZED",
    "INVALID API KEY",
    "INVALID JWT",
    "PERMISSION DENIED",
    "AUTHENTICATION",
    "AUTHORIZATION",
)

AUTHENTICATION_STATUS_CODES = (401, 403)

TRANSIENT_MARKERS = (
    "500",
    "502",
    "503",
    "504",
    "429",
    "BAD GATEWAY",
    "SERVICE UNAVAILABLE",
    "TEMPORARY",
    "TIMED OUT",
    "TIMEOUT",
    "CONNECTION RESET",
    "CONNECTION ABORTED",
    "CONNECTION ERROR",
    "NETWORK ERROR",
    "UNAVAILABLE",
)

TRANSIENT_EXCEPTION_TYPES = (
    requests.Timeout,
    requests.ConnectionError,
    TimeoutError,
    ConnectionError,
)

RESPONSE_MARKERS = (
    "INVALID JSON",
    "EMPTY RESPONSE",
    "UNEXPECTED RESPONSE SHAPE",
    "NON-TEXT RESPONSE",
    "MUST BE AN OBJECT",
    "MUST BE A LIST",
    "DOES NOT CONTAIN",
)

# KeyError/IndexError at the provider boundary are response-shape
# failures (e.g. a reply without "choices").
RESPONSE_EXCEPTION_TYPES = (KeyError, IndexError)


def _status_code_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status code from a provider exception."""

    code = getattr(exc, "code", None)

    if isinstance(code, int):
        return code

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)

    if isinstance(status, int):
        return status

    return None


def classify_provider_error(exc: BaseException) -> str:
    """Return the failure category for a provider exception.

    Pure function: same exception, same answer, no side effects.
    """

    message = str(exc).upper()

    # Existing rate-limit convention wins first, unchanged.
    if message.startswith(RATE_LIMIT_PREFIXES):
        return CATEGORY_RATE_LIMIT

    # Exception type outranks message text: these types are
    # unambiguous regardless of what their message contains.
    if isinstance(exc, TRANSIENT_EXCEPTION_TYPES):
        return CATEGORY_TRANSIENT

    if isinstance(exc, RESPONSE_EXCEPTION_TYPES):
        return CATEGORY_RESPONSE

    if any(
        marker in message
        for marker in CONFIGURATION_MARKERS
    ):
        return CATEGORY_CONFIGURATION

    if (
        _status_code_of(exc)
        in AUTHENTICATION_STATUS_CODES
        or any(
            marker in message
            for marker in AUTHENTICATION_MARKERS
        )
    ):
        return CATEGORY_AUTHENTICATION

    if any(
        marker in message
        for marker in TRANSIENT_MARKERS
    ):
        return CATEGORY_TRANSIENT

    if any(
        marker in message
        for marker in RESPONSE_MARKERS
    ):
        return CATEGORY_RESPONSE

    return CATEGORY_UNKNOWN
