"""Batch 7: provider reliability, failover and response validation.

Every test runs hermetically:

* no API keys exist unless a test sets a obviously-fake value itself,
* HTTP/SDK entry points are faked locally (never a real request),
* retry sleeps are neutralized.

No test in this file can perform a real provider API call.
"""

import pytest
import requests

from agent.config.logging_setup import (
    configure_logging,
    reset_logging,
)
from agent.config.settings import Settings
from agent.core.approval import NonInteractiveApproval
from agent.core.coding_loop import (
    CodingLoop,
    LazyGeminiProvider,
    LazyOpenRouterProvider,
)
from agent.providers.base import LLMProvider
from agent.providers.failures import (
    CATEGORY_AUTHENTICATION,
    CATEGORY_CONFIGURATION,
    CATEGORY_RATE_LIMIT,
    CATEGORY_RESPONSE,
    CATEGORY_TRANSIENT,
    CATEGORY_UNKNOWN,
    classify_provider_error,
)
from agent.providers.gemini import GeminiProvider
from agent.providers.manager import ProviderManager
from agent.providers.openrouter import OpenRouterProvider


# =========================================================
# Fixtures
# =========================================================


@pytest.fixture(autouse=True)
def _logging_isolation():
    """Batch 6 invariant: no handler stacking between tests."""

    reset_logging()
    yield
    reset_logging()


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Bounded retry delays must not slow the test run."""

    monkeypatch.setattr(
        "agent.core.error_recovery.time.sleep",
        lambda delay: None,
    )


@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch):
    """Every test runs as if no provider key exists."""

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


# =========================================================
# Local fakes (all LLMProvider implementations)
# =========================================================


class FailingProvider(LLMProvider):
    """Always raises the same exception; counts every attempt."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def generate(self, prompt):
        raise self.exc

    def generate_actions(self, task):
        self.calls += 1
        raise self.exc


class SequenceProvider(LLMProvider):
    """Working provider: run tests first, then finish.

    With ``actions`` supplied it returns exactly those actions first
    (one per call), then completion.
    """

    def __init__(self, actions=None):
        self.calls = 0
        self.actions = list(actions) if actions else None

    def generate(self, prompt):
        return '{"summary": "ok", "actions": []}'

    def generate_actions(self, task):
        self.calls += 1

        if self.actions is not None:
            if self.actions:
                return {
                    "summary": "Execute planned action",
                    "actions": [self.actions.pop(0)],
                }

            return {
                "summary": "Task completed",
                "actions": [],
            }

        if self.calls == 1:
            return {
                "summary": "Run tests",
                "actions": [{"tool": "run_tests"}],
            }

        return {
            "summary": "Task completed",
            "actions": [],
        }


class ReturningProvider(LLMProvider):
    """Returns a fixed value without raising (wrong-shape replies)."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return "irrelevant"

    def generate_actions(self, task):
        self.calls += 1
        return self.value


class FakeEngine:
    """Deterministic stand-in for the real ActionEngine."""

    def __init__(self):
        self.calls = 0

    def execute(self, action):
        self.calls += 1
        return {
            "success": True,
            "tool": action.get("tool"),
            "result": {
                "success": True,
                "return_code": 0,
                "output": "1 passed",
            },
        }


class FakeResponse:
    """Stand-in for requests.Response; never touches the network."""

    def __init__(
        self,
        status_code=200,
        json_data=None,
        json_error=None,
    ):
        self.status_code = status_code
        self._json_data = json_data
        self._json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: error",
                response=self,
            )

    def json(self):
        if self._json_error is not None:
            raise self._json_error

        return self._json_data


class _FakeGenaiResponse:
    def __init__(self, text):
        self.text = text


class _FakeGenaiClient:
    """Fake google-genai client; generate_content never blocks."""

    def __init__(self, text):
        payload = _FakeGenaiResponse(text)

        class _Models:
            def generate_content(self, model, contents):
                return payload

        self.models = _Models()


def _openrouter_provider():
    """OpenRouterProvider without running __init__ (no env, no HTTP)."""

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider.api_key = "sk-or-fake-test-key"
    provider.model = "test-model"
    return provider


def _gemini_provider(text):
    """GeminiProvider without running __init__ (no env, no client)."""

    provider = GeminiProvider.__new__(GeminiProvider)
    provider.api_key = "fake-test-key"
    provider.model = "test-model"
    provider.client = _FakeGenaiClient(text)
    return provider


def _make_loop(tmp_path, primary, fallback=None, default="openrouter"):
    manager = ProviderManager()
    manager.register("openrouter", primary)
    manager.register(
        "gemini",
        fallback if fallback is not None else SequenceProvider(),
    )
    manager.set_default(default)

    loop = CodingLoop(str(tmp_path), provider_manager=manager)
    loop.engine = FakeEngine()
    return loop


def _failover_entries(result):
    return [
        item
        for item in result["history"]
        if (item.get("result") or {}).get("status")
        == "provider_failover"
    ]


def _status_entries(result, status):
    return [
        item
        for item in result["history"]
        if (item.get("result") or {}).get("status") == status
    ]


# =========================================================
# Failure classification (Phase 2)
# =========================================================


def test_missing_keys_are_configuration_failures():
    assert (
        classify_provider_error(
            RuntimeError("OPENROUTER_API_KEY is not set.")
        )
        == CATEGORY_CONFIGURATION
    )
    assert (
        classify_provider_error(
            RuntimeError("GEMINI_API_KEY is not set.")
        )
        == CATEGORY_CONFIGURATION
    )


def test_auth_messages_are_authentication_failures():
    assert (
        classify_provider_error(
            RuntimeError("401 invalid API key")
        )
        == CATEGORY_AUTHENTICATION
    )
    assert (
        classify_provider_error(
            requests.HTTPError(
                "403 Client Error: Forbidden for url: https://x"
            )
        )
        == CATEGORY_AUTHENTICATION
    )


def test_exception_status_codes_are_authentication_failures():
    class StatusError(Exception):
        """Mimics google.genai.errors.APIError with a code."""

    status_error = StatusError("unauthorized")
    status_error.code = 401

    assert (
        classify_provider_error(status_error)
        == CATEGORY_AUTHENTICATION
    )

    http_error = requests.HTTPError(
        "client error",
        response=FakeResponse(status_code=403),
    )

    assert (
        classify_provider_error(http_error)
        == CATEGORY_AUTHENTICATION
    )


def test_timeout_and_connection_errors_are_transient():
    assert (
        classify_provider_error(
            requests.Timeout("Read timed out. (read timeout=60)")
        )
        == CATEGORY_TRANSIENT
    )
    assert (
        classify_provider_error(
            requests.ConnectionError("Connection aborted.")
        )
        == CATEGORY_TRANSIENT
    )


def test_http_5xx_and_rate_bursts_are_transient():
    assert (
        classify_provider_error(
            RuntimeError("HTTP 503 service unavailable")
        )
        == CATEGORY_TRANSIENT
    )
    assert (
        classify_provider_error(
            requests.HTTPError("429 Too Many Requests")
        )
        == CATEGORY_TRANSIENT
    )


def test_client_errors_are_unusable_configuration():
    assert (
        classify_provider_error(
            requests.HTTPError(
                "400 Client Error: Bad Request for url: https://x"
            )
        )
        == CATEGORY_CONFIGURATION
    )
    assert (
        classify_provider_error(
            requests.HTTPError(
                "404 Client Error: Not Found for url: https://x"
            )
        )
        == CATEGORY_CONFIGURATION
    )


def test_malformed_responses_are_response_failures():
    assert (
        classify_provider_error(KeyError("choices"))
        == CATEGORY_RESPONSE
    )
    assert (
        classify_provider_error(
            ValueError("Model returned invalid JSON:\nnope")
        )
        == CATEGORY_RESPONSE
    )
    assert (
        classify_provider_error(
            RuntimeError("Gemini returned an empty response.")
        )
        == CATEGORY_RESPONSE
    )


def test_programming_errors_are_unknown():
    assert (
        classify_provider_error(
            ValueError("invalid internal argument")
        )
        == CATEGORY_UNKNOWN
    )
    assert (
        classify_provider_error(TypeError("bad call"))
        == CATEGORY_UNKNOWN
    )


def test_rate_limit_prefixes_keep_their_existing_category():
    assert (
        classify_provider_error(
            RuntimeError("OPENROUTER_RATE_LIMIT: quota")
        )
        == CATEGORY_RATE_LIMIT
    )
    assert (
        classify_provider_error(
            RuntimeError("GEMINI_RATE_LIMIT: quota")
        )
        == CATEGORY_RATE_LIMIT
    )


def test_unregistered_default_provider_is_configuration():
    assert (
        classify_provider_error(
            RuntimeError("Provider not available: ")
        )
        == CATEGORY_CONFIGURATION
    )


# =========================================================
# Configuration failure: missing-key failover (Phase 3)
# =========================================================


def test_missing_openrouter_key_fails_over_to_gemini(tmp_path):
    fallback = SequenceProvider()

    # Default construction: lazy providers, and the autouse
    # fixture guarantees no keys exist.
    loop = CodingLoop(str(tmp_path))
    loop.provider_manager.register("gemini", fallback)
    loop.engine = FakeEngine()

    result = loop.run("build the feature")

    assert result["status"] == "completed"
    assert result["provider"] == "gemini"

    failovers = _failover_entries(result)
    assert len(failovers) == 1
    assert failovers[0]["result"]["from_provider"] == "openrouter"
    assert failovers[0]["result"]["to_provider"] == "gemini"
    assert (
        failovers[0]["result"]["error"]
        == "OPENROUTER_API_KEY is not set."
    )


def test_missing_gemini_key_fails_over_to_openrouter(tmp_path):
    fallback = SequenceProvider()

    manager = ProviderManager()
    manager.register("gemini", LazyGeminiProvider())
    manager.register("openrouter", fallback)
    manager.set_default("gemini")

    loop = CodingLoop(str(tmp_path), provider_manager=manager)
    loop.engine = FakeEngine()

    result = loop.run("build the feature")

    assert result["status"] == "completed"
    assert result["provider"] == "openrouter"
    assert len(_failover_entries(result)) == 1


def test_both_keys_missing_returns_controlled_ai_error(tmp_path):
    loop = CodingLoop(str(tmp_path))
    loop.engine = FakeEngine()

    result = loop.run("impossible without keys")

    # No crash: both providers were attempted, then a controlled
    # result (unchanged shape) came back.
    assert result["status"] == "ai_error"
    assert result["success"] is False
    assert result["error"] == "GEMINI_API_KEY is not set."
    assert len(_failover_entries(result)) == 1


def test_no_registered_provider_returns_controlled_ai_error(
    tmp_path,
):
    manager = ProviderManager()

    loop = CodingLoop(str(tmp_path), provider_manager=manager)
    loop.engine = FakeEngine()

    result = loop.run("no providers at all")

    assert result["status"] == "ai_error"
    assert "Provider not available" in result["error"]


# =========================================================
# Authentication failures (Phase 4)
# =========================================================


def test_authentication_failure_fails_over(tmp_path):
    primary = FailingProvider(
        requests.HTTPError(
            "401 Client Error: Unauthorized for url: https://x"
        )
    )
    fallback = SequenceProvider()

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("auth probe")

    assert result["status"] == "completed"
    assert result["provider"] == "gemini"
    # Permanent failure: exactly one attempt on the primary.
    assert primary.calls == 1
    assert fallback.calls == 2


def test_authentication_failure_without_fallback_stays_controlled(
    tmp_path,
):
    primary = FailingProvider(RuntimeError("401 invalid API key"))

    manager = ProviderManager()
    manager.register("openrouter", primary)
    manager.set_default("openrouter")

    loop = CodingLoop(str(tmp_path), provider_manager=manager)
    loop.engine = FakeEngine()

    result = loop.run("auth probe")

    assert result["status"] == "ai_error"
    assert primary.calls == 1


# =========================================================
# Transient failures (Phases 4 + 5): bounded retry, then failover
# =========================================================


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout("Read timed out. (read timeout=60)"),
        requests.ConnectionError("Connection aborted."),
        RuntimeError("HTTP 503 service unavailable"),
    ],
    ids=["timeout", "connection-error", "http-503"],
)
def test_transient_failures_retry_bounded_then_fail_over(
    tmp_path,
    error,
):
    primary = FailingProvider(error)
    fallback = SequenceProvider()

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("transient probe")

    assert result["status"] == "completed"
    assert result["provider"] == "gemini"
    # Exact attempt count: 1 attempt + 2 bounded retries, no more.
    assert primary.calls == 3
    assert fallback.calls == 2


def test_transient_exhaustion_without_fallback_is_bounded(tmp_path):
    primary = FailingProvider(
        requests.Timeout("Read timed out. (read timeout=60)")
    )

    manager = ProviderManager()
    manager.register("openrouter", primary)
    manager.set_default("openrouter")

    loop = CodingLoop(str(tmp_path), provider_manager=manager)
    loop.engine = FakeEngine()

    result = loop.run("no fallback available")

    assert result["status"] == "ai_error"
    # Bounded: never loops forever, never more than 3 attempts.
    assert primary.calls == 3


def test_client_error_is_not_retried_and_fails_over(tmp_path):
    primary = FailingProvider(
        requests.HTTPError(
            "400 Client Error: Bad Request for url: https://x"
        )
    )
    fallback = SequenceProvider()

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("client error probe")

    assert result["status"] == "completed"
    # Permanent failure: exactly one attempt, then failover.
    assert primary.calls == 1
    assert len(_failover_entries(result)) == 1


# =========================================================
# Malformed provider responses (Phase 6)
# =========================================================


def test_keyerror_response_fails_over(tmp_path):
    primary = FailingProvider(KeyError("choices"))
    fallback = SequenceProvider()

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("malformed probe")

    assert result["status"] == "completed"
    assert primary.calls == 1
    assert len(_failover_entries(result)) == 1


def test_invalid_json_valueerror_fails_over(tmp_path):
    primary = FailingProvider(
        ValueError("Model returned invalid JSON:\nnot json")
    )
    fallback = SequenceProvider()

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("malformed probe")

    assert result["status"] == "completed"
    assert primary.calls == 1


def test_empty_response_fails_over(tmp_path):
    primary = FailingProvider(
        RuntimeError("Gemini returned an empty response.")
    )
    fallback = SequenceProvider()

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("empty probe")

    assert result["status"] == "completed"
    assert primary.calls == 1


def test_wrong_shape_plan_fails_over(tmp_path):
    # generate_actions returns a non-dict without raising.
    primary = ReturningProvider(None)
    fallback = SequenceProvider()

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("wrong shape probe")

    assert result["status"] == "completed"
    assert primary.calls == 1
    assert len(_failover_entries(result)) == 1


def test_non_list_actions_fails_over(tmp_path):
    primary = ReturningProvider(
        {"summary": "broken", "actions": "not-a-list"}
    )
    fallback = SequenceProvider()

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("wrong actions probe")

    assert result["status"] == "completed"
    assert primary.calls == 1


# --- provider-level response validation -------------------


def test_openrouter_malformed_json_body_is_controlled(
    monkeypatch,
):
    provider = _openrouter_provider()

    monkeypatch.setattr(
        "agent.providers.openrouter.requests.post",
        lambda *args, **kwargs: FakeResponse(
            json_error=ValueError("Expecting value")
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        provider.generate("hi")

    assert "invalid JSON" in str(excinfo.value)
    assert (
        classify_provider_error(excinfo.value)
        == CATEGORY_RESPONSE
    )


def test_openrouter_missing_choices_is_controlled(monkeypatch):
    provider = _openrouter_provider()

    monkeypatch.setattr(
        "agent.providers.openrouter.requests.post",
        lambda *args, **kwargs: FakeResponse(json_data={}),
    )

    # A KeyError must not escape: it becomes a controlled failure.
    with pytest.raises(ValueError) as excinfo:
        provider.generate("hi")

    assert "unexpected response shape" in str(excinfo.value)
    assert (
        classify_provider_error(excinfo.value)
        == CATEGORY_RESPONSE
    )


def test_openrouter_non_text_content_is_controlled(monkeypatch):
    provider = _openrouter_provider()

    monkeypatch.setattr(
        "agent.providers.openrouter.requests.post",
        lambda *args, **kwargs: FakeResponse(
            json_data={
                "choices": [
                    {"message": {"content": 12345}}
                ]
            }
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        provider.generate("hi")

    assert "non-text response" in str(excinfo.value)
    assert (
        classify_provider_error(excinfo.value)
        == CATEGORY_RESPONSE
    )


def test_openrouter_unauthorized_status_is_authentication(
    monkeypatch,
):
    provider = _openrouter_provider()

    monkeypatch.setattr(
        "agent.providers.openrouter.requests.post",
        lambda *args, **kwargs: FakeResponse(status_code=401),
    )

    with pytest.raises(
        requests.HTTPError
    ) as excinfo:
        provider.generate("hi")

    assert (
        classify_provider_error(excinfo.value)
        == CATEGORY_AUTHENTICATION
    )


def test_openrouter_rate_limit_keeps_existing_message(
    monkeypatch,
):
    provider = _openrouter_provider()

    monkeypatch.setattr(
        "agent.providers.openrouter.requests.post",
        lambda *args, **kwargs: FakeResponse(
            status_code=429,
            json_data={"error": {"message": "quota exhausted"}},
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        provider.generate("hi")

    # Existing public contract: the message prefix is exact.
    assert str(excinfo.value) == (
        "OPENROUTER_RATE_LIMIT: quota exhausted"
    )
    assert (
        classify_provider_error(excinfo.value)
        == CATEGORY_RATE_LIMIT
    )


def test_gemini_non_text_response_is_controlled():
    provider = _gemini_provider(12345)

    with pytest.raises(RuntimeError) as excinfo:
        provider.generate("hi")

    assert "non-text response" in str(excinfo.value)
    assert (
        classify_provider_error(excinfo.value)
        == CATEGORY_RESPONSE
    )


def test_gemini_empty_response_is_controlled():
    provider = _gemini_provider("")

    with pytest.raises(RuntimeError) as excinfo:
        provider.generate("hi")

    assert "empty response" in str(excinfo.value)
    assert (
        classify_provider_error(excinfo.value)
        == CATEGORY_RESPONSE
    )


def test_gemini_malformed_json_is_response_failure():
    provider = _gemini_provider("")

    # Exercise the parser directly with an empty payload.
    with pytest.raises(ValueError) as excinfo:
        provider._parse_json_response("   ")

    assert (
        classify_provider_error(excinfo.value)
        == CATEGORY_RESPONSE
    )


# =========================================================
# Failover boundaries (Phases 3 + 4)
# =========================================================


def test_primary_success_does_not_call_fallback(tmp_path):
    primary = SequenceProvider()
    fallback = SequenceProvider()

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("easy task")

    assert result["status"] == "completed"
    assert result["provider"] == "openrouter"
    assert primary.calls == 2
    assert fallback.calls == 0
    assert _failover_entries(result) == []


def test_fallback_also_fails_returns_controlled_ai_error(tmp_path):
    primary = FailingProvider(
        RuntimeError("OPENROUTER_API_KEY is not set.")
    )
    fallback = FailingProvider(
        RuntimeError("GEMINI_API_KEY is not set.")
    )

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("nothing works")

    assert result["status"] == "ai_error"
    # Each provider attempted exactly once; the final error is the
    # fallback's, proving the chain ran to exhaustion.
    assert primary.calls == 1
    assert fallback.calls == 1
    assert result["error"] == "GEMINI_API_KEY is not set."
    assert len(_failover_entries(result)) == 1


def test_unknown_error_never_fails_over(tmp_path):
    primary = FailingProvider(
        ValueError("invalid internal argument")
    )
    fallback = SequenceProvider()

    loop = _make_loop(tmp_path, primary, fallback)
    result = loop.run("programming bug")

    # Programming errors are not failover material.
    assert result["status"] == "ai_error"
    assert primary.calls == 1
    assert fallback.calls == 0
    assert _failover_entries(result) == []


# =========================================================
# Safety and approval boundaries (Phase 8)
# =========================================================


def test_provider_failure_does_not_bypass_safety(
    tmp_path,
    monkeypatch,
):
    executed = []

    def _no_subprocess(*args, **kwargs):
        executed.append(args)
        raise AssertionError("terminal must never run")

    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run",
        _no_subprocess,
    )

    dangerous = SequenceProvider(
        actions=[
            {
                "tool": "run_command",
                "command": "del important.txt",
            }
        ]
    )

    loop = CodingLoop(str(tmp_path))
    loop.provider_manager.register(
        "openrouter",
        FailingProvider(
            RuntimeError("OPENROUTER_API_KEY is not set.")
        ),
    )
    loop.provider_manager.register("gemini", dangerous)

    result = loop.run("do something dangerous")

    blocked = _status_entries(result, "blocked")
    assert blocked, "SafetyManager must still block the action"
    assert blocked[0]["result"]["success"] is False
    assert executed == []


def test_provider_failure_does_not_bypass_approval(
    tmp_path,
    monkeypatch,
):
    executed = []

    def _no_subprocess(*args, **kwargs):
        executed.append(args)
        raise AssertionError("terminal must never run")

    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run",
        _no_subprocess,
    )

    risky = SequenceProvider(
        actions=[
            {
                "tool": "run_command",
                "command": "pip install example-package",
            }
        ]
    )

    loop = CodingLoop(
        str(tmp_path),
        approval_callback=NonInteractiveApproval(),
    )
    loop.provider_manager.register(
        "openrouter",
        FailingProvider(
            RuntimeError("OPENROUTER_API_KEY is not set.")
        ),
    )
    loop.provider_manager.register("gemini", risky)

    result = loop.run("install a package")

    denied = _status_entries(result, "approval_required")
    assert denied, "ApprovalManager must still gate the action"
    assert executed == []


def test_provider_failure_cannot_execute_actions_by_itself(
    tmp_path,
    monkeypatch,
):
    def _no_input(*args, **kwargs):
        raise AssertionError("approval must not be prompted")

    monkeypatch.setattr("builtins.input", _no_input)

    engine = FakeEngine()

    loop = CodingLoop(str(tmp_path))
    loop.engine = engine
    loop.provider_manager.register(
        "openrouter",
        FailingProvider(
            RuntimeError("OPENROUTER_API_KEY is not set.")
        ),
    )
    loop.provider_manager.register(
        "gemini",
        FailingProvider(
            RuntimeError("GEMINI_API_KEY is not set.")
        ),
    )

    result = loop.run("do anything")

    # A failure path alone never executes anything.
    assert result["status"] == "ai_error"
    assert engine.calls == 0


# =========================================================
# Logging integration (Phase 7)
# =========================================================


def test_provider_failure_is_logged(tmp_path, capsys):
    configure_logging(Settings())

    loop = _make_loop(
        tmp_path,
        FailingProvider(
            RuntimeError("OPENROUTER_API_KEY is not set.")
        ),
    )
    loop.run("log probe")

    captured = capsys.readouterr()
    assert (
        "provider failure: provider=openrouter "
        "category=configuration"
    ) in captured.err


def test_failover_event_is_logged(tmp_path, capsys):
    configure_logging(Settings())

    loop = _make_loop(
        tmp_path,
        FailingProvider(
            RuntimeError("OPENROUTER_API_KEY is not set.")
        ),
    )
    loop.run("log probe")

    captured = capsys.readouterr()
    assert (
        "provider failover: from=openrouter to=gemini"
    ) in captured.err


def test_exhausted_failover_and_final_outcome_are_logged(
    tmp_path,
    capsys,
):
    configure_logging(Settings())

    loop = _make_loop(
        tmp_path,
        FailingProvider(
            RuntimeError("OPENROUTER_API_KEY is not set.")
        ),
        fallback=FailingProvider(
            RuntimeError("GEMINI_API_KEY is not set.")
        ),
    )
    loop.run("log probe")

    captured = capsys.readouterr()
    assert (
        "no fallback provider available: "
        "provider=gemini category=configuration"
    ) in captured.err
    assert (
        "provider run failed: provider=gemini "
        "category=configuration outcome=ai_error"
    ) in captured.err


def test_api_keys_are_not_present_in_logs(
    tmp_path,
    capsys,
    monkeypatch,
):
    fake_openrouter_key = "sk-or-batch7-never-log-this-000"
    fake_gemini_key = "AIza-batch7-never-log-this-111"

    monkeypatch.setenv("OPENROUTER_API_KEY", fake_openrouter_key)
    monkeypatch.setenv("GEMINI_API_KEY", fake_gemini_key)

    configure_logging(Settings())

    loop = _make_loop(
        tmp_path,
        FailingProvider(
            requests.HTTPError(
                "401 Client Error: Unauthorized"
            ),
        ),
        fallback=FailingProvider(
            RuntimeError("GEMINI_API_KEY is not set.")
        ),
    )
    loop.run("secret probe")

    captured = capsys.readouterr()

    assert fake_openrouter_key not in captured.err
    assert fake_gemini_key not in captured.err
    assert "Bearer" not in captured.err
    # The failure itself was still logged.
    assert "provider failure: provider=openrouter" in captured.err
