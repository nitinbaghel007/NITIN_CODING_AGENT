import pytest

from agent.core.coding_loop import CodingLoop, LazyProvider
from agent.providers.base import LLMProvider


class DummyProvider(LLMProvider):
    """Local provider used to test CodingLoop integration."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return '{"summary": "Test response", "actions": []}'

    def generate_actions(self, task: str) -> dict:
        self.calls += 1
        return {
            "summary": "Test response",
            "actions": [],
        }


class RateLimitProvider(LLMProvider):
    """Local provider that simulates a Gemini rate-limit error."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        raise RuntimeError(
            "GEMINI_RATE_LIMIT: simulated quota exhaustion"
        )

    def generate_actions(self, task: str) -> dict:
        self.calls += 1
        raise RuntimeError(
            "GEMINI_RATE_LIMIT: simulated quota exhaustion"
        )


class SequenceProvider(LLMProvider):
    """Local provider that returns a predefined action sequence."""

    def __init__(self):
        self.calls = 0
        self.responses = [
            {
                "summary": "Inspect calculator",
                "actions": [
                    {
                        "tool": "read_file",
                        "path": "calculator.py",
                    }
                ],
            },
            {
                "summary": "Update calculator",
                "actions": [
                    {
                        "tool": "write_file",
                        "path": "calculator.py",
                        "content": "updated",
                    }
                ],
            },
            {
                "summary": "Run tests",
                "actions": [
                    {
                        "tool": "run_tests",
                    }
                ],
            },
            {
                "summary": "Task completed",
                "actions": [],
            },
        ]

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return '{"summary": "Test response", "actions": []}'

    def generate_actions(self, task: str) -> dict:
        self.calls += 1

        index = min(
            self.calls - 1,
            len(self.responses) - 1,
        )

        return self.responses[index]


class FakeEngine:
    """Local fake engine used to test CodingLoop state tracking."""

    def execute(self, action: dict) -> dict:
        tool = action.get("tool")

        if tool == "read_file":
            return {
                "success": True,
                "tool": "read_file",
                "path": action.get("path"),
                "content": "calculator content",
            }

        if tool == "write_file":
            return {
                "success": True,
                "tool": "write_file",
                "path": action.get("path"),
            }

        if tool == "run_tests":
            return {
                "success": True,
                "tool": "run_tests",
                "result": {
                    "success": True,
                    "return_code": 0,
                    "output": "1 passed",
                },
            }

        return {
            "success": False,
            "tool": tool,
            "error": "Unsupported fake tool.",
        }


def test_coding_loop_registers_openrouter():
    loop = CodingLoop("workspace")

    assert "openrouter" in (
        loop.provider_manager.available()
    )


def test_default_manager_uses_lazy_providers(
    monkeypatch, tmp_path
):
    """Batch 8 dedup regression: the merged LazyProvider class
    keeps the default manager's behaviour exactly as before -
    real providers are constructed only on first use, and a
    missing key surfaces the same controlled error."""

    monkeypatch.delenv(
        "OPENROUTER_API_KEY", raising=False
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    loop = CodingLoop(str(tmp_path))

    for name in ("openrouter", "gemini"):
        provider = loop.provider_manager.providers[name]
        assert isinstance(provider, LazyProvider)
        # Nothing has been constructed at registration time.
        assert provider._provider is None

    with pytest.raises(
        RuntimeError,
        match="OPENROUTER_API_KEY is not set.",
    ):
        loop.provider_manager.get(
            "openrouter"
        ).generate("hello")

    # A failed construction must not be cached.
    assert (
        loop.provider_manager.providers[
            "openrouter"
        ]._provider
        is None
    )


def test_lazy_provider_constructs_once_and_delegates():
    """Batch 8 dedup regression: LazyProvider constructs the
    real provider exactly once, then delegates every call."""

    constructions = []
    received = []

    class DelegatingProvider(LLMProvider):
        def generate(self, prompt: str) -> str:
            received.append(prompt)
            return "generated"

        def generate_actions(self, task: str) -> dict:
            received.append(task)
            return {"actions": []}

    def factory():
        constructions.append("built")
        return DelegatingProvider()

    lazy = LazyProvider(factory)

    # No construction before the first call.
    assert constructions == []

    assert lazy.generate("prompt") == "generated"
    assert lazy.generate_actions("task") == {
        "actions": []
    }

    # One construction total, then reuse of the same instance.
    assert constructions == ["built"]
    assert received == ["prompt", "task"]


def test_coding_loop_uses_provider_manager():
    loop = CodingLoop("workspace")

    dummy = DummyProvider()

    loop.provider_manager.register(
        "dummy",
        dummy,
    )

    loop.provider_manager.set_default(
        "dummy"
    )

    provider = loop._get_provider()

    assert provider is dummy


def test_coding_loop_provider_generates_actions():
    loop = CodingLoop("workspace")

    dummy = DummyProvider()

    loop.provider_manager.register(
        "dummy",
        dummy,
    )

    loop.provider_manager.set_default(
        "dummy"
    )

    result = dummy.generate_actions(
        "Test coding task"
    )

    assert result["summary"] == "Test response"
    assert result["actions"] == []
    assert dummy.calls == 1


def test_coding_loop_completion_with_dummy_provider():
    loop = CodingLoop("workspace")

    dummy = DummyProvider()

    loop.provider_manager.register(
        "dummy",
        dummy,
    )

    loop.provider_manager.set_default(
        "dummy"
    )

    result = loop.run(
        "Test completed task"
    )

    assert result["success"] is False
    assert result["status"] != "ai_error"
    assert result["provider"] == "dummy"
    assert result["status"] == "max_steps_reached"


def test_coding_loop_handles_gemini_rate_limit():
    loop = CodingLoop("workspace")

    rate_limit_provider = RateLimitProvider()

    loop.provider_manager.register(
        "gemini_test",
        rate_limit_provider,
    )

    loop.provider_manager.set_default(
        "gemini_test"
    )

    result = loop.run(
        "Test Gemini quota handling"
    )

    assert result["success"] is False
    assert result["status"] == "rate_limit"
    assert result["provider"] == "gemini_test"
    assert (
        result["provider_status"]
        == "rate_limited"
    )
    assert result["step"] == 1
    assert (
        "GEMINI_RATE_LIMIT"
        in result["error"]
    )
    assert rate_limit_provider.calls == 1


def test_coding_loop_context_tracks_full_task():
    loop = CodingLoop("workspace")

    provider = SequenceProvider()

    loop.provider_manager.register(
        "context_test",
        provider,
    )

    loop.provider_manager.set_default(
        "context_test"
    )

    loop.engine = FakeEngine()

    result = loop.run(
        "Test ContextManager integration."
    )

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["provider"] == "context_test"
    assert result["steps"] == 4

    context = result["context"]

    assert (
        context["task"]
        == "Test ContextManager integration."
    )
    assert context["workspace"] == "workspace"
    assert context["provider"] == "context_test"
    assert context["current_step"] == 4
    assert context["status"] == "completed"

    assert context["files_inspected"] == [
        "calculator.py"
    ]

    assert context["files_changed"] == [
        "calculator.py"
    ]

    assert context["tests_run"] == 1
    assert context["tests_verified"] is True
    assert context["changes_since_test"] is False
    assert context["last_error"] is None

    assert len(context["history"]) == 4

    assert (
        context["history"][0]["action"]["tool"]
        == "read_file"
    )

    assert (
        context["history"][1]["action"]["tool"]
        == "write_file"
    )

    assert (
        context["history"][2]["action"]["tool"]
        == "run_tests"
    )

    assert (
        context["history"][3]["action"] is None
    )


def test_coding_loop_context_tracks_provider_error():
    loop = CodingLoop("workspace")

    provider = RateLimitProvider()

    loop.provider_manager.register(
        "context_error_test",
        provider,
    )

    loop.provider_manager.set_default(
        "context_error_test"
    )

    result = loop.run(
        "Test ContextManager error state."
    )

    context = result["context"]

    assert result["success"] is False
    assert result["status"] == "rate_limit"

    assert context["status"] == "rate_limit"
    assert (
        context["provider"]
        == "context_error_test"
    )
    assert context["current_step"] == 1
    assert (
        context["last_error"]
        is not None
    )
    assert (
        "GEMINI_RATE_LIMIT"
        in context["last_error"]
    )
    assert context["history"] == []