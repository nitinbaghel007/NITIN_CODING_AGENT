from agent.core.coding_loop import CodingLoop
from agent.providers.base import LLMProvider
from agent.providers.manager import ProviderManager


class FakeProvider(LLMProvider):
    def __init__(self, actions=None, error=None):
        self.actions = list(actions or [])
        self.error = error

    def generate(self, prompt: str) -> str:
        return "fake"

    def generate_actions(self, task: str) -> dict:
        if self.error is not None:
            raise RuntimeError(self.error)

        if self.actions:
            return self.actions.pop(0)

        return {
            "summary": "Task completed",
            "actions": [],
        }


def test_provider_manager_failover_switches_to_next_free_provider():
    manager = ProviderManager()
    manager.register("openrouter", FakeProvider())
    manager.register("gemini", FakeProvider())
    manager.set_default("openrouter")
    manager.set_status("openrouter", "rate_limited")

    result = manager.failover("openrouter")

    assert result == "gemini"
    assert manager.default_provider == "gemini"


def test_provider_manager_failover_returns_none_when_all_providers_unavailable():
    manager = ProviderManager()
    manager.register("openrouter", FakeProvider())
    manager.register("gemini", FakeProvider())
    manager.set_default("openrouter")
    manager.set_status("openrouter", "rate_limited")
    manager.set_status("gemini", "rate_limited")

    result = manager.failover("openrouter")

    assert result is None
    assert manager.default_provider == "openrouter"


def test_coding_loop_fails_over_from_openrouter_to_gemini(monkeypatch, tmp_path):
    openrouter = FakeProvider(
        error="OPENROUTER_RATE_LIMIT: quota exhausted"
    )
    gemini = FakeProvider(
        actions=[
            {
                "summary": "Run tests",
                "actions": [
                    {"tool": "run_tests"}
                ],
            },
            {
                "summary": "Task completed",
                "actions": [],
            },
        ]
    )

    manager = ProviderManager()
    manager.register("openrouter", openrouter)
    manager.register("gemini", gemini)
    manager.set_default("openrouter")

    loop = CodingLoop(
        str(tmp_path),
        provider_manager=manager,
        approval_callback=lambda action, decision: True,
    )

    class FakeEngine:
        def execute(self, action):
            return {
                "success": True,
                "tool": "run_tests",
                "result": {
                    "success": True,
                    "return_code": 0,
                    "output": "1 passed",
                },
            }

    loop.engine = FakeEngine()

    result = loop.run("Run the test suite and finish the task.")

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["provider"] == "gemini"
    assert any(
        item["result"].get("status") == "provider_failover"
        for item in result["history"]
    )
