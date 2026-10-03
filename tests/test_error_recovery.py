from agent.core.coding_loop import CodingLoop
from agent.core.error_recovery import ErrorRecoveryManager
from agent.providers.base import LLMProvider
from agent.providers.manager import ProviderManager


class RecoveryProvider(LLMProvider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, prompt: str) -> str:
        return "fake"

    def generate_actions(self, task: str) -> dict:
        self.calls += 1

        if len(self.responses) > 1:
            response = self.responses.pop(0)
        else:
            response = self.responses[0]

        if isinstance(response, Exception):
            raise response

        return response


def test_error_recovery_retries_transient_error():
    manager = ErrorRecoveryManager()

    assert manager.should_retry(
        "HTTP 503 service unavailable",
        0,
    ) is True

    assert manager.should_retry(
        "HTTP 503 service unavailable",
        1,
    ) is True

    assert manager.should_retry(
        "HTTP 503 service unavailable",
        2,
    ) is False


def test_error_recovery_rejects_auth_errors():
    manager = ErrorRecoveryManager()

    assert manager.should_retry(
        "401 invalid API key",
        0,
    ) is False

    assert manager.should_retry(
        "OPENROUTER_API_KEY is not set",
        0,
    ) is False


def test_error_recovery_delay_is_bounded():
    manager = ErrorRecoveryManager()

    assert manager.retry_delay(1) == 1
    assert manager.retry_delay(2) == 2
    assert manager.retry_delay(3) == 2


def test_coding_loop_retries_transient_provider_error(
    monkeypatch,
    tmp_path,
):
    provider = RecoveryProvider(
        [
            RuntimeError(
                "HTTP 503 service unavailable"
            ),
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
    manager.register(
        "openrouter",
        provider,
    )
    manager.set_default(
        "openrouter"
    )

    loop = CodingLoop(
        str(tmp_path),
        provider_manager=manager,
        approval_callback=lambda action, decision: True,
    )

    monkeypatch.setattr(
        "agent.core.error_recovery.time.sleep",
        lambda delay: None,
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

    result = loop.run(
        "Recover from a transient provider error."
    )

    assert result["success"] is True
    assert result["status"] == "completed"
    assert provider.calls == 3


def test_coding_loop_does_not_retry_auth_error(
    tmp_path,
):
    provider = RecoveryProvider(
        [
            RuntimeError(
                "401 invalid API key"
            ),
        ]
    )

    manager = ProviderManager()
    manager.register(
        "openrouter",
        provider,
    )
    manager.set_default(
        "openrouter"
    )

    loop = CodingLoop(
        str(tmp_path),
        provider_manager=manager,
        approval_callback=lambda action, decision: True,
    )

    result = loop.run(
        "Test authentication failure handling."
    )

    assert result["success"] is False
    assert result["status"] == "ai_error"
    assert provider.calls == 1


def test_coding_loop_does_not_retry_rate_limit(
    tmp_path,
):
    provider = RecoveryProvider(
        [
            RuntimeError(
                "OPENROUTER_RATE_LIMIT: quota exhausted"
            ),
        ]
    )

    manager = ProviderManager()
    manager.register(
        "openrouter",
        provider,
    )
    manager.set_default(
        "openrouter"
    )

    loop = CodingLoop(
        str(tmp_path),
        provider_manager=manager,
        approval_callback=lambda action, decision: True,
    )

    result = loop.run(
        "Test rate-limit handling."
    )

    assert result["success"] is False
    assert result["status"] == "rate_limit"
    assert provider.calls == 1
