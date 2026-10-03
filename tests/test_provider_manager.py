from agent.providers.base import LLMProvider
from agent.providers.manager import ProviderManager


class TestProvider(LLMProvider):
    """Dummy provider for ProviderManager tests."""

    def generate(self, prompt: str) -> str:
        return "test response"


def test_register_provider():
    manager = ProviderManager()

    provider = TestProvider()

    manager.register(
        "test",
        provider,
    )

    assert "test" in manager.available()


def test_get_provider():
    manager = ProviderManager()

    provider = TestProvider()

    manager.register(
        "test",
        provider,
    )

    assert manager.get("test") is provider


def test_default_provider():
    manager = ProviderManager()

    provider = TestProvider()

    manager.register(
        "test",
        provider,
    )

    manager.set_default("test")

    assert manager.get() is provider


def test_manager_status():
    manager = ProviderManager()

    provider = TestProvider()

    manager.register(
        "test",
        provider,
    )

    status = manager.status()

    assert status["default_provider"] == "openrouter"
    assert "test" in status["available_providers"]
    assert status["provider_status"]["test"] == "registered"


def test_provider_available_status():
    manager = ProviderManager()

    provider = TestProvider()

    manager.register(
        "test",
        provider,
    )

    manager.set_status(
        "test",
        "available",
    )

    assert manager.get_status("test") == "available"


def test_provider_rate_limited_status():
    manager = ProviderManager()

    provider = TestProvider()

    manager.register(
        "test",
        provider,
    )

    manager.set_status(
        "test",
        "rate_limited",
    )

    assert manager.get_status("test") == "rate_limited"


def test_provider_error_status():
    manager = ProviderManager()

    provider = TestProvider()

    manager.register(
        "test",
        provider,
    )

    manager.set_status(
        "test",
        "error",
    )

    assert manager.get_status("test") == "error"


def test_provider_disabled_status():
    manager = ProviderManager()

    provider = TestProvider()

    manager.register(
        "test",
        provider,
    )

    manager.set_status(
        "test",
        "disabled",
    )

    assert manager.get_status("test") == "disabled"


def test_invalid_provider_status():
    manager = ProviderManager()

    provider = TestProvider()

    manager.register(
        "test",
        provider,
    )

    try:
        manager.set_status(
            "test",
            "invalid",
        )
        assert False
    except ValueError:
        assert True