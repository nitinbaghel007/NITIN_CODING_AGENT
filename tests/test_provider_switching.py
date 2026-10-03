from agent.providers.base import LLMProvider
from agent.providers.manager import ProviderManager


class DummyProvider(LLMProvider):
    """Local provider for switching tests."""

    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str) -> str:
        return f"{self.name}: {prompt}"

    def generate_actions(self, task: str) -> dict:
        return {
            "summary": f"{self.name} response",
            "actions": [],
        }


def create_manager():
    manager = ProviderManager()

    openrouter = DummyProvider(
        "openrouter"
    )

    gemini = DummyProvider(
        "gemini"
    )

    manager.register(
        "openrouter",
        openrouter,
    )

    manager.register(
        "gemini",
        gemini,
    )

    return manager


def test_both_providers_are_registered():
    manager = create_manager()

    assert manager.available() == [
        "gemini",
        "openrouter",
    ]


def test_openrouter_is_default_provider():
    manager = create_manager()

    assert (
        manager.default_provider
        == "openrouter"
    )


def test_get_openrouter_provider():
    manager = create_manager()

    provider = manager.get(
        "openrouter"
    )

    assert provider.name == "openrouter"


def test_switch_to_gemini():
    manager = create_manager()

    manager.set_default(
        "gemini"
    )

    assert (
        manager.default_provider
        == "gemini"
    )

    provider = manager.get()

    assert provider.name == "gemini"


def test_switch_back_to_openrouter():
    manager = create_manager()

    manager.set_default(
        "gemini"
    )

    manager.set_default(
        "openrouter"
    )

    assert (
        manager.default_provider
        == "openrouter"
    )

    provider = manager.get()

    assert provider.name == "openrouter"


def test_explicit_provider_selection():
    manager = create_manager()

    manager.set_default(
        "openrouter"
    )

    provider = manager.get(
        "gemini"
    )

    assert provider.name == "gemini"


def test_provider_status_is_independent():
    manager = create_manager()

    manager.set_status(
        "openrouter",
        "rate_limited",
    )

    manager.set_status(
        "gemini",
        "available",
    )

    assert (
        manager.get_status(
            "openrouter"
        )
        == "rate_limited"
    )

    assert (
        manager.get_status(
            "gemini"
        )
        == "available"
    )


def test_provider_response_after_switch():
    manager = create_manager()

    manager.set_default(
        "gemini"
    )

    provider = manager.get()

    result = provider.generate_actions(
        "test task"
    )

    assert (
        result["summary"]
        == "gemini response"
    )

    assert result["actions"] == []