"""Tests for the central Settings object (Batch 3)."""

import dataclasses
import os

import pytest

from agent.config.settings import (
    DEFAULT_GEMINI_MODEL,
    ENV_GEMINI_API_KEY,
    ENV_GEMINI_MODEL,
    ENV_OPENROUTER_API_KEY,
    Settings,
)
from agent.providers.gemini import GeminiProvider
from agent.providers.openrouter import OpenRouterProvider

# Sample secrets. They never touch the network and are never real.
OPENROUTER_SAMPLE = "sk-or-sample-key-000111"
GEMINI_SAMPLE = "AIza-sample-key-222333"


def clear_env(monkeypatch):
    """Remove every variable Settings reads."""

    for name in (
        ENV_OPENROUTER_API_KEY,
        ENV_GEMINI_API_KEY,
        ENV_GEMINI_MODEL,
    ):
        monkeypatch.delenv(name, raising=False)


# -------------------------------------------------------------
# 1-3. Reading the three environment variables
# -------------------------------------------------------------


def test_settings_reads_openrouter_api_key(monkeypatch):
    monkeypatch.setenv(
        ENV_OPENROUTER_API_KEY,
        OPENROUTER_SAMPLE,
    )

    settings = Settings.from_env()

    assert settings.openrouter_api_key == OPENROUTER_SAMPLE
    assert settings.has_openrouter_key is True


def test_settings_reads_gemini_api_key(monkeypatch):
    monkeypatch.setenv(
        ENV_GEMINI_API_KEY,
        GEMINI_SAMPLE,
    )

    settings = Settings.from_env()

    assert settings.gemini_api_key == GEMINI_SAMPLE
    assert settings.has_gemini_key is True


def test_settings_reads_gemini_model(monkeypatch):
    monkeypatch.setenv(ENV_GEMINI_MODEL, "custom-model-x")

    settings = Settings.from_env()

    assert settings.gemini_model == "custom-model-x"


def test_settings_from_env_accepts_explicit_mapping():
    """Tests can inject a plain dict instead of touching os.environ."""

    settings = Settings.from_env(
        {
            ENV_OPENROUTER_API_KEY: "or-value",
            ENV_GEMINI_API_KEY: "gem-value",
            ENV_GEMINI_MODEL: "model-value",
        }
    )

    assert settings.openrouter_api_key == "or-value"
    assert settings.gemini_api_key == "gem-value"
    assert settings.gemini_model == "model-value"


# -------------------------------------------------------------
# 4. Missing variables behave like the previous os.getenv code
# -------------------------------------------------------------


def test_settings_missing_env_vars_are_none(monkeypatch):
    clear_env(monkeypatch)

    settings = Settings.from_env()

    assert settings.openrouter_api_key is None
    assert settings.gemini_api_key is None
    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.has_openrouter_key is False
    assert settings.has_gemini_key is False


def test_settings_defaults_to_original_gemini_model(monkeypatch):
    clear_env(monkeypatch)

    assert (
        Settings.from_env().gemini_model
        == "gemini-3.8-flash"
    )


def test_settings_empty_key_matches_previous_getenv_behaviour(
    monkeypatch,
):
    """An empty key must stay falsy so providers still raise."""

    monkeypatch.setenv(ENV_OPENROUTER_API_KEY, "")
    monkeypatch.setenv(ENV_GEMINI_API_KEY, "")

    settings = Settings.from_env()

    assert settings.openrouter_api_key == ""
    assert settings.gemini_api_key == ""
    assert settings.has_openrouter_key is False
    assert settings.has_gemini_key is False


def test_settings_empty_gemini_model_keeps_previous_semantics(
    monkeypatch,
):
    """os.getenv(key, default) only applies when the variable is absent."""

    monkeypatch.setenv(ENV_GEMINI_MODEL, "")

    assert Settings.from_env().gemini_model == ""


def test_settings_does_not_modify_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_OPENROUTER_API_KEY, OPENROUTER_SAMPLE)
    before = dict(os.environ)

    Settings.from_env()

    assert dict(os.environ) == before


# -------------------------------------------------------------
# 4b. Providers keep their original error and model behaviour
# -------------------------------------------------------------


def test_openrouter_provider_raises_when_key_missing(monkeypatch):
    clear_env(monkeypatch)

    with pytest.raises(RuntimeError) as exc:
        OpenRouterProvider()

    assert str(exc.value) == "OPENROUTER_API_KEY is not set."


def test_gemini_provider_raises_when_key_missing(monkeypatch):
    clear_env(monkeypatch)

    with pytest.raises(RuntimeError) as exc:
        GeminiProvider()

    assert str(exc.value) == "GEMINI_API_KEY is not set."


def test_openrouter_provider_reads_key_via_settings(monkeypatch):
    monkeypatch.setenv(
        ENV_OPENROUTER_API_KEY,
        OPENROUTER_SAMPLE,
    )

    provider = OpenRouterProvider()

    assert provider.api_key == OPENROUTER_SAMPLE
    assert provider.model == "openrouter/free"


def test_gemini_provider_reads_key_and_default_model(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv(ENV_GEMINI_API_KEY, GEMINI_SAMPLE)

    provider = GeminiProvider()

    assert provider.api_key == GEMINI_SAMPLE
    assert provider.model == "gemini-3.8-flash"


def test_gemini_provider_env_model_is_used(monkeypatch):
    monkeypatch.setenv(ENV_GEMINI_API_KEY, GEMINI_SAMPLE)
    monkeypatch.setenv(ENV_GEMINI_MODEL, "env-model-y")

    assert GeminiProvider().model == "env-model-y"


def test_gemini_provider_explicit_argument_still_wins(monkeypatch):
    monkeypatch.setenv(ENV_GEMINI_API_KEY, GEMINI_SAMPLE)
    monkeypatch.setenv(ENV_GEMINI_MODEL, "env-model-y")

    provider = GeminiProvider(model="argument-model")

    assert provider.model == "argument-model"


# -------------------------------------------------------------
# 5. Secrets are never printed or logged by Settings
# -------------------------------------------------------------


def test_settings_repr_hides_api_keys():
    settings = Settings(
        openrouter_api_key=OPENROUTER_SAMPLE,
        gemini_api_key=GEMINI_SAMPLE,
    )

    text = repr(settings)

    assert OPENROUTER_SAMPLE not in text
    assert GEMINI_SAMPLE not in text
    assert "***" in text


def test_settings_str_hides_api_keys():
    settings = Settings(
        openrouter_api_key=OPENROUTER_SAMPLE,
        gemini_api_key=GEMINI_SAMPLE,
    )

    text = str(settings)

    assert OPENROUTER_SAMPLE not in text
    assert GEMINI_SAMPLE not in text


def test_settings_never_prints_api_keys(capsys):
    settings = Settings(
        openrouter_api_key=OPENROUTER_SAMPLE,
        gemini_api_key=GEMINI_SAMPLE,
    )

    print(settings)
    print(f"{settings!s}")
    print(format(settings))
    print(settings.masked())

    out = capsys.readouterr().out

    assert OPENROUTER_SAMPLE not in out
    assert GEMINI_SAMPLE not in out
    assert OPENROUTER_SAMPLE not in repr(settings.masked())
    assert GEMINI_SAMPLE not in repr(settings.masked())


def test_settings_from_env_prints_nothing(capsys, monkeypatch):
    monkeypatch.setenv(
        ENV_OPENROUTER_API_KEY,
        OPENROUTER_SAMPLE,
    )
    monkeypatch.setenv(ENV_GEMINI_API_KEY, GEMINI_SAMPLE)

    Settings.from_env()

    captured = capsys.readouterr().out

    assert captured == ""
    assert OPENROUTER_SAMPLE not in captured
    assert GEMINI_SAMPLE not in captured


def test_settings_is_immutable():
    settings = Settings()

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.gemini_model = "changed"
