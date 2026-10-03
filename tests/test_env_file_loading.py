"""Focused tests for local .env loading (dependency + CLI wiring).

Hermetic rules for this module:
* no test reads the user's real .env file - every case builds its own
  temporary .env under tmp_path (or none at all)
* only obviously-fake sentinel values are used, never real credentials
* an autouse fixture restores os.environ exactly, so values injected by
  load_dotenv() during a test can never leak into other tests
"""

import os
import re
from pathlib import Path

import pytest

import main
from agent.config.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAKE_OPENROUTER = "fake-envfile-openrouter-key-not-real"
FAKE_GEMINI = "fake-envfile-gemini-key-not-real"
FAKE_MODEL = "fake-envfile-model-not-real"


@pytest.fixture(autouse=True)
def restore_environ():
    """Snapshot os.environ and restore it bit-for-bit afterwards."""

    before = dict(os.environ)
    yield
    for key in [k for k in os.environ if k not in before]:
        del os.environ[key]
    for key, value in before.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


def test_python_dotenv_is_declared_as_runtime_dependency():
    """Both dependency files pin python-dotenv consistently."""

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    )

    assert re.search(
        r'"python-dotenv==\d+\.\d+\.\d+"', pyproject
    ), "pyproject.toml must pin python-dotenv"
    assert re.search(
        r"^python-dotenv==\d+\.\d+\.\d+$", requirements, re.MULTILINE
    ), "requirements.txt must pin python-dotenv"

    pins = set(re.findall(r"python-dotenv==(\d+\.\d+\.\d+)", pyproject))
    pins |= set(
        re.findall(r"python-dotenv==(\d+\.\d+\.\d+)", requirements)
    )
    assert len(pins) == 1, f"version mismatch between files: {pins}"


def test_load_dotenv_makes_env_file_values_available_to_settings(
    tmp_path,
):
    """Values written to a .env file reach Settings.from_env()."""

    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        f"OPENROUTER_API_KEY={FAKE_OPENROUTER}\n"
        f"GEMINI_API_KEY={FAKE_GEMINI}\n"
        f"GEMINI_MODEL={FAKE_MODEL}\n",
        encoding="utf-8",
    )

    # Explicit path: the test never touches the project .env.
    from dotenv import load_dotenv

    loaded = load_dotenv(dotenv_path=str(dotenv_file))
    assert loaded is True

    settings = Settings.from_env()
    assert settings.openrouter_api_key == FAKE_OPENROUTER
    assert settings.gemini_api_key == FAKE_GEMINI
    assert settings.gemini_model == FAKE_MODEL


def test_load_dotenv_never_overrides_existing_os_environment(
    tmp_path, monkeypatch
):
    """OS environment wins: .env must not overwrite what is set."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "os-already-set-value")
    monkeypatch.setenv("GEMINI_MODEL", "os-model-already-set")

    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        "OPENROUTER_API_KEY=different-value-from-dotenv\n"
        "GEMINI_MODEL=different-model-from-dotenv\n"
        "GEMINI_API_KEY=only-in-dotenv-value\n",
        encoding="utf-8",
    )

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=str(dotenv_file))  # default: no override

    settings = Settings.from_env()
    # Already-set OS values are preserved exactly ...
    assert settings.openrouter_api_key == "os-already-set-value"
    assert settings.gemini_model == "os-model-already-set"
    # ... while previously-unset variables are filled in.
    assert settings.gemini_api_key == "only-in-dotenv-value"


def test_load_dotenv_with_missing_file_is_a_noop(tmp_path):
    """A missing .env file changes nothing (the common case)."""

    from dotenv import load_dotenv

    before = dict(os.environ)
    loaded = load_dotenv(
        dotenv_path=str(tmp_path / "does-not-exist.env")
    )
    assert loaded is False
    assert dict(os.environ) == before


def test_settings_from_env_still_reads_environment_only(tmp_path):
    """Settings itself never reads files - only the given mapping.

    Existing behaviour guard: even when a .env file exists on disk,
    Settings.from_env() must consult exactly the mapping it is given
    and nothing else.
    """

    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        f"OPENROUTER_API_KEY={FAKE_OPENROUTER}\n",
        encoding="utf-8",
    )
    assert dotenv_file.exists()

    settings = Settings.from_env(environ={})
    assert settings.openrouter_api_key is None
    assert settings.gemini_api_key is None
    assert settings.gemini_model == Settings.from_env(
        environ={}
    ).gemini_model
    assert not settings.has_openrouter_key
    assert not settings.has_gemini_key


def test_main_loads_dotenv_before_reading_settings(
    tmp_path, monkeypatch
):
    """CLI entry point wires load_dotenv() in ahead of from_env()."""

    calls = []
    monkeypatch.setattr(
        main, "load_dotenv", lambda *a, **k: calls.append("load_dotenv")
    )

    real_settings = main.Settings
    original_from_env = real_settings.from_env.__func__

    def spying_from_env(cls, environ=None):
        calls.append("from_env")
        return original_from_env(cls, environ)

    monkeypatch.setattr(
        real_settings, "from_env", classmethod(spying_from_env)
    )

    # Deterministic keyless run: no real provider call can happen.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LOG_FILE", raising=False)

    exit_code = main.main(
        [
            "run",
            "noop",
            "--workspace",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ]
    )

    assert calls[0] == "load_dotenv", (
        "load_dotenv must run before any Settings.from_env() call"
    )
    assert "from_env" in calls
    # Keyless offline run ends as ai_error with exit code 1.
    assert exit_code == 1


def test_env_example_has_required_empty_placeholder_entries():
    """.env.example carries the three required entries, values empty."""

    text = (PROJECT_ROOT / ".env.example").read_text(
        encoding="utf-8"
    )
    lines = [line.strip() for line in text.splitlines()]

    for entry in (
        "OPENROUTER_API_KEY=",
        "GEMINI_API_KEY=",
        "GEMINI_MODEL=",
    ):
        assert entry in lines, f"missing exact entry: {entry}"

    # No line may assign a non-empty value: placeholders must stay
    # empty so a copied-but-unfilled .env cannot carry a credential.
    assigned = [
        line
        for line in lines
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\S", line)
    ]
    assert assigned == [], f"non-empty values found: {assigned}"
