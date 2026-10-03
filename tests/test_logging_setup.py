"""Batch 6: centralized logging setup and configuration."""

import logging
import sys
from unittest.mock import patch

import pytest

import main
from agent.config.logging_setup import (
    LOGGER_NAME,
    configure_logging,
    reset_logging,
)
from agent.config.settings import (
    ENV_LOG_FILE,
    ENV_LOG_LEVEL,
    Settings,
)

AGENT_LOGGER = logging.getLogger(LOGGER_NAME)
PROBE = logging.getLogger("agent.tests.probe")


@pytest.fixture(autouse=True)
def _clean_logging():
    """Start and end every test with no leftover handlers."""

    reset_logging()
    yield
    reset_logging()


# -------------------------------------------------------------
# Handler installation and idempotency
# -------------------------------------------------------------


def test_configure_logging_installs_one_stderr_handler():
    configure_logging(Settings())

    assert len(AGENT_LOGGER.handlers) == 1

    console = AGENT_LOGGER.handlers[0]

    assert isinstance(console, logging.StreamHandler)
    assert console.stream is sys.stderr
    assert console.stream is not sys.stdout


def test_console_handler_never_targets_stdout():
    configure_logging(Settings(log_level="DEBUG"))

    for handler in AGENT_LOGGER.handlers:
        stream = getattr(handler, "stream", None)

        if stream is not None:
            assert stream is not sys.stdout
            assert stream is sys.stderr


def test_repeated_configuration_does_not_stack_handlers():
    configure_logging(Settings())
    configure_logging(Settings())
    configure_logging(Settings())

    assert len(AGENT_LOGGER.handlers) == 1


def test_configure_logging_returns_central_logger():
    logger = configure_logging(Settings(log_level="DEBUG"))

    assert logger is AGENT_LOGGER
    assert logger.level == logging.DEBUG
    assert logger.propagate is False


def test_reset_logging_cleans_up():
    configure_logging(Settings())
    assert AGENT_LOGGER.handlers

    reset_logging()

    assert AGENT_LOGGER.handlers == []
    assert AGENT_LOGGER.propagate is True


# -------------------------------------------------------------
# Destinations and levels
# -------------------------------------------------------------


def test_info_diagnostic_reaches_stderr_not_stdout(capsys):
    configure_logging(Settings())

    PROBE.info("stderr-destination-marker")

    captured = capsys.readouterr()

    assert "stderr-destination-marker" in captured.err
    assert "stderr-destination-marker" not in captured.out


def test_single_record_produces_single_line(capsys):
    configure_logging(Settings())

    PROBE.warning("no-duplicate-marker")

    captured = capsys.readouterr()

    lines = [
        line
        for line in captured.err.splitlines()
        if "no-duplicate-marker" in line
    ]

    assert len(lines) == 1
    assert "WARNING" in lines[0]


def test_info_level_hides_debug_records(capsys):
    configure_logging(Settings(log_level="INFO"))

    PROBE.debug("hidden-debug-marker")
    PROBE.info("visible-info-marker")

    captured = capsys.readouterr()

    assert "hidden-debug-marker" not in captured.err
    assert "visible-info-marker" in captured.err


def test_debug_level_emits_debug_records(capsys):
    configure_logging(Settings(log_level="DEBUG"))

    PROBE.debug("visible-debug-marker")

    captured = capsys.readouterr()

    assert "visible-debug-marker" in captured.err
    assert "DEBUG" in captured.err


def test_unknown_level_falls_back_to_info(capsys):
    configure_logging(Settings(log_level="LOUD"))

    PROBE.debug("unknown-level-hidden")
    PROBE.info("unknown-level-visible")

    captured = capsys.readouterr()

    assert "Unknown log level" in captured.err
    assert "unknown-level-hidden" not in captured.err
    assert "unknown-level-visible" in captured.err


def test_child_logger_propagates_to_central_handlers(capsys):
    configure_logging(Settings())

    logging.getLogger("agent.core.sample").error(
        "child-logger-marker"
    )

    captured = capsys.readouterr()

    lines = [
        line
        for line in captured.err.splitlines()
        if "child-logger-marker" in line
    ]

    assert len(lines) == 1
    assert "ERROR" in lines[0]


# -------------------------------------------------------------
# File destination
# -------------------------------------------------------------


def test_log_file_receives_records(tmp_path):
    path = tmp_path / "agent.log"
    configure_logging(Settings(log_file=str(path)))

    PROBE.info("file-destination-marker")

    content = path.read_text(encoding="utf-8")

    assert "file-destination-marker" in content
    assert "INFO" in content


def test_file_and_console_each_get_one_copy(capsys, tmp_path):
    path = tmp_path / "agent.log"
    configure_logging(Settings(log_file=str(path)))

    PROBE.info("both-destinations-marker")

    captured = capsys.readouterr()
    content = path.read_text(encoding="utf-8")

    assert (
        captured.err.count("both-destinations-marker") == 1
    )
    assert content.count("both-destinations-marker") == 1


def test_unwritable_log_file_degrades_to_console(
    capsys,
    tmp_path,
):
    # Parent directory does not exist -> FileHandler raises.
    bad_path = tmp_path / "missing" / "agent.log"
    configure_logging(Settings(log_file=str(bad_path)))

    PROBE.info("console-still-works-marker")

    captured = capsys.readouterr()

    assert "log file unavailable" in captured.err
    assert "console-still-works-marker" in captured.err
    # Console-only: the failed file handler was not added.
    assert len(AGENT_LOGGER.handlers) == 1


# -------------------------------------------------------------
# Secret safety
# -------------------------------------------------------------


def test_settings_secrets_never_appear_in_logs(capsys, tmp_path):
    openrouter_key = "sk-or-vsecret-TESTMASKVALUE00000000"
    gemini_key = "AIzaSyTESTMASKVALUE0000000000000000"

    settings = Settings(
        openrouter_api_key=openrouter_key,
        gemini_api_key=gemini_key,
        log_file=str(tmp_path / "secrets.log"),
    )
    configure_logging(settings)

    PROBE.info("config snapshot: %s", settings)
    PROBE.info("masked view: %s", settings.masked())

    captured = capsys.readouterr()
    file_content = (tmp_path / "secrets.log").read_text(
        encoding="utf-8"
    )

    for destination in (captured.err, file_content):
        assert openrouter_key not in destination
        assert gemini_key not in destination

    # The keys were replaced by the mask, not merely omitted.
    assert "***" in captured.err


def test_main_does_not_log_task_text(capsys):
    """Full prompts stay out of the log (Batch 6 rule)."""

    task_marker = "super-secret-task-text-xyz"

    with patch(
        "main.run_task",
        return_value={"success": True, "status": "completed"},
    ):
        main.main(
            [
                "run",
                task_marker,
                "--workspace",
                "workspace",
                "--json",
            ]
        )

    captured = capsys.readouterr()

    assert task_marker not in captured.err
    assert "run start" in captured.err


# -------------------------------------------------------------
# CLI integration: logging initialized once, never duplicated
# -------------------------------------------------------------


def test_repeated_cli_runs_emit_each_log_line_once(capsys):
    with patch(
        "main.run_task",
        return_value={"success": True, "status": "completed"},
    ):
        main.main(
            ["run", "t", "--workspace", "workspace", "--json"]
        )
        main.main(
            ["run", "t", "--workspace", "workspace", "--json"]
        )

    captured = capsys.readouterr()

    assert captured.err.count("run start") == 2
    assert captured.err.count("run finished") == 2
    # Each run replaced the previous handlers instead of adding
    # a second console handler.
    assert len(AGENT_LOGGER.handlers) == 1


# -------------------------------------------------------------
# Settings: log configuration fields
# -------------------------------------------------------------


def test_settings_logging_defaults():
    settings = Settings()

    assert settings.log_level == "INFO"
    assert settings.log_file is None


def test_settings_reads_log_env_vars():
    settings = Settings.from_env(
        {
            ENV_LOG_LEVEL: "debug",
            ENV_LOG_FILE: "logs/agent.log",
        }
    )

    assert settings.log_level == "debug"
    assert settings.log_file == "logs/agent.log"


def test_settings_empty_log_env_falls_back_to_defaults():
    settings = Settings.from_env(
        {
            ENV_LOG_LEVEL: "",
            ENV_LOG_FILE: "",
        }
    )

    assert settings.log_level == "INFO"
    assert settings.log_file is None
