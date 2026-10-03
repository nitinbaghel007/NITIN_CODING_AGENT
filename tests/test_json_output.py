"""Batch 6: --json stdout contract and stream separation.

Contract under test:

* ``--json`` -> stdout carries exactly one valid JSON document,
  with no non-JSON prefix or suffix,
* all diagnostics (banners, progress, approval notices, provider
  errors, logging) go to stderr and never touch stdout,
* the CLI error path still emits valid JSON,
* normal (non-JSON) mode keeps its existing human-readable output.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import main
from agent.config.logging_setup import reset_logging

REPO_ROOT = Path(__file__).resolve().parents[1]

SUCCESS_RESULT = {
    "success": True,
    "status": "completed",
    "steps": 1,
}

FAILURE_RESULT = {
    "success": False,
    "status": "max_steps_reached",
}


def print_agent_diagnostics() -> None:
    """Emulate every kind of print the real agent makes mid-run."""

    print("\n========================================")
    print(" NITIN CODING AGENT")
    print("========================================")
    print("\nTask:")
    print("noop")
    print("\n========== STEP 1/10 ==========")
    print("\n========================================")
    print(" APPROVAL REQUIRED")
    print("========================================")
    print("  DENIED (approval is never automatic)")
    print("\n[AI ERROR]")
    print("OPENROUTER_API_KEY is not set.")


def chatty_success(task, workspace, non_interactive=False):
    print_agent_diagnostics()
    return dict(SUCCESS_RESULT)


def chatty_failure(task, workspace, non_interactive=False):
    print_agent_diagnostics()
    return dict(FAILURE_RESULT)


def chatty_error(task, workspace, non_interactive=False):
    print_agent_diagnostics()
    raise RuntimeError("provider unavailable")


@pytest.fixture(autouse=True)
def _clean_logging():
    """Leave the global logger without stale stream handlers."""

    yield
    reset_logging()


def invoke(fake_run_task, *extra_args, **patch_kwargs):
    """Run the CLI with main.run_task replaced by ``fake_run_task``."""

    with patch(
        "main.run_task",
        side_effect=fake_run_task,
        **patch_kwargs,
    ) as run_task:
        code = main.main(
            [
                "run",
                "noop",
                "--workspace",
                "workspace",
                *extra_args,
            ]
        )

    return code, run_task


# -------------------------------------------------------------
# 1-3. stdout is exactly one valid JSON document
# -------------------------------------------------------------


def test_json_stdout_is_exactly_one_valid_json_document(capsys):
    code, _ = invoke(chatty_success, "--json")

    captured = capsys.readouterr()

    expected = json.dumps(
        SUCCESS_RESULT,
        indent=2,
        ensure_ascii=False,
    )

    assert code == 0
    # Whitespace-stripped stdout equals the serialized result, so
    # there is neither a prefix nor a suffix around the document.
    assert captured.out.strip() == expected
    assert json.loads(captured.out) == SUCCESS_RESULT


def test_json_stdout_has_no_non_json_prefix_or_suffix(capsys):
    invoke(chatty_success, "--json")

    captured = capsys.readouterr()

    # Nothing but whitespace may appear before the document...
    assert captured.out.split("{", 1)[0].strip() == ""

    # ...and nothing but whitespace after it.
    _, end = json.JSONDecoder().raw_decode(captured.out)
    assert captured.out[end:].strip() == ""


def test_json_progress_prints_do_not_reach_stdout(capsys):
    invoke(chatty_success, "--json")

    captured = capsys.readouterr()

    assert "NITIN CODING AGENT" not in captured.out
    assert "STEP 1/10" not in captured.out
    assert "APPROVAL REQUIRED" not in captured.out
    assert "OPENROUTER_API_KEY is not set." not in captured.out
    assert "[AI ERROR]" not in captured.out


# -------------------------------------------------------------
# 4-5. diagnostics are preserved on stderr, logging included
# -------------------------------------------------------------


def test_json_progress_prints_are_preserved_on_stderr(capsys):
    """Stream separation, not suppression: output still exists."""

    invoke(chatty_success, "--json")

    captured = capsys.readouterr()

    assert "NITIN CODING AGENT" in captured.err
    assert "STEP 1/10" in captured.err
    assert "APPROVAL REQUIRED" in captured.err
    assert "OPENROUTER_API_KEY is not set." in captured.err


def test_json_logging_does_not_contaminate_stdout(capsys):
    invoke(chatty_success, "--json")

    captured = capsys.readouterr()

    assert json.loads(captured.out) == SUCCESS_RESULT
    assert "run start" in captured.err
    assert "run finished" in captured.err
    assert "run start" not in captured.out
    assert "run finished" not in captured.out


# -------------------------------------------------------------
# 6-7. JSON error path
# -------------------------------------------------------------


def test_json_error_path_emits_valid_json_and_exit_one(capsys):
    code, _ = invoke(chatty_error, "--json")

    captured = capsys.readouterr()

    assert code == 1
    assert json.loads(captured.out) == {
        "success": False,
        "status": "cli_error",
        "error": "provider unavailable",
    }


def test_json_error_path_keeps_traceback_and_progress_off_stdout(
    capsys,
):
    code, _ = invoke(chatty_error, "--json")

    captured = capsys.readouterr()

    assert code == 1
    assert "Traceback" not in captured.out
    assert "NITIN CODING AGENT" not in captured.out
    # Diagnostics made before the failure survive on stderr, and
    # the failure itself is logged there too.
    assert "NITIN CODING AGENT" in captured.err
    assert "run failed" in captured.err
    assert "provider unavailable" in captured.err


# -------------------------------------------------------------
# 8-9. representative result paths
# -------------------------------------------------------------


def test_json_failure_result_still_emits_json_and_exit_one(capsys):
    code, _ = invoke(chatty_failure, "--json")

    captured = capsys.readouterr()

    assert code == 1
    assert json.loads(captured.out) == FAILURE_RESULT


def test_json_non_interactive_flag_passed_and_stdout_clean(capsys):
    code, run_task = invoke(
        chatty_success,
        "--json",
        "--non-interactive",
    )

    captured = capsys.readouterr()

    assert code == 0
    run_task.assert_called_once_with(
        task="noop",
        workspace="workspace",
        non_interactive=True,
    )
    assert json.loads(captured.out) == SUCCESS_RESULT


# -------------------------------------------------------------
# 10-11. normal mode keeps its existing behaviour
# -------------------------------------------------------------


def test_non_json_mode_output_is_preserved(capsys):
    code, _ = invoke(chatty_success)

    captured = capsys.readouterr()

    expected_json = json.dumps(
        SUCCESS_RESULT,
        indent=2,
        ensure_ascii=False,
    )

    assert code == 0
    assert "NITIN CODING AGENT" in captured.out
    assert "APPROVAL REQUIRED" in captured.out
    assert " FINAL RESULT" in captured.out
    assert expected_json in captured.out


def test_non_json_mode_error_output_is_preserved(capsys):
    code, _ = invoke(chatty_error)

    captured = capsys.readouterr()

    assert code == 1
    assert "[CLI ERROR]" in captured.out
    assert "provider unavailable" in captured.out


# -------------------------------------------------------------
# 12. end-to-end: the real CLI binary
# -------------------------------------------------------------


def test_real_cli_json_stdout_parses_and_diagnostics_hit_stderr(
    tmp_path,
):
    """Run the actual entry point with no API keys available.

    The provider raises "key is not set" before any network call,
    so the run is hermetic: it prints banners, an AI error, then
    the result document.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = ""
    env["GEMINI_API_KEY"] = ""

    proc = subprocess.run(
        [
            sys.executable,
            "main.py",
            "run",
            "noop",
            "--workspace",
            str(workspace),
            "--json",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
    )

    assert proc.returncode == 1

    parsed = json.loads(proc.stdout)
    assert parsed["success"] is False
    assert parsed["status"] == "ai_error"

    # Diagnostics were preserved, on stderr only.
    assert "NITIN CODING AGENT" in proc.stderr
    assert "run start" in proc.stderr
    assert "NITIN CODING AGENT" not in proc.stdout
