"""Batch 10: integration / end-to-end tests.

Every test drives the real application stack in-process through
``main.main()``: CLI parsing, Settings, logging, CodingLoop,
provider selection and failover, response parsing, ActionEngine,
safety, approval, tool execution, diagnosis and recovery - wired
exactly as the shipped entry point wires them.

Hermetic guarantees:

* providers are scripted fakes (no network, no API keys),
* the subprocess-backed tool layers are monkeypatched (no agent
  tool ever starts a process; the single deliberate ``--help``
  smoke invocation only parses arguments),
* every workspace is a pytest ``tmp_path``,
* the recovery sleep is suppressed.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import main
from agent.providers.base import LLMProvider

REPO_ROOT = Path(__file__).resolve().parent.parent
DECODER = json.JSONDecoder()

# -------------------------------------------------------------
# Scripted plans and suite results
# -------------------------------------------------------------

LIST_FILES_PLAN = {
    "summary": "Inspect the workspace",
    "actions": [{"tool": "list_files"}],
}

RUN_TESTS_PLAN = {
    "summary": "Run the test suite",
    "actions": [{"tool": "run_tests"}],
}

COMPLETE_PLAN = {
    "summary": "Task completed",
    "actions": [],
}

APPROVAL_PLAN = {
    "summary": "Install the requested dependency",
    "actions": [
        {
            "tool": "run_command",
            "command": "pip install requests",
        }
    ],
}

HAPPY_SCRIPT = [
    LIST_FILES_PLAN,
    RUN_TESTS_PLAN,
    COMPLETE_PLAN,
]

PASSING_SUITE = {
    "return_code": 0,
    "stdout": "2 passed",
    "stderr": "",
    "success": True,
}

FAILING_SUITE = {
    "return_code": 1,
    "stdout": "1 failed",
    "stderr": "SyntaxError: invalid syntax",
    "success": False,
}


# -------------------------------------------------------------
# Hermetic environment
# -------------------------------------------------------------


@pytest.fixture(autouse=True)
def hermetic_environment(monkeypatch):
    """No API keys, no log file, no real sleeping, ever."""

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LOG_FILE", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    monkeypatch.setattr(
        "agent.core.error_recovery.time.sleep",
        lambda delay: None,
    )


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------


def scripted_provider(plans, errors=()):
    """Build a fresh zero-argument provider class.

    ``errors`` are raised one per call before any plan is served;
    the last plan repeats. The class carries its call state on
    ``ScriptedProvider.state`` for assertions.
    """

    state = {
        "plans": [dict(plan) for plan in plans],
        "errors": list(errors),
        "calls": 0,
    }

    class ScriptedProvider(LLMProvider):
        def generate(self, prompt: str) -> str:
            return "scripted response"

        def generate_actions(self, task: str) -> dict:
            state["calls"] += 1

            if state["errors"]:
                raise state["errors"].pop(0)

            if len(state["plans"]) > 1:
                return state["plans"].pop(0)

            return state["plans"][0]

    ScriptedProvider.state = state
    return ScriptedProvider


def patch_openrouter(monkeypatch, provider_class):
    monkeypatch.setattr(
        "agent.core.coding_loop.OpenRouterProvider",
        provider_class,
    )


def patch_gemini(monkeypatch, provider_class):
    monkeypatch.setattr(
        "agent.core.coding_loop.GeminiProvider",
        provider_class,
    )


def patch_suite(monkeypatch, results):
    """Force ``run_tests`` results; the last entry repeats."""

    queue = [dict(entry) for entry in results]

    def run_pytest(self, timeout=120):
        if len(queue) > 1:
            return dict(queue.pop(0))

        return dict(queue[0])

    monkeypatch.setattr(
        "agent.tools.test_tool.TestTool.run_pytest",
        run_pytest,
    )


def spy_terminal(monkeypatch):
    """Record every command that would reach the shell."""

    executed = []

    def run(self, command, **kwargs):
        executed.append(command)
        return {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "success": True,
        }

    monkeypatch.setattr(
        "agent.tools.terminal_tool.TerminalTool.run",
        run,
    )
    return executed


def input_returning(monkeypatch, value):
    calls = []

    def fake_input(prompt=""):
        calls.append(prompt)
        return value

    monkeypatch.setattr("builtins.input", fake_input)
    return calls


def input_raising(monkeypatch, error):
    calls = []

    def fake_input(prompt=""):
        calls.append(prompt)
        raise error

    monkeypatch.setattr("builtins.input", fake_input)
    return calls


def invoke_cli(workspace, *flags, task="Make the tests pass."):
    args = ["run", task, "--workspace", str(workspace)]
    args.extend(flags)
    return main.main(args)


def parse_single_json(stdout: str) -> dict:
    """Parse stdout that must contain exactly one JSON document."""

    text = stdout.strip()
    parsed, end = DECODER.raw_decode(text)
    assert text[end:].strip() == ""
    return parsed


def history_with_status(result, status):
    return [
        entry["result"]
        for entry in result.get("history", [])
        if isinstance(entry.get("result"), dict)
        and entry["result"].get("status") == status
    ]


def history_tool_results(result, tool):
    return [
        entry["result"]
        for entry in result.get("history", [])
        if isinstance(entry.get("result"), dict)
        and entry["result"].get("tool") == tool
    ]


# =============================================================
# Phase 2: happy-path end-to-end
# =============================================================


def test_happy_path_workflow_completes_end_to_end(
    tmp_path, monkeypatch, capsys
):
    """CLI -> loop -> fake provider -> SAFE tool -> tests ->
    completion, with results propagated through history."""

    patch_openrouter(
        monkeypatch, scripted_provider(HAPPY_SCRIPT)
    )
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")

    captured = capsys.readouterr()
    result = parse_single_json(captured.out)

    assert code == 0
    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["context"]["tests_verified"] is True

    # Multiple loop steps, as the architecture requires.
    assert result["steps"] == 3

    # The SAFE action really ran against the real workspace.
    listings = [
        entry["result"]
        for entry in result["history"]
        if isinstance(entry.get("result"), dict)
        and "files" in entry["result"]
    ]
    assert len(listings) == 1
    assert listings[0]["success"] is True
    assert listings[0]["files"] == []
    assert (
        Path(listings[0]["workspace"]) == tmp_path.resolve()
    )

    # The tool result reached the caller through history.
    test_runs = history_tool_results(result, "run_tests")
    assert len(test_runs) == 1
    assert test_runs[0]["success"] is True
    assert test_runs[0]["result"]["stdout"] == "2 passed"

    # JSON mode separation: pure stdout, diagnostics on stderr.
    assert "NITIN CODING AGENT" not in captured.out
    assert "NITIN CODING AGENT" in captured.err
    assert "run start" in captured.err


def test_happy_path_plain_mode_reports_final_result(
    tmp_path, monkeypatch, capsys
):
    patch_openrouter(
        monkeypatch, scripted_provider(HAPPY_SCRIPT)
    )
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path)

    captured = capsys.readouterr()

    assert code == 0
    assert "NITIN CODING AGENT" in captured.out
    assert "FINAL RESULT" in captured.out
    assert '"status": "completed"' in captured.out

    # Logging stays on stderr in plain mode too.
    assert "run start" in captured.err
    assert "run start" not in captured.out


# =============================================================
# Phase 3: approval end-to-end
# =============================================================


def approval_script():
    return [
        dict(APPROVAL_PLAN),
        dict(RUN_TESTS_PLAN),
        dict(COMPLETE_PLAN),
    ]


def test_approval_accepted_executes_exactly_once(
    tmp_path, monkeypatch, capsys
):
    patch_openrouter(
        monkeypatch, scripted_provider(approval_script())
    )
    patch_suite(monkeypatch, [PASSING_SUITE])
    executed = spy_terminal(monkeypatch)
    prompts = input_returning(monkeypatch, "yes")

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"

    # Approved action executed exactly once, after one prompt.
    assert executed == ["pip install requests"]
    assert len(prompts) == 1

    runs = history_tool_results(result, "run_command")
    assert len(runs) == 1
    assert runs[0]["success"] is True
    assert runs[0]["command"] == "pip install requests"


def test_approval_denied_executes_zero_times(
    tmp_path, monkeypatch, capsys
):
    patch_openrouter(
        monkeypatch, scripted_provider(approval_script())
    )
    patch_suite(monkeypatch, [PASSING_SUITE])
    executed = spy_terminal(monkeypatch)
    prompts = input_returning(monkeypatch, "no")

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    assert executed == []
    assert len(prompts) == 1

    denials = history_with_status(result, "approval_required")
    assert len(denials) == 1
    assert denials[0]["success"] is False


def test_approval_eof_denies_and_run_continues(
    tmp_path, monkeypatch, capsys
):
    patch_openrouter(
        monkeypatch, scripted_provider(approval_script())
    )
    patch_suite(monkeypatch, [PASSING_SUITE])
    executed = spy_terminal(monkeypatch)
    prompts = input_raising(
        monkeypatch, EOFError("EOF when reading a line")
    )

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    assert len(prompts) == 1
    assert executed == []
    assert (
        len(history_with_status(result, "approval_required"))
        == 1
    )


def test_approval_keyboard_interrupt_denies_and_run_continues(
    tmp_path, monkeypatch, capsys
):
    patch_openrouter(
        monkeypatch, scripted_provider(approval_script())
    )
    patch_suite(monkeypatch, [PASSING_SUITE])
    executed = spy_terminal(monkeypatch)
    prompts = input_raising(monkeypatch, KeyboardInterrupt())

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    assert len(prompts) == 1
    assert executed == []
    assert (
        len(history_with_status(result, "approval_required"))
        == 1
    )


def test_approval_callback_failure_denies_without_crash(
    tmp_path, monkeypatch, capsys
):
    def broken_request(self, action, decision):
        raise RuntimeError("approval subsystem failed")

    monkeypatch.setattr(
        "agent.core.approval.ApprovalManager.request",
        broken_request,
    )

    patch_openrouter(
        monkeypatch, scripted_provider(approval_script())
    )
    patch_suite(monkeypatch, [PASSING_SUITE])
    executed = spy_terminal(monkeypatch)

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    assert executed == []
    assert (
        len(history_with_status(result, "approval_required"))
        == 1
    )


# =============================================================
# Phase 4: blocked / unsafe end-to-end
# =============================================================

UNSAFE_ACTIONS = [
    pytest.param(
        {"tool": "run_command", "command": "del important.txt"},
        None,
        id="blocked-command",
    ),
    pytest.param(
        {
            "tool": "run_command",
            "command": "git status && del important.txt",
        },
        None,
        id="shell-chain",
    ),
    pytest.param(
        {"tool": "format_disk"},
        "Unknown tool is blocked: format_disk",
        id="unknown-tool",
    ),
    pytest.param(
        {"tool": 123},
        "Action is missing a valid tool.",
        id="malformed-tool",
    ),
    pytest.param(
        {"tool": "run_command"},
        "run_command requires a non-empty command.",
        id="missing-command",
    ),
    pytest.param(
        {},
        "Action is missing a valid tool.",
        id="empty-action",
    ),
]


@pytest.mark.parametrize("action,error_fragment", UNSAFE_ACTIONS)
def test_unsafe_action_never_executes_end_to_end(
    action, error_fragment, tmp_path, monkeypatch, capsys
):
    patch_openrouter(
        monkeypatch,
        scripted_provider(
            [
                {
                    "summary": "Try an unsafe action",
                    "actions": [action],
                },
                dict(RUN_TESTS_PLAN),
                dict(COMPLETE_PLAN),
            ]
        ),
    )
    patch_suite(monkeypatch, [PASSING_SUITE])
    executed = spy_terminal(monkeypatch)
    prompts = input_raising(
        monkeypatch,
        AssertionError("approval must not be requested"),
    )

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    # The loop stays stable and reaches a controlled completion.
    assert code == 0
    assert result["status"] == "completed"

    blocked = history_with_status(result, "blocked")
    assert len(blocked) == 1
    assert blocked[0]["success"] is False
    assert blocked[0]["safety_level"] == "blocked"
    assert blocked[0]["error"]
    if error_fragment is not None:
        assert error_fragment in blocked[0]["error"]

    # Blocked actions never execute and never prompt for approval.
    assert executed == []
    assert prompts == []


def test_workspace_escape_leaves_filesystem_untouched(
    tmp_path, monkeypatch, capsys
):
    patch_openrouter(
        monkeypatch,
        scripted_provider(
            [
                {
                    "summary": "Escape the workspace",
                    "actions": [
                        {
                            "tool": "write_file",
                            "path": "../evil.py",
                            "content": "bad",
                        }
                    ],
                },
                dict(RUN_TESTS_PLAN),
                dict(COMPLETE_PLAN),
            ]
        ),
    )
    patch_suite(monkeypatch, [PASSING_SUITE])
    executed = spy_terminal(monkeypatch)

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert len(history_with_status(result, "blocked")) == 1
    assert not (tmp_path.parent / "evil.py").exists()
    assert executed == []


# =============================================================
# Phase 5: provider failover end-to-end (through the CLI)
# =============================================================


def test_primary_success_does_not_touch_fallback(
    tmp_path, monkeypatch, capsys
):
    primary = scripted_provider(HAPPY_SCRIPT)
    fallback = scripted_provider([])

    patch_openrouter(monkeypatch, primary)
    patch_gemini(monkeypatch, fallback)
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    assert primary.state["calls"] == 3
    assert fallback.state["calls"] == 0
    assert (
        history_with_status(result, "provider_failover") == []
    )


def test_missing_primary_key_falls_back_and_completes(
    tmp_path, monkeypatch, capsys
):
    primary = scripted_provider(
        HAPPY_SCRIPT,
        errors=[
            RuntimeError("OPENROUTER_API_KEY is not set.")
        ],
    )
    fallback = scripted_provider(
        [dict(RUN_TESTS_PLAN), dict(COMPLETE_PLAN)]
    )

    patch_openrouter(monkeypatch, primary)
    patch_gemini(monkeypatch, fallback)
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    # Configuration failures fail over without retrying.
    assert primary.state["calls"] == 1
    assert fallback.state["calls"] == 2

    failovers = history_with_status(result, "provider_failover")
    assert len(failovers) == 1
    assert failovers[0]["from_provider"] == "openrouter"
    assert failovers[0]["to_provider"] == "gemini"


def test_authentication_failure_falls_back_without_retry(
    tmp_path, monkeypatch, capsys
):
    primary = scripted_provider(
        HAPPY_SCRIPT,
        errors=[RuntimeError("401 Unauthorized")],
    )
    fallback = scripted_provider(
        [dict(RUN_TESTS_PLAN), dict(COMPLETE_PLAN)]
    )

    patch_openrouter(monkeypatch, primary)
    patch_gemini(monkeypatch, fallback)
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    assert primary.state["calls"] == 1
    assert (
        len(history_with_status(result, "provider_failover"))
        == 1
    )


@pytest.mark.parametrize(
    "message",
    [
        pytest.param("400 Bad Request", id="http-400"),
        pytest.param("404 Not Found", id="http-404"),
        pytest.param(
            "422 Unprocessable Entity", id="http-422"
        ),
    ],
)
def test_client_errors_fall_back_without_retry(
    message, tmp_path, monkeypatch, capsys
):
    primary = scripted_provider(
        HAPPY_SCRIPT,
        errors=[RuntimeError(message)],
    )
    fallback = scripted_provider(
        [dict(RUN_TESTS_PLAN), dict(COMPLETE_PLAN)]
    )

    patch_openrouter(monkeypatch, primary)
    patch_gemini(monkeypatch, fallback)
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    # Permanent request rejections are never retried.
    assert primary.state["calls"] == 1
    assert (
        len(history_with_status(result, "provider_failover"))
        == 1
    )


def test_transient_failure_retries_bounded_then_falls_back(
    tmp_path, monkeypatch, capsys
):
    primary = scripted_provider(
        HAPPY_SCRIPT,
        errors=[
            TimeoutError("connection timed out"),
            TimeoutError("connection timed out"),
            TimeoutError("connection timed out"),
        ],
    )
    fallback = scripted_provider(
        [dict(RUN_TESTS_PLAN), dict(COMPLETE_PLAN)]
    )

    patch_openrouter(monkeypatch, primary)
    patch_gemini(monkeypatch, fallback)
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    # 1 initial attempt + MAX_RETRIES (2) retries, then failover.
    assert primary.state["calls"] == 3
    assert (
        len(history_with_status(result, "provider_failover"))
        == 1
    )


def test_malformed_plan_shape_falls_back(
    tmp_path, monkeypatch, capsys
):
    primary = scripted_provider(
        [
            {
                "summary": "bad shape",
                "actions": "not-a-list",
            }
        ]
    )
    fallback = scripted_provider(
        [dict(RUN_TESTS_PLAN), dict(COMPLETE_PLAN)]
    )

    patch_openrouter(monkeypatch, primary)
    patch_gemini(monkeypatch, fallback)
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    assert (
        len(history_with_status(result, "provider_failover"))
        == 1
    )


def test_malformed_action_entry_falls_back(
    tmp_path, monkeypatch, capsys
):
    """A non-object action entry is a classified response
    failure, not an obscure crash of the loop."""

    primary = scripted_provider(
        [
            {
                "summary": "bad action entry",
                "actions": ["not-an-object"],
            }
        ]
    )
    fallback = scripted_provider(
        [dict(RUN_TESTS_PLAN), dict(COMPLETE_PLAN)]
    )

    patch_openrouter(monkeypatch, primary)
    patch_gemini(monkeypatch, fallback)
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    assert primary.state["calls"] == 1
    assert (
        len(history_with_status(result, "provider_failover"))
        == 1
    )


def test_malformed_action_without_healthy_fallback_is_ai_error(
    tmp_path, monkeypatch, capsys
):
    malformed = scripted_provider(
        [
            {
                "summary": "bad action entry",
                "actions": [42],
            }
        ]
    )

    patch_openrouter(monkeypatch, malformed)
    patch_gemini(monkeypatch, malformed)

    code = invoke_cli(tmp_path, "--json")

    captured = capsys.readouterr()
    result = parse_single_json(captured.out)

    # Controlled loop-level failure - not a cli_error escape.
    assert code == 1
    assert result["success"] is False
    assert result["status"] == "ai_error"
    assert "response shape" in result["error"]
    assert "Traceback" not in captured.err


def test_both_providers_unavailable_is_controlled_ai_error(
    tmp_path, monkeypatch, capsys
):
    primary = scripted_provider(
        HAPPY_SCRIPT,
        errors=[
            RuntimeError("OPENROUTER_API_KEY is not set.")
        ],
    )

    # The fallback stays the real GeminiProvider, which raises
    # hermetically because the environment has no keys.
    patch_openrouter(monkeypatch, primary)

    code = invoke_cli(tmp_path, "--json")
    captured = capsys.readouterr()
    result = parse_single_json(captured.out)

    assert code == 1
    assert result["success"] is False
    assert result["status"] == "ai_error"
    assert "GEMINI_API_KEY is not set." in result["error"]
    assert "run finished: status=ai_error" in captured.err


def test_provider_failure_does_not_leak_credentials(
    tmp_path, monkeypatch, capsys
):
    fake_key = "FAKE_E2E_SECRET_KEY_MUST_NOT_LEAK"
    monkeypatch.setenv("OPENROUTER_API_KEY", fake_key)

    primary = scripted_provider(
        HAPPY_SCRIPT,
        errors=[RuntimeError("401 Unauthorized")],
    )
    fallback = scripted_provider(
        [dict(RUN_TESTS_PLAN), dict(COMPLETE_PLAN)]
    )

    patch_openrouter(monkeypatch, primary)
    patch_gemini(monkeypatch, fallback)
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")
    captured = capsys.readouterr()

    assert code == 0
    assert fake_key not in captured.out
    assert fake_key not in captured.err


# =============================================================
# Phase 6: diagnosis + recovery end-to-end
# =============================================================


def test_diagnosed_failure_recovers_and_completes(
    tmp_path, monkeypatch, capsys
):
    patch_openrouter(
        monkeypatch,
        scripted_provider(
            [
                dict(RUN_TESTS_PLAN),
                dict(RUN_TESTS_PLAN),
                dict(COMPLETE_PLAN),
            ]
        ),
    )
    patch_suite(
        monkeypatch, [dict(FAILING_SUITE), dict(PASSING_SUITE)]
    )

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    assert result["context"]["tests_verified"] is True
    assert result["steps"] == 3

    diagnoses = history_with_status(result, "diagnosis")
    assert len(diagnoses) == 1
    assert diagnoses[0]["diagnosis"]["category"] == "syntax_error"

    fixes = history_with_status(result, "fix_attempt_allowed")
    assert len(fixes) == 1
    assert fixes[0]["failure_count"] == 1


def test_repeated_failure_stops_in_controlled_final_state(
    tmp_path, monkeypatch, capsys
):
    provider = scripted_provider(
        [dict(RUN_TESTS_PLAN), dict(RUN_TESTS_PLAN)]
    )
    patch_openrouter(monkeypatch, provider)
    patch_suite(monkeypatch, [dict(FAILING_SUITE)])

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 1
    assert result["success"] is False
    assert result["status"] == "fix_loop_stopped"
    assert result["failure_count"] == 2
    assert result["repeated_count"] == 2
    assert "same failure diagnosis repeated" in result["error"]

    # No infinite loop: one plan per step, two steps total.
    assert provider.state["calls"] == 2
    assert (
        len(history_with_status(result, "diagnosis")) == 2
    )


def test_transient_provider_error_recovers_in_place(
    tmp_path, monkeypatch, capsys
):
    provider = scripted_provider(
        [dict(RUN_TESTS_PLAN), dict(COMPLETE_PLAN)],
        errors=[TimeoutError("connection reset")],
    )
    patch_openrouter(monkeypatch, provider)
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"
    # The same provider recovered: no failover was needed.
    assert provider.state["calls"] == 3
    assert (
        history_with_status(result, "provider_failover") == []
    )


def test_tool_failure_is_controlled_and_loop_continues(
    tmp_path, monkeypatch, capsys
):
    def exploding_list(self):
        raise OSError("workspace unavailable")

    monkeypatch.setattr(
        "agent.tools.workspace_tool.WorkspaceTool.list_files",
        exploding_list,
    )

    patch_openrouter(
        monkeypatch,
        scripted_provider(
            [
                dict(LIST_FILES_PLAN),
                dict(RUN_TESTS_PLAN),
                dict(COMPLETE_PLAN),
            ]
        ),
    )
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")
    result = parse_single_json(capsys.readouterr().out)

    assert code == 0
    assert result["status"] == "completed"

    tool_failures = [
        entry["result"]
        for entry in result["history"]
        if isinstance(entry.get("result"), dict)
        and entry["result"].get("success") is False
        and "workspace unavailable" in str(
            entry["result"].get("error", "")
        )
    ]
    assert len(tool_failures) == 1
    assert tool_failures[0]["tool"] == "list_files"


# =============================================================
# Phase 7: CLI contract end-to-end
# =============================================================


def test_help_flag_exits_zero_with_usage_on_stdout(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "--help",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    assert proc.returncode == 0
    assert "usage:" in proc.stdout
    assert "run" in proc.stdout
    assert proc.stderr == ""


def test_plain_mode_without_keys_reports_ai_error(
    tmp_path, capsys
):
    # Real providers, clean environment: both raise hermetically
    # before any network access could happen.
    code = invoke_cli(tmp_path)

    captured = capsys.readouterr()

    assert code == 1
    assert '"status": "ai_error"' in captured.out
    assert "[AI ERROR]" in captured.out
    assert "[CLI ERROR]" not in captured.out
    # Failover reaches the second provider, whose key error is
    # the one finally reported.
    assert "API_KEY is not set." in captured.out
    assert "run start" in captured.err


def test_non_interactive_denies_approval_without_prompting(
    tmp_path, monkeypatch, capsys
):
    patch_openrouter(
        monkeypatch, scripted_provider(approval_script())
    )
    patch_suite(monkeypatch, [PASSING_SUITE])
    executed = spy_terminal(monkeypatch)

    def boom(prompt=""):
        raise AssertionError("input() must not be called")

    monkeypatch.setattr("builtins.input", boom)

    code = invoke_cli(
        tmp_path, "--json", "--non-interactive"
    )
    captured = capsys.readouterr()
    result = parse_single_json(captured.out)

    assert code == 0
    assert result["status"] == "completed"
    assert executed == []
    assert (
        len(history_with_status(result, "approval_required"))
        == 1
    )

    # The denial notice stayed off stdout; the document is pure.
    assert "NON-INTERACTIVE MODE" in captured.err
    assert "NON-INTERACTIVE MODE" not in captured.out


def test_log_file_environment_variable_writes_log(
    tmp_path, monkeypatch, capsys
):
    log_file = tmp_path / "agent.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))

    patch_openrouter(
        monkeypatch, scripted_provider(HAPPY_SCRIPT)
    )
    patch_suite(monkeypatch, [PASSING_SUITE])

    code = invoke_cli(tmp_path, "--json")

    captured = capsys.readouterr()

    assert code == 0
    assert json.loads(captured.out)["status"] == "completed"

    content = log_file.read_text(encoding="utf-8")
    assert "run start" in content
    assert "run finished" in content
    assert "NITIN CODING AGENT" not in content


def test_invalid_workspace_is_a_controlled_cli_error(
    tmp_path, capsys
):
    missing = tmp_path / "no-such-workspace"

    code = invoke_cli(missing, "--json")

    captured = capsys.readouterr()
    result = parse_single_json(captured.out)

    assert code == 1
    assert result["success"] is False
    assert result["status"] == "cli_error"
    assert "Workspace does not exist" in result["error"]
    assert "Traceback" not in captured.out


# =============================================================
# Phase 9: failure injection at subsystem boundaries
# =============================================================


def test_diagnosis_failure_becomes_controlled_cli_error(
    tmp_path, monkeypatch, capsys
):
    def exploding_diagnose(self, result):
        raise RuntimeError("diagnosis exploded")

    monkeypatch.setattr(
        "agent.core.diagnosis.DiagnosisEngine.diagnose",
        exploding_diagnose,
    )

    patch_openrouter(
        monkeypatch,
        scripted_provider(
            [dict(RUN_TESTS_PLAN), dict(COMPLETE_PLAN)]
        ),
    )
    patch_suite(monkeypatch, [dict(FAILING_SUITE)])

    code = invoke_cli(tmp_path, "--json")

    captured = capsys.readouterr()
    result = parse_single_json(captured.out)

    assert code == 1
    assert result["status"] == "cli_error"
    assert "diagnosis exploded" in result["error"]
    assert "Traceback" not in captured.out
    assert "run failed" in captured.err


def test_recovery_failure_becomes_controlled_ai_error(
    tmp_path, monkeypatch, capsys
):
    def broken_should_retry(self, error_message, attempt):
        raise RuntimeError("recovery exploded")

    monkeypatch.setattr(
        "agent.core.error_recovery."
        "ErrorRecoveryManager.should_retry",
        broken_should_retry,
    )

    provider = scripted_provider(
        [dict(RUN_TESTS_PLAN), dict(COMPLETE_PLAN)],
        errors=[TimeoutError("connection timed out")],
    )
    patch_openrouter(monkeypatch, provider)

    code = invoke_cli(tmp_path, "--json")

    captured = capsys.readouterr()
    result = parse_single_json(captured.out)

    assert code == 1
    assert result["status"] == "ai_error"
    assert "recovery exploded" in result["error"]
    # Unknown-category failures never fail over.
    assert (
        history_with_status(result, "provider_failover") == []
    )
    assert "Traceback" not in captured.out
