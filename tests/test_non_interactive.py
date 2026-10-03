"""Tests for --non-interactive mode (Batch 3)."""

from unittest.mock import patch

import main
from agent.core.action_engine import ActionEngine
from agent.core.approval import ApprovalManager, NonInteractiveApproval
from agent.core.coding_loop import CodingLoop
from agent.core.safety import SafetyDecision, SafetyLevel

# An action SafetyManager classifies as APPROVAL (not on the safe
# allowlist), so it can only run if approval is granted.
APPROVAL_ACTION = {
    "tool": "run_command",
    "command": "python -m pytest -q",
}

SAFE_ACTION = {
    "tool": "list_files",
}


def make_decision() -> SafetyDecision:
    return SafetyDecision(
        SafetyLevel.APPROVAL,
        "Command is not on the safe allowlist.",
    )


# -------------------------------------------------------------
# 6. CLI recognizes the flag
# -------------------------------------------------------------


def test_parser_recognises_non_interactive():
    parser = main.build_parser()

    args = parser.parse_args(
        ["run", "Fix the tests", "--non-interactive"]
    )

    assert args.non_interactive is True


def test_parser_non_interactive_defaults_to_false():
    parser = main.build_parser()

    args = parser.parse_args(["run", "Fix the tests"])

    assert args.non_interactive is False


def test_main_passes_non_interactive_flag():
    with patch(
        "main.run_task",
        return_value={"success": True, "status": "completed"},
    ) as run_task:
        code = main.main(
            [
                "run",
                "Fix the tests",
                "--workspace",
                "workspace",
                "--non-interactive",
                "--json",
            ]
        )

    assert code == 0
    run_task.assert_called_once_with(
        task="Fix the tests",
        workspace="workspace",
        non_interactive=True,
    )


def test_main_passes_interactive_flag_by_default():
    """Existing CLI behaviour stays compatible."""

    with patch(
        "main.run_task",
        return_value={"success": True, "status": "completed"},
    ) as run_task:
        code = main.main(
            [
                "run",
                "Fix the tests",
                "--workspace",
                "workspace",
                "--json",
            ]
        )

    assert code == 0
    run_task.assert_called_once_with(
        task="Fix the tests",
        workspace="workspace",
        non_interactive=False,
    )


# -------------------------------------------------------------
# 10. Interactive wiring is unchanged
# -------------------------------------------------------------


def test_interactive_run_task_builds_plain_coding_loop(tmp_path):
    workspace = tmp_path / "demo"
    workspace.mkdir()

    with patch("main.CodingLoop") as loop_class:
        loop_class.return_value.run.return_value = {
            "success": True
        }

        main.run_task("Fix the tests", str(workspace))

    loop_class.assert_called_once_with(str(workspace))


def test_non_interactive_run_task_injects_deny_callback(tmp_path):
    workspace = tmp_path / "demo"
    workspace.mkdir()

    with patch("main.CodingLoop") as loop_class:
        loop_class.return_value.run.return_value = {
            "success": True
        }

        main.run_task(
            "Fix the tests",
            str(workspace),
            non_interactive=True,
        )

    loop_class.assert_called_once()
    args, kwargs = loop_class.call_args

    assert args == (str(workspace),)
    assert isinstance(
        kwargs["approval_callback"],
        NonInteractiveApproval,
    )


# -------------------------------------------------------------
# 7. Non-interactive mode never calls input()
# -------------------------------------------------------------


def test_non_interactive_never_calls_input(tmp_path, monkeypatch):
    calls = []

    def spy_input(prompt=""):
        calls.append(prompt)
        raise AssertionError("input() must not be called.")

    monkeypatch.setattr("builtins.input", spy_input)

    engine = ActionEngine(
        str(tmp_path),
        approval_callback=NonInteractiveApproval(),
    )

    result = engine.execute(APPROVAL_ACTION)

    assert calls == []
    assert result["status"] == "approval_required"
    assert result["success"] is False


# -------------------------------------------------------------
# 8. EOFError / approval failure cannot crash a non-interactive run
# -------------------------------------------------------------


def test_non_interactive_survives_unavailable_stdin(
    tmp_path,
    monkeypatch,
):
    calls = []

    def eof_input(prompt=""):
        calls.append(prompt)
        raise EOFError("stdin closed")

    monkeypatch.setattr("builtins.input", eof_input)

    engine = ActionEngine(
        str(tmp_path),
        approval_callback=NonInteractiveApproval(),
    )

    result = engine.execute(APPROVAL_ACTION)

    assert calls == []
    assert result["status"] == "approval_required"
    assert result["success"] is False


def test_non_interactive_callback_swallows_eoferror():
    class ExplodingApproval(NonInteractiveApproval):
        def _deny(self, action, decision):
            raise EOFError("stdin unavailable")

    approval = ExplodingApproval()

    approved = approval(
        {"tool": "run_command"},
        make_decision(),
    )

    assert approved is False
    assert approval.denied_count == 1
    assert approval.last_error is not None
    assert "EOFError" in approval.last_error


def test_non_interactive_callback_never_raises():
    approval = NonInteractiveApproval()

    approved = approval(
        {"tool": "run_command"},
        make_decision(),
    )

    assert approved is False
    assert approval.last_error is None


# -------------------------------------------------------------
# Denials are safe: nothing dangerous runs, safe work still works
# -------------------------------------------------------------


def test_non_interactive_does_not_execute_denied_command(tmp_path):
    side_effect = tmp_path / "approved.txt"
    action = {
        "tool": "run_command",
        "command": (
            f"python -c \"open('{side_effect.name}','w')"
            ".write('x')\""
        ),
    }

    engine = ActionEngine(
        str(tmp_path),
        approval_callback=NonInteractiveApproval(),
    )

    result = engine.execute(action)

    assert result["status"] == "approval_required"
    assert not side_effect.exists()


def test_non_interactive_allows_safe_actions(tmp_path):
    engine = ActionEngine(
        str(tmp_path),
        approval_callback=NonInteractiveApproval(),
    )

    result = engine.execute(SAFE_ACTION)

    assert result["success"] is True
    assert engine.approval_callback.denied_count == 0


def test_coding_loop_wires_non_interactive_callback(tmp_path, monkeypatch):
    calls = []

    def spy_input(prompt=""):
        calls.append(prompt)
        raise AssertionError("input() must not be called.")

    monkeypatch.setattr("builtins.input", spy_input)

    loop = CodingLoop(
        str(tmp_path),
        approval_callback=NonInteractiveApproval(),
    )

    result = loop.engine.execute(APPROVAL_ACTION)

    assert calls == []
    assert result["status"] == "approval_required"
    assert loop.approval_manager is not None


# -------------------------------------------------------------
# 9. Interactive approval behaviour remains unchanged
# -------------------------------------------------------------


def test_interactive_approval_still_prompts(monkeypatch):
    calls = []

    def fake_input(prompt=""):
        calls.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", fake_input)

    manager = ApprovalManager()

    approved = manager.request(
        {"tool": "run_command", "command": "git push origin main"},
        make_decision(),
    )

    assert approved is True
    assert len(calls) == 1
    assert "Allow this action?" in calls[0]


def test_interactive_approval_denies_by_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    manager = ApprovalManager()

    approved = manager.request(
        {"tool": "run_command", "command": "git push origin main"},
        make_decision(),
    )

    assert approved is False
