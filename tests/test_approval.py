from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.core.action_engine import ActionEngine
from agent.core.approval import ApprovalManager, NonInteractiveApproval
from agent.core.safety import SafetyDecision, SafetyLevel


def test_approval_manager_rejects_by_default(monkeypatch):
    manager = ApprovalManager()

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "",
    )

    decision = SafetyDecision(
        SafetyLevel.APPROVAL,
        "Test approval",
    )

    result = manager.request(
        {
            "tool": "run_command",
            "command": "pip install requests",
        },
        decision,
    )

    assert result is False


def test_approval_manager_accepts_yes(monkeypatch):
    manager = ApprovalManager()

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "y",
    )

    decision = SafetyDecision(
        SafetyLevel.APPROVAL,
        "Test approval",
    )

    result = manager.request(
        {
            "tool": "run_command",
            "command": "pip install requests",
        },
        decision,
    )

    assert result is True


def test_approval_manager_accepts_yes_word(monkeypatch):
    manager = ApprovalManager()

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "YES",
    )

    decision = SafetyDecision(
        SafetyLevel.APPROVAL,
        "Test approval",
    )

    result = manager.request(
        {
            "tool": "run_command",
            "command": "git push origin main",
        },
        decision,
    )

    assert result is True


# -------------------------------------------------------------
# Batch 5 - approval robustness
# -------------------------------------------------------------
# Approval must be deterministic: a failed prompt denies, unexpected
# input denies, an approved action runs exactly once, and a denied or
# blocked action never reaches subprocess.


APPROVAL_ACTION = {
    "tool": "run_command",
    "command": "pip install example-package",
}
SAFE_ACTION = {"tool": "list_files"}
BLOCKED_ACTION = {
    "tool": "run_command",
    "command": "format C:",
}
CHAIN_ACTION = {
    "tool": "run_command",
    "command": "python --version & whoami",
}


def make_decision() -> SafetyDecision:
    return SafetyDecision(SafetyLevel.APPROVAL, "Test approval")


def approve_everything(action, decision) -> bool:
    """A callback that always approves - used to prove approval alone
    cannot make a blocked or unknown action executable."""
    return True


def spy_subprocess(monkeypatch, executed: list) -> None:
    """Record subprocess calls instead of running anything real."""

    def fake_run(cmd, **kwargs):
        executed.append(cmd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run", fake_run
    )


def make_engine(tmp_path, callback):
    return ActionEngine(str(tmp_path), approval_callback=callback)


def request_with(raw, action=APPROVAL_ACTION):
    """Feed `raw` to the approval prompt and return the decision."""
    manager = ApprovalManager()

    with patch("builtins.input", return_value=raw):
        return manager.request(action, make_decision())


# -------------------------------------------------------------
# 1. EOF / input robustness
# -------------------------------------------------------------


def test_approval_denies_on_eoferror():
    """Closed stdin must deny, never raise."""
    manager = ApprovalManager()

    with patch(
        "builtins.input",
        side_effect=EOFError("EOF when reading a line"),
    ):
        assert manager.request(APPROVAL_ACTION, make_decision()) is False


def test_approval_denies_on_keyboard_interrupt():
    """Ctrl+C at the prompt must deny, never raise."""
    manager = ApprovalManager()

    with patch("builtins.input", side_effect=KeyboardInterrupt):
        assert manager.request(APPROVAL_ACTION, make_decision()) is False


def test_approval_denies_on_broken_stdin():
    manager = ApprovalManager()

    with patch(
        "builtins.input",
        side_effect=OSError("stdin broken pipe"),
    ):
        assert manager.request(APPROVAL_ACTION, make_decision()) is False


def test_approval_denies_on_closed_stdin_value_error():
    manager = ApprovalManager()

    with patch(
        "builtins.input",
        side_effect=ValueError("I/O operation on closed file"),
    ):
        assert manager.request(APPROVAL_ACTION, make_decision()) is False


def test_closed_stdin_end_to_end_denies_action(tmp_path, monkeypatch):
    """Full path: closed stdin -> denial -> nothing executes."""
    executed = []
    spy_subprocess(monkeypatch, executed)

    engine = make_engine(tmp_path, ApprovalManager().request)

    with patch(
        "builtins.input",
        side_effect=EOFError("EOF when reading a line"),
    ):
        result = engine.execute(APPROVAL_ACTION)

    assert result["success"] is False
    assert result["status"] == "approval_required"
    assert executed == []


# -------------------------------------------------------------
# 2. Approval decision normalization
# -------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["y", "Y", "yes", "YES", "Yes", " y ", "  Y  ", "\ty\n", "YeS"],
)
def test_approval_accepts_only_explicit_yes(raw):
    assert request_with(raw) is True


@pytest.mark.parametrize(
    "raw",
    ["n", "N", "no", "NO", "No", " n ", ""],
)
def test_approval_denies_no_and_empty(raw):
    assert request_with(raw) is False


@pytest.mark.parametrize(
    "raw",
    ["   ", "\n", "\t", "\t \n"],
)
def test_approval_denies_whitespace_only(raw):
    assert request_with(raw) is False


@pytest.mark.parametrize(
    "raw",
    ["maybe", "1", "true", "ok", "yess", "Y E S", "sure", "yeah", "yy"],
)
def test_approval_denies_unexpected_input(raw):
    """Unexpected input must never approve an action."""
    assert request_with(raw) is False


@pytest.mark.parametrize("raw", [None, 1, 0, [], {}])
def test_approval_denies_non_text_input(raw):
    """A non-text answer must deny instead of crashing on .strip()."""
    assert request_with(raw) is False


# -------------------------------------------------------------
# 3. Non-interactive behaviour
# -------------------------------------------------------------


def test_non_interactive_denies_without_prompting():
    prompts = []

    def spy_input(prompt=""):
        prompts.append(prompt)
        raise AssertionError("input() must not be called")

    with patch("builtins.input", side_effect=spy_input):
        approved = NonInteractiveApproval()(
            APPROVAL_ACTION, make_decision()
        )

    assert approved is False
    assert prompts == []


def test_non_interactive_denial_is_recorded():
    approval = NonInteractiveApproval()

    assert approval(APPROVAL_ACTION, make_decision()) is False
    assert approval.denied_count == 1
    assert approval.last_error is None


def test_non_interactive_survives_keyboard_interrupt():
    """KeyboardInterrupt is not an Exception subclass - it must still
    be swallowed by the non-interactive contract."""

    class Interrupted(NonInteractiveApproval):
        def _deny(self, action, decision):
            raise KeyboardInterrupt

    approval = Interrupted()

    assert approval(APPROVAL_ACTION, make_decision()) is False
    assert approval.denied_count == 1
    assert approval.last_error is not None
    assert "KeyboardInterrupt" in approval.last_error


# -------------------------------------------------------------
# 4. Approval callback invocation
# -------------------------------------------------------------


def test_safe_action_skips_approval(tmp_path, monkeypatch):
    executed = []
    spy_subprocess(monkeypatch, executed)
    calls = []

    def callback(action, decision):
        calls.append(action)
        return False

    engine = make_engine(tmp_path, callback)
    result = engine.execute(SAFE_ACTION)

    assert result["success"] is True
    assert calls == []


def test_blocked_action_skips_approval(tmp_path, monkeypatch):
    """Even an approving callback must never be consulted for a
    blocked action."""
    executed = []
    spy_subprocess(monkeypatch, executed)
    calls = []

    def callback(action, decision):
        calls.append(action)
        return True

    engine = make_engine(tmp_path, callback)
    result = engine.execute(BLOCKED_ACTION)

    assert result["status"] == "blocked"
    assert result["success"] is False
    assert calls == []
    assert executed == []


def test_approval_invoked_exactly_once_when_approved(
    tmp_path, monkeypatch,
):
    executed = []
    spy_subprocess(monkeypatch, executed)
    calls = []

    def callback(action, decision):
        calls.append(action)
        return True

    engine = make_engine(tmp_path, callback)
    result = engine.execute(APPROVAL_ACTION)

    assert len(calls) == 1
    assert result["success"] is True
    assert len(executed) == 1
    assert executed[0] == "pip install example-package"


def test_denied_action_executes_zero_times(tmp_path, monkeypatch):
    executed = []
    spy_subprocess(monkeypatch, executed)
    calls = []

    def callback(action, decision):
        calls.append(action)
        return False

    engine = make_engine(tmp_path, callback)
    result = engine.execute(APPROVAL_ACTION)

    assert len(calls) == 1
    assert result["status"] == "approval_required"
    assert result["success"] is False
    assert executed == []


def test_raising_approval_callback_denies_without_crash(
    tmp_path, monkeypatch,
):
    executed = []
    spy_subprocess(monkeypatch, executed)

    def boom(action, decision):
        raise EOFError("approval path exploded")

    engine = make_engine(tmp_path, boom)
    result = engine.execute(APPROVAL_ACTION)

    assert result["success"] is False
    assert result["status"] == "approval_required"
    assert executed == []
    assert engine.last_approval_error is not None
    assert "EOFError" in engine.last_approval_error


def test_keyboard_interrupt_in_callback_denies(
    tmp_path, monkeypatch,
):
    executed = []
    spy_subprocess(monkeypatch, executed)

    def boom(action, decision):
        raise KeyboardInterrupt

    engine = make_engine(tmp_path, boom)
    result = engine.execute(APPROVAL_ACTION)

    assert result["success"] is False
    assert result["status"] == "approval_required"
    assert executed == []
    assert "KeyboardInterrupt" in engine.last_approval_error


# -------------------------------------------------------------
# 5. Action execution integration (full path)
# -------------------------------------------------------------


def test_full_path_non_interactive_denial(tmp_path, monkeypatch):
    """action -> safety -> denial -> no execution, no input()."""
    executed = []
    spy_subprocess(monkeypatch, executed)

    engine = make_engine(tmp_path, NonInteractiveApproval())

    with patch(
        "builtins.input",
        side_effect=AssertionError("input() must not be called"),
    ):
        result = engine.execute(APPROVAL_ACTION)

    assert result["status"] == "approval_required"
    assert executed == []
    assert engine.approval_callback.denied_count == 1


def test_full_path_blocked_chain_cannot_be_approved(
    tmp_path, monkeypatch,
):
    """Approval cannot launder a shell chain into execution."""
    executed = []
    spy_subprocess(monkeypatch, executed)

    engine = make_engine(tmp_path, approve_everything)
    result = engine.execute(CHAIN_ACTION)

    assert result["status"] == "blocked"
    assert result["success"] is False
    assert executed == []


def test_full_path_unknown_executable_cannot_execute_with_approval(
    tmp_path, monkeypatch,
):
    """Approval alone must not let an unknown executable run."""
    executed = []
    spy_subprocess(monkeypatch, executed)

    engine = make_engine(tmp_path, approve_everything)
    result = engine.execute(
        {"tool": "run_command", "command": "curl http://example.com"}
    )

    assert result["success"] is False
    assert executed == []
    assert "not allowed" in result["result"]["stderr"]


# -------------------------------------------------------------
# 7. Existing security guarantees still hold
# -------------------------------------------------------------


@pytest.mark.parametrize(
    "action, expected",
    [
        (CHAIN_ACTION, SafetyLevel.BLOCKED),
        (BLOCKED_ACTION, SafetyLevel.BLOCKED),
        (
            {"tool": "run_command", "command": "rm -rf /"},
            SafetyLevel.BLOCKED,
        ),
        (
            {"tool": "write_file", "path": "../escape.txt", "content": "x"},
            SafetyLevel.BLOCKED,
        ),
        ({"tool": "run_shell", "command": "ls"}, SafetyLevel.BLOCKED),
        (
            {"tool": "run_command", "command": "python -m pytest -q"},
            SafetyLevel.SAFE,
        ),
        (APPROVAL_ACTION, SafetyLevel.APPROVAL),
    ],
)
def test_security_guarantees_still_hold(tmp_path, action, expected):
    engine = make_engine(tmp_path, approve_everything)

    decision = engine.safety.validate(action)

    assert decision.level is expected, action
