"""Batch 9 Phase 2: direct contract tests for every tool.

Every subprocess call is monkeypatched - no real process is ever
started, nothing reaches the network, and no real filesystem
operation happens outside the pytest tmp workspace.
"""

import json
import subprocess
from pathlib import Path

import pytest

from agent.core.action_engine import ActionEngine
from agent.tools.file_tool import FileTool
from agent.tools.terminal_tool import TerminalTool
from agent.tools.test_tool import TestTool
from agent.tools.workspace_tool import WorkspaceTool

# pytest must not try to collect TestTool as a test class.
TestTool.__test__ = False


class FakeCompleted:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# =========================================================
# FileTool
# =========================================================


def test_file_tool_write_and_read_roundtrip(tmp_path):
    tool = FileTool(str(tmp_path))

    tool.write("notes/todo.txt", "step one")

    assert tool.read("notes/todo.txt") == "step one"
    assert tool.exists("notes/todo.txt") is True


def test_file_tool_write_creates_missing_parent_dirs(tmp_path):
    tool = FileTool(str(tmp_path))

    tool.write("deep/nested/dir/file.py", "print(1)")

    assert (
        tmp_path / "deep" / "nested" / "dir" / "file.py"
    ).read_text(encoding="utf-8") == "print(1)"


def test_file_tool_exists_is_false_for_missing_file(tmp_path):
    tool = FileTool(str(tmp_path))

    assert tool.exists("missing.py") is False


def test_file_tool_read_missing_file_raises(tmp_path):
    tool = FileTool(str(tmp_path))

    with pytest.raises(FileNotFoundError):
        tool.read("missing.py")


def test_file_tool_blocks_parent_traversal(tmp_path):
    tool = FileTool(str(tmp_path))

    with pytest.raises(ValueError, match="outside"):
        tool.read("../outside.txt")

    with pytest.raises(ValueError, match="outside"):
        tool.write("../outside.txt", "data")

    with pytest.raises(ValueError, match="outside"):
        tool.exists("../outside.txt")

    assert not (tmp_path.parent / "outside.txt").exists()


def test_file_tool_blocks_absolute_path(tmp_path):
    tool = FileTool(str(tmp_path))

    with pytest.raises(ValueError, match="outside"):
        tool.write("C:/evil.txt", "data")


# =========================================================
# WorkspaceTool
# =========================================================


def test_workspace_tool_list_files_shape(tmp_path):
    tool = WorkspaceTool(str(tmp_path))

    (tmp_path / "app.py").write_text(
        "print(1)", encoding="utf-8"
    )
    (tmp_path / "pkg").mkdir()

    result = tool.list_files()

    assert result["success"] is True
    assert result["files"] == ["app.py"]
    assert result["directories"] == ["pkg"]
    assert result["workspace"] == str(tmp_path.resolve())


def test_workspace_tool_ignores_cache_directories(tmp_path):
    tool = WorkspaceTool(str(tmp_path))

    (tmp_path / "keep.txt").write_text(
        "x", encoding="utf-8"
    )
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".git").mkdir()

    result = tool.list_files()

    assert result["files"] == ["keep.txt"]
    assert result["directories"] == []


def test_workspace_tool_empty_workspace_lists_nothing(
    tmp_path,
):
    tool = WorkspaceTool(str(tmp_path))

    result = tool.list_files()

    assert result["success"] is True
    assert result["files"] == []
    assert result["directories"] == []


# =========================================================
# TerminalTool (subprocess always mocked)
# =========================================================


def test_terminal_tool_success_result_shape(
    tmp_path, monkeypatch
):
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls.update(kwargs)
        return FakeCompleted(
            returncode=0,
            stdout="hello",
            stderr="",
        )

    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run",
        fake_run,
    )

    tool = TerminalTool(str(tmp_path))
    result = tool.run("git status")

    assert result == {
        "return_code": 0,
        "stdout": "hello",
        "stderr": "",
        "success": True,
    }

    # Dispatch contract: runs inside the workspace, captured.
    assert calls["command"] == "git status"
    assert Path(calls["cwd"]) == tmp_path.resolve()
    assert calls["shell"] is True
    assert calls["capture_output"] is True


def test_terminal_tool_reports_failing_command(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run",
        lambda *a, **k: FakeCompleted(
            returncode=1, stdout="", stderr="boom"
        ),
    )

    result = TerminalTool(str(tmp_path)).run("git log")

    assert result["success"] is False
    assert result["return_code"] == 1
    assert result["stderr"] == "boom"


def test_terminal_tool_blocked_command_never_executes(
    tmp_path, monkeypatch
):
    executed = []

    def fake_run(command, **kwargs):
        executed.append(command)
        return FakeCompleted()

    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run",
        fake_run,
    )

    result = TerminalTool(str(tmp_path)).run(
        "powershell -Command Get-Date"
    )

    assert result["success"] is False
    assert result["return_code"] == -2
    assert "blocked for safety" in result["stderr"]
    assert executed == []


def test_terminal_tool_unknown_command_never_executes(
    tmp_path, monkeypatch
):
    executed = []

    def fake_run(command, **kwargs):
        executed.append(command)
        return FakeCompleted()

    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run",
        fake_run,
    )

    result = TerminalTool(str(tmp_path)).run(
        "unknown-tool --flag"
    )

    assert result["success"] is False
    assert result["return_code"] == -2
    assert "not allowed" in result["stderr"]
    assert executed == []


def test_terminal_tool_empty_command_rejected_before_exec(
    tmp_path, monkeypatch
):
    """Malformed input is rejected before any process starts."""

    executed = []

    def fake_run(command, **kwargs):
        executed.append(command)
        return FakeCompleted()

    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run",
        fake_run,
    )

    with pytest.raises(ValueError, match="empty"):
        TerminalTool(str(tmp_path)).run("   ")

    assert executed == []


def test_terminal_tool_timeout_is_controlled_failure(
    tmp_path, monkeypatch
):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=command, timeout=kwargs.get("timeout")
        )

    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run",
        fake_run,
    )

    result = TerminalTool(str(tmp_path)).run(
        "git status", timeout=1
    )

    assert result == {
        "return_code": -1,
        "stdout": "",
        "stderr": "Command timed out.",
        "success": False,
    }


# =========================================================
# TestTool (subprocess always mocked)
# =========================================================


def test_test_tool_success_result_shape(
    tmp_path, monkeypatch
):
    calls = {}

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls.update(kwargs)
        return FakeCompleted(
            returncode=0,
            stdout="3 passed",
            stderr="",
        )

    monkeypatch.setattr(
        "agent.tools.test_tool.subprocess.run",
        fake_run,
    )

    result = TestTool(str(tmp_path)).run_pytest()

    assert result == {
        "return_code": 0,
        "stdout": "3 passed",
        "stderr": "",
        "success": True,
    }
    assert calls["args"] == [
        "python",
        "-m",
        "pytest",
    ]
    assert Path(calls["cwd"]) == tmp_path.resolve()


def test_test_tool_reports_failing_suite(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "agent.tools.test_tool.subprocess.run",
        lambda *a, **k: FakeCompleted(
            returncode=1, stdout="1 failed", stderr="err"
        ),
    )

    result = TestTool(str(tmp_path)).run_pytest()

    assert result["success"] is False
    assert result["return_code"] == 1
    assert result["stdout"] == "1 failed"
    assert result["stderr"] == "err"


def test_test_tool_timeout_is_controlled_failure(
    tmp_path, monkeypatch
):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args, timeout=kwargs.get("timeout")
        )

    monkeypatch.setattr(
        "agent.tools.test_tool.subprocess.run",
        fake_run,
    )

    result = TestTool(str(tmp_path)).run_pytest()

    assert result == {
        "return_code": -1,
        "stdout": "",
        "stderr": "Pytest timed out.",
        "success": False,
    }


# =========================================================
# Cross-tool guarantees
# =========================================================


def test_tools_print_nothing_to_stdout(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "agent.tools.terminal_tool.subprocess.run",
        lambda *a, **k: FakeCompleted(),
    )
    monkeypatch.setattr(
        "agent.tools.test_tool.subprocess.run",
        lambda *a, **k: FakeCompleted(),
    )

    file_tool = FileTool(str(tmp_path))
    file_tool.write("x.txt", "data")
    file_tool.read("x.txt")
    file_tool.exists("x.txt")

    WorkspaceTool(str(tmp_path)).list_files()
    TerminalTool(str(tmp_path)).run("git status")
    TestTool(str(tmp_path)).run_pytest()

    captured = capsys.readouterr()
    assert captured.out == ""


def test_tool_errors_do_not_leak_environment_secrets(
    tmp_path, monkeypatch
):
    """Controlled failure dicts never embed environment values."""

    secret = "FAKE_KEY_BATCH9_MUST_NOT_LEAK"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    engine = ActionEngine(str(tmp_path))

    results = [
        engine.execute({"tool": "hack_the_planet"}),
        engine.execute(
            {"tool": "read_file", "path": "missing.py"}
        ),
        engine.execute(
            {
                "tool": "run_command",
                "command": "del important.txt",
            }
        ),
    ]

    blob = json.dumps(results)
    assert secret not in blob
