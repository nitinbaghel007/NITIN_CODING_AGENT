from agent.core.action_engine import ActionEngine
from agent.core.safety import SafetyLevel


def test_action_engine_blocks_dangerous_command(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute(
        {
            "tool": "run_command",
            "command": "del important.txt",
        }
    )

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert result["safety_level"] == SafetyLevel.BLOCKED.value


def test_action_engine_requires_approval(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute(
        {
            "tool": "run_command",
            "command": "pip install requests",
        }
    )

    assert result["success"] is False
    assert result["status"] == "approval_required"
    assert result["safety_level"] == SafetyLevel.APPROVAL.value


def test_action_engine_executes_after_approval(tmp_path):
    engine = ActionEngine(
        str(tmp_path),
        approval_callback=lambda action, decision: True,
    )

    engine.terminal.run = lambda command: {
        "return_code": 0,
        "stdout": "approved",
        "stderr": "",
        "success": True,
    }

    result = engine.execute(
        {
            "tool": "run_command",
            "command": "custom-safe-command",
        }
    )

    assert result["success"] is True
    assert result["tool"] == "run_command"


def test_action_engine_safe_file_action_still_works(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute(
        {
            "tool": "write_file",
            "path": "hello.py",
            "content": "print('hello')",
        }
    )

    assert result["success"] is True
    assert (tmp_path / "hello.py").read_text(
        encoding="utf-8"
    ) == "print('hello')"


def test_action_engine_blocks_workspace_escape(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute(
        {
            "tool": "write_file",
            "path": "../outside.py",
            "content": "bad",
        }
    )

    assert result["success"] is False
    assert result["status"] == "blocked"


# =========================================================
# Batch 9 Phase 3: dispatch, malformed input, tool failures
# =========================================================


def test_list_files_dispatch_returns_stable_shape(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute({"tool": "list_files"})

    assert result["success"] is True
    assert result["files"] == []
    assert result["directories"] == []
    assert result["workspace"] == str(tmp_path.resolve())


def test_read_file_dispatch_returns_content(tmp_path):
    engine = ActionEngine(str(tmp_path))
    (tmp_path / "app.py").write_text(
        "print(1)", encoding="utf-8"
    )

    result = engine.execute(
        {"tool": "read_file", "path": "app.py"}
    )

    assert result["success"] is True
    assert result["tool"] == "read_file"
    assert result["content"] == "print(1)"
    assert result["path"] == "app.py"


def test_run_command_dispatch_propagates_tool_result(
    tmp_path, monkeypatch
):
    engine = ActionEngine(str(tmp_path))

    monkeypatch.setattr(
        engine.terminal,
        "run",
        lambda command: {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "success": True,
        },
    )

    result = engine.execute(
        {"tool": "run_command", "command": "git status"}
    )

    assert result == {
        "success": True,
        "tool": "run_command",
        "command": "git status",
        "result": {
            "return_code": 0,
            "stdout": "ok",
            "stderr": "",
            "success": True,
        },
    }


def test_run_tests_dispatch_propagates_tool_result(
    tmp_path, monkeypatch
):
    engine = ActionEngine(str(tmp_path))

    monkeypatch.setattr(
        engine.tests,
        "run_pytest",
        lambda timeout=120: {
            "return_code": 0,
            "stdout": "2 passed",
            "stderr": "",
            "success": True,
        },
    )

    result = engine.execute({"tool": "run_tests"})

    assert result == {
        "success": True,
        "tool": "run_tests",
        "result": {
            "return_code": 0,
            "stdout": "2 passed",
            "stderr": "",
            "success": True,
        },
    }


def test_safe_action_executes_exactly_once(
    tmp_path, monkeypatch
):
    engine = ActionEngine(str(tmp_path))
    calls = []

    monkeypatch.setattr(
        engine.terminal,
        "run",
        lambda command: calls.append(command)
        or {
            "return_code": 0,
            "stdout": "",
            "stderr": "",
            "success": True,
        },
    )

    result = engine.execute(
        {"tool": "run_command", "command": "git status"}
    )

    assert result["success"] is True
    assert calls == ["git status"]


def test_unknown_tool_is_blocked_deterministically(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute({"tool": "hack"})

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert result["safety_level"] == "blocked"
    assert result["error"] == (
        "Unknown tool is blocked: hack"
    )


def test_blocked_action_never_reaches_the_tool(
    tmp_path, monkeypatch
):
    engine = ActionEngine(str(tmp_path))
    calls = []

    monkeypatch.setattr(
        engine.terminal,
        "run",
        lambda command: calls.append(command),
    )

    result = engine.execute(
        {
            "tool": "run_command",
            "command": "del important.txt",
        }
    )

    assert result["status"] == "blocked"
    assert calls == []


def test_non_dict_action_is_blocked(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute("not an action")

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert result["error"] == (
        "Action must be a JSON object."
    )


def test_action_without_tool_is_blocked(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute({})

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert result["error"] == (
        "Action is missing a valid tool."
    )


def test_non_string_tool_is_blocked(tmp_path):
    engine = ActionEngine(str(tmp_path))

    for tool_value in (123, None, ["read_file"]):
        result = engine.execute({"tool": tool_value})

        assert result["success"] is False
        assert result["status"] == "blocked"


def test_write_file_without_content_is_rejected(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute(
        {"tool": "write_file", "path": "a.py"}
    )

    assert result["success"] is False
    assert result["error"] == (
        "write_file requires content."
    )
    assert not (tmp_path / "a.py").exists()


def test_write_file_without_path_is_blocked(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute(
        {"tool": "write_file", "content": "x"}
    )

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert result["error"] == (
        "File action requires a relative path."
    )


def test_read_file_without_path_is_blocked(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute({"tool": "read_file"})

    assert result["success"] is False
    assert result["status"] == "blocked"


def test_run_command_without_command_is_blocked(tmp_path):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute({"tool": "run_command"})

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert result["error"] == (
        "run_command requires a non-empty command."
    )


def test_read_file_missing_file_is_controlled_failure(
    tmp_path,
):
    engine = ActionEngine(str(tmp_path))

    result = engine.execute(
        {"tool": "read_file", "path": "missing.py"}
    )

    assert result["success"] is False
    assert result["tool"] == "read_file"
    assert result["path"] == "missing.py"
    assert result["error"]


def test_write_file_tool_exception_is_controlled(
    tmp_path, monkeypatch
):
    engine = ActionEngine(str(tmp_path))

    def exploding_write(path, content):
        raise OSError("disk full")

    monkeypatch.setattr(
        engine.files, "write", exploding_write
    )

    result = engine.execute(
        {
            "tool": "write_file",
            "path": "a.py",
            "content": "x",
        }
    )

    assert result["success"] is False
    assert result["tool"] == "write_file"
    assert result["error"] == "disk full"


def test_read_file_tool_exception_is_controlled(
    tmp_path, monkeypatch
):
    engine = ActionEngine(str(tmp_path))

    def exploding_read(path):
        raise OSError("io failure")

    monkeypatch.setattr(
        engine.files, "read", exploding_read
    )

    result = engine.execute(
        {"tool": "read_file", "path": "a.py"}
    )

    assert result["success"] is False
    assert result["tool"] == "read_file"
    assert result["error"] == "io failure"


def test_list_files_tool_exception_is_controlled(
    tmp_path, monkeypatch
):
    """An unexpected tool exception must become a controlled
    failure result, never a crash of the action path."""

    engine = ActionEngine(str(tmp_path))

    def exploding_list():
        raise OSError("workspace vanished")

    monkeypatch.setattr(
        engine.workspace_tool,
        "list_files",
        exploding_list,
    )

    result = engine.execute({"tool": "list_files"})

    assert result["success"] is False
    assert result["tool"] == "list_files"
    assert "workspace vanished" in result["error"]


def test_run_command_tool_exception_is_controlled(
    tmp_path, monkeypatch
):
    """An unexpected terminal exception must become a
    controlled failure result, never a crash."""

    engine = ActionEngine(str(tmp_path))

    def exploding_run(command):
        raise OSError("cannot spawn shell")

    monkeypatch.setattr(
        engine.terminal, "run", exploding_run
    )

    result = engine.execute(
        {"tool": "run_command", "command": "git status"}
    )

    assert result["success"] is False
    assert result["tool"] == "run_command"
    assert result["command"] == "git status"
    assert "cannot spawn shell" in result["error"]


def test_run_tests_tool_exception_is_controlled(
    tmp_path, monkeypatch
):
    """An unexpected test-runner exception must become a
    controlled failure result, never a crash."""

    engine = ActionEngine(str(tmp_path))

    def exploding_pytest(timeout=120):
        raise OSError("python missing from PATH")

    monkeypatch.setattr(
        engine.tests, "run_pytest", exploding_pytest
    )

    result = engine.execute({"tool": "run_tests"})

    assert result["success"] is False
    assert result["tool"] == "run_tests"
    assert "python missing from PATH" in result["error"]
