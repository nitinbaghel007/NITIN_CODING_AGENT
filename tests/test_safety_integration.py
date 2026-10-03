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
