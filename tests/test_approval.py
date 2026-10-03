from agent.core.approval import ApprovalManager
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
