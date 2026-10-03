"""Batch 9 Phase 6: CodingLoop integration for the diagnosis,
fix-loop, and tool-failure paths.

All providers are fakes, subprocess runs are monkeypatched,
and no network or destructive filesystem operation occurs.
"""

from agent.core.coding_loop import CodingLoop
from agent.providers.base import LLMProvider
from agent.providers.manager import ProviderManager


class PlanProvider(LLMProvider):
    """Returns a fixed sequence of plans; the last one repeats."""

    def __init__(self, plans):
        self.plans = list(plans)
        self.calls = 0

    def generate(self, prompt: str) -> str:
        return "fake"

    def generate_actions(self, task: str) -> dict:
        self.calls += 1

        if len(self.plans) > 1:
            return self.plans.pop(0)

        return self.plans[0]


RUN_TESTS_PLAN = {
    "summary": "Run the test suite",
    "actions": [{"tool": "run_tests"}],
}

COMPLETE_PLAN = {
    "summary": "Task completed",
    "actions": [],
}


def make_loop(tmp_path, plans):
    provider = PlanProvider(plans)

    manager = ProviderManager()
    manager.register("openrouter", provider)
    manager.set_default("openrouter")

    loop = CodingLoop(
        str(tmp_path),
        provider_manager=manager,
        approval_callback=lambda action, decision: True,
    )
    return loop, provider


def failing_suite(marker="SyntaxError: invalid syntax"):
    def run_pytest(timeout=120):
        return {
            "return_code": 1,
            "stdout": "1 failed",
            "stderr": marker,
            "success": False,
        }

    return run_pytest


def passing_suite():
    def run_pytest(timeout=120):
        return {
            "return_code": 0,
            "stdout": "2 passed",
            "stderr": "",
            "success": True,
        }

    return run_pytest


def history_results(result, status):
    return [
        entry["result"]
        for entry in result["history"]
        if isinstance(entry.get("result"), dict)
        and entry["result"].get("status") == status
    ]


def test_failed_tests_flow_into_diagnosis_then_fix_then_success(
    tmp_path, monkeypatch
):
    """failure -> diagnosis -> fix allowed -> retry -> pass."""

    monkeypatch.setattr(
        "agent.core.error_recovery.time.sleep",
        lambda delay: None,
    )

    loop, provider = make_loop(
        tmp_path,
        [
            dict(RUN_TESTS_PLAN, actions=[{"tool": "run_tests"}]),
            dict(RUN_TESTS_PLAN, actions=[{"tool": "run_tests"}]),
            dict(COMPLETE_PLAN),
        ],
    )

    suite_outcomes = iter(
        [
            failing_suite()(),
            passing_suite()(),
        ]
    )
    monkeypatch.setattr(
        loop.engine.tests,
        "run_pytest",
        lambda timeout=120: next(suite_outcomes),
    )

    result = loop.run("Fix the failing tests.")

    assert result["status"] == "completed"

    diagnoses = history_results(result, "diagnosis")
    assert len(diagnoses) == 1
    assert (
        diagnoses[0]["diagnosis"]["category"]
        == "syntax_error"
    )

    fix_events = history_results(result, "fix_attempt_allowed")
    assert len(fix_events) == 1
    assert fix_events[0]["failure_count"] == 1
    assert fix_events[0]["repeated_count"] == 1

    assert result["context"]["tests_verified"] is True
    assert provider.calls == 3


def test_repeated_identical_failure_stops_fix_loop(
    tmp_path, monkeypatch
):
    """The same failing diagnosis twice ends the run with a
    controlled fix_loop_stopped result - no infinite loop."""

    loop, provider = make_loop(
        tmp_path,
        [
            dict(RUN_TESTS_PLAN, actions=[{"tool": "run_tests"}]),
            dict(RUN_TESTS_PLAN, actions=[{"tool": "run_tests"}]),
        ],
    )

    executions = []

    def spy_suite(timeout=120):
        executions.append(1)
        return failing_suite()()

    monkeypatch.setattr(
        loop.engine.tests, "run_pytest", spy_suite
    )

    result = loop.run("Fail repeatedly.")

    assert result["success"] is False
    assert result["status"] == "fix_loop_stopped"
    assert result["failure_count"] == 2
    assert result["repeated_count"] == 2
    assert (
        "same failure diagnosis repeated"
        in result["error"]
    )
    assert result["tests_verified"] is False

    # Exactly one test run per step, no duplicate execution.
    assert len(executions) == 2
    # Exactly one plan fetched per step as well.
    assert provider.calls == 2

    assert len(history_results(result, "diagnosis")) == 2
    assert (
        len(history_results(result, "fix_loop_stopped")) == 1
    )


def test_passing_tests_verify_then_complete(
    tmp_path, monkeypatch
):
    loop, provider = make_loop(
        tmp_path,
        [
            dict(RUN_TESTS_PLAN, actions=[{"tool": "run_tests"}]),
            dict(COMPLETE_PLAN),
        ],
    )

    monkeypatch.setattr(
        loop.engine.tests, "run_pytest", passing_suite()
    )

    result = loop.run("Verify and finish.")

    assert result["success"] is True
    assert result["status"] == "completed"
    assert result["context"]["tests_verified"] is True
    assert history_results(result, "diagnosis") == []
    assert provider.calls == 2


def test_blocked_tool_action_does_not_crash_the_loop(
    tmp_path, monkeypatch
):
    """A tool failure is recorded and the loop keeps going."""

    loop, provider = make_loop(
        tmp_path,
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
            dict(RUN_TESTS_PLAN, actions=[{"tool": "run_tests"}]),
            dict(COMPLETE_PLAN),
        ],
    )

    monkeypatch.setattr(
        loop.engine.tests, "run_pytest", passing_suite()
    )

    result = loop.run("Survive a bad action.")

    assert result["status"] == "completed"
    assert result["context"]["tests_verified"] is True

    blocked = history_results(result, "blocked")
    assert len(blocked) == 1
    assert blocked[0]["success"] is False

    # The blocked write never touched the filesystem.
    assert not (tmp_path.parent / "evil.py").exists()
    assert provider.calls == 3
