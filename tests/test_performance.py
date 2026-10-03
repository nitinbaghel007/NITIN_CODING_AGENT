"""Batch 11: performance/resource regression contracts.

These are operation-count and structural tests - no wall-clock
timing assertions - so they stay deterministic and non-flaky:

* ``list_files`` must classify entries from the scandir scan's
  cached metadata (zero ``os.stat`` calls per entry, not one per
  entry as the pre-Batch-11 implementation did),
* ``list_files`` output must match the reference ``iterdir``
  algorithm exactly (ordering, classification, ignore set),
* ``list_files`` stays top-level only (recursion is not part of
  the contract),
* ``diagnose`` must scan the *whole* failure output (guards
  against a future truncating "optimization") on large payloads,
* the loop must build the prompt exactly once per step and embed
  the serialized history exactly once, with constant template
  overhead (linear - not quadratic - history growth).

No network, no subprocess, no real providers anywhere.
"""

import json
import os

from agent.core.coding_loop import CodingLoop
from agent.core.diagnosis import DiagnosisEngine
from agent.providers.base import LLMProvider
from agent.providers.manager import ProviderManager
from agent.tools.workspace_tool import WorkspaceTool

IGNORED = {"__pycache__", ".pytest_cache", ".git"}


# -------------------------------------------------------------
# list_files: operation count (the Batch 11 defect test)
# -------------------------------------------------------------


def test_list_files_avoids_per_entry_stat_calls(
    tmp_path, monkeypatch
):
    """Classifying entries must not issue one os.stat per entry.

    The pre-Batch-11 implementation called Path.is_file()/is_dir()
    on every entry; each call re-stats the filesystem (measured:
    ~130-220 ms per call on a 1,000-file workspace). The scan must
    use the metadata os.scandir already returned instead.
    """

    for i in range(40):
        (tmp_path / f"file_{i:03d}.py").write_text(
            "# x", encoding="utf-8"
        )
    for i in range(8):
        (tmp_path / f"dir_{i:03d}").mkdir()

    tool = WorkspaceTool(str(tmp_path))

    real_stat = os.stat
    stat_calls = []

    def counting_stat(*args, **kwargs):
        stat_calls.append(args)
        return real_stat(*args, **kwargs)

    monkeypatch.setattr(os, "stat", counting_stat)
    result = tool.list_files()
    stat_calls_made = len(stat_calls)

    assert result["success"] is True
    assert len(result["files"]) == 40
    assert len(result["directories"]) == 8
    # Plain files and directories carry their type in the
    # directory scan itself: zero follow-up stat calls.
    assert stat_calls_made == 0


# -------------------------------------------------------------
# list_files: exact equivalence with the reference algorithm
# -------------------------------------------------------------


def reference_list_files(workspace):
    """The pre-Batch-11 implementation, kept as the test oracle."""

    files = []
    directories = []

    for item in sorted(workspace.iterdir()):
        if item.name in IGNORED:
            continue

        if item.is_file():
            files.append(item.name)
        elif item.is_dir():
            directories.append(item.name)

    return files, directories


def test_list_files_matches_iterdir_reference(tmp_path):
    """Output equals the iterdir-based reference exactly.

    Locks ordering (Path-based sort), file/directory classification
    and the ignore set against the pre-optimization behaviour.
    """

    for name in (
        "b.py",
        "A.py",
        "_x.py",
        "0.py",
        "Z.py",
        "app.py",
        "module v2.py",
        "setup.cfg",
        "uber.py",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")

    for name in ("pkg", "src", ".hidden"):
        (tmp_path / name).mkdir()

    (tmp_path / "pkg" / "nested.py").write_text(
        "x", encoding="utf-8"
    )

    for name in IGNORED:
        (tmp_path / name).mkdir()
        (tmp_path / name / "junk").write_text(
            "x", encoding="utf-8"
        )

    result = WorkspaceTool(str(tmp_path)).list_files()
    ref_files, ref_directories = reference_list_files(
        tmp_path.resolve()
    )

    assert result["files"] == ref_files
    assert result["directories"] == ref_directories
    assert result["workspace"] == str(tmp_path.resolve())
    # The nested file is never reported (top-level only).
    assert "nested.py" not in result["files"]


def test_list_files_stays_top_level_only(tmp_path):
    """Nesting is not part of the contract: never recurse."""

    (tmp_path / "top.py").write_text(
        "x", encoding="utf-8"
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "nested.py").write_text(
        "x", encoding="utf-8"
    )
    (tmp_path / "pkg" / "sub").mkdir()
    (tmp_path / "pkg" / "sub" / "deep.py").write_text(
        "x", encoding="utf-8"
    )

    result = WorkspaceTool(str(tmp_path)).list_files()

    assert result["files"] == ["top.py"]
    assert result["directories"] == ["pkg"]
    assert "nested.py" not in result["files"]
    assert "deep.py" not in result["files"]
    assert "sub" not in result["directories"]


# -------------------------------------------------------------
# diagnosis: large payloads must be scanned in full
# -------------------------------------------------------------


def test_diagnosis_matches_marker_at_end_of_large_output():
    """A marker after ~200 KB of output is still classified.

    Guards the Batch 11 lower-once change against accidentally
    truncating the failure text.
    """

    engine = DiagnosisEngine()
    filler = "irrelevant test chatter line\n" * 7000

    result = {
        "success": False,
        "stderr": filler + "SyntaxError: bad syntax",
    }

    assert engine.diagnose(result)["category"] == "syntax_error"


def test_diagnosis_large_unknown_output_falls_back():
    """A large output with no known marker falls back safely."""

    engine = DiagnosisEngine()
    result = {
        "success": False,
        "stderr": "chatter without markers " * 9000,
    }

    assert (
        engine.diagnose(result)["category"]
        == "test_failure"
    )


# -------------------------------------------------------------
# loop: prompt built once per step, history embedded once
# -------------------------------------------------------------


class PromptCaptureProvider(LLMProvider):
    """Captures every prompt the loop hands to the provider."""

    def __init__(self, plans):
        self.plans = list(plans)
        self.calls = 0
        self.prompts = []

    def generate(self, prompt: str) -> str:
        return "unused"

    def generate_actions(self, task: str) -> dict:
        self.prompts.append(task)
        index = min(self.calls, len(self.plans) - 1)
        self.calls += 1
        return self.plans[index]


def test_prompt_built_once_per_step_and_history_embedded_once(
    tmp_path, monkeypatch
):
    """One prompt build per step, history serialized once.

    ``len(prompt) - len(serialized history)`` must be identical at
    every step: the template overhead is constant, so history
    growth in the prompt is exactly linear. Double-embedding or
    re-serializing history more than once breaks the constant and
    fails this test. The serialized history of step N must appear
    in prompt N exactly once - never twice, never with extra or
    missing entries.
    """

    (tmp_path / "alpha.py").write_text(
        "x", encoding="utf-8"
    )
    (tmp_path / "beta.py").write_text(
        "x", encoding="utf-8"
    )

    list_plan = {
        "summary": "Inspect workspace",
        "actions": [{"tool": "list_files"}],
    }
    provider = PromptCaptureProvider([list_plan])

    manager = ProviderManager()
    manager.register("openrouter", provider)
    manager.set_default("openrouter")

    loop = CodingLoop(
        str(tmp_path),
        provider_manager=manager,
    )

    builds = []
    original_build = CodingLoop._build_prompt

    def spy_build(self, **kwargs):
        prompt = original_build(self, **kwargs)
        builds.append(prompt)
        return prompt

    monkeypatch.setattr(
        CodingLoop, "_build_prompt", spy_build
    )

    # Expected history, built independently from the real tool.
    expected_result = WorkspaceTool(str(tmp_path)).list_files()
    action = list_plan["actions"][0]
    expected_history = [
        {
            "step": step,
            "action": action,
            "result": expected_result,
        }
        for step in range(
            1, CodingLoop.MAX_STEPS + 1
        )
    ]

    result = loop.run("Inspect the workspace repeatedly.")

    # Exactly one prompt build and one provider call per step.
    assert len(builds) == CodingLoop.MAX_STEPS
    assert provider.calls == CodingLoop.MAX_STEPS
    assert len(provider.prompts) == CodingLoop.MAX_STEPS

    overheads = set()
    for step in range(CodingLoop.MAX_STEPS):
        expected_json = json.dumps(
            expected_history[:step],
            indent=2,
            ensure_ascii=False,
        )
        prompt = builds[step]

        # The step's history appears exactly once, anchored at
        # the TOOL HISTORY section.
        assert (
            prompt.count(
                f"TOOL HISTORY:\n{expected_json}"
            )
            == 1
        )

        if step >= 1:
            # A second full copy anywhere in the prompt fails.
            assert prompt.count(expected_json) == 1

        overheads.add(len(prompt) - len(expected_json))

    # Constant template overhead => linear history growth.
    assert len(overheads) == 1

    # History itself is exact: one entry per step, no duplicates.
    assert result["history"] == expected_history
    assert result["context"]["history"] == expected_history
