import json
import logging

from agent.core.action_engine import ActionEngine
from agent.core.approval import ApprovalManager
from agent.core.context_manager import ContextManager
from agent.core.diagnosis import DiagnosisEngine
from agent.core.error_recovery import ErrorRecoveryManager
from agent.core.fix_loop import FixLoopGuard
from agent.providers.base import LLMProvider
from agent.providers.failures import (
    CATEGORY_RATE_LIMIT,
    FAILOVER_CATEGORIES,
    RATE_LIMIT_PREFIXES,
    classify_provider_error,
)
from agent.providers.gemini import GeminiProvider
from agent.providers.manager import ProviderManager
from agent.providers.openrouter import OpenRouterProvider

# Part of the Batch 6 "agent" logger hierarchy (central config).
logger = logging.getLogger(__name__)


class LazyProvider(LLMProvider):
    """Create the real provider only when needed."""

    def __init__(self, provider_factory):
        self._provider_factory = provider_factory
        self._provider = None

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = self._provider_factory()

        return self._provider

    def generate(self, prompt: str) -> str:
        """Generate a normal model response."""

        return self._get_provider().generate(
            prompt
        )

    def generate_actions(
        self,
        task: str,
    ) -> dict:
        """Generate structured coding actions."""

        return self._get_provider().generate_actions(
            task
        )


class CodingLoop:
    """Reliable AI coding loop with provider management."""

    MAX_STEPS = 10

    def __init__(
        self,
        workspace: str,
        provider_manager: ProviderManager | None = None,
        approval_manager: ApprovalManager | None = None,
        approval_callback=None,
    ):
        self.workspace = workspace

        self.provider_manager = (
            provider_manager
            if provider_manager is not None
            else self._create_default_provider_manager()
        )

        self.approval_manager = (
            approval_manager
            if approval_manager is not None
            else ApprovalManager()
        )

        if approval_callback is None:
            approval_callback = (
                self.approval_manager.request
            )

        self.engine = ActionEngine(
            workspace,
            approval_callback=approval_callback,
        )

        self.context_manager: ContextManager | None = None
        self.diagnosis_engine = DiagnosisEngine()
        self.fix_loop_guard = FixLoopGuard()

    def _create_default_provider_manager(
        self,
    ) -> ProviderManager:
        """Create the default provider manager."""

        manager = ProviderManager()

        manager.register(
            "openrouter",
            LazyProvider(OpenRouterProvider),
        )

        manager.register(
            "gemini",
            LazyProvider(GeminiProvider),
        )

        manager.set_default(
            "openrouter"
        )

        return manager

    def _get_provider(self):
        """Get the currently selected provider."""

        return self.provider_manager.get()

    def _build_prompt(
        self,
        task: str,
        history: list[dict],
        tests_verified: bool,
        changes_since_test: bool,
    ) -> str:
        history_text = json.dumps(
            history,
            indent=2,
            ensure_ascii=False,
        )

        verification_state = (
            "YES - tests have already passed."
            if tests_verified
            else "NO - tests have not yet passed."
        )

        change_state = (
            "YES - files changed after the last test."
            if changes_since_test
            else "NO - no file changes since the last test."
        )

        return f"""
You are Nitin Coding Agent.

WORKSPACE:
{self.workspace}

USER TASK:
{task}

TEST VERIFICATION STATE:
{verification_state}

CHANGES SINCE LAST TEST:
{change_state}

TOOL HISTORY:
{history_text}

Your job is to complete the coding task using
the available workspace tools.

AVAILABLE TOOLS:

1. list_files
{{
  "tool": "list_files"
}}

2. read_file
{{
  "tool": "read_file",
  "path": "relative/path.py"
}}

3. write_file
{{
  "tool": "write_file",
  "path": "relative/path.py",
  "content": "complete file content"
}}

4. run_command
{{
  "tool": "run_command",
  "command": "python --version"
}}

5. run_tests
{{
  "tool": "run_tests"
}}

RULES:
- Return ONLY valid JSON.
- Do NOT use Markdown.
- Do NOT use ```json.
- Use exactly ONE action per response.
- Put the action inside the "actions" list.
- Use list_files instead of ls or dir.
- Paths must be relative.
- Never use absolute paths.
- Never access files outside the workspace.
- Never use dangerous commands.
- Inspect existing files before modifying them.
- After code changes, run tests.
- If tests fail, diagnose and fix the actual problem.
- After a failed test, inspect the diagnosed affected area before changing code.
- Make a targeted fix based on the diagnosis.
- Run tests again after the fix.
- Do not repeat the same failed fix without changing your diagnosis or approach.
- Do not rewrite unrelated files.

IMPORTANT COMPLETION RULE:
If the task is already complete AND tests have passed,
return:
{{
  "summary": "Task completed",
  "actions": []
}}

Do NOT run the same tests repeatedly when:
- tests have already passed,
- no files have changed after the successful test,
- and the task requirements are already satisfied.

Only run tests again if code or tests changed after
the previous successful test.

If tests have passed but the task is NOT complete,
continue working on the missing requirement.

RESPONSE FORMAT:
{{
  "summary": "what you are doing now",
  "actions": [
    {{
      "tool": "...",
      "path": "...",
      "content": "..."
    }}
  ]
}}
"""

    def _execute_action(
        self,
        action: dict,
    ) -> dict:
        print("\nAction:")
        print(
            json.dumps(
                action,
                indent=2,
                ensure_ascii=False,
            )
        )

        result = self.engine.execute(
            action
        )

        print("\nTool Result:")
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        return result

    def _tests_passed(
        self,
        result: dict,
    ) -> bool:
        if not result.get("success"):
            return False

        if result.get("tool") != "run_tests":
            return False

        test_result = result.get(
            "result",
            {},
        )

        return (
            test_result.get("success") is True
            and test_result.get("return_code") == 0
        )

    @staticmethod
    def _validate_plan_shape(plan) -> None:
        """Reject a malformed provider plan with a controlled error.

        A wrong-shaped plan must surface as a provider response
        failure (classified, failover-eligible), never as an
        obscure AttributeError while unrelated code processes it.
        """

        if not isinstance(plan, dict):
            raise ValueError(
                "Provider returned an unexpected response shape "
                "(plan is not an object)."
            )

        actions = plan.get("actions")

        if actions is not None and not isinstance(
            actions, list
        ):
            raise ValueError(
                "Provider returned an unexpected response shape "
                "('actions' is not a list)."
            )

    def _generate_plan_with_recovery(
        self,
        provider,
        prompt: str,
        provider_name: str,
    ) -> dict:
        '''Generate a coding plan with bounded transient-error recovery.'''

        recovery = ErrorRecoveryManager()
        attempt = 0

        while True:
            try:
                plan = provider.generate_actions(prompt)
            except Exception as exc:
                error_message = str(exc)

                if error_message.startswith(
                    RATE_LIMIT_PREFIXES
                ):
                    raise

                if not recovery.should_retry(
                    error_message,
                    attempt,
                ):
                    raise

                attempt += 1

                delay = recovery.retry_delay(
                    attempt
                )

                print(
                    "\n[AI RETRY]"
                )
                print(
                    f"Provider: {provider_name}"
                )
                print(
                    f"Transient error: {error_message}"
                )
                print(
                    f"Retry {attempt}/{recovery.MAX_RETRIES} "
                    f"in {delay} seconds..."
                )

                recovery.sleep(delay)
                continue

            self._validate_plan_shape(plan)
            return plan

    def _mark_provider_status(
        self,
        provider_name: str,
        status: str,
    ) -> None:
        """Set a provider status only when it is registered.

        A misconfigured default (no such registered provider) must
        not raise inside the failure handler: the run still has to
        end in a controlled result instead of crashing.
        """

        if (
            provider_name
            in self.provider_manager.providers
        ):
            self.provider_manager.set_status(
                provider_name,
                status,
            )

    def _record_failover(
        self,
        provider_name: str,
        next_provider: str,
        error_message: str,
    ) -> None:
        """Switch the run to the fallback provider after a failure.

        Shared by the pre-existing rate-limit path and the Batch 7
        failure-category path so bookkeeping, history, logging and
        the user-visible notice stay identical for both.
        """

        failover_result = {
            "success": False,
            "status": "provider_failover",
            "from_provider": provider_name,
            "to_provider": next_provider,
            "error": error_message,
        }

        self.context_manager.add_history(
            None,
            failover_result,
        )

        self.context_manager.set_provider(
            next_provider
        )
        self.context_manager.clear_error()
        self.context_manager.set_status(
            "running"
        )

        logger.info(
            "provider failover: from=%s to=%s",
            provider_name,
            next_provider,
        )

        print(
            "\n========================================"
        )
        print(
            " PROVIDER FAILOVER"
        )
        print(
            "========================================"
        )
        print(
            f"\nFailed provider: {provider_name}"
        )
        print(
            f"Next provider: {next_provider}"
        )
        print(
            "\nContinuing the same coding task "
            "with the free provider."
        )

    def run(
        self,
        task: str,
    ) -> dict:
        print("\n========================================")
        print(" NITIN CODING AGENT")
        print("========================================")

        print("\nTask:")
        print(task)

        provider_name = (
            self.provider_manager.default_provider
        )

        self.context_manager = ContextManager(
            task=task,
            workspace=self.workspace,
            provider=provider_name,
        )

        context = self.context_manager.get()

        for step in range(
            1,
            self.MAX_STEPS + 1,
        ):
            self.context_manager.set_step(
                step
            )

            provider_name = (
                self.provider_manager.default_provider
            )

            self.context_manager.set_provider(
                provider_name
            )

            context = self.context_manager.get()

            print(
                f"\n========== STEP {step}/{self.MAX_STEPS} =========="
            )

            prompt = self._build_prompt(
                task=task,
                history=context.history,
                tests_verified=context.tests_verified,
                changes_since_test=(
                    context.changes_since_test
                ),
            )

            try:
                provider = self._get_provider()

                plan = self._generate_plan_with_recovery(
                    provider,
                    prompt,
                    provider_name,
                )

            except Exception as exc:
                error_message = str(exc)
                category = classify_provider_error(exc)

                self.context_manager.set_error(
                    error_message
                )

                logger.warning(
                    "provider failure: provider=%s "
                    "category=%s error=%s",
                    provider_name,
                    category,
                    error_message,
                )

                if category == CATEGORY_RATE_LIMIT:
                    self._mark_provider_status(
                        provider_name,
                        "rate_limited",
                    )

                    if (
                        provider_name
                        in ProviderManager.FAILOVER_ORDER
                    ):
                        next_provider = (
                            self.provider_manager.failover(
                                provider_name
                            )
                        )
                    else:
                        next_provider = None

                    if next_provider is not None:
                        self._record_failover(
                            provider_name,
                            next_provider,
                            error_message,
                        )
                        continue

                    logger.error(
                        "no fallback provider available: "
                        "provider=%s category=%s",
                        provider_name,
                        category,
                    )

                    self.context_manager.set_status(
                        "rate_limit"
                    )

                    print(
                        "\n========================================"
                    )
                    print(
                        " ALL FREE PROVIDERS EXHAUSTED"
                    )
                    print(
                        "========================================"
                    )
                    print(
                        "\nNo available free provider remains."
                    )
                    print(
                        "\nAgent stopped safely."
                    )

                    return {
                        "success": False,
                        "status": "rate_limit",
                        "provider": provider_name,
                        "provider_status": (
                            self.provider_manager.get_status(
                                provider_name
                            )
                        ),
                        "step": step,
                        "tests_verified": (
                            context.tests_verified
                        ),
                        "error": error_message,
                        "history": list(
                            context.history
                        ),
                        "context": (
                            self.context_manager.snapshot()
                        ),
                    }

                if category in FAILOVER_CATEGORIES:
                    self._mark_provider_status(
                        provider_name,
                        "error",
                    )

                    next_provider = (
                        self.provider_manager.failover(
                            provider_name
                        )
                        if provider_name
                        in ProviderManager.FAILOVER_ORDER
                        else None
                    )

                    if next_provider is not None:
                        self._record_failover(
                            provider_name,
                            next_provider,
                            error_message,
                        )
                        continue

                    logger.error(
                        "no fallback provider available: "
                        "provider=%s category=%s",
                        provider_name,
                        category,
                    )

                self._mark_provider_status(
                    provider_name,
                    "error",
                )

                self.context_manager.mark_failed(
                    error_message
                )

                logger.error(
                    "provider run failed: provider=%s "
                    "category=%s outcome=ai_error",
                    provider_name,
                    category,
                )

                print("\n[AI ERROR]")
                print(error_message)

                return {
                    "success": False,
                    "status": "ai_error",
                    "provider": provider_name,
                    "provider_status": (
                        self.provider_manager.get_status(
                            provider_name
                        )
                        if provider_name
                        in self.provider_manager.providers
                        else "error"
                    ),
                    "step": step,
                    "error": error_message,
                    "history": list(
                        context.history
                    ),
                    "context": (
                        self.context_manager.snapshot()
                    ),
                }

            else:
                self.provider_manager.set_status(
                    provider_name,
                    "available",
                )

                self.context_manager.clear_error()
                self.context_manager.set_status(
                    "running"
                )

            print("\nAI:")
            print(
                plan.get(
                    "summary",
                    "No summary",
                )
            )

            actions = plan.get(
                "actions",
                [],
            )

            if not actions:
                if context.tests_verified:
                    print(
                        "\n[VERIFIED] "
                        "Tests passed and AI completed "
                        "the task."
                    )

                    completion_result = {
                        "success": True,
                        "status": "completed",
                        "summary": plan.get(
                            "summary",
                            "Task completed",
                        ),
                    }

                    self.context_manager.add_history(
                        None,
                        completion_result,
                    )

                    self.context_manager.mark_completed()

                    return {
                        "success": True,
                        "status": "completed",
                        "provider": provider_name,
                        "steps": step,
                        "history": list(
                            context.history
                        ),
                        "context": (
                            self.context_manager.snapshot()
                        ),
                    }

                print(
                    "\n[WARNING] "
                    "AI claimed completion without "
                    "verified tests."
                )

                completion_result = {
                    "success": False,
                    "error": (
                        "Completion rejected. "
                        "Tests have not been verified."
                    ),
                }

                self.context_manager.add_history(
                    None,
                    completion_result,
                )

                continue

            action = actions[0]

            tool = action.get(
                "tool"
            )

            context = self.context_manager.get()

            if (
                tool == "run_tests"
                and context.tests_verified
                and not context.changes_since_test
            ):
                print(
                    "\n[SKIP] Tests already passed "
                    "and no changes were made."
                )

                skipped_result = {
                    "success": True,
                    "skipped": True,
                    "reason": (
                        "Tests already passed and "
                        "no changes occurred."
                    ),
                }

                self.context_manager.add_history(
                    action,
                    skipped_result,
                )

                continue

            result = self._execute_action(
                action
            )

            self.context_manager.add_history(
                action,
                result,
            )

            if tool == "read_file":
                if result.get("success"):
                    path = action.get(
                        "path",
                        "",
                    )

                    self.context_manager.record_file_inspected(
                        path
                    )

            if tool == "write_file":
                if result.get("success"):
                    path = action.get(
                        "path",
                        "",
                    )

                    self.context_manager.record_file_changed(
                        path
                    )

                    print(
                        "\n[STATE] "
                        "Code changed. Previous test "
                        "verification is invalid."
                    )

            if tool == "run_tests":
                tests_verified = self._tests_passed(
                    result
                )

                self.context_manager.record_test_run(
                    tests_verified
                )

                if tests_verified:
                    self.fix_loop_guard.record_success()

                    print(
                        "\n[TESTS PASS] "
                        "Test suite verified successfully."
                    )
                else:
                    diagnosis = (
                        self.diagnosis_engine.diagnose(
                            result
                        )
                    )

                    diagnosis_event = {
                        "success": False,
                        "status": "diagnosis",
                        "diagnosis": diagnosis,
                    }

                    self.context_manager.add_history(
                        None,
                        diagnosis_event,
                    )

                    print(
                        "\n[TESTS FAIL] "
                        "AI must diagnose and fix "
                        "the failure."
                    )

                    print(
                        "\n[DIAGNOSIS]"
                    )

                    print(
                        f"Category: "
                        f"{diagnosis['category']}"
                    )

                    print(
                        f"Summary: "
                        f"{diagnosis['summary']}"
                    )

                    print(
                        f"Likely cause: "
                        f"{diagnosis['likely_cause']}"
                    )

                    print(
                        f"Suggested action: "
                        f"{diagnosis['suggested_action']}"
                    )

                    fix_decision = (
                        self.fix_loop_guard.record_failure(
                            diagnosis
                        )
                    )

                    fix_event = {
                        "success": fix_decision.should_continue,
                        "status": (
                            "fix_attempt_allowed"
                            if fix_decision.should_continue
                            else "fix_loop_stopped"
                        ),
                        "failure_count": (
                            fix_decision.failure_count
                        ),
                        "repeated_count": (
                            fix_decision.repeated_count
                        ),
                        "reason": fix_decision.reason,
                    }

                    self.context_manager.add_history(
                        None,
                        fix_event,
                    )

                    if not fix_decision.should_continue:
                        self.context_manager.set_status(
                            "fix_loop_stopped"
                        )

                        print(
                            "\n[FIX LOOP STOPPED]"
                        )
                        print(
                            fix_decision.reason
                        )

                        return {
                            "success": False,
                            "status": "fix_loop_stopped",
                            "provider": provider_name,
                            "step": step,
                            "tests_verified": False,
                            "failure_count": (
                                fix_decision.failure_count
                            ),
                            "repeated_count": (
                                fix_decision.repeated_count
                            ),
                            "error": fix_decision.reason,
                            "history": list(
                                context.history
                            ),
                            "context": (
                                self.context_manager.snapshot()
                            ),
                        }

                    print(
                        "\n[FIX LOOP]"
                    )
                    print(
                        f"Autonomous fix attempt "
                        f"{fix_decision.failure_count}/"
                        f"{FixLoopGuard.MAX_FIX_ATTEMPTS}"
                    )

            if not result.get("success"):
                print(
                    "\n[TOOL FAILURE] "
                    "Returning failure to AI."
                )

        context = self.context_manager.get()

        self.context_manager.set_status(
            "max_steps_reached"
        )

        print(
            "\n[MAX STEPS REACHED] "
            "Agent stopped safely."
        )

        return {
            "success": False,
            "status": "max_steps_reached",
            "provider": (
                self.provider_manager.default_provider
            ),
            "steps": self.MAX_STEPS,
            "tests_verified": (
                context.tests_verified
            ),
            "history": list(
                context.history
            ),
            "context": (
                self.context_manager.snapshot()
            ),
        }


if __name__ == "__main__":
    loop = CodingLoop(
        "workspace"
    )

    result = loop.run(
        "Create a Python calculator with add "
        "and subtract functions and tests."
    )

    print(
        "\n========================================"
    )

    print(
        " FINAL RESULT"
    )

    print(
        "========================================"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )