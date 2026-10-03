from dataclasses import dataclass, field


@dataclass
class AgentContext:
    """Runtime state for a single coding task."""

    task: str
    workspace: str
    provider: str = ""
    current_step: int = 0
    status: str = "initialized"

    history: list[dict] = field(
        default_factory=list
    )

    files_inspected: list[str] = field(
        default_factory=list
    )

    files_changed: list[str] = field(
        default_factory=list
    )

    tests_run: int = 0
    tests_verified: bool = False
    changes_since_test: bool = False

    last_error: str | None = None


class ContextManager:
    """Manage runtime state for the coding agent."""

    def __init__(
        self,
        task: str,
        workspace: str,
        provider: str = "",
    ):
        self.context = AgentContext(
            task=task,
            workspace=workspace,
            provider=provider,
        )

    def set_provider(
        self,
        provider: str,
    ) -> None:
        self.context.provider = provider

    def set_step(
        self,
        step: int,
    ) -> None:
        if step < 0:
            raise ValueError(
                "Step cannot be negative."
            )

        self.context.current_step = step

    def set_status(
        self,
        status: str,
    ) -> None:
        if not status:
            raise ValueError(
                "Status cannot be empty."
            )

        self.context.status = status

    def add_history(
        self,
        action: dict | None,
        result: dict,
    ) -> None:
        self.context.history.append(
            {
                "step": self.context.current_step,
                "action": action,
                "result": result,
            }
        )

    def record_file_inspected(
        self,
        path: str,
    ) -> None:
        if (
            path
            and path not in self.context.files_inspected
        ):
            self.context.files_inspected.append(
                path
            )

    def record_file_changed(
        self,
        path: str,
    ) -> None:
        if (
            path
            and path not in self.context.files_changed
        ):
            self.context.files_changed.append(
                path
            )

        if path:
            self.context.changes_since_test = True
            self.context.tests_verified = False

    def record_test_run(
        self,
        passed: bool,
    ) -> None:
        self.context.tests_run += 1
        self.context.tests_verified = passed
        self.context.changes_since_test = False

    def set_error(
        self,
        error: str | None,
    ) -> None:
        self.context.last_error = error

    def clear_error(self) -> None:
        self.context.last_error = None

    def mark_completed(self) -> None:
        self.context.status = "completed"
        self.context.tests_verified = True
        self.context.changes_since_test = False

    def mark_failed(
        self,
        error: str | None = None,
    ) -> None:
        self.context.status = "failed"
        self.context.last_error = error

    def get(self) -> AgentContext:
        return self.context

    def snapshot(self) -> dict:
        return {
            "task": self.context.task,
            "workspace": self.context.workspace,
            "provider": self.context.provider,
            "current_step": self.context.current_step,
            "status": self.context.status,
            "history": list(
                self.context.history
            ),
            "files_inspected": list(
                self.context.files_inspected
            ),
            "files_changed": list(
                self.context.files_changed
            ),
            "tests_run": self.context.tests_run,
            "tests_verified": (
                self.context.tests_verified
            ),
            "changes_since_test": (
                self.context.changes_since_test
            ),
            "last_error": self.context.last_error,
        }

    def reset(self) -> None:
        self.context = AgentContext(
            task=self.context.task,
            workspace=self.context.workspace,
            provider=self.context.provider,
        )