import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SafetyLevel(str, Enum):
    """Safety classification for an agent action."""

    SAFE = "safe"
    APPROVAL = "approval"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class SafetyDecision:
    """Result of validating one agent action."""

    level: SafetyLevel
    reason: str


class SafetyManager:
    """Validate coding-agent actions before execution."""

    SAFE_TOOLS = {
        "list_files",
        "read_file",
        "write_file",
        "run_tests",
    }

    APPROVAL_COMMAND_PREFIXES = {
        "pip install",
        "pip uninstall",
        "python -m pip install",
        "python -m pip uninstall",
        "git add",
        "git commit",
        "git checkout",
        "git restore",
        "git clean",
        "git reset",
        "git push",
        "git pull",
        "npm install",
        "npm uninstall",
        "npm update",
    }

    SAFE_COMMANDS = {
        "python --version",
        "python -m pytest",
        "pytest",
        "pytest -q",
        "pytest -v",
        "git status",
        "git diff",
        "git log",
    }

    BLOCKED_COMMAND_MARKERS = (
        "format",
        "shutdown",
        "restart-computer",
        "stop-computer",
        "diskpart",
        "cipher /w",
        "vssadmin delete",
        "bcdedit",
        "takeown",
        "icacls ",
        "reg delete",
        "reg add",
        "net user",
        "net localgroup",
        "del ",
        "erase ",
        "rmdir ",
        "rd ",
        "rm ",
        "remove-item",
        "remove-itemproperty",
        "set-content ",
        "clear-content ",
        "invoke-expression",
        "iex ",
    )

    SHELL_CONTROL_CHARS = (
        "&",
        "|",
        ";",
        ">",
        "<",
        "`",
    )

    def __init__(self, workspace: str):
        self.workspace = Path(workspace).resolve()

    def validate(self, action: dict) -> SafetyDecision:
        """Classify an action as safe, approval-required, or blocked."""

        if not isinstance(action, dict):
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Action must be a JSON object.",
            )

        tool = action.get("tool")

        if not isinstance(tool, str) or not tool.strip():
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Action is missing a valid tool.",
            )

        if tool in self.SAFE_TOOLS:
            if tool in {"read_file", "write_file"}:
                return self._validate_path_action(action)

            return SafetyDecision(
                SafetyLevel.SAFE,
                f"{tool} is an approved safe tool.",
            )

        if tool == "run_command":
            return self._validate_command(
                action.get("command")
            )

        return SafetyDecision(
            SafetyLevel.BLOCKED,
            f"Unknown tool is blocked: {tool}",
        )

    def _validate_path_action(
        self,
        action: dict,
    ) -> SafetyDecision:
        path_value = action.get("path")

        if not isinstance(path_value, str) or not path_value.strip():
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "File action requires a relative path.",
            )

        path_text = path_value.strip()

        if Path(path_text).is_absolute():
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Absolute paths are blocked.",
            )

        if re.match(
            r"^[A-Za-z]:[\\/]",
            path_text,
        ):
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Windows absolute paths are blocked.",
            )

        candidate = (
            self.workspace / Path(path_text)
        ).resolve()

        try:
            candidate.relative_to(self.workspace)
        except ValueError:
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Path escapes the workspace.",
            )

        return SafetyDecision(
            SafetyLevel.SAFE,
            "Path stays inside the workspace.",
        )

    def _validate_command(
        self,
        command: object,
    ) -> SafetyDecision:
        if not isinstance(command, str) or not command.strip():
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "run_command requires a non-empty command.",
            )

        normalized = re.sub(
            r"\s+",
            " ",
            command.strip().lower(),
        )

        for marker in self.BLOCKED_COMMAND_MARKERS:
            if (
                normalized == marker
                or normalized.startswith(marker + " ")
                or marker in normalized
            ):
                return SafetyDecision(
                    SafetyLevel.BLOCKED,
                    f"Dangerous command pattern is blocked: {marker}",
                )

        if any(
            char in command
            for char in self.SHELL_CONTROL_CHARS
        ):
            return SafetyDecision(
                SafetyLevel.APPROVAL,
                "Shell control characters require approval.",
            )

        if normalized in self.SAFE_COMMANDS:
            return SafetyDecision(
                SafetyLevel.SAFE,
                "Command is on the safe command allowlist.",
            )

        for prefix in self.APPROVAL_COMMAND_PREFIXES:
            if normalized == prefix or normalized.startswith(
                prefix + " "
            ):
                return SafetyDecision(
                    SafetyLevel.APPROVAL,
                    "Command can change the environment or repository and requires approval.",
                )

        return SafetyDecision(
            SafetyLevel.APPROVAL,
            "Command is not on the safe allowlist.",
        )
