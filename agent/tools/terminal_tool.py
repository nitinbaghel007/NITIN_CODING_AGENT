import re
import subprocess
from pathlib import Path


class TerminalTool:
    """Safely execute approved commands inside the agent workspace."""

    ALLOWED_COMMANDS = {
        "python",
        "pytest",
        "git",
        # pip/npm are never SAFE in SafetyManager: they always go
        # through approval first, and approval must be able to reach
        # the execution layer afterwards.
        "pip",
        "npm",
    }

    BLOCKED_COMMANDS = {
        "del",
        "erase",
        "rmdir",
        "rd",
        "format",
        "shutdown",
        "restart",
        "taskkill",
        "diskpart",
        "powershell",
        "cmd",
    }

    def __init__(self, workspace: str):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _validate_command(self, command: str) -> None:
        command_lower = command.strip().lower()

        if not command_lower:
            raise ValueError("Command cannot be empty.")

        first_word = command_lower.split()[0]

        # Match on the executable itself, not its file extension, so
        # "python.exe" is the same executable as "python" and
        # "PowerShell.exe" cannot dodge the blocked list.
        executable = re.sub(
            r"\.(exe|cmd|bat|com|ps1)$",
            "",
            first_word,
        )

        if executable in self.BLOCKED_COMMANDS:
            raise PermissionError(
                f"Command blocked for safety: {executable}"
            )

        if executable not in self.ALLOWED_COMMANDS:
            raise PermissionError(
                f"Command not allowed: {executable}"
            )

    def run(self, command: str, timeout: int = 60) -> dict:
        try:
            self._validate_command(command)

            result = subprocess.run(
                command,
                cwd=self.workspace,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

        except PermissionError as exc:
            return {
                "return_code": -2,
                "stdout": "",
                "stderr": str(exc),
                "success": False,
            }

        except subprocess.TimeoutExpired:
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": "Command timed out.",
                "success": False,
            }