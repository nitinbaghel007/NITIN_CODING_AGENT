import subprocess
from pathlib import Path


class TerminalTool:
    """Safely execute approved commands inside the agent workspace."""

    ALLOWED_COMMANDS = {
        "python",
        "pytest",
        "git",
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

        if first_word in self.BLOCKED_COMMANDS:
            raise PermissionError(
                f"Command blocked for safety: {first_word}"
            )

        if first_word not in self.ALLOWED_COMMANDS:
            raise PermissionError(
                f"Command not allowed: {first_word}"
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