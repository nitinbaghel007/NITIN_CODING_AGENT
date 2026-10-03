import subprocess
from pathlib import Path


class TestTool:
    """Run the project's test suite inside the agent workspace."""

    def __init__(self, workspace: str):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run_pytest(self, timeout: int = 120) -> dict:
        try:
            result = subprocess.run(
                ["python", "-m", "pytest"],
                cwd=self.workspace,
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

        except subprocess.TimeoutExpired:
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": "Pytest timed out.",
                "success": False,
            }