from agent.tools.file_tool import FileTool
from agent.tools.terminal_tool import TerminalTool
from agent.tools.test_tool import TestTool
from agent.tools.workspace_tool import WorkspaceTool
from agent.core.safety import SafetyLevel, SafetyManager


class ActionEngine:
    """Execute validated coding actions using agent tools."""

    ALLOWED_TOOLS = {
        "list_files",
        "write_file",
        "read_file",
        "run_command",
        "run_tests",
    }

    def __init__(
        self,
        workspace: str,
        approval_callback=None,
    ):
        self.workspace = workspace

        self.files = FileTool(workspace)
        self.terminal = TerminalTool(workspace)
        self.tests = TestTool(workspace)
        self.workspace_tool = WorkspaceTool(workspace)

        self.safety = SafetyManager(workspace)
        self.approval_callback = approval_callback

    def execute(self, action: dict) -> dict:
        """Execute a single action after safety validation."""

        decision = self.safety.validate(action)

        if decision.level == SafetyLevel.BLOCKED:
            return {
                "success": False,
                "status": "blocked",
                "safety_level": decision.level.value,
                "error": decision.reason,
            }

        if decision.level == SafetyLevel.APPROVAL:
            approved = False

            if self.approval_callback is not None:
                approved = bool(
                    self.approval_callback(
                        action,
                        decision,
                    )
                )

            if not approved:
                return {
                    "success": False,
                    "status": "approval_required",
                    "safety_level": decision.level.value,
                    "error": decision.reason,
                    "action": action,
                }

        if not isinstance(action, dict):
            return {
                "success": False,
                "error": "Action must be a dictionary.",
            }

        tool = action.get("tool")

        if tool not in self.ALLOWED_TOOLS:
            return {
                "success": False,
                "error": f"Unknown or disallowed tool: {tool}",
            }

        # -----------------------------------------------------
        # LIST WORKSPACE FILES
        # -----------------------------------------------------

        if tool == "list_files":
            return self.workspace_tool.list_files()

        # -----------------------------------------------------
        # WRITE FILE
        # -----------------------------------------------------

        if tool == "write_file":
            path = action.get("path")
            content = action.get("content")

            if not path:
                return {
                    "success": False,
                    "error": "write_file requires path.",
                }

            if content is None:
                return {
                    "success": False,
                    "error": "write_file requires content.",
                }

            try:
                self.files.write(path, content)

                return {
                    "success": True,
                    "tool": "write_file",
                    "path": path,
                }

            except Exception as exc:
                return {
                    "success": False,
                    "tool": "write_file",
                    "path": path,
                    "error": str(exc),
                }

        # -----------------------------------------------------
        # READ FILE
        # -----------------------------------------------------

        if tool == "read_file":
            path = action.get("path")

            if not path:
                return {
                    "success": False,
                    "error": "read_file requires path.",
                }

            try:
                content = self.files.read(path)

                return {
                    "success": True,
                    "tool": "read_file",
                    "path": path,
                    "content": content,
                }

            except Exception as exc:
                return {
                    "success": False,
                    "tool": "read_file",
                    "path": path,
                    "error": str(exc),
                }

        # -----------------------------------------------------
        # RUN COMMAND
        # -----------------------------------------------------

        if tool == "run_command":
            command = action.get("command")

            if not command:
                return {
                    "success": False,
                    "error": "run_command requires command.",
                }

            result = self.terminal.run(command)

            return {
                "success": result["success"],
                "tool": "run_command",
                "command": command,
                "result": result,
            }

        # -----------------------------------------------------
        # RUN TESTS
        # -----------------------------------------------------

        if tool == "run_tests":
            result = self.tests.run_pytest()

            return {
                "success": result["success"],
                "tool": "run_tests",
                "result": result,
            }

        return {
            "success": False,
            "error": "Action could not be executed.",
        }

    # ---------------------------------------------------------
    # EXECUTE MULTIPLE ACTIONS
    # ---------------------------------------------------------

    def execute_actions(self, actions: list[dict]) -> list[dict]:
        """Execute multiple actions sequentially."""

        if not isinstance(actions, list):
            return [
                {
                    "success": False,
                    "error": "Actions must be a list.",
                }
            ]

        results = []

        for index, action in enumerate(actions, start=1):
            result = self.execute(action)

            result["action_number"] = index

            results.append(result)

            if not result["success"]:
                break

        return results