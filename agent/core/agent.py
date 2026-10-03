from dataclasses import dataclass

from agent.providers.openrouter import OpenRouterProvider
from agent.tools.file_tool import FileTool
from agent.tools.terminal_tool import TerminalTool
from agent.tools.test_tool import TestTool


@dataclass
class AgentTask:
    """Represents a task given to Nitin Coding Agent."""

    request: str
    workspace: str


class NitinCodingAgent:
    """Core controller for the Nitin Coding Agent."""

    def __init__(self, workspace: str):
        self.workspace = workspace

        self.provider = OpenRouterProvider()
        self.files = FileTool(workspace)
        self.terminal = TerminalTool(workspace)
        self.tests = TestTool(workspace)

    def create_task(self, request: str) -> AgentTask:
        return AgentTask(
            request=request,
            workspace=self.workspace,
        )

    def plan(self, task: AgentTask) -> list[str]:
        return [
            "Understand the coding task",
            "Inspect the workspace",
            "Plan the required changes",
            "Implement the changes",
            "Run tests",
            "Fix failures if necessary",
            "Verify the final result",
        ]

    def ask_model(self, task: AgentTask) -> str:
        prompt = f"""
You are Nitin Coding Agent.

Workspace:
{task.workspace}

Coding task:
{task.request}

Create a concise implementation plan.
Do not modify files yet.
"""

        return self.provider.generate(prompt)

    def workspace_status(self) -> dict:
        return {
            "workspace": self.workspace,
            "files_tool": "ready",
            "terminal_tool": "ready",
            "test_tool": "ready",
            "provider": "OpenRouter",
        }

    def status(self) -> str:
        return "Nitin Coding Agent is ready."


if __name__ == "__main__":
    agent = NitinCodingAgent("workspace")

    print(agent.status())

    print("\nSystem Status:")
    for key, value in agent.workspace_status().items():
        print(f"  {key}: {value}")

    task = agent.create_task(
        "Create a Python calculator with tests."
    )

    print("\nTask:")
    print(task.request)

    print("\nAgent Plan:")

    response = agent.ask_model(task)

    print(response)