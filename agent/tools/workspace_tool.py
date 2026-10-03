from pathlib import Path


class WorkspaceTool:
    """Safely inspect files inside the agent workspace."""

    def __init__(self, workspace: str):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def list_files(self) -> dict:
        """Return files and directories inside the workspace."""

        files = []
        directories = []

        for item in sorted(self.workspace.iterdir()):
            if item.name in {
                "__pycache__",
                ".pytest_cache",
                ".git",
            }:
                continue

            if item.is_file():
                files.append(item.name)

            elif item.is_dir():
                directories.append(item.name)

        return {
            "success": True,
            "workspace": str(self.workspace),
            "files": files,
            "directories": directories,
        }