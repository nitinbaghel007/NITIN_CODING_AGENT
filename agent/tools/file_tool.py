from pathlib import Path


class FileTool:
    """Safe file operations restricted to the agent workspace."""

    def __init__(self, workspace: str):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, relative_path: str) -> Path:
        target = (self.workspace / relative_path).resolve()

        if target != self.workspace and self.workspace not in target.parents:
            raise ValueError("Path is outside the agent workspace.")

        return target

    def read(self, relative_path: str) -> str:
        path = self._safe_path(relative_path)
        return path.read_text(encoding="utf-8")

    def write(self, relative_path: str, content: str) -> None:
        path = self._safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def exists(self, relative_path: str) -> bool:
        return self._safe_path(relative_path).exists()