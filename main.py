import argparse
import json
import sys
from pathlib import Path

from agent.core.coding_loop import CodingLoop


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface parser."""

    parser = argparse.ArgumentParser(
        prog="nitin-agent",
        description=(
            "Nitin Coding Agent - autonomous coding assistant."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run an autonomous coding task.",
    )

    run_parser.add_argument(
        "task",
        help="Coding task to execute.",
    )

    run_parser.add_argument(
        "--workspace",
        default="workspace",
        help=(
            "Workspace directory for the task. "
            "Defaults to 'workspace'."
        ),
    )

    run_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Print only the final result as JSON.",
    )

    return parser


def run_task(
    task: str,
    workspace: str,
) -> dict:
    """Run one coding task through the CodingLoop."""

    workspace_path = Path(workspace)

    if not workspace_path.exists():
        raise FileNotFoundError(
            f"Workspace does not exist: {workspace}"
        )

    if not workspace_path.is_dir():
        raise NotADirectoryError(
            f"Workspace is not a directory: {workspace}"
        )

    loop = CodingLoop(
        str(workspace_path)
    )

    return loop.run(task)


def main(
    argv: list[str] | None = None,
) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            result = run_task(
                task=args.task,
                workspace=args.workspace,
            )
        except Exception as exc:
            if args.output_json:
                print(
                    json.dumps(
                        {
                            "success": False,
                            "status": "cli_error",
                            "error": str(exc),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print(
                    "\n[CLI ERROR]"
                )
                print(str(exc))

            return 1

        if args.output_json:
            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
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

        return 0 if result.get("success") else 1

    parser.error(
        f"Unsupported command: {args.command}"
    )

    return 2


if __name__ == "__main__":
    sys.exit(main())
