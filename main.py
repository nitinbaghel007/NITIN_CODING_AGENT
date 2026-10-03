import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.config.logging_setup import configure_logging
from agent.config.settings import Settings
from agent.core.approval import NonInteractiveApproval
from agent.core.coding_loop import CodingLoop

# Child logger of the centrally configured "agent" logger; records
# are emitted by the handlers installed in main(), always on stderr.
logger = logging.getLogger("agent.cli")


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

    run_parser.add_argument(
        "--non-interactive",
        action="store_true",
        dest="non_interactive",
        help=(
            "Never prompt for approval. "
            "Approval-required actions are denied safely."
        ),
    )

    return parser


def run_task(
    task: str,
    workspace: str,
    non_interactive: bool = False,
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

    if non_interactive:
        loop = CodingLoop(
            str(workspace_path),
            approval_callback=NonInteractiveApproval(),
        )
    else:
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

    # Logging is initialized exactly once, here, at the CLI
    # boundary: console diagnostics go to stderr (never stdout)
    # plus Settings.log_file when one is configured.
    #
    # Local development convenience: load a project .env file (if
    # present) before reading configuration. load_dotenv()'s default
    # behaviour never overrides variables that are already set in
    # the OS environment, so existing environment-variable
    # configuration keeps precedence and keeps working unchanged.
    # A missing .env file is a no-op.
    load_dotenv()
    settings = Settings.from_env()
    configure_logging(settings)
    logger.debug("settings: %s", settings)

    if args.command == "run":
        logger.info(
            "run start: workspace=%s json=%s "
            "non_interactive=%s",
            args.workspace,
            args.output_json,
            args.non_interactive,
        )

        # --json reserves stdout for exactly one JSON document, so
        # every print() made while the task runs (banners,
        # progress, approval notices, provider diagnostics) is
        # treated as a diagnostic and redirected to stderr. The
        # redirect is torn down before the result is printed, so
        # the JSON itself always reaches the real stdout.
        run_stdout = (
            sys.stderr if args.output_json else sys.stdout
        )

        try:
            with contextlib.redirect_stdout(run_stdout):
                result = run_task(
                    task=args.task,
                    workspace=args.workspace,
                    non_interactive=args.non_interactive,
                )
        except Exception as exc:
            # The redirect has already been restored here, so only
            # the deterministic response below reaches stdout.
            logger.error(
                "run failed: %s: %s",
                type(exc).__name__,
                exc,
            )

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

        logger.info(
            "run finished: status=%s success=%s",
            result.get("status"),
            result.get("success"),
        )

        return 0 if result.get("success") else 1

    parser.error(
        f"Unsupported command: {args.command}"
    )

    return 2


if __name__ == "__main__":
    sys.exit(main())
