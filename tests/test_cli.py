from unittest.mock import patch

import main


def test_parser_requires_command():
    parser = main.build_parser()

    try:
        parser.parse_args([])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError(
            "Parser should require a command."
        )


def test_parser_accepts_run_task():
    parser = main.build_parser()

    args = parser.parse_args(
        [
            "run",
            "Fix the tests",
            "--workspace",
            "workspace/demo",
        ]
    )

    assert args.command == "run"
    assert args.task == "Fix the tests"
    assert args.workspace == "workspace/demo"
    assert args.output_json is False


def test_parser_supports_json_output():
    parser = main.build_parser()

    args = parser.parse_args(
        [
            "run",
            "Fix the tests",
            "--json",
        ]
    )

    assert args.output_json is True


def test_run_task_rejects_missing_workspace():
    try:
        main.run_task(
            "Fix the tests",
            "workspace/does-not-exist",
        )
    except FileNotFoundError as exc:
        assert "Workspace does not exist" in str(exc)
    else:
        raise AssertionError(
            "Missing workspace should be rejected."
        )


def test_run_task_calls_coding_loop(tmp_path):
    expected = {
        "success": True,
        "status": "completed",
    }

    workspace = tmp_path / "demo"
    workspace.mkdir()

    with patch(
        "main.CodingLoop"
    ) as loop_class:
        loop_instance = loop_class.return_value
        loop_instance.run.return_value = expected

        result = main.run_task(
            "Fix the tests",
            str(workspace),
        )

    loop_class.assert_called_once_with(
        str(workspace)
    )
    loop_instance.run.assert_called_once_with(
        "Fix the tests"
    )

    assert result == expected


def test_main_returns_zero_for_success(tmp_path):
    workspace = tmp_path / "demo"
    workspace.mkdir()

    expected = {
        "success": True,
        "status": "completed",
    }

    with patch(
        "main.run_task",
        return_value=expected,
    ):
        result = main.main(
            [
                "run",
                "Fix the tests",
                "--workspace",
                str(workspace),
                "--json",
            ]
        )

    assert result == 0


def test_main_returns_one_for_failed_task(tmp_path):
    workspace = tmp_path / "demo"
    workspace.mkdir()

    expected = {
        "success": False,
        "status": "max_steps_reached",
    }

    with patch(
        "main.run_task",
        return_value=expected,
    ):
        result = main.main(
            [
                "run",
                "Fix the tests",
                "--workspace",
                str(workspace),
                "--json",
            ]
        )

    assert result == 1


def test_main_returns_one_for_cli_error(tmp_path):
    workspace = tmp_path / "demo"
    workspace.mkdir()

    with patch(
        "main.run_task",
        side_effect=RuntimeError(
            "provider unavailable"
        ),
    ):
        result = main.main(
            [
                "run",
                "Fix the tests",
                "--workspace",
                str(workspace),
                "--json",
            ]
        )

    assert result == 1
