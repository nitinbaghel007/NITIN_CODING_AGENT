from agent.core.context_manager import (
    ContextManager,
)


def create_manager():
    return ContextManager(
        task="Add multiplication to calculator.",
        workspace="workspace",
        provider="gemini",
    )


def test_context_initial_state():
    manager = create_manager()

    context = manager.get()

    assert (
        context.task
        == "Add multiplication to calculator."
    )
    assert context.workspace == "workspace"
    assert context.provider == "gemini"
    assert context.current_step == 0
    assert context.status == "initialized"
    assert context.history == []
    assert context.files_inspected == []
    assert context.files_changed == []
    assert context.tests_run == 0
    assert context.tests_verified is False
    assert context.last_error is None


def test_set_provider_and_step():
    manager = create_manager()

    manager.set_provider("openrouter")
    manager.set_step(3)

    context = manager.get()

    assert context.provider == "openrouter"
    assert context.current_step == 3


def test_negative_step_rejected():
    manager = create_manager()

    try:
        manager.set_step(-1)
        assert False
    except ValueError:
        assert True


def test_history_is_recorded():
    manager = create_manager()

    manager.set_step(2)

    manager.add_history(
        {
            "tool": "read_file",
            "path": "calculator.py",
        },
        {
            "success": True,
        },
    )

    history = manager.get().history

    assert len(history) == 1
    assert history[0]["step"] == 2
    assert (
        history[0]["action"]["tool"]
        == "read_file"
    )


def test_file_tracking():
    manager = create_manager()

    manager.record_file_inspected(
        "calculator.py"
    )
    manager.record_file_inspected(
        "calculator.py"
    )

    manager.record_file_changed(
        "calculator.py"
    )
    manager.record_file_changed(
        "test_calculator.py"
    )

    context = manager.get()

    assert context.files_inspected == [
        "calculator.py"
    ]

    assert context.files_changed == [
        "calculator.py",
        "test_calculator.py",
    ]


def test_test_tracking():
    manager = create_manager()

    manager.record_test_run(False)

    assert manager.get().tests_run == 1
    assert (
        manager.get().tests_verified is False
    )

    manager.record_test_run(True)

    assert manager.get().tests_run == 2
    assert (
        manager.get().tests_verified is True
    )


def test_error_tracking():
    manager = create_manager()

    manager.set_error(
        "pytest failed."
    )

    assert (
        manager.get().last_error
        == "pytest failed."
    )

    manager.clear_error()

    assert manager.get().last_error is None


def test_completion():
    manager = create_manager()

    manager.mark_completed()

    context = manager.get()

    assert context.status == "completed"
    assert context.tests_verified is True


def test_failure():
    manager = create_manager()

    manager.mark_failed(
        "Provider unavailable."
    )

    context = manager.get()

    assert context.status == "failed"
    assert (
        context.last_error
        == "Provider unavailable."
    )


def test_snapshot():
    manager = create_manager()

    manager.set_step(4)
    manager.record_file_changed(
        "calculator.py"
    )
    manager.record_test_run(True)

    snapshot = manager.snapshot()

    assert snapshot["current_step"] == 4
    assert snapshot["files_changed"] == [
        "calculator.py"
    ]
    assert snapshot["tests_run"] == 1
    assert snapshot["tests_verified"] is True


def test_reset():
    manager = create_manager()

    manager.set_step(5)
    manager.record_file_changed(
        "calculator.py"
    )
    manager.record_test_run(True)
    manager.set_error("temporary error")

    manager.reset()

    context = manager.get()

    assert context.current_step == 0
    assert context.status == "initialized"
    assert context.files_changed == []
    assert context.tests_run == 0
    assert context.tests_verified is False
    assert context.last_error is None

    assert context.task == (
        "Add multiplication to calculator."
    )
    assert context.workspace == "workspace"
    assert context.provider == "gemini"