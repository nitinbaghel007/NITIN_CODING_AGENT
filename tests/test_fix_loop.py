from agent.core.fix_loop import FixLoopGuard


def make_diagnosis(
    category="assertion_failure",
    affected_area="calculator.py",
    summary="The assertion failed.",
):
    return {
        "category": category,
        "affected_area": affected_area,
        "summary": summary,
    }


def test_fix_loop_allows_first_failure():
    guard = FixLoopGuard()

    decision = guard.record_failure(
        make_diagnosis()
    )

    assert decision.should_continue is True
    assert decision.failure_count == 1
    assert decision.repeated_count == 1


def test_fix_loop_allows_different_failure():
    guard = FixLoopGuard()

    first = guard.record_failure(
        make_diagnosis()
    )

    second = guard.record_failure(
        make_diagnosis(
            category="type_error",
            affected_area="calculator.py",
            summary="A type mismatch occurred.",
        )
    )

    assert first.should_continue is True
    assert second.should_continue is True
    assert second.failure_count == 2
    assert second.repeated_count == 1


def test_fix_loop_stops_repeated_failure():
    guard = FixLoopGuard()

    guard.record_failure(
        make_diagnosis()
    )

    decision = guard.record_failure(
        make_diagnosis()
    )

    assert decision.should_continue is False
    assert decision.reason.startswith(
        "The same failure diagnosis repeated"
    )


def test_fix_loop_stops_at_max_attempts():
    guard = FixLoopGuard()

    guard.MAX_REPEATED_FAILURES = 99

    guard.record_failure(
        make_diagnosis("assertion_failure")
    )
    guard.record_failure(
        make_diagnosis("type_error")
    )

    decision = guard.record_failure(
        make_diagnosis("name_error")
    )

    assert decision.should_continue is False
    assert decision.failure_count == 3
    assert decision.reason.startswith(
        "Maximum autonomous fix attempts"
    )


def test_fix_loop_success_resets_state():
    guard = FixLoopGuard()

    guard.record_failure(
        make_diagnosis()
    )

    guard.record_success()

    decision = guard.record_failure(
        make_diagnosis()
    )

    assert decision.should_continue is True
    assert decision.failure_count == 1
    assert decision.repeated_count == 1


def test_fix_loop_fingerprint_ignores_case_and_whitespace():
    guard = FixLoopGuard()

    first = guard.record_failure(
        make_diagnosis(
            category="Assertion_Failure",
            affected_area="  Calculator.py  ",
            summary="  The assertion failed.  ",
        )
    )

    second = guard.record_failure(
        make_diagnosis(
            category="assertion_failure",
            affected_area="calculator.py",
            summary="the assertion failed.",
        )
    )

    assert first.should_continue is True
    assert second.should_continue is False
