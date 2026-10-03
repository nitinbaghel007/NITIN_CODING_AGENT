from agent.core.diagnosis import DiagnosisEngine


def test_diagnosis_passed_result():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": True,
            "tool": "run_tests",
        }
    )

    assert diagnosis["category"] == "passed"
    assert diagnosis["severity"] == "info"


def test_diagnosis_syntax_error():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "tool": "run_tests",
            "result": {
                "output": "SyntaxError: invalid syntax",
            },
        }
    )

    assert diagnosis["category"] == "syntax_error"
    assert diagnosis["severity"] == "high"


def test_diagnosis_import_error():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "result": {
                "output": "ModuleNotFoundError: No module named 'pandas'",
            },
        }
    )

    assert diagnosis["category"] == "import_error"


def test_diagnosis_assertion_failure():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "result": {
                "output": "E AssertionError: assert 3 == 4",
            },
        }
    )

    assert diagnosis["category"] == "assertion_failure"


def test_diagnosis_type_error():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "result": {
                "output": "TypeError: missing required positional argument",
            },
        }
    )

    assert diagnosis["category"] == "type_error"


def test_diagnosis_name_error():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "result": {
                "output": "NameError: name 'value' is not defined",
            },
        }
    )

    assert diagnosis["category"] == "name_error"


def test_diagnosis_file_not_found():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "result": {
                "output": "FileNotFoundError: missing.py",
            },
        }
    )

    assert diagnosis["category"] == "file_not_found"


def test_diagnosis_unknown_failure():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "result": {
                "output": "Something unexpected failed",
            },
        }
    )

    assert diagnosis["category"] == "test_failure"
    assert diagnosis["severity"] == "medium"


# =========================================================
# Batch 9 Phase 4: remaining diagnosis contracts
# =========================================================


def test_diagnosis_permission_error():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "result": {
                "output": "PermissionError: [Errno 13]",
            },
        }
    )

    assert diagnosis["category"] == "permission_error"
    assert diagnosis["severity"] == "high"


def test_diagnosis_invalid_result_for_non_dict():
    engine = DiagnosisEngine()

    for bad_input in ("a string", [], None, 42):
        diagnosis = engine.diagnose(bad_input)

        assert diagnosis["category"] == "invalid_result"
        assert diagnosis["severity"] == "high"


def test_diagnosis_insufficient_context_falls_back():
    engine = DiagnosisEngine()

    for empty in ({}, {"success": False}):
        diagnosis = engine.diagnose(empty)

        assert diagnosis["category"] == "test_failure"
        assert diagnosis["severity"] == "medium"


def test_diagnosis_reads_top_level_failure_keys():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "stderr": "NameError: name 'x' is undefined",
        }
    )

    assert diagnosis["category"] == "name_error"


def test_diagnosis_ignores_non_dict_nested_result():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "result": "raw non-dict payload",
            "error": "TypeError: bad operand type",
        }
    )

    assert diagnosis["category"] == "type_error"


def test_diagnosis_marker_matching_is_case_insensitive():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "output": "syntaxerror: invalid syntax",
        }
    )

    assert diagnosis["category"] == "syntax_error"


def test_diagnosis_pattern_order_prefers_syntax_error():
    """PATTERNS are evaluated in declaration order, so the
    first matching pattern wins when several markers appear."""

    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "output": "TypeError while handling SyntaxError",
        }
    )

    assert diagnosis["category"] == "syntax_error"


def test_diagnosis_to_dict_has_stable_keys():
    diagnosis = DiagnosisEngine().diagnose(
        {"success": False, "output": "boom"}
    )

    assert set(diagnosis) == {
        "category",
        "summary",
        "likely_cause",
        "affected_area",
        "suggested_action",
        "severity",
    }


def test_diagnosis_is_deterministic():
    engine = DiagnosisEngine()
    result = {
        "success": False,
        "result": {"stderr": "AssertionError: 1 == 2"},
    }

    first = engine.diagnose(result)
    second = engine.diagnose(result)

    assert first == second
    assert first["category"] == "assertion_failure"


def test_diagnosis_provider_failure_is_controlled():
    diagnosis = DiagnosisEngine().diagnose(
        {
            "success": False,
            "error": "HTTP 500 internal server error",
        }
    )

    assert diagnosis["category"] == "test_failure"
    assert diagnosis["severity"] == "medium"
