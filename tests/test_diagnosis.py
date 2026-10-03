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
