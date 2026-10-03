from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnosis:
    """Structured diagnosis of a coding/test failure."""

    category: str
    summary: str
    likely_cause: str
    affected_area: str
    suggested_action: str
    severity: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "summary": self.summary,
            "likely_cause": self.likely_cause,
            "affected_area": self.affected_area,
            "suggested_action": self.suggested_action,
            "severity": self.severity,
        }


class DiagnosisEngine:
    """Convert tool/test failures into structured coding diagnoses."""

    PATTERNS = (
        (
            "syntax_error",
            ("SyntaxError", "IndentationError", "TabError"),
            "Python syntax or indentation is invalid.",
            "The changed Python code contains invalid syntax or indentation.",
            "The affected file or line shown by the test output.",
            "Inspect the reported line and correct the syntax/indentation.",
            "high",
        ),
        (
            "import_error",
            ("ModuleNotFoundError", "ImportError"),
            "A required module or import could not be resolved.",
            "An import name, package, or module path is missing or incorrect.",
            "The import statement or dependency named in the error.",
            "Inspect the import and verify the dependency or module path.",
            "high",
        ),
        (
            "assertion_failure",
            ("AssertionError", "assert "),
            "A test assertion did not match the expected result.",
            "The implementation behavior differs from the test expectation.",
            "The failing test assertion and the code it exercises.",
            "Inspect the failing assertion and trace the expected versus actual value.",
            "medium",
        ),
        (
            "type_error",
            ("TypeError",),
            "A value was used with an incompatible type or signature.",
            "A function call, operation, or data value has an unexpected type.",
            "The traceback location and function signature involved.",
            "Inspect the failing call and verify argument types and return values.",
            "high",
        ),
        (
            "name_error",
            ("NameError", "UnboundLocalError"),
            "A variable or name is unavailable in the current scope.",
            "A referenced name is undefined or incorrectly scoped.",
            "The line containing the unresolved name.",
            "Inspect the name definition and its scope before changing behavior.",
            "high",
        ),
        (
            "file_not_found",
            ("FileNotFoundError",),
            "A required file or path could not be found.",
            "The requested file/path is missing or incorrect.",
            "The path shown in the failure output.",
            "Inspect workspace files and correct the path or create the required file.",
            "medium",
        ),
        (
            "permission_error",
            ("PermissionError",),
            "The operation was blocked by file or system permissions.",
            "The process does not have the required permission for the operation.",
            "The file or operation named in the error.",
            "Inspect the operation and avoid changing permissions unless explicitly required.",
            "high",
        ),
    )

    def diagnose(self, result: dict) -> dict:
        """Return a structured diagnosis for a failed tool/test result."""

        if not isinstance(result, dict):
            return Diagnosis(
                category="invalid_result",
                summary="The tool returned an invalid result object.",
                likely_cause="The tool result was not a dictionary.",
                affected_area="tool result",
                suggested_action="Inspect the tool integration before retrying.",
                severity="high",
            ).to_dict()

        if result.get("success") is True:
            return Diagnosis(
                category="passed",
                summary="The test/tool result passed successfully.",
                likely_cause="No failure was reported.",
                affected_area="none",
                suggested_action="Continue the coding task.",
                severity="info",
            ).to_dict()

        text = self._failure_text(result)

        # Lowercase once for the whole scan (Batch 11). The failure
        # text can be the full test output; lowering it inside the
        # per-marker check reallocated a complete lowercase copy of
        # that text once per marker (up to 12 copies per call, and
        # one extra per pattern prefix). Matching semantics are
        # unchanged: marker.lower() in lowered_text is exactly
        # marker.lower() in text.lower().
        lowered_text = text.lower()

        for (
            category,
            markers,
            summary,
            likely_cause,
            affected_area,
            suggested_action,
            severity,
        ) in self.PATTERNS:
            if any(marker.lower() in lowered_text for marker in markers):
                return Diagnosis(
                    category=category,
                    summary=summary,
                    likely_cause=likely_cause,
                    affected_area=affected_area,
                    suggested_action=suggested_action,
                    severity=severity,
                ).to_dict()

        return Diagnosis(
            category="test_failure",
            summary="The test or tool operation failed.",
            likely_cause="The available failure output does not match a known error category.",
            affected_area="The failing test, command, or traceback location.",
            suggested_action="Inspect the complete failure output before changing code.",
            severity="medium",
        ).to_dict()

    def _failure_text(self, result: dict) -> str:
        parts: list[str] = []

        for key in (
            "error",
            "output",
            "stderr",
            "stdout",
            "message",
        ):
            value = result.get(key)
            if value:
                parts.append(str(value))

        nested = result.get("result")
        if isinstance(nested, dict):
            for key in (
                "error",
                "output",
                "stderr",
                "stdout",
                "message",
            ):
                value = nested.get(key)
                if value:
                    parts.append(str(value))

        return "\n".join(parts)
