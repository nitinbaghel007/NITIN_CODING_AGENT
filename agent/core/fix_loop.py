from dataclasses import dataclass


@dataclass(frozen=True)
class FixLoopDecision:
    """Decision returned after recording a test failure."""

    should_continue: bool
    reason: str
    failure_count: int
    repeated_count: int


class FixLoopGuard:
    """Bound autonomous test-fix cycles and prevent repeated failures."""

    MAX_FIX_ATTEMPTS = 3
    MAX_REPEATED_FAILURES = 2

    def __init__(self):
        self.failure_count = 0
        self.repeated_count = 0
        self._last_fingerprint = ""

    def record_failure(
        self,
        diagnosis: dict,
    ) -> FixLoopDecision:
        """Record a diagnosis and decide whether another fix is allowed."""

        self.failure_count += 1

        fingerprint = self._fingerprint(
            diagnosis
        )

        if fingerprint == self._last_fingerprint:
            self.repeated_count += 1
        else:
            self.repeated_count = 1
            self._last_fingerprint = fingerprint

        if (
            self.repeated_count
            >= self.MAX_REPEATED_FAILURES
        ):
            return FixLoopDecision(
                should_continue=False,
                reason=(
                    "The same failure diagnosis repeated "
                    "without a successful verification."
                ),
                failure_count=self.failure_count,
                repeated_count=self.repeated_count,
            )

        if self.failure_count >= self.MAX_FIX_ATTEMPTS:
            return FixLoopDecision(
                should_continue=False,
                reason=(
                    "Maximum autonomous fix attempts reached."
                ),
                failure_count=self.failure_count,
                repeated_count=self.repeated_count,
            )

        return FixLoopDecision(
            should_continue=True,
            reason=(
                "Another targeted fix attempt is allowed."
            ),
            failure_count=self.failure_count,
            repeated_count=self.repeated_count,
        )

    def record_success(self) -> None:
        """Reset failure tracking after a verified successful test."""

        self.failure_count = 0
        self.repeated_count = 0
        self._last_fingerprint = ""

    def _fingerprint(
        self,
        diagnosis: dict,
    ) -> str:
        """Build a stable fingerprint from the diagnosis category and area."""

        category = str(
            diagnosis.get(
                "category",
                "unknown",
            )
        ).strip().lower()

        affected_area = str(
            diagnosis.get(
                "affected_area",
                "",
            )
        ).strip().lower()

        summary = str(
            diagnosis.get(
                "summary",
                "",
            )
        ).strip().lower()

        return "|".join(
            (
                category,
                affected_area,
                summary,
            )
        )
