from agent.core.safety import SafetyDecision


class ApprovalManager:
    """Handle interactive approval for safety-sensitive actions."""

    APPROVE_VALUES = {
        "y",
        "yes",
    }

    def request(
        self,
        action: dict,
        decision: SafetyDecision,
    ) -> bool:
        """Ask the user whether an approval-required action may run."""

        print("\n========================================")
        print(" APPROVAL REQUIRED")
        print("========================================")

        print("\nSafety level:")
        print(f"  {decision.level.value}")

        print("\nReason:")
        print(f"  {decision.reason}")

        print("\nRequested action:")
        print(f"  Tool: {action.get('tool')}")

        if action.get("command"):
            print(
                f"  Command: {action.get('command')}"
            )

        if action.get("path"):
            print(
                f"  Path: {action.get('path')}"
            )

        response = input(
            "\nAllow this action? [y/N]: "
        )

        return response.strip().lower() in self.APPROVE_VALUES


class NonInteractiveApproval:
    """Approval callback for ``--non-interactive`` runs.

    Guarantees for automated runs:

    * ``input()`` is never called, so a closed or missing stdin cannot
      block or break the run.
    * Approval is never granted automatically - approval-required actions
      come back as ``approval_required`` and are simply not executed.
    * Any failure inside the approval path (``EOFError``, ``OSError``,
      a broken pipe, ...) is swallowed so it can never escape into the
      agent loop and abort the task.

    Interactive behaviour is untouched: :class:`ApprovalManager` still
    prompts exactly as before when ``--non-interactive`` is not supplied.
    """

    def __init__(self):
        self.denied_count = 0
        self.last_error: str | None = None

    def _deny(self, action: dict, decision: SafetyDecision) -> bool:
        """Report the denial. Always returns ``False``."""

        print("\n========================================")
        print(" NON-INTERACTIVE MODE")
        print("========================================")

        print("\nApproval:")
        print("  DENIED (approval is never automatic)")

        print("\nSafety level:")
        print(f"  {decision.level.value}")

        print("\nReason:")
        print(f"  {decision.reason}")

        print("\nRequested action:")
        print(f"  Tool: {action.get('tool')}")

        if action.get("command"):
            print(f"  Command: {action.get('command')}")

        if action.get("path"):
            print(f"  Path: {action.get('path')}")

        return False

    def __call__(
        self,
        action: dict,
        decision: SafetyDecision,
    ) -> bool:
        """Deny safely; never raises and never prompts."""

        try:
            approved = bool(
                self._deny(action, decision)
            )
        except Exception as exc:
            # The approval path must never crash an unattended run.
            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            approved = False

        if not approved:
            self.denied_count += 1

        return approved
