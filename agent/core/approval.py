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
        """Ask the user whether an approval-required action may run.

        Reading the answer is best-effort. A closed stdin
        (``EOFError``), Ctrl+C (``KeyboardInterrupt``), a broken pipe or
        closed file (``OSError`` / ``ValueError``) all deny the action
        instead of raising into the agent loop. Approval is never
        implied by a failed prompt, and :meth:`_parse_response` decides
        every answer so unexpected input cannot approve by accident.
        """

        try:
            return self._prompt(action, decision)

        except KeyboardInterrupt:
            self._notice(
                "\n  Approval prompt interrupted - action DENIED."
            )

        except EOFError:
            self._notice(
                "\n  No input available (closed stdin) - action DENIED."
            )

        except (OSError, ValueError) as exc:
            self._notice(
                "\n  Input unavailable "
                f"({type(exc).__name__}) - action DENIED."
            )

        return False

    @staticmethod
    def _notice(message: str) -> None:
        """Report a failed prompt without risking a second failure.

        Reporting the denial must never be the thing that raises,
        including when stdout itself is broken.
        """

        try:
            print(message)
        except Exception:
            pass

    @classmethod
    def _parse_response(cls, response: object) -> bool:
        """Return True only for an explicit ``y`` or ``yes``.

        Case, surrounding whitespace and trailing newlines are ignored.
        Empty input, unexpected words and non-text answers all deny, so
        nothing can approve an action by accident.
        """

        if not isinstance(response, str):
            return False

        return response.strip().lower() in cls.APPROVE_VALUES

    def _prompt(
        self,
        action: dict,
        decision: SafetyDecision,
    ) -> bool:
        """Print the approval prompt and read the user's answer."""

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

        return self._parse_response(response)


class NonInteractiveApproval:
    """Approval callback for ``--non-interactive`` runs.

    Guarantees for automated runs:

    * ``input()`` is never called, so a closed or missing stdin cannot
      block or break the run.
    * Approval is never granted automatically - approval-required actions
      come back as ``approval_required`` and are simply not executed.
    * Any failure inside the approval path (``EOFError``, ``OSError``,
      a broken pipe, ``KeyboardInterrupt``, ...) is swallowed so it can
      never escape into the agent loop and abort the task.

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
        except (Exception, KeyboardInterrupt) as exc:
            # The approval path must never crash an unattended run.
            # KeyboardInterrupt is not an Exception subclass, so it is
            # listed explicitly; SystemExit still propagates.
            self.last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            approved = False

        if not approved:
            self.denied_count += 1

        return approved
