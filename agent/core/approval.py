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
