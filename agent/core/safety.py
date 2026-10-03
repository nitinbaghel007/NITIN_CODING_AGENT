import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SafetyLevel(str, Enum):
    """Safety classification for an agent action."""

    SAFE = "safe"
    APPROVAL = "approval"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class SafetyDecision:
    """Result of validating one agent action."""

    level: SafetyLevel
    reason: str


class SafetyManager:
    """Validate coding-agent actions before execution."""

    SAFE_TOOLS = {
        "list_files",
        "read_file",
        "write_file",
        "run_tests",
    }

    APPROVAL_COMMAND_PREFIXES = {
        "pip install",
        "pip uninstall",
        "python -m pip install",
        "python -m pip uninstall",
        "git add",
        "git commit",
        "git checkout",
        "git restore",
        "git clean",
        "git reset",
        "git push",
        "git pull",
        "npm install",
        "npm uninstall",
        "npm update",
    }

    # -----------------------------------------------------
    # SAFE COMMAND STRUCTURE
    # -----------------------------------------------------
    # Commands are matched on their *structure* (executable + expected
    # subcommand), not on the whole raw string, so legitimate arguments
    # such as flags and relative test paths stay SAFE.
    #
    # A command is only SAFE when:
    #   * its executable matches one of the rules below,
    #   * every argument looks like an ordinary relative argument, and
    #   * it contains no unquoted shell control character.
    SAFE_GIT_SUBCOMMANDS = {
        "status",
        "diff",
        "log",
    }

    # Ordinary argument characters: flags, relative paths, selectors.
    SAFE_ARG_PATTERN = re.compile(
        r"^[A-Za-z0-9_.\-/\\:=,\[\]'+~#@]+$"
    )

    # A Windows drive reference anywhere in an argument
    # (e.g. C:\evil.txt or --output=C:\evil.txt).
    DRIVE_REFERENCE_PATTERN = re.compile(
        r"[A-Za-z]:[\\/]"
    )

    # -----------------------------------------------------
    # DANGEROUS COMMAND TOKENS
    # -----------------------------------------------------
    # Matching is on WHOLE tokens only - never on arbitrary substrings.
    # The first two tokens are the command and its first argument, so:
    #     format C:            -> token "format"      -> BLOCKED
    #     git rm file          -> token "rm" (index 1)-> BLOCKED
    #     python -m pytest tests/test_format.py
    #                         -> token is the path    -> SAFE
    #     git log --format=%h  -> token is --format=%h-> not blocked
    DANGEROUS_TOKENS = {
        # Windows system / destructive commands
        "format",
        "shutdown",
        "restart-computer",
        "stop-computer",
        "diskpart",
        "bcdedit",
        "takeown",
        "icacls",
        "reg",
        "net",
        "cipher",
        "vssadmin",
        "taskkill",
        "del",
        "erase",
        "rmdir",
        "rd",
        # Shells (TerminalTool refuses these too)
        "cmd",
        "powershell",
        "powershell_ise",
        # Unix destructive commands
        "rm",
        "mkfs",
        "dd",
        # PowerShell cmdlets
        "remove-item",
        "remove-itemproperty",
        "set-content",
        "clear-content",
        "invoke-expression",
        "iex",
    }

    # -----------------------------------------------------
    # SHELL CONTROL CHARACTERS
    # -----------------------------------------------------
    # Unquoted occurrences of these make a shell chain: ``&``/``&&``
    # and ``|``/``||`` sequence commands, ``;`` separates them, ``>``/
    # ``<`` redirect, ``\n``/``\r`` start a new command line. Any of
    # them outside quotes blocks the command outright.
    # Detected outside quotes only: cmd.exe does not treat these as
    # operators while they are inside a quoted argument or path.
    SHELL_CONTROL_CHARS = (
        "&",
        "|",
        ";",
        ">",
        "<",
        "`",
        "\n",
        "\r",
    )

    def __init__(self, workspace: str):
        self.workspace = Path(workspace).resolve()

    def validate(self, action: dict) -> SafetyDecision:
        """Classify an action as safe, approval-required, or blocked."""

        if not isinstance(action, dict):
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Action must be a JSON object.",
            )

        tool = action.get("tool")

        if not isinstance(tool, str) or not tool.strip():
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Action is missing a valid tool.",
            )

        if tool in self.SAFE_TOOLS:
            if tool in {"read_file", "write_file"}:
                return self._validate_path_action(action)

            return SafetyDecision(
                SafetyLevel.SAFE,
                f"{tool} is an approved safe tool.",
            )

        if tool == "run_command":
            return self._validate_command(
                action.get("command")
            )

        return SafetyDecision(
            SafetyLevel.BLOCKED,
            f"Unknown tool is blocked: {tool}",
        )

    def _validate_path_action(
        self,
        action: dict,
    ) -> SafetyDecision:
        path_value = action.get("path")

        if not isinstance(path_value, str) or not path_value.strip():
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "File action requires a relative path.",
            )

        path_text = path_value.strip()

        if Path(path_text).is_absolute():
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Absolute paths are blocked.",
            )

        if re.match(
            r"^[A-Za-z]:[\\/]",
            path_text,
        ):
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Windows absolute paths are blocked.",
            )

        candidate = (
            self.workspace / Path(path_text)
        ).resolve()

        try:
            candidate.relative_to(self.workspace)
        except ValueError:
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Path escapes the workspace.",
            )

        return SafetyDecision(
            SafetyLevel.SAFE,
            "Path stays inside the workspace.",
        )

    def _validate_command(
        self,
        command: object,
    ) -> SafetyDecision:
        if not isinstance(command, str) or not command.strip():
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "run_command requires a non-empty command.",
            )

        normalized = re.sub(
            r"\s+",
            " ",
            command.strip().lower(),
        )

        tokens = self._tokenize(command)

        # 1. Dangerous command tokens (whole-token match, not substring).
        #    Checked first so dangerous commands stay blocked even when
        #    they also contain shell control characters.
        for token in tokens[:2]:
            bare = self._normalize_executable(token)
            if bare in self.DANGEROUS_TOKENS:
                return SafetyDecision(
                    SafetyLevel.BLOCKED,
                    f"Dangerous command is blocked: {bare}",
                )

        # 1b. Every segment of a chained command is checked, so
        #     "python --version & rm -rf /" cannot smuggle a dangerous
        #     executable past the two-token check above.
        for segment in self._split_segments(command):
            segment_tokens = self._tokenize(segment)
            if not segment_tokens:
                continue
            bare = self._normalize_executable(segment_tokens[0])
            if bare in self.DANGEROUS_TOKENS:
                return SafetyDecision(
                    SafetyLevel.BLOCKED,
                    f"Dangerous command is blocked: {bare}",
                )

        # 2. Shell chains are BLOCKED, not approval-gated. Chaining,
        #    piping and redirection let a second command run under the
        #    cover of an approved one, and TerminalTool executes with
        #    shell=True, so approval is not an acceptable answer here.
        #    Operators inside properly quoted arguments are not shell
        #    syntax and do not trigger this rule; unbalanced quotes are
        #    treated as unquoted (conservative direction).
        if self._unquoted_control_chars(command):
            return SafetyDecision(
                SafetyLevel.BLOCKED,
                "Shell command chaining, piping, or redirection is blocked.",
            )

        # 3. Structurally safe commands (allowlist of command shapes).
        if self._matches_safe_command(command, tokens):
            return SafetyDecision(
                SafetyLevel.SAFE,
                "Command is on the safe command allowlist.",
            )

        # 4. Known prefixes that change the environment or repository.
        for prefix in self.APPROVAL_COMMAND_PREFIXES:
            if normalized == prefix or normalized.startswith(
                prefix + " "
            ):
                return SafetyDecision(
                    SafetyLevel.APPROVAL,
                    "Command can change the environment or repository and requires approval.",
                )

        # 5. Default deny: unknown commands require approval.
        return SafetyDecision(
            SafetyLevel.APPROVAL,
            "Command is not on the safe allowlist.",
        )

    @staticmethod
    def _tokenize(command: str) -> list:
        """Split a command into tokens, honouring quoted arguments.

        Quote characters are kept on the token so callers can strip
        them again. Quotes never hide a token boundary: a quoted
        argument such as ``"a b"`` stays a single token.
        """
        tokens = []
        current = []
        in_double = False
        in_single = False

        for char in command:
            if char == '"' and not in_single:
                in_double = not in_double
                current.append(char)
            elif char == "'" and not in_double:
                in_single = not in_single
                current.append(char)
            elif (
                char.isspace()
                and not in_double
                and not in_single
            ):
                if current:
                    tokens.append("".join(current))
                    current = []
            else:
                current.append(char)

        if current:
            tokens.append("".join(current))

        return tokens

    @staticmethod
    def _strip_quotes(token: str) -> str:
        """Remove one matching pair of surrounding quotes."""
        if (
            len(token) >= 2
            and token[0] == token[-1]
            and token[0] in {'"', "'"}
        ):
            return token[1:-1]
        return token

    @classmethod
    def _normalize_executable(cls, token: str) -> str:
        """Quote-strip, lower-case and drop a trailing executable suffix.

        ``PowerShell.exe`` and ``powershell`` must classify the same way,
        otherwise an extension is enough to dodge a rule.
        """
        bare = cls._strip_quotes(token).lower()
        return re.sub(
            r"\.(exe|cmd|bat|com|ps1)$",
            "",
            bare,
        )

    @classmethod
    def _split_segments(cls, command: str) -> list:
        """Split a command on unquoted shell control characters.

        Returns the individual commands a shell would run in sequence,
        so each one can be classified on its own. Quoted operators are
        not separators (cmd.exe ignores them inside quotes), and
        unbalanced quotes fall back to splitting everywhere.
        """
        segments = []
        current = []
        in_double = False
        in_single = False

        for char in command:
            if char == '"' and not in_single:
                in_double = not in_double
                current.append(char)
            elif char == "'" and not in_double:
                in_single = not in_single
                current.append(char)
            elif (
                not in_double
                and not in_single
                and char in cls.SHELL_CONTROL_CHARS
            ):
                segments.append("".join(current))
                current = []
            else:
                current.append(char)

        segments.append("".join(current))

        return [
            segment
            for segment in segments
            if segment.strip()
        ]

    @classmethod
    def _unquoted_control_chars(cls, command: str) -> list:
        """Return shell control characters that sit outside quotes.

        cmd.exe only honours ``&``, ``|``, ``;``, ``>`` and ``<`` as
        operators when they are unquoted, so a ``>`` inside a quoted
        argument or path must not be treated as redirection. If the
        quotes are unbalanced the command cannot be analysed safely,
        so every control character counts (conservative direction).
        """
        found = []
        in_double = False
        in_single = False

        for char in command:
            if char == '"' and not in_single:
                in_double = not in_double
            elif char == "'" and not in_double:
                in_single = not in_single
            elif (
                not in_double
                and not in_single
                and char in cls.SHELL_CONTROL_CHARS
            ):
                found.append(char)

        if in_double or in_single:
            return [
                char
                for char in command
                if char in cls.SHELL_CONTROL_CHARS
            ]

        return found

    @classmethod
    def _matches_safe_command(
        cls,
        command: str,
        tokens: list,
    ) -> bool:
        """True when the command matches an allowlisted shape.

        Only read-only, workspace-local command shapes are allowed:
        ``python --version``, ``python -m pytest ...``, ``pytest ...``
        and the read-only ``git`` subcommands. Arguments are checked
        individually so flags and relative test paths stay safe while
        absolute paths and parent traversal do not.

        The shell-chain rule is re-checked here so the allowlist can
        never hand back SAFE for a chained or redirected command, no
        matter which check runs first.
        """
        if not tokens:
            return False

        if cls._unquoted_control_chars(command):
            return False

        head = cls._normalize_executable(tokens[0])
        args = [
            cls._strip_quotes(token).lower()
            for token in tokens[1:]
        ]

        if head == "python":
            if args == ["--version"]:
                return True

            if args[:2] == ["-m", "pytest"]:
                return cls._safe_args(args[2:])

            return False

        if head == "pytest":
            return cls._safe_args(args)

        if (
            head == "git"
            and args
            and args[0] in cls.SAFE_GIT_SUBCOMMANDS
        ):
            return cls._safe_args(args[1:])

        return False

    @classmethod
    def _safe_args(cls, args: list) -> bool:
        """True when every argument is an ordinary relative argument."""
        for arg in args:
            if not cls.SAFE_ARG_PATTERN.match(arg):
                return False

            if arg.startswith("/"):
                return False

            if arg.startswith("\\"):
                return False

            if cls.DRIVE_REFERENCE_PATTERN.search(arg):
                return False

            if Path(arg).is_absolute():
                return False

            if ".." in Path(arg).parts:
                return False

        return True
