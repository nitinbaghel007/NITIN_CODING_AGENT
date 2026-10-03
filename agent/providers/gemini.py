import json
import os
import re
import time

from google import genai
from google.genai import errors

from agent.providers.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini provider for Nitin Coding Agent."""

    MAX_JSON_RETRIES = 1

    # Additional application-level retries for transient
    # Gemini service errors such as HTTP 503.
    MAX_TRANSIENT_RETRIES = 3

    RETRY_DELAYS = (
        2,
        4,
        8,
    )

    def __init__(
        self,
        model: str | None = None,
    ):
        self.api_key = os.getenv("GEMINI_API_KEY")

        self.model = (
            model
            or os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash",
            )
        )

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set."
            )

        self.client = genai.Client(
            api_key=self.api_key
        )

    def _is_rate_limit_error(
        self,
        exc: Exception,
    ) -> bool:
        """Return True when Gemini reports a 429/quota error."""

        status_code = getattr(
            exc,
            "code",
            None,
        )

        if status_code == 429:
            return True

        error_text = str(exc).upper()

        rate_limit_markers = (
            "429",
            "RESOURCE_EXHAUSTED",
            "RATE LIMIT",
            "QUOTA EXCEEDED",
            "QUOTA",
        )

        return any(
            marker in error_text
            for marker in rate_limit_markers
        )

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Send a prompt to Gemini and return its response."""

        last_error = None

        for attempt in range(
            self.MAX_TRANSIENT_RETRIES + 1
        ):
            try:
                response = (
                    self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                    )
                )

                text = getattr(
                    response,
                    "text",
                    None,
                )

                if not text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return text

            except errors.ServerError as exc:
                last_error = exc

                # Only retry transient server errors.
                # Do not retry indefinitely.
                if attempt >= self.MAX_TRANSIENT_RETRIES:
                    raise

                delay = self.RETRY_DELAYS[
                    min(
                        attempt,
                        len(self.RETRY_DELAYS) - 1,
                    )
                ]

                print(
                    "\n[GEMINI RETRY]"
                )

                print(
                    f"Gemini server error: {exc}"
                )

                print(
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

            except errors.APIError as exc:
                if self._is_rate_limit_error(exc):
                    raise RuntimeError(
                        "GEMINI_RATE_LIMIT: "
                        f"{exc}"
                    ) from exc

                # Client/API errors such as 400, 401, 403,
                # 404, etc. should not be blindly retried.
                raise

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "Gemini request failed unexpectedly."
        )

    def _parse_json_response(
        self,
        raw: str,
    ) -> dict:
        """Parse Gemini response into a JSON object."""

        if not raw or not raw.strip():
            raise ValueError(
                "Gemini returned an empty response."
            )

        text = raw.strip()

        try:
            result = json.loads(text)

            if not isinstance(result, dict):
                raise ValueError(
                    "Gemini JSON response must be an object."
                )

            return result

        except json.JSONDecodeError:
            pass

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()

        try:
            result = json.loads(text)

            if not isinstance(result, dict):
                raise ValueError(
                    "Gemini JSON response must be an object."
                )

            return result

        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            candidate = text[
                start:end + 1
            ]

            try:
                result = json.loads(
                    candidate
                )

                if not isinstance(result, dict):
                    raise ValueError(
                        "Extracted JSON must be an object."
                    )

                return result

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Gemini returned invalid JSON:\n"
            f"{raw}"
        )

    def _normalize_actions(
        self,
        result: dict,
    ) -> dict:
        """Normalize Gemini output into standard action format."""

        if "actions" in result:
            actions = result["actions"]

            if not isinstance(actions, list):
                raise ValueError(
                    "Gemini 'actions' field must be a list."
                )

            return {
                "summary": result.get(
                    "summary",
                    "No summary provided.",
                ),
                "actions": actions,
            }

        if "tool" in result:
            return {
                "summary": (
                    "Single action generated by Gemini."
                ),
                "actions": [result],
            }

        raise ValueError(
            "Gemini response does not contain "
            "'actions' or a valid 'tool'."
        )

    def _build_action_prompt(
        self,
        task: str,
        strict_retry: bool = False,
    ) -> str:
        """Build structured coding-action prompt."""

        retry_instruction = ""

        if strict_retry:
            retry_instruction = """
IMPORTANT RETRY:

Your previous response was NOT valid JSON.

Return ONLY a normal JSON object.

DO NOT use:
- Markdown
- ```json
- function calls
- tool-call syntax
- special tokens

Example:

{
  "summary": "Inspect workspace",
  "actions": [
    {
      "tool": "list_files"
    }
  ]
}
"""

        return f"""
You are Nitin Coding Agent.

Convert the user's coding task into safe,
structured coding actions.

USER TASK:
{task}

AVAILABLE TOOLS:

1. list_files

{{
  "tool": "list_files"
}}

2. write_file

{{
  "tool": "write_file",
  "path": "relative/path.py",
  "content": "complete file content"
}}

3. read_file

{{
  "tool": "read_file",
  "path": "relative/path.py"
}}

4. run_command

{{
  "tool": "run_command",
  "command": "python --version"
}}

5. run_tests

{{
  "tool": "run_tests"
}}

RESPONSE FORMAT:

{{
  "summary": "short description",
  "actions": [
    {{
      "tool": "list_files"
    }}
  ]
}}

RULES:

- Return ONLY valid JSON.
- Do NOT use Markdown.
- Do NOT use ```json.
- Do NOT use function-call syntax.
- Do NOT use special tool-call tokens.
- Use exactly the JSON structure shown above.
- If one action is needed, still use the actions list.
- Paths must be relative.
- Never use absolute paths.
- Never access files outside the workspace.
- Never use dangerous commands.
- Use list_files instead of ls or dir.
- Inspect existing files before modifying them.
- Run tests after code changes.
- Keep changes focused on the task.

{retry_instruction}
"""

    def generate_actions(
        self,
        task: str,
    ) -> dict:
        """Generate structured coding actions."""

        for attempt in range(
            self.MAX_JSON_RETRIES + 1
        ):
            prompt = self._build_action_prompt(
                task,
                strict_retry=(
                    attempt > 0
                ),
            )

            raw = self.generate(
                prompt
            )

            try:
                result = self._parse_json_response(
                    raw
                )

                return self._normalize_actions(
                    result
                )

            except ValueError:
                if (
                    attempt
                    >= self.MAX_JSON_RETRIES
                ):
                    raise

                continue

        raise ValueError(
            "Unable to generate valid Gemini actions."
        )


if __name__ == "__main__":
    provider = GeminiProvider()

    result = provider.generate_actions(
        "Inspect the workspace and list the files."
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )