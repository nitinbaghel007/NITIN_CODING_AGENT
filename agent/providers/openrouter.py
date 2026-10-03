import json
import os
import re

import requests

from agent.providers.base import LLMProvider


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider for Nitin Coding Agent."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    MAX_JSON_RETRIES = 1

    def __init__(self, model: str = "openrouter/free"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model

        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set."
            )

    # =========================================================
    # BASIC LLM GENERATION
    # =========================================================

    def generate(self, prompt: str) -> str:
        """Send a prompt to OpenRouter and return model response."""

        response = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            },
            timeout=60,
        )

        if response.status_code == 429:
            try:
                error_data = response.json().get(
                    "error",
                    {},
                )

                message = error_data.get(
                    "message",
                    "OpenRouter rate limit exceeded.",
                )

            except ValueError:
                message = (
                    "OpenRouter rate limit exceeded."
                )

            raise RuntimeError(
                "OPENROUTER_RATE_LIMIT: "
                f"{message}"
            )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]

    # =========================================================
    # JSON PARSER
    # =========================================================

    def _parse_json_response(self, raw: str) -> dict:
        """Parse a model response into a JSON object."""

        if not raw or not raw.strip():
            raise ValueError(
                "Model returned an empty response."
            )

        text = raw.strip()

        # -----------------------------------------------------
        # Direct JSON
        # -----------------------------------------------------

        try:
            result = json.loads(text)

            if not isinstance(result, dict):
                raise ValueError(
                    "Model JSON response must be an object."
                )

            return result

        except json.JSONDecodeError:
            pass

        # -----------------------------------------------------
        # Markdown JSON
        # -----------------------------------------------------

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
                    "Model JSON response must be an object."
                )

            return result

        except json.JSONDecodeError:
            pass

        # -----------------------------------------------------
        # JSON embedded in surrounding text
        # -----------------------------------------------------

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:

            candidate = text[start:end + 1]

            try:
                result = json.loads(candidate)

                if not isinstance(result, dict):
                    raise ValueError(
                        "Extracted JSON must be an object."
                    )

                return result

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Model returned invalid JSON:\n"
            f"{raw}"
        )

    # =========================================================
    # ACTION NORMALIZATION
    # =========================================================

    def _normalize_actions(self, result: dict) -> dict:
        """Normalize model output into standard action format."""

        # -----------------------------------------------------
        # Standard format
        # -----------------------------------------------------

        if "actions" in result:

            actions = result["actions"]

            if not isinstance(actions, list):
                raise ValueError(
                    "Model 'actions' field must be a list."
                )

            return {
                "summary": result.get(
                    "summary",
                    "No summary provided.",
                ),
                "actions": actions,
            }

        # -----------------------------------------------------
        # Single action format
        # -----------------------------------------------------

        if "tool" in result:

            return {
                "summary": (
                    "Single action generated by the model."
                ),
                "actions": [result],
            }

        raise ValueError(
            "Model response does not contain "
            "'actions' or a valid 'tool'."
        )

    # =========================================================
    # ACTION PROMPT
    # =========================================================

    def _build_action_prompt(
        self,
        task: str,
        strict_retry: bool = False,
    ) -> str:

        retry_instruction = ""

        if strict_retry:

            retry_instruction = """
IMPORTANT RETRY:

Your previous response was NOT valid JSON.

DO NOT use:
<|tool_call_start|>
<|tool_call_end|>
[list_files()]
function calls
Markdown
or any other tool-call syntax.

Return ONLY a normal JSON object.

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

    # =========================================================
    # ACTION GENERATION
    # =========================================================

    def generate_actions(self, task: str) -> dict:
        """Generate structured coding actions."""

        for attempt in range(
            self.MAX_JSON_RETRIES + 1
        ):

            prompt = self._build_action_prompt(
                task,
                strict_retry=(attempt > 0),
            )

            raw = self.generate(prompt)

            try:

                result = self._parse_json_response(
                    raw
                )

                return self._normalize_actions(
                    result
                )

            except ValueError:

                if attempt >= self.MAX_JSON_RETRIES:
                    raise

                # One controlled retry.
                continue

        raise ValueError(
            "Unable to generate valid actions."
        )


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    provider = OpenRouterProvider()

    result = provider.generate_actions(
        "Inspect the workspace and fix any bug "
        "in the existing calculator."
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )