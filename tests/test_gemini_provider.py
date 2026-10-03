import pytest

from agent.providers.gemini import GeminiProvider


def create_provider_without_api_key():
    provider = GeminiProvider.__new__(
        GeminiProvider
    )

    provider.api_key = None
    provider.model = "test-model"
    provider.client = None

    return provider


def test_parse_direct_json():
    provider = create_provider_without_api_key()

    raw = """
{
  "summary": "Inspect workspace",
  "actions": [
    {
      "tool": "list_files"
    }
  ]
}
"""

    result = provider._parse_json_response(raw)

    assert result["summary"] == "Inspect workspace"
    assert result["actions"][0]["tool"] == "list_files"


def test_parse_markdown_json():
    provider = create_provider_without_api_key()

    raw = '''```json
{
  "summary": "Read calculator",
  "actions": [
    {
      "tool": "read_file",
      "path": "calculator.py"
    }
  ]
}
```'''

    result = provider._parse_json_response(raw)

    assert result["summary"] == "Read calculator"
    assert result["actions"][0]["tool"] == "read_file"
    assert result["actions"][0]["path"] == "calculator.py"


def test_parse_embedded_json():
    provider = create_provider_without_api_key()

    raw = """
Here is the requested action:

{
  "summary": "Run tests",
  "actions": [
    {
      "tool": "run_tests"
    }
  ]
}

End of response.
"""

    result = provider._parse_json_response(raw)

    assert result["summary"] == "Run tests"
    assert result["actions"][0]["tool"] == "run_tests"


def test_normalize_multiple_actions():
    provider = create_provider_without_api_key()

    result = provider._normalize_actions(
        {
            "summary": "Inspect project",
            "actions": [
                {
                    "tool": "list_files"
                },
                {
                    "tool": "read_file",
                    "path": "calculator.py",
                },
            ],
        }
    )

    assert result["summary"] == "Inspect project"
    assert len(result["actions"]) == 2
    assert (
        result["actions"][1]["path"]
        == "calculator.py"
    )


def test_normalize_single_action():
    provider = create_provider_without_api_key()

    result = provider._normalize_actions(
        {
            "tool": "list_files"
        }
    )

    assert len(result["actions"]) == 1
    assert (
        result["actions"][0]["tool"]
        == "list_files"
    )


def test_normalize_invalid_response():
    provider = create_provider_without_api_key()

    with pytest.raises(ValueError):
        provider._normalize_actions(
            {
                "summary": "Invalid response"
            }
        )


def test_parse_invalid_json():
    provider = create_provider_without_api_key()

    with pytest.raises(ValueError):
        provider._parse_json_response(
            "This is not JSON."
        )


def test_empty_response():
    provider = create_provider_without_api_key()

    with pytest.raises(ValueError):
        provider._parse_json_response("")