# Nitin Coding Agent

An autonomous CLI coding agent. You give it a natural-language coding task and a
workspace directory; it drives a bounded loop that inspects files, edits them,
runs the test suite, diagnoses failures, and only reports success after the tests
have actually passed.

The agent talks to a large language model to decide **one tool action per step**,
validates every action through a safety layer before executing it, and stops
safely when its step budget or fix budget is exhausted.

---

## What it does

| Stage | Module |
|---|---|
| CLI entry point (`run` subcommand) | `main.py` |
| 10-step orchestration loop | `agent/core/coding_loop.py` |
| Safety classification (`safe` / `approval` / `blocked`) | `agent/core/safety.py` |
| Interactive approval gate | `agent/core/approval.py` |
| Action validation and dispatch | `agent/core/action_engine.py` |
| Tools: files, terminal, tests, listing | `agent/tools/` |
| Failure diagnosis | `agent/core/diagnosis.py` |
| Fix-loop budgeting | `agent/core/fix_loop.py` |
| Transient provider-error retries | `agent/core/error_recovery.py` |
| Runtime state (history, files, test status) | `agent/core/context_manager.py` |
| LLM providers + failover | `agent/providers/` |

**Available tools:** `list_files`, `read_file`, `write_file`, `run_command`,
`run_tests`.

**Providers:** `openrouter` (default) and `gemini`. On a rate-limit error the loop
fails over to the next free provider and continues the same task.

---

## Requirements

* **Python 3.13** (developed and verified on **3.13.14**)
* Windows is the currently supported platform — the safety rules in
  `agent/core/safety.py` are written for Windows commands and paths
* Network access to `openrouter.ai` and/or `generativelanguage.googleapis.com`
  when running real tasks
* An API key for at least one provider

---

## Setup

### 1. Create the virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1

# cmd
.venv\Scripts\activate.bat
```

### 3. Install the pinned dependencies

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` contains the exact pins from the audited environment
(`requests==2.34.2`, `google-genai==2.25.0`, `pytest==9.1.1` plus their
transitive dependencies).

---

## API keys and configuration

Configuration is read from **environment variables only**. See `.env.example`
for the template — it contains placeholders, never real keys.

| Variable | Required | Used by | Notes |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes* | `agent/providers/openrouter.py` | Default provider. If unset, the loop fails over to Gemini |
| `GEMINI_API_KEY` | Yes* | `agent/providers/gemini.py` | Fallback provider used during failover |
| `GEMINI_MODEL` | No | `agent/providers/gemini.py` | Defaults to `gemini-3.8-flash` |

\* At least one must be set for a real run. With only `GEMINI_API_KEY` set, the
loop fails over to Gemini on the first step — a missing OpenRouter key is
treated as a failover error (verified offline: the keyless run switches
`openrouter → gemini` at step 1).

```powershell
# PowerShell - session scoped
$env:OPENROUTER_API_KEY = "your-openrouter-api-key"
$env:GEMINI_API_KEY      = "your-gemini-api-key"
$env:GEMINI_MODEL        = "gemini-3.8-flash"
```

```bash
# bash - session scoped
export OPENROUTER_API_KEY="your-openrouter-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export GEMINI_MODEL="gemini-3.8-flash"
```

Keys are never written to disk or logged by the agent.

---

## CLI usage

```text
python main.py run "<task>" [--workspace <dir>] [--json] [--non-interactive]
```

### Basic run

```powershell
python main.py run "Inspect the project, fix the failing tests, and verify them" --workspace workspace
```

Exit code `0` means the task completed with verified tests; `1` means the run
failed, hit the step limit, ran out of providers, or a CLI error occurred.

### Workspace

`--workspace` points at the directory the agent is allowed to touch. It defaults
to `workspace` and **must already exist** — the CLI rejects a missing path with
`Workspace does not exist`.

```powershell
mkdir mytask
python main.py run "Add a multiply function with tests" --workspace mytask
```

Everything the agent reads or writes is confined to that directory:

* relative paths only, absolute paths are blocked
* `..` traversal and symlinks that escape the directory are blocked
* unknown tools are blocked by default
* commands not on the safe allowlist require interactive approval (`[y/N]`);
  pass `--non-interactive` to deny approval-required actions safely instead
  of prompting (unattended runs never block)

### JSON output

```powershell
python main.py run "List the files" --workspace workspace --json
```

`--json` prints **exactly one JSON document on stdout**; all human-readable
progress (`STEP n/10`, actions, tool results) and diagnostics go to stderr, so
stdout is machine-parseable. The exit code reflects `success` (`0` only when
the task completed with verified tests).

---

## Running the tests

All three invocations collect only `tests/` and should report **468 passed**:

```powershell
python -m pytest          # via python -m
pytest                    # via the venv console script
python -m pytest tests    # explicit path
```

The configuration lives in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

* `testpaths` keeps `pytest` from collecting the sandbox copies under
  `workspace/` and the fixture under `AUTONOMOUS_TEST_PROJECT/`, which contain
  test files with duplicate module names
* `pythonpath` puts the project root on `sys.path` so `import main` and
  `from agent...` work no matter which of the three commands you use

### Other useful commands

```powershell
python -m pytest tests -q                      # quiet
python -m pytest tests/test_fix_loop.py        # one module
python -m pytest AUTONOMOUS_TEST_PROJECT       # the buggy demo fixture (expected: 1 failed, 1 passed)
```

### Smoke checks

```powershell
python main.py --help
python main.py run "noop" --workspace workspace --json
```

The second command returns exit code `1` when no API key is configured — that is
the expected error path, not a crash.

---

## Continuous integration

`.github/workflows/ci.yml` runs on every **push** and **pull request**:

* Windows runner (the documented supported platform), **Python 3.13**
* `python -m pip install -r requirements.txt` — the same pinned dependencies
  used locally, into a clean environment
* `python -m pytest` — the exact local command; **any** test failure fails the
  workflow (nothing is skipped, filtered, or allowed to fail)

The workflow adds no coverage upload, matrix, services, or deployment jobs —
the repository does not require them.

---

## Repository layout

```text
NITIN_CODING_AGENT/
├── main.py                    CLI entry point
├── pyproject.toml             project metadata + pytest configuration
├── requirements.txt           pinned dependencies
├── .env.example               environment variable template (placeholders)
├── .gitignore
├── .github/workflows/ci.yml   GitHub Actions CI (pytest on push / pull request)
├── agent/
│   ├── config/                settings + centralized logging setup
│   ├── core/                  loop, safety, approval, diagnosis, state
│   ├── prompts/               (empty, reserved)
│   ├── providers/             openrouter, gemini, manager, base
│   └── tools/                 file, terminal, test, workspace tools
├── tests/                     468 tests across 22 modules
├── workspace/                 default agent sandbox (runtime output)
├── AUTONOMOUS_TEST_PROJECT/   intentionally buggy demo task
└── logs/                      (empty, reserved; logging writes to stderr)
```

---

## Current limitations

Known issues from the code audit, current as of the CI/release batch. Items
that later batches demonstrably fixed (machine-readable `--json`, the
`--non-interactive` flag, missing-key failover to Gemini, centralized logging,
direct tests for providers and tools, version control, and CI) have been
removed; the rest are tracked as prioritized technical debt.

**Correctness / behaviour**

1. **Safety blocklist uses substring matching**, so a legitimate command such as
   `python -m pytest tests/test_format.py` is blocked because it contains
   `format`.
2. **Safe-command allowlist is exact-match**, so `python -m pytest` is safe but
   `python -m pytest -q` requires approval.
3. **`pip install` / `npm install` can be approved but still cannot run** — the
   approval layer allows them, the terminal tool's first-word allowlist rejects
   them.
4. **`list_files` is not recursive** — it returns only the top level of the
   workspace.
5. **Diagnosis `affected_area` is a static string**, not parsed from the actual
   traceback.

**Testing**

6. `workspace/` contains `test_calculator.py` in two places with the same module
   name, so running `python -m pytest` *inside* `workspace/` fails collection
   (pre-existing issue, unrelated to the project suite).
7. No coverage measurement is configured — CI runs the full suite but does not
   report coverage.

**Engineering hygiene**

8. Dead code removed in Batch 8: `agent/core/agent.py` (`NitinCodingAgent`) and
   the unused `ActionEngine.execute_actions()` helper were deleted after a
   repository-wide search confirmed nothing referenced them.
9. `agent/prompts/` is an empty placeholder; prompts are hard-coded.
10. Prompt/token usage grows with the full tool history embedded in every prompt
    (bounded by the 10-step loop — see the Batch 11 performance tests).

---

## License

Not specified.
