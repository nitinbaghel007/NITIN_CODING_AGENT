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
| `OPENROUTER_API_KEY` | Yes* | `agent/providers/openrouter.py` | Default provider. If unset, the run stops with `OPENROUTER_API_KEY is not set.` |
| `GEMINI_API_KEY` | Yes* | `agent/providers/gemini.py` | Fallback provider used on rate-limit failover |
| `GEMINI_MODEL` | No | `agent/providers/gemini.py` | Defaults to `gemini-3.8-flash` |

\* At least one must be set for a real run. Setting only the Gemini key is **not**
currently enough, because OpenRouter is the default provider and a missing key
does not trigger failover — see *Current limitations*.

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
python main.py run "<task>" [--workspace <dir>] [--json]
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
* commands not on the safe allowlist require interactive approval (`[y/N]`)

### JSON output

```powershell
python main.py run "List the files" --workspace workspace --json
```

`--json` prints the final result object as JSON and makes the exit code reflect
`success`. **Known limitation:** the loop still prints its human-readable progress
(`STEP n/10`, actions, tool results) to stdout before the JSON document, so
stdout is not yet machine-parseable — see *Current limitations*.

---

## Running the tests

All three invocations collect only `tests/` and should report **82 passed**:

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

## Repository layout

```text
NITIN_CODING_AGENT/
├── main.py                    CLI entry point
├── pyproject.toml             project metadata + pytest configuration
├── requirements.txt           pinned dependencies
├── .env.example               environment variable template (placeholders)
├── .gitignore
├── agent/
│   ├── config/                (empty, planned configuration module)
│   ├── core/                  loop, safety, approval, diagnosis, state
│   ├── prompts/               (empty, planned prompt registry)
│   ├── providers/             openrouter, gemini, manager, base
│   └── tools/                 file, terminal, test, workspace tools
├── tests/                     82 unit tests
├── workspace/                 default agent sandbox (runtime output)
├── AUTONOMOUS_TEST_PROJECT/   intentionally buggy demo task
└── logs/                      (empty, reserved for future logging)
```

---

## Current limitations

Known issues from the code audit. None are fixed yet; they are tracked as
prioritized technical debt.

**Correctness / behaviour**

1. **`--json` is not machine-readable yet.** Progress lines are printed to stdout
   before the JSON payload.
2. **Missing API key does not fail over.** With only `GEMINI_API_KEY` set, the run
   stops at step 1 with `OPENROUTER_API_KEY is not set.` instead of switching to
   Gemini. Failover currently happens only on rate-limit errors.
3. **Safety blocklist uses substring matching**, so a legitimate command such as
   `python -m pytest tests/test_format.py` is blocked because it contains
   `format`.
4. **Safe-command allowlist is exact-match**, so `python -m pytest` is safe but
   `python -m pytest -q` requires approval.
5. **`pip install` / `npm install` can be approved but still cannot run** — the
   approval layer allows them, the terminal tool's first-word allowlist rejects
   them.
6. **Approval is interactive and blocking.** There is no `--yes` /
   `--non-interactive` flag, so unattended runs can stall or abort when an action
   needs approval.
7. **`list_files` is not recursive** — it returns only the top level of the
   workspace.
8. **Diagnosis `affected_area` is a static string**, not parsed from the actual
   traceback.

**Testing**

9. `workspace/` contains `test_calculator.py` in two places with the same module
   name, so running `python -m pytest` *inside* `workspace/` fails collection
   (pre-existing issue, unrelated to the project suite).
10. `agent/providers/openrouter.py`, `agent/tools/*` and the network paths of both
    providers have no direct unit tests.
11. No coverage measurement or CI is configured yet.

**Engineering hygiene**

12. No version control has been initialised yet (planned as Batch 2).
13. No logging framework — the loop prints to stdout; the `logs/` directory is
    reserved but unused.
14. Dead code still present: `agent/core/agent.py` (`NitinCodingAgent`) is never
    imported, and `ActionEngine.execute_actions()` is never called.
15. `agent/config/` and `agent/prompts/` are empty placeholders; prompts are
    hard-coded in three files.
16. Prompt/token usage grows with the full tool history embedded in every prompt.

---

## License

Not specified.
