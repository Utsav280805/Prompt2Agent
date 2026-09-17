# Commands

Every command you need, in the order you'd actually need them. Run all of these from the
repo root (`D:\CHARUSAT\Sem-7\multi-agent-generator`).

Windows note: PowerShell needs the quotes around extras (`".[test]"`). In `cmd.exe` the
bare form (`.[test]`) also works. Everything below is written for PowerShell.

## 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Core generator only. Small install; no agent frameworks, no torch.
pip install -e .

# Plus the test runner, so you can verify the project itself.
pip install -e ".[test]"
```

The remaining extras are opt-in, because they are what made the install a gigabyte.
Install only the framework you actually generate code for, and only the provider you use:

```powershell
pip install -e ".[crewai]"              # or [langgraph] / [react] / [agno]
pip install -e ".[huggingface]"         # hosted open-source models. Small.
pip install -e ".[huggingface-local]"   # transformers + torch. Large (~2 GB).
pip install -e ".[ollama]"              # local Ollama server
pip install -e ".[openai]"              # or [anthropic] / [groq] / [watsonx]

pip install -e ".[open-source]"         # all frameworks + HF + Ollama, no paid SDKs
pip install -e ".[all]"                 # everything
```

## 2. Verify the project

```powershell
pip install -e ".[test,api]"
pytest
```

The suite is split by marker, and `pytest` alone runs everything except `live`:

```powershell
pytest -m unit           # pure logic: errors, secrets, the acceptance gate, selection, config
pytest -m integration    # real SQLite in a temp dir, real subprocesses, the ASGI app
pytest -m end_to_end     # the full generated-bundle proof
pytest -m live           # opt-in. Real provider, real key, real money. Excluded by default.
```

`tests/test_generated_bundles.py` compiles every framework x provider combination, then
generates a test bundle and runs it in a child `pytest` process with credentials stripped
from the environment. Exit code 0 means a generated bundle really does pass with no API
key, no network and no agent framework installed.

To run just the end-to-end proof:

```powershell
pytest tests/test_generated_bundles.py::test_generated_bundle_actually_passes -v
```

If something fails, the assertion message contains the child process's full stdout and
stderr, so the failing generated test is named directly in the output.

Every test carries a wall-clock timeout (`--timeout=60`). That is deliberate: a test that
reaches for a real model turns into a named failing test instead of a suite that never
finishes, which is the bug this project shipped once already.

## 2.1 The web application

Build the frontend once, then serve everything from one origin:

```powershell
cd frontend
npm install
npm run build
cd ..
multi-agent-generator --serve                  # http://127.0.0.1:8000
```

FastAPI serves the built React app itself, so there is no CORS in production and no second
process to manage. Interactive API docs are at http://127.0.0.1:8000/docs.

For frontend work, run the two dev servers side by side instead:

```powershell
multi-agent-generator --serve                  # terminal 1: API on :8000
cd frontend; npm run dev                       # terminal 2: UI on :5173
```

Vite proxies `/api` through to :8000, so the app talks to a same-origin API in development
exactly as it does in production.

Useful checks without a browser:

```powershell
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/providers
curl http://127.0.0.1:8000/api/frameworks
```

`/api/health` reports whether persistence is working, whether encryption is available and
whether a credential is configured. It never returns a key.

## 2.2 Generate a whole project

This is the v2 path: analyse, choose a framework, generate, review, improve, test.

```powershell
# Everything default. Writes to generated\<slug>\ and prints the verdict.
multi-agent-generator "Create a research assistant that finds papers and summarises them"

# Pick where it lands, and cap the improvement loop.
multi-agent-generator "Create a support triage team" --out-dir .\out\triage --max-iterations 2

# Force a framework instead of letting the selector choose and explain itself.
multi-agent-generator "Create a research assistant" --framework langgraph

# Choose the model that *designs* the agents.
multi-agent-generator "Create a research assistant" --provider openai --model gpt-4o-mini

# No model calls at all - heuristics only. No key needed, nothing billed.
multi-agent-generator "Create a research assistant" --offline

# Generate but do not run the tests, or skip generating them entirely.
multi-agent-generator "Create a research assistant" --no-run-tests
multi-agent-generator "Create a research assistant" --no-tests

# For CI: exit 1 unless the review passed AND the generated tests ran and passed.
multi-agent-generator "Create a research assistant" --strict
```

The last line is the one that matters. Without `--strict` the command writes the files and
tells you honestly whether they were verified; with it, "not verified" is a build failure.
When a project is not verified the reasons are printed verbatim - "the generated tests did
not run", "review scored 4.1" - rather than smoothed into a warning.

No credential handy? Set `DEFAULT_PROVIDER=mock` in `.env` and the whole pipeline runs with
no key, no network and no cost. The generated code is real; only the designing model is
simulated.

## 3. Credentials

Put these in a `.env` file at the repo root; it is loaded automatically.

```
OPENAI_API_KEY=sk-...          # provider: openai
ANTHROPIC_API_KEY=...          # provider: anthropic
GROQ_API_KEY=...               # provider: groq
GOOGLE_API_KEY=...             # provider: google
HF_TOKEN=hf_...                # provider: huggingface  (free)
WATSONX_API_KEY=...            # provider: watsonx
WATSONX_APIKEY=...             # same key again - this is the spelling the IBM SDK reads
WATSONX_PROJECT_ID=...
# huggingface-local, ollama and mock need no key at all.
```

`.env.example` at the repo root lists every variable with a comment, including the pipeline
and sandbox timeouts. Copy it rather than writing a `.env` from scratch:

```powershell
Copy-Item .env.example .env
```

A key can also be saved through the web app's Settings page, where it is encrypted at rest
and never sent back to the browser. Environment variables win over a saved key, which is
what makes CI predictable.

Free HF token: https://huggingface.co/settings/tokens

## 4. Orientation

```powershell
multi-agent-generator --help
multi-agent-generator --list-providers      # models, credentials, install hints, live status
multi-agent-generator --list-tools
multi-agent-generator --list-patterns
```

## 5. Generate agent code

```powershell
# Default: crewai + openai
multi-agent-generator "Create a research assistant that finds and summarises papers"

# Pick framework and provider
multi-agent-generator "Create a research assistant" --framework langgraph --provider huggingface

# Open-source, hosted (needs HF_TOKEN, no GPU)
multi-agent-generator "Create a research assistant" --provider huggingface

# Open-source, fully on this machine (downloads weights on first run)
multi-agent-generator "Create a research assistant" --provider huggingface-local

# Explicit model
multi-agent-generator "Create a research assistant" --provider huggingface --model Qwen/Qwen2.5-7B-Instruct

# Write to a file
multi-agent-generator "Create a research assistant" --framework crewai --output agent_system.py

# Config JSON instead of code, or both
multi-agent-generator "Create a research assistant" --format json --output config.json
multi-agent-generator "Create a research assistant" --format both
```

Frameworks: `crewai`, `crewai-flow`, `langgraph`, `react`, `react-lcel`, `agno`.
Providers: `openai`, `anthropic`, `watsonx`, `ollama`, `huggingface`, `huggingface-local`, `groq`.

CrewAI process type:

```powershell
multi-agent-generator "..." --framework crewai --process hierarchical
```

## 6. Generate a runnable test bundle

```powershell
# Alongside the agent code
multi-agent-generator "Create a research assistant" --framework crewai `
    --output agent_system.py --generate-tests

# From an existing config, with no LLM call and no credentials at all
multi-agent-generator --generate-tests --config config.json --framework crewai

# Choose the output directory (default: generated_tests/)
multi-agent-generator --generate-tests --config config.json --test-dir my_tests
```

That writes six files: the test module, `conftest.py`, `_helpers.py`, `pytest.ini`,
`requirements-test.txt` and a `README.md`.

Run the generated bundle:

```powershell
pip install -r generated_tests\requirements-test.txt
cd generated_tests
pytest
```

Offline by default: a deterministic fake model stands in for the real one and sockets are
blocked, so no key and no network are needed. Tests that need a real model are marked
`live` and deselected:

```powershell
pytest -m live                      # opt in, costs money
pytest -m contract                  # config validation only; needs nothing but pytest
$env:AGENT_MODULE="my_agents"; pytest -m live
```

## 7. Preflight dependency check

```powershell
multi-agent-generator --check-deps --framework crewai --provider huggingface
```

Prints what is installed, what is missing and the exact `pip install` to fix it. Exits
non-zero when something is missing, so it works as a CI gate. To get a requirements file:

```powershell
multi-agent-generator --emit-requirements --framework crewai --provider huggingface
multi-agent-generator --emit-requirements --framework crewai --provider huggingface --output requirements.txt
```

## 8. Other features

```powershell
# Custom tool
multi-agent-generator --tool "Fetch weather data from an API" --output weather_tool.py

# Orchestration
multi-agent-generator --orchestrate "agents that debate and reach consensus"
multi-agent-generator --pattern hierarchical --num-agents 4 --framework langgraph

# Evaluate an agent response
multi-agent-generator --evaluate --query "What is AI?" --response "AI is..." --threshold 0.8
```

## 9. Streamlit UI (legacy)

Kept working for anyone who relies on it, but it only drives the v1 single-file generator -
no pipeline, no review, no test run, no Playground. The React app in section 2.1 is the
supported interface.

```powershell
pip install -e ".[streamlit]"
streamlit run streamlit_app.py
```

## 10. Ollama, if you use it

```powershell
ollama serve
ollama pull llama3.2:3b
multi-agent-generator "Create a research assistant" --provider ollama
```
