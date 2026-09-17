# Generated tests for a crewai agent system

Provider: **openai** (OpenAI)

## Running the tests

```bash
pip install -r requirements-test.txt
pytest
```

That runs everything except the `live` tests. No API key and no network access are
needed, because a deterministic fake model stands in for the real one and sockets are
blocked for the duration of each offline test.

## What runs, and what needs installing

The suite is layered so that something meaningful runs in every environment.

Contract tests validate the agent configuration itself, checking that agent names are
unique and sanitise to valid Python identifiers, that every task points at an agent that
exists, and that required fields are present. They need no framework packages at all, so
they always run.

Construction tests build real `crewai` objects and assert on their wiring. They
need `crewai` installed; without it they *skip* with a stated reason rather than
failing collection.

Live tests actually call a model. They are marked `live` and deselected by default:

```bash
export OPENAI_API_KEY=...
pytest -m live
```

Live tests import your generated agent module. Save it next to these tests as
`agent_system.py`, or point `AGENT_MODULE` at it:

```bash
AGENT_MODULE=my_agents pytest -m live
```

## Credentials

Environment variables for this provider: OPENAI_API_KEY, API_KEY



## Checking dependencies

```bash
multi-agent-generator --check-deps --framework crewai --provider openai
```
