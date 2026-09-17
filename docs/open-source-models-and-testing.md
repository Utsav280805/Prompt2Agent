# Open-Source Models and Testable Generated Agents

This document covers two changes to `multi-agent-generator`: adding Hugging Face as a
first-class source of models (both hosted and on-device), and fixing the reason that
generated agent code cannot currently be tested.

The two problems are related more closely than they first appear. Both come from the same
root cause: the project assumes OpenAI everywhere, and it never writes down what a
generated project actually needs in order to run. Fixing the second problem properly
requires the provider abstraction introduced by the first.

## Part 1 — Hugging Face support

### Where OpenAI is currently hardcoded

There are two separate places a model gets chosen, and they are easy to conflate.

The first is the **generator's own model** — the LLM that reads your plain-English prompt
and returns an agent configuration as JSON. This lives in `AgentGenerator._initialize_model`
and already routes through LiteLLM, so it is provider-agnostic in principle. In practice its
`default_models` dictionary knows only `openai`, `watsonx` and `ollama`, and anything else
falls through to a branch that passes the provider *name* where a model *id* belongs.

The second is the **model written into the generated code** — the LLM the created agents will
actually use at runtime. This is hardcoded to OpenAI in every single framework generator:
`langgraph_generator.py` and both ReAct generators emit `ChatOpenAI`, `agno_generator.py`
emits `OpenAIChat`, and the two CrewAI generators emit no LLM at all and therefore inherit
CrewAI's OpenAI default.

The consequence is that even if you generated your agents using a local Ollama model, the
file you got back still demanded an `OPENAI_API_KEY` before it would run. The project was
provider-agnostic for itself but not for its output. Both halves need to change for the
project to be meaningfully open-source, which is why the chosen scope covers both.

### Two Hugging Face paths, not one

"Use Hugging Face" is ambiguous, and the two readings have very different footprints, so
both are implemented as distinct providers rather than one blurred option.

The `huggingface` provider is the **hosted** path. Requests go out to Hugging Face's
inference service over HTTP, authenticated with an `HF_TOKEN`. It needs no GPU, no model
download and no extra heavyweight dependency, because LiteLLM already speaks this protocol
via its `huggingface/` model prefix. This is the path most users should start on, and it is
what `--provider huggingface` selects.

The `huggingface-local` provider is the **on-device** path. Weights are downloaded once from
the Hub and executed in-process through `transformers`, so it runs fully offline and costs
nothing per token. The price is a multi-gigabyte download and real hardware requirements.
Because `torch` and `transformers` are far too heavy to force on every user, this path lives
behind an optional extra and its imports happen lazily, inside the call that needs them,
rather than at module import time. The default model is deliberately tiny
(`Qwen/Qwen2.5-0.5B-Instruct`) so that a first run on a student laptop completes instead of
exhausting the disk; larger models are a config change, not a code change.

### A single provider registry

Rather than scatter provider knowledge across six generators plus the inference layer, all of
it moves into one module, `multi_agent_generator/providers.py`. For each provider that module
records the default model id, which environment variables carry credentials, which pip extras
are required, and — crucially — how to construct that provider's LLM object in each of the
three framework families the project emits code for.

This is what makes the change tractable. Adding a provider becomes a matter of adding one
entry to one table, and the generator and the generated code can no longer disagree about
what a provider means, because they read the same table.

### Native classes versus OpenAI-compatible endpoints

One deliberate engineering decision deserves calling out, because it looks like a shortcut
and is not.

For the LangChain-based frameworks (LangGraph, ReAct, ReAct-LCEL) the generated code uses the
official `langchain-huggingface` partner package: `HuggingFaceEndpoint` wrapped in
`ChatHuggingFace` for the hosted path, and `HuggingFacePipeline.from_model_id` wrapped the
same way for the local path. These are the documented, stable, native integrations.

For CrewAI, the generated code uses CrewAI's own `LLM` class with a `huggingface/`-prefixed
model string. CrewAI routes through LiteLLM internally, so this is native too.

For Agno, the generated code points Agno's existing `OpenAIChat` class at Hugging Face's
OpenAI-compatible router (`https://router.huggingface.co/v1`) with the HF token as the API
key, instead of importing an Agno-specific Hugging Face class. This is intentional. Hugging
Face's router genuinely implements the OpenAI chat-completions API, so the request is correct
and the response parses; meanwhile the approach depends only on a class that is already
present in the current generated output and therefore known to exist at the pinned Agno
version. Guessing at a native class name that may or may not exist in `agno==2.3.24` would
trade a working integration for a plausible-looking `ImportError`. The generated file carries
a comment saying exactly this, so the choice is visible to whoever reads the output.

The same reasoning covers a genuine gap: neither CrewAI nor Agno can execute a `transformers`
pipeline in-process. For `huggingface-local` with those two frameworks, the generated code
points at a local OpenAI-compatible server (vLLM, TGI or llama.cpp) via `HF_LOCAL_BASE_URL`,
defaulting to `http://localhost:8000/v1`, and says so in a comment. Emitting code that
pretended to run local weights inside CrewAI would be worse than emitting code that tells you
what you actually need to start.

## Part 2 — Why generated agents cannot be tested

### The root cause

The headline problem is that `crewai` is not a dependency of this project. It appears nowhere
in `pyproject.toml`. Yet `crewai_generator.py` emits `from crewai import Agent, Task, Crew,
Process`, and `test_generator.py` emits the very same import inside the pytest fixture it
generates. So the tool generates a test suite whose first action is to import a package the
tool never told anyone to install. On a clean machine that is a collection-time `ImportError`,
before a single test body runs.

That is the largest instance of a general pattern: **the generated project has dependencies,
and nothing writes them down.** A generated file is handed over as a bare `.py` with no
`requirements.txt`, no `conftest.py`, no pytest configuration, and no statement of which
provider credentials it expects.

### Four more failures behind the first

Removing the missing-package error is not enough, because several independent problems sit
directly behind it.

Generated tests are decorated with `@pytest.mark.timeout(...)`, but `pytest-timeout` is not
declared either — not even in the `dev` extra. Without that plugin the mark is silently
inert, and under `--strict-markers` it is a hard error.

Generated tests call real models. The setup fixture does
`os.environ.setdefault("OPENAI_API_KEY", "test-key")` and then constructs live agents. A fake
key does not produce a passing test; it produces a `401` on every test that touches a model.
With a real key it produces a bill. Neither is a test suite anyone can run in CI.

The generated test bodies do not test anything. Every one of them ends in
`result = None` followed by `assert True`, with the intended assertions emitted as comments.
The suite reports green while exercising nothing, which is worse than reporting red.

And the emitted assertions were never executable to begin with. Strings like
`"response contains information from both agents"` and `"no memory leaks detected"` are English
prose, not Python expressions, so they could never have been uncommented into working code.

Finally, the framework packages that *are* declared are declared too aggressively. `langgraph`,
`streamlit` and a hard-pinned `agno==2.3.24` are all mandatory installs, so everyone pays for
every framework and the pin is free to collide with whatever else is in the environment. There
is also a latent bug waiting there: `generator.py` does a top-level `import streamlit as st`
and then guards its uses with `if st is not None`, which shows the intent was for Streamlit to
be optional, but the unguarded import means the CLI dies without a UI package installed.

### The fix

The strategy has three parts, matching the root causes rather than the symptoms.

**Declare dependencies honestly.** Heavy framework packages move out of mandatory
`dependencies` into per-framework extras (`crewai`, `agno`, `langgraph`, `ui`), with `crewai`
finally added and the `agno` pin loosened to a compatible range. New `huggingface` and
`huggingface-local` extras cover the two new providers, `pytest-timeout` joins a proper `test`
extra, and an `all` extra keeps one-line installation available. The Streamlit import in
`generator.py` becomes a guarded optional import so the CLI works without it.

**Make generated tests runnable offline.** Each generated suite begins with
`pytest.importorskip` on its framework, so on a machine without CrewAI the suite *skips* with
a clear reason instead of erroring — an honest signal rather than a crash. A generated
`conftest.py` supplies a `FakeLLM` double and stubs credentials, so tests exercise
construction, wiring and error handling with no network and no API key. Real assertions
replace the `assert True` placeholders, and the assertion strings become actual Python
expressions. Tests that genuinely need a live model are marked `live` and deselected by
default, so opting into cost is explicit.

**Ship a bundle, not a file.** Test generation now emits `conftest.py`, `pytest.ini` (with
markers registered, killing the strict-markers failure), `requirements-test.txt` generated
from the same provider registry that drove code generation, and a short README — alongside the
test module itself. A generated project can be unzipped and run with `pip install -r` then
`pytest`.

**Tell the user before they hit it.** A new `dependencies.py` maps each framework and provider
to its pip requirements and import names, checks what is actually installed, and reports what
is missing with the exact `pip install` line to fix it. `--check-deps` exposes this on the
command line, and code generation warns when it emits code for a framework that is not
installed. The `pip install -r requirements-test.txt` line in the emitted bundle comes from
this same map, so the checker, the extras and the generated manifest cannot drift apart.

## Verification

The changes are checked by generating a bundle for every framework and provider combination
and confirming three things: that the emitted Python compiles, that the emitted test suite
skips cleanly on a machine with no framework packages and no credentials, and that with the
generated `conftest.py` fakes in place the suite executes real assertions and passes. The
first of these also covers the pre-existing codegen bugs found during this work — an
undefined `BaseMessage` in the LangGraph output, unannotated pydantic-v2 `BaseTool` fields in
the LangGraph and ReAct output, and bare tool-name strings passed where CrewAI expects tool
objects — each of which would have broken any generated test at import or validation time
regardless of dependencies.
