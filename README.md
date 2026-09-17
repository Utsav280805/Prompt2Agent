# Multi-Agent Generator

<p align="center">
  <img alt="mag-banner" src="https://raw.githubusercontent.com/Utsav280805/Prompt2Agent/main/mag-banner.png" />
</p>

<p align="center">
  <a href="https://pypi.org/project/multi-agent-generator/"><img src="https://img.shields.io/pypi/v/multi-agent-generator?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pepy.tech/projects/multi-agent-generator"><img src="https://static.pepy.tech/personalized-badge/multi-agent-generator?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads" alt="Downloads"></a>
  <a href="https://github.com/Utsav280805/Prompt2Agent"><img src="https://img.shields.io/github/stars/Utsav280805/Prompt2Agent?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/Utsav280805/Prompt2Agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License MIT"></a>
</p>
<p align="center">
  <a href="https://docs.pydantic.dev/"><img src="https://img.shields.io/badge/Pydantic-v2-E92063.svg" alt="Pydantic v2"></a>
  <a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit"></a>
  <a href="https://Utsav280805.github.io/Prompt2Agent/"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="Docs"></a>
</p>

Describe a task in plain English and get a **working multi-agent project**: analysed, designed,
generated as real files, reviewed, improved, tested, and runnable — through a web app, an HTTP API,
or the CLI. Provider-agnostic (OpenAI, Anthropic, Google, Groq, Hugging Face, Ollama, WatsonX)
with a Hugging Face default that has a free tier.

### What's New in v2.0.0

The tool used to hand you a file and wish you luck. It now takes responsibility for the result.

- **A real pipeline, not a single prompt.** Analyse → select framework → select architecture →
  configure the LLM → generate → review → improve → test → execute in a sandbox → validate.
  Every stage streams its progress and every stage is recorded.
- **A reviewer and an improvement loop.** Generated code is scored 0–10 with specific issues, and
  failures are fed back for up to `MAX_GENERATION_ITERATIONS` attempts.
- **Verification you can trust.** A project is only marked verified when the review passed *and*
  the generated tests actually ran and passed. "Skipped", "could not install" and "timed out" are
  each reported as themselves — never as success. When it is not verified, the reasons are listed
  verbatim.
- **A React + TypeScript web app.** Create a project, watch the pipeline, browse the generated
  files, read the review, read the test output, and run the agent in a Playground. Started with
  `multi-agent-generator --serve`.
- **A FastAPI backend.** Every endpoint returns the same error shape — `code`, `message`,
  `action` — so a client needs one error component, not a guess per endpoint.
- **A provider abstraction.** One `LLMProvider` interface, one `LLMConfig` that is the only object
  holding a credential, and that credential is wrapped so it cannot be printed by accident.
- **Nothing hangs.** Every subprocess has a wall clock and is killed along with its whole process
  tree. Every model call has a timeout.
- **Persistent state.** Projects, runs and per-stage logs are stored in SQLite. Saved API keys are
  encrypted at rest and are never sent back to the browser.

### What's New in v1.1.0
- **Open-Source Models via Hugging Face** - hosted inference or fully local `transformers`, backing both the generator and the code it writes
- **Generated Tests That Actually Run** - offline by default, with a complete bundle (conftest, pytest.ini, requirements) instead of a lone file
- **Dependency Preflight** - `--check-deps` tells you what is missing before you hit it
- **Slimmer Install** - frameworks moved to extras; `pip install multi-agent-generator` no longer pulls a gigabyte of packages you may not use

### What's New in v1.0.0
- **Tool Auto-Discovery & Generation** - 15+ pre-built tools + natural language tool creation
- **Multi-Agent Orchestration Patterns** - Supervisor, Debate, Voting, Pipeline, MapReduce
- **Evaluation & Testing Framework** - Auto-generated tests + output quality metrics
- **CLI Support** - Full CLI commands for tools, evaluation, and orchestration

---

## Quick start

```bash
pip install -e ".[api,huggingface,crewai,test]"
cp .env.example .env          # then set HF_TOKEN (free: huggingface.co/settings/tokens)
```

Generate from the command line:

```bash
multi-agent-generator "Create a research assistant that finds papers and summarises them"
```

That runs the whole pipeline, writes the project to `generated/<slug>/`, and prints whether it was
verified. Add `--strict` to exit non-zero when it was not — that is the flag for CI.

Run the web app:

```bash
cd frontend && npm install && npm run build && cd ..
multi-agent-generator --serve                 # http://127.0.0.1:8000
```

During frontend development, run the API and Vite separately:

```bash
multi-agent-generator --serve                 # terminal 1: API on :8000
cd frontend && npm run dev                    # terminal 2: UI on :5173, proxying to the API
```

No key yet? Set `DEFAULT_PROVIDER=mock` and the entire pipeline runs offline, with no key, no
network and no cost. The generated code is real; only the model designing it is simulated.

### The CLI, briefly

| Flag | Effect |
| --- | --- |
| *(prompt only)* | Run the full pipeline: analyse, generate, review, improve, test |
| `--framework` | Force a framework instead of letting the selector choose and explain |
| `--provider`, `--model` | Choose the model that *designs* the agents |
| `--out-dir DIR` | Where to write the project (default `generated/<slug>`) |
| `--max-iterations N` | Cap the review/improve loop |
| `--no-tests`, `--no-run-tests` | Skip generating tests, or generate but do not run them |
| `--offline` | No model calls at all — heuristics only |
| `--strict` | Exit 1 unless the project was verified |
| `--serve`, `--host`, `--port` | Start the web application |
| `--config PATH` | Generate from an existing configuration, no credential needed |
| `--legacy` | The v1 single-file generator, unchanged |
| `--check-deps` | Report which framework and provider packages are missing |

### Running the test suite

```bash
pytest                 # unit + integration; no network, no credentials, no cost
pytest -m unit         # pure logic only
pytest -m integration  # SQLite, subprocesses, the ASGI app
pytest -m live         # opt-in: real providers, real keys, real money
```

`live` tests are excluded by default and every test carries a wall-clock timeout, so a test that
reaches for a real model becomes a named failure rather than a suite that never finishes.

---

## Features

### Agent Generation

* Generate agent code for multiple frameworks:

  * **CrewAI**: Structured workflows for multi-agent collaboration
  * **CrewAI Flow**: Event-driven workflows with state management
  * **LangGraph**: LangChain's framework for stateful, multi-actor applications
  * **Agno**: Agno framework for Agents Team orchestration
  * **ReAct (classic)**: Reasoning + Acting agents using `AgentExecutor`
  * **ReAct (LCEL)**: Future-proof ReAct built with LangChain Expression Language (LCEL)

* **Provider-Agnostic Inference** via LiteLLM:

  * Supports OpenAI, IBM WatsonX, Ollama, Anthropic, and more
  * Swap providers with a single CLI flag or environment variable

* **Flexible Output**:

  * Generate Python code
  * Generate JSON configs
  * Or both combined

### Tool Auto-Discovery & Generation (NEW!)

Create tools for your agents using plain English — no coding required:

```python
from multi_agent_generator.tools import ToolRegistry, ToolGenerator

# Browse 15+ pre-built tools across 10 categories
registry = ToolRegistry()
web_tools = registry.list_by_category("web_search")
all_tools = registry.list_all()

# Generate custom tools from natural language
generator = ToolGenerator()
tool = generator.generate_from_description("Create a tool that fetches weather data for a city")
print(tool.code)  # Ready-to-use Python code!
```

**Pre-built Tool Categories:**
| Category | Examples |
|----------|----------|
| Web Search | Google search, web scraper |
| File Operations | Read, write, list files |
| Data Processing | CSV parser, JSON transformer |
| Code Execution | Python executor, shell runner |
| API Integration | REST client, webhook handler |
| Database | SQL query, document store |
| Communication | Email sender, Slack notifier |
| Math | Calculator, statistics |
| Text Processing | Summarizer, translator |
| Image Processing | Resizer, format converter |

### Multi-Agent Orchestration Patterns (NEW!)

Choose from 5 battle-tested patterns to coordinate your agents:

```python
from multi_agent_generator.orchestration import Orchestrator, PatternType

orchestrator = Orchestrator()

# Generate orchestrated system from description
result = orchestrator.generate_from_description(
    "I need a research team where a manager delegates to specialists"
)
print(result["code"])  # Complete LangGraph/CrewAI code!

# Or configure manually
config = orchestrator.create_pattern_config(
    pattern_type=PatternType.SUPERVISOR,
    agents=["researcher", "writer", "reviewer"],
    task_description="Analyze market trends"
)
```

**Available Patterns:**

| Pattern | Use Case | How It Works |
|---------|----------|--------------|
| **Supervisor** | Delegating tasks to specialists | Central coordinator routes work |
| **Debate** | Reaching consensus | Agents discuss & refine answers |
| **Voting** | Democratic decisions | Agents vote on best response |
| **Pipeline** | Sequential processing | Chain of specialized steps |
| **MapReduce** | Parallel processing | Split, process, aggregate |

### Evaluation & Testing Framework (NEW!)

Auto-generate runnable tests and evaluate agent quality:

```python
from multi_agent_generator.evaluation import TestGenerator, AgentEvaluator

# Generate a complete, runnable test bundle
test_gen = TestGenerator()
bundle = test_gen.generate_bundle(
    config=your_config,
    framework="crewai",
    provider="huggingface",
)
bundle.save("generated_tests/")   # test module + conftest + pytest.ini + requirements

# Or build the suite object first, if you want to inspect or filter it
from multi_agent_generator.evaluation import TestType

suite = test_gen.generate_test_suite(
    your_config,
    framework="crewai",
    include_types=[TestType.CONTRACT, TestType.UNIT, TestType.INTEGRATION],
)
print(len(suite.test_cases))
suite.save("generated_tests/")

# Evaluate agent output quality
evaluator = AgentEvaluator()
result = evaluator.evaluate(
    "Analyze Q4 sales data",                  # the query
    "The analysis shows revenue grew 12%...", # the agent's response
    ground_truth="Market trends indicate...", # optional, enables accuracy scoring
)
print(result.metrics.overall_score())  # 0.0 - 1.0
print(result.metrics.relevance_score, result.metrics.coherence_score)
```

**Test Types:**
- Contract Tests - Validate the agent configuration itself (no dependencies needed)
- Unit Tests - Individual agent construction
- Integration Tests - Multi-agent wiring and handoffs
- End-to-End Tests - Full workflow validation (`live`)
- Performance Tests - Response time & peak memory (`live`)
- Reliability Tests - Error handling & recovery
- Quality Tests - Output quality metrics (`live`)

### Streamlit UI

* Interactive prompt entry
* Framework selection
* **Tool discovery & generation** (NEW!)
* **Orchestration pattern configuration** (NEW!)
* **Evaluation & testing dashboard** (NEW!)
* Config visualization
* Copy or download generated code

---

## Installation

### Basic Installation

```bash
pip install multi-agent-generator
```

That gives you the generator, the CLI and the Streamlit UI. It deliberately does **not**
install the agent frameworks, because this tool *writes* CrewAI, LangGraph, ReAct and
Agno code without ever importing those packages itself. Install the one you actually
generate for:

```bash
pip install 'multi-agent-generator[crewai]'        # CrewAI and CrewAI Flow
pip install 'multi-agent-generator[langgraph]'     # LangGraph
pip install 'multi-agent-generator[react]'         # ReAct classic and LCEL
pip install 'multi-agent-generator[agno]'          # Agno
```

...and the provider you want to run them on:

| Extra | Installs | Use it for |
|-------|----------|------------|
| `openai` | `langchain-openai` | OpenAI models |
| `anthropic` | `langchain-anthropic` | Claude models |
| `groq` | `langchain-groq` | Groq-hosted open models |
| `ollama` | `langchain-ollama` | A local Ollama server |
| `watsonx` | `ibm-watsonx-ai`, `langchain-ibm` | IBM WatsonX |
| `huggingface` | `huggingface-hub`, `langchain-huggingface` | **Open-source models, hosted by Hugging Face** |
| `huggingface-local` | `transformers`, `torch`, `accelerate` | **Open-source models on your own machine** |
| `test` | `pytest`, `pytest-timeout`, `langchain-core` | Running generated test suites |

Two convenience bundles:

```bash
pip install 'multi-agent-generator[open-source]'   # all frameworks + Hugging Face + Ollama
pip install 'multi-agent-generator[all]'           # everything except local torch
```

If you would rather not memorise any of this, ask the tool:

```bash
multi-agent-generator --check-deps --framework crewai --provider huggingface
```

It prints exactly what is missing and the `pip install` line that fixes it, and exits
non-zero when something is absent so you can use it as a CI preflight gate.

---

## Open-Source Models via Hugging Face

You are not tied to OpenAI. Two Hugging Face paths are supported, and both work for the
generator's own analysis step *and* for the code it writes, so the agents you generate
actually run on open weights.

### Hosted inference

The model runs on Hugging Face's infrastructure, so the install stays small.

```bash
pip install 'multi-agent-generator[crewai,huggingface]'
export HF_TOKEN=hf_...

multi-agent-generator "Research assistant that summarizes papers" \
  --framework crewai --provider huggingface
```

Pick a specific model with `--model`, or set `HF_MODEL_ID`:

```bash
multi-agent-generator "Research assistant" --provider huggingface \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### Fully local inference

Nothing leaves your machine, and no API key is needed. The default model is deliberately
tiny (`Qwen/Qwen2.5-0.5B-Instruct`) so a first run does not download several gigabytes;
point `--model` at something larger once you know it works.

```bash
pip install 'multi-agent-generator[crewai,huggingface-local]'

multi-agent-generator "Research assistant" \
  --framework crewai --provider huggingface-local
```

A caveat worth knowing: small open-weight models are less reliable at emitting strict
JSON than the large hosted ones, so prompt analysis occasionally needs a retry. The
generator recovers from near-valid JSON where it can and falls back to a sensible default
configuration when it cannot, rather than crashing.

To see every provider, its default models, whether its credentials are configured and
what to install:

```bash
multi-agent-generator --list-providers
```

---

## Testing Generated Agents

Generating agents is the point of this tool, so the tests it generates have to actually
run. Ask for them alongside the code:

```bash
multi-agent-generator "Research assistant" --framework crewai \
  --output agent_system.py --generate-tests
```

That writes a complete, runnable test project rather than a lone file:

```
generated_tests/
├── test_crewai_agents.py     # the suite
├── conftest.py               # offline fixtures: fake model, blocked sockets
├── _helpers.py               # sanitisation helpers the contract tests use
├── pytest.ini               # registered markers, live tests deselected
├── requirements-test.txt     # every package the suite needs, pinned by floor
└── README.md                 # how to run it
```

```bash
pip install -r generated_tests/requirements-test.txt
cd generated_tests && pytest
```

The suite runs **offline**: no API key, no network, no billed calls. A deterministic fake
model stands in for the real one, and sockets are blocked for the duration of each test,
so an accidental live call fails loudly instead of quietly costing money.

Tests are layered so something meaningful runs in every environment:

* **Contract tests** validate the configuration itself — agent names unique and
  sanitising to valid Python identifiers, tasks pointing at agents that exist, required
  fields present. No framework packages needed, so these always run.
* **Construction tests** build real framework objects and assert on their wiring. They
  need the framework installed and *skip* with a stated reason when it is not, rather
  than failing collection.
* **Live tests** call real models. Marked `live` and deselected by default:
  `pytest -m live`.

Generate tests from a config you already have, with no LLM call at all:

```bash
multi-agent-generator --generate-tests --config config.json --framework crewai
```

### Verifying the generator itself

The claim above — that generated bundles run on a clean machine — is checked by this
project's own suite, which needs nothing but `pytest`:

```bash
pip install -e ".[test]"
pytest
```

It compiles every framework × provider combination, then generates a bundle and executes
it in a child `pytest` process with credentials stripped from the environment. That last
test is the one that matters: a green run means a handed-over bundle really does pass with
no API key, no network, and no agent framework installed.

---

## Prerequisites

One supported LLM provider, and its credentials in the environment:

| Provider | Environment variables |
|----------|----------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Hugging Face (hosted) | `HF_TOKEN` (or `HUGGINGFACEHUB_API_TOKEN`) |
| Hugging Face (local) | none — nothing leaves your machine |
| Ollama | `OLLAMA_URL` (defaults to `http://localhost:11434`) |
| IBM WatsonX | `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` |

Run `multi-agent-generator --list-providers` to see which of these are already
configured on your machine, along with each provider's default model.

Switch providers with `--provider`, and override the model with `--model`.

> Agno is reached through its OpenAI-compatible model class, so non-OpenAI providers are
> wired up by pointing it at that provider's OpenAI-compatible endpoint. For Hugging Face
> that is the HF router; the generated code sets `base_url` accordingly.

---

## Usage

### Command Line

Basic usage with OpenAI (default):

```bash
multi-agent-generator "I need a research assistant that summarizes papers and answers questions" --framework crewai
```

Using WatsonX instead:

```bash
multi-agent-generator "I need a research assistant that summarizes papers and answers questions" --framework crewai --provider watsonx
```

Using Agno:

```bash
multi_agent_generator "build a researcher and writer" --framework agno --provider openai --output agno.py --format code
```

Using Ollama locally:

```bash
multi-agent-generator "Build me a ReAct assistant for customer support" --framework react-lcel --provider ollama
```

Save output to a file:

```bash
multi-agent-generator "I need a team to create viral social media content" --framework langgraph --output social_team.py
```

Get JSON configuration only:

```bash
multi-agent-generator "I need a team to analyze customer data" --framework react --format json
```

### Tool Generation via CLI

Generate custom tools from natural language:

```bash
# Generate a custom tool
multi-agent-generator --tool "Create a tool to fetch weather data from an API"

# Save to file
multi-agent-generator --tool "Create a web scraper tool" --output scraper_tool.py

# List all available tools
multi-agent-generator --list-tools

# List tools by category
multi-agent-generator --list-tools --tool-category api_integration
```

### Evaluation via CLI

Evaluate agent outputs directly from the command line:

```bash
# Basic evaluation
multi-agent-generator --evaluate --query "What is AI?" --response "AI is artificial intelligence..."

# With expected output for accuracy scoring
multi-agent-generator --evaluate \
  --query "Summarize machine learning" \
  --response "ML is a subset of AI that learns from data" \
  --expected "Machine learning is an AI technique" \
  --threshold 0.8

# Save results to file
multi-agent-generator --evaluate --query "Test" --response "Response" --output results.json
```

### Orchestration via CLI

Create orchestrated multi-agent systems:

```bash
# Get pattern suggestion from description
multi-agent-generator --orchestrate "I need agents to debate and reach consensus"

# Generate code for a specific pattern
multi-agent-generator --pattern supervisor --framework langgraph --output supervisor.py

# List all available patterns
multi-agent-generator --list-patterns

# Customize number of agents
multi-agent-generator --pattern voting --num-agents 5 --framework crewai
```

### Streamlit UI

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

Navigate between pages:
- **Agent Generator** - Generate agent code from natural language
- **Tool Discovery** - Browse and create tools
- **Orchestration Patterns** - Configure multi-agent coordination
- **Evaluation & Testing** - Generate tests and evaluate outputs

---

## Examples

### Research Assistant

```
I need a research assistant that summarizes papers and answers questions
```

### Content Creation Team

```
I need a team to create viral social media content and manage our brand presence
```

### Customer Support (LangGraph)

```
Build me a LangGraph workflow for customer support
```

### Orchestrated Team (NEW!)

```python
from multi_agent_generator.orchestration import Orchestrator

orchestrator = Orchestrator()
result = orchestrator.generate_from_description(
    "Build a content team with a supervisor managing writers and editors"
)
```

---

## Frameworks

### CrewAI

Role-playing autonomous AI agents with goals, roles, and backstories.

### CrewAI Flow

Event-driven workflows with sequential, parallel, or conditional execution.

### LangGraph

Directed graph of agents/tools with stateful execution.

### Agno

Role-playing Team orchestration AI agents with goals, roles, backstories and instructions.

### ReAct (classic)

Reasoning + Acting agents built with `AgentExecutor`.

### ReAct (LCEL)

Modern ReAct implementation using LangChain Expression Language — better for debugging and future-proof orchestration.

---

## LLM Providers

### OpenAI

State-of-the-art GPT models (default: `gpt-4o-mini`).

### IBM WatsonX

Enterprise-grade access to Llama and other foundation models (default: `llama-3-70b-instruct`).

### Ollama

Run Llama and other models locally.

### Anthropic

Use Claude models for agent generation.

### Hugging Face

Open-source models two ways: hosted inference through Hugging Face's API
(`--provider huggingface`, default `Qwen/Qwen2.5-7B-Instruct`), or fully local execution
with `transformers` (`--provider huggingface-local`, default `Qwen/Qwen2.5-0.5B-Instruct`).
Both back the generator's own analysis *and* the code it writes.

...and more, via LiteLLM.

---

## API Reference

### Tools Module

```python
from multi_agent_generator.tools import (
    ToolRegistry,      # Browse pre-built tools
    ToolGenerator,     # Generate custom tools
    ToolCategory,      # Tool category enum
    ToolDefinition,    # Tool data class
)
```

### Orchestration Module

```python
from multi_agent_generator.orchestration import (
    Orchestrator,      # High-level orchestration interface
    PatternType,       # Pattern type enum
    SupervisorPattern, # Supervisor pattern
    DebatePattern,     # Debate pattern
    VotingPattern,     # Voting pattern
    PipelinePattern,   # Pipeline pattern
    MapReducePattern,  # MapReduce pattern
)
```

### Evaluation Module

```python
from multi_agent_generator.evaluation import (
    TestGenerator,     # Auto-generate runnable test bundles
    TestCase,          # Individual test case
    TestSuite,         # Collection of tests
    TestBundle,        # Test module + conftest + pytest.ini + requirements
    TestType,          # Test type enum (CONTRACT, UNIT, INTEGRATION, ...)
    AgentEvaluator,    # Evaluate agent outputs
    EvaluationResult,  # Evaluation results
    Benchmark,         # Performance benchmarking
)
```

### Providers and Dependencies

```python
from multi_agent_generator.providers import PROVIDERS, get_provider, llm_snippet
from multi_agent_generator.dependencies import check_dependencies, requirements_txt

get_provider("hf").default_model            # aliases resolve: hf -> huggingface
check_dependencies("crewai", "huggingface") # what is missing, and how to install it
print(requirements_txt("crewai", "huggingface"))
```

---

## License

MIT

Maintainers: **[Nabarko Roy](https://github.com/Nabarko)**

Made with love. If you like star the repo and share it with AI Enthusiasts.
