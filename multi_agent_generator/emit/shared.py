# multi_agent_generator/emit/shared.py
"""
The parts of a generated project that do not depend on which framework was chosen.

Settings, credential handling, tool-call telemetry, the entry point, the runtime wrapper, the
memory store, the test suite, the README, ``requirements.txt``, ``.env.example`` and
``pytest.ini`` are written once here. Only the three genuinely framework-shaped questions -
how a tool is declared, how an agent is built, how agents are wired into a workflow - live in
the framework emitters.

Two rules govern everything in this module.

Nothing invents information. The README's file table, the architecture summary, the agent
list and the environment variable list are all rendered from the manifest, so a README cannot
claim four agents when three were planned.

No credential is ever written into a generated file. The manifest records environment variable
*names* and descriptions; the settings module reads them at run time and, when it has to
describe one, prints a masked form. A generated project is therefore safe to commit, to
export and to paste into a bug report.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from ..core.manifest import FileKind, ProjectManifest
from .base import Fragment, docstring, indent, join_blocks, literal

__all__ = [
    "settings_module",
    "telemetry_module",
    "state_module",
    "runner_module",
    "memory_module",
    "entrypoint_module",
    "package_init",
    "tool_registry_init",
    "agents_init",
    "requirements_txt",
    "env_example",
    "pytest_ini",
    "readme",
    "conftest_module",
    "test_imports_module",
    "test_config_module",
    "test_agents_module",
    "test_tools_module",
    "test_workflow_module",
    "test_simple_module",
    "dockerfile",
    "dockerignore",
]


#: Providers that load model weights in-process when a client is constructed. Building one of
#: these inside a unit test would either download gigabytes or take minutes, so the tests that
#: construct agents skip themselves and say why.
_LOCAL_MODEL_PROVIDERS = frozenset({"huggingface-local"})

#: The fake credential the generated tests set in order to prove ``describe()`` hides secrets.
#:
#: It has to read as a stand-in to a machine as well as to a person. The previous value spelled
#: its intent in English only ("do-not-print-me"): it matched the OpenAI key shape
#: ``core.validation`` scans for and carried none of the markers that excuse a placeholder - so
#: every generated project was reported as leaking a credential in its own test suite, a
#: BLOCKER under section 34 on all six frameworks. The scanner was right: shipping a
#: realistic-looking key inside generated source is the thing section 34 forbids, and the test
#: does not need a realistic one. ``not-real`` keeps the key *shape* the masking test wants
#: while saying plainly what it is.
_SECRET_SENTINEL = "sk-not-real-sentinel-0123456789"


def _module(manifest: ProjectManifest, path: str) -> str:
    """Dotted module name for a planned file, or an empty string if it was not planned."""
    return manifest.module_for(path) or ""


def _first_module(manifest: ProjectManifest, kind: str) -> str:
    files = manifest.files_of_kind(kind)
    return _module(manifest, files[0].path) if files else ""


def _workflow_module(manifest: ProjectManifest) -> str:
    """Where ``run_workflow`` lives, which differs by tier."""
    workflow = manifest.primary_workflow
    if workflow is not None and workflow.module:
        return workflow.module
    return _first_module(manifest, FileKind.WORKFLOW) or _first_module(manifest, FileKind.AGENT)


def _settings_module(manifest: ProjectManifest) -> str:
    return _first_module(manifest, FileKind.CONFIG)


def _llm_module(manifest: ProjectManifest) -> str:
    """
    Where ``get_llm`` lives.

    In the flat layout there is no service package, so the model factory is emitted into
    ``agent.py`` alongside the agents that use it.
    """
    for spec in manifest.files_of_kind(FileKind.SERVICE):
        if "llm" in spec.path:
            return _module(manifest, spec.path)
    return _first_module(manifest, FileKind.AGENT)


def _telemetry_module(manifest: ProjectManifest) -> str:
    """
    Where the tool-call recorders live.

    A dedicated service module in the package layout. In the flat layout they sit in
    ``tools.py`` next to the calls they record - or, when the project has no tools at all, in
    the module that holds the workflow, which still clears and reads the record.
    """
    for spec in manifest.files_of_kind(FileKind.SERVICE):
        if "telemetry" in spec.path:
            return _module(manifest, spec.path)
    return _first_module(manifest, FileKind.TOOL) or _workflow_module(manifest)


def _tools_registry_module(manifest: ProjectManifest) -> str:
    """Where the ``TOOLS`` dict lives: the tools *package* init, not an individual tool file."""
    first = _first_module(manifest, FileKind.TOOL)
    if not first:
        return _workflow_module(manifest)
    parts = first.split(".")
    # Package layout: app.tools.web_search -> app.tools (TOOLS lives in __init__.py).
    # Flat layout: tools -> tools (TOOLS lives in the same single-file module).
    return ".".join(parts[:-1]) if len(parts) >= 3 else first


# ============================================================================== settings
def settings_module(manifest: ProjectManifest) -> Fragment:
    """
    The settings module: every value the project reads from its environment, in one place.

    Credentials are read here and nowhere else, and :meth:`Settings.describe` masks them, so
    printing the configuration - which is the obvious thing to do when something is not
    working - cannot leak a key into a terminal, a log file or a screenshot.
    """
    required = [v for v in manifest.env_vars if v.required]
    optional = [v for v in manifest.env_vars if not v.required]

    fields: List[str] = []
    loads: List[str] = []
    for var in manifest.env_vars:
        attribute = var.name.lower()
        fields.append(f"    #: {var.purpose}\n    {attribute}: str = \"\"")
        loads.append(f"        {attribute}=os.getenv({literal(var.name)}, \"\"),")

    secret_names = [v.name for v in manifest.env_vars if "KEY" in v.name or "TOKEN" in v.name]

    body = (
        f"#: The environment variables this project reads. Names only - never values.\n"
        f"REQUIRED_ENV = {literal([v.name for v in required])}\n"
        f"OPTIONAL_ENV = {literal([v.name for v in optional])}\n"
        f"#: Variables whose values must never be printed in full.\n"
        f"SECRET_ENV = {literal(secret_names)}\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Settings:\n"
        + indent(
            docstring(
                "Everything this project needs from its environment.",
                [
                    "Frozen, so a value cannot be changed after it has been validated.",
                ],
            )
        )
        + "\n\n"
        + ("\n\n".join(fields) if fields else "    pass")
        + "\n\n"
        f"    #: The model id the agents use. Overridable with AGENT_MODEL.\n"
        f"    model: str = {literal(manifest.model or '')}\n\n"
        "    def missing(self) -> List[str]:\n"
        + indent(
            docstring(
                "Names of required variables that are not set.",
                [
                    "Returned as a list rather than raised, so the caller can decide whether a "
                    "missing credential is fatal. Importing a module must not fail because an "
                    "environment variable is absent - that would make the whole project "
                    "untestable without a real key.",
                ],
            ),
            "        ",
        )
        + "\n"
        "        return [\n"
        "            name\n"
        "            for name in REQUIRED_ENV\n"
        "            if not str(getattr(self, name.lower(), \"\") or \"\").strip()\n"
        "        ]\n\n"
        "    def require(self) -> None:\n"
        + indent(
            docstring(
                "Raise if a required variable is missing, naming it and what to do.",
                [
                    "Called at the start of a run, not at import time.",
                ],
            ),
            "        ",
        )
        + "\n"
        "        absent = self.missing()\n"
        "        if absent:\n"
        "            raise RuntimeError(\n"
        "                \"Missing required environment variable(s): \"\n"
        "                + \", \".join(absent)\n"
        "                + \". Copy .env.example to .env and fill them in.\"\n"
        "            )\n\n"
        "    def describe(self) -> Dict[str, str]:\n"
        + indent(
            docstring(
                "A printable summary of the configuration, with secrets masked.",
                [
                    "Secrets are reported as 'set' or 'not set' and never as a value, not even "
                    "a truncated one: the first characters of a key are enough to identify an "
                    "account, so a partial key is still a leaked key.",
                ],
            ),
            "        ",
        )
        + "\n"
        "        summary: Dict[str, str] = {\"model\": self.model}\n"
        "        for name in [*REQUIRED_ENV, *OPTIONAL_ENV]:\n"
        "            value = str(getattr(self, name.lower(), \"\") or \"\")\n"
        "            if name in SECRET_ENV:\n"
        "                summary[name] = \"set\" if value else \"not set\"\n"
        "            else:\n"
        "                summary[name] = value or \"not set\"\n"
        "        return summary\n\n\n"
        "@lru_cache(maxsize=1)\n"
        "def get_settings() -> Settings:\n"
        + indent(
            docstring(
                "Load the settings once and reuse them.",
                [
                    "`load_dotenv` is called first so a local `.env` works without exporting "
                    "anything by hand. It never overrides a variable that is already set, so a "
                    "real deployment's environment always wins over a checked-in file.",
                ],
            )
        )
        + "\n"
        "    load_dotenv(override=False)\n"
        "    return Settings(\n"
        + ("\n".join(loads) + "\n" if loads else "")
        + "        model=os.getenv(\"AGENT_MODEL\", \"\") or "
        + literal(manifest.model or "")
        + ",\n"
        "    )\n"
    )

    return Fragment(
        body=body,
        imports=[
            "import os",
            "from dataclasses import dataclass",
            "from functools import lru_cache",
            "from typing import Dict, List",
            "from dotenv import load_dotenv",
        ],
        provides=["Settings", "get_settings", "REQUIRED_ENV", "OPTIONAL_ENV", "SECRET_ENV"],
    )


# ============================================================================= telemetry
def telemetry_module(manifest: ProjectManifest) -> Fragment:
    """
    Records what the tools actually did during a run.

    This is what lets the platform show a real list of tool calls instead of a plausible one.
    A run that used no tools shows no tool calls, which is information rather than an
    omission.
    """
    body = (
        "#: Tool calls recorded during the current run, in the order they happened.\n"
        "_CALLS: List[Dict[str, Any]] = []\n\n\n"
        "def reset_calls() -> None:\n"
        + indent(
            docstring(
                "Clear the record. Called at the start of every run.",
                [
                    "Without this, a second run in the same process would report the first "
                    "run's tool calls as its own.",
                ],
            )
        )
        + "\n"
        "    _CALLS.clear()\n\n\n"
        "def record_call(name: str, arguments: Any, result: Any) -> None:\n"
        + indent(
            docstring(
                "Record one tool call.",
                [
                    "Arguments and results are stored as text and truncated, because a tool "
                    "that returns a large document would otherwise make the run record too "
                    "big to send back to the browser.",
                ],
            )
        )
        + "\n"
        "    _CALLS.append(\n"
        "        {\n"
        "            \"tool\": str(name),\n"
        "            \"arguments\": _shorten(arguments),\n"
        "            \"result\": _shorten(result),\n"
        "            \"at\": time.time(),\n"
        "        }\n"
        "    )\n\n\n"
        "def recorded_calls() -> List[Dict[str, Any]]:\n"
        + indent(docstring("Return a copy of the calls recorded so far."))
        + "\n"
        "    return list(_CALLS)\n\n\n"
        "def _shorten(value: Any, limit: int = 2000) -> str:\n"
        + indent(docstring("Render a value as text, truncated with the amount dropped stated."))
        + "\n"
        "    text = str(value)\n"
        "    if len(text) <= limit:\n"
        "        return text\n"
        "    return text[:limit] + f\"... [{len(text) - limit} more characters]\"\n"
    )
    return Fragment(
        body=body,
        imports=["import time", "from typing import Any, Dict, List"],
        provides=["record_call", "recorded_calls", "reset_calls"],
    )


# ================================================================================= state
def state_module(manifest: ProjectManifest, framework_fragment: Fragment) -> Fragment:
    """
    The state module.

    LangGraph supplies its own ``AgentState`` through the emitter, because there the state
    type is the framework's contract. Every other framework gets a plain dataclass that
    records the same thing without pretending to be a graph state.

    ``RunOutput`` is added in both cases: it is the shape ``run_workflow`` returns, written
    down as a type so the entry point and the tests agree about it.
    """
    run_output = (
        "@dataclass\n"
        "class RunOutput:\n"
        + indent(
            docstring(
                "The result of one run.",
                [
                    "`steps` and `tool_calls` are what actually happened, collected while the "
                    "workflow ran. An empty list means nothing of that kind happened, not that "
                    "the information is unavailable.",
                ],
            )
        )
        + "\n"
        "    output: str = \"\"\n"
        "    steps: List[Dict[str, Any]] = field(default_factory=list)\n"
        "    tool_calls: List[Dict[str, Any]] = field(default_factory=list)\n"
        "    duration_s: float = 0.0\n"
        "    error: str = \"\"\n\n"
        "    @property\n"
        "    def ok(self) -> bool:\n"
        "        \"\"\"True when the run finished without an error.\"\"\"\n"
        "        return not self.error\n\n"
        "    def as_dict(self) -> Dict[str, Any]:\n"
        "        \"\"\"A JSON-serialisable form, for printing or returning over an API.\"\"\"\n"
        "        return {\n"
        "            \"output\": self.output,\n"
        "            \"steps\": self.steps,\n"
        "            \"tool_calls\": self.tool_calls,\n"
        "            \"duration_s\": self.duration_s,\n"
        "            \"error\": self.error,\n"
        "            \"ok\": self.ok,\n"
        "        }\n"
    )

    if framework_fragment.body.strip():
        return Fragment(
            body=join_blocks([framework_fragment.body, run_output]),
            imports=list(dict.fromkeys([
                *framework_fragment.imports,
                "from dataclasses import dataclass, field",
                "from typing import Any, Dict, List",
            ])),
            provides=[*framework_fragment.provides, "RunOutput"],
        )

    workflow = manifest.primary_workflow
    extra = ""
    for name in (workflow.state_fields if workflow else []):
        if name in ("messages", "results", "next"):
            continue
        extra += f"    {name}: str = \"\"\n"

    agent_state = (
        "@dataclass\n"
        "class AgentState:\n"
        + indent(
            docstring(
                "What is carried from one step to the next.",
                [
                    "`results` holds each completed step's output keyed by step name, which is "
                    "what the run's breakdown is built from.",
                ],
            )
        )
        + "\n"
        "    query: str = \"\"\n"
        "    results: Dict[str, str] = field(default_factory=dict)\n"
        "    next: str = \"\"\n"
        + extra
    )
    return Fragment(
        body=join_blocks([agent_state, run_output]),
        imports=["from dataclasses import dataclass, field", "from typing import Any, Dict, List"],
        provides=["AgentState", "RunOutput"],
    )


# =============================================================================== runtime
def runner_module(manifest: ProjectManifest) -> Fragment:
    """
    The runtime wrapper for the modular tier.

    Turns an exception into a described failure rather than a traceback that reaches the user.
    The traceback is still captured - it goes into ``RunOutput.error`` and into the log - so
    nothing is hidden; it is just not the first thing a non-technical person sees.
    """
    workflow_module = _workflow_module(manifest)
    state_files = manifest.files_of_kind(FileKind.STATE)
    state_module_name = _module(manifest, state_files[0].path) if state_files else ""
    body = (
        "LOGGER = logging.getLogger(__name__)\n\n\n"
        "def run(query: str) -> RunOutput:\n"
        + indent(
            docstring(
                "Run the workflow for one query and always return a RunOutput.",
                [
                    "A failure is reported as a populated `error` rather than as a raised "
                    "exception, so the caller can show what went wrong alongside whatever "
                    "partial detail was collected.",
                    "The full traceback is logged at exception level. Nothing is swallowed: "
                    "the error text names the exception type and message, and the log holds "
                    "the stack for whoever needs it.",
                ],
            )
        )
        + "\n"
        "    text = str(query or \"\").strip()\n"
        "    if not text:\n"
        "        # An empty query is a user mistake, not a system failure. Answering it with a\n"
        "        # clear message costs nothing and avoids a pointless model call.\n"
        "        return RunOutput(error=\"No query was provided. Enter a question or task.\")\n"
        "\n"
        "    settings = get_settings()\n"
        "    absent = settings.missing()\n"
        "    if absent:\n"
        "        # Checked before the model is touched, so a missing key is reported as a\n"
        "        # configuration problem instead of surfacing later as an auth error.\n"
        "        return RunOutput(\n"
        "            error=(\n"
        "                \"Missing configuration: \"\n"
        "                + \", \".join(absent)\n"
        "                + \". Copy .env.example to .env and fill it in.\"\n"
        "            )\n"
        "        )\n"
        "\n"
        "    started = time.monotonic()\n"
        "    try:\n"
        "        result = run_workflow(text)\n"
        "    except Exception as exc:  # noqa: BLE001 - reported, not hidden\n"
        "        LOGGER.exception(\"Workflow failed\")\n"
        "        return RunOutput(\n"
        "            duration_s=round(time.monotonic() - started, 3),\n"
        "            error=f\"{type(exc).__name__}: {exc}\",\n"
        "        )\n"
        "\n"
        "    if not isinstance(result, dict):\n"
        "        # The contract is a dict with an 'output' key. Anything else is a bug in the\n"
        "        # workflow module, and saying so is more useful than str()-ing it silently.\n"
        "        return RunOutput(\n"
        "            output=str(result),\n"
        "            duration_s=round(time.monotonic() - started, 3),\n"
        "            error=\"run_workflow did not return a dict; showing its value as text.\",\n"
        "        )\n"
        "\n"
        "    return RunOutput(\n"
        "        output=str(result.get(\"output\") or \"\"),\n"
        "        steps=list(result.get(\"steps\") or []),\n"
        "        tool_calls=list(result.get(\"tool_calls\") or []),\n"
        "        duration_s=float(result.get(\"duration_s\") or round(time.monotonic() - started, 3)),\n"
        "    )\n"
    )
    return Fragment(
        body=body,
        imports=[
            "import logging",
            "import time",
            f"from {_settings_module(manifest)} import get_settings",
            f"from {workflow_module} import run_workflow",
        ]
        + ([f"from {state_module_name} import RunOutput"] if state_module_name else []),
        provides=["run"],
    )


def memory_module(manifest: ProjectManifest) -> Fragment:
    """A small JSON-backed store, for projects whose plan asked to carry state between runs."""
    body = (
        "LOGGER = logging.getLogger(__name__)\n\n\n"
        "class MemoryStore:\n"
        + indent(
            docstring(
                "Keeps results between runs in a JSON file.",
                [
                    "A file rather than a database, because that is proportional to what the "
                    "plan asked for and needs nothing installed. Swap the two `_load`/`_save` "
                    "methods for a real store when this project outgrows it.",
                    "The path stays inside the project directory. A corrupt or unreadable file "
                    "is reported and then treated as empty, so a bad memory file cannot stop "
                    "the agent from running.",
                ],
            )
        )
        + "\n\n"
        "    def __init__(self, path: Optional[Path] = None) -> None:\n"
        "        self.path = Path(path) if path else Path(__file__).resolve().parent.parent.parent / \".memory.json\"\n"
        "        self._data: Dict[str, Any] = self._load()\n\n"
        "    def _load(self) -> Dict[str, Any]:\n"
        "        if not self.path.exists():\n"
        "            return {}\n"
        "        try:\n"
        "            return json.loads(self.path.read_text(encoding=\"utf-8\")) or {}\n"
        "        except (OSError, ValueError) as exc:\n"
        "            LOGGER.warning(\"Could not read memory file %s: %s\", self.path, exc)\n"
        "            return {}\n\n"
        "    def _save(self) -> None:\n"
        "        try:\n"
        "            self.path.write_text(json.dumps(self._data, indent=2), encoding=\"utf-8\")\n"
        "        except OSError as exc:\n"
        "            LOGGER.warning(\"Could not write memory file %s: %s\", self.path, exc)\n\n"
        "    def get(self, key: str, default: Any = None) -> Any:\n"
        "        \"\"\"Read one remembered value.\"\"\"\n"
        "        return self._data.get(key, default)\n\n"
        "    def set(self, key: str, value: Any) -> None:\n"
        "        \"\"\"Remember one value and persist it immediately.\"\"\"\n"
        "        self._data[key] = value\n"
        "        self._save()\n\n"
        "    def append(self, key: str, value: Any) -> None:\n"
        "        \"\"\"Add to a remembered list, creating it if needed.\"\"\"\n"
        "        items = list(self._data.get(key) or [])\n"
        "        items.append(value)\n"
        "        self.set(key, items)\n\n"
        "    def clear(self) -> None:\n"
        "        \"\"\"Forget everything.\"\"\"\n"
        "        self._data = {}\n"
        "        self._save()\n"
    )
    return Fragment(
        body=body,
        imports=[
            "import json",
            "import logging",
            "from pathlib import Path",
            "from typing import Any, Dict, Optional",
        ],
        provides=["MemoryStore"],
    )


# ============================================================================ entry point
def entrypoint_module(manifest: ProjectManifest) -> Fragment:
    """
    ``main.py``: the one command a person runs.

    Takes the query from the command line or from stdin, prints the answer, and returns a
    non-zero exit status when the run failed - so a script wrapping this can tell.
    """
    runner_files = manifest.files_of_kind(FileKind.RUNTIME)
    uses_runner = bool(runner_files)
    imports = ["import argparse", "import json", "import logging", "import sys"]

    if uses_runner:
        runner = _module(manifest, runner_files[0].path)
        imports.append(f"from {runner} import run")
        call = "    result = run(query).as_dict()\n"
    else:
        imports.append(f"from {_workflow_module(manifest)} import run_workflow")
        imports.append(f"from {_settings_module(manifest)} import get_settings")
        call = (
            "    absent = get_settings().missing()\n"
            "    if absent:\n"
            "        # Reported as configuration, before any model call, so the message names\n"
            "        # the variable to set instead of surfacing as an authentication error.\n"
            "        return _fail(\n"
            "            \"Missing configuration: \"\n"
            "            + \", \".join(absent)\n"
            "            + \". Copy .env.example to .env and fill it in.\",\n"
            "            as_json,\n"
            "        )\n"
            "\n"
            "    try:\n"
            "        raw = run_workflow(query)\n"
            "    except Exception as exc:  # noqa: BLE001 - reported, not hidden\n"
            "        LOGGER.exception(\"Workflow failed\")\n"
            "        return _fail(f\"{type(exc).__name__}: {exc}\", as_json)\n"
            "\n"
            "    result = dict(raw) if isinstance(raw, dict) else {\"output\": str(raw)}\n"
            "    result.setdefault(\"error\", \"\")\n"
            "    result.setdefault(\"ok\", True)\n"
        )

    body = (
        "LOGGER = logging.getLogger(__name__)\n\n\n"
        "def _fail(message: str, as_json: bool) -> int:\n"
        + indent(
            docstring(
                "Print a failure the way the caller asked for it and return exit code 1.",
                [
                    "The message says what went wrong in plain language. The traceback, when "
                    "there was one, is in the log rather than on stdout.",
                ],
            )
        )
        + "\n"
        "    if as_json:\n"
        "        print(json.dumps({\"output\": \"\", \"error\": message, \"ok\": False}, indent=2))\n"
        "    else:\n"
        "        print(f\"Failed: {message}\", file=sys.stderr)\n"
        "    return 1\n\n\n"
        "def main(argv: list = None) -> int:\n"
        + indent(
            docstring(
                f"Run {manifest.project_name} once and print the answer.",
                [
                    "Returns a process exit status: 0 when the run succeeded, 1 when it did "
                    "not, so this works inside a shell script as well as by hand.",
                ],
            )
        )
        + "\n"
        "    parser = argparse.ArgumentParser(\n"
        f"        description={literal(manifest.description or manifest.project_name)},\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"query\",\n"
        "        nargs=\"*\",\n"
        "        help=\"What to ask the agent. Read from stdin when omitted.\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--json\",\n"
        "        action=\"store_true\",\n"
        "        help=\"Print the full result as JSON, including steps and tool calls.\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--verbose\",\n"
        "        action=\"store_true\",\n"
        "        help=\"Show debug logging, including framework output.\",\n"
        "    )\n"
        "    args = parser.parse_args(argv)\n"
        "\n"
        "    logging.basicConfig(\n"
        "        level=logging.DEBUG if args.verbose else logging.INFO,\n"
        "        format=\"%(asctime)s %(levelname)s %(name)s %(message)s\",\n"
        "    )\n"
        "\n"
        "    as_json = bool(args.json)\n"
        "    query = \" \".join(args.query).strip()\n"
        "    if not query and not sys.stdin.isatty():\n"
        "        # Allows `echo \"question\" | python main.py`, which is how this gets used\n"
        "        # from a script or another process.\n"
        "        query = sys.stdin.read().strip()\n"
        "    if not query:\n"
        "        return _fail(\"No query given. Pass one as an argument or on stdin.\", as_json)\n"
        "\n"
        + call
        + "\n"
        "    if as_json:\n"
        "        print(json.dumps(result, indent=2, default=str))\n"
        "    elif result.get(\"error\"):\n"
        "        return _fail(str(result[\"error\"]), as_json)\n"
        "    else:\n"
        "        print(result.get(\"output\") or \"\")\n"
        "\n"
        "    return 0 if not result.get(\"error\") else 1\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    raise SystemExit(main())\n"
    )
    return Fragment(body=body, imports=imports, provides=["main"])


# ============================================================================== packages
def package_init(exports: Sequence[str] = ()) -> Fragment:
    """
    A package marker.

    Carries an ``__all__`` when the package re-exports anything, and otherwise holds only the
    docstring the assembler prepends. It is not empty: a file with no explanation is a file
    the next reader has to guess about.
    """
    body = f"__all__ = {literal(list(exports))}\n" if exports else ""
    return Fragment(body=body, provides=list(exports))


def tool_registry_init(
    manifest: ProjectManifest,
    registry: Fragment,
    symbols: Dict[str, List[str]],
) -> Fragment:
    """
    ``tools/__init__.py``: imports every tool module and exposes the registry.

    ``symbols`` maps a tool name to the names its own module defined - a class for the
    LangChain-family frameworks, a bare function for Agno. It is passed in rather than
    guessed, because the emitter is the only thing that knows which shape it produced, and
    guessing here would emit an import of a name that does not exist.
    """
    package = manifest.package or ""
    imports: List[str] = []
    for tool in manifest.tools:
        module = tool.module or (f"{package}.tools.{tool.name}" if package else "tools")
        provided = symbols.get(tool.name) or [tool.class_name]
        imports.append(f"from {module} import {', '.join(provided)}")
    return Fragment(body=registry.body, imports=imports, provides=registry.provides)


def agents_init(manifest: ProjectManifest) -> Fragment:
    """``agents/__init__.py``: one import per agent builder, so the package is the public API."""
    imports = [
        f"from {agent.module} import {agent.factory_name}"
        for agent in manifest.agents
        if agent.module
    ]
    exports = [agent.factory_name for agent in manifest.agents]
    return Fragment(
        body=f"__all__ = {literal(exports)}\n" if exports else "",
        imports=imports,
        provides=exports,
    )


# ================================================================================== meta
def requirements_txt(manifest: ProjectManifest) -> str:
    """``requirements.txt``, from the dependency registry rather than from guesswork."""
    lines = [
        f"# Dependencies for {manifest.project_name}",
        f"# Framework: {manifest.framework}. Provider: {manifest.provider}.",
        "# Install with: pip install -r requirements.txt",
        "",
    ]
    lines.extend(manifest.dependencies)
    return "\n".join(lines) + "\n"


def env_example(manifest: ProjectManifest) -> str:
    """
    ``.env.example``: every variable the project reads, with no value filled in.

    Values are deliberately blank. A placeholder that looks like a key gets committed by
    accident and then gets mistaken for a real one during debugging.
    """
    lines = [
        f"# Environment for {manifest.project_name}",
        "# Copy this file to .env and fill in the values. Never commit .env.",
        "",
    ]
    for var in manifest.env_vars:
        lines.append(f"# {var.purpose}")
        lines.append(f"# Required: {'yes' if var.required else 'no'}")
        if var.example and "KEY" not in var.name and "TOKEN" not in var.name:
            lines.append(f"{var.name}={var.example}")
        else:
            lines.append(f"{var.name}=")
        lines.append("")
    return "\n".join(lines)


def pytest_ini(manifest: ProjectManifest) -> str:
    """
    ``pytest.ini``.

    The timeout is only written when ``pytest-timeout`` is actually a dependency, because a
    setting for a plugin that is not installed is silently ignored - and a silently ignored
    safety net is worse than none, since it looks like protection.

    The timeout exists to turn a hang into a named failure. It is not there to make a slow
    test pass: a test that needs longer than this is doing something a unit test should not.
    """
    has_timeout = any("pytest-timeout" in dep for dep in manifest.dependencies)
    lines = [
        "[pytest]",
        "testpaths = tests",
        "addopts = -q --strict-markers",
        "markers =",
        "    live: needs a real provider credential and network access; deselected by default",
        "    slow: takes more than a few seconds",
    ]
    if has_timeout:
        lines.append("timeout = 10")
    return "\n".join(lines) + "\n"


def dockerfile(manifest: ProjectManifest) -> str:
    """A Dockerfile that runs the project's real entry point."""
    entry = manifest.entrypoint or "main.py"
    return "\n".join(
        [
            "# syntax=docker/dockerfile:1",
            f"# Container image for {manifest.project_name}.",
            "FROM python:3.11-slim",
            "",
            "# Unbuffered output so logs appear immediately in `docker logs` rather than",
            "# arriving in blocks after the process exits.",
            "ENV PYTHONUNBUFFERED=1 \\",
            "    PYTHONDONTWRITEBYTECODE=1",
            "",
            "WORKDIR /app",
            "",
            "# Requirements are copied on their own first so the install layer is cached and",
            "# a code change does not reinstall every package.",
            "COPY requirements.txt ./",
            "RUN pip install --no-cache-dir -r requirements.txt",
            "",
            "COPY . .",
            "",
            "# A non-root user: the agent has no reason to be able to write to the image.",
            "RUN useradd --create-home --uid 1000 agent && chown -R agent:agent /app",
            "USER agent",
            "",
            "# No credential is baked in. Pass them at run time:",
            "#   docker run --env-file .env <image> \"your question\"",
            f"ENTRYPOINT [\"python\", \"{entry}\"]",
            "",
        ]
    )


def dockerignore() -> str:
    """Keeps secrets and local clutter out of the build context."""
    return "\n".join(
        [
            "# Never send a real .env into the build context - it would end up in the image.",
            ".env",
            "*.env",
            "!.env.example",
            "",
            ".git",
            ".gitignore",
            "__pycache__/",
            "*.py[cod]",
            ".pytest_cache/",
            ".memory.json",
            ".venv/",
            "venv/",
            "",
        ]
    )


# ================================================================================== tests
def conftest_module(manifest: ProjectManifest) -> Fragment:
    """
    ``tests/conftest.py``.

    Three jobs, and the first is the important one.

    **Network is blocked.** Every test in this suite runs with outbound sockets disabled. A
    unit test that quietly calls a real provider costs money, needs a key that CI does not
    have, and fails for reasons unrelated to the code. Worse, it usually *hangs* rather than
    failing, and a hanging suite is a suite nobody runs. Blocking the socket turns any
    accidental call into an immediate, named error that points at the test that made it.
    Tests that genuinely need a live provider are marked ``live`` and get the socket back.

    **Placeholder credentials are set.** Clients are constructed, never called, so a
    syntactically valid fake key is enough. This is what lets agent construction be tested
    without a real account - and the fake is obviously fake, so it cannot be mistaken for one.

    **The project root is importable.** So ``pytest`` works from the project directory
    without an install step or a ``PYTHONPATH`` incantation.
    """
    credential_lines = "\n".join(
        f"    {literal(var.name)}: {literal('test-' + var.name.lower().replace('_', '-'))},"
        for var in manifest.env_vars
        if var.required
    )

    body = (
        "#: The project root, so `import app` / `import agent` works from anywhere.\n"
        "ROOT = Path(__file__).resolve().parent.parent\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n\n"
        "#: Obviously-fake credentials. Enough to construct a client, useless for a real call,\n"
        "#: and recognisable as a placeholder if one ever shows up in output.\n"
        "PLACEHOLDER_ENV = {\n"
        + (credential_lines + "\n" if credential_lines else "")
        + "}\n\n"
        "#: True when this project's provider loads model weights in-process. Constructing a\n"
        "#: client then means downloading or loading a model, which a unit test must not do.\n"
        f"PROVIDER_LOADS_MODEL_LOCALLY = {manifest.provider in _LOCAL_MODEL_PROVIDERS}\n\n"
        f"#: Named in skip messages so the reason is specific rather than generic.\n"
        f"PROVIDER = {literal(manifest.provider)}\n\n\n"
        "class NetworkBlocked(RuntimeError):\n"
        + indent(
            docstring(
                "Raised when a test tries to open a network connection.",
                [
                    "Its own class, so the traceback says what the problem is instead of "
                    "surfacing as a connection refused error that looks like flakiness.",
                ],
            )
        )
        + "\n\n\n"
        "@pytest.fixture(autouse=True)\n"
        "def _no_network(request, monkeypatch):\n"
        + indent(
            docstring(
                "Block outbound sockets for every test not marked `live`.",
                [
                    "`socket.socket.connect` is patched rather than the whole module, so "
                    "loopback-free local work still behaves and only an actual connection "
                    "attempt raises.",
                    "This exists to make an accidental provider call fail loudly and fast. It "
                    "is not a substitute for the `live` marker: a test that needs a real model "
                    "should ask for one explicitly.",
                ],
            ),
            "    ",
        )
        + "\n"
        "    if request.node.get_closest_marker(\"live\"):\n"
        "        return\n"
        "\n"
        "    def _blocked(self, *args, **kwargs):\n"
        "        raise NetworkBlocked(\n"
        "            \"This test tried to open a network connection. Unit tests must not call \"\n"
        "            \"a model provider. Mark the test with @pytest.mark.live if it genuinely \"\n"
        "            \"needs one.\"\n"
        "        )\n"
        "\n"
        "    monkeypatch.setattr(socket.socket, \"connect\", _blocked)\n"
        "    monkeypatch.setattr(socket.socket, \"connect_ex\", _blocked)\n\n\n"
        "@pytest.fixture(autouse=True)\n"
        "def _placeholder_credentials(monkeypatch):\n"
        + indent(
            docstring(
                "Set placeholder credentials so clients can be constructed.",
                [
                    "`setenv` rather than a default in the code: the project must still fail "
                    "with a clear message when a real deployment has no key, and putting a "
                    "fallback in the source would hide that.",
                ],
            ),
            "    ",
        )
        + "\n"
        "    for name, value in PLACEHOLDER_ENV.items():\n"
        "        monkeypatch.setenv(name, value)\n"
        "    get_settings.cache_clear()\n"
        "    yield\n"
        "    get_settings.cache_clear()\n\n\n"
        "@pytest.fixture\n"
        "def skip_if_local_model():\n"
        + indent(
            docstring(
                "Skip a test that would have to construct a locally-loaded model.",
                [
                    "A real reason to skip, stated in the message: with a local provider, "
                    "building a client loads model weights, which is minutes and gigabytes "
                    "rather than a unit test. The skip is narrow - only the tests that "
                    "construct agents use it - and every other test still runs.",
                ],
            ),
            "    ",
        )
        + "\n"
        "    if PROVIDER_LOADS_MODEL_LOCALLY:\n"
        "        pytest.skip(\n"
        "            f\"Provider {PROVIDER} loads model weights in-process; constructing a \"\n"
        "            \"client is not a unit-test operation. Run the live suite instead.\"\n"
        "        )\n"
    )
    return Fragment(
        body=body,
        imports=[
            "import socket",
            "import sys",
            "from pathlib import Path",
            "import pytest",
            f"from {_settings_module(manifest)} import get_settings",
        ],
        provides=["ROOT", "PLACEHOLDER_ENV", "NetworkBlocked", "skip_if_local_model"],
    )


def test_imports_module(manifest: ProjectManifest) -> Fragment:
    """
    ``tests/test_imports.py``: every module imports.

    The cheapest test in the suite and the one that catches the most. A missing import, a
    typo in a symbol name, a circular import between the agents and the workflow - all of
    them show up here, by module name, before anything tries to run.

    The module list comes from the manifest, so a file added to the plan is covered without
    anyone remembering to add it to a list.
    """
    modules = [
        m
        for m in (_module(manifest, spec.path) for spec in manifest.files)
        if m and not m.startswith("tests")
    ]
    modules = list(dict.fromkeys(modules))

    body = (
        "#: Every module this project defines, from the plan rather than from a hand-kept list.\n"
        f"MODULES = {literal(modules)}\n\n\n"
        "@pytest.mark.parametrize(\"module_name\", MODULES)\n"
        "def test_module_imports(module_name):\n"
        + indent(
            docstring(
                "Each module imports cleanly.",
                [
                    "Parametrised so a failure names the module that broke instead of stopping "
                    "at the first one and hiding the rest.",
                ],
            )
        )
        + "\n"
        "    importlib.import_module(module_name)\n\n\n"
        "def test_workflow_entry_point_exists():\n"
        + indent(
            docstring(
                "The workflow module exposes run_workflow.",
                [
                    "This is the contract the runtime calls. If it is missing or is not "
                    "callable, the project cannot be run, no matter how well the rest imports.",
                ],
            )
        )
        + "\n"
        f"    module = importlib.import_module({literal(_workflow_module(manifest))})\n"
        "    assert hasattr(module, \"run_workflow\"), (\n"
        "        \"The workflow module must define run_workflow(query).\"\n"
        "    )\n"
        "    assert callable(module.run_workflow)\n\n\n"
        "def test_entrypoint_is_runnable():\n"
        + indent(
            docstring(
                "The entry point defines main().",
                [
                    "Checked by import rather than by running it, so this test needs no model.",
                ],
            )
        )
        + "\n"
        f"    module = importlib.import_module({literal(_module(manifest, manifest.entrypoint) or 'main')})\n"
        "    assert callable(getattr(module, \"main\", None))\n"
    )
    return Fragment(body=body, imports=["import importlib", "import pytest"])


def test_config_module(manifest: ProjectManifest) -> Fragment:
    """
    ``tests/test_config.py``: configuration loads, reports what is missing, and hides secrets.

    The last of those is a real test rather than a comment: it asserts that no secret value
    appears in :meth:`Settings.describe`, so a future change that starts printing keys fails
    the suite instead of shipping.
    """
    settings_module_name = _settings_module(manifest)
    body = (
        "def test_settings_load():\n"
        + indent(docstring("Settings load from the environment without raising."))
        + "\n"
        "    settings = get_settings()\n"
        "    assert isinstance(settings.model, str)\n\n\n"
        "def test_required_variables_are_satisfied_by_the_fixture():\n"
        + indent(
            docstring(
                "With placeholder credentials set, nothing is reported missing.",
                [
                    "Confirms the placeholder fixture actually covers every required variable. "
                    "If a variable is added to the plan and not to the fixture, this fails - "
                    "which is better than the rest of the suite failing for a reason that "
                    "looks unrelated.",
                ],
            )
        )
        + "\n"
        "    assert get_settings().missing() == []\n\n\n"
        "def test_missing_credential_is_reported_not_raised(monkeypatch):\n"
        + indent(
            docstring(
                "An absent required variable is listed, and require() explains it.",
                [
                    "Importing must never fail because of configuration, so the absence is "
                    "reported by a method the caller chooses to call.",
                ],
            )
        )
        + "\n"
        "    for name in REQUIRED_ENV:\n"
        "        monkeypatch.setenv(name, \"\")\n"
        "    get_settings.cache_clear()\n"
        "    settings = get_settings()\n"
        "\n"
        "    if not REQUIRED_ENV:\n"
        "        pytest.skip(\"This project reads no required environment variables.\")\n"
        "\n"
        "    assert settings.missing() == list(REQUIRED_ENV)\n"
        "    with pytest.raises(RuntimeError) as info:\n"
        "        settings.require()\n"
        "    message = str(info.value)\n"
        "    for name in REQUIRED_ENV:\n"
        "        assert name in message, \"The error must name the variable that is missing.\"\n\n\n"
        "def test_describe_never_reveals_a_secret(monkeypatch):\n"
        + indent(
            docstring(
                "No secret value appears in the configuration summary.",
                [
                    "A distinctive sentinel is set as each secret and then searched for in "
                    "every value describe() returns. Even a prefix would be a leak - the first "
                    "characters of a key identify the account - so the assertion is on the "
                    "whole value and on a short prefix of it.",
                ],
            )
        )
        + "\n"
        "    if not SECRET_ENV:\n"
        "        pytest.skip(\"This project reads no secret environment variables.\")\n"
        "\n"
        f"    sentinel = {literal(_SECRET_SENTINEL)}\n"
        "    for name in SECRET_ENV:\n"
        "        monkeypatch.setenv(name, sentinel)\n"
        "    get_settings.cache_clear()\n"
        "\n"
        "    summary = get_settings().describe()\n"
        "    rendered = \" \".join(f\"{k}={v}\" for k, v in summary.items())\n"
        "    assert sentinel not in rendered\n"
        "    assert sentinel[:12] not in rendered\n"
        "    for name in SECRET_ENV:\n"
        "        assert summary[name] == \"set\"\n"
    )
    return Fragment(
        body=body,
        imports=[
            "import pytest",
            f"from {settings_module_name} import REQUIRED_ENV, SECRET_ENV, get_settings",
        ],
    )


def test_tools_module(manifest: ProjectManifest) -> Fragment:
    """
    ``tests/test_tools.py``: every tool is registered, callable, and records its call.

    Tools are the one part of a generated project that can be exercised for real without a
    model, so these are proper behavioural tests rather than construction checks.
    """
    tools_module = _tools_registry_module(manifest)
    telemetry = _telemetry_module(manifest)
    expected = [t.name for t in manifest.tools]

    body = (
        "#: The tools the plan asked for. Compared against what the code registers.\n"
        f"PLANNED_TOOLS = {literal(expected)}\n\n\n"
        "def test_every_planned_tool_is_registered():\n"
        + indent(
            docstring(
                "The registry contains exactly the planned tools.",
                [
                    "Both directions matter: a missing tool means a capability the agent was "
                    "promised and does not have, and an extra one means the plan and the code "
                    "have drifted apart.",
                ],
            )
        )
        + "\n"
        "    assert sorted(TOOLS) == sorted(PLANNED_TOOLS)\n\n\n"
        "@pytest.mark.parametrize(\"tool_name\", PLANNED_TOOLS)\n"
        "def test_tool_runs_and_is_recorded(tool_name):\n"
        + indent(
            docstring(
                "Calling a tool returns text and appends exactly one telemetry record.",
                [
                    "The telemetry assertion is the point: the platform shows a run's tool "
                    "calls from these records, so a tool that works but does not record is a "
                    "tool the user cannot see the agent using.",
                ],
            )
        )
        + "\n"
        "    reset_calls()\n"
        "    tool = TOOLS[tool_name]\n"
        "    result = _invoke(tool, \"a test question\")\n"
        "\n"
        "    assert isinstance(result, str) and result, \"A tool must return non-empty text.\"\n"
        "    calls = recorded_calls()\n"
        "    assert len(calls) == 1, f\"Expected one recorded call, got {len(calls)}.\"\n"
        "    assert calls[0][\"tool\"] == tool_name\n\n\n"
        "def _invoke(tool, query):\n"
        + indent(
            docstring(
                "Call a tool whichever shape it has.",
                [
                    "Frameworks differ: LangChain tools are objects with `invoke`, Agno tools "
                    "are plain callables. The test covers the tool's behaviour, not the "
                    "calling convention, so it adapts here rather than being written twice.",
                ],
            )
        )
        + "\n"
        "    invoke = getattr(tool, \"invoke\", None)\n"
        "    if callable(invoke):\n"
        "        return str(invoke(query))\n"
        "    return str(tool(query))\n\n\n"
        "def test_reset_clears_previous_calls():\n"
        + indent(
            docstring(
                "reset_calls() empties the record.",
                [
                    "Without this, a second run in the same process would report the first "
                    "run's tool calls as its own - which would look like the agent doing work "
                    "it never did.",
                ],
            )
        )
        + "\n"
        "    reset_calls()\n"
        "    record_call(\"probe\", {\"query\": \"x\"}, \"y\")\n"
        "    assert len(recorded_calls()) == 1\n"
        "    reset_calls()\n"
        "    assert recorded_calls() == []\n"
    )
    return Fragment(
        body=body,
        imports=[
            "import pytest",
            f"from {tools_module} import TOOLS",
            f"from {telemetry} import record_call, recorded_calls, reset_calls",
        ],
    )


def test_agents_module(manifest: ProjectManifest) -> Fragment:
    """
    ``tests/test_agents.py``: every agent builder constructs, and holds the tools it was given.

    Construction is where framework mistakes surface - a pydantic model with an unannotated
    field, a tool passed in the wrong shape, a model object a framework will not accept - so
    calling the builders is a genuine check even though no model is invoked.
    """
    imports = ["import pytest"]
    for agent in manifest.agents:
        if agent.module:
            imports.append(f"from {agent.module} import {agent.factory_name}")
    if not manifest.agents:
        imports.append("# This project plans no agents, so there is nothing to construct.")

    builders = "\n".join(
        f"    {literal(a.name)}: {a.factory_name}," for a in manifest.agents
    )
    tool_counts = {a.name: len([t for t in a.tools if manifest.tool(t)]) for a in manifest.agents}

    body = (
        "#: Agent name -> its builder, from the plan.\n"
        f"BUILDERS = {{\n{builders}\n}}\n\n"
        "#: How many tools each agent was planned to hold.\n"
        f"PLANNED_TOOL_COUNTS = {literal(tool_counts)}\n\n\n"
        "@pytest.mark.parametrize(\"agent_name\", sorted(BUILDERS))\n"
        "def test_agent_builds(agent_name, skip_if_local_model):\n"
        + indent(
            docstring(
                "Each agent constructs without error.",
                [
                    "No model call is made - the client is built and handed over, not used. "
                    "What this catches is construction-time validation, which is where most "
                    "framework mistakes actually land.",
                ],
            )
        )
        + "\n"
        "    agent = BUILDERS[agent_name]()\n"
        "    assert agent is not None\n\n\n"
        "@pytest.mark.parametrize(\"agent_name\", sorted(BUILDERS))\n"
        "def test_agent_has_its_planned_tools(agent_name, skip_if_local_model):\n"
        + indent(
            docstring(
                "Each agent holds the number of tools the plan gave it.",
                [
                    "Read from the constructed object, so this fails if the tools are declared "
                    "in the plan and dropped on the way into the framework - which is exactly "
                    "the bug that made an earlier generator produce agents with no tools at "
                    "all.",
                    "Frameworks expose the collection differently, and some wrap an executor "
                    "around the agent. When the attribute genuinely cannot be found the test "
                    "says so rather than passing quietly.",
                ],
            )
        )
        + "\n"
        "    expected = PLANNED_TOOL_COUNTS[agent_name]\n"
        "    if not expected:\n"
        "        pytest.skip(f\"{agent_name} was planned without tools.\")\n"
        "\n"
        "    built = BUILDERS[agent_name]()\n"
        "    # Some emitters return (runnable, tools); others return an object holding them.\n"
        "    if isinstance(built, tuple) and len(built) == 2:\n"
        "        tools = built[1]\n"
        "    else:\n"
        "        tools = getattr(built, \"tools\", None)\n"
        "\n"
        "    assert tools is not None, (\n"
        "        f\"Could not read the tools off the constructed {agent_name}. \"\n"
        "        \"The agent may have been built without them.\"\n"
        "    )\n"
        "    assert len(list(tools)) == expected\n"
    )
    if not manifest.agents:
        body = (
            "def test_project_plans_no_agents():\n"
            + indent(
                docstring(
                    "This project's plan contains no agents.",
                    [
                        "Recorded as a passing test rather than an empty file, so the absence "
                        "is visible in the run output instead of looking like missing coverage.",
                    ],
                )
            )
            + "\n"
            "    assert True\n"
        )
    return Fragment(body=body, imports=imports)


def test_workflow_module(manifest: ProjectManifest) -> Fragment:
    """
    ``tests/test_workflow.py``: the workflow assembles, and its shape matches the plan.

    ``build_workflow`` is called for real - the graph is compiled, the crew is constructed,
    the executor is built - because that is where a wiring mistake shows up. ``run_workflow``
    is not called, because running it needs a model; the platform's runtime validation does
    that separately with a real credential.
    """
    workflow = manifest.primary_workflow
    module = _workflow_module(manifest)
    steps = [n.id for n in workflow.agent_nodes()] if workflow else []

    body = (
        "#: The steps the plan describes, in order.\n"
        f"PLANNED_STEPS = {literal(steps)}\n\n\n"
        "def test_workflow_builds(skip_if_local_model):\n"
        + indent(
            docstring(
                "The workflow assembles without error.",
                [
                    "For a graph framework this compiles the graph, which is where an edge to "
                    "a node that was never added is caught. For a crew it validates every "
                    "agent and task together. Either way the failure is found here rather "
                    "than on the user's first run.",
                ],
            )
        )
        + "\n"
        "    assert build_workflow() is not None\n\n\n"
        "def test_run_workflow_is_callable():\n"
        + indent(
            docstring(
                "run_workflow exists and takes one argument.",
                [
                    "Not invoked: calling it needs a real model. This checks the contract the "
                    "runtime depends on, and the platform's runtime validation checks the "
                    "behaviour with a live credential.",
                ],
            )
        )
        + "\n"
        "    assert callable(run_workflow)\n"
        "    parameters = inspect.signature(run_workflow).parameters\n"
        "    assert len(parameters) == 1, \"run_workflow takes exactly one argument: the query.\"\n\n\n"
        "def test_steps_match_the_plan():\n"
        + indent(
            docstring(
                "The step names in the code are the step names in the plan.",
                [
                    "This is what keeps the workflow diagram honest. The picture is drawn from "
                    "the plan and the run is driven by STEPS, so if the two ever disagree the "
                    "diagram is showing something that does not happen.",
                    "Skipped for frameworks that build their graph from nodes rather than an "
                    "ordered step list, where there is no STEPS table to compare.",
                ],
            )
        )
        + "\n"
        "    steps = getattr(module, \"STEPS\", None)\n"
        "    if steps is None:\n"
        "        pytest.skip(\n"
        "            \"This framework's workflow module builds its graph directly and has no \"\n"
        "            \"ordered STEPS table.\"\n"
        "        )\n"
        "    assert [str(step[\"name\"]) for step in steps] == PLANNED_STEPS\n"
    )
    return Fragment(
        body=body,
        imports=[
            "import inspect",
            "import pytest",
            f"import {module} as module",
            f"from {module} import build_workflow, run_workflow",
        ],
    )


def test_simple_module(manifest: ProjectManifest) -> Fragment:
    """
    The flat tier's single test file.

    A small project gets one test file rather than five, for the same reason it gets three
    modules rather than fifteen: five files each holding two tests is harder to read than one
    file holding ten. The coverage is the same - imports, configuration, secret masking,
    tools, agent construction, workflow shape.
    """
    workflow_module_name = _workflow_module(manifest)
    settings_name = _settings_module(manifest)
    telemetry = _telemetry_module(manifest)
    tools_module = _tools_registry_module(manifest) or workflow_module_name
    planned_tools = [t.name for t in manifest.tools]

    body = (
        f"PLANNED_TOOLS = {literal(planned_tools)}\n"
        f"PLANNED_AGENTS = {literal([a.name for a in manifest.agents])}\n\n\n"
        "def test_modules_import():\n"
        + indent(docstring("Every module in this project imports cleanly."))
        + "\n"
        f"    for name in {literal([m for m in dict.fromkeys(_module(manifest, s.path) for s in manifest.files) if m and not m.startswith('test')])}:\n"
        "        importlib.import_module(name)\n\n\n"
        "def test_settings_report_missing_rather_than_raising(monkeypatch):\n"
        + indent(
            docstring(
                "A missing credential is reported by name, not raised at import.",
                [
                    "Import must never depend on configuration, or the project cannot be "
                    "tested without a real key.",
                ],
            )
        )
        + "\n"
        "    if not REQUIRED_ENV:\n"
        "        pytest.skip(\"This project reads no required environment variables.\")\n"
        "    for name in REQUIRED_ENV:\n"
        "        monkeypatch.setenv(name, \"\")\n"
        "    get_settings.cache_clear()\n"
        "    assert get_settings().missing() == list(REQUIRED_ENV)\n\n\n"
        "def test_describe_masks_secrets(monkeypatch):\n"
        + indent(
            docstring(
                "No secret value appears in the configuration summary.",
                [
                    "Asserted rather than assumed, so a change that starts printing keys fails "
                    "here instead of shipping.",
                ],
            )
        )
        + "\n"
        "    if not SECRET_ENV:\n"
        "        pytest.skip(\"This project reads no secret environment variables.\")\n"
        f"    sentinel = {literal(_SECRET_SENTINEL)}\n"
        "    for name in SECRET_ENV:\n"
        "        monkeypatch.setenv(name, sentinel)\n"
        "    get_settings.cache_clear()\n"
        "    rendered = \" \".join(str(v) for v in get_settings().describe().values())\n"
        "    assert sentinel not in rendered and sentinel[:12] not in rendered\n\n\n"
        "@pytest.mark.parametrize(\"tool_name\", PLANNED_TOOLS)\n"
        "def test_tool_runs_and_records(tool_name):\n"
        + indent(
            docstring(
                "Each tool returns text and records exactly one call.",
                [
                    "The run's tool-call list is built from these records, so a tool that runs "
                    "without recording is invisible to the user watching it work.",
                ],
            )
        )
        + "\n"
        "    reset_calls()\n"
        "    tool = TOOLS[tool_name]\n"
        "    invoke = getattr(tool, \"invoke\", None)\n"
        "    result = invoke(\"a test question\") if callable(invoke) else tool(\"a test question\")\n"
        "    assert str(result)\n"
        "    calls = recorded_calls()\n"
        "    assert len(calls) == 1 and calls[0][\"tool\"] == tool_name\n\n\n"
        "def test_workflow_builds(skip_if_local_model):\n"
        + indent(
            docstring(
                "The workflow assembles, and run_workflow takes one argument.",
                [
                    "run_workflow is not invoked: that needs a real model, and the platform "
                    "validates it separately with a live credential.",
                ],
            )
        )
        + "\n"
        "    assert build_workflow() is not None\n"
        "    assert len(inspect.signature(run_workflow).parameters) == 1\n"
    )
    return Fragment(
        body=body,
        imports=[
            "import importlib",
            "import inspect",
            "import pytest",
            f"from {settings_name} import REQUIRED_ENV, SECRET_ENV, get_settings",
            f"from {tools_module} import TOOLS",
            f"from {telemetry} import recorded_calls, reset_calls",
            f"from {workflow_module_name} import build_workflow, run_workflow",
        ],
    )


# ================================================================================= readme
def _readme_tree(manifest: ProjectManifest) -> str:
    """The file list with each file's purpose, so the tree explains itself."""
    width = max((len(f.path) for f in manifest.files), default=0)
    lines = []
    for spec in sorted(manifest.files, key=lambda f: f.path):
        marker = "  <- run this" if spec.is_entrypoint else ""
        lines.append(f"{spec.path.ljust(width)}  {spec.purpose}{marker}")
    return "\n".join(lines)


def readme(manifest: ProjectManifest, entry_command: str) -> str:
    """
    ``README.md``, written from the manifest.

    Every number, name and path here is read from the plan. There is no template sentence
    that asserts a fact the project might not have: the agent table has one row per planned
    agent, the tool list has one entry per planned tool, and a project with no tools says so
    rather than describing tools it does not have.

    It documents what the project does, how to run it, what each file is for, and - honestly -
    what is left to implement, because a generated tool body is a stub and a README that
    pretends otherwise wastes the reader's afternoon.
    """
    summary = manifest.architecture_summary()
    workflow = manifest.primary_workflow
    unimplemented = [t for t in manifest.tools if not t.implemented]

    parts: List[str] = []
    parts.append(f"# {manifest.project_name}\n")
    if manifest.description:
        parts.append(f"{manifest.description}\n")

    if manifest.requirement:
        parts.append("## What was asked for\n")
        parts.append(
            "> "
            + manifest.requirement.strip().replace("\n", "\n> ")
            + "\n\n"
            + "This project was generated from that request. The sections below describe what "
            + "was built to satisfy it.\n"
        )

    # --- capabilities
    if manifest.capabilities:
        parts.append("## What it can do\n")
        parts.append(
            "\n".join(f"- {c}" for c in manifest.capabilities)
            + "\n\nEach of these is implemented by the agents and workflow described below.\n"
        )

    # --- architecture
    parts.append("## How it works\n")
    kind = workflow.kind if workflow else "single-agent"
    parts.append(
        f"Built on **{manifest.framework}**, talking to the **{manifest.provider}** provider"
        + (f" using `{manifest.model}`" if manifest.model else "")
        + f". The workflow is **{kind}**: "
        + (workflow.description if workflow and workflow.description else "one agent handles the request.")
        + "\n"
    )
    parts.append(
        f"It is laid out in the **{manifest.tier}** structure - "
        f"{summary['python_files']} Python files across "
        f"{summary['agents']} agent{'s' if summary['agents'] != 1 else ''}, "
        f"{summary['tools']} tool{'s' if summary['tools'] != 1 else ''} and "
        f"{summary['workflow_steps']} workflow step"
        f"{'s' if summary['workflow_steps'] != 1 else ''}.\n"
    )

    # --- agents
    if manifest.agents:
        parts.append("### Agents\n")
        parts.append("| Agent | Role | Tools |")
        parts.append("| --- | --- | --- |")
        for agent in manifest.agents:
            tools = ", ".join(f"`{t}`" for t in agent.tools) or "none"
            role = (agent.role or agent.goal or "").replace("|", "\\|")
            parts.append(f"| **{agent.label}** | {role} | {tools} |")
        parts.append("")

    # --- workflow steps
    if workflow and workflow.agent_nodes():
        parts.append("### Workflow\n")
        for index, node in enumerate(workflow.agent_nodes(), start=1):
            owner = manifest.agent(node.agent or "")
            who = owner.label if owner else (node.agent or "unassigned")
            parts.append(f"{index}. **{node.label}** ({who}) - {node.description}")
            if node.outputs:
                parts.append(f"   - Produces: {node.outputs}")
        parts.append("")
        if workflow.has_branching:
            conditional = [e for e in workflow.edges if e.conditional]
            parts.append(
                f"The workflow branches: {len(conditional)} of its transitions are "
                "conditional, so the path taken depends on what an earlier step produced.\n"
            )

    # --- tools
    if manifest.tools:
        parts.append("### Tools\n")
        for tool in manifest.tools:
            state = "implemented" if tool.implemented else "**stub - needs a real implementation**"
            parts.append(f"- `{tool.name}` - {tool.purpose} ({state})")
        parts.append("")

    # --- setup
    parts.append("## Setup\n")
    parts.append("```bash")
    parts.append("python -m venv .venv")
    parts.append("source .venv/bin/activate        # Windows: .venv\\Scripts\\activate")
    parts.append("pip install -r requirements.txt")
    parts.append("cp .env.example .env             # then fill in the values")
    parts.append("```\n")

    if manifest.env_vars:
        parts.append("### Configuration\n")
        parts.append("| Variable | Required | Purpose |")
        parts.append("| --- | --- | --- |")
        for var in manifest.env_vars:
            parts.append(
                f"| `{var.name}` | {'yes' if var.required else 'no'} | {var.purpose} |"
            )
        parts.append(
            "\nThese are read from the environment at run time. No value is stored in the "
            "source, so this project is safe to commit - but `.env` is not: keep it out of "
            "version control.\n"
        )

    # --- running
    parts.append("## Running it\n")
    parts.append("```bash")
    parts.append(f"{entry_command} \"your question here\"")
    parts.append("")
    parts.append("# Full detail, including each step and every tool call:")
    parts.append(f"{entry_command} --json \"your question here\"")
    parts.append("")
    parts.append("# From another process or a script:")
    parts.append(f"echo \"your question here\" | {entry_command}")
    parts.append("```\n")
    parts.append(
        "The exit status is 0 when the run succeeded and 1 when it did not, so this works "
        "inside a shell script.\n"
    )

    # --- tests
    parts.append("## Tests\n")
    parts.append("```bash")
    parts.append("pytest")
    parts.append("```\n")
    parts.append(
        "The suite runs offline. Outbound network connections are blocked and placeholder "
        "credentials are supplied, so it needs no API key, costs nothing, and cannot hang "
        "waiting on a provider. It checks that every module imports, that configuration "
        "reports missing values clearly, that no secret appears in the configuration summary, "
        "that every tool runs and records its call, that every agent constructs with the tools "
        "it was given, and that the workflow assembles with the steps the plan describes.\n"
    )
    parts.append(
        "It deliberately does **not** call the model. A passing suite means the project is "
        "correctly wired, not that the agent gave a good answer - those are different "
        "questions, and only the first one can be answered without a credential.\n"
    )

    # --- what is left
    if unimplemented:
        parts.append("## What still needs your attention\n")
        parts.append(
            f"{len(unimplemented)} tool"
            + ("s are" if len(unimplemented) != 1 else " is")
            + " a stub. Each returns a placeholder that says so, so the agent will run "
            "end to end but cannot do real work through "
            + ("them" if len(unimplemented) != 1 else "it")
            + " until you replace the body:\n"
        )
        for tool in unimplemented:
            module = tool.module.replace(".", "/") + ".py" if tool.module else "tools.py"
            parts.append(f"- `{tool.name}` in `{module}` - {tool.purpose}")
        parts.append("")

    # --- file map
    parts.append("## Files\n")
    parts.append("```")
    parts.append(_readme_tree(manifest))
    parts.append("```\n")

    # --- troubleshooting
    parts.append("## If something goes wrong\n")
    parts.append(
        "**A missing-configuration error.** The message names the variable. Copy "
        "`.env.example` to `.env` and fill it in.\n"
    )
    parts.append(
        "**An authentication error from the provider.** The credential is present but not "
        "accepted. Check the key itself, and check which account it belongs to.\n"
    )
    parts.append(
        "**A `NetworkBlocked` error while running tests.** Something in the code path being "
        "tested tried to call the provider. That is the block doing its job - unit tests must "
        "not make real calls. Find the call rather than removing the block.\n"
    )
    parts.append(
        "**The agent runs but the answer is a placeholder.** A stub tool was called. See the "
        "section above.\n"
    )
    parts.append(
        "**The agent stops after several rounds without answering.** There is an iteration "
        "ceiling so a model that keeps calling tools cannot loop forever. The run output lists "
        "what it tried, which usually points at a tool description the model is "
        "misunderstanding.\n"
    )

    if manifest.notes:
        parts.append("## Notes from generation\n")
        parts.append("\n".join(f"- {n}" for n in manifest.notes) + "\n")

    parts.append("---\n")
    parts.append(
        f"Generated by multi-agent-generator for {manifest.framework}. "
        "The code is yours to edit - it is ordinary Python with no generator runtime.\n"
    )
    return "\n".join(parts)
