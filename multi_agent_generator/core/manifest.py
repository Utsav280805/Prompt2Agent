# multi_agent_generator/core/manifest.py
"""
The project manifest: what the system decided to build, before it builds it.

The old flow went straight from an agent configuration to one Python string. That is why a
generated "project" was a 400-line ``agent.py`` and why nothing downstream could answer
simple questions about it - which agents exist, which tools they use, what the workflow
looks like, which file is responsible for what, what breaks if one file changes. All of
that information existed only as text inside the emitted source, where the only way to
recover it is to read Python.

A manifest fixes that by making the plan explicit and structured *first*:

    requirement -> analysis -> framework choice -> MANIFEST -> files

Everything after the manifest is a rendering of it. The workflow graph the UI draws, the
agent and tool inspectors, the dependency graph, the README, the per-file "purpose" and
"used by" lines, the requirement traceability - none of those are derived by parsing
generated code. They are read off the manifest, which is the same object the emitters were
handed. That is the difference between a UI that describes the project and a UI that
guesses about it.

The manifest is also what makes the architecture *proportional*. A one-agent summariser
and a five-agent research crew both get a plan; they just get different plans, chosen by
:mod:`multi_agent_generator.core.architecture`. A tiny agent must not be spread across
thirty files to look impressive, and a large one must not be crammed into a single module
to keep the file count down.

Nothing in here imports a framework or calls a model. It is pure data plus a little
naming logic, so it can be constructed in a test in one line and serialised into the
database without a translation layer.
"""
from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .models import relative_path

__all__ = [
    "Tier",
    "FileKind",
    "NodeKind",
    "FUNCTION_TOOL_FRAMEWORKS",
    "EnvVarSpec",
    "ToolSpec",
    "AgentSpec",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowSpec",
    "FileSpec",
    "ProjectManifest",
    "slugify",
    "identifier",
    "class_name",
]


# ================================================================================ naming
def identifier(name: str, prefix: str = "x") -> str:
    """
    Turn an arbitrary label into a valid, non-reserved Python identifier.

    Duplicated in spirit from :func:`multi_agent_generator.frameworks._common.sanitize_identifier`
    and deliberately not imported from it: this module must stay free of framework imports
    so that manifests can be built and tested without any agent framework installed.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", (name or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def slugify(name: str, fallback: str = "agent_project") -> str:
    """A filesystem- and import-safe project slug."""
    slug = identifier(name, prefix="p")
    return slug or fallback


def class_name(name: str, suffix: str = "") -> str:
    """``web_search`` -> ``WebSearch``, or ``WebSearchTool`` with ``suffix="Tool"``."""
    parts = [p for p in re.split(r"[^0-9a-zA-Z]+", name or "") if p]
    if not parts:
        parts = ["custom"]
    camel = "".join(p[:1].upper() + p[1:] for p in parts)
    if camel[0].isdigit():
        camel = f"X{camel}"
    if suffix and not camel.endswith(suffix):
        camel += suffix
    return camel


def _titleise(name: str) -> str:
    """``research_agent`` -> ``Research Agent``. For labels shown to a person."""
    parts = [p for p in re.split(r"[^0-9a-zA-Z]+", name or "") if p]
    if not parts:
        return "Untitled"
    return " ".join(p[:1].upper() + p[1:] for p in parts)


# ============================================================================ vocabulary
class Tier:
    """
    How much structure a project gets.

    Three tiers, not a continuum, because the point of the tier is to pick a *layout*, and
    there are only three layouts worth having. Section 3 of the brief is explicit in both
    directions: a complex agent must not collapse into one file, and a tiny agent must not
    be exploded into thirty.

    ``SIMPLE``
        Flat. ``main.py``, ``agent.py``, ``config.py``. One agent, at most one tool, no
        branching, no state. Splitting this further would create files that exist only to
        be imported once.

    ``STANDARD``
        A package with one module per responsibility: ``app/agents/``, ``app/tools/``,
        ``app/workflows/``, ``app/services/``, ``app/config/``. This is the default and
        covers most real requests.

    ``MODULAR``
        ``STANDARD`` plus the pieces that only earn their place at size: a typed state
        model, an explicit runtime runner, and a memory store when state has to persist.
    """

    SIMPLE = "simple"
    STANDARD = "standard"
    MODULAR = "modular"

    ALL = (SIMPLE, STANDARD, MODULAR)

    @staticmethod
    def rank(tier: str) -> int:
        try:
            return Tier.ALL.index(tier)
        except ValueError:
            return Tier.ALL.index(Tier.STANDARD)


class FileKind:
    """
    What a file is *for*.

    Section 70 asks that every generated file have a clear reason to exist. The kind is how
    that requirement is enforced mechanically rather than hoped for: the planner cannot add
    a file without saying what category of responsibility it carries, and the UI groups the
    file tree by these categories instead of dumping a flat list.
    """

    ENTRYPOINT = "entrypoint"
    AGENT = "agent"
    TOOL = "tool"
    WORKFLOW = "workflow"
    SERVICE = "service"
    CONFIG = "config"
    STATE = "state"
    MEMORY = "memory"
    RUNTIME = "runtime"
    PACKAGE = "package"
    TEST = "test"
    DOC = "doc"
    META = "meta"


class NodeKind:
    """Node types in a workflow graph. Kept small so the UI can style each one."""

    START = "start"
    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    END = "end"


#: Frameworks whose tools are plain functions rather than classes.
#:
#: Agno's ``Agent(tools=[...])`` takes the callables themselves, so its emitter writes
#: ``def search(query: str) -> str`` where the others write ``class SearchTool(BaseTool)``.
#: See :meth:`ProjectManifest.tool_symbol` for why the plan has to know this.
FUNCTION_TOOL_FRAMEWORKS = frozenset({"agno"})


# ============================================================================ leaf specs
@dataclass
class EnvVarSpec:
    """
    One environment variable the generated project reads.

    Recorded as a name plus a description and never a value. This is what
    ``.env.example`` is rendered from and what the "environment variables documented"
    validation check verifies, which means the documentation cannot drift from the code:
    both come from the same list.
    """

    name: str
    purpose: str = ""
    required: bool = True
    example: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "required": self.required,
            "example": self.example,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvVarSpec":
        return cls(
            name=str(data.get("name") or ""),
            purpose=str(data.get("purpose") or ""),
            required=bool(data.get("required", True)),
            example=str(data.get("example") or ""),
        )


@dataclass
class ToolSpec:
    """
    One tool, as a plan rather than as emitted source.

    ``module`` is filled in by the planner once the layout tier is known, so the same tool
    lands at ``tools.py`` in a simple project and ``app/tools/web_search.py`` in a modular
    one without the tool description caring which.
    """

    name: str
    purpose: str = ""
    label: str = ""
    #: Parameter name -> short description. Rendered into the tool's docstring and schema.
    parameters: Dict[str, str] = field(default_factory=dict)
    #: Environment variable names this tool needs before it can do real work.
    requires_env: List[str] = field(default_factory=list)
    #: Extra pip requirements this tool implies, beyond the framework's own.
    dependencies: List[str] = field(default_factory=list)
    module: str = ""
    #: True when the emitted implementation is a real integration rather than a stub the
    #: user is expected to complete. Surfaced in the UI so nobody mistakes one for the
    #: other - a stub that claims to search the web is worse than no tool at all.
    implemented: bool = False

    def __post_init__(self) -> None:
        self.name = identifier(self.name, prefix="tool")
        if not self.label:
            self.label = _titleise(self.name)
        if not self.purpose:
            self.purpose = f"{self.label} capability used by the agents."

    @property
    def class_name(self) -> str:
        return class_name(self.name, suffix="Tool")

    @property
    def function_name(self) -> str:
        return self.name if self.name.startswith("run_") else f"{self.name}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "purpose": self.purpose,
            "class_name": self.class_name,
            "parameters": dict(self.parameters),
            "requires_env": list(self.requires_env),
            "dependencies": list(self.dependencies),
            "module": self.module,
            "implemented": self.implemented,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolSpec":
        return cls(
            name=str(data.get("name") or "tool"),
            purpose=str(data.get("purpose") or ""),
            label=str(data.get("label") or ""),
            parameters={
                str(k): str(v) for k, v in (data.get("parameters") or {}).items()
            },
            requires_env=[str(v) for v in (data.get("requires_env") or [])],
            dependencies=[str(v) for v in (data.get("dependencies") or [])],
            module=str(data.get("module") or ""),
            implemented=bool(data.get("implemented", False)),
        )


@dataclass
class AgentSpec:
    """
    One agent, as a plan.

    The fields are exactly what section 12 says the workflow node inspector must show, so
    the inspector reads this object directly instead of scraping the generated file. When a
    field is unknown it stays empty rather than being filled with a plausible-sounding
    default - an invented backstory presented as the system's decision is a small lie that
    makes every other number on the page less trustworthy.
    """

    name: str
    role: str = ""
    goal: str = ""
    backstory: str = ""
    label: str = ""
    responsibilities: List[str] = field(default_factory=list)
    #: Tool names, matching :attr:`ToolSpec.name`.
    tools: List[str] = field(default_factory=list)
    model: str = ""
    allow_delegation: bool = False
    verbose: bool = True
    #: What this agent expects to receive, in one plain-language line.
    inputs: str = ""
    #: What it produces.
    outputs: str = ""
    module: str = ""

    def __post_init__(self) -> None:
        self.name = identifier(self.name, prefix="agent")
        if not self.label:
            self.label = _titleise(self.name)
        if not self.role:
            self.role = self.label
        if not self.goal:
            self.goal = f"Carry out the {self.label} part of the request."

    @property
    def class_name(self) -> str:
        return class_name(self.name)

    @property
    def factory_name(self) -> str:
        """Name of the ``build_*`` function the emitters generate for this agent."""
        return f"build_{self.name}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "responsibilities": list(self.responsibilities),
            "tools": list(self.tools),
            "model": self.model,
            "allow_delegation": self.allow_delegation,
            "verbose": self.verbose,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "module": self.module,
            "factory": self.factory_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentSpec":
        return cls(
            name=str(data.get("name") or "agent"),
            role=str(data.get("role") or ""),
            goal=str(data.get("goal") or ""),
            backstory=str(data.get("backstory") or ""),
            label=str(data.get("label") or ""),
            responsibilities=[str(v) for v in (data.get("responsibilities") or [])],
            tools=[str(v) for v in (data.get("tools") or [])],
            model=str(data.get("model") or ""),
            allow_delegation=bool(data.get("allow_delegation", False)),
            verbose=bool(data.get("verbose", True)),
            inputs=str(data.get("inputs") or ""),
            outputs=str(data.get("outputs") or ""),
            module=str(data.get("module") or ""),
        )


# ============================================================================== workflow
@dataclass
class WorkflowNode:
    """One box in the workflow graph."""

    id: str
    kind: str = NodeKind.AGENT
    label: str = ""
    description: str = ""
    #: Agent name, when ``kind`` is ``AGENT``.
    agent: Optional[str] = None
    #: Tool name, when ``kind`` is ``TOOL``.
    tool: Optional[str] = None
    inputs: str = ""
    outputs: str = ""

    def __post_init__(self) -> None:
        self.id = identifier(self.id, prefix="node")
        if not self.label:
            self.label = _titleise(self.id)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "description": self.description,
            "agent": self.agent,
            "tool": self.tool,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowNode":
        return cls(
            id=str(data.get("id") or "node"),
            kind=str(data.get("kind") or NodeKind.AGENT),
            label=str(data.get("label") or ""),
            description=str(data.get("description") or ""),
            agent=(str(data["agent"]) if data.get("agent") else None),
            tool=(str(data["tool"]) if data.get("tool") else None),
            inputs=str(data.get("inputs") or ""),
            outputs=str(data.get("outputs") or ""),
        )


@dataclass
class WorkflowEdge:
    """
    One arrow in the workflow graph.

    ``condition`` is what makes a branch a branch. It is carried as text because it is both
    the label the graph draws and the comment the generated router function documents
    itself with - keeping one source for both stops the picture and the code disagreeing.
    """

    source: str
    target: str
    condition: Optional[str] = None
    label: str = ""

    def __post_init__(self) -> None:
        self.source = identifier(self.source, prefix="node")
        self.target = identifier(self.target, prefix="node")
        if not self.label and self.condition:
            self.label = self.condition

    @property
    def conditional(self) -> bool:
        return bool(self.condition)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "condition": self.condition,
            "label": self.label,
            "conditional": self.conditional,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowEdge":
        return cls(
            source=str(data.get("source") or "node"),
            target=str(data.get("target") or "node"),
            condition=(str(data["condition"]) if data.get("condition") else None),
            label=str(data.get("label") or ""),
        )


@dataclass
class WorkflowSpec:
    """
    The agent workflow, as a graph.

    Every framework's orchestration is projected onto this one shape - CrewAI's ordered
    task list, CrewAI Flow's ``@start``/``@listen`` chain, LangGraph's nodes and
    conditional edges, ReAct's reason/act cycle. That projection is what lets a single
    graph component render all five honestly, and it is why ``kind`` is recorded: the
    picture is drawn from the same data the code was emitted from, so it cannot be a
    decorative diagram that happens to resemble the implementation.
    """

    name: str
    #: "sequential" | "hierarchical" | "graph" | "reason_act"
    kind: str = "sequential"
    description: str = ""
    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)
    entry: str = ""
    exits: List[str] = field(default_factory=list)
    #: Names of fields carried in shared state, for frameworks that have any.
    state_fields: List[str] = field(default_factory=list)
    module: str = ""

    def __post_init__(self) -> None:
        self.name = identifier(self.name, prefix="workflow")

    # ------------------------------------------------------------------------- accessors
    def node(self, node_id: str) -> Optional[WorkflowNode]:
        wanted = identifier(node_id, prefix="node")
        return next((n for n in self.nodes if n.id == wanted), None)

    def agent_nodes(self) -> List[WorkflowNode]:
        return [n for n in self.nodes if n.kind == NodeKind.AGENT]

    def successors(self, node_id: str) -> List[WorkflowEdge]:
        wanted = identifier(node_id, prefix="node")
        return [e for e in self.edges if e.source == wanted]

    def predecessors(self, node_id: str) -> List[WorkflowEdge]:
        wanted = identifier(node_id, prefix="node")
        return [e for e in self.edges if e.target == wanted]

    @property
    def has_branching(self) -> bool:
        return any(e.conditional for e in self.edges)

    def ordered_agent_names(self) -> List[str]:
        """
        Agent names in workflow order, deduplicated.

        Uses graph order rather than the raw agent list because the two can differ: an
        agent may appear at two points in a flow, and the README, the overview strip and
        the emitted sequential runner all want the order the work actually happens in.
        """
        names: List[str] = []
        for node in self.nodes:
            if node.kind == NodeKind.AGENT and node.agent and node.agent not in names:
                names.append(node.agent)
        return names

    def dangling_edges(self) -> List[WorkflowEdge]:
        """Edges pointing at a node that does not exist. A planning bug, caught cheaply."""
        ids = {n.id for n in self.nodes}
        return [e for e in self.edges if e.source not in ids or e.target not in ids]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
            "entry": self.entry,
            "exits": list(self.exits),
            "state_fields": list(self.state_fields),
            "module": self.module,
            "has_branching": self.has_branching,
            "step_count": len(self.agent_nodes()),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowSpec":
        return cls(
            name=str(data.get("name") or "workflow"),
            kind=str(data.get("kind") or "sequential"),
            description=str(data.get("description") or ""),
            nodes=[WorkflowNode.from_dict(n) for n in (data.get("nodes") or [])],
            edges=[WorkflowEdge.from_dict(e) for e in (data.get("edges") or [])],
            entry=str(data.get("entry") or ""),
            exits=[str(v) for v in (data.get("exits") or [])],
            state_fields=[str(v) for v in (data.get("state_fields") or [])],
            module=str(data.get("module") or ""),
        )


# ================================================================================= files
@dataclass
class FileSpec:
    """
    One planned file: where it goes, why it exists, and what it needs.

    ``purpose`` is not decoration. It is what the code viewer shows above the source
    (section 26), what the README's file list is built from (section 30), and the thing
    that makes "every file has a clear reason to exist" checkable rather than aspirational
    - a planner that cannot state a purpose has no business adding the file.

    ``depends_on`` holds *planned* dependencies. The real graph is recovered from the
    emitted source by :mod:`multi_agent_generator.core.depgraph`, and comparing the two is
    how a file that imports something the plan never intended gets noticed.
    """

    path: str
    purpose: str
    kind: str = FileKind.META
    is_entrypoint: bool = False
    #: Top-level symbols this file is expected to define.
    provides: List[str] = field(default_factory=list)
    #: Paths of other planned files this one is expected to import.
    depends_on: List[str] = field(default_factory=list)
    #: Agent, tool or workflow name this file implements, when it implements exactly one.
    owner: Optional[str] = None

    def __post_init__(self) -> None:
        # Shared with GeneratedFile rather than reimplemented: the plan's path and the emitted
        # file's path are compared by string in the assembler, the dependency graph and the
        # validator, so two normalisers that disagree by one character break all three at once.
        self.path = relative_path(self.path)

    @property
    def module_name(self) -> Optional[str]:
        """Dotted import path, for ``.py`` files only."""
        if not self.path.endswith(".py"):
            return None
        stem = self.path[: -len(".py")]
        parts = [p for p in stem.split("/") if p]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "purpose": self.purpose,
            "kind": self.kind,
            "is_entrypoint": self.is_entrypoint,
            "provides": list(self.provides),
            "depends_on": list(self.depends_on),
            "owner": self.owner,
            "module": self.module_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FileSpec":
        return cls(
            path=str(data.get("path") or "unnamed.txt"),
            purpose=str(data.get("purpose") or ""),
            kind=str(data.get("kind") or FileKind.META),
            is_entrypoint=bool(data.get("is_entrypoint", False)),
            provides=[str(v) for v in (data.get("provides") or [])],
            depends_on=[str(v) for v in (data.get("depends_on") or [])],
            owner=(str(data["owner"]) if data.get("owner") else None),
        )


# ============================================================================== manifest
@dataclass
class ProjectManifest:
    """
    The whole plan for one generated project.

    Constructed by :func:`multi_agent_generator.core.architecture.plan_architecture`,
    consumed by the emitters, then stored alongside the project so that every later
    question - what does this build, how does it work, what can it do, what would break -
    is answered from the plan rather than re-derived from source text.
    """

    project_name: str
    framework: str
    provider: str
    model: str
    requirement: str = ""
    description: str = ""
    tier: str = Tier.STANDARD
    #: Import package root for the generated code. Empty string means a flat layout.
    package: str = "app"
    agents: List[AgentSpec] = field(default_factory=list)
    tools: List[ToolSpec] = field(default_factory=list)
    workflows: List[WorkflowSpec] = field(default_factory=list)
    files: List[FileSpec] = field(default_factory=list)
    #: pip requirement specifiers.
    dependencies: List[str] = field(default_factory=list)
    env_vars: List[EnvVarSpec] = field(default_factory=list)
    entrypoint: str = "main.py"
    #: Free-text notes about the plan, shown on the Overview tab.
    notes: List[str] = field(default_factory=list)
    #: Short plain-language capability labels, derived from agents/tools/workflow.
    capabilities: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.project_name = slugify(self.project_name)
        if self.tier not in Tier.ALL:
            self.tier = Tier.STANDARD

    # ------------------------------------------------------------------------- accessors
    @property
    def slug(self) -> str:
        return self.project_name

    def agent(self, name: str) -> Optional[AgentSpec]:
        wanted = identifier(name, prefix="agent")
        return next((a for a in self.agents if a.name == wanted), None)

    def tool(self, name: str) -> Optional[ToolSpec]:
        wanted = identifier(name, prefix="tool")
        return next((t for t in self.tools if t.name == wanted), None)

    def file(self, path: str) -> Optional[FileSpec]:
        wanted = relative_path(path)
        return next((f for f in self.files if f.path == wanted), None)

    @property
    def primary_workflow(self) -> Optional[WorkflowSpec]:
        return self.workflows[0] if self.workflows else None

    def files_of_kind(self, kind: str) -> List[FileSpec]:
        return [f for f in self.files if f.kind == kind]

    def module_for(self, path: str) -> Optional[str]:
        spec = self.file(path)
        return spec.module_name if spec else None

    def agents_using(self, tool_name: str) -> List[AgentSpec]:
        """Which agents reference a tool. Powers the Tools tab's 'used by' column."""
        wanted = identifier(tool_name, prefix="tool")
        return [a for a in self.agents if wanted in a.tools]

    def tools_for(self, agent_name: str) -> List[ToolSpec]:
        agent = self.agent(agent_name)
        if agent is None:
            return []
        found = [self.tool(name) for name in agent.tools]
        return [t for t in found if t is not None]

    def tool_symbol(self, tool: ToolSpec) -> str:
        """
        The name this tool is actually bound to in the emitted code.

        Framework-dependent, and that is not a detail the plan can ignore. CrewAI, LangGraph
        and the two ReAct variants each emit a tool *class*; Agno takes the callable itself and
        emits a plain *function*. The planner used to write ``tool.class_name`` into every
        planned file's ``provides`` regardless of framework, so an Agno project's plan claimed
        ``tools/search.py`` provides ``SearchTool`` while the file it generated provides
        ``search``. Nothing crashed - which is why it survived - but the import graph reported
        an unrealised dependency on every Agno project with a tool, and the code viewer's
        "provides" column named a symbol that does not exist.

        This is the plan-side statement of a convention whose definition lives in each
        emitter's ``tool_fragment``. Two statements of one rule can drift, so
        ``tests/test_validation.py`` asserts they agree for every framework rather than
        trusting that they do.
        """
        if self.framework in FUNCTION_TOOL_FRAMEWORKS:
            return tool.function_name
        return tool.class_name

    # --------------------------------------------------------------------------- mutation
    def add_file(self, spec: FileSpec) -> FileSpec:
        """Add or replace a planned file by path."""
        for index, existing in enumerate(self.files):
            if existing.path == spec.path:
                self.files[index] = spec
                break
        else:
            self.files.append(spec)
        if spec.is_entrypoint:
            self.entrypoint = spec.path
        return spec

    def add_dependency(self, requirement: str) -> None:
        text = str(requirement or "").strip()
        if text and text not in self.dependencies:
            self.dependencies.append(text)

    def add_env_var(self, spec: EnvVarSpec) -> None:
        if not spec.name:
            return
        if any(existing.name == spec.name for existing in self.env_vars):
            return
        self.env_vars.append(spec)

    # ---------------------------------------------------------------------------- summary
    def architecture_summary(self) -> Dict[str, Any]:
        """
        The counts section 69 asks for, computed rather than asserted.

        Every number here is ``len()`` of something that exists in the plan. There is no
        code path that can set "agents: 4" without there being four agent specs, which is
        the whole point - section 87 forbids hard-coded counts, and the cheapest way to
        honour that is to make the count structurally impossible to fake.
        """
        workflow = self.primary_workflow
        return {
            "project_name": self.project_name,
            "tier": self.tier,
            "framework": self.framework,
            "provider": self.provider,
            "model": self.model,
            "package": self.package,
            "agents": len(self.agents),
            "tools": len(self.tools),
            "workflows": len(self.workflows),
            "workflow_steps": len(workflow.agent_nodes()) if workflow else 0,
            "services": len(self.files_of_kind(FileKind.SERVICE)),
            "tests": len(self.files_of_kind(FileKind.TEST)),
            "files": len(self.files),
            "python_files": len([f for f in self.files if f.path.endswith(".py")]),
            "entrypoint": self.entrypoint,
            "runtime": "Python",
            "dependencies": len(self.dependencies),
            "env_vars": [v.name for v in self.env_vars],
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "framework": self.framework,
            "provider": self.provider,
            "model": self.model,
            "requirement": self.requirement,
            "description": self.description,
            "tier": self.tier,
            "package": self.package,
            "agents": [a.as_dict() for a in self.agents],
            "tools": [t.as_dict() for t in self.tools],
            "workflows": [w.as_dict() for w in self.workflows],
            "files": [f.as_dict() for f in self.files],
            "dependencies": list(self.dependencies),
            "env_vars": [v.as_dict() for v in self.env_vars],
            "entrypoint": self.entrypoint,
            "notes": list(self.notes),
            "capabilities": list(self.capabilities),
            "summary": self.architecture_summary(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectManifest":
        """
        Rebuild from :meth:`as_dict`.

        Needed because the manifest is persisted with the project and read back when the
        user opens a workspace in a later session. ``summary`` is deliberately ignored on
        the way in: it is derived, and honouring a stored copy would let a stale count
        outlive the plan it described.
        """
        return cls(
            project_name=str(data.get("project_name") or "agent_project"),
            framework=str(data.get("framework") or ""),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            requirement=str(data.get("requirement") or ""),
            description=str(data.get("description") or ""),
            tier=str(data.get("tier") or Tier.STANDARD),
            package=str(data.get("package") or ""),
            agents=[AgentSpec.from_dict(a) for a in (data.get("agents") or [])],
            tools=[ToolSpec.from_dict(t) for t in (data.get("tools") or [])],
            workflows=[WorkflowSpec.from_dict(w) for w in (data.get("workflows") or [])],
            files=[FileSpec.from_dict(f) for f in (data.get("files") or [])],
            dependencies=[str(d) for d in (data.get("dependencies") or [])],
            env_vars=[EnvVarSpec.from_dict(v) for v in (data.get("env_vars") or [])],
            entrypoint=str(data.get("entrypoint") or "main.py"),
            notes=[str(n) for n in (data.get("notes") or [])],
            capabilities=[str(c) for c in (data.get("capabilities") or [])],
        )
