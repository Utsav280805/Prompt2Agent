# multi_agent_generator/core/architecture.py
"""
Stage 2.5: decide what to build, and how much of it, before any code is emitted.

This module sits between "we understand the requirement" and "we write the files". It takes
the analysis, the chosen framework and the model-authored agent configuration, and turns
them into a :class:`~multi_agent_generator.core.manifest.ProjectManifest` - an explicit plan
naming every agent, every tool, the workflow graph and every file with the reason it exists.

Two problems are solved here.

The first is architecture. The old flow asked one model call for one enormous Python string,
so the "project" was a single ``agent.py`` no matter how large the request. Planning first
means the layout can be chosen deliberately, and chosen *proportionally*: a one-agent
summariser gets three flat files, a five-agent research crew gets a package with a module
per responsibility. Neither extreme - a 2,000-line monolith, or thirty files for a toy - is
an accident that can happen here, because the tier is a decision with a written rule
(:func:`decide_tier`) rather than an emergent property of a prompt.

The second is knowability. Because the plan exists as data, every later question can be
answered by reading it: which agents exist, which tools each one holds, what the workflow
looks like, which file does what, what depends on what. Nothing downstream has to parse
generated Python to describe the project it just built.

No model is called from this module. The one genuinely linguistic decision - which agents
and tools a requirement implies - was already made upstream by ``build_agent_config``. What
happens here is deterministic: given the same analysis and config, the same manifest comes
out, which is what makes the plan reviewable and the generation reproducible.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..dependencies import requirements_for
from ..providers import get_provider, resolve_runtime_model
from .manifest import (
    AgentSpec,
    EnvVarSpec,
    FileKind,
    FileSpec,
    NodeKind,
    ProjectManifest,
    Tier,
    ToolSpec,
    WorkflowEdge,
    WorkflowNode,
    WorkflowSpec,
    identifier,
    slugify,
)
from .models import FrameworkChoice, RequirementAnalysis

__all__ = [
    "plan_architecture",
    "decide_tier",
    "derive_project_name",
    "FRAMEWORK_LABELS",
]


#: Human-readable framework names, for labels and generated documentation.
FRAMEWORK_LABELS = {
    "crewai": "CrewAI",
    "crewai-flow": "CrewAI Flow",
    "langgraph": "LangGraph",
    "react": "ReAct",
    "react-lcel": "ReAct (LCEL)",
    "agno": "Agno",
}

#: Which orchestration shape each framework's workflow graph takes. This is what makes
#: framework selection structural rather than cosmetic: the graph the UI draws, the modules
#: the emitters write and the runner's control flow all follow from this one mapping.
FRAMEWORK_WORKFLOW_KIND = {
    "crewai": "sequential",  # overridden to "hierarchical" when the config says so
    "crewai-flow": "sequential",
    "langgraph": "graph",
    "react": "reason_act",
    "react-lcel": "reason_act",
    "agno": "sequential",
}

#: Words that carry no meaning in a project name. "Create a research assistant that
#: summarises papers" should become ``research_assistant``, not ``create_a_research``.
_NAME_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "app",
        "application",
        "build",
        "can",
        "create",
        "design",
        "for",
        "generate",
        "help",
        "i",
        "ial",
        "make",
        "me",
        "my",
        "need",
        "of",
        "please",
        "set",
        "setup",
        "some",
        "system",
        "that",
        "the",
        "to",
        "up",
        "want",
        "which",
        "will",
        "with",
        "would",
        "write",
        "you",
    }
)


# ========================================================================== project name
def derive_project_name(analysis: RequirementAnalysis) -> str:
    """
    Pick a short, meaningful slug from the requirement text.

    Used for the directory name, the README title and the export filename. Falls back to
    ``agent_project`` rather than to something invented, because a wrong-but-confident name
    is more annoying to correct than a neutral one.
    """
    text = (analysis.summary or analysis.requirement or "").strip()
    words = [w for w in re.split(r"[^0-9a-zA-Z]+", text.lower()) if w]
    kept = [w for w in words if w not in _NAME_STOPWORDS and len(w) > 2][:3]
    if not kept:
        return "agent_project"
    return slugify("_".join(kept))


# ================================================================================== tiers
def decide_tier(
    agent_count: int,
    tool_count: int,
    analysis: RequirementAnalysis,
) -> str:
    """
    Choose how much structure the project gets.

    The rule is written out rather than inferred so that the Overview tab can state *why*
    a project has the shape it has, and so that the same requirement always produces the
    same layout.

    ``SIMPLE``
        One agent, at most one tool, no branching, no shared state. Splitting this across a
        package would produce modules that exist only to be imported once each.

    ``MODULAR``
        Four or more agents, or four or more tools, or branching *and* state together, or
        an analysis that called the requirement complex. At this size a flat layout stops
        being readable and the extra modules start paying for themselves.

    ``STANDARD``
        Everything in between, which is most real requests.
    """
    complexity = (analysis.complexity or "moderate").strip().lower()

    if (
        agent_count <= 1
        and tool_count <= 1
        and not analysis.needs_branching
        and not analysis.needs_state
        and complexity in ("simple", "trivial", "low", "")
    ):
        return Tier.SIMPLE

    if (
        agent_count >= 4
        or tool_count >= 4
        or (analysis.needs_branching and analysis.needs_state)
        or complexity in ("complex", "high", "advanced")
    ):
        return Tier.MODULAR

    return Tier.STANDARD


def tier_reason(tier: str, agent_count: int, tool_count: int, analysis: RequirementAnalysis) -> str:
    """One sentence explaining the tier, for the Overview tab and the README."""
    facts: List[str] = [
        f"{agent_count} agent{'s' if agent_count != 1 else ''}",
        f"{tool_count} tool{'s' if tool_count != 1 else ''}",
    ]
    if analysis.needs_branching:
        facts.append("conditional branching")
    if analysis.needs_state:
        facts.append("shared state")
    if analysis.needs_delegation:
        facts.append("delegation")
    detail = ", ".join(facts)

    if tier == Tier.SIMPLE:
        return (
            f"Flat layout: {detail}. Splitting this into a package would create modules "
            "that are each imported exactly once."
        )
    if tier == Tier.MODULAR:
        return (
            f"Modular package layout: {detail}. At this size the workflow, state and "
            "runtime each need their own module to stay readable."
        )
    return (
        f"Package layout with one module per responsibility: {detail}."
    )


# ================================================================================= agents
def _plan_agents(
    config: Mapping[str, Any],
    analysis: RequirementAnalysis,
    runtime_model: str,
) -> List[AgentSpec]:
    """
    Read the agent list out of the model-authored config.

    Names are made unique here rather than later: two agents whose labels both reduce to
    the same identifier would otherwise collide into one module, silently losing an agent
    from the generated project. Falling back to ``suggested_roles`` and then to a single
    generic assistant means an unusable config degrades to a working small project instead
    of an empty one.
    """
    raw_agents = list(config.get("agents") or [])

    if not raw_agents and analysis.suggested_roles:
        raw_agents = [{"name": role, "role": role} for role in analysis.suggested_roles]

    if not raw_agents:
        raw_agents = [
            {
                "name": "assistant",
                "role": "Assistant",
                "goal": analysis.summary or analysis.requirement or "Help the user.",
            }
        ]

    specs: List[AgentSpec] = []
    used: set[str] = set()
    for index, raw in enumerate(raw_agents):
        if not isinstance(raw, Mapping):
            continue
        base = identifier(str(raw.get("name") or f"agent_{index + 1}"), prefix="agent")
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)

        tools = [
            identifier(str(t), prefix="tool")
            for t in (raw.get("tools") or [])
            if isinstance(t, str) and t.strip()
        ]

        specs.append(
            AgentSpec(
                name=name,
                role=str(raw.get("role") or "").strip(),
                goal=str(raw.get("goal") or "").strip(),
                backstory=str(raw.get("backstory") or "").strip(),
                label=str(raw.get("label") or "").strip(),
                responsibilities=[
                    str(r).strip()
                    for r in (raw.get("responsibilities") or [])
                    if str(r).strip()
                ],
                tools=list(dict.fromkeys(tools)),
                model=runtime_model,
                allow_delegation=bool(raw.get("allow_delegation", False)),
                verbose=bool(raw.get("verbose", True)),
            )
        )

    if not specs:
        specs = [AgentSpec(name="assistant", model=runtime_model)]

    # The analysis asked for delegation; honour it even when the model-authored config
    # forgot the flag. Without this, a 2-agent crew is scored as missing REQ coverage
    # while the workflow still runs the specialists in order.
    if analysis.needs_delegation:
        specs[0].allow_delegation = True

    return specs


def _plan_tools(
    config: Mapping[str, Any],
    analysis: RequirementAnalysis,
    agents: Sequence[AgentSpec],
) -> List[ToolSpec]:
    """
    Collect every tool the agents reference, plus any the config lists standalone.

    ReAct configs put tools at the top level; crew-style configs attach them to agents. Both
    are read so that a tool never goes missing just because of where it was declared.

    Every tool planned here is emitted as a working stub - a real function with a real
    signature that returns a clearly-labelled placeholder - and ``implemented`` stays False
    so the UI can say so. A stub that quietly pretends to search the web would be worse
    than no tool at all, because the failure would surface as a confidently wrong answer.
    """
    names: List[str] = []
    descriptions: Dict[str, str] = {}

    for raw in config.get("tools") or []:
        if isinstance(raw, str) and raw.strip():
            names.append(identifier(raw, prefix="tool"))
        elif isinstance(raw, Mapping):
            raw_name = str(raw.get("name") or "").strip()
            if raw_name:
                key = identifier(raw_name, prefix="tool")
                names.append(key)
                text = str(raw.get("description") or raw.get("purpose") or "").strip()
                if text:
                    descriptions[key] = text

    for agent in agents:
        names.extend(agent.tools)

    for suggested in analysis.suggested_tools:
        if isinstance(suggested, str) and suggested.strip():
            names.append(identifier(suggested, prefix="tool"))

    ordered = list(dict.fromkeys(n for n in names if n))

    specs: List[ToolSpec] = []
    seen_classes: set[str] = set()
    for name in ordered:
        spec = ToolSpec(
            name=name,
            purpose=descriptions.get(name, ""),
            parameters={"query": "What to act on, as free text."},
        )
        # Two differently-spelled tools can collapse to the same class name, which would
        # emit one class twice and shadow the first. Drop the duplicate rather than
        # generate a file that cannot be imported.
        if spec.class_name in seen_classes:
            continue
        seen_classes.add(spec.class_name)
        specs.append(spec)

    return specs


# =============================================================================== workflow
def _task_nodes(
    config: Mapping[str, Any],
    agents: Sequence[AgentSpec],
) -> List[WorkflowNode]:
    """
    Turn the config's ordered task list into agent nodes.

    A task's ``agent`` is matched by identifier, not by exact string, because the model that
    wrote the config is not consistent about capitalisation or spacing. When a task names an
    agent that does not exist the node is still created and pinned to the first agent, which
    is what the code emitters do too - the picture and the code stay in agreement.
    """
    by_name = {a.name: a for a in agents}
    fallback = agents[0].name if agents else "assistant"

    nodes: List[WorkflowNode] = []
    used: set[str] = set()
    raw_tasks = [t for t in (config.get("tasks") or []) if isinstance(t, Mapping)]

    for index, task in enumerate(raw_tasks):
        base = identifier(str(task.get("name") or f"step_{index + 1}"), prefix="step")
        node_id = base
        suffix = 2
        while node_id in used:
            node_id = f"{base}_{suffix}"
            suffix += 1
        used.add(node_id)

        owner = identifier(str(task.get("agent") or ""), prefix="agent")
        if owner not in by_name:
            owner = fallback

        nodes.append(
            WorkflowNode(
                id=node_id,
                kind=NodeKind.AGENT,
                agent=owner,
                description=str(task.get("description") or "").strip(),
                inputs="The user query, plus the previous step's output." if index else "The user query.",
                outputs=str(task.get("expected_output") or "").strip(),
            )
        )

    if nodes:
        return nodes

    # No tasks in the config: one step per agent, in declaration order.
    for index, agent in enumerate(agents):
        nodes.append(
            WorkflowNode(
                id=f"step_{index + 1}_{agent.name}",
                kind=NodeKind.AGENT,
                label=agent.label,
                agent=agent.name,
                description=agent.goal,
                inputs="The user query, plus the previous step's output." if index else "The user query.",
                outputs=agent.outputs or "Its contribution to the final answer.",
            )
        )
    return nodes


def _sequential_workflow(
    name: str,
    kind: str,
    config: Mapping[str, Any],
    agents: Sequence[AgentSpec],
    description: str,
) -> WorkflowSpec:
    """A straight chain: start -> step -> step -> ... -> end."""
    steps = _task_nodes(config, agents)
    nodes = [WorkflowNode(id="start", kind=NodeKind.START, label="Request")]
    nodes.extend(steps)
    nodes.append(WorkflowNode(id="end", kind=NodeKind.END, label="Response"))

    edges: List[WorkflowEdge] = []
    previous = "start"
    for node in steps:
        edges.append(WorkflowEdge(source=previous, target=node.id))
        previous = node.id
    edges.append(WorkflowEdge(source=previous, target="end"))

    return WorkflowSpec(
        name=name,
        kind=kind,
        description=description,
        nodes=nodes,
        edges=edges,
        entry=steps[0].id if steps else "end",
        exits=["end"],
    )


def _hierarchical_workflow(
    name: str,
    config: Mapping[str, Any],
    agents: Sequence[AgentSpec],
) -> WorkflowSpec:
    """
    A manager delegating to specialists.

    The first agent is the manager, which is exactly the convention the CrewAI emitter
    follows when it sets ``manager_agent``. Drawing it any other way would show a structure
    the generated code does not have.
    """
    if len(agents) < 2:
        return _sequential_workflow(
            name,
            "sequential",
            config,
            agents,
            "A single agent handles the request from start to finish.",
        )

    manager, specialists = agents[0], list(agents[1:])
    nodes = [
        WorkflowNode(id="start", kind=NodeKind.START, label="Request"),
        WorkflowNode(
            id=f"manager_{manager.name}",
            kind=NodeKind.AGENT,
            label=f"{manager.label} (manager)",
            agent=manager.name,
            description=manager.goal,
            inputs="The user query.",
            outputs="Work assignments for the specialists, then the consolidated answer.",
        ),
    ]
    edges = [WorkflowEdge(source="start", target=f"manager_{manager.name}")]

    for specialist in specialists:
        node_id = f"worker_{specialist.name}"
        nodes.append(
            WorkflowNode(
                id=node_id,
                kind=NodeKind.AGENT,
                label=specialist.label,
                agent=specialist.name,
                description=specialist.goal,
                inputs="A task delegated by the manager.",
                outputs=specialist.outputs or "The result of its assigned task.",
            )
        )
        edges.append(
            WorkflowEdge(
                source=f"manager_{manager.name}",
                target=node_id,
                label="delegates",
            )
        )
        edges.append(WorkflowEdge(source=node_id, target=f"manager_{manager.name}", label="reports back"))

    nodes.append(WorkflowNode(id="end", kind=NodeKind.END, label="Response"))
    edges.append(WorkflowEdge(source=f"manager_{manager.name}", target="end", label="final answer"))

    return WorkflowSpec(
        name=name,
        kind="hierarchical",
        description=(
            f"{manager.label} plans the work, delegates to "
            f"{len(specialists)} specialist{'s' if len(specialists) != 1 else ''}, "
            "then consolidates the result."
        ),
        nodes=nodes,
        edges=edges,
        entry=f"manager_{manager.name}",
        exits=["end"],
    )


def _graph_workflow(
    name: str,
    config: Mapping[str, Any],
    agents: Sequence[AgentSpec],
    analysis: RequirementAnalysis,
) -> WorkflowSpec:
    """
    LangGraph: the config's own nodes and edges, including conditional ones.

    ``condition`` on an edge was previously carried in the config and never read by
    anything, so a requirement that genuinely needed branching produced a straight line.
    Here it becomes both a labelled edge in the picture and a routing function in the
    emitted code.
    """
    raw_nodes = [n for n in (config.get("nodes") or []) if isinstance(n, Mapping)]
    raw_edges = [e for e in (config.get("edges") or []) if isinstance(e, Mapping)]

    if not raw_nodes:
        spec = _sequential_workflow(
            name,
            "graph",
            config,
            agents,
            "A state graph whose nodes run in order.",
        )
        spec.state_fields = ["messages", "next"]
        return spec

    by_agent = {a.name: a for a in agents}
    fallback = agents[0].name if agents else "assistant"

    nodes = [WorkflowNode(id="start", kind=NodeKind.START, label="Request")]
    node_ids: List[str] = []
    for index, raw in enumerate(raw_nodes):
        node_id = identifier(str(raw.get("name") or f"node_{index + 1}"), prefix="node")
        if node_id in node_ids:
            continue
        owner = identifier(str(raw.get("agent") or ""), prefix="agent")
        if owner not in by_agent:
            owner = fallback
        node_ids.append(node_id)
        owner_spec = by_agent.get(owner)
        nodes.append(
            WorkflowNode(
                id=node_id,
                kind=NodeKind.AGENT,
                agent=owner,
                description=str(
                    raw.get("description") or (owner_spec.goal if owner_spec else "")
                ),
                inputs="The graph state.",
                outputs="The updated graph state.",
            )
        )
    nodes.append(WorkflowNode(id="end", kind=NodeKind.END, label="Response"))

    entry = node_ids[0] if node_ids else "end"
    edges: List[WorkflowEdge] = [WorkflowEdge(source="start", target=entry)]
    seen: set[tuple[str, str, Optional[str]]] = set()
    for raw in raw_edges:
        source = identifier(str(raw.get("source") or ""), prefix="node")
        raw_target = str(raw.get("target") or "")
        target = "end" if raw_target.strip().upper() == "END" else identifier(raw_target, prefix="node")
        if source not in node_ids or (target not in node_ids and target != "end"):
            continue
        condition = str(raw.get("condition") or "").strip() or None
        key = (source, target, condition)
        if key in seen:
            continue
        seen.add(key)
        edges.append(WorkflowEdge(source=source, target=target, condition=condition))

    # Any node with no outgoing edge would strand the graph, so terminate it explicitly.
    for node_id in node_ids:
        if not any(e.source == node_id for e in edges):
            edges.append(WorkflowEdge(source=node_id, target="end"))

    state_fields = ["messages", "next"]
    if analysis.needs_state:
        state_fields.append("context")

    return WorkflowSpec(
        name=name,
        kind="graph",
        description=(
            f"A state graph with {len(node_ids)} node"
            f"{'s' if len(node_ids) != 1 else ''}"
            + (", including conditional routing." if any(e.conditional for e in edges) else ".")
        ),
        nodes=nodes,
        edges=edges,
        entry=entry,
        exits=["end"],
        state_fields=state_fields,
    )


def _reason_act_workflow(
    name: str,
    agents: Sequence[AgentSpec],
    tools: Sequence[ToolSpec],
) -> WorkflowSpec:
    """
    ReAct: think, optionally call a tool, think again, answer.

    The loop is drawn as a loop because that is what a ReAct agent does. Flattening it into
    a chain would misrepresent the one thing that distinguishes this framework from the
    others.
    """
    agent = agents[0] if agents else AgentSpec(name="assistant")
    nodes = [
        WorkflowNode(id="start", kind=NodeKind.START, label="Request"),
        WorkflowNode(
            id="reason",
            kind=NodeKind.AGENT,
            label=f"{agent.label} (reason)",
            agent=agent.name,
            description="Decides whether to answer directly or call a tool.",
            inputs="The user query and everything observed so far.",
            outputs="Either a tool call or the final answer.",
        ),
    ]
    edges = [WorkflowEdge(source="start", target="reason")]

    if tools:
        nodes.append(
            WorkflowNode(
                id="act",
                kind=NodeKind.TOOL,
                label="Call tool",
                tool=tools[0].name,
                description=(
                    "Runs the selected tool: "
                    + ", ".join(t.label for t in tools[:4])
                    + ("..." if len(tools) > 4 else "")
                ),
                inputs="The tool name and its arguments.",
                outputs="The tool's observation.",
            )
        )
        edges.append(
            WorkflowEdge(source="reason", target="act", condition="a tool call is requested")
        )
        edges.append(WorkflowEdge(source="act", target="reason", label="observation"))

    nodes.append(WorkflowNode(id="end", kind=NodeKind.END, label="Response"))
    edges.append(
        WorkflowEdge(
            source="reason",
            target="end",
            condition="the answer is ready" if tools else None,
        )
    )

    return WorkflowSpec(
        name=name,
        kind="reason_act",
        description=(
            "A reason/act loop: the agent thinks, optionally calls a tool, observes the "
            "result, and repeats until it can answer."
            if tools
            else "A single reasoning step with no tools."
        ),
        nodes=nodes,
        edges=edges,
        entry="reason",
        exits=["end"],
        state_fields=["scratchpad"] if tools else [],
    )


def _plan_workflow(
    framework: str,
    config: Mapping[str, Any],
    agents: Sequence[AgentSpec],
    tools: Sequence[ToolSpec],
    analysis: RequirementAnalysis,
) -> WorkflowSpec:
    """Project the chosen framework's orchestration onto one graph shape."""
    name = "main_workflow"
    process = str(config.get("process") or "").strip().lower()
    kind = FRAMEWORK_WORKFLOW_KIND.get(framework, "sequential")

    if framework in ("react", "react-lcel"):
        return _reason_act_workflow(name, agents, tools)

    if framework == "langgraph":
        return _graph_workflow(name, config, agents, analysis)

    if process == "hierarchical" or (analysis.needs_delegation and len(agents) >= 2):
        return _hierarchical_workflow(name, config, agents)

    description = (
        "Each agent runs in turn, and each one receives the previous agent's output."
        if framework != "crewai-flow"
        else "A flow whose steps are chained explicitly, each listening for the previous one."
    )
    return _sequential_workflow(name, kind, config, agents, description)


# ================================================================================ layouts
def _module_path(package: str, *parts: str) -> str:
    segments = [p for p in (package, *parts) if p]
    return "/".join(segments)


def _layout_simple(manifest: ProjectManifest, include_tests: bool) -> None:
    """
    Flat layout for the smallest projects.

    ``main.py`` / ``agent.py`` / ``config.py`` and nothing else. Three files a person can
    read in one sitting, which for a single-agent project is the honest architecture.
    """
    manifest.package = ""

    for tool in manifest.tools:
        tool.module = "tools"
    for agent in manifest.agents:
        agent.module = "agent"
    for workflow in manifest.workflows:
        workflow.module = "agent"

    manifest.add_file(
        FileSpec(
            path="config.py",
            purpose="Reads settings and the API credential from the environment. No secret is ever hard-coded here.",
            kind=FileKind.CONFIG,
            provides=["Settings", "get_settings"],
        )
    )

    if manifest.tools:
        manifest.add_file(
            FileSpec(
                path="tools.py",
                purpose=(
                    "The tools the agent can call: "
                    + ", ".join(t.label for t in manifest.tools)
                    + ". Each one is a working stub you replace with a real implementation. "
                    "Tool-call recording lives here too, next to the calls it records."
                ),
                kind=FileKind.TOOL,
                # The telemetry helpers live in this module in the flat layout. A separate
                # three-function module would be a file that exists only to be imported once,
                # which is exactly what the simple tier is meant to avoid.
                provides=[manifest.tool_symbol(t) for t in manifest.tools]
                + ["TOOLS", "all_tools", "record_call", "recorded_calls", "reset_calls"],
            )
        )

    agent_provides = [a.factory_name for a in manifest.agents] + [
        "build_llm",
        "get_llm",
        "build_workflow",
        "run_workflow",
    ]
    if not manifest.tools:
        # No tools means no tools.py, so the recording helpers go here - the workflow still
        # reports its tool calls, and reports an empty list honestly.
        agent_provides.extend(["record_call", "recorded_calls", "reset_calls"])

    agent_deps = ["config.py"] + (["tools.py"] if manifest.tools else [])
    manifest.add_file(
        FileSpec(
            path="agent.py",
            purpose=(
                f"Builds the {manifest.agents[0].label} agent and runs it. "
                "This is where the framework wiring lives."
            ),
            kind=FileKind.AGENT,
            provides=agent_provides,
            depends_on=agent_deps,
            owner=manifest.agents[0].name if manifest.agents else None,
        )
    )

    manifest.add_file(
        FileSpec(
            path="main.py",
            purpose="Command-line entry point. Reads a query from the arguments or stdin and prints the agent's answer.",
            kind=FileKind.ENTRYPOINT,
            is_entrypoint=True,
            provides=["main"],
            depends_on=["agent.py", "config.py"],
        )
    )

    if include_tests:
        manifest.add_file(
            FileSpec(
                path="tests/conftest.py",
                purpose="Test fixtures. Blocks network access and supplies a deterministic fake model, so the suite needs no API key.",
                kind=FileKind.TEST,
            )
        )
        manifest.add_file(
            FileSpec(
                path="tests/test_agent.py",
                purpose="Checks the module imports, the settings load, and the agent and its tools can be constructed offline.",
                kind=FileKind.TEST,
                depends_on=["agent.py", "config.py"],
            )
        )


def _layout_package(manifest: ProjectManifest, include_tests: bool) -> None:
    """
    Package layout for standard and modular tiers.

    One module per responsibility, so that changing a tool touches a tool file and changing
    the workflow touches a workflow file. The modular tier adds the pieces that only earn
    their keep at size: a typed state model, an explicit runtime runner, and - when the
    requirement needs memory - a store.
    """
    package = manifest.package or "app"
    manifest.package = package
    modular = manifest.tier == Tier.MODULAR

    manifest.add_file(
        FileSpec(
            path=_module_path(package, "__init__.py"),
            purpose=f"Marks `{package}` as a package and exposes the public entry function.",
            kind=FileKind.PACKAGE,
        )
    )

    # ------------------------------------------------------------------------- config
    settings_path = _module_path(package, "config", "settings.py")
    manifest.add_file(
        FileSpec(
            path=_module_path(package, "config", "__init__.py"),
            purpose="Configuration package.",
            kind=FileKind.PACKAGE,
        )
    )
    manifest.add_file(
        FileSpec(
            path=settings_path,
            purpose="Loads every setting and credential from the environment, validates them, and reports clearly when one is missing.",
            kind=FileKind.CONFIG,
            provides=["Settings", "get_settings"],
        )
    )

    # ------------------------------------------------------------------------ services
    llm_path = _module_path(package, "services", "llm_service.py")
    manifest.add_file(
        FileSpec(
            path=_module_path(package, "services", "__init__.py"),
            purpose="Service package.",
            kind=FileKind.PACKAGE,
        )
    )
    manifest.add_file(
        FileSpec(
            path=llm_path,
            purpose=(
                f"Builds the shared {manifest.provider} model client once, so every agent "
                "reuses one connection instead of re-initialising a model per step."
            ),
            kind=FileKind.SERVICE,
            provides=["build_llm", "get_llm"],
            depends_on=[settings_path],
        )
    )

    # --------------------------------------------------------------------------- tools
    tool_paths: List[str] = []
    # Telemetry is planned whether or not there are tools. The workflow module always clears
    # the record before a run and reads it afterwards, so a project with no tools still needs
    # these three functions - and reports an empty tool-call list honestly rather than being
    # unable to report one at all.
    telemetry_path = _module_path(package, "services", "telemetry.py")
    manifest.add_file(
        FileSpec(
            path=telemetry_path,
            purpose=(
                "Records every tool call - name, input and result - while the workflow "
                "runs, so what the agent actually did can be shown afterwards instead "
                "of guessed at."
            ),
            kind=FileKind.SERVICE,
            provides=["record_call", "recorded_calls", "reset_calls"],
        )
    )
    if manifest.tools:
        manifest.add_file(
            FileSpec(
                path=_module_path(package, "tools", "__init__.py"),
                purpose="Tool registry. Collects every tool so agents can be given them by name.",
                kind=FileKind.PACKAGE,
                provides=["TOOLS", "all_tools"],
                depends_on=[],
            )
        )
        for tool in manifest.tools:
            path = _module_path(package, "tools", f"{tool.name}.py")
            tool.module = path[: -len(".py")].replace("/", ".")
            tool_paths.append(path)
            depends = [telemetry_path]
            if tool.requires_env:
                depends.append(settings_path)
            manifest.add_file(
                FileSpec(
                    path=path,
                    purpose=f"{tool.label}: {tool.purpose}",
                    kind=FileKind.TOOL,
                    provides=[manifest.tool_symbol(tool)],
                    depends_on=depends,
                    owner=tool.name,
                )
            )
        registry = manifest.file(_module_path(package, "tools", "__init__.py"))
        if registry is not None:
            registry.depends_on = list(tool_paths)

    # -------------------------------------------------------------------------- state
    # LangGraph always gets its own state module, at every tier. The state type *is* the
    # framework's contract - every node function annotates it - so if it lived in the
    # workflow module the agent modules would have to import from the workflow while the
    # workflow imports the agents. That is a circular import, and it fails at run time
    # rather than at generation time, which is the worst place to find it.
    state_path = ""
    if modular or manifest.framework == "langgraph":
        manifest.add_file(
            FileSpec(
                path=_module_path(package, "models", "__init__.py"),
                purpose="Data model package.",
                kind=FileKind.PACKAGE,
            )
        )
        state_path = _module_path(package, "models", "state.py")
        workflow = manifest.primary_workflow
        fields = ", ".join(workflow.state_fields) if workflow and workflow.state_fields else "the run's inputs and outputs"
        manifest.add_file(
            FileSpec(
                path=state_path,
                purpose=f"Typed state passed between steps ({fields}), plus the result type the entry point returns.",
                kind=FileKind.STATE,
                provides=["AgentState", "RunOutput"],
            )
        )

    # ------------------------------------------------------------------------- agents
    manifest.add_file(
        FileSpec(
            path=_module_path(package, "agents", "__init__.py"),
            purpose="Agent package. Exposes one builder per agent.",
            kind=FileKind.PACKAGE,
            provides=[a.factory_name for a in manifest.agents],
        )
    )
    agent_paths: List[str] = []
    for agent in manifest.agents:
        path = _module_path(package, "agents", f"{agent.name}.py")
        agent.module = path[: -len(".py")].replace("/", ".")
        agent_paths.append(path)
        depends = [llm_path]
        # A LangGraph node function annotates the shared state type, so the agent module
        # genuinely imports it. Recording that here keeps the dependency graph honest.
        if state_path and manifest.framework == "langgraph":
            depends.append(state_path)
        depends.extend(
            _module_path(package, "tools", f"{name}.py")
            for name in agent.tools
            if manifest.tool(name) is not None
        )
        manifest.add_file(
            FileSpec(
                path=path,
                purpose=f"{agent.label} - {agent.goal}",
                kind=FileKind.AGENT,
                provides=[agent.factory_name],
                depends_on=depends,
                owner=agent.name,
            )
        )
    agents_init = manifest.file(_module_path(package, "agents", "__init__.py"))
    if agents_init is not None:
        agents_init.depends_on = list(agent_paths)

    # ----------------------------------------------------------------------- workflows
    manifest.add_file(
        FileSpec(
            path=_module_path(package, "workflows", "__init__.py"),
            purpose="Workflow package.",
            kind=FileKind.PACKAGE,
        )
    )
    workflow_paths: List[str] = []
    for workflow in manifest.workflows:
        path = _module_path(package, "workflows", f"{workflow.name}.py")
        workflow.module = path[: -len(".py")].replace("/", ".")
        workflow_paths.append(path)
        depends = list(agent_paths) + [llm_path]
        if telemetry_path:
            depends.append(telemetry_path)
        if state_path:
            depends.append(state_path)
        manifest.add_file(
            FileSpec(
                path=path,
                purpose=(
                    f"Wires the agents into the {workflow.kind.replace('_', '/')} workflow "
                    f"and exposes `run_workflow`. {workflow.description}"
                ),
                kind=FileKind.WORKFLOW,
                provides=["build_workflow", "run_workflow"],
                depends_on=depends,
                owner=workflow.name,
            )
        )

    # ------------------------------------------------------------------------- runtime
    entry_depends = list(workflow_paths) + [settings_path]
    if modular:
        manifest.add_file(
            FileSpec(
                path=_module_path(package, "runtime", "__init__.py"),
                purpose="Runtime package.",
                kind=FileKind.PACKAGE,
            )
        )
        runner_path = _module_path(package, "runtime", "runner.py")
        manifest.add_file(
            FileSpec(
                path=runner_path,
                purpose=(
                    "Runs the workflow with timing, error handling and a structured result, "
                    "so a failure comes back as a described problem rather than a traceback."
                ),
                kind=FileKind.RUNTIME,
                provides=["run"],
                depends_on=list(workflow_paths) + ([state_path] if state_path else []),
            )
        )
        entry_depends = [runner_path, settings_path]

        if manifest.tier == Tier.MODULAR and any(
            w.state_fields for w in manifest.workflows
        ):
            manifest.add_file(
                FileSpec(
                    path=_module_path(package, "memory", "__init__.py"),
                    purpose="Memory package.",
                    kind=FileKind.PACKAGE,
                )
            )
            manifest.add_file(
                FileSpec(
                    path=_module_path(package, "memory", "store.py"),
                    purpose="Keeps conversation and intermediate results across steps, in a JSON file under the project directory.",
                    kind=FileKind.MEMORY,
                    provides=["MemoryStore"],
                    depends_on=[settings_path],
                )
            )

    manifest.add_file(
        FileSpec(
            path="main.py",
            purpose="Command-line entry point. Reads a query from the arguments or stdin and prints the answer.",
            kind=FileKind.ENTRYPOINT,
            is_entrypoint=True,
            provides=["main"],
            depends_on=entry_depends,
        )
    )

    # --------------------------------------------------------------------------- tests
    if include_tests:
        manifest.add_file(
            FileSpec(
                path="tests/conftest.py",
                purpose="Test fixtures. Blocks network access and supplies a deterministic fake model, so the suite needs no API key.",
                kind=FileKind.TEST,
            )
        )
        manifest.add_file(
            FileSpec(
                path="tests/test_imports.py",
                purpose="Imports every generated module. Catches a syntax error or a missing import before anything else runs.",
                kind=FileKind.TEST,
            )
        )
        manifest.add_file(
            FileSpec(
                path="tests/test_config.py",
                purpose="Checks the settings load from the environment and that a missing credential is reported clearly rather than crashing.",
                kind=FileKind.TEST,
                depends_on=[settings_path],
            )
        )
        manifest.add_file(
            FileSpec(
                path="tests/test_agents.py",
                purpose=f"Builds each of the {len(manifest.agents)} agents offline and checks its role, goal and tools are set.",
                kind=FileKind.TEST,
                depends_on=list(agent_paths),
            )
        )
        if manifest.tools:
            manifest.add_file(
                FileSpec(
                    path="tests/test_tools.py",
                    purpose=f"Calls each of the {len(manifest.tools)} tools with sample input and checks it returns a string.",
                    kind=FileKind.TEST,
                    depends_on=list(tool_paths),
                )
            )
        manifest.add_file(
            FileSpec(
                path="tests/test_workflow.py",
                purpose="Constructs the workflow and checks its shape matches the plan: the right steps, in the right order.",
                kind=FileKind.TEST,
                depends_on=list(workflow_paths),
            )
        )


def _layout_shared(manifest: ProjectManifest, include_tests: bool) -> None:
    """Files every project gets, whatever the tier."""
    manifest.add_file(
        FileSpec(
            path="requirements.txt",
            purpose="Every package this project needs, so it can be installed without guesswork.",
            kind=FileKind.META,
        )
    )
    manifest.add_file(
        FileSpec(
            path=".env.example",
            purpose="Every environment variable the project reads, with a comment and no real values. Copy to `.env` and fill in.",
            kind=FileKind.META,
        )
    )
    manifest.add_file(
        FileSpec(
            path="README.md",
            purpose="What this project is, how it works, how to install and run it, and what each file does.",
            kind=FileKind.DOC,
        )
    )
    if include_tests:
        manifest.add_file(
            FileSpec(
                path="pytest.ini",
                purpose="Test configuration: markers and a per-test timeout, so a test that reaches for a real model fails by name instead of hanging.",
                kind=FileKind.META,
            )
        )
    # Planned for every project rather than only on request, because the Docker export option
    # is offered for every project. An export option that has to generate a missing file at
    # download time is an option that can fail at download time.
    manifest.add_file(
        FileSpec(
            path="Dockerfile",
            purpose="Container image for this project: installs the requirements and runs the entry point as a non-root user, with no credential baked in.",
            kind=FileKind.META,
        )
    )
    manifest.add_file(
        FileSpec(
            path=".dockerignore",
            purpose="Keeps `.env`, caches and virtual environments out of the image, so a build cannot accidentally copy a real credential into a layer.",
            kind=FileKind.META,
        )
    )


# ================================================================================== plan
def _plan_env_vars(manifest: ProjectManifest, provider_name: str) -> None:
    """
    Record the environment variables the generated project reads.

    Only names and descriptions, never values. This list is the single source for both
    ``.env.example`` and the settings module, which is what stops the documentation from
    drifting away from the code that reads it - and it is why no credential can end up
    baked into a generated file.
    """
    try:
        spec = get_provider(provider_name)
    except Exception:  # noqa: BLE001 - an unknown provider must not break planning
        spec = None

    if spec is not None:
        for index, env_name in enumerate(getattr(spec, "credential_env", ()) or ()):
            manifest.add_env_var(
                EnvVarSpec(
                    name=env_name,
                    purpose=f"Credential for {getattr(spec, 'label', provider_name)}.",
                    required=index == 0,
                )
            )

    if manifest.model:
        manifest.add_env_var(
            EnvVarSpec(
                name="AGENT_MODEL",
                purpose="Overrides the model id the agents use.",
                required=False,
                example=manifest.model,
            )
        )

    for tool in manifest.tools:
        for env_name in tool.requires_env:
            manifest.add_env_var(
                EnvVarSpec(
                    name=env_name,
                    purpose=f"Needed by the {tool.label} tool.",
                    required=False,
                )
            )


def _plan_dependencies(manifest: ProjectManifest, framework: str, provider: str, include_tests: bool) -> None:
    """Resolve pip requirements from the existing dependency registry."""
    try:
        requirements = requirements_for(framework, provider, include_tests=include_tests)
    except Exception:  # noqa: BLE001
        requirements = []
    for requirement in requirements:
        manifest.add_dependency(getattr(requirement, "spec", "") or "")
    for tool in manifest.tools:
        for extra in tool.dependencies:
            manifest.add_dependency(extra)


def _derive_capabilities(manifest: ProjectManifest, analysis: RequirementAnalysis) -> List[str]:
    """
    Plain-language statements of what the project can do.

    Every entry traces to something in the plan - an agent goal, a tool, a workflow
    property. Nothing is added because it would look good on a card: a capability the code
    does not have is the single most misleading thing this UI could show.
    """
    capabilities: List[str] = []

    for goal in analysis.goals[:4]:
        text = str(goal).strip()
        if text:
            capabilities.append(text)

    if not capabilities:
        for agent in manifest.agents[:4]:
            if agent.goal:
                capabilities.append(agent.goal)

    workflow = manifest.primary_workflow
    if workflow is not None:
        steps = len(workflow.agent_nodes())
        if steps > 1:
            capabilities.append(
                f"Runs {steps} steps in order, passing each result to the next."
            )
        if workflow.has_branching:
            capabilities.append("Chooses a different path depending on what it finds.")
        if workflow.kind == "reason_act":
            capabilities.append("Decides for itself when to use a tool and when to answer.")
        if workflow.kind == "hierarchical":
            capabilities.append("Delegates work to specialists and consolidates their answers.")

    if manifest.tools:
        capabilities.append(
            "Can call "
            + ", ".join(t.label for t in manifest.tools[:5])
            + ("." if len(manifest.tools) <= 5 else f", and {len(manifest.tools) - 5} more.")
            + " (Tool bodies are stubs you fill in.)"
        )

    return list(dict.fromkeys(c for c in capabilities if c))


def plan_architecture(
    analysis: RequirementAnalysis,
    choice: FrameworkChoice,
    provider: str,
    model: Optional[str] = None,
    *,
    config: Optional[Mapping[str, Any]] = None,
    include_tests: bool = True,
    project_name: str = "",
) -> ProjectManifest:
    """
    Turn an analysis, a framework choice and an agent config into a full project plan.

    Args:
        analysis: The structural reading of the requirement.
        choice: The selected framework.
        provider: LLM provider the *generated code* will call at run time.
        model: Optional explicit model id for the generated code.
        config: The model-authored agent configuration. When omitted, the plan is built
            from the analysis alone - which is what makes planning testable with no model,
            no key and no network.
        include_tests: Whether to plan a test suite. On by default: an unverifiable
            project is the problem this whole pipeline exists to solve.
        project_name: Override the derived slug.

    Returns:
        A :class:`ProjectManifest` with every agent, tool, workflow node, edge and file
        named, and every file carrying the reason it exists.
    """
    framework = choice.framework
    spec = get_provider(provider)
    runtime_model = resolve_runtime_model(spec.name, model) or ""
    data: Mapping[str, Any] = config or {}

    agents = _plan_agents(data, analysis, runtime_model)
    tools = _plan_tools(data, analysis, agents)

    # Only keep tool references that survived planning, so no agent is given a tool that
    # was dropped as a duplicate - that would emit a NameError into the generated code.
    tool_names = {t.name for t in tools}
    for agent in agents:
        agent.tools = [name for name in agent.tools if name in tool_names]

    # A ReAct agent with tools that were never attached to it still needs them: the whole
    # point of the loop is tool selection.
    if framework in ("react", "react-lcel") and tools and agents and not agents[0].tools:
        agents[0].tools = [t.name for t in tools]

    workflow = _plan_workflow(framework, data, agents, tools, analysis)

    # Workflow nodes carry inputs/outputs; copy them onto the agents so requirement
    # tracing can see that information actually moves between steps.
    by_agent = {a.name: a for a in agents}
    for node in workflow.agent_nodes():
        owner = by_agent.get(node.agent or "")
        if owner is None:
            continue
        if not owner.inputs and node.inputs:
            owner.inputs = node.inputs
        if not owner.outputs and node.outputs:
            owner.outputs = node.outputs

    tier = decide_tier(len(agents), len(tools), analysis)

    manifest = ProjectManifest(
        project_name=project_name or derive_project_name(analysis),
        framework=framework,
        provider=spec.name,
        model=runtime_model,
        requirement=analysis.requirement,
        description=analysis.summary or analysis.requirement,
        tier=tier,
        package="" if tier == Tier.SIMPLE else "app",
        agents=agents,
        tools=tools,
        workflows=[workflow],
    )

    # Assign each agent the workflow's ordering, so downstream consumers see agents in the
    # order the work actually happens rather than the order the model happened to list them.
    ordered = workflow.ordered_agent_names()
    if ordered:
        by_name = {a.name: a for a in manifest.agents}
        manifest.agents = [by_name[name] for name in ordered if name in by_name] + [
            a for a in manifest.agents if a.name not in ordered
        ]

    for index, agent in enumerate(manifest.agents):
        if not agent.inputs:
            agent.inputs = "The user's query." if index == 0 else "The previous step's output."
        if not agent.outputs:
            agent.outputs = "Its contribution to the final answer."
        if not agent.responsibilities:
            agent.responsibilities = [
                node.description or node.label
                for node in workflow.nodes
                if node.agent == agent.name and (node.description or node.label)
            ]

    if tier == Tier.SIMPLE:
        _layout_simple(manifest, include_tests)
    else:
        _layout_package(manifest, include_tests)
    _layout_shared(manifest, include_tests)

    _plan_dependencies(manifest, framework, spec.name, include_tests)
    _plan_env_vars(manifest, spec.name)
    manifest.capabilities = _derive_capabilities(manifest, analysis)

    label = FRAMEWORK_LABELS.get(framework, framework)
    manifest.notes = [
        f"Framework: {label}. {choice.reason}" if choice.reason else f"Framework: {label}.",
        tier_reason(tier, len(manifest.agents), len(manifest.tools), analysis),
    ]
    if manifest.tools:
        manifest.notes.append(
            "Tool bodies are generated as working stubs with real signatures. They return a "
            "clearly-labelled placeholder rather than pretending to have done the work."
        )
    for risk in analysis.risks[:3]:
        text = str(risk).strip()
        if text:
            manifest.notes.append(f"Risk noted during analysis: {text}")

    return manifest
