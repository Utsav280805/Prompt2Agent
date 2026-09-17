# multi_agent_generator/emit/langgraph_emitter.py
"""
LangGraph emitter.

LangGraph is the framework to pick when the path through the work is not a straight line, so
this emitter's job is to produce a graph that genuinely has the shape the plan describes -
including its branches. The old single-file generator carried ``condition`` on every edge and
never read it, so a requirement that needed branching came out as a straight chain. Here a
conditional edge in the plan becomes an ``add_conditional_edges`` call with a router function
that names the branches it can take.

Where the plan asks for a decision the analysis did not supply logic for, the generated router
says so in its docstring and falls through to a documented default. That is the honest shape: a
real mechanism, a stated gap, and a graph that stays connected. Silently dropping the branch
would hide the gap; inventing a heuristic would be worse, because it would look implemented.

Three details follow the framework's actual current API rather than a remembered one:
``add_messages`` as the state reducer, ``bind_tools`` guarded by ``getattr`` because not every
chat model supports it, and one node function per graph node - not per agent - so an agent that
appears at two points in the graph is invoked twice, as the picture shows.
"""
from __future__ import annotations

from typing import Dict, List

from ..core.manifest import AgentSpec, NodeKind, ToolSpec, WorkflowSpec
from .base import Emitter, Fragment, docstring, indent, literal

__all__ = ["LangGraphEmitter"]


def _tool_class(tool: ToolSpec) -> str:
    description = tool.purpose or f"{tool.label} capability."
    return (
        f"class {tool.class_name}(BaseTool):\n"
        + indent(
            docstring(
                f"{tool.label} - {description}",
                [
                    "A stub with the real signature and description, so the model can select "
                    "it and the graph runs. Replace the body of `_run` with a real "
                    "implementation; the placeholder it returns states plainly that no real "
                    "work was done.",
                ],
            )
        )
        + "\n\n"
        # Annotations are required: BaseTool is a pydantic v2 model and an unannotated class
        # attribute raises PydanticUserError at import time.
        f"    name: str = {literal(tool.name)}\n"
        f"    description: str = {literal(description)}\n\n"
        "    def _run(self, query: str) -> str:\n"
        "        # TODO: replace this with a real implementation.\n"
        f"        result = {literal('[' + tool.name + ' is not implemented yet] requested: ')} + str(query)\n"
        f"        record_call({literal(tool.name)}, {{\"query\": query}}, result)\n"
        "        return result\n\n"
        "    async def _arun(self, query: str) -> str:\n"
        "        return self._run(query)\n"
    )


def _agent_builder(emitter: Emitter, agent: AgentSpec) -> str:
    """
    A builder returning a LangGraph node function.

    The model client and the tool bindings are resolved once, outside the returned closure, so
    a graph that visits this node repeatedly does not rebuild them on every visit.
    """
    instructions = emitter.instructions_for(agent) or f"You are {agent.role or agent.label}."
    return (
        f"def {agent.factory_name}(step_name: str = {literal(agent.name)}):\n"
        + indent(
            docstring(
                f"Build the {agent.label} graph node.",
                [
                    f"Role: {agent.role}." if agent.role else "",
                    (
                        "Tools: " + ", ".join(agent.tools) + "."
                        if agent.tools
                        else "This node uses no tools."
                    ),
                    "`step_name` is the graph node this function is registered under. It is "
                    "recorded with the node's output so a run can be read back step by step, "
                    "and it is a parameter because one agent may appear at more than one "
                    "point in the graph.",
                ],
            )
        )
        + "\n"
        f"    instructions = {literal(instructions)}\n"
        f"    tools = {emitter.agent_tools_expression(agent)}\n"
        "    client = get_llm()\n"
        "    # Not every chat model implements tool calling. Bind when it is available and\n"
        "    # fall back to the plain client when it is not, rather than failing at import.\n"
        "    bind = getattr(client, \"bind_tools\", None)\n"
        "    model = bind(tools) if tools and callable(bind) else client\n"
        "\n"
        "    def node(state: AgentState) -> Dict[str, Any]:\n"
        f"        \"\"\"Run the {agent.label} step against the current graph state.\"\"\"\n"
        "        history = list(state.get(\"messages\") or [])\n"
        "        response = model.invoke([SystemMessage(content=instructions), *history])\n"
        "        text = str(getattr(response, \"content\", response))\n"
        "        results = dict(state.get(\"results\") or {})\n"
        "        results[step_name] = text\n"
        "        return {\"messages\": [response], \"results\": results}\n"
        "\n"
        "    node.__name__ = f\"{step_name}_node\"\n"
        "    return node\n"
    )


def _state_block(workflow: WorkflowSpec) -> str:
    extra = ""
    for field in workflow.state_fields:
        if field in ("messages", "next", "results"):
            continue
        extra += f"    {field}: str\n"
    return (
        "class AgentState(TypedDict, total=False):\n"
        + indent(
            docstring(
                "The state every node reads and updates.",
                [
                    "`messages` uses LangGraph's `add_messages` reducer, so a node returning "
                    "one message appends it instead of replacing the history - which is what "
                    "lets a later node see what an earlier one said.",
                    "`results` holds one entry per completed node, keyed by node name. It is "
                    "what the run's step-by-step breakdown is built from.",
                    "`next` is how a node signals a branch: it writes the name of the node it "
                    "wants to run next, and the router reads it.",
                ],
            )
        )
        + "\n"
        "    messages: Annotated[List[BaseMessage], add_messages]\n"
        "    results: Dict[str, str]\n"
        "    next: str\n"
        + extra
    )


def _routers(workflow: WorkflowSpec) -> str:
    """
    One router per branching node.

    Emitted only where the plan actually recorded a condition, so a linear graph gets no
    routers and stays readable.
    """
    blocks: List[str] = []
    for node in workflow.nodes:
        if node.kind in (NodeKind.START, NodeKind.END):
            continue
        outgoing = [e for e in workflow.successors(node.id) if e.target]
        if len(outgoing) < 2 or not any(e.conditional for e in outgoing):
            continue

        targets = {e.target: (e.condition or "otherwise") for e in outgoing}
        default = next(
            (e.target for e in outgoing if not e.conditional),
            outgoing[-1].target,
        )
        table = "\n".join(
            f"    {literal(target)}: {literal(condition)}," for target, condition in targets.items()
        )
        conditions = [f"`{target}` when {condition}" for target, condition in targets.items()]
        blocks.append(
            f"#: Where {node.id} can go next, and the condition the plan recorded for each.\n"
            f"ROUTES_{node.id.upper()} = {{\n{table}\n}}\n\n\n"
            f"def route_{node.id}(state: AgentState) -> str:\n"
            + indent(
                docstring(
                    f"Choose the node that runs after {node.id}.",
                    [
                        "Branches from the plan: " + "; ".join(conditions) + ".",
                        "A node signals its choice by writing the target node's name into "
                        "state['next']. The decision itself is not implemented yet - nothing "
                        "in the requirement said how to make it - so an unset or unknown "
                        f"value falls through to {default}, which keeps the graph connected "
                        "rather than dead-ending mid-run.",
                    ],
                )
            )
            + "\n"
            "    choice = str(state.get(\"next\") or \"\").strip()\n"
            f"    if choice in ROUTES_{node.id.upper()}:\n"
            "        return choice\n"
            f"    return {literal(default)}\n"
        )
    return "\n\n\n".join(blocks)


class LangGraphEmitter(Emitter):
    """A compiled ``StateGraph`` whose nodes are agent functions."""

    key = "langgraph"
    label = "LangGraph"
    workflow_imports = (
        "import time",
        "from typing import Any, Dict, List",
        "from langchain_core.messages import BaseMessage, HumanMessage",
        "from langgraph.graph import END, StateGraph",
    )

    state_imports = (
        "from typing import Annotated, Any, Dict, List, TypedDict",
        "from langchain_core.messages import BaseMessage",
        "from langgraph.graph.message import add_messages",
    )

    agent_imports = (
        "from typing import Any, Dict",
        "from langchain_core.messages import SystemMessage",
    )

    def tool_fragment(self, tool: ToolSpec) -> Fragment:
        return Fragment(
            body=_tool_class(tool),
            imports=["from langchain_core.tools import BaseTool"],
            provides=[tool.class_name],
        )

    def agent_fragment(self, agent: AgentSpec) -> Fragment:
        return Fragment(
            body=_agent_builder(self, agent),
            imports=list(self.agent_imports),
            provides=[agent.factory_name],
        )

    def state_fragment(self) -> Fragment:
        workflow = self.manifest.primary_workflow
        if workflow is None:
            return Fragment()
        return Fragment(
            body=_state_block(workflow),
            imports=list(self.state_imports),
            provides=["AgentState"],
        )

    def workflow_fragment(self, workflow: WorkflowSpec) -> Fragment:
        graph_nodes = [n for n in workflow.nodes if n.kind == NodeKind.AGENT]
        node_agents: Dict[str, str] = {
            n.id: (n.agent or (self.manifest.agents[0].name if self.manifest.agents else ""))
            for n in graph_nodes
        }

        steps_table = "\n".join(
            "    {\n"
            f"        \"name\": {literal(n.id)},\n"
            f"        \"agent\": {literal(node_agents[n.id])},\n"
            f"        \"description\": {literal(n.description or 'Handle the request.')},\n"
            "    },"
            for n in graph_nodes
        )
        builders = "\n".join(
            f"    {literal(a.name)}: {a.factory_name},"
            for a in self.manifest.agents
        )
        node_map = "\n".join(
            f"    {literal(node_id)}: {literal(agent_name)},"
            for node_id, agent_name in node_agents.items()
        )

        # ------------------------------------------------------------------- edge wiring
        edge_lines: List[str] = []
        routed: set[str] = set()
        for node in graph_nodes:
            outgoing = workflow.successors(node.id)
            if len(outgoing) >= 2 and any(e.conditional for e in outgoing):
                mapping = ", ".join(
                    f"{literal(e.target)}: "
                    + ("END" if e.target == "end" else literal(e.target))
                    for e in outgoing
                )
                edge_lines.append(
                    f"    graph.add_conditional_edges({literal(node.id)}, route_{node.id}, "
                    f"{{{mapping}}})"
                )
                routed.add(node.id)
                continue
            for edge in outgoing:
                target = "END" if edge.target == "end" else literal(edge.target)
                edge_lines.append(f"    graph.add_edge({literal(node.id)}, {target})")

        for node in graph_nodes:
            if node.id not in routed and not workflow.successors(node.id):
                edge_lines.append(f"    graph.add_edge({literal(node.id)}, END)")

        entry = workflow.entry if workflow.entry in node_agents else (
            graph_nodes[0].id if graph_nodes else ""
        )

        build = (
            "def build_workflow():\n"
            + indent(
                docstring(
                    "Compile the state graph.",
                    [
                        workflow.description,
                        "One node is registered per graph node rather than per agent, so an "
                        "agent that appears twice in the plan runs twice - which is what the "
                        "workflow diagram shows.",
                    ],
                )
            )
            + "\n"
            "    graph = StateGraph(AgentState)\n"
            "    for node_name, agent_name in NODE_AGENTS.items():\n"
            "        builder = AGENT_BUILDERS.get(agent_name) or next(iter(AGENT_BUILDERS.values()))\n"
            "        graph.add_node(node_name, builder(step_name=node_name))\n"
            "\n"
            + "\n".join(edge_lines)
            + "\n\n"
            + (f"    graph.set_entry_point({literal(entry)})\n" if entry else "")
            + "    return graph.compile()\n"
        )

        run = (
            "def run_workflow(query: str) -> Dict[str, Any]:\n"
            + indent(
                docstring(
                    "Run the graph for one query and return the result with its detail.",
                    [
                        "Returns `output` (the last message's content), `steps` (one entry per "
                        "node that ran, in order), `tool_calls` and `duration_s`.",
                        "Nothing is caught here: a provider or graph failure propagates, "
                        "because reporting a swallowed exception as an empty answer would be "
                        "worse than reporting the failure.",
                    ],
                )
            )
            + "\n"
            "    reset_calls()\n"
            "    started = time.monotonic()\n"
            "    app = build_workflow()\n"
            "    final = app.invoke({\n"
            "        \"messages\": [HumanMessage(content=str(query))],\n"
            "        \"results\": {},\n"
            "        \"next\": \"\",\n"
            "    })\n"
            "    duration = round(time.monotonic() - started, 3)\n"
            "\n"
            "    messages = list(final.get(\"messages\") or [])\n"
            "    output = str(getattr(messages[-1], \"content\", \"\")) if messages else \"\"\n"
            "    results = final.get(\"results\") or {}\n"
            "    steps = [\n"
            "        {\n"
            "            \"name\": name,\n"
            "            \"agent\": NODE_AGENTS.get(name, \"\"),\n"
            "            \"output\": value,\n"
            "        }\n"
            "        for name, value in results.items()\n"
            "    ]\n"
            "\n"
            "    return {\n"
            "        \"output\": output,\n"
            "        \"steps\": steps,\n"
            "        \"tool_calls\": recorded_calls(),\n"
            "        \"duration_s\": duration,\n"
            "        \"framework\": \"langgraph\",\n"
            "    }\n"
        )

        blocks = [
            "#: The graph's nodes, in plan order.\n"
            f"STEPS: List[Dict[str, str]] = [\n{steps_table}\n]",
            "#: Agent name -> its node builder.\n" f"AGENT_BUILDERS = {{\n{builders}\n}}",
            "#: Graph node name -> the agent that runs it.\n" f"NODE_AGENTS = {{\n{node_map}\n}}",
        ]
        routers = _routers(workflow)
        if routers:
            blocks.append(routers)
        blocks.extend([build, run])

        return Fragment(
            body="\n\n\n".join(blocks),
            imports=list(self.workflow_imports),
            provides=["STEPS", "AGENT_BUILDERS", "NODE_AGENTS", "build_workflow", "run_workflow"],
        )
