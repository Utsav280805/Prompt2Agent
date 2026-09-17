# multi_agent_generator/emit/agno_emitter.py
"""
Agno emitter.

Two things here follow Agno's real API rather than the shape the other frameworks use.

Tools are plain Python functions. Agno builds a tool's schema from the function signature and
its docstring, so a tool is a function with a typed parameter and a written docstring - not a
class. The old generator emitted ``tools=[]`` for every agent with a comment saying the config
only carried names, which meant an Agno project could never call a tool at all. The manifest
carries real tool specs, so the functions are emitted and passed.

The model is built per agent rather than shared. Agno attaches per-agent state to the model
object, so handing one instance to several agents and to the team leader couples them. That is
the one place where this emitter deliberately gives up the caching the other frameworks get,
and it is a correctness trade, not an oversight.

Sequential and hierarchical plans produce genuinely different code: a chain of ``agent.run``
calls that passes each result forward, or an ``agno.team.Team`` in coordinate mode. The
workflow diagram is drawn from the same plan, so the picture and the code cannot disagree.
"""
from __future__ import annotations

from typing import List

from ..core.manifest import AgentSpec, ToolSpec, WorkflowSpec
from .base import Emitter, Fragment, docstring, indent, literal

__all__ = ["AgnoEmitter"]


def _tool_function(tool: ToolSpec) -> str:
    """
    A tool as a plain function.

    The docstring matters: Agno sends it to the model as the tool description, so a tool with
    no docstring is a tool the model cannot choose sensibly.
    """
    description = tool.purpose or f"{tool.label} capability."
    arg_lines = "\n".join(
        f"        {name}: {purpose}" for name, purpose in (tool.parameters or {}).items()
    ) or "        query: What to act on, as free text."
    return (
        f"def {tool.function_name}(query: str) -> str:\n"
        f'    """{tool.label} - {description}\n'
        "\n"
        "    Agno reads this docstring as the tool description the model sees, so it\n"
        "    describes what the tool does rather than how it is written.\n"
        "\n"
        "    Args:\n"
        f"{arg_lines}\n"
        "\n"
        "    Returns:\n"
        "        The tool's result as text.\n"
        '    """\n'
        "    # TODO: replace this with a real implementation.\n"
        f"    result = {literal('[' + tool.name + ' is not implemented yet] requested: ')} + str(query)\n"
        f"    record_call({literal(tool.name)}, {{\"query\": query}}, result)\n"
        "    return result\n"
    )


def _agent_builder(emitter: "AgnoEmitter", agent: AgentSpec) -> str:
    instructions = emitter.instructions_for(agent) or f"You are {agent.role or agent.label}."
    return (
        f"def {agent.factory_name}():\n"
        + indent(
            docstring(
                f"Build the {agent.label} agent.",
                [
                    f"Role: {agent.role}." if agent.role else "",
                    (
                        "Tools: " + ", ".join(agent.tools) + "."
                        if agent.tools
                        else "This agent has no tools and answers directly."
                    ),
                    "A fresh model instance is built for this agent. Agno keeps per-agent "
                    "state on the model object, so sharing one instance across agents couples "
                    "them together in ways that are hard to see later.",
                ],
            )
        )
        + "\n"
        "    return Agent(\n"
        f"        name={literal(agent.label)},\n"
        "        model=build_llm(),\n"
        f"        role={literal(agent.role or agent.label)},\n"
        f"        instructions={literal(instructions)},\n"
        f"        tools={emitter.agent_tools_expression(agent)},\n"
        # markdown=False: the output is shown in a web playground that renders it itself, and
        # markdown wrappers leak into the response text when it is displayed as plain text.
        "        markdown=False,\n"
        "    )\n"
    )


class AgnoEmitter(Emitter):
    """Agno agents, run either as an ordered chain or as a coordinating team."""

    key = "agno"
    label = "Agno"
    workflow_imports = (
        "import time",
        "from typing import Any, Dict, List",
    )
    agent_imports = ("from agno.agent import Agent",)

    # Agno takes the callable itself, not an instance of a tool class.
    def tool_expression(self, tool: ToolSpec) -> str:
        return tool.function_name

    def tool_fragment(self, tool: ToolSpec) -> Fragment:
        return Fragment(body=_tool_function(tool), provides=[tool.function_name])

    def agent_fragment(self, agent: AgentSpec) -> Fragment:
        return Fragment(
            body=_agent_builder(self, agent),
            imports=list(self.agent_imports),
            provides=[agent.factory_name],
        )

    # --------------------------------------------------------------------------- workflow
    def _steps_table(self, workflow: WorkflowSpec) -> str:
        rows: List[str] = []
        for node in workflow.agent_nodes():
            rows.append(
                "    {\n"
                f"        \"name\": {literal(node.id)},\n"
                f"        \"agent\": {literal(node.agent)},\n"
                f"        \"description\": {literal(node.description or 'Handle the request.')},\n"
                f"        \"expected\": {literal(node.outputs or '')},\n"
                "    },"
            )
        return "\n".join(rows)

    def _builders_table(self) -> str:
        return "\n".join(
            f"    {literal(a.name)}: {a.factory_name}," for a in self.manifest.agents
        )

    def _sequential_run(self) -> str:
        return (
            "def run_workflow(query: str) -> Dict[str, Any]:\n"
            + indent(
                docstring(
                    "Run each step in order, passing every result forward.",
                    [
                        "Each step gets the original request plus what the previous steps "
                        "produced, which is what makes this a pipeline rather than several "
                        "unrelated calls.",
                        "Returns `output` (the last step's result), `steps` (one entry per "
                        "step that ran), `tool_calls` and `duration_s`.",
                    ],
                )
            )
            + "\n"
            "    reset_calls()\n"
            "    started = time.monotonic()\n"
            "    steps: List[Dict[str, Any]] = []\n"
            "    context = \"\"\n"
            "    output = \"\"\n"
            "\n"
            "    for step in STEPS:\n"
            "        builder = AGENT_BUILDERS.get(step[\"agent\"])\n"
            "        if builder is None:\n"
            "            # The plan named an agent this module does not build. Skip it and say\n"
            "            # so in the steps, rather than silently running the wrong agent.\n"
            "            steps.append({\n"
            "                \"name\": step[\"name\"],\n"
            "                \"agent\": step[\"agent\"],\n"
            "                \"status\": \"skipped\",\n"
            "                \"output\": f\"No builder is defined for agent {step['agent']}.\",\n"
            "            })\n"
            "            continue\n"
            "\n"
            "        prompt = step[\"description\"] + \"\\n\\nUser request: \" + str(query)\n"
            "        if step[\"expected\"]:\n"
            "            prompt += \"\\n\\nExpected output: \" + step[\"expected\"]\n"
            "        if context:\n"
            "            prompt += \"\\n\\nPrevious step's output:\\n\" + context\n"
            "\n"
            "        step_started = time.monotonic()\n"
            "        response = builder().run(prompt)\n"
            "        text = str(getattr(response, \"content\", response) or \"\")\n"
            "        steps.append({\n"
            "            \"name\": step[\"name\"],\n"
            "            \"agent\": step[\"agent\"],\n"
            "            \"status\": \"completed\",\n"
            "            \"output\": text,\n"
            "            \"duration_s\": round(time.monotonic() - step_started, 3),\n"
            "        })\n"
            "        context = text\n"
            "        output = text\n"
            "\n"
            "    return {\n"
            "        \"output\": output,\n"
            "        \"steps\": steps,\n"
            "        \"tool_calls\": recorded_calls(),\n"
            "        \"duration_s\": round(time.monotonic() - started, 3),\n"
            "        \"framework\": \"agno\",\n"
            "    }\n"
        )

    def _team_run(self, manager: AgentSpec, specialists: List[AgentSpec]) -> str:
        members = ", ".join(f"{a.factory_name}()" for a in specialists)
        instructions = self.instructions_for(manager) or "Coordinate the members to answer."
        return (
            "def build_workflow():\n"
            + indent(
                docstring(
                    "Build the coordinating team.",
                    [
                        f"{manager.label} leads and delegates to "
                        f"{len(specialists)} member"
                        f"{'s' if len(specialists) != 1 else ''}.",
                        "Coordinate mode is Agno's delegating mode: the leader decides which "
                        "member handles what, which is the behaviour the plan asked for.",
                    ],
                )
            )
            + "\n"
            "    return Team(\n"
            f"        name={literal(self.manifest.project_name)},\n"
            "        mode=\"coordinate\",\n"
            "        model=build_llm(),\n"
            f"        members=[{members}],\n"
            "        instructions=[\n"
            f"            {literal(instructions)},\n"
            "            \"Use the user's request as the shared context for every member.\",\n"
            "        ],\n"
            "        markdown=False,\n"
            "        show_members_responses=True,\n"
            "    )\n\n\n"
            "def run_workflow(query: str) -> Dict[str, Any]:\n"
            + indent(
                docstring(
                    "Run the team for one request.",
                    [
                        "Member results are read from the response only if Agno exposed them. "
                        "They are read with `getattr` rather than assumed, so a version that "
                        "does not carry them produces one step for the team instead of a "
                        "crash - and never an invented breakdown.",
                    ],
                )
            )
            + "\n"
            "    reset_calls()\n"
            "    started = time.monotonic()\n"
            "    response = build_workflow().run(str(query))\n"
            "    duration = round(time.monotonic() - started, 3)\n"
            "    output = str(getattr(response, \"content\", response) or \"\")\n"
            "\n"
            "    steps: List[Dict[str, Any]] = []\n"
            "    for member in (getattr(response, \"member_responses\", None) or []):\n"
            "        steps.append({\n"
            "            \"name\": str(getattr(member, \"agent_id\", None)\n"
            "                          or getattr(member, \"name\", \"member\")),\n"
            "            \"agent\": str(getattr(member, \"agent_id\", None)\n"
            "                           or getattr(member, \"name\", \"\")),\n"
            "            \"status\": \"completed\",\n"
            "            \"output\": str(getattr(member, \"content\", \"\") or \"\"),\n"
            "        })\n"
            "    if not steps:\n"
            f"        steps.append({{\"name\": \"team\", \"agent\": {literal(manager.name)},\n"
            "                      \"status\": \"completed\", \"output\": output})\n"
            "\n"
            "    return {\n"
            "        \"output\": output,\n"
            "        \"steps\": steps,\n"
            "        \"tool_calls\": recorded_calls(),\n"
            "        \"duration_s\": duration,\n"
            "        \"framework\": \"agno\",\n"
            "    }\n"
        )

    def workflow_fragment(self, workflow: WorkflowSpec) -> Fragment:
        blocks: List[str] = [
            "#: The steps this workflow runs, in plan order.\n"
            f"STEPS: List[Dict[str, str]] = [\n{self._steps_table(workflow)}\n]",
            "#: Agent name -> its builder.\n"
            f"AGENT_BUILDERS = {{\n{self._builders_table()}\n}}",
        ]
        imports = list(self.workflow_imports)

        if workflow.kind == "hierarchical" and len(self.manifest.agents) >= 2:
            manager, specialists = self.manifest.agents[0], list(self.manifest.agents[1:])
            imports.append("from agno.team import Team")
            blocks.append(self._team_run(manager, specialists))
            provides = ["STEPS", "AGENT_BUILDERS", "build_workflow", "run_workflow"]
        else:
            blocks.append(
                "def build_workflow():\n"
                + indent(
                    docstring(
                        "Build every agent this workflow uses, keyed by name.",
                        [
                            "Exposed so a test can construct the whole project offline without "
                            "running it, which is how `tests/test_workflow.py` checks the shape "
                            "of the plan against the code.",
                        ],
                    )
                )
                + "\n"
                "    return {name: builder() for name, builder in AGENT_BUILDERS.items()}\n"
            )
            blocks.append(self._sequential_run())
            provides = ["STEPS", "AGENT_BUILDERS", "build_workflow", "run_workflow"]

        return Fragment(body="\n\n\n".join(blocks), imports=imports, provides=provides)
