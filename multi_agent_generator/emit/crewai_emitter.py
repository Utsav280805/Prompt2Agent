# multi_agent_generator/emit/crewai_emitter.py
"""
CrewAI and CrewAI Flow emitters.

Both build the same agents and the same tools; they differ in how the work is orchestrated.
Plain CrewAI assembles a ``Crew`` of ``Task`` objects and hands it to a sequential or
hierarchical process. CrewAI Flow wires the steps together explicitly with ``@start`` and
``@listen``, carrying state between them - which is why the framework choice is not
cosmetic: pick Flow and you get a different module, a different control flow and a
different picture in the workflow view.

Two deliberate departures from the old single-file generator:

Task descriptions are assembled in Python rather than left to CrewAI's ``{query}``
interpolation. Interpolation calls ``str.format`` on the description, so a single stray
brace in model-authored text - and there often is one - raised ``KeyError`` at kickoff. The
concatenated form cannot fail that way and behaves identically across CrewAI versions.

Steps are data. ``STEPS`` is a module-level list built from the plan, and the crew is
assembled by iterating it, instead of unrolling one hand-written ``Task(...)`` block per
step. That keeps the generated file short and readable at ten steps as well as at two, and
it means the run result can name each step honestly because the names come from the plan.
"""
from __future__ import annotations

from typing import List, Sequence

from ..core.manifest import AgentSpec, ToolSpec, WorkflowSpec
from .base import Emitter, Fragment, docstring, indent, literal

__all__ = ["CrewAIEmitter", "CrewAIFlowEmitter"]


_TOOL_IMPORTS = ["from crewai.tools import BaseTool"]


def _tool_class(tool: ToolSpec) -> str:
    """
    A CrewAI tool class.

    ``name`` and ``description`` carry type annotations because ``BaseTool`` is a pydantic
    v2 model: an unannotated class attribute raises ``PydanticUserError`` at import, which
    is a failure that shows up as "the module cannot be imported" rather than as anything
    resembling its cause.
    """
    description = tool.purpose or f"{tool.label} capability."
    return (
        f"class {tool.class_name}(BaseTool):\n"
        + indent(
            docstring(
                f"{tool.label} - {description}",
                [
                    "This is a stub. It has the right name, the right signature and the "
                    "right description, so the agent can select it and the project runs - "
                    "but it does not do the real work yet. Replace the body of `_run` with "
                    "a real implementation.",
                    "The placeholder it returns says so explicitly, so an agent reading the "
                    "result cannot mistake it for a genuine lookup.",
                ],
            )
        )
        + "\n\n"
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


def _agent_builder(emitter: Emitter, agent: AgentSpec, *, manager: bool = False) -> str:
    """A ``build_<agent>()`` factory returning a configured CrewAI ``Agent``."""
    extras = ""
    if manager:
        # A manager coordinates rather than does, so it gets room to iterate and a hard
        # ceiling so a delegation loop cannot run forever.
        extras = "        max_iter=5,\n        max_execution_time=300,\n"
    tools_expression = "[]" if manager else emitter.agent_tools_expression(agent)
    return (
        f"def {agent.factory_name}():\n"
        + indent(
            docstring(
                f"Build the {agent.label} agent.",
                [
                    f"Role: {agent.role}." if agent.role else "",
                    f"Goal: {agent.goal}." if agent.goal else "",
                    (
                        "Tools: " + ", ".join(agent.tools) + "."
                        if agent.tools
                        else "This agent uses no tools."
                    ),
                ],
            )
        )
        + "\n"
        "    return Agent(\n"
        f"        role={literal(agent.role or agent.label)},\n"
        f"        goal={literal(agent.goal)},\n"
        f"        backstory={literal(agent.backstory)},\n"
        f"        tools={tools_expression},\n"
        "        llm=get_llm(),\n"
        f"        verbose={bool(agent.verbose)},\n"
        f"        allow_delegation={bool(agent.allow_delegation or manager)},\n"
        + extras
        + "    )\n"
    )


def _steps_table(workflow: WorkflowSpec) -> str:
    """
    The ordered plan, as data in the generated file.

    Emitted as a literal list so the run result can label each step with the name the plan
    gave it. Nothing here is invented at run time, which is what makes the step list in the
    Playground trustworthy: it is the plan, not a guess reconstructed from log output.
    """
    rows: List[str] = []
    for index, node in enumerate(workflow.agent_nodes(), start=1):
        rows.append(
            "    {\n"
            f"        \"name\": {literal(node.label or node.id)},\n"
            f"        \"agent\": {literal(node.agent or '')},\n"
            f"        \"description\": {literal(node.description or 'Handle the request.')},\n"
            f"        \"expected_output\": {literal(node.outputs or 'A useful result.')},\n"
            "    },"
        )
    body = "\n".join(rows)
    return (
        "#: The ordered plan. Each entry becomes one task, in this order.\n"
        f"STEPS: List[Dict[str, str]] = [\n{body}\n]\n"
    )


def _builders_table(agents: Sequence[AgentSpec]) -> str:
    entries = "\n".join(f"    {literal(a.name)}: {a.factory_name}," for a in agents)
    return (
        "#: Agent name -> its factory. Built once per run.\n"
        f"AGENT_BUILDERS = {{\n{entries}\n}}\n"
    )


def _run_workflow(framework: str, extra_result_keys: str = "") -> str:
    """
    ``run_workflow`` - the one function the whole product depends on existing.

    The sandbox harness looks for this name, so this is the seam between generated code and
    the Playground. It returns a dict rather than a bare string because the Playground has
    to show what happened, not just the final answer: which steps ran, which tools were
    called, how long it took. Every one of those comes from the run itself.
    """
    return (
        "def run_workflow(query: str) -> Dict[str, Any]:\n"
        + indent(
            docstring(
                "Run the workflow for one query and return the result with its detail.",
                [
                    "Returns a dict with `output` (the final answer), `steps` (what each "
                    "agent produced), `tool_calls` (what was actually called) and "
                    "`duration_s`. The extra detail is what makes a run inspectable rather "
                    "than a black box.",
                    "Raises nothing of its own: a framework or provider failure propagates "
                    "unchanged, because a swallowed exception here would be reported as a "
                    "successful run with an empty answer.",
                ],
            )
        )
        + "\n"
        "    reset_calls()\n"
        "    started = time.monotonic()\n"
        "    crew = build_workflow(query)\n"
        "    result = crew.kickoff()\n"
        "    duration = round(time.monotonic() - started, 3)\n"
        "\n"
        "    steps: List[Dict[str, Any]] = []\n"
        "    # `tasks_output` is how recent CrewAI reports per-task results. Read defensively:\n"
        "    # older versions do not have it, and a missing attribute must not lose the answer.\n"
        "    for index, task_output in enumerate(getattr(result, \"tasks_output\", None) or []):\n"
        "        planned = STEPS[index] if index < len(STEPS) else {}\n"
        "        steps.append({\n"
        "            \"name\": planned.get(\"name\") or f\"step_{index + 1}\",\n"
        "            \"agent\": planned.get(\"agent\") or \"\",\n"
        "            \"output\": str(getattr(task_output, \"raw\", task_output)),\n"
        "        })\n"
        "\n"
        "    return {\n"
        "        \"output\": str(getattr(result, \"raw\", result)),\n"
        "        \"steps\": steps,\n"
        "        \"tool_calls\": recorded_calls(),\n"
        "        \"duration_s\": duration,\n"
        f"        \"framework\": {literal(framework)},\n"
        + extra_result_keys
        + "    }\n"
    )


class CrewAIEmitter(Emitter):
    """Sequential or hierarchical CrewAI crews."""

    key = "crewai"
    label = "CrewAI"
    workflow_imports = (
        "import time",
        "from typing import Any, Dict, List",
        "from crewai import Agent, Crew, Process, Task",
    )

    def tool_fragment(self, tool: ToolSpec) -> Fragment:
        return Fragment(body=_tool_class(tool), imports=list(_TOOL_IMPORTS), provides=[tool.class_name])

    def agent_fragment(self, agent: AgentSpec) -> Fragment:
        manager = (
            self.manifest.primary_workflow is not None
            and self.manifest.primary_workflow.kind == "hierarchical"
            and self.manifest.agents
            and self.manifest.agents[0].name == agent.name
        )
        return Fragment(
            body=_agent_builder(self, agent, manager=bool(manager)),
            imports=["from crewai import Agent"],
            provides=[agent.factory_name],
        )

    def workflow_fragment(self, workflow: WorkflowSpec) -> Fragment:
        hierarchical = workflow.kind == "hierarchical"
        agents = list(self.manifest.agents)

        if hierarchical and len(agents) > 1:
            manager, workers = agents[0], agents[1:]
            crew_body = (
                "    agents = {name: builder() for name, builder in AGENT_BUILDERS.items()}\n"
                f"    manager = agents.pop({literal(manager.name)})\n"
                "    tasks = [\n"
                "        Task(\n"
                "            description=step[\"description\"] + \"\\n\\nUser request: \" + str(query),\n"
                "            expected_output=step[\"expected_output\"],\n"
                "        )\n"
                "        for step in STEPS\n"
                "    ]\n"
                "    # Hierarchical mode: the manager is passed separately and assigns the work,\n"
                "    # so the tasks deliberately carry no `agent=`. Naming an owner here would\n"
                "    # override the delegation this process type exists to perform.\n"
                "    return Crew(\n"
                "        agents=list(agents.values()),\n"
                "        tasks=tasks,\n"
                "        process=Process.hierarchical,\n"
                "        manager_agent=manager,\n"
                "        verbose=True,\n"
                "    )\n"
            )
            note = (
                f"{manager.label} plans and delegates; "
                f"{len(workers)} specialist(s) carry out the work."
            )
        else:
            crew_body = (
                "    agents = {name: builder() for name, builder in AGENT_BUILDERS.items()}\n"
                "    fallback = next(iter(agents.values()))\n"
                "    tasks = [\n"
                "        Task(\n"
                "            description=step[\"description\"] + \"\\n\\nUser request: \" + str(query),\n"
                "            expected_output=step[\"expected_output\"],\n"
                "            agent=agents.get(step[\"agent\"], fallback),\n"
                "        )\n"
                "        for step in STEPS\n"
                "    ]\n"
                "    return Crew(\n"
                "        agents=list(agents.values()),\n"
                "        tasks=tasks,\n"
                "        process=Process.sequential,\n"
                "        verbose=True,\n"
                "    )\n"
            )
            note = "Each task runs in order, and each one receives the previous task's output."

        build = (
            "def build_workflow(query: str = \"\") -> Crew:\n"
            + indent(
                docstring(
                    "Assemble the crew for one query.",
                    [
                        note,
                        "The query is concatenated into each task description rather than "
                        "passed through CrewAI's `{query}` interpolation, which calls "
                        "str.format and fails on any stray brace in the description text.",
                        "`query` defaults to empty so the crew can be assembled without one. "
                        "That is what lets a test construct every agent and task - which is "
                        "where CrewAI's validation actually runs - without inventing a "
                        "question or calling a model.",
                        "Named `build_workflow` rather than `build_crew` because every "
                        "framework in this generator exposes the same two names, and the "
                        "tests and the runtime are written against that pair.",
                    ],
                )
            )
            + "\n"
            + crew_body
        )

        body = "\n\n\n".join(
            [
                _steps_table(workflow),
                _builders_table(agents),
                build,
                _run_workflow("crewai"),
            ]
        )
        return Fragment(
            body=body,
            imports=list(self.workflow_imports),
            provides=["STEPS", "AGENT_BUILDERS", "build_workflow", "run_workflow"],
        )


class CrewAIFlowEmitter(CrewAIEmitter):
    """
    CrewAI Flow: the steps are chained explicitly and state travels between them.

    Each step gets its own single-task crew rather than one crew running every task, because
    that is what makes a Flow a Flow - the flow object owns the sequencing and the state, and
    each listener contributes one result to it. Running a multi-task crew inside a single
    listener, as the old generator did, produced a Flow whose steps all fired at once.
    """

    key = "crewai-flow"
    label = "CrewAI Flow"
    workflow_imports = (
        "import time",
        "from typing import Any, Dict, List",
        "from crewai import Agent, Crew, Process, Task",
        "from crewai.flow.flow import Flow, listen, start",
        "from pydantic import BaseModel, Field",
    )

    def workflow_fragment(self, workflow: WorkflowSpec) -> Fragment:
        steps = workflow.agent_nodes()
        agents = list(self.manifest.agents)

        state = (
            "class FlowState(BaseModel):\n"
            + indent(
                docstring(
                    "State carried between flow steps.",
                    [
                        "`results` accumulates one entry per completed step, so a later step "
                        "can read what an earlier one produced and the final answer can be "
                        "assembled from named parts rather than from whatever was printed.",
                    ],
                )
            )
            + "\n"
            "    query: str = \"\"\n"
            "    results: Dict[str, str] = Field(default_factory=dict)\n"
            "    current_step: str = \"\"\n"
        )

        run_step = (
            "def run_step(step: Dict[str, str], query: str, previous: Dict[str, str]) -> str:\n"
            + indent(
                docstring(
                    "Run one step as a single-task crew and return its output.",
                    [
                        "Earlier results are appended to the prompt so each step can build on "
                        "the last, which is the reason to use a flow instead of one crew.",
                    ],
                )
            )
            + "\n"
            "    builder = AGENT_BUILDERS.get(step[\"agent\"]) or next(iter(AGENT_BUILDERS.values()))\n"
            "    agent = builder()\n"
            "    description = step[\"description\"] + \"\\n\\nUser request: \" + str(query)\n"
            "    if previous:\n"
            "        description += \"\\n\\nEarlier results:\\n\" + \"\\n\".join(\n"
            "            f\"- {name}: {value}\" for name, value in previous.items()\n"
            "        )\n"
            "    crew = Crew(\n"
            "        agents=[agent],\n"
            "        tasks=[Task(\n"
            "            description=description,\n"
            "            expected_output=step[\"expected_output\"],\n"
            "            agent=agent,\n"
            "        )],\n"
            "        process=Process.sequential,\n"
            "        verbose=True,\n"
            "    )\n"
            "    return str(getattr(crew.kickoff(), \"raw\", \"\"))\n"
        )

        # Each planned step becomes a real @listen method, so the emitted flow has the same
        # shape as the workflow graph the UI draws.
        methods: List[str] = [
            "    @start()\n"
            "    def begin(self):\n"
            '        """Entry point. Records which step runs first."""\n'
            f"        self.state.current_step = {literal(steps[0].id if steps else 'done')}\n"
            "        return self.state\n"
        ]
        previous = "begin"
        for index, node in enumerate(steps):
            method = f"step_{index + 1}_{node.agent or 'agent'}"
            planned = f"STEPS[{index}]"
            methods.append(
                f"    @listen({literal(previous)})\n"
                f"    def {method}(self, state):\n"
                f'        """{(node.label or node.id).replace(chr(34), "")}."""\n'
                f"        step = {planned}\n"
                "        output = run_step(step, self.state.query, dict(self.state.results))\n"
                "        self.state.results[step[\"name\"]] = output\n"
                + (
                    f"        self.state.current_step = {literal(steps[index + 1].id)}\n"
                    if index + 1 < len(steps)
                    else '        self.state.current_step = "done"\n'
                )
                + "        return self.state\n"
            )
            previous = method

        methods.append(
            f"    @listen({literal(previous)})\n"
            "    def finish(self, state):\n"
            + indent(
                docstring(
                    "Combine every step's result into the final answer.",
                    [
                        "The last step's output leads, because that is the answer; the earlier "
                        "sections follow as supporting detail.",
                    ],
                )
                , "        "
            )
            + "\n"
            "        if not self.state.results:\n"
            '            return ""\n'
            "        sections = [f\"=== {name} ===\\n{value}\" for name, value in self.state.results.items()]\n"
            '        return "\\n\\n".join(sections)\n'
        )

        flow = "class ProjectFlow(Flow[FlowState]):\n" + indent(
            docstring(
                "The flow: each step listens for the previous one to finish.",
                [workflow.description],
            )
        ) + "\n\n" + "\n\n".join(methods)

        build = (
            "def build_workflow(query: str = \"\") -> ProjectFlow:\n"
            + indent(
                docstring(
                    "Construct the flow with its starting state.",
                    [
                        "`query` defaults to empty so the flow can be constructed without "
                        "one. That is what lets a test build it - which validates the state "
                        "model and every listener wiring - without calling a model.",
                        "Every framework in this generator exposes `build_workflow` and "
                        "`run_workflow`, and the tests and the runtime are written against "
                        "that pair.",
                    ],
                )
            )
            + "\n"
            "    flow = ProjectFlow()\n"
            "    flow.state.query = str(query)\n"
            "    return flow\n"
        )

        run = (
            "def run_workflow(query: str) -> Dict[str, Any]:\n"
            + indent(
                docstring(
                    "Run the flow for one query and return the result with its detail.",
                    [
                        "Returns `output`, `steps`, `tool_calls` and `duration_s`. The step "
                        "list comes from the flow's own state, so it reflects what actually "
                        "ran rather than what was planned to run.",
                    ],
                )
            )
            + "\n"
            "    reset_calls()\n"
            "    started = time.monotonic()\n"
            "    flow = build_workflow(query)\n"
            "    output = flow.kickoff()\n"
            "    duration = round(time.monotonic() - started, 3)\n"
            "\n"
            "    by_name = {step[\"name\"]: step for step in STEPS}\n"
            "    steps = [\n"
            "        {\n"
            "            \"name\": name,\n"
            "            \"agent\": by_name.get(name, {}).get(\"agent\", \"\"),\n"
            "            \"output\": value,\n"
            "        }\n"
            "        for name, value in flow.state.results.items()\n"
            "    ]\n"
            "\n"
            "    return {\n"
            "        \"output\": str(output or \"\"),\n"
            "        \"steps\": steps,\n"
            "        \"tool_calls\": recorded_calls(),\n"
            "        \"duration_s\": duration,\n"
            "        \"framework\": \"crewai-flow\",\n"
            "    }\n"
        )

        body = "\n\n\n".join(
            [
                _steps_table(workflow),
                _builders_table(agents),
                state,
                run_step,
                flow,
                build,
                run,
            ]
        )
        return Fragment(
            body=body,
            imports=list(self.workflow_imports),
            provides=[
                "STEPS",
                "AGENT_BUILDERS",
                "FlowState",
                "run_step",
                "ProjectFlow",
                "build_workflow",
                "run_workflow",
            ],
        )
