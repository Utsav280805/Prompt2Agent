# multi_agent_generator/emit/react_emitter.py
"""
ReAct emitters: the classic ``AgentExecutor`` loop, and a tool-calling LCEL loop.

Both are genuine reason/act cycles, and they differ in how the model is asked to choose a
tool - which is the real distinction between the two, so it is worth having both.

``react`` uses ``create_react_agent`` with the classic scratchpad prompt. It works with any
text model, including ones that have no structured tool-calling API at all, because the
choice is parsed out of the model's own text.

``react-lcel`` binds the tools to the model and executes the ``tool_calls`` it returns,
looping until the model answers. That needs a model with a tool-calling API, so the emitted
code checks for ``bind_tools`` and says plainly what it falls back to when it is absent
rather than failing at import.

The old LCEL generator emitted a plain prompt-to-model chain with the tools declared and
never called, which meant a project whose whole reason for existing was tool use could not
use a tool. That is why this file executes the loop.

Both report their steps from the framework's own record - ``intermediate_steps`` for the
executor, the accumulated messages for the LCEL loop - not from a reconstruction. The
Playground's step list is therefore what happened.
"""
from __future__ import annotations

from ..core.manifest import AgentSpec, ToolSpec, WorkflowSpec
from .base import Emitter, Fragment, docstring, indent, literal

__all__ = ["ReActEmitter", "ReActLCELEmitter"]


#: The classic ReAct scratchpad. `{tools}`, `{tool_names}`, `{input}` and
#: `{agent_scratchpad}` are LangChain's own variables and must survive into the generated
#: file untouched - which they do, because nothing here formats this string.
_SCRATCHPAD = '''

You have access to the following tools:
{tools}

Use exactly this format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Question: {input}
Thought:{agent_scratchpad}'''


def _escape_braces(text: str) -> str:
    """
    Double any brace in model-authored text.

    ``ChatPromptTemplate`` treats a single brace as a variable, so a role description
    containing one raises ``KeyError`` on the first call - a failure that surfaces at run
    time, well after the code looked fine.
    """
    return str(text or "").replace("{", "{{").replace("}", "}}")


def _tool_class(tool: ToolSpec) -> str:
    description = tool.purpose or f"{tool.label} capability."
    params = (
        "    # Declared inputs: " + ", ".join(tool.parameters.keys()) + "\n"
        if tool.parameters
        else ""
    )
    return (
        f"class {tool.class_name}(BaseTool):\n"
        + indent(
            docstring(
                f"{tool.label} - {description}",
                [
                    "A stub with the real name, signature and description, so the agent can "
                    "select it and the loop runs. Replace the body of `_run`; the placeholder "
                    "it returns states that no real work was done, so the agent cannot treat "
                    "it as a genuine result.",
                ],
            )
        )
        + "\n\n"
        # Annotated because BaseTool is a pydantic v2 model: an unannotated attribute raises
        # PydanticUserError at import.
        f"    name: str = {literal(tool.name)}\n"
        f"    description: str = {literal(description)}\n\n"
        + params
        # One string parameter, because LangChain calls `_run` with the tool input as a
        # single value. A parameter per declared input raises TypeError when invoked.
        + "    def _run(self, query: str) -> str:\n"
        "        # TODO: replace this with a real implementation.\n"
        f"        result = {literal('[' + tool.name + ' is not implemented yet] requested: ')} + str(query)\n"
        f"        record_call({literal(tool.name)}, {{\"query\": query}}, result)\n"
        "        return result\n\n"
        "    async def _arun(self, query: str) -> str:\n"
        "        return self._run(query)\n"
    )


class ReActEmitter(Emitter):
    """Classic ReAct: ``create_react_agent`` inside an ``AgentExecutor``."""

    key = "react"
    label = "ReAct"
    workflow_imports = (
        "import time",
        "from typing import Any, Dict, List",
    )
    agent_imports = (
        "from langchain.agents import AgentExecutor, create_react_agent",
        "from langchain_core.prompts import ChatPromptTemplate",
    )

    def tool_fragment(self, tool: ToolSpec) -> Fragment:
        return Fragment(
            body=_tool_class(tool),
            imports=["from langchain_core.tools import BaseTool"],
            provides=[tool.class_name],
        )

    def agent_fragment(self, agent: AgentSpec) -> Fragment:
        role = _escape_braces(agent.role or agent.label)
        goal = _escape_braces(agent.goal or "help the user")
        instructions = _escape_braces(self.instructions_for(agent))
        preface = f"You are {role}. Your goal is: {goal}."
        if instructions and instructions != goal:
            preface += f" {instructions}"

        body = (
            "#: LangChain's classic ReAct scratchpad. Its braces are LangChain's own template\n"
            "#: variables and must stay single, which is why nothing formats this string.\n"
            f"SCRATCHPAD = {literal(_SCRATCHPAD)}\n\n"
            f"#: The system preface for {agent.label}. Braces in the role text are doubled so\n"
            "#: ChatPromptTemplate does not read them as variables.\n"
            f"PREFACE = {literal(preface)}\n\n\n"
            f"def {agent.factory_name}():\n"
            + indent(
                docstring(
                    f"Build the {agent.label} ReAct executor.",
                    [
                        f"Role: {agent.role}." if agent.role else "",
                        (
                            "Tools: " + ", ".join(agent.tools) + "."
                            if agent.tools
                            else "This agent has no tools, so it answers directly."
                        ),
                        "`return_intermediate_steps` is on so a run can report which tools "
                        "were actually called, from the executor's own record rather than "
                        "from a reconstruction.",
                    ],
                )
            )
            + "\n"
            f"    tools = {self.agent_tools_expression(agent)}\n"
            "    prompt = ChatPromptTemplate.from_template(PREFACE + SCRATCHPAD)\n"
            "    agent = create_react_agent(get_llm(), tools, prompt)\n"
            "    return AgentExecutor(\n"
            "        agent=agent,\n"
            "        tools=tools,\n"
            f"        verbose={bool(agent.verbose)},\n"
            "        handle_parsing_errors=True,\n"
            "        # A hard ceiling: a model that never emits 'Final Answer' would otherwise\n"
            "        # loop until the process timeout, which reads as a hang rather than a limit.\n"
            "        max_iterations=6,\n"
            "        return_intermediate_steps=True,\n"
            "    )\n"
        )
        return Fragment(
            body=body,
            imports=list(self.agent_imports),
            provides=["SCRATCHPAD", "PREFACE", agent.factory_name],
        )

    def workflow_fragment(self, workflow: WorkflowSpec) -> Fragment:
        agent = self.manifest.agents[0] if self.manifest.agents else AgentSpec(name="assistant")
        body = (
            "def build_workflow():\n"
            + indent(
                docstring(
                    "Build the reason/act executor.",
                    [workflow.description],
                )
            )
            + "\n"
            f"    return {agent.factory_name}()\n\n\n"
            "def run_workflow(query: str) -> Dict[str, Any]:\n"
            + indent(
                docstring(
                    "Run the agent for one query and return the result with its detail.",
                    [
                        "`steps` is built from the executor's `intermediate_steps`, so each "
                        "entry is a tool the agent really chose, with the input it passed and "
                        "the observation it got back.",
                    ],
                )
            )
            + "\n"
            "    reset_calls()\n"
            "    started = time.monotonic()\n"
            "    executor = build_workflow()\n"
            "    response = executor.invoke({\"input\": str(query)})\n"
            "    duration = round(time.monotonic() - started, 3)\n"
            "\n"
            "    steps: List[Dict[str, Any]] = []\n"
            "    for index, entry in enumerate(response.get(\"intermediate_steps\") or []):\n"
            "        # Each entry is (action, observation). Read it defensively: a parsing\n"
            "        # error can leave a shape that does not unpack.\n"
            "        action = entry[0] if isinstance(entry, (list, tuple)) and entry else entry\n"
            "        observation = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else \"\"\n"
            "        steps.append({\n"
            "            \"name\": str(getattr(action, \"tool\", f\"step_{index + 1}\")),\n"
            f"            \"agent\": {literal(agent.name)},\n"
            "            \"input\": str(getattr(action, \"tool_input\", \"\")),\n"
            "            \"output\": str(observation),\n"
            "        })\n"
            "\n"
            "    return {\n"
            "        \"output\": str(response.get(\"output\") or \"\"),\n"
            "        \"steps\": steps,\n"
            "        \"tool_calls\": recorded_calls(),\n"
            "        \"duration_s\": duration,\n"
            "        \"framework\": \"react\",\n"
            "    }\n"
        )
        return Fragment(
            body=body,
            imports=list(self.workflow_imports),
            provides=["build_workflow", "run_workflow"],
        )


class ReActLCELEmitter(Emitter):
    """
    ReAct built from LCEL primitives, with a real tool-calling loop.

    The model is asked for tool calls through its structured API, the calls are executed, the
    observations are fed back as ``ToolMessage``s, and the loop repeats until the model
    answers or the iteration ceiling is hit.
    """

    key = "react-lcel"
    label = "ReAct (LCEL)"
    workflow_imports = (
        "import time",
        "from typing import Any, Dict, List",
        "from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage",
    )
    # The agent fragment declares its own imports; nothing here needs a prompt template,
    # because the tools are bound to the model rather than described in a prompt.
    agent_imports = ()

    def tool_fragment(self, tool: ToolSpec) -> Fragment:
        return Fragment(
            body=_tool_class(tool),
            imports=["from langchain_core.tools import BaseTool"],
            provides=[tool.class_name],
        )

    def agent_fragment(self, agent: AgentSpec) -> Fragment:
        instructions = self.instructions_for(agent) or f"You are {agent.role or agent.label}."
        body = (
            f"def {agent.factory_name}():\n"
            + indent(
                docstring(
                    f"Build the {agent.label} runnable and its tool set.",
                    [
                        "Returns `(runnable, tools)`. The runnable takes a message list and "
                        "returns the model's reply, with the tools bound when the model "
                        "supports tool calling.",
                        "Not every provider does. When `bind_tools` is missing the plain "
                        "client is returned instead, so the project still runs - it just "
                        "answers without calling a tool, and the run result shows no tool "
                        "calls rather than pretending otherwise.",
                    ],
                )
            )
            + "\n"
            f"    tools = {self.agent_tools_expression(agent)}\n"
            "    client = get_llm()\n"
            "    bind = getattr(client, \"bind_tools\", None)\n"
            "    runnable = bind(tools) if tools and callable(bind) else client\n"
            "    return runnable, tools\n\n\n"
            f"#: The system message for {agent.label}.\n"
            f"INSTRUCTIONS = {literal(instructions)}\n"
        )
        return Fragment(
            body=body,
            imports=[],
            provides=[agent.factory_name, "INSTRUCTIONS"],
        )

    def workflow_fragment(self, workflow: WorkflowSpec) -> Fragment:
        agent = self.manifest.agents[0] if self.manifest.agents else AgentSpec(name="assistant")
        body = (
            "#: How many reason/act rounds are allowed before the loop gives up. A model that\n"
            "#: keeps calling tools without answering would otherwise run to the process\n"
            "#: timeout, which reads as a hang instead of a limit.\n"
            "MAX_ROUNDS = 6\n\n\n"
            "def build_workflow():\n"
            + indent(docstring("Build the runnable and its tool set.", [workflow.description]))
            + "\n"
            f"    return {agent.factory_name}()\n\n\n"
            "def run_workflow(query: str) -> Dict[str, Any]:\n"
            + indent(
                docstring(
                    "Run the reason/act loop for one query.",
                    [
                        "Each round: ask the model, execute any tool calls it returned, feed "
                        "the observations back. Stops when the model replies without a tool "
                        "call, or when MAX_ROUNDS is reached.",
                        "`steps` records each round from the messages that were actually "
                        "exchanged.",
                    ],
                )
            )
            + "\n"
            "    reset_calls()\n"
            "    started = time.monotonic()\n"
            "    runnable, tools = build_workflow()\n"
            "    by_name = {getattr(tool, \"name\", \"\"): tool for tool in tools}\n"
            "\n"
            "    messages: List[Any] = [\n"
            "        SystemMessage(content=INSTRUCTIONS),\n"
            "        HumanMessage(content=str(query)),\n"
            "    ]\n"
            "    steps: List[Dict[str, Any]] = []\n"
            "    output = \"\"\n"
            "\n"
            "    for round_index in range(MAX_ROUNDS):\n"
            "        reply = runnable.invoke(messages)\n"
            "        messages.append(reply)\n"
            "        calls = list(getattr(reply, \"tool_calls\", None) or [])\n"
            "        text = str(getattr(reply, \"content\", \"\") or \"\")\n"
            "\n"
            "        if not calls:\n"
            "            output = text\n"
            "            steps.append({\n"
            "                \"name\": f\"round_{round_index + 1}\",\n"
            f"                \"agent\": {literal(agent.name)},\n"
            "                \"output\": text,\n"
            "            })\n"
            "            break\n"
            "\n"
            "        for call in calls:\n"
            "            name = str(call.get(\"name\") if isinstance(call, dict) else getattr(call, \"name\", \"\"))\n"
            "            args = call.get(\"args\") if isinstance(call, dict) else getattr(call, \"args\", {})\n"
            "            call_id = str(call.get(\"id\") if isinstance(call, dict) else getattr(call, \"id\", \"\") or name)\n"
            "            tool = by_name.get(name)\n"
            "            if tool is None:\n"
            "                observation = f\"No tool named {name} is available to this agent.\"\n"
            "            else:\n"
            "                observation = str(tool.invoke(args if args else \"\"))\n"
            "            messages.append(ToolMessage(content=observation, tool_call_id=call_id))\n"
            "            steps.append({\n"
            "                \"name\": name,\n"
            f"                \"agent\": {literal(agent.name)},\n"
            "                \"input\": str(args),\n"
            "                \"output\": observation,\n"
            "            })\n"
            "    else:\n"
            "        # The loop finished without the model settling on an answer. Say so rather\n"
            "        # than returning the last tool observation as if it were the response.\n"
            "        output = (\n"
            "            f\"The agent used all {MAX_ROUNDS} reason/act rounds without reaching a \"\n"
            "            \"final answer. The steps below show what it tried.\"\n"
            "        )\n"
            "\n"
            "    return {\n"
            "        \"output\": output,\n"
            "        \"steps\": steps,\n"
            "        \"tool_calls\": recorded_calls(),\n"
            "        \"duration_s\": round(time.monotonic() - started, 3),\n"
            "        \"framework\": \"react-lcel\",\n"
            "    }\n"
        )
        return Fragment(
            body=body,
            imports=list(self.workflow_imports),
            provides=["MAX_ROUNDS", "build_workflow", "run_workflow"],
        )
