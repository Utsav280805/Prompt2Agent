# multi_agent_generator/frameworks/crewai_generator.py
"""
Generator for CrewAI code.
"""
from typing import Any, Dict, List, Optional

from ._common import (
    collect_tools,
    header_comment,
    llm_snippet_for,
    render_llm_setup,
    sanitize_identifier,
    tool_class_name,
)


def _sanitize_var_name(name: str) -> str:
    """Convert an agent/task name to a valid Python variable name."""
    return sanitize_identifier(name, "item")


def _crewai_tool_classes(tool_names: List[str]) -> str:
    """
    Emit CrewAI tool classes for the tool names mentioned in the config.

    This replaces the previous behaviour of passing the raw name strings straight into
    ``Agent(tools=[...])``. CrewAI validates that entry as a list of tool objects, so
    ``tools=['search_tool']`` failed at construction - meaning no generated CrewAI file
    with tools could ever be imported, let alone tested.
    """
    if not tool_names:
        return ""

    code = "# Tool stubs - replace the bodies with real implementations\n"
    for tool in tool_names:
        cls = tool_class_name(tool)
        code += f"class {cls}(BaseTool):\n"
        code += f'    name: str = "{tool}"\n'
        code += f'    description: str = "Tool for {tool} operations"\n\n'
        code += "    def _run(self, query: str) -> str:\n"
        code += "        # TODO: implement actual functionality\n"
        code += f'        return f"Result from {tool}: {{query}}"\n\n\n'
    return code


def create_crewai_code(
    config: Dict[str, Any],
    provider: str = "openai",
    model: Optional[str] = None,
) -> str:
    """
    Generate CrewAI code from a configuration.

    Args:
        config: Agents, tasks and process type.
        provider: LLM provider backing the generated agents.
        model: Optional explicit model id overriding the provider default.

    Returns:
        Generated Python code as a string.
    """
    process_type = (config.get("process") or "sequential").lower()
    agents = config.get("agents") or []
    tasks = config.get("tasks") or []

    snippet = llm_snippet_for("crewai", provider, model, config)
    resolved_model = snippet.model

    code = header_comment("CrewAI", provider, resolved_model, "crewai")

    tool_names = collect_tools(agents)

    imports: List[str] = ["from crewai import Agent, Crew, Process, Task"]
    if tool_names:
        imports.append("from crewai.tools import BaseTool")
    imports.append("from typing import Any, Dict, List")
    for line in snippet.imports:
        if line not in imports:
            imports.append(line)
    code += "\n".join(imports) + "\n\n\n"

    # LLM
    code += "# Shared LLM\n"
    code += render_llm_setup(snippet) + "\n\n\n"

    # Tools
    code += _crewai_tool_classes(tool_names)

    tool_instances = {t: f"{tool_class_name(t)}()" for t in tool_names}
    if tool_names:
        code += "# Tool registry\n"
        code += "TOOLS = {\n"
        for tool in tool_names:
            code += f'    "{tool}": {tool_instances[tool]},\n'
        code += "}\n\n\n"

    # Agents
    agent_name_to_var: Dict[str, str] = {}
    for i, agent in enumerate(agents):
        agent_var = f"agent_{_sanitize_var_name(agent.get('name', f'agent_{i}'))}"
        agent_name_to_var[agent.get("name")] = agent_var

        agent_tools = [t for t in (agent.get("tools") or []) if isinstance(t, str)]
        # CrewAI's hierarchical manager is a coordinator only. CrewAI rejects manager tools
        # during kickoff, so keep tools on specialists and let the manager delegate to them.
        if process_type == "hierarchical" and i == 0:
            agent_tools = []
        tools_expr = (
            "[" + ", ".join(f'TOOLS["{t}"]' for t in agent_tools if t in tool_instances) + "]"
        )

        code += f"# Agent: {agent.get('name', f'agent_{i}')}\n"
        code += f"{agent_var} = Agent(\n"
        code += f"    role={agent.get('role', 'Specialist')!r},\n"
        code += f"    goal={agent.get('goal', 'Complete assigned tasks')!r},\n"
        code += f"    backstory={agent.get('backstory', '')!r},\n"
        code += f"    verbose={bool(agent.get('verbose', True))},\n"
        code += f"    allow_delegation={bool(agent.get('allow_delegation', False))},\n"
        code += f"    tools={tools_expr},\n"
        code += "    llm=llm,\n"
        if process_type == "hierarchical" and i == 0:
            code += "    max_iter=5,\n"
            code += "    max_execution_time=300,\n"
        code += ")\n\n"

    # Tasks
    task_vars: List[str] = []
    for i, task in enumerate(tasks):
        task_var = f"task_{_sanitize_var_name(task.get('name', f'task_{i}'))}"
        task_vars.append(task_var)

        code += f"# Task: {task.get('name', f'task_{i}')}\n"
        code += f"{task_var} = Task(\n"
        code += f"    description={task.get('description', '')!r},\n"

        agent_name = task.get("agent")
        if agent_name and agent_name in agent_name_to_var:
            code += f"    agent={agent_name_to_var[agent_name]},\n"
        elif agent_name_to_var:
            # Assign a sensible owner rather than emitting an undefined name.
            if process_type == "hierarchical" and len(agents) > 1:
                fallback = agents[1].get("name")
            else:
                fallback = agents[0].get("name")
            fallback_var = agent_name_to_var.get(fallback) or next(
                iter(agent_name_to_var.values())
            )
            code += f"    # Auto-assigned to: {fallback}\n"
            code += f"    agent={fallback_var},\n"

        code += f"    expected_output={task.get('expected_output', 'A useful result')!r},\n"
        code += ")\n\n"

    # Crew
    code += "# Crew configuration\n"
    code += "crew = Crew(\n"
    code += "    agents=[" + ", ".join(agent_name_to_var.values()) + "],\n"
    code += "    tasks=[" + ", ".join(task_vars) + "],\n"
    if process_type == "hierarchical":
        code += "    process=Process.hierarchical,\n"
        if agent_name_to_var:
            manager_var = next(iter(agent_name_to_var.values()))
            code += f"    manager_agent={manager_var},\n"
    else:
        code += "    process=Process.sequential,\n"
    code += "    verbose=True,\n"
    code += ")\n\n\n"

    code += '''def run_workflow(query: str):
    """Run the workflow using CrewAI."""
    return crew.kickoff(inputs={"query": query})


if __name__ == "__main__":
    print(run_workflow("Your query here"))
'''
    return code
