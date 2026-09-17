# multi_agent_generator/frameworks/crewai_flow_generator.py
"""
Generator for CrewAI Flow code.
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


def create_crewai_flow_code(
    config: Dict[str, Any],
    provider: str = "openai",
    model: Optional[str] = None,
) -> str:
    """
    Generate CrewAI Flow code from a configuration.

    Creates event-driven workflow code using the CrewAI Flow framework, with
    transitions between workflow steps.

    Args:
        config: Agents, tasks and workflow configuration.
        provider: LLM provider backing the generated agents.
        model: Optional explicit model id overriding the provider default.

    Returns:
        Generated Python code as a string.
    """
    agents = config.get("agents") or []
    tasks = config.get("tasks") or []

    snippet = llm_snippet_for("crewai-flow", provider, model, config)

    code = header_comment("CrewAI Flow", provider, snippet.model, "crewai-flow")

    tool_names = collect_tools(agents)

    imports: List[str] = [
        "from crewai import Agent, Crew, Task",
        "from crewai.flow.flow import Flow, listen, start",
    ]
    if tool_names:
        imports.append("from crewai.tools import BaseTool")
    imports += [
        "from typing import Any, Dict, List",
        "from pydantic import BaseModel, Field",
    ]
    for line in snippet.imports:
        if line not in imports:
            imports.append(line)
    code += "\n".join(imports) + "\n\n\n"

    # LLM
    code += "# Shared LLM\n"
    code += render_llm_setup(snippet) + "\n\n\n"

    # Flow state
    code += "# Define flow state\n"
    code += "class AgentState(BaseModel):\n"
    code += '    query: str = Field(default="")\n'
    code += "    results: Dict[str, Any] = Field(default_factory=dict)\n"
    code += '    current_step: str = Field(default="")\n\n\n'

    # Tools (CrewAI validates tools as objects, not name strings)
    tool_instances: Dict[str, str] = {}
    if tool_names:
        code += "# Tool stubs - replace the bodies with real implementations\n"
        for tool in tool_names:
            cls = tool_class_name(tool)
            tool_instances[tool] = f"{cls}()"
            code += f"class {cls}(BaseTool):\n"
            code += f'    name: str = "{tool}"\n'
            code += f'    description: str = "Tool for {tool} operations"\n\n'
            code += "    def _run(self, query: str) -> str:\n"
            code += "        # TODO: implement actual functionality\n"
            code += f'        return f"Result from {tool}: {{query}}"\n\n\n'

        code += "TOOLS = {\n"
        for tool in tool_names:
            code += f'    "{tool}": {tool_instances[tool]},\n'
        code += "}\n\n\n"

    # Agents. Names are sanitised: an LLM-supplied name like "Research Analyst" used to
    # be interpolated straight into an identifier, producing code that would not parse.
    agent_name_to_var: Dict[str, str] = {}
    for i, agent in enumerate(agents):
        agent_var = f"agent_{sanitize_identifier(agent.get('name', f'agent_{i}'), 'agent')}"
        agent_name_to_var[agent.get("name")] = agent_var

        agent_tools = [t for t in (agent.get("tools") or []) if isinstance(t, str)]
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
        code += ")\n\n"

    # Tasks
    task_vars: List[str] = []
    task_steps: List[Dict[str, str]] = []
    for i, task in enumerate(tasks):
        raw_name = task.get("name", f"task_{i}")
        safe = sanitize_identifier(raw_name, "task")
        task_var = f"task_{safe}"
        task_vars.append(task_var)
        task_steps.append({"raw": raw_name, "safe": safe, "var": task_var})

        code += f"# Task: {raw_name}\n"
        code += f"{task_var} = Task(\n"
        code += f"    description={task.get('description', '')!r},\n"

        agent_name = task.get("agent")
        if agent_name and agent_name in agent_name_to_var:
            code += f"    agent={agent_name_to_var[agent_name]},\n"
        elif agent_name_to_var:
            fallback = agents[0].get("name")
            code += f"    # Auto-assigned to: {fallback}\n"
            code += f"    agent={agent_name_to_var[fallback]},\n"

        code += f"    expected_output={task.get('expected_output', 'A useful result')!r},\n"
        code += ")\n\n"

    # Crew
    code += "# Crew configuration\n"
    code += "crew = Crew(\n"
    code += "    agents=[" + ", ".join(agent_name_to_var.values()) + "],\n"
    code += "    tasks=[" + ", ".join(task_vars) + "],\n"
    code += "    verbose=True,\n"
    code += ")\n\n\n"

    # Flow
    code += "# Define CrewAI Flow\n"
    code += "class WorkflowFlow(Flow[AgentState]):\n"
    code += "    @start()\n"
    code += "    def initial_input(self):\n"
    code += '        """Process the initial user query."""\n'
    code += '        print("Starting workflow...")\n'
    first_step = task_steps[0]["raw"] if task_steps else "completed"
    code += f'        self.state.current_step = "{first_step}"\n'
    code += "        return self.state\n\n"

    previous_step = "initial_input"
    for i, step in enumerate(task_steps):
        code += f"    @listen('{previous_step}')\n"
        code += f"    def execute_{step['safe']}(self, state):\n"
        code += f'        """Execute the {step["raw"]} task."""\n'
        code += f'        print("Executing task: {step["raw"]}")\n'
        code += f"        result = crew.kickoff(inputs={{\n"
        code += '            "query": self.state.query,\n'
        code += '            "previous_results": self.state.results,\n'
        code += "        })\n"
        code += f'        self.state.results["{step["raw"]}"] = result\n'
        if i < len(task_steps) - 1:
            code += f'        self.state.current_step = "{task_steps[i + 1]["raw"]}"\n'
        else:
            code += '        self.state.current_step = "completed"\n'
        code += "        return self.state\n\n"
        previous_step = f"execute_{step['safe']}"

    code += f"    @listen('{previous_step}')\n"
    code += "    def aggregate_results(self, state):\n"
    code += '        """Combine all results from tasks."""\n'
    code += '        print("Workflow completed, aggregating results...")\n'
    code += '        combined_result = ""\n'
    code += "        for task_name, result in state.results.items():\n"
    # Both placeholders must survive into the generated file. The previous version wrote
    # this line with a single-braced {task_name} inside an f-string, so the *generator's*
    # loop variable was substituted and every aggregated section was labelled with the
    # last task's name instead of the current one.
    code += '            combined_result += f"\\n\\n=== {task_name} ===\\n{result}"\n'
    code += "        return combined_result\n\n\n"

    code += '''def run_workflow(query: str):
    """Run the flow end to end."""
    flow = WorkflowFlow()
    flow.state.query = query
    return flow.kickoff()


def visualize_flow():
    """Write an HTML visualisation of the flow."""
    WorkflowFlow().plot("workflow_flow")
    print("Flow visualization saved to workflow_flow.html")


if __name__ == "__main__":
    print(run_workflow("Your query here"))
'''
    return code
