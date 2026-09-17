# multi_agent_generator/frameworks/agno_generator.py
"""
Generator for Agno team code.
"""
from typing import Any, Dict, List, Optional

from ._common import header_comment, llm_snippet_for, sanitize_identifier


def _sanitize(name: str) -> str:
    return sanitize_identifier(name, "item")


def create_agno_code(
    config: Dict[str, Any],
    provider: str = "openai",
    model: Optional[str] = None,
) -> str:
    """
    Generate Agno code from a configuration.

    Structure of the generated file:
      - imports
      - shared model factory
      - agents
      - task functions
      - team config
      - run_workflow(query)

    Args:
        config: Agents, tasks and process type.
        provider: LLM provider backing the generated agents.
        model: Optional explicit model id overriding the provider default.
    """
    process_type = (config.get("process") or "sequential").lower()
    agents = config.get("agents") or []
    tasks = config.get("tasks") or []

    snippet = llm_snippet_for("agno", provider, model, config)

    code = header_comment("Agno", provider, snippet.model, "agno")

    imports: List[str] = [
        "from agno.agent import Agent",
        "from agno.team import Team",
        "from typing import Any, Dict, List",
        "from dotenv import load_dotenv",
    ]
    for line in snippet.imports:
        if line not in imports:
            imports.append(line)
    code += "\n".join(imports) + "\n\n"
    code += "load_dotenv()  # Load environment variables from .env file\n\n\n"

    if snippet.comment:
        code += f"# NOTE: {snippet.comment}\n"

    # Credential resolution has to be emitted at module level, before the factory below
    # closes over the resulting name. Agno is the one framework that consumes only the
    # snippet's `expression` (it goes inside a function body), so unlike the other
    # generators it cannot rely on render_llm_setup to place this.
    if snippet.preamble:
        code += snippet.preamble + "\n\n"

    # A factory rather than a shared instance: Agno attaches per-agent state to a model
    # object, so handing the same instance to every agent and to the team can couple
    # them together in surprising ways.
    code += "def build_model():\n"
    code += '    """Construct the LLM used by every agent and by the team leader."""\n'
    code += f"    return {snippet.expression}\n\n\n"

    # Agents
    agent_vars: List[str] = []
    agent_name_to_var: Dict[str, str] = {}
    for i, agent in enumerate(agents):
        raw_name = agent.get("name", f"agent_{i}")
        var = f"agent_{_sanitize(raw_name)}"
        agent_vars.append(var)
        agent_name_to_var[raw_name] = var

        role = agent.get("role", "")
        goal = agent.get("goal", "")
        backstory = agent.get("backstory", "")
        instructions = " ".join(p for p in (goal, backstory) if p).strip()

        code += f"# Agent: {raw_name}\n"
        code += f"{var} = Agent(\n"
        code += f"    name={raw_name!r},\n"
        code += "    model=build_model(),\n"
        code += f"    role={role!r},\n"
        if instructions:
            code += f"    instructions={instructions!r},\n"
        # Agno expects tool objects; the config only carries names, so this stays empty
        # rather than emitting something that fails validation.
        code += "    tools=[],\n"
        code += "    markdown=True,\n"
        code += ")\n\n"

    if not agent_vars:
        # Guarantee at least one member so the emitted Team is constructible.
        code += "# No agents in config; emitting a single fallback agent\n"
        code += "agent_default = Agent(\n"
        code += "    name='default_assistant',\n"
        code += "    model=build_model(),\n"
        code += "    role='General Assistant',\n"
        code += "    tools=[],\n"
        code += "    markdown=True,\n"
        code += ")\n\n"
        agent_vars.append("agent_default")

    # Tasks
    task_vars: List[str] = []
    for i, task in enumerate(tasks):
        raw_name = task.get("name", f"task_{i}")
        tvar = f"task_{_sanitize(raw_name)}"
        task_vars.append(tvar)

        desc = task.get("description", "")
        expected = task.get("expected_output", "")

        assigned = task.get("agent")
        if assigned and assigned in agent_name_to_var:
            assigned_var = agent_name_to_var[assigned]
            auto_note = None
        else:
            if process_type == "hierarchical" and len(agent_vars) > 1:
                assigned_var = agent_vars[1]
            else:
                assigned_var = agent_vars[0]
            auto_note = assigned_var

        code += f"# Task: {raw_name}\n"
        code += f"def {tvar}(query: str) -> Any:\n"
        if auto_note:
            code += f"    # Auto-assigned to: {auto_note}\n"
        code += "    prompt = (\n"
        code += f"        {desc!r} + '\\n\\n' +\n"
        code += "        'User query: ' + str(query) + '\\n' +\n"
        code += f"        'Expected output: ' + {expected!r}\n"
        code += "    )\n"
        code += f"    return {assigned_var}.run(prompt).content\n\n"

    # Team
    code += "# Team configuration\n"
    code += "team = Team(\n"
    code += "    name='Auto Team',\n"
    code += "    mode='coordinate',\n"
    code += "    model=build_model(),\n"
    code += f"    members=[{', '.join(agent_vars)}],\n"
    code += "    instructions=[\n"
    code += "        'Coordinate members to complete the tasks in order.',\n"
    code += "        'Use the query as shared context.',\n"
    code += "    ],\n"
    code += "    markdown=True,\n"
    code += "    show_members_responses=True,\n"
    code += ")\n\n\n"

    # Runner
    code += "def run_workflow(query: str) -> Dict[str, Any]:\n"
    code += '    """Run the Agno team, executing tasks in order."""\n'
    code += "    results: Dict[str, Any] = {}\n"
    if task_vars:
        for tvar in task_vars:
            code += f"    results[{tvar!r}] = {tvar}(query)\n"
    else:
        code += "    results['team'] = team.run(query).content\n"
    code += "    return results\n\n\n"

    code += 'if __name__ == "__main__":\n'
    code += '    print(run_workflow("Your query here"))\n'

    return code
