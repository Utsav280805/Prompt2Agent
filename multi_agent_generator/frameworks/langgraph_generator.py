# multi_agent_generator/frameworks/langgraph_generator.py
"""
Generator for LangGraph code.
"""
from typing import Any, Dict, Optional

from ._common import (
    collect_tools,
    header_comment,
    llm_snippet_for,
    render_llm_setup,
    sanitize_identifier,
    tool_class_name,
)


def create_langgraph_code(
    config: Dict[str, Any],
    provider: str = "openai",
    model: Optional[str] = None,
) -> str:
    """
    Generate LangGraph code from a configuration.

    Args:
        config: Agents, nodes and edges describing the graph.
        provider: LLM provider backing the generated agents.
        model: Optional explicit model id overriding the provider default.

    Returns:
        Generated Python code as a string.
    """
    agents = config.get("agents") or []
    nodes = config.get("nodes") or []
    edges = config.get("edges") or []

    snippet = llm_snippet_for("langgraph", provider, model, config)
    resolved_model = snippet.model

    code = header_comment("LangGraph", provider, resolved_model, "langgraph")

    # `BaseMessage` is referenced by AgentState below; the previous version of this
    # generator annotated the state with it but never imported it, so every generated
    # file raised NameError on import.
    base_imports = [
        "from langgraph.graph import StateGraph, END",
        "from langchain_core.messages import BaseMessage, HumanMessage, AIMessage",
        "from langchain_core.tools import BaseTool",
        "from typing import Any, Dict, List, TypedDict",
    ]
    for line in snippet.imports:
        if line not in base_imports:
            base_imports.append(line)
    code += "\n".join(base_imports) + "\n\n\n"

    # State
    code += "# Define state\n"
    code += "class AgentState(TypedDict):\n"
    code += "    messages: List[BaseMessage]\n"
    code += "    next: str\n\n\n"

    # LLM, built once at import time. Building it inside each node function - as the
    # previous version did - would reload a local transformers model on every single
    # step, which is unusably slow for the huggingface-local provider.
    code += "# Shared LLM\n"
    code += render_llm_setup(snippet) + "\n\n\n"

    # Tools
    tool_names = collect_tools(agents)
    if tool_names:
        code += "# Define tools\n"
        for tool in tool_names:
            cls = tool_class_name(tool)
            # `name` and `description` need type annotations: BaseTool is a pydantic v2
            # model, and unannotated class attributes raise PydanticUserError at import.
            code += f"class {cls}(BaseTool):\n"
            code += f'    name: str = "{tool}"\n'
            code += f'    description: str = "Tool for {tool} operations"\n\n'
            code += "    def _run(self, query: str) -> str:\n"
            code += "        # TODO: implement actual functionality\n"
            code += f'        return f"Result from {tool} tool: {{query}}"\n\n'
            code += "    async def _arun(self, query: str) -> str:\n"
            code += "        return self._run(query)\n\n\n"

        code += "tools = [\n"
        for tool in tool_names:
            code += f"    {tool_class_name(tool)}(),\n"
        code += "]\n\n\n"
    else:
        code += "tools: List[BaseTool] = []\n\n\n"

    # Agent node functions
    agent_funcs = {}
    for agent in agents:
        func = f"{sanitize_identifier(agent.get('name', 'agent'), 'agent')}_agent"
        agent_funcs[agent.get("name")] = func

        role = agent.get("role", "an assistant")
        code += f"# Agent: {agent.get('name', 'agent')}\n"
        code += f"def {func}(state: AgentState) -> AgentState:\n"
        code += f'    """Agent that handles {role}."""\n'
        code += "    messages = state[\"messages\"]\n"
        code += "    response = llm.invoke(messages)\n"
        code += "    return {\n"
        code += '        "messages": messages + [response],\n'
        code += '        "next": state.get("next", ""),\n'
        code += "    }\n\n\n"

    # Routing
    code += "# Define routing logic\n"
    code += "def router(state: AgentState) -> str:\n"
    code += '    """Route to the next node."""\n'
    code += '    return state.get("next", "END")\n\n\n'

    # Graph
    code += "# Define the graph\n"
    code += "workflow = StateGraph(AgentState)\n\n"

    code += "# Add nodes to the graph\n"
    known_nodes = set()
    for node in nodes:
        node_name = node.get("name")
        if not node_name:
            continue
        known_nodes.add(node_name)
        func = agent_funcs.get(node.get("agent"))
        if func is None:
            # Fall back to the first agent rather than emitting a NameError.
            func = next(iter(agent_funcs.values()), None)
            if func is None:
                continue
            code += f"# Note: node '{node_name}' had no matching agent; using {func}\n"
        code += f'workflow.add_node("{node_name}", {func})\n'

    code += "\n# Add edges\n"
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        if target == "END":
            code += f'workflow.add_edge("{source}", END)\n'
        else:
            code += f'workflow.add_edge("{source}", "{target}")\n'

    if nodes:
        entry = nodes[0].get("name")
        if entry:
            code += f'\n# Set entry point\nworkflow.set_entry_point("{entry}")\n'

    code += """

# Compile the graph
app = workflow.compile()


def run_agent(query: str) -> List[BaseMessage]:
    \"\"\"Run the agent on a query.\"\"\"
    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "next": "",
    })
    return result["messages"]


if __name__ == "__main__":
    for message in run_agent("Your query here"):
        print(f"{message.type}: {message.content}")
"""

    return code
