# multi_agent_generator/frameworks/react_generator.py
"""
Generator for ReAct code (classic AgentExecutor and LCEL variants).
"""
from typing import Any, Dict, List, Optional

from ._common import (
    header_comment,
    llm_snippet_for,
    render_llm_setup,
    tool_class_name,
)


def _tool_classes(config: Dict[str, Any]) -> str:
    """
    Emit BaseTool subclasses for each configured tool.

    Two fixes over the previous version: the ``name``/``description`` fields carry type
    annotations (BaseTool is a pydantic v2 model and unannotated attributes raise
    PydanticUserError at import time), and ``_run`` takes a single string argument.
    LangChain calls ``_run`` with the tool input, so generating one positional parameter
    per config-declared "parameter" produced tools that raised TypeError when invoked.
    """
    tools = config.get("tools") or []
    if not tools:
        return "tools: List[BaseTool] = []\n\n\n"

    code = "# Define tools\n"
    for tool in tools:
        name = tool.get("name") or "custom_tool"
        cls = tool_class_name(name)
        description = str(tool.get("description") or f"Tool for {name}").replace('"', "'")
        params = tool.get("parameters") or {}

        code += f"class {cls}(BaseTool):\n"
        code += f'    name: str = "{name}"\n'
        code += f'    description: str = "{description}"\n\n'
        if params:
            code += "    # Declared parameters: " + ", ".join(params.keys()) + "\n"
        code += "    def _run(self, query: str) -> str:\n"
        code += "        try:\n"
        code += "            # TODO: implement actual functionality\n"
        code += f'            return f"Executed {name} with input: {{query}}"\n'
        code += "        except Exception as exc:\n"
        code += f'            return f"Error in {name}: {{exc}}"\n\n'
        code += "    async def _arun(self, query: str) -> str:\n"
        code += "        return self._run(query)\n\n\n"

    code += "tools = [\n"
    for tool in tools:
        code += f"    {tool_class_name(tool.get('name') or 'custom_tool')}(),\n"
    code += "]\n\n\n"
    return code


def _primary_agent(config: Dict[str, Any]) -> Dict[str, Any]:
    agents = config.get("agents") or []
    if agents and isinstance(agents[0], dict):
        return agents[0]
    return {"role": "a helpful assistant", "goal": "assist the user"}


def _escape(text: str) -> str:
    """Make text safe to embed inside a double-quoted generated string literal."""
    return str(text or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def create_react_code(
    config: Dict[str, Any],
    provider: str = "openai",
    model: Optional[str] = None,
) -> str:
    """Generate classic ReAct code built on ``AgentExecutor``."""
    snippet = llm_snippet_for("react", provider, model, config)
    resolved_model = snippet.model
    agent = _primary_agent(config)

    code = header_comment("ReAct (classic)", provider, resolved_model, "react")

    imports: List[str] = [
        "from langchain_core.tools import BaseTool",
        "from langchain_core.prompts import ChatPromptTemplate",
        "from langchain.agents import AgentExecutor, create_react_agent",
        "from typing import Any, Dict, List",
    ]
    for line in snippet.imports:
        if line not in imports:
            imports.append(line)
    code += "\n".join(imports) + "\n\n\n"

    code += _tool_classes(config)

    code += "# Shared LLM\n"
    code += render_llm_setup(snippet) + "\n\n\n"

    role = _escape(agent.get("role", "a helpful assistant"))
    goal = _escape(agent.get("goal", "assist the user"))

    # create_react_agent expects the classic ReAct scratchpad variables. Omitting
    # {tools}/{tool_names}/{agent_scratchpad} makes AgentExecutor raise at construction,
    # so the prompt below declares all three.
    code += f'''react_prompt = ChatPromptTemplate.from_template(
    """You are {role}. Your goal is {goal}.

You have access to the following tools:
{{tools}}

Use this format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Question: {{input}}
Thought:{{agent_scratchpad}}"""
)

agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)


def run_agent(query: str) -> str:
    """Run the ReAct agent on a query."""
    response = agent_executor.invoke({{"input": query}})
    if isinstance(response, dict):
        for step in response.get("intermediate_steps", []) or []:
            print(step)
        return response.get("output", "No response generated")
    return str(response)


if __name__ == "__main__":
    print(run_agent("Your query here"))
'''
    return code


def create_react_lcel_code(
    config: Dict[str, Any],
    provider: str = "openai",
    model: Optional[str] = None,
) -> str:
    """Generate ReAct code built with LangChain Expression Language."""
    snippet = llm_snippet_for("react-lcel", provider, model, config)
    resolved_model = snippet.model
    agent = _primary_agent(config)

    code = header_comment("ReAct (LCEL)", provider, resolved_model, "react-lcel")

    imports: List[str] = [
        "from typing import Any, Dict, List",
        "from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder",
        "from langchain_core.output_parsers import StrOutputParser",
        "from langchain_core.runnables import RunnableLambda, RunnablePassthrough",
        "from langchain_core.tools import BaseTool",
    ]
    for line in snippet.imports:
        if line not in imports:
            imports.append(line)
    code += "\n".join(imports) + "\n\n\n"

    code += _tool_classes(config)

    code += "# Shared LLM\n"
    code += render_llm_setup(snippet) + "\n\n\n"

    role = _escape(agent.get("role", "a helpful assistant"))
    goal = _escape(agent.get("goal", "assist the user"))

    # The previous version piped a dict of two RunnablePassthrough()s into the prompt,
    # which forwards the whole input dict to both slots instead of selecting keys, so
    # "history" received the full payload. Selecting explicitly with itemgetter-style
    # lambdas keeps each prompt variable bound to its own value.
    code += f'''react_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are {role}. Your goal is {goal}. Use tools when needed."),
    MessagesPlaceholder("history", optional=True),
    ("human", "{{input}}"),
])

chain = (
    {{
        "input": RunnableLambda(lambda payload: payload["input"]),
        "history": RunnableLambda(lambda payload: payload.get("history", [])),
    }}
    | react_prompt
    | llm
    | StrOutputParser()
)


def run_agent(query: str, history: List[Any] = None) -> str:
    """Run the LCEL chain on a query."""
    return chain.invoke({{"input": query, "history": history or []}})


if __name__ == "__main__":
    print(run_agent("Your query here"))
'''
    return code
