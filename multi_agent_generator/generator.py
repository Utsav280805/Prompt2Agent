# multi_agent_generator/generator.py
"""
Agent configuration generator that analyzes user requirements.
Unified across multiple LLM providers via the provider registry.
"""
import json
import warnings
from typing import Dict, Any, Optional, List

from .model_inference import Message, build_inference
from .providers import get_provider

# The generator can be driven from the Streamlit UI or from the CLI/API, and it should
# report warnings through whichever is active. The trap that produced "missing
# ScriptRunContext!" on every CLI run was equating "streamlit is importable" with
# "we are inside a Streamlit script run" - the former is true whenever the package is
# merely installed. `_in_streamlit_runtime()` asks the real question instead.
try:  # pragma: no cover - trivial import guard
    import streamlit as st
except Exception:  # ImportError, or Streamlit raising during import
    st = None


def _in_streamlit_runtime() -> bool:
    """True only when code is executing inside an active Streamlit script run."""
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        # Older/newer Streamlit may move this helper; fall back to the runtime probe.
        try:
            from streamlit.runtime import exists

            return bool(exists())
        except Exception:
            return False


def _report(level: str, message: str) -> None:
    """Surface a message through Streamlit when in the UI, else the warnings module."""
    if _in_streamlit_runtime():
        getattr(st, level, st.write)(message)
    elif level == "error":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    else:
        warnings.warn(message, UserWarning, stacklevel=2)


class AgentGenerator:
    """
    Generates agent configurations based on natural language descriptions.
    Provider-agnostic: see ``multi_agent_generator.providers`` for the registry.
    """

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize the generator with the specified provider.

        Args:
            provider: The LLM provider to use (openai, huggingface,
                huggingface-local, watsonx, ollama, anthropic, groq).
            model: Optional explicit model id, overriding the provider default.
        """
        self.provider = get_provider(provider).name
        self.model_id = model
        self.model = None

    def set_provider(self, provider: str, model: Optional[str] = None):
        """
        Change the LLM provider.

        Args:
            provider: The LLM provider to use.
            model: Optional explicit model id, overriding the provider default.
        """
        self.provider = get_provider(provider).name
        self.model_id = model
        self.model = None  # reset for re-init

    def _initialize_model(self):
        """
        Build the inference backend if not already done.

        Everything about *how* to reach the model - which credential satisfies this
        provider, which route prefix the id needs, what the default model is - belongs to
        the LLM layer and is resolved there. Two things this method used to do have been
        removed rather than moved:

        It no longer applies a route prefix. It called ``resolve_generator_model``, which
        returns ``huggingface/Qwen/Qwen2.5-7B-Instruct``, and the layer below then stripped
        that prefix off again before the provider class added its own. A value that is
        transformed and untransformed on its way through two modules is a bug waiting to
        be introduced; the plain id is passed straight through instead.

        It no longer warns about a missing credential and carry on. ``validate()`` raises
        :class:`~multi_agent_generator.errors.MissingCredentialError` - a real error with a
        message and a suggested action - *before* any request is built. A warning that was
        followed by a request that could not possibly succeed only moved the failure later
        and made it less legible.
        """
        if self.model is not None:
            return

        self.model = build_inference(
            self.provider,
            model=self.model_id,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.95,
        )
        self.model.validate()

    def analyze_prompt(self, user_prompt: str, framework: str) -> Dict[str, Any]:
        """
        Analyze a natural language prompt to generate agent configuration.

        Args:
            user_prompt: The natural language description
            framework: The agent framework to use

        Returns:
            A dictionary containing the agent configuration
        """
        self._initialize_model()
        system_prompt = self._get_system_prompt_for_framework(framework)

        try:
            messages: List[Message] = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt)
            ]

            response = self.model.generate_text(messages)

            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as exc:
                    # Smaller open-weight models often emit *almost* valid JSON. Say so
                    # plainly instead of silently returning the default config.
                    _report(
                        "warning",
                        f"Model returned malformed JSON ({exc}). "
                        "Using the default configuration. Try a larger model if this "
                        "keeps happening.",
                    )
                    return self._get_default_config(framework)
            else:
                _report(
                    "warning",
                    "Could not extract valid JSON from model response. "
                    "Using default configuration.",
                )
                return self._get_default_config(framework)

        except Exception as e:
            _report("error", f"Error in analyzing prompt: {e}")
            return self._get_default_config(framework)

    def _get_system_prompt_for_framework(self, framework: str) -> str:
        """
        Get the system prompt for the specified framework.
        
        Args:
            framework: The agent framework to use
            
        Returns:
            The system prompt for the framework
        """
        if framework == "crewai":
            return """
            You are an expert at creating AI research assistants using CrewAI. Based on the user's request,
            suggest appropriate agents, their roles, tools, and tasks. 
            
            CRITICAL REQUIREMENTS:
            1. Create specialized agents with distinct roles and expertise
            2. ALWAYS assign the most appropriate agent to each task based on their role/expertise
            3. Each task must have an "agent" field with the exact agent name
            4. Match agent specialization to task requirements
            
            Process Types:
            - Sequential: Tasks executed one after another in order
            - Hierarchical: A manager agent coordinates and delegates tasks to specialized agents
            
            Format your response as JSON with this structure:
            {
                "process": "sequential" or "hierarchical",
                "agents": [
                    {
                        "name": "agent_name",
                        "role": "specific specialized role",
                        "goal": "clear specific goal",
                        "backstory": "relevant professional backstory",
                        "tools": ["relevant_tool1", "relevant_tool2"],
                        "verbose": true,
                        "allow_delegation": true/false
                    }
                ],
                "tasks": [
                    {
                        "name": "task_name",
                        "description": "detailed task description",
                        "tools": ["required tools for this task"],
                        "agent": "exact_agent_name_from_above",
                        "expected_output": "specific expected output"
                    }
                ]
            }
            
            AGENT-TASK ASSIGNMENT RULES:
            - Research tasks → Research Specialist/Analyst
            - Data collection → Data Specialist/Collector  
            - Analysis tasks → Data Analyst/Statistician
            - Writing tasks → Content Writer/Technical Writer
            - Review tasks → Quality Reviewer/Editor
            - Coordination tasks → Project Manager/Coordinator
            
            ALWAYS ensure each task has the most suitable agent assigned based on the agent's role and expertise.
            Use exact agent names (matching the "name" field in agents array) in the "agent" field of tasks.
            """
        elif framework == "crewai-flow":
            return """
            You are an expert at creating AI research assistants using CrewAI Flow. Based on the user's request,
            suggest appropriate agents, their roles, tools, and tasks organized in a workflow. 
            
            CRITICAL REQUIREMENTS:
            1. Create specialized agents with distinct roles and expertise
            2. ALWAYS assign the most appropriate agent to each task based on their role/expertise
            3. Each task must have an "agent" field with the exact agent name
            4. Match agent specialization to task requirements
            
            Process Types:
            - Sequential: Tasks flow through a predefined sequence with specific agent assignments
            - Hierarchical: A manager coordinates the flow and delegates to specialized agents
            
            Format your response as JSON with this structure:
            {
                "process": "sequential" or "hierarchical",
                "agents": [
                    {
                        "name": "agent_name",
                        "role": "specific specialized role",
                        "goal": "clear specific goal",
                        "backstory": "relevant professional backstory",
                        "tools": ["relevant_tool1", "relevant_tool2"],
                        "verbose": true,
                        "allow_delegation": true/false
                    }
                ],
                "tasks": [
                    {
                        "name": "task_name",
                        "description": "detailed task description",
                        "tools": ["required tools for this task"],
                        "agent": "exact_agent_name_from_above",
                        "expected_output": "specific expected output"
                    }
                ]
            }
            
            ALWAYS ensure proper agent-to-task matching based on expertise and specialization.
            """
        elif framework == "langgraph":
            return """
            You are an expert at creating AI agents using LangChain's LangGraph framework. Based on the user's request,
            suggest appropriate agents, their roles, tools, and nodes for the graph. Format your response as JSON with this structure:
            {
                "agents": [
                    {
                        "name": "agent name",
                        "role": "specific role description",
                        "goal": "clear goal",
                        "tools": ["tool1", "tool2"],
                        "llm": "model name (e.g., gpt-4.1-mini)"
                    }
                ],
                "nodes": [
                    {
                        "name": "node name",
                        "description": "detailed description",
                        "agent": "agent name"
                    }
                ],
                "edges": [
                    {
                        "source": "source node name",
                        "target": "target node name",
                        "condition": "condition description (optional)"
                    }
                ]
            }
            """
        elif framework == "react":
            return """
            You are an expert at creating AI agents using the ReAct (Reasoning + Acting) framework. 
            Based on the user's request, design an agent with reasoning steps and tool usage.

            Format your response strictly as JSON with this structure:
            {
                "agents": [
                    {
                        "name": "agent name",
                        "role": "specific role description",
                        "goal": "clear goal",
                        "tools": ["tool1", "tool2"],
                        "llm": "model name (e.g., gpt-4.1-mini)"
                    }
                ],
                "tools": [
                    {
                        "name": "tool name",
                        "description": "detailed description of what the tool does",
                        "parameters": {
                            "param1": "parameter description",
                            "param2": "parameter description"
                        }
                    }
                ],
                "examples": [
                    {
                        "query": "example user query",
                        "thought": "single-step thought",
                        "action": "example action",
                        "observation": "example observation",
                        "final_answer": "example final answer"
                    }
                ]
            }
            """
        elif framework == "react-lcel":
            return """
            You are an expert at creating AI agents using the ReAct (Reasoning + Acting) framework, 
            implemented with LangChain Expression Language (LCEL). 
            The agent should demonstrate **multi-step reasoning** with clear intermediate steps.

            Format your response strictly as JSON with this structure:
            {
                "agents": [
                    {
                        "name": "agent name",
                        "role": "specific role description",
                        "goal": "clear goal",
                        "tools": ["tool1", "tool2"],
                        "llm": "model name (e.g., gpt-4.1-mini)"
                    }
                ],
                "tools": [
                    {
                        "name": "tool name",
                        "description": "detailed description of what the tool does",
                        "parameters": {
                            "param1": "parameter description",
                            "param2": "parameter description"
                        },
                        "examples": [
                            {"input": "example input", "output": "expected output"}
                        ]
                    }
                ],
                "examples": [
                    {
                        "query": "example user query",
                        "thoughts": [
                            "step 1 thought",
                            "step 2 thought"
                        ],
                        "actions": [
                            {"tool": "tool name", "input": "tool input"}
                        ],
                        "observations": [
                            "result from tool call"
                        ],
                        "final_answer": "example final answer"
                    }
                ]
            }
            """
        elif framework == "agno":
            return """
            You are an expert at creating AI agents using the Agno framework. Based on the user's request,
            suggest appropriate agents, their roles, tools, and tasks. 
            
            CRITICAL REQUIREMENTS:
            1. Create specialized agents with distinct roles and expertise
            2. ALWAYS assign the most appropriate agent to each task based on their role/expertise
            3. Each task must have an "agent" field with the exact agent name
            4. Match agent specialization to task requirements
            
            Process Types:
            - Sequential: Tasks executed one after another in order
            
            Format your response as JSON with this structure:
            {
                "model_id": "model name (e.g., gpt-4o)",
                "process": "sequential",
                "agents": [
                    {
                        "name": "agent_name",
                        "role": "specific specialized role",
                        "goal": "clear specific goal",
                        "backstory": "relevant professional backstory",
                        "tools": ["relevant_tool1", "relevant_tool2"],
                        "verbose": true,
                        "allow_delegation": true/false
                    }
                ],
                "tasks": [
                    {
                        "name": "task_name",
                        "description": "detailed task description",
                        "tools": ["required tools for this task"],
                        "agent": "exact_agent_name_from_above",
                        "expected_output": "specific expected output"
                    }
                ]
            }
            """
        else:
            return """
            You are an expert at creating AI research assistants. Based on the user's request,
            suggest appropriate agents, their roles, tools, and tasks.
            """

    def _get_default_config(self, framework: str) -> Dict[str, Any]:
        """
        Get a default configuration for the specified framework.
        
        Args:
            framework: The agent framework to use
            
        Returns:
            A default configuration dictionary
        """
        if framework == "crewai":
            return {
                "process": "sequential",  # Default to sequential
                "agents": [
                    {
                        "name": "research_specialist",
                        "role": "Research Specialist",
                        "goal": "Conduct thorough research and gather information",
                        "backstory": "Expert researcher with years of experience in data gathering and analysis",
                        "tools": ["search_tool", "web_scraper"],
                        "verbose": True,
                        "allow_delegation": False
                    },
                    {
                        "name": "content_writer",
                        "role": "Content Writer",
                        "goal": "Create clear and comprehensive written content",
                        "backstory": "Professional writer skilled in creating engaging and informative content",
                        "tools": ["writing_tool", "grammar_checker"],
                        "verbose": True,
                        "allow_delegation": False
                    }
                ],
                "tasks": [
                    {
                        "name": "research_task",
                        "description": "Gather information and conduct research on the given topic",
                        "tools": ["search_tool"],
                        "agent": "research_specialist",
                        "expected_output": "Comprehensive research findings and data"
                    },
                    {
                        "name": "writing_task",
                        "description": "Create written content based on research findings",
                        "tools": ["writing_tool"],
                        "agent": "content_writer",
                        "expected_output": "Well-written content document"
                    }
                ]
            }
        elif framework == "crewai-flow":
            return {
                "process": "sequential",  # Default to sequential
                "agents": [
                    {
                        "name": "research_specialist",
                        "role": "Research Specialist",
                        "goal": "Conduct thorough research and gather information",
                        "backstory": "Expert researcher with years of experience in data gathering and analysis",
                        "tools": ["search_tool", "web_scraper"],
                        "verbose": True,
                        "allow_delegation": False
                    },
                    {
                        "name": "content_writer",
                        "role": "Content Writer",
                        "goal": "Create clear and comprehensive written content",
                        "backstory": "Professional writer skilled in creating engaging and informative content",
                        "tools": ["writing_tool", "grammar_checker"],
                        "verbose": True,
                        "allow_delegation": False
                    }
                ],
                "tasks": [
                    {
                        "name": "research_task",
                        "description": "Gather information and conduct research on the given topic",
                        "tools": ["search_tool"],
                        "agent": "research_specialist",
                        "expected_output": "Comprehensive research findings and data"
                    },
                    {
                        "name": "writing_task",
                        "description": "Create written content based on research findings",
                        "tools": ["writing_tool"],
                        "agent": "content_writer",
                        "expected_output": "Well-written content document"
                    }
                ]
            }
        
        elif framework == "langgraph":
            return {
                "agents": [{
                    "name": "default_assistant",
                    "role": "General Assistant",
                    "goal": "Help with basic tasks",
                    "tools": ["basic_tool"],
                    "llm": "gpt-4.1-mini"
                }],
                "nodes": [{
                    "name": "process_input",
                    "description": "Process user input",
                    "agent": "default_assistant"
                }],
                "edges": [{
                    "source": "process_input",
                    "target": "END",
                    "condition": "task completed"
                }]
            }
        elif framework == "react":
            return {
                "agents": [{
                    "name": "default_assistant",
                    "role": "General Assistant",
                    "goal": "Help with basic tasks",
                    "tools": ["basic_tool"],
                    "llm": "gpt-4.1-mini"
                }],
                "tools": [{
                    "name": "basic_tool",
                    "description": "A basic utility tool",
                    "parameters": {"input": "User input to process"}
                }],
                # Must stay JSON-serialisable: `--format json` runs json.dumps on this.
                "examples": [{
                    "query": "Summarise the latest AI research",
                    "thought": "I should search for recent papers",
                    "action": "basic_tool",
                    "observation": "Found 3 relevant papers",
                    "final_answer": "Here are the latest AI papers..."
                }]
            }
        elif framework == "react-lcel":
            return {
                "agents": [{
                    "name": "default_assistant",
                    "role": "General Assistant",
                    "goal": "Help with multi-step tasks",
                    "tools": ["basic_tool"],
                    "llm": "gpt-4.1-mini"
                }],
                "tools": [{
                    "name": "basic_tool",
                    "description": "A basic utility tool",
                    "parameters": {"input": "User input to process"},
                    "examples": [{"input": "search cats", "output": "cat info"}]
                }],
                "examples": [{
                    "query": "Find trending AI research papers",
                    "thoughts": [
                        "I should search for trending AI papers",
                        "I should summarize the findings"
                    ],
                    "actions": [
                        {"tool": "basic_tool", "input": "trending AI papers"}
                    ],
                    "observations": [
                        "Found 3 relevant papers"
                    ],
                    "final_answer": "Here are the latest AI papers..."
                }]
            }
        elif framework == "agno":
            return {
                "model_id": "gpt-4o",
                "process": "sequential",  # Default to sequential
                "agents": [
                    {
                        "name": "research_specialist",
                        "role": "Research Specialist",
                        "goal": "Conduct thorough research and gather information",
                        "backstory": "Expert researcher with years of experience in data gathering and analysis",
                        "tools": ["search_tool", "web_scraper"],
                        "verbose": True,
                        "allow_delegation": False
                    },
                    {
                        "name": "content_writer",
                        "role": "Content Writer",
                        "goal": "Create clear and comprehensive written content",
                        "backstory": "Professional writer skilled in creating engaging and informative content",
                        "tools": ["writing_tool", "grammar_checker"],
                        "verbose": True,
                        "allow_delegation": False
                    }
                ],
                "tasks": [
                    {
                        "name": "research_task",
                        "description": "Gather information and conduct research on the given topic",
                        "tools": ["search_tool"],
                        "agent": "research_specialist",
                        "expected_output": "Comprehensive research findings and data"
                    },
                    {
                        "name": "writing_task",
                        "description": "Create written content based on research findings",
                        "tools": ["writing_tool"],
                        "agent": "content_writer",
                        "expected_output": "Well-written content document"
                    }
                ]
            }

        else:
            return {}
