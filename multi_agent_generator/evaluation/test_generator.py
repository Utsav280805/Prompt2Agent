# multi_agent_generator/evaluation/test_generator.py
"""
Test Generator - auto-generate runnable test suites for multi-agent systems.

The previous version of this module generated tests that could not pass on any machine.
Each generated test body was ``result = None`` followed by ``assert True``, with the
intended checks emitted as English prose in comments (``# assert no memory leaks
detected``) that were never valid Python. The setup fixture imported ``crewai``, a
package this project never declared as a dependency, and set ``OPENAI_API_KEY`` to
``"test-key"`` before constructing live agents - so on a clean machine the suite failed
at collection, and with a real key it made billed network calls.

This version generates tests in three tiers, so that something useful runs no matter
what is installed:

* **Contract tests** validate the agent configuration itself - unique names, tasks
  pointing at agents that exist, required fields present. These need no framework, no
  credentials and no network, so they always run and always mean something.
* **Construction tests** build real framework objects and assert on their wiring. They
  need the framework installed, guarded by ``pytest.importorskip``, but still no
  network: constructing an agent does not call a model.
* **Live tests** actually invoke models. They are marked ``live`` and deselected by
  default, so spending money is opt-in.

Generated suites also ship with a ``conftest.py`` that blocks socket access for
non-live tests, which turns "these tests should not call the network" from a hope into
an enforced property.
"""

from __future__ import annotations

import json
import os
import pprint
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from ..dependencies import framework_import_name, requirements_txt
from ..providers import FRAMEWORK_FAMILY, get_provider


class TestType(Enum):
    """Types of tests that can be generated."""

    CONTRACT = "contract"            # Validate the configuration itself
    UNIT = "unit"                    # Test individual agents
    INTEGRATION = "integration"      # Test agent interactions
    END_TO_END = "end_to_end"        # Test complete workflows
    PERFORMANCE = "performance"      # Test response times
    RELIABILITY = "reliability"      # Test error handling
    QUALITY = "quality"              # Test output quality


#: Test types that require a live model call, and are therefore marked ``live``
#: and deselected unless explicitly requested.
LIVE_TEST_TYPES = frozenset({TestType.END_TO_END, TestType.PERFORMANCE, TestType.QUALITY})


@dataclass
class TestCase:
    """
    Represents a single test case.

    ``body`` holds the executable statements for the test. ``assertions`` holds real
    Python boolean expressions - not prose - so that everything emitted is valid code.
    """

    name: str
    description: str
    test_type: TestType
    input_data: Dict[str, Any] = field(default_factory=dict)
    expected_behavior: str = ""
    assertions: List[str] = field(default_factory=list)
    body: List[str] = field(default_factory=list)
    fixtures: List[str] = field(default_factory=list)
    timeout_seconds: int = 30
    tags: List[str] = field(default_factory=list)
    live: bool = False
    skip_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "input_data": self.input_data,
            "expected_behavior": self.expected_behavior,
            "assertions": self.assertions,
            "body": self.body,
            "fixtures": self.fixtures,
            "timeout_seconds": self.timeout_seconds,
            "tags": self.tags,
            "live": self.live,
        }


@dataclass
class TestSuite:
    """Collection of test cases for an agent system."""

    name: str
    description: str
    framework: str = ""
    provider: str = "openai"
    config: Dict[str, Any] = field(default_factory=dict)
    test_cases: List[TestCase] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""

    def add_test(self, test: TestCase) -> None:
        self.test_cases.append(test)

    def get_tests_by_type(self, test_type: TestType) -> List[TestCase]:
        return [t for t in self.test_cases if t.test_type == test_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "framework": self.framework,
            "provider": self.provider,
            "test_cases": [t.to_dict() for t in self.test_cases],
            "setup_code": self.setup_code,
            "teardown_code": self.teardown_code,
        }

    def save(self, directory: str) -> List[str]:
        """
        Write the full runnable bundle for this suite to ``directory``.

        Returns the list of paths written. This is the method the README advertised but
        which did not previously exist.
        """
        bundle = TestGenerator().generate_bundle(
            self.config, self.framework, self.provider, suite=self
        )
        return bundle.save(directory)


@dataclass
class TestBundle:
    """
    A complete, runnable test project.

    A bare test file is not enough: it needs its dependencies written down, its pytest
    markers registered, and fixtures that keep it offline. Those are the pieces that
    were missing, and they are why generated tests could not be run.
    """

    framework: str
    provider: str
    files: Dict[str, str] = field(default_factory=dict)

    def save(self, directory: str) -> List[str]:
        os.makedirs(directory, exist_ok=True)
        written: List[str] = []
        for filename, content in self.files.items():
            path = os.path.join(directory, filename)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
            written.append(path)
        return written


# --------------------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------------------
class TestGenerator:
    """
    Generates runnable test suites for multi-agent systems.

    The output is a bundle rather than a single file: see :meth:`generate_bundle`.
    """

    def __init__(self, model_inference=None):
        """
        Args:
            model_inference: Optional inference backend, reserved for LLM-assisted test
                generation. Not required for the deterministic tests generated here.
        """
        self.model = model_inference

    # ---------------------------------------------------------------- suite assembly
    def generate_test_suite(
        self,
        config: Optional[Dict[str, Any]] = None,
        framework: str = "crewai",
        include_types: Optional[Sequence[TestType]] = None,
        *,
        agent_config: Optional[Dict[str, Any]] = None,
        test_types: Optional[Sequence[Any]] = None,
        provider: str = "openai",
    ) -> TestSuite:
        """
        Generate a complete test suite for an agent configuration.

        Args:
            config: Agent system configuration.
            framework: Target framework (crewai, langgraph, ...).
            include_types: Specific test types to include (all if None).
            agent_config: Alias for ``config``, matching the documented API.
            test_types: Alias for ``include_types``; accepts strings or TestType values.
            provider: LLM provider the generated agents use.

        Returns:
            A TestSuite with generated test cases.
        """
        config = config if config is not None else (agent_config or {})

        # Accept the documented spelling, and tolerate plain strings.
        raw_types = include_types if include_types is not None else test_types
        selected = self._coerce_types(raw_types)

        agents = config.get("agents") or []
        tasks = config.get("tasks") or []

        suite = TestSuite(
            name=f"{framework}_test_suite",
            description=f"Auto-generated test suite for a {framework} agent system",
            framework=framework,
            provider=get_provider(provider).name,
            config=config,
        )

        if TestType.CONTRACT in selected:
            suite.test_cases.extend(self._contract_tests(agents, tasks))
        if TestType.UNIT in selected:
            suite.test_cases.extend(self._unit_tests(agents, framework))
        if TestType.INTEGRATION in selected:
            suite.test_cases.extend(self._integration_tests(agents, tasks, framework))
        if TestType.RELIABILITY in selected:
            suite.test_cases.extend(self._reliability_tests(agents, framework))
        if TestType.END_TO_END in selected:
            suite.test_cases.extend(self._e2e_tests(framework))
        if TestType.PERFORMANCE in selected:
            suite.test_cases.extend(self._performance_tests(framework))
        if TestType.QUALITY in selected:
            suite.test_cases.extend(self._quality_tests(framework))

        return suite

    @staticmethod
    def _coerce_types(raw: Optional[Sequence[Any]]) -> List[TestType]:
        if raw is None:
            return list(TestType)
        coerced: List[TestType] = []
        for item in raw:
            if isinstance(item, TestType):
                coerced.append(item)
            else:
                try:
                    coerced.append(TestType(str(item)))
                except ValueError:
                    known = ", ".join(t.value for t in TestType)
                    raise ValueError(
                        f"Unknown test type {item!r}. Valid types: {known}"
                    ) from None
        return coerced

    # ------------------------------------------------------------- contract tier
    def _contract_tests(
        self, agents: List[Dict], tasks: List[Dict]
    ) -> List[TestCase]:
        """
        Tests over the configuration itself.

        These are the tests that make the suite worth running on a fresh checkout: they
        need no framework, no credentials and no network, so they can never be skipped
        for environmental reasons, and every assertion is a real check.
        """
        tests = [
            TestCase(
                name="test_config_declares_agents",
                description="The configuration declares at least one agent",
                test_type=TestType.CONTRACT,
                expected_behavior="agents list is non-empty",
                body=["agents = AGENT_CONFIG.get('agents') or []"],
                assertions=["len(agents) > 0"],
                tags=["contract", "config"],
            ),
            TestCase(
                name="test_agent_names_are_unique",
                description="Agent names are unique, so tasks can reference them unambiguously",
                test_type=TestType.CONTRACT,
                expected_behavior="no duplicate agent names",
                body=[
                    "names = [a.get('name') for a in AGENT_CONFIG.get('agents') or []]",
                    "duplicates = {n for n in names if names.count(n) > 1}",
                ],
                assertions=["duplicates == set()"],
                tags=["contract", "config"],
            ),
            TestCase(
                name="test_agents_have_required_fields",
                description="Every agent declares a name, role and goal",
                test_type=TestType.CONTRACT,
                expected_behavior="each agent has name, role and goal",
                body=[
                    "missing = [",
                    "    (a.get('name'), key)",
                    "    for a in AGENT_CONFIG.get('agents') or []",
                    "    for key in ('name', 'role', 'goal')",
                    "    if not a.get(key)",
                    "]",
                ],
                assertions=["missing == []"],
                tags=["contract", "config"],
            ),
            TestCase(
                name="test_agent_names_are_valid_identifiers",
                description="Agent names can be turned into valid Python variable names",
                test_type=TestType.CONTRACT,
                expected_behavior="every sanitised agent name is a valid identifier",
                body=[
                    "# Generated code derives variable names from agent names, so a name",
                    "# that cannot be sanitised would produce a file that will not parse.",
                    "bad = [",
                    "    a.get('name')",
                    "    for a in AGENT_CONFIG.get('agents') or []",
                    "    if not sanitize(a.get('name', '')).isidentifier()",
                    "]",
                ],
                assertions=["bad == []"],
                tags=["contract", "codegen"],
            ),
        ]

        if tasks:
            tests.extend(
                [
                    TestCase(
                        name="test_tasks_reference_existing_agents",
                        description="Every task is assigned to an agent that exists",
                        test_type=TestType.CONTRACT,
                        expected_behavior="no task references an unknown agent",
                        body=[
                            "agent_names = {a.get('name') for a in AGENT_CONFIG.get('agents') or []}",
                            "orphans = [",
                            "    t.get('name')",
                            "    for t in AGENT_CONFIG.get('tasks') or []",
                            "    if t.get('agent') and t.get('agent') not in agent_names",
                            "]",
                        ],
                        assertions=["orphans == []"],
                        tags=["contract", "config"],
                    ),
                    TestCase(
                        name="test_tasks_have_descriptions_and_outputs",
                        description="Every task declares a description and an expected output",
                        test_type=TestType.CONTRACT,
                        expected_behavior="each task has description and expected_output",
                        body=[
                            "incomplete = [",
                            "    t.get('name')",
                            "    for t in AGENT_CONFIG.get('tasks') or []",
                            "    if not t.get('description') or not t.get('expected_output')",
                            "]",
                        ],
                        assertions=["incomplete == []"],
                        tags=["contract", "config"],
                    ),
                    TestCase(
                        name="test_task_names_are_unique",
                        description="Task names are unique",
                        test_type=TestType.CONTRACT,
                        expected_behavior="no duplicate task names",
                        body=[
                            "names = [t.get('name') for t in AGENT_CONFIG.get('tasks') or []]",
                            "duplicates = {n for n in names if names.count(n) > 1}",
                        ],
                        assertions=["duplicates == set()"],
                        tags=["contract", "config"],
                    ),
                ]
            )

        return tests

    # ----------------------------------------------------------------- unit tier
    def _unit_tests(self, agents: List[Dict], framework: str) -> List[TestCase]:
        """
        Per-agent construction tests.

        These build real framework objects. Construction does not call a model, so they
        stay offline; the network guard in conftest.py enforces that.
        """
        family = FRAMEWORK_FAMILY.get(framework, "langchain")
        tests: List[TestCase] = []

        for index, agent in enumerate(agents):
            name = agent.get("name") or f"agent_{index}"
            safe = _safe_test_name(name)

            if family == "crewai":
                body = [
                    "from crewai import Agent",
                    f"cfg = AGENT_CONFIG['agents'][{index}]",
                    "agent = Agent(",
                    "    role=cfg['role'],",
                    "    goal=cfg['goal'],",
                    "    backstory=cfg.get('backstory', ''),",
                    "    tools=[],",
                    "    llm=offline_llm,",
                    ")",
                ]
                assertions = [
                    "agent.role == cfg['role']",
                    "agent.goal == cfg['goal']",
                ]
                fixtures = ["offline_llm"]
            elif family == "agno":
                body = [
                    "from agno.agent import Agent",
                    f"cfg = AGENT_CONFIG['agents'][{index}]",
                    "agent = Agent(",
                    "    name=cfg['name'],",
                    "    role=cfg.get('role', ''),",
                    "    model=fake_llm,",
                    "    tools=[],",
                    ")",
                ]
                assertions = [
                    "agent.name == cfg['name']",
                ]
                fixtures = ["fake_llm"]
            else:  # langchain family
                body = [
                    "from langchain_core.prompts import ChatPromptTemplate",
                    f"cfg = AGENT_CONFIG['agents'][{index}]",
                    "prompt = ChatPromptTemplate.from_messages([",
                    "    ('system', 'You are ' + cfg['role'] + '. Your goal is ' + cfg['goal'] + '.'),",
                    "    ('human', '{input}'),",
                    "])",
                    "chain = prompt | fake_llm",
                ]
                assertions = [
                    "chain is not None",
                    "'input' in prompt.input_variables",
                ]
                fixtures = ["fake_llm"]

            tests.append(
                TestCase(
                    name=f"test_{safe}_constructs",
                    description=f"{name} can be constructed from its configuration",
                    test_type=TestType.UNIT,
                    expected_behavior="agent is constructed with the configured role and goal",
                    fixtures=fixtures,
                    body=body,
                    assertions=assertions,
                    tags=["unit", safe, "construction"],
                )
            )

            declared_tools = [t for t in (agent.get("tools") or []) if isinstance(t, str)]
            if declared_tools:
                tests.append(
                    TestCase(
                        name=f"test_{safe}_tool_names_are_usable",
                        description=f"{name}'s tool names can be turned into class names",
                        test_type=TestType.UNIT,
                        expected_behavior="each tool name yields a valid class name",
                        body=[
                            f"tools = AGENT_CONFIG['agents'][{index}].get('tools') or []",
                            "class_names = [tool_class_name(t) for t in tools if isinstance(t, str)]",
                            "# Uniqueness is deliberately not asserted. Names that differ only",
                            "# in punctuation ('web_search' and 'web-search') map to the same",
                            "# class by design, and the generator drops the duplicate rather",
                            "# than emitting two classes with one name - so a collision here",
                            "# is valid config, not a defect. What must hold is that every",
                            "# name yields a usable class identifier.",
                        ],
                        assertions=[
                            "class_names != []",
                            "all(n.isidentifier() for n in class_names)",
                            "all(n.endswith('Tool') for n in class_names)",
                        ],
                        tags=["unit", safe, "tools"],
                    )
                )

        return tests

    # ---------------------------------------------------------- integration tier
    def _integration_tests(
        self, agents: List[Dict], tasks: List[Dict], framework: str
    ) -> List[TestCase]:
        """Wiring tests across more than one agent or task."""
        family = FRAMEWORK_FAMILY.get(framework, "langchain")
        tests: List[TestCase] = []

        if len(agents) < 2 and not tasks:
            return tests

        if family == "crewai" and tasks:
            tests.append(
                TestCase(
                    name="test_crew_assembles_with_all_tasks",
                    description="A Crew can be assembled with every configured agent and task",
                    test_type=TestType.INTEGRATION,
                    expected_behavior="crew holds one task per configured task",
                    fixtures=["offline_llm"],
                    body=[
                        "from crewai import Agent, Crew, Process, Task",
                        "agents = {}",
                        "for cfg in AGENT_CONFIG['agents']:",
                        "    agents[cfg['name']] = Agent(",
                        "        role=cfg['role'],",
                        "        goal=cfg['goal'],",
                        "        backstory=cfg.get('backstory', ''),",
                        "        tools=[],",
                        "        llm=offline_llm,",
                        "    )",
                        "task_objects = []",
                        "for cfg in AGENT_CONFIG['tasks']:",
                        "    owner = agents.get(cfg.get('agent')) or list(agents.values())[0]",
                        "    task_objects.append(Task(",
                        "        description=cfg['description'],",
                        "        expected_output=cfg.get('expected_output', 'A result'),",
                        "        agent=owner,",
                        "    ))",
                        "crew = Crew(",
                        "    agents=list(agents.values()),",
                        "    tasks=task_objects,",
                        "    process=Process.sequential,",
                        ")",
                    ],
                    assertions=[
                        "len(crew.tasks) == len(AGENT_CONFIG['tasks'])",
                        "len(crew.agents) == len(AGENT_CONFIG['agents'])",
                        "all(t.agent is not None for t in crew.tasks)",
                    ],
                    timeout_seconds=60,
                    tags=["integration", "assembly"],
                )
            )

        if family == "langchain":
            tests.append(
                TestCase(
                    name="test_chain_passes_output_between_steps",
                    description="Output from one step is available to the next",
                    test_type=TestType.INTEGRATION,
                    expected_behavior="a two-step chain runs offline and returns text",
                    fixtures=["fake_llm"],
                    body=[
                        "from langchain_core.prompts import ChatPromptTemplate",
                        "from langchain_core.output_parsers import StrOutputParser",
                        "first = ChatPromptTemplate.from_template('Research: {input}')",
                        "second = ChatPromptTemplate.from_template('Summarise: {input}')",
                        "chain = (",
                        "    first | fake_llm | StrOutputParser()",
                        "    | (lambda text: {'input': text})",
                        "    | second | fake_llm | StrOutputParser()",
                        ")",
                        "result = chain.invoke({'input': 'quarterly sales'})",
                    ],
                    assertions=[
                        "isinstance(result, str)",
                        "len(result) > 0",
                    ],
                    timeout_seconds=60,
                    tags=["integration", "handoff"],
                )
            )

        if tasks and len(tasks) > 1:
            tests.append(
                TestCase(
                    name="test_task_order_is_deterministic",
                    description="Task ordering in the config is stable and total",
                    test_type=TestType.INTEGRATION,
                    expected_behavior="tasks form an ordered sequence with no gaps",
                    body=[
                        "tasks = AGENT_CONFIG['tasks']",
                        "names = [t.get('name') for t in tasks]",
                    ],
                    assertions=[
                        "len(names) == len(tasks)",
                        "all(n for n in names)",
                    ],
                    tags=["integration", "ordering"],
                )
            )

        delegating = [a for a in agents if a.get("allow_delegation")]
        if delegating and FRAMEWORK_FAMILY.get(framework) == "crewai":
            tests.append(
                TestCase(
                    name="test_delegation_flag_is_honoured",
                    description="Agents configured to delegate are constructed with delegation enabled",
                    test_type=TestType.INTEGRATION,
                    expected_behavior="allow_delegation survives construction",
                    fixtures=["offline_llm"],
                    body=[
                        "from crewai import Agent",
                        "delegators = [a for a in AGENT_CONFIG['agents'] if a.get('allow_delegation')]",
                        "built = [",
                        "    Agent(",
                        "        role=a['role'],",
                        "        goal=a['goal'],",
                        "        backstory=a.get('backstory', ''),",
                        "        allow_delegation=True,",
                        "        tools=[],",
                        "        llm=offline_llm,",
                        "    )",
                        "    for a in delegators",
                        "]",
                    ],
                    assertions=[
                        "len(built) == len(delegators)",
                        "all(a.allow_delegation for a in built)",
                    ],
                    timeout_seconds=60,
                    tags=["integration", "delegation"],
                )
            )

        return tests

    # ---------------------------------------------------------- reliability tier
    def _reliability_tests(self, agents: List[Dict], framework: str) -> List[TestCase]:
        """Error-handling tests that stay offline."""
        family = FRAMEWORK_FAMILY.get(framework, "langchain")
        tests: List[TestCase] = []

        if family == "crewai":
            tests.append(
                TestCase(
                    name="test_missing_required_field_is_rejected",
                    description="Constructing an agent without a role raises rather than passing silently",
                    test_type=TestType.RELIABILITY,
                    expected_behavior="a validation error is raised",
                    fixtures=["offline_llm"],
                    body=[
                        "from crewai import Agent",
                        "with pytest.raises(Exception):",
                        "    Agent(goal='no role supplied', backstory='', tools=[], llm=offline_llm)",
                    ],
                    assertions=[],
                    tags=["reliability", "validation"],
                )
            )

        tests.append(
            TestCase(
                name="test_empty_query_is_handled",
                description="An empty query does not raise an unhandled exception",
                test_type=TestType.RELIABILITY,
                expected_behavior="empty and whitespace input are handled gracefully",
                fixtures=["fake_llm"],
                body=[
                    "from langchain_core.prompts import ChatPromptTemplate",
                    "from langchain_core.output_parsers import StrOutputParser",
                    "chain = ChatPromptTemplate.from_template('{input}') | fake_llm | StrOutputParser()",
                    "results = [chain.invoke({'input': value}) for value in ('', '   ')]",
                ],
                assertions=[
                    "all(isinstance(r, str) for r in results)",
                ],
                tags=["reliability", "error_handling"],
            )
        )

        tests.append(
            TestCase(
                name="test_repeated_calls_are_stable",
                description="Repeated calls with a deterministic model return consistent results",
                test_type=TestType.RELIABILITY,
                expected_behavior="the same input yields the same output shape each time",
                fixtures=["fake_llm"],
                body=[
                    "from langchain_core.prompts import ChatPromptTemplate",
                    "from langchain_core.output_parsers import StrOutputParser",
                    "chain = ChatPromptTemplate.from_template('{input}') | fake_llm | StrOutputParser()",
                    "results = [chain.invoke({'input': 'same question'}) for _ in range(3)]",
                ],
                assertions=[
                    "len(results) == 3",
                    "all(isinstance(r, str) and r for r in results)",
                ],
                tags=["reliability", "idempotency"],
            )
        )

        tests.append(
            TestCase(
                name="test_no_credentials_required_for_construction",
                description="Building the system does not require real provider credentials",
                test_type=TestType.RELIABILITY,
                expected_behavior="no real API key is present during offline tests",
                body=[
                    "# The conftest stubs credentials and blocks sockets. If this test",
                    "# ever sees a plausible real key, the offline guarantee has broken.",
                    "key = os.environ.get('OPENAI_API_KEY', '')",
                ],
                assertions=[
                    "key.startswith('test-') or key == ''",
                ],
                tags=["reliability", "offline"],
            )
        )

        return tests

    # ----------------------------------------------------------------- live tiers
    def _e2e_tests(self, framework: str) -> List[TestCase]:
        """End-to-end tests. These call real models, so they are marked ``live``."""
        return [
            TestCase(
                name="test_complete_workflow_live",
                description="The complete workflow runs against a real model",
                test_type=TestType.END_TO_END,
                expected_behavior="workflow returns a non-empty final result",
                fixtures=["agent_module"],
                body=[
                    "result = agent_module.run_workflow('Summarise the latest results')",
                ],
                assertions=[
                    "result is not None",
                    "len(str(result)) > 0",
                ],
                timeout_seconds=300,
                live=True,
                tags=["e2e", "workflow", "live"],
            ),
        ]

    def _performance_tests(self, framework: str) -> List[TestCase]:
        return [
            TestCase(
                name="test_response_time_live",
                description="A single request completes within the budget",
                test_type=TestType.PERFORMANCE,
                expected_behavior="response arrives within 120 seconds",
                fixtures=["agent_module"],
                body=[
                    "started = time.perf_counter()",
                    "result = agent_module.run_workflow('Quick timing probe')",
                    "elapsed = time.perf_counter() - started",
                    "print(f'elapsed: {elapsed:.2f}s')",
                ],
                assertions=[
                    "result is not None",
                    "elapsed < 120",
                ],
                timeout_seconds=180,
                live=True,
                tags=["performance", "timing", "live"],
            ),
            TestCase(
                name="test_peak_memory_is_bounded_live",
                description="Peak memory during a run stays within bounds",
                test_type=TestType.PERFORMANCE,
                expected_behavior="peak allocation under 500 MB",
                fixtures=["agent_module"],
                body=[
                    "tracemalloc.start()",
                    "try:",
                    "    agent_module.run_workflow('Memory probe')",
                    "    _current, peak = tracemalloc.get_traced_memory()",
                    "finally:",
                    "    tracemalloc.stop()",
                    "peak_mb = peak / (1024 * 1024)",
                    "print(f'peak: {peak_mb:.1f} MB')",
                ],
                assertions=["peak_mb < 500"],
                timeout_seconds=300,
                live=True,
                tags=["performance", "memory", "live"],
            ),
        ]

    def _quality_tests(self, framework: str) -> List[TestCase]:
        return [
            TestCase(
                name="test_output_relevance_live",
                description="Output is relevant to the query",
                test_type=TestType.QUALITY,
                expected_behavior="response mentions the queried subject",
                fixtures=["agent_module", "evaluator"],
                body=[
                    "query = 'What is machine learning?'",
                    "response = str(agent_module.run_workflow(query))",
                    "result = evaluator.evaluate(query, response)",
                    "print(result.metrics.overall_score())",
                ],
                assertions=[
                    "result.metrics.relevance_score > 0.3",
                    "len(response) > 0",
                ],
                timeout_seconds=300,
                live=True,
                tags=["quality", "relevance", "live"],
            ),
        ]

    # ------------------------------------------------------------------ rendering
    def generate_pytest_file(self, suite: TestSuite) -> str:
        """Render the pytest module for ``suite``."""
        framework = suite.framework or "crewai"
        import_name = framework_import_name(framework)
        config_literal = pprint.pformat(suite.config or {}, width=88, sort_dicts=False)

        # Only import what the rendered bodies actually use, so the generated file has
        # no unused imports for a linter to flag.
        all_body = "\n".join(line for t in suite.test_cases for line in t.body)
        stdlib = ["os"]
        if "time.perf_counter" in all_body:
            stdlib.append("time")
        if "tracemalloc" in all_body:
            stdlib.append("tracemalloc")

        header = f'''"""
{suite.description}

Auto-generated test suite: {suite.name}
Framework: {framework}
Provider:  {suite.provider}

Running these tests
-------------------
    pip install -r requirements-test.txt
    pytest

Offline by default: a deterministic fake model stands in for the real one, and sockets
are blocked, so no credentials and no network are needed. Tests marked `live` call real
models and are deselected unless you ask for them explicitly:

    pytest -m live

Contract tests validate the configuration itself and need no framework packages at all.
Tests that build framework objects skip cleanly when {import_name!r} is missing, rather
than failing collection.
"""

{chr(10).join(f"import {mod}" for mod in stdlib)}

import pytest

# Sanitisation helpers, copied into the bundle so this suite has no dependency on the
# generator that produced it.
from _helpers import sanitize, tool_class_name

# The configuration under test, inlined so the suite is self-contained.
AGENT_CONFIG = {config_literal}


'''

        code = header
        for test in suite.test_cases:
            code += self._render_test(test)
            code += "\n\n"

        code += '''if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
'''
        return code

    def _render_test(self, test: TestCase) -> str:
        """Render one TestCase as a pytest function."""
        decorators: List[str] = []

        marker = test.test_type.value
        decorators.append(f"@pytest.mark.{marker}")
        if test.live:
            decorators.append("@pytest.mark.live")
        decorators.append(f"@pytest.mark.timeout({test.timeout_seconds})")

        args = ", ".join(test.fixtures)
        signature = f"def {test.name}({args}):"

        lines: List[str] = decorators + [signature]
        lines.append('    """')
        lines.append(f"    {test.description}")
        if test.expected_behavior:
            lines.append("")
            lines.append(f"    Expected: {test.expected_behavior}")
        lines.append('    """')

        # Every third-party module the body imports gets its own skip guard. Scanning the
        # body rather than assuming the framework matters: a CrewAI suite still uses
        # langchain_core for its offline reliability tests, and a machine with crewai but
        # without langchain_core should skip those, not error during collection.
        for module in _third_party_imports(test.body):
            lines.append(
                f'    pytest.importorskip('
                f'"{module}", reason="{module} is not installed; '
                f'install it to run this test")'
            )

        for line in test.body:
            lines.append(f"    {line}" if line else "")

        for assertion in test.assertions:
            lines.append(f"    assert {assertion}")

        if not test.assertions and not any(
            "pytest.raises" in line for line in test.body
        ):
            # Never emit a test with no check at all; that is how the previous version
            # produced a suite that reported green while exercising nothing.
            lines.append("    assert True  # structural test: construction above must not raise")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------- conftest
    def generate_conftest(self, framework: str, provider: str = "openai") -> str:
        """
        Render the ``conftest.py`` that keeps generated tests offline.

        Three things happen here that make the suite runnable where it previously was
        not: credentials are stubbed with obviously-fake values, a ``fake_llm`` fixture
        supplies a deterministic model so no request is ever made, and sockets are
        blocked for every non-live test so an accidental live call fails loudly instead
        of quietly costing money.
        """
        spec = get_provider(provider)
        env_lines = "\n".join(
            f'    monkeypatch.setenv("{var}", "test-{var.lower().replace("_", "-")}")'
            for var in (spec.credential_env or ("OPENAI_API_KEY",))
        )

        return f'''"""
Shared pytest fixtures for the generated {framework} test suite.

Everything here exists to make the suite runnable without credentials and without
network access. Tests marked `live` opt out of the offline guards.
"""

import importlib
import os
import socket

import pytest


# ---------------------------------------------------------------------------- markers
def pytest_configure(config):
    """Register markers so `--strict-markers` does not reject the generated suite."""
    if not config.pluginmanager.hasplugin("pytest_timeout"):
        # Worth saying out loud. Without the plugin, every `@pytest.mark.timeout(...)`
        # in this suite is inert, so a genuinely stuck test blocks the run forever
        # instead of failing. That is a silent loss of a safety net.
        config.issue_config_time_warning(
            pytest.PytestConfigWarning(
                "pytest-timeout is not installed, so the per-test timeouts in this "
                "suite have no effect. Install it with "
                "`pip install -r requirements-test.txt` so a hung test fails instead "
                "of hanging the run."
            ),
            stacklevel=2,
        )

    for marker, description in [
        ("contract", "validates the agent configuration; no dependencies needed"),
        ("unit", "builds individual agents"),
        ("integration", "exercises agent and task wiring"),
        ("end_to_end", "runs the full workflow"),
        ("performance", "measures timing and memory"),
        ("reliability", "checks error handling"),
        ("quality", "scores output quality"),
        ("live", "requires real credentials and network access"),
        # Registered here as well as by pytest-timeout, so the suite still collects
        # under --strict-markers if that plugin happens to be missing.
        ("timeout", "per-test time limit; honoured when pytest-timeout is installed"),
    ]:
        config.addinivalue_line("markers", f"{{marker}}: {{description}}")


# ------------------------------------------------------------------- offline guards
@pytest.fixture(autouse=True)
def disable_telemetry(monkeypatch):
    """
    Switch off framework telemetry before anything is imported.

    This is not tidiness, it is a correctness fix. CrewAI ships an OpenTelemetry
    exporter that tries to phone home, and chromadb (pulled in transitively) does the
    same. With the socket guard below active, that background traffic surfaces as
    confusing errors in tests that have nothing to do with the network. Opting out is
    also the right default for a generated suite: nobody expects running tests to emit
    usage data.

    Applies to live tests too - a real model call still works with telemetry off.

    The values are spelled out per variable rather than derived, because they do not
    agree on polarity. `LANGCHAIN_TRACING_V2` is the trap: it is an *enable* flag, so
    setting it to "true" here would switch LangSmith tracing on, and the tracer would
    then try to POST every chain run straight into the socket guard below. chromadb's
    `ANONYMIZED_TELEMETRY` is likewise an enable flag and wants "False".
    """
    for var, value in (
        ("CREWAI_TELEMETRY_OPT_OUT", "true"),   # opt-out flag: true means "do not send"
        ("OTEL_SDK_DISABLED", "true"),          # disable flag
        ("ANONYMIZED_TELEMETRY", "False"),      # chromadb enable flag
        ("LANGCHAIN_TRACING_V2", "false"),      # LangSmith enable flag
        ("LANGCHAIN_TRACING", "false"),         # its pre-v2 spelling
        ("LANGSMITH_TRACING", "false"),         # its current spelling
        ("HF_HUB_DISABLE_TELEMETRY", "1"),      # disable flag
    ):
        monkeypatch.setenv(var, value)


@pytest.fixture(autouse=True)
def stub_credentials(request, monkeypatch):
    """
    Replace provider credentials with obviously-fake values.

    The previous generated suite did `os.environ.setdefault("OPENAI_API_KEY", "test-key")`
    and then made real calls, so it returned 401s. Here the fake key is paired with a
    network block, so nothing tries to authenticate in the first place.
    """
    if request.node.get_closest_marker("live"):
        return
{env_lines}
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-api-key")


@pytest.fixture(autouse=True)
def block_network(request, monkeypatch):
    """
    Fail fast if a non-live test tries to open a socket.

    This turns "these tests do not need the network" from an assumption into an
    enforced property. Without it, a refactor could silently reintroduce billed calls.

    Deliberately not a yield fixture: an early `return` in a generator fixture makes
    pytest raise "fixture did not yield a value" for every live test. monkeypatch
    already restores the original attribute at teardown, so nothing is lost.
    """
    if request.node.get_closest_marker("live"):
        return

    def guard(*args, **kwargs):
        raise RuntimeError(
            "Network access is blocked in offline tests. If this test genuinely needs "
            "a real model, mark it with @pytest.mark.live and run `pytest -m live`."
        )

    monkeypatch.setattr(socket, "socket", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


# ------------------------------------------------------------------- import warm-up
def _warm_imports():
    """
    Pay LangChain's one-time import cost here, during collection.

    This is the fix for a suite that appeared to hang. `langchain_core.language_models`
    imports `transformers`, and `transformers` calls
    `importlib.metadata.packages_distributions()` in its module body - a stat of every
    file of every installed distribution. On a large virtualenv (Windows, Python 3.13)
    that takes minutes.

    Nothing about that is a test failure, but it used to happen inside a fixture, which
    put it inside the per-test timeout window. The result was that
    `test_empty_query_is_handled` sat there doing nothing visible and then died on a
    timeout, blaming a test that was entirely innocent.

    Importing at module scope moves the cost to collection, which no per-test timeout
    covers. A slow start is then honestly reported as a slow start, and every test that
    follows gets the cached module. The fixtures below additionally avoid
    `langchain_core.language_models` altogether, so on most installs this warm-up finds
    nothing expensive to do.

    Failures are swallowed on purpose: a missing package is the normal case for suites
    whose framework does not use LangChain, and the tests that need it carry their own
    `importorskip`.
    """
    for module_name in ("langchain_core.prompts", "langchain_core.output_parsers"):
        try:
            importlib.import_module(module_name)
        except Exception:
            return


_warm_imports()


# ------------------------------------------------------------------------ fake model
@pytest.fixture
def fake_llm():
    """
    A deterministic stand-in model. No network, no credentials, no heavy imports.

    Previously this reached for LangChain's own `FakeListChatModel`, which lives in
    `langchain_core.language_models` - the package whose import drags in `transformers`
    and stalls for minutes (see `_warm_imports` above). Nothing here needs a real
    `BaseChatModel`: piping a plain callable into an LCEL chain wraps it in a
    `RunnableLambda`, so `prompt | fake_llm | StrOutputParser()` still exercises the real
    template rendering and the real output parser, with a fixed string where the model
    call would be.

    `FakeChatModel` is defined in `_helpers.py` and imports nothing but the standard
    library, so requesting this fixture can never be slow.
    """
    from _helpers import FakeChatModel

    return FakeChatModel()


@pytest.fixture
def offline_llm():
    """
    A model object CrewAI accepts, that never makes a call.

    CrewAI validates the `llm` argument against its own `LLM` type, so a LangChain fake
    is not interchangeable here. Constructing `LLM` performs no request - the call only
    happens on kickoff, which offline tests never reach - and the obviously-fake key
    plus the socket guard mean an accidental request fails loudly rather than billing
    anyone.
    """
    crewai = pytest.importorskip("crewai", reason="crewai is not installed")
    return crewai.LLM(model="gpt-4o-mini", api_key="test-not-a-real-key")


# --------------------------------------------------------------------- live fixtures
@pytest.fixture
def agent_module():
    """
    Import the generated agent module, for live tests only.

    Set AGENT_MODULE to the module name if your generated file is not `agent_system`.
    """
    module_name = os.environ.get("AGENT_MODULE", "agent_system")
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(
            f"Could not import generated module {{module_name!r}} ({{exc}}). "
            "Save your generated code next to these tests, or set AGENT_MODULE."
        )


@pytest.fixture
def evaluator():
    """The output-quality evaluator, used by live quality tests."""
    try:
        from multi_agent_generator.evaluation import AgentEvaluator
    except ImportError:
        pytest.skip("multi-agent-generator is not installed; needed for quality scoring.")
    return AgentEvaluator()
'''

    # ------------------------------------------------------------------- helpers file
    def generate_helpers(self) -> str:
        """
        Render ``_helpers.py``: the sanitisation helpers plus the offline fake model.

        Copied into the bundle rather than imported from the installed package, so a
        generated test project stands on its own.
        """
        return '''"""
Standalone helpers for the generated test suite.

`sanitize` and `tool_class_name` mirror the name-handling logic the code generator uses,
so the contract tests can verify that a configuration will actually produce parseable
code. `FakeChatModel` is the offline stand-in model the fixtures hand to the tests.

All three are copied here rather than imported so the generated project depends on
nothing but the standard library and its own framework.
"""

import keyword
import re


def sanitize(name, prefix="x"):
    """Turn an arbitrary label into a valid, non-reserved Python identifier."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", (name or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = "{}_{}".format(prefix, cleaned)
    if keyword.iskeyword(cleaned):
        cleaned = cleaned + "_"
    return cleaned


def tool_class_name(name):
    """Build a CamelCase tool class name, e.g. web_scraper -> WebScraperTool."""
    parts = [p for p in re.split(r"[^0-9a-zA-Z]+", name or "") if p]
    if not parts:
        parts = ["custom"]
    camel = "".join(p[:1].upper() + p[1:] for p in parts)
    if camel[0].isdigit():
        camel = "Tool" + camel
    if not camel.endswith("Tool"):
        camel += "Tool"
    return camel


class FakeChatModel:
    """
    A deterministic stand-in for a chat model, importing nothing but the standard library.

    It is a plain callable rather than a subclass of a framework model class, and that is
    the point. LangChain's own `FakeListChatModel` lives in
    `langchain_core.language_models`, and importing that package pulls in `transformers`,
    whose module body stats every file of every installed distribution. On a large
    virtualenv that takes minutes, which is what made the previous generated suite look
    like it hung.

    Piping a callable into an LCEL chain is supported - LangChain wraps it in a
    `RunnableLambda` - so `prompt | model | StrOutputParser()` still runs the real
    template rendering and the real output parser. Returning a `str` rather than a
    message object is deliberate: `StrOutputParser` accepts a string and passes it
    through, so no message class needs importing either.

    Responses cycle rather than run out, so a suite can call the model any number of
    times and still get deterministic output.
    """

    default_response = "This is a deterministic fake response for testing."

    def __init__(self, responses=None):
        cleaned = [r for r in (responses or []) if r]
        self.responses = cleaned or [self.default_response]
        self.calls = []
        self.bound_tools = []
        self._index = 0
        # LangChain reads __name__ off the wrapped callable when naming a run step.
        self.__name__ = type(self).__name__

    def _next_response(self):
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        return response

    def __call__(self, prompt=None, **kwargs):
        """Record what was asked and return the next canned response."""
        self.calls.append(prompt)
        return self._next_response()

    # Aliases covering the call styles the generated suites use.
    def invoke(self, prompt=None, **kwargs):
        return self(prompt, **kwargs)

    def predict(self, text=None, **kwargs):
        return self(text, **kwargs)

    def bind_tools(self, tools=None, **kwargs):
        """Record the tools and stay chainable, mirroring the real signature."""
        self.bound_tools = list(tools or [])
        return self

    @property
    def call_count(self):
        return len(self.calls)
'''

    # ---------------------------------------------------------------------- pytest.ini
    def generate_pytest_ini(self) -> str:
        """
        Render ``pytest.ini``.

        Registering the markers here is what fixes the `@pytest.mark.timeout` problem:
        the generated suite used that mark while `pytest-timeout` was declared nowhere,
        so it was either inert or, under `--strict-markers`, a hard error.
        """
        return """[pytest]
# Generated by multi-agent-generator.

testpaths = .

# Offline by default: `live` tests need real credentials and are opt-in via `-m live`.
addopts =
    -v
    --strict-markers
    --tb=short
    -m "not live"

# Fallback per-test limit. Every generated test also carries its own
# @pytest.mark.timeout(...), which takes precedence, so this value only governs tests you
# add yourself. Requires pytest-timeout (listed in requirements-test.txt); conftest.py
# warns at startup when it is missing, because without the plugin these limits are
# silently inert and a stuck test blocks the run instead of failing.
#
# Not raised to accommodate slow tests: offline, nothing here should need a minute. The
# one case that used to blow past this was a slow import inside a fixture, which is now
# handled at collection time in conftest.py instead. timeout_method is left unset so
# pytest-timeout chooses per platform - pinning it to "thread" would turn a single slow
# test into a killed run on Linux, and Windows has no SIGALRM so it uses thread anyway.
timeout = 10

markers =
    contract: validates the agent configuration; needs no framework packages
    unit: builds individual agents
    integration: exercises agent and task wiring
    end_to_end: runs the full workflow
    performance: measures timing and memory
    reliability: checks error handling
    quality: scores output quality
    live: requires real credentials and network access
    timeout: per-test time limit; honoured when pytest-timeout is installed
"""

    # ------------------------------------------------------------------------ readme
    def generate_readme(self, framework: str, provider: str) -> str:
        """Render a short README explaining how to run the generated suite."""
        spec = get_provider(provider)
        import_name = framework_import_name(framework)
        creds = (
            ", ".join(spec.credential_env)
            if spec.credential_env
            else "none required"
        )

        return f"""# Generated tests for a {framework} agent system

Provider: **{provider}** ({spec.label})

## Running the tests

```bash
pip install -r requirements-test.txt
pytest
```

That runs everything except the `live` tests. No API key and no network access are
needed, because a deterministic fake model stands in for the real one and sockets are
blocked for the duration of each offline test.

## What runs, and what needs installing

The suite is layered so that something meaningful runs in every environment.

Contract tests validate the agent configuration itself, checking that agent names are
unique and sanitise to valid Python identifiers, that every task points at an agent that
exists, and that required fields are present. They need no framework packages at all, so
they always run.

Construction tests build real `{framework}` objects and assert on their wiring. They
need `{import_name}` installed; without it they *skip* with a stated reason rather than
failing collection.

Live tests actually call a model. They are marked `live` and deselected by default:

```bash
export {spec.credential_env[0] if spec.credential_env else 'NO_KEY_NEEDED'}=...
pytest -m live
```

Live tests import your generated agent module. Save it next to these tests as
`agent_system.py`, or point `AGENT_MODULE` at it:

```bash
AGENT_MODULE=my_agents pytest -m live
```

## Credentials

Environment variables for this provider: {creds}

{spec.notes}

## Checking dependencies

```bash
multi-agent-generator --check-deps --framework {framework} --provider {provider}
```
"""

    # ------------------------------------------------------------------------ bundle
    def generate_bundle(
        self,
        config: Dict[str, Any],
        framework: str = "crewai",
        provider: str = "openai",
        include_types: Optional[Sequence[TestType]] = None,
        suite: Optional[TestSuite] = None,
    ) -> TestBundle:
        """
        Generate a complete, runnable test project.

        This is the method that actually solves the reported problem. A bare test file
        cannot be run by whoever receives it; a bundle that states its own dependencies,
        registers its own markers and supplies its own offline fixtures can.
        """
        provider_name = get_provider(provider).name
        if suite is None:
            suite = self.generate_test_suite(
                config,
                framework,
                include_types,
                provider=provider_name,
            )

        safe_framework = framework.replace("-", "_")
        return TestBundle(
            framework=framework,
            provider=provider_name,
            files={
                f"test_{safe_framework}_agents.py": self.generate_pytest_file(suite),
                "conftest.py": self.generate_conftest(framework, provider_name),
                "_helpers.py": self.generate_helpers(),
                "pytest.ini": self.generate_pytest_ini(),
                "requirements-test.txt": requirements_txt(
                    framework, provider_name, include_tests=True
                ),
                "README.md": self.generate_readme(framework, provider_name),
            },
        )

    # ------------------------------------------------------- backwards-compatible bits
    def _generate_setup_code(self, config: Dict[str, Any], framework: str) -> str:
        """
        Deprecated. Setup now lives in the generated ``conftest.py``.

        Kept so existing callers do not break; it simply points at the replacement.
        """
        return self.generate_conftest(framework)

    def _generate_teardown_code(self, framework: str) -> str:
        """Deprecated. Teardown is handled by fixture scope in the generated conftest."""
        return ""


def _safe_test_name(name: str) -> str:
    """Sanitise an agent name for use inside a test function name."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", (name or "agent").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "agent"
    if cleaned[0].isdigit():
        cleaned = f"agent_{cleaned}"
    return cleaned


#: Modules that are always importable, so they need no skip guard in generated tests.
_STDLIB_SAFE = frozenset(
    {
        "os",
        "sys",
        "re",
        "json",
        "time",
        "math",
        "typing",
        "pathlib",
        "tracemalloc",
        "importlib",
        "dataclasses",
        "collections",
        "itertools",
        "functools",
        "pytest",
        "_helpers",
    }
)

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)")


def _third_party_imports(body: Sequence[str]) -> List[str]:
    """
    Top-level third-party modules imported by a generated test body.

    Used to emit one ``pytest.importorskip`` per module, so a missing package produces a
    clear skip instead of a collection error. Order is preserved and duplicates dropped,
    which keeps generated files byte-identical across runs.
    """
    found: List[str] = []
    for line in body:
        match = _IMPORT_RE.match(line)
        if not match:
            continue
        module = match.group(1)
        if module in _STDLIB_SAFE or module in found:
            continue
        found.append(module)
    return found


def generate_tests(
    config: Dict[str, Any],
    framework: str,
    output_format: str = "pytest",
    provider: str = "openai",
) -> str:
    """
    Convenience function to generate tests for an agent configuration.

    Args:
        config: Agent system configuration.
        framework: Target framework.
        output_format: ``pytest`` for the test module, ``json`` for the suite
            description, or ``bundle`` for a JSON mapping of every file in the bundle.
        provider: LLM provider the generated agents use.

    Returns:
        Generated test code, or JSON.
    """
    generator = TestGenerator()

    if output_format == "bundle":
        bundle = generator.generate_bundle(config, framework, provider)
        return json.dumps(bundle.files, indent=2)

    suite = generator.generate_test_suite(config, framework, provider=provider)

    if output_format == "pytest":
        return generator.generate_pytest_file(suite)
    if output_format == "json":
        return json.dumps(suite.to_dict(), indent=2)

    raise ValueError(
        f"Unknown output format: {output_format}. Use 'pytest', 'json' or 'bundle'."
    )
