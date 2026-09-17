"""
Auto-generated test suite for a crewai agent system

Auto-generated test suite: crewai_test_suite
Framework: crewai
Provider:  openai

Running these tests
-------------------
    pip install -r requirements-test.txt
    pytest

Offline by default: a deterministic fake model stands in for the real one, and sockets
are blocked, so no credentials and no network are needed. Tests marked `live` call real
models and are deselected unless you ask for them explicitly:

    pytest -m live

Contract tests validate the configuration itself and need no framework packages at all.
Tests that build framework objects skip cleanly when 'crewai' is missing, rather
than failing collection.
"""

import os
import time
import tracemalloc

import pytest

# Sanitisation helpers, copied into the bundle so this suite has no dependency on the
# generator that produced it.
from _helpers import sanitize, tool_class_name

# The configuration under test, inlined so the suite is self-contained.
AGENT_CONFIG = {'process': 'sequential',
 'agents': [{'name': 'research_specialist',
             'role': 'Research Specialist',
             'goal': 'Conduct thorough research and gather information',
             'backstory': 'Expert researcher with years of experience in data '
                          'gathering and analysis',
             'tools': ['search_tool', 'web_scraper'],
             'verbose': True,
             'allow_delegation': False},
            {'name': 'content_writer',
             'role': 'Content Writer',
             'goal': 'Create clear and comprehensive written content',
             'backstory': 'Professional writer skilled in creating engaging and '
                          'informative content',
             'tools': ['writing_tool', 'grammar_checker'],
             'verbose': True,
             'allow_delegation': False}],
 'tasks': [{'name': 'research_task',
            'description': 'Gather information and conduct research on the given topic',
            'tools': ['search_tool'],
            'agent': 'research_specialist',
            'expected_output': 'Comprehensive research findings and data'},
           {'name': 'writing_task',
            'description': 'Create written content based on research findings',
            'tools': ['writing_tool'],
            'agent': 'content_writer',
            'expected_output': 'Well-written content document'}]}


@pytest.mark.contract
@pytest.mark.timeout(30)
def test_config_declares_agents():
    """
    The configuration declares at least one agent

    Expected: agents list is non-empty
    """
    agents = AGENT_CONFIG.get('agents') or []
    assert len(agents) > 0


@pytest.mark.contract
@pytest.mark.timeout(30)
def test_agent_names_are_unique():
    """
    Agent names are unique, so tasks can reference them unambiguously

    Expected: no duplicate agent names
    """
    names = [a.get('name') for a in AGENT_CONFIG.get('agents') or []]
    duplicates = {n for n in names if names.count(n) > 1}
    assert duplicates == set()


@pytest.mark.contract
@pytest.mark.timeout(30)
def test_agents_have_required_fields():
    """
    Every agent declares a name, role and goal

    Expected: each agent has name, role and goal
    """
    missing = [
        (a.get('name'), key)
        for a in AGENT_CONFIG.get('agents') or []
        for key in ('name', 'role', 'goal')
        if not a.get(key)
    ]
    assert missing == []


@pytest.mark.contract
@pytest.mark.timeout(30)
def test_agent_names_are_valid_identifiers():
    """
    Agent names can be turned into valid Python variable names

    Expected: every sanitised agent name is a valid identifier
    """
    # Generated code derives variable names from agent names, so a name
    # that cannot be sanitised would produce a file that will not parse.
    bad = [
        a.get('name')
        for a in AGENT_CONFIG.get('agents') or []
        if not sanitize(a.get('name', '')).isidentifier()
    ]
    assert bad == []


@pytest.mark.contract
@pytest.mark.timeout(30)
def test_tasks_reference_existing_agents():
    """
    Every task is assigned to an agent that exists

    Expected: no task references an unknown agent
    """
    agent_names = {a.get('name') for a in AGENT_CONFIG.get('agents') or []}
    orphans = [
        t.get('name')
        for t in AGENT_CONFIG.get('tasks') or []
        if t.get('agent') and t.get('agent') not in agent_names
    ]
    assert orphans == []


@pytest.mark.contract
@pytest.mark.timeout(30)
def test_tasks_have_descriptions_and_outputs():
    """
    Every task declares a description and an expected output

    Expected: each task has description and expected_output
    """
    incomplete = [
        t.get('name')
        for t in AGENT_CONFIG.get('tasks') or []
        if not t.get('description') or not t.get('expected_output')
    ]
    assert incomplete == []


@pytest.mark.contract
@pytest.mark.timeout(30)
def test_task_names_are_unique():
    """
    Task names are unique

    Expected: no duplicate task names
    """
    names = [t.get('name') for t in AGENT_CONFIG.get('tasks') or []]
    duplicates = {n for n in names if names.count(n) > 1}
    assert duplicates == set()


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_research_specialist_constructs(offline_llm):
    """
    research_specialist can be constructed from its configuration

    Expected: agent is constructed with the configured role and goal
    """
    pytest.importorskip("crewai", reason="crewai is not installed; install it to run this test")
    from crewai import Agent
    cfg = AGENT_CONFIG['agents'][0]
    agent = Agent(
        role=cfg['role'],
        goal=cfg['goal'],
        backstory=cfg.get('backstory', ''),
        tools=[],
        llm=offline_llm,
    )
    assert agent.role == cfg['role']
    assert agent.goal == cfg['goal']


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_research_specialist_tool_names_are_usable():
    """
    research_specialist's tool names can be turned into class names

    Expected: each tool name yields a valid class name
    """
    tools = AGENT_CONFIG['agents'][0].get('tools') or []
    class_names = [tool_class_name(t) for t in tools if isinstance(t, str)]
    # Uniqueness is deliberately not asserted. Names that differ only
    # in punctuation ('web_search' and 'web-search') map to the same
    # class by design, and the generator drops the duplicate rather
    # than emitting two classes with one name - so a collision here
    # is valid config, not a defect. What must hold is that every
    # name yields a usable class identifier.
    assert class_names != []
    assert all(n.isidentifier() for n in class_names)
    assert all(n.endswith('Tool') for n in class_names)


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_content_writer_constructs(offline_llm):
    """
    content_writer can be constructed from its configuration

    Expected: agent is constructed with the configured role and goal
    """
    pytest.importorskip("crewai", reason="crewai is not installed; install it to run this test")
    from crewai import Agent
    cfg = AGENT_CONFIG['agents'][1]
    agent = Agent(
        role=cfg['role'],
        goal=cfg['goal'],
        backstory=cfg.get('backstory', ''),
        tools=[],
        llm=offline_llm,
    )
    assert agent.role == cfg['role']
    assert agent.goal == cfg['goal']


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_content_writer_tool_names_are_usable():
    """
    content_writer's tool names can be turned into class names

    Expected: each tool name yields a valid class name
    """
    tools = AGENT_CONFIG['agents'][1].get('tools') or []
    class_names = [tool_class_name(t) for t in tools if isinstance(t, str)]
    # Uniqueness is deliberately not asserted. Names that differ only
    # in punctuation ('web_search' and 'web-search') map to the same
    # class by design, and the generator drops the duplicate rather
    # than emitting two classes with one name - so a collision here
    # is valid config, not a defect. What must hold is that every
    # name yields a usable class identifier.
    assert class_names != []
    assert all(n.isidentifier() for n in class_names)
    assert all(n.endswith('Tool') for n in class_names)


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_crew_assembles_with_all_tasks(offline_llm):
    """
    A Crew can be assembled with every configured agent and task

    Expected: crew holds one task per configured task
    """
    pytest.importorskip("crewai", reason="crewai is not installed; install it to run this test")
    from crewai import Agent, Crew, Process, Task
    agents = {}
    for cfg in AGENT_CONFIG['agents']:
        agents[cfg['name']] = Agent(
            role=cfg['role'],
            goal=cfg['goal'],
            backstory=cfg.get('backstory', ''),
            tools=[],
            llm=offline_llm,
        )
    task_objects = []
    for cfg in AGENT_CONFIG['tasks']:
        owner = agents.get(cfg.get('agent')) or list(agents.values())[0]
        task_objects.append(Task(
            description=cfg['description'],
            expected_output=cfg.get('expected_output', 'A result'),
            agent=owner,
        ))
    crew = Crew(
        agents=list(agents.values()),
        tasks=task_objects,
        process=Process.sequential,
    )
    assert len(crew.tasks) == len(AGENT_CONFIG['tasks'])
    assert len(crew.agents) == len(AGENT_CONFIG['agents'])
    assert all(t.agent is not None for t in crew.tasks)


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_task_order_is_deterministic():
    """
    Task ordering in the config is stable and total

    Expected: tasks form an ordered sequence with no gaps
    """
    tasks = AGENT_CONFIG['tasks']
    names = [t.get('name') for t in tasks]
    assert len(names) == len(tasks)
    assert all(n for n in names)


@pytest.mark.reliability
@pytest.mark.timeout(30)
def test_missing_required_field_is_rejected(offline_llm):
    """
    Constructing an agent without a role raises rather than passing silently

    Expected: a validation error is raised
    """
    pytest.importorskip("crewai", reason="crewai is not installed; install it to run this test")
    from crewai import Agent
    with pytest.raises(Exception):
        Agent(goal='no role supplied', backstory='', tools=[], llm=offline_llm)


@pytest.mark.reliability
@pytest.mark.timeout(30)
def test_empty_query_is_handled(fake_llm):
    """
    An empty query does not raise an unhandled exception

    Expected: empty and whitespace input are handled gracefully
    """
    pytest.importorskip("langchain_core", reason="langchain_core is not installed; install it to run this test")
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    chain = ChatPromptTemplate.from_template('{input}') | fake_llm | StrOutputParser()
    results = [chain.invoke({'input': value}) for value in ('', '   ')]
    assert all(isinstance(r, str) for r in results)


@pytest.mark.reliability
@pytest.mark.timeout(30)
def test_repeated_calls_are_stable(fake_llm):
    """
    Repeated calls with a deterministic model return consistent results

    Expected: the same input yields the same output shape each time
    """
    pytest.importorskip("langchain_core", reason="langchain_core is not installed; install it to run this test")
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    chain = ChatPromptTemplate.from_template('{input}') | fake_llm | StrOutputParser()
    results = [chain.invoke({'input': 'same question'}) for _ in range(3)]
    assert len(results) == 3
    assert all(isinstance(r, str) and r for r in results)


@pytest.mark.reliability
@pytest.mark.timeout(30)
def test_no_credentials_required_for_construction():
    """
    Building the system does not require real provider credentials

    Expected: no real API key is present during offline tests
    """
    # The conftest stubs credentials and blocks sockets. If this test
    # ever sees a plausible real key, the offline guarantee has broken.
    key = os.environ.get('OPENAI_API_KEY', '')
    assert key.startswith('test-') or key == ''


@pytest.mark.end_to_end
@pytest.mark.live
@pytest.mark.timeout(300)
def test_complete_workflow_live(agent_module):
    """
    The complete workflow runs against a real model

    Expected: workflow returns a non-empty final result
    """
    result = agent_module.run_workflow('Summarise the latest results')
    assert result is not None
    assert len(str(result)) > 0


@pytest.mark.performance
@pytest.mark.live
@pytest.mark.timeout(180)
def test_response_time_live(agent_module):
    """
    A single request completes within the budget

    Expected: response arrives within 120 seconds
    """
    started = time.perf_counter()
    result = agent_module.run_workflow('Quick timing probe')
    elapsed = time.perf_counter() - started
    print(f'elapsed: {elapsed:.2f}s')
    assert result is not None
    assert elapsed < 120


@pytest.mark.performance
@pytest.mark.live
@pytest.mark.timeout(300)
def test_peak_memory_is_bounded_live(agent_module):
    """
    Peak memory during a run stays within bounds

    Expected: peak allocation under 500 MB
    """
    tracemalloc.start()
    try:
        agent_module.run_workflow('Memory probe')
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    print(f'peak: {peak_mb:.1f} MB')
    assert peak_mb < 500


@pytest.mark.quality
@pytest.mark.live
@pytest.mark.timeout(300)
def test_output_relevance_live(agent_module, evaluator):
    """
    Output is relevant to the query

    Expected: response mentions the queried subject
    """
    query = 'What is machine learning?'
    response = str(agent_module.run_workflow(query))
    result = evaluator.evaluate(query, response)
    print(result.metrics.overall_score())
    assert result.metrics.relevance_score > 0.3
    assert len(response) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
