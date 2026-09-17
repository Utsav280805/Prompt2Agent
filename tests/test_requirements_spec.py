"""Requirement tracing: planned agents and tools must cover suggested roles."""
from __future__ import annotations

import pytest

from multi_agent_generator.core.analysis import heuristic_analysis
from multi_agent_generator.core.architecture import plan_architecture
from multi_agent_generator.core.manifest import (
    AgentSpec,
    NodeKind,
    ProjectManifest,
    ToolSpec,
    WorkflowEdge,
    WorkflowNode,
    WorkflowSpec,
)
from multi_agent_generator.core.models import FrameworkChoice, RequirementAnalysis
from multi_agent_generator.core.requirements_spec import (
    extract_requirements,
    requirement_checks,
    trace_requirements,
)

pytestmark = pytest.mark.unit


def _analysis(**overrides) -> RequirementAnalysis:
    data = dict(
        requirement="Create a research assistant that finds papers and writes a review",
        summary="Research then write.",
        goals=["Find papers", "Write a review"],
        suggested_roles=["researcher", "writer"],
        suggested_tools=["web_search"],
        needs_sequential_steps=True,
        needs_branching=False,
        needs_delegation=False,
        needs_state=False,
        needs_human_input=False,
        complexity="moderate",
    )
    data.update(overrides)
    return RequirementAnalysis(**data)


def _manifest(agents, tools=None, workflow=None) -> ProjectManifest:
    manifest = ProjectManifest(
        project_name="demo",
        framework="crewai",
        provider="huggingface",
        model="test-model",
        agents=list(agents),
        tools=list(tools or []),
    )
    if workflow is not None:
        manifest.workflows = [workflow]
    return manifest


class TestRoleAndToolCoverage:
    def test_researcher_matches_research_specialist(self):
        items = extract_requirements(_analysis())
        traces = trace_requirements(
            items,
            _manifest(
                [
                    AgentSpec(name="research_specialist", role="Research Specialist"),
                    AgentSpec(name="content_writer", role="Content Writer"),
                ],
                [ToolSpec(name="duckduckgo_search", purpose="Search the web")],
            ),
        )
        mechanical = [t for t in traces if t.item.kind in ("role", "tool")]
        assert all(t.satisfied is not False for t in mechanical), [
            (t.item.text, t.finding) for t in mechanical if t.satisfied is False
        ]

    def test_heuristic_research_prompt_is_fully_covered_by_planned_architecture(self):
        analysis = heuristic_analysis(
            "Create a research assistant that finds recent papers on a topic "
            "and writes a short literature review"
        )
        choice = FrameworkChoice(framework="crewai", reason="test", confidence=0.9)
        manifest = plan_architecture(analysis, choice, provider="huggingface")
        checks = requirement_checks(analysis, manifest)
        blocking = [c for c in checks if c.blocking]
        assert blocking == [], [c.summary for c in blocking]


class TestBehaviourCoverage:
    def test_delegation_passes_when_manager_may_delegate(self):
        analysis = _analysis(needs_delegation=True, suggested_roles=["researcher", "writer"])
        items = extract_requirements(analysis)
        traces = trace_requirements(
            items,
            _manifest(
                [
                    AgentSpec(name="researcher", allow_delegation=True),
                    AgentSpec(name="writer"),
                ]
            ),
        )
        delegation = next(t for t in traces if t.item.source == "analysis.needs_delegation")
        assert delegation.satisfied is True

    def test_state_passes_from_ordered_workflow_even_if_agent_inputs_blank(self):
        analysis = _analysis(needs_state=True)
        items = extract_requirements(analysis)
        workflow = WorkflowSpec(
            name="main",
            kind="sequential",
            nodes=[
                WorkflowNode(id="start", kind=NodeKind.START),
                WorkflowNode(
                    id="step_1",
                    kind=NodeKind.AGENT,
                    agent="researcher",
                    inputs="The user query.",
                ),
                WorkflowNode(
                    id="step_2",
                    kind=NodeKind.AGENT,
                    agent="writer",
                    inputs="Previous output.",
                ),
                WorkflowNode(id="end", kind=NodeKind.END),
            ],
            edges=[
                WorkflowEdge(source="start", target="step_1"),
                WorkflowEdge(source="step_1", target="step_2"),
                WorkflowEdge(source="step_2", target="end"),
            ],
        )
        traces = trace_requirements(
            items,
            _manifest(
                [AgentSpec(name="researcher"), AgentSpec(name="writer")],
                workflow=workflow,
            ),
        )
        state = next(t for t in traces if t.item.source == "analysis.needs_state")
        assert state.satisfied is True
