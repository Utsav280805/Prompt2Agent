# multi_agent_generator/core/__init__.py
"""
The core domain layer.

Everything the product knows how to *do* lives here, and nothing here knows how it is being
called. There are no imports of Streamlit, FastAPI, Click or any transport in this package -
which is what allows the CLI, the HTTP API and the test suite to share one implementation of
the pipeline rather than three subtly different ones.

The layering runs one way::

    models      value types: no logic, no I/O
    manifest    the plan: which agents, tools, workflows and files exist
    parsing     turning model output into those types
    analysis    stage 1: what did the user ask for
    selection   stage 2: which framework fits
    architecture stage 3: plan the project - tier, layout, file list
    generation  stage 4: render a runnable project
    depgraph    stage 5: recover the real import graph and validate it
    validation  stage 6: the named gates a project must pass to be called ready
    review      stage 7: is it actually any good
    improvement stage 8: fix what review found
    repair      stage 9: rewrite what validation found, bounded and root-caused
    requirements_spec REQ-001 onward: did the built system honour what was asked for
    pipeline    orchestration, event trace, acceptance verdict
    service     the entry point callers use

Import from this package, not from its modules: ``from multi_agent_generator.core import
run_pipeline`` stays stable even when the internals move.
"""
from __future__ import annotations

from .analysis import analyze_requirement, heuristic_analysis
from .architecture import plan_architecture
from .depgraph import DependencyGraph, GraphProblem, build_graph, validate_imports
from .generation import build_agent_config, generate_project
from .improvement import ImprovementOutcome, improve_project
from .manifest import ProjectManifest
from .models import (
    FrameworkChoice,
    GeneratedFile,
    GeneratedProject,
    IterationRecord,
    PipelineEvent,
    PipelineResult,
    ProjectStatus,
    RequirementAnalysis,
    ReviewIssue,
    ReviewResult,
    RunResult,
    RunStatus,
    Severity,
    Stage,
    TestReport,
    utcnow,
)
from .pipeline import Pipeline, PipelineRequest, run_pipeline
from .repair import (
    MAX_REPAIR_ITERATIONS,
    RepairAction,
    RepairOutcome,
    repair_project,
    root_cause_of,
)
from .requirements_spec import (
    RequirementItem,
    RequirementTrace,
    coverage_summary,
    extract_requirements,
    requirement_checks,
    trace_requirements,
)
from .review import model_review, review_project, static_review
from .selection import FRAMEWORK_PROFILES, score_frameworks, select_framework
from .service import GeneratorService, RunStore, build_service, get_service, set_service
from .validation import (
    CATEGORIES,
    CATEGORY_LABELS,
    RUN_ENTRY_NAMES,
    Check,
    CheckStatus,
    ValidationReport,
    static_checks,
    validate_project,
)

__all__ = [
    # value types
    "FrameworkChoice",
    "GeneratedFile",
    "GeneratedProject",
    "IterationRecord",
    "PipelineEvent",
    "PipelineResult",
    "ProjectManifest",
    "ProjectStatus",
    "RequirementAnalysis",
    "ReviewIssue",
    "ReviewResult",
    "RunResult",
    "RunStatus",
    "Severity",
    "Stage",
    "TestReport",
    "utcnow",
    # stages
    "analyze_requirement",
    "heuristic_analysis",
    "select_framework",
    "score_frameworks",
    "FRAMEWORK_PROFILES",
    "build_agent_config",
    "plan_architecture",
    "generate_project",
    "build_graph",
    "validate_imports",
    "DependencyGraph",
    "GraphProblem",
    "validate_project",
    "static_checks",
    "ValidationReport",
    "Check",
    "CheckStatus",
    "CATEGORIES",
    "CATEGORY_LABELS",
    "RUN_ENTRY_NAMES",
    "review_project",
    "static_review",
    "model_review",
    "improve_project",
    "ImprovementOutcome",
    "repair_project",
    "root_cause_of",
    "RepairAction",
    "RepairOutcome",
    "MAX_REPAIR_ITERATIONS",
    # requirement traceability
    "extract_requirements",
    "trace_requirements",
    "requirement_checks",
    "coverage_summary",
    "RequirementItem",
    "RequirementTrace",
    # orchestration
    "Pipeline",
    "PipelineRequest",
    "run_pipeline",
    "GeneratorService",
    "RunStore",
    "get_service",
    "set_service",
    "build_service",
]
