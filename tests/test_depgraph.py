# tests/test_depgraph.py
"""
Tests for the file dependency graph.

Two jobs. The first is to pin down every judgement
:mod:`multi_agent_generator.core.depgraph` makes, because most of them are judgements about
*severity* - whether a thing that looks wrong actually stops the project running - and a wrong
judgement in either direction is expensive. A missed blocker means the user is handed a project
that dies on its first import, which is the complaint this whole module exists to answer. A
false blocker means a project that would have run fine is withheld, which is worse, because
there is nothing for the user to fix.

The second job is the round trip at the bottom: plan a real project, assemble it, and check that
the emitted code is internally consistent - every import resolves, every imported name exists,
nothing depends on a package the generated ``requirements.txt`` does not list. Nothing had ever
checked that before. It is also the first test in this repository to exercise
``plan_architecture`` and ``assemble_project`` at all.

No network, no credentials, no agent framework installed:

    pip install pytest
    pytest tests/test_depgraph.py
"""
from __future__ import annotations

import textwrap
from typing import Dict, List, Mapping, Optional, Sequence

import pytest

from multi_agent_generator.core.analysis import heuristic_analysis
from multi_agent_generator.core.architecture import plan_architecture
from multi_agent_generator.core.depgraph import (
    CODE_AMBIGUOUS_MODULE,
    CODE_BAD_RELATIVE_IMPORT,
    CODE_CIRCULAR_IMPORT,
    CODE_MISSING_SYMBOL,
    CODE_ORPHAN_FILE,
    CODE_SYNTAX_ERROR,
    CODE_TEST_IMPORTED,
    CODE_UNDECLARED_DEPENDENCY,
    CODE_UNPLANNED_EDGE,
    CODE_UNREALISED_EDGE,
    CODE_UNRESOLVED_INTERNAL,
    DependencyGraph,
    GraphProblem,
    build_graph,
    import_roots_for,
    validate_imports,
)
from multi_agent_generator.core.models import GeneratedProject, Severity
from multi_agent_generator.core.selection import select_framework
from multi_agent_generator.dependencies import Requirement
from multi_agent_generator.emit.assemble import EMITTERS, assemble_project

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _project(
    files: Mapping[str, str],
    *,
    dependencies: Sequence[str] = (),
    entrypoint: Optional[str] = None,
    framework: str = "crewai",
    provider: str = "openai",
) -> GeneratedProject:
    """
    A GeneratedProject from a path -> source mapping.

    Sources are written as indented triple-quoted strings and dedented here, so the tests read
    like the files they describe. Insertion order is preserved, which one test below relies on.
    """
    project = GeneratedProject(
        framework=framework,
        provider=provider,
        model="gpt-4.1-mini",
        dependencies=list(dependencies),
    )
    for path, content in files.items():
        project.add_file(
            path,
            textwrap.dedent(content).strip() + "\n",
            is_entrypoint=entrypoint is not None and path == entrypoint,
        )
    return project


def _codes(graph: DependencyGraph) -> List[str]:
    return [problem.code for problem in graph.problems]


def _of_code(graph: DependencyGraph, code: str) -> List[GraphProblem]:
    return [problem for problem in graph.problems if problem.code == code]


def _one(graph: DependencyGraph, code: str) -> GraphProblem:
    found = _of_code(graph, code)
    assert len(found) == 1, f"expected exactly one {code}, got {_codes(graph)}"
    return found[0]


def _edge_pairs(graph: DependencyGraph) -> List[tuple]:
    return sorted({(edge.source, edge.target) for edge in graph.edges})


def _explain(graph: DependencyGraph) -> str:
    """A readable failure message. A bare `assert graph.ok` tells you nothing useful."""
    if not graph.problems:
        return "no problems"
    return "\n".join(
        f"  [{p.severity.value}] {p.code}: {p.message}" for p in graph.problems
    )


# --------------------------------------------------------------------------------------
# Impact analysis - section 7's question
# --------------------------------------------------------------------------------------
_CHAIN = {
    "main.py": """
        from app.workflow import run_workflow


        def main() -> None:
            print(run_workflow("hello"))
    """,
    "app/__init__.py": """
        from app.workflow import run_workflow
    """,
    "app/workflow.py": """
        from app.agents import build_researcher


        def run_workflow(query: str) -> dict:
            return {"output": build_researcher().run(query)}
    """,
    "app/agents.py": """
        from app.config import get_settings


        def build_researcher():
            return get_settings()
    """,
    "app/config.py": """
        import os


        def get_settings() -> dict:
            return {"model": os.getenv("AGENT_MODEL", "")}
    """,
}


def test_impact_of_is_the_transitive_reverse_closure():
    graph = build_graph(_project(_CHAIN, entrypoint="main.py"))

    assert graph.ok, _explain(graph)
    assert graph.dependents("app/config.py") == ["app/agents.py"]
    # Every file above config in the chain, and config itself excluded: it is the cause of the
    # change, not part of what the change breaks.
    assert graph.impact_of("app/config.py") == [
        "app/__init__.py",
        "app/agents.py",
        "app/workflow.py",
        "main.py",
    ]
    # Nothing imports the entry point, so changing it breaks nothing else.
    assert graph.impact_of("main.py") == []
    assert graph.dependencies_of("main.py") == [
        "app/agents.py",
        "app/config.py",
        "app/workflow.py",
    ]
    assert graph.imports("app/workflow.py") == ["app/agents.py"]


def test_graph_reports_the_entry_point_it_was_given():
    project = _project(_CHAIN, entrypoint="main.py")
    graph = build_graph(project)

    assert graph.entrypoint == "main.py"
    assert graph.nodes["main.py"].is_entrypoint is True
    assert graph.nodes["app/config.py"].is_entrypoint is False


def test_entry_point_is_recovered_from_the_file_flag_alone():
    """
    A project whose file carries ``is_entrypoint`` but whose ``entrypoint`` field was never set
    must still have one answer to "which file runs", not two that disagree.
    """
    project = _project(_CHAIN, entrypoint="main.py")
    project.entrypoint = None

    graph = build_graph(project)

    assert graph.entrypoint == "main.py"
    assert graph.nodes["main.py"].is_entrypoint is True


def test_as_dict_carries_the_reverse_edges_the_ui_needs():
    graph = build_graph(_project(_CHAIN, entrypoint="main.py"))

    payload = graph.as_dict()
    by_path = {entry["path"]: entry for entry in payload["files"]}

    assert by_path["app/config.py"]["used_by"] == ["app/agents.py"]
    assert "main.py" in by_path["app/config.py"]["breaks_if_changed"]
    assert payload["summary"]["files"] == 5
    assert payload["summary"]["ok"] is True
    assert payload["entrypoint"] == "main.py"
    # Whether an edge is load-bearing at import time is decided here, once, not in TypeScript.
    assert all("binds_names" in edge for edge in payload["edges"])


# --------------------------------------------------------------------------------------
# Imports that cannot be satisfied - section 28
# --------------------------------------------------------------------------------------
def test_import_of_a_module_the_project_does_not_contain_blocks():
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from app.missing import helper
                    from app.config import get_settings
                """,
                "app/config.py": """
                    def get_settings() -> dict:
                        return {}
                """,
            },
            entrypoint="main.py",
        )
    )

    problem = _one(graph, CODE_UNRESOLVED_INTERNAL)
    assert problem.severity is Severity.BLOCKER
    assert problem.path == "main.py"
    assert "app.missing" in problem.message
    assert graph.ok is False
    assert graph.unresolved_imports() == {"main.py": ["app.missing"]}


def test_import_of_a_name_the_target_does_not_define_blocks():
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from app.config import get_settings, load_profile
                """,
                "app/config.py": """
                    def get_settings() -> dict:
                        return {}
                """,
            },
            entrypoint="main.py",
        )
    )

    problem = _one(graph, CODE_MISSING_SYMBOL)
    assert problem.severity is Severity.BLOCKER
    assert "load_profile" in problem.message
    assert "get_settings" not in problem.message
    assert problem.detail == ("app/config.py",)


def test_names_bound_inside_module_level_blocks_are_found():
    """
    A guarded ``try: import x`` and a name assigned inside ``if`` are still module attributes.
    Missing them would report a false ImportError against code that works.
    """
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from app.config import FLAG, json_module, LATE
                """,
                "app/config.py": """
                    try:
                        import json as json_module
                    except ImportError:
                        json_module = None

                    if True:
                        FLAG = 1

                    for LATE in range(1):
                        pass
                """,
            },
            entrypoint="main.py",
        )
    )

    assert _of_code(graph, CODE_MISSING_SYMBOL) == []
    assert graph.ok, _explain(graph)


def test_a_function_local_name_is_not_importable():
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from app.config import helper
                """,
                "app/config.py": """
                    def build() -> int:
                        helper = 1
                        return helper
                """,
            },
            entrypoint="main.py",
        )
    )

    problem = _one(graph, CODE_MISSING_SYMBOL)
    assert problem.severity is Severity.BLOCKER
    assert "helper" in problem.message


def test_star_import_suppresses_the_name_check_rather_than_guessing():
    """
    ``from x import *`` makes the target's namespace unknowable without executing it, so names
    imported *from* such a module are not checked. Reporting them would be a guess.
    """
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from app.config import anything_at_all
                """,
                "app/config.py": """
                    from app.base import *
                """,
                "app/base.py": """
                    anything_at_all = 1
                """,
            },
            entrypoint="main.py",
        )
    )

    assert _of_code(graph, CODE_MISSING_SYMBOL) == []


# --------------------------------------------------------------------------------------
# Third-party imports
# --------------------------------------------------------------------------------------
def test_undeclared_third_party_import_is_reported_and_declared_one_is_not():
    graph = build_graph(
        _project(
            {
                "main.py": """
                    import os
                    import crewai
                    import mystery_pkg
                """,
                "app/config.py": """
                    STUB = 1
                """,
            },
            dependencies=["crewai>=0.80.0"],
            entrypoint="main.py",
        )
    )

    node = graph.nodes["main.py"]
    assert node.stdlib == ["os"]
    assert node.third_party == ["crewai"]
    assert node.unresolved == ["mystery_pkg"]

    problem = _one(graph, CODE_UNDECLARED_DEPENDENCY)
    assert problem.severity is Severity.MAJOR
    assert "mystery_pkg" in problem.message
    # MAJOR blocks a release: a project that cannot import is not READY.
    assert graph.ok is False


def test_requirement_lines_resolve_to_the_roots_they_make_importable():
    roots, guessed = import_roots_for(
        [
            "python-dotenv>=1.0.0",
            "langchain-core>=0.3.0",
            "crewai[tools]>=0.80.0",
            "uvloop; sys_platform != 'win32'",
            "--index-url https://example.invalid/simple",
            "# a comment",
            "",
        ]
    )

    # dotenv and langchain_core come from the requirement registry, which pairs every spec with
    # the module name used to detect it - the same source --check-deps reads.
    assert {"dotenv", "langchain_core", "crewai", "uvloop"} <= roots
    # An option line is not a distribution.
    assert not any(root.startswith("-") for root in roots)
    # Only the distribution the registry has never heard of was guessed at, and it says so.
    assert len(guessed) == 1
    assert "uvloop" in guessed[0]


def test_extras_bracket_does_not_leak_into_the_distribution_name():
    """
    ``crewai[tools]>=0.80.0`` contains both ``[`` and ``>=``. Splitting on whichever separator
    was checked first left ``crewai[tools]``, which then failed to match the registry and was
    reported as an undeclared dependency against code importing plain ``crewai``.
    """
    assert Requirement("crewai[tools]>=0.80.0", "crewai").package == "crewai"
    assert Requirement("crewai >= 0.80.0", "crewai").package == "crewai"
    assert Requirement("crewai", "crewai").package == "crewai"

    graph = build_graph(
        _project(
            {"main.py": "import crewai\n"},
            dependencies=["crewai[tools]>=0.80.0"],
            entrypoint="main.py",
        )
    )
    assert _of_code(graph, CODE_UNDECLARED_DEPENDENCY) == []
    assert graph.nodes["main.py"].third_party == ["crewai"]


# --------------------------------------------------------------------------------------
# Namespace packages - the false blocker that nearly shipped
# --------------------------------------------------------------------------------------
def test_namespace_package_import_is_not_reported_as_missing():
    """
    ``app/tools/`` with no ``__init__.py`` is still importable: Python 3 treats the directory as
    a namespace package. Reporting ``from app.tools import web_search`` as "no such module"
    would block a project that runs perfectly well.
    """
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from app.tools import web_search
                """,
                "app/tools/web_search.py": """
                    def web_search(query: str) -> str:
                        return query
                """,
            },
            entrypoint="main.py",
        )
    )

    assert graph.ok, _explain(graph)
    assert ("main.py", "app/tools/web_search.py") in _edge_pairs(graph)
    assert graph.impact_of("app/tools/web_search.py") == ["main.py"]


def test_namespace_package_cannot_provide_a_name_it_has_no_module_for():
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from app.tools import web_search, summarise
                """,
                "app/tools/web_search.py": """
                    def web_search(query: str) -> str:
                        return query
                """,
            },
            entrypoint="main.py",
        )
    )

    problem = _one(graph, CODE_MISSING_SYMBOL)
    assert problem.severity is Severity.BLOCKER
    assert "summarise" in problem.message
    assert "__init__.py" in problem.message


# --------------------------------------------------------------------------------------
# Cycles, and whether they actually fail
# --------------------------------------------------------------------------------------
def test_cycle_of_module_level_from_imports_blocks():
    graph = build_graph(
        _project(
            {
                "a.py": """
                    from b import beta

                    alpha = beta + 1
                """,
                "b.py": """
                    from a import alpha

                    beta = 2
                """,
            }
        )
    )

    problem = _one(graph, CODE_CIRCULAR_IMPORT)
    assert problem.severity is Severity.BLOCKER
    assert set(problem.detail) == {"a.py", "b.py"}
    assert graph.ok is False
    # Reported once, not once per rotation: it is one fix.
    assert len(graph.cycles()) == 1


def test_star_import_in_a_cycle_still_blocks():
    """
    ``from b import *`` binds names out of ``b`` and so needs ``b`` finished, exactly like
    ``from b import beta``. Testing for a non-empty name list alone would have called this
    harmless.
    """
    graph = build_graph(
        _project(
            {
                "a.py": """
                    from b import *

                    alpha = 1
                """,
                "b.py": """
                    from a import alpha

                    beta = 2
                """,
            }
        )
    )

    problem = _one(graph, CODE_CIRCULAR_IMPORT)
    assert problem.severity is Severity.BLOCKER


def test_cycle_through_a_function_level_import_is_minor():
    graph = build_graph(
        _project(
            {
                "a.py": """
                    from b import beta

                    alpha = beta + 1
                """,
                "b.py": """
                    beta = 2


                    def describe() -> int:
                        from a import alpha

                        return alpha
                """,
            }
        )
    )

    problem = _one(graph, CODE_CIRCULAR_IMPORT)
    assert problem.severity is Severity.MINOR
    assert "does not fail today" in problem.message
    # Worth showing, not worth withholding a working project over.
    assert graph.ok is True


def test_cycle_that_only_binds_modules_is_minor():
    graph = build_graph(
        _project(
            {
                "a.py": """
                    import b

                    alpha = 1
                """,
                "b.py": """
                    import a

                    beta = 2
                """,
            }
        )
    )

    problem = _one(graph, CODE_CIRCULAR_IMPORT)
    assert problem.severity is Severity.MINOR
    assert graph.ok is True


# --------------------------------------------------------------------------------------
# Dynamic imports
# --------------------------------------------------------------------------------------
def test_literal_dynamic_import_creates_a_real_edge():
    """
    The generated ``tests/test_imports.py`` reaches every module through
    ``importlib.import_module``. Without following those calls the graph would claim nothing
    depends on the agent modules - the section 7 question, answered wrongly.
    """
    graph = build_graph(
        _project(
            {
                "app/config.py": """
                    def get_settings() -> dict:
                        return {}
                """,
                "tests/test_imports.py": """
                    import importlib


                    def test_modules_import() -> None:
                        assert importlib.import_module("app.config") is not None
                """,
            }
        )
    )

    assert ("tests/test_imports.py", "app/config.py") in _edge_pairs(graph)
    assert graph.dependents("app/config.py") == ["tests/test_imports.py"]
    assert graph.impact_of("app/config.py") == ["tests/test_imports.py"]
    # It binds a module, not a name, so it can never be the step that makes a cycle raise.
    edge = next(e for e in graph.edges if e.source == "tests/test_imports.py")
    assert edge.dynamic is True
    assert edge.binds_names is False


def test_dynamic_import_of_a_variable_is_not_guessed_at():
    graph = build_graph(
        _project(
            {
                "app/config.py": """
                    STUB = 1
                """,
                "tests/test_imports.py": """
                    import importlib

                    MODULES = ["app.config"]


                    def test_modules_import() -> None:
                        for name in MODULES:
                            assert importlib.import_module(name) is not None
                """,
            }
        )
    )

    # An invented edge would show up in the impact analysis as a fact. Better absent.
    assert graph.edges == []


# --------------------------------------------------------------------------------------
# Structural problems
# --------------------------------------------------------------------------------------
def test_source_importing_the_test_suite_is_reported():
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from tests.helpers import fake_model
                """,
                "tests/helpers.py": """
                    def fake_model() -> str:
                        return "stub"
                """,
            },
            entrypoint="main.py",
        )
    )

    problem = _one(graph, CODE_TEST_IMPORTED)
    assert problem.severity is Severity.MAJOR
    assert problem.path == "main.py"
    assert problem.detail == ("tests/helpers.py",)


def test_test_files_may_import_each_other():
    graph = build_graph(
        _project(
            {
                "tests/helpers.py": """
                    def fake_model() -> str:
                        return "stub"
                """,
                "tests/test_agent.py": """
                    from tests.helpers import fake_model


                    def test_stub() -> None:
                        assert fake_model() == "stub"
                """,
            }
        )
    )

    assert _of_code(graph, CODE_TEST_IMPORTED) == []
    assert graph.ok, _explain(graph)


def test_two_files_claiming_the_same_module_are_reported():
    graph = build_graph(
        _project(
            {
                # Insertion order decides which one is described as the incumbent.
                "app/tools.py": """
                    TOOLS = []
                """,
                "app/tools/__init__.py": """
                    TOOLS = []
                """,
            }
        )
    )

    problem = _one(graph, CODE_AMBIGUOUS_MODULE)
    assert problem.severity is Severity.MAJOR
    assert problem.path == "app/tools/__init__.py"
    assert "app/tools.py" in problem.message
    assert graph.ok is False


def test_a_syntax_error_does_not_stop_the_rest_of_the_graph():
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from app.config import get_settings
                """,
                "app/config.py": """
                    def get_settings() -> dict:
                        return {}
                """,
                "app/broken.py": """
                    def oops(:
                """,
            },
            entrypoint="main.py",
        )
    )

    problem = _one(graph, CODE_SYNTAX_ERROR)
    assert problem.severity is Severity.BLOCKER
    assert problem.path == "app/broken.py"
    assert graph.nodes["app/broken.py"].syntax_error
    # The other three files were still analysed.
    assert len(graph.nodes) == 3
    assert ("main.py", "app/config.py") in _edge_pairs(graph)


def test_names_are_not_checked_against_a_file_that_would_not_parse():
    """One syntax error should produce one problem, not one per name imported from it."""
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from app.broken import alpha, beta, gamma
                """,
                "app/broken.py": """
                    def oops(:
                """,
            },
            entrypoint="main.py",
        )
    )

    assert _of_code(graph, CODE_MISSING_SYMBOL) == []
    assert len(_of_code(graph, CODE_SYNTAX_ERROR)) == 1


def test_relative_imports_resolve_within_their_package():
    graph = build_graph(
        _project(
            {
                "app/__init__.py": """
                    from .config import get_settings
                """,
                "app/config.py": """
                    def get_settings() -> dict:
                        return {}
                """,
                "app/agents.py": """
                    from . import config


                    def build():
                        return config.get_settings()
                """,
            }
        )
    )

    assert graph.ok, _explain(graph)
    assert ("app/__init__.py", "app/config.py") in _edge_pairs(graph)
    assert ("app/agents.py", "app/config.py") in _edge_pairs(graph)


@pytest.mark.parametrize(
    "path, source",
    [
        # One dot too many: the anchor for app/agents.py is already the outermost package.
        ("app/agents.py", "from ..outside import thing\n"),
        # A module with no package at all cannot import relative to anything.
        ("main.py", "from .config import get_settings\n"),
    ],
)
def test_relative_import_past_the_top_level_is_reported(path: str, source: str):
    graph = build_graph(_project({path: source, "app/config.py": "STUB = 1\n"}))

    problem = _one(graph, CODE_BAD_RELATIVE_IMPORT)
    assert problem.severity is Severity.BLOCKER
    assert problem.path == path


def test_orphan_file_is_minor_and_entry_points_are_not_orphans():
    graph = build_graph(
        _project(
            {
                "main.py": """
                    from app.config import get_settings
                """,
                "app/config.py": """
                    def get_settings() -> dict:
                        return {}
                """,
                "app/unused.py": """
                    def never_called() -> None:
                        return None
                """,
            },
            entrypoint="main.py",
        )
    )

    problem = _one(graph, CODE_ORPHAN_FILE)
    assert problem.severity is Severity.MINOR
    assert problem.path == "app/unused.py"
    # Untidy, not broken.
    assert graph.ok is True


def test_package_markers_and_conftest_are_not_orphans():
    graph = build_graph(
        _project(
            {
                "app/__init__.py": "",
                "tests/conftest.py": """
                    import pytest


                    @pytest.fixture()
                    def query() -> str:
                        return "hello"
                """,
            },
            dependencies=["pytest>=7.0.0"],
        )
    )

    assert _of_code(graph, CODE_ORPHAN_FILE) == []


# --------------------------------------------------------------------------------------
# Review-issue conversion
# --------------------------------------------------------------------------------------
def test_validate_imports_returns_review_issues_worst_first():
    issues = validate_imports(
        _project(
            {
                "main.py": """
                    from app.missing import helper
                    import mystery_pkg
                """,
                "app/config.py": """
                    STUB = 1
                """,
                "app/unused.py": """
                    OTHER = 2
                """,
            },
            entrypoint="main.py",
        )
    )

    assert issues, "an unimportable project must produce issues"
    assert all(issue.category == "imports" for issue in issues)
    assert issues[0].severity is Severity.BLOCKER
    severities = [issue.severity for issue in issues]
    assert severities == sorted(
        severities,
        key=lambda s: [Severity.BLOCKER, Severity.MAJOR, Severity.MINOR, Severity.INFO].index(s),
    )
    assert any(issue.suggestion for issue in issues)


def test_a_clean_project_produces_no_issues():
    assert validate_imports(_project(_CHAIN, entrypoint="main.py")) == []


# --------------------------------------------------------------------------------------
# The round trip: plan, assemble, and check the emitted project against itself
# --------------------------------------------------------------------------------------
_REQUIREMENT = (
    "Research a topic using web search, verify the sources, and write a short report "
    "with citations."
)


def _assembled(framework: str):
    analysis = heuristic_analysis(_REQUIREMENT)
    choice = select_framework(analysis, framework)
    manifest = plan_architecture(analysis, choice, provider="openai")
    return manifest, assemble_project(manifest)


@pytest.mark.parametrize("framework", sorted(EMITTERS))
def test_assembled_project_imports_cleanly(framework: str):
    """
    Every emitted project must be internally consistent before anything calls it READY.

    This is section 28's gate applied to real output: every internal import resolves to a file
    that exists, every imported name is defined where it is imported from, and every third-party
    root is covered by the project's own requirements.txt. A failure here is a generator defect,
    not a test to relax.
    """
    manifest, project = _assembled(framework)

    assert manifest.framework == framework
    graph = build_graph(project, manifest)

    assert graph.blocking_problems == [], (
        f"{framework} assembled a project that cannot import:\n{_explain(graph)}"
    )
    assert graph.unresolved_imports() == {}, _explain(graph)
    assert graph.summary()["files"] == len(project.source_files())


@pytest.mark.parametrize("framework", sorted(EMITTERS))
def test_assembled_project_has_a_reachable_entry_point(framework: str):
    manifest, project = _assembled(framework)
    graph = build_graph(project, manifest)

    assert graph.entrypoint, f"{framework} produced no entry point"
    assert graph.entrypoint in graph.nodes
    assert graph.nodes[graph.entrypoint].is_entrypoint is True
    # An entry point that imports nothing from the project would mean the modules around it are
    # decoration. Every tier places the workflow outside the entry point.
    assert graph.dependencies_of(graph.entrypoint), (
        f"{framework} entry point {graph.entrypoint} imports nothing from its own project"
    )


@pytest.mark.parametrize("framework", sorted(EMITTERS))
def test_no_emitted_file_is_unreachable(framework: str):
    manifest, project = _assembled(framework)
    graph = build_graph(project, manifest)

    orphans = [problem.path for problem in _of_code(graph, CODE_ORPHAN_FILE)]
    assert orphans == [], f"{framework} emitted files nothing imports: {orphans}"


@pytest.mark.parametrize("framework", sorted(EMITTERS))
def test_plan_and_emitted_code_disagreements_are_information_only(framework: str):
    """
    The plan is a prediction. Where it and the emitted imports differ, the graph reports the
    difference without letting it withhold a working project - and the graph the user is shown
    is read from the code, so it stays right either way.
    """
    manifest, project = _assembled(framework)
    graph = build_graph(project, manifest)

    drift = _of_code(graph, CODE_UNPLANNED_EDGE) + _of_code(graph, CODE_UNREALISED_EDGE)
    assert all(problem.severity is Severity.INFO for problem in drift)
    for path in graph.drift:
        assert path in graph.nodes


@pytest.mark.parametrize("framework", sorted(EMITTERS))
def test_graph_is_stable_across_two_builds(framework: str):
    """
    Two builds of one project must produce the same graph. The Files tab, the impact analysis
    and the repair engine all read this, and a set-iteration order leaking into the output would
    make them disagree with each other for no reason.
    """
    _, project = _assembled(framework)

    first = build_graph(project).as_dict()
    second = build_graph(project).as_dict()

    assert first == second


def test_declared_dependencies_cover_every_third_party_import():
    """
    A generated project must state its own requirements. This caught python-dotenv: the emitted
    config module imports it at module level, and nothing declared it - it happened to be
    installed because the generator itself depends on it, so an exported project only failed on
    someone else's machine.
    """
    missing: Dict[str, List[str]] = {}
    for framework in sorted(EMITTERS):
        manifest, project = _assembled(framework)
        graph = build_graph(project, manifest)
        undeclared = sorted(
            {
                root
                for node in graph.nodes.values()
                for root in node.unresolved
            }
        )
        if undeclared:
            missing[framework] = undeclared

    assert missing == {}, f"imports no requirement provides: {missing}"
