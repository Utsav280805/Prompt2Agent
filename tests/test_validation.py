# tests/test_validation.py
"""
Tests for the readiness gates.

:mod:`multi_agent_generator.core.validation` answers one question - may this project be
described as ready - and the cost of getting it wrong is asymmetric in both directions. A missed
blocker hands the user a project that dies on its first import, which is the complaint the whole
rewrite exists to answer. A false blocker withholds a project that would have worked, and leaves
the user nothing to fix. So most of what follows is not "does the check fire" but "does it fire
on exactly the right thing", with a matching test for the near miss that must stay quiet.

Four groups.

The first pins the status algebra: what blocks, what does not, and specifically that a *skipped*
required check blocks, because "we never ran the tests" and "the tests passed" must not render
as the same green tick.

The second is section 17's central demand - a project that passes every readable gate is still
not ready until something has actually run it.

The third walks each category with a hand-written project small enough to reason about, always
in pairs: the broken case, and the innocent lookalike that must stay quiet.

The fourth is the round trip. Every framework's real assembled output goes through the static
gates and must pass all of them. That is the group that would have caught the Agno tool-symbol
bug, and it is what stops this module drifting away from the emitters.

No network, no credentials, no agent framework installed::

    pip install pytest
    pytest tests/test_validation.py
"""
from __future__ import annotations

import textwrap
from typing import Dict, Mapping, Optional, Sequence, Tuple

import pytest

from multi_agent_generator.core.analysis import heuristic_analysis
from multi_agent_generator.core.architecture import plan_architecture
from multi_agent_generator.core.depgraph import build_graph
from multi_agent_generator.core.manifest import (
    FUNCTION_TOOL_FRAMEWORKS,
    AgentSpec,
    EnvVarSpec,
    FileSpec,
    ProjectManifest,
    ToolSpec,
    WorkflowEdge,
)
from multi_agent_generator.core.models import (
    GeneratedProject,
    RunResult,
    RunStatus,
    Severity,
    # Aliased on import for the same reason as in tests/test_generated_bundles.py: pytest tries
    # to *collect* any module-level name beginning with "Test", and a dataclass with an
    # __init__ it cannot construct produces a PytestCollectionWarning on every run. The class
    # is named for the report it holds, not for pytest, so the alias belongs here rather than a
    # rename in core/models.py.
    TestReport as Report,
    relative_path,
)
from multi_agent_generator.core.selection import select_framework
from multi_agent_generator.core.validation import (
    CATEGORIES,
    CATEGORY_LABELS,
    MAX_REASONABLE_FILE_LINES,
    RUN_ENTRY_NAMES,
    Check,
    CheckStatus,
    ValidationReport,
    static_checks,
    validate_project,
)
from multi_agent_generator.emit.assemble import EMITTERS, assemble_project

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _project(
    files: Mapping[str, str],
    *,
    dependencies: Sequence[str] = ("crewai>=0.80.0", "python-dotenv>=1.0.0"),
    entrypoint: Optional[str] = "main.py",
    framework: str = "crewai",
) -> GeneratedProject:
    """A GeneratedProject from a path -> source mapping, dedented so the tests read like files."""
    project = GeneratedProject(
        framework=framework,
        provider="openai",
        model="gpt-4.1-mini",
        dependencies=list(dependencies),
        entrypoint=entrypoint,
    )
    for path, content in files.items():
        project.add_file(
            path,
            textwrap.dedent(content).strip() + "\n",
            is_entrypoint=path == entrypoint,
        )
    return project


#: A minimal project that satisfies every static gate.
#:
#: Each test below starts from this and breaks exactly one thing, so a failure names the thing
#: that broke rather than producing a pile of unrelated findings. It deliberately imports no
#: third-party package: the gates being tested are structural, and a stub that needed ``crewai``
#: installed in order to be parsed would drag the framework into a unit test.
_SOUND: Dict[str, str] = {
    "config.py": '''
        """Settings."""
        import os


        class Settings:
            def __init__(self, api_key: str, model: str) -> None:
                self.api_key = api_key
                self.model = model


        def get_settings() -> Settings:
            return Settings(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("AGENT_MODEL", "gpt-4.1-mini"),
            )
    ''',
    "tools.py": '''
        """Tools."""
        from typing import List


        class SearchTool:
            name = "search"

            def run(self, query: str) -> str:
                return f"results for {query}"


        TOOLS: List[object] = [SearchTool()]


        def all_tools() -> List[object]:
            return list(TOOLS)
    ''',
    "workflow.py": '''
        """Workflow."""
        from config import get_settings
        from tools import all_tools


        def build_researcher():
            return {"tools": all_tools(), "model": get_settings().model}


        def build_workflow():
            return [build_researcher()]


        def run_workflow(query: str) -> str:
            steps = build_workflow()
            return f"{len(steps)} step(s) answered: {query}"
    ''',
    "main.py": '''
        """Entry point."""
        import sys

        from workflow import run_workflow


        def main() -> int:
            try:
                print(run_workflow(" ".join(sys.argv[1:]) or "hello"))
            except Exception as exc:
                print(f"The run failed: {exc}")
                return 1
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
    ''',
    "tests/test_workflow.py": '''
        """Offline test."""
        from workflow import run_workflow


        def test_workflow_runs():
            assert "answered" in run_workflow("ping")
    ''',
    "requirements.txt": "crewai>=0.80.0\npython-dotenv>=1.0.0\n",
    ".env.example": "# Environment\nOPENAI_API_KEY=\n\nAGENT_MODEL=gpt-4.1-mini\n",
}


def _sound(**overrides: Optional[str]) -> GeneratedProject:
    """
    The sound project, with named files replaced or removed.

    Keyword names are paths with ``__`` for ``/`` and ``_dot_`` for ``.``, so
    ``tests__test_workflow_dot_py`` means ``tests/test_workflow.py``. Passing ``None`` deletes
    the file, which is how the "missing file" cases are written.
    """
    files = dict(_SOUND)
    for key, content in overrides.items():
        path = key.replace("__", "/").replace("_dot_", ".")
        if content is None:
            files.pop(path, None)
        else:
            files[path] = content
    return _project(files)


def _manifest(**overrides: object) -> ProjectManifest:
    """A manifest with the four required fields filled in, so a test states only what it means."""
    fields: Dict[str, object] = {
        "project_name": "research-agent",
        "framework": "crewai",
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "requirement": "Research a topic and write it up.",
    }
    fields.update(overrides)
    return ProjectManifest(**fields)  # type: ignore[arg-type]


def _by_id(checks: Sequence[Check]) -> Dict[str, Check]:
    return {check.id: check for check in checks}


def _explain(checks: Sequence[Check]) -> str:
    """A readable failure message. ``assert report.ready`` on its own tells you nothing."""
    interesting = [c for c in checks if c.status is not CheckStatus.PASSED]
    if not interesting:
        return "every check passed"
    return "\n".join(
        f"  [{c.status.value}/{c.severity.value}{'/blocking' if c.blocking else ''}] "
        f"{c.id}: {c.summary}"
        for c in interesting
    )


def _check(
    status: CheckStatus,
    *,
    severity: Severity = Severity.MAJOR,
    required: bool = True,
) -> Check:
    return Check(
        id="example.check",
        category="imports",
        title="An example",
        status=status,
        summary="Something was found.",
        severity=severity,
        required=required,
    )


# --------------------------------------------------------------------------------------
# The status algebra - what blocks, and what deliberately does not
# --------------------------------------------------------------------------------------
def test_a_failed_blocker_blocks():
    assert _check(CheckStatus.FAILED, severity=Severity.BLOCKER).blocking is True


def test_a_failed_major_blocks():
    assert _check(CheckStatus.FAILED, severity=Severity.MAJOR).blocking is True


def test_a_failed_minor_does_not_block():
    """
    Severity decides, not status.

    A stub tool and a missing framework package are both failures, and only one of them stops
    the agent running. Collapsing the two would either withhold working projects or ship broken
    ones, depending on which way the collapse went.
    """
    assert _check(CheckStatus.FAILED, severity=Severity.MINOR).blocking is False
    assert _check(CheckStatus.FAILED, severity=Severity.INFO).blocking is False


def test_a_warning_never_blocks():
    for severity in Severity:
        assert _check(CheckStatus.WARNING, severity=severity).blocking is False


def test_a_required_skipped_check_blocks():
    """
    The reason the enum has four members instead of two.

    An unverified property has not been established. Reporting it as ready is the exact
    behaviour section 59 forbids, so a required check with nothing to look at blocks just as a
    failure would - regardless of how mild its severity is.
    """
    assert _check(CheckStatus.SKIPPED, severity=Severity.INFO, required=True).blocking is True


def test_an_optional_skipped_check_does_not_block():
    assert _check(CheckStatus.SKIPPED, required=False).blocking is False


def test_an_empty_report_is_not_ready():
    """No checks means nothing was verified, which is not the same as nothing being wrong."""
    assert ValidationReport().ready is False


def test_ready_is_derived_from_the_checks():
    report = ValidationReport(checks=[_check(CheckStatus.PASSED, severity=Severity.INFO)])
    assert report.ready is True
    report.checks.append(_check(CheckStatus.FAILED, severity=Severity.BLOCKER))
    assert report.ready is False


def test_every_check_is_filed_under_a_known_category():
    """
    A typo in a category name would silently drop the check from the dashboard.

    ``by_category`` only walks :data:`CATEGORIES`, so a check filed under "entrypoints" instead
    of "entrypoint" would still block readiness while being invisible in the UI - leaving the
    user with a project reported as not ready and no stated reason why.
    """
    report = validate_project(_sound(), tests=None)
    unknown = sorted({c.category for c in report.checks} - set(CATEGORIES))
    assert unknown == [], f"checks filed under unknown categories: {unknown}"
    assert set(CATEGORIES) <= set(CATEGORY_LABELS)


def test_by_category_shows_every_check_exactly_once():
    report = validate_project(_sound(), tests=None)
    rendered = [c["id"] for group in report.by_category() for c in group["checks"]]
    assert sorted(rendered) == sorted(c.id for c in report.checks)
    assert len(rendered) == len(set(rendered))


def test_check_ids_are_unique():
    """Two checks with one id makes ``report.get`` ambiguous and the rendered list inconsistent."""
    report = validate_project(_sound(), tests=None)
    ids = [c.id for c in report.checks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert duplicates == [], f"duplicate check ids: {duplicates}"


# --------------------------------------------------------------------------------------
# Generation succeeding is not readiness - section 17's central demand
# --------------------------------------------------------------------------------------
def test_a_perfect_project_is_not_ready_until_its_tests_have_run():
    """
    The single most important assertion in this file.

    ``_SOUND`` passes every gate that can be decided by reading it. It is still not ready,
    because nothing has run it. That is the difference between "we generated code" and "we
    delivered a working agent", and it is enforced here by data rather than by a comment.
    """
    report = validate_project(_sound(), tests=None)

    readable = [c for c in report.checks if c.category not in ("tests", "runtime")]
    assert [c for c in readable if c.blocking] == [], _explain(report.checks)

    assert report.ready is False
    executed = report.get("tests.executed")
    assert executed is not None
    assert executed.status is CheckStatus.SKIPPED
    assert executed.blocking is True
    assert "not been run" in executed.summary


def test_a_project_becomes_ready_once_its_tests_pass():
    report = validate_project(
        _sound(),
        tests=Report(status="passed", exit_code=0, passed_count=3, duration_s=1.2),
    )
    assert report.ready is True, _explain(report.checks)
    assert report.get("tests.passed").status is CheckStatus.PASSED
    assert report.get("runtime.offline").status is CheckStatus.PASSED
    assert report.entrypoint == "main.py"


def test_failing_tests_block_and_the_remedy_does_not_suggest_hiding_them():
    """
    Section 18 in assertion form.

    The remedy text is part of the product: a suggestion to raise the timeout or delete the test
    is how a suite becomes green while the project stays broken.
    """
    report = validate_project(
        _sound(),
        tests=Report(
            status="failed",
            exit_code=1,
            passed_count=1,
            failed_count=2,
            failed_tests=["tests/test_workflow.py::test_workflow_runs"],
        ),
    )
    failed = report.get("tests.passed")
    assert failed.status is CheckStatus.FAILED
    assert failed.severity is Severity.BLOCKER
    assert report.ready is False
    assert "Do not raise the timeout" in failed.fix
    # The suite did run, and saying otherwise would send the repair engine after the wrong thing.
    assert report.get("tests.executed").status is CheckStatus.PASSED


def test_a_timed_out_suite_is_not_reported_as_having_run():
    report = validate_project(
        _sound(),
        tests=Report(status="timeout", timed_out=True, duration_s=300.0),
    )
    executed = report.get("tests.executed")
    assert executed.status is CheckStatus.SKIPPED
    assert executed.blocking is True
    assert report.ready is False


def test_a_suite_that_could_not_run_is_skipped_not_passed():
    report = validate_project(
        _sound(),
        tests=Report(
            status="error",
            exit_code=2,
            unavailable_reason="pytest could not be installed, so the suite did not run.",
        ),
    )
    executed = report.get("tests.executed")
    assert executed.status is CheckStatus.SKIPPED
    assert executed.blocking is True
    assert "could not be installed" in executed.summary
    assert report.ready is False


def test_an_all_skipped_suite_has_verified_nothing():
    """
    A suite that exits zero having skipped everything is the quietest way to fake readiness.

    ``TestReport.ok`` is true here - nothing failed - so the naive reading is a pass. What
    actually happened is that no assertion ran, and both the tests gate and the runtime gate
    have to say so.
    """
    report = validate_project(
        _sound(),
        tests=Report(status="passed", exit_code=0, passed_count=0, skipped_count=7),
    )
    meaningful = report.get("tests.meaningful")
    assert meaningful.status is CheckStatus.SKIPPED
    assert meaningful.blocking is True

    offline = report.get("runtime.offline")
    assert offline.status is CheckStatus.SKIPPED
    assert offline.blocking is True
    # Not "executed and did not complete": nothing was executed, and inventing an observed
    # failure is the same dishonesty as inventing an observed pass, only in the other direction.
    assert "skipped rather than executed" in offline.summary

    assert report.ready is False


def test_a_missing_test_suite_blocks_but_can_be_waived():
    without = _sound(tests__test_workflow_dot_py=None)
    assert _by_id(static_checks(without))["tests.present"].blocking is True
    waived = _by_id(static_checks(without, expect_tests=False))["tests.present"]
    assert waived.status is CheckStatus.SKIPPED
    assert waived.blocking is False


# --------------------------------------------------------------------------------------
# Runtime: the offline gate is required, the paid one is not
# --------------------------------------------------------------------------------------
def test_a_passing_suite_that_never_runs_the_workflow_does_not_prove_it_works():
    """
    Imports resolving is not the workflow running.

    A generated suite that only asserts modules import would pass while the agent is incapable
    of answering anything, so the offline runtime gate looks for a test that actually calls into
    the workflow.
    """
    imports_only = _sound(
        tests__test_workflow_dot_py='''
            import workflow


            def test_module_imports():
                assert workflow is not None
        '''
    )
    report = validate_project(
        imports_only,
        tests=Report(status="passed", exit_code=0, passed_count=1),
    )
    offline = report.get("runtime.offline")
    assert offline.status is CheckStatus.FAILED
    assert offline.blocking is True
    assert report.ready is False


def test_a_real_provider_run_is_never_required_for_readiness():
    """
    Section 59 cuts both ways.

    A live call costs money and needs a credential the machine may not have, so gating readiness
    on it would mean nothing is ever ready in CI. But flattening "has had a real run" into "has
    not" would overclaim, so it stays a visible, optional check.
    """
    report = validate_project(
        _sound(),
        tests=Report(status="passed", exit_code=0, passed_count=2),
        runtime=None,
    )
    live = report.get("runtime.live")
    assert live.status is CheckStatus.SKIPPED
    assert live.required is False
    assert live.blocking is False
    assert report.ready is True, _explain(report.checks)


def test_a_failed_live_run_is_reported_without_blocking_readiness():
    report = validate_project(
        _sound(),
        tests=Report(status="passed", exit_code=0, passed_count=2),
        runtime=RunResult(
            status=RunStatus.FAILED,
            exit_code=1,
            error={"message": "The provider rejected the request: invalid API key."},
            stderr="AuthenticationError: invalid api key",
        ),
    )
    live = report.get("runtime.live")
    assert live.status is CheckStatus.FAILED
    assert "invalid API key" in live.summary
    assert live.blocking is False
    assert report.ready is True, _explain(report.checks)


def test_a_successful_live_run_reports_derived_counts_only():
    """Section 87: the numbers shown are lengths of real lists, not decoration."""
    report = validate_project(
        _sound(),
        tests=Report(status="passed", exit_code=0, passed_count=2),
        runtime=RunResult(
            status=RunStatus.SUCCEEDED,
            exit_code=0,
            duration_s=4.25,
            output="done",
            steps=[{"agent": "researcher"}, {"agent": "writer"}],
            tool_calls=[{"tool": "search"}],
            entry="workflow:run_workflow",
        ),
    )
    live = report.get("runtime.live")
    assert live.status is CheckStatus.PASSED
    assert "2 steps" in live.evidence
    assert "1 tool calls" in live.evidence


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
def test_a_project_with_no_settings_module_fails():
    checks = _by_id(static_checks(_sound(config_dot_py=None)))
    assert checks["configuration.settings_module"].status is CheckStatus.FAILED


def test_the_settings_check_names_the_file_that_defines_get_settings_not_one_that_imports_it():
    """
    The near miss for the test above, and the bug that made that test pass when it should fail.

    ``workflow.py`` says ``from config import get_settings``, which makes the name importable
    *from* ``workflow.py``. It does not make ``workflow.py`` the configuration module. The
    validator was reading the graph's ``provides`` list, which includes imported bindings by
    design, so deleting ``config.py`` left the check green and its summary announcing that
    configuration was loaded from one place - ``workflow.py``. That is a fabricated answer to
    "where does this project read its settings", which section 9 rules out, so the check reads
    ``defines`` instead. Here the module is renamed rather than deleted: the answer must follow
    the definition.
    """
    project = _sound(
        config_dot_py=None,
        settings_dot_py='''
            import os


            def get_settings():
                return {
                    "key": os.getenv("OPENAI_API_KEY", ""),
                    "model": os.getenv("AGENT_MODEL", "gpt-4.1-mini"),
                }
        ''',
        workflow_dot_py='''
            from settings import get_settings
            from tools import all_tools


            def build_researcher():
                return {"role": "Researcher", "settings": get_settings()}


            def build_workflow():
                return [build_researcher(), all_tools()]


            def run_workflow(query: str) -> str:
                return str(build_workflow()) + query
        ''',
    )
    check = _by_id(static_checks(project))["configuration.settings_module"]
    assert check.status is CheckStatus.PASSED
    assert "settings.py" in check.summary + " ".join(check.evidence)
    assert "workflow.py" not in check.summary


def test_a_planned_variable_missing_from_env_example_fails():
    stripped = _sound(_dot_env_dot_example="# Environment\nAGENT_MODEL=gpt-4.1-mini\n")
    manifest = _manifest(env_vars=[EnvVarSpec(name="OPENAI_API_KEY", purpose="The credential.")])
    check = _by_id(static_checks(stripped, manifest))["configuration.env_documented"]
    assert check.status is CheckStatus.FAILED
    assert "OPENAI_API_KEY" in check.summary


def test_an_undocumented_variable_warns_rather_than_blocks():
    """
    A variable the code reads but nobody wrote down is worth saying and not worth withholding
    the project over - the agent still runs, it just has an undocumented knob.
    """
    extra = _sound(
        config_dot_py='''
            import os


            def get_settings():
                return {
                    "key": os.getenv("OPENAI_API_KEY", ""),
                    "model": os.getenv("AGENT_MODEL", "gpt-4.1-mini"),
                    "region": os.environ["SEARCH_REGION"],
                }
        '''
    )
    check = _by_id(static_checks(extra))["configuration.env_documented"]
    assert check.status is CheckStatus.WARNING
    assert "SEARCH_REGION" in check.summary
    assert check.blocking is False


def test_a_variable_only_a_test_reads_is_not_demanded_in_env_example():
    """
    The near miss for the check above.

    ``source_files()`` includes the tests, so the naive implementation asked ``.env.example`` to
    document variables only ``conftest.py`` sets - a finding about nothing, on every project.
    """
    project = _sound(
        tests__conftest_dot_py='''
            import os


            def pytest_configure():
                os.environ.setdefault("PYTEST_ONLY_FLAG", "1")
                assert os.getenv("PYTEST_ONLY_FLAG")
        '''
    )
    check = _by_id(static_checks(project))["configuration.env_documented"]
    assert check.status is CheckStatus.PASSED, check.summary
    assert "PYTEST_ONLY_FLAG" not in check.detail


def test_a_missing_env_example_fails():
    checks = _by_id(static_checks(_sound(_dot_env_dot_example=None)))
    assert checks["configuration.env_documented"].status is CheckStatus.FAILED


# --------------------------------------------------------------------------------------
# Credentials in source - section 34
# --------------------------------------------------------------------------------------
def test_a_hard_coded_provider_key_is_a_blocker_and_is_never_quoted_back():
    leaked = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2"
    project = _sound(
        config_dot_py=f'''
            import os


            def get_settings():
                return {{"key": "{leaked}", "model": os.getenv("AGENT_MODEL", "x")}}
        '''
    )
    check = _by_id(static_checks(project))["configuration.secrets"]

    assert check.status is CheckStatus.FAILED
    assert check.severity is Severity.BLOCKER
    assert check.blocking is True
    assert "config.py" in check.paths
    assert "config.py:5 contains an OpenAI-style secret key" in check.detail
    # The whole point: the report proves a leak without republishing it.
    rendered = " ".join([check.summary, check.detail, check.fix, *check.evidence])
    assert leaked not in rendered


def test_a_configured_platform_secret_found_in_source_is_reported():
    """
    The high-confidence pass.

    This value has no vendor prefix and no recognisable shape - it is caught only because the
    caller said it is a credential the platform holds.
    """
    project = _sound(
        config_dot_py='''
            def get_settings():
                return {"key": "9f2c7d1e4b6a8350"}
        '''
    )
    check = _by_id(static_checks(project, secrets=["9f2c7d1e4b6a8350"]))["configuration.secrets"]
    assert check.status is CheckStatus.FAILED
    assert "a configured credential value" in check.detail
    assert "9f2c7d1e4b6a8350" not in check.detail


def test_the_generated_conftest_placeholders_are_not_reported_as_leaks():
    """
    The false positive this check was one line away from having on every project.

    The emitted ``tests/conftest.py`` fills the environment with deliberately fake credentials
    of the form ``test-openai-api-key``. Flagging those would mean every project ever generated
    reported a BLOCKER credential leak - and a check that fires on everything is one whose real
    findings stop being read.
    """
    project = _sound(
        tests__conftest_dot_py='''
            import os

            PLACEHOLDER_ENV = {
                "OPENAI_API_KEY": "test-openai-api-key",
                "HUGGINGFACEHUB_API_TOKEN": "test-huggingfacehub-api-token",
            }

            for name, value in PLACEHOLDER_ENV.items():
                os.environ.setdefault(name, value)
        '''
    )
    check = _by_id(static_checks(project))["configuration.secrets"]
    assert check.status is CheckStatus.PASSED, check.detail


def test_prose_about_api_keys_is_not_reported_as_a_leak():
    """
    The other near miss.

    Generated configuration modules document their fields, and a comment reading
    ``API_KEY: "the credential used to authenticate"`` matches the generic assignment shape
    unless the pattern refuses values containing spaces.
    """
    project = _sound(
        config_dot_py='''
            import os

            #: API_KEY: "the provider credential read from the environment at startup"
            NOTE = "not a real value, just documentation prose"


            def get_settings():
                return {"key": os.getenv("OPENAI_API_KEY", ""), "note": NOTE}
        '''
    )
    check = _by_id(static_checks(project))["configuration.secrets"]
    assert check.status is CheckStatus.PASSED, check.detail


def test_reading_a_key_from_the_environment_is_not_a_leak():
    check = _by_id(static_checks(_sound()))["configuration.secrets"]
    assert check.status is CheckStatus.PASSED, check.detail


# --------------------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------------------
def test_a_missing_requirements_file_is_a_blocker():
    check = _by_id(static_checks(_sound(requirements_dot_txt=None)))["dependencies.file"]
    assert check.status is CheckStatus.FAILED
    assert check.severity is Severity.BLOCKER


def test_a_project_that_does_not_declare_its_own_framework_is_a_blocker():
    """The failure mode is total: nothing in the project imports on a clean machine."""
    check = _by_id(
        static_checks(_sound(requirements_dot_txt="python-dotenv>=1.0.0\n"))
    )["dependencies.framework"]
    assert check.status is CheckStatus.FAILED
    assert check.severity is Severity.BLOCKER
    assert "crewai" in check.summary


def test_an_uninstallable_requirement_line_fails():
    """
    The execution layer refuses URL, path and alternative-index installs.

    A project containing one would be reported ready and then fail to install, so the rule is
    applied here from the same place that enforces it there.
    """
    check = _by_id(
        static_checks(
            _sound(
                requirements_dot_txt=(
                    "crewai @ git+https://example.invalid/repo\n"
                    "python-dotenv>=1.0.0\n"
                    "--index-url https://example.invalid/simple\n"
                )
            )
        )
    )["dependencies.installable"]
    assert check.status is CheckStatus.FAILED
    assert check.blocking is True


def test_an_unknown_framework_skips_the_framework_check_without_blocking():
    """
    A framework nobody registered is a gap in this repository, not a defect in the user's
    project. Blocking on it would withhold a project for a reason the user cannot act on.
    """
    project = _project(dict(_SOUND), framework="some-new-framework")
    check = _by_id(static_checks(project))["dependencies.framework"]
    assert check.status is CheckStatus.SKIPPED
    assert check.required is False
    assert check.blocking is False


# --------------------------------------------------------------------------------------
# Imports and structure - the graph folded into checks
# --------------------------------------------------------------------------------------
def test_a_clean_project_shows_a_pass_for_every_graph_code():
    """
    Every code the graph can report becomes a visible row even when it found nothing.

    A dashboard that only lists what went wrong cannot be read as "we checked this", which is
    the difference between a report and a pile of errors.
    """
    checks = _by_id(static_checks(_sound()))
    for check_id in (
        "structure.syntax",
        "imports.internal",
        "imports.third_party",
        "imports.symbols",
        "imports.cycles",
        "imports.relative",
        "imports.ambiguous",
        "imports.test_isolation",
        "structure.orphans",
    ):
        assert checks[check_id].status is CheckStatus.PASSED, checks[check_id].summary


def test_an_import_of_a_file_that_does_not_exist_fails():
    """
    A module that looks like part of this project and is not there is a blocker.

    ``tools`` is a real module here, so ``tools.helpers`` is unambiguously a claim about this
    project's own layout - and the whole complaint that started this work was generation
    reporting success and the first import dying. Note the import target: an unresolvable *bare*
    name is a different finding, which the next test pins down.
    """
    project = _sound(
        workflow_dot_py='''
            from config import get_settings
            from tools.helpers import helper
            from tools import all_tools


            def build_workflow():
                return [helper, get_settings(), all_tools()]


            def run_workflow(query: str) -> str:
                return str(build_workflow()) + query
        '''
    )
    check = _by_id(static_checks(project))["imports.internal"]
    assert check.status is CheckStatus.FAILED
    assert check.blocking is True


def test_an_unresolvable_bare_import_is_reported_as_a_dependency_not_a_missing_file():
    """
    The near-miss for the test above, and the reason it had to be retargeted.

    ``from missing_module import helper`` has a root that is neither a module in this project,
    nor stdlib, nor a declared requirement. The graph cannot tell whether the author meant a
    file that was never generated or a package that was never declared, so it says the honest
    thing - the import will not resolve, add the distribution or drop it - and files it under
    dependencies. Either way it blocks, which is what actually matters; asserting the *category*
    is what keeps the two findings from being quietly merged into one vague one later.
    """
    project = _sound(
        workflow_dot_py='''
            from config import get_settings
            from missing_module import helper
            from tools import all_tools


            def build_workflow():
                return [helper, get_settings(), all_tools()]


            def run_workflow(query: str) -> str:
                return str(build_workflow()) + query
        '''
    )
    checks = _by_id(static_checks(project))
    third_party = checks["imports.third_party"]
    assert third_party.status is CheckStatus.FAILED
    assert third_party.blocking is True
    assert "missing_module" in (third_party.detail or "") + third_party.summary
    # And it must not be misfiled as a missing project file.
    assert checks["imports.internal"].status is CheckStatus.PASSED


def test_a_syntax_error_is_reported_under_structure_and_does_not_crash_validation():
    report = validate_project(_sound(workflow_dot_py="def build_workflow(:\n    pass\n"))
    assert report.get("structure.syntax").status is CheckStatus.FAILED
    assert report.ready is False


def test_one_enormous_file_fails_the_structure_gate():
    """Sections 3 and 4: a single thousand-line ``agent.py`` is not an acceptable deliverable."""
    filler = "\n".join(f"X{i} = {i}" for i in range(MAX_REASONABLE_FILE_LINES + 40))
    project = _project(
        {
            "main.py": (
                "import os\n\n\n"
                f"{filler}\n\n\n"
                'def run_workflow(query):\n'
                '    return os.getenv("AGENT_MODEL", "") + query\n\n\n'
                "def main():\n"
                "    try:\n"
                '        print(run_workflow("hi"))\n'
                "    except Exception as exc:\n"
                "        print(exc)\n"
            ),
            "requirements.txt": "crewai>=0.80.0\npython-dotenv>=1.0.0\n",
            ".env.example": "AGENT_MODEL=gpt-4.1-mini\n",
        }
    )
    check = _by_id(static_checks(project, expect_tests=False))["structure.multi_file"]
    assert check.status is CheckStatus.FAILED
    assert "single" in check.summary


def test_a_long_readme_does_not_make_a_small_project_look_oversized():
    """
    The near miss, and the reason this gate counts application lines rather than every line.

    ``GeneratedProject.total_lines`` sums the README and the Dockerfile too, so a single-module
    project with thorough documentation was reported as one enormous Python file - telling the
    user to split up code that was thirty lines long.
    """
    project = _project(
        {
            "main.py": (
                "import os\n\n\n"
                'def run_workflow(query):\n'
                '    return os.getenv("AGENT_MODEL", "") + query\n\n\n'
                "def main():\n"
                "    try:\n"
                '        print(run_workflow("hi"))\n'
                "    except Exception as exc:\n"
                "        print(exc)\n"
            ),
            "README.md": "\n".join(f"Documentation line {i}." for i in range(900)) + "\n",
            "requirements.txt": "crewai>=0.80.0\npython-dotenv>=1.0.0\n",
            ".env.example": "AGENT_MODEL=gpt-4.1-mini\n",
        }
    )
    check = _by_id(static_checks(project, expect_tests=False))["structure.multi_file"]
    assert check.status is CheckStatus.PASSED, check.summary


def test_a_planned_file_that_was_never_written_fails():
    manifest = _manifest(
        files=[
            FileSpec(path="workflow.py", purpose="The workflow"),
            FileSpec(path="memory.py", purpose="Conversation memory"),
        ]
    )
    check = _by_id(static_checks(_sound(), manifest))["structure.planned_files"]
    assert check.status is CheckStatus.FAILED
    assert "memory.py" in check.detail


def test_a_project_with_no_plan_is_not_declared_broken():
    """
    A project restored from an older record has no stored manifest. Reporting it as failed
    because its plan is absent would be a finding about our own storage, not about the project.
    """
    report = validate_project(
        _sound(),
        manifest=None,
        tests=Report(status="passed", exit_code=0, passed_count=2),
    )
    planned = report.get("structure.planned_files")
    assert planned.status is CheckStatus.SKIPPED
    assert planned.blocking is False
    assert report.ready is True, _explain(report.checks)
    assert any("no architecture plan" in note.lower() for note in report.notes)


# --------------------------------------------------------------------------------------
# Entry point - the contract the execution layer depends on
# --------------------------------------------------------------------------------------
def test_the_runner_and_the_validator_agree_on_the_run_function_names():
    """
    One list, two consumers.

    ``core.validation`` gates a project on having one of these functions; ``execution.runner``
    is what actually goes looking for one. A second copy of the list would eventually let a
    project pass the gate and then fail to launch, with each side insisting the other was wrong.
    """
    from multi_agent_generator.execution import runner

    assert runner._RUN_NAMES is RUN_ENTRY_NAMES


def test_a_project_with_no_run_function_is_a_blocker():
    project = _sound(
        workflow_dot_py='''
            from config import get_settings
            from tools import all_tools


            def build_workflow():
                return [get_settings(), all_tools()]
        ''',
        main_dot_py='''
            from workflow import build_workflow

            if __name__ == "__main__":
                try:
                    print(build_workflow())
                except Exception as exc:
                    print(f"The run failed: {exc}")
        ''',
        tests__test_workflow_dot_py='''
            from workflow import build_workflow


            def test_builds():
                assert build_workflow()
        ''',
    )
    check = _by_id(static_checks(project))["entrypoint.runnable"]
    assert check.status is CheckStatus.FAILED
    assert check.severity is Severity.BLOCKER


def test_a_run_function_defined_only_in_a_test_does_not_satisfy_the_gate():
    """
    The near miss.

    The runner imports application modules, never the tests, so a ``run_workflow`` that exists
    only under ``tests/`` is a function nothing will ever call.
    """
    project = _sound(
        workflow_dot_py='''
            from config import get_settings
            from tools import all_tools


            def build_workflow():
                return [get_settings(), all_tools()]
        ''',
        main_dot_py='''
            from workflow import build_workflow

            if __name__ == "__main__":
                try:
                    print(build_workflow())
                except Exception as exc:
                    print(f"The run failed: {exc}")
        ''',
        tests__test_workflow_dot_py='''
            from workflow import build_workflow


            def run_workflow(query):
                return query


            def test_builds():
                assert build_workflow()
                assert run_workflow("x") == "x"
        ''',
    )
    assert _by_id(static_checks(project))["entrypoint.runnable"].status is CheckStatus.FAILED


def test_a_project_that_only_prints_its_answer_is_flagged_without_being_withheld():
    """
    A workflow that prints instead of returning cannot be driven by the Playground, because
    there is nothing to capture. That is worth reporting - and it is not worth withholding a
    project the user can still run from a terminal, so it does not block.
    """
    project = _sound(
        workflow_dot_py='''
            from config import get_settings
            from tools import all_tools


            def build_workflow():
                return [get_settings(), all_tools()]


            def main() -> None:
                print(f"{len(build_workflow())} step(s) ran")
        ''',
        main_dot_py='''
            from workflow import main

            if __name__ == "__main__":
                try:
                    main()
                except Exception as exc:
                    print(f"The run failed: {exc}")
        ''',
        tests__test_workflow_dot_py='''
            from workflow import main


            def test_main_runs():
                main()
        ''',
    )
    checks = _by_id(static_checks(project))
    # main() is a name the runner will find, so the project can be started at all.
    assert checks["entrypoint.runnable"].status is CheckStatus.PASSED
    contract = checks["entrypoint.result_contract"]
    assert contract.status is CheckStatus.FAILED
    assert contract.severity is Severity.MINOR
    assert contract.blocking is False


def test_an_entrypoint_naming_a_file_that_was_not_generated_is_a_blocker():
    project = _project(dict(_SOUND), entrypoint="does_not_exist.py")
    checks = _by_id(static_checks(project))
    declared = checks["entrypoint.declared"]
    assert declared.status is CheckStatus.FAILED
    assert declared.severity is Severity.BLOCKER
    assert "does_not_exist.py" in declared.summary


# --------------------------------------------------------------------------------------
# Error handling - section 80
# --------------------------------------------------------------------------------------
def test_an_exception_handler_that_does_nothing_is_reported_with_its_line():
    project = _sound(
        workflow_dot_py='''
            from config import get_settings
            from tools import all_tools


            def build_workflow():
                return [get_settings(), all_tools()]


            def run_workflow(query: str) -> str:
                try:
                    return str(build_workflow()) + query
                except ValueError:
                    pass
                return ""
        '''
    )
    check = _by_id(static_checks(project))["error_handling.no_silent_failures"]
    assert check.status is CheckStatus.FAILED
    assert "workflow.py" in check.paths
    # The line reported is the ``except`` clause itself - line 12 of the file above - because
    # that is the line the reader has to go and look at.
    assert "workflow.py:12 catches ValueError and does nothing" in check.detail


def test_a_handler_that_logs_is_not_reported():
    """The near miss: handling an error visibly is the behaviour we want, not the one we flag."""
    project = _sound(
        workflow_dot_py='''
            from config import get_settings
            from tools import all_tools


            def build_workflow():
                return [get_settings(), all_tools()]


            def run_workflow(query: str) -> str:
                try:
                    return str(build_workflow()) + query
                except ValueError as exc:
                    print(f"could not build the workflow: {exc}")
                    raise
        '''
    )
    check = _by_id(static_checks(project))["error_handling.no_silent_failures"]
    assert check.status is CheckStatus.PASSED, check.detail


def test_a_silent_handler_inside_a_test_is_not_reported_as_application_code():
    """
    The other near miss, and the reason ``_app_files`` exists.

    ``except Exception: pass`` in a generated test is usually an optional-import guard. Flagging
    it would attach a fix written for application code to a file the fix does not apply to.
    """
    project = _sound(
        tests__conftest_dot_py='''
            try:
                import crewai
            except Exception:
                pass
        '''
    )
    check = _by_id(static_checks(project))["error_handling.no_silent_failures"]
    assert check.status is CheckStatus.PASSED, check.detail


def test_an_entrypoint_with_no_error_handling_is_flagged_without_being_withheld():
    project = _sound(
        main_dot_py='''
            from workflow import run_workflow

            if __name__ == "__main__":
                print(run_workflow("hello"))
        '''
    )
    check = _by_id(static_checks(project))["error_handling.entrypoint_guard"]
    assert check.status is CheckStatus.FAILED
    assert check.blocking is False
    assert "traceback" in check.summary


# --------------------------------------------------------------------------------------
# Agents, tools and workflow, checked against the plan
# --------------------------------------------------------------------------------------
def test_a_planned_agent_with_no_builder_in_the_code_is_a_blocker():
    present = AgentSpec(name="researcher")
    absent = AgentSpec(name="summariser")

    # _SOUND defines build_researcher() and nothing else, so the naming convention is itself
    # part of what this test pins down.
    assert present.factory_name == "build_researcher"

    check = _by_id(static_checks(_sound(), _manifest(agents=[present, absent])))["agents.emitted"]
    assert check.status is CheckStatus.FAILED
    assert check.severity is Severity.BLOCKER
    assert absent.factory_name in check.detail


def test_an_agent_referencing_a_tool_the_project_does_not_have_fails():
    manifest = _manifest(agents=[AgentSpec(name="researcher", tools=["nonexistent_tool"])])
    check = _by_id(static_checks(_sound(), manifest))["agents.tools_exist"]
    assert check.status is CheckStatus.FAILED
    assert "nonexistent_tool" in check.detail


def test_a_stub_tool_is_named_plainly_without_withholding_the_project():
    """
    Section 88: a stub that claims to search the web is worse than no tool at all, so it is
    stated loudly - but the agent does run, so it does not block.
    """
    manifest = _manifest(tools=[ToolSpec(name="search", implemented=False)])
    check = _by_id(static_checks(_sound(), manifest))["tools.implemented"]
    assert check.status is CheckStatus.WARNING
    assert check.blocking is False
    assert "placeholder" in check.summary
    assert "not real" in check.fix


def test_a_tool_registry_the_agents_can_read_is_required():
    """
    Tools that exist but are never collected cannot be handed to an agent.

    The class is defined, the import resolves, nothing is broken by any other measure - and the
    agent has no tools, which is exactly the sort of quiet gap that gets shipped as working.
    """
    manifest = _manifest(tools=[ToolSpec(name="search", implemented=True)])
    project = _sound(
        tools_dot_py='''
            class SearchTool:
                name = "search"

                def run(self, query: str) -> str:
                    return f"results for {query}"
        ''',
        workflow_dot_py='''
            from config import get_settings
            from tools import SearchTool


            def build_researcher():
                return {"tools": [SearchTool()], "model": get_settings().model}


            def build_workflow():
                return [build_researcher()]


            def run_workflow(query: str) -> str:
                return f"{len(build_workflow())} step(s) answered: {query}"
        ''',
    )
    checks = _by_id(static_checks(project, manifest))
    assert checks["tools.emitted"].status is CheckStatus.PASSED
    assert checks["tools.registry"].status is CheckStatus.FAILED


def test_a_workflow_edge_pointing_nowhere_fails():
    manifest, project = _assembled("crewai")
    workflow = manifest.primary_workflow
    assert workflow is not None
    workflow.edges.append(WorkflowEdge(source=workflow.entry, target="no_such_node"))

    check = _by_id(static_checks(project, manifest))["workflow.spec"]
    assert check.status is CheckStatus.FAILED
    assert "no_such_node" in check.summary


def test_a_project_that_does_not_build_its_workflow_is_a_blocker():
    manifest, _ = _assembled("crewai")
    project = _sound(
        workflow_dot_py='''
            from config import get_settings
            from tools import all_tools


            def run_workflow(query: str) -> str:
                return f"{len(all_tools())} tool(s), {get_settings().model}: {query}"
        '''
    )
    check = _by_id(static_checks(project, manifest))["workflow.emitted"]
    assert check.status is CheckStatus.FAILED
    assert check.severity is Severity.BLOCKER


def test_a_manifest_with_no_agents_is_a_blocker():
    check = _by_id(static_checks(_sound(), _manifest()))["agents.emitted"]
    assert check.status is CheckStatus.FAILED
    assert check.severity is Severity.BLOCKER


# --------------------------------------------------------------------------------------
# Robustness: validation is the last thing that may crash
# --------------------------------------------------------------------------------------
def test_validation_never_raises_on_an_empty_project():
    report = validate_project(GeneratedProject(framework="crewai", provider="openai", model="m"))
    assert report.ready is False
    assert report.checks
    assert report.unmet()


def test_validation_never_raises_on_unparseable_content():
    project = GeneratedProject(framework="crewai", provider="openai", model="m")
    project.add_file("main.py", "\x00\x01 not python at all")
    project.add_file("tools.py", "def (((")
    report = validate_project(project)
    assert report.ready is False
    assert isinstance(report.as_dict()["categories"], list)


def test_the_report_serialises_without_the_graph():
    """
    ``as_dict`` feeds an HTTP response. The graph is deliberately excluded: it has its own
    ``as_dict`` and belongs under its own key, not nested inside the verdict.
    """
    report = validate_project(_sound(), tests=None)
    payload = report.as_dict()
    assert "graph" not in payload
    assert report.graph is not None
    assert set(payload) >= {"ready", "checks", "failed", "skipped", "categories", "unmet", "notes"}


def test_unmet_reads_as_sentences_a_person_could_act_on():
    report = validate_project(_sound(requirements_dot_txt=None), tests=None)
    assert report.unmet()
    for line in report.unmet():
        label, _, rest = line.partition(":")
        assert label in CATEGORY_LABELS.values(), line
        assert rest.strip(), line


def test_failures_convert_to_review_issues_for_the_repair_engine():
    report = validate_project(_sound(requirements_dot_txt=None), tests=None)
    issues = report.to_issues()
    assert issues
    assert all(issue.message for issue in issues)
    assert {issue.category for issue in issues} <= set(CATEGORIES)


# --------------------------------------------------------------------------------------
# The round trip: every framework's real output, through every static gate
# --------------------------------------------------------------------------------------
_REQUIREMENT = (
    "Research a topic using web search, verify the sources, and write a short report "
    "with citations."
)


def _assembled(framework: str) -> Tuple[ProjectManifest, GeneratedProject]:
    analysis = heuristic_analysis(_REQUIREMENT)
    choice = select_framework(analysis, framework)
    manifest = plan_architecture(analysis, choice, provider="openai")
    return manifest, assemble_project(manifest)


def test_a_planned_dotfile_keeps_the_dot_in_its_name():
    """
    The bug that made every framework fail to assemble, in one assertion.

    ``FileSpec`` normalised its path with ``lstrip("./")``, and ``lstrip`` takes a *set of
    characters* rather than a prefix, so the planner's ``.env.example`` arrived at the assembler
    as ``env.example`` and ``.dockerignore`` as ``dockerignore``. The assembler's writer table is
    keyed by the real names, so it correctly refused to emit a file it had no writer for - and
    ``plan_architecture`` into ``assemble_project`` had therefore never once worked. Both layers
    now share ``relative_path``, and this is the round trip that proves the plan and the emitted
    file agree about the name.
    """
    spec = FileSpec(path=".env.example", purpose="Documents the variables.")
    assert spec.path == ".env.example"
    assert FileSpec(path=".dockerignore", purpose="Keeps the image small.").path == ".dockerignore"

    manifest = _manifest(files=[spec])
    assert manifest.file(".env.example") is spec

    for framework in sorted(EMITTERS):
        planned, project = _assembled(framework)
        for path in (".env.example", ".dockerignore"):
            if planned.file(path) is None:
                continue
            assert project.get(path) is not None, (
                f"{framework}: the plan asked for {path} and the project has no such file"
            )


def test_crewai_hierarchical_manager_is_emitted_without_tools():
    """CrewAI's manager coordinates workers and must never receive worker tools."""
    requirement = "Make a support triage crew that classifies an incoming ticket and drafts a reply"
    analysis = heuristic_analysis(requirement)
    choice = select_framework(analysis, "crewai")
    manifest = plan_architecture(analysis, choice, provider="openai")
    workflow = manifest.primary_workflow
    assert workflow is not None and workflow.kind == "hierarchical"

    project = assemble_project(manifest)
    manager = manifest.agents[0]
    manager_path = manager.module.replace(".", "/") + ".py"
    manager_file = project.get(manager_path)
    assert manager_file is not None
    assert "tools=[]" in manager_file.content


def test_a_path_that_tries_to_leave_the_project_is_confined():
    """
    The near miss for the test above: keeping dotfiles must not also stop confining paths.

    Generated paths come from model output, so ``relative_path`` is a security boundary before
    it is a tidiness helper - a file written to ``../../.ssh/authorized_keys`` would escape the
    workspace. Leading ``./`` goes, ``..`` segments and drive letters go, and a path that is
    nothing but traversal still has to come out as a usable name rather than an empty string.
    """
    assert relative_path("./main.py") == "main.py"
    assert relative_path("/etc/passwd") == "etc/passwd"
    assert relative_path("../../secret.txt") == "secret.txt"
    assert relative_path("C:/Windows/system32/x.py") == "Windows/system32/x.py"
    assert relative_path("..") == "unnamed.txt"
    # And the plan normalises through the same function, so a spec cannot smuggle one past it.
    assert FileSpec(path="../../secret.txt", purpose="No.").path == "secret.txt"


@pytest.mark.parametrize("framework", sorted(EMITTERS))
def test_every_assembled_project_passes_every_static_gate(framework: str):
    """
    The test that matters most, and the one that catches drift between plan and emitter.

    Everything a project can be judged on without running it is judged here, on the generator's
    real output. A blocking failure is a defect in this repository - in the planner, in an
    emitter, or in the shared helpers - and the fix belongs there, never in the gate.
    """
    manifest, project = _assembled(framework)
    checks = static_checks(project, manifest)

    blocking = [c for c in checks if c.blocking]
    assert blocking == [], (
        f"{framework} produced a project that fails its own gates:\n{_explain(checks)}"
    )


@pytest.mark.parametrize("framework", sorted(EMITTERS))
def test_the_planned_tool_symbol_is_the_one_the_emitter_writes(framework: str):
    """
    The convention this locks down is stated twice, and must not drift.

    ``ProjectManifest.tool_symbol`` is the plan's statement of how a tool is named; each
    emitter's ``tool_fragment`` is the code's. CrewAI, LangGraph and both ReAct variants emit a
    tool *class*; Agno takes the callable itself and emits a plain *function*. The planner used
    to record ``class_name`` for all of them, so every Agno project claimed its tool module
    provided ``SearchTool`` while the file actually defined ``search``. Nothing crashed - which
    is why it survived - but the import graph reported an unrealised dependency on every Agno
    project with a tool, and the code viewer named a symbol that did not exist.

    Asserting agreement is the only thing that keeps two statements of one rule honest.
    """
    manifest, project = _assembled(framework)
    assert manifest.tools, f"{framework} planned no tools, so this test would prove nothing"

    graph = build_graph(project, manifest)
    # ``defines``, not ``provides``: the assertion below says "no application file *defines*
    # that name", and provides would also count a file that merely re-exported it from
    # somewhere else, which would let the plan and the code disagree undetected.
    defined = {
        name
        for node in graph.nodes.values()
        if not node.is_test
        for name in node.defines
    }

    for tool in manifest.tools:
        symbol = manifest.tool_symbol(tool)
        expected = tool.function_name if framework in FUNCTION_TOOL_FRAMEWORKS else tool.class_name
        assert symbol == expected
        assert symbol in defined, (
            f"{framework}: the plan says {tool.name} is bound to {symbol!r}, but no generated "
            f"application file defines that name."
        )


@pytest.mark.parametrize("framework", sorted(EMITTERS))
def test_every_assembled_project_offers_a_run_function_the_runner_will_find(framework: str):
    manifest, project = _assembled(framework)
    checks = _by_id(static_checks(project, manifest))
    assert checks["entrypoint.runnable"].status is CheckStatus.PASSED
    assert checks["entrypoint.result_contract"].status is CheckStatus.PASSED


@pytest.mark.parametrize("framework", sorted(EMITTERS))
def test_no_assembled_project_contains_a_credential(framework: str):
    manifest, project = _assembled(framework)
    check = _by_id(static_checks(project, manifest))["configuration.secrets"]
    assert check.status is CheckStatus.PASSED, check.detail


@pytest.mark.parametrize("framework", sorted(EMITTERS))
def test_no_assembled_project_is_ready_before_its_tests_run(framework: str):
    """Section 17, applied to real output rather than to a fixture."""
    manifest, project = _assembled(framework)
    report = validate_project(project, manifest, tests=None)
    assert report.ready is False
    assert report.unmet()
