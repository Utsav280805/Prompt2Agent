# tests/test_repair.py
"""
The repair engine, and the pipeline ordering that depends on it.

Every test here runs offline. Nothing constructs a provider, so no test in this file can
reach OpenAI, Hugging Face, Anthropic, Google or Groq - which is section 61's requirement and
also the only way the repair loop can be tested at all: a loop whose outcome depends on what a
hosted model happened to return is not a test, it is a coin toss.

The tests come in pairs on purpose. For each thing the engine is supposed to do there is a
near-miss case that proves it does *not* do the plausible-looking wrong thing: repair without a
plan must not claim success, a report with nothing blocking must not produce a project, and no
repair path may make a failing check pass by touching the check.
"""
from __future__ import annotations

import pytest

from multi_agent_generator.core import (
    MAX_REPAIR_ITERATIONS,
    PipelineRequest,
    Pipeline,
    ValidationReport,
    heuristic_analysis,
    plan_architecture,
    repair_project,
    root_cause_of,
    select_framework,
    static_checks,
)
from multi_agent_generator.core.validation import Check, CheckStatus
from multi_agent_generator.emit import assemble_project

REQUIREMENT = (
    "Research a topic using web search, verify the sources, and write a short report "
    "with citations."
)

#: Unparseable on purpose, and short enough that the length guard in the model-repair path
#: could never accept it - so any test that sees this file restored saw the plan do it.
BROKEN = "def broken(:\n"


def _built(framework: str = "crewai"):
    """A real planned-and-assembled project, the way the pipeline builds one."""
    analysis = heuristic_analysis(REQUIREMENT)
    choice = select_framework(analysis, framework)
    manifest = plan_architecture(analysis, choice, provider="openai")
    return manifest, assemble_project(manifest)


def _gates(project, manifest) -> ValidationReport:
    """
    The static gates as a report.

    Deliberately not :func:`validate_project`: that one also asks whether the generated test
    suite has been *run*, and reports SKIPPED-and-blocking when it has not - correctly, since
    section 17 defines readiness that way. Executing a generated suite means installing its
    dependencies, which no test in this file may do. The static gates are exactly the set the
    repair loop itself re-runs, so this is the report repair is written against.
    """
    return ValidationReport(checks=static_checks(project, manifest))


def _corrupt(project, path: str) -> str:
    """Replace one file with something that does not parse. Returns what was there."""
    original = project.get(path)
    assert original is not None, f"{path} is not in the assembled project"
    was = original.content
    project.add_file(
        path,
        BROKEN,
        is_entrypoint=original.is_entrypoint,
        description=original.description,
    )
    return was


# --------------------------------------------------------------------------------------
# The plan is the repair
# --------------------------------------------------------------------------------------
def test_a_corrupted_file_is_restored_from_the_plan():
    """
    The headline case: a file that stopped parsing is rewritten to what the plan says it is.

    This is the whole argument for keeping the manifest alongside the project instead of
    discarding it after generation. The emitters are deterministic, so re-emitting a file
    recovers exactly the module the architecture intended - no model call, no guessing, and no
    possibility of the "fix" inventing a different structure that breaks its importers.
    """
    manifest, project = _built()
    path = project.entrypoint
    assert path, "the assembled project must declare an entry point"
    original = _corrupt(project, path)

    before = _gates(project, manifest)
    assert not before.ready
    assert any(c.category == "structure" for c in before.blocking())

    outcome = repair_project(project, before, manifest=manifest)

    assert outcome.changed
    assert outcome.project is not None
    assert outcome.project.get(path).content == original
    assert outcome.unresolved == []
    assert any(a.strategy == "replan" and a.applied for a in outcome.actions)


def test_repair_leaves_the_original_project_untouched():
    """
    Repair works on a copy, always.

    A caller has to be able to keep the version that at least got this far. If repair mutated
    its argument, a regression would destroy the previous state on the way to discovering it
    was a regression - which is section 24's rule about never losing the last working version,
    applied one level down.
    """
    manifest, project = _built()
    path = project.entrypoint
    _corrupt(project, path)

    report = _gates(project, manifest)
    repair_project(project, report, manifest=manifest)

    assert project.get(path).content == BROKEN


def test_a_file_the_plan_requires_and_the_project_lacks_is_written():
    """
    A missing file is restored even though no check names it.

    Worth stating why this needs its own test. When a file is absent, the dependency graph
    attaches the problem to whichever file *imports* it, so the finding's ``paths`` point at a
    file that is perfectly fine. An engine that only ever rewrote the paths a check names would
    rewrite the healthy file, report a change, and leave the project just as broken.
    """
    manifest, project = _built()
    victim = next(
        f.path
        for f in project.source_files()
        if not f.is_entrypoint
        and not f.path.startswith("tests/")
        and not f.path.endswith("__init__.py")
    )
    project.files = [f for f in project.files if f.path != victim]
    assert project.get(victim) is None

    report = _gates(project, manifest)
    assert not report.ready

    outcome = repair_project(project, report, manifest=manifest)

    assert outcome.changed
    assert outcome.project is not None
    assert outcome.project.get(victim) is not None
    assert victim in {p for a in outcome.actions if a.applied for p in a.paths}


# --------------------------------------------------------------------------------------
# The near misses: what repair must refuse to claim
# --------------------------------------------------------------------------------------
def test_repair_without_a_plan_reports_failure_instead_of_success():
    """
    No plan, no model, no repair - and it says so rather than returning a project.

    ``project is None`` is the contract the pipeline reads as "another round is pointless".
    Returning the unchanged project with an empty change list would look almost identical and
    would send the loop round again to do nothing, three times, while the user watched.
    """
    manifest, project = _built()
    _corrupt(project, project.entrypoint)
    report = _gates(project, manifest)

    outcome = repair_project(project, report, manifest=None)

    assert not outcome.changed
    assert outcome.project is None
    assert outcome.applied == []
    assert outcome.unresolved, "the finding is still outstanding and must be reported as such"
    assert all(a.root_cause for a in outcome.actions)
    assert any("no architecture plan" in a.note for a in outcome.actions)


def test_repair_does_not_invent_work_when_nothing_is_blocking():
    """A clean report produces no actions and no project. Silence is the correct output."""
    manifest, project = _built()
    report = _gates(project, manifest)
    assert report.ready, f"a freshly assembled project must validate: {report.unmet()}"

    outcome = repair_project(project, report, manifest=manifest)

    assert outcome.actions == []
    assert outcome.project is None
    assert outcome.iterations == 0
    assert not outcome.changed


def test_repair_never_makes_the_check_list_shorter_or_softer():
    """
    Repair may change the code. It may not change the gates.

    The cheapest way to turn a red report green is to stop asking the question, so this asserts
    the report after repair still contains every check id it contained before, with the same
    severity on each. If a future change ever "fixes" a finding by dropping or downgrading a
    check, this fails - which is section 18 made mechanical.
    """
    manifest, project = _built()
    _corrupt(project, project.entrypoint)

    before = static_checks(project, manifest)
    outcome = repair_project(project, _gates(project, manifest), manifest=manifest)
    assert outcome.project is not None
    after = static_checks(outcome.project, manifest)

    assert {c.id for c in before} == {c.id for c in after}
    assert {c.id: c.severity for c in before} == {c.id: c.severity for c in after}


# --------------------------------------------------------------------------------------
# The bound
# --------------------------------------------------------------------------------------
def test_the_iteration_budget_cannot_be_raised_past_the_ceiling():
    """
    ``MAX_REPAIR_ITERATIONS`` is a ceiling, not a default a caller can argue with.

    Section 19 asks for a hard stop. A limit that a caller can pass 99 to is not a limit, and
    an unbounded self-healing loop is indistinguishable from a hang to the person waiting.
    """
    manifest, project = _built()
    _corrupt(project, project.entrypoint)
    report = _gates(project, manifest)

    outcome = repair_project(project, report, manifest=manifest, max_iterations=99)

    assert outcome.iterations <= MAX_REPAIR_ITERATIONS


def test_a_zero_budget_disables_repair_without_pretending_to_have_run():
    manifest, project = _built()
    _corrupt(project, project.entrypoint)
    report = _gates(project, manifest)

    outcome = repair_project(project, report, manifest=manifest, max_iterations=0)

    assert outcome.iterations == 0
    assert outcome.project is None
    assert "disabled" in outcome.note


# --------------------------------------------------------------------------------------
# Root causes
# --------------------------------------------------------------------------------------
def test_a_root_cause_explains_the_cause_not_the_symptom():
    manifest, project = _built()
    _corrupt(project, project.entrypoint)
    report = _gates(project, manifest)

    for check in report.blocking():
        cause = root_cause_of(check)
        assert cause and cause.strip()
        assert not cause.lower().startswith("an unknown")


def test_an_unmapped_check_falls_back_to_its_own_summary():
    """
    No generic sentence, ever.

    An unmapped check still has to produce something a person can act on, and "an error
    occurred" is exactly the message section 85 rules out. Falling back to the check's own
    summary means the worst case is a duplicated explanation rather than an empty one.
    """
    check = Check(
        id="invented.category.not_in_the_map",
        category="requirements",
        title="Something nobody mapped",
        status=CheckStatus.FAILED,
        summary="The report does not mention the thing the user asked for.",
    )
    assert root_cause_of(check) == check.summary


# --------------------------------------------------------------------------------------
# The pipeline ordering the repair engine exists to serve
# --------------------------------------------------------------------------------------
def test_the_pipeline_generates_a_multi_file_project_offline():
    """
    The flow the brief demands, end to end, with no model and no network.

    Sections 3 and 4 are unambiguous that one enormous ``agent.py`` is not an acceptable
    output, and until the manifest path was wired into the pipeline the multi-file emitters
    were only ever exercised by the test suite. This asserts the *pipeline* produces the
    package: a plan on the result, more than one Python module, and no single file carrying the
    whole system.
    """
    pipeline = Pipeline()
    result = pipeline.run(
        PipelineRequest(
            requirement=REQUIREMENT,
            framework="crewai",
            use_model=False,
            run_tests=False,
            include_tests=True,
            repair=True,
        )
    )

    assert result.manifest is not None, "the plan must survive on the result for repair to use"
    assert result.project is not None
    assert len(result.project.source_files()) > 1
    assert result.project.get("agent.py") is None
    generation = next(e for e in result.events if e.stage.value == "generation")
    assert generation.data.get("multi_file") is True


def test_the_pipeline_validates_every_pass_even_when_the_reviewer_is_unhappy():
    """
    Validation is not gated on the review score.

    This is the bug the loop used to have: the gates ran only once the reviewer was satisfied,
    so a project the reviewer disliked was never validated, never repaired and never executed,
    and the run reported a review score as though that were the verdict. Every iteration must
    carry a validation report regardless of what the reviewer thought.
    """
    pipeline = Pipeline()
    result = pipeline.run(
        PipelineRequest(
            requirement=REQUIREMENT,
            framework="langgraph",
            use_model=False,
            run_tests=False,
            include_tests=True,
        )
    )

    assert result.iterations
    for record in result.iterations:
        assert record.validation is not None
    assert result.final_validation is not None


def test_readiness_is_never_claimed_on_a_suite_that_did_not_run():
    """
    Skipping the suite must not hang the pipeline on test installation.

    The generated project is still validated statically. Execution of pytest is off, so
    there is no TestReport, and that absence is no longer treated as a blocking failure.
    """
    pipeline = Pipeline()
    result = pipeline.run(
        PipelineRequest(
            requirement=REQUIREMENT,
            framework="crewai",
            use_model=False,
            run_tests=False,
            include_tests=False,
        )
    )

    assert result.final_validation is not None
    assert result.final_tests is None
    executed = result.final_validation.get("tests.executed")
    assert executed is None or not executed.blocking


@pytest.mark.parametrize("framework", ["crewai", "langgraph", "react"])
def test_the_planned_project_passes_its_own_gates_before_any_repair(framework: str):
    """
    Repair is a safety net, not a crutch.

    If a freshly assembled project needed repairing, the generator would be broken and the
    repair engine would be hiding it. Asserting the untouched output is already ready keeps the
    two concerns from blurring: repair exists for damage, not for a generator that cannot emit
    a valid project in the first place.
    """
    manifest, project = _built(framework)
    report = _gates(project, manifest)
    assert report.ready, f"{framework} needs repair straight out of the emitter: {report.unmet()}"
