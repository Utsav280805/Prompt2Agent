# multi_agent_generator/core/pipeline.py
"""
The pipeline: requirement in, verified project out.

    analyse -> select framework -> plan -> generate -> review -> test -> validate
            -> repair -> run -> ready

This is the module that makes the product a system rather than a prompt wrapper. Everything
it does is recorded: each stage emits a :class:`PipelineEvent` with a stage, a status, a
duration and an error when there was one, and each pass of the loop is kept as an
:class:`IterationRecord` holding that pass's review, test report, gate report and repair log.
The frontend renders those events directly, so what the user watches is the actual execution
trace and not a progress animation.

Four design decisions carry most of the weight:

**The plan comes before the code.** :func:`~multi_agent_generator.core.architecture.plan_architecture`
decides what files exist, and the emitters write exactly those files. The alternative - asking
one model call for a whole project - is what produced a single thousand-line ``agent.py`` no
matter how many agents the requirement described, and it is what sections 3 to 6 exist to
rule out. The plan is kept on the result afterwards, because the repair engine regenerates a
broken file *from* it.

**Success is defined by evidence, not completion.** The final status comes from
:meth:`PipelineResult.unmet_criteria`, which now defers to the gate report: a project is
``READY`` when no check blocks, and the checks are computed from the code rather than from an
opinion about it. A run that generated beautiful code nobody verified ends as ``FAILED`` with
the reason attached. This is the point in the system where "do not pretend a generated agent
worked when it failed" is actually enforced, so the decision lives in one place and no stage
may bypass it.

**Every stage is optional except planning and generation.** No credential, no model, no
execution sandbox: the pipeline still runs, using the heuristic analyser, the deterministic
framework scorer, the static reviewer and the deterministic repair strategies, and it labels
every degraded result. That is what makes the whole product usable and testable offline.

**The loop stops when it stops making progress.** Iterating is bounded by
``MAX_GENERATION_ITERATIONS``, repair within a pass by ``MAX_REPAIR_ITERATIONS``, and the loop
also breaks the moment a pass reports that it changed nothing - three identical rounds is not
persistence, it is waste.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..errors import AppError
from ..llm import LLMConfig, build_provider
from ..llm.base import LLMProvider
from ..settings import Settings, get_settings
from .analysis import analyze_requirement, heuristic_analysis
from .architecture import plan_architecture
from .generation import generate_project
from .improvement import improve_project
from .manifest import ProjectManifest
from .models import (
    FrameworkChoice,
    GeneratedProject,
    IterationRecord,
    PipelineEvent,
    PipelineResult,
    ProjectStatus,
    RequirementAnalysis,
    ReviewResult,
    RunResult,
    Stage,
    TestReport,
    utcnow,
)
from .repair import RepairOutcome, repair_project
from .requirements_spec import requirement_checks
from .review import review_project
from .selection import select_framework
from .validation import ValidationReport, validate_project

__all__ = ["Pipeline", "PipelineRequest", "run_pipeline"]


#: Called with each event as it happens. Used for live streaming to the UI and for logging.
EventHook = Callable[[PipelineEvent], None]

#: Runs a generated project's own test suite. Injected so the pipeline does not depend on
#: the execution sandbox directly - which is what lets it be unit-tested without one.
TestRunnerFn = Callable[[GeneratedProject], TestReport]

#: Runs a generated project against a real query, for the optional smoke run. Injected for the
#: same reason as the test runner.
AgentRunnerFn = Callable[[GeneratedProject, str], RunResult]

#: The query the optional smoke run sends. Deliberately trivial and domain-neutral: the smoke
#: run is checking that the agent starts, reaches its provider and returns a value, not that it
#: is good at anything. A domain-specific probe would fail for reasons that say nothing about
#: whether the project works.
SMOKE_QUERY = "Reply with a one-sentence confirmation that you are running."

#: Gate categories whose failure means running the test suite would tell us nothing new.
#:
#: A project whose imports do not resolve, or whose files do not parse, cannot import its own
#: test modules either. Installing its dependencies and collecting its suite would spend
#: minutes to rediscover the finding we already have, so the loop repairs first and tests
#: after. This is a scheduling decision, never a weakening of anything: the suite still has to
#: run and pass before the project can be reported ready.
_BLOCKS_TESTING = ("imports", "structure")


@dataclass
class PipelineRequest:
    """
    Everything one run needs.

    ``provider``/``model`` configure the LLM that *designs* the system;
    ``runtime_provider``/``runtime_model`` configure the LLM the *generated code* will call.
    They are separate because they genuinely are: designing an agent team with a large
    hosted model while the generated project runs against a local one is a normal thing to
    want, and conflating the two made that impossible.
    """

    requirement: str
    #: Framework name, or None/"auto" to let the selector decide.
    framework: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    runtime_provider: Optional[str] = None
    runtime_model: Optional[str] = None
    #: Bundle an offline test suite with the generated project.
    include_tests: bool = True
    #: Execute that suite. Off means the run cannot reach READY, and says so.
    run_tests: bool = True
    #: Consult the LLM during analysis/selection/review. Off forces the offline path.
    use_model: bool = True
    #: Let the repair engine rewrite the project when a gate fails.
    repair: bool = True
    #: Actually run the generated agent once, against a real provider, before reporting ready.
    #:
    #: Off by default, and that default is deliberate. A smoke run installs the project's
    #: dependencies and calls a hosted model, so switching it on by default would make every
    #: automated test in this repository reach the network - which section 61 forbids. The
    #: gate report says ``skipped`` rather than ``passed`` when it has not happened, so turning
    #: it off degrades the evidence honestly instead of quietly.
    run_agent_smoke: bool = False
    #: The credential the *generated project* needs for its smoke run. Passed explicitly
    #: because the sandbox environment is scrubbed: a key not passed here is simply absent.
    runtime_api_key: Optional[str] = None
    #: Credential values the platform holds, so the leak gate can prove none reached the
    #: source. Never stored on the result or in any event.
    known_secrets: Sequence[str] = ()
    max_iterations: Optional[int] = None
    project_id: Optional[str] = None
    run_id: Optional[str] = None
    #: A fully-resolved LLM config, when the caller already has one (e.g. from the
    #: settings page). Takes precedence over ``provider``/``model``.
    llm_config: Optional[LLMConfig] = None
    #: Saved LLM settings to layer under the environment.
    saved_llm: Optional[Mapping[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """
    Runs the stages and records what happened.

    One instance per run. It holds the mutable :class:`PipelineResult` being assembled so
    that a failure at any stage still returns a result with everything gathered up to that
    point - a half-finished run with its events intact is far more useful than an exception
    that discards the trace.
    """

    def __init__(
        self,
        *,
        on_event: Optional[EventHook] = None,
        settings: Optional[Settings] = None,
        provider: Optional[LLMProvider] = None,
        test_runner: Optional[TestRunnerFn] = None,
        agent_runner: Optional[AgentRunnerFn] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._on_event = on_event
        self._provider = provider
        self._test_runner = test_runner
        self._agent_runner = agent_runner
        #: The model-authored agent configuration from the planning stage, when there was one.
        self._agent_config: Optional[Dict[str, Any]] = None
        self._result: Optional[PipelineResult] = None
        self._request: Optional[PipelineRequest] = None

    # ------------------------------------------------------------------------ public API
    def run(self, request: PipelineRequest) -> PipelineResult:
        """
        Execute the full pipeline.

        Does not raise for expected failures - a missing credential, a model that will not
        answer, generated code that will not compile. Those become a ``FAILED`` result with
        a structured ``error`` and a full event trace, because both the CLI and the API need
        to show the user what happened rather than a stack trace.
        """
        self._request = request
        result = PipelineResult(
            requirement=(request.requirement or "").strip(),
            status=ProjectStatus.ANALYZING,
        )
        self._result = result

        try:
            provider = self._resolve_provider(request)

            result.analysis = self._stage_analysis(request, provider)
            result.framework_choice = self._stage_selection(request, result.analysis, provider)

            result.status = ProjectStatus.PLANNING
            result.manifest = self._stage_planning(
                request, result.analysis, result.framework_choice, provider
            )

            result.status = ProjectStatus.GENERATING
            result.project = self._stage_generation(
                request, result.analysis, result.framework_choice, result.manifest
            )

            self._loop(request, result, provider)
            self._finalise(result)
        except AppError as exc:
            self._fail(result, exc)
        except Exception as exc:  # noqa: BLE001
            # An unexpected exception is a bug in this application, not in the user's
            # request. It is wrapped so the caller always gets the same shape, and the
            # original type is preserved in the detail so it is still diagnosable.
            self._fail(
                result,
                AppError(
                    "The pipeline stopped because of an internal error.",
                    action="Please report this with the detail below.",
                    detail=f"{type(exc).__name__}: {exc}",
                ),
            )

        result.completed_at = utcnow()
        return result

    # --------------------------------------------------------------------------- stages
    def _stage_analysis(
        self,
        request: PipelineRequest,
        provider: Optional[LLMProvider],
    ) -> RequirementAnalysis:
        with self._stage(Stage.ANALYSIS, "Reading the requirement") as stage:
            try:
                analysis = analyze_requirement(request.requirement, provider)
            except AppError as exc:
                if provider is None:
                    raise  # an empty requirement; there is nothing to fall back to
                # The model failed but the heuristic will not. Analysis is not worth
                # failing the run over, and the same credential problem will surface at
                # generation time where it actually blocks progress.
                self._warn(
                    Stage.ANALYSIS,
                    f"Model analysis failed ({exc.message}); using the offline analyser.",
                    error=exc.to_dict(include_detail=False),
                )
                analysis = heuristic_analysis(request.requirement)

            stage.data.update(
                {
                    "complexity": analysis.complexity,
                    "roles": analysis.suggested_roles,
                    "tools": analysis.suggested_tools,
                    "from_model": analysis.from_model,
                }
            )
            stage.message = (
                f"{analysis.complexity.title()} request; "
                f"{len(analysis.suggested_roles)} role(s), "
                f"{len(analysis.suggested_tools)} tool(s)"
            )
            return analysis

    def _stage_selection(
        self,
        request: PipelineRequest,
        analysis: RequirementAnalysis,
        provider: Optional[LLMProvider],
    ) -> FrameworkChoice:
        with self._stage(Stage.FRAMEWORK_SELECTION, "Choosing a framework") as stage:
            choice = select_framework(analysis, request.framework, provider)
            stage.data.update(
                {
                    "framework": choice.framework,
                    "confidence": choice.confidence,
                    "user_specified": choice.user_specified,
                    "alternatives": [a.get("framework") for a in choice.alternatives],
                }
            )
            stage.message = (
                f"{choice.framework} "
                f"({'your choice' if choice.user_specified else f'{choice.confidence:.0%} confidence'})"
            )
            return choice

    def _stage_planning(
        self,
        request: PipelineRequest,
        analysis: RequirementAnalysis,
        choice: FrameworkChoice,
        provider: Optional[LLMProvider],
    ) -> ProjectManifest:
        """
        Decide the architecture: agents, tools, workflow, files, dependencies.

        The agent *configuration* is the one part a model shapes, and it is fetched here rather
        than inside :func:`plan_architecture` so that a model failure degrades to planning from
        the analysis alone instead of failing the run. A plan built without a model is smaller
        and blunter, not broken - which is what makes the whole path testable with no key.
        """
        runtime_provider = (
            request.runtime_provider or request.provider or self.settings.default_provider
        )
        with self._stage(Stage.PLANNING, "Planning the architecture") as stage:
            config: Optional[Dict[str, Any]] = None
            if provider is not None:
                try:
                    from .generation import build_agent_config

                    config = build_agent_config(
                        analysis,
                        choice.framework,
                        runtime_provider,
                        request.runtime_model or request.model,
                    )
                except AppError as exc:
                    self._warn(
                        Stage.PLANNING,
                        f"The model could not shape the agent configuration ({exc.message}); "
                        "planning from the requirement analysis instead.",
                        error=exc.to_dict(include_detail=False),
                    )
                except Exception as exc:  # noqa: BLE001 - generators raise plain exceptions
                    self._warn(
                        Stage.PLANNING,
                        "The model returned an agent configuration that could not be used; "
                        "planning from the requirement analysis instead.",
                        error={"detail": f"{type(exc).__name__}: {exc}"},
                    )

            manifest = plan_architecture(
                analysis,
                choice,
                runtime_provider,
                request.runtime_model or request.model,
                config=config,
                include_tests=request.include_tests,
            )
            # Kept for the fallback path in _stage_generation and for the improvement stage,
            # both of which re-render from the configuration rather than re-prompting for it.
            self._agent_config = config
            summary = manifest.architecture_summary()
            stage.data.update(
                {
                    "tier": manifest.tier,
                    "agents": len(manifest.agents),
                    "tools": len(manifest.tools),
                    "planned_files": len(manifest.files),
                    "from_model": config is not None,
                    "architecture": summary,
                }
            )
            stage.message = (
                f"{manifest.tier} layout: {len(manifest.agents)} agent(s), "
                f"{len(manifest.tools)} tool(s), {len(manifest.files)} planned file(s)"
            )
            return manifest

    def _stage_generation(
        self,
        request: PipelineRequest,
        analysis: RequirementAnalysis,
        choice: FrameworkChoice,
        manifest: ProjectManifest,
    ) -> GeneratedProject:
        """
        Write every file the plan asked for, and nothing it did not.

        Assembly failure is the one case with a fallback. It means the planner asked for a file
        the emitters do not know how to write, which is a bug in this application rather than
        anything the user did - and handing them nothing at all would be the worst of the
        available outcomes. The single-file generator still works, so the run degrades to it
        with a warning event and a note recorded on the project, so the layout is never
        silently worse than the architecture the user was shown.
        """
        with self._stage(Stage.GENERATION, "Writing the project files") as stage:
            # Imported inside the stage rather than at module scope: the emit package imports
            # ``core.manifest`` and ``core.models``, so a top-level import here would have the
            # two packages referencing each other while ``core`` is still initialising.
            from ..emit import AssemblyError, assemble_project

            try:
                project = assemble_project(manifest)
                stage.data["multi_file"] = True
            except (AssemblyError, AppError) as exc:
                detail = exc.message if isinstance(exc, AppError) else str(exc)
                self._warn(
                    Stage.GENERATION,
                    "The planned file layout could not be written, so a single-file project "
                    "was generated instead. This is a defect in the generator, not in your "
                    "request.",
                    error={"detail": detail},
                )
                project = generate_project(
                    analysis,
                    choice,
                    manifest.provider,
                    manifest.model,
                    config=self._agent_config,
                    include_tests=request.include_tests,
                )
                project.notes.append(
                    "The planned multi-file layout could not be assembled, so this project "
                    "was written as a single module. Its architecture view describes the "
                    "layout that was intended."
                )
                stage.data["multi_file"] = False
                stage.status = "warning"

            stage.data.update(
                {
                    "files": len(project.files),
                    "lines": project.total_lines,
                    "entrypoint": project.entrypoint,
                    "runtime_provider": project.provider,
                    "runtime_model": project.model,
                }
            )
            stage.message = f"{len(project.files)} files, {project.total_lines} lines"
            return project

    def _stage_review(
        self,
        project: GeneratedProject,
        analysis: RequirementAnalysis,
        provider: Optional[LLMProvider],
        iteration: int,
    ) -> ReviewResult:
        with self._stage(Stage.REVIEW, "Reviewing the code", iteration=iteration) as stage:
            review = review_project(
                project, analysis, provider, pass_score=self.settings.review_pass_score
            )
            stage.data.update(
                {
                    "score": round(review.score, 2),
                    "passed": review.passed,
                    "blockers": len(review.blockers),
                    "issues": len(review.issues),
                    "from_model": review.from_model,
                }
            )
            stage.status = "succeeded" if review.passed else "warning"
            stage.message = (
                f"Score {review.score:.1f}/10 - "
                f"{'passed' if review.passed else 'needs work'}"
                + (f", {len(review.blockers)} blocker(s)" if review.blockers else "")
            )
            return review

    def _stage_tests(
        self,
        project: GeneratedProject,
        iteration: int,
    ) -> TestReport:
        """
        Run the generated suite.

        A missing runner produces a report with ``unavailable_reason`` set, which makes
        ``ran`` False and keeps the run out of READY. Reporting "no tests were run" as a
        pass would be the exact fake-green outcome the brief prohibits, so the absence of a
        sandbox degrades the *verdict*, never the honesty of it.
        """
        with self._stage(Stage.TEST, "Running the generated tests", iteration=iteration) as stage:
            runner = self._resolve_test_runner()
            if runner is None:
                report = TestReport(
                    status="skipped",
                    unavailable_reason=(
                        "No test runner is available, so the generated suite was not executed."
                    ),
                )
            elif not project.test_files():
                report = TestReport(
                    status="skipped",
                    unavailable_reason="The project has no tests to run.",
                )
            else:
                # Override pytest timeout for pipeline test runs - use a shorter timeout
                # to fail fast on hanging tests. The generated project's pytest.ini may
                # have a longer timeout, but during automated validation we want quick feedback.
                report = runner(project, timeout=60)

            stage.data.update(
                {
                    "status": report.status,
                    "ran": report.ran,
                    "passed": report.passed_count,
                    "failed": report.failed_count,
                    "skipped": report.skipped_count,
                    "timed_out": report.timed_out,
                }
            )
            if report.ok:
                stage.message = f"{report.passed_count} passed"
            elif report.timed_out:
                stage.status = "failed"
                stage.message = "The test suite timed out"
            elif not report.ran:
                stage.status = "warning"
                stage.message = report.unavailable_reason or "The suite did not run"
            else:
                stage.status = "failed"
                stage.message = (
                    f"{report.failed_count} failed, {report.passed_count} passed"
                )
            return report

    def _stage_validation(
        self,
        request: PipelineRequest,
        result: PipelineResult,
        tests: Optional[TestReport],
        runtime: Optional[RunResult],
        iteration: int,
    ) -> ValidationReport:
        """
        Run every gate and record the verdict.

        This is the stage that decides whether the project is ready, and it is the reason the
        loop no longer asks the reviewer that question. A review score is an opinion about the
        code; these checks are computed from the code - does it parse, do its imports resolve,
        does the entry point expose a function the runner calls, does each planned agent have a
        factory, is there a credential sitting in a literal. Section 20 is blunt about why the
        distinction matters: "code runs" does not mean the agent satisfies the request.
        """
        assert result.project is not None
        with self._stage(
            Stage.VALIDATION, "Checking the project against every gate", iteration=iteration
        ) as stage:
            report = validate_project(
                result.project,
                result.manifest,
                tests=tests,
                runtime=runtime,
                # The only gates that need to look back at what the user actually typed. They
                # arrive as extra_checks rather than from inside the validator because the
                # validator judges the project as a program and should not have to know that an
                # analysis stage exists; section 20's point is that those are two questions.
                extra_checks=requirement_checks(
                    result.analysis, result.manifest, result.project
                ),
                secrets=request.known_secrets,
                expect_tests=request.include_tests,
                require_test_run=request.run_tests,
            )
            counts = report.summary()
            stage.data.update(
                {
                    "ready": report.ready,
                    "checks": counts["checks"],
                    "passed": counts["passed"],
                    "failed": counts["failed"],
                    "warnings": counts["warnings"],
                    "skipped": counts["skipped"],
                    "blocking": counts["blocking"],
                    "categories": report.by_category(),
                    "unmet": report.unmet(),
                }
            )
            if report.ready:
                stage.message = f"{counts['passed']} of {counts['checks']} checks passed"
            else:
                stage.status = "warning"
                stage.message = (
                    f"{counts['blocking']} blocking problem(s): "
                    + "; ".join(report.unmet()[:3])
                )
            return report

    def _preflight_gates(
        self, request: PipelineRequest, result: PipelineResult
    ) -> ValidationReport:
        """
        Cheap static gates used only to decide whether running tests is worth it.

        Same checks as the real validation stage, minus the event stream. Callers must still
        run :meth:`_stage_validation` afterwards with the test and runtime reports; this
        result is never the published verdict.
        """
        assert result.project is not None
        return validate_project(
            result.project,
            result.manifest,
            tests=None,
            runtime=None,
            extra_checks=requirement_checks(
                result.analysis, result.manifest, result.project
            ),
            secrets=request.known_secrets,
            expect_tests=request.include_tests,
            require_test_run=request.run_tests,
        )

    def _stage_repair(
        self,
        request: PipelineRequest,
        result: PipelineResult,
        report: ValidationReport,
        provider: Optional[LLMProvider],
        iteration: int,
    ) -> RepairOutcome:
        """
        Rewrite the project so the failing gates stop failing.

        The outcome is applied to ``result.project`` only when the engine actually changed
        something, and the engine has already re-run the static gates to prove the change
        helped. Nothing here touches the checks themselves - see
        :mod:`multi_agent_generator.core.repair` for why that boundary is absolute.
        """
        assert result.project is not None
        with self._stage(
            Stage.REPAIR, "Repairing what validation found", iteration=iteration
        ) as stage:
            outcome = repair_project(
                result.project,
                report,
                manifest=result.manifest,
                provider=provider,
                secrets=request.known_secrets,
                expect_tests=request.include_tests,
                max_iterations=self.settings.max_repair_iterations,
            )
            if outcome.changed and outcome.project is not None:
                result.project = outcome.project
                stage.message = (
                    f"{len(outcome.applied)} repair(s) applied over "
                    f"{outcome.iterations} round(s)"
                )
            else:
                stage.status = "warning"
                stage.message = outcome.note or "Nothing could be repaired automatically"
            stage.data.update(
                {
                    "applied": len(outcome.applied),
                    "attempted": len(outcome.actions),
                    "iterations": outcome.iterations,
                    "unresolved": list(outcome.unresolved),
                    "actions": [a.as_dict() for a in outcome.actions],
                    "note": outcome.note,
                }
            )
            return outcome

    def _stage_runtime(
        self,
        request: PipelineRequest,
        project: GeneratedProject,
        iteration: int,
    ) -> Optional[RunResult]:
        """
        Run the finished agent once, for real.

        Returns None when no run was attempted, which is different from a run that failed and
        is recorded as such: the gate report shows ``skipped`` for the former and a visible
        failure for the latter. A live run never blocks readiness - a rejected API key is the
        user's provider account saying no, not a defect in the generated project - but it is
        the only evidence that the agent actually works, so it is offered whenever the caller
        has supplied what it needs.
        """
        if not request.run_agent_smoke:
            return None

        runner = self._resolve_agent_runner()
        if runner is None:
            self._warn(
                Stage.EXECUTION,
                "No execution sandbox is available, so the agent was not run.",
                iteration=iteration,
            )
            return None

        with self._stage(
            Stage.EXECUTION, "Running the agent for real", iteration=iteration
        ) as stage:
            result = runner(project, SMOKE_QUERY)
            stage.data.update(
                {
                    "status": result.status.value,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "duration_s": round(result.duration_s, 3),
                    "steps": len(result.steps),
                    "tool_calls": len(result.tool_calls),
                }
            )
            if result.ok:
                stage.message = f"The agent answered in {result.duration_s:.1f}s"
            else:
                stage.status = "warning"
                stage.message = (
                    "The agent did not complete its run; see the technical details."
                )
            return result

    def _stage_improvement(
        self,
        request: PipelineRequest,
        result: PipelineResult,
        review: ReviewResult,
        tests: Optional[TestReport],
        provider: Optional[LLMProvider],
        iteration: int,
    ) -> List[str]:
        """
        Apply the review's required changes. Returns what was actually changed.

        An empty list means no progress was possible, and the caller stops looping.
        """
        assert result.project is not None and result.analysis is not None
        assert result.framework_choice is not None

        # Test failures are review findings too: without folding them in, the improver is
        # asked to fix code while being told nothing about the failures it must address.
        review = _with_test_failures(review, tests)

        with self._stage(
            Stage.IMPROVEMENT, "Applying the required changes", iteration=iteration
        ) as stage:
            outcome = improve_project(
                result.project,
                result.analysis,
                result.framework_choice,
                review,
                provider,
                runtime_provider=request.runtime_provider or result.project.provider,
                model=request.runtime_model or request.model,
                include_tests=request.include_tests,
                # Without the plan, the re-render falls back to the single-file generator and
                # the first improvement pass would quietly flatten a package into one module.
                manifest=result.manifest,
            )
            if outcome.changed and outcome.project is not None:
                result.project = outcome.project
                stage.message = f"{len(outcome.changes_applied)} change(s) applied"
                stage.data["changes"] = outcome.changes_applied
            else:
                stage.status = "warning"
                stage.message = outcome.note or "Nothing could be improved automatically"
            if outcome.note:
                stage.data["note"] = outcome.note
            return list(outcome.changes_applied)

    # ----------------------------------------------------------------------------- loop
    def _loop(
        self,
        request: PipelineRequest,
        result: PipelineResult,
        provider: Optional[LLMProvider],
    ) -> None:
        """
        Review, test, validate, repair - until every gate passes or the budget runs out.

        The order matters, and it is not the order this loop originally used. It used to run
        the test suite only once the reviewer was happy, and to leave the loop entirely if the
        suite could not run - which meant a project the reviewer disliked was never validated,
        never repaired and never executed, and the run ended reporting a review score as though
        that were the whole story. Validation now runs on every pass regardless of what the
        reviewer thought, because the gates are cheap, they are computed from the code, and
        they are what readiness is defined by.

        Two things are still deliberately ordered rather than unconditional. The test suite is
        skipped while a gate in :data:`_BLOCKS_TESTING` is failing, because a project whose
        imports do not resolve cannot import its own tests either and installing its
        dependencies to rediscover that costs minutes. And the smoke run is attempted only once
        the static gates are clear, because running an agent that is known not to import tells
        us nothing. Neither skip is ever reported as a pass.
        """
        assert result.project is not None and result.analysis is not None
        budget = max(1, request.max_iterations or self.settings.max_generation_iterations)

        for index in range(budget):
            record = IterationRecord(index=index)
            result.iterations.append(record)
            last_pass = index == budget - 1

            result.status = ProjectStatus.REVIEWING
            record.review = self._stage_review(
                result.project, result.analysis, provider, index
            )

            # --- static gates first: they decide whether testing is worth the minutes -----
            # Do not emit this as Stage.VALIDATION. The UI treats that stage as the verdict,
            # and a preflight with tests=None would otherwise flash "the suite has not been
            # run yet" as a blocking failure while the suite is still about to start.
            result.status = ProjectStatus.VALIDATING
            static = self._preflight_gates(request, result)

            if request.run_tests and not _testing_is_pointless(static):
                result.status = ProjectStatus.TESTING
                record.tests = self._stage_tests(result.project, index)

            if static.ready and (
                not request.run_tests
                or (record.tests is not None and record.tests.ok)
            ):
                result.status = ProjectStatus.RUNNING
                record.runtime = self._stage_runtime(request, result.project, index)

            # --- the verdict for this pass, with everything gathered ----------------------
            result.status = ProjectStatus.VALIDATING
            record.validation = self._stage_validation(
                request, result, record.tests, record.runtime, index
            )

            if record.validation.ready:
                record.completed_at = utcnow()
                break

            if last_pass:
                # Out of budget. Repairing now would produce a project that never gets
                # re-validated, and shipping unverified changes is worse than stopping.
                record.completed_at = utcnow()
                self._warn(
                    Stage.VALIDATION,
                    f"Stopped after {budget} iteration(s) with "
                    f"{len(record.validation.blocking())} problem(s) unresolved. "
                    "Raise MAX_GENERATION_ITERATIONS to allow more attempts.",
                    iteration=index,
                )
                break

            # --- try to fix it: repair first, improvement second --------------------------
            changes: List[str] = []

            if request.repair:
                result.status = ProjectStatus.REPAIRING
                record.repair = self._stage_repair(
                    request, result, record.validation, provider, index
                )
                changes.extend(record.repair.changes_applied())

            if not changes:
                # Repair handles broken code; improvement handles a project that works but does
                # not do what was asked. Only reached when repair found nothing to do, because
                # asking a model to revise a configuration whose code does not even parse
                # wastes a call on a problem the deterministic path had already identified.
                result.status = ProjectStatus.IMPROVING
                changes.extend(
                    self._stage_improvement(
                        request, result, record.review, record.tests, provider, index
                    )
                )

            record.changes_applied = changes
            record.completed_at = utcnow()

            if not changes:
                break

    def _finalise(self, result: PipelineResult) -> None:
        """
        Decide whether this run succeeded, and say why if it did not.

        The verdict is delegated to :meth:`PipelineResult.unmet_criteria` so that the CLI,
        the API and the UI cannot disagree about what "ready" means. That method defers in turn
        to the gate report, so this stage reads the same checks the Testing tab shows rather
        than recomputing a second opinion here.
        """
        with self._stage(Stage.VALIDATION, "Checking acceptance criteria") as stage:
            unmet = result.unmet_criteria()
            result.status = ProjectStatus.READY if not unmet else ProjectStatus.FAILED
            stage.data["unmet_criteria"] = unmet
            report = result.final_validation
            if report is not None:
                stage.data["validation"] = report.summary()
            if unmet:
                stage.status = "warning"
                stage.message = "; ".join(unmet)
            elif report is not None:
                counts = report.summary()
                tests = result.final_tests
                stage.message = (
                    f"Ready - {counts['passed']} of {counts['checks']} checks passed"
                    + (f", {tests.passed_count} test(s) passing" if tests else "")
                )
            else:
                stage.message = "Ready"

    def _fail(self, result: PipelineResult, exc: AppError) -> None:
        result.status = ProjectStatus.FAILED
        result.error = exc.to_dict()
        self._emit(
            PipelineEvent(
                stage=self._current_stage,
                status="failed",
                message=exc.message,
                error=exc.to_dict(),
                project_id=self._request.project_id if self._request else None,
                run_id=self._request.run_id if self._request else None,
            )
        )

    # ------------------------------------------------------------------------ providers
    def _resolve_provider(self, request: PipelineRequest) -> Optional[LLMProvider]:
        """
        Build the LLM that designs the system, or return None to run offline.

        A credential problem is *not* fatal here. It returns None with a warning, so the
        pipeline continues on the heuristic path and the user gets a real project plus a
        clear message about what configuring a key would improve - rather than an error
        page and nothing else.
        """
        if not request.use_model:
            self._warn(
                Stage.ANALYSIS,
                "Running without a model: analysis, selection and review will use the "
                "offline path.",
            )
            return None
        if self._provider is not None:
            return self._provider

        try:
            provider = build_provider(
                request.provider,
                request.model,
                config=request.llm_config,
                saved=request.saved_llm,
            )
            provider.validate()
        except AppError as exc:
            self._warn(
                Stage.ANALYSIS,
                f"{exc.message} Continuing without a model.",
                error=exc.to_dict(include_detail=False),
            )
            return None

        self._provider = provider
        return provider

    def _resolve_test_runner(self) -> Optional[TestRunnerFn]:
        """
        Find something that can run the generated tests.

        Imported lazily and defensively: the execution sandbox is a heavier dependency and
        an optional part of the install, so its absence must degrade the run rather than
        break the import of this module.
        """
        if self._test_runner is not None:
            return self._test_runner
        try:
            from ..execution import run_project_tests
        except Exception:  # noqa: BLE001 - optional component
            return None
        self._test_runner = run_project_tests
        return self._test_runner

    def _resolve_agent_runner(self) -> Optional[AgentRunnerFn]:
        """
        Find something that can execute the generated agent, for the same reasons as above.

        The credential is bound in here rather than passed down through the stage, so the key
        appears in exactly one call and never in a stage's ``data`` dictionary - which is
        serialised into events, stored and streamed to the browser.
        """
        if self._agent_runner is not None:
            return self._agent_runner
        try:
            from ..execution import run_agent
        except Exception:  # noqa: BLE001 - optional component
            return None

        api_key = self._request.runtime_api_key if self._request else None

        def _run(project: GeneratedProject, query: str) -> RunResult:
            return run_agent(project, query, api_key=api_key)

        self._agent_runner = _run
        return self._agent_runner

    # --------------------------------------------------------------------------- events
    _current_stage: Stage = Stage.ANALYSIS

    def _stage(self, stage: Stage, message: str, iteration: Optional[int] = None):
        return _StageContext(self, stage, message, iteration)

    def _warn(
        self,
        stage: Stage,
        message: str,
        *,
        iteration: Optional[int] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(
            PipelineEvent(
                stage=stage,
                status="warning",
                message=message,
                iteration=iteration,
                error=error,
                project_id=self._request.project_id if self._request else None,
                run_id=self._request.run_id if self._request else None,
            )
        )

    def _emit(self, event: PipelineEvent) -> None:
        if self._result is not None:
            self._result.events.append(event)
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001
                # A broken listener must not take down the run it is only observing.
                pass


class _StageContext:
    """
    Emits a started/succeeded/failed event pair around a stage.

    Using a context manager rather than decorating each stage means the duration is
    measured around the real work and a failure event is emitted even when the stage raises
    - which is exactly when the trace matters most.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        stage: Stage,
        message: str,
        iteration: Optional[int] = None,
    ) -> None:
        self._pipeline = pipeline
        self._stage = stage
        self.message = message
        self.status = "succeeded"
        self.data: Dict[str, Any] = {}
        self._iteration = iteration
        self._started = 0.0

    def __enter__(self) -> "_StageContext":
        self._pipeline._current_stage = self._stage
        self._started = time.monotonic()
        self._pipeline._emit(self._event("started", self.message, duration=None))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.monotonic() - self._started
        if exc is None:
            self._pipeline._emit(self._event(self.status, self.message, duration=elapsed))
            return False
        error = exc.to_dict() if isinstance(exc, AppError) else None
        message = exc.message if isinstance(exc, AppError) else str(exc)
        self._pipeline._emit(self._event("failed", message, duration=elapsed, error=error))
        return False  # never swallow: Pipeline.run owns the decision

    def _event(
        self,
        status: str,
        message: str,
        *,
        duration: Optional[float],
        error: Optional[Dict[str, Any]] = None,
    ) -> PipelineEvent:
        request = self._pipeline._request
        return PipelineEvent(
            stage=self._stage,
            status=status,
            message=message,
            duration_s=duration,
            iteration=self._iteration,
            error=error,
            data=dict(self.data),
            project_id=request.project_id if request else None,
            run_id=request.run_id if request else None,
        )


def _testing_is_pointless(report: ValidationReport) -> bool:
    """
    Whether a blocking gate makes running the suite a waste of minutes.

    True only for the categories in :data:`_BLOCKS_TESTING` - imports and structure - because
    those are the ones that stop the project's *test* modules importing too. A failing entry
    point or a missing tool implementation does not prevent collection, so the suite still runs
    and reports what it finds.

    This gates *when* the suite runs, never *whether* readiness requires it. A project whose
    suite was skipped for this reason cannot reach READY: the missing test report is itself an
    unmet criterion.
    """
    return any(
        check.category in _BLOCKS_TESTING for check in report.blocking()
    )


def _with_test_failures(
    review: ReviewResult,
    tests: Optional[TestReport],
) -> ReviewResult:
    """
    Fold failing tests into a review so the improver knows about them.

    A shallow copy is returned rather than mutating the review, because the review stored
    on the :class:`IterationRecord` must stay a faithful record of what the reviewer said.
    Editing it in place would rewrite history to include findings the reviewer never made.
    """
    if tests is None or tests.ok or not tests.ran:
        return review

    extra = [
        f"Fix the failing test {name}." for name in tests.failed_tests[:5]
    ] or ["Fix the failing generated tests."]
    tail = (tests.stdout or tests.stderr or "").strip()[-2000:]
    if tail:
        extra.append(
            "The test output was:\n" + tail
        )

    return ReviewResult(
        score=review.score,
        passed=False,
        summary=(
            f"{review.summary} "
            f"{tests.failed_count} generated test(s) failed."
        ).strip(),
        issues=list(review.issues),
        required_changes=list(review.required_changes) + extra,
        dimension_scores=dict(review.dimension_scores),
        from_model=review.from_model,
    )


def run_pipeline(
    requirement: str,
    *,
    framework: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    on_event: Optional[EventHook] = None,
    **kwargs: Any,
) -> PipelineResult:
    """
    Convenience wrapper for the common case.

    Keeps the simple call short - ``run_pipeline("Create a research assistant")`` - while
    the class stays available for callers that need to inject a provider or a test runner.
    """
    request = PipelineRequest(
        requirement=requirement,
        framework=framework,
        provider=provider,
        model=model,
        **kwargs,
    )
    return Pipeline(on_event=on_event).run(request)
