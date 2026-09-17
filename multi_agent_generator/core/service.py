# multi_agent_generator/core/service.py
"""
The service layer: one entry point every caller shares.

The CLI, the FastAPI backend and any future caller all go through :class:`GeneratorService`.
That is the whole reason it exists. The moment two callers each assemble a pipeline their
own way, they drift - one forgets to pass the runtime provider, another defaults tests off -
and the product behaves differently depending on how it was invoked. A single service makes
"generate a project" mean exactly one thing.

The service is deliberately thin. It owns no policy the pipeline does not already own; it
resolves defaults from settings, runs the pipeline, and offers a couple of narrower entry
points (analysis only, a connection check) for the parts of the UI that do not need a full
run. Anything requiring real judgement lives one layer down, in the stage modules, where it
can be tested without a service around it.

Persistence is injected, not imported. A caller that wants runs saved passes a ``store``;
the default is no persistence at all, so the CLI can generate a project without a database
existing. This keeps the dependency arrow pointing the right way: the service knows the
*shape* of a store (a tiny protocol) but not SQLite, and the storage layer depends on the
service's models rather than the other way round.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from ..errors import AppError
from ..llm import LLMConfig, build_provider
from ..llm.base import ConnectionCheck
from ..settings import Settings, get_settings
from .analysis import analyze_requirement, heuristic_analysis
from .models import PipelineEvent, PipelineResult, RequirementAnalysis
from .pipeline import EventHook, Pipeline, PipelineRequest, TestRunnerFn

__all__ = ["GeneratorService", "RunStore", "get_service", "set_service", "build_service"]


@runtime_checkable
class RunStore(Protocol):
    """
    The persistence surface the service depends on.

    A protocol rather than a concrete class so the storage layer can satisfy it without the
    service importing anything from it. The methods are intentionally coarse - one call at
    the start of a run, one at the end - because finer-grained persistence would couple the
    store to the pipeline's internal stage boundaries, which are free to change.
    """

    def start_run(self, request: "PipelineRequest") -> Optional[str]:
        """Record that a run began. Returns a run id, or None if not persisted."""

    def finish_run(self, run_id: Optional[str], result: "PipelineResult") -> None:
        """Record a finished run and its result."""

    def record_event(self, run_id: Optional[str], event: "PipelineEvent") -> None:
        """Record one pipeline event as it happens (optional; may be a no-op)."""


class GeneratorService:
    """
    Shared façade over the pipeline.

    Construct one and reuse it: a service instance may cache a validated provider across
    calls, which matters for the API, where rebuilding and re-validating a provider on every
    request would add a network round-trip to each call.
    """

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        store: Optional[RunStore] = None,
        test_runner: Optional[TestRunnerFn] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._store = store
        self._test_runner = test_runner

    # --------------------------------------------------------------------------- runs
    def generate(
        self,
        requirement: str,
        *,
        framework: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        runtime_provider: Optional[str] = None,
        runtime_model: Optional[str] = None,
        include_tests: bool = True,
        run_tests: bool = True,
        use_model: bool = True,
        max_iterations: Optional[int] = None,
        llm_config: Optional[LLMConfig] = None,
        saved_llm: Optional[Mapping[str, Any]] = None,
        on_event: Optional[EventHook] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline for ``requirement`` and return its result.

        This never raises for an expected failure: the pipeline converts a missing
        credential or unrunnable code into a ``FAILED`` result carrying the reason and the
        event trace. It is the single method the CLI's ``generate`` command and the API's
        ``POST /api/projects`` both call, so their behaviour cannot diverge.

        ``project_id`` ties the run to a stored project. ``run_id`` lets the caller name the
        run *before* it starts, which the API needs: ``POST /api/projects`` has to return an
        id the client can immediately subscribe to for progress, and it cannot do that if the
        id is only decided once the worker reaches the database.
        """
        request = PipelineRequest(
            requirement=requirement,
            framework=framework,
            provider=provider,
            model=model,
            runtime_provider=runtime_provider,
            runtime_model=runtime_model,
            include_tests=include_tests,
            run_tests=run_tests,
            use_model=use_model,
            max_iterations=max_iterations,
            project_id=project_id,
            run_id=run_id,
            llm_config=llm_config,
            saved_llm=saved_llm,
            metadata=dict(metadata or {}),
        )

        # The store honours a pre-assigned id and only invents one when the caller did not
        # supply it, so the id the caller was handed stays the id in the database.
        if self._store is not None:
            run_id = self._store.start_run(request) or run_id
        request.run_id = run_id

        # Persist each event as it happens *and* pass it to the caller's own hook, so the
        # database record and the live UI stream see the same sequence.
        def _sink(event: PipelineEvent) -> None:
            if self._store is not None:
                try:
                    self._store.record_event(run_id, event)
                except Exception:  # noqa: BLE001 - persistence must not break a run
                    pass
            if on_event is not None:
                on_event(event)

        pipeline = Pipeline(
            on_event=_sink,
            settings=self.settings,
            test_runner=self._test_runner,
        )
        result = pipeline.run(request)

        if self._store is not None:
            try:
                self._store.finish_run(run_id, result)
            except Exception:  # noqa: BLE001
                # A run that succeeded but could not be saved is still a run that succeeded;
                # losing the record is a smaller failure than discarding a good result.
                pass
        return result

    # ------------------------------------------------------------------ narrower calls
    def analyze(
        self,
        requirement: str,
        *,
        use_model: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> RequirementAnalysis:
        """
        Just the analysis stage.

        The project-creation form uses this to show the user what was understood - roles,
        tools, complexity - before they commit to a full generation run. It degrades to the
        heuristic on any model problem, because a preview that fails is worse than a preview
        labelled "offline".
        """
        if not use_model:
            return heuristic_analysis(requirement)
        try:
            provider_obj = self._provider(provider, model, llm_config)
        except AppError:
            return heuristic_analysis(requirement)
        try:
            return analyze_requirement(requirement, provider_obj)
        except AppError:
            return heuristic_analysis(requirement)

    def check_connection(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        saved: Optional[Mapping[str, Any]] = None,
    ) -> ConnectionCheck:
        """
        Test that a provider configuration actually works.

        This backs the "Test connection" button on the settings page. It returns a
        :class:`ConnectionCheck` (ok / message / latency) rather than raising, because the
        settings page wants to *render* a failure, not handle an exception - and a failed
        credential test is an expected outcome there, not an error.
        """
        try:
            provider_obj = build_provider(
                provider,
                model,
                api_key=api_key,
                base_url=base_url,
                config=llm_config,
                saved=saved,
            )
        except AppError as exc:
            # A bad provider name or an unresolvable config never reaches a provider
            # instance, so there is nothing to ask - report the configuration error in the
            # same shape a failed live check would use.
            return ConnectionCheck(
                ok=False,
                provider=str(provider or self.settings.default_provider),
                model=str(model or ""),
                message=exc.message,
                detail=exc.action,
                code=exc.code,
            )
        return provider_obj.test_connection()

    def describe_provider(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        saved: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        """A redacted description of a resolved provider config, for the settings UI."""
        config = llm_config or LLMConfig.resolve(
            provider=provider, model=model, saved=saved
        )
        return config.redacted()

    # -------------------------------------------------------------------------- helper
    def _provider(
        self,
        provider: Optional[str],
        model: Optional[str],
        llm_config: Optional[LLMConfig],
    ):
        built = build_provider(provider, model, config=llm_config)
        built.validate()
        return built


#: Process-wide default service. Most callers want the same one; the API replaces it with a
#: store-backed instance at startup. Kept behind a function so tests can reset it.
_DEFAULT_SERVICE: Optional[GeneratorService] = None


def build_service(
    *,
    persist: bool = True,
    settings: Optional[Settings] = None,
) -> GeneratorService:
    """
    Construct a service, wiring in persistence and the real test runner.

    This is the constructor the CLI and the API actually use. It lives here, rather than in
    each caller, so that "a service that saves its runs and can run tests for real" is
    assembled in exactly one place - the same single-definition argument that motivates the
    service itself.

    The storage import is deferred to call time on purpose: the service *module* must not
    depend on SQLite, so that importing the pipeline never drags in a database. ``persist``
    can be turned off (tests, a dry run) to get a service that generates without writing
    anything, which is also the graceful fallback if the storage layer cannot start.
    """
    store: Optional[RunStore] = None
    if persist:
        try:
            from ..storage import Storage

            store = Storage(settings=settings)
        except Exception:  # noqa: BLE001 - a missing/unwritable DB must not block generating
            store = None

    test_runner: Optional[TestRunnerFn] = None
    try:
        from ..execution import run_project_tests

        test_runner = run_project_tests
    except Exception:  # noqa: BLE001 - execution sandbox optional; pipeline degrades cleanly
        test_runner = None

    return GeneratorService(settings=settings, store=store, test_runner=test_runner)


def get_service() -> GeneratorService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = GeneratorService()
    return _DEFAULT_SERVICE


def set_service(service: Optional[GeneratorService]) -> None:
    """Replace (or reset, with None) the process-wide service. Used by the API and by tests."""
    global _DEFAULT_SERVICE
    _DEFAULT_SERVICE = service
