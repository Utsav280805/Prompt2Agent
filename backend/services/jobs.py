# backend/services/jobs.py
"""
Running generation in the background.

A generation run makes several model calls, then installs nothing but does execute a test
suite in a subprocess. Thirty seconds is a good run; three minutes is not unusual. That is far
too long for a request to hold open, so ``POST /api/projects`` starts the work and returns
immediately with a run id, and the client watches progress on ``/api/runs/{id}/events``.

The design decisions here, and why:

**A bounded thread pool, not unbounded threads.** Each run spawns a subprocess for the test
suite, so N concurrent runs means N concurrent pytest processes. A pool with a small cap turns
"the machine falls over under ten clicks" into "the tenth run waits its turn". The cap is
deliberately low because the work is heavy, not because threads are expensive.

**Threads, not ``async``.** The pipeline is synchronous, blocking code - it calls provider SDKs
and ``subprocess.communicate``. Running it on the event loop would freeze every other request;
rewriting it as async would mean an async and a sync copy of the pipeline, one of which would
rot. A worker thread is the honest way to host blocking work in an async server.

**Events are buffered per run, not just streamed.** A client that connects after a run started -
a page refresh mid-generation - must still see the earlier stages. So every event is appended
to the job's history *and* pushed to any live subscriber, and a new subscriber is replayed the
history first. Streaming only would make the progress view depend on lucky timing.

**A finished job is not forgotten immediately.** Results stay in memory briefly after
completion so a client that polls a moment late still gets the terminal state rather than a
404 that looks like "the run vanished".
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from multi_agent_generator.core.models import PipelineEvent, PipelineResult, ProjectStatus, Stage
from multi_agent_generator.core.pipeline import PipelineRequest
from multi_agent_generator.errors import AppError
from multi_agent_generator.logging_config import StageLogger

__all__ = ["Job", "JobManager", "get_job_manager", "reset_job_manager"]


#: How many pipelines may run at once. Each one can hold a pytest subprocess, so this is a
#: resource limit rather than a concurrency tuning knob.
DEFAULT_MAX_WORKERS = 2

#: How long a finished job stays in memory after completion.
RETENTION_SECONDS = 30 * 60

#: Per-subscriber queue depth. A subscriber that stops reading (a closed browser tab) is
#: dropped rather than allowed to grow without bound.
_SUBSCRIBER_QUEUE_SIZE = 512

#: Sentinel pushed to subscribers when a run finishes, so a stream can close cleanly instead
#: of waiting for a timeout.
_DONE = object()


@dataclass
class Job:
    """One in-flight or recently finished generation run."""

    run_id: str
    project_id: Optional[str]
    requirement: str
    status: str = "queued"  # queued | running | done | error
    events: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    _subscribers: List["queue.Queue[Any]"] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def finished(self) -> bool:
        return self.status in ("done", "error")

    def snapshot(self, *, include_events: bool = True) -> Dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "project_id": self.project_id,
                "requirement": self.requirement,
                "status": self.status,
                "finished": self.finished,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duration_s": round((self.finished_at or time.time()) - self.started_at, 3),
                "event_count": len(self.events),
                "events": list(self.events) if include_events else [],
                "result": self.result,
                "error": self.error,
            }

    # ---------------------------------------------------------------------- event fan-out
    def publish(self, payload: Dict[str, Any]) -> None:
        """Record an event and hand it to every live subscriber."""
        with self._lock:
            self.events.append(payload)
            targets = list(self._subscribers)
        for sink in targets:
            try:
                sink.put_nowait(payload)
            except queue.Full:
                # The subscriber is not keeping up - almost always an abandoned connection.
                # Dropping it protects the run; the event is still in `events`, so a
                # reconnecting client loses nothing.
                self._detach(sink)

    def subscribe(self) -> "queue.Queue[Any]":
        """
        Attach a listener, pre-loaded with everything that already happened.

        The history is copied under the same lock that appends new events, so a subscriber
        cannot miss an event that lands between "read history" and "start listening", nor see
        one twice.
        """
        sink: "queue.Queue[Any]" = queue.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        with self._lock:
            for payload in self.events:
                try:
                    sink.put_nowait(payload)
                except queue.Full:
                    break
            if self.finished:
                sink.put_nowait(_DONE)
            else:
                self._subscribers.append(sink)
        return sink

    def unsubscribe(self, sink: "queue.Queue[Any]") -> None:
        self._detach(sink)

    def _detach(self, sink: "queue.Queue[Any]") -> None:
        with self._lock:
            if sink in self._subscribers:
                self._subscribers.remove(sink)

    def close(self) -> None:
        """Mark the stream complete and release every subscriber."""
        with self._lock:
            targets = list(self._subscribers)
            self._subscribers.clear()
        for sink in targets:
            try:
                sink.put_nowait(_DONE)
            except queue.Full:
                pass


class JobManager:
    """
    Owns the worker pool and the job registry.

    One instance per process, held by the app context.
    """

    #: Exposed so the SSE route can recognise the end-of-stream sentinel.
    DONE = _DONE

    def __init__(self, *, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="magen-run"
        )
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------- lifecycle
    def submit(
        self,
        *,
        run_id: str,
        project_id: Optional[str],
        request: PipelineRequest,
        service: Any,
        storage: Any = None,
    ) -> Job:
        """
        Queue a pipeline run and return its job immediately.

        ``run_id`` is generated by the caller before submitting, so the HTTP response can
        name the run the client should watch. Letting the worker allocate it would create a
        window in which the client has nothing to subscribe to.
        """
        job = Job(run_id=run_id, project_id=project_id, requirement=request.requirement)
        self._reap()
        with self._lock:
            self._jobs[run_id] = job

        self._pool.submit(self._execute, job, request, service, storage)
        return job

    def get(self, run_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(run_id)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if not job.finished)

    def shutdown(self) -> None:
        # wait=False: shutdown happens on server stop, and a run's own timeouts already
        # bound how long its subprocess can live. Blocking here would make Ctrl-C hang.
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---------------------------------------------------------------------------- worker
    def _execute(
        self,
        job: Job,
        request: PipelineRequest,
        service: Any,
        storage: Any,
    ) -> None:
        log = StageLogger("backend.jobs", project_id=job.project_id, run_id=job.run_id)
        job.status = "running"
        log.info("Generation run started.", status="started")

        def on_event(event: PipelineEvent) -> None:
            payload = event.as_dict()
            payload.setdefault("project_id", job.project_id)
            payload["run_id"] = job.run_id
            job.publish(payload)
            log.event(event)

        try:
            result: PipelineResult = service.generate(
                request.requirement,
                framework=request.framework,
                provider=request.provider,
                model=request.model,
                runtime_provider=request.runtime_provider,
                runtime_model=request.runtime_model,
                include_tests=request.include_tests,
                run_tests=request.run_tests,
                use_model=request.use_model,
                max_iterations=request.max_iterations,
                llm_config=request.llm_config,
                saved_llm=request.saved_llm,
                on_event=on_event,
                # Passing the ids the caller already handed the client keeps the job, the
                # database row and the log lines all naming the same run.
                project_id=job.project_id,
                run_id=job.run_id,
                metadata=request.metadata,
            )
        except AppError as exc:
            # An expected failure that escaped the pipeline. Recorded on the job so the UI
            # can render it, exactly as if it had been returned synchronously.
            job.error = exc.to_dict()
            job.status = "error"
            log.error(exc.message, status="failed", error_code=exc.code)
            self._persist_failure(storage, job)
        except Exception as exc:  # noqa: BLE001 - a bug; must not kill the worker thread
            job.error = {
                "code": "internal_error",
                "message": "The generation run failed unexpectedly.",
                "detail": f"{type(exc).__name__}: {exc}",
            }
            job.status = "error"
            log.exception("Generation run crashed.", status="failed")
            self._persist_failure(storage, job)
        else:
            job.result = result.as_dict(include_content=True)
            job.status = "done"
            self._persist_result(storage, job, result)
            log.info(
                "Generation run finished.",
                status="succeeded" if result.succeeded else "failed",
                duration_s=result.duration_s,
            )
        finally:
            job.finished_at = time.time()
            job.close()

    # ------------------------------------------------------------------------ persistence
    def _persist_result(self, storage: Any, job: Job, result: PipelineResult) -> None:
        if storage is None or not job.project_id:
            return
        try:
            storage.projects.save_result(job.project_id, result)
        except Exception:  # noqa: BLE001
            # A result that cannot be saved is still a result. It is in the job and will be
            # returned to the client; losing the row is the lesser failure.
            StageLogger("backend.jobs", project_id=job.project_id, run_id=job.run_id).warning(
                "The run finished but its result could not be saved."
            )

    def _persist_failure(self, storage: Any, job: Job) -> None:
        """Mark the project failed so the workspace list does not show it stuck 'generating'."""
        if storage is None or not job.project_id:
            return
        failed = PipelineResult(
            requirement=job.requirement,
            status=ProjectStatus.FAILED,
            error=job.error,
        )
        try:
            storage.projects.save_result(job.project_id, failed)
            storage.runs.finish_run(job.run_id, failed)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------------ retention
    def _reap(self) -> None:
        """Drop finished jobs older than the retention window."""
        cutoff = time.time() - RETENTION_SECONDS
        with self._lock:
            stale = [
                run_id
                for run_id, job in self._jobs.items()
                if job.finished and (job.finished_at or 0) < cutoff
            ]
            for run_id in stale:
                self._jobs.pop(run_id, None)


def make_queued_event(run_id: str, project_id: Optional[str], message: str) -> Dict[str, Any]:
    """
    A synthetic 'queued' event, so the UI shows something the instant a run is created.

    Without it the progress panel is blank until the analysis stage emits, which reads as a
    broken button. It is marked ``status="info"`` rather than ``started`` so it cannot be
    mistaken for a real stage having begun.
    """
    return PipelineEvent(
        stage=Stage.ANALYSIS,
        status="info",
        message=message,
        project_id=project_id,
        run_id=run_id,
    ).as_dict()


_manager: Optional[JobManager] = None
_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """The process-wide job manager."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager


def reset_job_manager() -> None:
    """
    Shut the manager down and discard it.

    Discarding matters as much as shutting down: a ``ThreadPoolExecutor`` that has been shut
    down rejects new work forever, so leaving the dead one in place would make every run in a
    second application instance fail with ``cannot schedule new futures``. That is not a
    hypothetical - it is what happens to the second test that builds an app, and to any
    process that stops and restarts the server in place.
    """
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.shutdown()
        _manager = None
