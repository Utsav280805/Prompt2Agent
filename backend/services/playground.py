"""Background execution for Playground requests.

Playground runs can spend minutes installing a framework before the generated agent starts.
Keeping that work behind a short submit request prevents browser and Vite proxy connections from
being held open for the entire install and execution lifetime.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from multi_agent_generator.core.models import GeneratedProject, RunResult
from multi_agent_generator.execution import run_agent


@dataclass
class PlaygroundJob:
    run_id: str
    project_id: str
    status: str = "queued"
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def finished(self) -> bool:
        return self.status in ("done", "error")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "status": self.status,
            "finished": self.finished,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(
                (self.finished_at or time.time()) - self.started_at, 3
            ),
            "result": self.result,
            "error": self.error,
        }


class PlaygroundJobManager:
    """Bounded in-process workers for generated-agent execution."""

    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="magen-playground")
        self._jobs: Dict[str, PlaygroundJob] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        project: GeneratedProject,
        project_id: str,
        query: str,
        *,
        timeout: Optional[int],
        api_key: Optional[str],
        install: bool,
    ) -> PlaygroundJob:
        job = PlaygroundJob(run_id=f"pgr_{uuid.uuid4().hex[:16]}", project_id=project_id)
        with self._lock:
            self._jobs[job.run_id] = job
        self._pool.submit(self._execute, job, project, query, timeout, api_key, install)
        return job

    def get(self, run_id: str) -> Optional[PlaygroundJob]:
        with self._lock:
            return self._jobs.get(run_id)

    def _execute(
        self,
        job: PlaygroundJob,
        project: GeneratedProject,
        query: str,
        timeout: Optional[int],
        api_key: Optional[str],
        install: bool,
    ) -> None:
        job.status = "running"
        try:
            outcome: RunResult = run_agent(
                project,
                query,
                timeout=timeout,
                api_key=api_key,
                install=install,
            )
            job.result = outcome.as_dict()
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - surface worker failure as API data
            job.error = {
                "code": "playground_worker_failed",
                "message": "The Playground worker failed before it returned a result.",
                "detail": f"{type(exc).__name__}: {exc}",
            }
            job.status = "error"
        finally:
            job.finished_at = time.time()


_manager: Optional[PlaygroundJobManager] = None
_manager_lock = threading.Lock()


def get_playground_manager() -> PlaygroundJobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = PlaygroundJobManager()
        return _manager


def reset_playground_manager() -> None:
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager._pool.shutdown(wait=False, cancel_futures=True)
        _manager = None
