# backend/api/runs.py
"""
Watching a run.

Two ways to follow the same work, because a UI needs both.

``GET /api/runs/{id}`` is a snapshot: status, events so far, and the result once there is one.
It answers "what happened" for a page that has just loaded, and it is the only thing a client
needs if it is happy to poll.

``GET /api/runs/{id}/events`` is a live stream over server-sent events. SSE rather than a
WebSocket because the traffic is strictly one-directional - the server narrates, the client
listens - and SSE gets that over plain HTTP with automatic browser reconnection and no
protocol upgrade for a proxy to mishandle.

Both are backed by the same in-memory job, and both fall back to the database when the job is
gone: after a server restart, or thirty minutes after a run finished, the run is still fully
readable from its stored logs. Without that fallback a completed run would 404 once memory was
reclaimed, which looks exactly like losing the user's work.
"""
from __future__ import annotations

import asyncio
import functools
import json
import queue
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from multi_agent_generator.errors import NotFoundError

from ..config import AppContext
from ..deps import context
from ..services.jobs import Job, JobManager, get_job_manager

router = APIRouter(tags=["runs"])

#: How long to wait for an event before emitting a keepalive comment. Proxies and browsers
#: drop idle connections, and a long generation stage can legitimately be silent for a while.
_HEARTBEAT_SECONDS = 15.0


@router.get("/runs/{run_id}", summary="Run status, events and result")
def get_run(run_id: str, ctx: AppContext = Depends(context)) -> Dict[str, Any]:
    job = get_job_manager().get(run_id)
    if job is not None:
        return job.snapshot()

    stored = _stored_run(ctx, run_id)
    if stored is None:
        raise NotFoundError(
            "That run does not exist.",
            action="It may have been deleted with its project.",
            context={"run_id": run_id},
        )
    return stored


@router.get("/runs/{run_id}/events", summary="Live run progress (server-sent events)")
async def stream_run_events(
    run_id: str,
    request: Request,
    ctx: AppContext = Depends(context),
) -> StreamingResponse:
    job = get_job_manager().get(run_id)

    if job is None:
        stored = _stored_run(ctx, run_id)
        if stored is None:
            raise NotFoundError(
                "That run does not exist.",
                action="It may have been deleted with its project.",
                context={"run_id": run_id},
            )
        # The run is over and out of memory. Replay its stored log and close, so a client
        # that reconnects to a finished run still renders the full history.
        return StreamingResponse(
            _replay(stored),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    return StreamingResponse(
        _live(job, request),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )


#: ``no-cache`` and ``X-Accel-Buffering: no`` together are what actually make streaming
#: work end to end - without them nginx buffers the whole response and the client sees
#: nothing until the run is over, which defeats the point.
_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


async def _live(job: Job, request: Request) -> AsyncIterator[str]:
    """
    Stream a running job's events.

    The job hands out a queue pre-loaded with everything that already happened, so a client
    connecting mid-run is caught up before it sees anything new.
    """
    sink = job.subscribe()
    loop = asyncio.get_running_loop()
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                # queue.get blocks, so it runs in a worker thread. The timeout is what lets
                # this loop notice a disconnected client and emit heartbeats during a long
                # silent stage.
                item = await loop.run_in_executor(
                    None, functools.partial(sink.get, True, _HEARTBEAT_SECONDS)
                )
            except queue.Empty:
                yield ": keepalive\n\n"
                continue

            if item is JobManager.DONE:
                break
            yield _sse("stage", item)

        snapshot = job.snapshot(include_events=False)
        yield _sse("done", snapshot)
    finally:
        # Always detach: a client that closes its tab must not leave a queue behind that
        # the worker keeps filling.
        job.unsubscribe(sink)


async def _replay(stored: Dict[str, Any]) -> AsyncIterator[str]:
    """Replay a finished run from the database, then close."""
    for entry in stored.get("logs") or []:
        yield _sse("stage", entry)
    yield _sse("done", {k: v for k, v in stored.items() if k != "logs"})


def _stored_run(ctx: AppContext, run_id: str) -> Optional[Dict[str, Any]]:
    if ctx.storage is None:
        return None
    return ctx.storage.runs.get(run_id)
