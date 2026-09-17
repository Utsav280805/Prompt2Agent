# backend/api/playground.py
"""
The Playground: running a generated agent against a real question.

This is the endpoint that proves the product works. Everything before it produces code that
*looks* right; this executes it and returns what the agent actually said.

It is also the only endpoint that runs untrusted code, so it is the one place the sandbox
guarantees matter. All of them are enforced by :mod:`multi_agent_generator.execution`, not
re-implemented here:

- the project is written to a fresh workspace, with every path re-validated against its root;
- the child process gets an allowlisted environment plus **at most one** credential, the one
  its own provider needs, so the application's other secrets are absent from the environment
  rather than merely unmentioned;
- the project's dependencies are installed into a content-keyed directory on the child's
  import path, never into the interpreter serving this request;
- a wall-clock timeout kills the whole process tree, so a runaway agent cannot outlive the
  request that started it;
- the credential is passed in memory for that one run and is never written into the generated
  source, so a downloaded project contains no keys;
- the captured stdout and stderr in the response body have already been scrubbed of that
  credential's literal value. This matters more than it sounds: provider SDKs put the key into
  their authentication error messages, and without the scrub a wrong-key run would return the
  key to the browser.

The run is synchronous. Unlike generation it is a single bounded step, the user is watching a
spinner they asked for, and ``AGENT_EXECUTION_TIMEOUT`` already caps how long it can take.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends

from multi_agent_generator.core.models import GeneratedProject
from multi_agent_generator.errors import NotFoundError
from multi_agent_generator.llm.config import LLMConfig
from multi_agent_generator.logging_config import StageLogger

from ..config import AppContext
from ..deps import context, project_record
from ..schemas import RunAgentRequest
from ..services.playground import get_playground_manager

router = APIRouter(tags=["playground"])


@router.post("/projects/{project_id}/run", summary="Run the generated agent")
def run_project_agent(
    payload: RunAgentRequest,
    record: dict = Depends(project_record),
    ctx: AppContext = Depends(context),
) -> Dict[str, Any]:
    result = (record.get("result") or {})
    data = result.get("project") if isinstance(result, dict) else None
    if not data:
        raise NotFoundError(
            "This project has no code to run yet.",
            action="Generate the project first, then try again.",
            context={"project_id": record.get("id"), "status": record.get("status")},
        )

    project = GeneratedProject.from_dict(data)
    if not project.files:
        raise NotFoundError(
            "This project's files were not saved, so there is nothing to run.",
            action="Regenerate the project.",
            context={"project_id": record.get("id")},
        )

    log = StageLogger("backend.playground", project_id=str(record.get("id")))
    # Resolve the credential before queueing. It is held only in the worker's memory and is
    # scrubbed from the generated process output by run_agent.
    api_key = _credential_for(ctx, project.provider)
    job = get_playground_manager().submit(
        project,
        str(record.get("id")),
        payload.query,
        timeout=payload.timeout,
        api_key=api_key,
        install=payload.install,
    )
    return job.snapshot()


@router.get("/playground-runs/{run_id}", summary="Get Playground run status")
def get_playground_run(run_id: str) -> Dict[str, Any]:
    job = get_playground_manager().get(run_id)
    if job is None:
        raise NotFoundError(
            "That Playground run does not exist.",
            action="Start the query again from the Playground.",
            context={"run_id": run_id},
        )
    return job.snapshot()


def _credential_for(ctx: AppContext, provider: str) -> Optional[str]:
    """
    The plaintext key for ``provider``, or None.

    One of the few deliberate :meth:`~multi_agent_generator.errors.Secret.reveal` call sites.
    The value goes straight into the sandbox's environment for one run and is scrubbed out of
    anything that run produced before the response is built, so it is never stored, logged or
    returned.

    Returning None is a normal outcome, not a failure: a local provider needs no key, and a
    missing one for a hosted provider surfaces as the agent's own clear "credential missing"
    error from inside the sandbox. Raising here instead would deny the user the more useful
    message.
    """
    saved = ctx.storage.saved_llm(reveal=True) if ctx.storage is not None else None
    try:
        config = LLMConfig.resolve(provider=provider, saved=saved)
    except Exception:  # noqa: BLE001 - an unknown provider is the sandbox's story to tell
        return None
    return config.api_key.reveal()
