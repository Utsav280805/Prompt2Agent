# backend/api/projects.py
"""
Projects: create, list, inspect, download, delete.

``POST /api/projects`` is the endpoint the whole product turns on, and it is deliberately
*asynchronous*. It creates the project row, mints a run id, hands the work to a background
worker and returns ``202 Accepted`` immediately. The client then watches
``/api/runs/{run_id}/events``.

The alternative - holding the request open for the two minutes a real run takes - fails for
reasons that are not hypothetical: proxies and load balancers time out well before that, the
browser cannot show intermediate progress, and a refresh loses the run entirely. Returning a
run id makes progress observable and survivable.

Reads are split by cost. The list endpoint omits generated source (a project can be tens of
kilobytes and a workspace can hold dozens), the detail endpoint includes it, and the file tree
is available separately for the Files tab. That way opening the workspace list does not pull
every byte the user has ever generated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Response, status

from multi_agent_generator.core.models import GeneratedProject
from multi_agent_generator.core.pipeline import PipelineRequest
from multi_agent_generator.core.service import GeneratorService
from multi_agent_generator.errors import NotFoundError
from multi_agent_generator.storage import Storage, new_id

from ..config import AppContext
from ..deps import context, project_record, require_storage, service
from ..schemas import CreateProjectRequest, ProjectSummary
from ..services.export import project_zip_bytes, suggest_filename
from ..services.jobs import get_job_manager

router = APIRouter(tags=["projects"])


@router.post(
    "/projects",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a project and start generating",
)
def create_project(
    payload: CreateProjectRequest,
    ctx: AppContext = Depends(context),
    svc: GeneratorService = Depends(service),
) -> Dict[str, Any]:
    """
    Start a run. Returns at once with the ids needed to follow it.

    Works without a database: the run still executes and streams, it simply is not saved.
    Refusing to generate because persistence is down would turn a degraded deployment into a
    useless one.
    """
    store = ctx.storage
    project_id: Optional[str] = None
    if store is not None:
        project_id = store.projects.create(
            requirement=payload.requirement,
            name=payload.name,
            framework=payload.framework,
            provider=payload.provider,
            model=payload.model,
        )

    # Minted here, before the work is queued, so the response can name a run the client may
    # subscribe to immediately. The store honours this id rather than inventing its own.
    run_id = new_id("run")

    saved_llm = None
    if store is not None:
        # reveal=True: the pipeline needs the usable key. It travels in memory to the
        # provider and is never written to a response or a log.
        saved_llm = store.saved_llm(reveal=True)

    request = PipelineRequest(
        requirement=payload.requirement,
        framework=payload.framework,
        provider=payload.provider,
        model=payload.model,
        runtime_provider=payload.runtime_provider,
        runtime_model=payload.runtime_model,
        include_tests=payload.include_tests,
        run_tests=payload.run_tests,
        use_model=payload.use_model,
        max_iterations=payload.max_iterations,
        project_id=project_id,
        run_id=run_id,
        saved_llm=saved_llm,
    )

    job = get_job_manager().submit(
        run_id=run_id,
        project_id=project_id,
        request=request,
        service=svc,
        storage=store,
    )

    return {
        "project_id": project_id,
        "run_id": job.run_id,
        "status": job.status,
        "persisted": store is not None,
        "events_url": f"/api/runs/{job.run_id}/events",
        "run_url": f"/api/runs/{job.run_id}",
    }


@router.get("/projects", response_model=List[ProjectSummary], summary="List projects")
def list_projects(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    store: Storage = Depends(require_storage),
) -> List[ProjectSummary]:
    return [ProjectSummary(**row) for row in store.projects.list(limit=limit, offset=offset)]


@router.get("/projects/{project_id}", summary="One project, with generated files")
def get_project(record: dict = Depends(project_record)) -> Dict[str, Any]:
    return record


@router.get("/projects/{project_id}/files", summary="The project's file tree")
def get_project_files(record: dict = Depends(project_record)) -> Dict[str, Any]:
    project = _generated_project(record)
    return {
        "project_id": record.get("id"),
        "framework": project.framework,
        "entrypoint": project.entrypoint,
        "file_count": len(project.files),
        "total_lines": project.total_lines,
        "files": [f.as_dict() for f in project.files],
    }


@router.get("/projects/{project_id}/runs", summary="Run history for a project")
def get_project_runs(
    record: dict = Depends(project_record),
    store: Storage = Depends(require_storage),
) -> List[Dict[str, Any]]:
    return store.runs.list_for_project(str(record["id"]))


@router.get("/projects/{project_id}/download", summary="Download the project as a zip")
def download_project(record: dict = Depends(project_record)) -> Response:
    project = _generated_project(record)
    if not project.files:
        raise NotFoundError(
            "This project has no generated files to download.",
            action="Generate the project first, or regenerate it if the run failed.",
            context={"project_id": record.get("id")},
        )

    filename = suggest_filename(record.get("name"), project.framework)
    body = project_zip_bytes(project, root=filename[:-4])
    return Response(
        content=body,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(body)),
        },
    )


@router.delete(
    "/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a project",
)
def delete_project(
    record: dict = Depends(project_record),
    store: Storage = Depends(require_storage),
) -> Response:
    # Runs and logs are removed by the schema's cascading foreign keys, so there is no
    # second delete to forget here.
    store.projects.delete(str(record["id"]))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _generated_project(record: dict) -> GeneratedProject:
    """
    Rebuild the runnable project from its stored result.

    A project row holds the whole pipeline result as JSON; the generated project sits under
    ``result.project``. A row without one is a draft or a failed run, which is a 404 for
    file-shaped requests rather than an empty success - "here are your zero files" reads as
    data loss.
    """
    result = record.get("result") or {}
    data = result.get("project") if isinstance(result, dict) else None
    if not data:
        raise NotFoundError(
            "This project has not produced any code yet.",
            action=(
                "Wait for the run to finish, or start a new run if the previous one failed."
            ),
            context={"project_id": record.get("id"), "status": record.get("status")},
        )
    return GeneratedProject.from_dict(data)
