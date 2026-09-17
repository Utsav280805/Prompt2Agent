# multi_agent_generator/storage/repository.py
"""
Repositories: the only code that reads and writes the database.

Everything above this layer works in domain objects - a :class:`PipelineResult`, an
:class:`LLMConfig` - and everything below it works in rows. The repositories are the seam,
and keeping that seam narrow is what lets the storage backend change without the pipeline
noticing, and what lets the pipeline be tested with no database at all.

:class:`RunRepository` implements the :class:`~multi_agent_generator.core.service.RunStore`
protocol, so a :class:`GeneratorService` constructed with one will persist every run and
every event without the service knowing SQLite exists. That inversion is the point: the
service depends on the *shape* of a store, and this module supplies one.

The credential rule from :mod:`.secretbox` is enforced here and only here:
:class:`LLMConfigRepository` encrypts on the way in and decrypts on the way out, and no other
code path touches the ``api_key_enc`` column. A saved key is therefore never in plaintext at
rest and never returned to a caller that did not explicitly ask to reveal it.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Mapping, Optional

from ..core.models import PipelineEvent, PipelineResult, utcnow
from ..errors import ConfigurationError
from ..settings import Settings, get_settings
from .database import Database, from_json, to_json
from .secretbox import SecretBox

__all__ = ["ProjectRepository", "RunRepository", "LLMConfigRepository", "Storage"]


def new_id(prefix: str) -> str:
    """
    A prefixed, collision-resistant id.

    Public because the API mints a run id *before* submitting the run, so the HTTP response
    can name a run the client can subscribe to immediately. Both paths must produce the same
    format, so both call this.
    """
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class ProjectRepository:
    """Reads and writes the ``projects`` table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        *,
        requirement: str,
        name: Optional[str] = None,
        framework: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        project_id = new_id("prj")
        now = utcnow().isoformat()
        self._db.execute(
            """
            INSERT INTO projects
                (id, name, requirement, framework, provider, model, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, ?)
            """,
            (project_id, name or _default_name(requirement), requirement, framework,
             provider, model, now, now),
        )
        return project_id

    def save_result(self, project_id: str, result: PipelineResult) -> None:
        """Persist the outcome of a run against its project row."""
        choice = result.framework_choice
        project = result.project
        self._db.execute(
            """
            UPDATE projects
               SET status = ?, framework = ?, provider = ?, model = ?,
                   result_json = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                result.status.value,
                (choice.framework if choice else None),
                (project.provider if project else None),
                (project.model if project else None),
                to_json(result.as_dict(include_content=True)),
                utcnow().isoformat(),
                project_id,
            ),
        )

    def get(self, project_id: str) -> Optional[Dict[str, Any]]:
        row = self._db.query_one("SELECT * FROM projects WHERE id = ?", (project_id,))
        return _project_row(row) if row else None

    def list(self, *, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        rows = self._db.query_all(
            "SELECT * FROM projects ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        # The list view does not need the full generated source, which can be large; the
        # summary omits it and the detail endpoint fetches it via get().
        return [_project_row(row, include_result=False) for row in rows]

    def delete(self, project_id: str) -> bool:
        cursor = self._db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cursor.rowcount > 0


class RunRepository:
    """
    Reads and writes ``runs`` and ``logs``.

    Implements the :class:`RunStore` protocol, so a :class:`GeneratorService` can persist
    runs by holding an instance of this and nothing more.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ---- RunStore protocol -----------------------------------------------------------
    def start_run(self, request: Any) -> Optional[str]:
        """Record a run beginning. Accepts a PipelineRequest; reads only public fields."""
        run_id = getattr(request, "run_id", None) or new_id("run")
        self._db.execute(
            """
            INSERT INTO runs
                (id, project_id, kind, status, requirement, framework, provider, model, started_at)
            VALUES (?, ?, 'generation', 'running', ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                getattr(request, "project_id", None),
                getattr(request, "requirement", None),
                getattr(request, "framework", None),
                getattr(request, "provider", None),
                getattr(request, "model", None),
                utcnow().isoformat(),
            ),
        )
        return run_id

    def finish_run(self, run_id: Optional[str], result: PipelineResult) -> None:
        if not run_id:
            return
        self._db.execute(
            """
            UPDATE runs
               SET status = ?, completed_at = ?, duration_s = ?, succeeded = ?,
                   result_json = ?, error_json = ?
             WHERE id = ?
            """,
            (
                result.status.value,
                (result.completed_at or utcnow()).isoformat(),
                result.duration_s,
                1 if result.succeeded else 0,
                to_json(result.as_dict(include_content=False)),
                to_json(result.error),
                run_id,
            ),
        )

    def record_event(self, run_id: Optional[str], event: PipelineEvent) -> None:
        self._db.execute(
            """
            INSERT INTO logs
                (run_id, project_id, stage, status, level, message, iteration,
                 duration_s, timestamp, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                event.project_id,
                event.stage.value,
                event.status,
                event.level,
                event.message,
                event.iteration,
                event.duration_s,
                event.timestamp.isoformat(),
                to_json(event.data) if event.data else None,
            ),
        )

    # ---- reads -----------------------------------------------------------------------
    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self._db.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not row:
            return None
        record = _run_row(row)
        record["logs"] = self.logs(run_id)
        return record

    def list_for_project(self, project_id: str) -> List[Dict[str, Any]]:
        rows = self._db.query_all(
            "SELECT * FROM runs WHERE project_id = ? ORDER BY started_at DESC",
            (project_id,),
        )
        return [_run_row(row) for row in rows]

    def logs(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self._db.query_all(
            "SELECT * FROM logs WHERE run_id = ? ORDER BY id ASC", (run_id,)
        )
        return [_log_row(row) for row in rows]


class LLMConfigRepository:
    """
    Reads and writes the single-row ``llm_configs`` table.

    This is the one place a saved credential is encrypted and decrypted. The ``reveal``
    parameter on :meth:`get` defaults to False so that the common path - rendering the
    settings page - cannot leak the key, and a caller has to ask for it explicitly, in one
    greppable place, to get the plaintext for an actual request.
    """

    def __init__(self, db: Database, secretbox: SecretBox) -> None:
        self._db = db
        self._box = secretbox

    def save(self, config: Mapping[str, Any]) -> None:
        provider = config.get("provider")
        if not provider:
            # `provider` is NOT NULL in the schema, and it is also the one field the rest of
            # the app cannot do anything useful without. Refuse with a clear ConfigurationError
            # rather than letting a raw sqlite3.IntegrityError escape from the storage layer.
            raise ConfigurationError(
                "Cannot save LLM settings without a provider.",
                action="Choose a provider (for example 'openai' or 'huggingface') and save again.",
            )
        api_key = config.get("api_key")
        encrypted = self._box.encrypt(api_key) if api_key else None
        self._db.execute(
            """
            INSERT INTO llm_configs
                (id, provider, model, base_url, temperature, max_tokens, timeout, top_p,
                 api_key_enc, extra_json, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model,
                base_url = excluded.base_url,
                temperature = excluded.temperature,
                max_tokens = excluded.max_tokens,
                timeout = excluded.timeout,
                top_p = excluded.top_p,
                -- Keep the stored key if the caller did not supply a new one, so saving a
                -- temperature change does not silently wipe the credential.
                api_key_enc = COALESCE(excluded.api_key_enc, llm_configs.api_key_enc),
                extra_json = excluded.extra_json,
                updated_at = excluded.updated_at
            """,
            (
                str(provider),
                config.get("model"),
                config.get("base_url"),
                config.get("temperature"),
                config.get("max_tokens"),
                config.get("timeout"),
                config.get("top_p"),
                encrypted,
                to_json(dict(config.get("extra") or {})),
                utcnow().isoformat(),
            ),
        )

    def get(self, *, reveal: bool = False) -> Optional[Dict[str, Any]]:
        row = self._db.query_one("SELECT * FROM llm_configs WHERE id = 1")
        if not row:
            return None
        saved: Dict[str, Any] = {
            "provider": row["provider"],
            "model": row["model"],
            "base_url": row["base_url"],
            "temperature": row["temperature"],
            "max_tokens": row["max_tokens"],
            "timeout": row["timeout"],
            "top_p": row["top_p"],
            "extra": from_json(row["extra_json"]) or {},
            "credential_configured": bool(row["api_key_enc"]),
        }
        if reveal and row["api_key_enc"]:
            saved["api_key"] = self._box.decrypt(row["api_key_enc"])
        return saved

    def clear_key(self) -> None:
        self._db.execute("UPDATE llm_configs SET api_key_enc = NULL WHERE id = 1")


class Storage:
    """
    Bundles the repositories and the database behind one object.

    The application holds one :class:`Storage`; the API's dependency layer and the CLI's
    command functions reach through it for whichever repository they need. Constructing it
    is what creates the data directory and the schema, so wherever the app first touches
    storage, the database exists.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        path = self.settings.sqlite_path
        if path is None:
            # A non-SQLite DATABASE_URL is a configuration the storage layer cannot honour.
            # Defaulting to an in-memory database instead would "work" and then lose every
            # project on restart, which is a far worse failure than refusing to start.
            raise ConfigurationError(
                "Only SQLite databases are supported, and DATABASE_URL is set to something "
                "else.",
                action=(
                    "Set DATABASE_URL to a sqlite path like "
                    "`sqlite:///./data/magen.db`, or unset it to use the default."
                ),
                detail=f"DATABASE_URL={self.settings.database_url!r}",
            )
        self.db = Database(path)
        self.secretbox = SecretBox(self.settings.data_dir)
        self.projects = ProjectRepository(self.db)
        self.runs = RunRepository(self.db)
        self.llm_config = LLMConfigRepository(self.db, self.secretbox)

    def saved_llm(self, *, reveal: bool = False) -> Optional[Dict[str, Any]]:
        """Convenience for the common 'load saved provider settings' call."""
        return self.llm_config.get(reveal=reveal)

    def close(self) -> None:
        self.db.close()


# ------------------------------------------------------------------------------ mappers
def _project_row(row, *, include_result: bool = True) -> Dict[str, Any]:
    record = {
        "id": row["id"],
        "name": row["name"],
        "requirement": row["requirement"],
        "framework": row["framework"],
        "provider": row["provider"],
        "model": row["model"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if include_result:
        record["result"] = from_json(row["result_json"])
    return record


def _run_row(row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "kind": row["kind"],
        "status": row["status"],
        "requirement": row["requirement"],
        "framework": row["framework"],
        "provider": row["provider"],
        "model": row["model"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "duration_s": row["duration_s"],
        "succeeded": bool(row["succeeded"]) if row["succeeded"] is not None else None,
        "result": from_json(row["result_json"]),
        "error": from_json(row["error_json"]),
    }


def _log_row(row) -> Dict[str, Any]:
    return {
        "stage": row["stage"],
        "status": row["status"],
        "level": row["level"],
        "message": row["message"],
        "iteration": row["iteration"],
        "duration_s": row["duration_s"],
        "timestamp": row["timestamp"],
        "data": from_json(row["data_json"]) or {},
    }


def _default_name(requirement: str) -> str:
    """A readable project name from the first words of the requirement."""
    text = " ".join((requirement or "").split())
    if not text:
        return "Untitled project"
    return text[:57].rstrip() + "..." if len(text) > 60 else text
