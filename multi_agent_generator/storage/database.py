# multi_agent_generator/storage/database.py
"""
The SQLite connection and schema.

This layer uses the standard library's :mod:`sqlite3` directly rather than an ORM. That is a
deliberate choice for a local-first tool: an ORM would add a heavyweight dependency and a
migration framework to store perhaps a thousand rows, and the schema here is small enough
that hand-written SQL is clearer than the object mapping that would hide it. The cost is that
schema changes are manual, which :func:`_migrate` handles by version number.

Two decisions worth stating:

**WAL mode and a per-thread connection.** The API serves requests on a thread pool and a run
executes on a background thread, so the database is touched concurrently. SQLite handles that
only in WAL journal mode and only if connections are not shared across threads, so each
thread gets its own connection via thread-local storage. Sharing one connection is the single
most common way a SQLite-backed web app corrupts itself under load.

**Foreign keys on, with cascade.** A run belongs to a project and a log line belongs to a
run; deleting a project should take its runs and their logs with it. SQLite disables foreign
keys by default, so it is turned on per connection - forgetting to would let the cascade
rules in the schema silently do nothing.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

__all__ = ["Database", "SCHEMA_VERSION"]


#: Bump when the schema changes; :func:`_migrate` compares against ``PRAGMA user_version``.
SCHEMA_VERSION = 1


_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    requirement   TEXT NOT NULL,
    framework     TEXT,
    provider      TEXT,
    model         TEXT,
    status        TEXT NOT NULL DEFAULT 'draft',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    -- The full generated project and result, as JSON. Stored rather than reconstructed so
    -- the Files tab and review can be shown without re-running the pipeline.
    result_json   TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    project_id    TEXT REFERENCES projects(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL DEFAULT 'generation',  -- generation | test | agent
    status        TEXT NOT NULL DEFAULT 'pending',
    requirement   TEXT,
    framework     TEXT,
    provider      TEXT,
    model         TEXT,
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    duration_s    REAL,
    succeeded     INTEGER,
    result_json   TEXT,
    error_json    TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT REFERENCES runs(id) ON DELETE CASCADE,
    project_id    TEXT,
    stage         TEXT,
    status        TEXT,
    level         TEXT,
    message       TEXT,
    iteration     INTEGER,
    duration_s    REAL,
    timestamp     TEXT NOT NULL,
    data_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_logs_run ON logs(run_id);

CREATE TABLE IF NOT EXISTS llm_configs (
    id             INTEGER PRIMARY KEY CHECK (id = 1),  -- single-row table
    provider       TEXT NOT NULL,
    model          TEXT,
    base_url       TEXT,
    temperature    REAL,
    max_tokens     INTEGER,
    timeout        INTEGER,
    top_p          REAL,
    -- Encrypted by SecretBox before it reaches here. The schema cannot enforce that, so the
    -- repository is the only thing that writes this column and it always encrypts first.
    api_key_enc    TEXT,
    extra_json     TEXT,
    updated_at     TEXT NOT NULL
);
"""


class Database:
    """
    A thread-safe handle to one SQLite database.

    Construct once and share the instance; it hands out a per-thread connection internally,
    so the instance itself is safe to keep on the application object.
    """

    def __init__(self, path: Optional[Path | str]) -> None:
        # ":memory:" is honoured for tests, but note each thread then gets its *own* memory
        # database - fine for single-threaded tests, which is the only place it is used.
        self._path = str(path) if path is not None else ":memory:"
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialised = False

    @property
    def path(self) -> str:
        return self._path

    def connect(self) -> sqlite3.Connection:
        """Return this thread's connection, creating and initialising it on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            self._path,
            timeout=30.0,  # wait for a competing writer rather than raising "database is locked"
            check_same_thread=True,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if self._path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")

        # Initialise the schema *before* caching the connection. If the DDL fails - a full
        # disk, a corrupt file - the exception propagates and this thread is left with no
        # cached connection, so a retry runs the initialisation again. Caching first would
        # mean the retry took the early-return path above and quietly handed back a
        # connection to a database with no tables in it.
        try:
            self._ensure_schema(conn)
        except Exception:
            conn.close()
            raise

        self._local.conn = conn
        return conn

    # ------------------------------------------------------------------------ execution
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        conn = self.connect()
        cursor = conn.execute(sql, tuple(params))
        conn.commit()
        return cursor

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        return self.connect().execute(sql, tuple(params)).fetchone()

    def query_all(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.connect().execute(sql, tuple(params)).fetchall())

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -------------------------------------------------------------------------- schema
    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        # Guard with a lock and a flag: two threads opening their first connection at once
        # would otherwise both run the migration. The DDL is idempotent, but the version
        # bump is not, and racing it is how "already exists" errors appear under load.
        #
        # In-memory databases are the exception. Each connection gets its *own* private
        # database, so the "already initialised" flag would be a lie for every thread after
        # the first, handing it an empty schema. There is no shared file to race over, so
        # each connection simply initialises itself.
        if self._path == ":memory:":
            conn.executescript(_SCHEMA)
            _migrate(conn)
            conn.commit()
            return

        with self._init_lock:
            if self._initialised:
                return
            conn.executescript(_SCHEMA)
            _migrate(conn)
            conn.commit()
            self._initialised = True


def _migrate(conn: sqlite3.Connection) -> None:
    """
    Bring an existing database up to :data:`SCHEMA_VERSION`.

    Version 1 is the initial schema, created idempotently by the ``CREATE TABLE IF NOT
    EXISTS`` statements above, so there is nothing to migrate yet. The scaffolding is here so
    that the *first* schema change has an obvious home and does not require retrofitting a
    versioning scheme onto databases already in the wild.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= SCHEMA_VERSION:
        return
    # Future migrations go here, guarded by `if current < N:`.
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def to_json(value: Any) -> Optional[str]:
    """Serialise a value for a ``*_json`` column, tolerating anything non-serialisable."""
    if value is None:
        return None
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def from_json(text: Optional[str]) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None
