# multi_agent_generator/storage/__init__.py
"""
Persistence.

A generation run is expensive - it costs model calls, a review, an improvement loop and a
test execution - so losing the result when a browser tab closes is not acceptable. This
package is what makes a project a *thing that exists* rather than a transient response:
projects, their runs, the timeline of stage events inside each run, and the saved LLM
settings all live in a SQLite file under the data directory.

The layering is deliberately strict:

- :mod:`~multi_agent_generator.storage.database` owns the connection and the schema, and
  nothing else in the codebase imports :mod:`sqlite3`.
- :mod:`~multi_agent_generator.storage.repository` owns the SQL, translating between rows and
  the domain objects the rest of the app uses.
- :mod:`~multi_agent_generator.storage.secretbox` owns credential encryption, and enforces
  the rule that a saved API key is either encrypted or not stored at all.

Callers normally want just one thing::

    from multi_agent_generator.storage import Storage

    storage = Storage()
    project_id = storage.projects.create(requirement="Research assistant")

Constructing :class:`Storage` creates the data directory and the schema, so there is no
separate "initialise the database" step to forget.
"""
from __future__ import annotations

from .database import SCHEMA_VERSION, Database, from_json, to_json
from .repository import (
    LLMConfigRepository,
    ProjectRepository,
    RunRepository,
    Storage,
    new_id,
)
from .secretbox import SecretBox, encryption_available

__all__ = [
    "Storage",
    "ProjectRepository",
    "RunRepository",
    "LLMConfigRepository",
    "Database",
    "SCHEMA_VERSION",
    "SecretBox",
    "encryption_available",
    "to_json",
    "from_json",
    "new_id",
]
