# backend/services/__init__.py
"""
Orchestration that is genuinely web-specific.

The rule for this package: something belongs here only if the CLI would have no use for it.
Running a pipeline in a background thread so an HTTP request can return early, and packing a
project into a zip for a browser download, are both in that category. Framework selection,
review thresholds and generation are not - those live in ``multi_agent_generator.core`` and
are shared.
"""
from __future__ import annotations

from .export import project_zip_bytes, suggest_filename
from .jobs import Job, JobManager, get_job_manager

__all__ = [
    "Job",
    "JobManager",
    "get_job_manager",
    "project_zip_bytes",
    "suggest_filename",
]
