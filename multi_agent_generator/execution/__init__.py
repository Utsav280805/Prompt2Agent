# multi_agent_generator/execution/__init__.py
"""
The execution sandbox.

Generated code is untrusted code. It was written by a language model from a prompt a user
typed, and it may loop forever, spawn subprocesses, read the filesystem or try to print its
environment. This package is the boundary that makes running it acceptable anyway.

Five guarantees, each implemented in exactly one place:

- **A separate workspace.** Every run gets a fresh directory, written by
  :mod:`~multi_agent_generator.execution.workspace`, which re-validates every path against
  the workspace root before a byte is written.
- **A scrubbed environment.** The child sees an allowlisted environment plus at most one
  credential - the one its own provider needs. The application's other secrets are not in
  the dictionary at all, so no amount of environment-dumping in generated code can reveal
  them. On the way back, captured output is scrubbed of that credential's literal value.
- **A private import path.** :mod:`~multi_agent_generator.execution.deps` installs a generated
  project's requirements into a content-keyed directory and puts it on the child's
  ``PYTHONPATH``. Nothing is ever installed into the interpreter running this application, so
  a model-authored ``requirements.txt`` cannot change the packages the server depends on.
- **A hard wall-clock bound.** :mod:`~multi_agent_generator.execution.process` owns the only
  ``subprocess`` calls in the codebase and kills the whole process *tree* at the timeout, so
  a grandchild cannot outlive the run that started it.
- **An honest verdict.** :mod:`~multi_agent_generator.execution.pytest_report` distinguishes
  "passed", "failed" and "never actually ran". A suite that could not be collected is never
  reported as green.

What this is not: a container. The child runs as the same OS user with the same filesystem
permissions, and pip executes the build code of whatever it installs. For a deployment that
accepts requirements from untrusted users, the boundary needs to be a container or a VM, and
that belongs at the deployment level rather than in this package. Stating the limit is more
useful than implying a guarantee this does not give.

Public surface::

    from multi_agent_generator.execution import run_project_tests, run_agent

Both return structured results and neither raises for a failure of the code being run - a
crashing agent is data, not an exception.
"""
from __future__ import annotations

from .deps import (
    DependencySet,
    cache_entries,
    clear_dependency_cache,
    ensure_dependencies,
    python_path_for,
)
from .process import ProcessResult, python_executable, run_process
from .pytest_report import parse_pytest_result
from .runner import run_agent, run_project_tests
from .workspace import Workspace, build_child_env, prepare_workspace

__all__ = [
    "run_project_tests",
    "run_agent",
    "ensure_dependencies",
    "python_path_for",
    "cache_entries",
    "clear_dependency_cache",
    "DependencySet",
    "prepare_workspace",
    "build_child_env",
    "Workspace",
    "run_process",
    "python_executable",
    "ProcessResult",
    "parse_pytest_result",
]
