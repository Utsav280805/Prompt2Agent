# multi_agent_generator/execution/runner.py
"""
Running generated code: its test suite, and the agent itself.

Two entry points, and the difference between them is the whole design.

:func:`run_project_tests` runs the test suite that ships with every generated project. That
suite is offline by construction - its ``conftest.py`` blocks outbound sockets and sets
placeholder credentials - so it needs no key and cannot cost money, which is what makes it
cheap enough to run on every pipeline iteration. It is the check the acceptance criteria depend
on, so it has to be honest.

:func:`run_agent` executes the agent for real, from the Playground. This one *does* call a
provider, so it costs money and needs a key, and it is never run as part of generation - only
when a person asks for it.

Three guarantees hold for both, and each one exists because of a specific failure:

**Nothing is installed into this process's environment.** A generated ``requirements.txt`` is
model output. Installing it with the server's interpreter would let that output change the
packages the server runs on. :mod:`.deps` installs into a content-keyed directory instead and
puts it on the child's import path.

**One credential crosses the boundary, and nothing comes back.** The child environment is
built from an allowlist and holds at most the single key the generated project's own provider
needs. On the way out, captured stdout and stderr are scrubbed of that key's literal value,
because provider SDKs put credentials in their exception messages and those messages reach the
Playground.

**Neither call can hang the caller.** A wall-clock timeout enforced by a process-tree kill,
with stdin supplied rather than left open, so an ``input()`` in generated code fails instead of
waiting forever.
"""
from __future__ import annotations

import json
import os
import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import GeneratedProject, RunResult, RunStatus, TestReport
from ..core.validation import RUN_ENTRY_NAMES
from ..errors import AppError, ExecutionError, scrub_values
from ..providers import get_provider
from ..settings import get_settings
from .deps import DependencySet, ensure_dependencies, python_path_for
from .process import ProcessResult, python_executable, run_process
from .pytest_report import parse_pytest_result
from .workspace import build_child_env, prepare_workspace

__all__ = ["run_project_tests", "run_agent"]


def run_project_tests(
    project: GeneratedProject,
    *,
    timeout: Optional[int] = None,
    workspace_dir: Optional[os.PathLike] = None,
    install: bool = True,
) -> TestReport:
    """
    Run a generated project's own test suite in a sandbox.

    Args:
        project: the project to verify.
        timeout: wall-clock bound. Defaults to ``TEST_EXECUTION_TIMEOUT``.
        workspace_dir: parent directory for the sandbox; the configured workspaces dir by
            default.
        install: install the project's requirements first. **On by default**, and that default
            is the point: the suite's first test imports every module, and a module that
            imports ``crewai`` cannot be checked without ``crewai`` present. Running with
            ``install=False`` would report ``ModuleNotFoundError`` for a perfectly good project
            and call it a test failure. The cost is paid once per requirement set, not once per
            run, because :func:`~.deps.ensure_dependencies` caches by content.

    Returns:
        A :class:`TestReport`. Never raises for a test failure, a timeout or a missing
        pytest - all three are results the pipeline must be able to report rather than
        crash on.
    """
    settings = get_settings()
    limit = timeout or settings.test_execution_timeout

    if not project.test_files():
        return TestReport(
            status="skipped",
            unavailable_reason="The project has no tests to run.",
        )

    workspace = prepare_workspace(project, parent_dir=workspace_dir or settings.workspaces_dir)
    try:
        deps = DependencySet()
        if install:
            deps = ensure_dependencies(workspace, timeout=settings.install_timeout)
            if not deps.ok:
                result = deps.result
                return TestReport(
                    status="error",
                    exit_code=result.exit_code if result else None,
                    duration_s=result.duration_s if result else 0.0,
                    stdout=result.stdout if result else "",
                    stderr=result.stderr if result else "",
                    unavailable_reason=(
                        "The project's dependencies could not be installed, so its tests "
                        "could not run. " + deps.reason
                    ),
                    notes=list(deps.rejected),
                )

        env = build_child_env(
            extra={
                # The project root, plus the dependency directory when there is one. Set
                # explicitly rather than relying on pytest's rootdir inference, which changes
                # depending on whether an ini file is present.
                "PYTHONPATH": python_path_for(workspace, deps),
            }
        )

        result = run_process(
            [
                python_executable(),
                "-m",
                "pytest",
                "tests",
                "-q",
                # Live tests hit a real provider. Excluding them here is not hiding a
                # failure - it is running the suite the way its own README says to, since
                # the sandbox has no credential to run them with.
                "-m",
                "not live",
                # Force a short timeout to fail fast on hanging tests during pipeline validation.
                # This overrides whatever timeout is in the project's pytest.ini.
                "--timeout=5",
                # Stop after enough failures to be informative. A project whose every test
                # fails identically should not spend the whole timeout proving it.
                "--maxfail=10",
                "-p",
                "no:cacheprovider",
                "--color=no",
            ],
            cwd=str(workspace.root),
            env=env,
            timeout=limit,
        )
        report = parse_pytest_result(result)
        # A partial refusal still installed something, so the suite genuinely ran - but if it
        # failed on an import, the refused line is the explanation. Carried as a note rather
        # than as `unavailable_reason`, which would wrongly mark a suite that ran as one that
        # did not.
        if deps.rejected:
            report.notes.extend(deps.rejected)
        return report
    finally:
        workspace.cleanup()


def run_agent(
    project: GeneratedProject,
    query: str,
    *,
    timeout: Optional[int] = None,
    workspace_dir: Optional[os.PathLike] = None,
    api_key: Optional[str] = None,
    install: bool = True,
) -> RunResult:
    """
    Execute a generated agent against a real query.

    This is the Playground path. It runs a small harness in a subprocess: the harness imports
    the project's workflow module, calls ``run_workflow(query)``, and prints the result as JSON
    inside a delimiter so the answer can be separated from framework chatter.

    Args:
        project: the project to run.
        query: the user's question, passed to the run function.
        timeout: wall-clock bound. Defaults to ``AGENT_EXECUTION_TIMEOUT``.
        api_key: the credential for the *project's* provider. Passed explicitly - the sandbox
            environment is scrubbed, so a key not passed here simply is not present, and the
            agent reports a missing-credential error rather than silently using the
            application's own key.
        install: install requirements first. Required for a real run, since the framework has
            to be importable; the install is cached by requirement content, so this is fast
            after the first project that uses a given framework.

    Returns:
        A :class:`RunResult` whose captured output has been scrubbed of ``api_key``. Never
        raises for a failing agent.
    """
    settings = get_settings()
    limit = timeout or settings.agent_execution_timeout

    # Migrate projects generated before the Hugging Face router change in the disposable
    # workspace. The saved project remains untouched, but an old CrewAI literal such as
    # ``huggingface/Qwen/Qwen2.5-7B-Instruct`` no longer sends the run through a retired route.
    project = _migrate_runtime_provider(project)
    target = _run_target(project)
    if target is None:
        return RunResult(
            status=RunStatus.FAILED,
            error=ExecutionError(
                "This project has no module that exposes a run function, so there is "
                "nothing to run.",
                action="Regenerate the project.",
            ).to_dict(),
        )
    entry_path, entry_function = target

    workspace = prepare_workspace(project, parent_dir=workspace_dir or settings.workspaces_dir)
    try:
        deps = DependencySet()
        if install:
            deps = ensure_dependencies(workspace, timeout=settings.install_timeout)
            if not deps.ok:
                result = deps.result
                return RunResult(
                    status=RunStatus.FAILED,
                    stdout=result.stdout if result else "",
                    stderr=result.stderr if result else "",
                    exit_code=result.exit_code if result else None,
                    duration_s=result.duration_s if result else 0.0,
                    error=ExecutionError(
                        "The project's dependencies could not be installed, so the agent "
                        "could not start.",
                        action=(
                            "Check requirements.txt and your network connection, then try "
                            "again. The pip output below names the package that failed."
                        ),
                        # The refused lines are part of the detail, not a separate field the
                        # Playground would have to know to look for. Without them a refusal
                        # reads as a mysterious install failure.
                        detail="\n".join([deps.reason, *deps.rejected]),
                    ).to_dict(),
                )

        credential_name = _credential_name(project.provider)
        harness = workspace.path("_run_agent.py")
        harness.write_text(
            _HARNESS.replace("__ENTRY_MODULE__", _module_name(entry_path)).replace(
                "__ENTRY_FUNCTION__", entry_function
            ),
            encoding="utf-8",
        )

        env = build_child_env(
            credential_name=credential_name,
            credential_value=api_key,
            extra={"PYTHONPATH": python_path_for(workspace, deps)},
        )

        result = run_process(
            [python_executable(), str(harness)],
            cwd=str(workspace.root),
            env=env,
            timeout=limit,
            stdin_data=json.dumps({"query": query}),
        )
        return _to_run_result(result, limit, secrets=[api_key])
    finally:
        workspace.cleanup()


# There is deliberately no ``install_requirements`` helper here any more. It returned only pip's
# ProcessResult and discarded the directory the packages went into, which is the one thing a
# caller actually needs - an install whose location is thrown away is an install the child
# cannot import from. :func:`~.deps.ensure_dependencies` is the whole interface.


# ------------------------------------------------------------------------------ helpers
#: Marker the harness wraps its JSON in. A delimiter rather than "the last line of stdout" is
#: what makes the answer recoverable from frameworks that print progress bars, banners and
#: tool traces around it.
_BEGIN = "<<<AGENT_OUTPUT_BEGIN>>>"
_END = "<<<AGENT_OUTPUT_END>>>"

#: Function names the harness falls back to, best first. ``run_workflow`` is the contract every
#: project this generator emits exposes; the rest are there so a hand-edited project, or one
#: from an older version, still runs. Normally the caller names the function explicitly, which
#: matters because a module can *import* a run function as well as define one.
#:
#: Imported rather than restated. ``core.validation`` gates a project on having one of these
#: functions before calling it ready, and this module is what actually goes looking for one - so
#: a second copy of the list here would eventually let a project pass the gate and then fail to
#: launch, with the two lists each insisting the other was wrong.
_RUN_NAMES = RUN_ENTRY_NAMES

_HARNESS = f'''\
"""Sandbox harness. Written at run time; not part of the project."""
import importlib
import json
import sys
import time
import traceback

BEGIN = "{_BEGIN}"
END = "{_END}"
MODULE = "__ENTRY_MODULE__"
FUNCTION = "__ENTRY_FUNCTION__"
RUN_NAMES = {_RUN_NAMES!r}
LIMIT = 4000


def _emit(payload):
    print(BEGIN)
    print(json.dumps(payload, default=str))
    print(END)


def _find_callable(module):
    """
    The function to call, preferring the one the caller named.

    The explicit name matters: a runtime module both defines `run` and imports `run_workflow`
    from the workflow module it wraps, so a search by name alone would call the inner function
    and skip the error handling the wrapper exists to provide.
    """
    if FUNCTION:
        fn = getattr(module, FUNCTION, None)
        if callable(fn):
            return FUNCTION, fn
    for name in RUN_NAMES:
        fn = getattr(module, name, None)
        if callable(fn):
            return name, fn
    return None, None


def _as_mapping(result):
    """
    The result as a plain dict, or None if it is not one.

    Handles both shapes a generated project returns: the `run_workflow` contract returns a
    dict directly, and the runtime wrapper returns a small result object that knows how to
    flatten itself. Anything else is not forced into this shape - it is reported as the plain
    value it is.
    """
    if isinstance(result, dict):
        return dict(result)
    flatten = getattr(result, "as_dict", None)
    if callable(flatten):
        try:
            data = flatten()
        except Exception:
            return None
        if isinstance(data, dict):
            return data
    return None


def _detail_of(error):
    if isinstance(error, dict):
        return str(error.get("detail") or error.get("message") or error)
    return str(error)


def _message_of(error):
    if isinstance(error, dict):
        return str(error.get("message") or "The agent reported an error.")
    return str(error)


def main():
    try:
        payload = json.loads(sys.stdin.read() or "{{}}")
    except Exception:
        payload = {{}}
    query = payload.get("query") or ""

    try:
        module = importlib.import_module(MODULE)
    except Exception:
        _emit({{"error": "The agent module could not be imported.",
               "detail": traceback.format_exc()[-LIMIT:]}})
        return 1

    name, fn = _find_callable(module)
    if fn is None:
        _emit({{"error": "The agent module defines no run function.",
               "detail": "Looked for: " + ", ".join(RUN_NAMES)}})
        return 1

    started = time.monotonic()
    try:
        try:
            result = fn(query)
        except TypeError as exc:
            # Only retry with no argument when the signature is what rejected the call. A
            # TypeError raised *inside* the agent must surface as the failure it is, not be
            # retried and reported as a different one.
            if "argument" not in str(exc) and "positional" not in str(exc):
                raise
            result = fn()
    except Exception:
        _emit({{"error": "The agent raised an exception.",
               "detail": traceback.format_exc()[-LIMIT:],
               "entry": name,
               "duration_s": round(time.monotonic() - started, 3)}})
        return 1

    elapsed = round(time.monotonic() - started, 3)
    data = _as_mapping(result)

    if data is None:
        # A project that returns a bare string still produces an answer. The absence of step
        # detail is stated rather than filled in with something plausible.
        _emit({{"output": "" if result is None else str(result),
               "steps": [],
               "tool_calls": [],
               "duration_s": elapsed,
               "entry": name,
               "note": ("The run function returned a " + type(result).__name__ + " rather "
                        "than a result mapping, so no step or tool detail is available.")}})
        return 0

    error = data.get("error")
    if error:
        # The project reported its own failure through the result rather than by raising -
        # which is what the generated runtime wrapper does deliberately. Reported as a failure,
        # because a described error is still an error.
        _emit({{"error": _message_of(error),
               "detail": _detail_of(error)[-LIMIT:],
               "entry": name,
               "duration_s": data.get("duration_s") or elapsed}})
        return 1

    steps = data.get("steps")
    calls = data.get("tool_calls")
    _emit({{"output": str(data.get("output", "") or ""),
           "steps": steps if isinstance(steps, list) else [],
           "tool_calls": calls if isinstance(calls, list) else [],
           "duration_s": (data.get("duration_s")
                          if isinstance(data.get("duration_s"), (int, float))
                          else elapsed),
           "entry": name}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _run_target(project: GeneratedProject) -> Optional[Tuple[str, str]]:
    """
    Which module the harness imports, and which function it calls.

    Preference order, and each step has a reason:

    1. The runtime wrapper, when the project has one. It calls the workflow with timing and
       error handling and returns a described failure instead of a traceback, which is exactly
       what the Playground needs to show a non-technical user.
    2. The workflow module. This is where ``run_workflow`` is defined, and importing the module
       that defines a function beats importing one that merely calls it.
    3. The declared entry point, then the conventional flat-layout filenames - so a
       hand-edited or older project still runs rather than being refused.

    The function name is returned alongside the path because a module can import a run
    function as well as define one, and calling the imported inner function would skip the
    wrapper.
    """
    for file in project.source_files():
        if file.path.endswith("runtime/runner.py") or file.path.endswith("runtime\\runner.py"):
            return file.path, "run"

    for file in project.source_files():
        if (
            "workflow" in file.path
            and file.path.endswith(".py")
            and not file.path.endswith("/__init__.py")
            and not file.path.endswith("\\__init__.py")
        ):
            return file.path, "run_workflow"

    entry = project.entrypoint_file
    if entry is not None and entry.path != "main.py":
        return entry.path, ""

    for candidate in ("agent.py", "main.py"):
        if project.get(candidate) is not None:
            return candidate, "run_workflow" if candidate == "agent.py" else ""

    return (entry.path, "") if entry is not None else None


def _module_name(path: str) -> str:
    """agent.py -> agent; pkg/agent.py -> pkg.agent."""
    stem = path[:-3] if path.endswith(".py") else path
    return stem.replace("/", ".").replace("\\", ".")


def _credential_name(provider: str) -> Optional[str]:
    """The single env var the generated project's provider reads, if it needs one."""
    try:
        spec = get_provider(provider)
    except (ValueError, AppError):
        return None
    return spec.credential_env[0] if spec.credential_env else None


def _migrate_runtime_provider(project: GeneratedProject) -> GeneratedProject:
    """Make legacy hosted Hugging Face projects use the current CrewAI router."""
    if project.provider != "huggingface":
        return project

    from ..providers import HF_ROUTER_BASE_URL, get_provider

    model = os.environ.get("HF_MODEL_ID") or get_provider("huggingface").runtime_model
    legacy_prefix = "huggingface/"
    legacy_model = f"{legacy_prefix}Qwen/Qwen2.5-7B-Instruct"
    migrated = copy.deepcopy(project)
    for generated_file in migrated.files:
        if legacy_model not in generated_file.content:
            continue
        content = generated_file.content.replace(legacy_model, f"openai/{model}")
        if "base_url=" not in content and "from crewai import LLM" in content:
            content = re.sub(
                r'(api_key\s*=\s*HF_TOKEN,\n)',
                rf'\1    base_url="{HF_ROUTER_BASE_URL}",\n',
                content,
                count=1,
            )
        generated_file.content = content
    return migrated


def _to_run_result(
    result: ProcessResult,
    limit: int,
    *,
    secrets: Optional[List[Optional[str]]] = None,
) -> RunResult:
    """
    Turn a finished harness process into a :class:`RunResult`.

    Captured output is scrubbed of the credential that was handed to the child before it is put
    anywhere a caller can see it. This is the only place that can do it - the API layer that
    serialises the result no longer knows which key was used - and it has to be done, because a
    provider SDK raising ``AuthenticationError: ... key sk-live-...`` writes the credential to
    stderr, and stderr is shown in the Playground.
    """
    scrub = list(secrets or [])
    stdout = scrub_values(result.stdout, scrub)
    stderr = scrub_values(result.stderr, scrub)

    if result.timed_out:
        return RunResult(
            status=RunStatus.TIMED_OUT,
            stdout=stdout,
            stderr=stderr,
            exit_code=result.exit_code,
            timed_out=True,
            duration_s=result.duration_s,
            error={
                "code": "execution_timeout",
                "message": f"The agent did not finish within {limit} seconds.",
                "action": (
                    "Raise AGENT_EXECUTION_TIMEOUT, or check the output below for a hang."
                ),
            },
        )

    if result.start_error is not None:
        return RunResult(
            status=RunStatus.FAILED,
            duration_s=result.duration_s,
            error={
                "code": "execution_failed",
                "message": "The agent process could not be started.",
                "detail": scrub_values(result.start_error, scrub),
            },
        )

    combined = f"{stdout}\n{stderr}".lower()
    if "model_not_supported" in combined or "not supported by any provider" in combined:
        return RunResult(
            status=RunStatus.FAILED,
            stdout=stdout,
            stderr=stderr,
            exit_code=result.exit_code,
            duration_s=result.duration_s,
            error={
                "code": "model_not_supported",
                "message": "The selected Hugging Face model is not enabled for this account.",
                "action": (
                    "Set HF_MODEL_ID to a model enabled by your Hugging Face token, then "
                    "run the query again. The generated project now uses the Hugging Face "
                    "OpenAI-compatible router."
                ),
                "detail": (stderr or stdout).strip()[-2000:] or None,
            },
        )

    payload = _extract_payload(stdout)
    if payload is None:
        # No delimiter at all means the harness never got far enough to print one - the
        # interpreter died first. The captured stderr is the only real evidence, so it is
        # surfaced rather than replaced with a generic message.
        return RunResult(
            status=RunStatus.FAILED if result.exit_code != 0 else RunStatus.SUCCEEDED,
            stdout=stdout,
            stderr=stderr,
            exit_code=result.exit_code,
            duration_s=result.duration_s,
            output=stdout.strip() or None,
            error=(
                None
                if result.exit_code == 0
                else {
                    "code": "execution_failed",
                    "message": "The agent exited without producing an answer.",
                    "detail": (stderr or stdout).strip()[-2000:] or None,
                }
            ),
        )

    if "error" in payload:
        return RunResult(
            status=RunStatus.FAILED,
            stdout=stdout,
            stderr=stderr,
            exit_code=result.exit_code,
            duration_s=result.duration_s,
            entry=_opt_str(payload.get("entry")),
            error={
                "code": "execution_failed",
                "message": str(payload.get("error")),
                "detail": _opt_str(payload.get("detail")),
            },
        )

    steps = payload.get("steps")
    calls = payload.get("tool_calls")
    return RunResult(
        status=RunStatus.SUCCEEDED,
        stdout=stdout,
        stderr=stderr,
        exit_code=result.exit_code,
        # The harness's own measurement when it reported one: it timed the agent call, whereas
        # the process duration also includes interpreter startup and framework import, which
        # can be several seconds and would make every run look slower than it was.
        duration_s=_number(payload.get("duration_s"), result.duration_s),
        output=str(payload.get("output") or ""),
        steps=[s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else [],
        tool_calls=[c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else [],
        entry=_opt_str(payload.get("entry")),
    )


def _opt_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _number(value: Any, fallback: float) -> float:
    return float(value) if isinstance(value, (int, float)) else float(fallback)


def _extract_payload(stdout: str) -> Optional[Dict[str, Any]]:
    """Pull the harness's JSON out from between the delimiters."""
    if not stdout or _BEGIN not in stdout or _END not in stdout:
        return None
    body = stdout.rsplit(_BEGIN, 1)[1].split(_END, 1)[0].strip()
    try:
        loaded = json.loads(body)
    except (ValueError, TypeError):
        return None
    return loaded if isinstance(loaded, dict) else None
