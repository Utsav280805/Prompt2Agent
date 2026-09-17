# multi_agent_generator/execution/process.py
"""
Running a child process that cannot outlive its timeout.

This module exists because of one specific bug class. The old test suite hung for two
minutes on a single test, and the reason it *could* hang is that nothing in the system owned
a wall-clock bound on the work it started. A timeout on the parent's own call is not enough:
if the child has already spawned grandchildren, killing the child leaves them running, still
holding the pipes open, and the parent blocks forever on a read that will never end.

So the contract here is deliberately absolute: :func:`run_process` returns within
``timeout`` plus a small grace period, always, and when it returns nothing it started is
still alive. Everything else in the execution layer is built on that guarantee, which is why
this is the only place in the codebase allowed to call :mod:`subprocess`.

The kill is a *tree* kill, escalating from polite to forced:

1. SIGTERM to the whole process group (POSIX) or ``taskkill /T`` (Windows), giving the child
   a chance to flush output and clean up;
2. after a short grace period, SIGKILL to the group, which nothing can catch or ignore.

Grandchildren are the reason for the group. A pytest run spawns workers; an agent script
spawns an inference server. Sending a signal to one pid and calling it done is how "we added
a timeout" turns into "it still hangs".
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

__all__ = ["ProcessResult", "run_process"]


_IS_WINDOWS = os.name == "nt"

#: How long a terminated process gets to exit before it is killed outright.
_GRACE_SECONDS = 3.0

#: Cap on captured output per stream. Generated code in a loop can emit megabytes, and
#: holding all of it costs memory in the API process for no benefit - the tail is what
#: diagnoses a failure.
_MAX_CAPTURE = 400_000


@dataclass
class ProcessResult:
    """What a finished (or killed) child process produced."""

    command: List[str]
    exit_code: Optional[int]
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    #: Set when the process could not be started at all (missing interpreter, bad cwd).
    start_error: Optional[str] = None
    cwd: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.start_error is None

    @property
    def combined_output(self) -> str:
        """stdout and stderr in one blob, for parsers that do not care which is which."""
        if not self.stderr:
            return self.stdout
        if not self.stdout:
            return self.stderr
        return f"{self.stdout}\n{self.stderr}"


def run_process(
    command: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: float = 120.0,
    stdin_data: Optional[str] = None,
) -> ProcessResult:
    """
    Run ``command`` and return within ``timeout`` no matter what it does.

    Args:
        command: argv list. Never a shell string - a list means no shell, and no shell means
            no quoting bug can turn a generated filename into a command.
        cwd: working directory. Should always be the sandbox workspace.
        env: the complete environment for the child. Passed rather than inherited, so the
            caller decides exactly what the child can see.
        timeout: wall-clock bound in seconds.
        stdin_data: text to write to stdin, if the child expects input. Stdin is otherwise
            closed, which is what stops an interactive ``input()`` in generated code from
            hanging the run forever waiting for a keystroke nobody will type.

    Returns:
        A :class:`ProcessResult`. Never raises for a process-level failure - a crash, a
        timeout and a missing interpreter are all normal outcomes here and are reported in
        the result so the caller can explain them.
    """
    argv = [str(part) for part in command]
    started = time.monotonic()

    popen_kwargs: Dict[str, object] = {
        "cwd": cwd,
        "env": dict(env) if env is not None else None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        # Generated code prints whatever a model produced, which routinely contains
        # characters the console encoding cannot represent. Replacing them keeps a UnicodeError
        # in the *parent* from destroying an otherwise successful run.
        "errors": "replace",
    }

    if _IS_WINDOWS:
        # A new process group is what makes taskkill /T able to find the descendants.
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        # setsid: the child becomes a session leader, so its pid is also its process-group
        # id and one killpg reaches every descendant.
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(argv, **popen_kwargs)  # noqa: S603 - argv list, no shell
    except (OSError, ValueError) as exc:
        return ProcessResult(
            command=argv,
            exit_code=None,
            duration_s=time.monotonic() - started,
            start_error=f"{type(exc).__name__}: {exc}",
            cwd=cwd,
        )

    timed_out = False
    # A separate watchdog is intentional. On Windows, generated frameworks can leave a
    # descendant holding stdout/stderr open, preventing communicate() from reaching its own
    # timeout path. The watchdog owns the hard deadline and kills the whole process tree.
    watchdog_fired = threading.Event()
    watchdog = threading.Timer(timeout, _watchdog_kill, args=(process, watchdog_fired))
    watchdog.daemon = True
    watchdog.start()
    try:
        stdout, stderr = process.communicate(
            input=stdin_data, timeout=timeout + _GRACE_SECONDS * 2
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(process)
        # Collect whatever was written before the kill. The tree is dead by now, so the
        # pipes are closed and this cannot block indefinitely - but it is still bounded,
        # because "cannot block" is an assumption and the whole point of this module is not
        # to rely on one.
        try:
            stdout, stderr = process.communicate(timeout=_GRACE_SECONDS * 2)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
    except (OSError, ValueError) as exc:
        _kill_tree(process)
        return ProcessResult(
            command=argv,
            exit_code=None,
            duration_s=time.monotonic() - started,
            start_error=f"{type(exc).__name__}: {exc}",
            cwd=cwd,
        )

    timed_out = timed_out or watchdog_fired.is_set()
    if watchdog.is_alive():
        watchdog.cancel()

    return ProcessResult(
        command=argv,
        exit_code=process.returncode,
        stdout=_clip(stdout),
        stderr=_clip(stderr),
        duration_s=time.monotonic() - started,
        timed_out=timed_out,
        cwd=cwd,
    )


def _clip(text: Optional[str]) -> str:
    if not text:
        return ""
    if len(text) <= _MAX_CAPTURE:
        return text
    dropped = len(text) - _MAX_CAPTURE
    return f"... [{dropped} earlier characters omitted] ...\n{text[-_MAX_CAPTURE:]}"


def _watchdog_kill(process: "subprocess.Popen", fired: threading.Event) -> None:
    """Enforce the deadline even when descendant pipes confuse communicate()."""
    fired.set()
    if process.poll() is None:
        _kill_tree(process)


def _kill_tree(process: "subprocess.Popen") -> None:
    """
    Kill the process and everything it started.

    Escalates rather than going straight to SIGKILL: a terminated pytest run flushes its
    summary, which is often the only clue about *where* it hung. Every step is wrapped
    because the process may exit between one call and the next, and a race here must not
    replace a useful timeout report with an unrelated exception.
    """
    if process.poll() is not None:
        return

    if _IS_WINDOWS:
        _kill_tree_windows(process)
    else:
        _kill_tree_posix(process)

    try:
        process.wait(timeout=_GRACE_SECONDS)
    except Exception:  # noqa: BLE001
        pass


def _kill_tree_posix(process: "subprocess.Popen") -> None:
    try:
        group = os.getpgid(process.pid)
    except (ProcessLookupError, OSError):
        group = None

    for sig, wait in ((signal.SIGTERM, _GRACE_SECONDS), (signal.SIGKILL, 0.0)):
        try:
            if group is not None:
                os.killpg(group, sig)
            else:
                process.send_signal(sig)
        except (ProcessLookupError, OSError):
            return
        if wait:
            try:
                process.wait(timeout=wait)
                return  # exited politely; no need to escalate
            except subprocess.TimeoutExpired:
                continue


def _kill_tree_windows(process: "subprocess.Popen") -> None:
    """
    Windows has no process groups we can signal, so shell out to taskkill.

    ``/T`` is the important flag - it takes the descendants too. If taskkill is unavailable
    or refuses, we fall back to killing the direct child, which is worse than nothing only
    in the sense that a grandchild may survive; leaving the child alive as well would be
    strictly worse.
    """
    try:
        subprocess.run(  # noqa: S603,S607 - fixed argv, no user input
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            timeout=_GRACE_SECONDS * 2,
            check=False,
        )
    except Exception:  # noqa: BLE001 - taskkill missing or blocked
        pass
    if process.poll() is None:
        try:
            process.kill()
        except Exception:  # noqa: BLE001
            pass


def python_executable() -> str:
    """
    The interpreter to run generated code with.

    ``sys.executable`` rather than a bare ``"python"``: the sandbox must use the same
    interpreter the application is running under, or a project generated on Python 3.13
    quietly gets tested against whatever ``python`` happens to mean on the user's PATH.
    """
    return sys.executable or "python"
