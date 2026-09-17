# multi_agent_generator/execution/deps.py
"""
Installing a generated project's dependencies without touching the application's own.

The problem this solves is specific. A generated project ships a ``requirements.txt`` whose
contents were decided by a language model. Running ``pip install -r`` on that file with the
server's interpreter would let model output change the packages the server itself is running
on - a pinned ``pydantic==1.10`` in a generated project would break the running application,
and a hostile one could do considerably worse. So nothing here ever installs into the current
environment.

Instead each distinct ``requirements.txt`` gets its own directory, installed with pip's
``--target``, and that directory is put on the child's ``PYTHONPATH``. The application's
``site-packages`` is left alone.

The directory is keyed by a hash of the requirements content and kept, which is the other half
of the design. A fresh install per run would mean downloading CrewAI on every pipeline
iteration - minutes each time, and the reason the naive implementation installed into the
shared environment in the first place. Keying by content means the second project that needs
the same framework finds it already there, and a project that changes its requirements gets a
new directory rather than a half-upgraded old one.

What this is and is not
-----------------------
It is process isolation with a private import path, a scrubbed environment, a wall-clock
timeout and a process-tree kill. It is *not* a container: the child can still read the
filesystem it has user permissions for, and pip runs the setup code of whatever it installs.
For a deployment that runs untrusted requirements, the execution layer needs a container or a
VM boundary, and that belongs at the deployment level rather than here. Saying so plainly
matters more than implying a guarantee this does not give.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..dependencies import screen_requirements
from ..logging_config import get_logger
from ..settings import get_settings
from .process import ProcessResult, python_executable, run_process
from .workspace import Workspace, build_child_env

__all__ = [
    "DependencySet",
    "ensure_dependencies",
    "python_path_for",
    "clear_dependency_cache",
    "cache_entries",
]

LOGGER = get_logger("execution.deps")

#: Written into a completed cache directory. Its presence is the only thing that marks an
#: install as finished - a directory that exists without it is a crashed or killed install and
#: is rebuilt, rather than being reused and failing later with a half-installed package.
_MARKER = ".install-complete.json"

# A dependency install must never leave a Playground request waiting forever. The lock wait
# follows the same bounded install timeout, so a legitimate first install is not reported as a
# failure merely because downloading a framework took longer than two minutes.
_LOCK_WAIT_GRACE_SECONDS = 30.0

# Multiple Playground requests can need the same framework at once. Without an OS-level lock
# they all run pip against the same target directory, which leaves the cache incomplete and
# makes every waiting request look like a hung agent.

# The rule for which requirement lines may be installed lives in
# ``multi_agent_generator.dependencies`` - see :func:`screen_requirements` there - because
# validation needs to apply the same rule before a run, and a second copy of it here would
# eventually disagree with the first.


@dataclass
class DependencySet:
    """
    The outcome of making a project's dependencies available.

    ``path`` is the directory to add to ``PYTHONPATH``; None when there was nothing to install.
    ``ok`` is False only when an install was attempted and failed - "nothing to install" is a
    success, because a project with no dependencies has not failed to install them.
    """

    path: Optional[Path] = None
    ok: bool = True
    from_cache: bool = False
    #: The pip run, when one happened. Kept so a failure can be shown with its real output.
    result: Optional[ProcessResult] = None
    #: Plain-language explanation when ``ok`` is False, or when nothing was installed.
    reason: str = ""
    #: Requirement lines that were refused, with the reason for each.
    rejected: List[str] = field(default_factory=list)

    @property
    def installed(self) -> bool:
        return self.path is not None and self.path.is_dir()

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "installed": self.installed,
            "from_cache": self.from_cache,
            "reason": self.reason,
            "rejected": list(self.rejected),
            "exit_code": self.result.exit_code if self.result else None,
            "duration_s": round(self.result.duration_s, 3) if self.result else 0.0,
        }


def _read_requirements(workspace: Workspace) -> str:
    path = workspace.root / "requirements.txt"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _screen(text: str) -> Tuple[List[str], List[str]]:
    """
    Split requirement lines into the ones that may be installed and the ones that may not.

    Thin wrapper over :func:`~multi_agent_generator.dependencies.screen_requirements`, kept
    only so the call site below reads the same as it always did.
    """
    return screen_requirements(text)


def _cache_key(lines: List[str]) -> str:
    """
    A stable key for a set of requirement lines.

    Sorted before hashing so two projects that need the same packages in a different order
    share one install. The interpreter version is part of the key because a ``--target``
    directory holds compiled wheels for one Python.
    """
    payload = "\n".join(sorted(lines)) + f"\npy{sys.version_info.major}.{sys.version_info.minor}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def ensure_dependencies(
    workspace: Workspace,
    *,
    timeout: Optional[int] = None,
    cache_dir: Optional[os.PathLike] = None,
) -> DependencySet:
    """
    Make the workspace's requirements importable, installing them only if needed.

    Returns a :class:`DependencySet` and never raises for an install failure - a project whose
    dependencies cannot be resolved is a result the pipeline has to report, not a crash.
    """
    settings = get_settings()
    root = Path(cache_dir) if cache_dir is not None else settings.deps_cache_dir
    text = _read_requirements(workspace)
    allowed, rejected = _screen(text)

    if rejected:
        # Logged as well as returned. A refusal that only ever appears in a return value is one
        # a caller can forget to look at, and the symptom - an unexplained ImportError several
        # steps later - is much harder to trace back than a line in the log.
        LOGGER.warning(
            "Refused %d requirement line(s) from a generated project",
            len(rejected),
            extra={"stage": "install"},
        )

    if not allowed:
        if rejected:
            # Not a success. Every line was refused, so the project's imports will fail, and
            # returning ok=True here would hand the caller a green install followed by a
            # ModuleNotFoundError with no stated cause.
            return DependencySet(
                ok=False,
                reason=(
                    "None of this project's requirements could be installed: every line asks "
                    "to install from a URL, a local path or an alternative index, which is not "
                    "allowed for generated code. The refused lines are listed below."
                ),
                rejected=rejected,
            )
        return DependencySet(
            ok=True,
            reason="This project lists no installable dependencies, so nothing was installed.",
            rejected=rejected,
        )

    key = _cache_key(allowed)
    target = root / key
    install_timeout = timeout or settings.install_timeout
    try:
        with _cache_lock(
            target.with_name(target.name + ".lock"),
            timeout=install_timeout + _LOCK_WAIT_GRACE_SECONDS,
        ):
            return _ensure_cached_dependencies(
                target,
                allowed,
                rejected,
                timeout=install_timeout,
            )
    except TimeoutError:
        return DependencySet(
            ok=False,
            reason=(
                "Another process is installing this project's dependencies and did not "
                "release the cache lock in time. Try the Playground again."
            ),
            rejected=rejected,
        )


@contextmanager
def _cache_lock(path: Path, *, timeout: float):
    """Hold a cross-process lock using atomic directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = Path(str(path) + ".dir")
    started = time.monotonic()
    owner = f"{os.getpid()}\n".encode("ascii")
    acquired = False

    while not acquired:
        try:
            lock_dir.mkdir()
            lock_dir.joinpath("owner").write_bytes(owner)
            acquired = True
        except FileExistsError:
            try:
                age = time.time() - lock_dir.stat().st_mtime
                contents = lock_dir.joinpath("owner").read_text(encoding="ascii").strip()
                owner_pid = int(contents) if contents else None
            except (OSError, ValueError):
                age = 0.0
                owner_pid = None

            # The previous implementation could leave an empty lock file behind. Treat an
            # empty file that has had time to finish creation as stale, and recover locks from
            # workers that no longer exist.
            owner_alive = owner_pid is not None and _pid_is_alive(owner_pid)
            if (not owner_alive and age >= 1.0) or age > 3600.0:
                try:
                    shutil.rmtree(lock_dir)
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() - started >= timeout:
                raise TimeoutError(str(path))
            time.sleep(0.1)

    try:
        yield
    finally:
        try:
            shutil.rmtree(lock_dir)
        except FileNotFoundError:
            pass


def _pid_is_alive(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _ensure_cached_dependencies(
    target: Path,
    allowed: List[str],
    rejected: List[str],
    *,
    timeout: int,
) -> DependencySet:
    """Install one cache entry while its per-target lock is held."""
    marker = target / _MARKER

    if marker.exists():
        LOGGER.debug("Dependency cache hit", extra={"stage": "install"})
        return DependencySet(
            path=target,
            ok=True,
            from_cache=True,
            reason="Dependencies were already installed for this requirement set.",
            rejected=rejected,
        )

    # A directory without the marker is the residue of an install that was killed. Removing it
    # is not optional: pip --target into a partially populated directory produces an
    # environment that imports and then fails halfway through, which is far harder to diagnose
    # than a clean reinstall.
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    # Written from the screened lines rather than reusing the project's file, so exactly what
    # was allowed is what gets installed.
    spec_file = target / "requirements.screened.txt"
    spec_file.write_text("\n".join(allowed) + "\n", encoding="utf-8")

    result = run_process(
        [
            python_executable(),
            "-m",
            "pip",
            "install",
            "--target",
            str(target),
            "--quiet",
            "--disable-pip-version-check",
            "--no-input",
            "--timeout",
            "30",
            "--retries",
            "2",
            # No build isolation opt-out and no --no-deps: the project's own transitive
            # dependencies are part of what it needs, and resolving them here is the point.
            "-r",
            str(spec_file),
        ],
        cwd=str(target),
        env=build_child_env(),
        timeout=timeout,
    )

    if not result.ok:
        # The marker is deliberately not written, so the next attempt starts clean instead of
        # importing a partial install and failing with a confusing error.
        return DependencySet(
            path=None,
            ok=False,
            result=result,
            reason=(
                "The project's dependencies could not be installed. The pip output below "
                "names the package that failed."
            ),
            rejected=rejected,
        )

    marker.write_text(
        json.dumps({"requirements": allowed}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "Installed dependencies for a generated project",
        extra={"stage": "install", "duration_s": result.duration_s},
    )
    return DependencySet(
        path=target,
        ok=True,
        result=result,
        reason="Dependencies installed.",
        rejected=rejected,
    )


def python_path_for(workspace: Workspace, deps: Optional[DependencySet] = None) -> str:
    """
    The ``PYTHONPATH`` a child running this workspace needs.

    The workspace root comes first so the project's own modules always win over anything with
    the same name in the dependency directory - a generated project containing ``tools.py``
    must import its own, not a package that happens to be called ``tools``.
    """
    parts = [str(workspace.root)]
    if deps is not None and deps.installed and deps.path is not None:
        parts.append(str(deps.path))
    return os.pathsep.join(parts)


def cache_entries(cache_dir: Optional[os.PathLike] = None) -> List[Dict[str, object]]:
    """
    What is currently in the dependency cache, for a settings screen or a support question.

    Reports only completed installs. A directory without its marker is residue, and listing it
    as if it were usable would be the same class of lie as a green tick on a failed run.
    """
    settings = get_settings()
    root = Path(cache_dir) if cache_dir is not None else settings.deps_cache_dir
    if not root.is_dir():
        return []

    entries: List[Dict[str, object]] = []
    for child in sorted(root.iterdir()):
        marker = child / _MARKER
        if not marker.is_file():
            continue
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        entries.append(
            {
                "key": child.name,
                "requirements": payload.get("requirements") or [],
                "size_mb": round(_directory_size(child) / (1024 * 1024), 1),
            }
        )
    return entries


def clear_dependency_cache(cache_dir: Optional[os.PathLike] = None) -> int:
    """
    Delete every cached install and report how many were removed.

    Safe at any time: the cache is rebuilt on the next run that needs it, which is why it can
    be offered as a real button rather than a warning-laden one.
    """
    settings = get_settings()
    root = Path(cache_dir) if cache_dir is not None else settings.deps_cache_dir
    if not root.is_dir():
        return 0
    removed = 0
    for child in list(root.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


def _directory_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            # A file that vanished mid-walk is not worth failing a size report over.
            continue
    return total
