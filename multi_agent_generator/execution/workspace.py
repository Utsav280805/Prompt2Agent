# multi_agent_generator/execution/workspace.py
"""
Writing a generated project to disk, safely, and building the environment it runs in.

Two jobs, both security boundaries.

**Materialising the project.** A :class:`GeneratedProject` is a list of files whose paths
came, ultimately, from model output. Writing them to disk is exactly the moment a path like
``../../.ssh/authorized_keys`` would do damage, so every write is re-checked against the
workspace root here even though :class:`GeneratedProject` already normalised the paths once.
Two independent checks on a filesystem-escape is not redundancy worth removing; it is the
cheap half of defence in depth.

**Building the child environment.** The generated code runs in a *scrubbed* environment, not
the application's. The brief is explicit - "No accidental access to application secrets", "Do
not expose the host environment unnecessarily" - and the failure it guards against is real:
if the child inherited ``os.environ`` it would inherit the application's own OpenAI key, and
generated code that logs its environment (plenty does) would print it. So the child gets a
minimal PATH-and-locale environment plus *only* the one credential its own provider needs,
and nothing else crosses the boundary.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from ..core.models import GeneratedProject
from ..errors import ExecutionError

__all__ = ["Workspace", "prepare_workspace", "build_child_env"]


#: Environment variable name prefixes that may cross into the sandbox. Anything not matching
#: one of these is dropped, so an application secret with an unexpected name cannot leak by
#: accident - the list is an allowlist, which fails safe, rather than a denylist, which fails
#: open the moment someone adds a new secret.
_SAFE_ENV_PREFIXES = (
    "PATH",
    "PYTHON",
    "LC_",
    "LANG",
    "TERM",
    "TMP",
    "TEMP",
    "TZ",
    "HOME",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "APPDATA",
    "LOCALAPPDATA",
)

#: Names that must never cross, even if they somehow matched a prefix above. Belt and
#: braces: the allowlist should already exclude these, but naming them makes the intent
#: auditable and survives someone loosening the prefix list later.
_BLOCKED_ENV_NAMES = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WATSONX_APIKEY",
    "IBM_CLOUD_API_KEY",
    "DATABASE_URL",
}


@dataclass
class Workspace:
    """
    A directory holding one materialised project, plus the paths a runner needs.

    ``owns_dir`` records whether this workspace created the directory, so :meth:`cleanup`
    only deletes what it made. A caller that pointed a workspace at an existing directory
    (to inspect a run afterwards) does not want it swept away underneath them.
    """

    root: Path
    entrypoint: Optional[str]
    files: List[str]
    owns_dir: bool = True

    @property
    def entrypoint_path(self) -> Optional[Path]:
        return (self.root / self.entrypoint) if self.entrypoint else None

    def path(self, relative: str) -> Path:
        """Resolve a project-relative path, refusing anything that escapes the root."""
        return _safe_join(self.root, relative)

    def read(self, relative: str) -> str:
        return self.path(relative).read_text(encoding="utf-8")

    def cleanup(self) -> None:
        if self.owns_dir and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> bool:
        self.cleanup()
        return False


def prepare_workspace(
    project: GeneratedProject,
    *,
    parent_dir: Optional[os.PathLike] = None,
    keep: bool = False,
) -> Workspace:
    """
    Write ``project`` to a fresh directory and return a handle to it.

    Args:
        project: the project to materialise.
        parent_dir: where to create the workspace. Defaults to a new temp directory. The
            API points this at its configured workspaces dir so runs can be inspected.
        keep: unused sentinel reserved for callers that manage their own lifetime; the
            ``owns_dir`` flag is what actually controls cleanup.

    Raises:
        ExecutionError: if the project has no files, or a path escapes the workspace. Both
            are refusals rather than best-effort writes, because a half-written project is
            not something worth trying to run.
    """
    if not project.files:
        raise ExecutionError(
            "There is nothing to run: the project has no files.",
            action="Generate a project before executing it.",
        )

    base = Path(parent_dir) if parent_dir is not None else None
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="mag_run_", dir=str(base) if base else None))

    written: List[str] = []
    for file in project.files:
        target = _safe_join(root, file.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file.content, encoding="utf-8")
        written.append(file.path)

    entry = project.entrypoint_file
    return Workspace(
        root=root,
        entrypoint=entry.path if entry else None,
        files=written,
        owns_dir=True,
    )


def build_child_env(
    *,
    credential_name: Optional[str] = None,
    credential_value: Optional[str] = None,
    base_env: Optional[Mapping[str, str]] = None,
    extra: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """
    Build the scrubbed environment a sandboxed child runs in.

    Starts from an allowlist of the host environment - enough to find an interpreter and a
    temp dir, nothing more - then adds back exactly one credential: the one the generated
    code's own provider needs. The application's other secrets never enter the dictionary,
    so they cannot be read, logged or exfiltrated by whatever the model wrote.

    Args:
        credential_name: env var the generated provider reads, e.g. ``HF_TOKEN``. The only
            secret allowed across the boundary.
        credential_value: its value. Passed explicitly rather than read from ``os.environ``
            so the caller - which already resolved it - stays the single source of truth,
            and so this function never has to touch the host's secrets itself.
        base_env: the host environment to filter. Defaults to ``os.environ``.
        extra: additional non-secret variables to set (e.g. ``PYTHONPATH``).
    """
    source = base_env if base_env is not None else os.environ
    child: Dict[str, str] = {}

    for name, value in source.items():
        if name in _BLOCKED_ENV_NAMES:
            continue
        if _is_allowed(name):
            child[name] = value

    # Keep child output unbuffered and importable-from-here. Unbuffered matters for the
    # timeout path: a buffered child that is killed loses the very output that would explain
    # why it hung.
    child.setdefault("PYTHONUNBUFFERED", "1")
    child.setdefault("PYTHONIOENCODING", "utf-8")
    # Stop the child writing .pyc files into the sandbox - noise in the Files view and a
    # needless write into a directory meant to be disposable.
    child.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    if extra:
        child.update({str(k): str(v) for k, v in extra.items()})

    if credential_name and credential_value:
        # Added last and unconditionally: this is the one secret that is supposed to be
        # here, and it must not be filtered out by the allowlist above.
        child[credential_name] = credential_value

    return child


def _is_allowed(name: str) -> bool:
    upper = name.upper()
    return any(upper == prefix or upper.startswith(prefix) for prefix in _SAFE_ENV_PREFIXES)


def _safe_join(root: Path, relative: str) -> Path:
    """
    Join ``relative`` onto ``root`` and refuse to leave ``root``.

    The check is on the *resolved* path, so it catches escapes that only appear after
    symlinks and ``..`` are collapsed - the ones a naive string prefix check would miss.
    This is the last line before bytes hit the disk, so it raises rather than sanitising:
    at this point an out-of-bounds path is a bug or an attack, and neither should be papered
    over by silently relocating the file.
    """
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ExecutionError(
            "A generated file path tried to escape the workspace.",
            action="This is a bug in code generation; please report it.",
            context={"path": relative},
        ) from exc
    return candidate
