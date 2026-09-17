# multi_agent_generator/settings.py
"""
Application settings.

One place that reads the environment, so nothing else has to. The precedence rule is
the one users expect:

    explicit argument  >  saved user config  >  environment / .env  >  built-in default

``.env`` is loaded once, here, at import - and *only* here. Previously ``load_dotenv()``
sat in ``model_inference``, which meant importing that module had a side effect on the
process environment, and any code path that did not import it (notably a generated
standalone script) silently saw no ``HF_TOKEN`` at all. That mismatch is exactly the bug
behind "I configured Hugging Face and it still says no api_key".
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "Settings",
    "get_settings",
    "reload_settings",
    "project_root",
    "load_env_file",
]


def project_root() -> Path:
    """Repository root - the directory containing this package."""
    return Path(__file__).resolve().parent.parent


def load_env_file(path: Optional[Path] = None, override: bool = False) -> bool:
    """
    Load a ``.env`` file into ``os.environ``.

    Uses python-dotenv when available and falls back to a small parser otherwise, so
    this never becomes a hard dependency for generated code that wants the same
    behaviour. Returns True if a file was read.
    """
    env_path = Path(path) if path else project_root() / ".env"
    if not env_path.is_file():
        return False

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=override)
        return True
    except Exception:
        pass

    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and (override or key not in os.environ):
                os.environ[key] = value
        return True
    except OSError:
        return False


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, ignoring blanks and junk rather than crashing."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw.strip()))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """Resolved application configuration."""

    # --- storage -----------------------------------------------------------------
    data_dir: Path
    database_url: str
    workspaces_dir: Path
    #: Where a generated project's dependencies are installed. Never the application's own
    #: interpreter: model-authored ``requirements.txt`` files must not be able to change the
    #: packages this server is running on. Keyed by requirements content and reused, so the
    #: framework is downloaded once rather than once per run.
    deps_cache_dir: Path

    # --- pipeline ----------------------------------------------------------------
    max_generation_iterations: int = 3
    #: How many times the repair engine may rewrite a project within one iteration.
    #:
    #: Bounded on purpose. A repair loop with no ceiling is the failure mode of every
    #: self-healing generator: it burns the budget re-attempting a fix that cannot work,
    #: and the user watches a spinner instead of reading the real error. Three is the
    #: figure section 19 names, and :mod:`multi_agent_generator.core.repair` treats it as
    #: a hard stop rather than a suggestion.
    max_repair_iterations: int = 3
    review_pass_score: float = 7.5
    agent_execution_timeout: int = 120
    test_execution_timeout: int = 300
    install_timeout: int = 900

    # --- LLM defaults -------------------------------------------------------------
    default_provider: str = "huggingface"
    default_model: Optional[str] = None
    request_timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 2000

    # --- server -------------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")
    log_level: str = "INFO"
    log_json: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file()

        root = project_root()
        data_dir = Path(os.getenv("DATA_DIR") or (root / ".magen_data")).resolve()
        workspaces = Path(
            os.getenv("WORKSPACES_DIR") or (data_dir / "workspaces")
        ).resolve()
        deps_cache = Path(
            os.getenv("DEPS_CACHE_DIR") or (data_dir / "deps")
        ).resolve()

        db_url = os.getenv("DATABASE_URL") or f"sqlite:///{(data_dir / 'magen.db').as_posix()}"

        origins_raw = os.getenv("CORS_ORIGINS", "")
        origins = tuple(o.strip() for o in origins_raw.split(",") if o.strip()) or (
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
        )

        return cls(
            data_dir=data_dir,
            database_url=db_url,
            workspaces_dir=workspaces,
            deps_cache_dir=deps_cache,
            max_generation_iterations=_env_int("MAX_GENERATION_ITERATIONS", 3),
            max_repair_iterations=_env_int("MAX_REPAIR_ITERATIONS", 3),
            review_pass_score=_env_float("REVIEW_PASS_SCORE", 7.5),
            agent_execution_timeout=_env_int("AGENT_EXECUTION_TIMEOUT", 120),
            test_execution_timeout=_env_int("TEST_EXECUTION_TIMEOUT", 300),
            install_timeout=_env_int("DEPENDENCY_INSTALL_TIMEOUT", 900),
            default_provider=os.getenv("DEFAULT_PROVIDER", "huggingface"),
            default_model=os.getenv("DEFAULT_MODEL") or None,
            request_timeout=_env_int("LLM_REQUEST_TIMEOUT", 120),
            temperature=_env_float("LLM_TEMPERATURE", 0.7),
            max_tokens=_env_int("LLM_MAX_TOKENS", 2000),
            host=os.getenv("HOST", "127.0.0.1"),
            port=_env_int("PORT", 8000),
            cors_origins=origins,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_json=_env_bool("LOG_JSON", False),
        )

    def ensure_dirs(self) -> None:
        """Create the directories this configuration promises exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        self.deps_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sqlite_path(self) -> Optional[Path]:
        """Filesystem path when ``database_url`` is SQLite, else None."""
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url[len("sqlite:///") :])
        return None


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """The process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
        _settings.ensure_dirs()
    return _settings


def reload_settings() -> Settings:
    """Re-read the environment. Used by tests and by the settings API endpoint."""
    global _settings
    _settings = None
    return get_settings()
