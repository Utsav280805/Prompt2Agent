# multi_agent_generator/core/parsing.py
"""
Reading structured data out of model output.

Every stage that asks a model for JSON has to cope with the same four habits: a ```json
fence, a sentence of preamble, a trailing comma, and single quotes. Each of those was
handled ad hoc at a different call site, with a different fallback, which is why one stage
would silently return a default configuration while another raised.

This module does it once, and - importantly - distinguishes *"the model returned prose"*
from *"the model returned JSON that says something I did not expect"*. Only the first is a
parsing problem. The second is a content problem and belongs to the caller.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

__all__ = [
    "extract_json",
    "extract_json_object",
    "as_str_list",
    "as_float",
    "as_bool",
    "strip_code_fences",
]

_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def strip_code_fences(text: str) -> str:
    """
    Return the contents of the first fenced block, or the text unchanged.

    Applied to generated *code* as well as JSON: models asked for a Python file very often
    wrap it in a fence, and writing that fence into a ``.py`` file produces a syntax error
    on line one.
    """
    if not text:
        return ""
    match = _FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def _balanced_slice(text: str, opener: str, closer: str) -> Optional[str]:
    """
    Find the first balanced ``opener``/``closer`` region, ignoring braces inside strings.

    The old approach - ``text[text.find('{') : text.rfind('}') + 1]`` - breaks on two
    common shapes: a model that emits two JSON objects (it splices them together into
    something invalid), and a model that mentions ``}`` inside a prose sentence after the
    object. Tracking depth and string state costs a few lines and removes both failures.
    """
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _repair(candidate: str) -> str:
    """
    Fix the small syntax slips small models make, and nothing else.

    Deliberately conservative. Trailing commas and Python's ``True``/``False``/``None``
    are unambiguous mistakes with exactly one correct reading. Anything more aggressive -
    rewriting single quotes, say - risks corrupting string content that was legitimately
    quoted, and turning a visible parse failure into a silently wrong value.
    """
    repaired = _TRAILING_COMMA.sub(r"\1", candidate)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    return repaired


def extract_json(text: str) -> Optional[Any]:
    """
    Pull the first JSON value out of ``text``. Returns None when there is none.

    None means "no JSON here", which the caller must handle explicitly. Returning an empty
    dict instead would be indistinguishable from a model that genuinely answered ``{}``,
    and that ambiguity is how an empty configuration once passed for a real one.
    """
    if not text or not text.strip():
        return None

    for candidate in _candidates(text):
        for attempt in (candidate, _repair(candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _candidates(text: str) -> List[str]:
    """Progressively less literal readings of the text, best first."""
    found: List[str] = []
    stripped = text.strip()

    fenced = strip_code_fences(text)
    if fenced and fenced != stripped:
        found.append(fenced)
    found.append(stripped)

    for source in (fenced or stripped, stripped):
        for opener, closer in (("{", "}"), ("[", "]")):
            sliced = _balanced_slice(source, opener, closer)
            if sliced and sliced not in found:
                found.append(sliced)
    return found


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Like :func:`extract_json`, but only accepts an object.

    A bare list where an object was requested is a different failure from prose, and
    callers that want ``{"score": ...}`` should not have to re-check the type themselves.
    """
    value = extract_json(text)
    if isinstance(value, dict):
        return value
    # A model asked for one object sometimes returns a single-element array containing it.
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    return None


def as_str_list(value: Any, limit: int = 50) -> List[str]:
    """
    Coerce a field that should be a list of strings.

    Models return a list, a newline-joined string, a comma-joined string, or a list of
    ``{"description": ...}`` objects, all for the same prompt. Normalising here means the
    stages consuming these fields do not each grow their own coercion.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip(" -*\t") for p in re.split(r"[\n;]+", value)]
        if len(parts) == 1:
            parts = [p.strip() for p in value.split(",")]
        return [p for p in (part.strip() for part in parts) if p][:limit]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                text = (
                    item.get("description")
                    or item.get("message")
                    or item.get("name")
                    or item.get("text")
                )
                if text and str(text).strip():
                    out.append(str(text).strip())
        return out[:limit]
    return [str(value).strip()][:limit]


def as_float(value: Any, default: float = 0.0) -> float:
    """Read a number that may arrive as ``"8.5"``, ``"8.5/10"`` or ``8``."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return default
    return default


def as_bool(value: Any, default: bool = False) -> bool:
    """Read a boolean that may arrive as a string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes", "y", "1", "pass", "passed", "ok"):
            return True
        if text in ("false", "no", "n", "0", "fail", "failed"):
            return False
    return default
