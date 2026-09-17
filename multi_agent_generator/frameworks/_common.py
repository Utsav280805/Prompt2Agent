# multi_agent_generator/frameworks/_common.py
"""
Shared helpers for the framework code generators.

These exist so that every generator sanitises names, names tool classes and renders
LLM setup the same way. Previously each generator did its own thing, which is how the
output ended up with invalid identifiers for multi-word agent names and a
non-deterministic tool order.
"""
from __future__ import annotations

import keyword
import re
from typing import Any, Dict, Iterable, List, Optional

from ..providers import LLMSnippet, get_provider, llm_snippet

__all__ = [
    "sanitize_identifier",
    "tool_class_name",
    "dedupe",
    "collect_tools",
    "render_llm_setup",
    "llm_snippet_for",
    "header_comment",
]


def sanitize_identifier(name: str, prefix: str = "x") -> str:
    """
    Turn an arbitrary label into a valid, non-reserved Python identifier.

    Agent names come from an LLM, so they routinely contain spaces, punctuation or
    leading digits. The old generators only replaced spaces and hyphens, which meant a
    name like "Research & Analysis" produced code that would not parse.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", (name or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def tool_class_name(name: str) -> str:
    """Build a CamelCase tool class name, e.g. ``web_scraper`` -> ``WebScraperTool``."""
    parts = [p for p in re.split(r"[^0-9a-zA-Z]+", name or "") if p]
    if not parts:
        parts = ["custom"]
    camel = "".join(p[:1].upper() + p[1:] for p in parts)
    if camel[0].isdigit():
        camel = f"Tool{camel}"
    if not camel.endswith("Tool"):
        camel += "Tool"
    return camel


def dedupe(items: Iterable[str]) -> List[str]:
    """
    De-duplicate while preserving first-seen order.

    The generators used to build tool lists with a ``set``, so the emitted file differed
    between runs for the same input. Deterministic output matters here: generated code
    gets committed and diffed.
    """
    return list(dict.fromkeys(i for i in items if i))


def collect_tools(agents: Iterable[Dict[str, Any]]) -> List[str]:
    """
    Every tool name mentioned by any agent, in stable order.

    Also drops names that would collide once turned into a class name: ``search`` and
    ``search_tool`` both become ``SearchTool``, and emitting that class twice produces a
    file where the second definition silently shadows the first. De-duplicating here
    rather than in each generator means every framework gets the fix.
    """
    names: List[str] = []
    for agent in agents or []:
        for tool in agent.get("tools") or []:
            if isinstance(tool, str):
                names.append(tool)
            elif isinstance(tool, dict) and tool.get("name"):
                names.append(tool["name"])

    unique: List[str] = []
    seen_classes: set = set()
    for name in dedupe(names):
        class_name = tool_class_name(name)
        if class_name in seen_classes:
            continue
        seen_classes.add(class_name)
        unique.append(name)
    return unique


#: Providers whose model ids are plain names rather than Hub repo ids. A config-suggested
#: model is only trusted for these, or for the two Hugging Face providers when it looks
#: like a repo id.
_PLAIN_MODEL_PROVIDERS = ("openai",)
_HUB_MODEL_PROVIDERS = ("huggingface", "huggingface-local")


def llm_snippet_for(
    framework: str,
    provider: str = "openai",
    model: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> LLMSnippet:
    """
    Resolve the LLM snippet for a framework, honouring a model named in the config.

    When the caller did not pass an explicit model, a model id suggested by the config
    (``model_id``, or the first agent's ``llm``) is used - but only for providers where
    such a name is meaningful.

    This guard has to be narrow. The analysis prompts ask the model for a name like
    ``gpt-4.1-mini``, and the fallback configs hardcode exactly that, so a permissive
    rule would emit ``ChatGroq(model="gpt-4.1-mini")`` or
    ``ChatAnthropic(model="gpt-4o")`` - code that fails on its first call. A suggestion
    is therefore accepted only for OpenAI, or for Hugging Face when it actually looks
    like a Hub repo id. Every other provider falls through to its registry default.
    """
    resolved = model
    if resolved is None and config:
        suggested = config.get("model_id")
        if not suggested:
            agents = config.get("agents") or []
            if agents and isinstance(agents[0], dict):
                suggested = agents[0].get("llm")
        if suggested:
            # Normalise first: aliases like "hf" must not bypass the Hub-id check.
            spec_name = get_provider(provider).name
            looks_like_hub_id = "/" in suggested
            if spec_name in _HUB_MODEL_PROVIDERS:
                if looks_like_hub_id:
                    resolved = suggested
            elif spec_name in _PLAIN_MODEL_PROVIDERS and not looks_like_hub_id:
                resolved = suggested

    return llm_snippet(framework, provider, resolved)


def render_llm_setup(snippet: LLMSnippet, indent: str = "") -> str:
    """Render a snippet's preamble + setup block, optionally indented, with its comment."""
    lines: List[str] = []
    if snippet.comment:
        lines.append(f"# NOTE: {snippet.comment}")
    if snippet.preamble:
        lines.extend(snippet.preamble.splitlines())
        lines.append("")
    if snippet.setup:
        lines.extend(snippet.setup.splitlines())
    if not indent:
        return "\n".join(lines)
    return "\n".join(f"{indent}{line}" if line else "" for line in lines)


def header_comment(
    label: str,
    provider: str,
    model: str,
    framework_key: Optional[str] = None,
) -> str:
    """
    A short provenance banner for generated files.

    ``label`` is the human-readable framework name shown in the banner ("CrewAI Flow"),
    while ``framework_key`` is the canonical CLI value ("crewai-flow"). They have to be
    separate: the banner ends with a copy-pasteable ``--check-deps`` command, and passing
    the display label there produced commands argparse rejects, since ``--framework``
    only accepts the lowercase hyphenated keys.
    """
    key = framework_key or label
    return (
        f'"""\n'
        f"Auto-generated {label} multi-agent system.\n\n"
        f"Provider: {provider}\n"
        f"Model:    {model}\n\n"
        f"Generated by multi-agent-generator. Run `multi-agent-generator --check-deps "
        f"--framework {key} --provider {provider}`\n"
        f"to verify the packages this file needs are installed.\n"
        f'"""\n'
    )
