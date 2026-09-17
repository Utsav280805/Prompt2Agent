# multi_agent_generator/core/analysis.py
"""
Stage 1: work out what the user actually asked for.

The output feeds framework selection, so it deliberately asks about *structural*
properties - is there an ordering, is there branching, is there delegation, is there state
- rather than producing a prose summary. Those are the properties that distinguish a crew
from a graph, and asking about them directly is what makes the subsequent choice
defensible instead of a keyword guess.

There are two paths and they are honest about which one ran. With a working LLM the model
answers; with no credential (or a model that returns prose) a heuristic answers and sets
``from_model=False``, which the UI shows. What does *not* happen is a silent fall back that
presents heuristic output as model output.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..errors import AppError
from ..llm.base import LLMProvider
from .models import RequirementAnalysis
from .parsing import as_bool, as_str_list, extract_json_object

__all__ = ["analyze_requirement", "heuristic_analysis", "ANALYSIS_SYSTEM_PROMPT"]


ANALYSIS_SYSTEM_PROMPT = """\
You analyse a request for an AI agent system and describe its structure. You do not write
code and you do not choose a framework.

Reply with ONLY a JSON object, no prose and no code fence:

{
  "summary": "one or two sentences describing what the system must do",
  "goals": ["concrete outcome", "..."],
  "suggested_roles": ["researcher", "writer"],
  "suggested_tools": ["web_search", "file_read"],
  "needs_sequential_steps": true,
  "needs_branching": false,
  "needs_delegation": false,
  "needs_state": false,
  "needs_human_input": false,
  "complexity": "simple" | "moderate" | "complex",
  "risks": ["what could make this system fail or behave badly"]
}

Definitions - be strict about these, they decide the architecture:
- needs_sequential_steps: distinct steps that must run in a fixed order.
- needs_branching: a step's result changes which step runs next, or work loops/retries.
- needs_delegation: a coordinator decides which specialist handles each piece of work.
- needs_state: information must persist and be updated across steps or turns.
- needs_human_input: a person must approve or supply something mid-run.

Use 2 to 5 roles. Prefer fewer, well-separated roles over many overlapping ones.
"""


def analyze_requirement(
    requirement: str,
    provider: Optional[LLMProvider] = None,
) -> RequirementAnalysis:
    """
    Describe the structure of ``requirement``.

    Args:
        requirement: What the user typed.
        provider: LLM to consult. When None, the heuristic is used directly - which is how
            the pipeline runs with no credentials at all.

    Never raises for model-side problems. Analysis is the first stage, and failing the
    whole run because a model returned prose would be worse than proceeding on a clearly
    labelled heuristic reading; the credential error, if there is one, surfaces at
    generation time where it genuinely blocks progress.
    """
    text = (requirement or "").strip()
    if not text:
        # An empty requirement is a caller bug, not a model problem, and no heuristic can
        # invent an answer. Say so rather than returning a plausible-looking empty record.
        raise AppError(
            "No requirement was provided.",
            action="Describe what you want the agent system to do, then try again.",
        )

    if provider is None:
        return heuristic_analysis(text)

    try:
        raw = provider.complete_text(text, system=ANALYSIS_SYSTEM_PROMPT)
    except AppError:
        # Re-raised, not swallowed: a missing credential or unreachable provider is
        # actionable and the pipeline decides what to do about it.
        raise

    payload = extract_json_object(raw)
    if payload is None:
        fallback = heuristic_analysis(text)
        fallback.risks.append(
            "Requirement analysis fell back to heuristics: the model did not return JSON."
        )
        return fallback
    return _from_payload(text, payload)


def _from_payload(requirement: str, payload: Dict[str, Any]) -> RequirementAnalysis:
    """Build an analysis from parsed model output, with the heuristic filling any gaps."""
    guess = heuristic_analysis(requirement)

    complexity = str(payload.get("complexity") or "").strip().lower()
    if complexity not in ("simple", "moderate", "complex"):
        complexity = guess.complexity

    roles = as_str_list(payload.get("suggested_roles"), limit=8) or guess.suggested_roles
    return RequirementAnalysis(
        requirement=requirement,
        summary=str(payload.get("summary") or "").strip() or guess.summary,
        goals=as_str_list(payload.get("goals"), limit=10) or guess.goals,
        suggested_roles=roles,
        suggested_tools=as_str_list(payload.get("suggested_tools"), limit=10)
        or guess.suggested_tools,
        needs_sequential_steps=as_bool(
            payload.get("needs_sequential_steps"), guess.needs_sequential_steps
        ),
        needs_branching=as_bool(payload.get("needs_branching"), guess.needs_branching),
        needs_delegation=as_bool(payload.get("needs_delegation"), guess.needs_delegation),
        needs_state=as_bool(payload.get("needs_state"), guess.needs_state),
        needs_human_input=as_bool(
            payload.get("needs_human_input"), guess.needs_human_input
        ),
        complexity=complexity,
        risks=as_str_list(payload.get("risks"), limit=10),
        from_model=True,
    )


# ============================================================================ heuristic
#: Signals for each structural property. Multi-word phrases are checked as substrings and
#: single words on word boundaries, so "state" does not match "statement".
_SIGNALS: Dict[str, tuple] = {
    "sequential": (
        "step by step", "stepwise", "pipeline", "workflow", "then", "after that",
        "first", "finally", "stage", "sequence", "in order", "followed by",
        "multi-step", "process",
    ),
    "branching": (
        "if", "depending", "decide", "conditional", "branch", "route", "retry",
        "loop", "until", "fallback", "otherwise", "escalate", "classify", "triage",
    ),
    "delegation": (
        "delegate", "supervisor", "manager", "coordinate", "orchestrate", "assign",
        "team of", "crew", "specialist", "hand off", "handoff", "sub-agent", "subagent",
    ),
    "state": (
        "remember", "memory", "conversation", "chat", "history", "context across",
        "session", "persist", "accumulate", "track progress", "stateful", "follow-up",
    ),
    "human": (
        "approve", "approval", "human in the loop", "human-in-the-loop", "review by",
        "confirm", "sign off", "sign-off", "ask the user", "manual check",
    ),
}

#: Role hints. The key is the role name emitted; the values are what suggests it.
_ROLE_HINTS: Dict[str, tuple] = {
    "researcher": ("research", "find", "search", "gather", "investigate", "source", "look up"),
    "analyst": ("analyse", "analyze", "analysis", "evaluate", "assess", "compare", "insight"),
    "writer": ("write", "draft", "summarise", "summarize", "report", "article", "content", "blog"),
    "editor": ("edit", "proofread", "review", "polish", "refine", "quality"),
    "planner": ("plan", "organise", "organize", "schedule", "roadmap", "strategy", "itinerary"),
    "coder": ("code", "program", "script", "implement", "develop", "refactor", "debug"),
    "data_engineer": ("data", "csv", "database", "sql", "etl", "dataset", "spreadsheet"),
    "support_agent": ("support", "customer", "ticket", "helpdesk", "faq", "complaint"),
    "reviewer": ("verify", "validate", "fact check", "fact-check", "audit", "critique"),
}

#: Tool hints, same shape.
_TOOL_HINTS: Dict[str, tuple] = {
    "web_search": ("search", "web", "internet", "google", "news", "online", "research"),
    "web_scrape": ("scrape", "crawl", "website", "url", "page content"),
    "file_read": ("file", "document", "pdf", "read from disk", "upload"),
    "file_write": ("save", "export", "write to file", "output file"),
    "data_analysis": ("csv", "excel", "spreadsheet", "dataframe", "statistics", "chart"),
    "api_request": ("api", "endpoint", "rest", "webhook", "http"),
    "database_query": ("database", "sql", "postgres", "mysql", "query the"),
    "calculator": ("calculate", "compute", "math", "arithmetic", "sum of"),
    "email": ("email", "send mail", "inbox", "smtp"),
    "code_execution": ("run code", "execute", "interpreter", "sandbox"),
}


def _mentions(text: str, needles: tuple) -> bool:
    """
    Whether ``text`` contains any needle, respecting word boundaries for single words.

    The boundary rule is what stops "if" matching "specific" and "then" matching
    "strengthen" - both of which made the previous keyword matching fire on almost every
    input, so every requirement looked like it needed branching.
    """
    for needle in needles:
        if " " in needle or "-" in needle:
            if needle in text:
                return True
        elif re.search(rf"\b{re.escape(needle)}\b", text):
            return True
    return False


def _count_matches(text: str, needles: tuple) -> int:
    return sum(1 for n in needles if _mentions(text, (n,)))


def heuristic_analysis(requirement: str) -> RequirementAnalysis:
    """
    Structural reading of a requirement with no model call.

    Used when there is no provider, and as the gap-filler for a model that answered
    partially. It is a genuine analysis rather than a stub: the pipeline can run entirely
    offline on it, which is what makes the whole product testable without credentials.
    """
    text = (requirement or "").strip()
    lowered = text.lower()

    roles = [role for role, hints in _ROLE_HINTS.items() if _mentions(lowered, hints)]
    tools = [tool for tool, hints in _TOOL_HINTS.items() if _mentions(lowered, hints)]

    # Two roles rather than one: almost every useful agent system separates gathering
    # information from producing the answer, and a single-role "assistant" is the config
    # that generated the least useful code.
    if len(roles) < 2:
        roles = list(dict.fromkeys(roles + ["researcher", "writer"]))[:2]
    if not tools:
        tools = ["web_search"]

    sequential = _mentions(lowered, _SIGNALS["sequential"]) or len(roles) > 1
    branching = _count_matches(lowered, _SIGNALS["branching"]) >= 2
    delegation = _mentions(lowered, _SIGNALS["delegation"]) or len(roles) >= 4
    stateful = _mentions(lowered, _SIGNALS["state"])
    human = _mentions(lowered, _SIGNALS["human"])

    signal_count = sum([sequential, branching, delegation, stateful, human])
    word_count = len(lowered.split())
    if signal_count >= 3 or len(roles) >= 4 or word_count > 90:
        complexity = "complex"
    elif signal_count <= 1 and len(roles) <= 2 and word_count < 25:
        complexity = "simple"
    else:
        complexity = "moderate"

    summary = text if len(text) <= 200 else text[:197].rstrip() + "..."

    return RequirementAnalysis(
        requirement=text,
        summary=summary,
        goals=_split_goals(text),
        suggested_roles=roles,
        suggested_tools=tools[:5],
        needs_sequential_steps=sequential,
        needs_branching=branching,
        needs_delegation=delegation,
        needs_state=stateful,
        needs_human_input=human,
        complexity=complexity,
        risks=[],
        from_model=False,
    )


def _split_goals(text: str) -> List[str]:
    """
    Break a requirement into candidate goals on explicit separators only.

    Explicit list markers, "and then", and sentence boundaries are real signals of
    separate goals. Splitting on bare "and" is not - it appears inside single goals far
    more often than between two - so it is left alone.
    """
    if not text:
        return []
    parts = re.split(r"(?:\n\s*[-*\d.)]+\s*)|(?:\band then\b)|(?:[.;]\s+)", text)
    goals = [p.strip(" -*\t.") for p in parts if p and len(p.strip()) > 12]
    return goals[:6] if goals else ([text.strip()] if text.strip() else [])
