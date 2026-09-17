# multi_agent_generator/core/selection.py
"""
Stage 2: choose the framework.

Three rules, in order of precedence:

1. If the user named a framework, that is the answer. Confidence 1.0, no model call, no
   second-guessing. A tool that overrides an explicit instruction because it thinks it
   knows better is a tool people stop trusting.
2. Otherwise score each framework against the structural analysis, using what the
   frameworks are actually good at.
3. Ask a model to check the scored recommendation, and let it override with a reason.

Rule 2 is not a fallback for rule 3 - it runs first and always. That ordering matters: the
scored result is a defensible answer on its own, so the model is being asked to improve a
decision rather than to make one from nothing, and a model that returns prose costs the
pipeline nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..errors import AppError, UnsupportedCombinationError
from ..frameworks import FRAMEWORKS
from ..llm.base import LLMProvider
from .models import FrameworkChoice, RequirementAnalysis
from .parsing import as_float, extract_json_object

__all__ = [
    "select_framework",
    "score_frameworks",
    "FRAMEWORK_PROFILES",
    "SELECTION_SYSTEM_PROMPT",
]


@dataclass(frozen=True)
class FrameworkProfile:
    """What a framework is for, in terms the selector can reason about."""

    name: str
    label: str
    summary: str
    #: Structural properties this framework handles well.
    strengths: Tuple[str, ...]
    #: Properties it handles awkwardly, if at all.
    weaknesses: Tuple[str, ...]
    #: Baseline preference, breaking ties between otherwise equal candidates. CrewAI leads
    #: because role-based sequential work is by far the most common request.
    base_score: float = 0.0


#: Every framework the generator can emit, with the properties that should select it.
#: Adding a framework means adding an entry here - the selector has no per-name special
#: cases, so nothing else needs to change.
FRAMEWORK_PROFILES: Dict[str, FrameworkProfile] = {
    "crewai": FrameworkProfile(
        name="crewai",
        label="CrewAI",
        summary=(
            "Role-playing agents with explicit goals and backstories, executing ordered "
            "tasks as a crew. The most direct fit for 'a team of specialists produces a "
            "deliverable'."
        ),
        strengths=("roles", "sequential", "delegation", "collaboration"),
        weaknesses=("branching", "state", "cycles"),
        base_score=1.0,
    ),
    "crewai-flow": FrameworkProfile(
        name="crewai-flow",
        label="CrewAI Flow",
        summary=(
            "Event-driven CrewAI: steps are triggered by decorated listeners, so routing "
            "and conditional continuation are first-class while keeping crew-style roles."
        ),
        strengths=("roles", "sequential", "branching", "events", "state"),
        weaknesses=("cycles",),
    ),
    "langgraph": FrameworkProfile(
        name="langgraph",
        label="LangGraph",
        summary=(
            "An explicit state graph. Nodes mutate typed shared state and edges may be "
            "conditional or cyclic, which makes retries, loops and human checkpoints "
            "expressible rather than simulated."
        ),
        strengths=("branching", "state", "cycles", "human_input", "sequential"),
        weaknesses=("simplicity",),
    ),
    "react": FrameworkProfile(
        name="react",
        label="LangChain ReAct agent",
        summary=(
            "One agent that reasons and calls tools in a loop via AgentExecutor. The right "
            "shape when the work is 'answer questions using these tools', not 'coordinate "
            "a team'."
        ),
        strengths=("tools", "simplicity", "single_agent"),
        weaknesses=("delegation", "roles", "state"),
    ),
    "react-lcel": FrameworkProfile(
        name="react-lcel",
        label="LangChain (LCEL chain)",
        summary=(
            "A composed prompt -> model -> parser chain. Lowest overhead and easiest to "
            "reason about, for straight-through transformations with no agency."
        ),
        strengths=("simplicity", "single_agent", "deterministic"),
        weaknesses=("delegation", "branching", "state", "tools"),
    ),
    "agno": FrameworkProfile(
        name="agno",
        label="Agno",
        summary=(
            "Lightweight agents and coordinated teams with built-in memory. Good when "
            "several agents must share context cheaply."
        ),
        strengths=("roles", "collaboration", "state", "sequential"),
        weaknesses=("branching", "cycles"),
    ),
}

#: Weight per structural property. Branching and cycles dominate because getting those
#: wrong forces the generated code to fake control flow, which is the failure that
#: produces unrunnable projects; roles and tools are cheaper to approximate.
_WEIGHTS: Dict[str, float] = {
    "branching": 3.0,
    "cycles": 2.5,
    "state": 2.0,
    "delegation": 2.0,
    "human_input": 1.5,
    "roles": 1.5,
    "sequential": 1.0,
    "tools": 1.0,
    "collaboration": 1.0,
    "single_agent": 1.5,
    "simplicity": 1.0,
    "deterministic": 0.5,
    "events": 1.0,
}


SELECTION_SYSTEM_PROMPT = """\
You choose the best framework for building a multi-agent system. Reply with ONLY a JSON
object, no prose and no code fence:

{"framework": "<name>", "reason": "<one or two sentences>", "confidence": 0.0-1.0}

Choose from exactly these names:
{framework_list}

Guidance:
- langgraph: conditional edges, loops, retries, typed shared state, human checkpoints.
- crewai: a team of role-playing specialists producing a deliverable in a fixed order.
- crewai-flow: crew-style roles but the path through the work depends on results.
- react: a single agent that answers questions by calling tools in a reasoning loop.
- react-lcel: a straight-through transformation with no agency and no tools.
- agno: several lightweight agents sharing memory and context cheaply.

A recommendation has already been computed from a structural analysis. Agree with it
unless it is clearly wrong; if you disagree, say specifically which structural requirement
it fails to serve. Set confidence below 0.5 if the request is too vague to tell.
"""


def _properties(analysis: RequirementAnalysis) -> Dict[str, bool]:
    """Turn the analysis into the property vocabulary the profiles are written in."""
    role_count = len(analysis.suggested_roles)
    return {
        "sequential": analysis.needs_sequential_steps,
        "branching": analysis.needs_branching,
        "cycles": analysis.needs_branching and analysis.needs_state,
        "state": analysis.needs_state,
        "delegation": analysis.needs_delegation,
        "human_input": analysis.needs_human_input,
        "roles": role_count >= 2,
        "collaboration": role_count >= 3,
        "tools": bool(analysis.suggested_tools),
        "single_agent": role_count <= 1,
        "simplicity": analysis.complexity == "simple",
        "deterministic": analysis.complexity == "simple"
        and not analysis.suggested_tools,
        "events": analysis.needs_branching and analysis.needs_sequential_steps,
    }


def score_frameworks(
    analysis: RequirementAnalysis,
    allowed: Optional[List[str]] = None,
) -> List[Tuple[str, float, List[str]]]:
    """
    Score every candidate framework, best first.

    Returns ``(name, score, matched_properties)`` per framework. Scoring rewards a strength
    that the requirement needs and penalises a weakness that the requirement exercises -
    the penalty is what stops "supports roles" alone from selecting CrewAI for work that is
    fundamentally a loop.
    """
    properties = _properties(analysis)
    needed = {key for key, present in properties.items() if present}
    candidates = allowed or list(FRAMEWORK_PROFILES)

    scored: List[Tuple[str, float, List[str]]] = []
    for name in candidates:
        profile = FRAMEWORK_PROFILES.get(name)
        if profile is None:
            continue
        score = profile.base_score
        matched: List[str] = []
        for strength in profile.strengths:
            if strength in needed:
                score += _WEIGHTS.get(strength, 1.0)
                matched.append(strength)
        for weakness in profile.weaknesses:
            if weakness in needed:
                score -= _WEIGHTS.get(weakness, 1.0) * 0.9
        scored.append((name, round(score, 2), matched))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def _confidence_from_scores(scored: List[Tuple[str, float, List[str]]]) -> float:
    """
    Turn a score gap into a confidence.

    Confidence is about *separation*, not magnitude: a top score of 9 that the runner-up
    also reaches means the requirement does not distinguish them, and reporting 0.9 there
    would be a lie. A clear gap earns high confidence even if both scores are modest.
    """
    if not scored:
        return 0.0
    if len(scored) == 1:
        return 0.75
    best, runner_up = scored[0][1], scored[1][1]
    spread = max(1.0, abs(best))
    gap = (best - runner_up) / spread
    return max(0.35, min(0.95, 0.5 + gap))


def _explain(name: str, matched: List[str], analysis: RequirementAnalysis) -> str:
    """A reason a user can check, naming the requirements that drove the choice."""
    profile = FRAMEWORK_PROFILES[name]
    if matched:
        readable = ", ".join(m.replace("_", " ") for m in matched)
        return (
            f"{profile.label} was selected because this request needs {readable}. "
            f"{profile.summary}"
        )
    return (
        f"{profile.label} was selected as the best general fit for a "
        f"{analysis.complexity} request with {len(analysis.suggested_roles)} distinct "
        f"role(s). {profile.summary}"
    )


def select_framework(
    analysis: RequirementAnalysis,
    requested: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
) -> FrameworkChoice:
    """
    Choose a framework for ``analysis``.

    Args:
        analysis: Structural reading of the requirement.
        requested: An explicit framework name, or None/"auto" to decide.
        provider: LLM used to sanity-check the scored choice. Optional; without it the
            scored result stands on its own.

    Raises:
        UnsupportedCombinationError: if ``requested`` names a framework that does not exist.
            This is raised rather than silently corrected, because generating CrewAI code
            for someone who asked for LangGraph is worse than telling them the name is
            wrong.
    """
    if requested and str(requested).strip().lower() not in ("", "auto", "auto-detect", "any"):
        name = str(requested).strip().lower()
        if name not in FRAMEWORK_PROFILES or name not in FRAMEWORKS:
            raise UnsupportedCombinationError(
                f"Unknown framework {requested!r}.",
                action=f"Choose one of: {', '.join(FRAMEWORKS)}.",
                context={"requested": requested},
            )
        return FrameworkChoice(
            framework=name,
            reason=(
                f"You selected {FRAMEWORK_PROFILES[name].label} explicitly. "
                f"{FRAMEWORK_PROFILES[name].summary}"
            ),
            confidence=1.0,
            user_specified=True,
        )

    scored = score_frameworks(analysis, allowed=[f for f in FRAMEWORKS if f in FRAMEWORK_PROFILES])
    if not scored:
        # Only reachable if the profile table and the generator table have diverged, which
        # is a packaging error rather than a user error.
        raise AppError(
            "No framework is available to generate code with.",
            action="Reinstall multi-agent-generator; its framework registry is incomplete.",
        )

    best_name, _, matched = scored[0]
    choice = FrameworkChoice(
        framework=best_name,
        reason=_explain(best_name, matched, analysis),
        confidence=_confidence_from_scores(scored),
        user_specified=False,
        alternatives=[
            {
                "framework": name,
                "score": score,
                "note": FRAMEWORK_PROFILES[name].summary,
            }
            for name, score, _ in scored[1:4]
        ],
    )

    if provider is None:
        return choice
    return _consult_model(analysis, choice, scored, provider)


def _consult_model(
    analysis: RequirementAnalysis,
    choice: FrameworkChoice,
    scored: List[Tuple[str, float, List[str]]],
    provider: LLMProvider,
) -> FrameworkChoice:
    """
    Let a model confirm or override the scored choice.

    Model failures are absorbed here on purpose: there is already a valid, explained answer,
    so an unreachable provider should downgrade the *quality* of the decision, not fail the
    run. That is different from the generation stage, where there is no non-model answer and
    the error must propagate.
    """
    allowed = [name for name, _, _ in scored]
    prompt = _selection_prompt(analysis, choice, scored)
    system = SELECTION_SYSTEM_PROMPT.replace(
        "{framework_list}", "\n".join(f"- {name}" for name in allowed)
    )

    try:
        raw = provider.complete_text(prompt, system=system, temperature=0.1, max_tokens=400)
    except AppError as exc:
        choice.alternatives.append(
            {"framework": None, "score": None, "note": f"Model review skipped: {exc.message}"}
        )
        return choice

    payload = extract_json_object(raw)
    if not payload:
        return choice

    proposed = str(payload.get("framework") or "").strip().lower()
    reason = str(payload.get("reason") or "").strip()
    confidence = as_float(payload.get("confidence"), choice.confidence)
    # Some models answer with the label rather than the key.
    if proposed not in allowed:
        proposed = next(
            (n for n in allowed if FRAMEWORK_PROFILES[n].label.lower() == proposed), ""
        )
    if proposed not in allowed:
        return choice

    if proposed == choice.framework:
        # Agreement is evidence, so confidence rises - but not to certainty, because a
        # model agreeing with a heuristic that shares its blind spots is weak evidence.
        choice.confidence = max(choice.confidence, min(0.95, confidence))
        if reason:
            choice.reason = reason
        return choice

    scores = {name: score for name, score, _ in scored}
    return FrameworkChoice(
        framework=proposed,
        reason=reason
        or f"{FRAMEWORK_PROFILES[proposed].label} was chosen on review of the requirement.",
        confidence=confidence,
        user_specified=False,
        alternatives=[
            {
                "framework": choice.framework,
                "score": scores.get(choice.framework),
                "note": (
                    "Structural scoring preferred this, but the reviewing model "
                    "disagreed: " + (reason or "no reason given.")
                ),
            }
        ]
        + choice.alternatives[:2],
    )


def _selection_prompt(
    analysis: RequirementAnalysis,
    choice: FrameworkChoice,
    scored: List[Tuple[str, float, List[str]]],
) -> str:
    """Give the model the analysis, the current recommendation and the runners-up."""
    needs = [
        label
        for label, present in (
            ("ordered steps", analysis.needs_sequential_steps),
            ("conditional branching or loops", analysis.needs_branching),
            ("delegation by a coordinator", analysis.needs_delegation),
            ("state carried across steps", analysis.needs_state),
            ("human approval mid-run", analysis.needs_human_input),
        )
        if present
    ]
    lines = [
        f"Requirement: {analysis.requirement}",
        "",
        f"Summary: {analysis.summary or '(none)'}",
        f"Complexity: {analysis.complexity}",
        f"Roles implied: {', '.join(analysis.suggested_roles) or 'none identified'}",
        f"Tools implied: {', '.join(analysis.suggested_tools) or 'none identified'}",
        f"Structural needs: {', '.join(needs) or 'none identified'}",
        "",
        f"Current recommendation: {choice.framework} (score {scored[0][1]})",
        "Runners-up: "
        + ", ".join(f"{name} ({score})" for name, score, _ in scored[1:4]),
    ]
    return "\n".join(lines)
