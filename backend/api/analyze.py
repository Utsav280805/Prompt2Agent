# backend/api/analyze.py
"""
Previewing what the pipeline understood, before committing to a run.

A generation run costs real money and several minutes, and the most expensive failure mode is
discovering at the end that the system misread the requirement. This endpoint runs only the
analysis stage and returns the roles, tools and complexity it inferred, so the user can fix an
ambiguous sentence *before* paying for the rest.

``use_model: false`` runs the offline heuristic instead - instant, free, no credential. That
is what makes the create form usable on a fresh install where no key is configured yet: the
preview still works, and ``from_model: false`` in the response tells the UI to label it as the
offline reading rather than passing a keyword-matcher off as model output.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from multi_agent_generator.core.service import GeneratorService

from ..deps import service
from ..schemas import AnalyzeRequest

router = APIRouter(tags=["analyze"])


@router.post("/analyze", summary="Analyse a requirement without generating")
def analyze(
    payload: AnalyzeRequest,
    svc: GeneratorService = Depends(service),
) -> Dict[str, Any]:
    analysis = svc.analyze(
        payload.requirement,
        use_model=payload.use_model,
        provider=payload.provider,
        model=payload.model,
    )
    return analysis.as_dict()
