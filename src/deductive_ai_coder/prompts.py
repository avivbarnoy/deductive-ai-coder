from __future__ import annotations

import json
from typing import Iterable

from .models import ProtocolSnapshot


MAX_LITERATURE_CHARS = 60_000


def codebook_as_text(snapshot: ProtocolSnapshot) -> str:
    lines: list[str] = []
    for code in snapshot.codes:
        lines.append(f"ID: {code.code_id}")
        lines.append(f"Name: {code.name}")
        if code.parent_id:
            lines.append(f"Parent: {code.parent_id}")
        if code.definition:
            lines.append(f"Definition: {code.definition}")
        if code.include_when:
            lines.append(f"Include when: {code.include_when}")
        if code.exclude_when:
            lines.append(f"Exclude when: {code.exclude_when}")
        lines.append("")
    return "\n".join(lines).strip()


def coding_prompt(
    snapshot: ProtocolSnapshot,
    response_text: str,
    contextual_data: dict[str, object] | None = None,
) -> str:
    mode_instruction = (
        "Assign exactly one code."
        if snapshot.coding_mode == "single"
        else "Assign one or more codes only when each is independently supported."
    )
    context_json = json.dumps(contextual_data or {}, ensure_ascii=False, default=str)

    return f"""
You are applying a researcher-defined deductive qualitative coding protocol.
Your task is classification, not theory generation.

PROJECT: {snapshot.project_name}
RESEARCH QUESTION: {snapshot.research_question or 'Not specified'}
UNIT OF ANALYSIS: {snapshot.unit_of_analysis or 'Not specified'}
STUDY CONTEXT: {snapshot.study_context or 'Not specified'}

GENERAL CODING INSTRUCTIONS:
{snapshot.general_instructions}

CODING MODE:
{mode_instruction}

CODEBOOK:
{codebook_as_text(snapshot)}

CASE CONTEXT (may be empty):
{context_json}

TEXT TO CODE:
{response_text}

Return only the exact value(s) shown after `ID:` in the codebook.
Do not add words such as "CODE", labels, explanations, punctuation, or new identifiers.
""".strip()


def explanation_prompt(
    snapshot: ProtocolSnapshot,
    response_text: str,
    assigned_codes: list[str],
    user_question: str,
    contextual_data: dict[str, object] | None = None,
) -> str:
    context_json = json.dumps(contextual_data or {}, ensure_ascii=False, default=str)
    return f"""
You are explaining a prior deductive coding decision to a researcher.
Do not change the code unless explicitly asked to reconsider it. Explain the decision using the supplied protocol and case only.

PROJECT: {snapshot.project_name}
GENERAL INSTRUCTIONS: {snapshot.general_instructions}
CODEBOOK:
{codebook_as_text(snapshot)}

CASE CONTEXT:
{context_json}

TEXT:
{response_text}

ASSIGNED CODE(S): {', '.join(assigned_codes)}

RESEARCHER QUESTION:
{user_question}

Answer concisely and refer to the code definitions/boundaries that matter.
""".strip()


def codesign_prompt(
    snapshot: ProtocolSnapshot,
    user_request: str,
    literature_texts: Iterable[str] | None = None,
    calibration_evidence: str = "",
) -> str:
    literature = "\n\n--- SOURCE SEPARATOR ---\n\n".join(literature_texts or [])
    if len(literature) > MAX_LITERATURE_CHARS:
        literature = literature[:MAX_LITERATURE_CHARS] + "\n[TRUNCATED BY APP]"

    return f"""
You are helping a researcher refine a DEDUCTIVE qualitative coding protocol.
The researcher remains the methodological decision-maker. Do not silently add theoretical claims.

CURRENT PROJECT
Project: {snapshot.project_name}
Research question: {snapshot.research_question or 'Not specified'}
Unit of analysis: {snapshot.unit_of_analysis or 'Not specified'}
Study context: {snapshot.study_context or 'Not specified'}
Coding mode: {snapshot.coding_mode}
Hierarchy allowed: {snapshot.allow_hierarchy}

CURRENT GENERAL INSTRUCTIONS
{snapshot.general_instructions}

CURRENT CODEBOOK
{codebook_as_text(snapshot) or '[No codes defined yet]'}

OPTIONAL LITERATURE / SOURCE MATERIAL
{literature or '[None supplied]'}

OPTIONAL CALIBRATION EVIDENCE
{calibration_evidence or '[None supplied]'}

RESEARCHER REQUEST
{user_request}

Return a complete revised draft of the codebook and coding instructions, plus a short message explaining the proposed changes.
Preserve existing codes unless the researcher's request or evidence gives a reason to modify, merge, split, add, or remove them.
Code IDs are strings and may use any clear scheme.
If hierarchy is disabled, do not use parent IDs.
""".strip()
