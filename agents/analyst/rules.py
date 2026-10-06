"""Deterministic parts of the analyst: merging search results, checking the LLM's
table choices, and deciding the answer status."""

import json
from collections.abc import Iterable, Sequence
from typing import Any

from agents.analyst.schemas import Candidate, PartAssessment, Status
from agents.ingestion.schemas import ColumnInfo
from services.vector_store import Match


def candidate_from_match(match: Match) -> Candidate:
    """Rebuild a table's metadata from what ingestion stored in Pinecone."""
    m: dict[str, Any] = match.metadata
    columns = [ColumnInfo.model_validate(c) for c in json.loads(m["columns_json"])]
    return Candidate(
        table_id=match.id,
        score=match.score,
        name=m["name"],
        description=m["description"],
        filename=m["filename"],
        pages=[int(p) for p in m["pages"]],
        columns=columns,
        row_header_column=m.get("row_header_column"),  # absent when the table has none
        row_headers=list(m.get("row_headers", [])),
        n_rows=int(m["n_rows"]),
        preview=m.get("preview", ""),
    )


def merge_matches(results: Iterable[Sequence[Match]], limit: int) -> list[Match]:
    """Combine the results of several searches: each table once, with its best score,
    best first. A table found by two searches isn't counted twice."""
    best: dict[str, Match] = {}
    for matches in results:
        for match in matches:
            if match.id not in best or match.score > best[match.id].score:
                best[match.id] = match
    return sorted(best.values(), key=lambda m: m.score, reverse=True)[:limit]


def check_assessments(
    parts: list[PartAssessment], plan_questions: list[str], candidate_ids: set[str]
) -> list[PartAssessment]:
    """Keep the LLM's table choices honest.

    Ids that aren't among the candidates are dropped (an LLM can invent or mistype
    them). A part marked answerable that ends up with no tables can't be answered.
    When there's one assessment per planned part, each part's text is taken from the
    plan, so it's exactly what the user saw planned.
    """
    if len(parts) == len(plan_questions):
        parts = [p.model_copy(update={"question": q}) for p, q in zip(parts, plan_questions, strict=True)]
    checked: list[PartAssessment] = []
    for part in parts:
        ids = [i for i in dict.fromkeys(part.table_ids) if i in candidate_ids]  # dedupe, keep order
        if not part.answerable:
            # Keep its related tables: the answer can say what they do contain.
            # selected_table_ids() keeps them out of code generation.
            checked.append(part.model_copy(update={"table_ids": ids}))
        elif ids:
            checked.append(part.model_copy(update={"table_ids": ids, "missing": None}))
        else:
            checked.append(
                part.model_copy(
                    update={"table_ids": [], "answerable": False, "missing": "No indexed table contains this data."}
                )
            )
    return checked


def decide_status(parts: Sequence[PartAssessment]) -> Status:
    """§6.4: all parts computable -> answered; none -> cannot_answer; else partial."""
    answerable = sum(p.answerable for p in parts)  # True counts as 1
    if parts and answerable == len(parts):
        return "answered"
    if answerable == 0:
        return "cannot_answer"
    return "partial"


def selected_table_ids(parts: Sequence[PartAssessment]) -> list[str]:
    """Tables the code will load: those needed by the answerable parts, in order."""
    return list(dict.fromkeys(i for p in parts if p.answerable for i in p.table_ids))
