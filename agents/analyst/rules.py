"""Deterministic parts of the analyst: merging search results, checking the LLM's
table choices and the program's output, deciding the answer status, and citing sources."""

import json
import math
from collections.abc import Iterable, Sequence
from typing import Any, cast

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


# --- checking the program's output ---------------------------------------------------


def check_result(result: dict[str, Any], n_parts: int) -> list[str]:
    """Problems with the JSON a program printed, worded for the LLM to fix; empty if valid.

    Expected: {"parts": [{"value": ..., "unit": "...", "missing": null or "..."}, ...]},
    one entry per part being computed, in order.
    """
    raw = result.get("parts")
    if not isinstance(raw, list):
        return ['The JSON object must have a "parts" list, one entry per part.']
    entries = cast(list[object], raw)
    errors: list[str] = []
    if len(entries) != n_parts:
        errors.append(f'"parts" has {len(entries)} entries; it must have {n_parts}, one per part, in order.')
    for i, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not {"value", "unit", "missing"} <= entry.keys():
            errors.append(f'Part {i} must be an object with the keys "value", "unit" and "missing".')
            continue
        part = cast(dict[str, object], entry)
        if part["missing"] is None and part["value"] is None:
            errors.append(f'Part {i} has no value; if the data isn\'t there, explain it in "missing".')
    if _has_non_finite(result):
        errors.append(
            "The result contains NaN or Infinity, which usually means a value wasn't found "
            'or a division by zero. Report data that isn\'t there in "missing" instead.'
        )
    return errors


def _has_non_finite(value: object) -> bool:
    """NaN or Infinity anywhere in a JSON value (Python's json module accepts both)."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_non_finite(v) for v in cast(dict[str, object], value).values())
    if isinstance(value, list):
        return any(_has_non_finite(v) for v in cast(list[object], value))
    return False


def apply_result(parts: Sequence[PartAssessment], result: dict[str, Any]) -> list[PartAssessment]:
    """Mark the parts the program couldn't compute (it reported data as missing) as not
    answerable. A valid result has one entry per answerable part, in order."""
    entries = iter(cast(list[dict[str, Any]], result["parts"]))
    updated: list[PartAssessment] = []
    for part in parts:
        if not part.answerable:
            updated.append(part)
            continue
        missing = next(entries)["missing"]
        if missing is None:
            updated.append(part)
        else:
            updated.append(part.model_copy(update={"answerable": False, "missing": str(missing)}))
    return updated


def mark_failed(parts: Sequence[PartAssessment], attempts: int) -> list[PartAssessment]:
    """After the last program also failed, nothing could be computed."""
    reason = f"The calculation failed after {attempts} attempts."
    return [
        p.model_copy(update={"answerable": False, "missing": reason}) if p.answerable else p
        for p in parts
    ]


# --- the answer --------------------------------------------------------------------


def sources_footer(tables: Sequence[Candidate], language: str, used: bool) -> str:
    """The tables behind the answer, with file and pages, written by code so the
    citation is always exact."""
    if not tables:
        return ""
    if language == "es":
        title, pages = ("Tablas utilizadas" if used else "Tablas consultadas"), "pág."
    else:
        title, pages = ("Tables used" if used else "Tables consulted"), "p."
    lines = [f"- {t.name} ({t.filename}, {pages} {', '.join(str(p) for p in t.pages)})" for t in tables]
    return f"**{title}:**\n" + "\n".join(lines)


def format_value(value: object, language: str) -> str:
    """A computed value as text for the answer, with the language's number format:
    1.234.567,89 in Spanish, 1,234,567.89 in English. Up to 4 decimals, no trailing zeros."""
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, int | float):
        text = f"{value:,.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"
        if language == "es":  # swap the separators: 1,234.5 -> 1.234,5
            text = text.replace(",", "_").replace(".", ",").replace("_", ".")
        return text
    if isinstance(value, dict):
        items = cast(dict[str, object], value).items()
        return "{" + ", ".join(f"{k}: {format_value(v, language)}" for k, v in items) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(format_value(v, language) for v in cast(list[object], value)) + "]"
    return str(value)
