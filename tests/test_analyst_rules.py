from typing import Any

import pytest

from agents.analyst.rules import (
    apply_result,
    candidate_from_match,
    check_assessments,
    check_result,
    decide_status,
    format_value,
    mark_failed,
    merge_matches,
    selected_table_ids,
    sources_footer,
    tables_labelling,
)
from agents.analyst.schemas import Candidate, PartAssessment
from agents.ingestion.rules import pinecone_metadata
from agents.ingestion.schemas import ColumnInfo, Source, TableRecord
from services.vector_store import Match


def part(answerable: bool, table_ids: list[str] | None = None, question: str = "q") -> PartAssessment:
    return PartAssessment(
        question=question,
        table_ids=table_ids if table_ids is not None else ["t1"],
        answerable=answerable,
        missing=None if answerable else "2024 is not available",
    )


# --- status ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answerable", "status"),
    [
        ([True], "answered"),
        ([True, True], "answered"),
        ([True, False], "partial"),
        ([False], "cannot_answer"),
        ([False, False], "cannot_answer"),
        ([], "cannot_answer"),  # nothing planned: nothing can be answered
    ],
)
def test_decide_status(answerable: list[bool], status: str):
    assert decide_status([part(a) for a in answerable]) == status


# --- checking the LLM's choices ----------------------------------------------------


def test_unknown_table_ids_are_dropped():
    checked = check_assessments([part(True, ["t1", "invented", "t1"])], ["q"], {"t1", "t2"})
    assert checked[0].table_ids == ["t1"]  # unknown id dropped, duplicate removed
    assert checked[0].answerable


def test_answerable_part_without_valid_tables_becomes_unanswerable():
    checked = check_assessments([part(True, ["invented"])], ["q"], {"t1"})
    assert not checked[0].answerable
    assert checked[0].missing == "No indexed table contains this data."
    assert decide_status(checked) == "cannot_answer"


def test_part_text_comes_from_the_plan():
    llm_parts = [part(True, question="q1 (metrics: ['x'])"), part(False, question="q2")]
    checked = check_assessments(llm_parts, ["q1", "q2"], {"t1"})
    assert [p.question for p in checked] == ["q1", "q2"]


def test_only_answerable_parts_select_tables():
    parts = [part(True, ["t1", "t2"]), part(False, ["t3"]), part(True, ["t2"])]
    assert selected_table_ids(parts) == ["t1", "t2"]


# --- retrieval --------------------------------------------------------------------


def test_merge_matches_keeps_best_score_per_table():
    first = [Match("a", 0.5, {}), Match("b", 0.4, {})]
    second = [Match("b", 0.9, {}), Match("c", 0.1, {})]
    merged = merge_matches([first, second], limit=2)
    assert [(m.id, m.score) for m in merged] == [("b", 0.9), ("a", 0.5)]


def test_candidate_round_trips_through_pinecone_metadata():
    record = TableRecord(
        table_id="0123456789abcdef-p1-t0",
        name="Ventas",
        description="Una tabla. Dos oraciones.",
        columns=[ColumnInfo(name="Provincia", description="Provincia", unit="text")],
        row_header_column=None,
        source=Source(filename="informe.pdf", pages=[1, 2]),
        row_headers=[],
        n_rows=24,
        preview="| Provincia |",
    )
    match = Match(record.table_id, 0.8, dict(pinecone_metadata(record, "text")))
    candidate = candidate_from_match(match)

    assert candidate.pages == [1, 2]  # stored as strings, read back as ints
    assert candidate.columns == record.columns  # stored as JSON, read back as models
    assert candidate.row_header_column is None  # stored by leaving the key out
    assert candidate.score == 0.8


# --- the program's output ------------------------------------------------------------


def entry(value: object = 1, missing: str | None = None) -> dict[str, object]:
    return {"value": value, "unit": "x", "missing": missing}


def test_valid_result():
    assert check_result({"parts": [entry(), entry({"a": [1.5, "b"]})]}, 2) == []
    assert check_result({"parts": [entry(None, "Rosario no está")]}, 1) == []
    assert check_result({"parts": [{"value": "Formosa", "unit": "", "missing": None}]}, 1) == []


@pytest.mark.parametrize(
    ("result", "fragment"),
    [
        ({"answer": 1}, 'must have a "parts" list'),
        ({"parts": [entry()]}, "it must have 2"),
        ({"parts": [{"value": 1}, entry()]}, 'Part 1 must be an object with the keys'),
        ({"parts": [entry(None), entry()]}, "Part 1 has no value"),
        ({"parts": [entry(float("nan")), entry()]}, "NaN or Infinity"),
        ({"parts": [entry({"Salta": float("inf")}), entry()]}, "NaN or Infinity"),
        ({"parts": [entry("Formosa"), entry()]}, "so its number is missing"),
    ],
)
def test_invalid_result(result: dict[str, Any], fragment: str):
    assert any(fragment in e for e in check_result(result, 2))


def test_apply_result_lowers_parts_reported_missing():
    parts = [part(True), part(False), part(True)]
    result = {"parts": [entry(5), entry(None, "Rosario no está")]}  # entries for parts 1 and 3
    updated = apply_result(parts, result)

    assert [p.answerable for p in updated] == [True, False, False]
    assert updated[2].missing == "Rosario no está"
    assert decide_status(updated) == "partial"


def test_mark_failed():
    updated = mark_failed([part(True), part(False)], attempts=3)
    assert [p.answerable for p in updated] == [False, False]
    assert updated[0].missing == "The calculation failed after 3 attempts."
    assert updated[1].missing == "2024 is not available"  # unchanged


# --- the answer ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "language", "text"),
    [
        (1234567, "es", "1.234.567"),
        (1234567, "en", "1,234,567"),
        (1234.5, "es", "1.234,5"),
        (0.0864623, "en", "0.0865"),  # up to 4 decimals
        (2.0, "en", "2"),  # no trailing zeros
        ({"Salta": 1500.25}, "es", "{Salta: 1.500,25}"),
        ([1, None, "x"], "en", "[1, null, x]"),
    ],
)
def test_format_value(value: object, language: str, text: str):
    assert format_value(value, language) == text


def test_sources_footer():
    table = Candidate(
        table_id="t1", score=0.9, name="Población", description="", filename="anuario.pdf",
        pages=[3, 4], columns=[], row_header_column=None, row_headers=[], n_rows=1, preview="",
    )
    assert sources_footer([table], "es", used=True) == "**Tablas utilizadas:**\n- Población (anuario.pdf, pág. 3, 4)"
    assert sources_footer([table], "en", used=False) == "**Tables consulted:**\n- Población (anuario.pdf, p. 3, 4)"
    assert sources_footer([], "en", used=True) == ""


def test_tables_labelling_matches_row_labels_ignoring_case_and_accents():
    def record(table_id: str, labels: list[str]) -> TableRecord:
        return TableRecord(
            table_id=table_id, name="t", description="d", columns=[], row_header_column="Área",
            source=Source(filename="f.pdf", pages=[1]), row_headers=labels, n_rows=len(labels), preview="",
        )

    records = [record("a", ["Gran Rosario", "Gran Córdoba"]), record("b", ["Cuyo"]), record("c", [])]
    assert tables_labelling(["gran cordoba"], records) == ["a"]
    assert tables_labelling(["Cuyo", "Gran Rosario"], records) == ["a", "b"]
    assert tables_labelling(["Rosario", ""], records) == []  # whole labels only
