import pytest

from agents.analyst.rules import (
    candidate_from_match,
    check_assessments,
    decide_status,
    merge_matches,
    selected_table_ids,
)
from agents.analyst.schemas import PartAssessment
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
