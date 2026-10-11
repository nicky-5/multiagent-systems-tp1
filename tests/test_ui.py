"""Smoke tests of the Streamlit pages, run headless with Streamlit's AppTest.

No LLM, Pinecone or network calls: the analyst is replaced by a fake, and the table
store points at a temporary folder.
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from agents.analyst.graph import TableReview
from agents.analyst.schemas import Candidate, CodeAttempt, PartAssessment
from agents.analyst.state import AnalystState, initial_state
from agents.ingestion.schemas import ColumnInfo, Source, TableRecord
from services.config import get_settings
from services.table_store import table_store
from ui import ask_page

APP = Path(__file__).resolve().parent.parent / "app.py"
TABLE_ID = "0123456789abcdef-p1-t0"
RECORD = TableRecord(
    table_id=TABLE_ID,
    name="Población por provincia",
    description="Población residente. Dos oraciones.",
    columns=[ColumnInfo(name="Provincia", description="x", unit="texto")],
    row_header_column="Provincia",
    source=Source(filename="anuario.pdf", pages=[1, 2]),
    row_headers=["Salta"],
    n_rows=1,
    preview="",
)


@pytest.fixture(autouse=True)
def data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Every test gets an empty table store in a temporary folder."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    monkeypatch.undo()
    get_settings.cache_clear()


def tables_script() -> None:
    from ui import tables_page

    tables_page.render()


def ask_script() -> None:
    from ui import ask_page

    ask_page.render()


def test_app_starts():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception


def test_tables_page_when_empty():
    app = AppTest.from_function(tables_script).run()
    assert not app.exception
    assert "No tables yet" in app.info[0].value


def test_tables_page_lists_stored_tables():
    table_store().save(RECORD, pd.DataFrame({"Provincia": ["Salta"]}))
    app = AppTest.from_function(tables_script).run()

    assert not app.exception
    assert app.subheader[0].value == "anuario.pdf"
    assert app.expander[0].label == "Población por provincia · p. 1, 2 · 1 rows"


def answered_state(question: str) -> AnalystState:
    state = initial_state(question)
    state["status"] = "partial"
    state["answer"] = "La población de Salta en 2023 fue de 1.441.351 personas."
    state["assessments"] = [
        PartAssessment(question="Salta 2023", table_ids=[TABLE_ID], answerable=True, missing=None),
        PartAssessment(question="Salta 2024", table_ids=[TABLE_ID], answerable=False, missing="2024 no está"),
    ]
    state["candidates"] = [
        Candidate(
            table_id=TABLE_ID, score=0.8, name=RECORD.name, description="", filename="anuario.pdf",
            pages=[1, 2], columns=[], row_header_column="Provincia", row_headers=[], n_rows=1, preview="",
        )
    ]
    state["selected_ids"] = [TABLE_ID]
    state["attempts"] = [
        CodeAttempt(approach="x", code="raise KeyError('x')", error="KeyError: 'x'"),
        CodeAttempt(approach="Se filtra Salta.", code="print('{}')", result={"parts": []}, isolation=["timeout"]),
    ]
    return state


def test_ask_page_shows_the_answer_and_its_details(monkeypatch: pytest.MonkeyPatch):
    table_store().save(RECORD, pd.DataFrame({"Provincia": ["Salta"]}))
    def fake_ask(question: str, on_progress: object = None) -> AnalystState:
        return answered_state(question)

    monkeypatch.setattr(ask_page, "ask", fake_ask)

    app = AppTest.from_function(ask_script).run()
    app.chat_input[0].set_value("¿Población de Salta?").run()

    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "1.441.351" in text
    assert "2024 no está" in text  # the missing data is listed
    labels = [e.label for e in app.expander]
    assert "Generated code (1 correction)" in labels
    assert "Selected tables (1)" in labels


def test_ask_page_shows_errors_instead_of_crashing(monkeypatch: pytest.MonkeyPatch):
    def failing(question: str, on_progress: object = None) -> AnalystState:
        raise ConnectionError("Pinecone is unreachable")

    monkeypatch.setattr(ask_page, "ask", failing)
    app = AppTest.from_function(ask_script).run()
    app.chat_input[0].set_value("¿Población de Salta?").run()

    assert not app.exception
    assert "ConnectionError: Pinecone is unreachable" in app.error[0].value


def ingest_script() -> None:
    from ui import ingest_page

    ingest_page.render()


def test_ingest_page_shows_a_status_card_per_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from agents.ingestion.graph import IngestionOutcome, ProgressCallback
    from ui import ingest_page

    def fake_ingest(pdf_path: str, on_progress: ProgressCallback | None = None) -> IngestionOutcome:
        assert on_progress is not None
        on_progress("extract", "extract", "2 table(s) found")
        on_progress("table 1/2 (p. [1])", "pending", "waiting to start")
        on_progress("table 2/2 (p. [2])", "pending", "waiting to start")
        on_progress("table 1/2 (p. [1])", "enrich", "asking the LLM for metadata (attempt 1)")
        on_progress("table 2/2 (p. [2])", "skipped", "skipped")
        on_progress("table 1/2 (p. [1])", "done", "done: '<Población>'")
        on_progress("index", "index", "saved and indexed 1 table(s)")
        return IngestionOutcome(filename="anuario.pdf", records=[RECORD], warnings=[])

    (tmp_path / "anuario.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(ingest_page, "SAMPLE_PDFS", tmp_path)
    monkeypatch.setattr(ingest_page, "ingest", fake_ingest)
    app = AppTest.from_function(ingest_script).run()
    app.button[1].click().run()  # ingest the sample PDFs

    assert not app.exception
    cards = next(m.value for m in app.markdown if "ing-grid" in m.value)
    assert cards.count('class="ing-card"') == 2
    assert "Indexed" in cards and "Skipped" in cards
    assert "&lt;Población&gt;" in cards  # PDF and LLM text is escaped


def test_ask_page_pauses_for_review_and_resumes_with_the_chosen_tables(monkeypatch: pytest.MonkeyPatch):
    other = RECORD.model_copy(update={"table_id": "0123456789abcdef-p3-t0", "name": "Ventas por provincia"})
    table_store().save(RECORD, pd.DataFrame({"Provincia": ["Salta"]}))
    table_store().save(other, pd.DataFrame({"Provincia": ["Salta"]}))
    resumed_with: list[list[str]] = []

    def fake_ask_with_review(question: str, on_progress: object = None) -> TableReview:
        return TableReview(thread_id="t1", state=answered_state(question))

    def fake_resume(review: TableReview, table_ids: list[str], on_progress: object = None) -> AnalystState:
        resumed_with.append(table_ids)
        state = answered_state(review.state["question"])
        state["user_table_ids"], state["review_result"] = table_ids, "changed"
        return state

    monkeypatch.setattr(ask_page, "ask_with_review", fake_ask_with_review)
    monkeypatch.setattr(ask_page, "resume_review", fake_resume)
    app = AppTest.from_function(ask_script).run()
    app.toggle(key="review_tables").set_value(True).run()
    app.chat_input[0].set_value("¿Población de Salta?").run()

    assert not app.exception
    assert app.chat_input[0].disabled  # no new question while one waits for review
    tables = app.multiselect[0]
    assert tables.value == [TABLE_ID]  # what the analyst selected
    tables.set_value([other.table_id])
    next(b for b in app.button if b.label == "Continue").click().run()

    assert not app.exception
    assert resumed_with == [[other.table_id]]
    assert not app.multiselect  # the form is gone, the answer is in the history
    assert any("Tables chosen by you" in c.value for c in app.caption)
    assert "1.441.351" in " ".join(m.value for m in app.markdown)
