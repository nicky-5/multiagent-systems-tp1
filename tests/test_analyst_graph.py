"""Run the analyst graph with fakes for the LLM, embeddings and Pinecone.

The generated programs are real and run in the real sandbox, against a small table
in a temporary table store.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from agents.analyst import nodes
from agents.analyst.graph import build_graph
from agents.analyst.schemas import (
    GeneratedCode,
    PartAssessment,
    QueryPlan,
    QuestionPart,
    Search,
    Selection,
)
from agents.analyst.state import AnalystState, initial_state
from agents.ingestion.rules import pinecone_metadata
from agents.ingestion.schemas import ColumnInfo, Source, TableRecord
from services.table_store import TableStore
from services.vector_store import Match

POPULATION = "0123456789abcdef-p1-t0"
SALES = "0123456789abcdef-p2-t0"
DATA = pd.DataFrame({
    "Provincia": pd.Series(["Salta", "Jujuy"], dtype="string"),
    "Población 2023": pd.Series([1_441_351, 811_611], dtype="Int64"),
})


def make_record(table_id: str, name: str) -> TableRecord:
    return TableRecord(
        table_id=table_id,
        name=name,
        description="Una tabla. Dos oraciones.",
        columns=[
            ColumnInfo(name="Provincia", description="x", unit="texto"),
            ColumnInfo(name="Población 2023", description="x", unit="personas"),
        ],
        row_header_column="Provincia",
        source=Source(filename="anuario.pdf", pages=[3, 4]),
        row_headers=["Salta", "Jujuy"],
        n_rows=2,
        preview="",
    )


def plan(*questions: str) -> QueryPlan:
    return QueryPlan(
        language="es",
        parts=[QuestionPart(question=q, metrics=[], entities=[], periods=[], operation="lookup") for q in questions],
        searches=[Search(spanish="población por provincia", english="population by province")],
    )


def assess(answerable: bool, missing: str | None = None) -> PartAssessment:
    return PartAssessment(question="x", table_ids=[POPULATION], answerable=answerable, missing=missing)


def program(*parts: str) -> GeneratedCode:
    """A program that prints the given JSON entries as its result."""
    code = (
        "import json, pandas as pd\n"
        f"df = pd.read_parquet('{POPULATION}.parquet')\n"
        "salta = int(df.loc[df['Provincia'] == 'Salta', 'Población 2023'].iloc[0])\n"
        f"print(json.dumps({{'parts': [{', '.join(parts)}]}}))\n"
    )
    return GeneratedCode(approach="Se filtra la fila de Salta.", code=code)


SALTA = "{'value': salta, 'unit': 'personas', 'missing': None}"
MISSING = "{'value': None, 'unit': '', 'missing': 'Rosario no está en la tabla'}"
CRASH = GeneratedCode(approach="x", code="import pandas as pd\nraise KeyError('Poblacion')\n")
NO_JSON = GeneratedCode(approach="x", code="print('done')\n")


class FakeChatModel:
    """Answers structured-output requests with the scripted object(s) for their schema
    (a list is answered in order, its last item repeating) and plain requests (respond)
    with a fixed text. Records every request."""

    def __init__(self, answers: dict[type[BaseModel], BaseModel | list[BaseModel]]):
        self.answers = answers
        self.asked: list[type[BaseModel] | str] = []
        self.prompts: list[list[BaseMessage]] = []

    def with_structured_output(self, schema: type[BaseModel]) -> RunnableLambda[list[BaseMessage], BaseModel]:
        def answer(messages: list[BaseMessage]) -> BaseModel:
            self.asked.append(schema)
            self.prompts.append(messages)
            scripted = self.answers[schema]
            if isinstance(scripted, list):
                return scripted.pop(0) if len(scripted) > 1 else scripted[0]
            return scripted

        return RunnableLambda(answer)

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.asked.append("respond")
        self.prompts.append(messages)
        return AIMessage("Respuesta.")


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self, matches: list[Match]):
        self.matches = matches

    def query(self, vector: Sequence[float], top_k: int) -> list[Match]:
        return self.matches[:top_k]


def run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    llm: FakeChatModel,
    stored: list[TableRecord],
    indexed: list[TableRecord] | None = None,
    max_attempts: int = 3,
    coder: FakeChatModel | None = None,
) -> AnalystState:
    """stored: tables with data in the table store; indexed: tables in Pinecone
    (defaults to the stored ones); coder: the code-generation model (defaults to llm)."""
    tables = TableStore(tmp_path)
    for record in stored:
        tables.save(record, DATA)
    in_pinecone = indexed if indexed is not None else stored
    matches = [Match(r.table_id, 0.9 - i / 10, dict(pinecone_metadata(r, ""))) for i, r in enumerate(in_pinecone)]

    fakes: dict[str, Callable[[], Any]] = {
        "chat_model": lambda: llm,
        "code_model": lambda: coder or llm,
        "embeddings": FakeEmbeddings,
        "vector_store": lambda: FakeVectorStore(matches),
        "table_store": lambda: tables,
    }
    for name, factory in fakes.items():
        monkeypatch.setattr(nodes, name, factory)
    state_in = initial_state("¿pregunta?", max_attempts)
    output = build_graph().invoke(state_in)  # pyright: ignore[reportUnknownMemberType]
    return cast(AnalystState, output)


# --- retrieval and selection -------------------------------------------------------


def test_answerable_question(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({
        QueryPlan: plan("población de Salta 2023"),
        Selection: Selection(parts=[assess(True)]),
        GeneratedCode: program(SALTA),
    })
    state = run(monkeypatch, tmp_path, llm, [make_record(POPULATION, "Población"), make_record(SALES, "Ventas")])

    assert [c.table_id for c in state["candidates"]] == [POPULATION, SALES]
    assert state["selected_ids"] == [POPULATION]
    assert state["status"] == "answered"
    assert len(state["attempts"]) == 1
    assert state["attempts"][0].result == {"parts": [{"value": 1441351, "unit": "personas", "missing": None}]}
    # The LLM's text, then the citation added by code
    assert state["answer"] == "Respuesta.\n\n**Tablas utilizadas:**\n- Población (anuario.pdf, pág. 3, 4)"


def test_partial_question(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({
        QueryPlan: plan("población de Salta 2023", "población de Salta 2024"),
        Selection: Selection(parts=[assess(True), assess(False, "2024 no está")]),
        GeneratedCode: program(SALTA),  # one entry: only the answerable part is computed
    })
    state = run(monkeypatch, tmp_path, llm, [make_record(POPULATION, "Población")])

    assert state["status"] == "partial"
    assert [a.question for a in state["assessments"]] == ["población de Salta 2023", "población de Salta 2024"]


def test_cannot_answer_skips_code_generation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({
        QueryPlan: plan("población de Salta 2024"),
        Selection: Selection(parts=[assess(False, "La tabla cubre 2022-2023")]),
    })
    state = run(monkeypatch, tmp_path, llm, [make_record(POPULATION, "Población")])

    assert state["status"] == "cannot_answer"
    assert llm.asked == [QueryPlan, Selection, "respond"]  # no program was written or run
    assert state["attempts"] == []
    assert "**Tablas consultadas:**" in state["answer"]  # related table, not "used"


def test_no_candidates_skips_the_selection_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({QueryPlan: plan("desempleo en Rosario")})
    state = run(monkeypatch, tmp_path, llm, stored=[])

    assert state["status"] == "cannot_answer"
    assert llm.asked == [QueryPlan, "respond"]  # select didn't call the LLM


def test_indexed_table_without_data_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({QueryPlan: plan("población de Salta 2023")})
    stale = make_record(SALES, "Ventas")
    state = run(monkeypatch, tmp_path, llm, stored=[], indexed=[stale])

    assert state["candidates"] == []
    assert state["warnings"] == [f"Indexed tables without stored data were ignored: ['{SALES}']"]


# --- code generation, execution and self-correction --------------------------------


def test_failed_program_is_corrected_with_its_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({
        QueryPlan: plan("población de Salta 2023"),
        Selection: Selection(parts=[assess(True)]),
        GeneratedCode: [CRASH, program(SALTA)],
    })
    state = run(monkeypatch, tmp_path, llm, [make_record(POPULATION, "Población")])

    first, second = state["attempts"]
    assert first.error is not None and "KeyError: 'Poblacion'" in first.error
    assert second.error is None
    assert state["status"] == "answered"
    # The second request showed the LLM its failed program and the error
    retry_prompt = str(llm.prompts[3][-1].content)
    assert "raise KeyError('Poblacion')" in retry_prompt and "KeyError: 'Poblacion'" in retry_prompt


def test_invalid_output_counts_as_a_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({
        QueryPlan: plan("población de Salta 2023"),
        Selection: Selection(parts=[assess(True)]),
        GeneratedCode: [program(SALTA, SALTA), program(SALTA)],  # first: 2 entries for 1 part
    })
    state = run(monkeypatch, tmp_path, llm, [make_record(POPULATION, "Población")])

    first, second = state["attempts"]
    assert first.error is not None and "it must have 1" in first.error
    assert second.error is None and state["status"] == "answered"


def test_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({
        QueryPlan: plan("población de Salta 2023"),
        Selection: Selection(parts=[assess(True)]),
        GeneratedCode: NO_JSON,
    })
    state = run(monkeypatch, tmp_path, llm, [make_record(POPULATION, "Población")], max_attempts=2)

    assert len(state["attempts"]) == 2
    assert all(a.error for a in state["attempts"])
    assert state["status"] == "cannot_answer"
    assert state["assessments"][0].missing == "The calculation failed after 2 attempts."


def test_data_missing_at_run_time_lowers_the_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({
        QueryPlan: plan("población de Salta 2023", "población de Rosario 2023"),
        Selection: Selection(parts=[assess(True), assess(True)]),
        GeneratedCode: program(SALTA, MISSING),  # the program didn't find Rosario
    })
    state = run(monkeypatch, tmp_path, llm, [make_record(POPULATION, "Población")])

    assert state["status"] == "partial"
    assert not state["assessments"][1].answerable
    assert state["assessments"][1].missing == "Rosario no está en la tabla"
    assert len(state["attempts"]) == 1  # a valid result: no retry



def test_code_is_generated_by_the_code_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # The general model has no answer for GeneratedCode: asking it would fail
    general = FakeChatModel({QueryPlan: plan("población de Salta 2023"), Selection: Selection(parts=[assess(True)])})
    coder = FakeChatModel({GeneratedCode: program(SALTA)})
    state = run(monkeypatch, tmp_path, general, [make_record(POPULATION, "Población")], coder=coder)

    assert state["status"] == "answered"
    assert coder.asked == [GeneratedCode]
    assert general.asked == [QueryPlan, Selection, "respond"]
