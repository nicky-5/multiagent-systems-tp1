"""Run the analyst graph with fakes for the LLM, embeddings, Pinecone and the table store."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from agents.analyst import nodes
from agents.analyst.graph import build_graph
from agents.analyst.schemas import (
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


def make_record(table_id: str, name: str) -> TableRecord:
    return TableRecord(
        table_id=table_id,
        name=name,
        description="Una tabla. Dos oraciones.",
        columns=[ColumnInfo(name="Provincia", description="x", unit="text")],
        row_header_column="Provincia",
        source=Source(filename="anuario.pdf", pages=[1]),
        row_headers=["Salta"],
        n_rows=1,
        preview="",
    )


def plan(*questions: str) -> QueryPlan:
    return QueryPlan(
        language="es",
        parts=[QuestionPart(question=q, metrics=[], entities=[], periods=[], operation="lookup") for q in questions],
        searches=[Search(spanish="población por provincia", english="population by province")],
    )


class FakeChatModel:
    """Answers each structured-output request with the scripted object for its schema."""

    def __init__(self, answers: dict[type[BaseModel], BaseModel]):
        self.answers = answers
        self.asked: list[type[BaseModel]] = []

    def with_structured_output(self, schema: type[BaseModel]) -> RunnableLambda[list[BaseMessage], BaseModel]:
        def answer(_messages: list[BaseMessage]) -> BaseModel:
            self.asked.append(schema)
            return self.answers[schema]

        return RunnableLambda(answer)


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
) -> AnalystState:
    """stored: tables with data in the table store; indexed: tables in Pinecone
    (defaults to the stored ones)."""
    tables = TableStore(tmp_path)
    for record in stored:
        tables.save(record, pd.DataFrame({"Provincia": ["Salta"]}))
    in_pinecone = indexed if indexed is not None else stored
    matches = [Match(r.table_id, 0.9 - i / 10, dict(pinecone_metadata(r, ""))) for i, r in enumerate(in_pinecone)]

    fakes: dict[str, Callable[[], Any]] = {
        "chat_model": lambda: llm,
        "embeddings": FakeEmbeddings,
        "vector_store": lambda: FakeVectorStore(matches),
        "table_store": lambda: tables,
    }
    for name, factory in fakes.items():
        monkeypatch.setattr(nodes, name, factory)
    output = build_graph().invoke(initial_state("¿pregunta?"))  # pyright: ignore[reportUnknownMemberType]
    return cast(AnalystState, output)


def test_answerable_question(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({
        QueryPlan: plan("población de Salta 2023"),
        Selection: Selection(parts=[PartAssessment(question="x", table_ids=[POPULATION], answerable=True, missing=None)]),
    })
    state = run(monkeypatch, tmp_path, llm, [make_record(POPULATION, "Población"), make_record(SALES, "Ventas")])

    assert [c.table_id for c in state["candidates"]] == [POPULATION, SALES]
    assert state["status"] == "answered"
    assert state["selected_ids"] == [POPULATION]


def test_partial_question(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({
        QueryPlan: plan("población de Salta 2023", "población de Salta 2024"),
        Selection: Selection(parts=[
            PartAssessment(question="x", table_ids=[POPULATION], answerable=True, missing=None),
            PartAssessment(question="y", table_ids=[POPULATION], answerable=False, missing="2024 no está"),
        ]),
    })
    state = run(monkeypatch, tmp_path, llm, [make_record(POPULATION, "Población")])

    assert state["status"] == "partial"
    assert [a.question for a in state["assessments"]] == ["población de Salta 2023", "población de Salta 2024"]


def test_no_candidates_skips_the_selection_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({QueryPlan: plan("desempleo en Rosario")})
    state = run(monkeypatch, tmp_path, llm, stored=[])

    assert state["status"] == "cannot_answer"
    assert llm.asked == [QueryPlan]  # select didn't call the LLM


def test_indexed_table_without_data_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    llm = FakeChatModel({QueryPlan: plan("población de Salta 2023")})
    stale = make_record(SALES, "Ventas")
    state = run(monkeypatch, tmp_path, llm, stored=[], indexed=[stale])

    assert state["candidates"] == []
    assert state["warnings"] == [f"Indexed tables without stored data were ignored: ['{SALES}']"]
