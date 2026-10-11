"""Run the whole ingestion graph with fakes for the LLM, embeddings and Pinecone:
fan-out, retry loop, reject path and batch indexing."""

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from agents.ingestion import nodes
from agents.ingestion.graph import build_graph
from agents.ingestion.schemas import ColumnInfo, TableMetadata
from agents.ingestion.state import IngestionState, initial_state
from services.table_store import TableStore
from services.vector_store import MetadataValue

PDF = Path(__file__).parent.parent / "data" / "sample_pdfs" / "informe_anual_ferreteria_andina.pdf"
VALID = "Una tabla de prueba. Tiene dos oraciones."
INVALID = "Una sola oración."


class FakeChatModel:
    """Stands in for ChatOpenAI. Reads the column names from the prompt, so its
    metadata matches whichever table it's asked about, and asks `describe` for
    the description given (table prompt, attempt number)."""

    def __init__(self, describe: Callable[[str, int], str]):
        self.describe = describe
        self.calls: dict[str, int] = {}  # attempts per table, keyed by the prompt

    def with_structured_output(self, _schema: object) -> RunnableLambda[list[BaseMessage], TableMetadata]:
        return RunnableLambda(self._answer)

    def _answer(self, messages: list[BaseMessage]) -> TableMetadata:
        prompt = str(messages[1].content)  # [system, user, (retry feedback)]
        self.calls[prompt] = self.calls.get(prompt, 0) + 1
        columns = re.findall(r"^- '(.+)' \(dtype", prompt, flags=re.MULTILINE)
        return TableMetadata(
            name="Tabla",
            description=self.describe(prompt, self.calls[prompt]),
            columns=[ColumnInfo(name=c, description="x", unit="x") for c in columns],
            row_header_column=None,
        )


class FakeEmbeddings:
    def __init__(self):
        self.calls: list[list[str]] = []  # the texts of each embed_documents call

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(t)), 0.0, 1.0] for t in texts]


class FakeVectorStore:
    def __init__(self):
        self.upserts: list[list[str]] = []  # the ids of each upsert call
        self.records: dict[str, Mapping[str, MetadataValue]] = {}

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, MetadataValue]],
    ) -> int:
        self.upserts.append(list(ids))
        self.records.update(zip(ids, metadatas, strict=True))  # same id overwrites, like Pinecone
        return len(ids)

    def delete(self, ids: Sequence[str]) -> None:
        for i in ids:
            self.records.pop(i, None)


class Services:
    """The fakes a test can inspect after running the graph."""

    def __init__(self, tmp_path: Path):
        self.embeddings = FakeEmbeddings()
        self.vectors = FakeVectorStore()
        self.tables = TableStore(tmp_path / "tables")


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Services:
    """A pytest fixture: every test that names a `services` parameter gets a fresh one.
    tmp_path is a built-in fixture with a new temporary directory per test."""
    fakes = Services(tmp_path)
    # The nodes look up these factories when they run, so replacing them in the
    # nodes module is enough
    monkeypatch.setattr(nodes, "embeddings", lambda: fakes.embeddings)
    monkeypatch.setattr(nodes, "vector_store", lambda: fakes.vectors)
    monkeypatch.setattr(nodes, "table_store", lambda: fakes.tables)
    return fakes


def run(monkeypatch: pytest.MonkeyPatch, fake: FakeChatModel, max_attempts: int = 3) -> IngestionState:
    monkeypatch.setattr(nodes, "chat_model", lambda: fake)
    # invoke() is typed as returning a plain dict; it holds IngestionState's keys.
    # LangGraph's signature contains unknown types (see the note in graph.py).
    output = build_graph().invoke(initial_state(str(PDF), max_attempts))  # pyright: ignore[reportUnknownMemberType]
    return cast(IngestionState, output)


def test_every_table_gets_a_record(monkeypatch: pytest.MonkeyPatch, services: Services):
    state = run(monkeypatch, FakeChatModel(lambda _prompt, _attempt: VALID))

    ids = sorted(r.table_id for r in state["records"])
    assert [i.split("-", 1)[1] for i in ids] == ["p1-t0", "p1-t1", "p1-t2"]
    assert state["warnings"] == []


def test_invalid_answer_is_retried_with_feedback(monkeypatch: pytest.MonkeyPatch, services: Services):
    fake = FakeChatModel(lambda _prompt, attempt: INVALID if attempt == 1 else VALID)
    state = run(monkeypatch, fake)

    assert len(state["records"]) == 3
    assert sorted(fake.calls.values()) == [2, 2, 2]  # each table: one failure, one success


def test_table_is_skipped_after_max_attempts(monkeypatch: pytest.MonkeyPatch, services: Services):
    # Only the income statement table (it has a "Concepto" column) never validates
    fake = FakeChatModel(lambda prompt, _attempt: INVALID if "'Concepto'" in prompt else VALID)
    state = run(monkeypatch, fake, max_attempts=2)

    assert len(state["records"]) == 2
    assert len(state["warnings"]) == 1
    assert "skipped: metadata still invalid after 2 attempts" in state["warnings"][0]


def test_only_valid_tables_are_indexed_in_one_batch(monkeypatch: pytest.MonkeyPatch, services: Services):
    fake = FakeChatModel(lambda prompt, _attempt: INVALID if "'Concepto'" in prompt else VALID)
    state = run(monkeypatch, fake, max_attempts=1)

    indexed = state["indexed"]
    assert [i.split("-", 1)[1] for i in indexed] == ["p1-t1", "p1-t2"]  # p1-t0 was rejected
    assert len(services.embeddings.calls) == 1  # one embedding request...
    assert services.vectors.upserts == [indexed]  # ...and one upsert, for both tables
    for table_id in indexed:
        assert services.tables.load_df(table_id).shape[0] > 0  # data saved too
        assert services.tables.load_record(table_id).table_id == table_id


def test_reingesting_the_same_pdf_overwrites_its_records(monkeypatch: pytest.MonkeyPatch, services: Services):
    fake = FakeChatModel(lambda _prompt, _attempt: VALID)
    first = run(monkeypatch, fake)
    second = run(monkeypatch, fake)

    assert first["indexed"] == second["indexed"]  # same content, same ids
    assert len(services.vectors.records) == 3  # overwritten, not duplicated
    assert len(services.tables.list_records()) == 3


def test_reingesting_removes_tables_no_longer_extracted(monkeypatch: pytest.MonkeyPatch, services: Services):
    fake = FakeChatModel(lambda _prompt, _attempt: VALID)
    first = run(monkeypatch, fake)
    # A table an older extractor found at a position this one doesn't
    prefix = first["indexed"][0].split("-")[0]
    old_id = f"{prefix}-p9-t0"
    old = services.tables.load_record(first["indexed"][0]).model_copy(update={"table_id": old_id})
    services.tables.save(old, services.tables.load_df(first["indexed"][0]))
    services.vectors.records[old_id] = {}

    run(monkeypatch, fake)

    assert old_id not in services.vectors.records
    assert not services.tables.exists(old_id)
    assert len(services.tables.list_records()) == 3


def test_reingesting_keeps_a_rejected_tables_previous_version(monkeypatch: pytest.MonkeyPatch, services: Services):
    first = run(monkeypatch, FakeChatModel(lambda _prompt, _attempt: VALID))
    run(monkeypatch, FakeChatModel(lambda prompt, _attempt: INVALID if "'Concepto'" in prompt else VALID), max_attempts=1)

    assert sorted(services.vectors.records) == sorted(first["indexed"])  # p1-t0 still indexed
    assert len(services.tables.list_records()) == 3
