"""Run the whole ingestion graph with a fake LLM: fan-out, retry loop and reject path."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from agents.ingestion import nodes
from agents.ingestion.graph import build_graph
from agents.ingestion.schemas import ColumnInfo, TableMetadata
from agents.ingestion.state import IngestionState, initial_state

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


def run(monkeypatch: pytest.MonkeyPatch, fake: FakeChatModel, max_attempts: int = 3) -> IngestionState:
    # The nodes look up chat_model when they run, so replacing it in their module is enough
    monkeypatch.setattr(nodes, "chat_model", lambda: fake)
    # invoke() is typed as returning a plain dict; it holds IngestionState's keys.
    # LangGraph's signature contains unknown types (see the note in graph.py).
    output = build_graph().invoke(initial_state(str(PDF), max_attempts))  # pyright: ignore[reportUnknownMemberType]
    return cast(IngestionState, output)


def test_every_table_gets_a_record(monkeypatch: pytest.MonkeyPatch):
    state = run(monkeypatch, FakeChatModel(lambda _prompt, _attempt: VALID))

    ids = sorted(r.table_id for r in state["records"])
    assert [i.split("-", 1)[1] for i in ids] == ["p1-t0", "p1-t1", "p1-t2"]
    assert state["warnings"] == []


def test_invalid_answer_is_retried_with_feedback(monkeypatch: pytest.MonkeyPatch):
    fake = FakeChatModel(lambda _prompt, attempt: INVALID if attempt == 1 else VALID)
    state = run(monkeypatch, fake)

    assert len(state["records"]) == 3
    assert sorted(fake.calls.values()) == [2, 2, 2]  # each table: one failure, one success


def test_table_is_skipped_after_max_attempts(monkeypatch: pytest.MonkeyPatch):
    # Only the income statement table (it has a "Concepto" column) never validates
    fake = FakeChatModel(lambda prompt, _attempt: INVALID if "'Concepto'" in prompt else VALID)
    state = run(monkeypatch, fake, max_attempts=2)

    assert len(state["records"]) == 2
    assert len(state["warnings"]) == 1
    assert "skipped: metadata still invalid after 2 attempts" in state["warnings"][0]
