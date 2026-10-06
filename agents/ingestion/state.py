"""State of the ingestion graph."""

import operator
from typing import Annotated, TypedDict

from agents.ingestion.schemas import TableMetadata, TableRecord
from services.pdf_extract import ExtractedTable


class IngestionState(TypedDict):
    """Shared memory of the graph. Each node receives the whole state and returns a
    dict with only the keys it changes; LangGraph merges that into the state.

    Every key is always present (initial_state fills in empty values), so nodes can
    read state["key"] without checking.
    """

    # Input
    pdf_path: str  # a path, not bytes: the state is logged in every LangSmith span
    max_attempts: int

    # Set by extract
    filename: str
    doc_hash: str
    tables: list[ExtractedTable]

    # The table being processed: tables[current]
    current: int
    attempt: int  # LLM calls made for this table so far
    metadata: TableMetadata | None  # the LLM's latest answer
    errors: list[str]  # validation problems with that answer

    # Output. Annotated[..., operator.add] attaches a reducer: when a node returns
    # {"records": [r]}, LangGraph appends (old + [r]) instead of replacing the list.
    records: Annotated[list[TableRecord], operator.add]
    warnings: Annotated[list[str], operator.add]


def initial_state(pdf_path: str, max_attempts: int) -> IngestionState:
    return {
        "pdf_path": pdf_path,
        "max_attempts": max_attempts,
        "filename": "",
        "doc_hash": "",
        "tables": [],
        "current": 0,
        "attempt": 0,
        "metadata": None,
        "errors": [],
        "records": [],
        "warnings": [],
    }
