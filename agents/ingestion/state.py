"""State of the ingestion graph and of its per-table subgraph."""

import operator
from typing import Annotated, TypedDict

from agents.ingestion.schemas import TableMetadata, TableRecord
from services.pdf_extract import ExtractedTable

# Annotated[list, operator.add] attaches a reducer to a key: when a node returns
# {"records": [r]}, LangGraph stores old + [r] instead of replacing the list. Parallel
# branches rely on this: each table's subgraph returns its own record, and the
# reducer collects them in the parent state whatever order they finish in.
Records = Annotated[list[TableRecord], operator.add]
Warnings = Annotated[list[str], operator.add]


class IngestionState(TypedDict):
    """State of the parent graph: one PDF.

    Each node receives the whole state and returns a dict with only the keys it
    changes. Every key is always present (initial_state fills in empty values), so
    nodes can read state["key"] without checking.
    """

    # Input
    pdf_path: str  # a path, not bytes: the state is logged in every LangSmith span
    max_attempts: int

    # Set by extract
    filename: str
    doc_hash: str
    tables: list[ExtractedTable]

    # Output, filled in by the per-table subgraphs
    records: Records
    warnings: Warnings


class TableState(TypedDict):
    """State of the per-table subgraph. Each parallel branch has its own copy,
    created from the payload of the Send that started it."""

    table: ExtractedTable
    label: str  # e.g. "table 2/3 (p. [1])", for progress messages
    filename: str
    doc_hash: str
    max_attempts: int

    attempt: int  # LLM calls made for this table so far
    metadata: TableMetadata | None  # the LLM's latest answer
    errors: list[str]  # validation problems with that answer

    records: Records
    warnings: Warnings


class TableOutput(TypedDict):
    """What a finished subgraph hands back to the parent: only the shared keys."""

    records: Records
    warnings: Warnings


def initial_state(pdf_path: str, max_attempts: int) -> IngestionState:
    return {
        "pdf_path": pdf_path,
        "max_attempts": max_attempts,
        "filename": "",
        "doc_hash": "",
        "tables": [],
        "records": [],
        "warnings": [],
    }
