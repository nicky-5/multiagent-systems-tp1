"""Routing functions for the ingestion graph's conditional edges.

A routing function reads the state and returns where to go next. It doesn't change
the state; that is the nodes' job.
"""

from typing import Literal

from langgraph.types import Send

from agents.ingestion.state import IngestionState, TableState


def fan_out(state: IngestionState) -> list[Send] | Literal["__end__"]:
    """Start one process_table subgraph per table, all in parallel.

    Returning a list of Send objects (instead of a node name) tells LangGraph to run
    the target node once per Send, each with its own input. That input becomes the
    branch's TableState, so it must contain every key that TableState declares.
    """
    tables = state["tables"]
    if not tables:
        return "__end__"  # the value of langgraph's END constant, spelled out for the type checker
    sends: list[Send] = []
    for i, table in enumerate(tables, start=1):
        branch: TableState = {
            "table": table,
            "label": f"table {i}/{len(tables)} (p. {table.pages})",
            "filename": state["filename"],
            "doc_hash": state["doc_hash"],
            "max_attempts": state["max_attempts"],
            "attempt": 0,
            "metadata": None,
            "errors": [],
            "records": [],
            "warnings": [],
        }
        sends.append(Send("process_table", branch))
    return sends


def after_validate(state: TableState) -> Literal["build_record", "enrich", "reject"]:
    if not state["errors"]:
        return "build_record"
    if state["attempt"] < state["max_attempts"]:
        return "enrich"  # loop back; enrich sees the errors and sends them as feedback
    return "reject"
