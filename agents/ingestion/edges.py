"""Routing functions for the ingestion graph's conditional edges.

A routing function reads the state and returns the name of the next node. It
doesn't change the state; that is the nodes' job.
"""

from typing import Literal

from agents.ingestion.state import IngestionState


def after_validate(state: IngestionState) -> Literal["build_record", "enrich", "reject"]:
    if not state["errors"]:
        return "build_record"
    if state["attempt"] < state["max_attempts"]:
        return "enrich"  # loop back; enrich sees the errors and sends them as feedback
    return "reject"


def next_table(state: IngestionState) -> Literal["enrich", "__end__"]:
    """After a table is done (or after extract): process the next one, or finish."""
    if state["current"] < len(state["tables"]):
        return "enrich"
    return "__end__"  # the value of langgraph's END constant, spelled out for the type checker
