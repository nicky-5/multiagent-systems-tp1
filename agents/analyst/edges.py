"""Routing functions for the analyst graph's conditional edges."""

from typing import Literal

from agents.analyst.state import AnalystState


def after_select(state: AnalystState) -> Literal["generate_code", "respond"]:
    """Nothing to compute when no part is answerable: go straight to the answer."""
    if state["status"] == "cannot_answer":
        return "respond"
    return "generate_code"


def after_execute(state: AnalystState) -> Literal["generate_code", "respond"]:
    last = state["attempts"][-1]
    if last.error is None:
        return "respond"
    if len(state["attempts"]) < state["max_attempts"]:
        return "generate_code"  # self-correction: the next program sees this error
    return "respond"  # out of attempts; execute marked the parts as failed
