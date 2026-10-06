"""State of the analyst graph."""

from typing import TypedDict

from agents.analyst.schemas import Candidate, PartAssessment, QueryPlan, Status


class AnalystState(TypedDict):
    """Shared memory of the analyst graph. Every key is always present
    (initial_state fills in empty values)."""

    # Input
    question: str

    # Set by understand
    plan: QueryPlan | None

    # Set by retrieve: the tables found, best first
    candidates: list[Candidate]

    # Set by select
    assessments: list[PartAssessment]  # one per part of the question
    selected_ids: list[str]  # tables the code will load
    # Decided by code from the assessments (rules.decide_status), not by the LLM's
    # wording, and kept here so tests can check it. Later steps may only lower it.
    status: Status | None

    warnings: list[str]


def initial_state(question: str) -> AnalystState:
    return {
        "question": question,
        "plan": None,
        "candidates": [],
        "assessments": [],
        "selected_ids": [],
        "status": None,
        "warnings": [],
    }
