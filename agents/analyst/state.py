"""State of the analyst graph."""

from typing import TypedDict

from agents.analyst.schemas import (
    Candidate,
    CodeAttempt,
    PartAssessment,
    QueryPlan,
    Status,
)


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

    # Set by generate_code and execute: every program tried, the last one is current.
    # More than one means the first ones failed and were corrected.
    attempts: list[CodeAttempt]
    max_attempts: int

    # Set by respond
    answer: str

    warnings: list[str]


def initial_state(question: str, max_attempts: int = 3) -> AnalystState:
    return {
        "question": question,
        "plan": None,
        "candidates": [],
        "assessments": [],
        "selected_ids": [],
        "status": None,
        "attempts": [],
        "max_attempts": max_attempts,
        "answer": "",
        "warnings": [],
    }
