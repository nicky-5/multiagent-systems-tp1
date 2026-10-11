"""State of the analyst graph."""

from typing import Literal, TypedDict

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
    review: bool  # pause after select for the user to confirm or change the tables

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

    # Set by review (None without one): the tables the user chose, and whether they
    # are the ones select chose or others (then the selection was redone over them)
    user_table_ids: list[str] | None
    review_result: Literal["confirmed", "changed"] | None

    # Set by generate_code and execute: every program tried, the last one is current.
    # More than one means the first ones failed and were corrected.
    attempts: list[CodeAttempt]
    max_attempts: int

    # Set by respond
    answer: str

    warnings: list[str]


def initial_state(question: str, max_attempts: int = 3, review: bool = False) -> AnalystState:
    return {
        "question": question,
        "review": review,
        "plan": None,
        "candidates": [],
        "assessments": [],
        "selected_ids": [],
        "status": None,
        "user_table_ids": None,
        "review_result": None,
        "attempts": [],
        "max_attempts": max_attempts,
        "answer": "",
        "warnings": [],
    }
