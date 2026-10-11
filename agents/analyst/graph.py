"""The analyst graph:

    START -> understand -> retrieve -> select -> review -+-> generate_code -> execute -+-> respond -> END
                                                         |        ^                    |
                                                         |        +---- on error ------+  (up to max_attempts)
                                                         +-- cannot_answer ------------------> respond

review pauses for the user to confirm or change the tables, only when asked to
(ask_with_review); otherwise it passes straight through.

Try it from the command line:
    uv run python -m agents.analyst.graph "¿Cuántos turistas llegaron en julio de 2024?"
"""

# See the note in agents/ingestion/graph.py about LangGraph's types in strict mode.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, cast
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RetryPolicy

from agents.analyst import edges, nodes
from agents.analyst.schemas import Candidate, CodeAttempt, PartAssessment, QueryPlan
from agents.analyst.state import AnalystState, initial_state
from services.config import get_settings

Graph = CompiledStateGraph[AnalystState, None, AnalystState, AnalystState]


def build_graph(checkpointer: BaseCheckpointSaver[str] | None = None) -> Graph:
    """The analyst graph. A checkpointer is needed only to pause for a review."""
    graph = StateGraph(AnalystState)

    # Retry on exceptions such as network errors (LLM, embeddings, Pinecone). Not on
    # execute: a failing program isn't an exception there, it's a result that the
    # generate_code loop handles; an exception there is a setup problem.
    retry = RetryPolicy(max_attempts=3)
    graph.add_node("understand", nodes.understand, retry_policy=retry)
    graph.add_node("retrieve", nodes.retrieve, retry_policy=retry)
    graph.add_node("select", nodes.select, retry_policy=retry)
    # The retry policy covers the re-selection after a change; the pause itself (an
    # interrupt) is never retried
    graph.add_node("review", nodes.review, retry_policy=retry)
    graph.add_node("generate_code", nodes.generate_code, retry_policy=retry)
    graph.add_node("execute", nodes.execute)
    graph.add_node("respond", nodes.respond, retry_policy=retry)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "retrieve")
    graph.add_edge("retrieve", "select")
    graph.add_edge("select", "review")
    graph.add_conditional_edges("review", edges.after_review)
    graph.add_edge("generate_code", "execute")
    graph.add_conditional_edges("execute", edges.after_execute)
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=checkpointer)


@cache
def review_graph() -> Graph:
    """The graph for reviewed runs, built once so all of them share its checkpointer.

    The checkpointer saves the state when review pauses, and a later call resumes from
    it by the run's thread_id. InMemorySaver keeps it in this process (the Streamlit
    server), so a paused question lives as long as the app does.

    The checkpointer serializes the state, and restores only types it's told are safe
    (restoring arbitrary classes from stored data is a security risk), so the state's
    own classes are listed.
    """
    safe_types = [QueryPlan, Candidate, PartAssessment, CodeAttempt]
    return build_graph(InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=safe_types)))


def run_config(
    question: str, run_id: UUID | None = None, metadata: dict[str, str] | None = None
) -> RunnableConfig:
    """How the run appears in LangSmith. run_id becomes the trace's id, so a caller can
    find the trace afterwards (the evaluation script does)."""
    config: RunnableConfig = {
        "run_name": "analyst",
        "tags": ["analyst"],
        "metadata": {"question": question, **(metadata or {})},
    }
    if run_id is not None:
        config["run_id"] = run_id
    return config


# Called with (node, message) after each step of the graph
ProgressCallback = Callable[[str, str], None]


def _no_progress(node: str, message: str) -> None:
    pass


def ask(
    question: str,
    on_progress: ProgressCallback | None = None,
    *,
    run_id: UUID | None = None,
    metadata: dict[str, str] | None = None,
) -> AnalystState:
    """Run the analyst graph on a question, reporting each step; returns the final state.

    run_id and metadata are passed on to LangSmith (see run_config)."""
    state = initial_state(question, get_settings().analyst_max_code_attempts)
    return _stream(build_graph(), state, run_config(question, run_id, metadata), on_progress)


@dataclass
class TableReview:
    """A reviewed run, paused after select for the user to confirm the tables."""

    thread_id: str  # identifies the paused run in the checkpointer
    state: AnalystState  # the state so far: plan, candidates, assessments, selected_ids


def ask_with_review(question: str, on_progress: ProgressCallback | None = None) -> AnalystState | TableReview:
    """Like ask(), but pause after select so the user can confirm or change the tables.

    Returns a TableReview to pass to resume_review(), or the final state when there was
    nothing to review (no table is stored).
    """
    state = initial_state(question, get_settings().analyst_max_code_attempts, review=True)
    return _run_reviewed(state, question, str(uuid4()), on_progress)


def resume_review(
    review: TableReview, table_ids: list[str], on_progress: ProgressCallback | None = None
) -> AnalystState:
    """Continue a paused run with the tables the user chose (the same ones to confirm)."""
    result = _run_reviewed(Command(resume=table_ids), review.state["question"], review.thread_id, on_progress)
    assert not isinstance(result, TableReview)  # the graph pauses only once
    return result


def _run_reviewed(
    graph_input: AnalystState | Command[Any], question: str, thread_id: str, on_progress: ProgressCallback | None
) -> AnalystState | TableReview:
    config = run_config(question, metadata={"thread_id": thread_id})
    # The checkpointer stores each run's state under its thread_id. LangSmith also groups
    # traces with the same thread_id metadata, so the paused and resumed halves of a
    # question show up together.
    config["configurable"] = {"thread_id": thread_id}
    graph = review_graph()
    final = _stream(graph, graph_input, config, on_progress)
    if graph.get_state(config).interrupts:  # stopped at review's interrupt()
        return TableReview(thread_id=thread_id, state=final)
    return final


def _stream(
    graph: Graph, graph_input: AnalystState | Command[Any], config: RunnableConfig, on_progress: ProgressCallback | None
) -> AnalystState:
    """Run the graph, reporting each step; returns the last state (final or paused)."""
    report = on_progress or _no_progress
    final = cast(AnalystState, {})
    # "values" yields the whole state after each step (the last one is the result);
    # "updates" yields what each node changed, to describe the step
    for mode, data in graph.stream(graph_input, config, stream_mode=["updates", "values"]):
        if mode == "values":
            final = cast(AnalystState, data)
        else:
            for node, update in cast(dict[str, Any], data).items():
                if node != "__interrupt__":  # the pause itself, not a node's update
                    report(node, describe_step(node, update or {}))
    return final


def describe_step(node: str, update: dict[str, Any]) -> str:
    """One line on what a node just did, for progress displays."""
    match node:
        case "understand":
            plan = cast(QueryPlan, update["plan"])
            return f"{len(plan.parts)} part(s); searching for {len(plan.searches)} dataset(s)"
        case "retrieve":
            return f"{len(update['candidates'])} candidate table(s) found"
        case "select":
            return f"{len(update['selected_ids'])} table(s) selected; status: {update['status']}"
        case "review":
            if "review_result" not in update:
                return "skipped (no review asked for)"
            if update["review_result"] == "confirmed":
                return "tables confirmed"
            return f"you chose {len(update['user_table_ids'])} table(s); selection redone; status: {update['status']}"
        case "generate_code":
            return f"program {len(update['attempts'])} written"
        case "execute":
            last = cast(list[CodeAttempt], update["attempts"])[-1]
            if last.error is None:
                return "program ran; result is valid"
            return f"program failed: {last.error.splitlines()[0]}"
        case "respond":
            return "answer written"
        case _:
            return "done"


if __name__ == "__main__":
    question = sys.argv[1]
    state = ask(question, lambda node, message: print(f"[{node}] {message}"))

    print("\nAssessment:")
    for a in state["assessments"]:
        verdict = "answerable" if a.answerable else f"NOT answerable: {a.missing}"
        print(f"  - {a.question}\n    tables {a.table_ids}; {verdict}")
    for i, attempt in enumerate(state["attempts"], start=1):
        print(f"\n--- program {i}: {attempt.approach}\n{attempt.code}")
        if attempt.stdout.strip():
            print(f"--- output:\n{attempt.stdout.strip()}")
        if attempt.error:
            print(f"--- error:\n{attempt.error[-800:]}")
        else:
            print(f"--- result: {json.dumps(attempt.result, ensure_ascii=False)}")
    print(f"\nStatus: {state['status']}\n\n{state['answer']}")
    if state["warnings"]:
        print(f"\nWarnings: {state['warnings']}")
