"""The analyst graph:

    START -> understand -> retrieve -> select -+-> generate_code -> execute -+-> respond -> END
                                               |        ^                    |
                                               |        +---- on error ------+  (up to max_attempts)
                                               +-- cannot_answer ------------------> respond

Try it from the command line:
    uv run python -m agents.analyst.graph "¿Cuántos turistas llegaron en julio de 2024?"
"""

# See the note in agents/ingestion/graph.py about LangGraph's types in strict mode.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

import json
import sys
from collections.abc import Callable
from typing import Any, cast
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agents.analyst import edges, nodes
from agents.analyst.schemas import CodeAttempt, QueryPlan
from agents.analyst.state import AnalystState, initial_state
from services.config import get_settings


def build_graph() -> CompiledStateGraph[AnalystState, None, AnalystState, AnalystState]:
    graph = StateGraph(AnalystState)

    # Retry on exceptions such as network errors (LLM, embeddings, Pinecone). Not on
    # execute: a failing program isn't an exception there, it's a result that the
    # generate_code loop handles; an exception there is a setup problem.
    retry = RetryPolicy(max_attempts=3)
    graph.add_node("understand", nodes.understand, retry_policy=retry)
    graph.add_node("retrieve", nodes.retrieve, retry_policy=retry)
    graph.add_node("select", nodes.select, retry_policy=retry)
    graph.add_node("generate_code", nodes.generate_code, retry_policy=retry)
    graph.add_node("execute", nodes.execute)
    graph.add_node("respond", nodes.respond, retry_policy=retry)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "retrieve")
    graph.add_edge("retrieve", "select")
    graph.add_conditional_edges("select", edges.after_select)
    graph.add_edge("generate_code", "execute")
    graph.add_conditional_edges("execute", edges.after_execute)
    graph.add_edge("respond", END)

    return graph.compile()


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
    report = on_progress or _no_progress
    state = initial_state(question, get_settings().analyst_max_code_attempts)
    final = state
    # "values" yields the whole state after each step (the last one is the result);
    # "updates" yields what each node changed, to describe the step
    for mode, data in build_graph().stream(
        state, run_config(question, run_id, metadata), stream_mode=["updates", "values"]
    ):
        if mode == "values":
            final = cast(AnalystState, data)
        else:
            for node, update in cast(dict[str, dict[str, Any] | None], data).items():
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
