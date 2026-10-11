"""The ingestion graph.

    START -> extract -> (one process_table per table, in parallel) -> index -> END

process_table is itself a graph (a subgraph), run once per table:

    START -> enrich -> validate -> build_record -> END
               ^          |
               +-- retry -+-> reject -> END   (after max_attempts)

Try it from the command line:
    uv run python -m agents.ingestion.graph data/sample_pdfs/<file>.pdf
"""

# langgraph.graph has no py.typed marker of its own, and LangGraph's method signatures
# contain unannotated types (e.g. CachePolicy's type parameter), so strict mode would
# flag the imports and every add_node/compile/stream call.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agents.ingestion import edges, nodes, rules
from agents.ingestion.schemas import TableRecord
from agents.ingestion.state import (
    IngestionState,
    Step,
    TableOutput,
    TableState,
    initial_state,
)
from services.config import get_settings


def build_table_graph() -> CompiledStateGraph[TableState, None, TableState, TableOutput]:
    """The per-table subgraph: get metadata from the LLM, validate it, retry or give up."""
    # output_schema limits what the subgraph returns to the parent to TableOutput's
    # keys. The parent merges those through its reducers; table, attempt etc. stay
    # private to the branch.
    graph = StateGraph(TableState, output_schema=TableOutput)

    # RetryPolicy re-runs the node on exceptions such as network errors. That's
    # separate from the validation loop, which retries on *wrong answers*.
    graph.add_node("enrich", nodes.enrich, retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node("validate", nodes.validate)
    graph.add_node("build_record", nodes.build_record)
    graph.add_node("reject", nodes.reject)

    graph.add_edge(START, "enrich")
    graph.add_edge("enrich", "validate")
    graph.add_conditional_edges("validate", edges.after_validate)
    graph.add_edge("build_record", END)
    graph.add_edge("reject", END)

    return graph.compile()


def build_graph() -> CompiledStateGraph[IngestionState, None, IngestionState, IngestionState]:
    graph = StateGraph(IngestionState)

    graph.add_node("extract", nodes.extract)
    # A compiled graph can be used as a node; LangGraph runs it as a subgraph.
    graph.add_node("process_table", build_table_graph())
    graph.add_node("index", nodes.index, retry_policy=RetryPolicy(max_attempts=3))

    graph.add_edge(START, "extract")
    # fan_out returns a list of Send objects, so the possible destinations can't be
    # read from its return type; the third argument lists them (for drawing the graph).
    graph.add_conditional_edges("extract", edges.fan_out, ["process_table", END])
    # A node after a fan-out runs once, when every branch has finished
    graph.add_edge("process_table", "index")
    graph.add_edge("index", END)

    return graph.compile()


def run_config(pdf_path: str) -> RunnableConfig:
    return {
        # How the run is named and labeled in LangSmith
        "run_name": "ingestion",
        "tags": ["ingestion"],
        "metadata": {"filename": Path(pdf_path).name},
        # At most this many branches run at once, so a PDF with many tables doesn't
        # send them all to the LLM proxy at the same time.
        "max_concurrency": get_settings().ingest_max_concurrency,
    }


# Called with (source, step, message) as ingestion progresses; source is "extract",
# "index" or a table's label such as "table 2/3 (p. [1])"
ProgressCallback = Callable[[str, Step, str], None]


@dataclass
class IngestionOutcome:
    filename: str
    records: list[TableRecord]  # the tables indexed, sorted by table_id
    warnings: list[str]  # what was skipped, to show to the user


def _no_progress(source: str, step: Step, message: str) -> None:
    pass


def ingest(pdf_path: str, on_progress: ProgressCallback | None = None) -> IngestionOutcome:
    """Run the ingestion graph on one PDF, reporting progress as it goes."""
    report = on_progress or _no_progress
    state = initial_state(pdf_path, get_settings().ingest_max_attempts)
    records: list[TableRecord] = []
    warnings: list[str] = []

    # Two stream modes at once: "updates" yields each node's state update, "custom"
    # yields the progress events nodes emit. subgraphs=True includes events from
    # inside the subgraphs. Each item is then (namespace, mode, data); namespace is
    # () for the parent graph and identifies the branch for subgraph events.
    for _namespace, mode, data in build_graph().stream(
        state, run_config(pdf_path), stream_mode=["updates", "custom"], subgraphs=True
    ):
        if mode == "custom":
            event = cast(dict[str, Any], data)
            report(event["table"], event["step"], event["message"])
            continue
        for node, node_update in cast(dict[str, dict[str, Any] | None], data).items():
            update = node_update or {}
            if node == "extract":
                warnings += update["warnings"]
                report("extract", "extract", f"{len(update['tables'])} table(s) found")
                # Every table, so callers can list them before their branches start
                # (at most max_concurrency branches run at once)
                for label in rules.table_labels(update["tables"]):
                    report(label, "pending", "waiting to start")
            elif node == "process_table":  # a branch finished; its output reaches the parent
                records += update.get("records", [])
                warnings += update.get("warnings", [])
            elif node == "index":
                report("index", "index", f"saved and indexed {len(update['indexed'])} table(s)")

    records.sort(key=lambda r: r.table_id)
    return IngestionOutcome(filename=Path(pdf_path).name, records=records, warnings=warnings)


if __name__ == "__main__":
    started = time.perf_counter()
    outcome = ingest(
        sys.argv[1],
        lambda source, step, message: print(f"{time.perf_counter() - started:5.1f}s [{source}] {message}"),
    )
    print(f"\n{len(outcome.records)} records, warnings: {outcome.warnings}")
    for record in outcome.records:
        print(f"- {record.table_id}: {record.name!r}\n  {record.description}")
