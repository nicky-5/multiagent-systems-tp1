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
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agents.ingestion import edges, nodes
from agents.ingestion.schemas import TableRecord
from agents.ingestion.state import (
    IngestionState,
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


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    graph = build_graph()
    state = initial_state(pdf_path, get_settings().ingest_max_attempts)
    started = time.perf_counter()
    records: list[TableRecord] = []
    warnings: list[str] = []

    # Two stream modes at once: "updates" yields each node's state update, "custom"
    # yields the progress events nodes emit. subgraphs=True includes events from
    # inside the subgraphs. Each item is then (namespace, mode, data); namespace is
    # () for the parent graph and identifies the branch for subgraph events.
    # The Streamlit UI will use the same loop to show progress.
    for _namespace, mode, data in graph.stream(
        state,
        run_config(pdf_path),
        stream_mode=["updates", "custom"],
        subgraphs=True,
    ):
        elapsed = f"{time.perf_counter() - started:5.1f}s"
        if mode == "custom":
            event = cast(dict[str, str], data)
            print(f"{elapsed} [{event['table']}] {event['message']}")
            continue
        for node, node_update in cast(dict[str, dict[str, Any] | None], data).items():
            update = node_update or {}
            if node == "extract":
                print(f"{elapsed} [extract] {len(update['tables'])} tables found")
            elif node == "process_table":  # a branch finished; its output reaches the parent
                records += update.get("records", [])
                warnings += update.get("warnings", [])
            elif node == "index":
                print(f"{elapsed} [index] saved and upserted {len(update['indexed'])} tables")

    print(f"\n{len(records)} records, warnings: {warnings}")
    for record in records:
        print(f"- {record.table_id}: {record.name!r}\n  {record.description}")
