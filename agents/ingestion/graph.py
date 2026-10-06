"""The ingestion graph: PDF -> tables -> LLM metadata (validated, with retries) -> records.

Try it from the command line:
    uv run python -m agents.ingestion.graph data/sample_pdfs/<file>.pdf
"""

# langgraph.graph has no py.typed marker of its own, and LangGraph's method signatures
# contain unannotated types (e.g. CachePolicy's type parameter), so strict mode would
# flag the imports and every add_node/compile/stream call.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

import sys
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agents.ingestion import edges, nodes
from agents.ingestion.state import IngestionState, initial_state
from services.config import get_settings


def build_graph() -> CompiledStateGraph[IngestionState, None, IngestionState, IngestionState]:
    graph = StateGraph(IngestionState)

    graph.add_node("extract", nodes.extract)
    # RetryPolicy re-runs the node on exceptions such as network errors. That's
    # separate from the validation loop, which retries on *wrong answers*.
    graph.add_node("enrich", nodes.enrich, retry_policy=RetryPolicy(max_attempts=3))
    graph.add_node("validate", nodes.validate)
    graph.add_node("build_record", nodes.build_record)
    graph.add_node("reject", nodes.reject)

    graph.add_edge(START, "extract")
    graph.add_conditional_edges("extract", edges.next_table)  # a PDF may have no tables
    graph.add_edge("enrich", "validate")
    graph.add_conditional_edges("validate", edges.after_validate)
    graph.add_conditional_edges("build_record", edges.next_table)
    graph.add_conditional_edges("reject", edges.next_table)

    return graph.compile()


def run_config(pdf_path: str) -> RunnableConfig:
    return {
        "run_name": "ingestion",
        "tags": ["ingestion"],
        "metadata": {"filename": Path(pdf_path).name},
        # LangGraph stops a run after 25 steps by default, to catch infinite loops.
        # Each table takes at least 3 steps, so a PDF with many tables needs more.
        "recursion_limit": 500,
    }


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    graph = build_graph()
    state = initial_state(pdf_path, get_settings().ingest_max_attempts)

    # stream_mode="updates" yields {node_name: update} after each node finishes.
    # The Streamlit UI will use the same loop to show progress.
    for step in graph.stream(state, run_config(pdf_path), stream_mode="updates"):
        updates = cast(dict[str, dict[str, Any] | None], step)
        for node, node_update in updates.items():
            update = node_update or {}
            if node == "extract":
                print(f"[extract] {len(update['tables'])} tables, warnings={update['warnings']}")
            elif node == "enrich":
                print(f"[enrich] attempt {update['attempt']}: {update['metadata'].name!r}")
            elif node == "validate":
                print(f"[validate] errors={update['errors']}")
            elif node == "build_record":
                print(f"[build_record] {update['records'][0].model_dump_json(indent=2)}")
            else:
                print(f"[{node}] {update}")
