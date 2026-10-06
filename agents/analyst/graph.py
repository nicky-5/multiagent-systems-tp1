"""The analyst graph (first half):

    START -> understand -> retrieve -> select -> END

Code generation, execution with self-correction, and the answer come next.

Try it from the command line:
    uv run python -m agents.analyst.graph "¿Cuántos turistas llegaron en julio de 2024?"
"""

# See the note in agents/ingestion/graph.py about LangGraph's types in strict mode.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

import sys
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agents.analyst import nodes
from agents.analyst.state import AnalystState, initial_state


def build_graph() -> CompiledStateGraph[AnalystState, None, AnalystState, AnalystState]:
    graph = StateGraph(AnalystState)

    # Retry on exceptions such as network errors (LLM, embeddings, Pinecone)
    retry = RetryPolicy(max_attempts=3)
    graph.add_node("understand", nodes.understand, retry_policy=retry)
    graph.add_node("retrieve", nodes.retrieve, retry_policy=retry)
    graph.add_node("select", nodes.select, retry_policy=retry)

    graph.add_edge(START, "understand")
    graph.add_edge("understand", "retrieve")
    graph.add_edge("retrieve", "select")
    graph.add_edge("select", END)

    return graph.compile()


def run_config(question: str) -> RunnableConfig:
    return {"run_name": "analyst", "tags": ["analyst"], "metadata": {"question": question}}


if __name__ == "__main__":
    question = sys.argv[1]
    output = build_graph().invoke(initial_state(question), run_config(question))
    state = cast(AnalystState, output)

    plan = state["plan"]
    assert plan is not None
    print(f"Question ({plan.language}): {question}\n\nParts:")
    for part in plan.parts:
        print(f"  - {part.question}  [{part.operation}; {part.metrics}; {part.entities}; {part.periods}]")
    print("Searches:")
    for search in plan.searches:
        print(f"  - {search.spanish!r} / {search.english!r}")
    print("\nCandidates:")
    for c in state["candidates"]:
        print(f"  {c.score:.3f}  {c.table_id}  {c.name}")
    print("\nAssessment:")
    for a in state["assessments"]:
        verdict = "answerable" if a.answerable else f"NOT answerable: {a.missing}"
        print(f"  - {a.question}\n    tables {a.table_ids}; {verdict}")
    print(f"\nStatus: {state['status']}; selected: {state['selected_ids']}")
    if state["warnings"]:
        print(f"Warnings: {state['warnings']}")
