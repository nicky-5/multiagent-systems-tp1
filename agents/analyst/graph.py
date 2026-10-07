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
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from agents.analyst import edges, nodes
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


def run_config(question: str) -> RunnableConfig:
    return {"run_name": "analyst", "tags": ["analyst"], "metadata": {"question": question}}


if __name__ == "__main__":
    question = sys.argv[1]
    state_in = initial_state(question, get_settings().analyst_max_code_attempts)
    state = cast(AnalystState, build_graph().invoke(state_in, run_config(question)))

    print(f"Question: {question}\n\nAssessment:")
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
