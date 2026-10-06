"""Nodes of the analyst graph. Each takes the state and returns a partial update."""

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agents.analyst import prompts, rules
from agents.analyst.schemas import PartAssessment, QueryPlan, Selection
from agents.analyst.state import AnalystState
from services.config import get_settings
from services.llm import chat_model, embeddings, with_schema
from services.table_store import table_store
from services.vector_store import vector_store


def understand(state: AnalystState) -> dict[str, Any]:
    """Split the question into parts and phrase the searches for the data it needs."""
    messages: list[BaseMessage] = [
        SystemMessage(prompts.UNDERSTAND_SYSTEM),
        HumanMessage(prompts.understand_user_prompt(state["question"])),
    ]
    plan = with_schema(chat_model(), QueryPlan).invoke(messages)
    return {"plan": plan}


def retrieve(state: AnalystState) -> dict[str, Any]:
    """Search Pinecone with every query in both languages and merge the results (no LLM)."""
    plan = state["plan"]
    assert plan is not None  # understand always runs first
    settings = get_settings()

    queries = [q for s in plan.searches for q in (s.spanish, s.english)] or [state["question"]]
    vectors = embeddings().embed_documents(queries)  # one request for all the queries
    store = vector_store()
    results = [store.query(v, top_k=settings.analyst_top_k) for v in vectors]
    matches = rules.merge_matches(results, limit=settings.analyst_max_candidates)

    # A Pinecone record whose data file is gone (e.g. data/tables was deleted) can't be
    # used for calculations, so it isn't offered to the LLM.
    tables = table_store()
    candidates = [rules.candidate_from_match(m) for m in matches if tables.exists(m.id)]
    stale = [m.id for m in matches if not tables.exists(m.id)]
    warnings = [f"Indexed tables without stored data were ignored: {stale}"] if stale else []
    return {"candidates": candidates, "warnings": warnings}


def select(state: AnalystState) -> dict[str, Any]:
    """Decide which tables each part needs and whether it's answerable, then derive
    the status from those decisions."""
    plan = state["plan"]
    assert plan is not None
    candidates = state["candidates"]

    if candidates:
        messages: list[BaseMessage] = [
            SystemMessage(prompts.SELECT_SYSTEM),
            HumanMessage(prompts.select_user_prompt(state["question"], plan, candidates)),
        ]
        selection = with_schema(chat_model(), Selection).invoke(messages)
        parts = rules.check_assessments(
            selection.parts, [p.question for p in plan.parts], {c.table_id for c in candidates}
        )
    else:  # nothing indexed matched: no need to ask the LLM
        parts = [
            PartAssessment(
                question=p.question, table_ids=[], answerable=False,
                missing="No indexed table is related to this question.",
            )
            for p in plan.parts
        ]

    return {
        "assessments": parts,
        "selected_ids": rules.selected_table_ids(parts),
        "status": rules.decide_status(parts),
    }
