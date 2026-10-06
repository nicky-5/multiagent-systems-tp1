"""Nodes of the ingestion graph. Each takes the state and returns a partial update.

extract and index run once per PDF in the parent graph; the other nodes run inside
the per-table subgraph, once per table and in parallel across tables.
"""

import hashlib
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer

from agents.ingestion import prompts, rules
from agents.ingestion.schemas import TableMetadata
from agents.ingestion.state import IngestionState, TableState
from services.llm import chat_model, embeddings, with_schema
from services.pdf_extract import extract_tables
from services.table_store import table_store
from services.vector_store import vector_store

# --- parent graph --------------------------------------------------------------


def extract(state: IngestionState) -> dict[str, Any]:
    """Load the PDF and pull out its tables with their context (no LLM)."""
    path = Path(state["pdf_path"])
    pdf_bytes = path.read_bytes()
    result = extract_tables(pdf_bytes)
    warnings = list(result.warnings)
    if not result.tables:
        warnings.append(f"No tables found in {path.name}.")
    return {
        "filename": path.name,
        "doc_hash": hashlib.sha256(pdf_bytes).hexdigest(),
        "tables": result.tables,
        "warnings": warnings,
    }


def index(state: IngestionState) -> dict[str, Any]:
    """Save and index every table that got a valid record, in one batch.

    Runs once, after all the per-table branches have finished, so rejected tables
    never get here. The table store is written first: a Pinecone record must never
    point at data that isn't saved.
    """
    records = sorted(state["records"], key=lambda r: r.table_id)
    if not records:
        return {"indexed": []}

    # Find each record's DataFrame by recomputing the id of every extracted table
    frames = {
        rules.make_table_id(state["doc_hash"], t.pages[0], t.index_on_page): t.df
        for t in state["tables"]
    }
    store = table_store()
    for record in records:
        store.save(record, frames[record.table_id])

    texts = [rules.embedding_text(record) for record in records]
    vectors = embeddings().embed_documents(texts)  # one request for all the tables
    vector_store().upsert(
        ids=[record.table_id for record in records],
        vectors=vectors,
        metadatas=[rules.pinecone_metadata(record, text) for record, text in zip(records, texts, strict=True)],
    )
    return {"indexed": [record.table_id for record in records]}


# --- per-table subgraph --------------------------------------------------------


def enrich(state: TableState) -> dict[str, Any]:
    """Ask the LLM for the table's metadata, with feedback on a retry."""
    attempt = state["attempt"] + 1
    _progress(state, f"asking the LLM for metadata (attempt {attempt})")
    messages: list[BaseMessage] = [
        SystemMessage(prompts.ENRICH_SYSTEM),
        HumanMessage(prompts.enrich_user_prompt(state["table"], state["filename"])),
    ]
    previous = state["metadata"]
    if previous is not None and state["errors"]:  # a retry: show the LLM what was wrong
        feedback = prompts.retry_prompt(previous.model_dump_json(), state["errors"])
        messages.append(HumanMessage(feedback))

    metadata = with_schema(chat_model(), TableMetadata).invoke(messages)
    return {"metadata": metadata, "attempt": attempt}


def validate(state: TableState) -> dict[str, Any]:
    """Check the metadata against the table (no LLM). The edge after this node
    decides between accepting, retrying and giving up."""
    metadata = state["metadata"]
    assert metadata is not None  # enrich always runs first
    errors = rules.validate_metadata(metadata, state["table"].df)
    _progress(state, f"validation failed: {'; '.join(errors)}" if errors else "metadata is valid")
    return {"errors": errors}


def build_record(state: TableState) -> dict[str, Any]:
    """Turn the validated metadata into a complete record."""
    metadata = state["metadata"]
    assert metadata is not None
    record = rules.build_record(metadata, state["table"], state["doc_hash"], state["filename"])
    _progress(state, f"done: {record.name!r}")
    return {"records": [record]}


def reject(state: TableState) -> dict[str, Any]:
    """Give up on a table whose metadata never validated, and say so in the UI."""
    warning = (
        f"{state['label']} skipped: metadata still invalid after {state['attempt']} "
        f"attempts ({'; '.join(state['errors'])})."
    )
    _progress(state, "skipped")
    return {"warnings": [warning]}


def _progress(state: TableState, message: str) -> None:
    """Emit a progress event, received by callers that stream with stream_mode="custom".

    Parallel branches' node updates don't say which table they belong to, so the UI
    uses these labeled events to show per-table progress. Outside a stream it's a no-op.
    """
    get_stream_writer()({"table": state["label"], "message": message})
