"""Nodes of the ingestion graph. Each takes the state and returns a partial update."""

import hashlib
from pathlib import Path
from typing import Any, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from agents.ingestion import prompts, rules
from agents.ingestion.schemas import TableMetadata
from agents.ingestion.state import IngestionState
from services.llm import chat_model
from services.pdf_extract import extract_tables


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
        **_start_table(0),
    }


def enrich(state: IngestionState) -> dict[str, Any]:
    """Ask the LLM for the current table's metadata, with feedback on a retry."""
    table = state["tables"][state["current"]]
    messages: list[BaseMessage] = [
        SystemMessage(prompts.ENRICH_SYSTEM),
        HumanMessage(prompts.enrich_user_prompt(table, state["filename"])),
    ]
    previous = state["metadata"]
    if previous is not None and state["errors"]:  # a retry: show the LLM what was wrong
        feedback = prompts.retry_prompt(previous.model_dump_json(), state["errors"])
        messages.append(HumanMessage(feedback))

    # with_structured_output sends TableMetadata's JSON schema with the request and
    # parses the reply into a TableMetadata instance. LangChain annotates one of its
    # parameters as a bare `list`, which strict mode reports as partially unknown.
    llm = cast(
        Runnable[LanguageModelInput, TableMetadata],
        chat_model().with_structured_output(TableMetadata),  # pyright: ignore[reportUnknownMemberType]
    )
    metadata = llm.invoke(messages)
    return {"metadata": metadata, "attempt": state["attempt"] + 1}


def validate(state: IngestionState) -> dict[str, Any]:
    """Check the metadata against the table (no LLM). The edge after this node
    decides between accepting, retrying and giving up."""
    metadata = state["metadata"]
    assert metadata is not None  # enrich always runs first
    table = state["tables"][state["current"]]
    return {"errors": rules.validate_metadata(metadata, table.df)}


def build_record(state: IngestionState) -> dict[str, Any]:
    """Turn the validated metadata into a complete record, then move to the next table."""
    metadata = state["metadata"]
    assert metadata is not None
    table = state["tables"][state["current"]]
    record = rules.build_record(metadata, table, state["doc_hash"], state["filename"])
    return {"records": [record], **_start_table(state["current"] + 1)}


def reject(state: IngestionState) -> dict[str, Any]:
    """Give up on a table whose metadata never validated, and say so in the UI."""
    table = state["tables"][state["current"]]
    problems = "; ".join(state["errors"])
    warning = (
        f"Table on page(s) {table.pages} skipped: metadata still invalid after "
        f"{state['attempt']} attempts ({problems})."
    )
    return {"warnings": [warning], **_start_table(state["current"] + 1)}


def _start_table(index: int) -> dict[str, Any]:
    """State update that points the loop at tables[index] with a clean slate."""
    return {"current": index, "attempt": 0, "metadata": None, "errors": []}
