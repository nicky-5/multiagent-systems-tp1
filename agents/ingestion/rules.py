"""Deterministic parts of ingestion: metadata validation, table ids, and records."""

import json
import math
import re

import pandas as pd

from agents.ingestion.schemas import Source, TableMetadata, TableRecord
from services.pdf_extract import ExtractedTable
from services.vector_store import MetadataValue

# The prompt limits names in words, which a model can follow (it sees text as word
# pieces, not letters); validation checks characters.
# 12 words are about 80 characters, well within the hard limit.
MAX_NAME_WORDS = 12
MAX_NAME_LENGTH = 120
PREVIEW_ROWS = 5


def document_prefix(doc_hash: str) -> str:
    """The start every table_id of a document shares."""
    return doc_hash[:16]


def make_table_id(doc_hash: str, first_page: int, index_on_page: int) -> str:
    """Stable id: the same PDF content always yields the same ids, so re-ingesting
    it overwrites its Pinecone records instead of duplicating them."""
    return f"{document_prefix(doc_hash)}-p{first_page}-t{index_on_page}"


def table_labels(tables: list[ExtractedTable]) -> list[str]:
    """How each table is named in progress messages, e.g. "table 2/3 (p. [1])"."""
    return [f"table {i}/{len(tables)} (p. {t.pages})" for i, t in enumerate(tables, start=1)]


def count_sentences(text: str) -> int:
    """A sentence ends in . ! or ? followed by the end of the text or by whitespace and
    a capital letter, so the dots in "1.5" and "S.A. para" don't count."""
    return len(re.findall(r"[.!?](?=\s+[A-ZÁÉÍÓÚÑ¿¡]|\s*$)", text.strip()))


def validate_metadata(meta: TableMetadata, df: pd.DataFrame) -> list[str]:
    """Check the LLM's metadata against the extracted table.

    Returns a list of problems, written as feedback for the LLM; empty means valid.
    """
    errors: list[str] = []
    actual = [str(column) for column in df.columns]
    given = [column.name for column in meta.columns]

    if len(given) != len(actual):
        errors.append(f"The table has {len(actual)} columns but you described {len(given)}.")
    if given != actual:
        missing = [column for column in actual if column not in given]
        unknown = [column for column in given if column not in actual]
        if missing:
            errors.append(f"These columns are missing: {missing}.")
        if unknown:
            errors.append(f"These columns don't exist in the table: {unknown}.")
        if not missing and not unknown:
            errors.append(f"List the columns in the table's order: {actual}.")

    if meta.row_header_column is not None:
        if meta.row_header_column not in actual:
            errors.append(
                f"row_header_column {meta.row_header_column!r} is not a column; "
                f"use one of {actual} or null."
            )
        elif pd.api.types.is_numeric_dtype(df[meta.row_header_column]):
            errors.append(
                f"row_header_column {meta.row_header_column!r} is numeric; it must be a "
                "text column that labels the rows, or null."
            )

    if not meta.name.strip():
        errors.append("name is empty.")
    elif len(meta.name) > MAX_NAME_LENGTH:
        # Models count characters poorly, so say how much to cut, roughly in words too
        # (at this name's own average word length, counting the space after each word),
        # and what can go. The 10 extra characters keep the next try from landing just
        # over the limit again.
        to_cut = len(meta.name) - MAX_NAME_LENGTH + 10
        chars_per_word = (len(meta.name) + 1) / len(meta.name.split())
        errors.append(
            f"name has {len(meta.name)} characters; the maximum is {MAX_NAME_LENGTH}. Remove at "
            f"least {to_cut} characters (about {math.ceil(to_cut / chars_per_word)} words), "
            "e.g. the scope or period, which the description already covers."
        )

    sentences = count_sentences(meta.description)
    if not 2 <= sentences <= 4:
        errors.append(f"description has {sentences} sentence(s); write 2 to 4.")

    return errors


def build_record(
    meta: TableMetadata, table: ExtractedTable, doc_hash: str, filename: str
) -> TableRecord:
    """Combine validated LLM metadata with the fields we can compute exactly."""
    row_headers = []
    if meta.row_header_column:
        row_headers = table.df[meta.row_header_column].dropna().astype(str).tolist()
    return TableRecord(
        **meta.model_dump(),
        table_id=make_table_id(doc_hash, table.pages[0], table.index_on_page),
        source=Source(filename=filename, pages=table.pages),
        row_headers=row_headers,
        n_rows=len(table.df),
        preview=table.df.head(PREVIEW_ROWS).to_markdown(index=False),
    )


# --- indexing ------------------------------------------------------------------

MAX_EMBEDDED_ROW_HEADERS = 50
# Pinecone allows about 40 KB of metadata per record; keep a safety margin
MAX_METADATA_BYTES = 38_000



def embedding_text(record: TableRecord) -> str:
    """The text whose embedding represents the table in Pinecone.

    It holds what a question is likely to mention: the subject (name, description),
    the measures (columns with units) and the entities (row labels).
    """
    columns = "; ".join(f"{c.name} ({c.unit}): {c.description}" for c in record.columns)
    lines = [record.name, record.description, f"Columns: {columns}"]
    if record.row_headers:
        shown = record.row_headers[:MAX_EMBEDDED_ROW_HEADERS]
        more = len(record.row_headers) - len(shown)
        lines.append(f"Rows: {', '.join(shown)}" + (f" (+{more} more)" if more > 0 else ""))
    return "\n".join(lines)


def metadata_size(metadata: dict[str, MetadataValue]) -> int:
    return len(json.dumps(metadata, ensure_ascii=False).encode())


def pinecone_metadata(record: TableRecord, text: str) -> dict[str, MetadataValue]:
    """Flatten a record into Pinecone metadata, trimmed to fit the size limit.

    Everything the analyst needs to choose a table is here, so table selection
    doesn't have to open the table store. The full data stays in the table store.
    """
    metadata: dict[str, MetadataValue] = {
        "table_id": record.table_id,
        "name": record.name,
        "description": record.description,
        "filename": record.source.filename,
        "pages": [str(p) for p in record.source.pages],  # lists must hold strings
        "column_names": [c.name for c in record.columns],
        "columns_json": json.dumps([c.model_dump() for c in record.columns], ensure_ascii=False),
        "row_headers": record.row_headers,
        "n_rows": record.n_rows,
        "preview": record.preview,
        "text": text,
    }
    if record.row_header_column is not None:  # None isn't allowed, so leave the key out
        metadata["row_header_column"] = record.row_header_column

    # Too big (very long tables): drop the least important fields first. Halving
    # row_headers keeps as many labels as fit.
    row_headers = record.row_headers
    while metadata_size(metadata) > MAX_METADATA_BYTES and row_headers:
        row_headers = row_headers[: len(row_headers) // 2]
        metadata["row_headers"] = row_headers
    for key in ("preview", "text"):
        if metadata_size(metadata) > MAX_METADATA_BYTES:
            metadata[key] = ""
    return metadata
