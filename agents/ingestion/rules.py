"""Deterministic parts of ingestion: metadata validation, table ids, and records."""

import re

import pandas as pd

from agents.ingestion.schemas import Source, TableMetadata, TableRecord
from services.pdf_extract import ExtractedTable

MAX_NAME_LENGTH = 80


def make_table_id(doc_hash: str, first_page: int, index_on_page: int) -> str:
    """Stable id: the same PDF content always yields the same ids, so re-ingesting
    it overwrites its Pinecone records instead of duplicating them."""
    return f"{doc_hash[:16]}-p{first_page}-t{index_on_page}"


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
        errors.append(f"name has {len(meta.name)} characters; the maximum is {MAX_NAME_LENGTH}.")

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
    )
