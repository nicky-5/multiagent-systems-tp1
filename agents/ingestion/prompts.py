"""Prompts for the enrich node."""

from services.pdf_extract import ExtractedTable

PREVIEW_ROWS = 10

ENRICH_SYSTEM = """\
You document tables extracted from PDF reports. Your metadata is used to find the \
right table for a user's question (semantic search) and to write pandas code over it.

Rules:
- Use only the table and its surrounding text. Don't invent periods, units or scope \
that they don't support.
- unit: take units from the caption, headers and notes (e.g. "en miles de ARS").
- Write name and descriptions in the language of the document's prose (its title, \
headings, captions and paragraphs). Ignore the language of names inside the cells: \
an English report about places with Spanish names gets English metadata."""


def enrich_user_prompt(table: ExtractedTable, filename: str) -> str:
    context = table.context
    df = table.df
    columns = "\n".join(f"- {name!r} (dtype {dtype})" for name, dtype in df.dtypes.items())
    preview = df.head(PREVIEW_ROWS).to_markdown(index=False)  # uses the tabulate package
    lines = [
        f"File: {filename}, pages {table.pages}",
        f"Document title: {context.document_title or '(none)'}",
        f"Section heading: {context.heading or '(none)'}",
        "Text before the table:",
        *(f"  {text}" for text in context.above or ["(none)"]),
        "Text after the table:",
        *(f"  {text}" for text in context.below or ["(none)"]),
        "",
        f"Columns ({len(df.columns)}):",
        columns,
        "",
        f"First {min(PREVIEW_ROWS, len(df))} of {len(df)} rows:",
        preview,
    ]
    return "\n".join(lines)


def retry_prompt(previous_json: str, errors: list[str]) -> str:
    problems = "\n".join(f"- {error}" for error in errors)
    return (
        f"Your previous answer was:\n{previous_json}\n\n"
        f"It failed validation:\n{problems}\n\nReturn a corrected answer."
    )
