"""Prompts for the analyst's LLM nodes."""

from agents.analyst.schemas import Candidate, QueryPlan

MAX_ROW_HEADERS_SHOWN = 60

UNDERSTAND_SYSTEM = """\
You plan how to answer questions using statistical tables extracted from PDF reports \
(company reports, statistical yearbooks). You don't answer the question; you describe \
what data it needs.

- Split the question into parts only where it asks for separate results (e.g. "which \
region had the highest X, and by how much did it exceed Y?" has two parts). A single \
result is one part even if computing it takes several values: "sales per resident" is one \
part with operation "ratio", not a sales lookup, a population lookup and a ratio.
- For each part, list the metrics, entities, periods and the operation. Use only what the \
question says: don't carry an entity or period from one part into another, and don't add \
any the question doesn't mention.
- searches: one per distinct dataset needed, each in Spanish and in English. Describe the \
dataset, not the question: "ventas netas por provincia 2024", not "cuánto vendió Córdoba"."""


SELECT_SYSTEM = """\
You decide which tables can answer each part of a question. You may only use data in the \
candidate tables: no general knowledge, external sources or estimates.

For each part of the plan, list the table_ids it needs (several if the data must be \
combined) and decide whether it is answerable. A part is NOT answerable when:
- a needed period or entity isn't in the table (e.g. the question asks for 2024 and the \
table ends in 2023, or asks for a province the table doesn't list). Never interpolate, \
extrapolate or use the nearest year.
- a needed conversion requires external data: exchange rates, inflation adjustment, or a \
per-capita figure without a population table. Conversions that need no external data are \
fine (thousands -> millions, % <-> fraction, summing all 12 months into a year).
- a comparison value is missing (e.g. "compared to the national average" when no table \
reports one). Don't substitute a value you would compute yourself, unless the question \
defines it that way.
- the meaning or unit of a column is unclear and the choice changes the result (e.g. \
nominal vs real values not stated).

Check periods and entities against the column names and row labels listed for each table, \
not only against its description. When a part isn't answerable, say what's missing and \
what the closest related table does contain."""


def understand_user_prompt(question: str) -> str:
    return f"Question: {question}"


def select_user_prompt(question: str, plan: QueryPlan, candidates: list[Candidate]) -> str:
    parts = "\n".join(
        f"{i}. {p.question} (metrics: {p.metrics}; entities: {p.entities}; "
        f"periods: {p.periods}; operation: {p.operation})"
        for i, p in enumerate(plan.parts, start=1)
    )
    tables = "\n\n".join(_describe(c) for c in candidates) or "(no candidate tables were found)"
    return f"Question: {question}\n\nParts:\n{parts}\n\nCandidate tables:\n\n{tables}"


def _describe(c: Candidate) -> str:
    columns = "\n".join(f"  - {col.name!r} ({col.unit}): {col.description}" for col in c.columns)
    rows = ""
    if c.row_header_column:
        shown = c.row_headers[:MAX_ROW_HEADERS_SHOWN]
        more = f" ... (+{len(c.row_headers) - len(shown)} more)" if len(c.row_headers) > len(shown) else ""
        rows = f"Row labels (column {c.row_header_column!r}): {', '.join(shown)}{more}\n"
    return (
        f"table_id: {c.table_id}\n"
        f"Name: {c.name}\n"
        f"Source: {c.filename}, pages {c.pages}; {c.n_rows} rows\n"
        f"Description: {c.description}\n"
        f"Columns:\n{columns}\n"
        f"{rows}"
        f"First rows:\n{c.preview}"
    )
