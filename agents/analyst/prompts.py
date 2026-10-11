"""Prompts for the analyst's LLM nodes."""

from typing import Any

import pandas as pd

from agents.analyst.rules import format_value
from agents.analyst.schemas import Candidate, PartAssessment, QueryPlan, QuestionPart

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
- searches: describe the dataset, not the question: "ventas netas por provincia 2024", \
not "cuánto vendió Córdoba"."""


SELECT_SYSTEM = """\
You decide which tables can answer each part of a question. You may only use data in the \
candidate tables: no general knowledge, external sources or estimates.

For each part of the plan, list the table_ids whose data its calculation uses (several \
if the data must be combined; a table without the part's entity, e.g. only national \
totals, isn't used for that entity's value) and decide whether it is answerable. A part is NOT answerable when:
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
not only against its description."""


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


# --- generate_code -----------------------------------------------------------------

MAX_ERROR_CHARS = 2000
PREVIEW_ROWS_FOR_CODE = 5


GENERATE_SYSTEM = """\
You write a Python program that computes the answer to parts of a question from tables \
extracted from PDF reports.

Environment:
- Python 3 with the standard library and only the packages listed at the end. Nothing \
else is installed.
- Each table is a Parquet file in the current directory, named in the table list. Load it \
with pandas.read_parquet("<file>"). There's no network and no other file.
- pandas 3: text columns have the "str" dtype; numeric columns use the nullable "Int64" and \
"Float64" dtypes, whose missing values are pd.NA (check with pd.isna). Column names are \
exactly as listed, including spaces, accents and units.

Rules:
- Use only values loaded from the tables; never type numbers from the preview into the \
code. Filter rows by their labels exactly as written in the data.
- A part about a named entity (a city, region, company) takes its value from that entity's \
row; a total or average row is not a substitute. If no table has the entity's row, report \
it as missing.
- Report only what each part asks for, with no extra figures (e.g. no interval derived \
from a coefficient of variation).
- When a table already reports the figure asked for (e.g. a share or change column), \
read it instead of recomputing it: the report's own figure is the source.
- Never interpolate, extrapolate, estimate or use outside knowledge (exchange rates, \
inflation, population). Conversions that need no outside data are fine (thousands -> \
millions, % <-> fraction, summing all 12 months into a year); mention them in approach.
- Report results in readable units: when a ratio or conversion yields a small or awkward \
figure (e.g. 0.164 thousands of ARS per resident), express it in a unit that reads \
naturally (164 ARS per resident) and say so in approach. Keep percentages as percentages \
(8.65 with unit "%"), not fractions.
- When a part asks which entity has the highest or lowest value, report the entity with \
its value as an object ({"Formosa": 165.43}, unit "ARS per resident"), not the name alone. \
A text value has unit "".
- If a value a part needs isn't in the data (a row, column or period not found, or NA), \
don't guess: put the reason in that part's "missing" and leave "value" null.
- End by printing exactly one line of JSON, and nothing after it:
  print(json.dumps({"parts": [...]}, ensure_ascii=False))
  with one entry per part, in the given order: {"value": ..., "unit": "...", "missing": null}.
  value may be a number, string, list or object (e.g. {"Salta": 1.5, "Jujuy": 2.1}). \
Convert numpy and pandas values to plain Python first (int(...), float(...), .tolist()), \
and round to a sensible precision.
- You may print intermediate values on earlier lines; they're shown to the user."""


def generate_system(packages: dict[str, str]) -> str:
    """The code generator's instructions. packages is services.sandbox.available_packages():
    exactly what the sandbox can import, with versions. They go last so the fixed
    instructions before them stay an identical, cacheable prefix if the versions change."""
    available = "\n".join(f"- {name} {version}" for name, version in packages.items())
    return f"{GENERATE_SYSTEM}\n\nAvailable packages:\n{available}"


def generate_user_prompt(
    question: str, parts: list[QuestionPart], tables: list[tuple[Candidate, pd.DataFrame]]
) -> str:
    lines = [f"Question: {question}", "", "Compute these parts, in this order:"]
    for i, p in enumerate(parts, start=1):
        lines.append(
            f"{i}. {p.question} (metrics: {p.metrics}; entities: {p.entities}; "
            f"periods: {p.periods}; operation: {p.operation})"
        )
    lines += ["", "Tables:"]
    for candidate, df in tables:
        units = {c.name: c.unit for c in candidate.columns}
        columns = "\n".join(
            f"  - {name!r}: dtype {dtype}, unit {units.get(str(name), '?')}" for name, dtype in df.dtypes.items()
        )
        lines += [
            "",
            f"File: {candidate.table_id}.parquet",
            f"Name: {candidate.name} ({len(df)} rows)",
            f"Description: {candidate.description}",
            f"Columns:\n{columns}",
            f"First rows:\n{df.head(PREVIEW_ROWS_FOR_CODE).to_markdown(index=False)}",
        ]
        if candidate.row_header_column:
            labels = df[candidate.row_header_column].dropna().astype(str).unique().tolist()
            lines.append(f"All values of {candidate.row_header_column!r}: {', '.join(labels)}")
    return "\n".join(lines)



def code_retry_prompt(previous_code: str, error: str) -> str:
    # Tracebacks of pandas errors are long; the end has the line in main.py and the error
    tail = error if len(error) <= MAX_ERROR_CHARS else "...\n" + error[-MAX_ERROR_CHARS:]
    return (
        f"Your previous program was:\n```python\n{previous_code}\n```\n\n"
        f"It failed:\n{tail}\n\nWrite a corrected program."
    )


# --- respond -----------------------------------------------------------------------

RESPOND_SYSTEM = """\
You write the final answer to a user's question about tables from PDF reports, from \
results that were already computed. You don't compute anything yourself.

- Answer in the language given. Be brief: the result first, then one or two sentences on \
how it was computed (from "approach"), including any unit conversion.
- Use only numbers from the computed results, copied exactly as written (they're already \
rounded and formatted for the answer's language), with their units. Don't round or \
convert them, and don't add numbers, estimates or outside knowledge.
- For each part that couldn't be answered, say so and what's missing, using the reason \
given (it may say what a related table does contain).
- If nothing could be answered, give no numeric result at all.
- Don't list sources or table names at the end; they're added automatically."""


def respond_user_prompt(
    question: str,
    language: str,
    parts: list[PartAssessment],
    approach: str | None,
    result_parts: list[dict[str, Any]],
) -> str:
    lines = [f"Question: {question}", f"Language: {'Spanish' if language == 'es' else 'English'}", ""]
    results = iter(result_parts)
    for i, part in enumerate(parts, start=1):
        if part.answerable:
            r = next(results)
            value = format_value(r["value"], language)
            lines.append(f"Part {i}: {part.question}\n  Result: {value} {r['unit']}")
        else:
            lines.append(f"Part {i}: {part.question}\n  Not answerable: {part.missing}")
    if approach:
        lines += ["", f"Approach: {approach}"]
    return "\n".join(lines)
