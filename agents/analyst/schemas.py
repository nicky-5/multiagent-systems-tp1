"""Pydantic models for the analyst agent.

QueryPlan and Selection are LLM outputs (structured output); Candidate is built by
code from Pinecone results. As in ingestion, Field descriptions are part of the
JSON schema the LLM sees, so they double as instructions.
"""

from typing import Literal

from pydantic import BaseModel, Field

from agents.ingestion.schemas import ColumnInfo

Status = Literal["answered", "partial", "cannot_answer"]


# --- understand ------------------------------------------------------------------


class QuestionPart(BaseModel):
    question: str = Field(description="One self-contained sub-question, in the user's language.")
    metrics: list[str] = Field(description="Quantities needed, e.g. 'net sales', 'population'.")
    entities: list[str] = Field(description="Entities it's about, e.g. provinces, regions, concepts.")
    periods: list[str] = Field(description="Years, months or ranges needed, e.g. '2024'.")
    operation: str = Field(description="The calculation, e.g. 'lookup', 'sum', 'ratio', 'growth rate'.")


class Search(BaseModel):
    """One dataset to look for, phrased in both languages because tables may be
    documented in either."""

    spanish: str = Field(description="Short search query in Spanish describing the dataset.")
    english: str = Field(description="The same query in English.")


class QueryPlan(BaseModel):
    language: Literal["es", "en"] = Field(description="Language of the question.")
    parts: list[QuestionPart] = Field(
        description="The question split into parts that can be answered independently; one part if it can't be split."
    )
    searches: list[Search] = Field(
        description=(
            "One search per distinct dataset needed. A question combining two kinds of "
            "data (e.g. sales and population) needs two searches."
        )
    )


# --- retrieve --------------------------------------------------------------------


class Candidate(BaseModel):
    """A table found in Pinecone, with the metadata the LLM uses to choose tables."""

    table_id: str
    score: float  # best similarity over all the searches
    name: str
    description: str
    filename: str
    pages: list[int]
    columns: list[ColumnInfo]
    row_header_column: str | None
    row_headers: list[str]
    n_rows: int
    preview: str


# --- select ----------------------------------------------------------------------


class PartAssessment(BaseModel):
    question: str = Field(description="The part's question text, without its metrics or periods.")
    table_ids: list[str] = Field(description="table_ids of the candidates this part needs; may be several.")
    answerable: bool = Field(
        description="True only if every value the calculation needs is in those tables."
    )
    missing: str | None = Field(
        description=(
            "If not answerable: what's missing, and what the closest related table does "
            "contain (e.g. 'The population table covers 2022-2023; 2024 is not available'). "
            "Null if answerable."
        )
    )


class Selection(BaseModel):
    parts: list[PartAssessment] = Field(description="One assessment per part, in the plan's order.")
