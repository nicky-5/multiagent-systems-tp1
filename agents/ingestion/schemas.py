"""Pydantic models for table metadata.

TableMetadata is what the LLM produces (via structured output): only the fields that
need judgment. TableRecord is the complete metadata we index; code fills in the
fields that can be derived exactly (table_id, source, row_headers).
"""

from pydantic import BaseModel, Field

# Field descriptions are sent to the LLM as part of the JSON schema, so they double
# as instructions.


class ColumnInfo(BaseModel):
    name: str = Field(description="The column name exactly as given, unchanged.")
    description: str = Field(description="What the column holds, in one short sentence.")
    unit: str = Field(
        description=(
            "Type or unit of the values, e.g. 'thousands of ARS', 'persons', '%', "
            "'year', 'month name', 'text'."
        )
    )


class TableMetadata(BaseModel):
    name: str = Field(description="Short human-readable table name, at most 80 characters.")
    description: str = Field(
        description="2 to 4 sentences: what the table measures, units, period and scope."
    )
    columns: list[ColumnInfo] = Field(description="One entry per column, in the given order.")
    row_header_column: str | None = Field(
        description=(
            "Name of the column whose values label the rows (e.g. a province or concept "
            "column), or null if no column does."
        )
    )


class Source(BaseModel):
    filename: str
    pages: list[int]


class TableRecord(TableMetadata):
    """Everything we know about one indexed table. Inherits the LLM's fields."""

    table_id: str
    source: Source
    row_headers: list[str]  # values of row_header_column, read from the data
    n_rows: int
