from typing import Any

import pandas as pd
import pytest

from agents.ingestion.rules import (
    build_record,
    count_sentences,
    make_table_id,
    validate_metadata,
)
from agents.ingestion.schemas import ColumnInfo, TableMetadata
from services.pdf_extract import ExtractedTable, TableContext

DF = pd.DataFrame({
    "Provincia": pd.Series(["Salta", "Jujuy", None], dtype="string"),
    "Ventas 2024": pd.Series([1.5, 2.5, 3.0], dtype="Float64"),
})
GOOD_DESCRIPTION = "Ventas netas por provincia en 2024. Valores en miles de ARS."


def meta(**overrides: Any) -> TableMetadata:
    """Valid metadata for DF; tests override the field they want to break."""
    fields: dict[str, Any] = {
        "name": "Ventas por provincia 2024",
        "description": GOOD_DESCRIPTION,
        "columns": [
            ColumnInfo(name="Provincia", description="Provincia", unit="text"),
            ColumnInfo(name="Ventas 2024", description="Ventas netas", unit="miles de ARS"),
        ],
        "row_header_column": "Provincia",
    }
    return TableMetadata(**(fields | overrides))


def test_valid_metadata_has_no_errors():
    assert validate_metadata(meta(), DF) == []


def test_missing_column():
    errors = validate_metadata(meta(columns=meta().columns[:1]), DF)
    assert "The table has 2 columns but you described 1." in errors
    assert "These columns are missing: ['Ventas 2024']." in errors


def test_renamed_column():
    columns = [meta().columns[0], ColumnInfo(name="Ventas", description="x", unit="ARS")]
    errors = validate_metadata(meta(columns=columns), DF)
    assert "These columns don't exist in the table: ['Ventas']." in errors


def test_columns_out_of_order():
    errors = validate_metadata(meta(columns=meta().columns[::-1]), DF)
    assert errors == ["List the columns in the table's order: ['Provincia', 'Ventas 2024']."]


@pytest.mark.parametrize(
    ("row_header_column", "fragment"),
    [("Región", "is not a column"), ("Ventas 2024", "is numeric")],
)
def test_bad_row_header_column(row_header_column: str, fragment: str):
    errors = validate_metadata(meta(row_header_column=row_header_column), DF)
    assert len(errors) == 1 and fragment in errors[0]


def test_row_header_column_may_be_null():
    assert validate_metadata(meta(row_header_column=None), DF) == []


@pytest.mark.parametrize(
    ("description", "ok"),
    [
        ("One sentence only.", False),
        (GOOD_DESCRIPTION, True),
        ("Uno. Dos. Tres. Cuatro. Cinco.", False),
    ],
)
def test_description_length(description: str, ok: bool):
    assert (validate_metadata(meta(description=description), DF) == []) == ok


def test_name_rules():
    assert validate_metadata(meta(name="  "), DF) == ["name is empty."]
    assert "maximum is 80" in validate_metadata(meta(name="x" * 81), DF)[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Creció 1.5 % en 2024. Fuente: INDEC.", 2),  # decimal point
        ("Ventas de Ferretería Andina S.A. para 2024. En miles de ARS.", 2),  # abbreviation
        ("Sin punto final", 0),
        ("¿Qué mide? Mide ventas.", 2),
    ],
)
def test_count_sentences(text: str, expected: int):
    assert count_sentences(text) == expected


def test_table_id_is_deterministic():
    assert make_table_id("ab" * 32, 3, 1) == "abababababababab-p3-t1"
    assert make_table_id("ab" * 32, 3, 1) == make_table_id("ab" * 32, 3, 1)


def test_build_record_fills_computed_fields():
    table = ExtractedTable(pages=[2, 3], index_on_page=0, df=DF, context=TableContext())
    record = build_record(meta(), table, "f" * 64, "informe.pdf")

    assert record.table_id == "ffffffffffffffff-p2-t0"
    assert record.source.filename == "informe.pdf"
    assert record.source.pages == [2, 3]
    assert record.row_headers == ["Salta", "Jujuy"]  # read from the data; <NA> dropped
    assert record.n_rows == 3
    assert record.name == "Ventas por provincia 2024"  # LLM fields carried over
