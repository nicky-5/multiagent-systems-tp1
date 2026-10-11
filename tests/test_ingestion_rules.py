from typing import Any

import pandas as pd
import pytest

from agents.ingestion.rules import (
    MAX_METADATA_BYTES,
    build_record,
    count_sentences,
    embedding_text,
    make_table_id,
    metadata_size,
    pinecone_metadata,
    table_labels,
    validate_metadata,
)
from agents.ingestion.schemas import ColumnInfo, TableMetadata, TableRecord
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


# --- indexing --------------------------------------------------------------------


def record(**overrides: Any) -> TableRecord:
    table = ExtractedTable(pages=[2, 3], index_on_page=0, df=DF, context=TableContext())
    built = build_record(meta(), table, "f" * 64, "informe.pdf")
    return built.model_copy(update=overrides)  # a copy with some fields replaced


def test_build_record_includes_a_preview():
    assert record().preview.splitlines()[0].startswith("| Provincia")


def test_embedding_text_has_name_columns_and_rows():
    text = embedding_text(record())
    assert text.splitlines()[0] == "Ventas por provincia 2024"
    assert "Ventas 2024 (miles de ARS): Ventas netas" in text
    assert text.endswith("Rows: Salta, Jujuy")


def test_embedding_text_caps_row_labels():
    text = embedding_text(record(row_headers=[f"r{i}" for i in range(60)]))
    assert text.endswith("r49 (+10 more)")


def test_pinecone_metadata_uses_only_allowed_values():
    metadata = pinecone_metadata(record(), "text")
    assert metadata["pages"] == ["2", "3"]  # lists hold strings
    assert metadata["column_names"] == ["Provincia", "Ventas 2024"]
    assert metadata["row_header_column"] == "Provincia"
    assert "row_header_column" not in pinecone_metadata(record(row_header_column=None), "text")


def test_pinecone_metadata_is_trimmed_to_fit():
    labels = [f"Una etiqueta de fila bastante larga número {i}" for i in range(5000)]
    metadata = pinecone_metadata(record(row_headers=labels), "text")

    assert metadata_size(metadata) <= MAX_METADATA_BYTES
    kept = metadata["row_headers"]
    assert isinstance(kept, list) and 0 < len(kept) < len(labels)
    assert metadata["preview"] != ""  # trimming row labels was enough


def test_table_labels_number_the_tables_and_list_their_pages():
    tables = [
        ExtractedTable(pages=[1], index_on_page=0, df=DF, context=TableContext()),
        ExtractedTable(pages=[2, 3], index_on_page=0, df=DF, context=TableContext()),
    ]
    assert table_labels(tables) == ["table 1/2 (p. [1])", "table 2/2 (p. [2, 3])"]
