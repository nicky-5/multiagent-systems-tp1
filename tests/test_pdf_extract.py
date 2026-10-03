from pathlib import Path

import pandas as pd
import pytest

from services.pdf_extract import (
    RawTable,
    Row,
    Table,
    TextBlock,
    collect_context,
    extract_tables,
    flatten_header,
    merge_continuations,
    split_header,
    to_dataframe,
)

SAMPLES = Path(__file__).parent.parent / "data" / "sample_pdfs"


def raw(page, rows, index=0, last=True, bbox=(0, 100, 500, 300)):
    """Small factory so tests only spell out the fields they care about."""
    return RawTable(page=page, index_on_page=index, is_last_on_page=last, bbox=bbox, rows=rows)


# --- headers -----------------------------------------------------------------


def test_split_header_single_row():
    rows: list[Row] = [["Concepto", "2022", "2023"], ["Ventas", "$ 1.000", "$ 2.000"]]
    assert split_header(rows) == (rows[:1], rows[1:])


def test_split_header_two_rows_with_years():
    rows = [
        ["", "2023", None],
        ["Región", "Unidades", "Monto"],
        ["Centro", "337.084", "1.303.617,7"],
    ]
    assert split_header(rows) == (rows[:2], rows[2:])


def test_split_header_none_when_first_row_has_data():
    rows: list[Row] = [["2023", "April", "422,915"], ["2023", "May", "500,000"]]
    assert split_header(rows) == ([], rows)


def test_split_header_text_only_table_keeps_one_header_row():
    rows: list[Row] = [["Name", "Role"], ["Ana", "Owner"]]
    assert split_header(rows) == (rows[:1], rows[1:])


def test_flatten_header_merged_cells():
    header = [
        ["", "2023", None, "2024", None],
        ["Región", "Unidades", "Monto", "Unidades", "Monto"],
    ]
    assert flatten_header(header) == [
        "Región", "2023 Unidades", "2023 Monto", "2024 Unidades", "2024 Monto",
    ]


def test_flatten_header_names_empty_and_duplicate_columns():
    assert flatten_header([["", "Total", "Total"]]) == ["column_1", "Total", "Total (2)"]


# --- continuations -----------------------------------------------------------

HEADER = ["Provincia", "Ventas"]


def test_merge_continuation_with_repeated_header():
    tables = merge_continuations([
        raw(1, [HEADER, ["Salta", "1,5"]]),
        raw(2, [HEADER, ["Jujuy", "2,5"]]),
    ])
    assert len(tables) == 1
    assert tables[0].pages == [1, 2]
    assert tables[0].body == [["Salta", "1,5"], ["Jujuy", "2,5"]]  # header not repeated


def test_merge_continuation_without_header():
    tables = merge_continuations([
        raw(1, [HEADER, ["Salta", "1,5"]]),
        raw(2, [["Jujuy", "2,5"]]),
    ])
    assert len(tables) == 1
    assert tables[0].body == [["Salta", "1,5"], ["Jujuy", "2,5"]]


@pytest.mark.parametrize(
    "second",
    [
        raw(3, [HEADER, ["Jujuy", "2,5"]]),  # skips a page
        raw(2, [HEADER, ["Jujuy", "2,5"]], index=1),  # not the first table on its page
        raw(2, [["Year", "Arrivals"], ["2023", "100"]]),  # a different header
        raw(2, [["Jujuy", "2,5", "x"]]),  # a different column count
    ],
)
def test_not_a_continuation(second):
    assert len(merge_continuations([raw(1, [HEADER, ["Salta", "1,5"]]), second])) == 2


def test_not_a_continuation_when_previous_table_is_not_last_on_its_page():
    first = raw(1, [HEADER, ["Salta", "1,5"]], last=False)
    assert len(merge_continuations([first, raw(2, [["Jujuy", "2,5"]])])) == 2


# --- typing ------------------------------------------------------------------


def test_to_dataframe_types_columns():
    table = Table(
        parts=[raw(1, [])],
        header_rows=[["Región", "Unidades", "Monto", "Share"]],
        body=[
            ["Centro", "1.000", "$ 1.234,5", "60 %"],
            ["Online", "", "$ (10,0)", "40 %"],
        ],
    )
    table.parts[0].rows = [table.header_rows[0]]  # n_cols comes from the raw rows
    df = to_dataframe(table)

    assert list(df.columns) == ["Región", "Unidades", "Monto", "Share (%)"]
    assert df["Región"].dtype == "string"
    assert df["Unidades"].dtype == "Int64"
    assert df["Monto"].dtype == "Float64"
    assert df["Unidades"].isna().tolist() == [False, True]  # empty cell -> <NA>
    assert df["Monto"].tolist() == [1234.5, -10.0]
    assert df["Share (%)"].tolist() == [60, 40]


# --- context -----------------------------------------------------------------


def block(page, y, text, heading=False):
    return TextBlock(page=page, bbox=(0, y, 400, y + 12), text=text, is_heading=heading)


def test_collect_context_stops_at_heading_and_tables():
    blocks = [
        block(1, 10, "Annual report", heading=True),
        block(1, 40, "Note under the previous table"),
        block(1, 60, "2. Sales", heading=True),
        block(1, 80, "Intro paragraph"),
        block(1, 98, "Table 2: Sales by region"),  # overlaps the table top by 2pt
        block(1, 310, "Source: company data"),
        block(1, 330, "3. Costs", heading=True),
        block(1, 350, "Not about table 2"),
    ]
    ctx = collect_context(blocks, start=(1, 108), end=(1, 300), prev_end=(1, 30))

    assert ctx.document_title == "Annual report"
    assert ctx.heading == "2. Sales"
    assert ctx.above == ["Intro paragraph", "Table 2: Sales by region"]
    assert ctx.below == ["Source: company data"]


def test_collect_context_for_table_spanning_pages():
    blocks = [
        block(1, 10, "Chapter 2", heading=True),
        block(1, 600, "Table 2: Monthly arrivals"),
        block(2, 450, "Year-on-year change is not reported for 2023."),
    ]
    ctx = collect_context(blocks, start=(1, 610), end=(2, 440))
    assert ctx.above == ["Table 2: Monthly arrivals"]
    assert ctx.below == ["Year-on-year change is not reported for 2023."]


# --- end to end on the sample PDFs -------------------------------------------


def test_extract_informe_anual():
    result = extract_tables(SAMPLES / "informe_anual_ferreteria_andina.pdf")
    assert result.warnings == []
    assert [t.pages for t in result.tables] == [[1], [1], [1, 2]]

    income, regions, provinces = (t.df for t in result.tables)
    assert income.loc[income["Concepto"] == "Costo de ventas", "2022"].item() == -1147248
    assert list(regions.columns)[:3] == ["Región", "2023 Unidades", "2023 Monto (miles ARS)"]
    assert pd.isna(regions.loc[regions["Región"] == "Online", "2023 Unidades"].item())
    assert len(provinces) == 24
    assert provinces["Participación (%)"].sum() == pytest.approx(100, abs=0.5)


def test_extract_yearbook_continuation_without_header():
    result = extract_tables(SAMPLES / "provincial_statistics_yearbook.pdf")
    arrivals = result.tables[1]
    assert arrivals.pages == [1, 2]
    assert len(arrivals.df) == 24  # 2 years x 12 months; no data row lost as a header
    assert arrivals.df["Year"].tolist() == [2023] * 12 + [2024] * 12
    assert arrivals.context.heading == "Chapter 2. Tourism"


def test_extract_from_bytes():
    pdf_bytes = (SAMPLES / "provincial_statistics_yearbook.pdf").read_bytes()
    assert len(extract_tables(pdf_bytes).tables) == 2
