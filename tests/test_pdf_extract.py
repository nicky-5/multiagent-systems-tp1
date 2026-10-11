from pathlib import Path

import pandas as pd
import pytest

from services.pdf_extract import (
    BBox,
    RawTable,
    Row,
    Table,
    TextBlock,
    Word,
    collect_context,
    drop_running_text,
    extract_tables,
    flatten_header,
    header_from_words,
    merge_continuations,
    merge_range_rows,
    placements_to_rows,
    split_header,
    split_sections,
    tidy_header,
    to_dataframe,
)

SAMPLES = Path(__file__).parent.parent / "data" / "sample_pdfs"


def raw(
    page: int, rows: list[Row], index: int = 0, last: bool = True, bbox: BBox = (0, 100, 500, 300)
) -> RawTable:
    """Small factory so tests only spell out the fields they care about."""
    return RawTable(page=page, index_on_page=index, is_last_on_page=last, bbox=bbox, rows=rows)


# --- headers -----------------------------------------------------------------


def test_split_header_single_row():
    rows: list[Row] = [["Concepto", "2022", "2023"], ["Ventas", "$ 1.000", "$ 2.000"]]
    assert split_header(rows) == (rows[:1], rows[1:])


def test_split_header_two_rows_with_years():
    rows: list[Row] = [
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


def test_split_header_leaves_a_section_row_in_the_body():
    rows: list[Row] = [
        ["Indicador", "2024", "2025"],
        ["Pobreza", "", ""],  # labels the rows below it
        ["Hogares", "42,5", "24,1"],
    ]
    assert split_header(rows) == (rows[:1], rows[1:])


def test_split_header_keeps_a_row_label_header_between_header_rows():
    rows: list[Row] = [
        ["", "Hogares", "", "Personas"],
        ["Área geográfica", "", "", ""],  # the first column's name, above the sub-headers
        ["", "Total", "Incidencia", "Total"],
        ["Cuyo", "1.142", "23,8", "4.508"],
    ]
    assert split_header(rows) == (rows[:3], rows[3:])


def test_tidy_header_single_group_label_becomes_a_note():
    header: list[Row] = [["", "", "2026", "", ""], ["Región", "Enero", "Febrero", "Marzo", "Abril"]]
    rows, notes = tidy_header(header)
    assert notes == ["2026"]
    assert flatten_header(rows) == ["Región", "Enero", "Febrero", "Marzo", "Abril"]


def test_tidy_header_leaves_merged_cells_and_the_last_row_alone():
    header: list[Row] = [["", "2023", None, "2024", None], ["Región", "Unidades", "Monto", "", ""]]
    rows, notes = tidy_header(header)
    assert (rows, notes) == (header, [])


# --- headers from the page: words at (x0, y0, x1, y1) and rules at (y, x0, x1) ----

COLUMNS = [(0.0, 100.0), (100.0, 160.0), (160.0, 220.0), (220.0, 280.0)]  # row labels + 3 data


def word(x0: float, y0: float, text: str, width: float = 30) -> Word:
    return (x0, y0, x0 + width, y0 + 9, text)


def test_header_from_words_group_label_spans_the_columns_under_its_rule():
    words = [
        word(150, 0, "Año", 15), word(167, 0, "2025", 20),  # one label: the words are 2 pt apart
        word(10, 20, "Indicador"), word(115, 20, "1° trim"), word(175, 20, "2° trim"), word(235, 20, "3° trim"),
    ]
    rules = [(11.0, 100.0, 220.0)]  # under "Año 2025", over the first two quarters only
    names, notes = header_from_words(words, rules, COLUMNS)
    assert names == ["Indicador", "Año 2025 1° trim", "Año 2025 2° trim", "3° trim"]
    assert notes == []


def test_header_from_words_rule_closing_the_header_is_not_a_group():
    # A rule under every label, with only a lone "%" below it
    words = [word(10, 0, "Edad"), word(115, 0, "Total"), word(175, 0, "Pobres"), word(235, 0, "Otros"), word(185, 20, "%", 5)]
    names, notes = header_from_words(words, [(12.0, 0.0, 280.0)], COLUMNS)
    assert names == ["Edad", "Total", "Pobres", "Otros"]
    assert notes == ["%"]  # alone in its band: it describes the whole table


def test_header_from_words_wrapped_label_and_label_over_the_whole_table():
    words = [
        word(150, 0, "2026", 20),  # straddles two columns, alone in its band
        word(10, 15, "Región"), word(115, 15, "Total"), word(175, 15, "Pobres"), word(235, 15, "No pobres"),
        word(170, 25, "indigentes", 40),  # second line of "Pobres indigentes"
    ]
    names, notes = header_from_words(words, [], COLUMNS)
    assert names == ["Región", "Total", "Pobres indigentes", "No pobres"]
    assert notes == ["2026"]


def test_header_from_words_two_groups_on_one_rule_split_the_columns():
    columns = [(0.0, 100.0), (100.0, 150.0), (150.0, 200.0), (200.0, 250.0), (250.0, 300.0)]
    words = [
        word(140, 0, "2023", 20), word(240, 0, "2024", 20),
        word(10, 20, "Región"), word(110, 20, "U"), word(160, 20, "M"), word(210, 20, "U"), word(260, 20, "M"),
    ]
    names, _ = header_from_words(words, [(11.0, 0.0, 300.0)], columns)  # a grid: one rule under the row
    assert names == ["Región", "2023 U", "2023 M", "2024 U", "2024 M"]


def test_placements_to_rows_fills_spanned_positions():
    # Like HTML, the grid leaves out positions a span covers: "Región" spans 2 rows,
    # "2023" and "2024" 2 columns each
    placements = [
        [("Región", 1, 2), ("2023", 2, 1), ("2024", 2, 1)],
        [("Unidades", 1, 1), ("Monto", 1, 1), ("Unidades", 1, 1), ("Monto", 1, 1)],
    ]
    assert placements_to_rows(placements, 5) == [
        ["Región", "2023", None, "2024", None],
        ["", "Unidades", "Monto", "Unidades", "Monto"],
    ]
    assert placements_to_rows(placements, 6) is None  # doesn't fit: use extract() instead


def test_flatten_header_merged_cells():
    header: list[Row] = [
        ["", "2023", None, "2024", None],
        ["Región", "Unidades", "Monto", "Unidades", "Monto"],
    ]
    assert flatten_header(header) == [
        "Región", "2023 Unidades", "2023 Monto", "2024 Unidades", "2024 Monto",
    ]


def test_flatten_header_names_empty_and_duplicate_columns():
    assert flatten_header([["", "Total", "Total"]]) == ["column_1", "Total", "Total (2)"]


# --- continuations -----------------------------------------------------------

HEADER: Row = ["Provincia", "Ventas"]


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
def test_not_a_continuation(second: RawTable):
    assert len(merge_continuations([raw(1, [HEADER, ["Salta", "1,5"]]), second])) == 2


def test_not_a_continuation_after_a_caption():
    first = raw(1, [HEADER, ["Salta", "1,5"]])
    second = raw(2, [HEADER, ["Jujuy", "2,5"]], bbox=(0, 100, 500, 300))
    caption = (2, 80.0)  # "Cuadro 4.2", just above the second table
    assert len(merge_continuations([first, second], captions=[caption])) == 2


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


def test_split_sections_labels_rows_with_their_section():
    body = [
        ["Total", "10", "20"],
        ["Pobreza", "", ""],
        ["Hogares", "42,5", "24,1"],
        ["Indigencia", "", ""],
        ["Hogares", "13,6", "5,6"],
    ]
    sections, rows = split_sections(body)
    assert sections == ["", "Pobreza", "Indigencia"]
    assert rows == [body[0], body[2], body[4]]


def test_merge_range_rows_moves_intervals_into_columns():
    columns = ["Indicador", "2024", "2025"]
    body = [
        ["Hogares", "42,5", "24,1"],
        ["IC (90%)", "[41,4 ; 43,7]", "[23,0 ; 25,3]"],
        ["Personas", "52,9", "31,6"],
        ["IC (90%)", "[51,4 ; 54,5]", "[30,1 ; 33,1]"],
    ]
    new_columns, rows = merge_range_rows(columns, body)
    assert new_columns == [
        "Indicador", "2024", "2024: IC (90%) (min)", "2024: IC (90%) (max)",
        "2025", "2025: IC (90%) (min)", "2025: IC (90%) (max)",
    ]
    assert rows == [
        ["Hogares", "42,5", "41,4", "43,7", "24,1", "23,0", "25,3"],
        ["Personas", "52,9", "51,4", "54,5", "31,6", "30,1", "33,1"],
    ]


def test_merge_range_rows_leaves_tables_without_ranges_alone():
    body = [["Hogares", "42,5"], ["Personas", "[a ; b] text"]]
    assert merge_range_rows(["Indicador", "2024"], body) == (["Indicador", "2024"], body)


def test_to_dataframe_section_column():
    table = Table(
        parts=[raw(1, [["Indicador", "2024", "2025"]])],
        header_rows=[["Indicador", "2024", "2025"]],
        body=[["Pobreza", "", ""], ["Hogares", "42,5", "24,1"], ["Indigencia", "", ""], ["Hogares", "13,6", "5,6"]],
    )
    df = to_dataframe(table)
    assert list(df.columns) == ["Section", "Indicador", "2024", "2025"]
    assert df["Section"].tolist() == ["Pobreza", "Indigencia"]
    assert df["2024"].tolist() == [42.5, 13.6]


# --- context -----------------------------------------------------------------


def block(page: int, top: float, text: str, heading: bool = False) -> TextBlock:
    return TextBlock(page=page, bbox=(0, top, 400, top + 12), text=text, is_heading=heading)


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
    assert [table.pages for table in result.tables] == [[1], [1], [1, 2]]

    income, regions, provinces = (table.df for table in result.tables)
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


def test_drop_running_text_removes_headers_repeated_across_pages():
    footer = [block(page, 800, f"{page}/20 Incidencia de la pobreza") for page in (1, 2, 3)]
    caption = block(2, 100, "Cuadro 1. Pobreza")
    assert drop_running_text([*footer, caption]) == [caption]
