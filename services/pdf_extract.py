"""Extract tables and the text around them from a machine-readable PDF.

Only extract_tables() talks to PyMuPDF. It turns each page into plain data (RawTable,
TextBlock), and the rest of the pipeline is pure functions over that data:

    RawTable per page --merge_continuations--> Table --to_dataframe--> pandas DataFrame
                                                     --collect_context--> TableContext
"""

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import accumulate, groupby, takewhile
from pathlib import Path
from typing import Any, NamedTuple, Protocol, cast

import pandas as pd
import pymupdf

from services.numbers import infer_decimal_sep, is_blank, parse_number

# A cell is a string, or None when PyMuPDF reports it as covered by a merged neighbour.
Row = list[str | None]
BBox = tuple[float, float, float, float]  # (x0, y0, x1, y1), points, origin top-left
# A position in the document. Tuples compare element by element, so (page, y)
# orders by page first and then top-to-bottom within the page.
Pos = tuple[int, float]

MAX_HEADER_ROWS = 3
START_OF_DOC: Pos = (0, 0.0)
END_OF_DOC: Pos = (10**9, 0.0)
# A caption's box includes its font descent, so it can overlap the table's edge by a few points.
CAPTION_OVERLAP = 3
MIN_CONTENT_IMAGE_HEIGHT = 100  # smaller images are usually logos
MAX_HEADING_LENGTH = 150
HEADING_SIZE_MARGIN = 1.5


@dataclass
class RawTable:
    """One table exactly as PyMuPDF found it on a single page."""

    page: int  # 1-based
    index_on_page: int
    is_last_on_page: bool
    bbox: BBox
    rows: list[Row]


@dataclass
class Table:
    """One logical table: one RawTable, plus any continuations on the following pages."""

    parts: list[RawTable]
    header_rows: list[Row]
    body: list[Row]

    @property
    def pages(self) -> list[int]:
        return [part.page for part in self.parts]

    @property
    def n_cols(self) -> int:
        return len(self.parts[0].rows[0])

    @property
    def start(self) -> Pos:
        return (self.parts[0].page, self.parts[0].bbox[1])

    @property
    def end(self) -> Pos:
        return (self.parts[-1].page, self.parts[-1].bbox[3])


@dataclass
class TextBlock:
    page: int
    bbox: BBox
    text: str
    is_heading: bool = False

    @property
    def top(self) -> Pos:
        return (self.page, self.bbox[1])


@dataclass
class TableContext:
    document_title: str | None = None
    heading: str | None = None  # nearest section heading above the table
    above: list[str] = field(default_factory=list[str])  # caption / paragraphs before it
    below: list[str] = field(default_factory=list[str])  # notes / source lines after it


@dataclass
class ExtractedTable:
    pages: list[int]
    index_on_page: int  # position on its first page; with the page, identifies the table
    df: pd.DataFrame
    context: TableContext


@dataclass
class ExtractionResult:
    tables: list[ExtractedTable]
    warnings: list[str]  # things we skipped, to be shown in the UI


# --- cells and headers -------------------------------------------------------


def clean_cell(cell: str | None) -> str | None:
    """Collapse the line breaks and repeated spaces of wrapped cells. Keeps None."""
    return None if cell is None else " ".join(cell.split())


def _is_year(text: str) -> bool:
    return re.fullmatch(r"(19|20)\d\d", text.strip()) is not None


def _looks_like_header(row: Row) -> bool:
    """Header rows hold labels: no cell is a number, except years such as "2023"."""
    cells = [cell for cell in row if cell is not None and not is_blank(cell)]
    return bool(cells) and all(parse_number(cell) is None or _is_year(cell) for cell in cells)


def split_header(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    """Split rows into (header_rows, body). The header is the leading label-only rows."""
    header_size = 0
    while header_size < min(len(rows), MAX_HEADER_ROWS) and _looks_like_header(rows[header_size]):
        header_size += 1
    if header_size == len(rows):  # a table of text only: assume a single header row
        header_size = 1
    return rows[:header_size], rows[header_size:]


def flatten_header(header_rows: list[Row]) -> list[str]:
    """Combine a multi-row header into one name per column.

    ["",       "2023",     None,    "2024",     None   ]
    ["Región", "Unidades", "Monto", "Unidades", "Monto"]
    -> ["Región", "2023 Unidades", "2023 Monto", "2024 Unidades", "2024 Monto"]

    None marks a cell covered by a merged cell to its left, so we copy that text
    across. A vertical merge would also give None, but grouped column headers
    (one year spanning several columns) are by far the more common case.
    """
    filled_rows = [_fill_merged_cells(row) for row in header_rows]
    columns = zip(*filled_rows)  # zip(*rows) transposes rows into columns
    return unique_names([_join_labels(column_labels) for column_labels in columns])


def _fill_merged_cells(row: Row) -> list[str]:
    """Replace each None with the text to its left: ["2023", None] -> ["2023", "2023"]."""
    def carry_left(last_text: str, cell: str | None) -> str:
        return last_text if cell is None else cell

    return list(accumulate(row, carry_left, initial=""))[1:]  # drop the initial ""


def _join_labels(labels: Iterable[str]) -> str:
    """Join a column's header labels, skipping empty ones and repeats: ("2023", "2023", "Monto") -> "2023 Monto"."""
    non_empty = (label for label in labels if label)
    return " ".join(label for label, _ in groupby(non_empty))  # groupby collapses consecutive repeats


def unique_names(names: list[str]) -> list[str]:
    """Name empty columns column_N and number duplicates, so every name is a valid DataFrame column."""
    named = [name or f"column_{position}" for position, name in enumerate(names, start=1)]
    earlier_uses = [named[:index].count(name) for index, name in enumerate(named)]
    return [
        name if uses == 0 else f"{name} ({uses + 1})" for name, uses in zip(named, earlier_uses)
    ]


# --- continuation tables -----------------------------------------------------


def is_continuation(previous: Table, raw: RawTable, header: list[Row]) -> bool:
    """assignment.md §5.1: a table at the start of the next page, with the same number
    of columns, that either repeats the header row or has no header row."""
    previous_part = previous.parts[-1]
    return (
        raw.page == previous_part.page + 1
        and previous_part.is_last_on_page
        and raw.index_on_page == 0
        and len(raw.rows[0]) == previous.n_cols
        and (not header or header == previous.header_rows)
    )


def merge_continuations(raw_tables: list[RawTable]) -> list[Table]:
    """Group per-page tables (in reading order) into logical tables."""
    tables: list[Table] = []
    for raw in raw_tables:
        header, body = split_header(raw.rows)
        if tables and is_continuation(tables[-1], raw, header):
            tables[-1].parts.append(raw)
            tables[-1].body.extend(body)  # a repeated header is dropped here
        else:
            tables.append(Table(parts=[raw], header_rows=header, body=body))
    return tables


# --- typing ------------------------------------------------------------------


def to_dataframe(table: Table) -> pd.DataFrame:
    """Build a DataFrame, converting columns whose every non-blank cell is a number.

    Integer-only columns become "Int64" and other numeric columns "Float64". These are
    pandas' nullable dtypes, so empty cells become <NA> without turning ints into
    floats. Columns where every value had a % sign get " (%)" added to their name,
    because parse_number drops the sign.
    """
    if table.header_rows:
        columns = flatten_header(table.header_rows)
    else:
        columns = [f"column_{position}" for position in range(1, table.n_cols + 1)]
    body = [[cell or "" for cell in row] for row in table.body]  # merged body cells -> ""
    body = [row for row in body if any(row)]
    decimal_sep = infer_decimal_sep(cell for row in body for cell in row)

    return pd.DataFrame(
        dict(
            _typed_column(name, [row[column_index] for row in body], decimal_sep)
            for column_index, name in enumerate(columns)
        )
    )


def _typed_column(name: str, cells: list[str], decimal_sep: str | None) -> tuple[str, pd.Series]:
    """A (name, Series) pair: numeric if every non-blank cell parses as a number, text otherwise."""
    filled = [cell for cell in cells if not is_blank(cell)]
    numbers = [parse_number(cell, decimal_sep) for cell in cells]
    is_numeric = filled and all(parse_number(cell, decimal_sep) is not None for cell in filled)
    if not is_numeric:
        return name, pd.Series([None if is_blank(cell) else cell for cell in cells], dtype="string")

    if all("%" in cell for cell in filled) and "%" not in name:
        name = f"{name} (%)"
    is_integral = all(number is None or number.is_integer() for number in numbers)
    return name, pd.Series(numbers, dtype="Int64" if is_integral else "Float64")


# --- context -----------------------------------------------------------------


def collect_context(
    blocks: list[TextBlock],
    start: Pos,
    end: Pos,
    prev_end: Pos = START_OF_DOC,
    next_start: Pos = END_OF_DOC,
    max_above: int = 4,
    max_below: int = 2,
) -> TableContext:
    """Pick the text that describes the table spanning start..end.

    blocks must be in reading order and must not include the cells of any table.
    "Above" stops at the nearest heading or the previous table, whichever is
    closer; "below" stops at the next heading or table.
    """
    headings = [block for block in blocks if block.is_heading]
    # Compare tops rather than bottoms, because captions overlap the table slightly.
    before = [block for block in blocks if block.top < start]
    heading = next((block for block in reversed(before) if block.is_heading), None)

    above_starts_after = max(prev_end, heading.top if heading else START_OF_DOC)
    above = [
        block.text for block in before if not block.is_heading and block.top > above_starts_after
    ][-max_above:]

    end_page, end_y = end
    after_table = (block for block in blocks if block.top > (end_page, end_y - CAPTION_OVERLAP))
    before_next_section = takewhile(
        lambda block: block.top < next_start and not block.is_heading, after_table
    )
    below = [block.text for block in before_next_section][:max_below]

    return TableContext(
        document_title=headings[0].text if headings else None,
        heading=heading.text if heading else None,
        above=above,
        below=below,
    )


# --- PyMuPDF -----------------------------------------------------------------
# Most of PyMuPDF's API has no type annotations. These protocols declare the parts
# we call, so the type checker can still check our code against them.


class _PdfTable(Protocol):
    @property
    def bbox(self) -> Sequence[float]: ...
    def extract(self) -> list[Row]: ...


class _PdfTableFinder(Protocol):
    @property
    def tables(self) -> list[_PdfTable]: ...


class _PdfPage(Protocol):
    def get_text(self, option: str = "text") -> Any: ...
    def get_images(self) -> list[Any]: ...
    def find_tables(self) -> _PdfTableFinder | None: ...
    def get_image_info(self) -> list[dict[str, Any]]: ...


class _PdfDocument(Protocol):
    def pages(self) -> Iterable[_PdfPage]: ...


def _open_pdf(pdf: str | Path | bytes) -> _PdfDocument:
    doc = pymupdf.open(stream=pdf, filetype="pdf") if isinstance(pdf, bytes) else pymupdf.open(pdf)
    return cast(_PdfDocument, doc)


def _to_bbox(coordinates: Sequence[float]) -> BBox:
    x0, y0, x1, y1 = coordinates
    return (x0, y0, x1, y1)


class _StyledBlock(NamedTuple):
    block: TextBlock
    font_size: float  # the largest in the block
    is_bold: bool


def _center_is_inside(bbox: BBox, area: BBox) -> bool:
    left, top, right, bottom = area
    center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return left <= center_x <= right and top <= center_y <= bottom


def _page_blocks(page: _PdfPage, page_no: int, table_bboxes: list[BBox]) -> list[_StyledBlock]:
    """Text blocks outside tables, with their font size and boldness."""
    page_dict: dict[str, Any] = page.get_text("dict")
    styled_blocks = [_style_block(raw_block, page_no) for raw_block in page_dict["blocks"]]
    return [
        styled
        for styled in styled_blocks
        if styled is not None
        and not any(_center_is_inside(styled.block.bbox, table_bbox) for table_bbox in table_bboxes)
    ]


def _style_block(raw_block: dict[str, Any], page_no: int) -> _StyledBlock | None:
    """Convert one PyMuPDF text block, or None if it holds only whitespace."""
    lines = raw_block.get("lines", [])
    spans = [span for line in lines for span in line["spans"] if span["text"].strip()]
    if not spans:
        return None
    text = " ".join(" ".join(span["text"] for span in line["spans"]) for line in lines)
    return _StyledBlock(
        block=TextBlock(page=page_no, bbox=_to_bbox(raw_block["bbox"]), text=" ".join(text.split())),
        font_size=max(span["size"] for span in spans),
        is_bold=all(span["flags"] & pymupdf.TEXT_FONT_BOLD or "Bold" in span["font"] for span in spans),
    )


def _mark_headings(styled_blocks: list[_StyledBlock]) -> list[TextBlock]:
    """Headings are short and bold, or clearly larger than the document's median text size."""
    if not styled_blocks:
        return []
    body_size = statistics.median(styled.font_size for styled in styled_blocks)
    for styled in styled_blocks:
        is_large = styled.font_size > body_size + HEADING_SIZE_MARGIN
        is_short = len(styled.block.text) < MAX_HEADING_LENGTH
        styled.block.is_heading = (styled.is_bold or is_large) and is_short
    return [styled.block for styled in styled_blocks]


def extract_tables(pdf: str | Path | bytes) -> ExtractionResult:
    """Extract every table in the PDF as a DataFrame plus its surrounding text."""
    doc = _open_pdf(pdf)
    raw_tables: list[RawTable] = []
    styled_blocks: list[_StyledBlock] = []
    warnings: list[str] = []

    # page.number is typed int | None, so count pages ourselves (1-based)
    for page_no, page in enumerate(doc.pages(), start=1):
        page_text: str = page.get_text()
        if not page_text.strip():
            if page.get_images():
                warnings.append(f"Page {page_no} has no text layer (scanned image?); skipped.")
            continue

        finder = page.find_tables()
        found_tables = finder.tables if finder else []
        page_tables = [
            RawTable(
                page=page_no,
                index_on_page=table_index,
                is_last_on_page=table_index == len(found_tables) - 1,
                bbox=_to_bbox(found.bbox),
                rows=[[clean_cell(cell) for cell in row] for row in found.extract()],
            )
            for table_index, found in enumerate(found_tables)
        ]
        raw_tables += [table for table in page_tables if not _is_empty(table)]
        warnings += [
            f"Page {page_no}, table {table.index_on_page + 1}: no readable text; skipped."
            for table in page_tables
            if _is_empty(table)
        ]
        styled_blocks += _page_blocks(page, page_no, [table.bbox for table in page_tables])

        warnings += [
            f"Page {page_no} contains an image; any table inside it was not extracted."
            for image in page.get_image_info()
            if _is_content_image(image)
        ]

    blocks = _mark_headings(styled_blocks)
    tables = merge_continuations(raw_tables)
    extracted = [
        ExtractedTable(
            pages=table.pages,
            index_on_page=table.parts[0].index_on_page,
            df=to_dataframe(table),
            context=_context_of(tables, table_index, blocks),
        )
        for table_index, table in enumerate(tables)
    ]
    return ExtractionResult(extracted, warnings)


def _is_empty(table: RawTable) -> bool:
    return all(is_blank(cell) for row in table.rows for cell in row)


def _is_content_image(image: dict[str, Any]) -> bool:
    _, image_top, _, image_bottom = image["bbox"]
    return image_bottom - image_top > MIN_CONTENT_IMAGE_HEIGHT


def _context_of(tables: list[Table], table_index: int, blocks: list[TextBlock]) -> TableContext:
    """The text around tables[table_index], bounded by its neighbouring tables."""
    table = tables[table_index]
    is_first = table_index == 0
    is_last = table_index == len(tables) - 1
    return collect_context(
        blocks,
        table.start,
        table.end,
        prev_end=START_OF_DOC if is_first else tables[table_index - 1].end,
        next_start=END_OF_DOC if is_last else tables[table_index + 1].start,
    )
