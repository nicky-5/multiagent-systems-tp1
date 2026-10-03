"""Extract tables and the text around them from a machine-readable PDF.

Only extract_tables() talks to PyMuPDF. It turns each page into plain data (RawTable,
TextBlock), and the rest of the pipeline is pure functions over that data:

    RawTable per page --merge_continuations--> Table --to_dataframe--> pandas DataFrame
                                                     --collect_context--> TableContext
"""

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

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
END_OF_DOC: Pos = (10**9, 0.0)


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
        return [p.page for p in self.parts]

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
    above: list[str] = field(default_factory=list)  # caption / paragraphs before it
    below: list[str] = field(default_factory=list)  # notes / source lines after it


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
    cells = [c for c in row if c is not None and not is_blank(c)]
    return bool(cells) and all(parse_number(c) is None or _is_year(c) for c in cells)


def split_header(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    """Split rows into (header_rows, body). The header is the leading label-only rows."""
    n = 0
    while n < min(len(rows), MAX_HEADER_ROWS) and _looks_like_header(rows[n]):
        n += 1
    if n == len(rows):  # a table of text only: assume a single header row
        n = 1
    return rows[:n], rows[n:]


def flatten_header(header_rows: list[Row]) -> list[str]:
    """Combine a multi-row header into one name per column.

    ["",       "2023",     None,    "2024",     None   ]
    ["Región", "Unidades", "Monto", "Unidades", "Monto"]
    -> ["Región", "2023 Unidades", "2023 Monto", "2024 Unidades", "2024 Monto"]

    None marks a cell covered by a merged cell to its left, so we copy that text
    across. A vertical merge would also give None, but grouped column headers
    (one year spanning several columns) are by far the more common case.
    """
    filled = []
    for row in header_rows:
        out, last = [], ""
        for cell in row:
            last = last if cell is None else cell
            out.append(last)
        filled.append(out)

    names = []
    for col in zip(*filled):  # zip(*rows) transposes rows into columns
        parts = []
        for part in col:
            if part and (not parts or parts[-1] != part):
                parts.append(part)
        names.append(" ".join(parts))
    return unique_names(names)


def unique_names(names: list[str]) -> list[str]:
    """Name empty columns column_N and number duplicates, so every name is a valid DataFrame column."""
    seen: dict[str, int] = {}
    out = []
    for i, name in enumerate(names):
        name = name or f"column_{i + 1}"
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return out


# --- continuation tables -----------------------------------------------------


def is_continuation(prev: Table, raw: RawTable, header: list[Row]) -> bool:
    """assignment.md §5.1: a table at the start of the next page, with the same number
    of columns, that either repeats the header row or has no header row."""
    last = prev.parts[-1]
    return (
        raw.page == last.page + 1
        and last.is_last_on_page
        and raw.index_on_page == 0
        and len(raw.rows[0]) == prev.n_cols
        and (not header or header == prev.header_rows)
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
        columns = [f"column_{i + 1}" for i in range(table.n_cols)]
    body = [[c or "" for c in row] for row in table.body]  # merged body cells -> ""
    body = [row for row in body if any(row)]
    decimal_sep = infer_decimal_sep(c for row in body for c in row)

    data = {}
    for i, name in enumerate(columns):
        cells = [row[i] for row in body]
        filled = [c for c in cells if not is_blank(c)]
        numbers = [parse_number(c, decimal_sep) for c in cells]
        if filled and all(parse_number(c, decimal_sep) is not None for c in filled):
            if all("%" in c for c in filled) and "%" not in name:
                name = f"{name} (%)"
            integral = all(n is None or n.is_integer() for n in numbers)
            data[name] = pd.Series(numbers, dtype="Int64" if integral else "Float64")
        else:
            data[name] = pd.Series([None if is_blank(c) else c for c in cells], dtype="string")
    return pd.DataFrame(data)


# --- context -----------------------------------------------------------------


def collect_context(
    blocks: list[TextBlock],
    start: Pos,
    end: Pos,
    prev_end: Pos = (0, 0.0),
    next_start: Pos = END_OF_DOC,
    max_above: int = 4,
    max_below: int = 2,
) -> TableContext:
    """Pick the text that describes the table spanning start..end.

    blocks must be in reading order and must not include the cells of any table.
    "Above" stops at the nearest heading or the previous table, whichever is
    closer; "below" stops at the next heading or table.
    """
    headings = [b for b in blocks if b.is_heading]
    before = [b for b in blocks if b.top < start]  # compare tops: captions overlap the table slightly
    heading = next((b for b in reversed(before) if b.is_heading), None)

    lower = max(prev_end, heading.top if heading else (0, 0.0))
    above = [b.text for b in before if not b.is_heading and b.top > lower][-max_above:]

    below = []
    for b in blocks:
        if b.top <= (end[0], end[1] - 3):  # -3pt tolerance, as for captions
            continue
        if b.top >= next_start or b.is_heading or len(below) == max_below:
            break
        below.append(b.text)

    return TableContext(
        document_title=headings[0].text if headings else None,
        heading=heading.text if heading else None,
        above=above,
        below=below,
    )


# --- PyMuPDF -----------------------------------------------------------------


def _inside(bbox: BBox, area: BBox) -> bool:
    """True if bbox's centre lies within area."""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return area[0] <= cx <= area[2] and area[1] <= cy <= area[3]


def _page_blocks(
    page: pymupdf.Page, page_no: int, table_bboxes: list[BBox]
) -> list[tuple[TextBlock, float, bool]]:
    """Text blocks outside tables, each with its largest font size and whether it's all bold."""
    out = []
    # get_text() returns str, list or dict depending on its argument, so its declared
    # return type is a union. cast() tells the type checker which one we know we get;
    # it does nothing at runtime.
    page_dict = cast(dict[str, Any], page.get_text("dict"))
    for b in page_dict["blocks"]:
        spans = [s for line in b.get("lines", []) for s in line["spans"] if s["text"].strip()]
        x0, y0, x1, y1 = b["bbox"]
        bbox = (x0, y0, x1, y1)
        if not spans or any(_inside(bbox, t) for t in table_bboxes):
            continue
        text = " ".join(" ".join(s["text"] for s in line["spans"]) for line in b["lines"])
        bold = all(s["flags"] & pymupdf.TEXT_FONT_BOLD or "Bold" in s["font"] for s in spans)
        block = TextBlock(page=page_no, bbox=bbox, text=" ".join(text.split()))
        out.append((block, max(s["size"] for s in spans), bold))
    return out


def extract_tables(pdf: str | Path | bytes) -> ExtractionResult:
    """Extract every table in the PDF as a DataFrame plus its surrounding text."""
    doc = pymupdf.open(stream=pdf, filetype="pdf") if isinstance(pdf, bytes) else pymupdf.open(pdf)
    raw_tables: list[RawTable] = []
    sized_blocks: list[tuple[TextBlock, float, bool]] = []
    warnings: list[str] = []

    # page.number is typed int | None, so count pages ourselves (1-based)
    for page_no, page in enumerate(doc.pages(), start=1):
        if not cast(str, page.get_text()).strip():
            if page.get_images():
                warnings.append(f"Page {page_no} has no text layer (scanned image?); skipped.")
            continue

        finder = page.find_tables()  # typed as possibly None
        found = finder.tables if finder else []
        for i, tab in enumerate(found):
            rows = [[clean_cell(c) for c in row] for row in tab.extract()]
            if all(is_blank(c) for row in rows for c in row):
                warnings.append(f"Page {page_no}, table {i + 1}: no readable text; skipped.")
                continue
            raw_tables.append(RawTable(page_no, i, i == len(found) - 1, tuple(tab.bbox), rows))
        sized_blocks += _page_blocks(page, page_no, [tuple(t.bbox) for t in found])

        for img in page.get_image_info():
            if img["bbox"][3] - img["bbox"][1] > 100:  # small images are usually logos
                warnings.append(
                    f"Page {page_no} contains an image; any table inside it was not extracted."
                )

    # Headings are bold or clearly larger than the document's typical (median) text size.
    body_size = statistics.median(size for _, size, _ in sized_blocks) if sized_blocks else 0
    blocks = []
    for block, size, bold in sized_blocks:
        block.is_heading = (bold or size > body_size + 1.5) and len(block.text) < 150
        blocks.append(block)

    tables = merge_continuations(raw_tables)
    result = []
    for i, table in enumerate(tables):
        context = collect_context(
            blocks,
            table.start,
            table.end,
            prev_end=tables[i - 1].end if i > 0 else (0, 0.0),
            next_start=tables[i + 1].start if i + 1 < len(tables) else END_OF_DOC,
        )
        result.append(ExtractedTable(table.pages, table.parts[0].index_on_page, to_dataframe(table), context))
    return ExtractionResult(result, warnings)
