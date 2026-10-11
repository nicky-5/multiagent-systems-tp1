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
import pymupdf.layout  # pyright: ignore[reportUnusedImport]  # with it, find_tables() uses layout analysis

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
MIN_FILLED_BODY_CELLS = 2  # fewer: a fragment (e.g. on a cover page), not a data table
# "Cuadro 4.2", "Table 3": a caption, which starts a new table
CAPTION = re.compile(r"(cuadro|tabla|table|gráfico|grafico|figura|figure)\s+\d", re.IGNORECASE)
TOC_LEADER = re.compile(r"\.{5,}")  # the dots of a table of contents
RANGE = re.compile(r"\[\s*([^;\]]+?)\s*;\s*([^;\]]+?)\s*\]")  # "[41,4 ; 43,7]", e.g. a confidence interval
MAX_HEADING_LENGTH = 150
MIN_RUNNING_PAGES = 3  # text repeated on this many pages is a running header or footer
MAX_CAPTION_LENGTH = 300  # longer: a paragraph that starts with "Cuadro 1 ...", not a caption
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
    is_caption: bool = False  # "Cuadro 4.2 ...", "Table 2: ..."

    @property
    def top(self) -> Pos:
        return (self.page, self.bbox[1])


@dataclass
class TableContext:
    document_title: str | None = None
    heading: str | None = None  # nearest section heading above the table
    above: list[str] = field(default_factory=list[str])  # caption / paragraphs before it
    below: list[str] = field(default_factory=list[str])  # notes / source lines after it
    # header labels over the whole table, e.g. a year or unit ("2026", "Pesos")
    header_notes: list[str] = field(default_factory=list[str])


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
    # A last label-only row with text in the first column only ("Pobreza" above the
    # poverty rows) is the first section row of the body, not part of the header
    while header_size > 1 and _is_section_row(rows[header_size - 1]):
        header_size -= 1
    return rows[:header_size], rows[header_size:]


def _is_section_row(row: Sequence[str | None]) -> bool:
    """A row labelling the rows below it: text in the first column, nothing else."""
    first, *rest = row
    return (
        len(row) >= 3
        and first is not None
        and not is_blank(first)
        and all(cell is None or is_blank(cell) for cell in rest)
    )


def tidy_header(header_rows: list[Row]) -> tuple[list[Row], list[str]]:
    """Handle group labels: header labels that belong to several columns but sit in one.

    A header row with a single label over the data columns ("2026", "Pesos", "%")
    describes the whole table, so it's returned as a note instead of becoming part of
    one column's name. In a row above the last with several ("Hogares" ... "Personas"),
    each label also names the empty cells nearest to it; in the last row, empty cells
    are columns without a name. The first column (the row labels) is never given a
    group label. Returns (header_rows, notes).
    """
    rows: list[Row] = []
    notes: list[str] = []
    for row_index, row in enumerate(header_rows):
        data = _fill_merged_cells(row)[1:]  # merged cells carry their label right
        labels = [label for label, _ in groupby(label for label in data if label)]
        is_last = row_index == len(header_rows) - 1
        if len(labels) == 1 and "" in data:
            notes.append(labels[0])
            rows.append([row[0]] + [""] * len(data))
        elif len(labels) > 1 and "" in data and not is_last:
            rows.append([row[0], *_spread_to_nearest(data)])
        else:
            rows.append(row)
    return rows, notes


def _spread_to_nearest(cells: list[str]) -> list[str]:
    """Fill each empty cell with the nearest label, the left one on a tie:
    ["", "A", "", "", "B", ""] -> ["A", "A", "A", "B", "B", "B"]."""
    labelled = [index for index, cell in enumerate(cells) if cell]
    nearest = [min(labelled, key=lambda index: abs(index - position)) for position in range(len(cells))]
    return [cell or cells[nearest[position]] for position, cell in enumerate(cells)]


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


def is_continuation(
    previous: Table, raw: RawTable, header: list[Row], captions: Sequence[Pos] = ()
) -> bool:
    """assignment.md §5.1: a table at the start of the next page, with the same number
    of columns, that either repeats the header row or has no header row. A caption
    between the two ("Cuadro 4.2") means it's a new table with the same layout."""
    previous_part = previous.parts[-1]
    return (
        raw.page == previous_part.page + 1
        and previous_part.is_last_on_page
        and raw.index_on_page == 0
        and len(raw.rows[0]) == previous.n_cols
        and (not header or header == previous.header_rows)
        and not any(previous.end < caption < (raw.page, raw.bbox[1]) for caption in captions)
    )


def merge_continuations(raw_tables: list[RawTable], captions: Sequence[Pos] = ()) -> list[Table]:
    """Group per-page tables (in reading order) into logical tables. captions are the
    positions of table captions, which separate tables."""
    tables: list[Table] = []
    for raw in raw_tables:
        header, body = split_header(raw.rows)
        if tables and is_continuation(tables[-1], raw, header, captions):
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

    Section rows ("Pobreza", then its rows, then "Indigencia" and its rows) become a
    "Section" column, so rows with the same label in different sections stay apart.
    A single label row splits nothing, so it stays a row.
    """
    if table.header_rows:
        columns = flatten_header(tidy_header(table.header_rows)[0])
    else:
        columns = [f"column_{position}" for position in range(1, table.n_cols + 1)]
    body = [[cell or "" for cell in row] for row in table.body]  # merged body cells -> ""
    body = [row for row in body if any(row)]
    columns, body = merge_range_rows(columns, body)
    sections, rows = split_sections(body)
    if len({section for section in sections if section}) >= 2:
        columns = unique_names(["Section", *columns])
        body = [[section, *row] for section, row in zip(sections, rows, strict=True)]
    decimal_sep = infer_decimal_sep(cell for row in body for cell in row)

    return pd.DataFrame(
        dict(
            _typed_column(name, [row[column_index] for row in body], decimal_sep)
            for column_index, name in enumerate(columns)
        )
    )


def merge_range_rows(columns: list[str], body: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Move rows of ranges into the row above them, as two columns per value column.

    Statistical tables often follow each estimate with its confidence interval:
        Hogares                         42,5            24,4
        Intervalos de confianza (90%)   [41,4 ; 43,7]   [23,3 ; 25,5]
    Mixed in one column, the ranges would keep the estimates from being numbers, so
    "Hogares" gets the columns "<column>: <range label> (min)" and "(max)" instead.
    Only done when every range row has the same label and follows a value row.
    """
    range_rows = [index for index, row in enumerate(body) if _is_range_row(row)]
    labels = {body[index][0] for index in range_rows}
    if not range_rows or len(labels) != 1 or 0 in range_rows:
        return columns, body
    if any(index - 1 in range_rows for index in range_rows):
        return columns, body
    label = labels.pop()

    new_columns = [columns[0]]
    for name in columns[1:]:
        new_columns += [name, f"{name}: {label} (min)", f"{name}: {label} (max)"]
    merged: list[list[str]] = []
    for index, row in enumerate(body):
        if index in range_rows:
            continue
        ranges = body[index + 1] if index + 1 in range_rows else [""] * len(row)
        new_row = [row[0]]
        for value, range_cell in zip(row[1:], ranges[1:], strict=True):
            match = RANGE.fullmatch(range_cell.strip())
            new_row += [value, *(match.groups() if match else ("", ""))]
        merged.append(new_row)
    return unique_names(new_columns), merged


def _is_range_row(row: list[str]) -> bool:
    data = [cell for cell in row[1:] if not is_blank(cell)]
    return bool(data) and all(RANGE.fullmatch(cell.strip()) for cell in data)


def split_sections(body: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Remove section rows, returning (the section of each remaining row, the rows).
    Rows before the first section row have the section ""."""
    sections: list[str] = []
    rows: list[list[str]] = []
    current = ""
    for row in body:
        if _is_section_row(row):
            current = row[0]
        else:
            sections.append(current)
            rows.append(row)
    return sections, rows


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
    closer, and always keeps the table's caption; "below" stops at the next
    heading, caption or table.
    """
    headings = [block for block in blocks if block.is_heading]
    # Compare tops rather than bottoms, because captions overlap the table slightly.
    before = [block for block in blocks if block.top < start]
    heading = next((block for block in reversed(before) if block.is_heading), None)

    above_starts_after = max(prev_end, heading.top if heading else START_OF_DOC)
    candidates = [block for block in before if not block.is_heading and block.top > above_starts_after]
    above = candidates[-max_above:]
    caption = next((block for block in reversed(candidates) if block.is_caption), None)
    if caption is not None and caption not in above:  # a long paragraph sits between them
        above = [caption, *above[1:]]

    end_page, end_y = end
    after_table = (block for block in blocks if block.top > (end_page, end_y - CAPTION_OVERLAP))
    before_next_section = takewhile(
        lambda block: block.top < next_start and not block.is_heading and not block.is_caption,
        after_table,
    )
    below = [block.text for block in before_next_section][:max_below]

    return TableContext(
        document_title=headings[0].text if headings else None,
        heading=heading.text if heading else None,
        above=[block.text for block in above],
        below=below,
    )


# --- PyMuPDF -----------------------------------------------------------------
# Most of PyMuPDF's API has no type annotations. These protocols declare the parts
# we call, so the type checker can still check our code against them.


class _SpanCell(Protocol):
    text: str
    colspan: int
    rowspan: int


class _PdfTable(Protocol):
    @property
    def bbox(self) -> Sequence[float]: ...
    @property
    def col_count(self) -> int: ...
    @property
    def placements(self) -> list[list[_SpanCell]] | None: ...
    def extract(self) -> list[Row]: ...


class _PdfTableFinder(Protocol):
    @property
    def tables(self) -> list[_PdfTable]: ...


class _PdfPage(Protocol):
    def get_text(self, option: str = "text") -> Any: ...
    def get_images(self) -> list[Any]: ...
    def find_tables(self, *, union: bool = False, refine: bool = False) -> _PdfTableFinder | None: ...
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


def drop_running_text(blocks: list[TextBlock]) -> list[TextBlock]:
    """Remove running headers and footers ("9/20 Incidencia de la pobreza ..."): text
    that repeats, apart from its numbers, on several pages. It describes the document,
    not the table it happens to sit next to."""

    def key(block: TextBlock) -> str:
        return re.sub(r"\d+", "#", block.text)

    pages_by_text: dict[str, set[int]] = {}
    for block in blocks:
        pages_by_text.setdefault(key(block), set()).add(block.page)
    return [block for block in blocks if len(pages_by_text[key(block)]) < MIN_RUNNING_PAGES]


def _mark_headings(styled_blocks: list[_StyledBlock]) -> list[TextBlock]:
    """Headings are short and bold, or clearly larger than the document's median text size,
    and contain a word (a large "75,6%" in an infographic is not a heading).
    A table caption ("Cuadro 4.2 ...") is not a heading: it describes only its own table."""
    if not styled_blocks:
        return []
    body_size = statistics.median(styled.font_size for styled in styled_blocks)
    for styled in styled_blocks:
        is_large = styled.font_size > body_size + HEADING_SIZE_MARGIN
        is_short = len(styled.block.text) < MAX_HEADING_LENGTH
        styled.block.is_caption = (
            CAPTION.match(styled.block.text) is not None and len(styled.block.text) < MAX_CAPTION_LENGTH
        )
        has_word = re.search(r"[^\W\d_]", styled.block.text) is not None  # any letter
        styled.block.is_heading = (
            (styled.is_bold or is_large) and is_short and has_word and not styled.block.is_caption
        )
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

        # Layout analysis finds tables drawn without ruling lines (e.g. shaded bands
        # instead of cell borders); union combines it with the line-based finder,
        # and refine splits the rows and columns the line grid merged.
        finder = page.find_tables(union=True, refine=True)
        found_tables = [found for found in (finder.tables if finder else []) if not _is_toc(found)]
        page_tables = [
            RawTable(
                page=page_no,
                index_on_page=table_index,
                is_last_on_page=table_index == len(found_tables) - 1,
                bbox=_to_bbox(found.bbox),
                rows=[[clean_cell(cell) for cell in row] for row in _rows(found)],
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

    # PDFs store text in drawing order, which can differ from reading order (a
    # caption drawn after its table); collect_context needs reading order
    styled_blocks.sort(key=lambda styled: (styled.block.page, styled.block.bbox[1], styled.block.bbox[0]))
    blocks = drop_running_text(_mark_headings(styled_blocks))
    captions = [block.top for block in blocks if block.is_caption]
    tables = merge_continuations(raw_tables, captions)
    extracted: list[ExtractedTable] = []
    for table_index, table in enumerate(tables):
        if _is_fragment(table):
            warnings.append(f"Page {table.pages[0]}: a table with almost no data was skipped.")
            continue
        context = _context_of(tables, table_index, blocks)
        context.header_notes = tidy_header(table.header_rows)[1]
        extracted.append(
            ExtractedTable(
                pages=table.pages,
                index_on_page=table.parts[0].index_on_page,
                df=to_dataframe(table),
                context=context,
            )
        )
    return ExtractionResult(extracted, warnings)


def _rows(found: _PdfTable) -> list[Row]:
    """The table's cells, from refine's grid when there is one. Its cells keep labels
    that span a column boundary whole ("2026"), where extract() cuts them ("20", "26")."""
    if found.placements is None:
        return found.extract()
    return placements_to_rows(
        [[(cell.text, cell.colspan, cell.rowspan) for cell in row] for row in found.placements],
        found.col_count,
    ) or found.extract()


def placements_to_rows(placements: list[list[tuple[str, int, int]]], n_cols: int) -> list[Row] | None:
    """Turn an HTML-style grid of (text, colspan, rowspan) cells, which leaves out the
    positions a span covers, into full rows. A position covered from the left is None
    (as extract() reports merged cells) and one covered from above is "".
    None if the cells don't fit n_cols columns."""
    covered_from_above: set[tuple[int, int]] = set()  # (row, column)
    rows: list[Row] = []
    for row_index, cells in enumerate(placements):
        row: Row = []
        queue = iter(cells)
        while len(row) < n_cols:
            column = len(row)
            if (row_index, column) in covered_from_above:
                row.append("")
                continue
            cell = next(queue, None)
            if cell is None:
                return None
            text, colspan, rowspan = cell
            for spanned in range(column, column + colspan):
                for below in range(row_index + 1, row_index + rowspan):
                    covered_from_above.add((below, spanned))
            row += [text, *[None] * (colspan - 1)]
        if len(row) != n_cols or next(queue, None) is not None:
            return None
        rows.append(row)
    return rows


def _is_toc(found: _PdfTable) -> bool:
    """A table of contents: its entries end in dot leaders ("Introducción .......... 4")."""
    return any(cell and TOC_LEADER.search(cell) for row in found.extract() for cell in row)


def _is_fragment(table: Table) -> bool:
    filled = sum(1 for row in table.body for cell in row if cell is not None and not is_blank(cell))
    return filled < MIN_FILLED_BODY_CELLS


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
