"""Extract tables and the text around them from a machine-readable PDF.

Only extract_tables() talks to PyMuPDF. It turns each page into plain data (RawTable,
TextBlock), and the rest of the pipeline is pure functions over that data:

    RawTable per page --merge_continuations--> Table --to_dataframe--> pandas DataFrame
                                                     --collect_context--> TableContext
"""

import re
import statistics
from collections.abc import Callable, Iterable, Sequence
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
Word = tuple[float, float, float, float, str]  # (x0, y0, x1, y1, text), as PyMuPDF's "words"
Segment = tuple[float, float, float]  # a horizontal rule: (y, x0, x1)

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
LINE_TOLERANCE = 3  # words whose vertical centres differ less than this are on one line
PHRASE_GAP = 5  # words closer than this (points) form one label; columns are further apart
RULE_TOLERANCE = 2  # slack when comparing rule and text positions


@dataclass
class RawTable:
    """One table exactly as PyMuPDF found it on a single page."""

    page: int  # 1-based
    index_on_page: int
    is_last_on_page: bool
    bbox: BBox
    rows: list[Row]
    header_size: int | None = None  # known when the header was rebuilt from the page
    header_notes: list[str] = field(default_factory=list[str])  # labels over the whole table


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
    """Take out header rows with a single label over all the data columns ("2026",
    "Pesos", "%", "Total 31 aglomerados urbanos"): it describes the whole table, so it's
    returned as a note instead of becoming part of every column's name. The first
    column (the row labels) doesn't count. Returns (header_rows, notes)."""
    rows: list[Row] = []
    notes: list[str] = []
    for row in header_rows:
        data = _fill_merged_cells(row)[1:]  # merged cells carry their label right
        labels = [label for label, _ in groupby(label for label in data if label)]
        if len(labels) == 1 and len(data) > 1:
            notes.append(labels[0])
            rows.append([row[0]] + [""] * len(data))
        else:
            rows.append(row)
    return rows, notes


# --- headers from the page ----------------------------------------------------


class _Phrase(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def centre(self) -> float:
        return (self.x0 + self.x1) / 2


def header_from_words(
    words: Sequence[Word], rules: Sequence[Segment], columns: Sequence[tuple[float, float]]
) -> tuple[list[str], list[str]]:
    """Name each column from the header's words and horizontal rules.

    columns are the x-ranges of the table's columns. Words close together on a line
    form a label, which belongs to the column it's centred over. A column's name is:
      - the group labels over it: a label with a rule under it spanning several columns
        ("Año 2025" over three quarters) applies to every column under that rule;
      - then its own labels, top to bottom ("Pobres" + "indigentes", a wrapped label).
    A label alone in its band of the header that isn't part of a wrapped label ("2026",
    "Pesos") describes the whole table, as does a group over every data column. Those
    are returned as notes. The first column (row labels) never gets a group label.
    Returns (names, notes).
    """
    centres = [(left + right) / 2 for left, right in columns]
    data_columns = set(range(1, len(columns)))
    phrases = [phrase for line in _lines(words) for phrase in _phrases(line)]
    runs = _merge_rules(rules)

    def column_of(phrase: _Phrase) -> int:
        return min(range(len(columns)), key=lambda j: _distance(phrase.centre, columns[j]))

    def rule_under(phrase: _Phrase) -> Segment | None:
        """The nearest rule below the phrase, with no other label in between."""
        for y, x0, x1 in sorted(runs):
            if y < phrase.y1 - RULE_TOLERANCE or not x0 - RULE_TOLERANCE <= phrase.centre <= x1 + RULE_TOLERANCE:
                continue
            between = [
                p for p in phrases
                if p is not phrase and phrase.y1 - RULE_TOLERANCE < p.y0 and p.y1 < y + RULE_TOLERANCE
                and p.x0 < phrase.x1 and phrase.x0 < p.x1
            ]
            return None if between else (y, x0, x1)
        return None

    # Group labels and the columns they cover. A group has labels under its rule in at
    # least two of those columns (a rule closing the header, above a lone "%", doesn't
    # make one). Where two groups on one rule claim a column (a grid draws one rule
    # under the whole header row), the nearer one wins.
    groups: dict[_Phrase, set[int]] = {}
    for phrase in phrases:
        rule = rule_under(phrase)
        if rule is None or column_of(phrase) == 0:
            continue
        y, x0, x1 = rule
        spanned = {j for j in data_columns if x0 - RULE_TOLERANCE <= centres[j] <= x1 + RULE_TOLERANCE}
        labelled_below = {column_of(p) for p in phrases if p.y0 > y - RULE_TOLERANCE} & spanned
        if len(spanned) >= 2 and len(labelled_below) >= 2:
            groups[phrase] = spanned
    for phrase, spanned in groups.items():
        rivals = [other for other in groups if other is not phrase and abs(other.y0 - phrase.y0) < LINE_TOLERANCE]
        groups[phrase] = {
            j for j in spanned
            if all(abs(centres[j] - phrase.centre) <= abs(centres[j] - other.centre) for other in rivals if j in groups[other])
        }

    notes: list[str] = []
    stacks: list[list[_Phrase]] = [[] for _ in columns]
    for phrase in sorted(phrases, key=lambda p: (p.y0, p.x0)):
        column = column_of(phrase)
        if phrase in groups:
            if groups[phrase] >= data_columns:
                notes.append(phrase.text)
            continue
        if column != 0 and len(data_columns) >= 2 and _stands_alone(phrase, phrases, columns[column], column_of):
            notes.append(phrase.text)
        else:
            stacks[column].append(phrase)

    names: list[str] = []
    for j in range(len(columns)):
        own = " ".join(p.text for p in stacks[j])
        over = sorted((p for p, spanned in groups.items() if j in spanned and not spanned >= data_columns), key=lambda p: p.y0)
        # "Pobres" over "Pobres no indigentes" adds nothing
        labels = [p.text for p in over if not own.startswith(p.text)] + [p.text for p in stacks[j]]
        names.append(" ".join(label for label, _ in groupby(labels)))
    return unique_names(names), notes


def _stands_alone(
    phrase: _Phrase, phrases: Sequence[_Phrase], column: tuple[float, float], column_of: Callable[[_Phrase], int]
) -> bool:
    """True if no other label over the data columns shares the phrase's band of the
    header, and it isn't a line of a wrapped label (another label just above or below
    it in its column; a label straddling two columns isn't). The row labels' name
    ("Región") often spans the whole header height, so it doesn't count."""
    others = [p for p in phrases if p is not phrase]
    shares_band = any(p.y0 < phrase.y1 + 1 and phrase.y0 < p.y1 + 1 for p in others if column_of(p) != 0)
    height = phrase.y1 - phrase.y0
    left, right = column
    inside = left - RULE_TOLERANCE <= phrase.x0 and phrase.x1 <= right + RULE_TOLERANCE
    wrapped = inside and any(
        _distance(p.centre, column) == 0
        and (0 <= p.y0 - phrase.y1 < height / 2 or 0 <= phrase.y0 - p.y1 < height / 2)
        for p in others
    )
    return not shares_band and not wrapped


def _distance(x: float, column: tuple[float, float]) -> float:
    left, right = column
    return 0 if left <= x <= right else min(abs(x - left), abs(x - right))


def _lines(words: Sequence[Word]) -> list[list[Word]]:
    """Group words into lines of text, top to bottom."""
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        centre = (word[1] + word[3]) / 2
        if lines and abs(centre - (lines[-1][0][1] + lines[-1][0][3]) / 2) < LINE_TOLERANCE:
            lines[-1].append(word)
        else:
            lines.append([word])
    return lines


def _phrases(line: Sequence[Word]) -> list[_Phrase]:
    """Join a line's words into labels."""
    phrases: list[_Phrase] = []
    for x0, y0, x1, y1, text in sorted(line, key=lambda w: w[0]):
        if phrases and x0 - phrases[-1].x1 < PHRASE_GAP:
            last = phrases[-1]
            phrases[-1] = _Phrase(last.x0, min(last.y0, y0), x1, max(last.y1, y1), f"{last.text} {text}")
        else:
            phrases.append(_Phrase(x0, y0, x1, y1, text))
    return phrases


def _merge_rules(rules: Sequence[Segment]) -> list[Segment]:
    """Join rule segments that touch on the same line: a rule drawn cell by cell is one
    rule, but a gap between segments separates two groups."""
    merged: list[Segment] = []
    for y, x0, x1 in sorted(rules, key=lambda r: (round(r[0]), r[1])):
        if merged and abs(merged[-1][0] - y) < 1 and x0 - merged[-1][2] <= 1:
            merged[-1] = (merged[-1][0], merged[-1][1], max(merged[-1][2], x1))
        else:
            merged.append((y, x0, x1))
    return merged


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
        if raw.header_size is None:
            header, body = split_header(raw.rows)
        else:
            header, body = raw.rows[: raw.header_size], raw.rows[raw.header_size :]
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


class _PdfRow(Protocol):
    @property
    def bbox(self) -> Sequence[float]: ...
    @property
    def cells(self) -> list[Sequence[float] | None]: ...


class _PdfTable(Protocol):
    @property
    def bbox(self) -> Sequence[float]: ...
    @property
    def rows(self) -> list[_PdfRow]: ...
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
    def get_drawings(self) -> list[dict[str, Any]]: ...


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
        words: list[Word] = [cast(Word, tuple(w[:5])) for w in page.get_text("words")]
        rules = _horizontal_rules(page)
        page_tables = [
            _raw_table(found, page_no, table_index, len(found_tables), words, rules)
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
        context.header_notes = table.parts[0].header_notes or tidy_header(table.header_rows)[1]
        extracted.append(
            ExtractedTable(
                pages=table.pages,
                index_on_page=table.parts[0].index_on_page,
                df=to_dataframe(table),
                context=context,
            )
        )
    return ExtractionResult(extracted, warnings)


def _raw_table(
    found: _PdfTable, page_no: int, index: int, n_tables: int, words: list[Word], rules: list[Segment]
) -> RawTable:
    """One found table as plain data, with its header rebuilt from the page when it has one."""
    rows = [[clean_cell(cell) for cell in row] for row in _rows(found)]
    raw = RawTable(
        page=page_no,
        index_on_page=index,
        is_last_on_page=index == n_tables - 1,
        bbox=_to_bbox(found.bbox),
        rows=rows,
    )
    header, body = split_header(rows)
    if not header or not body or len(found.rows) != len(rows):
        return raw

    # Columns without data are spacers the grid split off, when the grid has clearly
    # more columns than the header has labels (an empty column with its own label,
    # like a change column with nothing for the first year, is kept)
    labels_per_row = max(sum(1 for cell in row if cell is not None and not is_blank(cell)) for row in header)
    kept = list(range(len(rows[0])))
    if len(rows[0]) > labels_per_row + 1:
        kept = [0] + [j for j in kept[1:] if any(not is_blank(row[j]) for row in body)]
    columns = _column_ranges(found, len(header), kept)
    if columns is None:
        return raw
    left, top, right, _ = found.bbox
    body_top = found.rows[len(header)].bbox[1]
    header_words = [
        w for w in words
        if left <= (w[0] + w[2]) / 2 <= right and top - RULE_TOLERANCE <= (w[1] + w[3]) / 2 < body_top
    ]
    header_rules = [r for r in rules if top - RULE_TOLERANCE <= r[0] <= body_top and r[2] > left and r[1] < right]
    if not header_words:
        return raw
    names, notes = header_from_words(header_words, header_rules, columns)
    raw.rows = [cast(Row, names)] + [[row[j] for j in kept] for row in body]
    raw.header_size = 1
    raw.header_notes = notes
    return raw


def _column_ranges(found: _PdfTable, header_size: int, kept: list[int]) -> list[tuple[float, float]] | None:
    """The x-range of each kept column, from its body cells. A dropped column's space
    is split between its neighbours. None if a column has no cell geometry."""
    extents: list[tuple[float, float]] = []
    for j in kept:
        boxes = [cells[j] for row in found.rows[header_size:] if (cells := row.cells)[j] is not None]
        if not boxes:
            return None
        extents.append((min(b[0] for b in boxes if b), max(b[2] for b in boxes if b)))
    left, _, right, _ = found.bbox
    bounds = [left] + [(extents[k][1] + extents[k + 1][0]) / 2 for k in range(len(extents) - 1)] + [right]
    return [(bounds[k], bounds[k + 1]) for k in range(len(extents))]


def _horizontal_rules(page: _PdfPage) -> list[Segment]:
    """Horizontal lines on the page, drawn as lines or as thin rectangles."""
    rules: list[Segment] = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l" and abs(item[1].y - item[2].y) < 1:
                rules.append((item[1].y, min(item[1].x, item[2].x), max(item[1].x, item[2].x)))
            elif item[0] == "re" and item[1].height < 2 and item[1].width > 2:
                rules.append((item[1].y0, item[1].x0, item[1].x1))
    return rules


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
