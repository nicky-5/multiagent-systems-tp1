"""Print what PyMuPDF finds in a PDF: tables, their headers, and nearby text.

Run with:  uv run python scripts/explore_pdf.py data/sample_pdfs/<file>.pdf
"""

import sys
from typing import Any

import pymupdf

CAPTION_OVERLAP = 3  # a caption's box includes font descent and can overlap the table
MAX_CAPTION_DISTANCE = 60

path = sys.argv[1]
# Most of PyMuPDF's API is unannotated; this script explores it dynamically.
doc: Any = pymupdf.open(path)

for page_no, page in enumerate(doc.pages(), start=1):
    print(f"\n{'=' * 30} page {page_no} {'=' * 30}")

    # Text blocks: tuples (x0, y0, x1, y1, text, block_no, block_type).
    # Coordinates are in points (1/72 inch), origin at the TOP-left, so y grows downward.
    blocks = page.get_text("blocks")

    finder = page.find_tables()  # default strategy "lines": uses the drawn grid lines
    tables: list[Any] = finder.tables if finder else []
    print(f"{len(tables)} table(s) found")

    for table_index, table in enumerate(tables):
        x0, y0, x1, y1 = table.bbox
        print(f"\n--- table {table_index}: {table.row_count} rows x {table.col_count} cols, "
              f"bbox=({x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f})")
        # header.external=True means the header was found ABOVE the table (not inside it)
        print(f"header.names   = {table.header.names}")
        print(f"header.external= {table.header.external}")

        above = [
            text.strip().replace("\n", " ")
            for _, block_top, _, block_bottom, text, *_ in blocks
            if block_top < y0
            and block_bottom <= y0 + CAPTION_OVERLAP
            and y0 - block_bottom < MAX_CAPTION_DISTANCE
        ]
        print(f"text above     = {above}")

        df = table.to_pandas()  # first row becomes the column names
        print(df.head(4).to_string())
        print(f"... dtypes: {dict(df.dtypes.astype(str))}")
