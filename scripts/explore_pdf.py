"""Print what PyMuPDF finds in a PDF: tables, their headers, and nearby text.

Run with:  uv run python scripts/explore_pdf.py data/sample_pdfs/<file>.pdf
"""

import sys

import pymupdf

path = sys.argv[1]
doc = pymupdf.open(path)

for page_no, page in enumerate(doc.pages(), start=1):  # doc.pages() yields one Page per page
    print(f"\n{'=' * 30} page {page_no} {'=' * 30}")

    # Text blocks: tuples (x0, y0, x1, y1, text, block_no, block_type).
    # Coordinates are in points (1/72 inch), origin at the TOP-left, so y grows downward.
    blocks = page.get_text("blocks")

    finder = page.find_tables()  # default strategy "lines": uses the drawn grid lines
    tables = finder.tables if finder else []
    print(f"{len(tables)} table(s) found")

    for i, tab in enumerate(tables):
        x0, y0, x1, y1 = tab.bbox
        print(f"\n--- table {i}: {tab.row_count} rows x {tab.col_count} cols, "
              f"bbox=({x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f})")
        # header.external=True means the header was found ABOVE the table (not inside it)
        print(f"header.names   = {tab.header.names}")
        print(f"header.external= {tab.header.external}")

        # Context: text blocks that end at most 60pt above the table's top edge.
        # +3pt tolerance: a caption's box includes font descent and can overlap the table.
        above = [b[4].strip().replace("\n", " ") for b in blocks
                 if b[3] <= y0 + 3 and y0 - b[3] < 60 and b[1] < y0]
        print(f"text above     = {above}")

        df = tab.to_pandas()  # first row becomes the column names
        print(df.head(4).to_string())
        print(f"... dtypes: {dict(df.dtypes.astype(str))}")
