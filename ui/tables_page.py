"""Browse the indexed tables: metadata, source and data."""

from itertools import groupby

import streamlit as st

from agents.ingestion.schemas import TableRecord
from services.table_store import table_store
from ui.widgets import page_header, show_table


def render() -> None:
    page_header("Indexed tables")
    store = table_store()
    records = store.list_records()
    if not records:
        st.info("No tables yet. Upload a PDF on the **Upload PDFs** page.")
        return

    query = st.text_input(
        "Filter", placeholder="Words in the name, description, columns or file name"
    )
    shown = [r for r in records if matches(r, query)]
    st.caption(f"{len(shown)} of {len(records)} table(s)")

    by_file = sorted(shown, key=lambda r: (r.source.filename, r.source.pages))
    for filename, group in groupby(by_file, key=lambda r: r.source.filename):
        st.subheader(filename)
        for record in group:
            pages = ", ".join(map(str, record.source.pages))
            with st.expander(f"{record.name} · p. {pages} · {record.n_rows} rows"):
                st.write(record.description)
                st.caption(f"table_id `{record.table_id}`")
                st.markdown("**Columns**")
                show_table(
                    [
                        {"Column": c.name, "Unit": c.unit, "Description": c.description}
                        for c in record.columns
                    ]
                )
                if record.row_header_column:
                    st.caption(
                        f"Rows are labelled by the column **{record.row_header_column}**."
                    )
                st.markdown("**Data**")
                show_table(store.load_df(record.table_id))


def matches(record: TableRecord, query: str) -> bool:
    """True if every word of the query appears somewhere in the table's metadata."""
    text = " ".join(
        [
            record.name,
            record.description,
            record.source.filename,
            *(c.name for c in record.columns),
            *record.row_headers,
        ]
    ).lower()
    return all(word in text for word in query.lower().split())
