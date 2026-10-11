"""Small helpers shared by the pages."""

from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# A container created with key="page-header" gets the CSS class st-key-page-header,
# and Streamlit wraps it in a div of its own (stLayoutWrapper). position: sticky keeps
# that wrapper in the flow but pins it to the top of the page's scrolling area
# (stMain, or stAppScrollToBottomContainer on pages with a chat input) while the page
# scrolls, below Streamlit's own toolbar (3.75rem high). It goes on the
# wrapper because a sticky element only moves within its parent, and the wrapper's
# parent is the whole page.
# The header needs a solid background, or the content would show through it. Only
# stApp has the theme's background color; the elements between it and the header are
# transparent, so they (every element inside stApp that contains the header, matched
# with :has) and the header inherit it, which follows light and dark themes.
HEADER_CSS = """
<style>
[data-testid="stApp"] :has(.st-key-page-header),
.st-key-page-header {
    background-color: inherit;
}
[data-testid="stLayoutWrapper"]:has(> .st-key-page-header) {
    position: sticky;
    top: 3.75rem;
    z-index: 100;
}
.st-key-page-header {
    padding-bottom: .5rem;
    border-bottom: 1px solid rgba(128, 128, 128, .25);
}
</style>
"""


def page_header(title: str, caption: str | None = None) -> DeltaGenerator:
    """The page's title (and caption), kept at the top of the page while scrolling.

    Returns the header's container, so a page can add controls that should stay in
    view too (header.toggle(...)).
    """
    st.html(HEADER_CSS)
    header = st.container(key="page-header")
    header.title(title)
    if caption:
        header.caption(caption)
    return header


def show_table(data: pd.DataFrame | Sequence[Mapping[str, object]]) -> None:
    """A scrollable table without the row index."""
    # One of the types in st.dataframe's signature comes from an optional dependency
    # that isn't installed, so strict mode reports the method as partially unknown.
    st.dataframe(data, hide_index=True)  # pyright: ignore[reportUnknownMemberType]
