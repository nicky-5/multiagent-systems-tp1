"""Small helpers shared by the pages."""

from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st


def show_table(data: pd.DataFrame | Sequence[Mapping[str, object]]) -> None:
    """A scrollable table without the row index."""
    # One of the types in st.dataframe's signature comes from an optional dependency
    # that isn't installed, so strict mode reports the method as partially unknown.
    st.dataframe(data, hide_index=True)  # pyright: ignore[reportUnknownMemberType]
