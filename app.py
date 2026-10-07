"""Streamlit entry point: setup checks, sidebar, and the three pages.

Run with:  uv run streamlit run app.py
"""

import os

import streamlit as st
from pydantic import ValidationError

from services.config import get_settings
from services.sandbox import (
    SandboxNotReady,
    bubblewrap_available,
    isolation_layers,
    sandbox_environment,
    seccomp_available,
)
from ui import ask_page, ingest_page, tables_page


def check_settings() -> bool:
    """Stop with a clear message if required settings (API keys...) are missing."""
    try:
        get_settings()
    except ValidationError as e:
        missing = ", ".join(str(err["loc"][0]).upper() for err in e.errors())
        st.error(
            f"Missing or invalid settings: **{missing}**. Copy `.env.example` to `.env`, "
            "fill in the values, and restart the app."
        )
        return False
    return True


def render_sidebar() -> None:
    settings = get_settings()
    st.sidebar.subheader("Configuration")
    code_model = settings.llm_code_model or settings.llm_model
    st.sidebar.caption(f"LLM: `{settings.llm_model}` · code: `{code_model}`")
    st.sidebar.caption(f"Pinecone index: `{settings.pinecone_index_name}`")
    tracing = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    st.sidebar.caption(
        f"LangSmith tracing: {'on, project `' + project + '`' if tracing else 'off'}"
    )

    st.sidebar.subheader("Code sandbox")
    try:
        sandbox_environment()
    except SandboxNotReady:
        st.sidebar.error(
            "Not set up: run `uv sync --project sandbox`. Questions can't be answered until then."
        )
        return
    bubblewrap, seccomp = bubblewrap_available(), seccomp_available()
    if not (bubblewrap and seccomp):
        st.sidebar.warning(
            "Reduced isolation: bubblewrap or libseccomp isn't available here."
        )
    st.sidebar.caption(" · ".join(isolation_layers(bubblewrap, seccomp)))


def main() -> None:
    st.set_page_config(
        page_title="PDF Tables QA", page_icon=":material/table_chart:", layout="wide"
    )
    if not check_settings():
        st.stop()
    render_sidebar()
    page = st.navigation(
        [
            st.Page(
                ask_page.render,
                title="Ask",
                icon=":material/chat:",
                url_path="ask",
                default=True,
            ),
            st.Page(
                ingest_page.render,
                title="Upload PDFs",
                icon=":material/upload_file:",
                url_path="upload",
            ),
            st.Page(
                tables_page.render,
                title="Indexed tables",
                icon=":material/table_chart:",
                url_path="tables",
            ),
        ]
    )
    page.run()


main()
