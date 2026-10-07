"""Upload PDFs and follow their ingestion as it happens."""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from agents.ingestion.graph import ingest
from services.config import ROOT, get_settings
from ui.widgets import show_table

SAMPLE_PDFS = ROOT / "data" / "sample_pdfs"


@dataclass
class IngestionLog:
    """What happened to one PDF, kept in the session so it stays on screen."""

    filename: str
    tables: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    warnings: list[str] = field(default_factory=list[str])
    error: str | None = None


def render() -> None:
    st.title("Upload PDFs")
    st.caption(
        "Tables are extracted, described by the LLM, validated, and indexed in Pinecone. "
        "Uploading the same PDF again updates its tables instead of duplicating them."
    )
    uploads = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    left, right = st.columns([1, 3])
    paths: list[Path] = []
    if left.button("Ingest", type="primary", disabled=not uploads):
        paths = [save_upload(f.name, f.getvalue()) for f in uploads or []]
    samples = sorted(SAMPLE_PDFS.glob("*.pdf"))
    if right.button(f"Ingest the {len(samples)} sample PDFs", disabled=not samples):
        paths = samples

    logs: list[IngestionLog] = st.session_state.setdefault("ingestion_logs", [])
    for path in paths:
        logs.insert(0, ingest_with_progress(path))

    for log in logs:
        render_log(log)


def ingest_with_progress(path: Path) -> IngestionLog:
    log = IngestionLog(filename=path.name)
    # st.status is a box whose label and icon change as the work progresses; anything
    # written inside the `with` block (here, by the progress callback) appears in it.
    with st.status(f"Ingesting {path.name}...", expanded=True) as status:
        try:
            outcome = ingest(
                str(path), lambda source, message: st.write(f"**{source}**: {message}")
            )
        except Exception as e:  # noqa: BLE001  (shown to the user instead of crashing the app)
            log.error = f"{type(e).__name__}: {e}"
            status.update(label=f"{path.name}: failed", state="error", expanded=True)
            return log
        log.warnings = outcome.warnings
        log.tables = [
            {
                "Table": r.name,
                "Pages": ", ".join(map(str, r.source.pages)),
                "Rows": r.n_rows,
                "table_id": r.table_id,
            }
            for r in outcome.records
        ]
        status.update(
            label=f"{path.name}: {len(outcome.records)} table(s) indexed",
            state="complete",
            expanded=False,
        )
    return log


def render_log(log: IngestionLog) -> None:
    st.subheader(log.filename)
    if log.error:
        st.error(f"Ingestion failed: {log.error}")
        return
    for warning in log.warnings:  # tables or pages that couldn't be read
        st.warning(warning)
    if log.tables:
        show_table(log.tables)
    else:
        st.info("No tables were indexed from this file.")


def save_upload(name: str, content: bytes) -> Path:
    """Store an upload under data/uploads/<content hash>/<file name>.

    The folder per content keeps the original file name, which ingestion records as the
    table's source, while two different files with the same name can't overwrite each other.
    """
    safe_name = re.sub(r"[^\w.\- ]", "_", Path(name).name) or "upload.pdf"
    folder = (
        get_settings().data_dir / "uploads" / hashlib.sha256(content).hexdigest()[:16]
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / safe_name
    path.write_bytes(content)
    return path
