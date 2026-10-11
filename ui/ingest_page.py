"""Upload PDFs and follow their ingestion as it happens."""

import hashlib
import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import streamlit as st
from streamlit.elements.lib.mutable_status_container import StatusContainer

from agents.ingestion.graph import ingest
from agents.ingestion.state import Step
from services.config import ROOT, get_settings
from ui.widgets import page_header, show_table

SAMPLE_PDFS = ROOT / "data" / "sample_pdfs"


@dataclass
class IngestionLog:
    """What happened to one PDF, kept in the session so it stays on screen."""

    filename: str
    tables: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    warnings: list[str] = field(default_factory=list[str])
    error: str | None = None


def render() -> None:
    page_header(
        "Upload PDFs",
        "Tables are extracted, described by the LLM, validated, and indexed in Pinecone. "
        "Uploading the same PDF again updates its tables instead of duplicating them.",
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


# A table card's state: the ingestion step it last reported, or one only the UI knows
# about ("indexed" once the index node has run, "failed" if the whole run crashed)
CardStep = Step | Literal["indexed", "failed"]

# step -> (text shown on the card, color, still in progress)
CARD_STEPS: dict[CardStep, tuple[str, str, bool]] = {
    "pending": ("Pending", "#9ca3af", True),
    "enrich": ("Generating metadata", "#3b82f6", True),
    "invalid": ("Validation failed", "#f59e0b", True),
    "valid": ("Metadata is valid", "#8b5cf6", True),
    "done": ("Waiting to be indexed", "#14b8a6", True),
    "indexed": ("Indexed", "#22c55e", False),
    "skipped": ("Skipped: metadata never validated", "#ef4444", False),
    "failed": ("Ingestion failed", "#ef4444", False),
}

CARD_CSS = (
    "<style>"
    ".ing-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));"
    "gap:.5rem;margin:.25rem 0 .5rem}"
    ".ing-card{border:1px solid rgba(128,128,128,.25);border-left:3px solid var(--c);"
    "border-radius:.5rem;padding:.5rem .75rem;background:rgba(128,128,128,.05)}"
    ".ing-label{font-weight:600;font-size:.9rem}"
    ".ing-step{display:flex;align-items:center;gap:.5rem;margin-top:.25rem;"
    "color:var(--c);font-size:.875rem;font-weight:500}"
    ".ing-detail{opacity:.65;font-size:.78rem;margin-top:.15rem;overflow:hidden;"
    "display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}"
    ".ing-spin{width:.8rem;height:.8rem;flex:none;border-radius:50%;"
    "border:2px solid rgba(128,128,128,.25);border-top-color:var(--c);"
    "animation:ing-spin .8s linear infinite}"
    ".ing-icon{width:.8rem;text-align:center;flex:none}"
    "@keyframes ing-spin{to{transform:rotate(360deg)}}"
    "</style>"
)


@dataclass
class TableCard:
    label: str  # e.g. "table 2/3 (p. [1])"
    step: CardStep
    message: str  # the event's own message, shown in small print


class IngestionProgress:
    """Live view of one PDF's ingestion: a status card per table, redrawn on each event.

    It's the progress callback passed to ingest(). The graph streams in this thread,
    so the callback can update Streamlit elements directly.
    """

    def __init__(self, filename: str, status: StatusContainer) -> None:
        self.filename = filename
        self.status = status
        self.cards: dict[str, TableCard] = {}  # by label, in order of first event
        # st.empty() holds one element that each draw() replaces
        self.view = status.empty()

    def __call__(self, source: str, step: Step, message: str) -> None:
        if step == "extract":
            # update() without expanded= clears the box's open state, so it closes
            self.status.update(
                label=f"Ingesting {self.filename}: {message}...", expanded=True
            )
        elif step == "index":
            for card in self.cards.values():
                if card.step == "done":
                    card.step = "indexed"
        else:
            self.cards[source] = TableCard(source, step, message)
        self.draw()

    def fail(self) -> None:
        """Mark the tables that were still in progress when the run crashed."""
        for card in self.cards.values():
            if CARD_STEPS[card.step][2]:
                card.step = "failed"
        self.draw()

    def draw(self) -> None:
        if self.cards:
            cards = "".join(card_html(card) for card in self.cards.values())
            self.view.markdown(
                f'{CARD_CSS}<div class="ing-grid">{cards}</div>', unsafe_allow_html=True
            )


def card_html(card: TableCard) -> str:
    text, color, working = CARD_STEPS[card.step]
    if card.step == "pending":  # not started yet: no spinner
        icon = '<span class="ing-icon">○</span>'
    elif working:
        icon = '<span class="ing-spin"></span>'
    else:
        icon = f'<span class="ing-icon">{"✓" if card.step == "indexed" else "✕"}</span>'
    # Labels and messages contain LLM output and PDF text, so they're escaped. The HTML
    # is kept on one line: Markdown would treat indented lines as a code block.
    return (
        f'<div class="ing-card" style="--c:{color}">'
        f'<div class="ing-label">{html.escape(card.label)}</div>'
        f'<div class="ing-step">{icon}{text}</div>'
        f'<div class="ing-detail" title="{html.escape(card.message)}">'
        f"{html.escape(card.message)}</div>"
        "</div>"
    )


def ingest_with_progress(path: Path) -> IngestionLog:
    log = IngestionLog(filename=path.name)
    # st.status is a box whose label and icon change as the work progresses; the
    # progress view draws its table cards inside it.
    with st.status(f"Ingesting {path.name}...", expanded=True) as status:
        progress = IngestionProgress(path.name, status)
        try:
            outcome = ingest(str(path), progress)
        except Exception as e:  # noqa: BLE001  (shown to the user instead of crashing the app)
            log.error = f"{type(e).__name__}: {e}"
            progress.fail()
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
