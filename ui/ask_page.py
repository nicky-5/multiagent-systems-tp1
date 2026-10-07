"""Ask questions about the indexed tables, with a chat history for the session."""

from dataclasses import dataclass
from typing import Literal

import streamlit as st

from agents.analyst.graph import ask
from agents.analyst.state import AnalystState
from services.table_store import table_store
from ui.widgets import show_table

BADGES: dict[str, tuple[str, Literal["green", "orange", "red"]]] = {
    "answered": ("Answered", "green"),
    "partial": ("Partially answered", "orange"),
    "cannot_answer": ("Cannot answer", "red"),
}


@dataclass
class Exchange:
    """One question and what came of it."""

    question: str
    state: AnalystState | None = None  # the analyst's final state
    error: str | None = None  # if the run itself failed (LLM, Pinecone...)


def render() -> None:
    st.title("Ask")
    st.caption(
        "Questions are answered only from the indexed tables, by code run in a sandbox."
    )

    history: list[Exchange] = st.session_state.setdefault("chat", [])
    for exchange in history:
        render_exchange(exchange)

    question = st.chat_input("Ask about the indexed tables (in English or Spanish)")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            exchange = run(question)
            render_result(exchange)
        history.append(exchange)


def run(question: str) -> Exchange:
    with st.status("Working on it...", expanded=True) as status:
        try:
            state = ask(
                question, lambda node, message: st.write(f"**{node}**: {message}")
            )
        except Exception as e:  # noqa: BLE001  (shown to the user instead of crashing the app)
            status.update(label="Failed", state="error", expanded=True)
            return Exchange(question, error=f"{type(e).__name__}: {e}")
        retries = len(state["attempts"]) - 1
        label = "Done" + (
            f" ({retries} code correction{'s' if retries > 1 else ''})"
            if retries > 0
            else ""
        )
        status.update(label=label, state="complete", expanded=False)
    return Exchange(question, state=state)


def render_exchange(exchange: Exchange) -> None:
    with st.chat_message("user"):
        st.markdown(exchange.question)
    with st.chat_message("assistant"):
        render_result(exchange)


def render_result(exchange: Exchange) -> None:
    if exchange.error or exchange.state is None:
        st.error(f"Something went wrong while answering: {exchange.error}")
        return
    state = exchange.state

    label, color = BADGES[state["status"] or "cannot_answer"]
    st.badge(label, color=color)
    st.markdown(state["answer"])

    missing = [p for p in state["assessments"] if not p.answerable]
    if missing:
        st.markdown("**Missing data**")
        st.markdown("\n".join(f"- *{p.question}*: {p.missing}" for p in missing))
    for warning in state["warnings"]:
        st.warning(warning)

    render_details(state)


def render_details(state: AnalystState) -> None:
    """Everything behind the answer: understanding, tables, code, output."""
    by_id = {c.table_id: c for c in state["candidates"]}

    plan = state["plan"]
    with st.expander("How the question was understood"):
        if plan:
            for i, (part, assessment) in enumerate(
                zip(plan.parts, state["assessments"], strict=False), start=1
            ):
                verdict = "answerable" if assessment.answerable else "not answerable"
                st.markdown(f"{i}. {part.question} ({part.operation}; {verdict})")
            st.markdown(
                "**Searches:** "
                + "; ".join(f"{s.spanish} / {s.english}" for s in plan.searches)
            )
        st.markdown("**Candidate tables** (similarity):")
        st.markdown(
            "\n".join(f"- {c.score:.2f} {c.name}" for c in state["candidates"])
            or "none"
        )

    if state["selected_ids"]:
        with st.expander(f"Selected tables ({len(state['selected_ids'])})"):
            store = table_store()
            for table_id in state["selected_ids"]:
                c = by_id[table_id]
                st.markdown(
                    f"**{c.name}** · {c.filename}, p. {', '.join(map(str, c.pages))}"
                )
                show_table(store.load_df(table_id))

    attempts = state["attempts"]
    if attempts:
        retries = len(attempts) - 1
        title = "Generated code" + (
            f" ({retries} correction{'s' if retries > 1 else ''})" if retries else ""
        )
        with st.expander(title):
            for i, attempt in enumerate(attempts, start=1):
                if len(attempts) > 1:
                    st.markdown(f"**Program {i}**")
                st.caption(attempt.approach)
                st.code(attempt.code, language="python")
                if attempt.stdout.strip():
                    st.markdown("Output:")
                    st.code(attempt.stdout.strip(), language="text")
                if attempt.error:
                    st.error(attempt.error[-1500:])
            st.caption("Sandbox: " + " · ".join(attempts[-1].isolation))
