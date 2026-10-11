"""Nodes of the analyst graph. Each takes the state and returns a partial update."""

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agents.analyst import prompts, rules
from agents.analyst.schemas import (
    CodeAttempt,
    GeneratedCode,
    PartAssessment,
    QueryPlan,
    Selection,
)
from agents.analyst.state import AnalystState
from services.config import get_settings
from services.llm import chat_model, code_model, embeddings, with_schema
from services.sandbox import available_packages, run_python
from services.table_store import table_store
from services.vector_store import vector_store


def understand(state: AnalystState) -> dict[str, Any]:
    """Split the question into parts and phrase the searches for the data it needs."""
    messages: list[BaseMessage] = [
        SystemMessage(prompts.UNDERSTAND_SYSTEM),
        HumanMessage(prompts.understand_user_prompt(state["question"])),
    ]
    plan = with_schema(chat_model(), QueryPlan).invoke(messages)
    return {"plan": plan}


def retrieve(state: AnalystState) -> dict[str, Any]:
    """Search Pinecone with every query in both languages and merge the results, then
    add tables with a row label the question names (no LLM)."""
    plan = state["plan"]
    assert plan is not None  # understand always runs first
    settings = get_settings()

    queries = [q for s in plan.searches for q in (s.spanish, s.english)] or [state["question"]]
    vectors = embeddings().embed_documents(queries)  # one request for all the queries
    store = vector_store()
    results = [store.query(v, top_k=settings.analyst_top_k) for v in vectors]
    matches = rules.merge_matches(results, limit=settings.analyst_max_candidates)

    # A Pinecone record whose data file is gone (e.g. data/tables was deleted) can't be
    # used for calculations, so it isn't offered to the LLM.
    tables = table_store()
    candidates = [rules.candidate_from_match(m) for m in matches if tables.exists(m.id)]
    stale = [m.id for m in matches if not tables.exists(m.id)]

    found = {c.table_id for c in candidates}
    entities = [entity for part in plan.parts for entity in part.entities]
    records = tables.list_records()
    labelled = [i for i in rules.tables_labelling(entities, records) if i not in found]
    by_id = {r.table_id: r for r in records}
    candidates += [rules.candidate_from_record(by_id[i]) for i in labelled[: rules.MAX_LABEL_CANDIDATES]]
    warnings = [f"Indexed tables without stored data were ignored: {stale}"] if stale else []
    return {"candidates": candidates, "warnings": warnings}


def select(state: AnalystState) -> dict[str, Any]:
    """Decide which tables each part needs and whether it's answerable, then derive
    the status from those decisions."""
    plan = state["plan"]
    assert plan is not None
    candidates = state["candidates"]

    if candidates:
        messages: list[BaseMessage] = [
            SystemMessage(prompts.SELECT_SYSTEM),
            HumanMessage(prompts.select_user_prompt(state["question"], plan, candidates)),
        ]
        selection = with_schema(chat_model(), Selection).invoke(messages)
        parts = rules.check_assessments(
            selection.parts, [p.question for p in plan.parts], {c.table_id for c in candidates}
        )
    else:  # nothing indexed matched: no need to ask the LLM
        parts = [
            PartAssessment(
                question=p.question, table_ids=[], answerable=False,
                missing="No indexed table is related to this question.",
            )
            for p in plan.parts
        ]

    return {
        "assessments": parts,
        "selected_ids": rules.selected_table_ids(parts),
        "status": rules.decide_status(parts),
    }


def generate_code(state: AnalystState) -> dict[str, Any]:
    """Write a program that computes the answerable parts; on a retry, fix the last one."""
    plan = state["plan"]
    assert plan is not None
    # The parts to compute: those select judged answerable, with their planned details
    answerable = [p.answerable for p in state["assessments"]]
    parts = [part for part, ok in zip(plan.parts, answerable, strict=False) if ok]

    # The real column names and dtypes come from the data itself, not from metadata
    store = table_store()
    by_id = {c.table_id: c for c in state["candidates"]}
    tables = [(by_id[i], store.load_df(i)) for i in state["selected_ids"]]

    messages: list[BaseMessage] = [
        SystemMessage(prompts.generate_system(available_packages())),
        HumanMessage(prompts.generate_user_prompt(state["question"], parts, tables)),
    ]
    attempts = state["attempts"]
    if attempts and attempts[-1].error:  # a retry: show the failed program and its error
        messages.append(HumanMessage(prompts.code_retry_prompt(attempts[-1].code, attempts[-1].error)))

    generated = with_schema(code_model(), GeneratedCode).invoke(messages)  # LLM_CODE_MODEL
    attempt = CodeAttempt(approach=generated.approach, code=generated.code)
    return {"attempts": [*attempts, attempt]}


def execute(state: AnalystState) -> dict[str, Any]:
    """Run the latest program in the sandbox and check its output (no LLM).

    A valid result may still report data as missing; those parts become unanswerable
    and the status is decided again. After the last failed attempt, nothing could be
    computed.
    """
    attempts = state["attempts"]
    store = table_store()
    files = {f"{i}.parquet": store.data_path(i) for i in state["selected_ids"]}
    run = run_python(attempts[-1].code, files)

    error = run.error
    if error is None:
        assert run.result is not None
        n_parts = sum(p.answerable for p in state["assessments"])
        problems = rules.check_result(run.result, n_parts)
        if problems:
            error = "The program ran, but its output is invalid:\n- " + "\n- ".join(problems)

    attempt = attempts[-1].model_copy(update={
        "stdout": run.stdout,
        "stderr": run.stderr,
        "error": error,
        "result": run.result if error is None else None,
        "isolation": run.isolation,
    })
    update: dict[str, Any] = {"attempts": [*attempts[:-1], attempt]}

    if error is None:
        assert run.result is not None
        parts = rules.apply_result(state["assessments"], run.result)
        update |= {"assessments": parts, "status": rules.decide_status(parts)}
    elif len(attempts) >= state["max_attempts"]:
        parts = rules.mark_failed(state["assessments"], len(attempts))
        update |= {"assessments": parts, "status": rules.decide_status(parts)}
    return update


def respond(state: AnalystState) -> dict[str, Any]:
    """Write the answer from the computed results and the assessments, then cite the
    tables (the citation is added by code, so it's exact)."""
    plan = state["plan"]
    assert plan is not None
    parts = state["assessments"]

    # What was computed: the last program's valid result, if any part is still answerable
    approach: str | None = None
    computed: list[dict[str, Any]] = []
    last = state["attempts"][-1] if state["attempts"] else None
    if last is not None and last.result is not None and any(p.answerable for p in parts):
        approach = last.approach
        # Entries of the parts still answerable (the others reported data as missing)
        computed = [r for r in last.result["parts"] if r["missing"] is None]

    messages: list[BaseMessage] = [
        SystemMessage(prompts.RESPOND_SYSTEM),
        HumanMessage(prompts.respond_user_prompt(state["question"], plan.language, parts, approach, computed)),
    ]
    text = chat_model().invoke(messages).text.strip()

    # Cite the tables the result came from; when nothing was computed, the related
    # tables the explanation refers to
    by_id = {c.table_id: c for c in state["candidates"]}
    used = bool(computed)
    cited = state["selected_ids"] if used else list(dict.fromkeys(i for p in parts for i in p.table_ids))
    footer = rules.sources_footer([by_id[i] for i in cited if i in by_id], plan.language, used=used)
    return {"answer": f"{text}\n\n{footer}".strip()}
