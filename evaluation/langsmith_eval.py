"""Score the test questions automatically in LangSmith: a dataset and evaluators.

    uv run python -m evaluation.langsmith_eval             # all questions
    uv run python -m evaluation.langsmith_eval Q05 Q06     # some of them

Needs the sample PDFs ingested first, like evaluation/evaluate.py, and LangSmith
configured in .env. Each run:

1. syncs evaluation/questions.json into the LangSmith dataset DATASET_NAME (each
   question is an example: the question as input, what's expected as reference output);
2. runs the analyst on every example, through langsmith.evaluate(), as an experiment;
3. scores each run with the evaluators below, which LangSmith stores as feedback, so
   experiments (e.g. after a prompt or model change) can be compared side by side.

The evaluators use the same checks as evaluation/evaluate.py.
"""

# langsmith's signatures use bare dict (e.g. metadata: dict | None), which strict mode
# reports as partially unknown on evaluate and on every Client method used here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

import json
import sys
import uuid
from collections.abc import Callable
from typing import Any

from langsmith import Client, evaluate

from agents.analyst.graph import ask
from evaluation.evaluate import (
    QUESTIONS,
    Check,
    check_status,
    check_tables,
    check_text,
    check_values,
    table_ids_by_position,
)
from services.config import get_settings

DATASET_NAME = "pdf-tables-qa"

# Example ids derived from the question id (uuid5: the same name always gives the same
# UUID), so syncing again updates each question's example instead of adding a copy
EXAMPLE_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "pdf-tables-qa/questions")

Evaluator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def example_id(question_id: str) -> str:
    return str(uuid.uuid5(EXAMPLE_NAMESPACE, question_id))


def to_example(item: dict[str, Any]) -> dict[str, Any]:
    """A question from questions.json as a LangSmith example."""
    reference = {k: v for k, v in item.items() if k.startswith("expected_")}
    return {
        "id": example_id(item["id"]),
        "inputs": {"question": item["question"]},
        "outputs": reference,
        "metadata": {"eval_id": item["id"], "categories": item["categories"]},
    }


def sync_dataset(client: Client, questions: list[dict[str, Any]]) -> None:
    """Create the dataset if needed, then add new questions and update changed ones."""
    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(
            DATASET_NAME, description="Test questions over the sample PDFs, from evaluation/questions.json"
        )
    existing = {str(e.id) for e in client.list_examples(dataset_name=DATASET_NAME)}
    examples = [to_example(item) for item in questions]
    new = [e for e in examples if e["id"] not in existing]
    if new:
        client.create_examples(dataset_name=DATASET_NAME, examples=new)
    if len(new) < len(examples):
        client.update_examples(dataset_name=DATASET_NAME, updates=[e for e in examples if e["id"] in existing])


def run_analyst(inputs: dict[str, Any]) -> dict[str, Any]:
    """The target: what the evaluators get from a run, as JSON.

    Inside evaluate() the analyst's own trace becomes a child of this run, so every
    graph node is visible from the experiment.
    """
    try:
        state = ask(inputs["question"])
    except Exception as e:  # noqa: BLE001  (a failing question is a result, scored as wrong)
        return {"error": f"{type(e).__name__}: {e}", "status": None, "answer": "", "selected_ids": [], "result": None}
    last = state["attempts"][-1] if state["attempts"] else None
    return {
        "status": state["status"],
        "answer": state["answer"],
        "selected_ids": state["selected_ids"],
        "result": last.result if last else None,
        "programs": len(state["attempts"]),  # more than 1 means self-correction happened
    }


def feedback(key: str, check: Check) -> dict[str, Any]:
    ok, notes = check
    return {"key": key, "score": ok, "comment": "; ".join(notes) or None}


def evaluators(ids_by_table: dict[tuple[str, str], str]) -> list[Evaluator]:
    """One evaluator per check, plus "correct" (all of them), the per-question verdict.

    LangSmith passes each evaluator the arguments it names: outputs is what run_analyst
    returned, reference_outputs the example's expected values.
    """

    def status(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        return feedback("status", check_status(reference_outputs, outputs["status"]))

    def values(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        return feedback("values", check_values(reference_outputs, outputs["result"]))

    def text(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        return feedback("text", check_text(reference_outputs, outputs["answer"]))

    def tables(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        return feedback("tables", check_tables(reference_outputs, outputs["selected_ids"], ids_by_table))

    def correct(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        checks = [e(outputs, reference_outputs) for e in (status, values, text, tables)]
        failed = [f"{c['key']}: {c['comment']}" for c in checks if not c["score"]]
        if outputs.get("error"):
            failed.insert(0, outputs["error"])
        return {"key": "correct", "score": not failed, "comment": "; ".join(failed) or None}

    return [status, values, text, tables, correct]


def main(only: list[str]) -> None:
    questions: list[dict[str, Any]] = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    client = Client()
    sync_dataset(client, questions)

    examples = list(client.list_examples(dataset_name=DATASET_NAME))
    if only:
        examples = [e for e in examples if (e.metadata or {}).get("eval_id") in only]
    settings = get_settings()
    results = evaluate(
        run_analyst,
        data=examples,
        evaluators=evaluators(table_ids_by_position()),
        experiment_prefix="analyst",
        # Recorded with the experiment, to tell apart runs with different settings
        metadata={
            "llm_model": settings.llm_model,
            "llm_code_model": settings.llm_code_model or settings.llm_model,
            "analyst_max_code_attempts": settings.analyst_max_code_attempts,
            "analyst_top_k": settings.analyst_top_k,
        },
        client=client,
    )
    # Each row holds one example, its run and the evaluators' results
    rows = sorted(results, key=lambda row: str((row["example"].metadata or {}).get("eval_id")))
    correct = 0
    for row in rows:
        scores = {r.key: r for r in row["evaluation_results"].get("results", [])}
        verdict = scores["correct"]
        correct += bool(verdict.score)
        eval_id = (row["example"].metadata or {}).get("eval_id")
        print(f"{eval_id}: {'correct' if verdict.score else 'INCORRECT'} {verdict.comment or ''}")
    print(f"\n{correct}/{len(rows)} correct. Experiment: {results.experiment_name}")


if __name__ == "__main__":
    main(sys.argv[1:])
