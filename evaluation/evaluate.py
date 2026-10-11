"""Run the test questions through the analyst and check its answers.

    uv run python -m evaluation.evaluate             # all questions
    uv run python -m evaluation.evaluate Q05 Q06     # some of them

Needs the sample PDFs ingested first. Writes evaluation/results.json and prints a
Markdown table for EVALUATION.md. Each run is a LangSmith trace tagged with the
question's id (metadata "eval_id").
"""

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from agents.analyst.graph import ask
from agents.analyst.state import AnalystState
from services.table_store import table_store

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.json"
RESULTS = HERE / "results.json"

# A computed number matches an expected one within 0.5%, also in a different but
# equivalent unit: thousands vs units, or a fraction vs a percentage.
TOLERANCE = 0.005
SCALES: dict[float, str | None] = {  # scale -> note on the unit change (None: same unit)
    1.0: None,
    1000.0: "same value x1000 (a 1000-times smaller unit, e.g. ARS instead of thousands)",
    0.001: "same value /1000 (a 1000-times larger unit, e.g. thousands instead of ARS)",
    0.01: "same value as a fraction instead of a percentage",
}


@dataclass
class Outcome:
    id: str
    question: str
    expected_status: str
    status: str | None = None
    correct: bool = False
    attempts: int = 0  # programs written; more than 1 means self-correction happened
    notes: list[str] = field(default_factory=list[str])
    answer: str = ""
    run_id: str = ""
    error: str | None = None


def numbers_in(value: object) -> list[float]:
    """Every number in a JSON value (results may be numbers, lists or objects)."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        return [n for v in cast(dict[str, object], value).values() for n in numbers_in(v)]
    if isinstance(value, list):
        return [n for v in cast(list[object], value) for n in numbers_in(v)]
    return []


def find_number(expected: float, computed: list[float]) -> tuple[bool, str | None]:
    """Whether expected is among the computed numbers, and a note if in another unit."""
    for scale, note in SCALES.items():
        target = expected * scale
        if any(math.isclose(n, target, rel_tol=TOLERANCE) for n in computed):
            return True, note
    return False, None


# Each check compares one aspect of a run with the question's expectations and returns
# whether it passed, plus notes explaining failures (or an accepted unit change). The
# LangSmith evaluators in evaluation/langsmith_eval.py use the same functions.
Check = tuple[bool, list[str]]


def check_status(item: dict[str, Any], status: str | None) -> Check:
    if status == item["expected_status"]:
        return True, []
    return False, [f"status {status}, expected {item['expected_status']}"]


def check_values(item: dict[str, Any], result: object) -> Check:
    """Every expected number is among those the last program computed."""
    computed = numbers_in(result)
    ok, notes = True, list[str]()
    # Each expected value is a number, or a list of acceptable alternatives
    for expected in item["expected_values"]:
        options = cast(list[float], expected) if isinstance(expected, list) else [cast(float, expected)]
        alternatives = [float(v) for v in options]
        matches = [(v, find_number(v, computed)) for v in alternatives]
        hit = next(((v, note) for v, (found, note) in matches if found), None)
        if hit is None:
            ok = False
            notes.append(f"value {expected} not in the result {computed}")
        elif hit[1]:
            notes.append(f"{hit[0]}: {hit[1]}")
    return ok, notes


def check_text(item: dict[str, Any], answer: str) -> Check:
    missing = [t for t in item["expected_text"] if t.casefold() not in answer.casefold()]
    return not missing, [f"{t!r} not in the answer" for t in missing]


def check_tables(
    item: dict[str, Any], selected_ids: list[str], ids_by_table: dict[tuple[str, str], str]
) -> Check:
    missing = [
        t for t in item["expected_tables"] if ids_by_table.get((t["file"], t["table"])) not in selected_ids
    ]
    return not missing, [f"table {t['file']} {t['table']} not selected" for t in missing]


def table_ids_by_position() -> dict[tuple[str, str], str]:
    """Stored table ids by (file, position such as "p1-t2").

    Questions name tables by file and position, because table ids include a hash of the
    PDF, which changes if the sample PDFs are regenerated.
    """
    return {(r.source.filename, r.table_id.split("-", 1)[1]): r.table_id for r in table_store().list_records()}


def check(item: dict[str, Any], state: AnalystState, ids_by_table: dict[tuple[str, str], str]) -> Outcome:
    outcome = Outcome(
        id=item["id"], question=item["question"], expected_status=item["expected_status"],
        status=state["status"], attempts=len(state["attempts"]), answer=state["answer"],
    )
    last = state["attempts"][-1] if state["attempts"] else None
    checks = [
        check_status(item, state["status"]),
        check_values(item, last.result if last else None),
        check_text(item, state["answer"]),
        check_tables(item, state["selected_ids"], ids_by_table),
    ]
    for _, notes in checks:
        outcome.notes += notes
    if outcome.attempts > 1:
        outcome.notes.append(f"{outcome.attempts - 1} code correction(s)")
    outcome.correct = all(ok for ok, _ in checks)
    return outcome


def main(only: list[str]) -> None:
    questions: list[dict[str, Any]] = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if only:
        questions = [q for q in questions if q["id"] in only]
    ids_by_table = table_ids_by_position()

    outcomes: list[Outcome] = []
    for item in questions:
        print(f"{item['id']}: {item['question']}", flush=True)
        run_id = uuid4()
        try:
            state = ask(item["question"], run_id=run_id, metadata={"eval_id": item["id"]})
            outcome = check(item, state, ids_by_table)
        except Exception as e:  # noqa: BLE001  (a failing question is a result, not a crash)
            outcome = Outcome(item["id"], item["question"], item["expected_status"], error=f"{type(e).__name__}: {e}")
        outcome.run_id = str(run_id)
        print(f"  -> {outcome.status} ({'correct' if outcome.correct else 'INCORRECT'}) {'; '.join(outcome.notes)}")
        outcomes.append(outcome)

    RESULTS.write_text(json.dumps([asdict(o) for o in outcomes], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    correct = sum(o.correct for o in outcomes)
    print(f"\n{correct}/{len(outcomes)} correct. Results in {RESULTS.relative_to(HERE.parent)}\n")
    print("| ID | Expected | Got | Correct | Programs | Notes |")
    print("|----|----------|-----|:-------:|:--------:|-------|")
    for o in outcomes:
        notes = "; ".join(o.notes) or (o.error or "")
        print(f"| {o.id} | `{o.expected_status}` | `{o.status}` | {'✅' if o.correct else '❌'} | {o.attempts} | {notes} |")


if __name__ == "__main__":
    main(sys.argv[1:])
