from typing import Any

import pytest

from evaluation.evaluate import find_number, numbers_in
from evaluation.langsmith_eval import evaluators, example_id, to_example


def test_numbers_in_nested_results():
    # Annotated: Pylance infers a partially unknown type for this mixed literal
    result: dict[str, Any] = {"parts": [{"value": {"Salta": 1.5, "n": [2, True, "x"]}}]}
    assert numbers_in(result) == [1.5, 2.0]


@pytest.mark.parametrize(
    ("expected", "computed", "found", "note"),
    [
        (162.02, [162.0236], True, None),  # within 0.5%
        (162.02, [0.16202], True, "same value /1000 (a 1000-times larger unit, e.g. thousands instead of ARS)"),
        (8.65, [0.0865], True, "same value as a fraction instead of a percentage"),
        (597850, [597850000.0], True, "same value x1000 (a 1000-times smaller unit, e.g. ARS instead of thousands)"),
        (165.43, [163.86], False, None),  # Salta's value, not Formosa's
    ],
)
def test_find_number(expected: float, computed: list[float], found: bool, note: str | None):
    assert find_number(expected, computed) == (found, note)


ITEM: dict[str, Any] = {
    "id": "Q05",
    "question": "¿Ventas por habitante en Córdoba?",
    "categories": ["multi-table"],
    "expected_status": "answered",
    "expected_answer": "162,02 ARS por habitante.",
    "expected_values": [162.02],
    "expected_text": ["Córdoba"],
    "expected_tables": [{"file": "a.pdf", "table": "p1-t0"}],
}
IDS_BY_TABLE = {("a.pdf", "p1-t0"): "abc-p1-t0"}


def test_examples_have_stable_ids_and_keep_only_expectations_as_reference():
    example = to_example(ITEM)
    assert example["id"] == example_id("Q05") == to_example(dict(ITEM))["id"]
    assert example["id"] != example_id("Q06")
    assert example["inputs"] == {"question": ITEM["question"]}
    assert set(example["outputs"]) == {
        "expected_status", "expected_answer", "expected_values", "expected_text", "expected_tables"
    }
    assert example["metadata"] == {"eval_id": "Q05", "categories": ["multi-table"]}


def score(outputs: dict[str, Any]) -> dict[str, tuple[bool, str | None]]:
    reference = to_example(ITEM)["outputs"]
    return {(f := e(outputs, reference))["key"]: (f["score"], f["comment"]) for e in evaluators(IDS_BY_TABLE)}


def test_evaluators_pass_a_correct_run():
    outputs: dict[str, Any] = {
        "status": "answered", "answer": "En Córdoba fueron 162,02 ARS.", "selected_ids": ["abc-p1-t0"],
        "result": {"parts": [{"value": 0.16202, "unit": "miles de ARS", "missing": None}]}, "programs": 1,
    }
    scores = score(outputs)
    assert scores["correct"] == (True, None)
    assert scores["values"][1] is not None and "/1000" in scores["values"][1]  # accepted unit change is noted


def test_evaluators_fail_each_check_and_a_crashed_run():
    scores = score({"status": "partial", "answer": "", "selected_ids": [], "result": None})
    assert {k: ok for k, (ok, _) in scores.items()} == {
        "status": False, "values": False, "text": False, "tables": False, "correct": False
    }
    crashed = score({"error": "ConnectionError: x", "status": None, "answer": "", "selected_ids": [], "result": None})
    assert crashed["correct"][1] is not None and crashed["correct"][1].startswith("ConnectionError: x")
