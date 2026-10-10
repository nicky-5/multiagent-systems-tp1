import pytest

from evaluation.evaluate import find_number, numbers_in


def test_numbers_in_nested_results():
    assert numbers_in({"parts": [{"value": {"Salta": 1.5, "n": [2, True, "x"]}}]}) == [1.5, 2.0]


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
