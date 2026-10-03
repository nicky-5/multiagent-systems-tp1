import pytest

from services.numbers import infer_decimal_sep, parse_number


# parametrize runs the test once per tuple, and each case is reported separately.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1234", 1234),
        ("1.850.400", 1850400),  # Spanish thousands
        ("11,016,024", 11016024),  # English thousands
        ("18.801,8", 18801.8),  # Spanish decimal
        ("1,234.5", 1234.5),  # English decimal
        ("0,2 %", 0.2),  # percentages keep their value
        ("1.43%", 1.43),
        ("+5.3%", 5.3),
        ("-12,5", -12.5),
        ("−7", -7),  # Unicode minus sign
        ("$ 1.850.400", 1850400),
        ("US$ 2,500.75", 2500.75),
        ("$ (1.147.248)", -1147248),  # parentheses mean negative
        ("(3,5) %", -3.5),
        ("1 234 567", 1234567),  # spaces as thousands separator
        ("2023", 2023),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", [None, "", "  ", "-", "—", "n/a", "s/d"])
def test_blank_cells_are_none(text):
    assert parse_number(text) is None


@pytest.mark.parametrize("text", ["Buenos Aires", "2023 Unidades", "1.2.3", "12a", "(", "%"])
def test_non_numbers_are_none(text):
    assert parse_number(text) is None


def test_ambiguous_number_uses_the_table_separator():
    # "1.234" alone is read as 1234; inside a table that uses "." for decimals it's 1.234
    assert parse_number("1.234") == 1234
    assert parse_number("1.234", decimal_sep=".") == 1.234
    assert parse_number("422,915", decimal_sep=".") == 422915


def test_infer_decimal_sep():
    assert infer_decimal_sep(["Centro", "337.084", "1.303.617,7"]) == ","
    assert infer_decimal_sep(["746,474", "+5.3%", "January"]) == "."
    assert infer_decimal_sep(["422,915", "2023", None]) is None  # nothing decisive
