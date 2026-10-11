"""Parse numbers that PDFs store as formatted text ("$ (1.234,5)", "12,5 %", "1,234.5")."""

import re
from collections.abc import Iterable

_CURRENCY = re.compile(r"US\$|AR\$|USD|ARS|EUR|[$€£]", re.IGNORECASE)
_BLANKS = {"", "-", "–", "—", "n/a", "n/d", "s/d", "...", "…"}
_THOUSANDS_GROUP_SIZE = 3


def is_blank(text: str | None) -> bool:
    """True for empty cells and the usual "no data" placeholders."""
    return text is None or text.strip().lower() in _BLANKS


def _strip_decorations(text: str) -> tuple[str, bool]:
    """Remove currency, %, whitespace and sign markers. Returns (digits_and_separators, is_negative)."""
    # \s also matches non-breaking and thin spaces, which some reports use as thousands separators
    number = re.sub(r"\s", "", _CURRENCY.sub("", text)).replace("−", "-")
    number = number.removeprefix("%").removesuffix("%")  # "12,5 %", or "% 12,5" in some tables
    is_negative = False
    if number.startswith("(") and number.endswith(")"):  # accounting style: (1.234) is -1234
        is_negative, number = True, number[1:-1]
    if number and number[0] in "+-":
        sign, number = number[0], number[1:]
        if sign == "-":
            is_negative = not is_negative
    return number, is_negative


def _decimal_evidence(number: str) -> str | None:
    """What a single cell tells us about the decimal separator, or None if it's ambiguous."""
    if "." in number and "," in number:  # "1.234,5" / "1,234.5": the last one is the decimal separator
        return "." if number.rfind(".") > number.rfind(",") else ","
    for separator, other_separator in ((".", ","), (",", ".")):
        occurrences = number.count(separator)
        if occurrences >= 2:  # "1.850.400": appears twice, so it must be the thousands separator
            return other_separator
        if occurrences == 1:
            digits_after = len(number) - number.index(separator) - 1
            if digits_after != _THOUSANDS_GROUP_SIZE:  # "0,2", "1.43"
                return separator
    return None  # "422,915" could be 422915 or 422.915


def infer_decimal_sep(cells: Iterable[str | None]) -> str | None:
    """Guess a table's decimal separator by letting every numeric-looking cell vote."""
    numbers = [_strip_decorations(cell)[0] for cell in cells if cell is not None and not is_blank(cell)]
    votes = [_decimal_evidence(number) for number in numbers if re.fullmatch(r"[\d.,]+", number)]
    dot_votes, comma_votes = votes.count("."), votes.count(",")
    if dot_votes == comma_votes:
        return None
    return "." if dot_votes > comma_votes else ","


def parse_number(text: str | None, decimal_sep: str | None = None) -> float | None:
    """Parse a formatted number. Returns None if the cell is blank or not a number.

    decimal_sep should come from infer_decimal_sep() over the whole table. Without it,
    each cell is judged on its own and an ambiguous "1.234" is read as 1234.
    Percentages are returned as written: "12,5 %" -> 12.5.
    """
    # The explicit None check lets the type checker narrow text to str below;
    # it can't see that inside is_blank().
    if text is None or is_blank(text):
        return None
    number, is_negative = _strip_decorations(text)

    decimal_separator = decimal_sep or _decimal_evidence(number)
    if decimal_separator is None:  # no decimal part; any separator present is a thousands separator
        integer_part, fraction = number, ""
        thousands_separator = "." if "." in number else ","
    else:
        thousands_separator = "," if decimal_separator == "." else "."
        integer_part, _, fraction = number.partition(decimal_separator)

    if thousands_separator in integer_part:
        # Thousands separators must split the digits into groups of 3: rejects "1.2.3"
        grouped_digits = rf"\d{{1,3}}(?:{re.escape(thousands_separator)}\d{{3}})+"
        if not re.fullmatch(grouped_digits, integer_part):
            return None
        integer_part = integer_part.replace(thousands_separator, "")
    if not re.fullmatch(r"\d+", integer_part) or not re.fullmatch(r"\d*", fraction):
        return None

    value = float(f"{integer_part}.{fraction or '0'}")
    return -value if is_negative else value
