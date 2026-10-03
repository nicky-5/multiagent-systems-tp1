"""Parse numbers that PDFs store as formatted text ("$ (1.234,5)", "12,5 %", "1,234.5")."""

import re
from collections.abc import Iterable

_CURRENCY = re.compile(r"US\$|AR\$|USD|ARS|EUR|[$€£]", re.IGNORECASE)
_BLANKS = {"", "-", "–", "—", "n/a", "n/d", "s/d", "...", "…"}


def is_blank(text: str | None) -> bool:
    """True for empty cells and the usual "no data" placeholders."""
    return text is None or text.strip().lower() in _BLANKS


def _strip_decorations(text: str) -> tuple[str, bool]:
    """Remove currency, %, whitespace and sign markers. Returns (digits_and_separators, negative)."""
    # \s also matches non-breaking and thin spaces, which some reports use as thousands separators
    s = re.sub(r"\s", "", _CURRENCY.sub("", text)).replace("−", "-").removesuffix("%")
    negative = False
    if s.startswith("(") and s.endswith(")"):  # accounting style: (1.234) is -1234
        negative, s = True, s[1:-1]
    if s and s[0] in "+-":
        negative, s = negative ^ (s[0] == "-"), s[1:]
    return s, negative


def _decimal_evidence(s: str) -> str | None:
    """What a single cell tells us about the decimal separator, or None if it's ambiguous."""
    if "." in s and "," in s:  # "1.234,5" / "1,234.5": the last one is the decimal separator
        return "." if s.rfind(".") > s.rfind(",") else ","
    for sep, other in ((".", ","), (",", ".")):
        count = s.count(sep)
        if count >= 2:  # "1.850.400": appears twice, so it must be the thousands separator
            return other
        if count == 1 and len(s) - s.index(sep) - 1 != 3:  # "0,2", "1.43": not a 3-digit group
            return sep
    return None  # "422,915" could be 422915 or 422.915


def infer_decimal_sep(cells: Iterable[str | None]) -> str | None:
    """Guess a table's decimal separator by letting every numeric-looking cell vote."""
    votes = {".": 0, ",": 0}
    for cell in cells:
        if cell is None or is_blank(cell):
            continue
        s, _ = _strip_decorations(cell)
        if re.fullmatch(r"[\d.,]+", s) and (sep := _decimal_evidence(s)):
            votes[sep] += 1
    if votes["."] == votes[","]:
        return None
    return "." if votes["."] > votes[","] else ","


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
    s, negative = _strip_decorations(text)

    decimal = decimal_sep or _decimal_evidence(s)
    if decimal is None:  # no decimal part; any separator present is a thousands separator
        int_part, frac = s, ""
        thousands = "." if "." in s else ","
    else:
        thousands = "," if decimal == "." else "."
        int_part, _, frac = s.partition(decimal)

    if thousands in int_part:
        # Thousands separators must split the digits into groups of 3: rejects "1.2.3"
        if not re.fullmatch(rf"\d{{1,3}}(?:{re.escape(thousands)}\d{{3}})+", int_part):
            return None
        int_part = int_part.replace(thousands, "")
    if not re.fullmatch(r"\d+", int_part) or not re.fullmatch(r"\d*", frac):
        return None

    value = float(f"{int_part}.{frac or '0'}")
    return -value if negative else value
