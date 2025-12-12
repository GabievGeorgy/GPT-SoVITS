import re
from num2words import num2words

# Numbers with optional decimal part.
RE_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

# Map common dash characters to ASCII hyphen-minus.
_DASH_TRANSLATION = {
    ord("-"): "-",
    ord("\u2010"): "-",  # hyphen
    ord("\u2011"): "-",  # non-breaking hyphen
    ord("\u2012"): "-",  # figure dash
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u2015"): "-",  # horizontal bar
    ord("\u2212"): "-",  # minus sign
}

# Word characters (Latin + Cyrillic) for hyphen protection.
_WORD = r"0-9A-Za-z\u0400-\u04FF"
# Allowable chars after cleanup: digits, Latin/Cyrillic letters, basic punct.
_CLEAN_RE = re.compile(rf"[^{_WORD},.!? -]")


def _replace_numbers(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(0).replace(",", ".")
        try:
            if "." in raw:
                number = float(raw)
            else:
                number = int(raw)
            return num2words(number, lang="ru")
        except Exception:
            return match.group(0)

    return RE_NUMBER.sub(repl, text)


def _replace_abbrev(text: str) -> str:
    return text


def _normalize_dashes_and_ellipsis(text: str) -> str:
    """
    - Normalize various dash characters to '-'.
    - Normalize ellipsis to three dots ("...") to avoid extra symbols.
    - Ensure standalone dashes (not inside word-word) are spaced as pauses.
    """
    text = text.translate(_DASH_TRANSLATION)

    # Normalize both "..." and Unicode ellipsis (U+2026) to "...".
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"\u2026+", "...", text)

    placeholder = "__HYPHEN__"
    # Protect true hyphenated words (word-word) so we don't force a pause there.
    text = re.sub(rf"([{_WORD}])-+(?=[{_WORD}])", rf"\1{placeholder}", text)
    # Remaining dashes become pauses.
    text = re.sub(r"-+", " - ", text)
    text = text.replace(placeholder, "-")
    return text


def normalize_ru(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    text = _replace_abbrev(text)
    text = _replace_numbers(text)
    text = _normalize_dashes_and_ellipsis(text)

    # Remove stray symbols, keep punctuation we model.
    text = _CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_normalize(text: str) -> str:
    return normalize_ru(text)
