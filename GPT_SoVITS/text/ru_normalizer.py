import re
from num2words import num2words

# Numbers with optional decimal part.
RE_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

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


def normalize_ru(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    text = _replace_abbrev(text)
    text = _replace_numbers(text)

    # Remove stray symbols, keep punctuation we model.
    text = re.sub(r"[^0-9A-Za-zА-Яа-яЁё,.!? -]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_normalize(text: str) -> str:
    return normalize_ru(text)
