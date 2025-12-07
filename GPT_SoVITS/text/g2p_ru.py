from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .ru_normalizer import normalize_ru
from . import symbols2 as symbols_mod

RU_PUNCT = ",.!?-"
RU_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"

# IPA vowels used to count vowel index within a word
IPA_VOWELS = {
    "a",
    "ɐ",
    "ə",
    "o",
    "ɔ",
    "ɵ",
    "e",
    "ɛ",
    "i",
    "ɪ",
    "u",
    "ʊ",
    "ɨ",
    "y",
    "æ",
    "ɜ",
}

VOWEL_MAP = {
    "a": "RU_a",
    "ä": "RU_a",
    "ɐ": "RU_a",
    "ə": "RU_a",
    "o": "RU_o",
    "ɔ": "RU_o",
    "ɵ": "RU_o",
    "e": "RU_e",
    "ɛ": "RU_e",
    "ɜ": "RU_e",
    "i": "RU_i",
    "ɪ": "RU_i",
    "u": "RU_u",
    "ʊ": "RU_u",
    "ɨ": "RU_y",
}

CONS_MAP = {
    "p": "RU_p",
    "b": "RU_b",
    "t": "RU_t",
    "d": "RU_d",
    "k": "RU_k",
    "g": "RU_g",
    "f": "RU_f",
    "v": "RU_v",
    "s": "RU_s",
    "z": "RU_z",
    "m": "RU_m",
    "n": "RU_n",
    "r": "RU_r",
    "l": "RU_l",
    "x": "RU_h",
    "ʂ": "RU_sh",
    "ʐ": "RU_zh",
    "j": "RU_j",
    "ɫ": "RU_l",
}

LETTER_FALLBACK = {
    "а": ["RU_a"],
    "б": ["RU_b"],
    "в": ["RU_v"],
    "г": ["RU_g"],
    "д": ["RU_d"],
    "е": ["RU_j", "RU_e"],
    "ё": ["RU_j", "RU_o"],
    "ж": ["RU_zh"],
    "з": ["RU_z"],
    "и": ["RU_i"],
    "й": ["RU_j"],
    "к": ["RU_k"],
    "л": ["RU_l"],
    "м": ["RU_m"],
    "н": ["RU_n"],
    "о": ["RU_o"],
    "п": ["RU_p"],
    "р": ["RU_r"],
    "с": ["RU_s"],
    "т": ["RU_t"],
    "у": ["RU_u"],
    "ф": ["RU_f"],
    "х": ["RU_h"],
    "ц": ["RU_ts"],
    "ч": ["RU_ch"],
    "ш": ["RU_sh"],
    "щ": ["RU_sh", "RU_soft"],
    "ъ": [],
    "ы": ["RU_y"],
    "ь": ["RU_soft"],
    "э": ["RU_e"],
    "ю": ["RU_j", "RU_u"],
    "я": ["RU_j", "RU_a"],
}

ALL_SYMBOLS = set(symbols_mod.symbols)
_SILERO_ACCENTOR = None
_TRYIPA_MODEL = None


def _load_silero_accentor():
    """
    Lazy-load silero-stress via the official API.
    Returns a callable accentor: accentor(text) -> str.
    """
    global _SILERO_ACCENTOR
    if _SILERO_ACCENTOR is not None:
        return _SILERO_ACCENTOR

    try:
        from silero_stress import load_accentor
    except ImportError as exc:
        raise RuntimeError(
            "silero-stress is required for Russian stress marks. "
            "Install via `pip install silero-stress`."
        ) from exc

    accentor = load_accentor()
    if accentor is None:
        raise RuntimeError("Failed to load silero-stress accentor")

    _SILERO_ACCENTOR = accentor
    return _SILERO_ACCENTOR


def run_silero_stress(norm_text: str) -> str:
    """
    Run silero-stress (or another stressor) on NORMALIZED text.
    Expected: string with `+` after a stressed vowel inside a word.
    Examples:
        'предыдущем опыте' -> 'предыд+ущем +опыте'
    Note: real implementation depends on how silero-stress is wired; this is the interface.
    """
    if not norm_text.strip():
        return ""

    accentor = _load_silero_accentor()
    stressed = accentor(norm_text)
    if stressed is None:
        return ""

    return str(stressed)


def extract_word_and_stress(word_with_plus: str) -> Tuple[str, Optional[int]]:
    """
    'предыд+ущем'  -> ('предыдущем', 2)   # + before stressed vowel
    'пред+ыдущем'  -> ('предыдущем', 2)   # + after stressed vowel
    'через'        -> ('через', None)
    '+опыте'       -> ('опыте', 0)        # edge case but handled

    stress_vowel_ordinal: index of stressed vowel among vowels in the word (0, 1, 2, ...)
    or None if `+` absent or cannot be aligned to a vowel.
    Supports both placements of `+`: before or after the stressed vowel.
    """
    if "+" not in word_with_plus:
        return word_with_plus, None

    plus_pos = word_with_plus.index("+")
    vowel_ord = -1
    stress_vowel_ordinal: Optional[int] = None

    clean_chars: List[str] = []

    for i, ch in enumerate(word_with_plus):
        if ch == "+":
            next_char = word_with_plus[i + 1] if i + 1 < len(word_with_plus) else ""
            if next_char in RU_VOWELS:
                stress_vowel_ordinal = vowel_ord + 1
            continue

        clean_chars.append(ch)

        if ch in RU_VOWELS:
            vowel_ord += 1

            if i + 1 == plus_pos and stress_vowel_ordinal is None:
                stress_vowel_ordinal = vowel_ord

    clean_word = "".join(clean_chars)
    return clean_word, stress_vowel_ordinal


def stress_pipeline(text: str) -> List[Tuple[str, Optional[int]]]:
    """
    High-level step:
      raw text -> normalize_ru -> silero-stress -> list of (word, stressed vowel index)

    Returns tuples per whitespace token in stressed_text.
    Punctuation is not separated here; g2p_ru handles it later.
    """
    text = text.strip()
    if not text:
        return []

    norm_text = normalize_ru(text)
    if not norm_text:
        return []

    norm_text_for_stress = norm_text.replace("-", " ")

    stressed = run_silero_stress(norm_text_for_stress)
    if not stressed:
        return []

    result: List[Tuple[str, Optional[int]]] = []
    for token in stressed.split():
        clean_word, stress_vowel_ordinal = extract_word_and_stress(token)
        if not clean_word:
            continue
        result.append((clean_word, stress_vowel_ordinal))

    return result


def load_tryipa_model():
    """
    Lazy-load TryIPaG2P model.
    Implemented according to the current TryIPaG2P API.
    """
    global _TRYIPA_MODEL
    if _TRYIPA_MODEL is not None:
        return _TRYIPA_MODEL

    try:
        from tryiparu import G2PModel
    except ImportError as exc:
        raise RuntimeError(
            "TryIPaG2P is required for Russian IPA conversion. "
            "Install via `pip install git+https://github.com/NikiPshg/TryIPaG2P.git`."
        ) from exc

    import torch

    def _noop_compile(module, *args, **kwargs):
        return module

    torch.compile = _noop_compile  # type: ignore[attr-defined]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _TRYIPA_MODEL = G2PModel(device=device, load_dataset=True)
    return _TRYIPA_MODEL


def word_to_ipa(word: str) -> List[str]:
    """
    Wrapper over TryIPaG2P: one word -> list of IPA tokens.
    Must return a **list** of phones, not a single string.
    """
    model = load_tryipa_model()

    ipa_seq = model(word.strip())

    if isinstance(ipa_seq, str):
        ipa_seq = ipa_seq.split()
    elif isinstance(ipa_seq, tuple):
        ipa_seq = list(ipa_seq)

    ipa_list = list(ipa_seq)
    return [token for token in ipa_list if token and not token.isspace()]


def ipa_to_ru_tokens(phone: str) -> List[str]:
    tokens: List[str] = []
    if not phone:
        return tokens

    stressed = False
    if any(m in phone for m in ("ˈ", "ˌ", "'")):
        stressed = True
        phone = phone.replace("ˈ", "").replace("ˌ", "").replace("'", "")

    phone = phone.replace("ː", "")

    if stressed:
        tokens.append("RU_STRESS")

    if not phone:
        return tokens

    if phone in VOWEL_MAP:
        tokens.append(VOWEL_MAP[phone])
        return tokens

    soft = False
    if "ʲ" in phone:
        soft = True
        phone = phone.replace("ʲ", "")

    if phone in ("t͡s", "ts"):
        base_token = "RU_ts"
    elif phone in ("t͡ɕ", "tɕ"):
        base_token = "RU_ch"
        soft = True
    elif phone in ("ɕ", "ɕː"):
        base_token = "RU_sh"
        soft = True
    elif phone == "ʑ":
        base_token = "RU_zh"
        soft = True
    else:
        base_token = CONS_MAP.get(phone)

    if base_token:
        tokens.append(base_token)
        if soft:
            tokens.append("RU_soft")

    return tokens


def ipa_word_to_ru_tokens(ipa_seq: List[str], stress_vowel_ordinal: Optional[int]) -> List[str]:
    """
    Core step: IPA sequence + stressed-vowel index -> list of RU_* tokens.
    Inserts 'RU_STRESS' BEFORE the stressed vowel.

    'RU_STRESS' is a normal token; its length contributes to word2ph.
    """
    ru_tokens: List[str] = []
    ipa_vowel_idx = 0

    for phone in ipa_seq:
        core = phone

        if core in IPA_VOWELS:
            stressed_here = stress_vowel_ordinal is not None and ipa_vowel_idx == stress_vowel_ordinal

            if stressed_here:
                ru_tokens.append("RU_STRESS")

            ru_tokens.extend(ipa_to_ru_tokens(phone))
            ipa_vowel_idx += 1
        else:
            ru_tokens.extend(ipa_to_ru_tokens(phone))

    return ru_tokens


def _fallback_letters_to_tokens(word: str) -> List[str]:
    tokens: List[str] = []
    for ch in word:
        tokens.extend(LETTER_FALLBACK.get(ch, []))
    return tokens


def _filter_unknown(phones: List[str]) -> List[str]:
    out: List[str] = []
    for ph in phones:
        if ph in ALL_SYMBOLS:
            out.append(ph)
        else:
            if "UNK" in ALL_SYMBOLS:
                out.append("UNK")
    return out


def text_to_phonemes(text: str) -> Tuple[List[str], List[int], str]:
    text = text.strip()
    if not text:
        return [], [], ""

    norm_text = normalize_ru(text)
    if not norm_text:
        return [], [], ""

    phones: List[str] = []
    word2ph: List[int] = []
    norm_words: List[str] = []

    stressed_words = stress_pipeline(text)

    for raw_word, stress_vowel_ordinal in stressed_words:
        if not raw_word or raw_word.isspace():
            continue

        if all(ch in RU_PUNCT for ch in raw_word):
            for ch in raw_word:
                phones.append(ch)
                word2ph.append(1)
                norm_words.append(ch)
            continue

        word = raw_word.lower()

        ipa_seq: Optional[List[str]] = None
        ru_tokens: List[str] = []

        try:
            ipa_seq = word_to_ipa(word)
        except Exception:
            ipa_seq = None

        if ipa_seq:
            ru_tokens = ipa_word_to_ru_tokens(ipa_seq, stress_vowel_ordinal)

        if not ru_tokens:
            ru_tokens = _fallback_letters_to_tokens(word)

        n_ph = len(ru_tokens)
        if n_ph <= 0:
            continue

        phones.extend(ru_tokens)
        word2ph.append(n_ph)
        norm_words.append(word)

    phones = _filter_unknown(phones)

    norm_text_out = " ".join(norm_words)
    return phones, word2ph, norm_text_out


def g2p(text: str):
    return text_to_phonemes(text)
