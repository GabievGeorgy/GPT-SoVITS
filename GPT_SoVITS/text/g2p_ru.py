from typing import List, Tuple

from gruut import sentences as gruut_sentences

from .ru_normalizer import normalize_ru

RU_PUNCT = ",.!?-"

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


def ipa_to_ru_tokens(phone: str) -> List[str]:
    tokens: List[str] = []

    if not phone:
        return tokens

    # Stress marker can be a separate symbol or inline.
    if "ˈ" in phone or "'" in phone:
        tokens.append("RU_STRESS")
        phone = phone.replace("ˈ", "").replace("'", "")

    if phone in ("ˈ", "ˌ", "'"):
        tokens.append("RU_STRESS")
        return tokens

    phone = phone.replace("ː", "")

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
    elif phone == "ɕ":
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

    # Unknown IPA symbol.
    return tokens


def _fallback_letters_to_tokens(word: str) -> List[str]:
    tokens: List[str] = []
    for ch in word:
        tokens.extend(LETTER_FALLBACK.get(ch, []))
    return tokens


def text_to_phonemes(text: str) -> Tuple[List[str], List[int], str]:
    """
    text -> (phones, word2ph, norm_text)
    """
    text = text.strip()
    if not text:
        return [], [], ""

    norm_text = normalize_ru(text)

    phones: List[str] = []
    word2ph: List[int] = []
    norm_words: List[str] = []

    for sent in gruut_sentences(norm_text, lang="ru"):
        for word in sent.words:
            raw = word.text
            if not raw or raw.isspace():
                continue

            if all(ch in RU_PUNCT for ch in raw):
                phones.extend(list(raw))
                continue

            start_len = len(phones)

            if getattr(word, "phonemes", None):
                for ph in word.phonemes:
                    ph_text = getattr(ph, "text", ph)
                    tokens = ipa_to_ru_tokens(ph_text)
                    phones.extend(tokens)
            else:
                phones.extend(_fallback_letters_to_tokens(raw.lower()))

            n_ph = len(phones) - start_len
            if n_ph > 0:
                word2ph.append(n_ph)
                norm_words.append(raw.lower())

    norm_text_out = " ".join(norm_words)
    return phones, word2ph, norm_text_out


def g2p(text: str):
    return text_to_phonemes(text)


if __name__ == "__main__":
    sample = "Привет, мир! 123 руб. ул. Ленина, д. 5."
    print(text_to_phonemes(sample))
