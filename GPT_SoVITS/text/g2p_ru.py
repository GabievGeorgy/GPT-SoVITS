from typing import List, Tuple

# Basic Russian alphabet and a small set of punctuation we explicitly keep.
RU_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
RU_PUNCT = ",.!?-"


def _is_ru_letter(ch: str) -> bool:
    return ch in RU_LETTERS


def _is_ru_punct(ch: str) -> bool:
    return ch in RU_PUNCT


def text_to_phonemes(text: str) -> Tuple[List[str], List[int], str]:
    """
    Grapheme-based g2p for Russian.
    Returns phones (tokens like RU_а), word2ph counts per word, and normalized text.
    """
    text = text.lower()

    phones: List[str] = []
    word2ph: List[int] = []
    norm_words: List[str] = []

    for raw_word in text.split():
        word = raw_word.strip()
        if not word:
            continue

        start_len = len(phones)
        for ch in word:
            if _is_ru_letter(ch) or _is_ru_punct(ch):
                phones.append(f"RU_{ch}")

        new_len = len(phones)
        n_ph = new_len - start_len
        if n_ph > 0:
            word2ph.append(n_ph)
            norm_words.append(word)

    norm_text = " ".join(norm_words)
    return phones, word2ph, norm_text


def g2p(text: str):
    return text_to_phonemes(text)


if __name__ == "__main__":
    sample = "Привет, мир!"
    print(text_to_phonemes(sample))
