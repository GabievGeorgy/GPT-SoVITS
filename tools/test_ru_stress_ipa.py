import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "GPT_SoVITS"))

from GPT_SoVITS.text.g2p_ru import stress_pipeline, word_to_ipa, ipa_word_to_ru_tokens


def main():
    samples = [
        "Предыдущем опыте через балалайку.",
        "Мука и мука.",
        "Ёж сидит ещё в чаще щавеля.",
    ]

    for text in samples:
        print("=" * 80)
        print("TEXT:", text)
        stressed_words = stress_pipeline(text)
        print("STRESSED WORDS:", stressed_words)

        for word, stress_vowel_ordinal in stressed_words:
            print("-" * 40)
            print(f"WORD: {word!r}, stress_vowel_ordinal={stress_vowel_ordinal}")
            ipa_seq = word_to_ipa(word)
            print("IPA:", ipa_seq)
            ru_tokens = ipa_word_to_ru_tokens(ipa_seq, stress_vowel_ordinal)
            print("RU TOKENS:", ru_tokens)


if __name__ == "__main__":
    main()
