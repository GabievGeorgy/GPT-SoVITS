import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "GPT_SoVITS"))

from GPT_SoVITS.text.cleaner import clean_text


def main():
    samples = [
        "Привет, мир! Как дела?",
        "123 руб. ул. Ленина, д. 5.",
    ]
    for text in samples:
        phones, word2ph, norm_text = clean_text(text, language="ru", version="v2")
        print("TEXT:", text)
        print("NORM:", norm_text)
        print("PHONES:", phones)
        print("WORD2PH:", word2ph)
        total = sum(word2ph) if word2ph is not None else None
        print("LEN phones =", len(phones), "sum(word2ph) =", total)
        print("-" * 40)


if __name__ == "__main__":
    main()
