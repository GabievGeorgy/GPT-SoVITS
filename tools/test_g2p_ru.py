import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "GPT_SoVITS"))

from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.text import symbols2 as symbols_mod
from GPT_SoVITS.text.ru_normalizer import normalize_ru


def check_normalizer_dash_and_ellipsis():
    text = "молотые зерна - это мука... ещё—пример"
    norm = normalize_ru(text)
    assert "..." in norm, norm
    assert "—" not in norm, norm  # typographic dash must be normalized to '-'
    parts = norm.split()
    assert "-" in parts, norm  # spaced dash kept for pause
    assert any(tok.endswith("...") for tok in parts), norm  # trailing ellipsis preserved as three dots


def main():
    check_normalizer_dash_and_ellipsis()

    samples = [
        "Привет, мир! Как дела?",
        "123 руб. ул. Ленина, д. 5.",
        "Ёж сидит ещё в чаще щавеля.",
        "Мука и мука.",
        "На горе стоит замок. Замок старый.",
        "В замке висел старый замок.",
        "молотые зерна— это мука... ещё пример",
    ]

    symset = set(symbols_mod.symbols)

    for text in samples:
        print("=" * 80)
        print("TEXT:", text)
        phones, word2ph, norm_text = clean_text(text, language="ru", version="v2")
        print("NORM:", norm_text)
        print("PHONES:", phones)
        print("WORD2PH:", word2ph)

        total = sum(word2ph) if word2ph is not None else None
        print("LEN phones =", len(phones), "sum(word2ph) =", total)

        if total is not None and total != len(phones):
            raise RuntimeError(
                f"len(phones) != sum(word2ph): {len(phones)} vs {total}"
            )

        missing = [p for p in phones if p not in symset]
        print("Missing in symbols2:", missing)
        if missing:
            raise RuntimeError(f"Found {len(missing)} tokens outside symbols2: {set(missing)}")

        unk = [p for p in phones if p == "UNK"]
        print("UNK tokens:", unk)

        stress_positions = [i for i, p in enumerate(phones) if p == "RU_STRESS"]
        print("RU_STRESS positions:", stress_positions)

        print("-" * 80)


if __name__ == "__main__":
    main()
