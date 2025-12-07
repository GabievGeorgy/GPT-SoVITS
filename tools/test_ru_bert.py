import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "GPT_SoVITS"))

from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.text.ru_bert import (
    get_ru_bert_feature,
    is_ru_bert_enabled,
    load_ru_bert,
    resolve_ru_bert_path,
)


def main():
    if not is_ru_bert_enabled():
        print("RU_BERT_ENABLED is disabled; skipping ruBERT test.")
        return

    bert_dir = resolve_ru_bert_path(str(ROOT / "GPT_SoVITS" / "pretrained_models" / "ruRoberta-large"))

    load_ru_bert(bert_dir, device="cpu", is_half=False)

    samples = [
        "Привет, мир! Как дела?",
        "123 руб. ул. Ленина, д. 5.",
        "Ёж сидит ещё в чаще щавеля.",
        "Мука и мука.",
    ]

    for text in samples:
        print("=" * 80)
        print("TEXT:", text)
        phones, word2ph, norm_text = clean_text(text, language="ru", version="v2")
        print("NORM:", norm_text)
        print("PHONES:", phones)
        print("WORD2PH:", word2ph)
        print("LEN phones =", len(phones), "sum(word2ph) =", sum(word2ph))

        bert = get_ru_bert_feature(
            norm_text=norm_text,
            word2ph=word2ph,
            bert_dir=bert_dir,
            device="cpu",
            is_half=False,
        )
        print("BERT SHAPE:", bert.shape)
        assert bert.shape[1] == len(phones)

        print("OK for this sample")

    print("ALL GOOD")


if __name__ == "__main__":
    main()
