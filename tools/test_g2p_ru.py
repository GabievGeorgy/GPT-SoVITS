from GPT_SoVITS.text.cleaner import clean_text


def main():
    text = "Привет, мир! Как дела?"
    phones, word2ph, norm_text = clean_text(text, language="ru", version="v2")
    print("TEXT:", text)
    print("NORM:", norm_text)
    print("PHONES:", phones)
    print("WORD2PH:", word2ph)
    print("LEN phones =", len(phones), "sum(word2ph) =", sum(word2ph) if word2ph is not None else None)


if __name__ == "__main__":
    main()
