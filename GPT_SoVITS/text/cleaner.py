from text import cleaned_text_to_sequence
import os
# if os.environ.get("version","v1")=="v1":
#     from text import chinese
#     from text.symbols import symbols
# else:
#     from text import chinese2 as chinese
#     from text.symbols2 import symbols

from text import symbols as symbols_v1
from text import symbols2 as symbols_v2

special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]


def clean_text(text, language, version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    language = (language or "en").lower()
    if version == "v1":
        symbols = symbols_v1.symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english", "ru": "g2p_ru"}
    else:
        symbols = symbols_v2.symbols
        language_module_map = {
            "zh": "chinese2",
            "ja": "japanese",
            "en": "english",
            "ko": "korean",
            "yue": "cantonese",
            "ru": "g2p_ru",
        }

    if language not in language_module_map:
        language = "en"
        text = " "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol, version)
    language_module = __import__("text." + language_module_map[language], fromlist=[language_module_map[language]])
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text
    g2p_fn = getattr(language_module, "text_to_phonemes", getattr(language_module, "g2p"))

    def _split_g2p_result(result, fallback_norm):
        if isinstance(result, tuple):
            if len(result) == 3:
                ph, word2ph_res, new_norm = result
                return list(ph), word2ph_res, new_norm
            if len(result) == 2:
                ph, word2ph_res = result
                return list(ph), word2ph_res, fallback_norm
        return list(result), None, fallback_norm

    g2p_result = g2p_fn(norm_text)
    if language == "zh" or language == "yue":  ##########
        phones, word2ph = g2p_result
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = list(g2p_result)
        if len(phones) < 4:
            phones = [","] + phones
        word2ph = None
    else:
        phones, word2ph, norm_text = _split_g2p_result(g2p_result, norm_text)
    phones = ["UNK" if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol, version=None):
    if version is None:
        version = os.environ.get("version", "v2")
    language = (language or "en").lower()
    if version == "v1":
        symbols = symbols_v1.symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english", "ru": "g2p_ru"}
    else:
        symbols = symbols_v2.symbols
        language_module_map = {
            "zh": "chinese2",
            "ja": "japanese",
            "en": "english",
            "ko": "korean",
            "yue": "cantonese",
            "ru": "g2p_ru",
        }

    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = __import__("text." + language_module_map[language], fromlist=[language_module_map[language]])
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language, version=None):
    version = os.environ.get("version", version)
    if version is None:
        version = "v2"
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones, version)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
