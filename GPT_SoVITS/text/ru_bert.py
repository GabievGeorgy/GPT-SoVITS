from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

# Cache to avoid reloading model/tokenizer on every call
_RU_BERT_TOKENIZER: Optional[AutoTokenizer] = None
_RU_BERT_MODEL: Optional[AutoModel] = None
_RU_BERT_DEVICE: str = "cpu"
_RU_BERT_HIDDEN: Optional[int] = None
RU_BERT_LOCAL_PATH = "GPT_SoVITS/pretrained_models/ruRoberta-large"
RU_BERT_REMOTE_PATH = "ai-forever/ruRoberta-large"


def resolve_ru_bert_path(path: Optional[str] = None) -> str:
    """
    Resolve the ruBERT/RuRoberta path in one place to avoid scattered literals.
    Priority: explicit argument -> env `ru_bert_path` -> local cache -> HF hub.
    """
    candidate = path if path not in [None, ""] else os.environ.get("ru_bert_path")
    if candidate not in [None, ""]:
        return candidate
    return RU_BERT_LOCAL_PATH if os.path.exists(RU_BERT_LOCAL_PATH) else RU_BERT_REMOTE_PATH


def load_ru_bert(
    bert_dir: Optional[str] = None,
    device: Optional[str] = None,
    is_half: bool = True,
) -> Tuple[AutoTokenizer, AutoModel, str, int]:
    """
    Load ruBERT/ruRoberta once and keep it in memory.
    Returns (tokenizer, model, device, hidden_size).
    """
    global _RU_BERT_TOKENIZER, _RU_BERT_MODEL, _RU_BERT_DEVICE, _RU_BERT_HIDDEN

    bert_dir = resolve_ru_bert_path(bert_dir)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    if _RU_BERT_MODEL is not None:
        return _RU_BERT_TOKENIZER, _RU_BERT_MODEL, _RU_BERT_DEVICE, _RU_BERT_HIDDEN  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(bert_dir, use_fast=True, add_prefix_space=True)
    model = AutoModel.from_pretrained(bert_dir)

    if is_half and device.startswith("cuda"):
        model = model.half()

    model = model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size

    _RU_BERT_TOKENIZER = tokenizer
    _RU_BERT_MODEL = model
    _RU_BERT_DEVICE = device
    _RU_BERT_HIDDEN = hidden_size

    return tokenizer, model, device, hidden_size


@torch.no_grad()
def get_ru_bert_feature(
    norm_text: str,
    word2ph: List[int],
    bert_dir: str,
    device: Optional[str] = None,
    is_half: bool = True,
) -> torch.Tensor:
    """
    norm_text: text after normalize_ru/clean_text(..., language="ru")
    word2ph: number of phonemes per word/punctuation in norm_text
    return: tensor [hidden_dim, sum(word2ph)] — phone-level BERT feature
    """
    tokenizer, model, device, hidden_size = load_ru_bert(
        bert_dir=bert_dir,
        device=device,
        is_half=is_half,
    )

    words = norm_text.split()
    if len(words) != len(word2ph):
        raise ValueError(
            f"len(words)={len(words)} != len(word2ph)={len(word2ph)}; norm_text='{norm_text}'"
        )

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    encoded = encoded.to(device)

    outputs = model(**encoded, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    token_emb = torch.stack(hidden_states[-4:], dim=0).mean(0)[0]

    word_ids = encoded.word_ids(0)

    word_vecs = []
    for word_index in range(len(words)):
        idxs = [i for i, wid in enumerate(word_ids) if wid == word_index]
        if not idxs:
            word_vecs.append(torch.zeros(hidden_size, device=device))
        else:
            vec = token_emb[idxs].mean(dim=0)
            word_vecs.append(vec)

    phone_level = []
    for vec, n_ph in zip(word_vecs, word2ph):
        if n_ph <= 0:
            continue
        phone_level.append(vec.unsqueeze(0).expand(n_ph, -1))

    if not phone_level:
        return torch.zeros(hidden_size, 0, device=device)

    phone_level = torch.cat(phone_level, dim=0)

    assert phone_level.shape[0] == sum(word2ph), (phone_level.shape, sum(word2ph))

    return phone_level.T
