import argparse
from pathlib import Path
from typing import Iterable, Tuple

import torch
import yaml


def _find_weight_tensor(weight_dict: dict, candidates: Iterable[str], suffix_hint: str) -> Tuple[str, torch.Tensor]:
    for key in candidates:
        if key in weight_dict:
            return key, weight_dict[key]
    for key in weight_dict:
        if key.endswith(suffix_hint):
            return key, weight_dict[key]
    raise KeyError(f"Cannot find embedding weight with suffix '{suffix_hint}'. Available keys: {list(weight_dict.keys())}")


def expand_gpt_embeddings(ckpt_path_in: Path, ckpt_path_out: Path, phoneme_vocab_size: int):
    ckpt = torch.load(ckpt_path_in, map_location="cpu")
    weight_dict = ckpt["weight"]
    key, emb = _find_weight_tensor(
        weight_dict,
        ["model.ar_text_embedding.word_embeddings.weight"],
        "ar_text_embedding.word_embeddings.weight",
    )
    old_vocab, emb_dim = emb.shape

    if old_vocab >= phoneme_vocab_size:
        print(f"[GPT] No expansion needed: old_vocab={old_vocab}, target={phoneme_vocab_size}")
        ckpt["config"]["model"]["phoneme_vocab_size"] = phoneme_vocab_size
        torch.save(ckpt, ckpt_path_out)
        return

    delta = phoneme_vocab_size - old_vocab
    std = emb.std().item()
    extra = torch.randn(delta, emb_dim) * std
    new_emb = torch.cat([emb, extra], dim=0)

    weight_dict[key] = new_emb
    ckpt["config"]["model"]["phoneme_vocab_size"] = phoneme_vocab_size

    torch.save(ckpt, ckpt_path_out)
    print(f"[GPT] Expanded embedding: {old_vocab} -> {phoneme_vocab_size} (delta={delta}, std={std:.6f})")


def expand_sovits_embeddings(ckpt_path_in: Path, ckpt_path_out: Path, phoneme_vocab_size: int):
    ckpt = torch.load(ckpt_path_in, map_location="cpu")
    weight_dict = ckpt["weight"]
    key, emb = _find_weight_tensor(weight_dict, ["enc_p.text_embedding.weight"], "enc_p.text_embedding.weight")
    old_vocab, emb_dim = emb.shape

    if old_vocab >= phoneme_vocab_size:
        print(f"[SoVITS] No expansion needed: old_vocab={old_vocab}, target={phoneme_vocab_size}")
        torch.save(ckpt, ckpt_path_out)
        return

    delta = phoneme_vocab_size - old_vocab
    std = emb.std().item()
    extra = torch.randn(delta, emb_dim) * std
    new_emb = torch.cat([emb, extra], dim=0)

    weight_dict[key] = new_emb
    torch.save(ckpt, ckpt_path_out)
    print(f"[SoVITS] Expanded embedding: {old_vocab} -> {phoneme_vocab_size} (delta={delta}, std={std:.6f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="GPT_SoVITS/configs/s1longer-v2.yaml")
    ap.add_argument("--gpt-in", type=str, required=True)
    ap.add_argument("--gpt-out", type=str, required=True)
    ap.add_argument("--sovits-in", type=str, required=True)
    ap.add_argument("--sovits-out", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    phoneme_vocab_size = int(cfg["model"]["phoneme_vocab_size"])

    expand_gpt_embeddings(Path(args.gpt_in), Path(args.gpt_out), phoneme_vocab_size)
    expand_sovits_embeddings(Path(args.sovits_in), Path(args.sovits_out), phoneme_vocab_size)


if __name__ == "__main__":
    main()
