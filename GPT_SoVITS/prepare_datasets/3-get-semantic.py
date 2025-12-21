import os

inp_text = os.environ.get("inp_text")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
pretrained_s2G = os.environ.get("pretrained_s2G")
s2config_path = os.environ.get("s2config_path")

import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

import logging
import traceback

import torch
from tools.my_utils import clean_path

pretrained_s2G = clean_path(pretrained_s2G or "")
if not pretrained_s2G:
    raise ValueError("Missing env var: pretrained_s2G")

# Backward-compat: some UI flows accidentally prefix root weight paths with "GPT_SoVITS/".
if not os.path.exists(pretrained_s2G) and pretrained_s2G.startswith(f"GPT_SoVITS{os.sep}SoVITS_weights"):
    alt = pretrained_s2G.replace(f"GPT_SoVITS{os.sep}", "", 1)
    if os.path.exists(alt):
        pretrained_s2G = alt

if not os.path.exists(pretrained_s2G):
    raise FileNotFoundError(pretrained_s2G)

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

from config import pretrained_sovits_name
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from module.quantize import ResidualVectorQuantizer
import utils
from torch import nn

logging.getLogger("numba").setLevel(logging.WARNING)
# from config import pretrained_s2G

# inp_text=sys.argv[1]
# exp_name=sys.argv[2]
# i_part=sys.argv[3]
# all_parts=sys.argv[4]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[5]
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name


hubert_dir = "%s/4-cnhubert" % (opt_dir)
semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
if os.path.exists(semantic_path) == False:
    os.makedirs(opt_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    hps = utils.get_hparams_from_file(s2config_path)

    # Detect v3/v4 (new header / lora) correctly. File-size heuristics misclassify v4 as v3.
    version, model_version, if_lora = get_sovits_version_from_path_fast(pretrained_s2G)

    semantic_frame_rate = getattr(getattr(hps, "model", None), "semantic_frame_rate", None) or "25hz"
    if semantic_frame_rate not in ("25hz", "50hz"):
        semantic_frame_rate = "25hz"

    class _SemanticExtractor(nn.Module):
        def __init__(self, semantic_frame_rate: str):
            super().__init__()
            ssl_dim = 768
            if semantic_frame_rate == "25hz":
                self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
            else:
                self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)
            self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)

        @torch.no_grad()
        def extract_latent(self, x):
            ssl = self.ssl_proj(x)
            _, codes, _, _ = self.quantizer(ssl)
            return codes.transpose(0, 1)

    vq_model = _SemanticExtractor(semantic_frame_rate)
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    # utils.load_checkpoint(utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth"), vq_model, None, True)
    # utils.load_checkpoint(pretrained_s2G, vq_model, None, True)
    def _strip_module_prefix(sd: dict) -> dict:
        if any(k.startswith("module.") for k in sd.keys()):
            return {k.replace("module.", "", 1): v for k, v in sd.items()}
        return sd

    dict_s2 = load_sovits_new(pretrained_s2G)
    weights = dict_s2.get("weight", {}) or {}

    # For LoRA weights (v3/v4), files may be delta-only. For semantic tokens we only need ssl_proj+quantizer,
    # so we can safely fall back to the base pretrained weights for those modules.
    need_prefixes = ("ssl_proj.", "quantizer.")
    has_needed = any(k.lstrip("module.").startswith(need_prefixes) for k in weights.keys())
    if (not has_needed) or if_lora:
        base_path = pretrained_sovits_name.get(model_version) or pretrained_sovits_name.get(version)
        if base_path and os.path.exists(base_path):
            base_weights = (load_sovits_new(base_path).get("weight", {}) or {})
            # Prefer user-provided weights over base.
            weights = {**base_weights, **weights}
        elif not has_needed:
            raise RuntimeError(
                f"pretrained_s2G does not contain ssl_proj/quantizer weights and base checkpoint is missing: {base_path}"
            )

    weights = _strip_module_prefix(weights)
    extractor_weights = {k: v for k, v in weights.items() if k.startswith(need_prefixes)}
    load_res = vq_model.load_state_dict(extractor_weights, strict=False)
    if any(k.startswith(need_prefixes) for k in getattr(load_res, "missing_keys", [])):
        raise RuntimeError(f"Failed to load ssl_proj/quantizer weights: {load_res}")
    print(load_res)

    def name2go(wav_name, lines):
        hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
        if os.path.exists(hubert_path) == False:
            return
        ssl_content = torch.load(hubert_path, map_location="cpu")
        if is_half == True:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append("%s\t%s" % (wav_name, semantic))

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines1 = []
    for line in lines[int(i_part) :: int(all_parts)]:
        # print(line)
        try:
            # wav_name,text=line.split("\t")
            wav_name, spk_name, language, text = line.split("|")
            wav_name = clean_path(wav_name)
            wav_name = os.path.basename(wav_name)
            # name2go(name,lines1)
            name2go(wav_name, lines1)
        except:
            print(line, traceback.format_exc())
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))
