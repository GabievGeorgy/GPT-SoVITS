#!/usr/bin/env bash
set -euo pipefail

#
# End-to-end CLI pipeline (webui-equivalent) for:
#   - dataset preparation from a .list file
#   - SoVITS training
#   - GPT (Text2Semantic) training
# with optional Backblaze B2 uploads of full training checkpoints.
#
# This script is designed for Linux (e.g. vast.ai). It only orchestrates existing
# repo scripts; it does not change training logic.
#
# Example:
#   bash scripts/train_from_list.sh \
#     --version v4 \
#     --exp 500h_v4 \
#     --list /data/raw/xxx.list \
#     --wav-dir /data/raw/wavs \
#     --train-gpus 0-1-2-3-4-5-6-7 \
#     --gpus-1aa 0-0 \
#     --gpus-1ab 0-0 \
#     --gpus-1ac 0-0 \
#     --sovits-epochs 30 \
#     --gpt-epochs 30 \
#     --sovits-save-every 2 \
#     --gpt-save-every 2 \
#     --b2-bucket-id <bucketId> \
#     --b2-prefix checkpoints/500h_v4/
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXP_ROOT="${EXP_ROOT:-logs}"
IS_HALF="${IS_HALF:-True}"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_from_list.sh [options]

Required:
  --version <v1|v2|v2Pro|v2ProPlus|v3|v4>
  --exp <experiment_name>
  --list <path/to/data.list>
  --train-gpus <gpu-ids-like-0-1-2-3> (must be unique; used for SoVITS+GPT training)

Notes:
  - For dataset preparation, you can use duplicates like 0-0 to run multiple processes on one GPU.
  - For training, GPUs must be unique (use --train-gpus 0, not 0-0).

Optional (dataset):
  --wav-dir <dir>              (default: empty, use paths from .list)
  --bert-dir <path>            (default: "", keep repo defaults)
  --ssl-dir <path>             (default: GPT_SoVITS/pretrained_models/chinese-hubert-base)
  --gpus-1aa <gpu-ids>          (default: 0-0) tokenization + BERT feature extraction
  --gpus-1ab <gpu-ids>          (default: 0-0) speech SSL feature extraction
  --gpus-1ac <gpu-ids>          (default: 0-0) semantics token extraction

Optional (weights):
  --pretrained-s2g <path>      (default: based on --version)
  --pretrained-s2d <path>      (default: empty)
  --pretrained-s1  <path>      (default: based on --version)

Optional (training):
  --sovits-batch <int>         (default: 11)
  --sovits-epochs <int>        (default: 30)
  --sovits-save-every <int>    (default: 2) save full checkpoints every N epochs
  --lora-rank <int>            (default: 32) for v3/v4 SoVITS
  --gpt-batch <int>            (default: 13)
  --gpt-epochs <int>           (default: 30)
  --gpt-save-every <int>       (default: 2) save full checkpoints every N epochs
  --save-infer-weights <0|1>   (default: 1) also saves inference weights in *weights/ folders

Optional (Backblaze B2 upload; requires `b2` CLI configured):
  --b2-bucket-id <id>
  --b2-prefix <remote/prefix/> (default: checkpoints/<exp>/)
  --b2-threads <int>           (default: 8)
  --b2-poll-sec <int>          (default: 60) upload watcher poll interval

Optional (pipeline control):
  --skip-prepare               skip 1A/1B/1C if outputs already exist in logs/<exp>/
  --skip-sovits                skip SoVITS training
  --skip-gpt                   skip GPT training

EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

VERSION=""
EXP_NAME=""
LIST_PATH=""
WAV_DIR=""
TRAIN_GPUS=""
GPUS_1AA="0-0"
GPUS_1AB="0-0"
GPUS_1AC="0-0"
BERT_DIR=""
SSL_DIR="GPT_SoVITS/pretrained_models/chinese-hubert-base"

PRETRAINED_S2G=""
PRETRAINED_S2D=""
PRETRAINED_S1=""

S2_BATCH=11
S2_EPOCHS=30
LORA_RANK=32
S1_BATCH=13
S1_EPOCHS=30
S2_SAVE_EVERY_EPOCH=2
S1_SAVE_EVERY_EPOCH=2
SAVE_INFER_WEIGHTS=1

B2_BUCKET_ID=""
B2_PREFIX=""
B2_THREADS=8
B2_POLL_SEC=60

SKIP_PREPARE=0
SKIP_SOVITS=0
SKIP_GPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="${2:-}"; shift 2;;
    --exp) EXP_NAME="${2:-}"; shift 2;;
    --list) LIST_PATH="${2:-}"; shift 2;;
    --wav-dir) WAV_DIR="${2:-}"; shift 2;;
    --train-gpus) TRAIN_GPUS="${2:-}"; shift 2;;
    --gpus-1aa) GPUS_1AA="${2:-}"; shift 2;;
    --gpus-1ab) GPUS_1AB="${2:-}"; shift 2;;
    --gpus-1ac) GPUS_1AC="${2:-}"; shift 2;;
    --bert-dir) BERT_DIR="${2:-}"; shift 2;;
    --ssl-dir) SSL_DIR="${2:-}"; shift 2;;
    --pretrained-s2g) PRETRAINED_S2G="${2:-}"; shift 2;;
    --pretrained-s2d) PRETRAINED_S2D="${2:-}"; shift 2;;
    --pretrained-s1) PRETRAINED_S1="${2:-}"; shift 2;;
    --sovits-batch) S2_BATCH="${2:-}"; shift 2;;
    --sovits-epochs) S2_EPOCHS="${2:-}"; shift 2;;
    --sovits-save-every) S2_SAVE_EVERY_EPOCH="${2:-}"; shift 2;;
    --lora-rank) LORA_RANK="${2:-}"; shift 2;;
    --gpt-batch) S1_BATCH="${2:-}"; shift 2;;
    --gpt-epochs) S1_EPOCHS="${2:-}"; shift 2;;
    --gpt-save-every) S1_SAVE_EVERY_EPOCH="${2:-}"; shift 2;;
    --save-infer-weights) SAVE_INFER_WEIGHTS="${2:-}"; shift 2;;
    --b2-bucket-id) B2_BUCKET_ID="${2:-}"; shift 2;;
    --b2-prefix) B2_PREFIX="${2:-}"; shift 2;;
    --b2-threads) B2_THREADS="${2:-}"; shift 2;;
    --b2-poll-sec) B2_POLL_SEC="${2:-}"; shift 2;;
    --skip-prepare) SKIP_PREPARE=1; shift 1;;
    --skip-sovits) SKIP_SOVITS=1; shift 1;;
    --skip-gpt) SKIP_GPT=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) die "Unknown argument: $1";;
  esac
done

[[ -n "$VERSION" ]] || die "--version is required"
[[ -n "$EXP_NAME" ]] || die "--exp is required"
[[ -n "$LIST_PATH" ]] || die "--list is required"
[[ -n "$TRAIN_GPUS" ]] || die "--train-gpus is required"

[[ -n "$TRAIN_GPUS" ]] || die "--train-gpus resolved empty"
[[ -n "$GPUS_1AA" ]] || die "--gpus-1aa resolved empty"
[[ -n "$GPUS_1AB" ]] || die "--gpus-1ab resolved empty"
[[ -n "$GPUS_1AC" ]] || die "--gpus-1ac resolved empty"

[[ -f "$LIST_PATH" ]] || die ".list file not found: $LIST_PATH"
if [[ -n "$WAV_DIR" ]]; then
  [[ -d "$WAV_DIR" ]] || die "wav dir not found: $WAV_DIR"
fi
[[ -d "$SSL_DIR" ]] || die "ssl model dir not found: $SSL_DIR"

EXP_DIR="$EXP_ROOT/$EXP_NAME"
mkdir -p "$EXP_DIR"

if [[ -z "$B2_PREFIX" ]]; then
  B2_PREFIX="checkpoints/${EXP_NAME}/"
fi

default_pretrained_s2g() {
  case "$VERSION" in
    v4) echo "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth";;
    v3) echo "GPT_SoVITS/pretrained_models/s2Gv3.pth";;
    v2Pro) echo "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth";;
    v2ProPlus) echo "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth";;
    v2) echo "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth";;
    v1) echo "GPT_SoVITS/pretrained_models/s2G488k.pth";;
    *) echo "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth";;
  esac
}

default_pretrained_s1() {
  case "$VERSION" in
    v2) echo "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt";;
    v1) echo "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt";;
    *) echo "GPT_SoVITS/pretrained_models/s1v3.ckpt";;
  esac
}

if [[ -z "$PRETRAINED_S2G" ]]; then PRETRAINED_S2G="$(default_pretrained_s2g)"; fi
if [[ -z "$PRETRAINED_S1" ]]; then PRETRAINED_S1="$(default_pretrained_s1)"; fi

[[ -f "$PRETRAINED_S2G" ]] || die "pretrained_s2G not found: $PRETRAINED_S2G"
if [[ -n "$PRETRAINED_S2D" ]]; then
  [[ -f "$PRETRAINED_S2D" ]] || die "pretrained_s2D not found: $PRETRAINED_S2D"
fi
[[ -f "$PRETRAINED_S1" ]] || die "pretrained_s1 not found: $PRETRAINED_S1"

[[ "$S2_SAVE_EVERY_EPOCH" -ge 1 ]] || die "--sovits-save-every must be >= 1"
[[ "$S1_SAVE_EVERY_EPOCH" -ge 1 ]] || die "--gpt-save-every must be >= 1"

echo "== Config =="
echo "version=$VERSION exp=$EXP_NAME exp_dir=$EXP_DIR"
echo "gpus_1aa=$GPUS_1AA gpus_1ab=$GPUS_1AB gpus_1ac=$GPUS_1AC train_gpus=$TRAIN_GPUS is_half=$IS_HALF"
echo "list=$LIST_PATH wav_dir=${WAV_DIR:-<from list>}"
echo "ssl_dir=$SSL_DIR bert_dir=${BERT_DIR:-<repo default>}"
echo "pretrained_s2G=$PRETRAINED_S2G pretrained_s2D=${PRETRAINED_S2D:-<none>}"
echo "pretrained_s1=$PRETRAINED_S1"
echo "sovits: batch=$S2_BATCH epochs=$S2_EPOCHS save_every_epoch=$S2_SAVE_EVERY_EPOCH save_infer_weights=$SAVE_INFER_WEIGHTS"
if [[ "$VERSION" == "v3" || "$VERSION" == "v4" ]]; then
  echo "sovits: lora_rank=$LORA_RANK (v3/v4)"
fi
echo "gpt:    batch=$S1_BATCH epochs=$S1_EPOCHS save_every_epoch=$S1_SAVE_EVERY_EPOCH save_infer_weights=$SAVE_INFER_WEIGHTS"
echo "skip_prepare=$SKIP_PREPARE skip_sovits=$SKIP_SOVITS skip_gpt=$SKIP_GPT"

train_gpu_csv="${TRAIN_GPUS//-/,}"
train_gpu_parts=(${TRAIN_GPUS//-/ })

# Training GPUs must be unique; duplicates like 0-0 are useful for prep, but harmful for training.
declare -A _seen_train=()
for g in "${train_gpu_parts[@]}"; do
  if [[ -n "${_seen_train[$g]+x}" ]]; then
    die "Duplicate GPU id in --train-gpus: '$TRAIN_GPUS' (use 0-0 only for --gpus-1aa/--gpus-1ab/--gpus-1ac)"
  fi
  _seen_train[$g]=1
done

run_parts() {
  local gpus="$1"
  local script="$2"
  shift 2
  local gpu_parts=()
  IFS="-" read -r -a gpu_parts <<< "$gpus"
  local parts="${#gpu_parts[@]}"
  [[ "$parts" -ge 1 ]] || die "invalid gpu list: '$gpus'"

  local pids=()
  for ((i=0; i<parts; i++)); do
    (
      export inp_text="$LIST_PATH"
      export inp_wav_dir="$WAV_DIR"
      export exp_name="$EXP_NAME"
      export opt_dir="$EXP_DIR"
      export i_part="$i"
      export all_parts="$parts"
      export _CUDA_VISIBLE_DEVICES="${gpu_parts[$i]}"
      export is_half="$IS_HALF"
      "$PYTHON_BIN" -s "$script" "$@"
    ) &
    pids+=($!)
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  [[ "$failed" -eq 0 ]] || die "one or more workers failed for $script"
}

merge_parts() {
  local parts="$1"
  local pattern="$2"
  local out="$3"
  local tmp_files=()
  for ((i=0; i<parts; i++)); do
    tmp_files+=("$EXP_DIR/${pattern//\{i\}/$i}")
  done
  : > "$EXP_DIR/$out"
  for f in "${tmp_files[@]}"; do
    [[ -f "$f" ]] || die "missing expected output: $f"
    cat "$f" >> "$EXP_DIR/$out"
    # Ensure files are separated even if a part file has no trailing newline.
    echo >> "$EXP_DIR/$out"
    rm -f "$f"
  done
}

if [[ "$SKIP_PREPARE" -eq 0 ]]; then
  echo "== Stage 1A: text/phones/bert =="
  export bert_pretrained_dir="$BERT_DIR"
  run_parts "$GPUS_1AA" "GPT_SoVITS/prepare_datasets/1-get-text.py"
  IFS="-" read -r -a _g1aa <<< "$GPUS_1AA"
  merge_parts "${#_g1aa[@]}" "2-name2text-{i}.txt" "2-name2text.txt"

  echo "== Stage 1B: ssl/hubert + wav32k =="
  export cnhubert_base_dir="$SSL_DIR"
  run_parts "$GPUS_1AB" "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"

  if [[ "$VERSION" == *"Pro"* ]]; then
    echo "== Stage 1B(extra): speaker embedding for Pro* versions =="
    run_parts "$GPUS_1AB" "GPT_SoVITS/prepare_datasets/2-get-sv.py"
  fi

  echo "== Stage 1C: semantic tokens =="
  export pretrained_s2G="$PRETRAINED_S2G"
  export s2config_path=$(
    if [[ "$VERSION" == "v2Pro" || "$VERSION" == "v2ProPlus" ]]; then
      echo "GPT_SoVITS/configs/s2${VERSION}.json"
    else
      echo "GPT_SoVITS/configs/s2.json"
    fi
  )
  run_parts "$GPUS_1AC" "GPT_SoVITS/prepare_datasets/3-get-semantic.py"

  # Merge semantic parts into 6-name2semantic.tsv (same as webui)
  IFS="-" read -r -a _g1ac <<< "$GPUS_1AC"
  parts_1ac="${#_g1ac[@]}"
  {
    echo -e "item_name\tsemantic_audio"
    for ((i=0; i<parts_1ac; i++)); do
      f="$EXP_DIR/6-name2semantic-$i.tsv"
      [[ -f "$f" ]] || die "missing expected output: $f"
      cat "$f"
      # Ensure files are separated even if a part file has no trailing newline.
      echo
      rm -f "$f"
    done
  } > "$EXP_DIR/6-name2semantic.tsv"
else
  [[ -f "$EXP_DIR/2-name2text.txt" ]] || die "missing $EXP_DIR/2-name2text.txt (required when --skip-prepare)"
  [[ -f "$EXP_DIR/6-name2semantic.tsv" ]] || die "missing $EXP_DIR/6-name2semantic.tsv (required when --skip-prepare)"
  [[ -d "$EXP_DIR/4-cnhubert" ]] || die "missing $EXP_DIR/4-cnhubert (required when --skip-prepare)"
  [[ -d "$EXP_DIR/5-wav32k" ]] || die "missing $EXP_DIR/5-wav32k (required when --skip-prepare)"
fi

echo "== Training configs =="
S2_CFG="$EXP_DIR/s2_train_config.json"
S1_CFG="$EXP_DIR/s1_train_config.yaml"

export VERSION="$VERSION"
export EXP_NAME="$EXP_NAME"
export EXP_DIR="$EXP_DIR"
export TRAIN_GPUS="$TRAIN_GPUS"
export PRETRAINED_S2G="$PRETRAINED_S2G"
export PRETRAINED_S2D="$PRETRAINED_S2D"
export PRETRAINED_S1="$PRETRAINED_S1"
export S2_BATCH="$S2_BATCH"
export S2_EPOCHS="$S2_EPOCHS"
export LORA_RANK="$LORA_RANK"
export S1_BATCH="$S1_BATCH"
export S1_EPOCHS="$S1_EPOCHS"
export S2_SAVE_EVERY_EPOCH="$S2_SAVE_EVERY_EPOCH"
export S1_SAVE_EVERY_EPOCH="$S1_SAVE_EVERY_EPOCH"
export S2_CFG="$S2_CFG"
export S1_CFG="$S1_CFG"
export SAVE_INFER_WEIGHTS="$SAVE_INFER_WEIGHTS"
export IS_HALF="$IS_HALF"

"$PYTHON_BIN" - <<PY
import json, os
from config import SoVITS_weight_version2root, SoVITS_weight_finetune_version2root

version = os.environ["VERSION"]
exp_name = os.environ["EXP_NAME"]
exp_dir = os.environ["EXP_DIR"]
is_half = os.environ.get("IS_HALF", "True") == "True"
gpu_numbers = os.environ["TRAIN_GPUS"]
pretrained_s2G = os.environ["PRETRAINED_S2G"]
pretrained_s2D = os.environ.get("PRETRAINED_S2D", "")
batch_size = int(os.environ["S2_BATCH"])
epochs = int(os.environ["S2_EPOCHS"])
save_every_epoch = int(os.environ["S2_SAVE_EVERY_EPOCH"])
save_infer = os.environ.get("SAVE_INFER_WEIGHTS", "1") not in ("0", "False", "false")

config_file = "GPT_SoVITS/configs/s2.json" if version not in {"v2Pro", "v2ProPlus"} else f"GPT_SoVITS/configs/s2{version}.json"
with open(config_file, "r", encoding="utf-8") as f:
    data = json.load(f)

if not is_half:
    data["train"]["fp16_run"] = False
    batch_size = max(1, batch_size // 2)

data["train"]["batch_size"] = batch_size
data["train"]["epochs"] = epochs
data["train"]["text_low_lr_rate"] = float(data["train"].get("text_low_lr_rate", 0.4))
data["train"]["pretrained_s2G"] = pretrained_s2G
data["train"]["pretrained_s2D"] = pretrained_s2D
data["train"]["if_save_latest"] = 0
data["train"]["if_save_every_weights"] = bool(save_infer)
data["train"]["save_every_epoch"] = save_every_epoch
data["train"]["gpu_numbers"] = gpu_numbers
if version in {"v3", "v4"}:
    data["train"]["lora_rank"] = str(os.environ.get("LORA_RANK", "32"))
data["model"]["version"] = version
data["data"]["exp_dir"] = exp_dir
data["s2_ckpt_dir"] = exp_dir
data["name"] = exp_name
data["version"] = version

# Save small weights to a "weights" folder. For v2ProPlus, always use the finetune folder
# to ensure the saved weights are loadable for later finetuning (torch.save format).
if version == "v2ProPlus":
    weight_root = SoVITS_weight_finetune_version2root.get(version, SoVITS_weight_version2root.get(version, "SoVITS_weights"))
else:
    weight_root = SoVITS_weight_version2root.get(version, "SoVITS_weights")

os.makedirs(weight_root, exist_ok=True)
data["save_weight_dir"] = weight_root

# Match webui behavior for where it saves inference weights
if save_infer:
    # fallback mapping lives in webui; we just keep default weight dir field if present
    pass

out_path = os.environ["S2_CFG"]
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
print(out_path)
PY

"$PYTHON_BIN" - <<PY
import os, yaml
from config import GPT_weight_version2root, GPT_weight_finetune_version2root

version = os.environ["VERSION"]
exp_name = os.environ["EXP_NAME"]
exp_dir = os.environ["EXP_DIR"]
is_half = os.environ.get("IS_HALF", "True") == "True"
gpu_numbers = os.environ["TRAIN_GPUS"]
pretrained_s1 = os.environ["PRETRAINED_S1"]
batch_size = int(os.environ["S1_BATCH"])
epochs = int(os.environ["S1_EPOCHS"])
save_every_n_epoch = int(os.environ["S1_SAVE_EVERY_EPOCH"])
save_infer = os.environ.get("SAVE_INFER_WEIGHTS", "1") not in ("0", "False", "false")

cfg_path = "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

if not is_half:
    data["train"]["precision"] = "32"
    batch_size = max(1, batch_size // 2)

data["train"]["batch_size"] = batch_size
data["train"]["epochs"] = epochs
data["pretrained_s1"] = pretrained_s1
data["train"]["save_every_n_epoch"] = save_every_n_epoch
data["train"]["if_save_every_weights"] = bool(save_infer)
data["train"]["if_save_latest"] = False
data["train"]["if_dpo"] = False
data["train"]["exp_name"] = exp_name

if version == "v2ProPlus":
    gpt_root = GPT_weight_finetune_version2root.get(version, GPT_weight_version2root.get(version, "GPT_weights"))
else:
    gpt_root = GPT_weight_version2root.get(version, "GPT_weights")

os.makedirs(gpt_root, exist_ok=True)
data["train"]["half_weights_save_dir"] = gpt_root

data["train_semantic_path"] = f"{exp_dir}/6-name2semantic.tsv"
data["train_phoneme_path"] = f"{exp_dir}/2-name2text.txt"
data["output_dir"] = f"{exp_dir}/logs_s1_{version}"

out_path = os.environ["S1_CFG"]
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
print(out_path)
PY

upload_enabled=0
if [[ -n "$B2_BUCKET_ID" ]]; then
  command -v b2 >/dev/null 2>&1 || die "b2 CLI not found but --b2-bucket-id was provided"
  upload_enabled=1
fi

upload_file() {
  local local_path="$1"
  local remote_path="$2"
  b2 file upload --threads "$B2_THREADS" "$B2_BUCKET_ID" "$local_path" "$remote_path" >/dev/null
}

watch_and_upload() {
  local label="$1"
  local dir="$2"
  local glob="$3"
  local manifest="$4"

  mkdir -p "$(dirname "$manifest")"
  touch "$manifest"

  while true; do
    if [[ -f "$EXP_DIR/.stop_upload" ]]; then
      break
    fi
    # Upload config once if present
    if [[ -f "$EXP_DIR/config.json" ]] && ! grep -qF "$EXP_DIR/config.json" "$manifest"; then
      upload_file "$EXP_DIR/config.json" "${B2_PREFIX}config.json"
      echo "$EXP_DIR/config.json" >> "$manifest"
    fi
    if [[ -f "$S1_CFG" ]] && ! grep -qF "$S1_CFG" "$manifest"; then
      upload_file "$S1_CFG" "${B2_PREFIX}s1_train_config.yaml"
      echo "$S1_CFG" >> "$manifest"
    fi
    if [[ -f "$S2_CFG" ]] && ! grep -qF "$S2_CFG" "$manifest"; then
      upload_file "$S2_CFG" "${B2_PREFIX}s2_train_config.json"
      echo "$S2_CFG" >> "$manifest"
    fi

    # Upload new checkpoint files
    shopt -s nullglob
    local files=( "$dir"/$glob )
    shopt -u nullglob
    for f in "${files[@]}"; do
      [[ -f "$f" ]] || continue
      if grep -qF "$f" "$manifest"; then
        continue
      fi
      base="$(basename "$f")"
      upload_file "$f" "${B2_PREFIX}${label}/${base}"
      echo "$f" >> "$manifest"
    done

    sleep "$B2_POLL_SEC"
  done
}

uploader_pids=()
if [[ "$upload_enabled" -eq 1 ]]; then
  echo "== B2 upload enabled: bucket=$B2_BUCKET_ID prefix=$B2_PREFIX =="
  mkdir -p "$EXP_DIR/.upload"
  # SoVITS full checkpoints
  if [[ "$VERSION" == "v3" || "$VERSION" == "v4" ]]; then
    s2_ckpt_dir="$EXP_DIR/logs_s2_${VERSION}_lora_${LORA_RANK}"
  else
    s2_ckpt_dir="$EXP_DIR/logs_s2_${VERSION}"
  fi
  # GPT lightning checkpoints
  s1_ckpt_dir="$EXP_DIR/logs_s1_${VERSION}/ckpt"
  (
    watch_and_upload "s2" "$s2_ckpt_dir" "G_*.pth" "$EXP_DIR/.upload/manifest_s2.txt"
  ) & uploader_pids+=($!)
  (
    watch_and_upload "s2" "$s2_ckpt_dir" "D_*.pth" "$EXP_DIR/.upload/manifest_s2.txt"
  ) & uploader_pids+=($!)
  (
    watch_and_upload "s1" "$s1_ckpt_dir" "*.ckpt" "$EXP_DIR/.upload/manifest_s1.txt"
  ) & uploader_pids+=($!)

  # Small finetuneable weights (for later finetune/inference)
  s2_weights_dir="$("$PYTHON_BIN" - <<'PY'
import json, os
p = os.environ.get("S2_CFG", "")
if not p:
    raise SystemExit(0)
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("save_weight_dir", ""))
PY
)"
  s1_weights_dir="$("$PYTHON_BIN" - <<'PY'
import os, yaml
p = os.environ.get("S1_CFG", "")
if not p:
    raise SystemExit(0)
with open(p, "r", encoding="utf-8") as f:
    d = yaml.safe_load(f)
print(((d or {}).get("train") or {}).get("half_weights_save_dir", ""))
PY
)"

  if [[ -n "$s2_weights_dir" && -d "$s2_weights_dir" ]]; then
    (
      watch_and_upload "s2_weights" "$s2_weights_dir" "*.pth" "$EXP_DIR/.upload/manifest_s2_weights.txt"
    ) & uploader_pids+=($!)
  fi
  if [[ -n "$s1_weights_dir" && -d "$s1_weights_dir" ]]; then
    (
      watch_and_upload "s1_weights" "$s1_weights_dir" "*.ckpt" "$EXP_DIR/.upload/manifest_s1_weights.txt"
    ) & uploader_pids+=($!)
  fi
fi

if [[ "$SKIP_SOVITS" -eq 0 ]]; then
  echo "== Stage 1Ba: SoVITS train =="
  if [[ "$VERSION" == "v1" || "$VERSION" == "v2" || "$VERSION" == "v2Pro" || "$VERSION" == "v2ProPlus" ]]; then
    "$PYTHON_BIN" -s GPT_SoVITS/s2_train.py --config "$S2_CFG"
  else
    "$PYTHON_BIN" -s GPT_SoVITS/s2_train_v3_lora.py --config "$S2_CFG"
  fi
else
  echo "== Stage 1Ba: SoVITS train skipped =="
fi

if [[ "$SKIP_GPT" -eq 0 ]]; then
  echo "== Stage 1Bb: GPT train =="
  export _CUDA_VISIBLE_DEVICES="$train_gpu_csv"
  export hz="25hz"
  "$PYTHON_BIN" -s GPT_SoVITS/s1_train.py --config_file "$S1_CFG"
else
  echo "== Stage 1Bb: GPT train skipped =="
fi

touch "$EXP_DIR/.stop_upload" || true
if [[ "${#uploader_pids[@]}" -gt 0 ]]; then
  for pid in "${uploader_pids[@]}"; do
    wait "$pid" || true
  done
fi

echo "== Done =="
echo "Experiment folder: $EXP_DIR"
