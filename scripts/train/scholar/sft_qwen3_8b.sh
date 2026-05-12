#!/bin/bash
# Qwen3-8B Scholar SFT (LOCAL: NO BEAKER)
#
# Trains the model on shape-only multi-turn tool-use trajectories distilled
# from a GPT-5.x teacher (see scripts/data/scholar/distill.sh).
#
# The distill JSONL is structurally tulu-compatible:
#   {example_id, messages, dataset, ground_truth, _meta}
# Tokenization uses sft_tulu_tokenize_and_truncate_v1, which auto-masks
# every non-assistant turn (-100). Reward in `_meta` is NOT used by SFT.
#
# Usage:
#   ./scripts/train/scholar/sft_qwen3_8b.sh
#
# Common env overrides:
#   BASE_MODEL_NAME_OR_PATH   (default: /mlx_devbox/.../models/Qwen3-8B)
#   SFT_DATA_PATH             (default: data/scholar/sft_distill_score.jsonl)
#   PREPARED_SFT_DATA_PATH    (default: OUTPUT_DIR/sft_messages_only.jsonl)
#   OUTPUT_DIR                (default: outputs/qwen3_8b_scholar_sft)
#   EXP_NAME                  (default: qwen3_8b_scholar_sft)
#   NUM_PROCESSES             (default: 8)
#   MAX_SEQ_LENGTH            (default: 16384, matches RL --pack_length)
#   LEARNING_RATE             (default: 5e-6)
#   NUM_TRAIN_EPOCHS          (default: 2)
#   PER_DEVICE_TRAIN_BATCH_SIZE (default: 1)
#   GRADIENT_ACCUMULATION_STEPS (default: 8)  -> effective batch 64 with 8 GPU
#   DEEPSPEED_CONFIG          (default: configs/ds_configs/stage3_no_offloading_accelerate.conf)
#   KEEP_LAST_N_CHECKPOINTS   (default: 1; only keep the latest training state)
#   WANDB_MODE                (default: offline)
#   PUSH_TO_HUB               (default: false; keep local runs HF-token-free)

set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

export PATH="/home/tiger/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
ACCELERATE_BIN="${UV_PROJECT_ENVIRONMENT}/bin/accelerate"

if [ ! -x "${PYTHON_BIN}" ] || [ ! -x "${ACCELERATE_BIN}" ]; then
    echo "[scholar-sft] ERROR: ${PYTHON_BIN} / ${ACCELERATE_BIN} not found. Run: uv sync" >&2
    exit 1
fi

# --------------------------------------------------
# Config
# --------------------------------------------------
EXP_NAME="${EXP_NAME:=qwen3_8b_scholar_sft_3}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:=/mlx_devbox/users/luoyunze/playground/open-instruct/models/Qwen3-8B}"
SFT_DATA_PATH="${SFT_DATA_PATH:=${REPO_ROOT}/data/scholar/sft_distill_full.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:=${REPO_ROOT}/outputs/${EXP_NAME}}"
PREPARED_SFT_DATA_PATH="${PREPARED_SFT_DATA_PATH:=${OUTPUT_DIR}/sft_messages_only.jsonl}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:=configs/ds_configs/stage3_no_offloading_accelerate.conf}"

NUM_PROCESSES="${NUM_PROCESSES:=8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:=16384}"
LEARNING_RATE="${LEARNING_RATE:=5e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:=20}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:=8}"
WARMUP_RATIO="${WARMUP_RATIO:=0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:=0.0}"
LOGGING_STEPS="${LOGGING_STEPS:=5}"
KEEP_LAST_N_CHECKPOINTS="${KEEP_LAST_N_CHECKPOINTS:=1}"
SEED="${SEED:=1}"
WANDB_MODE="${WANDB_MODE:=offline}"
PUSH_TO_HUB="${PUSH_TO_HUB:=false}"
TRY_LAUNCH_BEAKER_EVAL_JOBS="${TRY_LAUNCH_BEAKER_EVAL_JOBS:=false}"
export WANDB_MODE

if [ ! -f "${SFT_DATA_PATH}" ]; then
    echo "[scholar-sft] ERROR: SFT data not found at ${SFT_DATA_PATH}" >&2
    echo "[scholar-sft]        Run scripts/data/scholar/distill.sh first." >&2
    exit 1
fi

if [ ! -f "${BASE_MODEL_NAME_OR_PATH}/config.json" ]; then
    echo "[scholar-sft] WARNING: ${BASE_MODEL_NAME_OR_PATH}/config.json missing. Treating as a HF id."
fi

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/sft_train.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[scholar-sft] Preparing schema-stable SFT JSONL: ${PREPARED_SFT_DATA_PATH}"
"${PYTHON_BIN}" - "${SFT_DATA_PATH}" "${PREPARED_SFT_DATA_PATH}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)

kept = 0
with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line_no, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        messages = obj.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{src}:{line_no} missing non-empty messages")
        # Keep only schema-stable fields needed by SFT. The original _meta
        # contains variable nested reward_breakdown keys, which breaks
        # datasets' JSON schema inference.
        out = {
            "messages": messages,
            "dataset": obj.get("dataset", "scholar_distill"),
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        kept += 1

print(f"[scholar-sft] Wrote {kept} examples to {dst}")
PY

EXTRA_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE}" != "disabled" ]; then
    EXTRA_ARGS+=(--with_tracking --report_to wandb)
fi

echo "[scholar-sft] EXP_NAME=${EXP_NAME}"
echo "[scholar-sft] MODEL=${BASE_MODEL_NAME_OR_PATH}"
echo "[scholar-sft] DATA=${SFT_DATA_PATH}"
echo "[scholar-sft] PREPARED_DATA=${PREPARED_SFT_DATA_PATH}"
echo "[scholar-sft] OUTPUT=${OUTPUT_DIR}"
echo "[scholar-sft] NUM_PROCESSES=${NUM_PROCESSES}  EPOCHS=${NUM_TRAIN_EPOCHS}  LR=${LEARNING_RATE}  MAX_SEQ=${MAX_SEQ_LENGTH}"

"${ACCELERATE_BIN}" launch \
    --mixed_precision bf16 \
    --num_processes "${NUM_PROCESSES}" \
    --use_deepspeed \
    --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name "${EXP_NAME}" \
    --model_name_or_path "${BASE_MODEL_NAME_OR_PATH}" \
    --tokenizer_name_or_path "${BASE_MODEL_NAME_OR_PATH}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type linear \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --gradient_checkpointing \
    --dataset_mixer_list "${PREPARED_SFT_DATA_PATH}" 1.0 \
    --dataset_mixer_list_splits train \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps "${LOGGING_STEPS}" \
    --checkpointing_steps epoch \
    --keep_last_n_checkpoints "${KEEP_LAST_N_CHECKPOINTS}" \
    --push_to_hub "${PUSH_TO_HUB}" \
    --try_launch_beaker_eval_jobs "${TRY_LAUNCH_BEAKER_EVAL_JOBS}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"

echo "[scholar-sft] Done. Final checkpoint at ${OUTPUT_DIR}"
