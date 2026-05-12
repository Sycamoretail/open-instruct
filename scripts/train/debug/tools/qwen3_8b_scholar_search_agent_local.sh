#!/bin/bash
# Qwen3-8B scholar search agent training (LOCAL: NO BEAKER, NO JQ)
#
# Requirements before running:
# 1. export PATH="/home/tiger/.local/bin:$PATH"
# 2. export MODEL_NAME_OR_PATH=/path/to/Qwen3-8B (or your downloaded model path)
# 3. export HF_HOME (optional, where model is cached)
#
# Two-stage training:
# Phase1 (format-only: mock rewards, mock tools):
# 4. SCHOLAR_REWARD_MODE=mock SCHOLAR_TOOL_BACKEND=mock ./scripts/train/debug/tools/qwen3_8b_scholar_search_agent_local.sh
#
# Phase2 (real evaluator + internal tools):
# 4. SCHOLAR_REWARD_MODE=real SCHOLAR_TOOL_BACKEND=internal ./scripts/train/debug/tools/qwen3_8b_scholar_search_agent_local.sh
#
# By default, this runs Phase2: real rewards, internal tools.
#

set -euo pipefail

# Always run from repo root so relative paths and uv env resolution are stable.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

# --------------------------------------------------
# Default env and configuration
# --------------------------------------------------
SCHOLAR_REWARD_MODE="${SCHOLAR_REWARD_MODE:=real}"
SCHOLAR_TOOL_BACKEND="${SCHOLAR_TOOL_BACKEND:=internal}"
EXP_NAME="${EXP_NAME:=qwen3_8b_scholar_search_agent_en_real}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:=/mlx_devbox/users/luoyunze/playground/open-instruct/models/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:=/mlx_devbox/users/luoyunze/playground/open-instruct/outputs/${EXP_NAME}}"
SCHOLAR_RAW_DIR="${SCHOLAR_RAW_DIR:=/mlx_devbox/users/luoyunze/playground/open-instruct/data/scholar_en}"
SCHOLAR_RLVR_DIR="${SCHOLAR_RLVR_DIR:=/mlx_devbox/users/luoyunze/playground/open-instruct/data/scholar_en_rlvr}"
WANDB_MODE="${WANDB_MODE:=offline}"
INIT_FROM_MOCK="${INIT_FROM_MOCK:=0}"
MOCK_INIT_EXP_NAME="${MOCK_INIT_EXP_NAME:=qwen3_8b_scholar_search_agent}"
MOCK_INIT_OUTPUT_DIR="${MOCK_INIT_OUTPUT_DIR:=/mlx_devbox/users/luoyunze/playground/open-instruct/outputs/${MOCK_INIT_EXP_NAME}}"
SAVE_FREQ="${SAVE_FREQ:=50}"
CHECKPOINT_STATE_FREQ="${CHECKPOINT_STATE_FREQ:=20}"
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:=/mlx_devbox/users/luoyunze/playground/open-instruct/outputs/${EXP_NAME}_training_state}"
MAX_RETRIES="${MAX_RETRIES:=5}"
RETRY_DELAY_SECS="${RETRY_DELAY_SECS:=30}"
MAX_PROMPT_TOKEN_LENGTH="${MAX_PROMPT_TOKEN_LENGTH:=2048}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:=16384}"
PACK_LENGTH="${PACK_LENGTH:=18500}"
NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:=16}"
NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:=8}"
MAX_STEPS="${MAX_STEPS:=10}"
PER_TURN_MAX_TOKENS="${PER_TURN_MAX_TOKENS:=2048}"
TOOL_CALL_TIMEOUT="${TOOL_CALL_TIMEOUT:=300}"

# Set up PATH for uv
export PATH="/home/tiger/.local/bin:$PATH"

# Avoid uv spamming Ray logs about VIRTUAL_ENV mismatch by making the project
# env path absolute (matches VIRTUAL_ENV if you previously activated it).
export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[scholar] ERROR: ${PYTHON_BIN} not found. Run: uv sync"
    exit 1
fi

# Less Ray noise.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# Required by vLLM in some Ray/spawn serialization paths (e.g. torch.dtype).
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Force single-node local runs onto IPv4 loopback. This avoids vLLM/NCCL
# bind failures on hosts where Ray only reports an IPv6 address.
export OPEN_INSTRUCT_HOST_IP="${OPEN_INSTRUCT_HOST_IP:=127.0.0.1}"
export VLLM_HOST_IP="${VLLM_HOST_IP:=${OPEN_INSTRUCT_HOST_IP}}"
export MASTER_ADDR="${MASTER_ADDR:=${OPEN_INSTRUCT_HOST_IP}}"
export WANDB_MODE

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"
export RAY_TMPDIR="${RAY_TMPDIR:=/tmp/r}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# W&B auth: prefer passing WANDB_API_KEY from your shell instead of hardcoding
# a secret in the repository. The script will forward it when present.
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
elif [ "${WANDB_MODE}" = "online" ]; then
    echo "[scholar] WARNING: WANDB_MODE=online but WANDB_API_KEY is not set."
fi

# Set up mock reward if needed
if [ "${SCHOLAR_REWARD_MODE}" = "mock" ]; then
    echo "[scholar] Mock reward (format-only training)"
    export SCHOLAR_MOCK_REWARD=1
else
    echo "[scholar] Real evaluator reward"
    unset SCHOLAR_MOCK_REWARD
fi

resolve_latest_mock_model_dir() {
    if [ -n "${MOCK_INIT_MODEL_NAME_OR_PATH:-}" ] && [ -f "${MOCK_INIT_MODEL_NAME_OR_PATH}/config.json" ]; then
        echo "${MOCK_INIT_MODEL_NAME_OR_PATH}"
        return 0
    fi
    if [ ! -d "${MOCK_INIT_OUTPUT_DIR}" ]; then
        return 1
    fi
    while IFS= read -r dir; do
        if [ -f "${dir}/.checkpoint_complete" ] && [ -f "${dir}/config.json" ]; then
            echo "${dir}"
            return 0
        fi
    done < <(
        find "${MOCK_INIT_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name '*_checkpoints' -printf '%T@ %p\n' \
            | sort -nr \
            | awk '{ $1=""; sub(/^ /, ""); print }'
    )
    return 1
}

if [ -n "${MODEL_NAME_OR_PATH:-}" ]; then
    EFFECTIVE_MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
elif [ "${SCHOLAR_REWARD_MODE}" = "real" ] && [ "${INIT_FROM_MOCK}" != "0" ] && [ "${INIT_FROM_MOCK}" != "false" ]; then
    if EFFECTIVE_MODEL_NAME_OR_PATH="$(resolve_latest_mock_model_dir)"; then
        echo "[scholar] Initializing real training from mock model: ${EFFECTIVE_MODEL_NAME_OR_PATH}"
    else
        EFFECTIVE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH}"
        echo "[scholar] WARNING: mock init model not found, falling back to base model: ${EFFECTIVE_MODEL_NAME_OR_PATH}"
    fi
else
    EFFECTIVE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH}"
fi

# --------------------------------------------------
# 1. Prepare scholar RLVR data if not already done
# --------------------------------------------------
if [ ! -f "${SCHOLAR_RLVR_DIR}/.prep.${SCHOLAR_TOOL_BACKEND}.done" ]; then
    echo "[scholar] Preparing scholar RLVR data (backend=${SCHOLAR_TOOL_BACKEND})"
    "${PYTHON_BIN}" scripts/data/scholar/prepare_scholar_data.py \
        --input_dir "${SCHOLAR_RAW_DIR}" \
        --output_dir "${SCHOLAR_RLVR_DIR}" \
        --tool_backend "${SCHOLAR_TOOL_BACKEND}"
    touch "${SCHOLAR_RLVR_DIR}/.prep.${SCHOLAR_TOOL_BACKEND}.done"
    echo "[scholar] Done preparing data"
fi

# --------------------------------------------------
# 2. Tool selection
# --------------------------------------------------
if [ "${SCHOLAR_TOOL_BACKEND}" = "internal" ]; then
    TRAIN_TOOLS=("scholar_search" "general_search" "fetch")
    TRAIN_TOOL_NAMES=("ScholarSearch" "GeneralSearch" "Fetch")
    TRAIN_TOOL_CONFIGS=("{}" "{}" "{}")
else
    TRAIN_TOOLS=("mock_scholar_search")
    TRAIN_TOOL_NAMES=("ScholarSearch")
    TRAIN_TOOL_CONFIGS=("{}")
fi

# --------------------------------------------------
# 3. Run training (8 GPUs, Deepspeed Stage3
# --------------------------------------------------
echo "[scholar] Starting training on 8 GPUs..."
echo "[scholar] Experiment name: ${EXP_NAME}"
echo "[scholar] Model init path: ${EFFECTIVE_MODEL_NAME_OR_PATH}"
if [ -f "${CHECKPOINT_STATE_DIR}/latest" ]; then
    echo "[scholar] Found checkpoint state at ${CHECKPOINT_STATE_DIR}; training should resume from the latest saved step."
elif [ -d "${CHECKPOINT_STATE_DIR}" ]; then
    if [ -z "$(find "${CHECKPOINT_STATE_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        rmdir "${CHECKPOINT_STATE_DIR}"
        echo "[scholar] Removed empty stale checkpoint state dir: ${CHECKPOINT_STATE_DIR}"
        echo "[scholar] No checkpoint state found at ${CHECKPOINT_STATE_DIR}; training will start from step 1."
    else
        echo "[scholar] ERROR: ${CHECKPOINT_STATE_DIR} exists but has no latest file."
        echo "[scholar] Refusing to start because grpo_fast would try to load an invalid checkpoint state directory."
        echo "[scholar] Either remove the invalid directory or set CHECKPOINT_STATE_DIR to a clean path."
        exit 1
    fi
else
    echo "[scholar] No checkpoint state found at ${CHECKPOINT_STATE_DIR}; training will start from step 1."
fi
echo "[scholar] Checkpoint state dir: ${CHECKPOINT_STATE_DIR}"
echo "[scholar] Checkpoint state freq: ${CHECKPOINT_STATE_FREQ}"

# --------------------------------------------------
# Retry wrapper: restarts training on failure up to MAX_RETRIES times.
# The checkpoint_state_dir mechanism ensures training resumes from the
# last saved step rather than restarting from scratch.
# --------------------------------------------------
attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[scholar] ===== Training attempt ${attempt}/${MAX_RETRIES} ($(date)) ====="

    set +e
    "${PYTHON_BIN}" -u open_instruct/grpo_fast.py \
    --dataset_mixer_list \
        "${SCHOLAR_RLVR_DIR}/search_data.2023.parquet" 1.0 \
        "${SCHOLAR_RLVR_DIR}/understanding_data.2023_new.parquet" 1.0 \
        "${SCHOLAR_RLVR_DIR}/write_section_data.2023.parquet" 1.0 \
        "${SCHOLAR_RLVR_DIR}/write_survey_data.2023.parquet" 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list "${SCHOLAR_RLVR_DIR}/search_data.2023.parquet" 32 \
    --dataset_mixer_eval_list_splits train \
    --sft_messages_key messages \
    --ground_truths_key ground_truth \
    --max_prompt_token_length "${MAX_PROMPT_TOKEN_LENGTH}" \
    --response_length "${RESPONSE_LENGTH}" \
    --pack_length "${PACK_LENGTH}" \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS_ROLLOUT}" \
    --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" \
    --model_name_or_path "${EFFECTIVE_MODEL_NAME_OR_PATH}" \
    --apply_verifiable_reward true \
    --temperature 1.0 \
    --exp_name "${EXP_NAME}" \
    --learning_rate 5e-7 \
    --total_episodes $((800 * 32 * 8)) \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 6 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 1 \
    --local_eval_every 20 \
    --gradient_checkpointing \
    --push_to_hub false \
    --output_dir "${OUTPUT_DIR}" \
    --kl_estimator 2 \
    --non_stop_penalty false \
    --num_mini_batches 1 \
    --lr_scheduler_type constant \
    --with_tracking \
    --wandb_project scholar_search_agent \
    --wandb_group_name qwen3_8b_scholar_en_real \
    --save_freq "${SAVE_FREQ}" \
    --checkpoint_state_freq "${CHECKPOINT_STATE_FREQ}" \
    --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}" \
    --keep_last_n_checkpoints 2 \
    --vllm_enable_prefix_caching true \
    --tools "${TRAIN_TOOLS[@]}" \
    --tool_call_names "${TRAIN_TOOL_NAMES[@]}" \
    --tool_configs "${TRAIN_TOOL_CONFIGS[@]}" \
    --max_steps "${MAX_STEPS}" \
    --per_turn_max_tokens "${PER_TURN_MAX_TOKENS}" \
    --pass_tools_to_chat_template false \
    --tool_parser_type dr_tulu
    exit_code=$?
    set -e

    if [ ${exit_code} -eq 0 ]; then
        echo "[scholar] Training completed successfully on attempt ${attempt}."
        break
    fi

    echo "[scholar] WARNING: Training failed with exit code ${exit_code} on attempt ${attempt}."

    if [ ${attempt} -ge ${MAX_RETRIES} ]; then
        echo "[scholar] ERROR: Exhausted all ${MAX_RETRIES} retry attempts. Giving up."
        exit ${exit_code}
    fi

    echo "[scholar] Waiting ${RETRY_DELAY_SECS}s before retry..."
    sleep "${RETRY_DELAY_SECS}"

    # Clean up stale Ray processes that may linger after a crash.
    ray stop --force 2>/dev/null || true
    sleep 5
done

echo "[scholar] Done training. Checkpoints saved to ${OUTPUT_DIR}"
