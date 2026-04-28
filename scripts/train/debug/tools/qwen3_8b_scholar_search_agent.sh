#!/bin/bash
# Train Qwen3-8B as a scholar search agent with GRPO.
#
# Two-phase plan:
#   PHASE 1 (this script, default): bootstrap the protocol.
#     - Uses the company-internal tools from ./tools (ScholarSearch /
#       GeneralSearch / Fetch) wrapped via
#       open_instruct/environments/tools/scholar_tools.py.
#     - Uses the ``dr_tulu`` tool parser so <call_tool name="...">...</call_tool>
#       is the wire format (matches the figure in our spec).
#     - Uses the mock scholar_verifier reward (SCHOLAR_MOCK_REWARD=1), which
#       only rewards protocol compliance (<think>/<call_tool>/<answer>/<cite>).
#       This lets the policy learn to emit well-formed tool calls before we
#       pay the cost of the full evaluator.py judge.
#
#   PHASE 2: unset SCHOLAR_MOCK_REWARD and rerun. Reward then flows through
#            evaluator.py::evaluate_scholar_score_verifier (nugget coverage /
#            reference coverage / citation precision / relevance rate).
#
# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------
# Qwen3-8B is not bundled here. Download it once, then point
# ``MODEL_NAME_OR_PATH`` at the local snapshot:
#
#     huggingface-cli download Qwen/Qwen3-8B \
#         --local-dir /mlx_devbox/users/luoyunze/models/Qwen3-8B \
#         --local-dir-use-symlinks False
#
# (or use ``modelscope download`` if HF is blocked). Exporting
# ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE`` also works if you prefer the
# standard HF cache location instead of --local-dir.
# ---------------------------------------------------------------------------

set -e

SCHOLAR_RAW_DIR=${SCHOLAR_RAW_DIR:-scholar}
SCHOLAR_RLVR_DIR=${SCHOLAR_RLVR_DIR:-scholar_rlvr}

# Tool backend: "internal" wires ./tools/*.py; "mock" uses the offline
# deterministic fake scholar search (no network / no internal auth needed).
TOOL_BACKEND=${TOOL_BACKEND:-internal}

# Reward mode: "mock" rewards format only; "real" runs evaluator.py.
# Phase 1 default: mock.
SCHOLAR_REWARD_MODE=${SCHOLAR_REWARD_MODE:-mock}
if [ "${SCHOLAR_REWARD_MODE}" = "mock" ]; then
    export SCHOLAR_MOCK_REWARD=1
    echo "[scholar] Reward: mock (format-only)."
else
    unset SCHOLAR_MOCK_REWARD
    echo "[scholar] Reward: evaluator.py (nugget/reference/citation)."
fi

# Default model path (override by exporting MODEL_NAME_OR_PATH).
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-models/Qwen3-8B}
if [ ! -d "${MODEL_NAME_OR_PATH}" ]; then
    echo "[scholar][warn] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH} does not exist."
    echo "[scholar][warn] Download Qwen3-8B first (see the header of this script)."
fi

# Step 1: regenerate RLVR parquet if missing (system prompt depends on backend).
if [ ! -f "${SCHOLAR_RLVR_DIR}/.prep.${TOOL_BACKEND}.done" ]; then
    echo "[scholar] Preparing RLVR parquet (backend=${TOOL_BACKEND}) -> ${SCHOLAR_RLVR_DIR}..."
    python3 scripts/data/scholar/prepare_scholar_data.py \
        --input_dir "${SCHOLAR_RAW_DIR}" \
        --output_dir "${SCHOLAR_RLVR_DIR}" \
        --tool_backend "${TOOL_BACKEND}"
    touch "${SCHOLAR_RLVR_DIR}/.prep.${TOOL_BACKEND}.done"
fi

# ---------------------------------------------------------------------------
# Tool selection for GRPO.
# ---------------------------------------------------------------------------
# config_name (from scholar_tools.py) -> CLI --tools flag:
#   scholar_search / general_search / fetch   (backend=internal)
#   mock_scholar_search                       (backend=mock)
if [ "${TOOL_BACKEND}" = "internal" ]; then
    TOOLS=(scholar_search general_search fetch)
    TOOL_CALL_NAMES=(ScholarSearch GeneralSearch Fetch)
    TOOL_CONFIGS=('{}' '{}' '{}')
else
    TOOLS=(mock_scholar_search)
    TOOL_CALL_NAMES=(ScholarSearch)
    TOOL_CONFIGS=('{}')
fi

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

python open_instruct/grpo_fast.py \
    --dataset_mixer_list \
        "${SCHOLAR_RLVR_DIR}/search_train.2023.parquet" 1.0 \
        "${SCHOLAR_RLVR_DIR}/understanding_train.2023.parquet" 1.0 \
        "${SCHOLAR_RLVR_DIR}/write_section_train.2023.parquet" 1.0 \
        "${SCHOLAR_RLVR_DIR}/write_survey_train.2023.parquet" 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list "${SCHOLAR_RLVR_DIR}/search_train.2023.parquet" 32 \
    --dataset_mixer_eval_list_splits train \
    --sft_messages_key messages \
    --ground_truths_key ground_truth \
    --dataset_source_key dataset \
    --max_prompt_token_length 4096 \
    --response_length 8192 \
    --pack_length 16384 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 8 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --exp_name qwen3_8b_scholar_search_agent \
    --learning_rate 3e-7 \
    --total_episodes $((500 * 32 * 8)) \
    --deepspeed_stage 3 \
    --num_epochs 1 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.01 \
    --seed 1 \
    --local_eval_every 20 \
    --gradient_checkpointing \
    --push_to_hub false \
    --output_dir /output/qwen3_8b_scholar_search_agent \
    --kl_estimator 2 \
    --non_stop_penalty False \
    --num_mini_batches 1 \
    --lr_scheduler_type constant \
    --save_freq 100 \
    --vllm_enable_prefix_caching \
    --tools "${TOOLS[@]}" \
    --tool_call_names "${TOOL_CALL_NAMES[@]}" \
    --tool_configs "${TOOL_CONFIGS[@]}" \
    --tool_parser_type dr_tulu
