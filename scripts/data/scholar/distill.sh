#!/bin/bash
# Distill multi-turn tool-use SFT trajectories from a GPT-5.x teacher.
# Wraps scripts/data/scholar/distill_sft_trajectories.py.
#
# Three preset profiles (selected via PROFILE env var):
#   PROFILE=pilot    -> 20 examples, no scoring/filtering, 1 input file
#   PROFILE=score    -> all 5 RLVR parquets, --always-score, no filter (default)
#   PROFILE=filter   -> all 5 RLVR parquets + per-data_type reward thresholds
#
# Usage:
#   AK_LIST="$AK1,$AK2,$AK3,$AK4" \
#       ./scripts/data/scholar/distill.sh
#   PROFILE=pilot AK_LIST="$AK1" ./scripts/data/scholar/distill.sh
#   PROFILE=filter AK_LIST="$AK1,$AK2" ./scripts/data/scholar/distill.sh
#
# Required env:
#   AK_LIST              Comma-separated Azure OpenAI API keys
#
# Common env overrides:
#   PROFILE              pilot | score | filter   (default: score)
#   MODEL_LIST           Comma-separated model names aligned to AK_LIST
#                        (default: gpt-5-2025-08-07 broadcast to all AKs)
#   RLVR_DIR             default: data/scholar_en_rlvr
#   OUTPUT_JSONL         absolute output path
#                        (default: data/scholar/sft_distill_${PROFILE}.jsonl)
#   NUM_EXAMPLES         only first N rows (handy for debug)
#   MAX_CONCURRENCY      default: pilot=4 / score=16 / filter=16
#   MAX_STEPS            agent loop step cap   (default: 10)
#   MAX_COMPLETION_TOKENS default: 8000 (GPT-5 includes reasoning tokens!)
#   REASONING_EFFORT     low | medium | high   (default: high)
#   TOOL_BACKEND         internal | mock        (default: internal)
#   MIN_REWARD_PER_TYPE  used only by PROFILE=filter
#       default: search=0.3,understanding=0.5,write_section=0.4,write_survey=0.4
#   OVERWRITE=1          truncate OUTPUT_JSONL before running

set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

export PATH="/home/tiger/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
DISTILL_SCRIPT="${REPO_ROOT}/scripts/data/scholar/distill_sft_trajectories.py"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[distill] ERROR: ${PYTHON_BIN} not found. Run: uv sync" >&2
    exit 1
fi

if [ -z "${AK_LIST:-}" ]; then
    echo "[distill] ERROR: AK_LIST is required (comma-separated Azure API keys)." >&2
    exit 1
fi

PROFILE="${PROFILE:=score}"
RLVR_DIR="${RLVR_DIR:=${REPO_ROOT}/data/scholar_en_rlvr}"
TOOL_BACKEND="${TOOL_BACKEND:=internal}"
MAX_STEPS="${MAX_STEPS:=10}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:=8000}"
REASONING_EFFORT="${REASONING_EFFORT:=high}"

# Default model_list to the same gpt-5-2025-08-07 for every AK in AK_LIST.
if [ -z "${MODEL_LIST:-}" ]; then
    NUM_AKS=$(awk -F, '{print NF}' <<< "${AK_LIST}")
    MODEL_LIST=$(printf 'gpt-5-2025-08-07%.0s,' $(seq 1 "${NUM_AKS}"))
    MODEL_LIST="${MODEL_LIST%,}"
fi

case "${PROFILE}" in
    pilot)
        DEFAULT_INPUTS=("${RLVR_DIR}/search_data.2023.parquet")
        DEFAULT_OUT="${REPO_ROOT}/data/scholar/sft_distill_pilot.jsonl"
        DEFAULT_NUM_EXAMPLES="${NUM_EXAMPLES:=20}"
        DEFAULT_CONC="${MAX_CONCURRENCY:=4}"
        EXTRA=()
        ;;
    score)
        DEFAULT_INPUTS=(
            "${RLVR_DIR}/search_data.2023.parquet"
            "${RLVR_DIR}/understanding_data.2023.parquet"
            "${RLVR_DIR}/understanding_data.2023_new.parquet"
            "${RLVR_DIR}/write_section_data.2023.parquet"
            "${RLVR_DIR}/write_survey_data.2023.parquet"
        )
        DEFAULT_OUT="${REPO_ROOT}/data/scholar/sft_distill_score.jsonl"
        DEFAULT_NUM_EXAMPLES="${NUM_EXAMPLES:-}"
        DEFAULT_CONC="${MAX_CONCURRENCY:=16}"
        EXTRA=(--always-score)
        ;;
    filter)
        DEFAULT_INPUTS=(
            "${RLVR_DIR}/search_data.2023.parquet"
            "${RLVR_DIR}/understanding_data.2023.parquet"
            "${RLVR_DIR}/understanding_data.2023_new.parquet"
            "${RLVR_DIR}/write_section_data.2023.parquet"
            "${RLVR_DIR}/write_survey_data.2023.parquet"
        )
        DEFAULT_OUT="${REPO_ROOT}/data/scholar/sft_distill_filter.jsonl"
        DEFAULT_NUM_EXAMPLES="${NUM_EXAMPLES:-}"
        DEFAULT_CONC="${MAX_CONCURRENCY:=16}"
        MIN_REWARD_PER_TYPE="${MIN_REWARD_PER_TYPE:=search=0.3,understanding=0.5,write_section=0.4,write_survey=0.4}"
        EXTRA=(--min-reward-per-type "${MIN_REWARD_PER_TYPE}")
        ;;
    *)
        echo "[distill] ERROR: unknown PROFILE=${PROFILE} (expected: pilot|score|filter)" >&2
        exit 1
        ;;
esac

OUTPUT_JSONL="${OUTPUT_JSONL:=${DEFAULT_OUT}}"
mkdir -p "$(dirname "${OUTPUT_JSONL}")"

if [ -n "${INPUTS:-}" ]; then
    # Allow caller to override with a space-separated INPUTS list.
    # shellcheck disable=SC2206
    INPUT_FILES=(${INPUTS})
else
    INPUT_FILES=("${DEFAULT_INPUTS[@]}")
fi

# Sanity-check that inputs exist.
for f in "${INPUT_FILES[@]}"; do
    if [ ! -f "${f}" ]; then
        echo "[distill] ERROR: input file missing: ${f}" >&2
        echo "[distill]        Run scripts/data/scholar/prepare_scholar_data.py first." >&2
        exit 1
    fi
done

cmd=(
    "${PYTHON_BIN}" "${DISTILL_SCRIPT}"
    --input "${INPUT_FILES[@]}"
    --output-jsonl "${OUTPUT_JSONL}"
    --ak-list "${AK_LIST}"
    --model-list "${MODEL_LIST}"
    --tool-backend "${TOOL_BACKEND}"
    --max-steps "${MAX_STEPS}"
    --max-completion-tokens "${MAX_COMPLETION_TOKENS}"
    --reasoning-effort "${REASONING_EFFORT}"
    --max-concurrency "${DEFAULT_CONC}"
)
if [ -n "${DEFAULT_NUM_EXAMPLES}" ]; then
    cmd+=(--num-examples "${DEFAULT_NUM_EXAMPLES}")
fi
if [ -n "${MIN_REWARD:-}" ]; then
    cmd+=(--min-reward "${MIN_REWARD}")
fi
if [ -n "${MIN_TOOL_CALLS:-}" ]; then
    cmd+=(--min-tool-calls "${MIN_TOOL_CALLS}")
fi
if [ "${REQUIRE_ANSWER:-1}" = "0" ]; then
    cmd+=(--no-require-answer)
fi
if [ "${OVERWRITE:-0}" = "1" ]; then
    cmd+=(--overwrite)
fi
cmd+=("${EXTRA[@]}")

echo "[distill] PROFILE=${PROFILE}"
echo "[distill] INPUT_FILES=(${#INPUT_FILES[@]} files)"
echo "[distill] OUTPUT_JSONL=${OUTPUT_JSONL}"
echo "[distill] AK_COUNT=$(awk -F, '{print NF}' <<< "${AK_LIST}")  MAX_CONCURRENCY=${DEFAULT_CONC}"
exec "${cmd[@]}"
