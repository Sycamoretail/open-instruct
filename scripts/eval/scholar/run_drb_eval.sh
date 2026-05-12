#!/bin/bash
# Run Deep Research Bench (DRB) evaluation: RACE (article quality)
# + FACT (citation verification).
# Wraps scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/run_eval.py.
#
# Usage:
#   ./scripts/eval/scholar/run_drb_eval.sh /abs/path/to/drb_output.jsonl my_run_name
#   INPUT_FILE=... TASK_NAME=... ./scripts/eval/scholar/run_drb_eval.sh
#
# Common env overrides:
#   INPUT_FILE          DRB-format input JSONL (positional arg #1)
#   TASK_NAME           run id (positional arg #2; default: derived from input)
#   OUTPUT_DIR          where to write race/, fact/ subdirs
#                       (default: eval_output/dr_tulu/drb_eval_${TASK_NAME})
#   MAX_WORKERS         judge concurrency       (default: 10)
#   LIMIT               only first N records    (default: all)
#   ONLY_EN=1           English subset only
#   ONLY_ZH=1           Chinese subset only
#   SKIP_RACE=1         skip RACE
#   SKIP_FACT=1         skip FACT
#   FORCE=1             ignore previously-cached results
#
#   Backend selection (mutually exclusive):
#     LLM_BACKEND       openai_compatible (default) | internal_http | gemini
#
#   For LLM_BACKEND=openai_compatible:
#     LLM_API_BASE      e.g. https://api.openai.com/v1 or Azure deployment
#     LLM_API_KEY       Bearer token
#     RACE_MODEL        e.g. gpt-4.1-2025-04-14 or gpt-5-2025-08-07
#     FACT_MODEL        defaults to RACE_MODEL
#     CLEAN_MODEL       defaults to RACE_MODEL
#
#   For LLM_BACKEND=internal_http:
#     DRB_INTERNAL_JUDGE_URL=https://ivavmlgq.fn.bytedance.net  (script-default)
#     DRB_INTERNAL_PROXY=http://sys-proxy-rd-relay.byted.org:8118  (script-default)
#     RACE_MODEL / FACT_MODEL still required to pick the backend model name
#
#   For LLM_BACKEND=gemini:
#     GEMINI_API_KEY    required

set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

export PATH="/home/tiger/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
RUN_EVAL="${REPO_ROOT}/scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/run_eval.py"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[drb-eval] ERROR: ${PYTHON_BIN} not found. Run: uv sync (or uv sync --extra dr-tulu)" >&2
    exit 1
fi
if [ ! -f "${RUN_EVAL}" ]; then
    echo "[drb-eval] ERROR: ${RUN_EVAL} missing." >&2
    exit 1
fi

INPUT_FILE="${1:-${INPUT_FILE:-}}"
TASK_NAME="${2:-${TASK_NAME:-}}"

if [ -z "${INPUT_FILE}" ]; then
    echo "[drb-eval] ERROR: INPUT_FILE is required (positional arg or env)." >&2
    echo "          Example: ./scripts/eval/scholar/run_drb_eval.sh path/to/drb.jsonl my_model" >&2
    exit 1
fi
if [ ! -f "${INPUT_FILE}" ]; then
    echo "[drb-eval] ERROR: input file not found: ${INPUT_FILE}" >&2
    exit 1
fi

if [ -z "${TASK_NAME}" ]; then
    TASK_NAME="$(basename "${INPUT_FILE}" .jsonl)_eval"
fi

OUTPUT_DIR="${OUTPUT_DIR:=${REPO_ROOT}/eval_output/dr_tulu/drb_eval_${TASK_NAME}}"
mkdir -p "${OUTPUT_DIR}"

MAX_WORKERS="${MAX_WORKERS:=10}"
LLM_BACKEND="${LLM_BACKEND:=openai_compatible}"
export DRB_JUDGE_BACKEND="${LLM_BACKEND}"

# Backend-specific env wiring (mirror run_eval.py's expectations).
case "${LLM_BACKEND}" in
    openai_compatible)
        if [ -n "${LLM_API_BASE:-}" ]; then export DRB_JUDGE_API_BASE="${LLM_API_BASE}"; fi
        if [ -n "${LLM_API_KEY:-}" ]; then export DRB_JUDGE_API_KEY="${LLM_API_KEY}"; fi
        ;;
    internal_http)
        export DRB_INTERNAL_JUDGE_URL="${DRB_INTERNAL_JUDGE_URL:=https://ivavmlgq.fn.bytedance.net}"
        export DRB_INTERNAL_PROXY="${DRB_INTERNAL_PROXY:=http://sys-proxy-rd-relay.byted.org:8118}"
        ;;
    gemini)
        if [ -z "${GEMINI_API_KEY:-}" ]; then
            echo "[drb-eval] ERROR: LLM_BACKEND=gemini requires GEMINI_API_KEY." >&2
            exit 1
        fi
        ;;
    *)
        echo "[drb-eval] ERROR: unknown LLM_BACKEND=${LLM_BACKEND}." >&2
        exit 1
        ;;
esac

cmd=(
    "${PYTHON_BIN}" "${RUN_EVAL}"
    --input_file "${INPUT_FILE}"
    --task_name "${TASK_NAME}"
    --output_dir "${OUTPUT_DIR}"
    --max_workers "${MAX_WORKERS}"
    --llm_backend "${LLM_BACKEND}"
)
if [ -n "${LIMIT:-}" ]; then cmd+=(--limit "${LIMIT}"); fi
if [ "${ONLY_EN:-0}" = "1" ]; then cmd+=(--only_en); fi
if [ "${ONLY_ZH:-0}" = "1" ]; then cmd+=(--only_zh); fi
if [ "${SKIP_RACE:-0}" = "1" ]; then cmd+=(--skip_race); fi
if [ "${SKIP_FACT:-0}" = "1" ]; then cmd+=(--skip_fact); fi
if [ "${SKIP_CLEANING:-0}" = "1" ]; then cmd+=(--skip_cleaning); fi
if [ "${FORCE:-0}" = "1" ]; then cmd+=(--force); fi
if [ -n "${LLM_API_BASE:-}" ]; then cmd+=(--llm_api_base "${LLM_API_BASE}"); fi
if [ -n "${LLM_API_KEY:-}" ]; then cmd+=(--llm_api_key "${LLM_API_KEY}"); fi
if [ -n "${RACE_MODEL:-}" ]; then cmd+=(--race_model "${RACE_MODEL}"); fi
if [ -n "${FACT_MODEL:-}" ]; then cmd+=(--fact_model "${FACT_MODEL}"); fi
if [ -n "${CLEAN_MODEL:-}" ]; then cmd+=(--clean_model "${CLEAN_MODEL}"); fi

echo "[drb-eval] INPUT=${INPUT_FILE}"
echo "[drb-eval] TASK_NAME=${TASK_NAME}"
echo "[drb-eval] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[drb-eval] BACKEND=${LLM_BACKEND}  RACE_MODEL=${RACE_MODEL:-(default)}  FACT_MODEL=${FACT_MODEL:-(default)}"
exec "${cmd[@]}"
