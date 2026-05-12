#!/bin/bash
# Generate model responses against a vLLM endpoint, in the same multi-turn
# tool-use protocol the model was trained on.
#
# Wraps scripts/eval/dr_tulu/generate_responses.py.
#
# Usage:
#   ./scripts/data/scholar/get_response.sh deep_research_bench
#   ./scripts/data/scholar/get_response.sh simpleqa
#
# Common env overrides:
#   DATASET             positional arg or env (default: deep_research_bench)
#   MODEL               served-model-name on the vLLM endpoint
#                       (default: scholar-8b-sft)
#   API_BASE            vLLM endpoint base URL  (default: http://127.0.0.1:30001/v1)
#   API_KEY             optional Bearer token
#   OUTPUT_FILE         absolute output JSONL path
#                       (default: eval_output/scholar/${MODEL}_${DATASET}.jsonl)
#   AGENT_MODE          internal | none      (default: internal)
#   MAX_TURNS           agent loop turn cap  (default: 6)
#   MAX_TOKENS          per-call token cap   (default: 4096)
#   MAX_TOOL_RESPONSE_CHARS
#                       max raw tool-response chars fed back to model
#                       (default: 12000; <=0 disables truncation)
#   TEMPERATURE         sampling temperature (default: 0.2)
#   FORCE_ANSWER_AFTER_MAX_TURNS
#                       after MAX_TURNS, force one final no-tool answer call
#                       (default: 1)
#   FORCE_ANSWER_PROMPT optional custom prompt for the final no-tool answer call
#   NUM_EXAMPLES        N | ablation | final_run | final_run_100
#   MAX_CONCURRENCY     thread pool size     (default: 4)
#   OVERWRITE=1         re-create OUTPUT_FILE from scratch

set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

export PATH="/home/tiger/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
GENERATE_SCRIPT="${REPO_ROOT}/scripts/eval/dr_tulu/generate_responses.py"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[get-response] ERROR: ${PYTHON_BIN} not found. Run: uv sync" >&2
    exit 1
fi
if [ ! -f "${GENERATE_SCRIPT}" ]; then
    echo "[get-response] ERROR: ${GENERATE_SCRIPT} missing." >&2
    exit 1
fi

DATASET="${1:-${DATASET:-deep_research_bench}}"
MODEL="${MODEL:=scholar-8b-sft}"
API_BASE="${API_BASE:=http://127.0.0.1:30001/v1}"
AGENT_MODE="${AGENT_MODE:=internal}"
MAX_TURNS="${MAX_TURNS:=6}"
MAX_TOKENS="${MAX_TOKENS:=4096}"
MAX_TOOL_RESPONSE_CHARS="${MAX_TOOL_RESPONSE_CHARS:=12000}"
TEMPERATURE="${TEMPERATURE:=0.2}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:=4}"
TIMEOUT_S="${TIMEOUT_S:=3600}"
FORCE_ANSWER_AFTER_MAX_TURNS="${FORCE_ANSWER_AFTER_MAX_TURNS:=1}"

DEFAULT_OUT="${REPO_ROOT}/eval_output/scholar/${MODEL}_${DATASET}.jsonl"
OUTPUT_FILE="${OUTPUT_FILE:=${DEFAULT_OUT}}"
mkdir -p "$(dirname "${OUTPUT_FILE}")"

if ! curl --silent --fail --max-time 5 "${API_BASE%/v1}/v1/models" >/dev/null 2>&1 \
    && ! curl --silent --fail --max-time 5 "${API_BASE}/models" >/dev/null 2>&1; then
    echo "[get-response] WARNING: ${API_BASE} not reachable. Start vLLM first:"
    echo "[get-response]   ./scripts/serve/scholar/serve_vllm.sh"
fi

cmd=(
    "${PYTHON_BIN}" "${GENERATE_SCRIPT}"
    --dataset "${DATASET}"
    --model "${MODEL}"
    --api-base "${API_BASE}"
    --agent-mode "${AGENT_MODE}"
    --max-turns "${MAX_TURNS}"
    --max-tokens "${MAX_TOKENS}"
    --max-tool-response-chars "${MAX_TOOL_RESPONSE_CHARS}"
    --temperature "${TEMPERATURE}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --timeout-s "${TIMEOUT_S}"
    --output-file "${OUTPUT_FILE}"
)
if [ -n "${API_KEY:-}" ]; then
    cmd+=(--api-key "${API_KEY}")
fi
if [ -n "${NUM_EXAMPLES:-}" ]; then
    cmd+=(--num-examples "${NUM_EXAMPLES}")
fi
if [ -n "${SUBSET:-}" ]; then
    cmd+=(--subset "${SUBSET}")
fi
if [ -n "${TOP_P:-}" ]; then
    cmd+=(--top-p "${TOP_P}")
fi
if [ "${OVERWRITE:-0}" = "1" ]; then
    cmd+=(--overwrite)
fi
if [ "${FORCE_ANSWER_AFTER_MAX_TURNS}" = "0" ]; then
    cmd+=(--no-force-answer-after-max-turns)
else
    cmd+=(--force-answer-after-max-turns)
fi
if [ -n "${FORCE_ANSWER_PROMPT:-}" ]; then
    cmd+=(--force-answer-prompt "${FORCE_ANSWER_PROMPT}")
fi

echo "[get-response] DATASET=${DATASET}  MODEL=${MODEL}"
echo "[get-response] API_BASE=${API_BASE}  AGENT_MODE=${AGENT_MODE}"
echo "[get-response] MAX_TURNS=${MAX_TURNS}  MAX_TOOL_RESPONSE_CHARS=${MAX_TOOL_RESPONSE_CHARS}  FORCE_ANSWER_AFTER_MAX_TURNS=${FORCE_ANSWER_AFTER_MAX_TURNS}"
echo "[get-response] OUTPUT_FILE=${OUTPUT_FILE}"
echo "[get-response] Launching: ${cmd[*]}"
exec "${cmd[@]}"
