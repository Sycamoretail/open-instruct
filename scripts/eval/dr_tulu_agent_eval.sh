#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DR_TULU_EVAL_DIR="${REPO_ROOT}/scripts/eval/dr_tulu"

export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"

usage() {
    cat <<'EOF'
Usage:
  scripts/eval/dr_tulu_agent_eval.sh <task> <results.jsonl>

Examples:
  scripts/eval/dr_tulu_agent_eval.sh simpleqa /abs/path/to/simpleqa.jsonl
  GRADER_MODEL=gpt-4.1-mini-2025-04-14 \
  scripts/eval/dr_tulu_agent_eval.sh researchqa /abs/path/to/researchqa.jsonl

Environment overrides:
  EVAL_SAVE_PATH=/abs/path/to/task_eval_results.json
  TASK_TYPE=
  GRADER_MODEL=gpt-4.1-2025-04-14
  RUN_MODE=auto_reason_search
  DEBUG_EVAL=false
EOF
}

if ! command -v uv >/dev/null 2>&1; then
    echo "[dr-tulu] uv is required but was not found in PATH." >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[dr-tulu] Expected Python at ${PYTHON_BIN}, but it does not exist." >&2
    exit 1
fi

TASK="${1:-${TASK:-}}"
RESULTS_FILE="${2:-${RESULTS_FILE:-}}"
EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-}"
TASK_TYPE="${TASK_TYPE:-}"
GRADER_MODEL="${GRADER_MODEL:-gpt-4.1-2025-04-14}"
RUN_MODE="${RUN_MODE:-auto_reason_search}"
DEBUG_EVAL="${DEBUG_EVAL:-false}"

if [[ -z "${TASK}" || -z "${RESULTS_FILE}" ]]; then
    usage >&2
    exit 1
fi

if [[ ! -f "${RESULTS_FILE}" ]]; then
    echo "[dr-tulu] Results file not found: ${RESULTS_FILE}" >&2
    exit 1
fi

echo "[dr-tulu] Syncing evaluation dependencies into ${UV_PROJECT_ENVIRONMENT}..."
(
    cd "${REPO_ROOT}"
    uv sync --extra dr-tulu
)

eval_cmd=(
    "${PYTHON_BIN}" "${DR_TULU_EVAL_DIR}/evaluate.py" "${TASK}" "${RESULTS_FILE}"
    "--grader-model" "${GRADER_MODEL}"
    "--run_mode" "${RUN_MODE}"
)
if [[ -n "${EVAL_SAVE_PATH}" ]]; then
    eval_cmd+=("--save_path" "${EVAL_SAVE_PATH}")
fi
if [[ -n "${TASK_TYPE}" ]]; then
    eval_cmd+=("--task_type" "${TASK_TYPE}")
fi
if [[ "${DEBUG_EVAL}" == "true" ]]; then
    eval_cmd+=("--debug")
fi

echo "[dr-tulu] Evaluating ${RESULTS_FILE}..."
"${eval_cmd[@]}"

echo "[dr-tulu] Done."
