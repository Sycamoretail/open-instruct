#!/bin/bash
# Serve a Scholar checkpoint as an OpenAI-compatible vLLM endpoint.
#
# Used by scripts/data/scholar/get_response.sh (DRB / DR-Tulu rollout) and
# any other code that talks to --api-base http://127.0.0.1:${PORT}/v1.
#
# Usage:
#   ./scripts/serve/scholar/serve_vllm.sh
#   MODEL_NAME_OR_PATH=outputs/qwen3_8b_scholar_sft \
#       SERVED_MODEL_NAME=scholar-8b-sft \
#       PORT=30001 \
#       ./scripts/serve/scholar/serve_vllm.sh
#
# Env overrides:
#   MODEL_NAME_OR_PATH    (default: /mlx_devbox/.../models/Qwen3-8B)
#   SERVED_MODEL_NAME     (default: derived from MODEL_NAME_OR_PATH basename)
#   HOST                  (default: 127.0.0.1)
#   PORT                  (default: 30001)
#   TENSOR_PARALLEL_SIZE  (default: 1)
#   GPU_MEMORY_UTIL       (default: 0.9)
#   MAX_MODEL_LEN         (default: 16384, matches SFT/RL pack length)
#   DTYPE                 (default: bfloat16)
#   API_KEY               (optional: enables Bearer auth on /v1)
#   GPU_VISIBLE           (optional: sets CUDA_VISIBLE_DEVICES, e.g. "0")
#   EXTRA_VLLM_ARGS       (optional: extra raw flags forwarded to vllm serve)

set -e

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

export PATH="/home/tiger/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv"
PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[serve-vllm] ERROR: ${PYTHON_BIN} not found. Run: uv sync" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import vllm" >/dev/null 2>&1; then
    echo "[serve-vllm] ERROR: vllm is not installed in ${UV_PROJECT_ENVIRONMENT}." >&2
    echo "[serve-vllm]        Install with: uv sync (or uv pip install vllm)." >&2
    exit 1
fi

# --------------------------------------------------
# Config
# --------------------------------------------------
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:=/mlx_devbox/users/luoyunze/playground/open-instruct/models/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:=$(basename "${MODEL_NAME_OR_PATH}")}"
HOST="${HOST:=127.0.0.1}"
PORT="${PORT:=30001}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:=1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:=0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:=16384}"
DTYPE="${DTYPE:=bfloat16}"

if [ -n "${GPU_VISIBLE:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_VISIBLE}"
fi

LOG_DIR="${LOG_DIR:=${REPO_ROOT}/outputs/serve_vllm}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SERVED_MODEL_NAME}.port${PORT}.log"

echo "[serve-vllm] MODEL=${MODEL_NAME_OR_PATH}"
echo "[serve-vllm] SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "[serve-vllm] LISTEN=${HOST}:${PORT}  TP=${TENSOR_PARALLEL_SIZE}  MAX_LEN=${MAX_MODEL_LEN}"
echo "[serve-vllm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)}"
echo "[serve-vllm] Log -> ${LOG_FILE}"
echo

cmd=(
    "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
    --model "${MODEL_NAME_OR_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}"
    --max-model-len "${MAX_MODEL_LEN}"
    --dtype "${DTYPE}"
    --trust-remote-code
)
if [ -n "${API_KEY:-}" ]; then
    cmd+=(--api-key "${API_KEY}")
fi
if [ -n "${EXTRA_VLLM_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    cmd+=(${EXTRA_VLLM_ARGS})
fi

echo "[serve-vllm] Launching: ${cmd[*]}"
exec "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
