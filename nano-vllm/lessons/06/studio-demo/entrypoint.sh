#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
MINISGL_PORT="${MINISGL_PORT:-1919}"
MEMORY_RATIO="${MEMORY_RATIO:-0.70}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_PREFILL_LENGTH="${MAX_PREFILL_LENGTH:-2048}"
PAGE_SIZE="${PAGE_SIZE:-16}"

mkdir -p \
  /mnt/workspace/.cache/modelscope \
  /mnt/workspace/.cache/flashinfer \
  /mnt/workspace/.cache/tvm-ffi

export MODELSCOPE_CACHE="/mnt/workspace/.cache/modelscope"
export FLASHINFER_WORKSPACE_BASE="/mnt/workspace/.cache/flashinfer"
export TVM_FFI_CACHE_DIR="/mnt/workspace/.cache/tvm-ffi"
export MINISGL_API_BASE="http://127.0.0.1:${MINISGL_PORT}/v1"

python -m minisgl \
  --model "${MODEL_ID}" \
  --model-source modelscope \
  --host 127.0.0.1 \
  --port "${MINISGL_PORT}" \
  --cache-type radix \
  --memory-ratio "${MEMORY_RATIO}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --max-seq-len-override "${MAX_SEQ_LEN}" \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --max-prefill-length "${MAX_PREFILL_LENGTH}" \
  --page-size "${PAGE_SIZE}" &
backend_pid=$!

cleanup() {
  if kill -0 "${backend_pid}" 2>/dev/null; then
    kill "${backend_pid}"
    wait "${backend_pid}" || true
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if ! kill -0 "${backend_pid}" 2>/dev/null; then
    echo "mini-SGLang exited before becoming ready" >&2
    wait "${backend_pid}"
  fi
  if curl --fail --silent "http://127.0.0.1:${MINISGL_PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl --fail --silent "http://127.0.0.1:${MINISGL_PORT}/v1/models" >/dev/null; then
  echo "mini-SGLang readiness check timed out" >&2
  exit 1
fi

python /app/app.py
