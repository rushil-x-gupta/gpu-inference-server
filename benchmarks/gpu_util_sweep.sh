#!/usr/bin/env bash
set -euo pipefail

# Anchor every path to this script's own location (benchmarks/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL="Qwen/Qwen2.5-3B-Instruct"
HOST="127.0.0.1"
PORT="8000"
RESULTS_DIR="../results"
GPU_LOG="gpu_util_log.csv"

if [[ ! -f "config/concurrency_sweep.json" ]]; then
  echo "ERROR: expected config file not found: ${SCRIPT_DIR}/config/concurrency_sweep.json" >&2
  exit 1
fi
echo "Preflight OK: config file found."

cd analysis

# utilization.gpu = % time SM (compute) was busy over the sample window
# utilization.memory = % time memory controller was busy over the sample window
# These two together are the direct compute-bound vs memory-bound signal --
# if utilization.gpu saturates near 100% while utilization.memory has
# headroom, that config is compute-bound; if it's the reverse, memory-bound.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw \
  --format=csv -l 1 > "${GPU_LOG}" &
SMI_PID=$!
echo "Started nvidia-smi logging (PID ${SMI_PID}) -> ${GPU_LOG}"

cleanup() {
  echo "Stopping nvidia-smi logging (PID ${SMI_PID})"
  kill "${SMI_PID}" 2>/dev/null || true
}
trap cleanup EXIT

SERVE_CMD="vllm serve ${MODEL} --gpu-memory-utilization 0.85 --max-model-len 8192 --host ${HOST} --port ${PORT}"
BENCH_CMD="vllm bench serve --model ${MODEL} --backend vllm --endpoint /v1/completions --dataset-name random --host ${HOST} --port ${PORT} --num-warmups 5"

# Separate experiment name from the original concurrency_sweep -- keeps this
# run's data isolated rather than overwriting the already-analyzed results,
# and as a side benefit gives a rough reproducibility check: if this run's
# throughput numbers land far from the original concurrency_sweep numbers
# for the same configs, that's worth investigating before trusting either.
echo "=== Concurrency sweep with GPU telemetry ==="
vllm bench sweep serve \
  --serve-cmd "${SERVE_CMD}" \
  --bench-cmd "${BENCH_CMD}" \
  --bench-params ../config/concurrency_sweep.json \
  --output-dir "${RESULTS_DIR}" \
  --experiment-name gpu_util_sweep \
  --num-runs 3

echo "Done. GPU telemetry log: ${GPU_LOG}"
echo "Results: ${RESULTS_DIR}/gpu_util_sweep/"