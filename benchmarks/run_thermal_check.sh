#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-3B-Instruct"
HOST="127.0.0.1"
PORT="8000"
RESULTS_DIR="benchmarks/results"
TEMP_LOG="benchmarks/analysis/temp_log.csv"

mkdir -p benchmarks/analysis

# Start GPU telemetry logging in the background, 1 sample/sec, timestamped
# so we can align it against each run's actual wall-clock window afterward.
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw \
  --format=csv -l 1 > "${TEMP_LOG}" &
SMI_PID=$!
echo "Started nvidia-smi logging (PID ${SMI_PID}) -> ${TEMP_LOG}"

# Make sure we stop the logger even if the benchmark fails or the script
# is interrupted -- an orphaned nvidia-smi loop is an easy thing to forget
# about and leave running.
cleanup() {
  echo "Stopping nvidia-smi logging (PID ${SMI_PID})"
  kill "${SMI_PID}" 2>/dev/null || true
}
trap cleanup EXIT

SERVE_CMD="vllm serve ${MODEL} --gpu-memory-utilization 0.85 --max-model-len 8192 --host ${HOST} --port ${PORT}"
BENCH_CMD="vllm bench serve --model ${MODEL} --backend vllm --endpoint /v1/completions --dataset-name random --host ${HOST} --port ${PORT} --num-warmups 5"

echo "=== Thermal check: output-length 2048 and 4096, 6 repeats each ==="
vllm bench sweep serve \
  --serve-cmd "${SERVE_CMD}" \
  --bench-cmd "${BENCH_CMD}" \
  --bench-params benchmarks/thermal_check.json \
  --output-dir "${RESULTS_DIR}" \
  --experiment-name thermal_check \
  --num-runs 6

echo "Done. Temperature log: ${TEMP_LOG}"
echo "Results: ${RESULTS_DIR}/thermal_check/"