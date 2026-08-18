#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-3B-Instruct"
HOST="127.0.0.1"
PORT="8000"
RESULTS_DIR="benchmarks/results"

# ---------------------------------------------------------------------------
# BEFORE RUNNING: confirm these flags exist on your installed vLLM version.
#   vllm bench sweep serve --help
# This was written against docs for a recent vLLM release; --num-runs,
# --bench-params, and --serve-cmd behavior may differ slightly on 0.26.0.
# If `vllm bench sweep serve` doesn't behave as expected, skip to the
# FALLBACK section at the bottom, which only uses flags confirmed directly
# from `vllm bench serve --help` output.
# ---------------------------------------------------------------------------

SERVE_CMD="vllm serve ${MODEL} --gpu-memory-utilization 0.85 --max-model-len 8192 --host ${HOST} --port ${PORT}"
BENCH_CMD="vllm bench serve --model ${MODEL} --backend vllm --endpoint /v1/completions --dataset-name random --host ${HOST} --port ${PORT} --num-warmups 5"

echo "=== Concurrency sweep ==="
vllm bench sweep serve \
  --serve-cmd "${SERVE_CMD}" \
  --bench-cmd "${BENCH_CMD}" \
  --bench-params config/concurrency_sweep.json \
  --output-dir "${RESULTS_DIR}" \
  --experiment-name concurrency_sweep \
  --num-runs 3

echo "=== Input length sweep ==="
vllm bench sweep serve \
  --serve-cmd "${SERVE_CMD}" \
  --bench-cmd "${BENCH_CMD}" \
  --bench-params config/input_length_sweep.json \
  --output-dir "${RESULTS_DIR}" \
  --experiment-name input_length_sweep \
  --num-runs 3

echo "=== Output length sweep ==="
vllm bench sweep serve \
  --serve-cmd "${SERVE_CMD}" \
  --bench-cmd "${BENCH_CMD}" \
  --bench-params config/output_length_sweep.json \
  --output-dir "${RESULTS_DIR}" \
  --experiment-name output_length_sweep \
  --num-runs 3

# ---------------------------------------------------------------------------
# FALLBACK — plain loop, only using flags confirmed from `vllm bench serve
# --help`: --max-concurrency, --random-input-len, --random-output-len,
# --num-prompts, --save-result, --result-dir, --result-filename, --metadata.
# Requires a `vllm serve` instance already running in another terminal.
# Uncomment and run manually if the native sweep tool doesn't cooperate.
# ---------------------------------------------------------------------------

# mkdir -p "${RESULTS_DIR}/concurrency_sweep"
# for row in $(jq -c '.[]' benchmarks/concurrency_sweep.json); do
#   c=$(echo "$row" | jq -r '."max-concurrency"')
#   in_len=$(echo "$row" | jq -r '."random-input-len"')
#   out_len=$(echo "$row" | jq -r '."random-output-len"')
#   n=$(echo "$row" | jq -r '."num-prompts"')
#   for run in 1 2 3; do
#     vllm bench serve \
#       --model "${MODEL}" --backend vllm --endpoint /v1/completions \
#       --dataset-name random --host "${HOST}" --port "${PORT}" \
#       --max-concurrency "${c}" --random-input-len "${in_len}" \
#       --random-output-len "${out_len}" --num-prompts "${n}" \
#       --num-warmups 5 \
#       --save-result --result-dir "${RESULTS_DIR}/concurrency_sweep" \
#       --result-filename "c${c}_i${in_len}_o${out_len}_run${run}.json" \
#       --metadata "sweep=concurrency" "run=${run}"
#   done
# done