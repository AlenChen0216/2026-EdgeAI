#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
# Default is the grading environment. Local 5090 development uses its existing
# CUDA 12.8 environment with SKIP_INSTALL=1 (see README).
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    "$PYTHON" -m pip install -r requirements.txt >&2
fi
mkdir -p results
# Run compression and official PPL evaluation only, without development checks.
targets=(0.75 0.81 0.50 0.85)
# Case4 uses joint SparseGPT pruning/quantization with its own G64 default.
quantizers=("${CASE1_QUANTIZER:-gptq}" "${CASE2_QUANTIZER:-gptq}" rtn sparsegpt)
for case_id in 1 2 3 4; do
    "$PYTHON" run_eval.py --case "$case_id" --rate "${targets[$((case_id - 1))]}" \
        --quantizer "${quantizers[$((case_id - 1))]}" \
        --metrics-json "results/case${case_id}.json" \
        --submission-output "$@"
done
