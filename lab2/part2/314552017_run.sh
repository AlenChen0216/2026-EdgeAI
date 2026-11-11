#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
# The GPU runner selects one case per process; the submission default is unchanged.
read -r -a case_ids <<< "${CASE_IDS-1 2 3 4}"
if [[ ${#case_ids[@]} -eq 0 ]]; then
    echo "CASE_IDS must contain case numbers from 1 to 4" >&2
    exit 2
fi
for case_id in "${case_ids[@]}"; do
    if [[ ! "$case_id" =~ ^[1-4]$ ]]; then
        echo "Invalid CASE_IDS entry: $case_id" >&2
        exit 2
    fi
done
RESULTS_DIR="${RESULTS_DIR:-results}"
# Set SKIP_INSTALL=1 to use an existing environment.
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    "$PYTHON" -m pip install -r requirements.txt >&2
fi
mkdir -p -- "$RESULTS_DIR"
# Run each compression case and its PPL evaluation.
targets=(0.75 0.81 0.50 0.85)
# Case4 uses joint SparseGPT pruning/quantization with its own G64 default.
quantizers=("${CASE1_QUANTIZER:-gptq}" "${CASE2_QUANTIZER:-gptq}" rtn sparsegpt)
for case_id in "${case_ids[@]}"; do
    "$PYTHON" run_eval.py --case "$case_id" --rate "${targets[$((case_id - 1))]}" \
        --quantizer "${quantizers[$((case_id - 1))]}" \
        --metrics-json "$RESULTS_DIR/case${case_id}.json" \
        --submission-output "$@"
done
