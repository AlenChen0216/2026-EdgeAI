#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    "$PYTHON" -m pip install -r requirements.txt >&2
fi
# Run quantization and official PPL evaluation only, without development checks.
exec "$PYTHON" run_eval.py "$@"
