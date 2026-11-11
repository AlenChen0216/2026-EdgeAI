"""Evaluate a model and report AdaptQuant effective bit rates."""

import argparse
import contextlib
import json
import sys

import torch

from adaptquant import (bit_rate_report, format_report, quantize_model,
                        reset_activation_statistics, weight_storage_report)
from evaluate import DEFAULT_MODEL, eval_ppl_wikitext2, load_model
from thresholds import load_thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", help="frozen reference threshold JSON; required unless --baseline")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metrics-json")
    args = parser.parse_args()
    if args.model != DEFAULT_MODEL:
        parser.error("Part3 target must be Qwen/Qwen3-4B")
    if args.baseline and args.thresholds:
        parser.error("--baseline and --thresholds cannot be combined")
    if not args.baseline and not args.thresholds:
        parser.error("--thresholds is required for AdaptQuant simulation")
    source = None
    storage = None
    bit_rate = None
    bit_rate_error = None
    if not args.baseline:
        thresholds, source = load_thresholds(args.thresholds)
    with contextlib.redirect_stdout(sys.stderr):
        torch.manual_seed(args.seed)
        model, tokenizer = load_model(args.model, args.dtype)
        if not args.baseline:
            model = quantize_model(model, thresholds)
            storage = weight_storage_report(model)
            reset_activation_statistics(model)
        # Activation assignments are measured during the unchanged evaluation.
        # Weight bits alone cannot determine whether the combined budget passes.
        ppl = eval_ppl_wikitext2(model, tokenizer)
        if not args.baseline:
            try:
                bit_rate = bit_rate_report(model)
            except ValueError as exc:
                bit_rate_error = str(exc)
    if args.metrics_json:
        metrics = vars(args) | {"ppl": ppl, "threshold_source": source,
                               "format_counts": format_report(model), "weight_storage": storage,
                               "bit_rate": bit_rate, "bit_rate_error": bit_rate_error,
                               "torch": torch.__version__, "cuda": torch.version.cuda}
        with open(args.metrics_json, "w") as output:
            json.dump(metrics, output, indent=2)
    print(f"PPL: {ppl:.4f}", flush=True)
    if bit_rate_error is not None:
        raise SystemExit(f"Cannot verify realized effective bit rate: {bit_rate_error}")
    if bit_rate is not None:
        if not bit_rate["passes_limit"]:
            raise SystemExit("Realized effective bit rate exceeds the 5.3-bit budget.")


if __name__ == "__main__":
    main()
