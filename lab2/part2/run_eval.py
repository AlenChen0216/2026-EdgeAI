import argparse
import contextlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import DEFAULT_MODEL, compression_rate, eval_ppl_wikitext2, load_model  # noqa: E402
from compress import compress_model  # noqa: E402
from common import storage_report  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=None, help="omit for the base model")
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--wbits", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None,
                    help="default: Case2=512, Case4=64, otherwise 128")
    ap.add_argument("--svd-rate", type=float, default=0.21,
                    help="Case3 SVD-only storage reduction before factor quantization")
    ap.add_argument("--svd-method", default="whiten", help="Case3: whiten (default) or plain baseline")
    ap.add_argument("--svd-samples", type=int, default=128)
    ap.add_argument("--svd-seqlen", type=int, default=2048, help="train calibration window; official PPL length is unchanged")
    ap.add_argument("--svd-damp", type=float, default=1e-4)
    ap.add_argument("--quantizer", default=None,
                    help="Case2 defaults to gptq, Case1 to rtn; Case4 uses sparsegpt")
    ap.add_argument("--qparam-method", default=None,
                    help="default: mse for RTN, minmax for GPTQ; both preserve packed storage")
    ap.add_argument("--mse-grid-size", type=int, default=21)
    ap.add_argument("--mse-min-ratio", type=float, default=0.5)
    ap.add_argument("--gptq-samples", type=int, default=128)
    ap.add_argument("--gptq-seqlen", type=int, default=2048)
    ap.add_argument("--gptq-damp", type=float, default=0.01)
    ap.add_argument("--gptq-block-size", type=int, default=128)
    ap.add_argument("--metrics-json", help="optional detailed byte ledger and experiment metadata")
    ap.add_argument("--submission-output", action="store_true",
                    help="print only the Case, compression ratio and PPL required by run.sh")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.group_size is None:
        args.group_size = {2: 512, 4: 64}.get(args.case, 128)
    args.quantizer = ("sparsegpt" if args.case == 4 else
                      (args.quantizer or ("gptq" if args.case == 2 else "rtn")))
    if args.qparam_method is None:
        args.qparam_method = "minmax" if args.quantizer in ("gptq", "sparsegpt") else "mse"
    gptq_config = None
    svd_config = None
    if args.case == 3 and args.svd_method == "whiten":
        from svd_whitening import WhiteningConfig
        svd_config = WhiteningConfig(args.svd_samples, args.svd_seqlen, args.svd_damp, args.seed)
    if (args.quantizer == "gptq" and args.case in (1, 2)) or args.case == 4:
        from gptq import GPTQConfig
        gptq_config = GPTQConfig(args.gptq_samples, args.gptq_seqlen,
                                 args.gptq_damp, args.gptq_block_size, args.seed)
    if args.case is not None and args.rate is None:
        args.rate = {1: 0.75, 2: 0.81, 3: 0.5, 4: 0.85}[args.case]
    with contextlib.redirect_stdout(sys.stderr):
        torch.manual_seed(args.seed)
        model, tok = load_model(args.model, args.dtype)
        if args.case is not None:
            model, tok = compress_model(model, tok, args.case, args.rate,
                                       args.wbits, args.group_size,
                                       qparam_method=args.qparam_method,
                                       mse_grid_size=args.mse_grid_size,
                                       mse_min_ratio=args.mse_min_ratio,
                                       quantizer=args.quantizer, gptq_config=gptq_config,
                                       svd_rate=args.svd_rate, svd_method=args.svd_method,
                                       svd_config=svd_config)
        ppl = eval_ppl_wikitext2(model, tok)
        rate = compression_rate(model)
        report = storage_report(model) if args.case is not None else None

    if args.metrics_json:
        metrics = vars(args) | {"ppl": ppl, "compression_ratio": rate,
                               "torch": torch.__version__,
                               "cuda": torch.version.cuda}
        if report is not None:
            metrics["storage"] = report
        if hasattr(model, "_gptq_metadata"):
            metrics["gptq"] = model._gptq_metadata
        if hasattr(model, "_case3_metadata"):
            metrics["case3_details"] = model._case3_metadata
        with open(args.metrics_json, "w") as output:
            json.dump(metrics, output, indent=2)

    print(f"Case {args.case}" if args.case is not None else "Base")
    print(f"Compression Ratio: {rate:.6%}")
    if not args.submission_output:
        svd = None if report is None else report["svd"]
        pruning = None if report is None else report["pruning"]
        svd_text = "N/A" if svd is None else f"{svd['compression_ratio']:.6%}"
        pruning_text = "N/A" if pruning is None else f"{pruning['rate']:.6%}"
        print(f"SVD Compression Ratio: {svd_text}")
        print(f"Pruning Rate: {pruning_text}")
        if pruning is not None:
            print(f"Pruned Elements: {pruning['element_fraction']:.6%}")
    print(f"PPL: {ppl:.4f}", flush=True)

if __name__ == "__main__":
    main()
