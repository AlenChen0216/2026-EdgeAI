"""Derive frozen quantization thresholds from a reference model."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from thresholds import derive_thresholds


def main():
    parser = argparse.ArgumentParser(description="One-time AdaptQuant reference derivation")
    parser.add_argument("--token-ids", help="2048 reference token IDs; default: WikiText-2 train")
    parser.add_argument("--reference-path", help="optional local Llama-3.1-8B model/tokenizer directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-bits", type=float, default=5.3,
                        help="reference mean operand bit target including E8M0 scales")
    parser.add_argument("--token-offset", type=int, default=0,
                        help="start token in WikiText-2 train when --token-ids is omitted")
    parser.add_argument("--corpus", default="provided-token-ids", help="source of a custom token file")
    parser.add_argument("--split", default="unspecified", help="split of a custom token file")
    args = parser.parse_args()
    reference_path = args.reference_path or "meta-llama/Llama-3.1-8B"
    if args.token_ids:
        ids = json.loads(Path(args.token_ids).read_text())
        token_source = {"corpus": args.corpus, "split": args.split,
                        "token_file": args.token_ids}
    else:
        from datasets import load_dataset
        tokenizer = AutoTokenizer.from_pretrained(reference_path)
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tokens = tokenizer("\n\n".join(dataset["text"]), add_special_tokens=False).input_ids
        ids = tokens[args.token_offset:args.token_offset+2048]
        token_source = {"corpus": "wikitext/wikitext-2-raw-v1", "split": "train",
                        "dataset_fingerprint": dataset._fingerprint,
                        "token_offset": args.token_offset, "add_special_tokens": False}
    model = AutoModelForCausalLM.from_pretrained(
        reference_path, dtype=torch.bfloat16,
        device_map="cuda")
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    token_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        document = derive_thresholds(model, token_ids, target_bits=args.target_bits)
    source = document.setdefault("source", {})
    source.update(token_source)
    # Hash canonical JSON bytes of the token list, independent of file whitespace.
    source["sequence_sha256"] = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    source["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("adaptquant.py", "mx_formats.py", "tiles.py", "thresholds.py", "derive_thresholds.py")}
    Path(args.output).with_suffix(".tokens.json").write_text(json.dumps(ids))
    with open(args.output, "w") as output:
        json.dump(document, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
