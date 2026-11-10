"""Frozen reference-model thresholds. Never derive them from target-model PPL."""

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_thresholds(path):
    """Two operand tables, each with seven [t1,t2] pairs; 'CAP' means infinity.

    Paper Appendix B reports the derivation procedure but not all numeric cuts.
    Null/template entries remain unusable until real reference thresholds exist.
    """
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if document.get("frozen") is not True:
        raise NotImplementedError("P3-T1: generate frozen reference thresholds with derive_thresholds.py")
    source = document.get("source", {})
    for key in ("reference_model", "revision", "corpus", "split", "sequence_sha256", "procedure"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"threshold source must record {key}")
    if source["reference_model"] != "meta-llama/Llama-3.1-8B":
        raise ValueError("this reproduction expects the Appendix B reference model Llama-3.1-8B")
    if source.get("tokens") != 2048:
        raise ValueError("Appendix B uses a single 2048-token sequence")
    if len(source["sequence_sha256"]) != 64 or any(
        char not in "0123456789abcdef" for char in source["sequence_sha256"]
    ):
        raise ValueError("sequence_sha256 must identify the reference token sequence")
    result = {}
    for operand in ("weight", "activation"):
        entries = document.get(operand, {})
        if set(entries) != set(PROJECTIONS):
            raise ValueError(f"{operand} must contain exactly the seven projection types")
        result[operand] = {}
        for projection, pair in entries.items():
            if pair is None or (isinstance(pair, list) and None in pair):
                raise NotImplementedError(f"P3-T1: missing {operand}/{projection} reference thresholds")
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("each threshold entry must be [t1,t2]")
            converted = []
            for value in pair:
                if value == "CAP":
                    converted.append(math.inf)
                elif type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    converted.append(float(value))
                else:
                    raise ValueError("thresholds must be nonnegative finite numbers or CAP")
            if converted[0] > converted[1]:
                raise ValueError("thresholds require t1<=t2")
            result[operand][projection] = tuple(converted)
    return result, {"source": source, "sha256": hashlib.sha256(raw).hexdigest(),
                    "path": str(path)}


def nvfp4_quantize(tiles, valid, tensor_scale):
    """NVFP4 comparator: 16-value E2M1 blocks, E4M3 scales, FP32 tensor scale.

    tensor_scale is amax(full operand)/(6*448), fixed across tile chunks.
    E4M3 rounding uses the same portable element simulator as MXFP8.
    """
    from adaptquant import round_elements
    from mx_formats import FORMATS
    values = tiles.float().masked_fill(~valid, 0).reshape(-1, 16, 2, 16)
    keep = valid.reshape_as(values)
    maximum = values.abs().amax(dim=-1, keepdim=True)
    local_scale = maximum / (6 * tensor_scale)
    local_scale = round_elements(local_scale, torch.ones_like(local_scale, dtype=torch.bool), FORMATS[8])
    scale = local_scale * tensor_scale
    safe_scale = torch.where(scale > 0, scale, 1)
    result = round_elements(values / safe_scale, keep, FORMATS[4]) * scale
    return result.reshape_as(tiles)


@torch.no_grad()
def measure_histogram(matrix, channel_weights, tile_batch=256):
    """Accumulate counts and Eq.8 normalized errors on the fixed 0.02 grid."""
    from adaptquant import block_scales, round_elements, tile_statistic
    from mx_formats import FORMATS
    from tiles import to_tiles
    tiles, valid, layout = to_tiles(matrix)
    grid = torch.arange(226, device=matrix.device, dtype=torch.float32) / 50
    counts = torch.zeros(226, device=matrix.device, dtype=torch.int64)
    errors = torch.zeros(3, 226, device=matrix.device, dtype=torch.float64)
    reference = torch.zeros(226, device=matrix.device, dtype=torch.float64)
    weights = channel_weights.float()
    weights = weights / weights.mean().clamp_min(torch.finfo(torch.float32).tiny)
    weights = F.pad(weights, (0, (-layout.cols) % 32))
    tensor_scale = matrix.abs().amax().float() / (6 * 448)
    tensor_scale = torch.where(tensor_scale > 0, tensor_scale, 1)
    for start in range(0, len(tiles), tile_batch):
        values = tiles[start:start+tile_batch].float()
        keep = valid[start:start+tile_batch]
        values = values.masked_fill(~keep, 0)
        position = torch.bucketize(tile_statistic(values, keep), grid, right=True) - 1
        columns = ((torch.arange(start, start+len(values), device=matrix.device)
                    % layout.tile_cols)[:, None] * 32
                   + torch.arange(32, device=matrix.device))
        importance = weights[columns][:, None, :]
        energy = (values.square() * importance).sum(dim=(-2, -1))
        energy = energy.clamp_min(torch.finfo(torch.float32).tiny)
        counts.scatter_add_(0, position, torch.ones_like(position))
        for index, fmt in enumerate(FORMATS.values()):
            scale = block_scales(values, keep, fmt)
            quantized = round_elements(values / scale, keep, fmt) * scale
            error = ((values - quantized).square() * importance).sum(dim=(-2, -1)) / energy
            errors[index].scatter_add_(0, position, error.double())
        quantized = nvfp4_quantize(values, keep, tensor_scale)
        error = ((values - quantized).square() * importance).sum(dim=(-2, -1)) / energy
        reference.scatter_add_(0, position, error.double())
    return {"counts": counts.cpu(), "errors": errors.cpu(), "reference": reference.cpu()}


def candidate_table(histograms):
    """Score every ordered threshold pair once using exclusive prefix sums."""
    counts = torch.stack([h["counts"] for h in histograms]).sum(dim=0)
    errors = torch.stack([h["errors"] for h in histograms])
    count_prefix = F.pad(counts.cumsum(dim=-1), (1, 0))
    error_prefix = F.pad(errors.cumsum(dim=-1), (1, 0))
    first, second = torch.triu_indices(227, 227)
    n4 = count_prefix[first]
    n6 = count_prefix[second] - n4
    n8 = count_prefix[-1] - count_prefix[second]
    # Quarter-bit units: payload bits plus 8/32 scale overhead.
    cost = 17 * n4 + 25 * n6 + 33 * n8
    distortion = (error_prefix[:, 0, first]
                  + error_prefix[:, 1, second] - error_prefix[:, 1, first]
                  + error_prefix[:, 2, -1, None] - error_prefix[:, 2, second]).clamp_min(0)
    return {"first": first, "second": second, "cost": cost, "fp4": n4,
            "errors": distortion, "tiles": int(counts.sum()),
            "reference": torch.stack([h["reference"].sum() for h in histograms])}


def solve_projection(table, slack_db=0.0):
    """Minimum bits subject to every layer's NVFP4 bound; ties favor FP4."""
    reference = table["reference"]
    feasible = (table["errors"] <= reference[:, None] * 10**(slack_db / 10) + 1e-12).all(dim=0)
    reference_met = bool(feasible.any())
    if not reference_met:
        # Explicit fallback from Appendix B: minimize worst relative layer error.
        worst = (table["errors"] / reference[:, None].clamp_min(1e-30)).amax(dim=0)
        feasible = worst == worst.min()
    cost = table["cost"].masked_fill(~feasible, torch.iinfo(torch.int64).max)
    cheapest = feasible & (cost == cost.min())
    most_fp4 = table["fp4"].masked_fill(~cheapest, -1).max()
    preferred = cheapest & (table["fp4"] == most_fp4)
    # If cuts land in empty bins, prefer the later FP8 promotion (including CAP).
    index = int(table["second"].masked_fill(~preferred, -1).argmax())
    first, second = int(table["first"][index]), int(table["second"][index])
    grid = [i / 50 for i in range(226)] + ["CAP"]
    error = table["errors"][:, index]
    positive = (reference > 0) & (error > 0)
    margin = (10 * (reference[positive] / error[positive]).log10()).min().item() if positive.any() else None
    return {"thresholds": [grid[first], grid[second]],
            "quarter_bit_cost": int(table["cost"][index]), "tiles": table["tiles"],
            "effective_bits": int(table["cost"][index]) / (4 * table["tiles"]),
            "slack_db": slack_db, "reference_met": reference_met,
            "minimum_sqnr_margin_db": margin,
            "layer_errors": error.tolist(), "layer_reference_errors": reference.tolist()}


def solve_operand(tables, slack_db=0.0):
    solutions = {name: solve_projection(table, slack_db) for name, table in tables.items()}
    cost = sum(s["quarter_bit_cost"] for s in solutions.values())
    tiles = sum(s["tiles"] for s in solutions.values())
    return {"thresholds": {name: s["thresholds"] for name, s in solutions.items()},
            "effective_bits": cost / (4 * tiles), "slack_db": slack_db,
            "projections": solutions}


def relax_operand(tables, target_bits):
    """Algorithm 1: 48 bisection steps on reference-only SQNR slack [0,60] dB."""
    result = solve_operand(tables)
    if result["effective_bits"] <= target_bits:
        return result
    lower, upper = 0.0, 60.0
    result = solve_operand(tables, upper)
    for _ in range(48):
        middle = (lower + upper) / 2
        candidate = solve_operand(tables, middle)
        if candidate["effective_bits"] <= target_bits:
            upper, result = middle, candidate
        else:
            lower = middle
    return result


@torch.no_grad()
def derive_thresholds(reference_model, token_ids, *, target_bits=5.3):
    """P3-T1: one reference pass, Eq.8 histograms, exact grid search, bit target.

    Activations stay at NVFP4 parity; weights absorb the bit-budget slack.
    If this would require weights below FP4, exchange the operands' roles.
    No target-model weights, text, PPL or precision feedback enters this path.
    """
    layers = {name: layer for name, layer in reference_model.named_modules()
              if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in PROJECTIONS}
    histograms = {"weight": {}, "activation": {}}
    moments = {}

    def observe(name):
        def hook(layer, args):
            inputs = args[0].reshape(-1, layer.in_features)
            moments[name] = inputs.float().square().mean(dim=0)
            importance = layer.weight.float().square().sum(dim=0)
            histograms["activation"][name] = measure_histogram(inputs, importance)
            print(f"[thresholds activation {len(moments)}/{len(layers)}] {name}",
                  file=sys.stderr, flush=True)
        return hook

    handles = [layer.register_forward_pre_hook(observe(name)) for name, layer in layers.items()]
    reference_model.eval()
    try:
        reference_model.model(token_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    for index, (name, layer) in enumerate(layers.items(), 1):
        histograms["weight"][name] = measure_histogram(layer.weight, moments.pop(name))
        print(f"[thresholds weight {index}/{len(layers)}] {name}", file=sys.stderr, flush=True)

    tables = {operand: {projection: candidate_table([
        h for name, h in entries.items() if name.rsplit(".", 1)[-1] == projection])
        for projection in PROJECTIONS} for operand, entries in histograms.items()}
    activation = solve_operand(tables["activation"])
    weight = solve_operand(tables["weight"])
    if target_bits is not None:
        if 2 * target_bits - activation["effective_bits"] < 4.25:
            activation = relax_operand(tables["activation"], 2 * target_bits - weight["effective_bits"])
        else:
            weight = relax_operand(tables["weight"], 2 * target_bits - activation["effective_bits"])
    ids = token_ids.detach().cpu().reshape(-1).tolist()
    effective = (weight["effective_bits"] + activation["effective_bits"]) / 2
    return {"frozen": True, "source": {
        "reference_model": "meta-llama/Llama-3.1-8B",
        "revision": getattr(reference_model.config, "_commit_hash", None),
        "corpus": "provided-token-ids", "split": "unspecified", "tokens": len(ids),
        "sequence_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "procedure": "AdaptQuant Algorithm 1 / Appendix B; full threshold grid; reference-only SQNR slack"},
        "weight": weight["thresholds"], "activation": activation["thresholds"],
        "derivation": {"target_effective_bits": target_bits,
            "realized_reference_bits": effective,
            "reference_target_met": target_bits is None or effective <= target_bits,
            "bit_cost": "payload plus 8-bit E8M0 scale per 32 elements; excludes indices and shape",
            "grid": "0,0.02,...,4.50,CAP; both thresholds enumerated",
            "tie_break": "minimum bits, then more FP4 tiles, then larger t2",
            "weight": weight, "activation": activation,
            "nvfp4": {"element": "E2M1", "block_size": 16, "scale": "E4M3",
                       "tensor_scale": "FP32 amax(full operand)/(6*448)", "rounding": "nearest-even"},
            "assumptions": ["reference text segment and operating point are explicit reproduction choices",
                            "unattested custom token files require their corpus/split metadata from the caller",
                            "no target-model calibration or target-model feedback"]}}
