"""Part2 compression policies, SVD and quantized factor storage."""

import math
from dataclasses import replace
import sys

import torch
from torch import nn

from common import record_base, replace_module, target_linears
from quantization import MixedPrecisionLinear, PackedLinear, QuantConfig


def choose_layer_configs(case, targets, rate, wbits, group_size):
    """Layer precision policies for HW Part2 Cases 1/2/3.

    targets: dict[full name, original nn.Linear]. Return a dict with exactly
    these keys: QuantConfig(bits,group_size) to quantize, None to keep.
    wbits is an optional CLI override; honor it explicitly when supplied.
    rate is a requested reduction, not evidence that it was reached. Think about
    qparam/padding overhead and Case1's strict >75% boundary. Check measured
    bytes before PPL. Start by filling separate branches for cases 1 and 2.
    """
    if case == 1:
        high_selected_bits = 4 if wbits is None else wbits
        selected_bits = 3 if wbits is None else wbits
        configs = {}
        for name in targets:
            if name.find("self_attn") != -1:
                configs[name] = QuantConfig(bits = high_selected_bits, group_size=group_size)
            else:
                configs[name] = QuantConfig(bits = selected_bits, group_size = group_size)
        return configs
    elif case == 2:
        bits = 3 if wbits is None else wbits
        return {name: QuantConfig(bits, group_size, compact=True) for name in targets}
    elif case == 3:
        selected_bits = 8 if wbits is None else wbits
        configs = {}
        for name in targets:
            configs[name] = QuantConfig(bits = selected_bits, group_size= group_size)
        return configs


def apply_quantization_plan(model, configs, *, qparam_options=None):
    """Replace explicit targets; unchanged targets remain in the byte ledger."""
    for index, (name, config) in enumerate(configs.items(), 1):
        if config is not None:
            if qparam_options:
                config = replace(config, **qparam_options)
            print(f"[compress {index}/{len(configs)}] {name}: bits={config.bits}, "
                  f"group={config.group_size}, qparams={config.qparam_method}",
                  file=sys.stderr, flush=True)
            replace_module(model, name, PackedLinear.from_linear(model.get_submodule(name), config))
    return model


def case1(model, tok, rate=0.75, wbits=None, group_size=128, *, qparam_options=None):
    configs = choose_layer_configs(1, target_linears(model), rate, wbits, group_size)
    return apply_quantization_plan(model, configs, qparam_options=qparam_options), tok


def case2(model, tok, rate=0.81, wbits=None, group_size=512, *, qparam_options=None):
    """RTN baseline for the 3-bit/G512 compact plan; CLI defaults to GPTQ."""
    configs = choose_layer_configs(2, target_linears(model), rate, wbits, group_size)
    return apply_quantization_plan(model, configs, qparam_options=qparam_options), tok


@torch.no_grad()
def svd_linear(linear, rank):
    """P2-SVD: replace W[O,I] with two inference-only Linear factors.

    W ~= A @ B, A=U[:,:r]*S[:r] [O,r], B=Vh[:r,:] [r,I].
    Return nn.Sequential(first,second): first stores B, second stores A and
    the original bias once. SVD uses FP32 (FP64 for an FP64 source); retained
    factors use the original dtype/device. Only factor Parameters survive,
    with independent compact storage, no U/S/Vh views or dense W cache.

    Rank is explicit: this helper does not choose a Case3 compression budget.
    Full rank is permitted for reconstruction checks, even if it saves no bytes.
    """
    weight = linear.weight.detach()

    compute_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    U, S, Vh = torch.linalg.svd(weight.to(compute_dtype), full_matrices=False)
    kwargs = {"device": weight.device, "dtype": weight.dtype}
    first = nn.Linear(linear.in_features, rank, bias=False, **kwargs)
    second = nn.Linear(rank, linear.out_features,
                       bias=linear.bias is not None, **kwargs)
    # copy_ casts into independent Parameters; never keep views into full SVD outputs.
    first.weight.copy_(Vh[:rank, :])
    second.weight.copy_(U[:, :rank] * S[:rank])
    if linear.bias is not None:
        second.bias.copy_(linear.bias)
    return nn.Sequential(first, second).requires_grad_(False).train(linear.training)


def prune_linear(linear, hessian, config, *, fraction=0.25, damp=0.01, block_size=128):
    """Joint SparseGPT 3:4 pruning/quantization with actual sparse storage."""
    from gptq import gptq_linear
    return gptq_linear(linear, hessian, config, damp=damp, block_size=block_size,
                       pruning=fraction, compact=True, pruning_pattern="3:4")


def _select_byte_increments(costs, budget, *, cover=False):
    """Subset sum over small rank/row remainders, never over full model bytes.

    With cover=True choose the smallest reachable sum >= budget; otherwise
    choose the largest sum <= budget. Return selected cost indices and bytes.
    """
    if not costs or budget <= 0:
        return [], 0
    divisor = math.gcd(*costs)
    limit = budget // divisor
    if cover:
        limit = (budget + divisor - 1) // divisor + max(costs) // divisor
    mask = (1 << (limit + 1)) - 1
    reachable = [1]
    for cost in costs:
        previous = reachable[-1]
        reachable.append((previous | (previous << (cost // divisor))) & mask)
    chosen = reachable[-1].bit_length() - 1
    if cover:
        lower = (budget + divisor - 1) // divisor
        above = reachable[-1] >> lower
        if above:
            chosen = lower + (above & -above).bit_length() - 1
    selected, total = [], chosen * divisor
    for index in range(len(costs) - 1, -1, -1):
        if not ((reachable[index] >> chosen) & 1):
            selected.append(index)
            chosen -= costs[index] // divisor
    return selected, total


def case3_ranks(model, names, svd_rate=0.21):
    """Budget SVD-only W storage before quantization; svd_rate is a REDUCTION.

    W[O,I] -> B[r,I], A[O,r]. Bias is unchanged and excluded from this stage's
    ratio. Choose floor/ceil ranks at or below (1-svd_rate)*original W bytes.
    Integer ranks can make the measured reduction slightly exceed svd_rate.
    """
    from fractions import Fraction
    specs = []
    for name in names:
        layer = model.get_submodule(name)
        out_features, in_features = layer.weight.shape
        specs.append((name, out_features, in_features, layer.weight.element_size()))
    weight_bytes = sum(model._weight_base_bytes[name] for name in names)
    retained = Fraction(1) - Fraction(str(svd_rate))
    factor_budget = weight_bytes * retained.numerator // retained.denominator
    ranks, costs, factor_bytes = {}, [], 0
    for name, outputs, inputs, element_size in specs:
        rank = max(1, min(outputs, inputs,
                          factor_budget * outputs * inputs // (weight_bytes * (outputs + inputs))))
        increment = (outputs + inputs) * element_size
        ranks[name] = rank
        factor_bytes += rank * increment
        if rank < min(outputs, inputs):
            costs.append((name, increment))
    selected, added = _select_byte_increments([cost for _, cost in costs], factor_budget - factor_bytes)
    for index in selected:
        ranks[costs[index][0]] += 1
    factor_bytes += added
    fixed = sum(model._linear_base_bytes.values()) - weight_bytes
    return ranks, {"requested_svd_compression_ratio": svd_rate,
                   "target_factor_bytes": factor_budget,
                   "planned_factor_bytes": factor_bytes,
                   "planned_total_bytes": factor_bytes + fixed}


def case3_quantization_plan(model, names, ranks, configs, rate):
    """Allocate low-bit trailing rows of each factor to the overall byte target.

    One packed child uses 32 metadata bytes, FP32 scale/zero per group, and
    actual padded codes. Row units end on byte boundaries even for odd bits/G.
    A proportional allocation is refined by subset sum of next row increments.
    Unreachable targets are recorded with their actual signed byte difference.
    No model tensors or dummy padding buffers are created by this planner.
    """
    from fractions import Fraction
    original = sum(model._linear_base_bytes.values())
    retained = Fraction(1) - Fraction(str(rate))
    target = original * retained.numerator // retained.denominator
    specs, svd_total = [], 0
    for name in names:
        linear, rank, config = model.get_submodule(name), ranks[name], configs[name]
        element_size = linear.weight.element_size()
        svd_total += rank * (linear.in_features + linear.out_features) * element_size
        if linear.bias is not None:
            svd_total += linear.bias.numel() * linear.bias.element_size()
        for factor, outputs, inputs in ((0, rank, linear.in_features),
                                        (1, linear.out_features, rank)):
            groups = (inputs + config.group_size - 1) // config.group_size
            row_bits = groups * config.group_size * config.bits
            unit_rows = 8 // math.gcd(row_bits, 8)
            saving = (unit_rows * inputs * element_size - unit_rows * row_bits // 8
                      - unit_rows * groups * 8)
            specs.append({"name": name, "factor": factor, "rows": outputs,
                          "unit_rows": unit_rows, "units": outputs // unit_rows,
                          "saving_per_unit": saving, "count": 0})
    candidates = [spec for spec in specs
                  if spec["saving_per_unit"] > 0
                  and spec["units"] * spec["saving_per_unit"] > 32]
    desired = max(0, svd_total - target)
    capacity = sum(spec["units"] * spec["saving_per_unit"] for spec in candidates)
    # Every used PackedLinear stores shape/bits/G in four int64 values.
    numerator = min(capacity, desired + 32 * len(candidates)) if desired else 0
    saved = 0
    for spec in candidates:
        count = numerator * spec["units"] // capacity
        if count * spec["saving_per_unit"] > 32:
            spec["count"] = count
            saved += count * spec["saving_per_unit"] - 32

    def next_increments():
        increments = []
        for spec in candidates:
            count, saving = spec["count"], spec["saving_per_unit"]
            step = 1 if count else 32 // saving + 1
            if count + step <= spec["units"]:
                increments.append((spec, step, step * saving - (32 if count == 0 else 0)))
        return increments

    increments = next_increments()
    remaining = desired - saved
    # Tiny models/large padding may need more than one next-row pass.
    while increments and remaining > sum(cost for _, _, cost in increments):
        for spec, step, cost in increments:
            spec["count"] += step
            saved += cost
        remaining = desired - saved
        increments = next_increments()
    selected, added = _select_byte_increments([cost for _, _, cost in increments], remaining, cover=True)
    for index in selected:
        spec, step, _ = increments[index]
        spec["count"] += step
    saved += added
    plan = {(spec["name"], spec["factor"]): spec["count"] * spec["unit_rows"] for spec in specs}
    final_bytes = svd_total - saved
    return plan, {"target_total_bytes": target, "svd_total_bytes": svd_total,
                  "planned_total_bytes": final_bytes,
                  "target_difference_bytes": final_bytes - target,
                  "exact_target": final_bytes == target,
                  "allocation": "proportional trailing rows, then next-row subset sum"}


@torch.no_grad()
def case3(model, tok, rate=0.5, wbits=None, group_size=128, *, qparam_options=None,
          svd_rate=0.21, svd_method="whiten", svd_config=None):
    """Original W -> activation-weighted SVD -> packed/float mixed factors.

    Both stages retain their storage savings. Record SVD weights BEFORE any
    factor quantization; final storage includes bias, padding, qparams, metadata
    and the unquantized factor rows. No original dense W survives conversion.
    """
    from common import tensor_bytes
    names = list(target_linears(model))
    configs = choose_layer_configs(3, dict.fromkeys(names), rate, wbits, group_size)
    if qparam_options:
        configs = {name: replace(config, **qparam_options) for name, config in configs.items()}
    ranks, rank_budget = case3_ranks(model, names, svd_rate)
    plan, quant_budget = case3_quantization_plan(model, names, ranks, configs, rate)
    layers, whitening = [], None
    if svd_method == "whiten":
        from gptq import calibration_data
        from svd_whitening import WhiteningConfig, apply_whitened_svd
        svd_config = WhiteningConfig() if svd_config is None else svd_config
        batches, provenance = calibration_data(tok, svd_config)
        model, layers, whitening = apply_whitened_svd(
            model, ranks, configs, plan, batches, svd_config, provenance=provenance)
        del batches
    else:
        for index, name in enumerate(names, 1):
            linear, config, rank = model.get_submodule(name), configs[name], ranks[name]
            compressed = svd_linear(linear, rank)
            before = sum(factor.weight.numel() * factor.weight.element_size() for factor in compressed)
            quantized_rows = [plan[name, factor] for factor in (0, 1)]
            print(f"[case3 {index}/{len(names)}] {name}: rank={rank}, "
                  f"quant={config.bits}-bit, group={config.group_size}, "
                  f"quantized rows={quantized_rows}", file=sys.stderr, flush=True)
            for factor in (0, 1):
                compressed[factor] = MixedPrecisionLinear.from_linear(
                    compressed[factor], config, quantized_rows[factor])
            layers.append({"name": name, "rank": rank, "bits": config.bits,
                           "group_size": config.group_size, "qparam_method": config.qparam_method,
                           "svd_factor_bytes": before, "quantized_rows": quantized_rows,
                           "final_bytes": tensor_bytes(compressed)})
            replace_module(model, name, compressed)
            del linear, compressed
    model._svd_measurement = {"original_bytes": sum(model._weight_base_bytes.values()),
                              "factor_bytes": sum(row["svd_factor_bytes"] for row in layers)}
    model._svd_stage_definition = "1 - SVD-only factor bytes / original W bytes; before factor quantization; excludes bias"
    model._case3_metadata = {"order": "svd_quantize_factors", "svd_rate": svd_rate,
                             "svd_method": svd_method, "whitening": whitening,
                             "rank_budget": rank_budget, "quantization_budget": quant_budget,
                             "quantizer": "rtn", "layers": layers}
    return model, tok


def case4(model, tok, rate=0.85, wbits=None, group_size=64, *, qparam_options=None,
          gptq_config=None):
    """3:4 sparse storage (>20% pruning-only CR) + 2-bit/G64 quantization."""
    from gptq import GPTQConfig, apply_gptq, calibration_data
    config = GPTQConfig() if gptq_config is None else gptq_config
    quant_config = QuantConfig(2 if wbits is None else wbits, group_size, "minmax")
    if qparam_options:
        quant_config = replace(quant_config, **qparam_options)
    configs = {name: quant_config for name in target_linears(model)}
    batches, provenance = calibration_data(tok, config)
    model = apply_gptq(model, configs, batches, config, provenance=provenance,
                       linear_compressor=prune_linear)
    model._gptq_metadata.update(algorithm="SparseGPT joint pruning/quantization",
                                pruning_fraction=0.25, pruning_pattern="3:4", scale_dtype="float16",
                                zero_point_storage="bit_packed")
    layers = model._gptq_metadata["layers"]
    model._pruning_measurement = {
        "original_elements": sum(layer["original_elements"] for layer in layers),
        "pruned_elements": sum(layer["pruned_elements"] for layer in layers),
        "original_bytes": sum(layer["pruning_original_bytes"] for layer in layers),
        "compressed_bytes": sum(layer["pruning_only_bytes"] for layer in layers),
        "definition": "1 - (original-dtype 3:4 values + packed positions + metadata + bias) / original Linear bytes; before quantization",
    }
    return model, tok


def compress_model(model, tok, case, rate, wbits=None, group_size=None, *,
                   qparam_method=None, mse_grid_size=21, mse_min_ratio=0.5,
                   quantizer="rtn", gptq_config=None, svd_rate=0.21,
                   svd_method="whiten", svd_config=None):
    if group_size is None:
        group_size = {2: 512, 4: 64}.get(case, 128)
    if qparam_method is None:
        qparam_method = "minmax" if quantizer == "gptq" or case == 4 else "mse"
    options = {"qparam_method": qparam_method, "mse_grid_size": mse_grid_size,
               "mse_min_ratio": mse_min_ratio}
    record_base(model)
    if case == 3:
        model, tok = case3(model, tok, rate, wbits, group_size,
                           qparam_options=options, svd_rate=svd_rate,
                           svd_method=svd_method, svd_config=svd_config)
    elif case == 4:
        model, tok = case4(model, tok, rate, wbits, group_size,
                           qparam_options=options, gptq_config=gptq_config)
    elif quantizer == "gptq":
        from gptq import GPTQConfig, apply_gptq, calibration_data
        gptq_config = GPTQConfig() if gptq_config is None else gptq_config
        configs = choose_layer_configs(case, target_linears(model), rate, wbits, group_size)
        configs = {name: None if config is None else replace(config, **options)
                   for name, config in configs.items()}
        batches, provenance = calibration_data(tok, gptq_config)
        model = apply_gptq(model, configs, batches, gptq_config, provenance=provenance)
    else:
        functions = {1: case1, 2: case2, 3: case3, 4: case4}
        model, tok = functions[case](model, tok, rate, wbits, group_size, qparam_options=options)
    return model, tok
