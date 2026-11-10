"""AdaptQuant MX fake quantization using portable PyTorch operations."""

import re
import math

import torch
from torch import nn
from torch.nn import functional as F

from mx_formats import FORMATS
from bit_rate import combined_bit_rate, format_bit_counts, summarize_operand
from storage import matrix_storage, summarize_storage
from tiles import from_tiles, to_tiles


def tile_statistic(tiles, valid):
    """P3-S1 (Eq.4). Float tiles/bool valid [N,16,32] -> float [N].

    Ignore padding. Evaluate the max/RMS statistic in FP32; all-zero tiles use
    statistic 0. Check uniform magnitudes and a single nonzero in a full tile.
    """
    values = tiles.float().masked_fill(~valid, 0).abs()
    maximum = values.amax(dim=(-2, -1))
    # Normalize first so that squaring large/small values stays in FP32 range.
    relative = values / torch.where(maximum > 0, maximum, 1)[:, None, None]
    mean_square = relative.square().sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp_min(1)
    statistic = -0.5 * mean_square.clamp_min(torch.finfo(torch.float32).tiny).log2()
    return torch.where(maximum > 0, statistic.clamp_min(0), 0)


def select_formats(statistic, thresholds):
    """P3-S2 (Eq.5). statistic [N], (t1,t2) -> integer bits [N] in {4,6,8}.

    Preserve device and check equality at both thresholds. CAP is +infinity;
    when t2=CAP no finite statistic may select FP8. Do not inspect PPL here.
    """
    t1, t2 = thresholds
    return 4 + 2 * (statistic >= t1).long() + 2 * (statistic >= t2).long()


def block_scales(tiles, valid, fmt):
    """P3-Q1 (Eq.2). tiles/valid [N,16,32], MXFormat -> scales [N,16,1].

    One power-of-two E8M0 scale per ROW, not per tile. Use FP32 and ignore
    padding; empty/all-zero rows use scale 1. Check a row whose max crosses a
    power-of-two boundary. Consider the E8M0 exponent range [-127,127].
    """
    maximum = tiles.float().masked_fill(~valid, 0).abs().amax(dim=-1, keepdim=True)
    safe_maximum = torch.where(maximum > 0, maximum, 1)
    exponent = (torch.frexp(safe_maximum)[1] - 1 - math.floor(math.log2(fmt.max_finite))).clamp(-127, 127)
    return torch.where(maximum > 0, torch.exp2(exponent.float()), 1)


def round_elements(normalized, valid, fmt):
    """P3-Q2 (Section3.2, Table1). Float [N,16,32] -> same-shaped float.

    Return element-grid values, BEFORE multiplying by scale. Use round-to-nearest
    ties-to-even, subnormals, flush-to-zero, signed saturation. Padding returns 0.
    Inspect mx_formats.element_table; integer rounding alone cannot simulate FP4.
    Check every codebook value, midpoints, both signs and overflow endpoints.
    """
    values = normalized.float().masked_fill(~valid, 0)
    magnitude = values.abs().clamp_max(fmt.max_finite)
    # Subnormals share the step of the smallest normal binade.
    exponent = torch.frexp(magnitude.clamp_min(2.0 ** (1 - fmt.bias)))[1] - 1
    step = torch.exp2((exponent - fmt.mantissa_bits).float())
    rounded = ((magnitude / step).round() * step).clamp_max(fmt.max_finite)
    return torch.copysign(rounded, values).masked_fill(~valid, 0)


@torch.no_grad()
def fake_quantize_matrix(matrix, thresholds, tile_batch=256, *, return_statistics=False):
    """Numerical plumbing around exercises; returns float matrix and tile counts.

    Chunking bounds numerical temporaries. All stored values remain floating
    point, so this path claims neither low-bit storage nor hardware speedups.
    With return_statistics=True, also return int64 [3,2] counters: valid
    elements and nonempty 32-element blocks per format. Padding never enters
    element counts; a partial nonempty block still requires a full scale.
    """
    if type(tile_batch) is not int or tile_batch < 1:
        raise ValueError("tile_batch must be a positive integer")
    if not torch.isfinite(matrix).all():
        raise ValueError("quantization input must be finite")
    tiles, valid, layout = to_tiles(matrix)
    output = torch.empty_like(tiles)
    counts = torch.zeros(3, dtype=torch.int64, device=matrix.device)
    statistics = torch.zeros(3, 2, dtype=torch.int64, device=matrix.device)
    for start in range(0, tiles.shape[0], tile_batch):
        chunk = tiles[start:start + tile_batch].float()
        mask = valid[start:start + tile_batch]
        statistic = tile_statistic(chunk, mask)
        if statistic.shape != (len(chunk),) or statistic.device != matrix.device:
            raise ValueError("tile statistic must have shape [N] on the input device")
        if not torch.isfinite(statistic).all() or (statistic < 0).any():
            raise ValueError("tile statistic must be finite and nonnegative")
        bits = select_formats(statistic, thresholds)
        if bits.shape != statistic.shape or bits.device != matrix.device:
            raise ValueError("format selection must preserve statistic shape/device")
        if bits.is_floating_point() or bits.is_complex() or not ((bits == 4) | (bits == 6) | (bits == 8)).all():
            raise ValueError("selected formats must be integer 4,6,8")
        reconstructed = torch.zeros_like(chunk)
        for index, width in enumerate((4, 6, 8)):
            selected = bits == width
            counts[index] += selected.sum()
            if not selected.any():
                continue
            values, keep = chunk[selected], mask[selected]
            statistics[index, 0] += keep.sum()
            statistics[index, 1] += keep.any(dim=-1).sum()
            scales = block_scales(values, keep, FORMATS[width])
            if scales.shape != (*values.shape[:2], 1) or scales.device != matrix.device:
                raise ValueError("MX scales must have shape [N,16,1] on the input device")
            if not scales.is_floating_point() or not torch.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError("MX scales must be positive finite floats")
            elements = round_elements(values / scales, keep, FORMATS[width])
            if elements.shape != values.shape or elements.device != matrix.device:
                raise ValueError("rounded elements must preserve input shape/device")
            if not elements.is_floating_point() or not torch.isfinite(elements).all():
                raise ValueError("rounded elements must be finite floats")
            reconstructed[selected] = (elements * scales).masked_fill(~keep, 0)
        output[start:start + len(chunk)] = reconstructed.to(matrix.dtype)
    result = from_tiles(output, layout)
    if not torch.isfinite(result).all():
        raise ValueError("reconstruction overflowed the model dtype")
    return (result, counts, statistics) if return_statistics else (result, counts)


class MXLinear(nn.Module):
    """Weight fake quantization once; input fake quantization every forward."""

    def __init__(self, weight, bias, activation_thresholds, weight_counts, weight_statistics=None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.detach())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.register_buffer("activation_thresholds", torch.tensor(
            activation_thresholds, dtype=torch.float64, device=weight.device))
        self.register_buffer("weight_counts", weight_counts)
        self.register_buffer("weight_statistics", weight_statistics)
        self.register_buffer("activation_counts", torch.zeros_like(weight_counts), persistent=False)
        self.register_buffer("activation_statistics", torch.zeros(3, 2, dtype=torch.int64,
                                                                 device=weight.device), persistent=False)
        self.activation_elements = 0
        self.activation_calls = 0

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear, weight_thresholds, activation_thresholds):
        weight, counts, statistics = fake_quantize_matrix(
            linear.weight.detach(), weight_thresholds, return_statistics=True)
        return cls(weight, linear.bias, activation_thresholds, counts, statistics).eval()

    def reset_activation_statistics(self):
        self.activation_counts.zero_()
        self.activation_statistics.zero_()
        self.activation_elements = 0
        self.activation_calls = 0

    @torch.no_grad()
    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError("MXLinear input feature count mismatch")
        # Flatten leading batch/token axes to M; columns always remain K.
        matrix = inputs.reshape(-1, self.in_features)
        quantized, counts, statistics = fake_quantize_matrix(
            matrix, self.activation_thresholds, return_statistics=True)
        self.activation_counts.add_(counts)
        self.activation_statistics.add_(statistics)
        self.activation_elements += matrix.numel()
        self.activation_calls += 1
        return F.linear(quantized.reshape_as(inputs), self.weight.to(inputs.dtype),
                        None if self.bias is None else self.bias.to(inputs.dtype))


_TARGET = re.compile(
    r"model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$")


def quantize_model(model, thresholds):
    if any(isinstance(layer, MXLinear) for layer in model.modules()):
        raise ValueError("model was already fake quantized; reload the original model")
    names = [name for name, layer in model.named_modules(remove_duplicate=False)
             if _TARGET.fullmatch(name) and isinstance(layer, nn.Linear)]
    if not names:
        raise ValueError("no transformer-block Linear targets were found")
    for name in names:
        projection = name.rsplit(".", 1)[-1]
        replacement = MXLinear.from_linear(model.get_submodule(name),
                                           thresholds["weight"][projection],
                                           thresholds["activation"][projection])
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, replacement)
    return model.eval()


def format_report(model):
    report = {}
    for name, layer in model.named_modules():
        if isinstance(layer, MXLinear):
            entry = {}
            for operand, counts in (("weight", layer.weight_counts), ("activation", layer.activation_counts)):
                values = counts.tolist()
                total = sum(values)
                entry[f"{operand}_tiles"] = dict(zip(("4", "6", "8"), values))
                entry[f"{operand}_fractions"] = dict(zip(
                    ("4", "6", "8"), [value / total for value in values])) if total else None
            report[name] = entry
    return report


def weight_storage_report(model):
    """Account for the format representation separately from FP16 simulation.

    Weight assignments are measured once during quantization. Activation counts
    depend on inference calls and must never enter the weight-bit denominator.
    """
    layers = {}
    for name, layer in model.named_modules():
        if isinstance(layer, MXLinear):
            layers[name] = matrix_storage(tuple(layer.weight.shape), layer.weight_counts.tolist())
    report = summarize_storage(layers)
    report["simulation_weight_bytes"] = sum(
        layer.weight.numel() * layer.weight.element_size()
        for layer in model.modules() if isinstance(layer, MXLinear)
    )
    return report


def reset_activation_statistics(model):
    """Start a fresh measurement; do not mix warm-up/diagnostics into evaluation."""
    for layer in model.modules():
        if isinstance(layer, MXLinear):
            layer.reset_activation_statistics()


def bit_rate_report(model):
    """Use both operands; every quantized layer must have evaluation observations."""
    weights, activations = {}, {}
    for name, layer in model.named_modules():
        if isinstance(layer, MXLinear):
            try:
                statistics = None if layer.weight_statistics is None else layer.weight_statistics.tolist()
                weights[name] = format_bit_counts(statistics, layer.weight.numel())
                activations[name] = format_bit_counts(layer.activation_statistics.tolist(),
                                                     layer.activation_elements)
                activations[name]["forward_calls"] = layer.activation_calls
            except ValueError as exc:
                raise ValueError(f"incomplete bit-rate statistics at {name}: {exc}") from exc
    return combined_bit_rate(summarize_operand(weights), summarize_operand(activations))
