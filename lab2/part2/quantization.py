"""Group-wise affine quantization, bit packing and mixed precision Linear inference."""

from dataclasses import dataclass
from math import gcd

import torch
from torch import nn
from torch.nn import functional as F


def pack_codes(codes, bits):
    """Pack unsigned codes LSB first; code i starts at bit i*bits.

    Final-byte padding is zero. One bit plane at a time avoids an N*bits temporary.
    """
    values = codes.reshape(-1).to(torch.int64)
    result = torch.zeros((values.numel() * bits + 7) // 8,
                         dtype=torch.int64, device=codes.device)
    positions = torch.arange(values.numel(), device=codes.device) * bits
    for bit in range(bits):
        offsets = positions + bit
        result.scatter_add_(0, offsets // 8, ((values >> bit) & 1) << (offsets % 8))
    return result.to(torch.uint8)


def unpack_codes(packed, bits, count):
    """Inverse of pack_codes, decoding byte blocks without per-element int64 indices.

    A block contains 8/gcd(8,bits) codes in bits/gcd(8,bits) bytes. Each code
    occupies at most two bytes; only the tiny within-block offsets are indexed.
    Final-block padding is transient and never changes the stored packed stream.
    """
    if bits == 8 or count == 0:
        return packed.clone()
    if bits in (2, 4):
        shifts = torch.arange(0, 8, bits, dtype=torch.uint8, device=packed.device)
        return ((packed[:, None] >> shifts) & ((1 << bits) - 1)).reshape(-1)[:count]

    divisor = gcd(8, bits)
    codes_per_block, bytes_per_block = 8 // divisor, bits // divisor
    padding = (-packed.numel()) % bytes_per_block
    blocks = F.pad(packed, (0, padding)).reshape(-1, bytes_per_block) if padding else packed.reshape(-1, bytes_per_block)
    offsets = torch.arange(codes_per_block, dtype=torch.int32, device=packed.device) * bits
    byte_positions = offsets // 8
    low = blocks.index_select(1, byte_positions).to(torch.int32)
    high = blocks.index_select(1, (byte_positions + 1).clamp_max(bytes_per_block - 1)).to(torch.int32)
    # For non-crossing codes, any excess high bits are discarded by the mask.
    low.bitwise_or_(high.bitwise_left_shift_(8))
    low.bitwise_right_shift_(offsets % 8)
    low.bitwise_and_((1 << bits) - 1)
    return low.to(torch.uint8).reshape(-1)[:count]


def group_weight(weight, group_size):
    """[out,in] -> [out,ceil(in/G),G], plus bool mask excluding padding."""
    rows, cols = weight.shape
    padding = (-cols) % group_size
    grouped = F.pad(weight, (0, padding)).reshape(rows, -1, group_size)
    mask = (torch.arange(cols + padding, device=weight.device) < cols)
    return grouped, mask.reshape(1, -1, group_size).expand_as(grouped)


def choose_qparams(groups, valid, bits):
    """Calculate group-wise affine min/max quantization parameters.

    Input: float groups [O,N,G], bool valid of the same shape, bits in [2,8].
    Return: positive finite scales and finite zero_points, both [O,N,1].
    Codes use [0,2**bits-1]. Padding is excluded; zero is included in the range.
    All-zero groups use scale 1.
    """
    # Keep parameter calculations in FP32, including for FP16 weights.
    groups = groups.float()
    fmax = torch.where(valid, groups, -torch.inf).amax(dim=-1, keepdim=True)
    fmin = torch.where(valid, groups, torch.inf).amin(dim=-1, keepdim=True)
    # Include zero before choosing scale, so one-sided/constant groups work too.
    fmax = fmax.clamp(min=0)
    fmin = fmin.clamp(max=0)
    pmax, pmin = 2**bits - 1, 0
    scale = (fmax - fmin) / (pmax - pmin)
    # An all-zero group has zero range; scale=1 still reconstructs zero exactly.
    scale = torch.where(torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale))
    zero_point = torch.round(pmin - fmin / scale).clamp(pmin, pmax)
    return scale, zero_point


def quantize_values(groups, valid, scales, zero_points, bits):
    """Quantize groups using affine scales and zero points.

    Input: groups/valid [O,N,G], qparams [O,N,1]. Return integer codes [O,N,G]
    in [0,2**bits-1], on the input device. Padding encodes zero.
    """
    scaled_tensor = groups/scales
    rounded_tensor = torch.round(scaled_tensor)
    rounded_tensor = rounded_tensor + zero_points
    rounded_tensor = torch.where(valid, rounded_tensor, zero_points)
    qmax, qmin = 2**bits - 1, 0
    quantized_tensor = rounded_tensor.clamp_(qmin, qmax)
    quantized_tensor = quantized_tensor.to(torch.uint8)
    return quantized_tensor


def dequantize_values(codes, scales, zero_points):
    """Reconstruct float values using r = S * (q - Z).

    Input: codes [O,N,G], qparams [O,N,1]. Return finite float [O,N,G].
    Codes are converted to FP32 before subtracting zero points.
    """
    codes = codes.to(torch.float32)
    codes = codes - zero_points
    dequantized = codes * scales
    return dequantized


@torch.no_grad()
def choose_qparams_mse(groups, valid, bits, *, grid_size=21, min_ratio=0.5,
                       group_chunk_size=2048):
    """Search clipping ranges that minimize group-wise weight MSE.

    Keep the original min/max parameters as the baseline. Candidate i shrinks
    both zero-inclusive endpoints by a factor in linspace(1,min_ratio,grid_size).
    Score the original weights after quantization and reconstruction, excluding
    padding; clipping loss must count too. Strict ties retain the baseline (or
    the first, widest winning candidate). No threshold/mask is stored: the final
    scales and zero points already describe the selected quantization range.

    Candidates run in parallel over bounded groups, not over a whole LLM layer.
    FP32 intermediates are temporary; this is weight-only PTQ, with no text or
    activation calibration and no gradient updates. Lower weight MSE does not
    guarantee lower end-to-end PPL.
    """

    # Sanitize padding before any arithmetic, including padding containing NaN.
    source = groups.float().masked_fill(~valid, 0)
    scales, zeros = choose_qparams(source, valid, bits)
    if grid_size == 1 or min_ratio == 1:
        return scales, zeros
    width = source.shape[-1]
    flat = source.reshape(-1, width)
    masks = valid.reshape(-1, width)
    best_scales, best_zeros = scales.reshape(-1, 1).clone(), zeros.reshape(-1, 1).clone()
    # Alpha=1 is already present as the exact min/max baseline; search narrower
    # candidates without making the baseline depend on a recomputed round trip.
    factors = torch.linspace(1., min_ratio, grid_size, device=groups.device,
                             dtype=torch.float32)[1:, None, None]
    n_candidates = grid_size - 1
    # Bound candidate arithmetic to ~8M elements per temporary, even for large G.
    chunk_size = max(1, min(group_chunk_size, (8 * 1024**2) // (n_candidates * width)))
    qmax = 2**bits - 1
    for start in range(0, flat.shape[0], chunk_size):
        stop = min(start + chunk_size, flat.shape[0])
        values, keep = flat[start:stop][None], masks[start:stop][None]
        baseline_scale, baseline_zero = best_scales[start:stop][None], best_zeros[start:stop][None]
        codes = quantize_values(values, keep, baseline_scale, baseline_zero, bits)
        restored = dequantize_values(codes, baseline_scale, baseline_zero)
        baseline_error = ((restored - values).square().masked_fill(~keep, 0)).sum(dim=-1)[0]
        lower = values.masked_fill(~keep, torch.inf).amin(dim=-1, keepdim=True).clamp(max=0)
        upper = values.masked_fill(~keep, -torch.inf).amax(dim=-1, keepdim=True).clamp(min=0)
        candidate_lower, candidate_upper = lower * factors, upper * factors
        candidate_scales = (candidate_upper - candidate_lower) / qmax
        candidate_scales = torch.where(
            torch.isfinite(candidate_scales) & (candidate_scales > 0),
            candidate_scales, torch.ones_like(candidate_scales))
        candidate_zeros = (-candidate_lower / candidate_scales).round().clamp(0, qmax)
        expanded_values = values.expand(n_candidates, -1, -1)
        expanded_mask = keep.expand_as(expanded_values)
        codes = quantize_values(expanded_values, expanded_mask, candidate_scales, candidate_zeros, bits)
        restored = dequantize_values(codes, candidate_scales, candidate_zeros)
        errors = ((restored - expanded_values).square().masked_fill(~expanded_mask, 0)).sum(dim=-1)
        candidate_error, winner = errors.min(dim=0)
        selected_scale = candidate_scales.squeeze(-1).gather(0, winner[None]).squeeze(0)
        selected_zero = candidate_zeros.squeeze(-1).gather(0, winner[None]).squeeze(0)
        improves = candidate_error < baseline_error
        best_scales[start:stop, 0] = torch.where(improves, selected_scale, baseline_scale[0, :, 0])
        best_zeros[start:stop, 0] = torch.where(improves, selected_zero, baseline_zero[0, :, 0])
    return best_scales.reshape_as(scales), best_zeros.reshape_as(zeros)


@dataclass(frozen=True)
class QuantConfig:
    bits: int
    group_size: int
    qparam_method: str = "mse"
    mse_grid_size: int = 21
    mse_min_ratio: float = 0.5
    compact: bool = False


class PackedLinear(nn.Module):
    """Packed inference storage; no retained dense weight. Inference only.

    shape_bits_group is serialized and included in the byte ledger. A state_dict
    needs matching PackedLinear shells; it does not rebuild the architecture.
    """

    def __init__(self, packed, scales, zero_points, shape, config, bias=None):
        super().__init__()
        self.out_features, self.in_features = shape
        self.bits, self.group_size = config.bits, config.group_size
        self.register_buffer("packed", packed.detach())
        self.register_buffer("scales", scales.detach())
        self.register_buffer("zero_points", zero_points.detach())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.register_buffer("shape_bits_group", torch.tensor(
            [*shape, self.bits, self.group_size], dtype=torch.int64, device=packed.device))

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear, config):
        groups, valid = group_weight(linear.weight.detach(), config.group_size)
        if config.qparam_method == "mse":
            scales, zeros = choose_qparams_mse(
                groups, valid, config.bits, grid_size=config.mse_grid_size,
                min_ratio=config.mse_min_ratio)
        else:
            scales, zeros = choose_qparams(groups.float(), valid, config.bits)
        if config.compact:
            scales = scales.half().clamp_min(torch.finfo(torch.float16).tiny).float()
        codes = quantize_values(groups.float(), valid, scales, zeros, config.bits)
        packed_type = CompactPackedLinear if config.compact else cls
        result = packed_type(pack_codes(codes, config.bits), scales, zeros,
                             tuple(linear.weight.shape), config, linear.bias).eval()
        return result

    def _dequantized_groups(self):
        n_groups = (self.in_features + self.group_size - 1) // self.group_size
        codes = unpack_codes(self.packed, self.bits, self.out_features * n_groups * self.group_size)
        codes = codes.reshape(self.out_features, n_groups, self.group_size)
        groups = dequantize_values(codes, self.scales, self.zero_points)
        return groups

    @torch.no_grad()
    def forward(self, inputs):
        groups = self._dequantized_groups()
        weight = groups.reshape(self.out_features, -1)[:, :self.in_features].to(inputs.dtype)
        bias = None if self.bias is None else self.bias.to(inputs.dtype)
        return F.linear(inputs, weight, bias)


class CompactPackedLinear(PackedLinear):
    """FP16 scales and bit-packed integer zero points for Cases 2/4."""

    def __init__(self, packed, scales, zero_points, shape, config, bias=None):
        super().__init__(packed, scales.half(),
                         pack_codes(zero_points.to(torch.uint8), config.bits),
                         shape, config, bias)

    def _dequantized_groups(self):
        codes = unpack_codes(self.packed, self.bits,
                             self.scales.numel() * self.group_size)
        codes = codes.reshape(*self.scales.shape[:2], self.group_size)
        zeros = unpack_codes(self.zero_points, self.bits,
                             self.scales.numel()).reshape_as(self.scales)
        return dequantize_values(codes, self.scales, zeros)


class Sparse34PackedLinear(CompactPackedLinear):
    """Three stored values and a 2-bit omitted position per four weights.

    Quantization groups and input width must be divisible by four. Only the
    surviving codes, omitted positions and qparams persist; dense weights are
    reconstructed temporarily by the inherited forward.
    """

    def __init__(self, codes, scales, zero_points, shape, config, omitted, bias=None):
        rows, cols = shape
        if cols % 4 or config.group_size % 4:
            raise ValueError("3:4 storage requires input width and group size divisible by 4")
        if codes.shape != (rows, cols) or omitted.shape != (rows, cols // 4):
            raise ValueError("3:4 codes/omitted positions must match the unpadded weight shape")
        slots = torch.arange(4, device=codes.device, dtype=torch.uint8)
        keep = slots != omitted.unsqueeze(-1)
        surviving = codes.reshape(rows, cols // 4, 4)[keep]
        super().__init__(pack_codes(surviving, config.bits), scales, zero_points,
                         shape, config, bias)
        self.register_buffer("omitted", pack_codes(omitted, 2))

    def _dequantized_groups(self):
        rows, cols = self.out_features, self.in_features
        omitted = unpack_codes(self.omitted, 2, rows * cols // 4).reshape(rows, -1)
        zeros = unpack_codes(self.zero_points, self.bits,
                             self.scales.numel()).reshape_as(self.scales)
        # Initialize omitted positions and group padding to the exact zero code.
        codes = zeros.expand(-1, -1, self.group_size).clone().reshape(rows, -1)
        blocks = codes[:, :cols].reshape(rows, cols // 4, 4)
        slots = torch.arange(4, device=codes.device, dtype=torch.uint8)
        keep = slots != omitted.unsqueeze(-1)
        surviving = unpack_codes(self.packed, self.bits, rows * cols // 4 * 3)
        blocks.masked_scatter_(keep, surviving)
        return dequantize_values(codes.reshape(*self.scales.shape[:2], self.group_size),
                                 self.scales, zeros)


class MixedPrecisionLinear(nn.Module):
    """Leading floating rows and trailing packed rows of one SVD factor.

    Children own compact independent storage. Bias is applied once after row
    concatenation; no full floating copy of the factor is retained. A state_dict
    is loaded into a matching architecture, just as with PackedLinear.
    """

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear, config, quantized_rows):
        if quantized_rows == 0:
            return linear.requires_grad_(False)
        if quantized_rows == linear.out_features:
            return PackedLinear.from_linear(linear, config).train(linear.training)
        result = cls()
        result.in_features, result.out_features = linear.in_features, linear.out_features
        floating_rows = linear.out_features - quantized_rows
        kwargs = {"device": linear.weight.device, "dtype": linear.weight.dtype}
        result.floating = nn.Linear(linear.in_features, floating_rows, bias=False, **kwargs)
        result.floating.weight.copy_(linear.weight[:floating_rows])
        tail = nn.Linear(linear.in_features, quantized_rows, bias=False, **kwargs)
        tail.weight.copy_(linear.weight[floating_rows:])
        result.quantized = PackedLinear.from_linear(tail, config)
        result.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        return result.requires_grad_(False).train(linear.training)

    @torch.no_grad()
    def forward(self, inputs):
        outputs = torch.cat((self.floating(inputs), self.quantized(inputs)), dim=-1)
        if self.bias is not None:
            outputs = outputs + self.bias.to(inputs.dtype)
        return outputs
