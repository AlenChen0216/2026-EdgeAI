"""Element-format inspection tools from AdaptQuant Table 1 (not a quantizer)."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MXFormat:
    bits: int
    exponent_bits: int
    mantissa_bits: int
    bias: int
    max_finite: float
    min_positive: float


FORMATS = {
    4: MXFormat(4, 2, 1, 1, 6.0, 0.5),
    6: MXFormat(6, 2, 3, 1, 7.5, 0.125),
    8: MXFormat(8, 4, 3, 7, 448.0, 2**-9),
}


def element_table(bits):
    """Return (code,value) for finite element encodings, including signed zero.

    E4M3's exp=15,mantissa=7 is reserved for NaN; FP4/FP6 use all encodings.
    """
    fmt = FORMATS[bits]
    table = []
    for code in range(2**bits):
        sign = -1 if code & (1 << (bits - 1)) else 1
        exponent = (code >> fmt.mantissa_bits) & ((1 << fmt.exponent_bits) - 1)
        mantissa = code & ((1 << fmt.mantissa_bits) - 1)
        if bits == 8 and exponent == 15 and mantissa == 7:
            continue
        if exponent == 0:
            value = (mantissa / 2**fmt.mantissa_bits) * 2**(1 - fmt.bias)
        else:
            value = (1 + mantissa / 2**fmt.mantissa_bits) * 2**(exponent - fmt.bias)
        table.append((code, sign * value))
    return table


def compare_values(original, reconstructed):
    """Flatten small tensors for a report/debug table; does not quantize input."""
    if original.shape != reconstructed.shape:
        raise ValueError("comparison tensors must have the same shape")
    x, y = original.detach().float().cpu().flatten(), reconstructed.detach().float().cpu().flatten()
    return [{"original": a.item(), "reconstructed": b.item(), "error": (b - a).item()}
            for a, b in zip(x, y)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, choices=(4, 6, 8), default=4)
    element_table(parser.parse_args().bits)
