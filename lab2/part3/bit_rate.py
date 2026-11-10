"""HW Part 3: average effective weight/activation format widths, including scales.

Logical format accounting excludes padded element codes and empty rows, but
charges one full E8M0 scale for every nonempty 32-element block, even partial
blocks. Storage/packing/index costs are reported separately in storage.py.
"""

BIT_RATE_LIMIT = 5.3
WIDTHS = (4, 6, 8)


def format_bit_counts(statistics, expected_elements):
    """Rows: FP4/FP6/FP8. Columns: valid elements, nonempty scale blocks."""
    if (type(expected_elements) is not int or expected_elements <= 0
            or statistics is None or len(statistics) != 3):
        raise ValueError("missing format statistics or no observed elements")
    for row in statistics:
        if (len(row) != 2 or any(type(n) is not int or n < 0 for n in row)
                or not row[1] <= row[0] <= 32 * row[1]):
            raise ValueError("expected valid element/nonempty scale-block counts for each format")
    elements = sum(row[0] for row in statistics)
    if elements != expected_elements:
        raise ValueError("format statistics do not cover every observed element")
    payload_bits = sum(width * row[0] for width, row in zip(WIDTHS, statistics))
    scale_bits = 8 * sum(row[1] for row in statistics)
    total_bits = payload_bits + scale_bits
    return {
        "elements": elements,
        "format_elements": {str(w): row[0] for w, row in zip(WIDTHS, statistics)},
        "scale_blocks": {str(w): row[1] for w, row in zip(WIDTHS, statistics)},
        "payload_bits": payload_bits, "scale_bits": scale_bits,
        "total_bits": total_bits, "effective_bits": total_bits / elements,
    }


def summarize_operand(layers):
    if not layers:
        raise ValueError("no quantized layers available for bit-rate accounting")
    totals = {key: sum(row[key] for row in layers.values())
              for key in ("elements", "payload_bits", "scale_bits", "total_bits")}
    if totals["elements"] <= 0:
        raise ValueError("no observed elements for operand bit-rate accounting")
    return {"layers": layers, **totals,
            "effective_bits": totals["total_bits"] / totals["elements"]}


def meets_bit_rate_limit(weight_bits, weight_elements, activation_bits, activation_elements):
    """Compare (Wbits/Wnum + Abits/Anum)/2 <= 53/10 without rounding."""
    if (any(type(n) is not int or n <= 0
            for n in (weight_bits, weight_elements, activation_bits, activation_elements))):
        raise ValueError("both operands require positive integer bits and element counts")
    return (5 * (weight_bits * activation_elements + activation_bits * weight_elements)
            <= 53 * weight_elements * activation_elements)


def combined_bit_rate(weight, activation):
    return {
        "definition": "(Weight Bits + Activation Bits) / 2",
        "accounting_basis": "observed format assignments; valid codes + 8-bit E8M0 "
                            "scale per nonempty 32-element block",
        "weight_aggregation": "sum over quantized W elements, once per layer",
        "activation_aggregation": "sum over input elements to quantized layers in "
                                  "the evaluation forward calls since reset",
        "excluded_costs": "index, shape, padded codes, empty-row scales, FP16 "
                          "simulation containers; weight storage reported separately",
        "weight": weight, "activation": activation,
        "weight_bits": weight["effective_bits"],
        "activation_bits": activation["effective_bits"],
        "realized_effective_bit_rate": (weight["effective_bits"] + activation["effective_bits"]) / 2,
        "limit_bits": BIT_RATE_LIMIT,
        "passes_limit": meets_bit_rate_limit(weight["total_bits"], weight["elements"],
                                             activation["total_bits"], activation["elements"]),
    }
