"""MX format-storage accounting from observed assignments, not FP16 RAM usage.

Layout follows paper Appendix C.1: full 16x32 tiles, packed element pools,
one 8-bit E8M0 scale per 32-element row, and a 5-byte tag/offset per tile.
We additionally budget two uint32 shape dimensions per matrix. This module
does not serialize a checkpoint or claim that the simulation tensors are packed.
"""

LAYOUT = {
    "name": "AdaptQuant Appendix C.1 pools + uint32 matrix shape",
    "tile_rows": 16, "tile_cols": 32,
    "scale_bits_per_block": 8, "index_bytes_per_tile": 5,
    "shape_bytes_per_matrix": 8,
    "padding": "store full tiles, including padded codes and empty-row scales",
}


def matrix_storage(shape, tile_counts):
    """Count a matrix's declared encoded representation, including padding.

    Counts are numbers of full tiles assigned to FP4/FP6/FP8, in that order.
    The denominator is the original number of weights, without padded elements.
    Index uses uint8 format + uint32 element offset; shape uses two uint32s.
    No file/container header or runtime diagnostics are part of this layout.
    """
    if (len(shape) != 2 or any(type(n) is not int or not 0 < n < 2**32 for n in shape)
            or len(tile_counts) != 3
            or any(type(n) is not int or n < 0 for n in tile_counts)):
        raise ValueError("expected positive uint32 matrix shape and three nonnegative integer counts")
    rows, cols = shape
    tiles = ((rows + 15) // 16) * ((cols + 31) // 32)
    if sum(tile_counts) != tiles:
        raise ValueError("format counts must cover every weight tile exactly once")
    if tiles * 512 > 2**32:
        raise ValueError("matrix exceeds the declared uint32 element-offset layout")
    weight_elements = rows * cols
    payload_bits = sum(n * 512 * width for n, width in zip(tile_counts, (4, 6, 8)))
    scale_bits = tiles * 16 * 8
    index_bits = tiles * 5 * 8
    metadata_bits = 8 * 8  # The two uint32 shape dimensions.
    total_bits = payload_bits + scale_bits + index_bits + metadata_bits
    return {
        "shape": list(shape), "weight_elements": weight_elements,
        "format_tiles": dict(zip(("4", "6", "8"), tile_counts)),
        "stored_elements": tiles * 512,
        "padding_elements": tiles * 512 - weight_elements,
        "payload_bits": payload_bits,  # Includes codes for padding; do not add padding again.
        "scale_bits": scale_bits, "index_bits": index_bits,
        "metadata_bits": metadata_bits, "total_bits": total_bits,
        "total_bytes": total_bits // 8,
        "effective_weight_bits": total_bits / weight_elements,
    }


def summarize_storage(layers):
    """Sum costs first; small layers must not get the same weight as large ones."""
    if not layers:
        raise ValueError("no quantized weight layers available for storage accounting")
    keys = ("weight_elements", "stored_elements", "padding_elements", "payload_bits",
            "scale_bits", "index_bits", "metadata_bits", "total_bits", "total_bytes")
    totals = {key: sum(row[key] for row in layers.values()) for key in keys}
    return {
        "accounting_basis": "format_layout_from_observed_weight_assignments",
        "packed_checkpoint_written": False,
        "scope": "quantized transformer-block q/k/v/o and gate/up/down W matrices; "
                 "excludes bias, embeddings, norm, lm_head and activations",
        "layout": dict(LAYOUT), "layers": layers, **totals,
        "effective_weight_bits": totals["total_bits"] / totals["weight_elements"],
    }
