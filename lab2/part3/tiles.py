"""16x32 tile layout. Padding is masked out of statistic/scale measurements."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TileLayout:
    rows: int
    cols: int

    @property
    def tile_rows(self):
        return (self.rows + 15) // 16

    @property
    def tile_cols(self):
        return (self.cols + 31) // 32


def to_tiles(matrix):
    """[M,K] -> [tile_count,16,32], mask, layout; tile order is row-major."""
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not matrix.is_floating_point():
        raise ValueError("expected a nonempty floating matrix")
    layout = TileLayout(*matrix.shape)
    padded = F.pad(matrix, (0, (-layout.cols) % 32, 0, (-layout.rows) % 16))
    valid = (torch.arange(padded.shape[0], device=matrix.device)[:, None] < layout.rows)
    valid = valid & (torch.arange(padded.shape[1], device=matrix.device)[None, :] < layout.cols)

    def arrange(tensor):
        return tensor.reshape(layout.tile_rows, 16, layout.tile_cols, 32).permute(
            0, 2, 1, 3).reshape(-1, 16, 32)

    return arrange(padded), arrange(valid), layout


def from_tiles(tiles, layout):
    expected = (layout.tile_rows * layout.tile_cols, 16, 32)
    if tuple(tiles.shape) != expected:
        raise ValueError(f"expected tiles of shape {expected}")
    matrix = tiles.reshape(layout.tile_rows, layout.tile_cols, 16, 32).permute(
        0, 2, 1, 3).reshape(layout.tile_rows * 16, layout.tile_cols * 32)
    return matrix[:layout.rows, :layout.cols].contiguous()
