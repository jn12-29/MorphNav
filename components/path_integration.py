from __future__ import annotations

from dataclasses import dataclass

import torch as th
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PathIntegrationOutputs:
    pc_logits: th.Tensor
    bottleneck: th.Tensor


class PlaceCellTargetEncoder(nn.Module):
    """Fixed place-cell population encoder for path-integration supervision."""

    def __init__(
        self,
        n_cells: int = 256,
        stdev: float = 0.01,
        pos_min: float = -2.5,
        pos_max: float = 2.5,
        seed: int = 8341,
    ) -> None:
        super().__init__()
        if n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if stdev <= 0.0:
            raise ValueError("stdev must be positive")
        if pos_max <= pos_min:
            raise ValueError("pos_max must be greater than pos_min")

        generator = th.Generator()
        generator.manual_seed(int(seed))
        centers = th.empty((int(n_cells), 2), dtype=th.float32)
        centers.uniform_(float(pos_min), float(pos_max), generator=generator)

        self.n_cells = int(n_cells)
        self.stdev = float(stdev)
        self.pos_min = float(pos_min)
        self.pos_max = float(pos_max)
        self.seed = int(seed)
        self.register_buffer("centers", centers, persistent=True)

    @property
    def metadata(self) -> dict[str, int | float]:
        return {
            "n_cells": self.n_cells,
            "stdev": self.stdev,
            "pos_min": self.pos_min,
            "pos_max": self.pos_max,
            "seed": self.seed,
        }

    def extra_repr(self) -> str:
        return (
            f"n_cells={self.n_cells}, stdev={self.stdev}, "
            f"pos_min={self.pos_min}, pos_max={self.pos_max}, seed={self.seed}"
        )

    def forward(self, positions: th.Tensor) -> th.Tensor:
        positions = positions.float()
        if positions.shape[-1] < 2:
            raise ValueError(f"positions must have at least 2 coordinates, got shape {tuple(positions.shape)}")

        xy = positions[..., :2]
        diff = xy.unsqueeze(-2) - self.centers.to(device=xy.device, dtype=xy.dtype)
        logits = -0.5 * diff.square().sum(dim=-1) / (self.stdev**2)
        return F.softmax(logits, dim=-1)


class PathIntegrationHead(nn.Module):
    """Auxiliary bottleneck and place-cell decoder branching from actor LSTM states."""

    def __init__(
        self,
        lstm_output_dim: int,
        bottleneck_dim: int = 256,
        n_place_cells: int = 256,
        dropout_rate: float = 0.5,
        bottleneck_has_bias: bool = False,
    ) -> None:
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if n_place_cells <= 0:
            raise ValueError("n_place_cells must be positive")

        self.bottleneck = nn.Linear(lstm_output_dim, bottleneck_dim, bias=bottleneck_has_bias)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pc_head = nn.Linear(bottleneck_dim, n_place_cells)

    def forward(self, latent_pi: th.Tensor) -> PathIntegrationOutputs:
        bottleneck = self.bottleneck(latent_pi)
        pc_logits = self.pc_head(self.dropout(bottleneck))
        return PathIntegrationOutputs(pc_logits=pc_logits, bottleneck=bottleneck)


def soft_place_cell_cross_entropy(pc_logits: th.Tensor, pc_targets: th.Tensor, mask: th.Tensor | None = None) -> th.Tensor:
    if pc_logits.shape != pc_targets.shape:
        raise ValueError(f"pc_logits shape {tuple(pc_logits.shape)} != pc_targets shape {tuple(pc_targets.shape)}")

    per_step = -(pc_targets * F.log_softmax(pc_logits, dim=-1)).sum(dim=-1)
    if mask is None:
        return per_step.mean()

    mask = mask.bool()
    if mask.shape != per_step.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} != loss shape {tuple(per_step.shape)}")
    if not th.any(mask):
        return per_step.mean() * 0.0
    return per_step[mask].mean()
