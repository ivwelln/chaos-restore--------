from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, residual_ch: int, skip_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.filter_conv = nn.Conv1d(residual_ch, residual_ch, kernel, padding=pad, dilation=dilation)
        self.gate_conv = nn.Conv1d(residual_ch, residual_ch, kernel, padding=pad, dilation=dilation)
        self.residual_conv = nn.Conv1d(residual_ch, residual_ch, 1)
        self.skip_conv = nn.Conv1d(residual_ch, skip_ch, 1)

    def forward(self, x: torch.Tensor):
        z = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        # деление на √2 сохраняет дисперсию при сложении с residual-ветвью
        return (x + self.residual_conv(z)) / 2.0**0.5, self.skip_conv(z)


class WaveNet1D(nn.Module):
    """Стек residual-блоков с дилатационными свёртками (некаузальный вариант WaveNet)."""

    def __init__(
        self,
        residual_ch: int = 48,
        skip_ch: int = 96,
        kernel: int = 3,
        dilations: List[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv1d(1, residual_ch, kernel, padding=kernel // 2)
        self.blocks = nn.ModuleList(
            [ResidualBlock(residual_ch, skip_ch, kernel, d) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_ch, skip_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(x)
        skips = 0.0
        for block in self.blocks:
            h, skip = block(h)
            skips = skips + skip
        return self.head(skips)
