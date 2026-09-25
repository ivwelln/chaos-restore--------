from __future__ import annotations

import torch
import torch.nn as nn


def _conv(in_ch: int, out_ch: int, kernel: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


def _up(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=kernel, stride=2, padding=kernel // 2, output_padding=1
        ),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet1D(nn.Module):
    """1D U-Net: тот же кодер-декодер, что у автоэнкодера, плюс skip-связи между уровнями."""

    def __init__(self, base: int = 32, kernel: int = 7, merge_kernel: int = 3) -> None:
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.enc1 = _conv(1, c1, kernel)              # L
        self.enc2 = _conv(c1, c2, kernel, stride=2)   # L/2
        self.enc3 = _conv(c2, c3, kernel, stride=2)   # L/4, bottleneck

        self.up2 = _up(c3, c2, kernel)                # L/2
        self.merge2 = _conv(c2 * 2, c2, merge_kernel)
        self.up1 = _up(c2, c1, kernel)                # L
        self.merge1 = _conv(c1 * 2, c1, merge_kernel)
        self.head = nn.Conv1d(c1, 1, kernel_size=kernel, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.merge2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.merge1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
