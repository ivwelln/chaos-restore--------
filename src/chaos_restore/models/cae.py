from __future__ import annotations

import torch
import torch.nn as nn


def _block(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


def _up_block(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_ch,
            out_ch,
            kernel_size=kernel,
            stride=2,
            padding=kernel // 2,
            output_padding=1,
        ),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """1D-свёрточный автоэнкодер «искажённое окно → чистое окно»."""

    def __init__(self, channels=(32, 64, 128), kernel: int = 7) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.encoder = nn.Sequential(
            _block(1, c1, kernel, stride=1),   # L
            _block(c1, c2, kernel, stride=2),  # L/2
            _block(c2, c3, kernel, stride=2),  # L/4
        )
        self.decoder = nn.Sequential(
            _up_block(c3, c2, kernel),  # L/2
            _up_block(c2, c1, kernel),  # L
            nn.Conv1d(c1, 1, kernel_size=kernel, padding=kernel // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
