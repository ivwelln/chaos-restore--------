from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal.windows import hann


@dataclass
class Frames:
    values: np.ndarray
    starts: np.ndarray
    n_samples: int

    @property
    def n_frames(self) -> int:
        return self.values.shape[0]

    @property
    def length(self) -> int:
        return self.values.shape[1]


def frame_signal(x: np.ndarray, length: int, hop: int, pad: bool = True) -> Frames:
    """Режет ряд на окна длины ``length`` с шагом ``hop``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if length > n:
        raise ValueError(f"длина окна {length} больше длины сигнала {n}")
    if hop < 1:
        raise ValueError("hop должен быть >= 1")

    if pad:
        n_frames = int(np.ceil(max(n - length, 0) / hop)) + 1
        need = (n_frames - 1) * hop + length
        if need > n:
            x = np.pad(x, (0, need - n), mode="reflect")
    else:
        n_frames = (n - length) // hop + 1

    starts = np.arange(n_frames, dtype=np.int64) * hop
    values = np.lib.stride_tricks.sliding_window_view(x, length)[starts].copy()
    return Frames(values=values, starts=starts, n_samples=n)


def overlap_add(
    frames: Frames,
    hop: int,
    window: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Собирает ряд из окон обратно (WOLA) и обрезает до исходной длины."""
    values, starts = frames.values, frames.starts
    n_frames, length = values.shape
    total = int(starts[-1]) + length

    if window is None:
        window = hann(length, sym=False)
    window = np.asarray(window, dtype=np.float64)
    if window.size != length:
        raise ValueError("длина весового окна не совпадает с длиной кадра")

    num = np.zeros(total, dtype=np.float64)
    den = np.zeros(total, dtype=np.float64)
    for i in range(n_frames):
        s = int(starts[i])
        num[s : s + length] += values[i] * window
        den[s : s + length] += window

    out = np.zeros(total, dtype=np.float64)
    good = den > eps
    out[good] = num[good] / den[good]

    # на краях веса окна Ханна нулевые – там берём невзвешенное среднее окон
    if not good.all():
        num_u = np.zeros(total, dtype=np.float64)
        den_u = np.zeros(total, dtype=np.float64)
        for i in range(n_frames):
            s = int(starts[i])
            num_u[s : s + length] += values[i]
            den_u[s : s + length] += 1.0
        bad = ~good & (den_u > 0)
        out[bad] = num_u[bad] / den_u[bad]

    return out[: frames.n_samples]


def reconstruct(values: np.ndarray, starts: np.ndarray, n_samples: int, hop: int) -> np.ndarray:
    return overlap_add(Frames(values=np.asarray(values, dtype=np.float64), starts=np.asarray(starts), n_samples=n_samples), hop)


def check_roundtrip(x: np.ndarray, length: int, hop: int) -> float:
    frames = frame_signal(x, length, hop)
    back = overlap_add(frames, hop)
    return float(np.max(np.abs(back - np.asarray(x, dtype=np.float64).ravel())))
