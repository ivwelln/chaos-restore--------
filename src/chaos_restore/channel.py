from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import bessel, butter, cheby1, freqz, lfilter

from .config import ChannelCfg, FilterCfg


@dataclass
class ChannelOutput:
    clean: np.ndarray
    filtered: np.ndarray
    distorted: np.ndarray
    noise: np.ndarray
    snr_db_actual: float
    b: np.ndarray
    a: np.ndarray


def design_filter(cfg: FilterCfg, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Проектирует ФНЧ. fc задана в 1/t, scipy ждёт нормировку на частоту Найквиста."""
    wn = cfg.fc / (fs / 2.0)
    if not 0.0 < wn < 1.0:
        raise ValueError(
            f"частота среза {cfg.fc} вне диапазона: должна быть в (0, {fs / 2}) при fs={fs}"
        )
    if cfg.kind == "butter":
        b, a = butter(cfg.order, wn, btype="low")
    elif cfg.kind == "bessel":
        # norm='mag': срез по уровню -3 дБ, как у Баттерворта, иначе фильтры несравнимы по полосе
        b, a = bessel(cfg.order, wn, btype="low", norm="mag")
    elif cfg.kind == "cheby1":
        b, a = cheby1(cfg.order, cfg.ripple_db, wn, btype="low")
    else:
        raise ValueError(f"неизвестный тип фильтра: {cfg.kind}")
    return np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)


def lowpass(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    return lfilter(b, a, x)


def signal_power(x: np.ndarray) -> float:
    return float(np.var(x))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    p_noise = signal_power(noise)
    if p_noise <= 0:
        return float("inf")
    return 10.0 * np.log10(signal_power(signal) / p_noise)


def add_noise(
    x: np.ndarray, target_snr_db: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    sigma = np.sqrt(signal_power(x) / (10.0 ** (target_snr_db / 10.0)))
    noise = rng.normal(0.0, sigma, size=x.shape)
    return x + noise, noise


def apply_channel(
    x: np.ndarray, cfg: ChannelCfg, fs: float, seed: Optional[int] = None
) -> ChannelOutput:
    """Полная модель среды: x -> ФНЧ -> +шум."""
    b, a = design_filter(cfg.filter, fs)
    filtered = lowpass(x, b, a)
    rng = np.random.default_rng(cfg.noise_seed if seed is None else seed)
    distorted, noise = add_noise(filtered, cfg.snr_db, rng)
    return ChannelOutput(
        clean=x,
        filtered=filtered,
        distorted=distorted,
        noise=noise,
        snr_db_actual=snr_db(filtered, noise),
        b=b,
        a=a,
    )


def frequency_response(
    b: np.ndarray, a: np.ndarray, fs: float, n: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    w, h = freqz(b, a, worN=n, fs=fs)
    return w, h


def group_delay_samples(b: np.ndarray, a: np.ndarray, fs: float, f_max: Optional[float] = None) -> float:
    from scipy.signal import group_delay

    w, gd = group_delay((b, a), w=2048, fs=fs)
    if f_max is None:
        f_max = w[-1]
    mask = (w > 0) & (w <= f_max)
    return float(np.mean(gd[mask])) if mask.any() else 0.0
