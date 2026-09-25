from __future__ import annotations

from typing import Dict, Optional

import nolds
import numpy as np

from .config import LyapunovCfg


def mse(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean((x_true - x_hat) ** 2))


def rmse(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.sqrt(mse(x_true, x_hat)))


def nrmse(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """RMSE, нормированная на СКО эталона: безразмерная, сравнима между прогонами."""
    denom = float(np.std(x_true))
    return rmse(x_true, x_hat) / denom if denom > 1e-12 else float("inf")


def correlation(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    if np.std(x_true) < 1e-12 or np.std(x_hat) < 1e-12:
        return 0.0
    return float(np.corrcoef(x_true, x_hat)[0, 1])


def snr_out_db(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    err = x_hat - x_true
    p_err = float(np.var(err))
    if p_err <= 0:
        return float("inf")
    return 10.0 * np.log10(float(np.var(x_true)) / p_err)


def mutual_information(x: np.ndarray, lag: int, bins: int = 64) -> float:
    a, b = x[:-lag], x[lag:]
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pab = hist / hist.sum()
    pa = pab.sum(axis=1, keepdims=True)
    pb = pab.sum(axis=0, keepdims=True)
    nz = pab > 0
    return float(np.sum(pab[nz] * np.log(pab[nz] / (pa @ pb)[nz])))


def first_minimum_lag(x: np.ndarray, max_lag: int = 200, bins: int = 64) -> int:
    """Задержка τ по первому минимуму взаимной информации."""
    x = np.asarray(x, dtype=np.float64).ravel()
    max_lag = int(min(max_lag, x.size // 10))
    values = [mutual_information(x, lag, bins) for lag in range(1, max_lag + 1)]
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] <= values[i + 1]:
            return i + 1
    return int(np.argmin(values)) + 1


def lyapunov(x: np.ndarray, dt: float, cfg: LyapunovCfg) -> float:
    """Старший показатель Ляпунова λ₁ в единицах 1/t (ряд прореживается, tau = dt * subsample)."""
    x = np.asarray(x, dtype=np.float64).ravel()[:: max(1, cfg.subsample)]
    if cfg.series_len and x.size > cfg.series_len:
        x = x[: cfg.series_len]
    lag = cfg.lag if cfg.lag else first_minimum_lag(x)
    return float(
        nolds.lyap_r(
            x,
            emb_dim=cfg.emb_dim,
            lag=lag,
            min_tsep=cfg.min_tsep,
            tau=dt * max(1, cfg.subsample),
            trajectory_len=cfg.trajectory_len,
            fit=cfg.fit,
        )
    )


def evaluate(
    x_true: np.ndarray,
    x_hat: np.ndarray,
    dt: float,
    lyap_cfg: Optional[LyapunovCfg] = None,
    method: str = "",
    with_lyap: bool = True,
) -> Dict[str, float]:
    x_true = np.asarray(x_true, dtype=np.float64).ravel()
    x_hat = np.asarray(x_hat, dtype=np.float64).ravel()
    n = min(x_true.size, x_hat.size)
    x_true, x_hat = x_true[:n], x_hat[:n]

    row: Dict[str, float] = {
        "method": method,
        "mse": mse(x_true, x_hat),
        "rmse": rmse(x_true, x_hat),
        "nrmse": nrmse(x_true, x_hat),
        "corr": correlation(x_true, x_hat),
        "snr_out_db": snr_out_db(x_true, x_hat),
    }
    if with_lyap and lyap_cfg is not None:
        row["lyap"] = lyapunov(x_hat, dt, lyap_cfg)
    return row
