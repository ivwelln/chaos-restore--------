from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import (
    butter,
    correlate,
    correlation_lags,
    filtfilt,
    freqz,
    lfilter,
    savgol_filter,
    welch,
)

from .config import Config
from .datasets import PreparedData

MAX_SHIFT = 512


def moving_average(x: np.ndarray, size: int) -> np.ndarray:
    return uniform_filter1d(x, size=int(size), mode="reflect")


def savgol(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    window = int(window) | 1  # длина окна обязана быть нечётной
    polyorder = int(min(polyorder, window - 1))
    return savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp")


def zero_phase_lowpass(x: np.ndarray, fc: float, fs: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, fc / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def transfer_function(b: np.ndarray, a: np.ndarray, n: int) -> np.ndarray:
    _, h = freqz(b, a, worN=n, whole=True)
    return h


def direct_inverse(y: np.ndarray, b: np.ndarray, a: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Прямая инверсия X̂ = Y / H. Существует, чтобы показать усиление шума."""
    n = y.size
    h = transfer_function(b, a, n)
    h_safe = np.where(np.abs(h) < floor, floor, h)
    return np.real(np.fft.ifft(np.fft.fft(y) / h_safe))


def wiener_deconvolution(y: np.ndarray, b: np.ndarray, a: np.ndarray, lam: float) -> np.ndarray:
    """Регуляризованная инверсия со скалярным λ: X̂ = Y·conj(H) / (|H|² + λ)."""
    n = y.size
    h = transfer_function(b, a, n)
    return np.real(np.fft.ifft(np.fft.fft(y) * np.conj(h) / (np.abs(h) ** 2 + lam)))


@dataclass
class WienerPSD:
    """Винеровский фильтр с регуляризацией по СПМ: G(f) = conj(H) / (|H|² + P_n(f) / P_x(f))."""

    b: np.ndarray
    a: np.ndarray
    fs: float
    freqs: np.ndarray
    psd_signal: np.ndarray
    psd_noise: np.ndarray

    @classmethod
    def design(
        cls,
        b: np.ndarray,
        a: np.ndarray,
        fs: float,
        x_train: np.ndarray,
        y_train: np.ndarray,
        nperseg: int = 4096,
    ) -> "WienerPSD":
        nperseg = int(min(nperseg, x_train.size // 4))
        filtered = np.asarray(lfilter(b, a, x_train))
        noise = y_train - filtered
        freqs, psd_signal = welch(x_train, fs=fs, nperseg=nperseg)
        _, psd_noise = welch(noise, fs=fs, nperseg=nperseg)
        return cls(b=b, a=a, fs=fs, freqs=freqs, psd_signal=psd_signal, psd_noise=psd_noise)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        n = y.size
        grid = np.abs(np.fft.fftfreq(n, d=1.0 / self.fs))
        p_x = np.interp(grid, self.freqs, self.psd_signal)
        p_n = np.interp(grid, self.freqs, self.psd_noise)
        p_x = np.maximum(p_x, p_x.max() * 1e-12)
        h = transfer_function(self.b, self.a, n)
        gain = np.conj(h) / (np.abs(h) ** 2 + p_n / p_x)
        return np.real(np.fft.ifft(np.fft.fft(y) * gain))


@dataclass
class BlindInverse:
    """Слепой обратный фильтр: оценивает fc и порядок канала по СПМ искажённого сигнала, не зная b и a."""

    b: np.ndarray
    a: np.ndarray
    fs: float
    log_a: float
    slope_b: float
    psd_noise: float
    fc_est: float
    order_est: int

    @classmethod
    def design(cls, y_train: np.ndarray, fs: float, nperseg: int = 4096) -> "BlindInverse":
        from scipy.optimize import least_squares

        nperseg = int(min(nperseg, max(256, y_train.size // 4)))
        freqs, pyy = welch(np.asarray(y_train, dtype=np.float64), fs=fs, nperseg=nperseg)
        mask = freqs > 0
        freqs, pyy = freqs[mask], pyy[mask]
        log_pyy = np.log10(pyy)

        noise0 = float(np.median(pyy[freqs > 0.6 * freqs[-1]]))
        knee = freqs[np.argmax(pyy < 3.0 * noise0)] if np.any(pyy < 3.0 * noise0) else freqs[-1] / 4
        fc0 = float(max(knee * 0.8, freqs[1]))
        a0 = float(log_pyy[0])

        def model(p):
            log_a, slope, fc, order, log_noise = p
            h2 = 1.0 / (1.0 + (freqs / fc) ** (2.0 * order))
            return np.log10(10.0 ** (log_a - slope * freqs) * h2 + 10.0**log_noise)

        guess = [a0, 0.5, fc0, 4.0, np.log10(noise0)]
        bounds = (
            [a0 - 6, 0.0, freqs[1], 0.5, np.log10(noise0) - 3],
            [a0 + 6, 20.0, freqs[-1], 12.0, np.log10(noise0) + 3],
        )
        try:
            fit = least_squares(
                lambda p: model(p) - log_pyy, guess, bounds=bounds, max_nfev=2000
            ).x
        except Exception:
            fit = guess

        log_a, slope, fc, order, log_noise = fit
        order_int = int(np.clip(round(order), 1, 10))
        fc = float(np.clip(fc, 1e-6, 0.95 * fs / 2))
        b, a = butter(order_int, fc / (fs / 2.0), btype="low")
        return cls(
            b=np.asarray(b),
            a=np.asarray(a),
            fs=fs,
            log_a=float(log_a),
            slope_b=float(slope),
            psd_noise=float(10.0**log_noise),
            fc_est=fc,
            order_est=order_int,
        )

    def __call__(self, y: np.ndarray) -> np.ndarray:
        n = y.size
        grid = np.abs(np.fft.fftfreq(n, d=1.0 / self.fs))
        p_x = 10.0 ** (self.log_a - self.slope_b * grid)
        p_x = np.maximum(p_x, p_x.max() * 1e-12)
        h = transfer_function(self.b, self.a, n)
        gain = np.conj(h) / (np.abs(h) ** 2 + self.psd_noise / p_x)
        return np.real(np.fft.ifft(np.fft.fft(y) * gain))


def estimate_shift(x_hat: np.ndarray, x_true: np.ndarray, max_shift: int = MAX_SHIFT) -> int:
    """Сдвиг S по максимуму взаимной корреляции: ``apply_shift(x_hat, S)`` выравнивается по x_true."""
    a = np.asarray(x_hat, float) - np.mean(x_hat)
    c = np.asarray(x_true, float) - np.mean(x_true)
    corr = correlate(c, a, mode="full", method="fft")
    lags = correlation_lags(c.size, a.size, mode="full")
    mask = np.abs(lags) <= max_shift
    return -int(lags[mask][int(np.argmax(corr[mask]))])


def apply_shift(x: np.ndarray, shift: int) -> np.ndarray:
    """``shift>0`` – сигнал сдвигается вперёд во времени (задержка компенсируется)."""
    if shift == 0:
        return x.copy()
    out = np.roll(x, -shift)
    if shift > 0:
        out[-shift:] = x[-1]
    else:
        out[:-shift] = x[0]
    return out


@dataclass
class Baseline:
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    params: Dict[str, float] = field(default_factory=dict)
    shift: int = 0

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return apply_shift(self.func(np.asarray(y, dtype=np.float64)), self.shift)

    def describe(self) -> str:
        if not self.params:
            return self.name
        inner = ", ".join(f"{k}={v:g}" for k, v in self.params.items())
        return f"{self.name} ({inner})"


def _select(
    name: str,
    candidates: List[Tuple[Dict[str, float], Callable[[np.ndarray], np.ndarray]]],
    y_train: np.ndarray,
    x_train: np.ndarray,
) -> Baseline:
    best: Optional[Baseline] = None
    best_mse = np.inf
    for params, func in candidates:
        restored = func(y_train)
        shift = estimate_shift(restored, x_train)
        err = float(np.mean((apply_shift(restored, shift) - x_train) ** 2))
        if err < best_mse:
            best_mse, best = err, Baseline(name=name, func=func, params=params, shift=shift)
    assert best is not None
    return best


def fit_baselines(
    data: PreparedData,
    b: np.ndarray,
    a: np.ndarray,
    fs: float,
    cfg: Config,
    include_direct_inverse: bool = True,
) -> Dict[str, Baseline]:
    """Подбирает параметры всех простых методов на обучающей части."""
    y_train = data.splits.train.slice(data.distorted)
    x_train = data.splits.train.slice(data.clean)
    grids = cfg.baselines
    out: Dict[str, Baseline] = {}

    out["скользящее среднее"] = _select(
        "скользящее среднее",
        [({"size": k}, (lambda k=k: lambda y: moving_average(y, k))()) for k in grids.moving_average_grid],
        y_train,
        x_train,
    )

    savgol_candidates = [
        ({"window": w, "polyorder": p}, (lambda w=w, p=p: lambda y: savgol(y, w, p))())
        for w in grids.savgol_window_grid
        for p in grids.savgol_polyorder_grid
        if p < w
    ]
    out["Савицкий-Голей"] = _select("Савицкий-Голей", savgol_candidates, y_train, x_train)

    out["ФНЧ нулевой фазы"] = _select(
        "ФНЧ нулевой фазы",
        [
            ({"fc": fc}, (lambda fc=fc: lambda y: zero_phase_lowpass(y, fc, fs))())
            for fc in grids.filtfilt_fc_grid
            if 0 < fc < fs / 2
        ],
        y_train,
        x_train,
    )

    if include_direct_inverse:
        direct = Baseline(
            name="обратный фильтр (прямой)",
            func=lambda y: direct_inverse(y, b, a),
            params={},
        )
        direct.shift = estimate_shift(direct.func(y_train), x_train)
        out["обратный фильтр (прямой)"] = direct

    out["обратный фильтр (Винер)"] = _select(
        "обратный фильтр (Винер)",
        [
            ({"lambda": lam}, (lambda lam=lam: lambda y: wiener_deconvolution(y, b, a, lam))())
            for lam in grids.wiener_lambda_grid
        ],
        y_train,
        x_train,
    )

    out["обратный фильтр (Винер, СПМ)"] = _select(
        "обратный фильтр (Винер, СПМ)",
        [
            ({"nperseg": nperseg}, WienerPSD.design(b, a, fs, x_train, y_train, nperseg))
            for nperseg in (1024, 4096, 16384)
            if nperseg <= x_train.size // 4
        ],
        y_train,
        x_train,
    )

    # слепой метод не знает b, a и не подбирается по эталону, поэтому кандидат один
    blind = BlindInverse.design(y_train, fs)
    out["обратный фильтр (слепой)"] = Baseline(
        name="обратный фильтр (слепой)",
        func=blind,
        params={"fc_оценка": round(blind.fc_est, 3), "порядок_оценка": blind.order_est},
        shift=estimate_shift(blind(y_train), x_train),
    )
    return out
