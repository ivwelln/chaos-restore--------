from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint

from .config import Config, LorenzCfg
from .utils import ensure_dir, params_hash


@dataclass
class LorenzData:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dt: float

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    def series(self, name: str = "x") -> np.ndarray:
        if name not in ("x", "y", "z"):
            raise ValueError(f"переменной {name!r} в системе Лоренца нет")
        return getattr(self, name)

    def __len__(self) -> int:
        return self.x.size


def lorenz_rhs(state: np.ndarray, t: float, sigma: float, rho: float, beta: float) -> list:
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def simulate(cfg: LorenzCfg) -> LorenzData:
    """Интегрирует систему и отбрасывает переходный процесс."""
    n_transient = int(round(cfg.transient / cfg.dt))
    n_total = n_transient + cfg.n_samples
    t_full = np.arange(n_total, dtype=np.float64) * cfg.dt

    # ошибка интегрирования хаотической системы растёт экспоненциально
    sol = odeint(
        lorenz_rhs,
        np.asarray(cfg.x0, dtype=np.float64),
        t_full,
        args=(cfg.sigma, cfg.rho, cfg.beta),
        rtol=1e-10,
        atol=1e-12,
    )

    sol = sol[n_transient:]
    t = t_full[n_transient:] - t_full[n_transient]
    return LorenzData(t=t, x=sol[:, 0].copy(), y=sol[:, 1].copy(), z=sol[:, 2].copy(), dt=cfg.dt)


def get_signal(cfg: Config, cache: bool = True, verbose: bool = True) -> LorenzData:
    """Эталонный сигнал с кэшированием на диск."""
    key = params_hash(cfg.lorenz.trajectory_key())
    path = cfg.data_dir / f"lorenz_{key}.npz"

    if cache and path.exists():
        with np.load(path) as blob:
            return LorenzData(
                t=blob["t"], x=blob["x"], y=blob["y"], z=blob["z"], dt=float(blob["dt"])
            )

    if verbose:
        print(f"[lorenz] интегрирование {cfg.lorenz.n_samples} отсчётов (dt={cfg.lorenz.dt})…")
    data = simulate(cfg.lorenz)

    if cache:
        ensure_dir(path.parent)
        np.savez_compressed(path, t=data.t, x=data.x, y=data.y, z=data.z, dt=data.dt)
    return data


def delay_embedding(x: np.ndarray, lag: int, dim: int = 2) -> np.ndarray:
    """Задержечное вложение для фазового портрета: точки (x[i], x[i+lag], …, x[i+(dim-1)*lag])."""
    if lag < 1:
        raise ValueError("lag должен быть >= 1")
    n = x.size - (dim - 1) * lag
    if n <= 0:
        raise ValueError("ряд короче, чем требуется для вложения")
    return np.stack([x[i * lag : i * lag + n] for i in range(dim)], axis=1)
