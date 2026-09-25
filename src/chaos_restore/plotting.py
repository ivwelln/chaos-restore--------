from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

from .lorenz import delay_embedding  # noqa: E402
from .utils import ensure_dir  # noqa: E402

COLORS = {
    "clean": "#1a1a1a",
    "filtered": "#8c8c8c",
    "distorted": "#e07b39",
    "baseline": "#4f8a5b",
    "restored": "#2f6fb0",
    "noise": "#c4c4c4",
    "accent": "#a3312f",
}
DEFAULT_DPI = 150


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DEFAULT_DPI,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.1,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def _save(fig, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    return path


def _color(name: str) -> str:
    return COLORS.get(name, None) or COLORS["accent"]


def plot_timeseries(
    t: np.ndarray,
    series: Dict[str, np.ndarray],
    path: Path,
    title: str = "",
    xlim: Optional[Sequence[float]] = None,
    ylabel: str = "x(t)",
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for name, values in series.items():
        n = min(len(t), len(values))
        ax.plot(t[:n], values[:n], label=name, color=_color(_role(name)), alpha=0.9)
    ax.set_xlabel("время, ед.")
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    if title:
        ax.set_title(title)
    ax.legend(
        ncols=min(len(series), 2 if max(map(len, series)) > 22 else 4),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
    )
    return _save(fig, path)


def _role(name: str) -> str:
    low = name.lower()
    for key, role in (
        ("эталон", "clean"),
        ("чист", "clean"),
        ("шум", "noise"),
        ("после фнч", "filtered"),
        ("искаж", "distorted"),
        ("сеть", "restored"),
        ("автоэнкодер", "restored"),
        ("u-net", "restored"),
        ("wavenet", "restored"),
        ("простой", "baseline"),
        ("средн", "baseline"),
        ("голей", "baseline"),
        ("фнч нулевой", "baseline"),
        ("обратный фильтр", "baseline"),
    ):
        if key in low:
            return role
    return "accent"


def plot_spectra(
    series: Dict[str, np.ndarray],
    fs: float,
    path: Path,
    title: str = "",
    fc: Optional[float] = None,
    nperseg: int = 4096,
    fmax: Optional[float] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for name, values in series.items():
        f, pxx = welch(np.asarray(values, dtype=np.float64), fs=fs, nperseg=nperseg)
        ax.semilogy(f, pxx, label=name, color=_color(_role(name)), alpha=0.9)
    if fc is not None:
        ax.axvline(fc, color=COLORS["accent"], ls="--", lw=1.0, label=f"fc = {fc:g}")
    ax.set_xlabel("частота, 1/ед. времени")
    ax.set_ylabel("СПМ")
    ax.set_xlim(0, fmax if fmax else fs / 2)
    if title:
        ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_attractor(x: np.ndarray, z: np.ndarray, path: Path, title: str = "Аттрактор (x, z)") -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(x, z, lw=0.3, color=COLORS["clean"], alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    return _save(fig, path)


def plot_portraits(
    series: Dict[str, np.ndarray],
    lag: int,
    path: Path,
    dt: float = 1.0,
    max_points: int = 20000,
) -> Path:
    fig, axes = plt.subplots(1, len(series), figsize=(4.0 * len(series), 3.9), squeeze=False)
    lim = None
    for ax, (name, values) in zip(axes[0], series.items()):
        emb = delay_embedding(np.asarray(values, dtype=np.float64).ravel()[:max_points], lag, 2)
        ax.plot(emb[:, 0], emb[:, 1], lw=0.25, color=_color(_role(name)), alpha=0.75)
        ax.set_title(name)
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t + {lag * dt:.2f})")
        span = np.abs(emb).max() * 1.05
        lim = span if lim is None else max(lim, span)
    for ax in axes[0]:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return _save(fig, path)


def plot_training_curves(history, path: Path, title: str = "Кривые обучения") -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    epochs = [h["epoch"] for h in history]
    ax.semilogy(epochs, [h["train_loss"] for h in history], label="обучение", color=COLORS["restored"])
    ax.semilogy(epochs, [h["val_loss"] for h in history], label="валидация", color=COLORS["accent"])
    best = min(history, key=lambda h: h["val_loss"])
    ax.axvline(best["epoch"], color="gray", ls="--", lw=1.0, label=f"лучшая эпоха: {best['epoch']}")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("MSE (нормированные величины)")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_bars(
    values: Dict[str, float],
    path: Path,
    ylabel: str,
    title: str = "",
    reference: Optional[float] = None,
    reference_label: str = "эталон",
) -> Path:
    fig, ax = plt.subplots(figsize=(min(1.35 * len(values) + 2.2, 10.5), 3.8))
    names = list(values)
    ax.bar(names, [values[n] for n in names], color=[_color(_role(n)) for n in names], width=0.6)
    if reference is not None:
        ax.axhline(reference, color=COLORS["accent"], ls="--", lw=1.2, label=reference_label)
        ax.legend()
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    return _save(fig, path)


def plot_curves(
    x: Sequence[float],
    series: Dict[str, Sequence[float]],
    path: Path,
    xlabel: str,
    ylabel: str,
    title: str = "",
    logx: bool = False,
    logy: bool = False,
    reference: Optional[float] = None,
    reference_label: str = "эталон",
    markers: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (name, values) in enumerate(series.items()):
        ax.plot(
            x,
            values,
            marker="o" if markers else None,
            ms=4,
            label=name,
            color=plt.cm.tab10(i % 10),
        )
    if reference is not None:
        ax.axhline(reference, color=COLORS["accent"], ls="--", lw=1.2, label=reference_label)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(x))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    n_entries = len(series) + (1 if reference is not None else 0)
    if n_entries > 4:
        ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    else:
        ax.legend()
    return _save(fig, path)


def plot_hbar_table(
    labels: Iterable[str],
    values: Iterable[float],
    path: Path,
    xlabel: str,
    title: str = "",
    reference: Optional[float] = None,
) -> Path:
    labels = list(labels)
    values = list(values)
    fig, ax = plt.subplots(figsize=(7.0, 0.38 * len(labels) + 1.8))
    ax.barh(labels, values, color=COLORS["restored"], height=0.6)
    if reference is not None:
        ax.axvline(reference, color=COLORS["accent"], ls="--", lw=1.2, label="обучающий фильтр")
        ax.legend()
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)
