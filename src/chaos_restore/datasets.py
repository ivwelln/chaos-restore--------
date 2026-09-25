from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import Config
from .windows import frame_signal


@dataclass(frozen=True)
class Span:
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start

    def slice(self, x: np.ndarray) -> np.ndarray:
        return x[self.start : self.stop]


@dataclass(frozen=True)
class Splits:
    train: Span
    val: Span
    test: Span

    def as_dict(self) -> Dict[str, Tuple[int, int]]:
        return {
            "train": (self.train.start, self.train.stop),
            "val": (self.val.start, self.val.stop),
            "test": (self.test.start, self.test.stop),
        }


def make_splits(n_samples: int, cfg: Config) -> Splits:
    """Границы обучения / валидации / теста по времени."""
    n_fit = int(round(n_samples * cfg.split.train_frac))
    n_val = int(round(n_fit * cfg.split.val_frac))
    n_train = n_fit - n_val
    min_len = cfg.windows.length
    if min(n_train, n_val, n_samples - n_fit) < min_len:
        raise ValueError(
            "слишком короткий сигнал: одна из частей короче окна "
            f"L={min_len} (train={n_train}, val={n_val}, test={n_samples - n_fit})"
        )
    return Splits(
        train=Span(0, n_train),
        val=Span(n_train, n_fit),
        test=Span(n_fit, n_samples),
    )


@dataclass
class Normalizer:
    """z-score с сохранением статистик обучающей части."""

    mean: float
    std: float

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        std = float(np.std(x))
        return cls(mean=float(np.mean(x)), std=std if std > 1e-12 else 1.0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass
class PreparedData:
    clean: np.ndarray
    distorted: np.ndarray
    splits: Splits
    norm_in: Normalizer
    norm_out: Normalizer
    length: int
    hop_train: int
    hop_infer: int


def prepare(clean: np.ndarray, distorted: np.ndarray, cfg: Config) -> PreparedData:
    splits = make_splits(clean.size, cfg)
    return PreparedData(
        clean=np.asarray(clean, dtype=np.float64),
        distorted=np.asarray(distorted, dtype=np.float64),
        splits=splits,
        norm_in=Normalizer.fit(splits.train.slice(distorted)),
        norm_out=Normalizer.fit(splits.train.slice(clean)),
        length=cfg.windows.length,
        hop_train=cfg.windows.hop_train,
        hop_infer=cfg.windows.hop_infer,
    )


def _torch():
    import torch

    return torch


class WindowDataset:
    def __init__(
        self,
        data: PreparedData,
        span: Span,
        hop: int,
    ) -> None:
        torch = _torch()
        x = data.norm_in.transform(span.slice(data.distorted))
        y = data.norm_out.transform(span.slice(data.clean))
        fx = frame_signal(x, data.length, hop)
        fy = frame_signal(y, data.length, hop)
        self.x = torch.from_numpy(fx.values[:, None, :].astype(np.float32))
        self.y = torch.from_numpy(fy.values[:, None, :].astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def make_loaders(data: PreparedData, cfg: Config):
    _torch()
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, inner: WindowDataset) -> None:
            self.inner = inner

        def __len__(self) -> int:
            return len(self.inner)

        def __getitem__(self, idx):
            return self.inner[idx]

    train_ds = _DS(WindowDataset(data, data.splits.train, data.hop_train))
    val_ds = _DS(WindowDataset(data, data.splits.val, data.hop_train))
    common = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, shuffle=False, drop_last=False, **common),
    )
