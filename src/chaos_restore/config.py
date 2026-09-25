from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsCfg(_Base):
    data: str = "data"
    results: str = "results"


class LorenzCfg(_Base):
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01
    transient: float = 20.0
    duration: float = 4000.0
    x0: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    variable: Literal["x", "y", "z"] = "x"

    def trajectory_key(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if k != "variable"}

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    @property
    def n_samples(self) -> int:
        return int(round(self.duration / self.dt))


class FilterCfg(_Base):
    kind: Literal["butter", "bessel", "cheby1"] = "butter"
    order: int = 4
    fc: float = 3.0
    ripple_db: float = 1.0


class ChannelCfg(_Base):
    filter: FilterCfg = Field(default_factory=FilterCfg)
    snr_db: float = 10.0
    noise_seed: int = 0


class WindowsCfg(_Base):
    length: int = 256
    stride_div: int = 4
    infer_stride_div: int = 2

    @property
    def hop_train(self) -> int:
        return max(1, self.length // self.stride_div)

    @property
    def hop_infer(self) -> int:
        return max(1, self.length // self.infer_stride_div)


class SplitCfg(_Base):
    train_frac: float = 0.8
    val_frac: float = 0.1


class TrainCfg(_Base):
    model: Literal["cae", "unet", "wavenet"] = "cae"
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0
    num_workers: int = 0
    device: str = "auto"


class LyapunovCfg(_Base):
    subsample: int = 10
    series_len: int = 6000
    emb_dim: int = 8
    lag: Optional[int] = 2
    min_tsep: Optional[int] = 20
    trajectory_len: int = 20
    fit: str = "poly"


class BaselinesCfg(_Base):
    moving_average_grid: List[int] = Field(
        default_factory=lambda: [3, 5, 7, 9, 13, 17, 21, 31, 41, 61]
    )
    savgol_window_grid: List[int] = Field(
        default_factory=lambda: [7, 11, 15, 21, 31, 41, 61, 81]
    )
    savgol_polyorder_grid: List[int] = Field(default_factory=lambda: [2, 3, 4])
    filtfilt_fc_grid: List[float] = Field(
        default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    )
    wiener_lambda_grid: List[float] = Field(
        default_factory=lambda: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0]
    )


class Config(_Base):
    name: str = "base"
    seed: int = 42
    paths: PathsCfg = Field(default_factory=PathsCfg)
    lorenz: LorenzCfg = Field(default_factory=LorenzCfg)
    channel: ChannelCfg = Field(default_factory=ChannelCfg)
    windows: WindowsCfg = Field(default_factory=WindowsCfg)
    split: SplitCfg = Field(default_factory=SplitCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    lyapunov: LyapunovCfg = Field(default_factory=LyapunovCfg)
    baselines: BaselinesCfg = Field(default_factory=BaselinesCfg)

    @property
    def data_dir(self) -> Path:
        return _as_path(self.paths.data)

    @property
    def results_dir(self) -> Path:
        return _as_path(self.paths.results) / self.name

    @model_validator(mode="after")
    def _check(self) -> "Config":
        if not 0.5 <= self.split.train_frac < 1.0:
            raise ValueError("split.train_frac должен быть в [0.5, 1.0)")
        if not 0.0 < self.split.val_frac < 0.5:
            raise ValueError("split.val_frac должен быть в (0.0, 0.5)")
        if self.windows.length < 16:
            raise ValueError("windows.length слишком мал")
        if self.lorenz.n_samples < 10 * self.windows.length:
            raise ValueError(
                "слишком короткая траектория для выбранной длины окна: "
                f"{self.lorenz.n_samples} отсчётов при L={self.windows.length}"
            )
        return self

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.model_dump(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def copy_with(self, **overrides: Any) -> "Config":
        raw = self.model_dump()
        for dotted, value in overrides.items():
            _set_dotted(raw, dotted, value)
        return Config.model_validate(raw)


def _as_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _set_dotted(raw: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = raw
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"неизвестный параметр конфига: {dotted}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"неизвестный параметр конфига: {dotted}")
    node[keys[-1]] = value


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def load_config(path: Optional[str | Path] = None, overrides: Optional[List[str]] = None) -> Config:
    """Читает yaml и применяет переопределения вида ``ключ.подключ=значение``."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = copy.deepcopy(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"переопределение должно быть вида ключ=значение, получено: {item!r}")
        dotted, _, text = item.partition("=")
        _set_dotted(raw, dotted.strip(), _parse_scalar(text.strip()))
    return Config.model_validate(raw)
