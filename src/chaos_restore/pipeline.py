from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .baselines import Baseline, fit_baselines
from .channel import ChannelOutput, apply_channel
from .config import Config
from .datasets import PreparedData, prepare
from .lorenz import LorenzData, get_signal
from .metrics import evaluate, first_minimum_lag, lyapunov
from .models import HUMAN_NAMES
from .train import TrainResult, restore_signal, train_model
from .utils import set_seed

LYAP_REFERENCE = 0.906  # табличное значение λ₁ для системы Лоренца (σ=10, ρ=28, β=8/3)


@dataclass
class Case:
    cfg: Config
    signal: LorenzData
    channel: ChannelOutput
    data: PreparedData

    @property
    def dt(self) -> float:
        return self.signal.dt

    @property
    def fs(self) -> float:
        return self.signal.fs

    @property
    def crop(self) -> int:
        return self.cfg.windows.length

    @property
    def variable(self) -> str:
        return self.cfg.lorenz.variable

    def test_clean(self) -> np.ndarray:
        return self._cropped(self.data.splits.test.slice(self.data.clean))

    def test_distorted(self) -> np.ndarray:
        return self._cropped(self.data.splits.test.slice(self.data.distorted))

    def _cropped(self, x: np.ndarray) -> np.ndarray:
        c = self.crop
        return x[c:-c] if 2 * c < x.size else x

    def crop_like_test(self, x: np.ndarray) -> np.ndarray:
        return self._cropped(np.asarray(x, dtype=np.float64).ravel())


def build_case(cfg: Config, verbose: bool = True, seed: bool = True) -> Case:
    """Шаги 1–3 одной командой: сгенерировать, испортить, подготовить."""
    if seed:
        set_seed(cfg.seed)
    signal = get_signal(cfg, verbose=verbose)
    source = signal.series(cfg.lorenz.variable)
    channel = apply_channel(source, cfg.channel, signal.fs)
    data = prepare(source, channel.distorted, cfg)
    if verbose:
        spans = data.splits.as_dict()
        print(
            f"[данные] сигнал {cfg.lorenz.variable}(t), {len(signal)} отсчётов; "
            f"фактический SNR {channel.snr_db_actual:.2f} дБ; "
            f"обучение {spans['train']}, валидация {spans['val']}, тест {spans['test']}"
        )
    return data_case(cfg, signal, channel, data)


def data_case(cfg: Config, signal: LorenzData, channel: ChannelOutput, data: PreparedData) -> Case:
    return Case(cfg=cfg, signal=signal, channel=channel, data=data)


def embedding_lag(case: Case, max_lag: int = 200) -> int:
    return first_minimum_lag(case.test_clean()[:20000], max_lag=max_lag)


def evaluate_signal(
    case: Case,
    x_hat: np.ndarray,
    method: str,
    with_lyap: bool = True,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row = evaluate(
        case.test_clean(),
        case.crop_like_test(x_hat),
        dt=case.dt,
        lyap_cfg=case.cfg.lyapunov,
        method=method,
        with_lyap=with_lyap,
    )
    if extra:
        row.update(extra)
    return row


def reference_rows(case: Case, with_lyap: bool = True) -> List[Dict[str, object]]:
    clean = case.test_clean()
    rows: List[Dict[str, object]] = []
    row_clean: Dict[str, object] = {
        "method": "эталон",
        "mse": 0.0,
        "rmse": 0.0,
        "nrmse": 0.0,
        "corr": 1.0,
        "snr_out_db": float("inf"),
    }
    if with_lyap:
        row_clean["lyap"] = lyapunov(clean, case.dt, case.cfg.lyapunov)
    rows.append(row_clean)
    rows.append(evaluate_signal(case, case.test_distorted(), "искажённый", with_lyap))
    return rows


def run_baselines(
    case: Case, with_lyap: bool = True, verbose: bool = True
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray], Dict[str, Baseline]]:
    """Шаг 5: подобрать простые методы на обучении, применить к тесту."""
    fitted = fit_baselines(case.data, case.channel.b, case.channel.a, case.fs, case.cfg)
    test_distorted = case.data.splits.test.slice(case.data.distorted)
    rows, restored = [], {}
    for name, baseline in fitted.items():
        x_hat = baseline(test_distorted)
        restored[name] = x_hat
        rows.append(
            evaluate_signal(
                case, x_hat, name, with_lyap, extra={"params": baseline.describe(), "shift": baseline.shift}
            )
        )
        if verbose:
            print(f"[baseline] {baseline.describe()}: NRMSE = {rows[-1]['nrmse']:.4f}")
    return rows, restored, fitted


def run_network(
    case: Case,
    out_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    with_lyap: bool = True,
    verbose: bool = True,
    tag: str = "",
) -> Tuple[Dict[str, object], np.ndarray, TrainResult]:
    """Шаг 6: обучить сеть и восстановить тестовую часть."""
    name = model_name or case.cfg.train.model
    result = train_model(case.data, case.cfg, out_dir=out_dir, model_name=name, verbose=verbose, tag=tag)
    x_hat = restore_signal(result.model, case.data, case.data.splits.test, case.cfg)
    row = evaluate_signal(
        case,
        x_hat,
        HUMAN_NAMES.get(name, name),
        with_lyap,
        extra={
            "params": f"{result.n_params} весов",
            "n_params": result.n_params,
            "epochs": result.epochs_run,
            "best_epoch": result.best_epoch,
            "val_loss": result.best_val_loss,
            "seconds": result.seconds,
            "sec_per_epoch": result.seconds_per_epoch,
        },
    )
    if verbose:
        print(f"[сеть] {name}: NRMSE = {row['nrmse']:.4f}, корреляция = {row['corr']:.4f}")
    return row, x_hat, result
