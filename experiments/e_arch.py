from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from . import _SRC  # noqa: F401

from chaos_restore.config import Config
from chaos_restore.models import HUMAN_NAMES
from chaos_restore.pipeline import build_case, embedding_lag, run_network
from chaos_restore.plotting import plot_portraits, plot_spectra, plot_timeseries, setup_style
from chaos_restore.utils import ensure_dir, print_table, save_table

DEFAULT_MODELS = ["cae", "unet", "wavenet"]


def run(
    cfg: Config,
    models: Optional[List[str]] = None,
    seeds: int = 1,
    with_lyap: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    setup_style()
    models = models or DEFAULT_MODELS
    out_dir = ensure_dir(cfg.results_dir / "e_arch")
    cfg.dump(out_dir / "config.yaml")

    rows, restored = [], {}
    for run_idx in range(seeds):
        seed = cfg.seed + run_idx
        case = build_case(cfg.copy_with(seed=seed), verbose=verbose and run_idx == 0)
        for name in models:
            if verbose:
                print(f"\n===== {HUMAN_NAMES.get(name, name)}, seed = {seed} =====")
            net_row, x_hat, result = run_network(
                case, out_dir=ensure_dir(out_dir / "checkpoints"), model_name=name,
                with_lyap=with_lyap, verbose=verbose, tag=f"seed{seed}",
            )
            rows.append({**net_row, "model": name, "seed": seed})
            if run_idx == 0:
                restored[HUMAN_NAMES.get(name, name)] = x_hat
        last_case = case

    df = save_table(rows, out_dir / "metrics_arch.csv")

    agg = (
        df.groupby("model")
        .agg(
            n_params=("n_params", "first"),
            sec_per_epoch=("sec_per_epoch", "mean"),
            nrmse_mean=("nrmse", "mean"),
            nrmse_std=("nrmse", "std"),
            corr_mean=("corr", "mean"),
            **({"lyap_mean": ("lyap", "mean")} if "lyap" in df else {}),
        )
        .reset_index()
        .sort_values("nrmse_mean")
    )
    save_table(agg.to_dict("records"), out_dir / "metrics_arch_summary.csv")
    if verbose:
        print_table(agg, title="Шаг 9. Сравнение архитектур (среднее по seed'ам)")

    best_key = agg.iloc[0]["model"]
    best_name = HUMAN_NAMES.get(best_key, best_key)
    clean_t = last_case.test_clean()
    t = np.arange(clean_t.size) * last_case.dt
    series = {
        "эталон": clean_t,
        "искажённый": last_case.test_distorted(),
        f"восстановленный ({best_name})": last_case.crop_like_test(restored[best_name]),
    }
    n_show = min(clean_t.size, int(round(8 / last_case.dt)))
    plot_timeseries(t[:n_show], {k: v[:n_show] for k, v in series.items()},
                    out_dir / "fig_arch_time.png",
                    title=f"Шаг 9. Лучшая архитектура: {best_name}", xlim=(0, 8))
    plot_spectra(series, last_case.fs, out_dir / "fig_arch_spectra.png",
                 title=f"Шаг 9. Спектры, {best_name}", fc=cfg.channel.filter.fc, fmax=20)
    plot_portraits(series, lag=embedding_lag(last_case), path=out_dir / "fig_arch_portraits.png",
                   dt=last_case.dt)
    return df
