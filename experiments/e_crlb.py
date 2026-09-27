from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import _SRC  # noqa: F401

from chaos_restore.config import Config
from chaos_restore.metrics import crlb, effective_n
from chaos_restore.models import HUMAN_NAMES
from chaos_restore.pipeline import LYAP_REFERENCE, build_case, run_baselines, run_network
from chaos_restore.plotting import plot_crlb, plot_curves, setup_style
from chaos_restore.utils import ensure_dir, print_table, save_table

DEFAULT_LENGTHS = [16, 24, 32, 48, 64, 128, 256]
DEFAULT_MODELS = ["cae", "wavenet"]


def _bound_columns(mse: float, sigma2: float, dt: float, length: int) -> Dict[str, float]:
    bound = crlb(sigma2, LYAP_REFERENCE, dt, length)
    return {
        "sigma2": sigma2,
        "crlb": bound,
        "mse_to_crlb": mse / bound,
        "n_eff": effective_n(mse, sigma2, LYAP_REFERENCE, dt),
    }


def run(
    cfg: Config,
    lengths: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    with_lyap: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    setup_style()
    lengths = sorted(lengths or DEFAULT_LENGTHS)
    models = models or DEFAULT_MODELS
    out_dir = ensure_dir(cfg.results_dir / "e_crlb")
    cfg.dump(out_dir / "config.yaml")

    base_case = build_case(cfg, verbose=verbose)
    base_rows, _, _ = run_baselines(base_case, with_lyap=False, verbose=verbose)
    best = min((r for r in base_rows if "прямой" not in str(r["method"])), key=lambda r: r["mse"])

    rows = []
    for length in lengths:
        case = build_case(cfg.copy_with(**{"windows.length": length}), verbose=False)
        sigma2 = case.noise_var
        for name in models:
            if verbose:
                print(f"\n===== {HUMAN_NAMES.get(name, name)}, L = {length} =====")
            net_row, _, _ = run_network(
                case, out_dir=ensure_dir(out_dir / "checkpoints"), model_name=name,
                with_lyap=with_lyap, verbose=verbose, tag=f"L{length}",
            )
            rows.append({**net_row, "model": name, "length": length,
                         **_bound_columns(net_row["mse"], sigma2, case.dt, length)})
    rows.append({"method": best["method"], "model": "baseline", "mse": best["mse"], "nrmse": best["nrmse"],
                 "n_eff": effective_n(best["mse"], base_case.noise_var, LYAP_REFERENCE, base_case.dt)})

    df = save_table(rows, out_dir / "metrics_crlb.csv")
    if verbose:
        print_table(
            df[["method", "length", "mse", "crlb", "mse_to_crlb", "n_eff"]],
            title="Шаг 11. MSE и граница Крамера–Рао (N = длина окна)",
        )

    nets = df[df["model"] != "baseline"]
    n = np.arange(lengths[0], lengths[-1] + 1)
    plot_crlb(
        n, crlb(float(nets["sigma2"].mean()), LYAP_REFERENCE, base_case.dt, n),
        {m: (g["length"].tolist(), g["mse"].tolist()) for m, g in nets.groupby("method", sort=False)},
        out_dir / "fig_crlb_mse.png",
        title="Шаг 11. MSE сети и граница Крамера–Рао",
        reference=best["mse"], reference_label="лучший простой метод",
    )
    plot_curves(
        lengths, {m: g["mse_to_crlb"].tolist() for m, g in nets.groupby("method", sort=False)},
        out_dir / "fig_crlb_ratio.png",
        xlabel="длина окна N, отсчётов", ylabel="MSE / crlb", logx=True, logy=True,
        reference=1.0, reference_label="MSE = граница",
        title="Шаг 11. Отношение MSE к границе Крамера–Рао",
    )
    return df
