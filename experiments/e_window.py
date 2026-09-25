from __future__ import annotations

from typing import List, Optional

import pandas as pd

from . import _SRC  # noqa: F401

from chaos_restore.config import Config
from chaos_restore.pipeline import build_case, run_network
from chaos_restore.plotting import plot_curves, setup_style
from chaos_restore.utils import ensure_dir, print_table, save_table

DEFAULT_LENGTHS = [64, 128, 256, 512, 1024]


def run(
    cfg: Config,
    lengths: Optional[List[int]] = None,
    with_lyap: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    setup_style()
    lengths = lengths or DEFAULT_LENGTHS
    out_dir = ensure_dir(cfg.results_dir / "e_window")
    cfg.dump(out_dir / "config.yaml")

    rows = []
    for length in lengths:
        if verbose:
            print(f"\n===== L = {length} =====")
        case = build_case(cfg.copy_with(**{"windows.length": length}), verbose=verbose)
        net_row, _, result = run_network(
            case, out_dir=ensure_dir(out_dir / "checkpoints"), with_lyap=with_lyap,
            verbose=verbose, tag=f"L{length}",
        )
        rows.append({**net_row, "length": length, "sec_per_epoch": result.seconds_per_epoch})

    df = save_table(rows, out_dir / "metrics_window.csv")
    if verbose:
        print_table(
            df[["length", "nrmse", "corr", "epochs", "sec_per_epoch"] + (["lyap"] if "lyap" in df else [])],
            title="Шаг 8. Качество и время в зависимости от длины окна",
        )

    plot_curves(
        lengths, {"NRMSE": df["nrmse"].tolist()}, out_dir / "fig_window_nrmse.png",
        xlabel="длина окна L, отсчётов", ylabel="NRMSE",
        title="Шаг 8. Ошибка восстановления от длины окна", logx=True,
    )
    plot_curves(
        lengths, {"секунд на эпоху": df["sec_per_epoch"].tolist()}, out_dir / "fig_window_time.png",
        xlabel="длина окна L, отсчётов", ylabel="время эпохи, с",
        title="Шаг 8. Стоимость обучения от длины окна", logx=True,
    )
    return df
