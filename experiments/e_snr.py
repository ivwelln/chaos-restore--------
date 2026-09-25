from __future__ import annotations

from typing import List, Optional

import pandas as pd

from . import _SRC  # noqa: F401

from chaos_restore.config import Config
from chaos_restore.pipeline import (
    LYAP_REFERENCE,
    build_case,
    evaluate_signal,
    run_baselines,
    run_network,
)
from chaos_restore.plotting import plot_curves, setup_style
from chaos_restore.utils import ensure_dir, print_table, save_table

DEFAULT_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]


def run(
    cfg: Config,
    snr_list: Optional[List[float]] = None,
    with_lyap: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    setup_style()
    snr_list = snr_list or DEFAULT_SNRS
    out_dir = ensure_dir(cfg.results_dir / "e_snr")
    cfg.dump(out_dir / "config.yaml")

    rows = []
    for snr in snr_list:
        if verbose:
            print(f"\n===== SNR = {snr:g} дБ =====")
        case = build_case(cfg.copy_with(**{"channel.snr_db": snr}), verbose=verbose)

        rows.append({**evaluate_signal(case, case.test_distorted(), "искажённый", with_lyap), "snr_db": snr})
        base_rows, _, _ = run_baselines(case, with_lyap=with_lyap, verbose=verbose)
        rows += [{**r, "snr_db": snr} for r in base_rows]
        net_row, _, _ = run_network(
            case, out_dir=ensure_dir(out_dir / "checkpoints"), with_lyap=with_lyap,
            verbose=verbose, tag=f"snr{snr:g}",
        )
        rows.append({**net_row, "snr_db": snr})

    df = save_table(rows, out_dir / "metrics_snr.csv")
    if verbose:
        print_table(
            df.pivot_table(index="snr_db", columns="method", values="nrmse").reset_index(),
            title="Шаг 7. NRMSE в зависимости от SNR",
        )

    methods = [m for m in df["method"].unique() if "прямой" not in str(m)]
    plot_curves(
        snr_list,
        {m: df[df["method"] == m].set_index("snr_db").loc[snr_list, "nrmse"].tolist() for m in methods},
        out_dir / "fig_snr_nrmse.png",
        xlabel="SNR, дБ", ylabel="NRMSE (лог. шкала)", logy=True,
        title="Шаг 7. Ошибка восстановления от уровня шума",
    )
    if with_lyap and "lyap" in df:
        plot_curves(
            snr_list,
            {m: df[df["method"] == m].set_index("snr_db").loc[snr_list, "lyap"].tolist() for m in methods},
            out_dir / "fig_snr_lyap.png",
            xlabel="SNR, дБ", ylabel="λ₁, 1/ед. времени",
            title="Шаг 7. Показатель Ляпунова от уровня шума",
            reference=LYAP_REFERENCE, reference_label=f"табличное {LYAP_REFERENCE}",
        )
    return df
