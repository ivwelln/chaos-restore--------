from __future__ import annotations

import numpy as np
import pandas as pd

from . import _SRC  # noqa: F401

from chaos_restore.config import Config
from chaos_restore.pipeline import (
    LYAP_REFERENCE,
    build_case,
    embedding_lag,
    reference_rows,
    run_baselines,
    run_network,
)
from chaos_restore.plotting import (
    plot_attractor,
    plot_bars,
    plot_portraits,
    plot_spectra,
    plot_timeseries,
    plot_training_curves,
    setup_style,
)
from chaos_restore.utils import ensure_dir, print_table, save_table


def run(
    cfg: Config,
    with_lyap: bool = True,
    verbose: bool = True,
    figures: bool = True,
) -> pd.DataFrame:
    setup_style()
    out_dir = ensure_dir(cfg.results_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    cfg.dump(out_dir / "config.yaml")

    case = build_case(cfg, verbose=verbose)
    sig, ch = case.signal, case.channel
    var = case.variable
    clean = sig.series(var)

    if figures:
        n_frag = int(round(10 / sig.dt))
        plot_timeseries(
            sig.t[:n_frag], {f"эталон {var}(t)": clean[:n_frag]}, fig_dir / "fig01_signal_time.png",
            title="Шаг 1. Эталонный сигнал системы Лоренца", xlim=(0, 10), ylabel=f"{var}(t)",
        )
        plot_attractor(sig.x[:100000], sig.z[:100000], fig_dir / "fig02_attractor.png")
        plot_spectra(
            {f"эталон {var}(t)": clean}, sig.fs, fig_dir / "fig04_spectrum_clean.png",
            title="Шаг 1. Спектр эталонного сигнала", fmax=20,
        )

        plot_timeseries(
            sig.t[:n_frag],
            {"эталон": clean[:n_frag], "после ФНЧ": ch.filtered[:n_frag], "искажённый": ch.distorted[:n_frag]},
            fig_dir / "fig05_channel_time.png",
            title=f"Шаг 2. Искажение: {cfg.channel.filter.kind} {cfg.channel.filter.order}-го порядка, "
                  f"fc = {cfg.channel.filter.fc:g}, SNR = {ch.snr_db_actual:.1f} дБ",
            xlim=(0, 10),
        )
        plot_spectra(
            {"эталон": clean, "после ФНЧ": ch.filtered, "искажённый": ch.distorted, "шум": ch.noise},
            sig.fs,
            fig_dir / "fig06_channel_spectra.png",
            title="Шаг 2. Спектры: выше fc искажённый сигнал совпадает с шумом",
            fc=cfg.channel.filter.fc,
            fmax=20,
        )

    rows = reference_rows(case, with_lyap=with_lyap)
    base_rows, base_restored, _ = run_baselines(case, with_lyap=with_lyap, verbose=verbose)
    rows += base_rows
    net_row, net_restored, train_result = run_network(
        case, out_dir=ensure_dir(out_dir / "checkpoints"), with_lyap=with_lyap, verbose=verbose
    )
    rows.append(net_row)

    df = save_table(rows, out_dir / "metrics_main.csv")
    if verbose:
        print_table(df.drop(columns=[c for c in ("n_params", "val_loss") if c in df], errors="ignore"),
                    title="Шаги 5–6. Сравнение методов на тестовой части")

    if figures:
        plot_training_curves(train_result.history, fig_dir / "fig07_training_curves.png")

        best_baseline = min(
            (r for r in base_rows if "прямой" not in str(r["method"])), key=lambda r: r["rmse"]
        )["method"]
        clean_t = case.test_clean()
        t_test = np.arange(clean_t.size) * sig.dt
        series = {
            "эталон": clean_t,
            "искажённый": case.test_distorted(),
            f"лучший простой: {best_baseline}": case.crop_like_test(base_restored[best_baseline]),
            "восстановленный (сеть)": case.crop_like_test(net_restored),
        }
        n_show = min(clean_t.size, int(round(8 / sig.dt)))
        plot_timeseries(
            t_test[:n_show], {k: v[:n_show] for k, v in series.items()},
            fig_dir / "fig08_compare_time.png",
            title="Шаг 6. Восстановление на тестовой части", xlim=(0, 8),
        )
        plot_spectra(
            series, sig.fs, fig_dir / "fig09_compare_spectra.png",
            title="Шаг 6. Спектры восстановленных сигналов", fc=cfg.channel.filter.fc, fmax=20,
        )
        lag = embedding_lag(case)
        plot_portraits(
            {"эталон": clean_t, "искажённый": case.test_distorted(),
             "восстановленный (сеть)": case.crop_like_test(net_restored)},
            lag=lag, path=fig_dir / "fig10_portraits.png", dt=sig.dt,
        )
        if with_lyap and "lyap" in df:
            values = {str(r["method"]): float(r["lyap"]) for r in rows if "lyap" in r and r["method"] != "эталон"}
            lam_ref = float(df.loc[df["method"] == "эталон", "lyap"].iloc[0])
            plot_bars(
                values, fig_dir / "fig11_lyapunov.png", ylabel="λ₁, 1/ед. времени",
                title="Шаг 6. Показатель Ляпунова по методам",
                reference=lam_ref,
                reference_label=f"эталон: {lam_ref:.3f} (табличное {LYAP_REFERENCE})",
            )
        if verbose:
            print(f"[рисунки] сохранены в {fig_dir}")

    return df
