from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from . import _SRC  # noqa: F401

from chaos_restore.config import Config
from chaos_restore.pipeline import build_case, evaluate_signal, run_network
from chaos_restore.plotting import plot_hbar_table, setup_style
from chaos_restore.train import restore_signal
from chaos_restore.utils import ensure_dir, print_table, save_table


@dataclass
class Variant:
    label: str
    group: str
    overrides: Dict[str, object]


def default_variants(cfg: Config) -> List[Variant]:
    fc = cfg.channel.filter.fc
    order = cfg.channel.filter.order
    variants = [
        Variant(f"Баттерворт {order}, fc={fc:g} (обучающий)", "эталонный", {}),
        Variant(f"Бессель {order}, fc={fc:g}", "тип фильтра", {"channel.filter.kind": "bessel"}),
        Variant(f"Чебышев I {order}, fc={fc:g}", "тип фильтра", {"channel.filter.kind": "cheby1"}),
    ]
    for factor in (0.5, 0.75, 1.25, 1.5, 2.0):
        variants.append(
            Variant(
                f"Баттерворт {order}, fc={fc * factor:g} (x{factor:g})",
                "частота среза",
                {"channel.filter.fc": fc * factor},
            )
        )
    for other in (2, 6, 8):
        variants.append(
            Variant(
                f"Баттерворт {other}, fc={fc:g}",
                "порядок",
                {"channel.filter.order": other},
            )
        )
    return variants


def run(
    cfg: Config,
    with_lyap: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    setup_style()
    out_dir = ensure_dir(cfg.results_dir / "e_filters")
    cfg.dump(out_dir / "config.yaml")

    base_case = build_case(cfg, verbose=verbose)
    _, _, result = run_network(
        base_case, out_dir=ensure_dir(out_dir / "checkpoints"), with_lyap=False, verbose=verbose
    )
    model, norm_out = result.model, base_case.data.norm_out

    rows = []
    for variant in default_variants(cfg):
        if verbose:
            print(f"\n----- {variant.label} -----")
        case = build_case(cfg.copy_with(**variant.overrides), verbose=False)
        case.data.norm_out = norm_out  # выход остаётся в «родном» масштабе обучения
        x_hat = restore_signal(model, case.data, case.data.splits.test, case.cfg)

        row_net = evaluate_signal(case, x_hat, "сеть", with_lyap=with_lyap)
        distorted = case.data.splits.test.slice(case.data.distorted)
        row_in = evaluate_signal(case, distorted, "искажённый", with_lyap=False)
        rows.append(
            {
                "variant": variant.label,
                "group": variant.group,
                "nrmse_net": row_net["nrmse"],
                "nrmse_distorted": row_in["nrmse"],
                "улучшение": row_in["nrmse"] / row_net["nrmse"] if row_net["nrmse"] else float("nan"),
                "corr_net": row_net["corr"],
                **({"lyap_net": row_net["lyap"]} if with_lyap and "lyap" in row_net else {}),
            }
        )
        if verbose:
            print(f"    NRMSE сети {row_net['nrmse']:.4f} против {row_in['nrmse']:.4f} у входа")

    df = save_table(rows, out_dir / "metrics_filters.csv")
    if verbose:
        print_table(df, title="Шаг 10. Работа сети при рассогласовании фильтра")

    plot_hbar_table(
        df["variant"], df["nrmse_net"], out_dir / "fig_filters_nrmse.png",
        xlabel="NRMSE сети на тесте",
        title="Шаг 10. Обобщение сети на другие фильтры",
        reference=float(df["nrmse_net"].iloc[0]),
    )
    return df
