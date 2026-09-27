#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chaos_restore.config import Config, load_config  # noqa: E402

QUICK_OVERRIDES = {
    "name": "quick",
    "lorenz.duration": 400.0,
    "train.max_epochs": 8,
    "train.patience": 3,
    "lyapunov.series_len": 3000,
}


def build_config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config, args.set)
    if args.quick:
        cfg = cfg.copy_with(**QUICK_OVERRIDES)
    if args.name:
        cfg = cfg.copy_with(name=args.name)
    return cfg


def cmd_generate(cfg: Config, args: argparse.Namespace) -> None:
    from chaos_restore.lorenz import get_signal
    from chaos_restore.metrics import first_minimum_lag, lyapunov
    from chaos_restore.pipeline import LYAP_REFERENCE
    from chaos_restore.plotting import plot_attractor, plot_portraits, plot_spectra, plot_timeseries, setup_style
    from chaos_restore.utils import ensure_dir, set_seed

    setup_style()
    set_seed(cfg.seed)
    sig = get_signal(cfg)
    fig_dir = ensure_dir(cfg.results_dir / "figures")

    var = cfg.lorenz.variable
    clean = sig.series(var)
    n_frag = int(round(10 / sig.dt))
    plot_timeseries(sig.t[:n_frag], {f"эталон {var}(t)": clean[:n_frag]}, fig_dir / "fig01_signal_time.png",
                    title="Шаг 1. Эталонный сигнал системы Лоренца", xlim=(0, 10), ylabel=f"{var}(t)")
    plot_attractor(sig.x[:100000], sig.z[:100000], fig_dir / "fig02_attractor.png")
    lag = first_minimum_lag(clean[:20000])
    plot_portraits({"эталон": clean[:20000]}, lag=lag, path=fig_dir / "fig03_delay_portrait.png", dt=sig.dt)
    plot_spectra({f"эталон {var}(t)": clean}, sig.fs, fig_dir / "fig04_spectrum_clean.png",
                 title="Шаг 1. Спектр эталонного сигнала", fmax=20)

    if not args.no_lyap:
        lam = lyapunov(clean, sig.dt, cfg.lyapunov)
        print(f"\nλ₁ эталонного сигнала: {lam:.4f} (табличное {LYAP_REFERENCE}); "
              f"отклонение {abs(lam - LYAP_REFERENCE) / LYAP_REFERENCE * 100:.1f} %")
    print(f"τ по первому минимуму взаимной информации: {lag} отсчётов ({lag * sig.dt:.2f} ед. времени)")
    print(f"Рисунки: {fig_dir}")


def cmd_calibrate_lyap(cfg: Config, args: argparse.Namespace) -> None:
    import itertools

    from chaos_restore.lorenz import get_signal
    from chaos_restore.metrics import first_minimum_lag, lyapunov
    from chaos_restore.pipeline import LYAP_REFERENCE
    from chaos_restore.utils import ensure_dir, print_table, save_table, set_seed

    set_seed(cfg.seed)
    sig = get_signal(cfg)
    rows = []
    for subsample, emb_dim, min_tsep, traj in itertools.product(
        (5, 10, 20), (4, 6, 8, 10), (10, 20, 30), (20, 30)
    ):
        probe = cfg.lyapunov.model_copy(
            update={"subsample": subsample, "emb_dim": emb_dim, "min_tsep": min_tsep,
                    "trajectory_len": traj, "lag": None}
        )
        try:
            value = lyapunov(sig.x, sig.dt, probe)
        except Exception as exc:
            print(f"  пропуск (sub={subsample}, emb={emb_dim}): {exc}")
            continue
        rows.append({
            "subsample": subsample, "emb_dim": emb_dim, "min_tsep": min_tsep,
            "trajectory_len": traj, "lag": first_minimum_lag(sig.x[::subsample][:20000]),
            "lyap": value, "отклонение_%": abs(value - LYAP_REFERENCE) / LYAP_REFERENCE * 100,
        })

    df = save_table(rows, ensure_dir(cfg.results_dir) / "lyap_calibration.csv")
    df = df.sort_values("отклонение_%")
    print_table(df.head(12), title=f"Калибровка lyap_r (цель λ₁ = {LYAP_REFERENCE})")
    best = df.iloc[0]
    print("\nЛучший набор -> впишите в configs/base.yaml:")
    for key in ("subsample", "emb_dim", "lag", "min_tsep", "trajectory_len"):
        print(f"  {key}: {int(best[key])}")


def cmd_distort(cfg: Config, args: argparse.Namespace) -> None:
    from chaos_restore.channel import apply_channel
    from chaos_restore.lorenz import get_signal
    from chaos_restore.metrics import nrmse
    from chaos_restore.plotting import plot_spectra, plot_timeseries, setup_style
    from chaos_restore.utils import ensure_dir, set_seed

    setup_style()
    set_seed(cfg.seed)
    sig = get_signal(cfg)
    clean = sig.series(cfg.lorenz.variable)
    ch = apply_channel(clean, cfg.channel, sig.fs)
    fig_dir = ensure_dir(cfg.results_dir / "figures")
    n = int(round(10 / sig.dt))

    plot_timeseries(
        sig.t[:n], {"эталон": clean[:n], "после ФНЧ": ch.filtered[:n], "искажённый": ch.distorted[:n]},
        fig_dir / "fig05_channel_time.png",
        title=f"Шаг 2. {cfg.channel.filter.kind} {cfg.channel.filter.order}-го порядка, "
              f"fc = {cfg.channel.filter.fc:g}, SNR = {ch.snr_db_actual:.1f} дБ",
        xlim=(0, 10),
    )
    plot_spectra(
        {"эталон": clean, "после ФНЧ": ch.filtered, "искажённый": ch.distorted, "шум": ch.noise},
        sig.fs, fig_dir / "fig06_channel_spectra.png",
        title="Шаг 2. Выше fc искажённый сигнал совпадает с шумом",
        fc=cfg.channel.filter.fc, fmax=20,
    )
    print(f"Заданный SNR: {cfg.channel.snr_db:g} дБ, фактический: {ch.snr_db_actual:.2f} дБ")
    print(f"NRMSE искажённого относительно эталона: {nrmse(clean, ch.distorted):.4f}")
    print(f"Рисунки: {fig_dir}")


def cmd_main(cfg: Config, args: argparse.Namespace) -> None:
    from experiments import e_main

    e_main.run(cfg, with_lyap=not args.no_lyap)


def cmd_exp(cfg: Config, args: argparse.Namespace) -> None:
    from experiments import e_arch, e_crlb, e_filters, e_snr, e_window

    with_lyap = not args.no_lyap
    if args.which == "snr":
        e_snr.run(cfg, snr_list=args.snr, with_lyap=with_lyap)
    elif args.which == "window":
        e_window.run(cfg, lengths=args.lengths, with_lyap=with_lyap)
    elif args.which == "arch":
        e_arch.run(cfg, models=args.models, seeds=args.seeds, with_lyap=with_lyap)
    elif args.which == "filters":
        e_filters.run(cfg, with_lyap=with_lyap)
    elif args.which == "crlb":
        e_crlb.run(cfg, lengths=args.lengths, models=args.models, with_lyap=with_lyap)


def cmd_all(cfg: Config, args: argparse.Namespace) -> None:
    cmd_generate(cfg, args)
    cmd_distort(cfg, args)
    cmd_main(cfg, args)
    for which in ("snr", "window", "arch", "filters", "crlb"):
        args.which = which
        cmd_exp(cfg, args)


def _add_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Общие ключи. default=SUPPRESS – чтобы они работали и до, и после команды."""
    parser.add_argument("--config", default=argparse.SUPPRESS, help="путь к yaml-конфигу")
    parser.add_argument("--set", action="append", default=argparse.SUPPRESS,
                        metavar="КЛЮЧ=ЗНАЧЕНИЕ", help="точечное переопределение параметра конфига")
    parser.add_argument("--name", default=argparse.SUPPRESS, help="имя прогона (подпапка в results/)")
    parser.add_argument("--quick", action="store_true", default=argparse.SUPPRESS,
                        help="уменьшенный прогон для проверки")
    parser.add_argument("--no-lyap", action="store_true", default=argparse.SUPPRESS,
                        help="не считать показатель Ляпунова")
    return parser


DEFAULTS = {"config": None, "set": [], "name": None, "quick": False, "no_lyap": False}


def build_parser() -> argparse.ArgumentParser:
    parser = _add_common(
        argparse.ArgumentParser(
            description="Восстановление хаотического сигнала нейронной сетью",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("generate", "шаг 1: эталонный сигнал"),
        ("calibrate-lyap", "шаг 1: подбор параметров lyap_r"),
        ("distort", "шаг 2: искажение сигнала"),
        ("main", "шаги 5-6: простые методы и автоэнкодер"),
        ("all", "прогнать всё подряд"),
    ):
        _add_common(sub.add_parser(name, help=help_text))

    exp = _add_common(sub.add_parser("exp", help="шаги 7-11: дополнительные исследования"))
    exp.add_argument("which", choices=["snr", "window", "arch", "filters", "crlb"])
    exp.add_argument("--snr", type=float, nargs="*", default=None, help="список SNR в дБ")
    exp.add_argument("--lengths", type=int, nargs="*", default=None, help="список длин окна")
    exp.add_argument("--models", nargs="*", default=None, help="список архитектур")
    exp.add_argument("--seeds", type=int, default=1, help="сколько seed'ов на архитектуру")
    return parser


COMMANDS = {
    "generate": cmd_generate,
    "calibrate-lyap": cmd_calibrate_lyap,
    "distort": cmd_distort,
    "main": cmd_main,
    "exp": cmd_exp,
    "all": cmd_all,
}


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for key, value in DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    cfg = build_config(args)
    print(f"[конфиг] прогон «{cfg.name}», результаты в {cfg.results_dir}")
    COMMANDS[args.command](cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
