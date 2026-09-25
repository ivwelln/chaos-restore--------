from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chaos_restore.baselines import apply_shift, estimate_shift, wiener_deconvolution
from chaos_restore.channel import apply_channel, design_filter, lowpass
from chaos_restore.config import load_config
from chaos_restore.datasets import Normalizer, make_splits, prepare
from chaos_restore.lorenz import simulate
from chaos_restore.metrics import first_minimum_lag, lyapunov, nrmse
from chaos_restore.windows import check_roundtrip, frame_signal, overlap_add

TOL_MACHINE = 1e-9


def _cfg(duration: float = 200.0):
    return load_config(ROOT / "configs" / "base.yaml").copy_with(**{"lorenz.duration": duration})


def test_overlap_add_roundtrip():
    rng = np.random.default_rng(0)
    for n in (5000, 5123):
        x = rng.normal(size=n)
        for length, div in ((256, 4), (256, 2), (128, 4), (512, 8), (64, 4)):
            err = check_roundtrip(x, length, length // div)
            assert err < TOL_MACHINE, f"L={length}, hop=L/{div}: ошибка {err:.2e}"


def test_overlap_add_keeps_length():
    x = np.arange(1000, dtype=float)
    frames = frame_signal(x, 128, 32)
    back = overlap_add(frames, 32)
    assert back.shape == x.shape


def test_splits_are_disjoint_and_ordered():
    cfg = _cfg()
    n = 100000
    splits = make_splits(n, cfg)
    assert splits.train.start == 0
    assert splits.train.stop == splits.val.start
    assert splits.val.stop == splits.test.start
    assert splits.test.stop == n
    assert len(splits.test) > 0 and len(splits.val) > 0


def test_normalizer_is_invertible():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=3.0, scale=7.0, size=10000)
    norm = Normalizer.fit(x)
    assert abs(float(np.mean(norm.transform(x)))) < 1e-10
    assert abs(float(np.std(norm.transform(x))) - 1.0) < 1e-10
    assert np.max(np.abs(norm.inverse(norm.transform(x)) - x)) < 1e-9


def test_channel_hits_requested_snr():
    cfg = _cfg()
    data = simulate(cfg.lorenz)
    for target in (0.0, 10.0, 25.0):
        out = apply_channel(data.x, cfg.channel.model_copy(update={"snr_db": target}), data.fs)
        assert abs(out.snr_db_actual - target) < 0.2, f"SNR {target}: получили {out.snr_db_actual}"


def test_lowpass_suppresses_high_frequencies():
    cfg = _cfg()
    data = simulate(cfg.lorenz)
    b, a = design_filter(cfg.channel.filter, data.fs)
    filtered = lowpass(data.x, b, a)
    from scipy.signal import welch

    f, p_clean = welch(data.x, fs=data.fs, nperseg=4096)
    _, p_filt = welch(filtered, fs=data.fs, nperseg=4096)
    high = f > 3 * cfg.channel.filter.fc
    assert float(np.mean(p_filt[high]) / np.mean(p_clean[high])) < 1e-2


def test_shift_estimation_recovers_known_delay():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(size=20000))
    for delay in (-37, 0, 12, 91):
        delayed = apply_shift(x, -delay)
        estimated = estimate_shift(delayed, x)
        assert estimated == delay, f"ожидали {delay}, получили {estimated}"


def test_wiener_inverts_filter_without_noise():
    cfg = _cfg()
    data = simulate(cfg.lorenz)
    b, a = design_filter(cfg.channel.filter, data.fs)
    filtered = lowpass(data.x, b, a)
    restored = wiener_deconvolution(filtered, b, a, lam=1e-10)
    m = slice(2000, -2000)
    assert nrmse(data.x[m], restored[m]) < 0.05


def test_models_keep_shape_and_are_comparable():
    import torch

    from chaos_restore.models import build_model, count_parameters

    sizes = {}
    for name in ("cae", "unet", "wavenet"):
        model = build_model(name).eval()
        sizes[name] = count_parameters(model)
        for length in (64, 256):
            with torch.no_grad():
                out = model(torch.zeros(2, 1, length))
            assert out.shape == (2, 1, length), f"{name}, L={length}: получили {tuple(out.shape)}"
    biggest, smallest = max(sizes.values()), min(sizes.values())
    assert biggest / smallest < 2.0, f"число параметров несопоставимо: {sizes}"


def test_prepare_uses_only_train_statistics():
    cfg = _cfg()
    data = simulate(cfg.lorenz)
    out = apply_channel(data.x, cfg.channel, data.fs)
    prepared = prepare(data.x, out.distorted, cfg)
    train_only = Normalizer.fit(prepared.splits.train.slice(out.distorted))
    assert abs(prepared.norm_in.mean - train_only.mean) < 1e-12
    assert abs(prepared.norm_in.std - train_only.std) < 1e-12


def test_lyapunov_matches_reference():
    cfg = _cfg(duration=1000.0)
    data = simulate(cfg.lorenz)
    value = lyapunov(data.x, data.dt, cfg.lyapunov)
    assert 0.80 < value < 1.00, f"λ₁ = {value:.3f}, ожидали около 0.906"


def test_first_minimum_lag_is_positive():
    cfg = _cfg()
    data = simulate(cfg.lorenz)
    lag = first_minimum_lag(data.x[:20000])
    assert 1 <= lag <= 100


def test_lyapunov_is_reproducible():
    cfg = _cfg(duration=400.0)
    data = simulate(cfg.lorenz)
    out = apply_channel(data.x, cfg.channel, data.fs)
    for signal in (data.x, out.distorted):
        first = lyapunov(signal, data.dt, cfg.lyapunov)
        second = lyapunov(signal, data.dt, cfg.lyapunov)
        assert first == second, f"λ₁ не воспроизводится: {first} против {second}"


def test_blind_inverse_recovers_channel_roughly():
    from chaos_restore.baselines import BlindInverse

    cfg = _cfg(duration=500.0)
    data = simulate(cfg.lorenz)
    out = apply_channel(data.x, cfg.channel, data.fs)
    blind = BlindInverse.design(out.distorted[: int(0.8 * data.x.size)], data.fs)

    fc_true = cfg.channel.filter.fc
    assert 0.5 * fc_true < blind.fc_est < 1.5 * fc_true, f"fc оценена как {blind.fc_est:.2f}"
    assert 2 <= blind.order_est <= 8, f"порядок оценён как {blind.order_est}"
    restored = blind(out.distorted)
    m = slice(2000, -2000)
    assert nrmse(data.x[m], restored[m]) < nrmse(data.x[m], out.distorted[m])


def test_signal_variable_is_switchable():
    from chaos_restore.pipeline import build_case

    for name in ("x", "y", "z"):
        case = build_case(_cfg().copy_with(**{"lorenz.variable": name}), verbose=False)
        expected = case.signal.series(name)
        assert np.array_equal(case.data.clean, expected), f"переменная {name} не доехала до данных"
        assert case.variable == name


def test_unknown_config_key_is_rejected():
    cfg = _cfg()
    try:
        cfg.copy_with(**{"channel.snr_dB": 10})
    except KeyError:
        return
    raise AssertionError("опечатка в имени параметра должна приводить к ошибке")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ПРОВАЛ  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok      {name}")
    print(f"\nПройдено {len(tests) - failed} из {len(tests)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
