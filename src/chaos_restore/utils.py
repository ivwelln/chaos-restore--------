from __future__ import annotations

import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def resolve_device(name: str = "auto") -> "Any":
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def params_hash(payload: Dict[str, Any], length: int = 10) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


@contextmanager
def timer(label: str = "", verbose: bool = True):
    store: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield store
    finally:
        store["seconds"] = time.perf_counter() - start
        if verbose and label:
            print(f"[время] {label}: {store['seconds']:.1f} с")


def save_table(rows: Iterable[Dict[str, Any]], path: Path, float_format: str = "%.6g") -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    ensure_dir(path.parent)
    df.to_csv(path, index=False, float_format=float_format, encoding="utf-8")
    return df


def print_table(df: pd.DataFrame, title: Optional[str] = None, float_format: str = "{:.4g}") -> None:
    if title:
        print(f"\n=== {title} ===")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.to_string(index=False, float_format=lambda v: float_format.format(v)))
