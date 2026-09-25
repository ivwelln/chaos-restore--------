from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .datasets import PreparedData, Span, make_loaders
from .models import build_model, count_parameters
from .utils import ensure_dir, resolve_device, save_table


@dataclass
class TrainResult:
    model: nn.Module
    history: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_run: int = 0
    seconds: float = 0.0
    seconds_per_epoch: float = 0.0
    n_params: int = 0
    checkpoint: Optional[Path] = None


def _run_epoch(model, loader, criterion, optimizer, device) -> float:
    """Одна эпоха. optimizer=None -> режим оценки (валидация)."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]
    return total / max(count, 1)


def train_model(
    data: PreparedData,
    cfg: Config,
    out_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    verbose: bool = True,
    tag: str = "",
) -> TrainResult:
    """Обучает модель на обучающей части и возвращает лучшую по валидации."""
    import copy
    import time

    device = resolve_device(cfg.train.device)
    name = model_name or cfg.train.model
    model = build_model(name).to(device)
    n_params = count_parameters(model)

    train_loader, val_loader = make_loaders(data, cfg)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    result = TrainResult(model=model, n_params=n_params)
    best_state = copy.deepcopy(model.state_dict())
    since_improve = 0
    started = time.perf_counter()

    if verbose:
        print(
            f"[обучение] {name}{' ' + tag if tag else ''}: {n_params} параметров, "
            f"{len(train_loader.dataset)} окон обучения / {len(val_loader.dataset)} валидации, "
            f"устройство {device}"
        )

    for epoch in range(1, cfg.train.max_epochs + 1):
        t0 = time.perf_counter()
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)
        result.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": time.perf_counter() - t0,
            }
        )
        improved = val_loss < result.best_val_loss - cfg.train.min_delta
        if improved:
            result.best_val_loss, result.best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_improve = 0
        else:
            since_improve += 1
        if verbose and (epoch <= 3 or epoch % 5 == 0 or improved):
            print(
                f"  эпоха {epoch:3d}  обучение {train_loss:.5f}  валидация {val_loss:.5f}"
                f"{'  <- лучшая' if improved else ''}"
            )
        if since_improve >= cfg.train.patience:
            if verbose:
                print(f"  ранняя остановка на эпохе {epoch} (лучшая – {result.best_epoch})")
            break

    result.epochs_run = len(result.history)
    result.seconds = time.perf_counter() - started
    result.seconds_per_epoch = result.seconds / max(result.epochs_run, 1)
    model.load_state_dict(best_state)
    model.eval()

    if out_dir is not None:
        ensure_dir(out_dir)
        suffix = f"_{tag}" if tag else ""
        result.checkpoint = out_dir / f"{name}{suffix}.pt"
        torch.save(
            {"model": name, "state_dict": best_state, "config": cfg.model_dump()},
            result.checkpoint,
        )
        save_table(result.history, out_dir / f"training_log_{name}{suffix}.csv")
    return result


@torch.no_grad()
def restore_signal(
    model: nn.Module,
    data: PreparedData,
    span: Span,
    cfg: Config,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Прогоняет через сеть непрерывный кусок сигнала и собирает его обратно."""
    from .windows import frame_signal, overlap_add

    device = device or next(model.parameters()).device
    model.eval()

    y = data.norm_in.transform(span.slice(data.distorted))
    frames = frame_signal(y, data.length, data.hop_infer)

    outputs = np.empty_like(frames.values)
    tensor = torch.from_numpy(frames.values[:, None, :].astype(np.float32))
    for start in range(0, tensor.shape[0], batch_size):
        chunk = tensor[start : start + batch_size].to(device)
        outputs[start : start + chunk.shape[0]] = model(chunk).squeeze(1).cpu().numpy()

    frames.values = outputs
    return data.norm_out.inverse(overlap_add(frames, data.hop_infer))
