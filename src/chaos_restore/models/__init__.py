from __future__ import annotations

from typing import Any

from .cae import ConvAutoencoder
from .unet import UNet1D
from .wavenet import WaveNet1D

REGISTRY = {
    "cae": ConvAutoencoder,
    "unet": UNet1D,
    "wavenet": WaveNet1D,
}

HUMAN_NAMES = {
    "cae": "автоэнкодер",
    "unet": "U-Net",
    "wavenet": "WaveNet",
}


def build_model(name: str, **kwargs: Any):
    if name not in REGISTRY:
        raise KeyError(f"неизвестная модель {name!r}; доступны: {', '.join(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["REGISTRY", "HUMAN_NAMES", "build_model", "count_parameters", "ConvAutoencoder", "UNet1D", "WaveNet1D"]
