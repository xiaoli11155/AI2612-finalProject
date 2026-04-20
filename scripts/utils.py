from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(paths: list[str]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_tensor_grid(images: torch.Tensor, out_path: str, nrow: int = 8) -> None:
    save_image(images, out_path, nrow=nrow, normalize=True, value_range=(-1, 1))
