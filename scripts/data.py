from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_dataloader(
    data_root: str,
    image_size: int,
    batch_size: int,
    workers: int,
    max_samples: int = 0,
) -> DataLoader:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.ImageFolder(root=str(root), transform=transform)
    if max_samples > 0:
        take_n = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(take_n)))

    drop_last = len(dataset) >= batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
