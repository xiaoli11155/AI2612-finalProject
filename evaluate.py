from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
from tqdm import tqdm

from models import load_generator
from scripts.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DCGAN with FID.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True, help="ImageFolder root.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-gen", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = (images.clamp(-1, 1) + 1) * 127.5
    return images.to(torch.uint8)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(args.data_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    net_g, model_args = load_generator(args.checkpoint, device)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    seen_real = 0
    for real, _ in tqdm(loader, desc="FID real"):
        real = to_uint8(real.to(device))
        fid.update(real, real=True)
        seen_real += real.size(0)
        if seen_real >= args.num_gen:
            break

    remaining = args.num_gen
    while remaining > 0:
        bsz = min(args.batch_size, remaining)
        noise = torch.randn(bsz, model_args["nz"], 1, 1, device=device)
        with torch.no_grad():
            fake = net_g(noise)
        fake = to_uint8(fake)
        fid.update(fake, real=False)
        remaining -= bsz

    fid_score = fid.compute().item()
    print(f"FID: {fid_score:.4f}")


if __name__ == "__main__":
    main()
