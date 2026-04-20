from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from models import load_generator
from scripts.utils import ensure_dirs, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent interpolation for DCGAN.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/interpolation")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    ensure_dirs([args.out_dir])

    net_g, model_args = load_generator(args.checkpoint, device)
    nz = model_args["nz"]

    z1 = torch.randn(1, nz, 1, 1, device=device)
    z2 = torch.randn(1, nz, 1, 1, device=device)
    seq = []
    alphas = torch.linspace(0, 1, steps=args.steps, device=device)

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            z = (1 - alpha) * z1 + alpha * z2
            img = net_g(z).cpu()
            seq.append(img)
            save_image(
                img,
                str(Path(args.out_dir) / f"interp_{i:03d}.png"),
                normalize=True,
                value_range=(-1, 1),
            )

    grid = torch.cat(seq, dim=0)
    save_image(
        grid,
        str(Path(args.out_dir) / "interpolation_grid.png"),
        nrow=args.steps,
        normalize=True,
        value_range=(-1, 1),
    )


if __name__ == "__main__":
    main()
