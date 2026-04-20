from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import load_generator
from scripts.utils import ensure_dirs, get_device, save_tensor_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with trained DCGAN.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/generated")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    ensure_dirs([args.out_dir])
    out_path = Path(args.out_dir) / "generated_grid.png"

    net_g, model_args = load_generator(args.checkpoint, device)
    noise = torch.randn(args.num_images, model_args["nz"], 1, 1, device=device)
    with torch.no_grad():
        fake = net_g(noise).cpu()
    save_tensor_grid(fake, str(out_path), nrow=args.nrow)


if __name__ == "__main__":
    main()
