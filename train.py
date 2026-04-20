from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import build_models
from scripts.data import build_dataloader
from scripts.utils import ensure_dirs, get_device, save_tensor_grid, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline DCGAN.")
    parser.add_argument(
        "--data-root", type=str, required=True, help="ImageFolder root."
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--nc", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-interval", type=int, default=200)
    parser.add_argument("--save-epoch-interval", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Use only first N samples for quick debugging. 0 means full dataset.",
    )
    return parser.parse_args()


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    net_g: torch.nn.Module,
    net_d: torch.nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "epoch": epoch,
        "net_g": net_g.state_dict(),
        "net_d": net_d.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "args": {
            "nz": args.nz,
            "ngf": args.ngf,
            "ndf": args.ndf,
            "nc": args.nc,
            "image_size": args.image_size,
        },
    }
    torch.save(ckpt, out_dir / f"dcgan_epoch_{epoch:03d}.pt")
    torch.save(ckpt, out_dir / "latest.pt")


def main() -> None:
    args = parse_args()
    if args.image_size != 128:
        raise ValueError("Current DCGAN architecture is fixed for 128x128. Please set --image-size 128.")
    set_seed(args.seed)

    out_root = Path(args.output_dir)
    sample_dir = out_root / "samples"
    ckpt_dir = out_root / "checkpoints"
    ensure_dirs([str(sample_dir), str(ckpt_dir)])

    device = get_device()
    dataloader = build_dataloader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(dataloader.dataset)} samples, {len(dataloader)} batches/epoch.")
    net_g, net_d = build_models(args.nz, args.ngf, args.ndf, args.nc, device)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    opt_d = optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_g = optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0
    global_step = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, (real, _) in enumerate(pbar):
            net_d.zero_grad(set_to_none=True)
            real = real.to(device)
            bsz = real.size(0)

            label_real = torch.full(
                (bsz,), real_label, dtype=torch.float, device=device
            )
            out_real = net_d(real)
            loss_d_real = criterion(out_real, label_real)
            loss_d_real.backward()
            d_x = out_real.mean().item()

            noise = torch.randn(bsz, args.nz, 1, 1, device=device)
            fake = net_g(noise)
            label_fake = torch.full(
                (bsz,), fake_label, dtype=torch.float, device=device
            )
            out_fake = net_d(fake.detach())
            loss_d_fake = criterion(out_fake, label_fake)
            loss_d_fake.backward()
            d_g_z1 = out_fake.mean().item()

            loss_d = loss_d_real + loss_d_fake
            opt_d.step()

            net_g.zero_grad(set_to_none=True)
            label_gen = torch.full((bsz,), real_label, dtype=torch.float, device=device)
            out_gen = net_d(fake)
            loss_g = criterion(out_gen, label_gen)
            loss_g.backward()
            d_g_z2 = out_gen.mean().item()
            opt_g.step()

            global_step += 1
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "iter": i,
                    "loss_d": float(loss_d.item()),
                    "loss_g": float(loss_g.item()),
                    "d_x": float(d_x),
                    "d_g_z1": float(d_g_z1),
                    "d_g_z2": float(d_g_z2),
                }
            )
            pbar.set_postfix(
                loss_d=f"{loss_d.item():.3f}", loss_g=f"{loss_g.item():.3f}"
            )

            if global_step % args.sample_interval == 0:
                with torch.no_grad():
                    fake_fixed = net_g(fixed_noise).detach().cpu()
                save_tensor_grid(
                    fake_fixed,
                    str(sample_dir / f"sample_e{epoch:03d}_s{global_step:06d}.png"),
                    nrow=8,
                )

        if epoch % args.save_epoch_interval == 0:
            save_checkpoint(ckpt_dir, epoch, net_g, net_d, opt_g, opt_d, args)

    with open(out_root / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with torch.no_grad():
        fake_fixed = net_g(fixed_noise).detach().cpu()
    save_tensor_grid(fake_fixed, str(sample_dir / "final_samples.png"), nrow=8)


if __name__ == "__main__":
    main()
