from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare flat image folder for ImageFolder.")
    parser.add_argument("--src", type=str, required=True, help="Source folder with images.")
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Destination ImageFolder root. Images will be copied to dst/faces.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only copy first N images (for quick experiments).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst_faces = Path(args.dst) / "faces"
    dst_faces.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    images.sort()
    if args.limit > 0:
        images = images[: args.limit]

    for p in tqdm(images, desc="Copying images"):
        target = dst_faces / p.name
        if not target.exists():
            shutil.copy2(p, target)

    print(f"Done. Copied {len(images)} images to {dst_faces}")


if __name__ == "__main__":
    main()
