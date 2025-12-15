from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _iter_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def _load_grayscale(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def build_npz_from_folder(folder: str, *, image_size: int, out_npz: str):
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(folder)

    # expected structure:
    # folder/0/*.png
    # folder/1/*.png
    x_list = []
    y_list = []
    for label in [0, 1]:
        class_dir = folder_path / str(label)
        if not class_dir.exists():
            continue
        for img_path in _iter_images(class_dir):
            x_list.append(_load_grayscale(img_path, image_size))
            y_list.append(label)

    if not x_list:
        raise RuntimeError(
            f"No images found under {folder}. Expected {folder}/0 and {folder}/1 with image files."
        )

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, x=x, y=y)
    print(f"Saved {out_path} with x={x.shape}, y={y.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()

    raw = Path(args.raw_root)
    build_npz_from_folder(str(raw / "train"), image_size=args.image_size, out_npz="data/processed/train.npz")
    build_npz_from_folder(str(raw / "val"), image_size=args.image_size, out_npz="data/processed/val.npz")
    build_npz_from_folder(str(raw / "test"), image_size=args.image_size, out_npz="data/processed/test.npz")


if __name__ == "__main__":
    main()
