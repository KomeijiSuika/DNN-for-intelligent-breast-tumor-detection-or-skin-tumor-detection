"""Generate a CST macro (.cstmacro) that builds a letter-shaped source mask
using an EMNIST letter sample at adjustable resolution.

The macro creates a thin brick array in the plane z = z0_m, with one brick per
"on" pixel, which you can use as an aperture / mask in CST.

Usage examples (from project root):

  # 1) Random letter from EMNIST test set, 128x128, original pixel size
  python scripts/generate_cst_source_macro.py \
      --config configs/emnist_letters.yaml \
      --out_macro outputs/source_macros/source_128.cstmacro

  # 2) Random letter, downsampled to 64x64, keeping total size ~128 mm
  python scripts/generate_cst_source_macro.py \
      --config configs/emnist_letters.yaml \
      --resolution 64 \
      --out_macro outputs/source_macros/source_64.cstmacro

  # 3) Specific index and resolution 32x32
  python scripts/generate_cst_source_macro.py \
      --config configs/emnist_letters.yaml \
      --index 10 \
      --resolution 32 \
      --out_macro outputs/source_macros/source_32_idx10.cstmacro

Notes:
  - By default this reads EMNIST npz from config's data.test_npz.
  - Resolution can be 128, 64 or 32; total physical aperture stays ~128 mm.
  - Pixels above a threshold (default 0.5) are treated as "on" and turned
    into bricks in component "SourceLetter".
"""

from __future__ import annotations

import argparse
import string
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _load_emnist_sample(npz_path: Path, index: int | None) -> Tuple[np.ndarray, int]:
    data = np.load(npz_path)
    x = data["x"]  # (N,H,W)
    y = data["y"]  # (N,)
    n = x.shape[0]
    if n == 0:
        raise SystemExit(f"No samples found in {npz_path}")
    if index is None or index < 0 or index >= n:
        index = int(np.random.randint(0, n))
    img = x[index]  # (H,W) float32 in [0,1]
    label = int(y[index])
    return img, label


def _resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape
    if h == size and w == size:
        return img
    ys = (np.linspace(0, h - 1, size)).astype(np.int64)
    xs = (np.linspace(0, w - 1, size)).astype(np.int64)
    return img[np.ix_(ys, xs)]


def _build_macro(img: np.ndarray, pixel_size_m: float, z0_m: float, thickness_m: float, threshold: float, label: int) -> str:
    """Build a CST macro that creates bricks for pixels >= threshold.

    img: 2D array in [0,1], shape (H,W).
    """
    h, w = img.shape
    pixel_mm = float(pixel_size_m) * 1e3
    z0_mm = float(z0_m) * 1e3
    dz_mm = float(thickness_m) * 1e3

    # thresholding
    vmax = float(img.max()) if img.size > 0 else 0.0
    if vmax <= 0.0:
        mask = np.zeros_like(img, dtype=bool)
    else:
        thr = threshold * vmax
        mask = img >= thr

    letters = string.ascii_uppercase
    letter_char = letters[label] if 0 <= label < len(letters) else "?"

    lines = []
    lines.append("Sub Main()")
    lines.append("  ' Auto-generated letter source mask")
    lines.append(f"  ' EMNIST label index: {label} (letter {letter_char})")
    lines.append(f"  Const pixelSize As Double = {pixel_mm:.6f} ' mm")
    lines.append(f"  Const z0 As Double = {z0_mm:.6f} ' mm")
    lines.append(f"  Const dz As Double = {dz_mm:.6f} ' mm")
    lines.append("")
    lines.append("  With Brick")
    lines.append("    .Reset")
    lines.append("    .Component \"SourceLetter\"")
    lines.append("  End With")
    lines.append("")

    for iy in range(h):
        for ix in range(w):
            if not mask[iy, ix]:
                continue
            x0 = ix * pixel_mm
            x1 = (ix + 1) * pixel_mm
            y0 = iy * pixel_mm
            y1 = (iy + 1) * pixel_mm
            z0 = z0_mm
            z1 = z0_mm + dz_mm
            name = f"Src_x{ix}_y{iy}"
            lines.append("  With Brick")
            lines.append("    .Reset")
            lines.append(f"    .Name \"{name}\"")
            lines.append("    .Component \"SourceLetter\"")
            # Use PEC or a dedicated material as needed; here we use PEC mask
            lines.append("    .Material \"PEC\"")
            lines.append(f"    .Xrange {x0:.6f}, {x1:.6f}")
            lines.append(f"    .Yrange {y0:.6f}, {y1:.6f}")
            lines.append(f"    .Zrange {z0:.6f}, {z1:.6f}")
            lines.append("    .Create")
            lines.append("  End With")

    lines.append("End Sub")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    DEFAULT_OUT = "outputs/source_macros/source_letter.cstmacro"

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/emnist_letters.yaml")
    ap.add_argument("--npz", default=None, help="EMNIST letters npz; default from config.data.test_npz")
    ap.add_argument("--index", type=int, default=-1, help="sample index; <0 for random")
    ap.add_argument("--resolution", type=int, default=128, help="target resolution: 128, 64, or 32")
    ap.add_argument("--threshold", type=float, default=0.5, help="amplitude threshold fraction for mask")
    ap.add_argument("--z0_m", type=float, default=0.0, help="z position of mask (m)")
    ap.add_argument("--thickness_m", type=float, default=1.0e-3, help="mask thickness (m)")
    ap.add_argument("--augment", action="store_true", help="slightly perturb image (noise) instead of raw dataset only")
    ap.add_argument("--out_macro", default=DEFAULT_OUT)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {}) or {}

    npz_path = Path(args.npz) if args.npz is not None else Path(data_cfg.get("test_npz", "data/processed/emnist_letters_test.npz"))
    if not npz_path.exists():
        raise SystemExit(f"EMNIST npz not found: {npz_path}")

    img, label = _load_emnist_sample(npz_path, args.index)
    h, w = img.shape

    # original physical pixel size from config
    pixel_size_m_orig = float(cfg.get("physics", {}).get("pixel_size_m", 1.0e-3))

    # adjust pixel size so total aperture stays the same when changing resolution
    target_res = int(args.resolution)
    if target_res not in (128, 64, 32):
        raise SystemExit("--resolution must be one of 128, 64, 32")

    img_resized = _resize_to_square(img, target_res)
    scale = h / float(target_res)
    pixel_size_m = pixel_size_m_orig * scale

    # optional small perturbation on top of dataset image
    img_used = img_resized.astype(np.float32)
    if args.augment:
        noise = np.random.normal(scale=0.05, size=img_used.shape).astype(np.float32)
        img_used = np.clip(img_used + noise, 0.0, 1.0)

    macro_text = _build_macro(img_used, pixel_size_m=pixel_size_m, z0_m=args.z0_m, thickness_m=args.thickness_m, threshold=float(args.threshold), label=label)

    # decide output macro name; if user kept default path, enrich it with resolution and letter
    from string import ascii_uppercase

    letter_char = ascii_uppercase[label] if 0 <= label < len(ascii_uppercase) else "X"
    if args.out_macro == DEFAULT_OUT:
        idx_str = str(args.index) if args.index >= 0 else "rand"
        out_path = Path(f"outputs/source_macros/source_{target_res}_{letter_char}_idx{idx_str}.cstmacro")
    else:
        out_path = Path(args.out_macro)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(macro_text, encoding="utf-8")

    # also save a PNG preview next to the macro so you can see the letter
    png_path = out_path.with_suffix(".png")
    plt.figure(figsize=(3, 3))
    plt.imshow(img_used, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(f"Letter {letter_char} (idx={args.index if args.index>=0 else 'rand'})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Wrote source macro to {out_path} (letter {letter_char}, label {label})")
    print(f"Saved preview image to {png_path}")


if __name__ == "__main__":
    main()
