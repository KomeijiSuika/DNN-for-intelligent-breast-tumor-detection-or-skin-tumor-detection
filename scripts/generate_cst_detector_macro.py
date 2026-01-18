"""Generate a CST macro (.cstmacro) that builds the 2x13 detector
regions used for EMNIST letters classification.

The macro creates 26 very thin bricks at the detector plane position,
following the classifier grid defined in configs/emnist_letters.yaml.
You can use these bricks as visualization of focal regions or as
post-processing regions (integrate |E|^2 over each brick).

Usage examples (from project root):

  # 128x128 effective pixel grid (no downsample)
  python scripts/generate_cst_detector_macro.py \
      --config configs/emnist_letters.yaml \
      --out_macro outputs/detector_macros/detector_128.cstmacro

  # 64x64 version (when metasurface was exported with downsample=2)
  python scripts/generate_cst_detector_macro.py \
      --config configs/emnist_letters.yaml \
      --downsample 2 \
      --out_macro outputs/detector_macros/detector_64.cstmacro

  # 32x32 version (downsample=4)
  python scripts/generate_cst_detector_macro.py \
      --config configs/emnist_letters.yaml \
      --downsample 4 \
      --out_macro outputs/detector_macros/detector_32.cstmacro
"""

from __future__ import annotations

import argparse
import string
from pathlib import Path
from typing import List, Tuple

import yaml


def _build_regions_from_grid(size: int, rows: int, cols: int, square: int, margin: int, gap: int) -> List[Tuple[int, int, int, int]]:
    """Reproduce _regions_grid layout: returns list of (x0,x1,y0,y1) pixel indices."""
    regions: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if len(regions) >= rows * cols:
                break
            x0 = margin + r * (square + gap)
            y0 = margin + c * (square + gap)
            x1 = x0 + square
            y1 = y0 + square
            regions.append((x0, x1, y0, y1))
        if len(regions) >= rows * cols:
            break
    return regions


def _downsample_regions(regions: List[Tuple[int, int, int, int]], factor: int) -> List[Tuple[int, int, int, int]]:
    """Map 128x128-pixel regions to a coarser grid.

    For downsample factor d (e.g. 2 -> 64x64, 4 -> 32x32), we
    approximate each region by integer-dividing the pixel indices
    and expanding the upper bound so that the physical coverage is
    preserved as much as possible.
    """
    if factor <= 1:
        return regions
    new: List[Tuple[int, int, int, int]] = []
    for x0, x1, y0, y1 in regions:
        nx0 = x0 // factor
        nx1 = (x1 - 1) // factor + 1
        ny0 = y0 // factor
        ny1 = (y1 - 1) // factor + 1
        new.append((nx0, nx1, ny0, ny1))
    return new


def _build_macro(
    pixel_size_m: float,
    detector_z_m: float,
    grid_regions: List[Tuple[int, int, int, int]],
    downsample: int,
) -> str:
    pixel_mm = float(pixel_size_m) * 1e3 * float(downsample)
    z_mm = float(detector_z_m) * 1e3
    dz_mm = 0.01  # very thin indicator layer

    letters = list(string.ascii_uppercase)  # A..Z

    lines: List[str] = []
    lines.append("Sub Main()")
    lines.append("  ' Auto-generated detector regions (2x13 grid) for EMNIST letters")
    lines.append(f"  Const pixelSize As Double = {pixel_mm:.6f} ' effective mm per pixel after downsample")
    lines.append(f"  Const zdet As Double = {z_mm:.6f} ' detector plane position (mm)")
    lines.append(f"  Const dz As Double = {dz_mm:.6f} ' detector brick thickness (mm)")
    lines.append("")

    lines.append("  With Brick")
    lines.append("    .Reset")
    lines.append("    .Component \"DetectorRegions\"")
    lines.append("  End With")
    lines.append("")

    for idx, (x0_pix, x1_pix, y0_pix, y1_pix) in enumerate(grid_regions):
        if idx >= 26:
            break
        letter = letters[idx]
        name = f"Det_{letter}"
        x0_mm = x0_pix * pixel_mm
        x1_mm = x1_pix * pixel_mm
        y0_mm = y0_pix * pixel_mm
        y1_mm = y1_pix * pixel_mm
        z0 = z_mm
        z1 = z_mm + dz_mm

        lines.append("  With Brick")
        lines.append("    .Reset")
        lines.append(f"    .Name \"{name}\"")
        lines.append("    .Component \"DetectorRegions\"")
        lines.append("    .Material \"Vacuum\"")
        lines.append(f"    .Xrange {x0_mm:.6f}, {x1_mm:.6f}")
        lines.append(f"    .Yrange {y0_mm:.6f}, {y1_mm:.6f}")
        lines.append(f"    .Zrange {z0:.6f}, {z1:.6f}")
        lines.append("    .Create")
        lines.append("  End With")
        lines.append("")

    lines.append("End Sub")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/emnist_letters.yaml")
    ap.add_argument("--downsample", type=int, default=1, help="pixel downsample factor (1=128, 2=64, 4=32)")
    ap.add_argument("--out_macro", default="outputs/detector_macros/detector_128.cstmacro")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    phys = cfg.get("physics", {}) or {}
    pixel_size_m = float(phys.get("pixel_size_m", 1.0e-3))
    z_list = list(phys.get("z_list_m", [0.10, 0.10, 0.10, 0.10]))
    if len(z_list) < 4:
        raise SystemExit("physics.z_list_m must have at least 4 entries (input->L1->L2->L3->detector)")
    # detector plane position (relative to input plane)
    detector_z_m = float(z_list[0] + z_list[1] + z_list[2] + z_list[3])

    clf = cfg.get("classifier", {}) or {}
    if clf.get("scheme", "grid") != "grid":
        raise SystemExit("This detector macro assumes classifier.scheme == 'grid'")
    grid = clf.get("grid", {}) or {}
    rows = int(grid.get("rows", 2))
    cols = int(grid.get("cols", 13))
    square = int(grid.get("square", 7))
    margin = int(grid.get("margin", 6))
    gap = int(grid.get("gap", 2))

    size = 128  # training detector plane size in pixels
    regions = _build_regions_from_grid(size=size, rows=rows, cols=cols, square=square, margin=margin, gap=gap)

    downsample = int(args.downsample) if args.downsample and args.downsample > 0 else 1

    # remap regions to coarse pixel indices when using downsample, so that
    # physical size matches the downsampled metasurface aperture
    regions = _downsample_regions(regions, downsample)

    macro_text = _build_macro(pixel_size_m=pixel_size_m, detector_z_m=detector_z_m, grid_regions=regions, downsample=downsample)

    out_path = Path(args.out_macro)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(macro_text, encoding="utf-8")

    print(f"Wrote detector macro to {out_path}")


if __name__ == "__main__":
    main()
