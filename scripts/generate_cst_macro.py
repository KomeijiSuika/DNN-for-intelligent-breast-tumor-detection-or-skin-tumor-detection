"""Generate a CST macro (.cstmacro) that builds 3 dielectric layers
from thickness maps exported by metasurface_dnn.phase_to_thickness.

Usage (example):

    python scripts/generate_cst_macro.py \
        --config configs/emnist_letters.yaml \
        --thickness_dir outputs/printable \
        --out_macro outputs/printable/build_3layers.cstmacro

Then, in CST 2020:
  - File -> Macros -> Open..., choose the generated .cstmacro
  - Run the macro; it will create three components (Layer1/2/3)
    filled with bricks (one per pixel) using material "Vero_White_Plus".

Notes:
  - This macro assumes pixel_size_m from the config (physics.pixel_size_m).
  - It uses z positions derived from physics.z_list_m (cumulative distances
    input->L1->L2->L3) converted to mm.
  - Thickness maps are expected to be `layer{1,2,3}_phase_rad.thickness_m.npy`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import yaml


def _load_thickness_layers(thickness_dir: Path, n_layers: int = 3) -> List[np.ndarray]:
    layers: List[np.ndarray] = []
    for i in range(1, n_layers + 1):
        npy_path = thickness_dir / f"layer{i}_phase_rad.thickness_m.npy"
        if not npy_path.exists():
            raise SystemExit(f"Missing thickness file: {npy_path}")
        arr = np.load(npy_path)
        if arr.ndim != 2:
            raise SystemExit(f"Thickness array must be 2D, got shape {arr.shape} for {npy_path}")
        layers.append(arr.astype(float))
    return layers


def _downsample_average(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average downsampling so total volume is preserved.

    Example: 128x128, factor=4 -> 32x32, each new pixel is mean of 4x4 block.
    With pixel area scaled by factor^2, integrated volume stays the same.
    """
    if factor <= 1:
        return arr
    h, w = arr.shape
    if h % factor != 0 or w % factor != 0:
        raise SystemExit(f"Downsample factor {factor} must divide array shape {arr.shape}")
    new_h, new_w = h // factor, w // factor
    arr_reshaped = arr.reshape(new_h, factor, new_w, factor)
    return arr_reshaped.mean(axis=(1, 3))


def _build_macro_content(
    layers: List[np.ndarray],
    pixel_size_m: float,
    layer_offsets_m: List[float],
    material_name: str = "Vero_White_Plus",
) -> str:
    if len(layer_offsets_m) < len(layers):
        raise SystemExit("layer_offsets_m must have at least as many entries as layers")

    pixel_size_mm = float(pixel_size_m) * 1e3
    layer_offsets_mm = [float(z) * 1e3 for z in layer_offsets_m]

    # assume all layers have same grid size
    ny, nx = layers[0].shape
    for arr in layers[1:]:
        if arr.shape != (ny, nx):
            raise SystemExit(f"All layers must have same shape; got {layers[0].shape} and {arr.shape}")

    lines: List[str] = []
    lines.append("Sub Main()")
    lines.append("  ' Auto-generated macro: build 3 dielectric layers from thickness maps")
    lines.append(f"  Const materialName As String = \"{material_name}\"")
    lines.append(f"  Const pixelSize As Double = {pixel_size_mm:.6f} ' mm")
    lines.append("")

    for li, thick_m in enumerate(layers, start=1):
        offset_mm = layer_offsets_mm[li - 1]
        lines.append(f"  ' Layer {li}")
        lines.append(f"  With Brick")
        lines.append("    .Reset")
        lines.append(f"    .Component \"Layer{li}\"")
        lines.append("  End With")
        lines.append("")
        # create one brick per pixel
        for iy in range(ny):
            for ix in range(nx):
                t_mm = float(thick_m[iy, ix]) * 1e3
                if t_mm <= 0.0:
                    continue
                x0 = ix * pixel_size_mm
                x1 = (ix + 1) * pixel_size_mm
                y0 = iy * pixel_size_mm
                y1 = (iy + 1) * pixel_size_mm
                z0 = offset_mm
                z1 = offset_mm + t_mm
                name = f"L{li}_x{ix}_y{iy}"
                lines.append("  With Brick")
                lines.append("    .Reset")
                lines.append(f"    .Name \"{name}\"")
                lines.append(f"    .Component \"Layer{li}\"")
                lines.append(f"    .Material \"{material_name}\"")
                lines.append(f"    .Xrange {x0:.6f}, {x1:.6f}")
                lines.append(f"    .Yrange {y0:.6f}, {y1:.6f}")
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
    ap.add_argument("--thickness_dir", default=None)
    ap.add_argument("--out_macro", default=None)
    ap.add_argument("--material_name", default="Vero_White_Plus")
    ap.add_argument("--downsample", type=int, default=1, help="block downsample factor (e.g. 4 for 128->32)")
    args = ap.parse_args()

    cfg = None
    pixel_size_m: float | None = None
    layer_offsets_m: List[float] | None = None

    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        phys = cfg.get("physics", {}) or {}
        out_root = str((cfg.get("output", {}) or {}).get("root_dir", "outputs"))
        pixel_size_m = float(phys.get("pixel_size_m", 1.0e-3))
        z_list = list(phys.get("z_list_m", [0.10, 0.10, 0.10, 0.10]))
        # layer positions: cumulative distances input->L1->L2->L3
        if len(z_list) >= 3:
            z1 = float(z_list[0])
            z2 = z1 + float(z_list[1])
            z3 = z2 + float(z_list[2])
            layer_offsets_m = [z1, z2, z3]
    if pixel_size_m is None:
        pixel_size_m = 1.0e-3
    if layer_offsets_m is None:
        # fallback: simple equally spaced offsets
        layer_offsets_m = [0.0, 0.02, 0.04]

    # Default output locations are derived from config.output.root_dir if present.
    out_root = out_root if "out_root" in locals() else "outputs"
    thickness_dir = Path(args.thickness_dir) if args.thickness_dir else Path(out_root) / "printable"
    thickness_dir.mkdir(parents=True, exist_ok=True)

    layers = _load_thickness_layers(thickness_dir, n_layers=3)

    # optional downsampling (e.g. 128x128 -> 32x32 with factor=4)
    if args.downsample and args.downsample > 1:
        factor = int(args.downsample)
        layers = [_downsample_average(l, factor) for l in layers]
        # each coarse pixel covers (factor x factor) fine pixels
        pixel_size_m *= factor

    macro = _build_macro_content(layers, pixel_size_m=pixel_size_m, layer_offsets_m=layer_offsets_m, material_name=args.material_name)

    out_macro_path = Path(args.out_macro) if args.out_macro else (Path(out_root) / "cst_macro" / "build_3layers.cstmacro")
    out_macro_path.parent.mkdir(parents=True, exist_ok=True)
    out_macro_path.write_text(macro, encoding="utf-8")

    print(f"Wrote CST macro to {out_macro_path}")


if __name__ == "__main__":
    main()
