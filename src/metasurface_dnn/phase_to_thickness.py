from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def phase_to_thickness_m(phase_rad: np.ndarray, *, wavelength_m: float, n_material: float, n0: float = 1.0) -> np.ndarray:
    """Convert phase (0..2π) to thickness for a uniform dielectric slab.

    For a slab of thickness t, phase delay (relative to background) is:
      phi = k0 * (n_material - n0) * t
    => t = phi * λ / (2π (n_material - n0))

    This gives a thickness in [0, t_period) where t_period = λ/(n_material-n0).
    """
    dn = float(n_material - n0)
    if dn <= 0:
        raise ValueError("n_material must be > n0")

    # Wrap to [0, 2π)
    phi = np.mod(phase_rad, 2.0 * np.pi)
    t = phi * float(wavelength_m) / (2.0 * np.pi * dn)
    return t.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase_npz", required=True, help="phase_masks.npz from training")
    p.add_argument("--wavelength_m", type=float, required=True)
    p.add_argument("--n_material", type=float, default=None, help="Refractive index of 3D-print material")
    p.add_argument("--eps_r", type=float, default=None, help="Relative permittivity (use n≈sqrt(eps_r))")
    p.add_argument("--tan_delta", type=float, default=None, help="Loss tangent (recorded in metadata)")
    p.add_argument("--n0", type=float, default=1.0)
    p.add_argument("--out_dir", default="outputs/printable")
    p.add_argument("--export_csv", action="store_true", help="Also export thickness maps as CSV for CST")
    args = p.parse_args()

    if args.n_material is None:
        if args.eps_r is None:
            raise SystemExit("Provide either --n_material or --eps_r")
        n_material = float(np.sqrt(float(args.eps_r)))
    else:
        n_material = float(args.n_material)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.phase_npz)
    for key in ["layer1_phase_rad", "layer2_phase_rad", "layer3_phase_rad"]:
        phase = data[key]
        thick = phase_to_thickness_m(phase, wavelength_m=args.wavelength_m, n_material=n_material, n0=args.n0)
        np.save(out_dir / f"{key}.thickness_m.npy", thick)

        if args.export_csv:
            csv_path = out_dir / f"{key}.thickness_m.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["thickness_m"])
                # Write as a matrix: one row per line, CST can import as ASCII grid.
                for row in thick:
                    writer.writerow([f"{v:.8e}" for v in row])

    meta = {
        "phase_npz": args.phase_npz,
        "wavelength_m": args.wavelength_m,
        "n_material": n_material,
        "eps_r": args.eps_r,
        "tan_delta": args.tan_delta,
        "n0": args.n0,
        "thickness_period_m": float(args.wavelength_m) / float(n_material - args.n0),
    }
    (out_dir / "meta.json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved thickness maps to {out_dir}")


if __name__ == "__main__":
    main()
