"""Generate CST-ready parameter exports from project config, LUT and phase masks.

Usage examples:
  python scripts/generate_cst_params.py --config configs/emnist_letters.yaml --phase_npz outputs/.../phase_masks.npz --lut_csv assets/meta_atom_lut_example.csv --out_dir outputs/cst_params
  python scripts/generate_cst_params.py --config configs/emnist_letters.yaml --out_dir outputs/cst_params

Outputs:
  - out_dir/cst_spec.md        : copy of CST simulation spec + computed material params
  - out_dir/cst_params.json    : JSON summary (frequency, wavelength, eps_r, tan_delta, sigma, pixel_size)
  - out_dir/layer{1,2,3}_geom.csv : (optional) per-layer geometry param grids if --phase_npz provided
  - out_dir/layer{1,2,3}_lut_index.csv : corresponding LUT index grids
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

EPS0 = 8.8541878128e-12
C = 299792458.0


def _load_lut_csv(path: str):
    phases = []
    geom = []
    with open(path, "r", encoding="utf-8") as f:
        # skip commented lines
        header = None
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in line.strip().split(",")]
                continue
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 2:
                continue
            phases.append(float(parts[0]))
            geom.append(float(parts[1]))
    phases = np.asarray(phases, dtype=np.float64)
    geom = np.asarray(geom, dtype=np.float64)
    order = np.argsort(phases)
    return phases[order], geom[order]


def _quantize_phase_to_lut(phase_rad: np.ndarray, lut_phase: np.ndarray, lut_geom: np.ndarray):
    phase_flat = phase_rad.reshape(-1)
    idx = np.abs(phase_flat[:, None] - lut_phase[None, :]).argmin(axis=1)
    geom_flat = lut_geom[idx]
    return geom_flat.reshape(phase_rad.shape), idx.reshape(phase_rad.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/emnist_letters.yaml")
    parser.add_argument("--lut_csv", default="assets/meta_atom_lut_example.csv")
    parser.add_argument("--phase_npz", default=None, help="phase_masks.npz from training")
    parser.add_argument("--out_dir", default="outputs/cst_params")
    parser.add_argument("--freq_hz", default=None, type=float, help="override frequency in Hz (optional)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Explicitly read YAML as UTF-8 to avoid platform-dependent decoding errors on Windows
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # derive physical params
    wavelength_m = cfg.get("physics", {}).get("wavelength_m")
    pixel_size_m = cfg.get("physics", {}).get("pixel_size_m")
    material = cfg.get("material", {}) or {}
    eps_r = material.get("eps_r", None)
    tan_delta = material.get("tan_delta", None)

    if args.freq_hz is not None:
        freq_hz = args.freq_hz
        wavelength_m_cfg = C / freq_hz
    elif wavelength_m is not None:
        wavelength_m_cfg = wavelength_m
        freq_hz = C / wavelength_m_cfg
    else:
        raise SystemExit("Either physics.wavelength_m in config or --freq_hz must be provided")

    omega = 2 * np.pi * freq_hz
    sigma = None
    if eps_r is not None and tan_delta is not None:
        sigma = float(tan_delta * omega * EPS0 * eps_r)

    summary = {
        "frequency_hz": float(freq_hz),
        "wavelength_m": float(wavelength_m_cfg),
        "pixel_size_m": float(pixel_size_m) if pixel_size_m is not None else None,
        "eps_r": float(eps_r) if eps_r is not None else None,
        "tan_delta": float(tan_delta) if tan_delta is not None else None,
        "sigma_siemens_per_m": float(sigma) if sigma is not None else None,
    }

    # write JSON summary
    (out_dir / "cst_params.json").write_text(json.dumps(summary, indent=2))

    # copy/write a short CST spec.md
    spec_lines = []
    spec_lines.append("# Generated CST Parameters\n")
    spec_lines.append(f"- frequency_hz: {summary['frequency_hz']}")
    spec_lines.append(f"- wavelength_m: {summary['wavelength_m']}")
    spec_lines.append(f"- pixel_size_m: {summary['pixel_size_m']}")
    spec_lines.append(f"- eps_r: {summary['eps_r']}")
    spec_lines.append(f"- tan_delta: {summary['tan_delta']}")
    spec_lines.append(f"- sigma_siemens_per_m: {summary['sigma_siemens_per_m']}")
    spec_lines.append("")
    spec_lines.append("## Notes:\n- Units: meters (m), Siemens/m for conductivity.\n- Use pixel_size_m to set unit cell size in CST.\n")
    (out_dir / "cst_spec.md").write_text("\n".join(spec_lines))

    # if phase_npz given, quantize and export per-layer CSVs
    if args.phase_npz is not None:
        lut_phase, lut_geom = _load_lut_csv(args.lut_csv)
        data = np.load(args.phase_npz)
        for key_idx, key in enumerate(["layer1_phase_rad", "layer2_phase_rad", "layer3_phase_rad"], start=1):
            if key not in data:
                continue
            phase = data[key]
            geom_map, idx_map = _quantize_phase_to_lut(phase, lut_phase, lut_geom)
            # save geom_map as CSV (rows x cols), values in same units as LUT (user must ensure LUT units)
            np.savetxt(out_dir / f"layer{key_idx}_geom.csv", geom_map, delimiter=",", fmt="%.6e")
            np.savetxt(out_dir / f"layer{key_idx}_lut_index.csv", idx_map, delimiter=",", fmt="%d")

        print(f"Exported per-layer geom CSVs to {out_dir}")
    else:
        print(f"Wrote CST params summary to {out_dir}/cst_params.json and {out_dir}/cst_spec.md")


if __name__ == "__main__":
    main()
