from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_lut_csv(path: str):
    phases = []
    geom = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in reader:
            phases.append(float(row["phase_rad"]))
            geom.append(float(row["geom_param"]))
    phases = np.asarray(phases, dtype=np.float64)
    geom = np.asarray(geom, dtype=np.float64)
    order = np.argsort(phases)
    return phases[order], geom[order]


def _quantize_phase_to_lut(phase_rad: np.ndarray, lut_phase: np.ndarray, lut_geom: np.ndarray):
    # nearest neighbor quantization
    phase_flat = phase_rad.reshape(-1)
    idx = np.abs(phase_flat[:, None] - lut_phase[None, :]).argmin(axis=1)
    geom_flat = lut_geom[idx]
    return geom_flat.reshape(phase_rad.shape), idx.reshape(phase_rad.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase_npz", required=True, help="phase_masks.npz exported by training")
    parser.add_argument("--lut_csv", default="assets/meta_atom_lut_example.csv")
    parser.add_argument("--out_dir", default="outputs/meta_atom_exports")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lut_phase, lut_geom = _load_lut_csv(args.lut_csv)

    data = np.load(args.phase_npz)
    for key in ["layer1_phase_rad", "layer2_phase_rad", "layer3_phase_rad"]:
        phase = data[key]
        geom_map, idx_map = _quantize_phase_to_lut(phase, lut_phase, lut_geom)
        np.save(out_dir / f"{key}.geom_param.npy", geom_map)
        np.save(out_dir / f"{key}.lut_index.npy", idx_map)

    print(f"Exported meta-atom maps to {out_dir}")


if __name__ == "__main__":
    main()
