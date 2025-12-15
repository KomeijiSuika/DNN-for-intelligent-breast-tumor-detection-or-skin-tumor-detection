from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import NPZAmplitudeDataset, SyntheticTumorDataset
from .model import DetectorRegion, ThreeLayerMetasurfaceDNN
from .physics import angular_spectrum_propagate
from .utils import ensure_dir, load_yaml, save_json, save_npz, select_device, timestamp


def _regions_from_cfg(regions_cfg):
    return [DetectorRegion(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in regions_cfg]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = select_device(cfg.get("device", "auto"))

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # dataset (prefer test set if exists, else synthetic)
    test_path = cfg["data"]["test_npz"]
    if Path(test_path).exists():
        ds = NPZAmplitudeDataset(test_path)
    else:
        syn = cfg["data"]["synthetic"]
        ds = SyntheticTumorDataset(max(args.n, 64), int(syn["image_size"]), seed=cfg.get("seed", 44))

    size = int(ds[0][0].shape[-1])

    model = ThreeLayerMetasurfaceDNN(
        size=size,
        wavelength_m=cfg["physics"]["wavelength_m"],
        pixel_size_m=cfg["physics"]["pixel_size_m"],
        z_list_m=list(cfg["physics"]["z_list_m"]),
        n0=cfg["physics"].get("n0", 1.0),
        phase_init=cfg["model"].get("phase_init", "uniform"),
        detector_regions=_regions_from_cfg(cfg["classifier"]["regions"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    out_dir = ensure_dir(str(Path(cfg["output"]["root_dir"]) / "simulations" / timestamp()))

    # pick first n samples
    xs = []
    ys = []
    for i in range(min(args.n, len(ds))):
        x, y = ds[i]
        xs.append(x.numpy())
        ys.append(int(y.item()))
    x = torch.from_numpy(np.stack(xs, axis=0)).to(device)
    y = np.array(ys, dtype=np.int64)

    with torch.no_grad():
        logits = model(x).cpu().numpy()
        pred = np.argmax(logits, axis=1)

    # also save detector-plane intensity for the first sample (use internal forward pieces)
    amp0 = x[0:1].to(device).to(torch.float32)
    field = amp0.to(torch.complex64)
    z_list = list(cfg["physics"]["z_list_m"])
    field = angular_spectrum_propagate(field, wavelength_m=cfg["physics"]["wavelength_m"], pixel_size_m=cfg["physics"]["pixel_size_m"], z_m=z_list[0], n0=cfg["physics"].get("n0", 1.0))
    field = model.l1(field)
    field = angular_spectrum_propagate(field, wavelength_m=cfg["physics"]["wavelength_m"], pixel_size_m=cfg["physics"]["pixel_size_m"], z_m=z_list[1], n0=cfg["physics"].get("n0", 1.0))
    field = model.l2(field)
    field = angular_spectrum_propagate(field, wavelength_m=cfg["physics"]["wavelength_m"], pixel_size_m=cfg["physics"]["pixel_size_m"], z_m=z_list[2], n0=cfg["physics"].get("n0", 1.0))
    field = model.l3(field)
    field = angular_spectrum_propagate(field, wavelength_m=cfg["physics"]["wavelength_m"], pixel_size_m=cfg["physics"]["pixel_size_m"], z_m=z_list[3], n0=cfg["physics"].get("n0", 1.0))
    intensity = (field.real**2 + field.imag**2)[0].cpu().numpy()

    # plots
    plt.figure(figsize=(5, 4))
    plt.imshow(intensity, cmap="magma")
    plt.title("Detector intensity (sample 0)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "intensity_sample0.png"), dpi=150)
    plt.close()

    save_npz(str(Path(out_dir) / "batch_inputs.npz"), {"x": np.stack(xs, 0), "y": y, "logits": logits, "pred": pred})
    save_json(str(Path(out_dir) / "summary.json"), {"n": int(len(xs)), "acc": float((pred == y).mean())})

    print(f"Saved simulation to {out_dir}")


if __name__ == "__main__":
    main()
