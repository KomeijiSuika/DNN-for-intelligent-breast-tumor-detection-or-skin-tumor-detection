from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import NPZAmplitudeDataset
from .model import DetectorRegion, ThreeLayerMetasurfaceDNN
from .utils import ensure_dir, load_yaml, select_device, timestamp


def _regions_from_cfg(regions_cfg):
    return [DetectorRegion(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in regions_cfg]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = select_device(cfg.get("device", "auto"))

    split_path = cfg["data"][f"{args.split}_npz"]
    ds = NPZAmplitudeDataset(split_path)
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

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    loader = DataLoader(ds, batch_size=64, shuffle=False)

    out_dir = ensure_dir(str(Path(cfg["output"]["root_dir"]) / "predictions"))
    out_csv = args.out_csv or str(Path(out_dir) / f"pred_{args.split}_{timestamp()}.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "y_true", "y_pred", "logit_0", "logit_1"])

        idx0 = 0
        with torch.no_grad():
            for x, y in loader:
                b = x.shape[0]
                x = x.to(device)
                logits = model(x).cpu().numpy()
                pred = np.argmax(logits, axis=1)
                y = y.numpy()
                for i in range(b):
                    writer.writerow([idx0 + i, int(y[i]), int(pred[i]), float(logits[i, 0]), float(logits[i, 1])])
                idx0 += b

    print(f"Saved predictions to {out_csv}")


if __name__ == "__main__":
    main()
