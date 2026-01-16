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


def _regions_grid(*, size: int, rows: int, cols: int, square: int, margin: int, gap: int, num_classes: int):
    if rows * cols < num_classes:
        raise ValueError(f"grid {rows}x{cols} has {rows*cols} cells < num_classes={num_classes}")
    needed_w = cols * square + (cols - 1) * gap + 2 * margin
    needed_h = rows * square + (rows - 1) * gap + 2 * margin
    if needed_w > size or needed_h > size:
        raise ValueError(f"grid does not fit into detector plane size={size}: need {needed_h}x{needed_w}")

    regions: list[DetectorRegion] = []
    for r in range(rows):
        for c in range(cols):
            if len(regions) >= num_classes:
                break
            x0 = margin + r * (square + gap)
            y0 = margin + c * (square + gap)
            regions.append(DetectorRegion(x0=x0, x1=x0 + square, y0=y0, y1=y0 + square))
        if len(regions) >= num_classes:
            break
    return regions


def _build_regions(cfg: dict, *, size: int):
    clf = cfg.get("classifier", {})
    scheme = clf.get("scheme", "explicit")
    if scheme == "grid":
        grid = clf["grid"]
        return _regions_grid(
            size=size,
            rows=int(grid["rows"]),
            cols=int(grid["cols"]),
            square=int(grid["square"]),
            margin=int(grid.get("margin", 0)),
            gap=int(grid.get("gap", 0)),
            num_classes=int(clf["num_classes"]),
        )
    return _regions_from_cfg(clf["regions"])


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
        detector_regions=_build_regions(cfg, size=size),
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
