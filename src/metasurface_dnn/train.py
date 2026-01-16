from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import NPZAmplitudeDataset, SyntheticTumorDataset
from .model import DetectorRegion, ThreeLayerMetasurfaceDNN
from .utils import ensure_dir, load_yaml, save_json, save_npz, select_device, timestamp


def _regions_from_cfg(regions_cfg):
    regions = []
    for r in regions_cfg:
        regions.append(DetectorRegion(x0=int(r[0]), x1=int(r[1]), y0=int(r[2]), y1=int(r[3])))
    return regions


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


def _try_load_dataset(npz_path: str):
    if os.path.exists(npz_path):
        return NPZAmplitudeDataset(npz_path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--require-real-data", action="store_true", help="Fail if data/processed/*.npz not found")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    device = select_device(cfg.get("device", "auto"))

    out_root = cfg["output"]["root_dir"]
    run_name = cfg["output"].get("run_name") or timestamp()
    run_dir = ensure_dir(str(Path(out_root) / run_name))
    ckpt_dir = ensure_dir(str(Path(out_root) / "checkpoints"))

    # data
    train_ds = _try_load_dataset(cfg["data"]["train_npz"])
    val_ds = _try_load_dataset(cfg["data"]["val_npz"])
    test_ds = _try_load_dataset(cfg["data"]["test_npz"])

    if train_ds is None or val_ds is None or test_ds is None:
        if args.require_real_data or not cfg["data"].get("use_synthetic_if_missing", True):
            raise FileNotFoundError(
                "Missing data/processed/*.npz. Provide train/val/test.npz or run without --require-real-data."
            )
        syn = cfg["data"]["synthetic"]
        size = int(syn["image_size"])
        train_ds = SyntheticTumorDataset(syn["n_train"], size, seed=cfg.get("seed", 42))
        val_ds = SyntheticTumorDataset(syn["n_val"], size, seed=cfg.get("seed", 43))
        test_ds = SyntheticTumorDataset(syn["n_test"], size, seed=cfg.get("seed", 44))

    size = int(train_ds[0][0].shape[-1])

    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    # model
    regions = _build_regions(cfg, size=size)
    model = ThreeLayerMetasurfaceDNN(
        size=size,
        wavelength_m=cfg["physics"]["wavelength_m"],
        pixel_size_m=cfg["physics"]["pixel_size_m"],
        z_list_m=list(cfg["physics"]["z_list_m"]),
        n0=cfg["physics"].get("n0", 1.0),
        phase_init=cfg["model"].get("phase_init", "uniform"),
        detector_regions=regions,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    def eval_loader(loader: DataLoader):
        model.eval()
        total = 0
        correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                total_loss += float(loss.item()) * x.shape[0]
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
                total += int(x.shape[0])
        return total_loss / max(total, 1), correct / max(total, 1)

    best_val_acc = -1.0
    history = []

    epochs = int(cfg["training"]["epochs"])
    log_every = int(cfg["training"].get("log_every", 50))

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * x.shape[0]
            seen += int(x.shape[0])

            if step % log_every == 0:
                pbar.set_postfix(loss=running / max(seen, 1))

        train_loss = running / max(seen, 1)
        val_loss, val_acc = eval_loader(val_loader)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(record)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = str(Path(ckpt_dir) / "best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "size": size,
                },
                ckpt_path,
            )

    test_loss, test_acc = eval_loader(test_loader)

    # export
    metrics = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
        "run_dir": run_dir,
        "checkpoint": str(Path(ckpt_dir) / "best.pt"),
        "device": str(device),
    }
    save_json(str(Path(run_dir) / "metrics.json"), metrics)

    phase = model.export_phase_masks()
    save_npz(
        str(Path(run_dir) / "phase_masks.npz"),
        {k: v.numpy() for k, v in phase.items()},
    )

    print(f"Saved run to {run_dir}")
    print(f"Best val acc: {best_val_acc:.4f} | Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
