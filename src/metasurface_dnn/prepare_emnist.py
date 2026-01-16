from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torchvision import datasets


def _fix_emnist_orientation(img: np.ndarray) -> np.ndarray:
    # EMNIST images are rotated/flipped compared to standard MNIST display.
    # Common fix: rotate 90 degrees and flip left-right.
    # Input is (H,W) uint8.
    return np.fliplr(np.rot90(img, k=1))


def _resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    # Nearest-neighbor resize without extra deps; good enough for pipeline.
    # img: (H,W)
    h, w = img.shape
    if h == size and w == size:
        return img
    ys = (np.linspace(0, h - 1, size)).astype(np.int64)
    xs = (np.linspace(0, w - 1, size)).astype(np.int64)
    return img[np.ix_(ys, xs)]


def export_emnist_letters_npz(
    *,
    root: str,
    out_dir: str,
    image_size: int = 128,
    n_train: int | None = None,
    n_test: int | None = None,
    val_split: float = 0.1,
    seed: int = 42,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train = datasets.EMNIST(root=root, split="letters", train=True, download=True)
    test = datasets.EMNIST(root=root, split="letters", train=False, download=True)

    # Labels for EMNIST letters are 1..26; convert to 0..25
    x_train = train.data.numpy()  # (N,28,28) uint8
    y_train = (train.targets.numpy().astype(np.int64) - 1)

    x_test = test.data.numpy()
    y_test = (test.targets.numpy().astype(np.int64) - 1)

    if n_train is not None:
        x_train = x_train[:n_train]
        y_train = y_train[:n_train]
    if n_test is not None:
        x_test = x_test[:n_test]
        y_test = y_test[:n_test]

    rng = np.random.default_rng(seed)
    idx = np.arange(x_train.shape[0])
    rng.shuffle(idx)

    n_val = int(round(val_split * len(idx)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    def make_xy(x_raw: np.ndarray, y: np.ndarray, indices: np.ndarray | None):
        if indices is not None:
            x_raw = x_raw[indices]
            y = y[indices]

        xs = []
        for i in range(x_raw.shape[0]):
            img = _fix_emnist_orientation(x_raw[i])
            img = _resize_to_square(img, image_size)
            amp = (img.astype(np.float32) / 255.0)
            xs.append(amp)
        x = np.stack(xs, axis=0).astype(np.float32)
        return x, y.astype(np.int64)

    x_tr, y_tr = make_xy(x_train, y_train, tr_idx)
    x_val, y_val = make_xy(x_train, y_train, val_idx)
    x_te, y_te = make_xy(x_test, y_test, None)

    np.savez_compressed(out_path / "emnist_letters_train.npz", x=x_tr, y=y_tr)
    np.savez_compressed(out_path / "emnist_letters_val.npz", x=x_val, y=y_val)
    np.savez_compressed(out_path / "emnist_letters_test.npz", x=x_te, y=y_te)

    print("Saved:")
    print(" -", out_path / "emnist_letters_train.npz", x_tr.shape, y_tr.shape)
    print(" -", out_path / "emnist_letters_val.npz", x_val.shape, y_val.shape)
    print(" -", out_path / "emnist_letters_test.npz", x_te.shape, y_te.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw", help="torchvision download root")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_test", type=int, default=None)
    args = p.parse_args()

    export_emnist_letters_npz(
        root=args.root,
        out_dir=args.out_dir,
        image_size=args.image_size,
        n_train=args.n_train,
        n_test=args.n_test,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
