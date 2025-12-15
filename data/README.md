# Dataset placeholder

This project expects **binary classification** data for tumor detection.

## Recommended processed format
Put the following files under `data/processed/`:

- `train.npz`
- `val.npz`
- `test.npz`

Each `.npz` should contain:
- `x`: float32 array with shape `(N, H, W)` (grayscale amplitude map, normalized to `[0, 1]`)
- `y`: int64 array with shape `(N,)` where `0/1` are class labels

## Notes
- If these files are missing, the code can generate a **synthetic** dataset (for pipeline verification only).
- Do not commit real medical datasets to git.
