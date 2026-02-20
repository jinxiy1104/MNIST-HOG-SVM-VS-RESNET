from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None


def load_mnist_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    if pd is not None:
        df = pd.read_csv(p, header=None)
        if df.shape[1] != 785:
            raise ValueError(f"Expected 785 columns, got {df.shape[1]} in {p}")
        arr = df.to_numpy()
    else:
        arr = np.loadtxt(p, delimiter=",", dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != 785:
            raise ValueError(f"Expected 785 columns, got {arr.shape[1] if arr.ndim == 2 else arr.shape} in {p}")
    y = arr[:, 0].astype(np.int64)
    X = arr[:, 1:].astype(np.uint8)

    if y.min() < 0 or y.max() > 9:
        raise ValueError(f"Labels out of range [0,9] in {p}: min={y.min()}, max={y.max()}")
    if X.min() < 0 or X.max() > 255:
        raise ValueError(f"Pixels out of range [0,255] in {p}: min={X.min()}, max={X.max()}")

    return X, y


def pixels784_to_image28(pixels784: np.ndarray) -> np.ndarray:
    if pixels784.shape != (784,):
        raise ValueError(f"Expected shape (784,), got {pixels784.shape}")
    img = pixels784.reshape(28, 28)
    return img.astype(np.uint8)


def batch_pixels_to_images28(X_pixels784: np.ndarray) -> np.ndarray:
    if X_pixels784.ndim != 2 or X_pixels784.shape[1] != 784:
        raise ValueError(f"Expected shape (N,784), got {X_pixels784.shape}")
    imgs = X_pixels784.reshape(-1, 28, 28)
    return imgs.astype(np.uint8)
