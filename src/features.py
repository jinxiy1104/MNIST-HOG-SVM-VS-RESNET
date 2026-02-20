import numpy as np
from skimage.feature import hog


def image28_to_hog(image28_f32: np.ndarray) -> np.ndarray:
    if image28_f32.shape != (28, 28):
        raise ValueError(f"Expected image shape (28,28), got {image28_f32.shape}")
    feat = hog(
        image28_f32,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def images28_to_hog_features(images28_f32: np.ndarray) -> np.ndarray:
    if images28_f32.ndim != 3 or images28_f32.shape[1:] != (28, 28):
        raise ValueError(f"Expected shape (N,28,28), got {images28_f32.shape}")
    feats = [image28_to_hog(img) for img in images28_f32]
    return np.stack(feats, axis=0).astype(np.float32)
